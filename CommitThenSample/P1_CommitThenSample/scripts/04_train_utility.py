"""S4: utility models per LOSO fold and seed, calibration on the TRAINING LP streams, utilities on the test streams.

Folds (DESIGN §5.6): test fold k = each LOSO-eligible IoT-23 scenario; models are trained on the pcap-derived
payload segments of every OTHER IoT-23 malware scenario with a pcap, the honeypots (first 50% of each
honeypot scenario by time) and the CIC training half (BenignTraffic). Fold "all" (zero-shot, applied unchanged to
LP-Edge): all eligible scenarios + honeypots + CIC training half. Three LightGBM seeds per fold.

Stages (resumable; every output atomic):
  A  feature cache  results/models/_cache/<ds>/<source>.npz      features of the payload units of every training
                    source (rolling statistics computed online over the WHOLE source), f_q, eligibility, labels
  B  models         results/models/<fold>/<seed>/{gmal.json, gmal.lgb.txt, hit_<q>.json, hit_<q>.lgb.txt,
                    m2.parquet, u8.parquet, bundle.json, train_info.json}
  C  calibration    .../{Qnorm.json, phi0.json, lamref.json, grid.json, boost.json} pooled over the fold's training LP streams
                    trainlp_<j>_p<ppm> (j = its training scenarios): Q̃_q, φ0 per ρ, λ_ref per (ρ, design) by
                    bisection on the mean expected cost, adaptive-arm λ_ref^A by boost replay, V per epoch;
                    boost trigger set F = detectors firing on <= 1e-3 of the benign training payload units
  D  application    util_<U3|U5>_s<seed>, util_U8, util_U7 columns on the fold's test streams (LP-IoT23 / LP-Edge,
                    natural); .../{auc.json, dups.json, ks.json}; DONE
Run under memguard from the project root::

    .venv/bin/python memguard/memguard.py run --name p1_s4 --workers 3 -- \
        .venv/bin/python -I -u P1_CommitThenSample/scripts/04_train_utility.py --folds all --seeds 3

LightGBM: 8 threads per fit, <= 3 concurrent fits (process pool of min(3, workers)).

Recipe variants (rev1 sensitivity, review R3-M1(a); config/streams.yaml zero_payload_variants)::

    ... 04_train_utility.py --variant zq --folds 1-1,3-1,7-1,34-1,48-1,49-1 --stages A,B,C --seeds 3

writes everything under results/models_<variant>/ (own feature cache, models, calibration; results/models/ is only
read by nothing and never written), calibrates on trainlp_<j>_p<ppm>_<variant> and applies to
lp_iot23_<k>_p<ppm>_<variant> at the variant's prevalences (natural streams and the zero-shot fold are out of
scope). Training sources (whole captures, natural prevalence) follow the variant's rule: keep 'q' -> unchanged
(payload q = 1); keep 0 -> the pure zero-payload connections of each IoT-23 malware scenario's seed host are
removed before the rolling features are computed. LightGBM settings and seeds are unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctsample.utils.io import (RESULTS, atomic_save_npz, atomic_write_json, atomic_write_text,  # noqa: E402
                               is_done, mark_done)
from ctsample.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory, workers  # noqa: E402

MODELS = RESULTS / "models"
CACHE = MODELS / "_cache"
STREAMS = RESULTS / "streams"
VARIANT = None             # recipe variant (--variant): models root, stream ids, prevalences, training-source rule
PREVALENCES = None         # None: config prevalences
TRAIN_ZERO_KEEP = 1.0      # keep probability of the seed host's pure zero-payload connections in training sources
RHOS = (0.01, 0.02, 0.05, 0.10)
GAMMA = 0.3
LGB_THREADS = 8
ALPHA_SIDE = 0.025
V_REF_EPOCHS = 2880
# CtS-family static designs calibrated here: name -> (kind, utility key | None, tau_mix)
KAPPA_B, W_B, ADAPT_R = 5.0, 20, 4
FORCE_CALIB = False
ADAPTIVE = ("U6", "U6tau0.2", "U6tau0.5")          # boost designs: also calibrated with the boost replayed
DESIGNS = {"U0": ("uniform", None, 0.0), "U1": ("S", None, 0.0), "U3": ("A", "U3", 0.0), "U5": ("A", "U5", 0.0),
           "U6": ("A", "U5", 0.0), "U6tau0.2": ("mixture", "U5", 0.2), "U6tau0.5": ("mixture", "U5", 0.5), "U8": ("A", "U8", 0.0),
           "U7": ("A", "U7", 0.0)}


# ------------------------------------------------------------------------------------------ #
def _set_variant(variant: str | None) -> None:
    """Point every path / stream id of this script at a recipe variant (also called in each pool worker)."""
    global VARIANT, MODELS, CACHE, PREVALENCES, TRAIN_ZERO_KEEP
    if not variant:
        return
    from ctsample.streams.recipe import load_config, zero_variant_cfg
    v = zero_variant_cfg(load_config(), variant)
    VARIANT = variant
    MODELS = RESULTS / f"models_{variant}"
    CACHE = MODELS / "_cache"
    PREVALENCES = [float(x) for x in v["prevalences"]]
    TRAIN_ZERO_KEEP = 1.0 if v["keep"] == "q" else float(v["keep"])     # 'q' at natural prevalence = 1
    if TRAIN_ZERO_KEEP not in (0.0, 1.0):
        raise ValueError("training-source zero-payload thinning is defined for keep in {q, 0} only")


def _init_worker(ppid: int, variant: str | None = None) -> None:
    import os
    import threading

    def watch():
        while True:
            if os.getppid() != ppid:
                os._exit(1)
            time.sleep(2)

    threading.Thread(target=watch, daemon=True).start()
    _set_variant(variant)


def _pool(n):
    import os
    return ProcessPoolExecutor(max_workers=n, mp_context=get_context("spawn"), initializer=_init_worker,
                               initargs=(os.getpid(), VARIANT))


def _run_pool(tasks, nw, label):
    """tasks: list of (fn, args); returns results list; raises on worker errors."""
    if not tasks:
        return []
    out, errors = [], []
    ex = _pool(nw)
    try:
        pend = {ex.submit(fn, *a): i for i, (fn, a) in enumerate(tasks)}
        while pend:
            done, _ = wait(set(pend), timeout=30, return_when=FIRST_COMPLETED)
            for f in done:
                pend.pop(f)
                r = f.result()
                if isinstance(r, dict) and r.get("error"):
                    errors.append(r)
                    print(f"[{label}] ERROR {r['error']}\n{r.get('traceback', '')}", flush=True)
                else:
                    out.append(r)
            check_memory()
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    if errors:
        raise RuntimeError(f"{label}: {len(errors)} task(s) failed")
    return out


def fold_name(scenario: str | None) -> str:
    return "all" if scenario is None else scenario.replace("CTU-IoT-Malware-Capture-", "")


# ------------------------------------------------------------------------------------------ #
# stage A: feature cache
# ------------------------------------------------------------------------------------------ #
def training_sources(inv: dict, fold: str | None, eligible: list[str]) -> list[dict]:
    """[{ds, source, epoch_hi}] (epoch_hi = None: all epochs)."""
    out = []
    io = inv["datasets"]["iot23"]
    by_sc = {}
    for s in io["sources"]:
        by_sc.setdefault(s["scenario"], []).append(s)
    for sc, info in sorted(io["scenarios"].items()):
        if not info.get("has_pcap"):
            continue
        role = info.get("role")
        if role == "malware":
            if fold is None and sc not in eligible:
                continue
            if fold is not None and sc == fold:
                continue
            for s in by_sc.get(sc, []):
                out.append({"ds": "iot23", "source": s["source"], "epoch_hi": None, "scenario": sc})
        elif role == "honeypot":
            srcs = sorted(by_sc.get(sc, []), key=lambda s: s["first_ts"])
            t_mid = 0.5 * (min(s["first_ts"] for s in srcs) + max(s["last_ts"] for s in srcs))
            for s in srcs:
                if s["first_ts"] >= t_mid:
                    continue
                e_hi = None if s["last_ts"] < t_mid else int((t_mid - s["first_ts"]) // 30) + 1
                out.append({"ds": "iot23", "source": s["source"], "epoch_hi": e_hi, "scenario": sc, "t_mid": t_mid})
    out.append({"ds": "ciciot2023_benign", "source": "BenignTraffic", "epoch_hi": None, "scenario": "CIC-train"})
    return out


def _cache_path(ds, source):
    return CACHE / ds / f"{source}.npz"


def _seed_zero_rule(ds: str, source: str) -> dict | None:
    """[variant, keep 0] The training source's seed (infected) host whose pure zero-payload connections are removed:
    IoT-23 malware scenarios only (honeypots and the CIC half have no infected host); None = source unchanged."""
    if ds != "iot23":
        return None
    from ctsample.streams.recipe import load_inventory, seed_host
    inv = load_inventory()
    io = inv["datasets"]["iot23"]
    src = next((s for s in io["sources"] if s["source"] == source), None)
    if src is None or io["scenarios"].get(src["scenario"], {}).get("role") != "malware":
        return None
    ip, mac = seed_host(src["scenario"], inv)
    return {"variant": VARIANT, "keep": TRAIN_ZERO_KEEP, "scenario": src["scenario"], "seed_ip": ip, "mac": mac,
            "rule": "pure zero-payload connections (flow_uid without payload unit) of the seed host removed"}


def build_cache(ds: str, source: str) -> dict:
    """Features of the payload units of one source (rolling stats over all its units) + targets."""
    try:
        import polars as pl
        from ctsample.design.utility import DEFAULT_QUERIES, cell_index, eligibility_matrix, stratum_index
        from ctsample.units import features as F
        out = _cache_path(ds, source)
        if out.exists():
            return {"source": source, "skipped": True}
        t0 = time.time()
        zinfo = _seed_zero_rule(ds, source) if TRAIN_ZERO_KEEP < 1.0 else None
        u = pl.read_parquet(str(RESULTS / "units" / ds / source / "part-*.parquet"),
                            columns=["unit_gid", "epoch", "index", "pay", "meta"]
                            + (["device_mac", "flow_uid"] if zinfo else [])).sort(["epoch", "index"])
        extra_z = {}
        if zinfo:                     # [variant] drop the seed host's pure zero-payload connections (keep 0)
            n0 = u.height
            pf = u.filter(pl.col("pay"))["flow_uid"].unique().implode()
            drop = ((u["device_mac"].fill_null("") == zinfo["mac"]) & ~u["pay"] & ~u["flow_uid"].is_in(pf))
            u = u.filter(~drop).drop(["device_mac", "flow_uid"])
            zinfo.update({"n_units_before": n0, "n_units_dropped": n0 - u.height})
            extra_z = {"zero_thin": np.array(json.dumps(zinfo))}
        c = F.cols_from_meta(u["meta"].to_arrow())
        ep = u["epoch"].to_numpy().astype(np.int64)
        gid = u["unit_gid"].to_numpy()
        pay = u["pay"].to_numpy().astype(bool)
        del u
        X = F.features_batch(c, ep)
        idx = np.nonzero(pay)[0]
        Xp = X[idx]
        del X
        cp = F.take_cols(c, idx)
        del c
        tq = [q for q in DEFAULT_QUERIES]
        tb = pl.read_parquet(str(RESULTS / "tables" / ds / f"{source}.parquet"),
                             columns=["unit_gid"] + [f"f_{q}" for q in tq] + ["suricata_ready"])
        lab = pl.read_parquet(str(RESULTS / "labels" / ds / f"{source}.parquet"),
                              columns=["unit_gid", "label", "is_malicious", "label_source"])
        g = pl.DataFrame({"unit_gid": gid[idx]}).join(tb, on="unit_gid", how="left").join(lab, on="unit_gid", how="left")
        fm = np.stack([g[f"f_{q}"].fill_null(0).to_numpy().astype(np.int32) for q in tq], axis=1)
        known = (g["label"].fill_null("unknown") != "unknown").to_numpy()
        mal = g["is_malicious"].fill_null(False).to_numpy().astype(bool)
        el = eligibility_matrix(tq, cp)
        atomic_save_npz(out, X=Xp.astype(np.int32), unit_gid=gid[idx], epoch=ep[idx], cell=cell_index(cp),
                        stratum=stratum_index(cp), elig=el, f=fm, mal=mal, known=known,
                        label_source=g["label_source"].fill_null("none").to_numpy().astype(str),
                        queries=np.array(tq), suricata_ready=np.array(bool(g["suricata_ready"].fill_null(False).all())),
                        **extra_z)
        return {"source": source, "n_pay": int(len(idx)), "n_units": int(len(pay)), "s": round(time.time() - t0, 1),
                **({"zero_thin": zinfo} if zinfo else {})}
    except MemoryBudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        return {"error": f"{source}: {type(e).__name__}: {e}", "traceback": traceback.format_exc()}


def load_training_set(srcs: list[dict]) -> dict:
    parts = []
    for s in srcs:
        z = np.load(_cache_path(s["ds"], s["source"]), allow_pickle=False)
        keep = np.ones(len(z["epoch"]), bool) if s["epoch_hi"] is None else z["epoch"] < s["epoch_hi"]
        d = {k: z[k][keep] for k in ("X", "cell", "stratum", "elig", "f", "mal", "known")}
        d["src"] = np.full(int(keep.sum()), len(parts), np.int32)
        d["queries"] = list(z["queries"])
        parts.append(d)
    qs = parts[0]["queries"]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "cell", "stratum", "elig", "f", "mal", "known", "src")}
    out["queries"] = qs
    out["sources"] = [s["source"] for s in srcs]
    return out


# ------------------------------------------------------------------------------------------ #
# stage B: models
# ------------------------------------------------------------------------------------------ #
def _fit_task(fold: str, seed: int, name: str, train_npz: str, qi: int | None) -> dict:
    """One forest: name = 'gmal' or q; trained on the fold's training set (memory-mapped npz)."""
    try:
        from ctsample.design.utility import train_forest
        d = MODELS / fold / str(seed)
        if (d / f"{'gmal' if qi is None else 'hit_' + name}.json").exists():
            return {"name": name, "skipped": True}
        z = np.load(train_npz, mmap_mode="r")
        if qi is None:
            m = z["known"]
            X, y = z["X"][m], z["mal"][m].astype(np.int8)
        else:
            m = z["elig"][:, qi]
            X, y = z["X"][m], (z["f"][m, qi] > 0).astype(np.int8)
        f, bst, info = train_forest(np.asarray(X), np.asarray(y), seed=seed, threads=LGB_THREADS,
                                    meta={"fold": fold, "target": name})
        stem = "gmal" if qi is None else f"hit_{name}"
        if bst is not None:
            atomic_write_text(d / f"{stem}.lgb.txt", bst.model_to_string())
        atomic_write_text(d / f"{stem}.json", f.to_json())
        return {"name": name, **info}
    except MemoryBudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        return {"error": f"{fold}/{seed}/{name}: {type(e).__name__}: {e}", "traceback": traceback.format_exc()}


def stage_models(fold: str, seeds: list[int], srcs: list[dict], nw: int) -> None:
    from ctsample.design import utility as U
    from ctsample.design.treeeval import IntForest
    todo_seeds = [s for s in seeds if not (MODELS / fold / str(s) / "bundle.json").exists()]
    if not todo_seeds:
        return
    tr = load_training_set(srcs)
    qs = tr["queries"]
    tp = MODELS / fold / "_train.npz"
    if not tp.exists():
        atomic_save_npz_uncompressed(tp, X=tr["X"], elig=tr["elig"], f=tr["f"], mal=tr["mal"], known=tr["known"])
    n_pos = {q: int((tr["f"][tr["elig"][:, j], j] > 0).sum()) for j, q in enumerate(qs)}
    with_model = [q for q in qs if n_pos[q] > 0]
    # lookup tables (seed-independent)
    M, mf, P8, pf, mt = {}, {}, {}, {}, {}
    for j, q in enumerate(qs):
        el = tr["elig"][:, j]
        t = U.second_moment_table(tr["cell"][el], tr["f"][el, j], U.kind_of(q))
        M[q], mf[q] = U.m_fix(t["m"]), t["m"]
        st = U.stratum_table(tr["stratum"], tr["f"][:, j] > 0, el)
        P8[q], pf[q] = U.p_fix(st["p"]), st["p"]
        mt[q] = {"global_m2": t["global"], "rate": st["rate"], "n_elig": int(el.sum()), "n_pos": n_pos[q]}
    tasks = []
    for s in todo_seeds:
        (MODELS / fold / str(s)).mkdir(parents=True, exist_ok=True)
        tasks.append((_fit_task, (fold, s, "gmal", str(tp), None)))
        for q in with_model:
            tasks.append((_fit_task, (fold, s, q, str(tp), qs.index(q))))
    res = _run_pool(tasks, min(nw, 3), f"fit {fold}")
    for s in todo_seeds:
        d = MODELS / fold / str(s)
        hit = {q: IntForest.from_json((d / f"hit_{q}.json").read_text()) for q in with_model}
        gmal = IntForest.from_json((d / "gmal.json").read_text())
        info = {"fold": fold, "seed": s, "sources": tr["sources"], "n_train_payload_units": int(len(tr["X"])),
                "n_known": int(tr["known"].sum()), "n_mal": int((tr["mal"] & tr["known"]).sum()), "tables": mt,
                "fits": {r["name"]: {k: v for k, v in r.items() if k != "name"} for r in res
                         if isinstance(r, dict) and not r.get("skipped")}}
        b = U.UtilityBundle(queries=with_model, hit=hit, gmal=gmal, M={q: M[q] for q in with_model},
                            P8={q: P8[q] for q in with_model}, m_float={q: mf[q] for q in with_model},
                            p8_float={q: pf[q] for q in with_model}, info=info)
        b.save(d)
        atomic_write_json(d / "train_info.json", info)
        print(f"[fit] {fold}/{s}: {len(with_model)} hit models ({sum(1 for q in with_model if hit[q].n_trees)} trees, "
              f"rest constant), gmal {'lgbm' if gmal.n_trees else 'constant'}", flush=True)


def atomic_save_npz_uncompressed(path: Path, **arrays) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------ #
# stages C/D: calibration and application
# ------------------------------------------------------------------------------------------ #
def _stream_inputs(sid: str):
    from ctsample.sim.tables import StreamTables
    import polars as pl
    from ctsample.units import features as F
    d = STREAMS / sid
    T = StreamTables.load(d)
    pay = np.asarray(T.pay, dtype=bool)
    Xp = np.load(d / "features_pay.npy")
    # metadata columns of the payload units: from units.parquet (test streams) or re-derived from the tables
    return T, pay, Xp


def _cols_pay(sid: str, T, pay):
    """Canonical columns of the stream's payload units (from units.parquet when present, else from the
    source metadata via the stream recipe; training streams store the payload cols cache)."""
    from ctsample.units import features as F
    import polars as pl
    d = STREAMS / sid
    p = d / "cols_pay.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    if (d / "units.parquet").exists():
        u = pl.read_parquet(d / "units.parquet", columns=["pay", "meta"])
        idx = np.nonzero(u["pay"].to_numpy())[0]
        return F.cols_from_meta(u["meta"].to_arrow().take(idx))
    raise FileNotFoundError(f"{sid}: no cols_pay.npz / units.parquet")


def _f_true(T, queries, pay_idx):
    out = {}
    for q in queries:
        if q in T.query_names:
            out[q] = T.f_dense(q)[pay_idx].astype(np.int64)
        else:
            out[q] = np.zeros(len(pay_idx), np.int64)
    return out


def utilities_for_stream(b, T, pay, Xp, cp, active, W, Q2, which):
    """Integer utilities of the payload units -> dict name -> float array over ALL units (â 2^-16)."""
    from ctsample.design import utility as U
    pay_idx = np.nonzero(pay)[0]
    cp = dict(cp)
    cp["pay"] = np.ones(len(pay_idx), bool)
    ft = _f_true(T, active, pay_idx) if "U7" in which else None
    u = b.utilities(cp, Xp, active, W, Q2, f_true=ft, which=which)
    out = {}
    for k in which:
        a = np.zeros(T.n_units)
        a[pay_idx] = u[k] / 65536.0
        out[k] = a
    out["_G"] = u.get("_G_U5")
    return out


def calibrate(fold: str, scen: str | None, seed: int, p_list, b, cfg, inv) -> dict:
    """Q̃, φ0, λ_ref (static and adaptive), V per epoch, pooled over the fold's TRAINING LP streams
    (one per training scenario: CIC training half + that scenario, the test recipe). The forests are evaluated
    once per distinct (feature row, eligibility) over every training stream of the fold (all p): the CIC
    background rows are identical in all of them."""
    from ctsample.design.controller import calibrate_lambda
    from ctsample.design.floor import phi0_from_training
    from ctsample.design.probs import design_pi
    from ctsample.design import utility as U
    from ctsample.streams.recipe import p_ppm, training_stream_ids
    from ctsample.units import features as F
    sids_by_p = {p: training_stream_ids(cfg, inv, scen, p, VARIANT) for p in p_list}
    sids = sorted({x for v in sids_by_p.values() for x in v})
    ST, Xs, Cs, off = {}, [], [], 0
    for sid in sids:
        T, pay, Xp = _stream_inputs(sid)
        cp = dict(_cols_pay(sid, T, pay))
        cp["pay"] = np.ones(len(Xp), bool)
        ST[sid] = {"T": T, "pay": pay, "pidx": np.nonzero(pay)[0], "rows": np.arange(off, off + len(Xp))}
        off += len(Xp)
        Xs.append(Xp)
        Cs.append(cp)
    Xcat = np.concatenate(Xs)
    Ccat = F.concat_cols(Cs)
    del Xs, Cs
    P_cat, pmal_cat = U.predict_all(b, Xcat, U.eligibility_matrix(b.queries, Ccat))
    Qn, ph, lam, grid, boost = {}, {}, {}, {}, {}
    for p in p_list:
        key = f"p{p_ppm(p)}"
        sp = sids_by_p[p]
        Ts = [ST[x]["T"] for x in sp]
        E_tot = sum(T.n_epochs for T in Ts)
        Qt = {q: sum(T.query_total(q) for T in Ts if q in T.query_names) / E_tot for q in b.queries}
        allq = sorted({q for T in Ts for q in T.query_names})
        active = [q for q in b.queries if Qt[q] > 0]
        W = U.equal_weights(len(active))
        Q2 = U.q2_fix([Qt[q] for q in active])
        Qn[key] = {"streams": sp, "n_epochs": E_tot, "Qt": Qt, "active": active, "W": W, "Q2": [int(x) for x in Q2],
                   "Qt_all": {q: sum(T.query_total(q) for T in Ts if q in T.query_names) / E_tot for q in allq}}
        bt = U.boost_trigger_set(Ts)                      # pre-registered training-data rule for F
        rows = np.concatenate([ST[x]["rows"] for x in sp])
        cr = F.take_cols(Ccat, rows)
        ft = {q: np.concatenate([(ST[x]["T"].f_dense(q)[ST[x]["pidx"]] if q in ST[x]["T"].query_names
                                  else np.zeros(len(ST[x]["pidx"]))).astype(np.int64) for x in sp]) for q in active}
        us = U.utilities_from_P(b, cr, P_cat[rows], pmal_cat[rows], active, W, Q2, f_true=ft)
        A = {k: us[k] / 65536.0 for k in ("U3", "U5", "U8", "U7")}
        S, DEV, EP, TRIG, HITS = [], [], [], [], []
        e_off = d_off = n_off = 0
        s_all_sum = s_pay_sum = 0.0
        for x in sp:
            T, pidx = ST[x]["T"], ST[x]["pidx"]
            s_all = np.asarray(T.s, dtype=float)
            s_all_sum += float(s_all.sum())
            s_pay_sum += float(s_all[pidx].sum())
            S.append(s_all[pidx])
            DEV.append(np.asarray(T.device)[pidx].astype(np.int64) + d_off)
            EP.append(np.asarray(T.epoch)[pidx].astype(np.int64) + e_off)
            trig = np.zeros(T.n_units, bool)
            for q in bt["queries"]:
                if q in T.query_names:
                    trig[T.hits(q)[0]] = True
            TRIG.append(trig[pidx])
            pos = np.full(T.n_units, -1, np.int64)
            pos[pidx] = np.arange(len(pidx)) + n_off
            HITS.append({q: (pos[T.hits(q)[0]], T.hits(q)[1]) for q in T.query_names})
            e_off += T.n_epochs
            d_off += T.n_devices
            n_off += len(pidx)
        s_pay = np.concatenate(S)
        dev_pay, ep_pay, trig_pay = np.concatenate(DEV), np.concatenate(EP), np.concatenate(TRIG)
        ph[key] = {"inputs": {"sum_s_all": s_all_sum, "sum_s_pay": s_pay_sum, "gamma": GAMMA, "streams": sp,
                              "n_epochs": E_tot}}
        lam[key], grid[key] = {}, {}
        adapt_cfgs, adapt_keys = [], []
        for rho in RHOS:
            rk = f"{rho:g}"
            phi0 = phi0_from_training(GAMMA, rho, s_all_sum, s_pay_sum)
            Bbar = rho * s_all_sum / E_tot
            ph[key][rk] = phi0
            lam[key][rk], grid[key][rk] = {}, {}
            for dn, (kind, ukey, tau) in DESIGNS.items():
                a = np.ones(len(s_pay)) if ukey is None else A[ukey]
                comps = {"mixture": ("A", "S"), "A": ("A",), "S": ("S",), "uniform": ("uniform",)}[kind]
                lr, status = [], []
                for cpn in comps:
                    l, ok = calibrate_lambda(s_pay, a, Bbar, phi0, E_tot, component=cpn, iters=64, return_status=True)
                    lr.append(l)
                    status.append(bool(ok))
                lam[key][rk][dn] = {"lam_ref": lr, "kind": kind, "util": ukey, "tau_mix": tau, "Bbar": Bbar,
                                    "phi0": phi0, "calibrated": status}
                if dn in ADAPTIVE:
                    adapt_cfgs.append({"lam_A": lr[0], "lam_S": lr[1] if kind == "mixture" else None,
                                       "tau_mix": tau, "phi0": phi0, "Bbar": Bbar})
                    adapt_keys.append((rk, dn))
                pi = design_pi(kind, lr, a, s_pay, phi0, tau)
                V = {}
                for q in allq:
                    num = 0.0
                    for H in HITS:
                        if q in H and len(H[q][0]):
                            num += float(np.sum(H[q][1] ** 2 / np.maximum(pi[H[q][0]], 1e-300)))
                    V[q] = num / E_tot
                grid[key][rk][dn] = {"V_per_epoch": V, "V_ref_epochs": V_REF_EPOCHS,
                                     "eta_star": {q: (math.sqrt(2 * math.log(1 / ALPHA_SIDE) / (v * V_REF_EPOCHS))
                                                      if v > 0 else None) for q, v in V.items()}}
        res = U.calibrate_lambda_adaptive(s_pay, A["U5"], dev_pay, ep_pay, trig_pay, E_tot, adapt_cfgs,
                                          kappa_b=KAPPA_B, W_b=W_B, R=ADAPT_R, seed=seed)
        for (rk, dn), (lA, info) in zip(adapt_keys, res):
            ent = lam[key][rk][dn]
            ent["lamref_adaptive"] = [lA] + ent["lam_ref"][1:]
            ent["adaptive_info"] = info
            ent["boost_queries"] = list(bt["queries"])
            ent["boost_detectors"] = list(bt["F"])
        boost[key] = bt
        print(f"[calib] {fold}/{seed} {key}: active {len(active)}, phi0(5%) {ph[key]['0.05']:.5f}, "
              f"boost F {bt['F']}", flush=True)
    return {"Qnorm": Qn, "phi0": ph, "lamref": lam, "grid": grid, "boost": boost}


def _ks_shift(Xa: np.ndarray, Xb: np.ndarray, n_max: int = 200_000, seed: int = 0) -> dict:
    from scipy.stats import ks_2samp
    from ctsample.units.features import FEATURE_NAMES
    rng = np.random.default_rng(seed)
    a = Xa if len(Xa) <= n_max else Xa[rng.choice(len(Xa), n_max, replace=False)]
    b = Xb if len(Xb) <= n_max else Xb[rng.choice(len(Xb), n_max, replace=False)]
    out = {}
    for j, nm in enumerate(FEATURE_NAMES):
        r = ks_2samp(np.asarray(a[:, j], dtype=np.float64), np.asarray(b[:, j], dtype=np.float64))
        out[nm] = float(r.statistic)
    return out


def apply_to_tests(fold: str, seed: int, b, Qn: dict, test_ids: list[tuple[str, str]], train_rows: set,
                   train_X: np.ndarray) -> dict:
    """Utilities, AUC, duplicate rows and KS shift for the fold's test streams; test_ids = [(stream_id, p_key)]."""
    from ctsample.design import utility as U
    from ctsample.streams.lp import add_utilities
    aucs, dups, ks = {}, {}, {}
    for sid, key in test_ids:
        if not (STREAMS / sid / "DONE").exists():
            print(f"[apply] {fold}/{seed}: {sid} not built, skipped", flush=True)
            continue
        T, pay, Xp = _stream_inputs(sid)
        cp = _cols_pay(sid, T, pay)
        q = Qn[key]
        active, W, Q2 = q["active"], q["W"], q["Q2"]
        pay_idx = np.nonzero(pay)[0]
        cpa = dict(cp)
        cpa["pay"] = np.ones(len(pay_idx), bool)
        el = U.eligibility_matrix(b.queries, cpa)
        P, pmal = U.predict_all(b, Xp, el)
        u = U.utilities_from_P(b, cpa, P, pmal, active, W, Q2, f_true=_f_true(T, active, pay_idx))
        us = {}
        for k in ("U3", "U5", "U8", "U7"):
            arr = np.zeros(T.n_units)
            arr[pay_idx] = u[k] / 65536.0
            us[k] = arr
        add_utilities(STREAMS / sid, {f"U3_s{seed}": us["U3"], f"U5_s{seed}": us["U5"], "U8": us["U8"],
                                      "U7": us["U7"]})
        # metadata predictability: AUC of ĝ_q on the eligible payload units (all queries with a model)
        G = U.ghat_int(P, b.M_cells(), U.cell_index(cpa))
        a = {}
        for j, qq in enumerate(b.queries):
            if qq not in T.query_names:
                continue
            m = el[:, j]
            y = T.f_dense(qq)[pay_idx][m] > 0
            a[qq] = {"auc": U.auc(G[m, j], y), "n_elig": int(m.sum()), "n_pos": int(y.sum())}
        lab = np.asarray(T.label)[pay_idx]
        known = lab >= 0
        a["g_mal"] = {"auc": U.auc(us["U3"][pay_idx][known], lab[known] > 0), "n_elig": int(known.sum()),
                      "n_pos": int((lab[known] > 0).sum())}
        aucs[sid] = a
        # per-feature train/test shift (two-sample KS statistic on the payload units' features)
        ks[sid] = _ks_shift(train_X, Xp)
        rows = {r.tobytes() for r in np.ascontiguousarray(Xp)}
        n_dup_rows = sum(1 for r in np.ascontiguousarray(Xp) if r.tobytes() in train_rows)
        dups[sid] = {"n_test_payload_units": int(len(Xp)), "n_in_training": int(n_dup_rows),
                     "frac_in_training": n_dup_rows / max(len(Xp), 1), "n_distinct_test_rows": len(rows),
                     "n_distinct_in_training": int(sum(1 for r in rows if r in train_rows))}
    return {"auc": aucs, "dups": dups, "ks": ks}


def stage_calibrate_apply(fold: str, scen: str | None, seeds: list[int], cfg, inv) -> None:
    from ctsample.design.utility import UtilityBundle
    from ctsample.streams.recipe import p_ppm, stream_name, variant_stream_id
    ps = [float(x) for x in cfg["prevalences"]] if PREVALENCES is None else list(PREVALENCES)
    if VARIANT:                       # recipe variant: its LP-IoT23 test streams only (no natural / zero-shot)
        if scen is None:
            raise ValueError("recipe variants are defined for the LOSO folds only")
        tests = [(variant_stream_id(stream_name("lp_iot23", scenario=scen, p=p), VARIANT), f"p{p_ppm(p)}") for p in ps]
    elif scen is None:
        tests = [(stream_name("lp_edge", p=p), f"p{p_ppm(p)}") for p in ps]
        if cfg.get("lp_gotham", {}).get("enabled", False):
            tests += [(stream_name("lp_gotham", p=p), f"p{p_ppm(p if p is not None else 0.01)}")
                      for p in cfg["lp_gotham"].get("prevalences", [None, 0.01])]
    else:
        tests = [(stream_name("lp_iot23", scenario=scen, p=p), f"p{p_ppm(p)}") for p in ps]
        tests.append((stream_name("natural", scenario=scen), f"p{p_ppm(0.05)}"))
    train_rows = train_X = None
    for s in seeds:
        d = MODELS / fold / str(s)
        if is_done(d) and not FORCE_CALIB:
            continue
        from ctsample.streams.recipe import training_stream_ids
        missing = [sid for p in ps for sid in training_stream_ids(cfg, inv, scen, p, VARIANT)
                   if not (STREAMS / sid / "DONE").exists()]
        if missing:
            print(f"[calib] {fold}/{s}: training streams missing {missing}; later", flush=True)
            return
        b = UtilityBundle.load(d, attach_lgbm=True)
        t0 = time.time()
        cal = calibrate(fold, scen, s, ps, b, cfg, inv)
        atomic_write_json(d / "Qnorm.json", cal["Qnorm"])
        atomic_write_json(d / "phi0.json", cal["phi0"])
        atomic_write_json(d / "lamref.json", cal["lamref"])
        atomic_write_json(d / "grid.json", cal["grid"])
        atomic_write_json(d / "boost.json", cal["boost"])
        if train_rows is None:
            z = np.load(MODELS / fold / "_train.npz", mmap_mode="r")
            train_X = np.ascontiguousarray(z["X"])
            train_rows = {r.tobytes() for r in train_X}
        ap = apply_to_tests(fold, s, b, cal["Qnorm"], tests, train_rows, train_X)
        atomic_write_json(d / "auc.json", ap["auc"])
        atomic_write_json(d / "dups.json", ap["dups"])
        atomic_write_json(d / "ks.json", ap["ks"])
        built = [sid for sid, _ in tests if (STREAMS / sid / "DONE").exists()]
        if len(built) == len(tests):
            mark_done(d, {"tests": built, "s": round(time.time() - t0, 1)})
        print(f"[calib] {fold}/{s}: done in {time.time() - t0:.0f}s (tests {built})", flush=True)
        check_memory()


def _calib_task(fold: str, scen: str | None, seeds: list[int], force: bool = False) -> dict:
    global FORCE_CALIB
    FORCE_CALIB = bool(force)
    try:
        from ctsample.streams.recipe import load_config, load_inventory
        t0 = time.time()
        stage_calibrate_apply(fold, scen, seeds, load_config(), load_inventory())
        return {"fold": fold, "s": round(time.time() - t0, 1),
                "done": [s for s in seeds if is_done(MODELS / fold / str(s))]}
    except MemoryBudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        return {"error": f"calib {fold}: {type(e).__name__}: {e}", "traceback": traceback.format_exc()}


# ------------------------------------------------------------------------------------------ #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all", help="'all' (6 LOSO folds + zero-shot) or comma list, e.g. 34-1,all")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--stages", default="A,B,C")
    ap.add_argument("--force-calib", action="store_true", help="redo stage C even where DONE exists")
    ap.add_argument("--variant", default=None, help="recipe variant (config zero_payload_variants), e.g. zq, z0: "
                                                    "everything under results/models_<variant>/")
    a = ap.parse_args()
    _set_variant(a.variant)
    from ctsample.streams.recipe import load_config, load_inventory, lp_folds
    cfg, inv = load_config(), load_inventory()
    folds = lp_folds(cfg, inv)
    if a.folds == "all":
        sel = folds + [None]
    else:
        names = a.folds.split(",")
        sel = [s for s in folds if fold_name(s) in names] + ([None] if "zs" in names or "all" in names else [])
    if VARIANT:
        if None in sel:
            print(f"[s4] variant {VARIANT}: the zero-shot fold is out of scope, skipped", flush=True)
        sel = [s for s in sel if s is not None]
        print(f"[s4] variant {VARIANT}: models root {MODELS}, prevalences {PREVALENCES}, training-source zero-payload "
              f"keep {TRAIN_ZERO_KEEP}", flush=True)
    seeds = list(range(a.seeds))
    nw = max(1, min(a.workers, workers(default=a.workers)))
    stages = set(a.stages.split(","))
    t0 = time.time()
    try:
        if "A" in stages:
            need = {}
            for sc in sel:
                for s in training_sources(inv, sc, folds):
                    need[(s["ds"], s["source"])] = s
            todo = [(build_cache, (ds, src)) for (ds, src) in sorted(need) if not _cache_path(ds, src).exists()]
            print(f"[cache] {len(need)} sources, {len(todo)} to featurise ({min(nw, 2)} workers)", flush=True)
            for r in _run_pool(todo, min(nw, 2), "cache"):
                print(f"[cache] {r}", flush=True)
        if "B" in stages:
            for sc in sel:
                fn = fold_name(sc)
                stage_models(fn, seeds, training_sources(inv, sc, folds), nw)
                check_memory()
        if "C" in stages:
            tasks = [(_calib_task, (fold_name(sc), sc, seeds, a.force_calib)) for sc in sel]
            for r in _run_pool(tasks, min(nw, 4), "calib"):
                print(f"[calib] {r}", flush=True)
    except MemoryBudgetExceeded as e:
        print(f"[mem] {e}; exiting {EXIT_MEMORY}", flush=True)
        return EXIT_MEMORY
    print(f"[s4] finished in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
