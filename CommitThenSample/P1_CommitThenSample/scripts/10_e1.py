"""E1: evidence retained at equal storage (DESIGN §6.1 E1, §6.2 baselines, §5.5 incidents).

Per distinct test stream (LP-IoT23 per fold x realised prevalence, LP-Edge) and ρ ∈ {1, 2, 5, 10}%:
  analytic job   every static / deterministic policy (budget-matched CtS designs CtS-AI-static = U6tau0.2_bm,
                 CtS-Size = B10_bm, CtS-Unif = B8_bm, B13; baselines B2, B3, B4, B6, B15; seeds 0-2 where the
                 policy uses the utility model): per incident presence and content-evidence recall
                 (1 − Π(1 − π) or the deterministic mask), median time to evidence, retained bytes by label,
                 spend/target and max |D|/B̄ of the per-epoch spend, bias / RMSE of the natural estimator per
                 evaluated estimand (≥ 50 eligible units).
  adaptive job   CtS-AI as deployed (U6τ: realised-cost controller + boost, P1-D's adaptive λ_ref^A and trigger
                 set), 200 reps of an exact full-unit simulation (experiments_core.AdaptiveFullSim: a uniform for
                 every payload unit), per seed.
Outputs: results/E1/<stream>/<cfg_hash>/{inc,pol,est}.parquet (analytic) or rep<a>_<b>_{runs,inc,est}.parquet
(adaptive) + config.json + DONE.

    .venv/bin/python memguard/memguard.py run --name p1_e1 --workers 16 -- \
        .venv/bin/python -I -u P1_CommitThenSample/scripts/10_e1.py --workers 12
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctsample.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory, workers  # noqa: E402

P1 = Path(__file__).resolve().parents[1]
CFG_PATH = P1 / "config" / "experiments" / "E1.yaml"
EXP_ROOT = "E1"            # --exp-root E1_rev1 (with --config config/experiments/E1_rev1.yaml) for revision 1
VARIANT = ""               # --variant zq|z0: stream-recipe sensitivity streams (index_<v>.json, models_<v>/)


def _use_variant(v: str) -> None:
    """Point experiments_core at the variant streams' index and models (P1-X-SV's layout)."""
    global VARIANT
    VARIANT = v or ""
    if not VARIANT:
        return
    import json as _j
    from ctsample.sim import experiments_core as X
    from ctsample.utils.io import RESULTS
    X.load_index = lambda: _j.loads((X.STREAMS / f"index_{VARIANT}.json").read_text())["streams"]
    X.MODELS = RESULTS / f"models_{VARIANT}"
    X._read_json.cache_clear()


def _init_worker(ppid: int) -> None:
    import threading

    def watch():
        while True:
            if os.getppid() != ppid:
                os._exit(1)
            time.sleep(2)

    threading.Thread(target=watch, daemon=True).start()


def load_cfg(path=None) -> dict:
    from ctsample.config import load_yaml
    return load_yaml(path or CFG_PATH).raw


def build_jobs(raw: dict) -> tuple[list[dict], list[dict]]:
    from ctsample.sim import experiments_core as X
    kept, dropped = X.distinct_streams(X.test_streams(tuple(raw["stream_kinds_e1"])))
    jobs = []
    for st in kept:
        for rho in raw["rho"]:
            base = {"stream": st["stream_id"], "kind": st["kind"], "bin": st["bin"], "fold": st["fold_short"],
                    "realised_prevalence": st.get("realised_prevalence"), "rho": float(rho)}
            jobs.append(dict(base, job="analytic"))
            for s in raw["seeds"]:
                jobs.append(dict(base, job="adaptive", seed=int(s), reps=int(raw["mc_adaptive"]["reps"])))
    return jobs, dropped


def _settings(raw: dict) -> dict:
    return {k: raw.get(k) for k in ("policies", "mc_adaptive", "incidents", "estimands", "seeds", "policy")}


def _run_dir(job: dict, raw: dict) -> Path:
    from ctsample.config import cfg_hash
    from ctsample.utils.io import RESULTS
    return RESULTS / EXP_ROOT / job["stream"] / cfg_hash({"exp": "E1", "job": job, "settings": _settings(raw)})


def _policy_arrays(sid: str, rho: float, pol: dict, seed: int | None):
    """(pi_pay, mask_full, mask_any, kept_bytes_pay, p_any) for a non-adaptive policy (payload order)."""
    from ctsample.sim import experiments_core as X
    T = X.load_tables(sid)
    s = np.asarray(T.s, dtype=float)[X.pay_index(T)]
    stem = pol["file"].format(s=seed)
    pi = X.design_pi(sid, stem, rho)
    if pol["kind"] == "pi":
        return pi, None, None, pi * s, pi
    if pol["kind"] == "mask":
        mf = pi > 0.5
        if pol.get("partial"):
            ma = X.design_pi(sid, stem, rho, ".mask_any") > 0
            kb = X.design_pi(sid, stem, rho, ".retained_bytes")
        else:
            ma, kb = mf, mf * s
        return mf.astype(float), mf, ma, kb, mf.astype(float)
    if pol["kind"] == "packet":
        pa = X.design_pi(sid, stem, rho, ".p_unit_any_packet")
        return pi, None, None, pi * s, pa
    if pol["kind"] == "mask_bytes":
        # rev1 byte-level B3: content evidence iff the unit's first payload packet is within the cut-off N
        from ctsample.baselines.policies import b3_evidence
        aux = np.load(X.DESIGNS / sid / "aux_b3_offsets.npz")
        N = float(X.design_meta(sid, stem, rho)["params"]["N_bytes"])
        ev = b3_evidence(N, aux["before"], aux["off_pay"])
        ma = X.design_pi(sid, stem, rho, ".mask_any") > 0
        kb = X.design_pi(sid, stem, rho, ".retained_bytes")
        return ev.astype(float), ev, ma | ev, kb, ev.astype(float)
    raise ValueError(pol["kind"])


def _b3_hit_p(T, sid: str, stem: str, rho: float, names) -> dict:
    """Per-hit retention (0/1) of byte-level B3 for the estimator: the first hit packet within the cut-off."""
    from ctsample.baselines.policies import b3_evidence
    from ctsample.sim import experiments_core as X
    aux = np.load(X.DESIGNS / sid / "aux_b3_offsets.npz")
    N = float(X.design_meta(sid, stem, rho)["params"]["N_bytes"])
    pidx = X.pay_index(T)
    upos = np.full(T.n_units, -1, dtype=np.int64)
    upos[pidx] = np.arange(len(pidx))
    out = {}
    for nm in names:
        u, _ = T.hits(T.query_index(nm))
        out[nm] = b3_evidence(N, aux["before"][upos[u]], aux[f"off_hit_{nm}"]).astype(float)
    return out


def analytic_job(job: dict, raw: dict) -> dict:
    import pandas as pd
    from ctsample.config import src_hash
    from ctsample.sim import experiments_core as X
    from ctsample.utils.io import atomic_write_json, atomic_write_parquet, mark_done
    import pyarrow as pa
    out = _run_dir(job, raw)
    if (out / "DONE").exists():
        return {"job": job, "skipped": True}
    out.mkdir(parents=True, exist_ok=True)
    sid, rho = job["stream"], job["rho"]
    T = X.load_tables(sid)
    pidx = X.pay_index(T)
    ep_pay = np.asarray(T.epoch)[pidx].astype(np.int64)
    s_all = float(np.asarray(T.s, dtype=float).sum())
    Bbar = rho * s_all / T.n_epochs
    incs = X.incident_sets(T)
    qrows = [r for r in X.evaluated_queries(T, int(raw["estimands"]["min_eligible"])) if r["Q"] > 0]
    qnames = [r["estimand"] for r in qrows]
    qd = [r["estimand"] for r in qrows if r["class"] == "D"]
    inc_rows, pol_rows, est_rows = [], [], []
    for pname, pol in raw["policies"].items():
        if pol["kind"] == "adaptive":
            continue
        for seed in (raw["seeds"] if pol.get("seeded") else [None]):
            pi, mf, ma, kb, p_any = _policy_arrays(sid, rho, pol, seed)
            if mf is not None:
                rec = X.recall_from_masks(mf, ma, incs)
                tte = X.tte_from_mask(mf, ep_pay, incs)
            else:
                rec = X.recall_from_pi(p_any, incs)
                tte = X.tte_from_pi(p_any, ep_pay, incs)
            for d, r, t in zip(incs, rec, tte):
                inc_rows.append({"policy": pname, "seed": -1 if seed is None else seed, "incident": d["incident"],
                                 "k": d["k"], "stratum": d["stratum"], "family": d["family"], "n_presence": len(d["presence_pos"]),
                                 "first_epoch": d["first_epoch"], "last_epoch": d["last_epoch"],
                                 "content_recall": r["content_recall"], "presence_recall": r["presence_recall"],
                                 "tte_median_epochs": t})
            bb = X.benign_bytes(T, kb)
            sp = X.spend_stats(T, kb, Bbar)
            meta = X.design_meta(sid, pol["file"].format(s=seed), rho)
            pol_rows.append({"policy": pname, "seed": -1 if seed is None else seed, "benign_bytes": bb["benign"],
                             "malicious_bytes": bb["malicious"], "unknown_bytes": bb["unknown"], "total_bytes": bb["total"],
                             "Bbar": Bbar, "budget_bytes": Bbar * T.n_epochs, "spend_over_target": sp["spend_over_target"],
                             "max_abs_D_over_Bbar": sp["max_abs_D_over_Bbar"],
                             "spend_over_target_json": meta.get("spend_over_target"), "saturated": json.dumps(meta.get("saturated"))})
            est = pol.get("estimator", "none")
            if est == "none":
                continue
            names = qd if pol.get("d_only") else qnames
            variants = [("natural", False)] + ([("scaled", True)] if pol.get("scaled_variant") else [])
            hit_p = None
            if est == "sum_bytes":
                hit_p = _b3_hit_p(T, sid, pol["file"].format(s=seed), rho, names)
                est = "sum"
            for vname, scale in variants:
                rows = X.natural_estimator_stats(T, names, est, pi_pay=pi, scale_fraction=scale, hit_p=hit_p)
                for r in rows:
                    r.update({"policy": pname, "seed": -1 if seed is None else seed, "variant": vname,
                              "rel_bias": r["bias"] / r["Q"] if r["Q"] > 0 else float("nan"),
                              "rel_rmse": r["rmse"] / r["Q"] if r["Q"] > 0 else float("nan"),
                              "cls": X.query_class(r["estimand"])})
                    est_rows.append(r)
    for nm, rows in (("inc", inc_rows), ("pol", pol_rows), ("est", est_rows)):
        atomic_write_parquet(out / f"{nm}.parquet", pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False))
    atomic_write_json(out / "config.json", {"exp": "E1", "job": job, "settings": _settings(raw), "src_hash": src_hash(),
                                            "n_incidents": len(incs), "estimands": qrows, "Bbar": Bbar,
                                            "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    mark_done(out, {"n_inc_rows": len(inc_rows)})
    return {"job": job, "out": str(out)}


def adaptive_job(job: dict, raw: dict) -> dict:
    import pandas as pd
    import pyarrow as pa
    from ctsample.config import src_hash
    from ctsample.sim import experiments_core as X
    from ctsample.utils.io import atomic_write_json, atomic_write_parquet, mark_done
    out = _run_dir(job, raw)
    if (out / "DONE").exists():
        return {"job": job, "skipped": True}
    out.mkdir(parents=True, exist_ok=True)
    sid, rho, seed = job["stream"], job["rho"], job["seed"]
    pol = raw["policies"]["CtS-AI"]
    T = X.load_tables(sid)
    meta = X.design_meta(sid, pol["source"].format(s=seed), rho)
    fold, sd, pkey = X.calibration_of(meta)
    lam = X.lamref_entry(fold, sd, pkey, rho, pol["grid_key"])
    spec = X.adaptive_spec(meta, lam, gamma=float(raw["policy"]["gamma"]))
    incs = X.incident_sets(T)
    qrows = [r for r in X.evaluated_queries(T, int(raw["estimands"]["min_eligible"])) if r["Q"] > 0]
    qnames = [r["estimand"] for r in qrows]
    atomic_write_json(out / "config.json", {"exp": "E1", "job": job, "settings": _settings(raw), "design": asdict(spec),
                                            "src_hash": src_hash(), "estimands": qrows, "n_incidents": len(incs),
                                            "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    sim = X.AdaptiveFullSim(T, spec, queries=qnames, incs=incs)
    chunk = int(raw["mc_adaptive"]["chunk"])
    Q = {r["estimand"]: r["Q"] for r in qrows}
    for a in range(0, int(job["reps"]), chunk):
        b = min(a + chunk, int(job["reps"]))
        tag = f"rep{a:06d}_{b:06d}"
        if (out / f"{tag}_runs.parquet").exists():
            continue
        reps = np.arange(a, b)
        r = sim.run(reps, int(T.stream_id), chunk=chunk)
        R = len(reps)
        runs = {"rep": reps, **{k: r[k] for k in ("spend_ratio", "max_abs_D_over_B", "boosted_frac", "benign_bytes",
                                                    "malicious_bytes", "total_bytes")}}
        nI = len(incs)
        inc = {"rep": np.repeat(reps, nI), "incident": np.tile([d["incident"] for d in incs], R),
               "k": np.tile([d["k"] for d in incs], R), "content": r["content"].ravel(), "presence": r["presence"].ravel(),
               "tte": r["tte"].ravel()}
        nq = len(qnames)
        est = {"rep": np.repeat(reps, nq), "estimand": np.tile(np.asarray(qnames, dtype=object), R).astype(str),
               "Qhat": r["Qhat"].ravel(), "Q": np.tile([Q[q] for q in qnames], R)}
        for nm, dct in (("est", est), ("inc", inc), ("runs", runs)):      # runs last = chunk marker
            atomic_write_parquet(out / f"{tag}_{nm}.parquet", pa.Table.from_pandas(pd.DataFrame(dct), preserve_index=False))
        check_memory()
    mark_done(out, {"reps": int(job["reps"])})
    return {"job": job, "out": str(out)}


def run_job(job: dict, raw: dict, exp_root: str = "E1", variant: str = "") -> dict:
    global EXP_ROOT
    EXP_ROOT = exp_root                     # spawned workers re-import the module: pass the root explicitly
    _use_variant(variant)
    t0 = time.time()
    try:
        r = analytic_job(job, raw) if job["job"] == "analytic" else adaptive_job(job, raw)
        r["seconds"] = time.time() - t0
        return r
    except MemoryBudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        return {"job": job, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", default=None)
    ap.add_argument("--jobs", default="analytic,adaptive")
    ap.add_argument("--rho", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default=None, help="experiment YAML (default config/experiments/E1.yaml)")
    ap.add_argument("--exp-root", default="E1", help="results/<exp-root>/ (E1_rev1 for revision 1)")
    ap.add_argument("--variant", default="", help="zq / z0 stream-recipe sensitivity streams")
    a = ap.parse_args()
    global EXP_ROOT, CFG_PATH
    EXP_ROOT = a.exp_root
    _use_variant(a.variant)
    if a.config:
        CFG_PATH = Path(a.config).resolve()
    raw = load_cfg()
    jobs, dropped = build_jobs(raw)
    jobs = [j for j in jobs if j["job"] in a.jobs.split(",")]
    if a.only:
        jobs = [j for j in jobs if any(k in j["stream"] for k in a.only.split(","))]
    if a.rho:
        rs = [float(x) for x in a.rho.split(",")]
        jobs = [j for j in jobs if any(abs(j["rho"] - r) < 1e-12 for r in rs)]
    jobs.sort(key=lambda j: (j["job"] != "adaptive", j["kind"] != "lp_edge"))
    print(f"[e1] {len(jobs)} jobs; duplicates skipped: {[(d['stream_id'], d['duplicate_of']) for d in dropped]}", flush=True)
    if a.dry_run:
        for j in jobs:
            print(json.dumps(j))
        return 0
    nw = max(1, min(a.workers, workers(default=a.workers)))
    t0 = time.time()
    failed, done = [], 0
    ex = ProcessPoolExecutor(max_workers=nw, mp_context=get_context("spawn"), initializer=_init_worker,
                             initargs=(os.getpid(),))
    try:
        pend = {ex.submit(run_job, j, raw, EXP_ROOT, VARIANT): j for j in jobs}
        while pend:
            fin, _ = wait(set(pend), timeout=30, return_when=FIRST_COMPLETED)
            for f in fin:
                j = pend.pop(f)
                r = f.result()
                done += 1
                tag = f"{j['job']}/{j['stream']}/rho={j['rho']:g}" + (f"/s{j['seed']}" if "seed" in j else "")
                if r.get("error"):
                    failed.append(tag)
                    print(f"[e1] FAILED {tag}: {r['error']}\n{r['traceback']}", flush=True)
                else:
                    print(f"[e1] {'skip' if r.get('skipped') else 'done'} {tag} ({r.get('seconds', 0):.0f}s) "
                          f"[{done}/{len(jobs)}, {time.time() - t0:.0f}s]", flush=True)
            check_memory()
    except MemoryBudgetExceeded as e:
        print(f"[mem] {e}; exiting {EXIT_MEMORY}", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        return EXIT_MEMORY
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    print(f"[e1] {done - len(failed)} ok, {len(failed)} failed in {time.time() - t0:.0f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
