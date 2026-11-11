"""S5b (rev1): faithful / online baselines next to the existing designs (review R3-M2, fix list B4 and M6).

For every LP test stream of E1 and E4 (LP-IoT23 all folds and prevalences, LP-Edge, LP-Gotham) and
ρ ∈ {1, 2, 5, 10}% this adds NEW files to results/designs/<stream>/ (nothing existing is overwritten):
  B4on_s<k>_<ρ>      B4 with the online horizon threshold controller, θ_ref calibrated on the fold's TRAINING
                     streams (pooled spend = pooled budget), seed k of the g_mal model; mask; spend reported
  B13on_s<k>_<ρ>     B13 online: per epoch top part (controller, share 1 − γ, training θ_ref) with π = 1, keyed
                     reserve π_r,e = max(φ0, min(1, γB̄/Σ_rest,e s)) (φ0 = the committed CtS floor of the fold);
                     sidecars .reserve_pi.npy (the reserve rate of every payload unit's epoch) and .mask_top.npy
  B6s_<ρ>            B6 triggered ONLY by the 36 pre-registered strict-audit sids (rules/audit_all.json kept_sids)
  B4ep_s<k>_<ρ>      B4 online and budget-exact: per-epoch greedy top-k by P_mal within B̄ + carried deficit
  B13ep_s<k>_<ρ>     B13 online and budget-exact (in expectation): per-epoch top part within (1 − γ)(B̄ + D) +
                     reserve π_r,e = max(φ0, min(1, (B_e − top)/Σ_rest s)); sidecars as B13on
  DLT_<ρ>            Duffield–Lund–Thorup threshold sampling π = min(1, s/z), budget-matched (no floor)
  aux_b3_offsets.npz byte-level B3 scoring inputs (independent of ρ; the cut-off N of each ρ is in B3_<ρ>.json):
                     before = connection bytes before each payload unit, off_pay = bytes of the unit up to and
                     including its first payload-carrying packet, off_hit_<q> = bytes up to its first hit packet
                     for every detector query (aligned with StreamTables.hits(q)); Suricata (S*) and label (L*)
                     estimands use off_pay (an upper bound on B3's retention, i.e. favourable to B3)
Calibrations: results/baselines_rev1/calib/<fold>_<seed>.json (θ_ref per p_key, ρ, policy).

    .venv/bin/python memguard/memguard.py run --name p1_rev_s5b --workers 6 -- \\
        .venv/bin/python -I -u P1_CommitThenSample/scripts/05b_baselines_rev1.py --workers 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctsample.utils.io import RESULTS, atomic_save_npy, atomic_write_json  # noqa: E402
from ctsample.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory, workers  # noqa: E402

RHOS = (0.01, 0.02, 0.05, 0.10)
GAMMA = 0.3
STREAMS = RESULTS / "streams"
MODELS = RESULTS / "models"
DESIGNS = RESULTS / "designs"
CALIB = RESULTS / "baselines_rev1" / "calib"
SCENARIOS = ("1-1", "3-1", "7-1", "34-1", "48-1", "49-1")
SEEDS = (0, 1, 2)
TAG = "rev1-baselines-v1"
VARIANT = ""              # --variant zq|z0: stream-recipe sensitivity streams (results/streams/index_<v>.json,
                          # models results/models_<v>/, training streams trainlp_<j>_<p>_<v>)


def _init_worker(ppid: int) -> None:
    import os
    import threading

    def watch():
        while True:
            if os.getppid() != ppid:
                os._exit(1)
            time.sleep(2)

    threading.Thread(target=watch, daemon=True).start()


def stream_jobs() -> list[tuple[str, str, str]]:
    """(stream id, fold, p_key) for every LP stream of E1/E4 (the E4 list is a superset of E1's); with a variant, the
    LP-IoT23 test streams of results/streams/index_<variant>.json."""
    if VARIANT:
        idx = json.loads((STREAMS / f"index_{VARIANT}.json").read_text())["streams"]
        return [(r["stream_id"], str(r["fold"]).replace("CTU-IoT-Malware-Capture-", ""), f"p{int(round(r['p'] * 1e6))}")
                for r in idx if r["kind"] == "lp_iot23"]
    out = []
    for fold in SCENARIOS:
        for ppm in (1000, 10000, 50000):
            out.append((f"lp_iot23_{fold}_p{ppm}", fold, f"p{ppm}"))
    for ppm in (1000, 10000, 50000):
        out.append((f"lp_edge_p{ppm}", "all", f"p{ppm}"))
    out += [("lp_gotham_p10000", "all", "p10000"), ("lp_gotham_nat", "all", "p10000")]
    return out


# ------------------------------------------------------------------------------------------- calibration
def calibrate(fold: str, seed: int) -> dict:
    """θ_ref (logit) of B4 (share 1) and B13 (share 1 − γ) per p_key and ρ on the fold's training LP streams."""
    from ctsample.baselines import policies as B
    from ctsample.design.utility import UtilityBundle, utility_int
    path = CALIB / f"{fold}_{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    b = UtilityBundle.load(MODELS / fold / str(seed), attach_lgbm=True)
    train = [j for j in SCENARIOS if j != fold]
    out = {"fold": fold, "seed": seed, "tag": TAG, "training_scenarios": train, "gamma": GAMMA, "p": {}}
    for ppm in (1000, 10000, 50000):
        pkey = f"p{ppm}"
        scores, sizes, sall = [], [], []
        for j in train:
            sd = STREAMS / (f"trainlp_{j}_{pkey}" + (f"_{VARIANT}" if VARIANT else ""))
            if not (sd / "DONE").exists():
                continue                                   # variants are built for p = 1 % only
            pay = np.load(sd / "pay.npy").astype(bool)
            s = np.load(sd / "s.npy").astype(np.float64)
            X = np.load(sd / "features_pay.npy")
            scores.append(utility_int(b.pmal_int(X)) / 65536.0)
            sizes.append(s[pay])
            sall.append(float(s.sum()))
        if not scores:
            continue
        blk = {}
        for rho in RHOS:
            budgets = [rho * x for x in sall]
            l4 = B.calibrate_threshold(scores, sizes, budgets)
            l13 = B.calibrate_threshold(scores, sizes, [(1 - GAMMA) * x for x in budgets])
            blk[f"{rho:g}"] = {"B4": l4, "B13": l13,
                               "train_spend_B4": float(sum(float(sz[B._logit(sc) >= l4].sum()) for sc, sz in zip(scores, sizes))),
                               "train_budget": float(sum(budgets))}
        out["p"][pkey] = blk
    CALIB.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, out)
    return out


# ------------------------------------------------------------------------------------------- B3 offsets
def b3_offsets(sid: str, T, before: np.ndarray) -> dict:
    """aux_b3_offsets.npz contents (payload order arrays; hit offsets aligned with T.hits(q))."""
    import polars as pl
    sd = STREAMS / sid
    pidx = np.nonzero(np.asarray(T.pay, dtype=bool))[0]
    u = pl.read_parquet(sd / "units.parquet", columns=["source", "unit_gid"])
    src = u["source"].to_numpy()[pidx]
    gid = u["unit_gid"].to_numpy()[pidx].astype(np.int64)
    cp = np.load(sd / "cols_pay.npz")
    sizes = np.asarray(cp["sizes"])
    has = sizes != 0
    jpay = np.where(has.any(axis=1), np.argmax(has, axis=1), 8)            # first payload packet (≤ 8: lower bound)
    s_pay = np.asarray(T.s, dtype=np.float64)[pidx]
    off_pay = np.full(len(pidx), np.nan)
    hit_first = {}                                                       # (gid) -> {query: first_hit_pkt}
    cum_by_gid = {}
    ds_of = {}
    inv = json.loads((RESULTS / "inventory.json").read_text())
    for ds, dd in inv["datasets"].items():
        for so in dd["sources"]:
            ds_of[so["source"]] = ds
    n_mismatch = 0
    for source in np.unique(src):
        m = src == source
        ds = ds_of[source]
        pm = np.load(RESULTS / "pktmap" / ds / f"{source}.npz")
        g = pm["unit_gid"]
        want = np.isin(g, gid[m])
        g2, j2, w2 = g[want], pm["pkt_idx"][want].astype(np.int64), pm["wirelen"][want].astype(np.int64) + 16
        o = np.lexsort((j2, g2))
        g2, j2, w2 = g2[o], j2[o], w2[o]
        cum = np.cumsum(w2)
        starts = np.r_[0, np.nonzero(g2[1:] != g2[:-1])[0] + 1]
        base = np.repeat(np.r_[0, cum[starts[1:] - 1]], np.diff(np.r_[starts, len(g2)]))
        within = cum - base                                              # bytes up to and including packet j
        ug = g2[starts]
        pos = {int(x): (int(a), int(b)) for x, a, b in zip(ug, starts, np.r_[starts[1:], len(g2)])}
        rows = np.nonzero(m)[0]
        for r in rows:
            a, b = pos.get(int(gid[r]), (0, 0))
            if b <= a:
                continue
            seg = within[a:b]
            if abs(seg[-1] - s_pay[r]) > 0.5:
                n_mismatch += 1                                          # caplen ≠ wirelen (truncated capture)
            cum_by_gid[int(gid[r])] = seg
            off_pay[r] = seg[min(int(jpay[r]), len(seg) - 1)]
        hp = RESULTS / "hits" / ds / f"{source}.parquet"
        if hp.exists():
            h = pl.read_parquet(hp, columns=["unit_gid", "query_id", "first_hit_pkt"]).filter(pl.col("unit_gid").is_in(gid[m]))
            for x, q, f in zip(h["unit_gid"].to_list(), h["query_id"].to_list(), h["first_hit_pkt"].to_list()):
                hit_first.setdefault(int(x), {})[q] = int(f)
    off_pay = np.where(np.isfinite(off_pay), off_pay, s_pay)               # no packet map: whole unit
    out = {"before": before.astype(np.float64), "off_pay": off_pay, "jpay": jpay.astype(np.int64)}
    upos = np.full(T.n_units, -1, dtype=np.int64)
    upos[pidx] = np.arange(len(pidx))
    stats = {"n_payload": int(len(pidx)), "caplen_wirelen_mismatch_units": int(n_mismatch),
             "first_payload_beyond_8": int((jpay >= 8).sum())}
    for qi, q in enumerate(T.query_names):
        uq, _ = T.hits(qi)
        pos_q = upos[uq]
        if q.startswith("D"):
            offs = np.empty(len(uq))
            for t, pq in enumerate(pos_q):
                gq = int(gid[pq])
                fh = hit_first.get(gq, {}).get(q)
                seg = cum_by_gid.get(gq)
                if fh is None or seg is None:
                    offs[t] = off_pay[pq]
                else:
                    offs[t] = seg[min(fh, len(seg) - 1)]
            out[f"off_hit_{q}"] = offs
        else:
            out[f"off_hit_{q}"] = off_pay[pos_q]
    return out, stats


# ------------------------------------------------------------------------------------------- per stream
def _save(d: Path, name: str, rho: float, pi_pay: np.ndarray, meta: dict) -> None:
    atomic_save_npy(d / f"{name}_{rho:g}.npy", np.asarray(pi_pay, dtype=np.float64))
    atomic_write_json(d / f"{name}_{rho:g}.json", meta)


def strict_severity(u, n: int) -> tuple[np.ndarray, int]:
    """Min Suricata severity per stream unit over alerts of the 36 strict-audit sids only (0 = none)."""
    import polars as pl
    kept = [int(x) for x in json.loads((RESULTS / "rules" / "audit_all.json").read_text())["kept_sids"]]
    sev = np.zeros(n, dtype=np.int64)
    tab = u.with_row_index("row").select(["row", "source", "unit_gid"])
    for src in u["source"].unique().to_list():
        for ds in ("ciciot2023_benign", "iot23", "edgeiiot", "gotham"):
            f = RESULTS / "alerts" / ds / f"{src}.parquet"
            if f.exists():
                al = pl.read_parquet(f, columns=["unit_gid", "severity", "sid"]).filter(
                    (pl.col("unit_gid") >= 0) & pl.col("sid").is_in(kept)).group_by("unit_gid").agg(pl.col("severity").min())
                j = tab.filter(pl.col("source") == src).join(al, on="unit_gid", how="inner")
                sev[j["row"].to_numpy()] = j["severity"].fill_null(3).to_numpy()
                break
    return sev, len(kept)


def run_stream(sid: str, fold: str, pkey: str, force: bool, variant: str = "") -> dict:
    set_variant(variant)                    # spawned workers re-import the module
    try:
        import importlib.util
        from ctsample.baselines import policies as B
        from ctsample.sim.tables import StreamTables
        d = DESIGNS / sid
        marker = d / "REV1_BASELINES_DONE"
        if marker.exists() and not force:
            return {"stream": sid, "skipped": True}
        if not (STREAMS / sid / "DONE").exists() or not (d / "DONE").exists():
            return {"stream": sid, "skipped": True, "reason": "stream / designs missing"}
        spec = importlib.util.spec_from_file_location("s5", Path(__file__).resolve().parent / "05_designs.py")
        s5 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(s5)
        T = StreamTables.load(STREAMS / sid)
        pay = np.asarray(T.pay, dtype=bool)
        flow, ts, u = s5._flow_codes(STREAMS / sid, T.n_units)
        sev, n_kept = strict_severity(u, T.n_units)
        cal = {k: calibrate(fold, k) for k in SEEDS}
        phi = json.loads((MODELS / fold / "0" / "phi0.json").read_text())[pkey]
        summary = []
        for rho in RHOS:
            rk = f"{rho:g}"
            phi0 = float(phi[rk])
            r6 = B.b6_alert_triggered(T, rho, sev, flow, ts)
            r6["name"] = "B6s"
            r6["params"].update({"sids": "strict audit (rules/audit_all.json kept_sids)", "n_sids": n_kept})
            rd = B.dlt_threshold(T, rho)
            res = [r6, rd]
            for k in SEEDS:
                pm = np.asarray(T.util(f"U3_s{k}"))
                c = cal[k]["p"][pkey][rk]
                r4 = B.b4_online(T, rho, pm, c["B4"])
                r4["name"] = f"B4on_s{k}"
                r13 = B.b13_online(T, rho, pm, c["B13"], phi0=phi0, gamma=GAMMA)
                r13["name"] = f"B13on_s{k}"
                r4e = B.b4_epoch(T, rho, pm)
                r4e["name"] = f"B4ep_s{k}"
                r13e = B.b13_epoch(T, rho, pm, phi0=phi0, gamma=GAMMA)
                r13e["name"] = f"B13ep_s{k}"
                res += [r4, r13, r4e, r13e]
            for r in res:
                meta = {"stream": sid, "design": r["name"], "rho": rho, "pi_over": "payload_units", "type": r["type"],
                        "spend_over_target": r["spend_ratio"], "params": r["params"], "phi0_committed": phi0,
                        "rev1": TAG, "calibration": f"baselines_rev1/calib/{fold}_*.json/{pkey}" if "on_s" in r["name"] else None}
                _save(d, r["name"], rho, r["pi"][pay], meta)
                for k2 in ("mask_triggered", "reserve_pi", "mask_top"):
                    if k2 in r:
                        atomic_save_npy(d / f"{r['name']}_{rho:g}.{k2}.npy", np.asarray(r[k2])[pay])
                summary.append({"design": r["name"], "rho": rho, "spend_over_target": r["spend_ratio"]})
            check_memory()
        before = B.conn_bytes_before(T, flow)
        aux, stats = b3_offsets(sid, T, before)
        import io
        buf = io.BytesIO()
        np.savez(buf, **aux)
        from ctsample.utils.io import atomic_write_bytes
        atomic_write_bytes(d / "aux_b3_offsets.npz", buf.getvalue())
        atomic_write_json(marker, {"tag": TAG, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "summary": summary,
                                   "b3_offsets": stats})
        return {"stream": sid, "n": len(summary), "b3": stats}
    except MemoryBudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        return {"stream": sid, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


def set_variant(v: str) -> None:
    global VARIANT, MODELS, CALIB
    VARIANT = v or ""
    if VARIANT:
        MODELS = RESULTS / f"models_{VARIANT}"
        CALIB = RESULTS / "baselines_rev1" / f"calib_{VARIANT}"


def main() -> int:
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--variant", default="", help="zq / z0 (stream-recipe sensitivity streams)")
    a = ap.parse_args()
    set_variant(a.variant)
    todo = stream_jobs()
    if a.only:
        todo = [t for t in todo if any(k in t[0] for k in a.only.split(","))]
    # calibrations first (one per fold and seed), sequentially in this process (shared by the stream workers)
    for fold in sorted({f for _, f, _ in todo}):
        for k in SEEDS:
            t0 = time.time()
            calibrate(fold, k)
            print(f"[s5b] calibration {fold}/{k} ({time.time() - t0:.0f}s)", flush=True)
    nw = max(1, min(a.workers, workers(default=a.workers)))
    t0 = time.time()
    res, failed = [], []
    ex = ProcessPoolExecutor(max_workers=nw, mp_context=get_context("spawn"), initializer=_init_worker,
                             initargs=(os.getpid(),))
    try:
        pend = {ex.submit(run_stream, sid, fold, pk, a.force, VARIANT): sid for sid, fold, pk in todo}
        while pend:
            done, _ = wait(set(pend), timeout=30, return_when=FIRST_COMPLETED)
            for f in done:
                sid = pend.pop(f)
                r = f.result()
                res.append(r)
                if r.get("error"):
                    failed.append(sid)
                    print(f"[s5b] FAILED {sid}: {r['error']}\n{r['traceback']}", flush=True)
                else:
                    print(f"[s5b] {r} ({time.time() - t0:.0f}s)", flush=True)
            check_memory()
    except MemoryBudgetExceeded as e:
        print(f"[mem] {e}; exiting {EXIT_MEMORY}", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        return EXIT_MEMORY
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    print(f"[s5b] {len(res) - len(failed)} ok, {len(failed)} failed in {time.time() - t0:.0f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
