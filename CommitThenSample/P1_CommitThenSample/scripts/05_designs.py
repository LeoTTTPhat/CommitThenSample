"""S5: static designs and budget-matched baselines per (test stream, design, ρ) (DESIGN §3.5, §6.2).

For every test stream (LP-IoT23(k, p), LP-Edge(p), LP-Gotham, natural IoT-23) and ρ ∈ {0.01, 0.02, 0.05, 0.10}:
  CtS designs: one sequential expected-cost controller pass (sim.montecarlo.static_pass, P1-C) with the fold's
  TRAINING calibration (φ0, λ_ref from results/models/<fold>/<seed>/{phi0,lamref}.json) and the test stream's
  own B̄ = ρ Σ_all s / E:
      U0 (= B8 uniform keyed), U1 (= B10 size-only), U3_s<k>, U5_s<k>, U6_s<k> (static U6 = U5: the boost needs
      the adaptive arm; documented), U6tau0.2_s<k>, U6tau0.5_s<k>, U8, U7 (oracle; headroom only)
  baselines (baselines/policies.py): B2, B3, B4_s<k> (global top-k, matched), B4ctl_s<k> (online threshold
      controller, spend reported), B6, B13_s<k>, B15 (+ B8/B10 aliases of U0/U1)
Budget-matched variants <design>_bm_<rho> (E1/E3/E4 comparisons, DESIGN §6.2): fixed λ over the stream, each
component by bisection on the stream's own metadata (sizes, payload flags, utilities; no labels/content except
the oracle U7) so that its expected spend is B̄E; 'saturated' marks streams whose payload content cannot absorb B̄E.
Outputs: results/designs/<stream_id>/<design>_<rho>.npy = π of the PAYLOAD units (stream order, i.e. index =
np.nonzero(pay); zero-payload units have π = 0), float64; <design>_<rho>.json = parameters (DesignSpec fields
for CtS designs), spend/target, max |D|/B̄, committed φ0 and the stream's own γρΣ_all/Σ_pay, share of payload
units at the floor; designs/<stream_id>/DONE when every file of the stream exists.

    .venv/bin/python memguard/memguard.py run --name p1_s5 --workers 4 -- \
        .venv/bin/python -I -u P1_CommitThenSample/scripts/05_designs.py [--only <substr>] [--force]

Recipe variants (rev1 sensitivity, review R3-M1(a)): ``--variant zq`` designs the variant test streams
lp_iot23_<k>_p<ppm>_<variant> (results/streams/index_<variant>.json) with the calibration of
results/models_<variant>/<fold>/<seed>/ (the design json's "calibration" then reads models_<variant>/...).
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

from ctsample.utils.io import RESULTS, atomic_save_npy, atomic_write_json, is_done, mark_done  # noqa: E402
from ctsample.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory, workers  # noqa: E402

RHOS = (0.01, 0.02, 0.05, 0.10)
GAMMA = 0.3
STREAMS = RESULTS / "streams"
MODELS = RESULTS / "models"
DESIGNS_DIR = RESULTS / "designs"
SEED_DESIGNS = ("U3", "U5", "U6", "U6tau0.2", "U6tau0.5")
FIXED_DESIGNS = ("U0", "U1", "U8", "U7")


def _set_variant(variant: str | None) -> None:
    """[recipe variant] calibration from results/models_<variant>/ (also called in each pool worker)."""
    global MODELS
    if variant:
        MODELS = RESULTS / f"models_{variant}"


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


def variant_test_streams(cfg, inv, variant: str):
    """[recipe variant] (stream id, fold, p key) of the variant's LP-IoT23 test streams (config zero_payload_variants)."""
    from ctsample.streams.recipe import lp_folds, p_ppm, stream_name, variant_stream_id, zero_variant_cfg
    v = zero_variant_cfg(cfg, variant)
    return [(variant_stream_id(stream_name("lp_iot23", scenario=sc, p=p), variant),
             sc.replace("CTU-IoT-Malware-Capture-", ""), f"p{p_ppm(p)}")
            for sc in lp_folds(cfg, inv) for p in v["prevalences"]]


def test_streams(cfg, inv):
    from ctsample.streams.recipe import lp_folds, p_ppm, stream_name
    out = []
    for sc in lp_folds(cfg, inv):
        fold = sc.replace("CTU-IoT-Malware-Capture-", "")
        for p in cfg["prevalences"]:
            out.append((stream_name("lp_iot23", scenario=sc, p=p), fold, f"p{p_ppm(p)}"))
        out.append((stream_name("natural", scenario=sc), fold, f"p{p_ppm(0.05)}"))
    for p in cfg["prevalences"]:
        out.append((stream_name("lp_edge", p=p), "all", f"p{p_ppm(p)}"))
    if cfg.get("lp_gotham", {}).get("enabled", False):
        for p in cfg["lp_gotham"].get("prevalences", [None, 0.01]):
            out.append((stream_name("lp_gotham", p=p), "all", f"p{p_ppm(p if p is not None else 0.01)}"))
    return out


def _save(d: Path, name: str, rho: float, pi_pay: np.ndarray, meta: dict) -> None:
    atomic_save_npy(d / f"{name}_{rho:g}.npy", np.asarray(pi_pay, dtype=np.float64))
    atomic_write_json(d / f"{name}_{rho:g}.json", meta)


def _flow_codes(stream_dir: Path, n: int):
    import polars as pl
    u = pl.read_parquet(stream_dir / "units.parquet", columns=["flow_uid_orig", "source", "first_ts_us", "unit_gid"])
    key = u.select((pl.col("source") + pl.lit("|") + pl.col("flow_uid_orig").bin.encode("hex")).alias("k"))["k"]
    codes = key.cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
    return codes, u["first_ts_us"].to_numpy(), u


def _alert_severity(u, n: int) -> np.ndarray:
    """Min Suricata severity over all alerts of each stream unit (0 = no alert), from the S2 alerts."""
    import polars as pl
    sev = np.zeros(n, dtype=np.int64)
    ds_of = {}
    for src in u["source"].unique().to_list():
        for ds in ("ciciot2023_benign", "iot23", "edgeiiot", "gotham"):
            if (RESULTS / "alerts" / ds / f"{src}.parquet").exists():
                ds_of[src] = ds
                break
    tab = u.with_row_index("row").select(["row", "source", "unit_gid"])
    for src, ds in ds_of.items():
        al = pl.read_parquet(RESULTS / "alerts" / ds / f"{src}.parquet", columns=["unit_gid", "severity"])
        al = al.filter(pl.col("unit_gid") >= 0).group_by("unit_gid").agg(pl.col("severity").min())
        j = tab.filter(pl.col("source") == src).join(al, on="unit_gid", how="inner")
        sev[j["row"].to_numpy()] = j["severity"].fill_null(3).to_numpy()
    return sev


def design_stream(sid: str, fold: str, pkey: str, seeds: list[int], force: bool) -> dict:
    try:
        from ctsample.baselines import policies as B
        from ctsample.design.controller import calibrate_lambda
        from ctsample.design.floor import phi0_from_training
        from ctsample.design.probs import design_pi
        from ctsample.sim.montecarlo import DesignSpec, stable_id, static_pass
        from ctsample.sim.tables import StreamTables
        d = DESIGNS_DIR / sid
        if is_done(d) and not force:
            return {"stream": sid, "skipped": True}
        sd = STREAMS / sid
        if not is_done(sd):
            return {"stream": sid, "skipped": True, "reason": "stream not built"}
        need = [MODELS / fold / str(s) / "DONE" for s in seeds]
        if not all(p.exists() for p in need):
            return {"stream": sid, "skipped": True, "reason": "models not calibrated"}
        T = StreamTables.load(sd)
        pay = np.asarray(T.pay, dtype=bool)
        s = np.asarray(T.s, dtype=np.float64)
        E = T.n_epochs
        sum_all, sum_pay = float(s.sum()), float(s[pay].sum())
        d.mkdir(parents=True, exist_ok=True)
        lam = {k: json.loads((MODELS / fold / str(k) / "lamref.json").read_text())[pkey] for k in seeds}
        phi = json.loads((MODELS / fold / str(seeds[0]) / "phi0.json").read_text())[pkey]
        summary = []
        flow, ts, u = _flow_codes(sd, T.n_units)
        sev = _alert_severity(u, T.n_units)
        cp = np.load(sd / "cols_pay.npz")
        n_pk = np.zeros(T.n_units)
        n_pk[pay] = cp["pk0"] + cp["pk1"]
        for rho in RHOS:
            rk = f"{rho:g}"
            Bbar = rho * sum_all / E
            phi0 = float(phi[rk])
            phi0_stream = phi0_from_training(GAMMA, rho, sum_all, sum_pay)
            jobs = [(dn, None) for dn in FIXED_DESIGNS] + [(dn, k) for dn in SEED_DESIGNS for k in seeds]
            for dn, k in jobs:
                cal = lam[seeds[0] if k is None else k][rk][dn]
                util = None if cal["util"] is None else (cal["util"] if cal["util"] in ("U8", "U7")
                                                          else f"{cal['util']}_s{k}")
                name = dn if k is None else f"{dn}_s{k}"
                if util is not None and util not in T.utils:
                    raise RuntimeError(f"{sid}: utility {util} missing (run 04 stage C)")
                spec = DesignSpec(name=name, design_id=stable_id(name), phi0=phi0, Bbar=Bbar,
                                  lam_ref=tuple(cal["lam_ref"]), kind=cal["kind"], util=util,
                                  tau_mix=float(cal["tau_mix"]), controller="expected", gamma=GAMMA, rho=rho)
                st = static_pass(T, spec)
                pi = st["pi_pay"]
                spend = float(np.sum(pi * s[pay]))
                meta = {"stream": sid, "design": name, "rho": rho, "pi_over": "payload_units",
                        "spec": {"phi0": phi0, "Bbar": Bbar, "lam_ref": list(cal["lam_ref"]), "kind": cal["kind"],
                                 "util": util, "tau_mix": float(cal["tau_mix"]), "controller": "expected",
                                 "design_id": spec.design_id},
                        "lamref_adaptive": cal.get("lamref_adaptive"), "calibration": f"{MODELS.name}/{fold}/{k or seeds[0]}/{pkey}",
                        "spend_over_target": spend / (Bbar * E), "max_abs_D_over_Bbar": float(np.max(np.abs(st["D"])) / Bbar),
                        "final_D_over_Bbar": float(st["D"][-1] / Bbar), "phi0_committed": phi0,
                        "phi0_stream": phi0_stream, "share_at_floor": float(np.mean(pi <= phi0 * (1 + 1e-9))),
                        "note": "static U6 = U5 (the boost needs the adaptive arm)" if dn == "U6" else None}
                _save(d, name, rho, pi, meta)
                summary.append({k2: meta[k2] for k2 in ("design", "rho", "spend_over_target", "max_abs_D_over_Bbar",
                                                         "phi0_committed", "phi0_stream", "share_at_floor")})
                if dn in ("U0", "U1"):
                    alias = {"U0": "B8", "U1": "B10"}[dn]
                    _save(d, alias, rho, pi, dict(meta, design=alias, alias_of=dn))
                # budget-matched static variant (E1/E3/E4 comparisons): fixed λ for the whole stream, each
                # component set by bisection on THIS stream's metadata (s, pay, utility; no labels or content
                # except the oracle U7) so that it alone spends B̄E in expectation; φ0 stays the committed value
                a_pay = np.ones(int(pay.sum())) if util is None else np.asarray(T.util(util))[pay]
                comps = {"mixture": ("A", "S"), "A": ("A",), "S": ("S",), "uniform": ("uniform",)}[cal["kind"]]
                lam_bm, sat = [], []
                for cpn in comps:
                    l, ok = calibrate_lambda(s[pay], a_pay, Bbar, phi0, E, component=cpn, iters=64, return_status=True)
                    lam_bm.append(l)
                    sat.append(not ok)
                pi_bm = design_pi(cal["kind"], lam_bm, a_pay, s[pay], phi0, float(cal["tau_mix"]))
                sp_bm = float(np.sum(pi_bm * s[pay]))
                mb = {"stream": sid, "design": f"{name}_bm", "rho": rho, "pi_over": "payload_units",
                      "budget_matched": True, "spec": {"phi0": phi0, "Bbar": Bbar, "lam": lam_bm, "kind": cal["kind"],
                                                       "util": util, "tau_mix": float(cal["tau_mix"]),
                                                       "controller": "none (fixed λ, matched on the stream's metadata)"},
                      "saturated": sat, "spend_over_target": sp_bm / (Bbar * E), "phi0_committed": phi0,
                      "phi0_stream": phi0_stream, "share_at_floor": float(np.mean(pi_bm <= phi0 * (1 + 1e-9))),
                      "uses_content": dn == "U7"}
                _save(d, f"{name}_bm", rho, pi_bm, mb)
                summary.append({k2: mb[k2] for k2 in ("design", "rho", "spend_over_target", "phi0_committed",
                                                       "phi0_stream", "share_at_floor")})
                if dn in ("U0", "U1"):
                    alias = {"U0": "B8", "U1": "B10"}[dn]
                    _save(d, f"{alias}_bm", rho, pi_bm, dict(mb, design=f"{alias}_bm", alias_of=f"{dn}_bm"))
            # deterministic / non-CtS baselines
            base = [B.b2_ring_buffer(T, rho), B.b3_time_machine(T, rho, flow),
                    B.b6_alert_triggered(T, rho, sev, flow, ts), B.b15_packet_sampling(T, rho, n_pk)]
            for k in seeds:
                pm = np.asarray(T.util(f"U3_s{k}"))
                r4 = B.b4_topk_pmal(T, rho, pm)
                r4["name"] = f"B4_s{k}"
                r13 = B.b13_topk_reserve(T, rho, pm, gamma=GAMMA)
                r13["name"] = f"B13_s{k}"
                rc = B.b4_controller(T, rho, pm)
                rc["name"] = f"B4ctl_s{k}"
                base += [r4, r13, rc]
            for r in base:
                meta = {"stream": sid, "design": r["name"], "rho": rho, "pi_over": "payload_units", "type": r["type"],
                        "spend_over_target": r["spend_ratio"], "params": r["params"], "phi0_committed": None,
                        "phi0_stream": phi0_stream}
                _save(d, r["name"], rho, r["pi"][pay], meta)
                extra = {k2: v for k2, v in r.items() if k2 in ("mask_any", "mask_triggered", "p_unit_any_packet",
                                                               "retained_bytes")}
                for k2, v in extra.items():
                    atomic_save_npy(d / f"{r['name']}_{rho:g}.{k2}.npy", np.asarray(v)[pay])
                summary.append({"design": r["name"], "rho": rho, "spend_over_target": r["spend_ratio"]})
            check_memory()
        atomic_write_json(d / "summary.json", summary)
        mark_done(d, {"n": len(summary)})
        return {"stream": sid, "n": len(summary)}
    except MemoryBudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        return {"stream": sid, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


def main() -> int:
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variant", default=None, help="recipe variant (config zero_payload_variants), e.g. zq: its "
                                                    "test streams, calibration from results/models_<variant>/")
    a = ap.parse_args()
    from ctsample.streams.recipe import load_config, load_inventory
    cfg, inv = load_config(), load_inventory()
    _set_variant(a.variant)
    todo = test_streams(cfg, inv) if not a.variant else variant_test_streams(cfg, inv, a.variant)
    if a.only:
        todo = [t for t in todo if any(k in t[0] for k in a.only.split(","))]
    seeds = list(range(a.seeds))
    nw = max(1, min(a.workers, workers(default=a.workers)))
    t0 = time.time()
    res, failed = [], []
    ex = ProcessPoolExecutor(max_workers=nw, mp_context=get_context("spawn"), initializer=_init_worker,
                             initargs=(os.getpid(), a.variant))
    try:
        pend = {ex.submit(design_stream, sid, fold, pk, seeds, a.force): sid for sid, fold, pk in todo}
        while pend:
            done, _ = wait(set(pend), timeout=30, return_when=FIRST_COMPLETED)
            for f in done:
                sid = pend.pop(f)
                r = f.result()
                res.append(r)
                if r.get("error"):
                    failed.append(sid)
                    print(f"[s5] FAILED {sid}: {r['error']}\n{r['traceback']}", flush=True)
                else:
                    print(f"[s5] {r} ({time.time() - t0:.0f}s)", flush=True)
            check_memory()
    except MemoryBudgetExceeded as e:
        print(f"[mem] {e}; exiting {EXIT_MEMORY}", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        return EXIT_MEMORY
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    print(f"[s5] {len(res) - len(failed)} ok, {len(failed)} failed in {time.time() - t0:.0f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
