"""E2: validity of the error statements on real streams (DESIGN §6.1 E2, §6.5 MC engine, §6.6 reporting).

Jobs = (part, stream, ρ, design arm). Parts (config/experiments/E2.yaml):
  main       LP-IoT23 (distinct per fold x realised prevalence), LP-Edge, natural IoT-23 and LP-Gotham
             (supplement); ρ ∈ {1, 2, 5}%; static U5τ (P1-D's U6tau0.2 design: expected-cost controller, no
             boost), adaptive U6τ (realised-cost controller + boost; λ_ref^A from the boost replay) and CtS-Size
             (U1/B10) static; R = 1,000 for LP-IoT23 at ρ = 5%, 500 otherwise.
  a11        selective abort (4 withholding rules): X := 0 + UCS voiding vs 'compliant epochs only'.
  reactor    [SHOULD] population-adaptive stress test: a C&C device goes silent for 20 epochs after a retained hit.
  familywise one report of |F| statements at α/|F| each (Bonferroni) per stream.
Every job runs P1-C's engine (sim.montecarlo.run_mc) with the pre-registered estimands (≥ 50 eligible units,
whole stream) and grid centres (training V_ref; experiments_core.build_estimands). Outputs:
results/E2/<stream>/<cfg_hash>/rep<a>_<b>_{cov,width,ctrl}.parquet + spec.json + config.json + DONE.

    .venv/bin/python memguard/memguard.py run --name p1_e2 --workers 16 -- \
        .venv/bin/python -I -u P1_CommitThenSample/scripts/11_e2.py --workers 14 [--parts main,a11] [--only 34-1]

rev1 (review R1-M3): ``--exp-root E2_rev1`` writes to results/E2_rev1/ and deep-merges config/experiments/E2_rev1.yaml
over E2.yaml: its ``fixed_time`` block switches on the engine's valid fixed-time comparators (MCSpec.fixed_time;
cov columns Lft1/Uft1/ft1_lo/ft1_hi, fth, ftb, ftx) and the horizon-tuned centre V_ref_ft of variant 'fth'.
Same seeds as rev0, so every rev0 column is reproduced bit for bit.

    .venv/bin/python memguard/memguard.py run --name p1_rev_e2 --workers 12 -- \
        .venv/bin/python -I -u P1_CommitThenSample/scripts/11_e2.py --workers 12 --exp-root E2_rev1
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctsample.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory, workers  # noqa: E402

P1 = Path(__file__).resolve().parents[1]
CFG_PATH = P1 / "config" / "experiments" / "E2.yaml"


def _init_worker(ppid: int) -> None:
    import threading

    def watch():
        while True:
            if os.getppid() != ppid:
                os._exit(1)
            time.sleep(2)

    threading.Thread(target=watch, daemon=True).start()


def load_cfg(exp_root: str = "E2") -> dict:
    """E2.yaml (over defaults.yaml); for exp_root != 'E2' also the overlay config/experiments/<exp_root>.yaml."""
    import yaml

    from ctsample.config import load_yaml
    over = None
    if exp_root != "E2":
        p = CFG_PATH.parent / f"{exp_root}.yaml"
        if not p.exists():
            raise FileNotFoundError(f"overlay config {p} missing for --exp-root {exp_root}")
        over = yaml.safe_load(p.read_text()) or {}
    raw = load_yaml(CFG_PATH, overrides=over).raw
    raw["exp_root"] = exp_root
    return raw


def fixed_time_on(raw: dict) -> bool:
    return bool((raw.get("fixed_time") or {}).get("enabled", False))


def horizon_v_ref(erow: dict, n_window_epochs: int, v_ref_epochs: int) -> float:
    """V_ref_ft of the horizon-tuned fixed-time variant 'fth': the training per-epoch V scaled to the evaluation
    window (V_ref x window/v_ref_epochs); the fallback (one floor-level hit, horizon-free) is kept as is."""
    if erow.get("V_ref_source") == "training":
        return float(erow["V_ref"]) * float(n_window_epochs) / float(v_ref_epochs)
    return float(erow["V_ref"])


def build_jobs(raw: dict, parts=("main", "a11", "reactor", "familywise")) -> tuple[list[dict], list[dict]]:
    from ctsample.sim import experiments_core as X
    rows = X.test_streams(("lp_iot23", "lp_edge", "lp_gotham", "natural"))
    kept, dropped = X.distinct_streams(rows)
    jobs = []
    for part in parts:
        pc = raw["parts"][part]
        rhos = pc.get("rho", raw["rho"])
        for st in kept:
            if st["kind"] not in pc["kinds"]:
                continue
            for rho in rhos:
                for dkey in pc["designs"]:
                    if part == "main":
                        reps = raw["reps_by"]["lp_iot23_headline"] if (st["kind"] == "lp_iot23" and abs(rho - 0.05) < 1e-12) \
                            else raw["reps_by"]["default"]
                    else:
                        reps = int(pc["reps"])
                    job = {"part": part, "stream": st["stream_id"], "kind": st["kind"], "bin": st["bin"],
                           "fold": st["fold_short"], "realised_prevalence": st.get("realised_prevalence"),
                           "rho": float(rho), "dkey": dkey, "reps": int(reps)}
                    if part == "a11":
                        job["withhold"] = list(pc["withhold"])
                        job["compliant_only"] = bool(pc.get("compliant_only", True))
                    if part == "reactor":
                        job["silence"] = int(pc["silence"])
                        job["families"] = list(pc["families"])
                    if part == "familywise":
                        job["familywise"] = True
                    jobs.append(job)
    return jobs, dropped


def _cost(job) -> float:
    c = job["reps"] * (2.2 if job["dkey"] == "adaptive" else 1.0)
    if job["part"] == "a11":
        c *= 5
    if job["kind"] == "lp_edge":
        c *= 3
    return c


def _settings(raw: dict) -> dict:
    keys = ("estimands", "arms", "chunk", "alpha", "cs", "mc", "policy")
    out = {k: raw.get(k) for k in keys}
    if raw.get("fixed_time") is not None:          # rev1 only (keeps the rev0 payload unchanged)
        out["fixed_time"] = raw["fixed_time"]
    return out


def run_job(job: dict, raw: dict) -> dict:
    t0 = time.time()
    try:
        from ctsample.config import src_hash
        from ctsample.sim import experiments_core as X
        from ctsample.sim.montecarlo import MCSpec, ReactiveSilence, run_dir, run_mc
        from ctsample.utils.io import atomic_write_json, is_done
        sid, rho, dkey = job["stream"], job["rho"], job["dkey"]
        dcfg = raw["arms"][dkey]
        T = X.load_tables(sid)
        meta = X.design_meta(sid, dcfg["source"], rho)
        spec = X.static_spec(meta, gamma=float(raw["policy"]["gamma"]))
        fold, seed, pkey = X.calibration_of(meta)
        lam = None
        if dcfg["arm"] == "adaptive":
            lam = X.lamref_entry(fold, seed, pkey, rho, dcfg["grid_key"])
            spec = X.adaptive_spec(meta, lam, gamma=float(raw["policy"]["gamma"]))
        grid = X.grid_entry(fold, seed, pkey, rho, dcfg["grid_key"])
        est_cfg = raw["estimands"]
        fam = 1
        if job.get("familywise"):
            fam = len(X.evaluated_queries(T, int(est_cfg["min_eligible"])))
        ests, erows = X.build_estimands(T, grid, spec.phi0, int(est_cfg["min_eligible"]), family_size=fam,
                                        epochs=int(est_cfg["v_ref_epochs"]))
        ft = fixed_time_on(raw)
        if ft:   # rev1: horizon-tuned centre of the fixed-time variant 'fth' (whole-stream windows: E epochs)
            from dataclasses import replace as _replace
            ests = [_replace(e, V_ref_ft=horizon_v_ref(r, T.n_epochs, int(est_cfg["v_ref_epochs"])))
                    for e, r in zip(ests, erows)]
            for e, r in zip(ests, erows):
                r["V_ref_ft"] = e.V_ref_ft
                r["window_epochs"] = int(T.n_epochs)
        reactor = None
        if job["part"] == "reactor":
            devs = sorted({int(d["device"]) for d in T.extra.get("incidents", []) if d.get("family") in job["families"]})
            if not devs:
                return {"job": job, "skipped": True, "reason": "no C&C incident"}
            reactor = ReactiveSilence(devices=tuple(devs), silence=int(job["silence"]), queries=None)
        a = raw["cs"]
        mc = MCSpec(reps=int(job["reps"]), alpha=float(raw["alpha"]), K_lcs=int(a["lcs"]["K"]),
                    lcs_decades=float(a["lcs"]["decades"]), ucs_n_low=int(a["ucs"]["n_low"]),
                    ucs_decades=float(a["ucs"]["decades"]), ucs_w0=float(a["ucs"]["w0"]),
                    ucs_eta0_frac=float(a["ucs"]["eta0_frac"]), report_every=int(raw["mc"]["report_every_epochs"]),
                    width_reps=int(raw["mc"]["width_reps"]), withhold=tuple(job.get("withhold", ())),
                    compliant_only=bool(job.get("compliant_only", True)), void_random_p=float(raw["mc"]["void_random_p"]),
                    withhold_p=float(raw["mc"]["withhold_p"]), stream_id=int(T.stream_id),
                    dcap_mode=raw["mc"]["dcap_mode"], reactor=reactor, fixed_time=ft,
                    ft_exact_kmax=int((raw.get("fixed_time") or {}).get("exact_pb_kmax", 1000)))
        exp_root = raw.get("exp_root", "E2")
        payload = {"exp": exp_root, "job": job, "settings": _settings(raw)}
        out = run_dir(exp_root, sid, payload, spec, ests, mc)
        if (out / "DONE").exists():
            return {"job": job, "skipped": True, "out": str(out)}
        out.mkdir(parents=True, exist_ok=True)
        repro = None
        if dcfg["arm"] == "static":
            repro = X.check_static_reproduces(T, spec, X.design_pi(sid, dcfg["source"], rho))
        conf = {"exp": exp_root, "job": job, "settings": _settings(raw), "design": asdict(spec),
                "design_source": {"file": f"designs/{sid}/{dcfg['source']}_{rho:g}.json", "calibration": meta["calibration"],
                                  "spend_over_target_static": meta.get("spend_over_target"),
                                  "lamref_adaptive": None if lam is None else lam.get("lamref_adaptive"),
                                  "boost_queries": None if lam is None else lam.get("boost_queries")},
                "static_repro_max_abs_diff": repro, "estimands": erows, "family_size": fam,
                "mc": json.loads(json.dumps(asdict(mc), default=str)), "src_hash": src_hash(),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "load_avg": os.getloadavg()}
        atomic_write_json(out / "config.json", conf)
        res = run_mc(T, spec, ests, mc, out_dir=out, workers=1, chunk=int(raw["chunk"]), check_mem=True)
        return {"job": job, "out": str(out), "seconds": time.time() - t0, "mc_seconds": res["seconds"],
                "n_est": len(ests), "repro": repro}
    except MemoryBudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        return {"job": job, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--parts", default="main,a11,reactor,familywise")
    ap.add_argument("--only", default=None, help="comma-separated substrings of stream ids")
    ap.add_argument("--rho", default=None, help="comma-separated rho values")
    ap.add_argument("--designs", default=None)
    ap.add_argument("--exp-root", default="E2", help="results/<exp-root>/; e.g. E2_rev1 (+ overlay config "
                                                     "config/experiments/<exp-root>.yaml: fixed-time comparators)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    raw = load_cfg(a.exp_root)
    jobs, dropped = build_jobs(raw, tuple(a.parts.split(",")))
    if a.only:
        jobs = [j for j in jobs if any(k in j["stream"] for k in a.only.split(","))]
    if a.rho:
        rs = [float(x) for x in a.rho.split(",")]
        jobs = [j for j in jobs if any(abs(j["rho"] - r) < 1e-12 for r in rs)]
    if a.designs:
        jobs = [j for j in jobs if j["dkey"] in a.designs.split(",")]
    jobs.sort(key=_cost, reverse=True)
    print(f"[e2] {len(jobs)} jobs; duplicate streams skipped: {[(d['stream_id'], d['duplicate_of']) for d in dropped]}",
          flush=True)
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
        pend = {ex.submit(run_job, j, raw): j for j in jobs}
        while pend:
            fin, _ = wait(set(pend), timeout=30, return_when=FIRST_COMPLETED)
            for f in fin:
                j = pend.pop(f)
                r = f.result()
                done += 1
                tag = f"{j['part']}/{j['stream']}/{j['dkey']}/rho={j['rho']:g}"
                if r.get("error"):
                    failed.append(tag)
                    print(f"[e2] FAILED {tag}: {r['error']}\n{r['traceback']}", flush=True)
                elif r.get("skipped"):
                    print(f"[e2] skip {tag} ({r.get('reason', 'done')}) [{done}/{len(jobs)}]", flush=True)
                else:
                    print(f"[e2] done {tag}: {r['seconds']:.0f}s, {r['n_est']} estimands, repro={r['repro']} "
                          f"[{done}/{len(jobs)}, {time.time() - t0:.0f}s]", flush=True)
            check_memory()
    except MemoryBudgetExceeded as e:
        print(f"[mem] {e}; exiting {EXIT_MEMORY}", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        return EXIT_MEMORY
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    print(f"[e2] {done - len(failed)} ok, {len(failed)} failed in {time.time() - t0:.0f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
