"""E3: value of the AI utility at equal budget (DESIGN §6.1 E3, §5.6 splits, §6.6 reporting).

Unit of analysis = test scenario (stream). In-domain: LOSO LP-IoT23 at p = 1% (6 folds); zero-shot: LP-Edge,
LP-Gotham (thinned 1% and natural prevalence), all-scenario models applied unchanged. ρ ∈ {2, 5}%.
  analytic job  budget-matched static designs (P1-D's <design>_bm: fixed λ, expected spend = B̄E): U0, U1
                (reference), U3, U5, U5τ (τ = 0.2, 0.5; = U6τ without the boost), U8 (stratified Neyman, no ML),
                U7 (oracle), 3 seeds: per estimand Var(Q̂) = Σ f²(1−π)/π, with-replacement V^wr = Σ f²/π; B12 (PPI
                difference estimator on U5): Σ (f − ĝ)²(1−π)/π with ĝ from the fold's hit models
                (indicator: P(hit | m); count: P(hit | m)·√M_cell).
  mc job        ρ = 5%: 500 reps of P1-C's engine per design (static _bm with controller 'none'; adaptive U6 / U6τ
                as deployed: realised-cost controller + boost) → final-epoch L+ (LCB gap) and, for the adaptive
                arms, the empirical MSE.
Outputs: results/E3/<stream>/<cfg_hash>/{var,ppi}.parquet + config.json + DONE (analytic) and the engine's
rep<a>_<b>_{cov,width,ctrl}.parquet + config.json + DONE (mc).

    .venv/bin/python memguard/memguard.py run --name p1_e3 --workers 16 -- \
        .venv/bin/python -I -u P1_CommitThenSample/scripts/12_e3.py --workers 12

Recipe-variant runs (rev1, review R3-M1(a)): ``--config config/experiments/E3_rev1_zq.yaml``. A config with
``base: E3.yaml`` is deep-merged over that base (identical settings unless overridden); ``out_dir`` names the
results directory (default E3), ``groups_e3`` the stream groups (default in_domain, zero_shot). Every design's
models root is read from its calibration string ('<root>/<fold>/<seed>/<pkey>': models or models_<variant>).
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
CFG_PATH = P1 / "config" / "experiments" / "E3.yaml"
GRID_KEY = {"U0": "U0", "U1": "U1", "U3": "U3", "U5": "U5", "U5tau0.2": "U6tau0.2", "U5tau0.5": "U6tau0.5",
            "U8": "U8", "U7": "U7"}


def _init_worker(ppid: int) -> None:
    import threading

    def watch():
        while True:
            if os.getppid() != ppid:
                os._exit(1)
            time.sleep(2)

    threading.Thread(target=watch, daemon=True).start()


def load_cfg(path=None) -> dict:
    """E3.yaml (default) or a variant config; ``base: <file>`` = deep-merge the variant over that base config."""
    import yaml
    from ctsample.config import load_yaml
    if path is None or Path(path).resolve() == CFG_PATH.resolve():
        return load_yaml(CFG_PATH).raw
    over = yaml.safe_load(Path(path).read_text()) or {}
    base = over.pop("base", None)
    if base is None:
        return load_yaml(path).raw
    raw = load_yaml(CFG_PATH.parent / base, overrides=over).raw
    raw["config_file"] = str(Path(path).name)
    return raw


def _use_models_of(meta: dict) -> None:
    """Models root of a design's training calibration ('<root>/<fold>/<seed>/<pkey>' with root 'models' or a recipe
    variant's 'models_<variant>'); experiments_core reads lamref / grid / utility bundles from X.MODELS."""
    from ctsample.sim import experiments_core as X
    from ctsample.utils.io import RESULTS
    X.MODELS = RESULTS / meta["calibration"].split("/")[0]


def _settings(raw):
    return {k: raw.get(k) for k in ("designs_e3", "adaptive_e3", "ppi", "mc_e3", "estimands", "seeds", "reference",
                                    "policy", "cs", "alpha")}


def build_jobs(raw: dict) -> list[dict]:
    jobs = []
    for grp in raw.get("groups_e3") or ("in_domain", "zero_shot"):
        for sid in raw["streams_e3"][grp]:
            for rho in raw["rho"]:
                jobs.append({"job": "analytic", "group": grp, "stream": sid, "rho": float(rho)})
            for rho in raw["rho_mc"]:
                for lab, d in raw["designs_e3"].items():
                    for s in (raw["seeds"] if d["seeded"] else [None]):
                        jobs.append({"job": "mc", "arm": "static_bm", "group": grp, "stream": sid, "rho": float(rho),
                                     "design": lab, "seed": s})
                for lab in raw["adaptive_e3"]:
                    for s in raw["seeds"]:
                        jobs.append({"job": "mc", "arm": "adaptive", "group": grp, "stream": sid, "rho": float(rho),
                                     "design": lab, "seed": s})
    return jobs


def _run_dir(job, raw):
    from ctsample.config import cfg_hash
    from ctsample.utils.io import RESULTS
    return RESULTS / raw.get("out_dir", "E3") / job["stream"] / cfg_hash({"exp": "E3", "job": job, "settings": _settings(raw)})


def _calib_meta(sid, rho, file_stem):
    """The controller-driven design json of the same design (carries the calibration path)."""
    from ctsample.sim import experiments_core as X
    stem = file_stem[:-3] if file_stem.endswith("_bm") else file_stem
    return X.design_meta(sid, stem, rho)


def ghat_for_ppi(sid: str, fold: str, seed: str, names) -> dict:
    """ĝ_q per payload unit from the fold's hit models (P1-D bundle): indicators P(hit|m); counts P(hit|m)·√M_cell."""
    from ctsample.design import utility as U
    from ctsample.sim import experiments_core as X
    b = U.UtilityBundle.load(X.MODELS / fold / str(seed), attach_lgbm=True)
    d = X.STREAMS / sid
    Xp = np.load(d / "features_pay.npy")
    z = np.load(d / "cols_pay.npz")
    cp = {k: z[k] for k in z.files}
    cp["pay"] = np.ones(len(Xp), bool)
    el = U.eligibility_matrix(b.queries, cp)
    P, _ = U.predict_all(b, Xp, el)
    cell = U.cell_index(cp)
    T = X.load_tables(sid)
    out = {}
    for nm in names:
        if nm not in b.queries:
            continue
        j = b.queries.index(nm)
        p = P[:, j] / 65536.0
        if T.query_kinds[T.query_index(nm)] == "count":
            M = np.asarray(b.M[nm], dtype=float)[cell] / 65536.0
            p = p * np.sqrt(np.maximum(M, 0.0))
        out[nm] = p
    return out


def analytic_job(job, raw):
    import pandas as pd
    import pyarrow as pa
    from ctsample.config import src_hash
    from ctsample.sim import experiments_core as X
    from ctsample.utils.io import atomic_write_json, atomic_write_parquet, mark_done
    out = _run_dir(job, raw)
    if (out / "DONE").exists():
        return {"job": job, "skipped": True}
    out.mkdir(parents=True, exist_ok=True)
    sid, rho = job["stream"], job["rho"]
    T = X.load_tables(sid)
    qrows = [r for r in X.evaluated_queries(T, int(raw["estimands"]["min_eligible"])) if r["Q"] > 0]
    names = [r["estimand"] for r in qrows]
    info = {r["estimand"]: r for r in qrows}
    pidx = X.pay_index(T)
    upos = np.full(T.n_units, -1, dtype=np.int64)
    upos[pidx] = np.arange(len(pidx))
    rows, prow = [], []
    pis = {}
    for lab, d in raw["designs_e3"].items():
        for s in (raw["seeds"] if d["seeded"] else [None]):
            stem = d["file"].format(s=s)
            pi = X.design_pi(sid, stem, rho)
            pis[(lab, s)] = pi
            meta = X.design_meta(sid, stem, rho)
            var = X.ht_var_by_query(T, pi, names)
            for nm in names:
                u, f = T.hits(T.query_index(nm))
                p = pi[upos[u]]
                vwr = float(np.sum(f * f / p)) if np.all(p > 0) else math.inf
                rows.append({"design": lab, "seed": -1 if s is None else s, "estimand": nm, "var": var[nm], "vwr": vwr,
                             "Q": info[nm]["Q"], "kind_q": info[nm]["kind"], "cls": info[nm]["class"],
                             "n_elig": info[nm]["n_elig"], "n_hits": info[nm]["n_hits"],
                             "spend_over_target": meta.get("spend_over_target"), "saturated": json.dumps(meta.get("saturated")),
                             "share_at_floor": meta.get("share_at_floor")})
    # B12: PPI difference estimator on the U5 design, per seed
    base = raw["ppi"]["base_design"]
    for s in raw["seeds"]:
        cm = _calib_meta(sid, rho, raw["designs_e3"][base]["file"].format(s=s))
        _use_models_of(cm)
        fold, sd, _ = X.calibration_of(cm)
        g = ghat_for_ppi(sid, fold, sd, names)
        v = X.ppi_var_by_query(T, pis[(base, s)], g)
        for nm, val in v.items():
            prow.append({"design": "B12", "seed": s, "estimand": nm, "var": val, "Q": info[nm]["Q"],
                         "kind_q": info[nm]["kind"], "cls": info[nm]["class"], "ghat_sum": float(np.sum(g[nm]))})
    atomic_write_parquet(out / "var.parquet", pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False))
    atomic_write_parquet(out / "ppi.parquet", pa.Table.from_pandas(pd.DataFrame(prow), preserve_index=False))
    atomic_write_json(out / "config.json", {"exp": "E3", "job": job, "settings": _settings(raw), "estimands": qrows,
                                            "src_hash": src_hash(), "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    mark_done(out, {"rows": len(rows)})
    return {"job": job, "out": str(out)}


def mc_job(job, raw):
    from ctsample.config import src_hash
    from ctsample.sim import experiments_core as X
    from ctsample.sim.montecarlo import MCSpec, run_mc
    from ctsample.utils.io import atomic_write_json
    sid, rho, lab, s = job["stream"], job["rho"], job["design"], job["seed"]
    T = X.load_tables(sid)
    gamma = float(raw["policy"]["gamma"])
    if job["arm"] == "static_bm":
        d = raw["designs_e3"][lab]
        stem = d["file"].format(s=s)
        meta = X.design_meta(sid, stem, rho)
        spec = X.bm_spec(meta, gamma)
        cm = _calib_meta(sid, rho, stem)
        _use_models_of(cm)
        gkey = GRID_KEY[lab]
        repro = X.check_static_reproduces(T, spec, X.design_pi(sid, stem, rho))
    else:
        d = raw["adaptive_e3"][lab]
        cm = X.design_meta(sid, d["source"].format(s=s), rho)
        _use_models_of(cm)
        gkey = d["grid_key"]
        fold, sd, pkey = X.calibration_of(cm)
        spec = X.adaptive_spec(cm, X.lamref_entry(fold, sd, pkey, rho, gkey), gamma)
        repro = None
    fold, sd, pkey = X.calibration_of(cm)
    grid = X.grid_entry(fold, sd, pkey, rho, gkey)
    ests, erows = X.build_estimands(T, grid, spec.phi0, int(raw["estimands"]["min_eligible"]))
    mcc = raw["mc_e3"]
    a = raw["cs"]
    mc = MCSpec(reps=int(mcc["reps"]), alpha=float(raw["alpha"]), K_lcs=int(a["lcs"]["K"]), lcs_decades=float(a["lcs"]["decades"]),
                ucs_n_low=int(a["ucs"]["n_low"]), ucs_decades=float(a["ucs"]["decades"]), ucs_w0=float(a["ucs"]["w0"]),
                ucs_eta0_frac=float(a["ucs"]["eta0_frac"]), report_every=int(mcc["report_every"]),
                width_reps=int(mcc["width_reps"]), stream_id=int(T.stream_id), dcap_mode=raw["mc"]["dcap_mode"])
    out = _run_dir(job, raw)
    if (out / "DONE").exists():
        return {"job": job, "skipped": True}
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "config.json", {"exp": "E3", "job": job, "settings": _settings(raw), "design": asdict(spec),
                                            "static_repro_max_abs_diff": repro, "estimands": erows,
                                            "mc": json.loads(json.dumps(asdict(mc), default=str)), "src_hash": src_hash(),
                                            "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    run_mc(T, spec, ests, mc, out_dir=out, workers=1, chunk=int(mcc["chunk"]), check_mem=True)
    return {"job": job, "out": str(out), "repro": repro}


def run_job(job, raw):
    t0 = time.time()
    try:
        r = analytic_job(job, raw) if job["job"] == "analytic" else mc_job(job, raw)
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
    ap.add_argument("--jobs", default="analytic,mc")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default=None, help="experiment config (default config/experiments/E3.yaml), e.g. "
                                                   "config/experiments/E3_rev1_zq.yaml (recipe variant)")
    a = ap.parse_args()
    raw = load_cfg(None if a.config is None else (Path(a.config) if Path(a.config).is_absolute()
                                                  else (P1 / a.config if (P1 / a.config).exists() else Path(a.config))))
    jobs = [j for j in build_jobs(raw) if j["job"] in a.jobs.split(",")]
    if a.only:
        jobs = [j for j in jobs if any(k in j["stream"] for k in a.only.split(","))]
    jobs.sort(key=lambda j: (j["job"] != "analytic", j.get("arm") != "adaptive"))
    print(f"[e3] {len(jobs)} jobs", flush=True)
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
                tag = f"{j['job']}/{j['stream']}/rho={j['rho']:g}" + (f"/{j['arm']}/{j['design']}/s{j['seed']}" if j["job"] == "mc" else "")
                if r.get("error"):
                    failed.append(tag)
                    print(f"[e3] FAILED {tag}: {r['error']}\n{r['traceback']}", flush=True)
                else:
                    print(f"[e3] {'skip' if r.get('skipped') else 'done'} {tag} ({r.get('seconds', 0):.0f}s, "
                          f"repro={r.get('repro')}) [{done}/{len(jobs)}, {time.time() - t0:.0f}s]", flush=True)
            check_memory()
    except MemoryBudgetExceeded as e:
        print(f"[mem] {e}; exiting {EXIT_MEMORY}", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        return EXIT_MEMORY
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    print(f"[e3] {done - len(failed)} ok, {len(failed)} failed in {time.time() - t0:.0f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
