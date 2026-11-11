#!/usr/bin/env python
"""12 -- P2-G experiment driver (DESIGN §6; PREREG_2026-10-21, BLAKE3 955c768e...). Owner: P2-G.

Run from the project root under memguard (P2 budget 90 GB, <= 8 processes):

  .venv/bin/python memguard/memguard.py run --name p2_E1 --limit-gb 90 --workers 8 -- \
      .venv/bin/python -I -u P2_CertiChain/scripts/12_run_experiment.py --exp E1

  --exp E1,E7,E15,E2,E11,E5,E3,E4,validity   experiments (configs/exp/<E>.yaml); 'validity' = primary endpoint 1
  --workers N        process pool size (default: MEMGUARD_WORKERS or 8; capped at 8)
  --no-boot          skip bootstraps (cells only)
  --overwrite        recompute finished cells / bootstraps
  --only DS,DS       restrict to datasets
  --numbers-only     rebuild results/summary/<E>_numbers.json from finished cells

Outputs: results/exp/E<n>/<ds>/<method>__<scorer_id>/<cfg_hash>/{split_<sss>.json, per_split.parquet,
summary.parquet, cell.json, bootstrap_*.json}, results/summary/E<n>_numbers.json. Resumable: finished cells and
bootstraps are skipped. Exit 75 on MemoryBudgetExceeded (memguard restarts with halved workers).
Never touches the GPU, never runs scripts 08-11/16 (reads their outputs only).
"""
from __future__ import annotations

import os
import sys

for _v, _n in (("POLARS_MAX_THREADS", "4"), ("OMP_NUM_THREADS", "2"), ("OPENBLAS_NUM_THREADS", "2"),
               ("VECLIB_MAXIMUM_THREADS", "2"), ("MKL_NUM_THREADS", "2")):
    os.environ.setdefault(_v, _n)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import yaml  # noqa: E402

from certichain.eval import experiments as X  # noqa: E402
from certichain.certify.losses import LossBank as LossBankX  # noqa: E402
from certichain.eval.splits import random_splits  # noqa: E402
from certichain.io.paths import P2_ROOT, atomic_write_json, atomic_write_parquet  # noqa: E402
from certichain.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory, workers  # noqa: E402

CFG_DIR = P2_ROOT / "configs" / "exp"
EXPENSIVE = {"lr_a1avg": 9, "lr_a1avgb": 9, "b8": 7, "bh_iid_pooled": 6, "lr_a2": 5, "lr_a2b": 5, "lr_a1": 5,
             "lr_a1b": 5, "bh_iid": 3}
METRIC_DEF = {
    "fdr": "realised FDR = mean over test incidents of the strict per-incident FDP (AMB = false), mean over splits",
    "fdr_lenient": "mean per-incident lenient FDP (AMB excluded), mean over splits",
    "pooled_fdr": "pooled FDR = sum V / max(sum R, 1) over a split's test incidents (strict), mean over splits",
    "pooled_fdr_lenient": "pooled lenient FDR, mean over splits",
    "p_fdp_gt_2a": "P(per-incident strict FDP > 2 alpha)", "fdx_020": "P(strict FDP > 0.2)",
    "fdx_030": "P(strict FDP > 0.3)", "n_claims": "mean claims per test incident", "frac_empty": "share of test incidents with no claim",
    "abstain": "share of splits with lambda-hat = lambda_0 (no claims anywhere)",
    "str_mean": "STR = certified distinct transitions / |T*|, mean over test incidents with |T*| > 0",
    "str_pooled": "STR pooled over a split's test incidents", "ostr_mean": "observable STR (/|T*_obs|), mean over incidents",
    "cer_link_mean": "CER_link = certified true links / (|E*_link| + |T*_hid|), mean over incidents with a positive denominator",
    "cer_link_pooled": "CER_link pooled over a split", "recall_pooled": "certified true links / true links in C (pooled)",
    "bridge_recall": "certified bridges / bridges (pooled)", "ucr": "share of certified links with a decoy endpoint (pooled)",
    "wrongful_benign": "P(any claim | benign-only incident)",
    "dev_false_impl": "non-campaign devices endpoint of >= 1 false certified link / non-campaign devices (pooled)",
    "load_ratio": "analyst load = sum R / sum TP over all splits' test incidents (ratio of sums)",
    "maxchain_fdr": "maximal-chain FDR (strict), mean over test incidents", "maxchain_fdr_lenient": "maximal-chain FDR (lenient)",
    "prefix_fdr": "prefix-family FDR", "chain_fwer": "P(some reported maximal chain is wrong)",
    "ccr_mean": "CCR = GT root-leaf paths fully covered by one certified maximal chain / GT paths, mean over incidents",
    "ccr_pooled": "CCR pooled", "cert_gap_cer_link": "certification gap: oracle-threshold CER_link at the same realised test FDR minus CRC-BH CER_link (uses test labels)",
    "lam": "lambda-hat", "n_chains": "maximal chains per incident",
}
REPORT_METRICS = tuple(METRIC_DEF)


def log(msg: str) -> None:
    print(f"[12 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cfg(exp: str) -> dict:
    p = CFG_DIR / f"{exp}.yaml"
    cfg = yaml.safe_load(p.read_text())
    import blake3
    cfg["_file"] = str(p.relative_to(P2_ROOT))
    cfg["_blake3"] = blake3.blake3(p.read_bytes()).hexdigest()
    return cfg


# --------------------------------------------------------------------------- process pool
def _cost(t: dict) -> float:
    c = EXPENSIVE.get(t.get("method", ""), 1)
    return c * (20 if t["kind"] == "boot" else 1) * (2 if t.get("sid_b") else 1)


def pool_run(tasks: list[dict], n_workers: int, label: str) -> list[dict]:
    out: list[dict] = []
    if not tasks:
        return out
    tasks = sorted(tasks, key=_cost, reverse=True)
    t0 = time.time()
    if n_workers <= 1:
        for k, t in enumerate(tasks):
            check_memory()
            out.append(X.run_task(t))
            if (k + 1) % 25 == 0:
                log(f"{label}: {k + 1}/{len(tasks)} ({time.time() - t0:.0f}s)")
        return out
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(n_workers, mp_context=ctx) as ex:
        futs = [ex.submit(X.run_task, t) for t in tasks]
        for k, f in enumerate(as_completed(futs)):
            out.append(f.result())
            check_memory()
            if (k + 1) % 25 == 0 or k + 1 == len(tasks):
                log(f"{label}: {k + 1}/{len(tasks)} ({time.time() - t0:.0f}s)")
    return out


def no_claims(path: Path) -> bool:
    df = pl.read_parquet(Path(path) / "per_split.parquet", columns=["n_claims"])
    return bool((df["n_claims"].fill_nan(0).fill_null(0) == 0).all())


# --------------------------------------------------------------------------- numbers helpers
def add_cell_numbers(N: X.Numbers, key: str, path: Path, status: str, boot_name: str = "main",
                     metrics=REPORT_METRICS) -> None:
    df, meta = X.read_cell(path)
    boot = X.read_bootstrap(path, boot_name)
    summ = meta.get("summary", {})
    for m in metrics:
        if m == "load_ratio":
            v = summ.get("load_ratio")
            ci = None
            if boot and "load_ratio" in boot["metrics"]:
                b = boot["metrics"]["load_ratio"]
                ci = (b.get("lo"), b.get("hi"))
            N.add(f"{key}|{m}", v, definition=METRIC_DEF[m], ci=ci, source=X.rel(path), status=status)
            continue
        if m not in df.columns:
            continue
        s = summ.get(m)
        if not isinstance(s, dict) or s.get("mean") is None:
            continue
        ci = se = None
        if boot and m in boot["metrics"] and isinstance(boot["metrics"][m], dict) and "lo" in boot["metrics"][m]:
            b = boot["metrics"][m]
            ci, se = (b["lo"], b["hi"]), b["se"]
        N.add(f"{key}|{m}", s["mean"], definition=METRIC_DEF.get(m, m), ci=ci, se=se, source=X.rel(path),
              status=status, split_sd=s["sd"], split_q05=s["q05"], split_q95=s["q95"],
              ci_kind=("incident-pool cluster bootstrap, B=%d x %d splits, percentile 95%%" %
                       (boot["B"], boot["n_splits_per_replicate"])) if ci else None)
    for c in df.columns:
        if c.startswith("fdr|"):
            s = summ.get(c)
            if isinstance(s, dict) and s.get("mean") is not None:
                N.add(f"{key}|{c}", s["mean"], definition=f"realised strict FDR within stratum {c[4:]}",
                      source=X.rel(path), status="secondary", split_sd=s["sd"], n_splits=s["n"])


def cell_status(ds: str, method: str, sid: str, alpha: float) -> str:
    if method == "crc_bh" and alpha in X.CO_PRIMARY and ds in ("cf_ton", "cf_ton_h", "cf_gotham") and \
            (sid.endswith(".stat") or sid.endswith(".fuse_s0")):
        return "primary"
    return "secondary"


# =========================================================================== E1
def ds_override(cfg: dict, a, key: str = "datasets") -> None:
    if getattr(a, "datasets", None):
        cfg[key] = {ds: {"K": X.DATASETS[ds]["K"], "n_test": X.DATASETS[ds]["n_test"]} for ds in a.datasets}


def e1(cfg: dict, a) -> None:
    ds_override(cfg, a)
    tasks = []
    missing = []
    for ds, d in cfg["datasets"].items():
        if a.only and ds not in a.only:
            continue
        for kind in cfg["scorers"]:
            sid = X.scorer_id(ds, kind)
            if not X.has_bank(ds, sid):
                missing.append(f"{ds}/{sid}")
                continue
            bank = X.load_bank(ds, sid)
            for alpha in cfg["alphas"]:
                for m in cfg["methods"]:
                    if m.startswith("chain") and not bank.has_chains:
                        continue
                    tasks.append({"kind": "cell", "exp": "E1", "ds": ds, "sid": sid, "method": m, "alpha": alpha,
                                  "K": d["K"], "n_test": d["n_test"], "splits": cfg["splits"],
                                  "overwrite": a.overwrite})
    if missing:
        log(f"E1: banks not available yet (skipped): {missing}")
    res = pool_run(tasks, a.workers, "E1 cells")
    if not a.no_boot:
        bc = cfg["bootstrap"]
        bt = []
        for r in res:
            if "path" not in r:
                continue
            t = r["task"]
            want = (bc.get("crc_all") and t["method"] == "crc_bh") or (
                t["ds"] == bc["table_iv"]["ds"] and t["alpha"] in bc["table_iv"]["alphas"])
            if not want:
                continue
            if bc.get("skip_if_no_claims") and no_claims(Path(r["path"])):
                continue
            bt.append({"kind": "boot", "cell": r["path"], "name": "main", "ds": t["ds"], "sid": t["sid"],
                       "method": t["method"], "alpha": t["alpha"], "K": t["K"], "n_test": t["n_test"],
                       "B": bc["B"], "boot_splits": bc["splits"], "seed": bc["seed"], "overwrite": a.overwrite})
        pool_run(bt, a.workers, "E1 bootstraps")
    e1_numbers(cfg)


def e1_numbers(cfg: dict) -> None:
    N = X.Numbers("E1")
    idx = X.collect("E1")
    if idx.height == 0:
        return
    for r in idx.iter_rows(named=True):
        key = f"{r['ds']}|{r['scorer_id']}|{r['method']}|a={r['alpha']}"
        path = Path(r["path"])
        add_cell_numbers(N, key, path, cell_status(r["ds"], r["method"], r["scorer_id"], r["alpha"]))
        if r["method"] == "ltt":
            df, _ = X.read_cell(path)
            cov = float((df["fdr"].to_numpy() <= r["alpha"] + 1e-12).mean())
            N.add(f"{key}|ltt_coverage", cov, definition="share of splits whose test-mean strict FDP <= alpha "
                  "(empirical LTT coverage; target >= 1 - delta = 0.9)", source=X.rel(path), status="secondary")
        if r["method"] in ("crc_fdx020", "crc_fdx030"):
            df, _ = X.read_cell(path)
            g = "fdx_020" if r["method"] == "crc_fdx020" else "fdx_030"
            N.add(f"{key}|fdx_target", float(df[g].mean()), definition=f"FDX exceedance P(FDP > gamma) of the "
                  f"method's own gamma ({g}); guarantee <= alpha", source=X.rel(path), status="secondary")
    N.add("_meta|missing_banks", [f"{ds}/{X.scorer_id(ds, k)}" for ds in cfg["datasets"] for k in cfg["scorers"]
                                  if not X.has_bank(ds, X.scorer_id(ds, k))],
          definition="banks not available when the numbers were written", status="secondary")
    N.write()
    log(f"E1 numbers: {len(N.d)} entries -> {N.path}")


# =========================================================================== primary endpoint 1
def parse_cell(name: str) -> tuple[str, float, str, str]:
    ds, al, meth, kind = name.split("|")
    return ds, float(al.split("=")[1]), meth, ("stat" if kind == "stats" else "fuse_s0")


def find_cell(exp: str, ds: str, sid: str, method: str, alpha: float, K: int | None = None,
              split_tag=None) -> Path | None:
    idx = X.collect(exp)
    if idx.height == 0:
        return None
    sub = idx.filter((pl.col("ds") == ds) & (pl.col("scorer_id") == sid) & (pl.col("method") == method)
                     & ((pl.col("alpha") - alpha).abs() < 1e-9))
    if K is not None:
        sub = sub.filter(pl.col("K") == K)
    if split_tag is not None and "split_tag" in sub.columns:
        sub = sub.filter(pl.col("split_tag") == split_tag)
    elif "split_tag" in sub.columns:
        sub = sub.filter(pl.col("split_tag").is_null())
    if "test_scorer_id" in sub.columns:
        sub = sub.filter(pl.col("test_scorer_id").is_null())
    return Path(sub["path"][0]) if sub.height else None


def validity(cfg: dict, a) -> None:
    validity_family_run(cfg, a, ds_map={}, tag="")
    if getattr(a, "ds_map", None):
        validity_family_run(cfg, a, ds_map=a.ds_map, tag=a.tag or "map")


def validity_family_run(cfg: dict, a, ds_map: dict, tag: str) -> None:
    """Pre-registered family (tag ''); with ds_map, the documented post-PREREG deviation family in which the mapped
    datasets (e.g. cf_gotham -> cf_gotham_v2) replace the original cells (written next to, not over, the original)."""
    pr = X.load_prereg()
    names = pr["endpoints"]["primary_1_validity"]["cells"]
    cells = {}
    for name in names:
        ds, alpha, meth, kind = parse_cell(name)
        ds = ds_map.get(ds, ds)
        if ds == "iot23_loso":
            sid = X.scorer_id("iot23", kind)
            p = X.exp_root("E11") / "iot23" / "loso" / f"{meth}__{sid}" / f"a{alpha}" / "jackknife.json"
            if p.exists():
                j = json.loads(p.read_text())
                cells[name] = {"alpha": alpha, "mean": j["estimate"], "se": j["se_jack"], "se_kind": "jackknife",
                               "n_units": j["n_scenarios"], "source": X.rel(p)}
            else:
                cells[name] = {"alpha": alpha, "mean": None, "se": None, "source": None}
            continue
        if ds not in X.DATASETS:
            cells[name] = {"alpha": alpha, "mean": None, "se": None, "source": None}
            continue
        sid = X.scorer_id(ds, kind)
        path = find_cell("E1", ds, sid, meth, alpha)
        if path is None:
            cells[name] = {"alpha": alpha, "mean": None, "se": None, "source": None}
            continue
        df, _ = X.read_cell(path)
        boot = X.read_bootstrap(path, "main")
        se = boot["metrics"]["fdr"]["se"] if boot else None
        cells[name] = {"alpha": alpha, "mean": float(df["fdr"].mean()), "se": se, "se_kind": "bootstrap B=1000x50",
                       "n_splits": df.height, "source": X.rel(path)}
    fam = X.validity_family(cells)
    run = [c for c in fam if not c.get("missing")]
    zmax = max((c["z"] for c in run), default=float("nan"))
    out = {"endpoint": "primary_1_validity", "prereg_blake3": X.PREREG_BLAKE3,
           "definition": pr["endpoints"]["primary_1_validity"], "cells": fam,
           "n_cells_run": len(run), "n_missing": len(fam) - len(run),
           "max_standardised_excess": zmax, "n_violations": sum(c["violation"] for c in run),
           "complete": len(run) == len(names)}
    out["ds_map"] = ds_map
    sfx = f"_{tag}" if tag else ""
    atomic_write_json(X.summary_root() / f"primary1_validity{sfx}.json", out)
    N = X.Numbers("E1")
    key = f"primary1{sfx}"
    st = "primary" if not tag else "deviation"
    for c in fam:
        if c.get("missing"):
            N.add(f"{key}|{c['cell']}", None, definition="cell not run (bank missing); excluded from the Holm family",
                  status=st)
            continue
        N.add(f"{key}|{c['cell']}", c["mean"], definition="realised FDR (mean strict FDP over test incidents), "
              "200 splits (P-LOSO: mean over scenarios)" + (f"; datasets substituted {ds_map} (post-PREREG deviation)"
                                                            if tag else ""), se=c["se"], status=st, z=c["z"],
              p=c["p"], p_holm=c["p_holm"], violation=c["violation"], se_kind=c.get("se_kind"), source=c.get("source"))
    N.add(f"{key}|max_standardised_excess", zmax, definition="max over run cells of z = (mean - alpha)/SE",
          status=st, n_cells=len(run), n_missing=len(fam) - len(run))
    N.add(f"{key}|n_violations", out["n_violations"], definition="Holm-rejected cells at 0.05 (one-sided H0: FDR <= alpha)",
          status=st)
    N.write()
    log(f"validity{sfx}: {len(run)}/{len(names)} cells, max z {zmax:.2f}, violations {out['n_violations']}")


# =========================================================================== E7
def e7(cfg: dict, a) -> None:
    ds, sid = cfg["dataset"], X.scorer_id(cfg["dataset"], cfg["scorer"])
    tasks = []
    for K in cfg["K_grid"]:
        for alpha in cfg["alphas"]:
            for m in cfg["methods"]:
                tasks.append({"kind": "cell", "exp": "E7", "ds": ds, "sid": sid, "method": m, "alpha": alpha,
                              "K": K, "n_test": cfg["n_test"], "splits": cfg["splits"], "overwrite": a.overwrite})
    res = pool_run(tasks, a.workers, "E7 cells")
    if not a.no_boot:
        bt = [{"kind": "boot", "cell": r["path"], "name": "main", "ds": r["task"]["ds"], "sid": r["task"]["sid"],
               "method": r["task"]["method"], "alpha": r["task"]["alpha"], "K": r["task"]["K"],
               "n_test": r["task"]["n_test"], "B": 1000, "boot_splits": 50, "seed": 0,
               "metrics": ["fdr", "fdr_lenient", "pooled_fdr", "str_mean", "str_pooled", "abstain", "n_claims"],
               "overwrite": a.overwrite}
              for r in res if "path" in r and not no_claims(Path(r["path"]))]
        pool_run(bt, a.workers, "E7 bootstraps")
    sc = cfg["synthetic"]
    st = []
    for gen, gp in sc["generators"].items():
        for p in range(sc["pools"]):
            st.append({"kind": "synth_pool", "gen": gen, "params": gp, "pool": p, "cfg": sc, "K_grid": cfg["K_grid"],
                       "alphas": cfg["alphas"], "methods": cfg["methods"], "overwrite": a.overwrite})
    pool_run(st, a.workers, "E7 synthetic pools")
    e7_synth_cells(cfg)
    e7_numbers(cfg)


def e7_synth_cells(cfg: dict) -> None:
    sc = cfg["synthetic"]
    for gen in sc["generators"]:
        parts = []
        for p in range(sc["pools"]):
            f = X.exp_root("E7") / "synth" / gen / f"p{p}" / "rows.parquet"
            if f.exists():
                parts.append(pl.read_parquet(f))
        if len(parts) < sc["pools"]:
            log(f"E7 synth {gen}: {len(parts)}/{sc['pools']} pools finished")
            continue
        allr = pl.concat(parts, how="diagonal_relaxed")
        for (m, K, al), sub in allr.group_by(["method", "K", "alpha"], maintain_order=True):
            sub = sub.sort(["pool", "seed"]).with_columns((pl.col("pool") * 1000 + pl.col("seed")).alias("seed"))
            cfgc = {"exp": "E7", "ds": f"synth_{gen}", "method": m, "K": int(K), "alpha": float(al),
                    "generator": sc["generators"][gen], "pools": sc["pools"], "splits_per_pool": sc["splits_per_pool"],
                    "pool_size": sc["pool_size"], "n_test": sc["n_test"], "prereg": X.PREREG_BLAKE3, "code": X.CODE_VERSION}
            h = X.cfg_hash(cfgc)
            path = X.cell_dir("E7", f"synth_{gen}", m, "synth", h)
            if not (path / "cell.json").exists():
                X.write_cell(path, sub, cfgc, {"seconds": None, "prereg_blake3": X.PREREG_BLAKE3})


def e7_numbers(cfg: dict) -> None:
    N = X.Numbers("E7")
    idx = X.collect("E7")
    for r in idx.iter_rows(named=True):
        key = f"{r['ds']}|{r['scorer_id'] or 'synth'}|{r['method']}|K={r['K']}|a={r['alpha']}"
        add_cell_numbers(N, key, Path(r["path"]), "secondary",
                         metrics=("fdr", "fdr_lenient", "pooled_fdr", "str_mean", "str_pooled", "recall_pooled",
                                  "abstain", "n_claims", "frac_empty", "load_ratio"))
    N.add("_meta|abstention_rule", "CRC abstains iff (K + 1) alpha < B = 1, i.e. K < 1/alpha - 1 "
          "(K <= 18 at alpha = 0.05, K <= 8 at 0.1, K <= 3 at 0.2)", definition="Remark (identities)", status="secondary")
    N.write()
    log(f"E7 numbers: {len(N.d)} entries")


# =========================================================================== E15
def e15(cfg: dict, a) -> None:
    out_dir = X.exp_root("E15")
    out_dir.mkdir(parents=True, exist_ok=True)
    N = X.Numbers("E15")
    desc_ds = ["cf_ton", "cf_ton_h", "cf_gotham", "iot23", "gotham_real"]
    if getattr(a, "datasets", None):
        desc_ds = [d for d in a.datasets if X.has_bank(d, X.scorer_id(d, "stat"))]
        cfg["oracle_datasets"] = list(desc_ds)
        cfg["probe_datasets"] = list(a.datasets)
    # descriptors (Table III) from the stats banks (cal/test pools) + summary_07 where no bank exists
    desc = []
    for ds in desc_ds:
        sid = X.scorer_id(ds, "stat")
        if X.has_bank(ds, sid):
            d = X.descriptors(X.load_bank(ds, sid))
            d.update(ds=ds, source=f"losses/{ds}/{sid}")
            desc.append(d)
    gt_name = "cf_gotham_t" if not getattr(a, "datasets", None) else next(
        (d for d in a.datasets if d.startswith("cf_gotham_t")), None)
    s07p = P2_ROOT / "results" / "interim" / (gt_name or "cf_gotham_t") / "summary_07.json"
    s07 = json.loads(s07p.read_text()) if (gt_name and s07p.exists()) else None
    ob = s07["observability"] if s07 else {}
    if s07 and gt_name not in desc_ds: desc.append({"ds": gt_name, "n_incidents": s07["n_incidents"], "N_median": s07["N"]["p50"],
                 "N_mean": s07["N"]["mean"], "N_total": s07["N_total"], "N_over_r_pooled": math.nan,
                 "N_over_r_all": s07["N_over_r"]["pooled"],
                 "N_over_r_median": s07["N_over_r"]["per_incident"]["p50"], "r_CT": ob["r_CT_pooled"],
                 "rho_T": ob["rho_T_pooled"], "ceiling": ob["ceiling_pooled"], "n_T": ob["n_T"],
                 "n_T_obs": ob["n_T_obs"], "n_T_hid": ob["n_T_hid"], "n_T_obs_in_C": ob["n_T_obs_in_C"],
                 "n_true": s07["labels"]["n_pos"], "n_amb": s07["labels"]["n_amb"],
                 "amb_share": s07["labels"]["n_amb"] / max(s07["N_total"], 1),
                 "source": f"interim/{gt_name}/summary_07.json"})
    ddf = pl.DataFrame(desc, infer_schema_length=None)
    atomic_write_parquet(ddf, out_dir / f"descriptors{'_' + a.tag if getattr(a, 'tag', '') else ''}.parquet")
    for d in desc:
        for k in ("n_incidents", "N_median", "N_over_r_pooled", "N_over_r_all", "N_over_r_median", "r_CT", "rho_T", "ceiling",
                  "n_T", "n_T_obs", "n_T_hid", "n_T_obs_in_C", "n_true", "amb_share"):
            if k in d:
                N.add(f"descriptor|{d['ds']}|{k}", d[k], definition=f"{k} of the cal/test pool (pooled over incidents)",
                      source=d["source"], status="secondary")
    # rule-only oracles B1 / B2 (1 run over the pool + 50 splits' test sets)
    orows = []
    for ds in cfg["oracle_datasets"]:
        sid = X.scorer_id(ds, "stat")
        if not X.has_bank(ds, sid):
            continue
        bank = X.load_bank(ds, sid)
        K = X.DATASETS[ds]["K"]
        nt = X.DATASETS[ds]["n_test"]
        for b in cfg["oracles"]:
            if not X.has_baseline(ds, b):
                continue
            cnt = X.baseline_counts(bank, ds, b)
            one = X.Splits(np.array([0]), np.zeros((1, 0), np.int64), np.arange(bank.n)[None, :])
            r1 = X.fixed_metrics(bank, cnt, one, 0.1)
            sp = random_splits(bank.n, K, range(cfg["oracle_splits"]), nt)
            r50 = X.fixed_metrics(bank, cnt, sp, 0.1)
            row = {"ds": ds, "oracle": b}
            for m in ("fdr", "fdr_lenient", "pooled_fdr", "recall_pooled", "str_mean", "str_pooled", "cer_link_mean",
                      "cer_link_pooled", "ucr", "wrongful_benign", "n_claims", "bridge_recall"):
                v1 = float(np.nanmean(r1[m]))
                v50 = np.asarray(r50[m], float)
                row[m] = v1
                row[f"{m}_sd50"] = float(np.nanstd(v50, ddof=1))
                row[f"{m}_mean50"] = float(np.nanmean(v50))
                N.add(f"oracle|{ds}|{b}|{m}", v1, definition=f"{METRIC_DEF.get(m, m)}; rule-only oracle on the whole "
                      "cal/test pool (no level)", status="secondary", split_mean50=row[f"{m}_mean50"],
                      split_sd50=row[f"{m}_sd50"], source=f"baselines/{ds}/{b}")
            row["precision_pooled"] = 1 - row["pooled_fdr"]
            N.add(f"oracle|{ds}|{b}|precision_pooled", row["precision_pooled"], definition="1 - pooled strict FDR",
                  status="secondary", source=f"baselines/{ds}/{b}")
            orows.append(row)
    if orows:
        atomic_write_parquet(pl.DataFrame(orows), out_dir / "oracles.parquet")
    # artifact probes
    pt = []
    for ds in cfg["probe_datasets"]:
        for role in cfg["probe_roles"]:
            pt.append({"kind": "probe", "ds": ds, "role": role, "cfg": cfg["probe"], "overwrite": a.overwrite})
    pres = pool_run(pt, min(a.workers, 4), "E15 probes")
    probes = []
    for r in pres:
        for k, v in r.get("aucs", {}).items():
            probes.append(v | {"key": k})
            expl = v.get("target") in ("i_attack", "ii_typematched")
            N.add(f"probe|{k}|auc", v.get("auc"), ci=(v.get("lo"), v.get("hi")),
                  definition="pooled out-of-fold AUC of an incident-grouped 5-fold LightGBM probe; "
                  "(i) campaign vs decoy unit, (ii) D3 look-alike vs native background unit, i_attack = campaign vs "
                  "D1/D2 attack decoys only, ii_typematched = D3 vs background units of the same unit types; feature "
                  "set in key (all = timing/position + the scorer's unit features); CI = incident-cluster bootstrap "
                  "B=200", status="exploratory" if expl else "secondary", n=v.get("n"), n_pos=v.get("n_pos"),
                  top_features=[[a, float(b)] for a, b in (v.get("top_features") or [])])
    atomic_write_json(out_dir / "probes.json", probes)
    # reuse table
    rr = []
    if getattr(a, "datasets", None):
        cfg["reuse_variants"] = [d for d in a.datasets if d.startswith("cf_")]
    for v in cfg["reuse_variants"]:
        p = P2_ROOT / "results" / "interim" / v / "audit" / "reuse.parquet"
        if p.exists():
            t = pl.read_parquet(p).with_columns(pl.lit(v).alias("variant"))
            rr.append(t)
            for pool in ("train", "caltest"):
                tt = t.filter(pl.col("pool") == pool)
                if tt.height:
                    w = tt["n_uses"].sum() / max(tt["n_used"].sum(), 1)
                    N.add(f"reuse|{v}|{pool}|incidents_per_used_episode", float(w),
                          definition="pooled incidents per used library episode (sum uses / sum used episodes)",
                          status="secondary", max_per_episode=int(tt["max_incidents_per_episode"].max()),
                          source=f"interim/{v}/audit/reuse.parquet")
        pb = P2_ROOT / "results" / "interim" / v / "audit" / "background_reuse.json"
        if pb.exists():
            bj = json.loads(pb.read_text())
            for pool, d in bj.items():
                N.add(f"reuse|{v}|{pool}|incidents_per_background_record", d.get("mean_incidents_per_record"),
                      definition="mean incidents per background record (windows overlap)", status="secondary",
                      source=f"interim/{v}/audit/background_reuse.json")
    if rr:
        atomic_write_parquet(pl.concat(rr, how="diagonal_relaxed"), out_dir / "reuse.parquet")
    # leakage audit
    leak = leakage_audit(cfg, getattr(a, "datasets", None))
    atomic_write_json(out_dir / f"leakage{'_' + a.tag if getattr(a, 'tag', '') else ''}.json", leak)
    N.add("leakage|tests", leak["summary"], definition="pytest -q of the leakage test files (P2-A / P2-F)",
          status="secondary", files=cfg["leakage_tests"])
    for v, ok in leak["source_disjoint"].items():
        N.add(f"leakage|source_disjoint|{v}", ok, definition="train/caltest library episode overlap == 0 "
              "(P2-E generator check)", status="secondary")
    N.add("leakage|gotham_bank_case_record_overlap", "mirai-infection 49,472/84,044 (59%), Merlin 30,018/46,033 (65%), "
          "mirai-dos 0/550, benign windows 10 and 0 flows", definition="raw records of the Gotham-real windows that also "
          "occur in the CF-GOTHAM-T bank (train-pool bots as context; P2-E measurement, reported not removed)",
          status="secondary", source="notes/P2-E.md")
    N.write()
    log(f"E15 numbers: {len(N.d)} entries")


def leakage_audit(cfg: dict, datasets=None) -> dict:
    files = [str(P2_ROOT / f) for f in cfg["leakage_tests"]]
    env = dict(os.environ)
    try:
        r = subprocess.run([sys.executable, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", *files],
                           capture_output=True, text=True, timeout=1800, env=env, cwd=str(P2_ROOT.parent))
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        res = {"returncode": r.returncode, "summary": tail}
    except Exception as e:  # pragma: no cover
        res = {"returncode": None, "summary": f"not run: {e}"}
    sd = {}
    for v in (datasets or ("cf_ton", "cf_ton_h", "cf_gotham", "cf_gotham_t")):
        p = P2_ROOT / "results" / "interim" / v / "summary.json"
        if p.exists():
            sd[v] = bool(json.loads(p.read_text()).get("source_disjoint", {}).get("ok"))
    res["source_disjoint"] = sd
    return res


# =========================================================================== E4
def e4(cfg: dict, a) -> None:
    ds, K, nt = cfg["dataset"], cfg["K"], cfg["n_test"]
    base = cfg["base"]
    if not X.has_bank(ds, base):
        log(f"E4: {ds}/{base} missing")
        return
    tasks = []
    variants = [(name, prm) for name, ps in cfg["cal_and_test"].items() for prm in ps]
    sids = [base] + [f"{base}__E4-{n}-{p:g}" for n, p in variants]
    for sid in sids:
        if not X.has_bank(ds, sid):
            log(f"E4: missing bank {sid}")
            continue
        for alpha in cfg["alphas"]:
            for m in ("crc_bh",):
                tasks.append({"kind": "cell", "exp": "E4", "ds": ds, "sid": sid, "method": m, "alpha": alpha, "K": K,
                              "n_test": nt, "splits": cfg["splits"], "overwrite": a.overwrite})
    for q in cfg["test_only_inject"]:
        tsid = f"{base}__E4-inject-{q}"
        if not X.has_bank(ds, tsid):
            log(f"E4: missing bank {tsid}")
            continue
        for alpha in cfg["alphas"]:
            for m in cfg["families"].values():
                tasks.append({"kind": "cell", "exp": "E4", "ds": ds, "sid": base, "method": m, "alpha": alpha, "K": K,
                              "n_test": nt, "splits": cfg["splits"], "test_sid": tsid, "overwrite": a.overwrite})
                tasks.append({"kind": "cell", "exp": "E4", "ds": ds, "sid": base, "method": m, "alpha": alpha,
                              "K": K, "n_test": nt, "splits": cfg["splits"], "overwrite": a.overwrite})
    res = pool_run(tasks, a.workers, "E4 cells")
    if not a.no_boot:
        bc = cfg["bootstrap"]
        bt = [{"kind": "boot", "cell": r["path"], "name": "main", "ds": r["task"]["ds"], "sid": r["task"]["sid"],
               "method": r["task"]["method"], "alpha": r["task"]["alpha"], "K": K, "n_test": nt,
               "test_sid": r["task"].get("test_sid"), "B": bc["B"], "boot_splits": bc["splits"], "seed": bc["seed"],
               "metrics": ["fdr", "fdr_lenient", "pooled_fdr", "str_mean", "str_pooled", "cer_link_mean", "n_claims"],
               "overwrite": a.overwrite} for r in res if "path" in r]
        pool_run(bt, a.workers, "E4 bootstraps")
    e4_numbers(cfg)


def e4_numbers(cfg: dict) -> None:
    from certichain.eval.splits import random_splits as rs
    N = X.Numbers("E4")
    idx = X.collect("E4")
    ds, base, K, nt = cfg["dataset"], cfg["base"], cfg["K"], cfg["n_test"]
    for r in idx.iter_rows(named=True):
        tsid = r.get("test_scorer_id")
        tag = r["scorer_id"].split("__E4-")[1] if "__E4-" in r["scorer_id"] else ("clean" if not tsid else "?")
        if tsid:
            tag = tsid.split("__E4-")[1]
        key = f"{r['ds']}|{tag}|{r['method']}|a={r['alpha']}"
        add_cell_numbers(N, key, Path(r["path"]), "secondary",
                         metrics=("fdr", "fdr_lenient", "pooled_fdr", "str_mean", "str_pooled", "cer_link_mean",
                                  "n_claims", "abstain", "load_ratio"))
        if tsid and r["method"] == cfg["families"]["threshold"]:
            q = int(tsid.rsplit("-", 1)[1])
            bank = X.load_bank(ds, base)
            tb = X.load_bank(ds, tsid)
            sp = rs(bank.n, K, range(cfg["splits"]), nt)
            sel = X.select(bank, r["method"], sp, r["alpha"], test_bank=tb)
            bnd = X.r3_bound_terms(tb, sel, q, r["alpha"])
            df, _ = X.read_cell(Path(r["path"]))
            within = float((df["fdr"].to_numpy() <= bnd + 1e-12).mean())
            N.add(f"{key}|r3_bound", float(np.mean(bnd)), definition="Remark R3 bound alpha + E[min(1, q/max(|S'|,1))] "
                  "for the score-threshold family (mean over splits of the per-split test-incident mean)",
                  status="secondary", share_splits_fdr_within_bound=within, q=q)
    N.write()
    log(f"E4 numbers: {len(N.d)} entries")


# =========================================================================== E3 (pre-registered primary endpoint 2)
def e3(cfg: dict, a) -> None:
    P, S = cfg["primary"], cfg["secondary"]
    alphas = P["alphas"]
    groups = [("primary", P["dataset"], P["K"], P["n_test"], P["stats"], P["fusion"], P["placebos"])]
    it = S["in_template"]
    groups.append(("in_template", it["dataset"], it["K"], it["n_test"], it["stats"], it["fusion"], it["placebos"]))
    tasks, pairs = [], []
    for role, ds, K, nt, s_stat, s_fuse, plac in groups:
        scorer = s_stat.split(".")[0]
        others = [f"{scorer}.{v}" for v in S["seeds"]] + [f"{scorer}.{v}" for v in S.get("kappa_variants", [])]
        sids = [s_stat, s_fuse] + list(plac) + others
        sids += [f"{s}__stratum-unseen" for s in (s_stat, s_fuse, *plac, *others)]
        avail = [s for s in sids if X.has_bank(ds, s)]
        if s_fuse not in avail or s_stat not in avail:
            log(f"E3 {role}: {ds} fusion/stats bank missing -> not run yet")
            continue
        for sid in avail:
            for alpha in alphas:
                tasks.append({"kind": "cell", "exp": "E3", "ds": ds, "sid": sid, "method": "crc_bh", "alpha": alpha,
                              "K": K, "n_test": nt, "splits": P["splits"], "overwrite": a.overwrite})
        for alpha in alphas:
            for sb in [s_fuse, *plac, *[o for o in others if o in avail]]:
                if sb in avail:
                    pairs.append((role, ds, K, nt, s_stat, sb, alpha, None))
            for strat in ("decoy_rich", "content_rich"):
                for sb in (s_fuse, *plac):
                    if sb in avail:
                        pairs.append((role, ds, K, nt, s_stat, sb, alpha, strat))
            su, fu = f"{s_stat}__stratum-unseen", f"{s_fuse}__stratum-unseen"
            for sb in (fu, *[f"{p}__stratum-unseen" for p in (*plac, *others)]):
                if su in avail and sb in avail:
                    pairs.append((role, ds, K, nt, su, sb, alpha, "unseen_type_pair"))
    res = pool_run(tasks, a.workers, "E3 cells")
    cells = {(r["task"]["ds"], r["task"]["sid"], r["task"]["alpha"]): r["path"] for r in res if "path" in r}
    if not a.no_boot:
        bc = P["bootstrap"]
        bt = []
        for role, ds, K, nt, sa, sb, alpha, strat in pairs:
            path = cells.get((ds, sb, alpha))
            if path is None:
                continue
            mask = None
            if strat == "decoy_rich":
                mask = {"col": "decoy_rich", "value": True}
            elif strat == "content_rich":
                mask = {"kind": "content_rich"}
            name = f"paired_vs_{sa}" + (f"_{strat}" if strat in ("decoy_rich", "content_rich") else "")
            bt.append({"kind": "boot", "cell": path, "name": name, "ds": ds, "sid": sa, "sid_b": sb, "method": "crc_bh",
                       "alpha": alpha, "K": K, "n_test": nt, "B": bc["B"], "boot_splits": bc["splits"],
                       "seed": bc["seed"], "mask": mask, "metrics": S["metrics"], "overwrite": a.overwrite})
        pool_run(bt, a.workers, "E3 paired bootstraps")
    e3_numbers(cfg)


_STRAT_CACHE: dict = {}


def stratum_frames(ds: str, sa: str, sb: str, alpha: float, strat: str, metrics) -> tuple[pl.DataFrame, pl.DataFrame]:
    """200-split metrics of CRC-BH restricted to the test incidents of a stratum (calibration unchanged), for the
    point estimates of stratum deltas (the bootstrap CIs come from the paired bootstrap with the same mask)."""
    d = X.DATASETS[ds]
    out = []
    for sid in (sa, sb):
        key = (ds, sid, alpha, strat)
        if key not in _STRAT_CACHE:
            bank = X.load_bank(ds, sid)
            sp = random_splits(bank.n, d["K"], range(X.N_SPLITS), d["n_test"])
            sel = X.select(bank, "crc_bh", sp, alpha)
            mcol = X._mask_from_spec(bank, ds, {"col": "decoy_rich", "value": True} if strat == "decoy_rich"
                                     else {"kind": "content_rich"})
            res = X.selection_metrics(bank, sel, alpha, mask=mcol[sel.test_rows], strata=())
            _STRAT_CACHE[key] = pl.DataFrame({m: np.asarray(res[m], float) for m in metrics if m in res})
        out.append(_STRAT_CACHE[key])
    return out[0], out[1]


UNSEEN_STRATUM_METRICS = ("str_mean", "str_pooled")


def e3_numbers(cfg: dict) -> None:
    P, S = cfg["primary"], cfg["secondary"]
    N = X.Numbers("E3")
    # drop stale non-STR entries of the unseen-type-pair stratum (their bank columns are not stratum-restricted)
    for k in [k for k in N.d if "|unseen_type_pair|delta_" in k and k.rsplit("delta_", 1)[1] not in UNSEEN_STRATUM_METRICS]:
        del N.d[k]
    out = {"prereg": X.load_prereg()["endpoints"]["primary_2_llm"], "groups": {}}
    groups = [("primary", P["dataset"], P["stats"], P["fusion"], P["placebos"]),
              ("in_template", S["in_template"]["dataset"], S["in_template"]["stats"], S["in_template"]["fusion"],
               S["in_template"]["placebos"])]
    for role, ds, s_stat, s_fuse, plac in groups:
        g = {}
        for alpha in P["alphas"]:
            pa = find_cell("E3", ds, s_stat, "crc_bh", alpha)
            if pa is None:
                continue
            dfa, _ = X.read_cell(pa)
            scorer = s_stat.split(".")[0]
            comps = [s_fuse, *plac] + [f"{scorer}.{v}" for v in S["seeds"]] + \
                    [f"{scorer}.{v}" for v in S.get("kappa_variants", [])]
            for sb in comps:
                pb = find_cell("E3", ds, sb, "crc_bh", alpha)
                if pb is None:
                    continue
                dfb, _ = X.read_cell(pb)
                for strat in (None, "decoy_rich", "content_rich", "unseen_type_pair"):
                    if strat == "unseen_type_pair":
                        ua, ub = f"{s_stat}__stratum-unseen", f"{sb}__stratum-unseen"
                        pua, pub = find_cell("E3", ds, ua, "crc_bh", alpha), find_cell("E3", ds, ub, "crc_bh", alpha)
                        if pua is None or pub is None:
                            continue
                        da_, _ = X.read_cell(pua)
                        db_, _ = X.read_cell(pub)
                        boot = X.read_bootstrap(pub, f"paired_vs_{ua}")
                    elif strat is None:
                        da_, db_ = dfa, dfb
                        boot = X.read_bootstrap(pb, f"paired_vs_{s_stat}")
                    else:
                        boot = X.read_bootstrap(pb, f"paired_vs_{s_stat}_{strat}")
                        if boot is None:
                            continue
                        da_, db_ = stratum_frames(ds, s_stat, sb, alpha, strat, S["metrics"])
                    for m in S["metrics"]:
                        if strat == "unseen_type_pair" and m not in UNSEEN_STRATUM_METRICS:
                            continue                 # only the STR columns of the stratum banks are stratum-restricted
                        key = f"{role}|{ds}|{sb}-{s_stat}|a={alpha}|{strat or 'all'}|delta_{m}"
                        point = None
                        if da_ is not None and m in da_.columns and m in db_.columns:
                            point = float(np.nanmean(db_[m].to_numpy()) - np.nanmean(da_[m].to_numpy()))
                        bm = (boot or {}).get("metrics", {}).get(m)
                        ci = se = None
                        if bm:
                            d = bm["delta"]
                            ci, se = (d["lo"], d["hi"]), d["se"]
                            if point is None:
                                point = None
                        status = "primary" if (role == "primary" and sb == s_fuse and strat is None and m == P["metric"]) \
                            else ("primary" if role == "primary" and sb in plac and strat is None and m == P["metric"]
                                  else ("secondary" if strat in (None, "unseen_type_pair", "decoy_rich", "content_rich")
                                        else "exploratory"))
                        sdef = {None: "all test incidents",
                                "decoy_rich": "test incidents with >= 2 decoy campaigns (calibration unchanged; all metrics "
                                              "restricted to those incidents)",
                                "content_rich": "test incidents with >= 1 content-flagged unit (calibration unchanged; all "
                                                "metrics restricted to those incidents)",
                                "unseen_type_pair": "PRE-REGISTERED STRATUM: STR restricted to observable transitions whose "
                                                    "(type_u, type_v) unit-type pair has no training positive (P2-F "
                                                    "__stratum-unseen banks; only their STR columns are stratum-restricted, "
                                                    "so only STR is reported for this stratum)"}[strat]
                        N.add(key, point, ci=ci, se=se, status=status,
                              definition=f"Delta {m} = {m}(CRC-BH, {sb}) - {m}(CRC-BH, {s_stat}), 200 splits; "
                              f"CI: paired incident-pool bootstrap (same resampled incidents and splits), "
                              f"B=1000 x 50 splits, percentile 95%; stratum: {sdef}",
                              p_one_sided=(bm or {}).get("delta", {}).get("p_one_sided_le0") if bm else None,
                              stratum_mean_a=(bm["a"].get("estimate") if bm and "estimate" in bm["a"] else None))
                        g.setdefault(f"{sb}|{alpha}|{strat or 'all'}", {})[m] = {"point": point, "ci": ci, "se": se,
                                                                              "p": (bm or {}).get("delta", {}).get("p_one_sided_le0")}
        out["groups"][role] = g
    # the pre-registered decision (primary group only)
    reps, plac_ok = {}, {}
    ds, s_stat, s_fuse = P["dataset"], P["stats"], P["fusion"]
    for alpha in P["alphas"]:
        pb = find_cell("E3", ds, s_fuse, "crc_bh", alpha)
        if pb is None or not (pb / f"bootstrap_paired_vs_{s_stat}_b.npy").exists():
            continue
        js = X.read_bootstrap(pb, f"paired_vs_{s_stat}")
        j = js["metrics"]
        mi = list(j).index(P["metric"])
        ra = np.load(pb / f"bootstrap_paired_vs_{s_stat}_a.npy")[:, mi]
        rb = np.load(pb / f"bootstrap_paired_vs_{s_stat}_b.npy")[:, mi]
        reps[f"{ds}|alpha={alpha}"] = rb - ra
        for pl_ in P["placebos"]:
            pp = find_cell("E3", ds, pl_, "crc_bh", alpha)
            bpl = X.read_bootstrap(pp, f"paired_vs_{s_stat}") if pp else None
            if bpl:
                d = bpl["metrics"][P["metric"]]["delta"]
                plac_ok[f"{pl_}|alpha={alpha}"] = {"lo": d["lo"], "hi": d["hi"], "contains_0": d["lo"] <= 0 <= d["hi"]}
            else:
                plac_ok[f"{pl_}|alpha={alpha}"] = None
    if reps:
        hb = X.holm_bootstrap(reps, P["holm_level"])
        complete = len(reps) == len(P["alphas"]) and all(v is not None for v in plac_ok.values()) and \
            len(plac_ok) == len(P["alphas"]) * len(P["placebos"])
        placebos_null = complete and all(v["contains_0"] for v in plac_ok.values())
        any_rej = any(v["reject"] for v in hb.values())
        dec = {"holm": hb, "placebos": plac_ok, "complete": complete,
               "success": bool(complete and any_rej and placebos_null),
               "success_rule": "PREREG: Holm-adjusted lower confidence bound of Delta > 0 AND both placebos' two-sided "
                               "95% CIs contain 0 at both alphas; reported per alpha (success = at least one alpha "
                               "rejected by Holm with the placebo condition met)",
               "rejected_alphas": [k for k, v in hb.items() if v["reject"]], "placebos_null": placebos_null}
        out["decision"] = dec
        for k, v in hb.items():
            N.add(f"primary2|{k}|holm", v["lcb_holm"], definition="Holm-adjusted one-sided lower confidence bound of "
                  "Delta CER_link (percentile of the paired bootstrap at the Holm level)", status="primary",
                  p_boot=v["p_boot"], p_holm=v["p_holm"], reject=v["reject"], holm_level=v["holm_level"],
                  lcb_normal=v["lcb_holm_normal"])
        N.add("primary2|success", dec["success"], definition=dec["success_rule"], status="primary",
              complete=complete, placebos_null=placebos_null, rejected=dec["rejected_alphas"])
    # cost: realised kappa, LLM tokens / latency per incident (cal/test band shards of the frozen main model)
    for ds, sid in ((P["dataset"], P["stats"]), (S["in_template"]["dataset"], S["in_template"]["stats"])):
        llm_dir = X.results_root() / "llm" / ds / "caltest_band"
        parts = sorted(llm_dir.glob("*/*/part-*.parquet"))
        if not parts or not X.has_bank(ds, sid):
            continue
        sh = pl.concat([pl.read_parquet(p, columns=["incident", "cand_id", "n_tok", "ms"]) for p in parts])
        sh = sh.unique(["incident", "cand_id"])
        per = sh.group_by("incident").agg(pl.len().alias("calls"), pl.col("n_tok").sum().alias("tok"),
                                          pl.col("ms").sum().alias("ms"))
        bank = X.load_bank(ds, sid)
        Nmap = dict(zip([X.inc_of(i) for i in bank.index["incident_id"].to_list()], bank.col("N").tolist()))
        per = per.with_columns(pl.col("incident").replace_strict(Nmap, default=None).alias("N"))
        tot_N = float(sum(Nmap.values()))
        kap = float(per["calls"].sum()) / tot_N if tot_N else None
        N.add(f"cost|{ds}|kappa_realised", kap, definition="LLM calls / candidates over the cal/test pool (cascade "
              "band, kappa_target 0.1)", status="secondary", calls=int(per["calls"].sum()), candidates=tot_N)
        N.add(f"cost|{ds}|tokens_per_incident", float(per["tok"].sum()) / len(Nmap), definition="mean LLM prompt-suffix "
              "tokens per incident (the shared prompt prefix is cached and not counted)", status="secondary",
              median_incident=float(per["tok"].median()), tokens_per_call=float(sh["n_tok"].mean()))
        N.add(f"cost|{ds}|ms_per_call", float(sh["ms"].mean()), definition="mean LLM latency per candidate (M3 Ultra, "
              "MEC-class node)", status="secondary", seconds_per_incident=float(per["ms"].sum()) / 1000 / len(Nmap))
    atomic_write_json(X.summary_root() / "primary2_llm.json", out)
    N.write()
    log(f"E3 numbers: {len(N.d)} entries; decision: {out.get('decision', {}).get('success')}")


# =========================================================================== E2
def e2(cfg: dict, a) -> None:
    tasks = []
    for ds, d in cfg["datasets"].items():
        if a.only and ds not in a.only:
            continue
        for kind in cfg["fine_grid_scorers"]:
            sid = X.scorer_id(ds, kind)
            if not X.has_bank(ds, sid):
                continue
            for alpha in X.FINE_GRID:
                for m in cfg["fine_grid_methods"]:
                    tasks.append({"kind": "cell", "exp": "E2", "ds": ds, "sid": sid, "method": m, "alpha": alpha,
                                  "K": d["K"], "n_test": d["n_test"], "splits": cfg["splits"], "overwrite": a.overwrite,
                                  "cert_gap": False})
        lsid = X.scorer_id(ds, cfg["lag_llr_scorer"])
        if X.has_bank(ds, lsid):
            for alpha in cfg["alphas"]:
                tasks.append({"kind": "cell", "exp": "E2", "ds": ds, "sid": lsid, "method": "crc_bh", "alpha": alpha,
                              "K": d["K"], "n_test": d["n_test"], "splits": cfg["splits"], "overwrite": a.overwrite})
    pool_run(tasks, a.workers, "E2 cells")
    for ds, d in cfg["datasets"].items():
        if a.only and ds not in a.only:
            continue
        for name in [*cfg["fixed_baselines"], "oracle"]:
            if name != "oracle" and not X.has_baseline(ds, name):
                continue
            if name == "B5":
                continue
            for alpha in (0.1, 0.2):
                X.run_fixed_cell("E2", ds, name, alpha, K=d["K"], n_test=d["n_test"], seeds=range(cfg["splits"]),
                                 overwrite=a.overwrite)
    e2_numbers(cfg)


def e2_numbers(cfg: dict) -> None:
    from certichain.eval.metrics import claims_to_reach
    N = X.Numbers("E2")
    idx = X.collect("E2")
    curves = []
    for r in idx.iter_rows(named=True):
        if r["scorer_id"] == "fixed" or r["method"] == "crc_bh" and r["scorer_id"].endswith("lag_llr"):
            key = f"{r['ds']}|{r['scorer_id']}|{r['method']}|a={r['alpha']}"
            add_cell_numbers(N, key, Path(r["path"]), "secondary")
        else:
            curves.append(r)
    cdf = pl.DataFrame(curves, infer_schema_length=None) if curves else pl.DataFrame()
    if cdf.height:
        atomic_write_parquet(cdf, X.exp_root("E2") / "fine_grid_curves.parquet")
        # precision matching: at target realised FDR f*, the largest grid alpha with realised FDR <= f*
        for (ds, sid, m), sub in cdf.group_by(["ds", "scorer_id", "method"], maintain_order=True):
            sub = sub.sort("alpha")
            for f in cfg["precision_targets"]:
                ok = sub.filter(pl.col("fdr") <= f + 1e-12)
                if ok.height == 0:
                    N.add(f"pm|{ds}|{sid}|{m}|fdr<={f}", None, definition="no fine-grid alpha reaches this realised FDR",
                          status="secondary")
                    continue
                best = ok.sort("alpha").tail(1).row(0, named=True)
                N.add(f"pm|{ds}|{sid}|{m}|fdr<={f}|str_mean", best["str_mean"],
                      definition="precision matching: STR at the largest fine-grid alpha whose realised strict FDR <= target",
                      status="secondary", alpha=best["alpha"], fdr=best["fdr"], pooled_fdr=best["pooled_fdr"],
                      cer_link_mean=best["cer_link_mean"], load_ratio=best["load_ratio"], source=best["path"])
    # fixed baselines vs CRC-BH at matched FDR (the baseline's own realised FDR)
    for ds in cfg["datasets"]:
        for name in [b for b in cfg["fixed_baselines"] if b != "B5"]:
            pb = find_cell("E2", ds, "fixed", name, 0.1)
            if pb is None:
                continue
            dfb, _ = X.read_cell(pb)
            fb = float(dfb["fdr"].mean())
            for kind in cfg["fine_grid_scorers"]:
                sid = X.scorer_id(ds, kind)
                sub = cdf.filter((pl.col("ds") == ds) & (pl.col("scorer_id") == sid) & (pl.col("method") == "crc_bh")) \
                    if cdf.height else cdf
                if not sub.height:
                    continue
                ok = sub.filter(pl.col("fdr") <= fb + 1e-12).sort("alpha")
                best = (ok if ok.height else sub.sort("alpha")).tail(1).row(0, named=True)
                N.add(f"pm_fixed|{ds}|{name}|vs_crc_bh|{sid}", best["str_mean"],
                      definition="CRC-BH STR at the largest fine-grid alpha whose realised FDR <= the baseline's "
                      "realised FDR; grid_exhausted = the baseline's FDR exceeds CRC-BH's realised FDR at every grid "
                      "alpha (<= 0.5), i.e. CRC-BH cannot be run at the baseline's error rate on this grid",
                      status="secondary", baseline_fdr=fb, baseline_str=float(dfb["str_mean"].mean()),
                      crc_alpha=best["alpha"], crc_fdr=best["fdr"],
                      grid_exhausted=bool(abs(best["alpha"] - max(X.FINE_GRID)) < 1e-9))
    # B5 (LLM only, s_LLM > 0, kappa = 1 arm): one run over its 15 incidents + CRC-BH restricted to them
    if X.has_baseline("cf_ton", "B5") and X.has_bank("cf_ton", "S_TON.stat"):
        bank = X.load_bank("cf_ton", "S_TON.stat")
        cnt = X.baseline_counts(bank, "cf_ton", "B5")
        rows = np.flatnonzero(np.isfinite(cnt["R"]))
        one = X.Splits(np.array([0]), np.zeros((1, 0), np.int64), rows[None, :])
        r1 = X.fixed_metrics(bank, cnt, one, 0.1)
        for m in ("fdr", "fdr_lenient", "pooled_fdr", "str_mean", "str_pooled", "cer_link_mean", "n_claims", "ucr"):
            N.add(f"cf_ton|fixed|B5|{m}", float(np.nanmean(r1[m])), definition=f"B5 LLM-only (s_LLM > 0 on every "
                  f"candidate of the 15 kappa = 1 arm incidents), one run: {METRIC_DEF.get(m, m)}", status="secondary",
                  n_incidents=int(rows.size))
        lr = float(r1["sum_R"].sum() / r1["sum_TP"].sum()) if r1["sum_TP"].sum() > 0 else None
        N.add("cf_ton|fixed|B5|load_ratio", lr, definition="B5 analyst load (sum R / sum TP over its 15 incidents)",
              status="secondary")
        mask = np.zeros(bank.n, bool)
        mask[rows] = True
        for sid in ("S_TON.stat", "S_TON.fuse_s0"):
            if not X.has_bank("cf_ton", sid):
                continue
            bk = X.load_bank("cf_ton", sid)
            sp = random_splits(bk.n, 150, range(200), 150)
            for alpha in (0.1, 0.2):
                sel = X.select(bk, "crc_bh", sp, alpha)
                res = X.selection_metrics(bk, sel, alpha, mask=mask[sel.test_rows], strata=())
                for m in ("fdr", "pooled_fdr", "str_mean", "cer_link_mean", "n_claims"):
                    N.add(f"cf_ton|B5arm|{sid}|crc_bh|a={alpha}|{m}", float(np.nanmean(res[m])),
                          definition=f"CRC-BH restricted to the 15 B5-arm incidents when they are test incidents "
                          f"(200 splits): {METRIC_DEF.get(m, m)}", status="secondary")
    # ranked analyst load: candidates inspected to reach 80% observable STR (ranked by the scorer / lag-LLR)
    for ds in cfg["datasets"]:
        for kind in ("stat", "fuse_s0", "lag_llr"):
            sid = X.scorer_id(ds, kind)
            if not X.has_bank(ds, sid):
                continue
            bank = X.load_bank(ds, sid)
            c = claims_to_reach(bank, np.arange(bank.n), cfg["ranked_frac"])
            f = c[np.isfinite(c)]
            nobs = bank.col("n_trans_obs")
            N.add(f"ranked_load|{ds}|{sid}", float(np.median(f)) if f.size else None,
                  definition="median over incidents of the number of top-ranked candidates to inspect until 80% of "
                  "|T*_obs| is hit (ranked baseline analyst load); NaN where unreachable within C", status="secondary",
                  mean=float(f.mean()) if f.size else None, n_reachable=int(f.size),
                  n_with_T_obs=int(np.sum(nobs > 0)), total_inspected=float(f.sum()) if f.size else None)
    N.write()
    log(f"E2 numbers: {len(N.d)} entries")


# =========================================================================== E5
def e5(cfg: dict, a) -> None:
    ds_override(cfg, a)
    tasks = []
    for ds, d in cfg["datasets"].items():
        sid = X.scorer_id(ds, "stat")
        if not X.has_bank(ds, sid):
            continue
        for alpha in cfg["alphas"]:
            for m in cfg["methods"]:
                tasks.append({"kind": "cell", "exp": "E5", "ds": ds, "sid": sid, "method": m, "alpha": alpha,
                              "K": d["K"], "n_test": d["n_test"], "splits": cfg["splits"], "overwrite": a.overwrite})
    res = pool_run(tasks, a.workers, "E5 cells")
    if not a.no_boot:
        bc = cfg["bootstrap"]
        bt = [{"kind": "boot", "cell": r["path"], "name": "main", "ds": r["task"]["ds"], "sid": r["task"]["sid"],
               "method": r["task"]["method"], "alpha": r["task"]["alpha"], "K": r["task"]["K"],
               "n_test": r["task"]["n_test"], "B": bc["B"], "boot_splits": bc["splits"], "seed": bc["seed"],
               "metrics": ["fdr", "fdr_lenient", "maxchain_fdr", "maxchain_fdr_lenient", "prefix_fdr", "chain_fwer",
                           "ccr_mean", "ccr_pooled", "str_mean", "n_chains"], "overwrite": a.overwrite}
              for r in res if "path" in r]
        pool_run(bt, a.workers, "E5 bootstraps")
    # by GT chain length L (strata of test incidents; calibration unchanged)
    out_rows = []
    for ds, d in cfg["datasets"].items():
        sid = X.scorer_id(ds, "stat")
        if not X.has_bank(ds, sid):
            continue
        bank = X.load_bank(ds, sid)
        L = X.chain_length_strata(ds, bank)
        atomic_write_parquet(pl.DataFrame({"incident_id": bank.index["incident_id"], "L": L}),
                             X.exp_root("E5") / ds / "chain_length.parquet")
        sp = random_splits(bank.n, d["K"], range(cfg["splits"]), d["n_test"])
        for alpha in cfg["alphas"]:
            for m in cfg["methods"]:
                sel = X.select(bank, m, sp, alpha)
                for lb in cfg["L_bins"]:
                    mask = np.isin(L, lb)[sel.test_rows]
                    res_ = X.selection_metrics(bank, sel, alpha, mask=mask, strata=())
                    row = {"ds": ds, "method": m, "alpha": alpha, "L_bin": "+".join(map(str, lb)) if len(lb) == 1
                           else f"{lb[0]}+", "L_min": lb[0], "n_incidents_pool": int(np.isin(L, lb).sum())}
                    for k in ("fdr", "fdr_lenient", "maxchain_fdr", "maxchain_fdr_lenient", "prefix_fdr", "chain_fwer",
                              "ccr_mean", "ccr_pooled", "str_mean", "n_chains", "n_claims"):
                        v = np.asarray(res_.get(k, np.full(sp.S, np.nan)), float)
                        row[k] = float(np.nanmean(v)) if np.isfinite(v).any() else None
                        row[f"{k}_sd"] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else None
                    out_rows.append(row)
    if out_rows:
        new_df = pl.DataFrame(out_rows, infer_schema_length=None)
        bl = X.exp_root("E5") / "by_length.parquet"
        if bl.exists():
            old_df = pl.read_parquet(bl).filter(~pl.col("ds").is_in(new_df["ds"].unique().to_list()))
            new_df = pl.concat([old_df, new_df], how="diagonal_relaxed")
        atomic_write_parquet(new_df, bl)
    if not a.no_boot and out_rows:
        bc = cfg["bootstrap"]
        cells = {(r["task"]["ds"], r["task"]["method"], r["task"]["alpha"]): r for r in res if "path" in r}
        bt = []
        for row in out_rows:
            if row["n_incidents_pool"] < 10 or row["method"] not in ("chain_composite", "chain_naive", "chain_rt"):
                continue
            r = cells.get((row["ds"], row["method"], row["alpha"]))
            if r is None or row["alpha"] not in (0.1, 0.2):
                continue
            lb = next(b for b in cfg["L_bins"] if b[0] == row["L_min"])
            bt.append({"kind": "boot", "cell": r["path"], "name": f"L_{row['L_bin']}", "ds": row["ds"],
                       "sid": r["task"]["sid"], "method": row["method"], "alpha": row["alpha"], "K": r["task"]["K"],
                       "n_test": r["task"]["n_test"], "B": bc["B"], "boot_splits": bc["splits"], "seed": bc["seed"],
                       "mask": {"kind": "chain_len", "L": lb},
                       "metrics": ["fdr", "maxchain_fdr", "prefix_fdr", "ccr_mean", "str_mean"], "overwrite": a.overwrite})
        pool_run(bt, a.workers, "E5 stratum bootstraps")
    ex = cfg["example1"]
    exs = [X.example1_curve(ex["L"], n=ex["n"], K=ex["K"], alpha=al, n_incidents=ex["n_incidents"], splits=ex["splits"])
           for al in ex["alphas"]]
    atomic_write_parquet(pl.concat(exs), X.exp_root("E5") / "example1.parquet")
    e5_numbers(cfg)


def e5_numbers(cfg: dict) -> None:
    N = X.Numbers("E5")
    idx = X.collect("E5")
    for r in idx.iter_rows(named=True):
        key = f"{r['ds']}|{r['scorer_id']}|{r['method']}|a={r['alpha']}"
        add_cell_numbers(N, key, Path(r["path"]), "secondary",
                         metrics=("fdr", "fdr_lenient", "maxchain_fdr", "maxchain_fdr_lenient", "prefix_fdr", "chain_fwer",
                                  "ccr_mean", "ccr_pooled", "str_mean", "n_chains", "n_claims", "abstain"))
    p = X.exp_root("E5") / "by_length.parquet"
    if p.exists():
        for r in pl.read_parquet(p).iter_rows(named=True):
            cp = find_cell("E5", r["ds"], X.scorer_id(r["ds"], "stat"), r["method"], r["alpha"])
            boot = X.read_bootstrap(cp, f"L_{r['L_bin']}") if cp is not None else None
            for k in ("maxchain_fdr", "prefix_fdr", "fdr", "ccr_mean"):
                bm = (boot or {}).get("metrics", {}).get(k)
                N.add(f"byL|{r['ds']}|{r['method']}|a={r['alpha']}|L={r['L_bin']}|{k}", r[k],
                      definition=f"{METRIC_DEF.get(k, k)} over test incidents whose GT longest root-leaf path has "
                      f"L in {r['L_bin']} (calibration on all incidents), 200 splits; CI: incident-pool bootstrap with "
                      f"the stratum mask (B=1000 x 50)", status="secondary", ci=(bm["lo"], bm["hi"]) if bm else None,
                      se=bm["se"] if bm else None, split_sd=r.get(f"{k}_sd"), n_incidents_pool=r["n_incidents_pool"])
    p = X.exp_root("E5") / "example1.parquet"
    if p.exists():
        for r in pl.read_parquet(p).iter_rows(named=True):
            for k in ("chain_naive|maxchain_fdr", "chain_naive|fdr", "chain_composite|maxchain_fdr",
                      "chain_composite|fdr", "chain_rt|chain_fwer", "theory_naive_maxchain_fdr", "asymptote"):
                N.add(f"example1|a={r['alpha']}|L={r['L']}|{k}", r[k], definition="Example 1 simulation (P2-C "
                      "eval/synth.example1_incident; n = 20 chains, K = 150, 5 splits)", status="secondary", f=r["f"])
    N.write()
    log(f"E5 numbers: {len(N.d)} entries")


# =========================================================================== E11
def e11(cfg: dict, a) -> None:
    ic = cfg["iot23"]
    out_root = X.exp_root("E11")
    N = X.Numbers("E11")
    sids = [(X.scorer_id("iot23", k), k, "primary") for k in ic["scorers"]] + \
        [(s_, s_.split(".", 1)[1], "sensitivity") for s_ in ic.get("extra_sids", [])]
    for sid, kind, role_ in sids:
        if not X.has_bank("iot23", sid):
            log(f"E11: iot23/{sid} missing")
            continue
        bank = X.load_bank("iot23", sid)
        g = X.scenario_groups(bank)
        for alpha in ic["alphas"]:
            meths = list(ic["loso_methods"]) + (["B2"] if (kind == "stat" and role_ == "primary") else [])
            for m in meths:
                p = out_root / "iot23" / "loso" / f"{m}__{sid}" / f"a{alpha}"
                if (p / "jackknife.json").exists() and not a.overwrite:
                    j = json.loads((p / "jackknife.json").read_text())
                else:
                    if m == "B2":
                        if not X.has_baseline("iot23", "B2"):
                            continue
                        fixed = X.baseline_counts(bank, "iot23", "B2")
                        jk = X.jackknife_loso(bank, alpha, method="fixed", fixed=fixed)
                    else:
                        jk = X.jackknife_loso(bank, alpha, method=m)
                    per = jk.pop("per_scenario")
                    p.mkdir(parents=True, exist_ok=True)
                    atomic_write_parquet(per, p / "per_scenario.parquet")
                    jl = {}
                    for metric in ("fdp_lenient",):
                        jj = X.jackknife_loso(bank, alpha, method="fixed" if m == "B2" else m,
                                              fixed=X.baseline_counts(bank, "iot23", "B2") if m == "B2" else None,
                                              metric=metric)
                        jj.pop("per_scenario")
                        jl[metric] = jj
                    tot_tr = float(per["sum_TR"].sum())
                    tot_T = float(per["n_T"].sum())
                    j = {**jk, "lenient": jl["fdp_lenient"], "method": m, "scorer_id": sid, "alpha": alpha,
                         "str_pooled": tot_tr / tot_T if tot_T else None,
                         "ceiling_pooled": float(per["n_T_obs_in_C"].sum() / tot_T) if tot_T else None,
                         "rho_T_pooled": float(per["n_T_hid"].sum() / tot_T) if tot_T else None,
                         "r_CT_pooled": float(per["n_T_obs_in_C"].sum() / max(per["n_T_obs"].sum(), 1)),
                         "n_abstain": int(per["abstain"].fill_null(False).sum()) if m in ("crc_bh",) else None,
                         "sum_R": float(per["sum_R"].sum()), "sum_TP": float(per["sum_TP"].sum()),
                         "any_claim_benign_scenarios": float(per.filter(pl.col("benign_only"))["any_claim"].mean())
                         if per.filter(pl.col("benign_only")).height else None,
                         "prereg_blake3": X.PREREG_BLAKE3, "K": len(set(g.tolist())) - 1}
                    atomic_write_json(p / "jackknife.json", j)
                st = "primary" if (m == "crc_bh" and kind in ("stat", "fuse_s0") and role_ == "primary") else "secondary"
                N.add(f"loso|{sid}|{m}|a={alpha}|fdr", j["estimate"], ci=(j["lo"], j["hi"]), se=j["se_jack"],
                      definition="P-LOSO: mean over the 23 held-out scenarios of the scenario FDP (mean strict window "
                      "FDP), K = 22 scenarios; delete-one-scenario jackknife CI (t_22)", status=st,
                      se_naive=j["se_naive"], source=X.rel(p))
                N.add(f"loso|{sid}|{m}|a={alpha}|fdr_lenient", j["lenient"]["estimate"],
                      ci=(j["lenient"]["lo"], j["lenient"]["hi"]), se=j["lenient"]["se_jack"],
                      definition="P-LOSO lenient (AMB excluded)", status="secondary", source=X.rel(p))
                for k in ("str_pooled", "ceiling_pooled", "rho_T_pooled", "r_CT_pooled", "n_abstain",
                          "any_claim_benign_scenarios", "sum_R", "sum_TP"):
                    N.add(f"loso|{sid}|{m}|a={alpha}|{k}", j.get(k), definition=f"P-LOSO {k}", status="secondary",
                          source=X.rel(p))
        # P-SHIFT (stress; grouped scenario splits)
        ps = ic["p_shift"]
        sp = X.shift_splits(g, ps["splits"], salt=ps["salt"])
        for alpha in ic["alphas"]:
            for m in ("crc_bh", "bh_iid"):
                df, meta, path = X.run_cell("E11", "iot23", sid, m, alpha, K=-1, n_test=None, splits=sp,
                                            split_tag=f"P-SHIFT:ceil(G/2) scenarios cal, salt {ps['salt']}",
                                            overwrite=a.overwrite)
                add_cell_numbers(N, f"pshift|{sid}|{m}|a={alpha}", path, "secondary",
                                 metrics=("fdr", "fdr_lenient", "pooled_fdr", "str_mean", "str_pooled", "abstain",
                                          "n_claims", "wrongful_benign", "load_ratio"))
        # P-WIN cells live in E1 (window-level 50/50, 200 splits): referenced here (run in E11 for extra scorers)
        for alpha in ic["alphas"]:
            for m in ("crc_bh", "bh_iid"):
                pth = find_cell("E1", "iot23", sid, m, alpha)
                if pth is None and role_ == "sensitivity":
                    _, _, pth = X.run_cell("E11", "iot23", sid, m, alpha, K=ic["p_win"]["K"], n_test=ic["p_win"]["n_test"],
                                           seeds=range(ic["p_win"]["splits"]), overwrite=a.overwrite)
                if pth is not None:
                    add_cell_numbers(N, f"pwin|{sid}|{m}|a={alpha}", pth, "secondary",
                                     metrics=("fdr", "fdr_lenient", "pooled_fdr", "str_mean", "str_pooled", "abstain",
                                              "n_claims", "wrongful_benign", "load_ratio"))
        if kind == "stat" and X.has_baseline("iot23", "B2"):
            for alpha in (0.1,):
                df, meta, path = X.run_fixed_cell("E11", "iot23", "B2", alpha, K=83, n_test=83, overwrite=a.overwrite)
                add_cell_numbers(N, f"pwin|fixed|B2", path, "secondary",
                                 metrics=("fdr", "fdr_lenient", "pooled_fdr", "str_mean", "str_pooled", "n_claims",
                                          "wrongful_benign", "load_ratio"))
    N.write()
    e11_gotham(cfg, a)
    log("E11 done")


def e11_gotham(cfg: dict, a) -> None:
    gc = cfg["gotham"]
    if getattr(a, "gotham_cal", ""):
        d_, s_ = a.gotham_cal.split(":")
        gc["calibration"] = {"ds": d_, "scorer_id": s_}
    if getattr(a, "gotham_test", ""):
        d_, s_ = a.gotham_test.split(":")
        gc["test"] = {"ds": d_, "scorer_id": s_}
    tag = getattr(a, "tag", "")
    GK = "gotham" + (f"_{tag}" if tag else "")
    N = X.Numbers("E11")
    out = X.exp_root("E11") / GK
    out.mkdir(parents=True, exist_ok=True)
    # GT side by side (seed-priority = the bank's labels; default rule from P2-B's stats)
    from certichain.scoring import driver as D
    side = []
    for inc in D.list_incidents("gotham_real"):
        st = D.load_stats("gotham_real", inc)
        for rule, key in (("seed_priority", "transition_stats"), ("default", "transition_stats_default_rule")):
            t = st.get(key) or {}
            side.append({"incident": inc, "rule": rule, **{k: t.get(k) for k in
                         ("n_T", "n_T_obs", "n_T_hid", "n_T_obs_in_C", "r_CT", "rho_T", "ceiling", "n_bridges",
                          "n_bridges_in_C", "n_E_link", "n_candidates", "n_pos", "n_null", "n_amb", "n_pos_bridge")}})
    sdf = pl.DataFrame(side, infer_schema_length=None)
    atomic_write_parquet(sdf, out / "gt_rules_side_by_side.parquet")
    for rule in ("seed_priority", "default"):
        s = sdf.filter(pl.col("rule") == rule)
        nT, nTo, nTh, nin = (float(s[c].fill_null(0).sum()) for c in ("n_T", "n_T_obs", "n_T_hid", "n_T_obs_in_C"))
        N.add(f"{GK}|gt|{rule}|rho_T", nTh / nT if nT else None, definition="pooled rho_T over the 9 Gotham-real incidents",
              status="secondary", n_T=nT, n_T_obs=nTo, n_T_obs_in_C=nin,
              ceiling=nin / nT if nT else None, n_pos=float(s["n_pos"].fill_null(0).sum()),
              n_amb=float(s["n_amb"].fill_null(0).sum()))
    ov = gc["record_overlap_reported"]
    for k, (x, n) in ov.items():
        N.add(f"{GK}|record_overlap|{k}", x / n if n else None, definition="share of the case-study window's raw "
              "records that also occur in the CF-GOTHAM-T bank (P2-E measurement)", status="secondary", records=x, of=n)
    cal_ds, cal_sid = gc["calibration"]["ds"], gc["calibration"]["scorer_id"]
    te_ds, te_sid = gc["test"]["ds"], gc["test"]["scorer_id"]
    if not X.has_bank(cal_ds, cal_sid) or not X.has_bank(te_ds, te_sid):
        N.add(f"{GK}|status", "calibration bank missing", definition=f"needs losses/{cal_ds}/{cal_sid} (DESIGN §5.1.3: "
              "scorer S_TON, bank CF-GOTHAM-T); requested from P2-F", status="secondary")
        N.write()
        log(f"E11 gotham: calibration bank {cal_ds}/{cal_sid} missing")
        return
    cb, tb = X.load_bank(cal_ds, cal_sid), X.load_bank(te_ds, te_sid)
    rows = []
    ydef = {}
    for iid in tb.index["incident_id"].to_list():
        lab = X.gotham_default_labels(X.inc_of(iid))
        ydef[iid] = dict(zip(lab["cand_id"].to_list(), lab["y"].to_list()))
    for alpha in gc["alphas"]:
        col = X.crc_full_bank(cb, alpha)
        R = tb.t("R", "bh")[:, col].astype(np.int64)
        c = tb.counts(np.arange(tb.n), R)
        for k, iid in enumerate(tb.index["incident_id"].to_list()):
            Rk, Vk, Ak, TPk = int(c["R"][k]), int(c["V"][k]), int(c["A"][k]), int(c["TP"][k])
            sel = tb.csr["cand"][tb.slice(k)][:Rk]
            yd = np.array([ydef[iid][int(x)] for x in sel], dtype=np.int64)
            Vd, Ad = int((yd != 1).sum()), int((yd == -1).sum())
            lo_d, hi_d = X.clopper_pearson(Vd, Rk)
            lo, hi = X.clopper_pearson(Vk, Rk)
            V0 = Vk - Ak
            lo_l, hi_l = X.clopper_pearson(V0, Rk - Ak)
            nT = float(tb.col("n_trans")[k])
            rows.append({"alpha": alpha, "incident": X.inc_of(iid), "lam_idx": col, "lam": float(cb.grids["bh"][col]),
                         "K": cb.n, "R": Rk, "V": Vk, "A": Ak, "TP": TPk, "fdp": Vk / max(Rk, 1), "fdp_cp_lo": lo,
                         "fdp_cp_hi": hi, "fdp_lenient": V0 / max(Rk - Ak, 1), "fdp_lenient_cp_lo": lo_l,
                         "fdp_lenient_cp_hi": hi_l, "TR": int(c["TR"][k]), "n_T": nT,
                         "str": int(c["TR"][k]) / nT if nT else None,
                         "ceiling": float(X.transitions_in_C(tb)[k] / nT) if nT else None,
                         "benign_window": "benign" in iid, "fdp_default": Vd / max(Rk, 1), "fdp_default_cp_lo": lo_d,
                         "fdp_default_cp_hi": hi_d, "fdp_default_lenient": (Vd - Ad) / max(Rk - Ad, 1),
                         "TP_default": int((yd == 1).sum())})
    gdf = pl.DataFrame(rows, infer_schema_length=None)
    atomic_write_parquet(gdf, out / "per_incident.parquet")
    from certichain.eval.gates import bank_within_auc
    for ds_, sid_ in ((cal_ds, cal_sid), (te_ds, te_sid), ("cf_gotham_t", "S_GOT.stat")):
        if X.has_bank(ds_, sid_):
            w = bank_within_auc(X.load_bank(ds_, sid_))
            N.add(f"{GK}|within_auc|{ds_}|{sid_}", w["mean"], definition="mean within-incident AUC of true links vs "
                  "y = 0 nulls (scorer quality on the case-study bank / incidents)", status="secondary",
                  n_incidents=w["n_incidents"])
    # bank-vs-case-study two-sample AUC on phi for THIS calibration bank (E10 recipe, so v2 banks get their own value)
    try:
        Xa, _ = X.phi_matrix(cal_ds, "caltest")
        Xb, _ = X.phi_matrix(te_ds, None)
        r2 = X.two_sample_auc(Xa, Xb)
        N.add(f"{GK}|two_sample_auc|{cal_ds}->{te_ds}", r2["auc"], definition="cross-validated logistic two-sample "
              "classifier on phi(I) (42-d, label-free), calibration-bank incidents vs Gotham-real incidents (E10 recipe); "
              "0.5 = indistinguishable", status="secondary", **{k: v for k, v in r2.items() if k != "auc"})
    except Exception as e:  # pragma: no cover
        log(f"E11 gotham two-sample AUC ({cal_ds}): {e}")
    for alpha in gc["alphas"]:
        sub = gdf.filter((pl.col("alpha") == alpha) & ~pl.col("benign_window"))
        wc = sub.filter(pl.col("R") > 0)
        N.add(f"{GK}|summary|a={alpha}", int(wc.height), definition="attack incidents (of 7) that receive >= 1 claim",
              status="secondary", fdp_min=float(wc["fdp"].min()) if wc.height else None,
              fdp_max=float(wc["fdp"].max()) if wc.height else None,
              lam_idx=int(gdf.filter(pl.col("alpha") == alpha)["lam_idx"][0]),
              benign_windows_with_claims=int(gdf.filter((pl.col("alpha") == alpha) & pl.col("benign_window")
                                                         & (pl.col("R") > 0)).height),
              cal_incidents_with_claims=int((cb.t("R", "bh")[:, int(gdf.filter(pl.col("alpha") == alpha)["lam_idx"][0])] > 0).sum()))
    for r in gdf.iter_rows(named=True):
        N.add(f"{GK}|{r['incident']}|a={r['alpha']}|fdp", r["fdp"], ci=(r["fdp_cp_lo"], r["fdp_cp_hi"]),
              definition="per-incident strict FDP of CRC-BH (S_TON, calibrated on all CF-GOTHAM-T incidents) with an "
              "exact Clopper-Pearson 95% CI, conditional on the incident; no FDR claim (seed-priority GT; default-rule "
              "GT FDP of the same claims in fdp_default)", status="secondary",
              R=r["R"], V=r["V"], TP=r["TP"], str=r["str"], ceiling=r["ceiling"], fdp_lenient=r["fdp_lenient"],
              fdp_default=r["fdp_default"], fdp_default_ci=[r["fdp_default_cp_lo"], r["fdp_default_cp_hi"]],
              fdp_default_lenient=r["fdp_default_lenient"], TP_default=r["TP_default"])
    N.write()
    log("E11 gotham case study written")


# =========================================================================== E6 (SHOULD)
def e6(cfg: dict, a) -> None:
    from certichain.certify.elond import build_stream_bank, choose_look_levels, evaluate_stream_split
    ds, scorer = cfg["dataset"], cfg["scorer"]
    md = X.results_root() / "models" / scorer
    s0 = np.load(md / "null_pool_prefix.npy")
    out_dir = X.exp_root("E6") / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = X.stream_incidents(ds, "train", scorer)
    te = X.stream_incidents(ds, "caltest", scorer)
    check_memory()
    trb = build_stream_bank(tr, s0)
    teb = build_stream_bank(te, s0)
    log(f"E6: stream banks train {trb.n}, cal/test {teb.n}")
    N = X.Numbers("E6")
    rows = []
    for alpha in cfg["alphas"]:
        theta = choose_look_levels(trb, alpha)
        sp = random_splits(teb.n, cfg["K"], range(cfg["splits"]), cfg["n_test"])
        sched = None
        for i, (seed, cal, test) in enumerate(sp):
            res = evaluate_stream_split(teb, cal, test, alpha, theta, order_seed=int(seed))
            sched = res.pop("_schedule") if i == 0 else sched
            for meth, mets in res.items():
                if meth.startswith("_"):
                    continue
                rows.append({"alpha": alpha, "seed": int(seed), "method": meth, **mets})
            if i % 20 == 0:
                check_memory()
        atomic_write_json(out_dir / f"schedule_a{alpha}.json", {"theta_idx": theta.tolist(), "split0": sched})
    df = pl.DataFrame(rows, infer_schema_length=None)
    atomic_write_parquet(df, out_dir / "per_split.parquet")
    for (alpha, meth), sub in df.group_by(["alpha", "method"], maintain_order=True):
        for c in sub.columns:
            if c in ("alpha", "seed", "method"):
                continue
            v = sub[c].to_numpy().astype(float)
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            N.add(f"{ds}|{meth}|a={alpha}|{c}", float(v.mean()), definition=f"E6 streaming metric {c} (P2-C "
                  "certify/elond.stream_metrics), mean over 100 splits", status="exploratory" if False else "secondary",
                  split_sd=float(v.std(ddof=1)) if v.size > 1 else 0.0, n_splits=int(v.size))
    N.write()
    log(f"E6 numbers: {len(N.d)} entries")


# =========================================================================== E10 (SHOULD)
def e10(cfg: dict, a) -> None:
    from certichain.certify.crc import crc_threshold_from_sums
    N = X.Numbers("E10")
    out_dir = X.exp_root("E10")
    out_dir.mkdir(parents=True, exist_ok=True)
    S_ = cfg["splits"]
    mets = ("fdr", "fdr_lenient", "pooled_fdr", "str_mean", "cer_link_mean", "abstain", "n_claims")
    # (1) unweighted transfer: calibrate on the source bank, test on the target bank (same scorer / null pool)
    for t in cfg["transfer"]:
        (cds, csid), (tds, tsid) = t["cal"], t["test"]
        if not (X.has_bank(cds, csid) and X.has_bank(tds, tsid)):
            log(f"E10 transfer {t['name']}: bank missing")
            continue
        bc, bt = X.load_bank(cds, csid), X.load_bank(tds, tsid)
        cal = random_splits(bc.n, t["K_cal"], range(S_), None).cal
        test = np.stack([np.random.default_rng((s, 11)).permutation(bt.n)[: t["n_test"]] for s in range(S_)])
        for alpha in cfg["alphas"]:
            sel = X.cross_select(bc, cal, bt, test, "crc_bh", alpha)
            res = X.selection_metrics(bt, sel, alpha, strata=())
            for m in mets:
                v = np.asarray(res[m], float)
                N.add(f"transfer|{t['name']}|a={alpha}|{m}", float(np.nanmean(v)), definition=f"unweighted transfer "
                      f"({cds}/{csid} calibration, K = {t['K_cal']}; {tds}/{tsid} test, {t['n_test']} incidents), "
                      f"CRC-BH, {S_} splits: {METRIC_DEF.get(m, m)}", status="secondary",
                      split_sd=float(np.nanstd(v, ddof=1)))
    # (2) few-shot recalibration on K_target target incidents (exact: CRC on the target's own incidents)
    for fcfg in cfg["few_shot"]:
        ds, sid = fcfg["ds"], fcfg["sid"]
        if not X.has_bank(ds, sid):
            log(f"E10 few-shot {fcfg['name']}: bank missing")
            continue
        for K in fcfg["K_target"]:
            for alpha in cfg["alphas"]:
                df, meta, path = X.run_cell("E10", ds, sid, "crc_bh", alpha, K=int(K), n_test=fcfg["n_test"],
                                            seeds=range(S_), overwrite=a.overwrite)
                add_cell_numbers(N, f"fewshot|{ds}|{sid}|K={K}|a={alpha}", path, "secondary", metrics=mets)
    # (3) IoT-23 scenario-level few-shot: K scenarios calibrate (scenario loss paths), the others are tested
    ic = cfg["iot23_scenario_few_shot"]
    if X.has_bank("iot23", ic["sid"]):
        bank = X.load_bank("iot23", ic["sid"])
        g = X.scenario_groups(bank)
        scen = sorted(set(g.tolist()))
        T = bank.t("Lmono", "bh")
        Ls = np.stack([T[g == s_].mean(axis=0) for s_ in scen])
        for K in ic["K_scenarios"]:
            for alpha in cfg["alphas"]:
                vals, strs, abst = [], [], []
                for s in range(ic["splits"]):
                    perm = np.random.default_rng((s, 23)).permutation(len(scen))
                    cal, te = perm[:K], perm[K:]
                    col = int(crc_threshold_from_sums(Ls[cal].sum(0)[None, :], np.array([K]), alpha)[0])
                    fd, tr_, nT = [], 0.0, 0.0
                    for j in te:
                        rows = np.flatnonzero(g == scen[j])
                        R = bank.t("R", "bh")[rows, col].astype(np.int64)
                        c = bank.counts(rows, R)
                        fd.append(float(np.mean(c["V"] / np.maximum(c["R"], 1))))
                        tr_ += float(c["TR"].sum())
                        nT += float(np.nansum(bank.col("n_trans")[rows]))
                    vals.append(float(np.mean(fd)))
                    strs.append(tr_ / nT if nT else np.nan)
                    abst.append(float(col == 0))
                N.add(f"iot23_fewshot|K={K}|a={alpha}|fdr", float(np.mean(vals)), definition="IoT-23 scenario-level "
                      f"few-shot: CRC on {K} random scenarios' mean window losses, test = the other scenarios' "
                      f"windows; mean over test scenarios of the scenario FDP, {ic['splits']} draws", status="secondary",
                      split_sd=float(np.std(vals, ddof=1)), str_pooled=float(np.nanmean(strs)), abstain=float(np.mean(abst)))
    # (4) bank-vs-deployment two-sample AUC on phi
    for t in cfg["two_sample"]:
        (bds, brole), (dds, drole) = t["bank"], t["deploy"]
        try:
            Xa, _ = X.phi_matrix(bds, brole)
            Xb, _ = X.phi_matrix(dds, drole)
        except Exception as e:  # pragma: no cover
            log(f"E10 two-sample {t['name']}: {e}")
            continue
        r = X.two_sample_auc(Xa, Xb)
        N.add(f"two_sample|{t['name']}|auc", r["auc"], definition="cross-validated logistic two-sample classifier on "
              "phi(I) (42-d, label-free), bank vs deployment incidents; AUC 0.5 = indistinguishable", status="secondary",
              **{k: v for k, v in r.items() if k != "auc"})
        check_memory()
    N.write()
    log(f"E10 numbers: {len(N.d)} entries")


# =========================================================================== X1 (EXPLORATORY; coordinator request)
def x1(cfg: dict, a) -> None:
    from certichain.eval import exploratory as XP
    from certichain.eval.gates import within_incident_auc
    from certichain.scoring import driver as D
    N = X.Numbers("X1")
    out = XP.x_root()
    out.mkdir(parents=True, exist_ok=True)
    pc = cfg["probe"]
    if getattr(a, "datasets", None):
        pc["datasets"] = list(a.datasets)
    # (a) timing / position probes per feature and feature group
    for ds in pc["datasets"]:
        for role in pc["roles"]:
            f = X.exp_root("E15") / "probe_units" / f"{ds}_{role}.parquet"
            if not f.exists():
                continue
            jp = out / f"timing_probes_{ds}_{role}.json"
            if jp.exists() and not a.overwrite:
                diag = json.loads(jp.read_text())
            else:
                diag = XP.timing_diagnosis(pl.read_parquet(f), params=pc["lightgbm"], rounds=pc["rounds"],
                                           B=pc["bootstrap_B"])
                atomic_write_json(jp, diag)
            for tgt, r in diag.items():
                N.add(f"probe|{ds}|{role}|{tgt}|driver", r["driver_group"], definition="timing/position feature group "
                      "with the largest grouped-probe AUC (EXPLORATORY)", status="exploratory",
                      all_timing_auc=r["all_timing"]["auc"], n=r["n"], n_pos=r["n_pos"])
                for g, gr in r["groups"].items():
                    N.add(f"probe|{ds}|{role}|{tgt}|group|{g}|auc", gr["auc"], ci=(gr.get("lo"), gr.get("hi")),
                          definition=f"incident-grouped 5-fold LightGBM probe AUC on the '{g}' features only",
                          status="exploratory")
                for feat, ur in r["univariate"].items():
                    N.add(f"probe|{ds}|{role}|{tgt}|feature|{feat}|auc", ur["auc"], definition="univariate AUC of the "
                          "raw feature (positive class = campaign unit for i/i_attack, D3 unit for ii)",
                          status="exploratory", separability=ur["separability"], median_pos=ur["median_pos"],
                          median_neg=ur["median_neg"])
    log("X1 (a) timing probes done")
    # (b) timing-ablated statistical scorers on CF-TON
    ac = cfg["ablation"]
    ds, scorer = ac["dataset"], ac["scorer"]
    full_sid = f"{scorer}.stat"
    full_bank = X.load_bank(ds, full_sid)
    for variant in ac["variants"]:
        meta = XP.fit_ablated(variant, scorer, log=log, check=check_memory)
        sid = f"{scorer}.{variant}"
        bp = XP.bank_path(ds, sid)
        bank = LossBankX.load(bp) if (bp / "manifest.json").exists() and not a.overwrite else \
            XP.build_ablated_bank(variant, ds, scorer, n_jobs=min(8, a.workers), log=log, check=check_memory)
        X._BANKS[(ds, sid)] = bank
        if bank.index["incident_id"].to_list() != full_bank.index["incident_id"].to_list():
            raise RuntimeError("ablated and P2-F banks are not row-aligned")
        N.add(f"ablation|{variant}|oof_within_auc_strict", meta["oof_within_auc_strict"]["mean"],
              definition="OOF within-incident AUC (strict nulls) of the refit on CF-TON train", status="exploratory",
              n_features=meta["n_features"], dropped=meta["dropped"], final_rounds=meta["final_rounds"])
        if variant == "full":
            mine = pl.read_parquet(XP.x_root() / "models" / f"{scorer}.full" / "oof.parquet")
            theirs = pl.read_parquet(D.models_dir(scorer) / "oof.parquet")
            jj = mine.join(theirs, on=["inc_id", "cand_id"], suffix="_p2f")
            N.add("ablation|full|reproduction_max_abs_oof_diff", float((jj["s_oof"] - jj["s_oof_p2f"]).abs().max()),
                  definition="max |OOF score difference| between this refit with all features and P2-F's S_TON OOF "
                  "(0 = exact reproduction of the recipe)", status="exploratory", n_rows=jj.height)
        for alpha in ac["alphas"]:
            df, meta_c, path = X.run_cell("X1", ds, sid, "crc_bh", alpha, K=ac["K"], n_test=ac["n_test"],
                                          seeds=range(ac["splits"]), bank=bank, overwrite=a.overwrite)
            bc = ac["bootstrap"]
            if not (path / f"bootstrap_paired_vs_{full_sid}.json").exists() or a.overwrite:
                o = X.bootstrap_reps(full_bank, "crc_bh", alpha, K=ac["K"], n_test=ac["n_test"],
                                     metrics=("fdr", "fdr_lenient", "pooled_fdr", "str_mean", "ostr_mean", "cer_link_mean",
                                              "n_claims"), B=bc["B"], n_splits=bc["splits"], seed=bc["seed"], bank_b=bank)
                X.write_bootstrap(path, f"paired_vs_{full_sid}", o, extra={"status": "exploratory"})
            boot = X.read_bootstrap(path, f"paired_vs_{full_sid}")
            pf = find_cell("E1", ds, full_sid, "crc_bh", alpha)
            dff, _ = X.read_cell(pf)
            for m in ("fdr", "fdr_lenient", "pooled_fdr", "str_mean", "ostr_mean", "cer_link_mean", "n_claims"):
                bm = boot["metrics"][m]
                N.add(f"ablation|{variant}|a={alpha}|{m}", float(df[m].mean()), ci=(bm["b"]["lo"], bm["b"]["hi"]),
                      definition=f"CRC-BH with the '{variant}' refit of {scorer}, 200 splits: {METRIC_DEF.get(m, m)}",
                      status="exploratory", full_value=float(dff[m].mean()),
                      delta=float(df[m].mean() - dff[m].mean()), delta_ci=[bm["delta"]["lo"], bm["delta"]["hi"]],
                      source=X.rel(path))
    log("X1 (b) ablations done")
    # (c) Gotham: scorer transfer (S_TON) vs in-domain S_GOT on the same Gotham-real incidents
    gc = cfg["gotham"]
    incs = D.list_incidents("gotham_real")
    ys = {inc: D.load_labels("gotham_real", inc)["y"].to_numpy() for inc in incs}
    for sc_name in (gc["transfer_scorer"], gc["in_domain_scorer"]):
        sc = XP.score_with(sc_name, "gotham_real", incs)
        w0 = within_incident_auc([sc[i] for i in incs], [ys[i] for i in incs], null="zero")
        ws = within_incident_auc([sc[i] for i in incs], [ys[i] for i in incs], null="strict")
        N.add(f"gotham|within_auc|gotham_real|{sc_name}", w0["mean"], definition="mean within-incident AUC of true "
              "links vs y = 0 nulls on the Gotham-real incidents (label-free scores of the scorer's final model)",
              status="exploratory", median=w0["median"], n_incidents=w0["n_incidents"], auc_strict=ws["mean"])
        if sc_name == gc["in_domain_scorer"]:
            sid = f"{sc_name}.stat"
            gb = XP.build_scored_bank("gotham_real", sc_name, sc, sid)
            cds, csid = gc["bank_in_domain"]
            cb = X.load_bank(cds, csid)
            for alpha in gc["alphas"]:
                col = X.crc_full_bank(cb, alpha)
                R = gb.t("R", "bh")[:, col].astype(np.int64)
                c = gb.counts(np.arange(gb.n), R)
                for k, iid in enumerate(gb.index["incident_id"].to_list()):
                    Rk, Vk = int(c["R"][k]), int(c["V"][k])
                    lo, hi = X.clopper_pearson(Vk, Rk)
                    nT = float(gb.col("n_trans")[k])
                    N.add(f"gotham_in_domain|{X.inc_of(iid)}|a={alpha}|fdp", Vk / max(Rk, 1), ci=(lo, hi),
                          definition="per-incident strict FDP of CRC-BH with the IN-DOMAIN scorer S_GOT calibrated on "
                          "all CF-GOTHAM-T incidents (S_GOT bank), exact Clopper-Pearson CI (EXPLORATORY: the "
                          "pre-registered case-study scorer is S_TON)", status="exploratory", R=Rk, V=Vk,
                          TP=int(c["TP"][k]), str=int(c["TR"][k]) / nT if nT else None, lam_idx=col)
    e10 = json.loads((X.summary_root() / "E10_numbers.json").read_text())["numbers"] \
        if (X.summary_root() / "E10_numbers.json").exists() else {}
    t = e10.get("two_sample|CF-GOTHAM-T -> Gotham-real|auc")
    if t:
        N.add("gotham|two_sample_auc|CF-GOTHAM-T->Gotham-real", t["value"], definition="bank-vs-deployment two-sample "
              "AUC on phi (E10, cross-validated logistic)", status="exploratory", n_deploy=t.get("n_deploy"))
    N.write()
    log(f"X1 numbers: {len(N.d)} entries")


# =========================================================================== headline numbers for the writer
def headline(cfg: dict, a) -> None:
    def num(e, k):
        d = json.loads((X.summary_root() / f"{e}_numbers.json").read_text())["numbers"] if \
            (X.summary_root() / f"{e}_numbers.json").exists() else {}
        v = d.get(k)
        return None if v is None else {kk: v.get(kk) for kk in ("value", "ci", "se", "status") if v.get(kk) is not None}
    v1 = json.loads((X.summary_root() / "primary1_validity.json").read_text())
    p2 = json.loads((X.summary_root() / "primary2_llm.json").read_text()) if (X.summary_root() / "primary2_llm.json").exists() else {}
    out = {
        "prereg_blake3": X.PREREG_BLAKE3,
        "primary1_validity": {"cells_run": v1["n_cells_run"], "cells_total": len(v1["cells"]),
                              "max_standardised_excess": v1["max_standardised_excess"], "violations": v1["n_violations"],
                              "missing": [c["cell"] for c in v1["cells"] if c.get("missing")]},
        "primary2_llm": p2.get("decision"),
        "cf_ton_alpha0.2": {k: num("E1", f"cf_ton|{sid}|crc_bh|a=0.2|{m}") for sid in ("S_TON.stat", "S_TON.fuse_s0")
                            for m in ("fdr", "pooled_fdr", "str_mean", "ostr_mean", "cer_link_mean", "load_ratio")
                            for k in [f"{sid}|{m}"]},
        "cf_ton_alpha0.1": {k: num("E1", f"cf_ton|{sid}|crc_bh|a=0.1|{m}") for sid in ("S_TON.stat", "S_TON.fuse_s0")
                            for m in ("fdr", "pooled_fdr", "str_mean", "ostr_mean", "cer_link_mean", "load_ratio")
                            for k in [f"{sid}|{m}"]},
        "bh_iid_cf_ton": {f"a={al}": num("E1", f"cf_ton|S_TON.stat|bh_iid|a={al}|fdr") for al in (0.1, 0.2)},
        "iot23_loso": {f"a={al}": num("E11", f"loso|S_CF.stat|crc_bh|a={al}|fdr") for al in (0.1, 0.2)},
        "iot23_ceiling": num("E11", "loso|S_CF.stat|crc_bh|a=0.1|ceiling_pooled"),
        "e4_inject_bh_q10_a0.1": num("E4", "cf_ton|inject-10|crc_bh|a=0.1|fdr"),
        "e4_inject_threshold_q10_a0.1": num("E4", "cf_ton|inject-10|crc_st|a=0.1|fdr"),
        "e5_composite_maxchain": {ds: num("E5", f"{ds}|{X.scorer_id(ds, 'stat')}|chain_composite|a=0.1|maxchain_fdr")
                                  for ds in ("cf_ton", "cf_gotham")},
        "gotham_summary_a0.2": num("E11", "gotham|summary|a=0.2"),
    }
    # post-PREREG deviation P1 (CF-GOTHAM v2), reported next to the originals
    pv2 = X.summary_root() / "primary1_validity_v2.json"
    if pv2.exists():
        v2 = json.loads(pv2.read_text())
        out["deviation_P1_v2"] = {
            "primary1_validity_v2": {"cells_run": v2["n_cells_run"], "cells_total": len(v2["cells"]),
                                     "max_standardised_excess": v2["max_standardised_excess"],
                                     "violations": v2["n_violations"], "ds_map": v2.get("ds_map")},
            "cf_gotham_v2_crc_bh_fdr": {f"a={al}": num("E1", f"cf_gotham_v2|S_GOT2.stat|crc_bh|a={al}|fdr") for al in (0.1, 0.2)},
            "e5_composite_maxchain_cf_gotham_v2": num("E5", "cf_gotham_v2|S_GOT2.stat|chain_composite|a=0.1|maxchain_fdr"),
            "gotham_v2_summary_a0.2": num("E11", "gotham_v2|summary|a=0.2"),
            "gotham_v2_sgot2_summary_a0.2": num("E11", "gotham_v2_sgot2|summary|a=0.2"),
            "iot23_loso_S_CF2": {f"{k}|a={al}": num("E11", f"loso|S_CF2.{k}|crc_bh|a={al}|fdr")
                                 for k in ("stat", "fuse_s0") for al in (0.1, 0.2)}}
    atomic_write_json(X.summary_root() / "headline.json", out)
    log("headline numbers written")


# =========================================================================== main
EXPERIMENTS = {"E1": e1, "E7": e7, "E15": e15, "validity": validity, "E4": e4, "E3": e3, "E2": e2, "E5": e5,
               "E11": e11, "E6": e6, "E10": e10, "headline": headline, "X1": x1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-boot", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--numbers-only", action="store_true")
    ap.add_argument("--register", default="", help="extra datasets name:scorer:K:n_test[,...] (e.g. v2 banks)")
    ap.add_argument("--datasets", default="", help="override the dataset list of E1 / E5 / E15 (comma list)")
    ap.add_argument("--ds-map", default="", help="orig=new[,...]: validity substitution family and naming (v2)")
    ap.add_argument("--tag", default="", help="suffix for side-by-side outputs (e.g. v2)")
    ap.add_argument("--gotham-cal", default="", help="ds:scorer_id of the case-study calibration bank")
    ap.add_argument("--gotham-test", default="", help="ds:scorer_id of the case-study test bank")
    a = ap.parse_args()
    a.only = [s for s in a.only.split(",") if s]
    for spec in [x for x in a.register.split(",") if x]:
        nm, sc, K, nt = (spec.split(":") + ["", "", ""])[:4]
        X.DATASETS[nm] = {"scorer": sc, "K": int(K) if K else None, "n_test": int(nt) if nt else None}
    a.datasets = [x for x in a.datasets.split(",") if x]
    a.ds_map = dict(x.split("=", 1) for x in a.ds_map.split(",") if x)
    a.workers = max(1, min(8, a.workers or workers(default=8)))
    pr = X.load_prereg()
    X.check_registry(pr)
    log(f"PREREG verified ({X.PREREG_BLAKE3[:16]}...), workers {a.workers}")
    try:
        for exp in a.exp.split(","):
            cfg = load_cfg("E1" if exp in ("validity", "headline") else exp)
            log(f"=== {exp} ({cfg['_file']}, blake3 {cfg['_blake3'][:12]})")
            t0 = time.time()
            if a.numbers_only:
                {"E1": e1_numbers, "E7": e7_numbers}.get(exp, lambda c: EXPERIMENTS[exp](c, a))(cfg)
            else:
                EXPERIMENTS[exp](cfg, a)
            log(f"=== {exp} done in {time.time() - t0:.0f}s")
    except MemoryBudgetExceeded as e:
        log(f"[mem] {e}; exiting {EXIT_MEMORY}")
        return EXIT_MEMORY
    return 0


if __name__ == "__main__":
    sys.exit(main())
