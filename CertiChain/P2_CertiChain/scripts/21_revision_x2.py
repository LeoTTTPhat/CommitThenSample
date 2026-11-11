#!/usr/bin/env python
"""21 -- P2-X revision analyses "X2" (POST-PREREG, EXPLORATORY; mock review paper/review_mock.md). Owner: P2-X.

Run from the project root under memguard (P2 budget 90 GB, <= 8 processes):

  .venv/bin/python memguard/memguard.py run --name p2_rev_x2 --limit-gb 90 --workers 8 -- \
      .venv/bin/python -I -u P2_CertiChain/scripts/21_revision_x2.py --task conditional

  --task conditional   task 1 (+3): conditional reliability of every method x dataset x alpha (split cells, IoT-23
                       P-LOSO, Gotham-real) -> results/exp/X2/conditional{,_loso}/..., summary/X2_conditional.json
  --task iot23_audit   task 2: every certified IoT-23 claim (P-LOSO, P-WIN frequency), true links, rule baselines,
                       rho_T split -> results/exp/X2/iot23_audit/, summary/X2_iot23_audit.json
  --task chains        task 3: chain rules (composite / naive / RT) incl. per-incident chain-FDP exceedance, by L
                       -> results/exp/X2/chains/, summary/X2_chains.json
  --task stats         task 4: one-sided upper confidence bounds, two-sample AUC permutation / references, MDE of the
                       LLM endpoint, Mondrian CRC -> results/exp/X2/{mondrian,two_sample}/, summary/X2_stats.json
  --task easiness      task 5: two-way (incident x episode) bootstrap, long-lag STR, decoy-type strata
                       -> results/exp/X2/{twoway,lagstr,decoy}/, summary/X2_easiness.json
  --task tables        LaTeX tables figures/table_x2_*.tex from the summaries
  --workers N (default MEMGUARD_WORKERS or 8, capped at 8)   --B N (bootstrap replicates, default 1000)
  --overwrite (recompute finished cells)   --only DS,DS

Every output is new (results/exp/X2, results/summary/X2_*.json, figures/table_x2_*.tex); nothing pre-registered or
produced by P2-G is modified. All numbers carry status 'exploratory' (post-PREREG revision analysis).
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
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from certichain.eval import experiments as X  # noqa: E402
from certichain.eval import revision as RV  # noqa: E402
from certichain.eval import revision_tasks as RT  # noqa: E402
from certichain.io.paths import P2_ROOT, atomic_write_json, atomic_write_parquet, atomic_write_text  # noqa: E402
from certichain.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory, workers  # noqa: E402

COST = {"b8": 12, "bh_iid_pooled": 6, "bh_iid": 3, "lr_a2": 5}
SUMMARY = X.summary_root()
FIG = P2_ROOT / "figures"
STATUS = "exploratory (post-PREREG revision analysis X2, P2-X; not pre-registered)"


def log(msg: str) -> None:
    print(f"[21 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pool_run(tasks: list[dict], n_workers: int, label: str) -> list[dict]:
    out: list[dict] = []
    if not tasks:
        return out
    tasks = sorted(tasks, key=lambda t: COST.get(t.get("method", ""), 1) * (3 if t["kind"] == "loso" else 1),
                   reverse=True)
    t0 = time.time()
    if n_workers <= 1:
        for k, t in enumerate(tasks):
            check_memory()
            out.append(RT.run_task(t))
            if "error" in out[-1]:
                log(f"ERROR {t}: {out[-1]['error']}\n{out[-1].get('traceback')}")
        return out
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(n_workers, mp_context=ctx) as ex:
        futs = [ex.submit(RT.run_task, t) for t in tasks]
        for k, f in enumerate(as_completed(futs)):
            r = f.result()
            out.append(r)
            if "error" in r:
                log(f"ERROR {r['task']}: {r['error']}\n{r.get('traceback')}")
            check_memory()
            if (k + 1) % 20 == 0 or k + 1 == len(tasks):
                log(f"{label}: {k + 1}/{len(tasks)} ({time.time() - t0:.0f}s)")
    return out


def reg_entry(value, definition: str, *, ci=None, se=None, source=None, **extra) -> dict:
    e = {"value": RT._clean(value), "definition": definition, "status": "exploratory"}
    if ci is not None:
        e["ci"] = [RT._clean(ci[0]), RT._clean(ci[1])]
    if se is not None:
        e["se"] = RT._clean(se)
    if source is not None:
        e["source"] = str(source)
    for k, v in extra.items():
        e[k] = RT._clean(v)
    return e


def write_summary(name: str, body: dict) -> Path:
    out = {"exp": "X2", "status": STATUS, "prereg_blake3": X.PREREG_BLAKE3,
           "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **body}
    p = SUMMARY / f"{name}.json"
    atomic_write_json(p, RT._clean(out))
    log(f"wrote {X.rel(p)}")
    return p


# =========================================================================== task 1: conditional reliability
def conditional_tasks(a) -> list[dict]:
    tasks = []
    for ds, d in RT.X2_DATASETS.items():
        if a.only and ds not in a.only:
            continue
        for sid in d["sids"]:
            if not X.has_bank(ds, sid):
                log(f"missing bank {ds}/{sid}")
                continue
            for alpha in RT.ALPHAS:
                for m in RT.SPLIT_METHODS:
                    tasks.append({"kind": "cond", "ds": ds, "sid": sid, "method": m, "alpha": alpha, "K": d["K"],
                                  "n_test": d["n_test"], "B": min(a.B, a.B_b8) if m == "b8" else a.B,
                                  "overwrite": a.overwrite})
                if sid.endswith(".stat"):
                    for m in RT.FIXED_METHODS:
                        if X.has_baseline(ds, m):
                            tasks.append({"kind": "cond", "ds": ds, "sid": sid, "method": m, "alpha": alpha,
                                          "K": d["K"], "n_test": d["n_test"], "B": a.B, "overwrite": a.overwrite})
    if not a.only or "iot23" in a.only:
        for sid in ("S_CF.stat", "S_CF.fuse_s0"):
            for alpha in RT.ALPHAS:
                for m in RT.LOSO_METHODS + (RT.FIXED_METHODS if sid.endswith(".stat") else ()):
                    if m in RT.FIXED_METHODS and not X.has_baseline("iot23", m):
                        continue
                    tasks.append({"kind": "loso", "sid": sid, "method": m, "alpha": alpha,
                                  "B": min(a.B, a.B_b8) if m == "b8" else a.B, "overwrite": a.overwrite})
    return tasks


def task_conditional(a) -> None:
    pool_run(conditional_tasks(a), a.workers, "conditional cells")
    numbers_conditional(a)


GOTHAM_SETTINGS = (
    ("S_TON transfer, CF-GOTHAM-T v2 bank", ("cf_gotham_t_v2", "S_TON.stat"), ("gotham_real", "S_TON.stat"), "v2"),
    ("S_TON transfer, CF-GOTHAM-T v1 bank", ("cf_gotham_t", "S_TON.stat"), ("gotham_real", "S_TON.stat"), "v1"),
    ("S_GOT2 in-domain (confounded: bank from the scorer's training pools)", ("cf_gotham_t_v2", "S_GOT2.stat"),
     ("gotham_real", "S_GOT2.stat"), "v2_sgot2"),
)
GOTHAM_METHODS = ("crc_bh", "crc_st", "crc_fdx020", "crc_fdx030", "ltt", "by_ref", "bh_ref")


def gotham_block() -> list[dict]:
    rows = []
    for name, cal, test, tag in GOTHAM_SETTINGS:
        if not (X.has_bank(*cal) and X.has_bank(*test)):
            if tag == "v2_sgot2" and X.has_bank(*cal) and not X.has_bank(*test):
                # the S_GOT2 Gotham-real scores have no bank under results/losses: rebuild in memory is not done here
                log(f"Gotham setting {tag}: test bank {test} missing (skipped)")
            continue
        for r in RT.gotham_rows(cal, test, GOTHAM_METHODS):
            r["setting"] = name
            r["tag"] = tag
            rows.append(r)
    return rows


def _cell_metrics(path: Path) -> dict | None:
    p = path / "cell.json"
    return json.loads(p.read_text()) if p.exists() else None


def collect_conditional() -> list[dict]:
    rows = []
    root = RT.x2_root() / "conditional"
    for p in sorted(root.glob("*/*/a*/cell.json")):
        c = json.loads(p.read_text())
        cfg = c["config"]
        rows.append({"protocol": "split", "ds": cfg["ds"], "sid": cfg["sid"], "method": cfg["method"],
                     "alpha": cfg["alpha"], "K": cfg["K"], "n_test": cfg["n_test"], "metrics": c["metrics"],
                     "load_ratio": c.get("load_ratio"), "no_claims": c.get("no_claims_in_any_split"),
                     "source": X.rel(p.parent), "ci_kind": c.get("ci_kind")})
    root = RT.x2_root() / "conditional_loso"
    for p in sorted(root.glob("*/*/a*/cell.json")):
        c = json.loads(p.read_text())
        cfg = c["config"]
        rows.append({"protocol": "P-LOSO", "ds": "iot23_loso", "sid": cfg["sid"], "method": cfg["method"],
                     "alpha": cfg["alpha"], "metrics": c["metrics"], "source": X.rel(p.parent),
                     "ci_kind": c.get("ci_kind"), "lam_idx_per_scenario": c.get("lam_idx_per_scenario")})
    return rows


METHOD_LABEL = dict(X.LABELS)
GUAR = dict(X.GUARANTEE)


def numbers_conditional(a=None) -> None:
    rows = collect_conditional()
    gr = gotham_block()
    nums: dict[str, dict] = {}
    for r in rows:
        key0 = f"{r['ds']}|{r['sid']}|{r['method']}|a={r['alpha']}"
        for m, d in r["metrics"].items():
            if not isinstance(d, dict):
                continue
            ci = (d.get("lo"), d.get("hi")) if d.get("lo") is not None else None
            extra = {k: d.get(k) for k in ("ucb95_normal", "ucb95_pct", "split_sd") if d.get(k) is not None}
            if "jackknife" in d:
                extra["jackknife"] = d["jackknife"]
            nums[f"{key0}|{m}"] = reg_entry(d.get("point"), RV.DEFS.get(m, m) + (
                " [P-LOSO: window level unless *_scen]" if r["protocol"] == "P-LOSO" else ""),
                ci=ci, se=d.get("se"), source=r["source"], **extra)
    for g in gr:
        key0 = f"gotham_real|{g['tag']}|{g['method']}|a={g['alpha']}"
        for m in ("frac_empty_all", "frac_empty_attack", "fdr_all", "fdr_attack", "cond_fdp", "pooled_fdr", "sum_R",
                  "sum_V", "sum_TP", "claimed_attack", "claimed_benign", "n_claims", "n_false", "fdx_020",
                  "cond_fdx_020", "fdp_min_claimed", "fdp_max_claimed", "cal_incidents_with_claims", "lam_idx"):
            nums[f"{key0}|{m}"] = reg_entry(g.get(m), f"Gotham-real ({g['setting']}): {m}; lambda-hat from all "
                                            f"{g['K']} calibration-bank incidents; counts only (n = 9)",
                                            source=g["cal"] + " -> " + g["test"])
    # review cross-check (paper/review_mock.md section 9)
    def val(k):
        e = nums.get(k)
        return None if e is None else e["value"]
    check = {
        "cf_ton stat E[FDP|claims] 0.29/0.40": [val("cf_ton|S_TON.stat|crc_bh|a=0.1|cond_fdp_ratio"),
                                                 val("cf_ton|S_TON.stat|crc_bh|a=0.2|cond_fdp_ratio")],
        "cf_ton fused 0.24/0.36": [val("cf_ton|S_TON.fuse_s0|crc_bh|a=0.1|cond_fdp_ratio"),
                                   val("cf_ton|S_TON.fuse_s0|crc_bh|a=0.2|cond_fdp_ratio")],
        "cf_ton_h stat 0.48/0.49": [val("cf_ton_h|S_TON_H.stat|crc_bh|a=0.1|cond_fdp_ratio"),
                                    val("cf_ton_h|S_TON_H.stat|crc_bh|a=0.2|cond_fdp_ratio")],
        "cf_ton_h fused 0.45/0.55": [val("cf_ton_h|S_TON_H.fuse_s0|crc_bh|a=0.1|cond_fdp_ratio"),
                                     val("cf_ton_h|S_TON_H.fuse_s0|crc_bh|a=0.2|cond_fdp_ratio")],
        "cf_gotham_v2 0.10/0.20": [val("cf_gotham_v2|S_GOT2.stat|crc_bh|a=0.1|cond_fdp_ratio"),
                                   val("cf_gotham_v2|S_GOT2.stat|crc_bh|a=0.2|cond_fdp_ratio")],
        "P(FDP>2a) cf_ton 0.146/0.237": [val("cf_ton|S_TON.stat|crc_bh|a=0.1|p_fdp_gt_2a"),
                                         val("cf_ton|S_TON.stat|crc_bh|a=0.2|p_fdp_gt_2a")],
        "P(FDP>2a) cf_ton_h 0.112/0.225": [val("cf_ton_h|S_TON_H.stat|crc_bh|a=0.1|p_fdp_gt_2a"),
                                           val("cf_ton_h|S_TON_H.stat|crc_bh|a=0.2|p_fdp_gt_2a")],
        "pooled cf_ton 0.410/0.527 (lenient 0.142/0.241)": [val("cf_ton|S_TON.stat|crc_bh|a=0.1|pooled_fdr"),
                                                            val("cf_ton|S_TON.stat|crc_bh|a=0.2|pooled_fdr"),
                                                            val("cf_ton|S_TON.stat|crc_bh|a=0.1|pooled_fdr_lenient"),
                                                            val("cf_ton|S_TON.stat|crc_bh|a=0.2|pooled_fdr_lenient")],
        "pooled cf_ton_h 0.600/0.639 (0.221/0.279)": [val("cf_ton_h|S_TON_H.stat|crc_bh|a=0.1|pooled_fdr"),
                                                      val("cf_ton_h|S_TON_H.stat|crc_bh|a=0.2|pooled_fdr"),
                                                      val("cf_ton_h|S_TON_H.stat|crc_bh|a=0.1|pooled_fdr_lenient"),
                                                      val("cf_ton_h|S_TON_H.stat|crc_bh|a=0.2|pooled_fdr_lenient")],
        "pooled cf_gotham_v2 0.173/0.374 (0.063/0.224)": [val("cf_gotham_v2|S_GOT2.stat|crc_bh|a=0.1|pooled_fdr"),
                                                          val("cf_gotham_v2|S_GOT2.stat|crc_bh|a=0.2|pooled_fdr"),
                                                          val("cf_gotham_v2|S_GOT2.stat|crc_bh|a=0.1|pooled_fdr_lenient"),
                                                          val("cf_gotham_v2|S_GOT2.stat|crc_bh|a=0.2|pooled_fdr_lenient")],
        "dev_false_impl cf_ton 0.026/0.077": [val("cf_ton|S_TON.stat|crc_bh|a=0.1|dev_false_impl"),
                                              val("cf_ton|S_TON.stat|crc_bh|a=0.2|dev_false_impl")],
        "dev_false_impl cf_ton_h 0.048/0.122": [val("cf_ton_h|S_TON_H.stat|crc_bh|a=0.1|dev_false_impl"),
                                                val("cf_ton_h|S_TON_H.stat|crc_bh|a=0.2|dev_false_impl")],
        "crc_st cf_ton stat fdr/pooled/str 0.081/0.257/0.091; 0.151/0.320/0.183":
            [val("cf_ton|S_TON.stat|crc_st|a=0.1|fdr"), val("cf_ton|S_TON.stat|crc_st|a=0.1|pooled_fdr"),
             val("cf_ton|S_TON.stat|crc_st|a=0.1|str_mean"), val("cf_ton|S_TON.stat|crc_st|a=0.2|fdr"),
             val("cf_ton|S_TON.stat|crc_st|a=0.2|pooled_fdr"), val("cf_ton|S_TON.stat|crc_st|a=0.2|str_mean")],
        "crc_st cf_ton fused 0.085/0.239/0.111; 0.150/0.286/0.238":
            [val("cf_ton|S_TON.fuse_s0|crc_st|a=0.1|fdr"), val("cf_ton|S_TON.fuse_s0|crc_st|a=0.1|pooled_fdr"),
             val("cf_ton|S_TON.fuse_s0|crc_st|a=0.1|str_mean"), val("cf_ton|S_TON.fuse_s0|crc_st|a=0.2|fdr"),
             val("cf_ton|S_TON.fuse_s0|crc_st|a=0.2|pooled_fdr"), val("cf_ton|S_TON.fuse_s0|crc_st|a=0.2|str_mean")],
        "crc_st cf_ton_h stat 0.085/0.345/0.053; 0.157/0.385/0.111":
            [val("cf_ton_h|S_TON_H.stat|crc_st|a=0.1|fdr"), val("cf_ton_h|S_TON_H.stat|crc_st|a=0.1|pooled_fdr"),
             val("cf_ton_h|S_TON_H.stat|crc_st|a=0.1|str_mean"), val("cf_ton_h|S_TON_H.stat|crc_st|a=0.2|fdr"),
             val("cf_ton_h|S_TON_H.stat|crc_st|a=0.2|pooled_fdr"), val("cf_ton_h|S_TON_H.stat|crc_st|a=0.2|str_mean")],
        "crc_st cf_gotham_v2 0.079/0.104/0.521; 0.176/0.273/0.567":
            [val("cf_gotham_v2|S_GOT2.stat|crc_st|a=0.1|fdr"), val("cf_gotham_v2|S_GOT2.stat|crc_st|a=0.1|pooled_fdr"),
             val("cf_gotham_v2|S_GOT2.stat|crc_st|a=0.1|str_mean"), val("cf_gotham_v2|S_GOT2.stat|crc_st|a=0.2|fdr"),
             val("cf_gotham_v2|S_GOT2.stat|crc_st|a=0.2|pooled_fdr"), val("cf_gotham_v2|S_GOT2.stat|crc_st|a=0.2|str_mean")],
        "crc_st wrongful_benign cf_ton stat/fused (review: 0)": [val("cf_ton|S_TON.stat|crc_st|a=0.1|wrongful_benign"),
                                                                 val("cf_ton|S_TON.stat|crc_st|a=0.2|wrongful_benign"),
                                                                 val("cf_ton|S_TON.fuse_s0|crc_st|a=0.1|wrongful_benign"),
                                                                 val("cf_ton|S_TON.fuse_s0|crc_st|a=0.2|wrongful_benign")],
        "bh_ref stat per-incident 0.175/0.286, pooled 0.539/0.658": [val("cf_ton|S_TON.stat|bh_ref|a=0.1|fdr"),
                                                                     val("cf_ton|S_TON.stat|bh_ref|a=0.2|fdr"),
                                                                     val("cf_ton|S_TON.stat|bh_ref|a=0.1|pooled_fdr"),
                                                                     val("cf_ton|S_TON.stat|bh_ref|a=0.2|pooled_fdr")],
        "B7 fused a=0.2 fdr/pooled/str 0.154/0.289/0.251": [val("cf_ton|S_TON.fuse_s0|b7|a=0.2|fdr"),
                                                            val("cf_ton|S_TON.fuse_s0|b7|a=0.2|pooled_fdr"),
                                                            val("cf_ton|S_TON.fuse_s0|b7|a=0.2|str_mean")],
        "IoT-23 P-LOSO stat claims 86/0, 286/1": [val("iot23_loso|S_CF.stat|crc_bh|a=0.1|sum_R"),
                                                  val("iot23_loso|S_CF.stat|crc_bh|a=0.1|sum_TP"),
                                                  val("iot23_loso|S_CF.stat|crc_bh|a=0.2|sum_R"),
                                                  val("iot23_loso|S_CF.stat|crc_bh|a=0.2|sum_TP")],
        "IoT-23 P-LOSO fused 117/0, 204/0": [val("iot23_loso|S_CF.fuse_s0|crc_bh|a=0.1|sum_R"),
                                             val("iot23_loso|S_CF.fuse_s0|crc_bh|a=0.1|sum_TP"),
                                             val("iot23_loso|S_CF.fuse_s0|crc_bh|a=0.2|sum_R"),
                                             val("iot23_loso|S_CF.fuse_s0|crc_bh|a=0.2|sum_TP")],
        "IoT-23 P-WIN pooled 0.998/0.993": [val("iot23|S_CF.stat|crc_bh|a=0.1|pooled_fdr"),
                                            val("iot23|S_CF.stat|crc_bh|a=0.2|pooled_fdr")],
        "Gotham S_TON 92 claims / 5 true": [val("gotham_real|v2|crc_bh|a=0.1|sum_R"), val("gotham_real|v2|crc_bh|a=0.1|sum_TP"),
                                           val("gotham_real|v2|crc_bh|a=0.2|sum_R"), val("gotham_real|v2|crc_bh|a=0.2|sum_TP")],
    }
    write_summary("X2_conditional", {
        "definitions": RV.DEFS,
        "conventions": "strict FDP (AMB counted false) unless *_lenient; split cells: 200 PREREG split seeds 0-199, "
                       "PREREG K / test sizes; CI = incident-pool cluster bootstrap (B x 50 splits, P2-C stream); "
                       "P-LOSO: window-level metrics, K = 22 scenarios, CI = scenario-cluster bootstrap with LOSO "
                       "recomputation (bounded), jackknife as P2-G; Gotham-real: counts on 9 fixed incidents, no CI.",
        "numbers": nums, "gotham_real": gr, "review_section9_check": check,
        "n_cells": len(rows)})


# =========================================================================== task 2: IoT-23 claim audit
AUDIT_SIDS = ("S_CF.stat", "S_CF.fuse_s0", "S_CF2.stat", "S_CF2.fuse_s0")
AUDIT_METHODS = ("crc_bh", "crc_st", "crc_fdx020", "ltt")


def _claim_summary(df: pl.DataFrame) -> dict:
    if df.height == 0:
        return {"n_claims": 0, "n_true": 0, "n_amb": 0, "n_false_y0": 0, "n_windows": 0, "n_scenarios": 0,
                "n_transitions_hit": 0, "by_truth": {}}
    return {"n_claims": int(df.height), "n_true": int((df["y"] == 1).sum()), "n_amb": int((df["y"] == -1).sum()),
            "n_false_y0": int((df["y"] == 0).sum()), "n_windows": int(df["window"].n_unique()),
            "n_scenarios": int(df["scenario"].n_unique()),
            "n_transitions_hit": int(df.filter(pl.col("y") == 1).select("window", "transition").unique().height),
            "pooled_fdr": float((df["y"] != 1).sum() / df.height),
            "by_truth": dict(zip(*df.group_by("truth").len().sort("len", descending=True).to_dict(as_series=False).values())),
            "top_type_pairs": [dict(r) for r in df.group_by("u_type", "v_type").len().sort("len", descending=True)
                               .head(8).iter_rows(named=True)]}


def task_iot23_audit(a) -> None:
    from certichain.eval import revision_audit as RA
    out = RT.x2_root() / "iot23_audit"
    out.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    frames, summ = [], {}
    for sid in AUDIT_SIDS:
        if not X.has_bank("iot23", sid):
            continue
        bank = X.load_bank("iot23", sid)
        g = X.scenario_groups(bank)
        for alpha in RT.ALPHAS:
            for m in AUDIT_METHODS:
                rows_list, R_list, cols = RV.loso_window_R(bank, m, alpha, groups=g)
                rows = np.concatenate(rows_list)
                R = np.concatenate(R_list)
                lam = [c for c, r in zip(cols, rows_list) for _ in range(len(r))]
                df = RA.claims_frame(bank, rows, R, protocol="P-LOSO", method=m, sid=sid, alpha=alpha, lam_idx=lam,
                                     cache=cache)
                if df.height:
                    frames.append(df)
                s = _claim_summary(df)
                s["lam_idx_per_scenario"] = dict(zip(sorted(set(g.tolist())), cols))
                s["n_scenarios_abstaining"] = int(sum(1 for c in cols if c == 0))
                summ[f"P-LOSO|{sid}|{m}|a={alpha}"] = s
    claims = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    atomic_write_parquet(claims, out / "certified_claims_ploso.parquet")
    # fixed baselines (truth of every claim)
    bank = X.load_bank("iot23", "S_CF.stat")
    bl = {}
    for b, cl in RA.baseline_claim_sets().items():
        df = RA.fixed_claims_frame(bank, cl, b, cache=cache)
        atomic_write_parquet(df, out / f"baseline_{b}_claims.parquet")
        bl[b] = df
        summ[f"fixed|{b}"] = _claim_summary(df)
    # true links in C and who finds them
    tl = RA.true_links(bank, cache=cache)
    for key in [k for k in summ if k.startswith("P-LOSO|") and k.split("|")[2] == "crc_bh"]:
        _, sid, m, al = key.split("|")
        sub = claims.filter((pl.col("scorer_id") == sid) & (pl.col("method") == m) &
                            (pl.col("alpha") == float(al[2:])) & (pl.col("y") == 1)) if claims.height else pl.DataFrame()
        found = set(zip(sub["window"].to_list(), sub["cand_id"].to_list())) if sub.height else set()
        tl = tl.with_columns(pl.Series(f"{sid}|{al}", [(w, c) in found for w, c in zip(tl["window"].to_list(),
                                                                                         tl["cand_id"].to_list())]))
    for b, df in bl.items():
        found = set(zip(df.filter(pl.col("y") == 1)["window"].to_list(), df.filter(pl.col("y") == 1)["cand_id"].to_list()))
        tl = tl.with_columns(pl.Series(b, [(w, c) in found for w, c in zip(tl["window"].to_list(), tl["cand_id"].to_list())]))
    # P-WIN certification frequency (CRC-BH, S_CF stats / fused)
    pw = {}
    for sid in ("S_CF.stat", "S_CF.fuse_s0"):
        bk = X.load_bank("iot23", sid)
        for alpha in RT.ALPHAS:
            freq, tot = RA.pwin_frequency(bk, "crc_bh", alpha)
            pw[f"P-WIN|{sid}|crc_bh|a={alpha}"] = tot
            if freq.height:
                atomic_write_parquet(freq, out / f"pwin_freq_{sid}_a{alpha}.parquet")
                fr = dict(zip(zip(freq["row"].to_list(), freq["rank"].to_list()), freq["freq"].to_list()))
                tl = tl.with_columns(pl.Series(f"pwin_freq|{sid}|a={alpha}",
                                               [fr.get((r, k), 0.0) for r, k in zip(tl["row"].to_list(), tl["rank"].to_list())]))
    atomic_write_parquet(tl, out / "true_links.parquet")
    ts = RA.transition_split(bank)
    # observable-transition STR (pooled; transitions hit / observable transitions) per method
    ostr = {}
    for key, s in summ.items():
        ostr[key] = {"n_transitions_hit": s["n_transitions_hit"],
                     "ostr_pooled": s["n_transitions_hit"] / ts["n_T_obs"] if ts["n_T_obs"] else None,
                     "str_pooled_with_R0": s["n_transitions_hit"] / ts["n_T"] if ts["n_T"] else None,
                     "str_pooled_without_R0": s["n_transitions_hit"] / (ts["n_T"] - ts["n_T_hid_R0"])
                     if ts["n_T"] - ts["n_T_hid_R0"] else None}
    write_summary("X2_iot23_audit", {
        "definitions": {
            "n_claims": "certified claims summed over the 166 windows (P-LOSO: each scenario held out, lambda-hat from "
                        "the other 22 scenarios; fixed: the rule's claim set)",
            "n_true": "claims with Y = 1 (true direct-enabling links)", "n_amb": "claims with Y = AMB (counted false)",
            "n_false_y0": "claims with Y = 0", "by_truth": "claims by truth class (revision_audit.truth_class)",
            "R0": "the definitional hidden entry transition EXEC(hidden) -> first malicious seed episode added once per "
                  "malicious window by the IoT-23 GT (gt/iot23_episodes.py rule R0)",
            "ostr_pooled": "distinct observable transitions hit / observable transitions (15)",
            "ceiling_without_R0": "observable transitions with their enabling pair in C / (|T*| - #R0) = r_CT"},
        "claims": summ, "transition_split": ts, "observable_str": ostr, "pwin": pw,
        "true_links": [dict(r) for r in tl.iter_rows(named=True)],
        "files": {"claims_ploso": X.rel(out / "certified_claims_ploso.parquet"),
                  "true_links": X.rel(out / "true_links.parquet"),
                  "baselines": {b: X.rel(out / f"baseline_{b}_claims.parquet") for b in bl}}})


# =========================================================================== task 3: chains
CHAIN_DS = ("cf_ton", "cf_ton_h", "cf_gotham", "cf_gotham_v2")


def task_chains(a) -> None:
    tasks = []
    for ds in CHAIN_DS:
        if a.only and ds not in a.only:
            continue
        sid = RT.X2_DATASETS[ds]["sids"][0]
        if not X.has_bank(ds, sid) or not X.load_bank(ds, sid).has_chains:
            continue
        for alpha in RT.ALPHAS:
            for m in RT.CHAIN_METHODS + ("crc_bh", "crc_st"):
                tasks.append({"kind": "chain", "ds": ds, "sid": sid, "method": m, "alpha": alpha, "B": a.B,
                              "overwrite": a.overwrite})
    pool_run(tasks, a.workers, "chain cells")
    numbers_chains()


def numbers_chains() -> None:
    nums, rows = {}, []
    for p in sorted((RT.x2_root() / "chains").glob("*/*/a*/cell.json")):
        c = json.loads(p.read_text())
        cfg = c["config"]
        for bin_, mets in c["metrics"].items():
            for m, d in mets.items():
                key = f"{cfg['ds']}|{cfg['sid']}|{cfg['method']}|a={cfg['alpha']}|{bin_}|{m}"
                nums[key] = reg_entry(d.get("point"), RV.DEFS.get(m, m) + (f" (test incidents with GT chain length "
                                      f"{bin_[2:]})" if bin_ != "all" else ""), ci=(d.get("lo"), d.get("hi"))
                                      if d.get("lo") is not None else None, se=d.get("se"), source=X.rel(p.parent),
                                      n_pool=c["n_pool_per_bin"].get(bin_))
        rows.append({"ds": cfg["ds"], "method": cfg["method"], "alpha": cfg["alpha"], "n_pool_per_bin": c["n_pool_per_bin"]})
    write_summary("X2_chains", {"definitions": {k: RV.DEFS[k] for k in RV.CHAIN_X2 if k in RV.DEFS},
                                "conventions": "CRC-calibrated chain rules on the frozen banks, 200 PREREG splits; "
                                "chain_rt = coherent-factuality-style rule on the score-threshold family (FWER-type); "
                                "chain_rt_bh = the same on the BH family; crc_bh / crc_st rows = chains read off the "
                                "edge-calibrated set (= chain_naive for crc_bh); L bins restrict TEST incidents.",
                                "numbers": nums, "cells": rows})


# =========================================================================== task 4: statistical framing
def task_stats(a) -> None:
    out: dict = {}
    # (d) Mondrian cells first (pool)
    tasks = []
    for ds in ("cf_ton", "cf_ton_h", "cf_gotham", "cf_gotham_v2"):
        for sid in RT.X2_DATASETS[ds]["sids"]:
            if X.has_bank(ds, sid):
                for alpha in RT.ALPHAS:
                    tasks.append({"kind": "mondrian", "ds": ds, "sid": sid, "alpha": alpha, "B": a.B,
                                  "overwrite": a.overwrite})
    pool_run(tasks, a.workers, "mondrian cells")
    out["mondrian"] = collect_mondrian()
    out["ucb"] = ucb_table()
    out["mde"] = mde_block()
    out["two_sample_auc"] = two_sample_block(a)
    write_summary("X2_stats", out)


def collect_mondrian() -> dict:
    res = {}
    for p in sorted((RT.x2_root() / "mondrian").glob("*/*/a*/cell.json")):
        c = json.loads(p.read_text())
        cfg = c["config"]
        res[f"{cfg['ds']}|{cfg['sid']}|{cfg['method']}|a={cfg['alpha']}"] = {
            "metrics": c["metrics"], "stratum_abstain_split_mondrian": c["stratum_abstain_split_mondrian"],
            "N_cut_points": c["N_cut_points"], "n_pool_per_stratum": c["n_pool_per_stratum"], "source": X.rel(p.parent),
            "definition": "Mondrian CRC-BH: one lambda-hat per label-free N tercile (calibration incidents of the "
                          "stratum only, K ~ K/3); 'marginal' = the standard CRC-BH evaluated within the same strata; "
                          "keys '<mondrian|marginal>|<stratum or all>'"}
    return res


GUARANTEED = {"crc_bh": "fdr", "crc_st": "fdr", "crc_fdx020": "fdx_020", "crc_fdx030": "fdx_030", "ltt": "fdr",
              "bh_iid": "fdr", "by_ref": "fdr", "bh_ref": "fdr", "b7": "fdr", "b8": "fdr", "bh_iid_pooled": "pooled_fdr"}


def ucb_table(margin: float = 0.02) -> dict:
    """One-sided 95% upper confidence bounds of the guaranteed estimand of every calibrated cell (X2 bootstraps),
    the pre-registered validity cells (SE_boot of P2-G's primary1 files) and P-LOSO (jackknife t and scenario
    bootstrap). within_margin = UCB <= alpha + margin (equivalence-style reading of the implementation check)."""
    rows = []
    for r in collect_conditional():
        m = r["method"]
        if m not in GUARANTEED:
            continue
        est = GUARANTEED[m]
        key_m = est if r["protocol"] == "split" else ("fdr_scen" if est == "fdr" else est)
        d = r["metrics"].get(key_m) or {}
        if d.get("point") is None:
            continue
        if r["protocol"] == "P-LOSO" and est != "fdr":
            continue                    # P-LOSO FDX is guaranteed at the scenario level; window-level fdx differs
        al = float(r["alpha"])
        row = {"protocol": r["protocol"], "ds": r["ds"], "sid": r["sid"], "method": m, "alpha": al, "estimand": key_m,
               "point": d.get("point"), "ucb95_normal": d.get("ucb95_normal"), "ucb95_pct": d.get("ucb95_pct"),
               "se": d.get("se"), "source": r["source"]}
        if r["protocol"] == "P-LOSO" and "jackknife" in d:
            row["ucb95_jackknife_t"] = d["jackknife"].get("ucb95")
            row["se_jack"] = d["jackknife"].get("se_jack")
        u = [x for x in (row.get("ucb95_normal"), row.get("ucb95_pct"), row.get("ucb95_jackknife_t")) if x is not None]
        row["ucb95_max"] = max(u) if u else None
        row["within_margin"] = (row["ucb95_max"] <= al + margin + 1e-12) if u else None
        row["point_le_alpha"] = d.get("point") <= al + 1e-12
        rows.append(row)
    prereg = []
    for fam, fn in (("prereg", "primary1_validity.json"), ("deviation_v2", "primary1_validity_v2.json")):
        p = SUMMARY / fn
        if not p.exists():
            continue
        for c in json.loads(p.read_text())["cells"]:
            if c.get("missing"):
                prereg.append({"family": fam, "cell": c["cell"], "missing": True})
                continue
            al = float(c["alpha"])
            se = float(c["se"])
            q = 1.6448536269514722
            if c.get("se_kind") == "jackknife":
                from scipy.stats import t as _t
                q = float(_t.ppf(0.95, int(c.get("n_units", 23)) - 1))
            ucb = float(c["mean"]) + q * se
            prereg.append({"family": fam, "cell": c["cell"], "alpha": al, "mean": c["mean"], "se": se,
                           "se_kind": c.get("se_kind"), "ucb95": ucb, "within_margin": ucb <= al + margin + 1e-12,
                           "detectable_violation_80pct": (1.6448536269514722 + 0.8416212335729143) * se,
                           "z": c.get("z"), "p_holm": c.get("p_holm")})
    return {"margin": margin, "cells": rows, "prereg_validity_cells": prereg,
            "definition": "ucb95_normal = point + 1.645 SE_boot (X2 incident-pool bootstrap); ucb95_pct = 95th percentile "
                          "of the bootstrap replicates; P-LOSO: ucb95_jackknife_t = est + t_{22,0.95} SE_jack and the "
                          "scenario-bootstrap bounds; prereg cells use P2-G's SE (bootstrap; jackknife for P-LOSO); "
                          "detectable_violation_80pct = excess over alpha that a one-sided 5% z test would detect with "
                          "80% power at that SE (unadjusted; Holm makes it larger)"}


def mde_block() -> dict:
    """MDE of the pre-registered LLM endpoint (Delta CER_link, CF-TON-H) at the Holm level, from P2-G's bootstrap SE;
    with the between-seed variance of the fusion learner added (seeds 0-4)."""
    p2 = json.loads((SUMMARY / "primary2_llm.json").read_text())
    e3 = json.loads((SUMMARY / "E3_numbers.json").read_text())["numbers"]
    e1 = json.loads((SUMMARY / "E1_numbers.json").read_text())["numbers"]
    out = {}
    for al in (0.1, 0.2):
        h = p2["decision"]["holm"][f"cf_ton_h|alpha={al}"]
        se = float(h["se"])
        lev = float(h["holm_level"])
        seeds = [e3[f"primary|cf_ton_h|S_TON_H.fuse_s{s}-S_TON_H.stat|a={al}|all|delta_cer_link_mean"]["value"]
                 for s in range(5)]
        sd_seed = float(np.std(seeds, ddof=1))
        base = e1.get(f"cf_ton_h|S_TON_H.stat|crc_bh|a={al}|cer_link_mean", {}).get("value")
        se_tot1 = math.sqrt(se ** 2 + sd_seed ** 2)
        se_tot5 = math.sqrt(se ** 2 + sd_seed ** 2 / 5)
        obs = float(e3[f"primary|cf_ton_h|S_TON_H.fuse_s0-S_TON_H.stat|a={al}|all|delta_cer_link_mean"]["value"])
        out[f"a={al}"] = {
            "observed_delta_s0": obs, "se_boot": se, "holm_level": lev, "power": 0.8,
            "mde_holm": RV.mde(se, lev), "mde_unadjusted_0.05": RV.mde(se, 0.05),
            "seed_deltas_s0_s4": seeds, "sd_between_seeds": sd_seed,
            "se_with_seed_var_single_seed": se_tot1, "mde_holm_with_seed_var_single_seed": RV.mde(se_tot1, lev),
            "se_with_seed_var_5seed_mean_approx": se_tot5, "mde_holm_with_seed_var_5seed_mean_approx": RV.mde(se_tot5, lev),
            "power_at_observed_delta": RV.power_at(obs, se, lev),
            "baseline_cer_link_stats": base,
            "mde_relative_to_baseline": (RV.mde(se, lev) / base) if base else None,
            "definition": "MDE = (z_{1-level} + z_{0.8}) x SE for the one-sided paired bootstrap test of Delta CER_link "
                          "> 0 at the Holm step-1 level (0.05/2); SE_boot from P2-G (B = 1000 x 50, fitted models "
                          "fixed); seed variance = SD of the s0-s4 deltas (fusion-learner refits)"}
    return out


TWO_SAMPLE = (
    ("CF-GOTHAM-T v2 bank -> Gotham-real", ("cf_gotham_t_v2", "caltest"), ("gotham_real", None)),
    ("CF-GOTHAM-T v1 bank -> Gotham-real", ("cf_gotham_t", "caltest"), ("gotham_real", None)),
    ("CF-TON bank -> CF-TON-H cal/test (held-out templates; reference shift)", ("cf_ton", "caltest"),
     ("cf_ton_h", "caltest")),
)


def _phi(ds: str, role) -> np.ndarray:
    cache = RT.x2_root() / "two_sample" / f"phi_{ds}_{role or 'all'}.npy"
    if cache.exists():
        return np.load(cache)
    Xm, _ = X.phi_matrix(ds, role)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(f".{cache.name}.tmp-{os.getpid()}.npy")
    np.save(tmp, Xm)
    os.replace(tmp, cache)
    return Xm


def two_sample_block(a) -> dict:
    P = 2000 if a.B >= 1000 else 50
    cache = RT.x2_root() / "two_sample" / f"two_sample_P{P}.json"
    if cache.exists() and not a.overwrite:
        return json.loads(cache.read_text())
    prev = SUMMARY / "X2_stats.json"
    if prev.exists() and not a.overwrite:          # results of an earlier run of this task (same P)
        old = json.loads(prev.read_text()).get("two_sample_auc") or {}
        first = old.get(TWO_SAMPLE[0][0]) or {}
        if first.get("P") == P:
            atomic_write_json(cache, old)
            return old
    out = _two_sample_compute(a, P)
    atomic_write_json(cache, RT._clean(out))
    return out


def _two_sample_compute(a, P: int) -> dict:
    out = {}
    for name, (bds, brole), (dds, drole) in TWO_SAMPLE:
        try:
            Xa, Xb = _phi(bds, brole), _phi(dds, drole)
        except Exception as e:  # pragma: no cover
            out[name] = {"error": str(e)}
            continue
        t0 = time.time()
        perm = RV.permutation_auc(Xa, Xb, P=P, seed=0)
        perm5 = RV.permutation_auc(Xa, Xb, P=max(P // 4, 20), seed=1, cv_seeds=range(5))
        reps = RV.group_boot_auc(Xa, Xb, B=max(P // 4, 20), seed=0)
        cvs = [RV.two_sample_auc_cv(Xa, Xb, seed=s) for s in range(20)]
        ent = {"n_bank": int(len(Xa)), "n_deploy": int(len(Xb)), "auc_seed0": perm["auc_obs"],
               "p_perm_seed0": perm["p_perm"], "null_q95_seed0": perm["null_q95"], "null_mean_seed0": perm["null_mean"],
               "P": perm["P"], "auc_mean5": perm5["auc_obs"], "p_perm_mean5": perm5["p_perm"],
               "null_q95_mean5": perm5["null_q95"], "P_mean5": perm5["P"],
               "cv_seed_spread": {"min": float(np.min(cvs)), "max": float(np.max(cvs)), "mean": float(np.mean(cvs))},
               "boot_ci": [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))], "boot_B": int(reps.size),
               "seconds": time.time() - t0}
        if len(Xb) > 9:          # same-size (9 deployment incidents) subsample distribution under the real shift
            rng = np.random.default_rng(7)
            sub = [RV.two_sample_auc_cv(Xa, Xb[rng.choice(len(Xb), 9, replace=False)], seed=r)
                   for r in range(max(P // 8, 20))]
            ent["auc_with_9_deploy_subsamples"] = {"mean": float(np.mean(sub)), "q05": float(np.quantile(sub, 0.05)),
                                                   "q50": float(np.quantile(sub, 0.5)), "q95": float(np.quantile(sub, 0.95)),
                                                   "R": len(sub)}
        out[name] = ent
        log(f"two-sample {name}: AUC {perm['auc_obs']:.3f}, p_perm {perm['p_perm']:.4f} ({time.time() - t0:.0f}s)")
    # exchangeable references (same sizes as the case study: 9 deployment incidents)
    for name, ds, nb in (("exchangeable: CF-GOTHAM-T v2 bank 141 vs 9 of its own incidents", "cf_gotham_t_v2", 141),
                         ("exchangeable: CF-TON cal/test 150 vs 9", "cf_ton", 150)):
        Xp = _phi(ds, "caltest")
        null = RV.exchangeable_null_auc(Xp, nb, 9, R=max(P // 2, 20), seed=0)
        out[name] = {"n_bank": nb, "n_deploy": 9, "R": int(null.size), "mean": float(null.mean()),
                     "q50": float(np.quantile(null, 0.5)), "q90": float(np.quantile(null, 0.9)),
                     "q95": float(np.quantile(null, 0.95)), "q99": float(np.quantile(null, 0.99)),
                     "share_ge_0.739": float(np.mean(null >= 0.7393 - 1e-12))}
    out["definition"] = ("two-sample AUC = E10's 5-fold cross-validated logistic classifier on the 42-d label-free phi(I) "
                         "(bank = 0, deployment = 1); p_perm = permutation p-value of the group labels (sizes kept); "
                         "*_mean5 = statistic averaged over 5 CV seeds; boot_ci = stratified incident bootstrap with "
                         "duplicates kept in one fold; exchangeable = AUC of random 9-incident 'deployments' drawn from "
                         "the same pool (null reference; q95 = a pre-statable warning threshold)")
    return out


# =========================================================================== task 5: easiness
EASY_DS = ("cf_ton", "cf_gotham", "cf_gotham_v2")


def task_easiness(a) -> None:
    tasks = []
    for ds in ("cf_ton", "cf_ton_h", "cf_gotham", "cf_gotham_v2"):
        if a.only and ds not in a.only:
            continue
        for sid in RT.X2_DATASETS[ds]["sids"]:
            if not X.has_bank(ds, sid):
                continue
            for alpha in RT.ALPHAS:
                for kind in ("twoway", "lagstr", "decoy"):
                    if kind == "twoway":
                        for variant in ("episodes", "episodes+bgtiles"):
                            tasks.append({"kind": kind, "ds": ds, "sid": sid, "alpha": alpha, "B": a.B,
                                          "variant": variant, "overwrite": a.overwrite})
                    else:
                        tasks.append({"kind": kind, "ds": ds, "sid": sid, "alpha": alpha, "B": a.B,
                                      "overwrite": a.overwrite})
    pool_run(tasks, a.workers, "easiness cells")
    numbers_easiness()


def numbers_easiness() -> None:
    body = {"twoway": {}, "lagstr": {}, "decoy": {}}
    for grp in body:
        pat = "*/*/a*/*/cell.json" if grp == "twoway" else "*/*/a*/cell.json"
        for p in sorted((RT.x2_root() / grp).glob(pat)):
            c = json.loads(p.read_text())
            cfg = c["config"]
            key = f"{cfg['ds']}|{cfg['sid']}|{cfg['method']}|a={cfg['alpha']}" + (f"|{cfg['variant']}" if grp == "twoway"
                                                                                   else "")
            c = {k: v for k, v in c.items() if k not in ("config_hash",)}
            c["source"] = X.rel(p.parent)
            body[grp][key] = c
    body["definitions"] = {
        "twoway": "CIs of the incident-only (P2-C) bootstrap vs the two-way incident x source-episode multiple-membership "
                  "bootstrap (revision.episode_pool_sampler; clusters = library episodes + D3 look-alikes from gen.json; "
                  "variant episodes+bgtiles adds 30-min tiles of the background capture); se_ratio = SE_two-way / "
                  "SE_incident",
        "lagstr": "STR restricted to T* transitions by generator lag (gen.json edge lag_s; bridges over hidden "
                  "episodes = sum of hop lags): long = above the pool median, short = at or below; Q1-Q4 quartiles; "
                  "gt600 = lag > 600 s; oracle = r_C-limited ceiling of each group",
        "decoy": "strata of test incidents by decoy composition (gen.json n_D1/n_D2/n_D3; calibration unchanged) and the "
                 "endpoint-kind classes of certified FALSE claims (share over 200 splits)"}
    write_summary("X2_easiness", body)


# =========================================================================== LaTeX tables
DS_LABEL = {"cf_ton": "CF-TON", "cf_ton_h": "CF-TON-H", "cf_gotham": "CF-GOTHAM v1$^\\dagger$",
            "cf_gotham_v2": "CF-GOTHAM v2$^\\ddagger$", "iot23": "IoT-23 P-WIN", "iot23_loso": "IoT-23 P-LOSO"}
M_LABEL = {"crc_bh": "CRC-BH", "crc_st": "CRC-ST", "crc_fdx020": "CRC-FDX(.2)", "crc_fdx030": "CRC-FDX(.3)",
           "ltt": "LTT", "bh_iid": "BH-iid", "bh_iid_pooled": "BH-iid pooled", "by_ref": "BY-ref", "bh_ref": "BH-ref",
           "b7": "B7 iso@0.5", "b8": "B8 Platt", "B1": "B1 King", "B2": "B2 BotHunter", "B4": "B4 lag-LLR",
           "chain_composite": "composite CRC", "chain_naive": "chain-naive", "chain_rt": "chain-RT",
           "chain_rt_bh": "chain-RT (BH)"}
HEADER_NOTE = ("% generated by P2_CertiChain/scripts/21_revision_x2.py --task tables (P2-X); EXPLORATORY post-PREREG "
               "revision analysis; source results/summary/X2_*.json\n")


def _f(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)) or isinstance(x, str):
        return "--" if not isinstance(x, str) else ("$\\infty$" if x == "inf" else "--")
    return f"{x:.{nd}f}"


def _tx(x: str) -> str:
    return str(x).replace("_", "\\_")


def _pair(nums: dict, key0: str, m: str, nd=2) -> str:
    a = (nums.get(f"{key0.replace('{a}', '0.1')}|{m}") or {}).get("value")
    b = (nums.get(f"{key0.replace('{a}', '0.2')}|{m}") or {}).get("value")
    return f"{_f(a, nd)}/{_f(b, nd)}"


def _write_tex(name: str, body: str) -> None:
    p = FIG / name
    atomic_write_text(p, HEADER_NOTE + body)
    log(f"wrote {X.rel(p)}")


def table_conditional(nums: dict, gr: list[dict]) -> None:
    rows = []
    plan = [("cf_ton", ("S_TON.stat", "S_TON.fuse_s0")), ("cf_ton_h", ("S_TON_H.stat", "S_TON_H.fuse_s0")),
            ("cf_gotham", ("S_GOT.stat",)), ("cf_gotham_v2", ("S_GOT2.stat",)), ("iot23_loso", ("S_CF.stat", "S_CF.fuse_s0")),
            ("iot23", ("S_CF.stat",))]
    meths = ("crc_bh", "crc_st", "crc_fdx020", "ltt", "bh_iid", "B2")
    cols = ("frac_empty", "fdr", "cond_fdp", "pooled_fdr", "n_claims", "n_false", "fdx_020", "cond_fdx_020", "str_mean")
    for ds, sids in plan:
        first = True
        for sid in sids:
            for m in meths:
                if m == "B2" and not sid.endswith(".stat"):
                    continue
                if sid.endswith("fuse_s0") and m not in ("crc_bh", "crc_st"):
                    continue
                k0 = f"{ds}|{sid}|{m}|a={{a}}"
                if f"{k0.replace('{a}', '0.1')}|fdr" not in nums and f"{k0.replace('{a}', '0.1')}|fdr_scen" not in nums:
                    continue
                c = []
                for col in cols:
                    cc = col
                    if ds == "iot23_loso" and col == "str_mean":
                        cc = "str_pooled"
                    if ds == "iot23_loso" and col == "fdr":
                        cc = "fdr_scen"           # the guaranteed P-LOSO estimand (scenario mean)
                    c.append(_pair(nums, k0, cc, 2 if col not in ("n_claims", "n_false") else 2))
                lab = M_LABEL[m] + (" (fused)" if sid.endswith("fuse_s0") else "")
                rows.append(("\\multirow{1}{*}{" + DS_LABEL[ds] + "}" if first else "") + f" & {lab} & " + " & ".join(c)
                            + " \\\\")
                first = False
        rows.append("\\midrule")
    # Gotham-real
    for g in gr:
        if g["method"] != "crc_bh" or g["tag"] == "v1":
            continue
        a1 = next((x for x in gr if x["tag"] == g["tag"] and x["method"] == "crc_bh" and x["alpha"] == 0.1), None)
        a2 = next((x for x in gr if x["tag"] == g["tag"] and x["method"] == "crc_bh" and x["alpha"] == 0.2), None)
        if g["alpha"] != 0.1 or a1 is None or a2 is None:
            continue
        lab = "CRC-BH (S$_\\mathrm{TON}$)" if g["tag"] == "v2" else "CRC-BH (S$_\\mathrm{GOT2}$)$^\\S$"
        cells = [f"{_f(a1['frac_empty_all'])}/{_f(a2['frac_empty_all'])}", f"{_f(a1['fdr_all'])}/{_f(a2['fdr_all'])}",
                 f"{_f(a1['cond_fdp'])}/{_f(a2['cond_fdp'])}", f"{_f(a1['pooled_fdr'])}/{_f(a2['pooled_fdr'])}",
                 f"{_f(a1['n_claims'], 1)}/{_f(a2['n_claims'], 1)}", f"{_f(a1['n_false'], 1)}/{_f(a2['n_false'], 1)}",
                 f"{_f(a1['fdx_020'])}/{_f(a2['fdx_020'])}", f"{_f(a1['cond_fdx_020'])}/{_f(a2['cond_fdx_020'])}", "--"]
        rows.append(("Gotham-real (9)" if g["tag"] == "v2" else "") + f" & {lab} & " + " & ".join(cells) + " \\\\")
    body = ("\\begin{table*}[t]\n\\centering\\scriptsize\n\\caption{Conditional reliability (exploratory, post-PREREG): "
            "what an investigator receives. Each cell gives $\\alpha=0.1$/$0.2$. Abst.\\ = share of test incidents with no "
            "claim; $E[\\mathrm{FDP}]$ = mean per-incident FDP (the guaranteed estimand); $E[\\mathrm{FDP}\\mid R>0]$ = mean "
            "FDP over incidents with $\\ge1$ claim; pooled = share of all issued claims that are false; claims and false "
            "claims per incident; FDX exceedance overall and given $\\ge1$ claim; STR (P-LOSO: pooled). 200 PREREG splits; "
            "95\\% incident-pool bootstrap CIs in Table~S\\ref{tab:x2full}.}\n\\label{tab:x2cond}\n"
            "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{llccccccccc}\n\\toprule\n"
            "Dataset & Method & Abst. & $E[\\mathrm{FDP}]$ & $E[\\mathrm{FDP}\\mid R{>}0]$ & Pooled & Claims & False & "
            "$P(\\mathrm{FDP}{>}.2)$ & $P(\\cdot\\mid R{>}0)$ & STR \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n\\\\[2pt]\\raggedright\\footnotesize $^\\dagger$pre-registered; "
            "$^\\ddagger$post-PREREG regeneration (deviation P1); $^\\S$exploratory and confounded (bank built from the "
            "scorer's training pools). Gotham-real: 9 fixed incidents ($\\lambda$ from all 150 CF-GOTHAM-T v2 "
            "incidents), counts, no CI. P-LOSO: $E[\\mathrm{FDP}]$ = mean over the 23 scenarios of the scenario FDP (the "
            "guaranteed estimand), other columns at the window level; scenarios calibrate on the other 22.\n\\end{table*}\n")
    _write_tex("table_x2_conditional.tex", body)


def table_conditional_full(nums: dict) -> None:
    """Supplement: every method x dataset x alpha with 95% CIs of E[FDP], E[FDP | R>0], pooled FDR, STR."""
    out = []
    order_ds = ("cf_ton", "cf_ton_h", "cf_gotham", "cf_gotham_v2", "iot23", "iot23_loso")
    keys = sorted({k.rsplit("|", 1)[0] for k in nums if not k.startswith("gotham_real")})
    def ci(k):
        e = nums.get(k) or {}
        v, c = e.get("value"), e.get("ci")
        if v is None:
            return "--"
        return f"{_f(v)}" + (f" [{_f(c[0])},{_f(c[1])}]" if c and c[0] is not None else "")
    for ds in order_ds:
        for k0 in [k for k in keys if k.split("|")[0] == ds]:
            _, sid, m, al = k0.split("|")
            if m not in M_LABEL:
                continue
            fdr_k = "fdr_scen" if ds == "iot23_loso" else "fdr"
            str_k = "str_pooled" if ds == "iot23_loso" else "str_mean"
            out.append(f"{DS_LABEL[ds]} & {_tx(sid.split('.')[1])} & {M_LABEL[m]} & {al[2:]} & "
                       f"{_f((nums.get(k0 + '|frac_empty') or {}).get('value'))} & {ci(k0 + '|' + fdr_k)} & "
                       f"{ci(k0 + '|cond_fdp')} & {ci(k0 + '|pooled_fdr')} & "
                       f"{_f((nums.get(k0 + '|n_claims') or {}).get('value'))} & "
                       f"{_f((nums.get(k0 + '|n_false') or {}).get('value'))} & "
                       f"{_f((nums.get(k0 + '|fdx_020') or {}).get('value'))} & {ci(k0 + '|' + str_k)} \\\\")
    body = ("\\begin{table*}[p]\n\\centering\\scriptsize\n\\caption{Conditional reliability of every method (exploratory, "
            "post-PREREG), with 95\\% CIs (incident-pool cluster bootstrap, $B=1000\\times50$ splits; B8: $B=500$; "
            "P-LOSO: scenario-cluster bootstrap with LOSO recomputation).}\n\\label{tab:x2full}\n"
            "\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{llllcccccccc}\n\\toprule\n"
            "Dataset & Score & Method & $\\alpha$ & Abst. & $E[\\mathrm{FDP}]$ & $E[\\mathrm{FDP}\\mid R{>}0]$ & Pooled FDR & "
            "Claims & False & $P(\\mathrm{FDP}{>}.2)$ & STR \\\\\n\\midrule\n" + "\n".join(out) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    _write_tex("table_x2_conditional_full.tex", body)


def table_st(nums: dict) -> None:
    """Score-threshold CRC next to CRC-BH on every dataset (issue 6 / R1-M3)."""
    plan = [("cf_ton", "S_TON.stat"), ("cf_ton", "S_TON.fuse_s0"), ("cf_ton_h", "S_TON_H.stat"),
            ("cf_ton_h", "S_TON_H.fuse_s0"), ("cf_gotham", "S_GOT.stat"), ("cf_gotham_v2", "S_GOT2.stat"),
            ("iot23", "S_CF.stat"), ("iot23", "S_CF.fuse_s0"), ("iot23_loso", "S_CF.stat"), ("iot23_loso", "S_CF.fuse_s0")]
    cols = ("str_mean", "cer_link_mean", "fdr", "fdr_lenient", "pooled_fdr", "pooled_fdr_lenient", "cond_fdp",
            "wrongful_benign", "dev_false_impl", "n_claims")
    rows = []
    for ds, sid in plan:
        for m in ("crc_bh", "crc_st"):
            k0 = f"{ds}|{sid}|{m}|a={{a}}"
            cc = []
            for col in cols:
                c2 = col
                if ds == "iot23_loso":
                    c2 = {"str_mean": "str_pooled", "fdr": "fdr_scen", "cer_link_mean": None,
                          "dev_false_impl": None}.get(col, col)
                cc.append(_pair(nums, k0, c2) if c2 else "--")
            rows.append(f"{DS_LABEL[ds]} & {_tx(sid.split('.')[1])} & {M_LABEL[m]} & " + " & ".join(cc) + " \\\\")
        rows.append("\\addlinespace[1pt]")
    body = ("\\begin{table*}[t]\n\\centering\\footnotesize\n\\caption{Score-threshold CRC (CRC-ST, same exact per-incident "
            "guarantee) next to CRC-BH on every dataset ($\\alpha=0.1$/$0.2$; exploratory rows; 200 PREREG splits). CER = "
            "CER$_\\mathrm{link}$; wrongful = P(any claim $\\mid$ benign-only incident); dev.\\ = non-campaign devices "
            "implicated by a false claim.}\n\\label{tab:x2st}\n\\setlength{\\tabcolsep}{2.5pt}\n"
            "\\begin{tabular}{lllcccccccccc}\n\\toprule\n"
            "Dataset & Score & Rule & STR & CER & FDR & FDR$_\\mathrm{len}$ & Pooled & Pooled$_\\mathrm{len}$ & "
            "$E[\\mathrm{FDP}\\mid R{>}0]$ & Wrongful & Dev. & Claims \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    _write_tex("table_x2_st.tex", body)


def table_iot23(audit: dict) -> None:
    rows = []
    for key, lab in (("P-LOSO|S_CF.stat|crc_bh", "CRC-BH (stats)"), ("P-LOSO|S_CF.fuse_s0|crc_bh", "CRC-BH (fused)"),
                     ("P-LOSO|S_CF.stat|crc_st", "CRC-ST (stats)"), ("P-LOSO|S_CF.fuse_s0|crc_st", "CRC-ST (fused)"),
                     ("P-LOSO|S_CF.stat|ltt", "LTT (stats)"), ("P-LOSO|S_CF2.stat|crc_bh", "CRC-BH (S$_\\mathrm{CF2}$)")):
        for al in (0.1, 0.2):
            s = audit["claims"].get(f"{key}|a={al}")
            if not s:
                continue
            o = audit["observable_str"].get(f"{key}|a={al}", {})
            rows.append(f"{lab} & {al} & {s['n_claims']} & {s['n_true']} & {s['n_amb']} & {s['n_false_y0']} & "
                        f"{s['n_windows']} & {s['n_scenarios']} & {s['n_transitions_hit']} & {_f(o.get('ostr_pooled'), 3)} \\\\")
    rows.append("\\midrule")
    for b, lab in (("B1", "B1 King-style"), ("B2", "B2 BotHunter-style"), ("B4", "B4 lag-LLR$>0$")):
        s = audit["claims"].get(f"fixed|{b}")
        o = audit["observable_str"].get(f"fixed|{b}", {})
        if s:
            rows.append(f"{lab} & -- & {s['n_claims']} & {s['n_true']} & {s['n_amb']} & {s['n_false_y0']} & "
                        f"{s['n_windows']} & {s['n_scenarios']} & {s['n_transitions_hit']} & {_f(o.get('ostr_pooled'), 3)} \\\\")
    ts = audit["transition_split"]
    body = ("\\begin{table*}[t]\n\\centering\\footnotesize\n\\caption{IoT-23 claim audit (exploratory): every certified "
            "claim under P-LOSO (each scenario calibrated on the other 22) and the rule baselines' claim sets, with the "
            "truth. Obs.\\ STR = distinct observable transitions hit / " + str(ts["n_T_obs"]) + " observable transitions.}\n"
            "\\label{tab:x2iot}\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{lccccccccc}\n\\toprule\n"
            "Method & $\\alpha$ & Claims & True & AMB & False & Windows & Scen. & Trans. & Obs.\\ STR \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\\\[2pt]\\raggedright\\footnotesize "
            f"$|T^*|={ts['n_T']}$: {ts['n_T_hid_R0']} hidden transitions are the definitional entry transition R0 "
            f"(one per malicious window), {ts['n_T_hid_other']} other hidden; {ts['n_T_obs']} observable, "
            f"{ts['n_T_obs_in_C']} with the enabling pair in $C$ ($\\rho_T$ {_f(ts['rho_T'], 3)} with R0, "
            f"{_f(ts['rho_T_without_R0'], 3)} without; ceiling {_f(ts['ceiling_with_R0'], 3)} vs "
            f"{_f(ts['ceiling_without_R0'], 3)} without R0).\n\\end{{table*}}\n")
    _write_tex("table_x2_iot23.tex", body)


def table_chains(nums: dict) -> None:
    rows = []
    for ds in CHAIN_DS:
        sid = RT.X2_DATASETS[ds]["sids"][0]
        for m in ("chain_composite", "chain_rt", "chain_rt_bh", "chain_naive"):
            for bin_ in ("all", "L=5+"):
                k0 = f"{ds}|{sid}|{m}|a={{a}}|{bin_}"
                if f"{k0.replace('{a}', '0.1')}|maxchain_fdr" not in nums:
                    continue
                cc = [_pair(nums, k0, c) for c in ("maxchain_fdr", "maxchain_fdr_pooled", "chain_exceed_a",
                                                     "cond_chain_exceed_a", "ccr_mean", "str_mean", "fdr", "frac_no_chain")]
                n = (nums.get(f"{k0.replace('{a}', '0.1')}|maxchain_fdr") or {}).get("n_pool")
                rows.append(f"{DS_LABEL[ds]} & {M_LABEL[m]} & {bin_} ({n}) & " + " & ".join(cc) + " \\\\")
        rows.append("\\addlinespace[1pt]")
    body = ("\\begin{table*}[t]\n\\centering\\footnotesize\n\\caption{Chains (exploratory rows): composite CRC vs "
            "coherent-factuality chain-RT vs naive read-off, $\\alpha=0.1$/$0.2$, all test incidents and the $L\\ge5$ "
            "stratum (pool size in brackets). Exceed = share of incidents whose maximal-chain FDP $>\\alpha$ (overall / "
            "given $\\ge1$ chain).}\n\\label{tab:x2chains}\n\\setlength{\\tabcolsep}{2.5pt}\n"
            "\\begin{tabular}{lllcccccccc}\n\\toprule\n"
            "Dataset & Rule & Incidents & Max-chain FDR & Pooled & Exceed & Exceed$\\mid$chain & CCR & STR & Edge FDR & "
            "No chain \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    _write_tex("table_x2_chains.tex", body)


def task_tables(a) -> None:
    def load(n):
        p = SUMMARY / f"{n}.json"
        return json.loads(p.read_text()) if p.exists() else None
    c = load("X2_conditional")
    if c:
        table_conditional(c["numbers"], c.get("gotham_real", []))
        table_conditional_full(c["numbers"])
        table_st(c["numbers"])
    au = load("X2_iot23_audit")
    if au:
        table_iot23(au)
    ch = load("X2_chains")
    if ch:
        table_chains(ch["numbers"])


# =========================================================================== addition A: DESIGN enabling rule
RELABEL = {"cf_ton": ("S_TON.stat", "S_TON.fuse_s0"), "cf_ton_h": ("S_TON_H.stat", "S_TON_H.fuse_s0")}
RELABEL_METHODS = ("crc_bh", "crc_st", "crc_fdx020", "ltt")


def task_relabel(a) -> None:
    from certichain.eval import revision_relabel as RL
    for ds, sids in RELABEL.items():
        if a.only and ds not in a.only:
            continue
        meta = RT.x2_root() / "banks" / ds / f"relabel_{RL.RULE_TAG}.json"
        if meta.exists() and not a.overwrite and all((RL.bank_path(ds, s) / "manifest.json").exists() for s in sids):
            log(f"relabel banks for {ds} exist")
        else:
            info = RL.build_relabel_banks(ds, sids, n_jobs=a.workers)
            log(f"relabel {ds}: {info['n_incidents']} incidents, {info['seconds']:.0f}s")
    tasks = []
    for ds, sids in RELABEL.items():
        d = RT.X2_DATASETS[ds]
        for sid in sids:
            for alpha in RT.ALPHAS:
                for m in RELABEL_METHODS:
                    tasks.append({"kind": "cond", "group": "relabel", "ds": ds, "sid": f"{sid}__{RL.RULE_TAG}",
                                  "bank_path": str(RL.bank_path(ds, sid)), "method": m, "alpha": alpha, "K": d["K"],
                                  "n_test": d["n_test"], "B": a.B, "overwrite": a.overwrite, "descriptors": True})
    pool_run(tasks, a.workers, "relabel cells")
    numbers_relabel()


SIDE_METRICS = ("frac_empty", "fdr", "cond_fdp", "pooled_fdr", "fdr_lenient", "pooled_fdr_lenient", "str_mean",
                "cer_link_mean", "n_claims", "n_false", "fdx_020", "wrongful_benign", "dev_false_impl")


def numbers_relabel() -> None:
    from certichain.eval import revision_relabel as RL
    rows, nums = [], {}
    for ds, sids in RELABEL.items():
        mp = RT.x2_root() / "banks" / ds / f"relabel_{RL.RULE_TAG}.json"
        info = json.loads(mp.read_text()) if mp.exists() else {}
        for sid in sids:
            for alpha in RT.ALPHAS:
                for m in RELABEL_METHODS:
                    pd_ = RT.x2_root() / "relabel" / ds / f"{m}__{sid}__{RL.RULE_TAG}" / f"a{alpha}" / "cell.json"
                    pr_ = RT.x2_root() / "conditional" / ds / f"{m}__{sid}" / f"a{alpha}" / "cell.json"
                    if not (pd_.exists() and pr_.exists()):
                        continue
                    cd, cr = json.loads(pd_.read_text()), json.loads(pr_.read_text())
                    row = {"ds": ds, "sid": sid, "method": m, "alpha": alpha}
                    for k in SIDE_METRICS:
                        dd, rr = cd["metrics"].get(k) or {}, cr["metrics"].get(k) or {}
                        row[k] = {"relation_aware": rr.get("point"), "relation_aware_ci": [rr.get("lo"), rr.get("hi")],
                                  "design_rule": dd.get("point"), "design_rule_ci": [dd.get("lo"), dd.get("hi")]}
                        nums[f"{ds}|{sid}|{m}|a={alpha}|{k}|design_rule"] = reg_entry(
                            dd.get("point"), RV.DEFS.get(k, k) + " -- cal/test pairs relabelled with the DESIGN "
                            "enabling rule (no entity condition); scores unchanged", ci=(dd.get("lo"), dd.get("hi"))
                            if dd.get("lo") is not None else None, se=dd.get("se"), source=X.rel(pd_.parent),
                            relation_aware=rr.get("point"))
                    if cd.get("descriptors"):
                        row["descriptors_design_rule"] = cd["descriptors"]
                    rows.append(row)
        nums[f"{ds}|relabel_info"] = reg_entry(None, "pooled transition / label statistics of the cal/test pool under "
                                               "both rules (verified: the relation-aware relabel reproduces the stored "
                                               "labels exactly)", pooled_stats=info.get("pooled_stats"),
                                               label_reasons_design=info.get("label_reasons_design"))
    write_summary("X2_relabel", {"definition": "sensitivity to the GT enabling rule (coordinator addition A, review R2-M4): "
                                 "the ORIGINAL DESIGN rule enab(A,B) = last unit of A starting before first(B) (no "
                                 "shared-entity condition) vs the relation-aware rule of the frozen banks; scorers "
                                 "unchanged (trained under the relation-aware rule); 200 PREREG splits; incident-pool "
                                 "bootstrap CIs", "rows": rows, "numbers": nums})


# =========================================================================== addition C: strong test-only injection
INJECT = {"cf_ton": ("S_TON.fuse_s0", "S_TON.stat")}
INJECT_Q = (1, 3, 10)


def task_inject(a) -> None:
    from certichain.eval import revision_inject as RI
    from certichain.eval import revision_relabel as RL
    for ds, sids in INJECT.items():
        for sid in sids:
            meta = RT.x2_root() / "banks" / ds / f"inject_above_{sid}.json"
            if meta.exists() and not a.overwrite and all((RL.bank_path(ds, sid, RI.bank_tag(q)) / "manifest.json").exists()
                                                         for q in INJECT_Q):
                log(f"inject banks {ds}/{sid} exist")
            else:
                info = RI.build_inject_banks(ds, sid, INJECT_Q, n_jobs=a.workers)
                log(f"inject banks {ds}/{sid}: top {info['top_value']:.3f} ({info['seconds']:.0f}s)")
    tasks = []
    for ds, sids in INJECT.items():
        d = RT.X2_DATASETS[ds]
        for sid in sids:
            for alpha in RT.ALPHAS:
                for m in ("crc_bh", "crc_st", "crc_fdx020", "ltt"):
                    for q in INJECT_Q:
                        tasks.append({"kind": "cond", "group": "inject", "ds": ds, "sid": sid, "method": m,
                                      "alpha": alpha, "K": d["K"], "n_test": d["n_test"], "B": a.B,
                                      "test_sid": f"{sid}__{RI.bank_tag(q)}", "inject_q": q,
                                      "test_bank_path": str(RL.bank_path(ds, sid, RI.bank_tag(q))),
                                      "overwrite": a.overwrite})
                        weak = f"{sid}__E4-inject-{q}"
                        if X.has_bank(ds, weak):          # E4's weak injection on the same splits (comparison)
                            tasks.append({"kind": "cond", "group": "inject", "ds": ds, "sid": sid, "method": m,
                                          "alpha": alpha, "K": d["K"], "n_test": d["n_test"], "B": a.B,
                                          "test_sid": weak, "inject_q": q,
                                          "test_bank_path": str(X.bank_dir(ds, weak)), "overwrite": a.overwrite})
    pool_run(tasks, a.workers, "inject cells")
    numbers_inject()


def numbers_inject() -> None:
    rows, nums = [], {}
    for p in sorted((RT.x2_root() / "inject").glob("*/*/a*/cell.json")):
        c = json.loads(p.read_text())
        cfg = c["config"]
        tsid = cfg.get("test_sid", "")
        kind = "strong (above calibration-pool max)" if "X2-inject-above" in tsid else "weak (E4: incident max + 1e-3)"
        q = cfg.get("inject_q")
        clean = RT.x2_root() / "conditional" / cfg["ds"] / f"{cfg['method']}__{cfg['sid']}" / f"a{cfg['alpha']}" / "cell.json"
        cm = json.loads(clean.read_text())["metrics"] if clean.exists() else {}
        row = {"ds": cfg["ds"], "sid": cfg["sid"], "method": cfg["method"], "alpha": cfg["alpha"], "q": q,
               "injection": kind, "remark2_bound": c.get("remark2_bound"), "source": X.rel(p.parent)}
        for k in ("fdr", "pooled_fdr", "frac_empty", "cond_fdp", "n_claims", "str_mean", "fdx_020"):
            d = c["metrics"].get(k) or {}
            row[k] = d.get("point")
            row[f"{k}_ci"] = [d.get("lo"), d.get("hi")]
            row[f"{k}_clean"] = (cm.get(k) or {}).get("point")
        row["fdr_inflation"] = (row["fdr"] - row["fdr_clean"]) if row["fdr"] is not None and row["fdr_clean"] is not None \
            else None
        rows.append(row)
        tag = "strong" if "X2-inject-above" in tsid else "weak"
        for k in ("fdr", "pooled_fdr", "fdr_inflation", "remark2_bound", "frac_empty", "str_mean"):
            nums[f"{cfg['ds']}|{cfg['sid']}|{cfg['method']}|a={cfg['alpha']}|{tag}|q={q}|{k}"] = reg_entry(
                row.get(k), f"test-only injection ({kind}), q = {q} nulls per test incident: {k}"
                + (" (vs the clean bank, same splits)" if k == "fdr_inflation" else ""),
                ci=row.get(f"{k}_ci") if row.get(f"{k}_ci") and row.get(f"{k}_ci")[0] is not None else None,
                source=row["source"])
    write_summary("X2_inject", {"definition": "stronger test-only injection (coordinator addition C, review R2-M3): q "
                                "null candidates per TEST incident raised above the maximum score of the calibration "
                                "pool and of S0 (strong) vs E4's incident max + 1e-3 (weak; P2-F's banks, recomputed on "
                                "the same 200 splits); calibration rows from the clean bank; remark2_bound = alpha + "
                                "E[min(1, q/max(|S'|,1))] (meaningful for the score-threshold family)",
                                "rows": rows, "numbers": nums})


# =========================================================================== task 6: Gotham cross-fit (generation + 07)
def task_crossfit_gen(a) -> None:
    from certichain.chainforge import audit as AU
    from certichain.eval import revision_crossfit as XF
    tasks = []
    for variant, split, n in ((XF.VAR_A, "train", 100), (XF.VAR_B, "caltest", 150)):
        idx = list(range(n))
        for k in range(a.workers):
            ch = idx[k::a.workers]
            if ch:
                tasks.append({"variant": variant, "split": split, "idx": ch})
    ctx = mp.get_context("spawn")
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(a.workers, mp_context=ctx) as ex:
        for f in as_completed([ex.submit(XF.gen_chunk, t) for t in tasks]):
            r = f.result()
            res.append(r)
            for e in r["errors"]:
                log(f"generation error: {e}")
    log(f"generated {sum(r['done'] for r in res)} incidents ({time.time() - t0:.0f}s); "
        f"errors {sum(len(r['errors']) for r in res)}")
    _, _, members = XF.make_libraries()
    chk = XF.reuse_check(members)
    log(f"episode use: {chk}")
    for variant, as_v in ((XF.VAR_A, "cf_gotham"), (XF.VAR_B, "cf_gotham_t")):
        log(json.dumps(AU.run_p2b_07(variant, as_v, workers=a.workers)))
    atomic_write_json(RT.x2_root() / "crossfit" / "generation.json",
                      {"half_A": XF.HALF_A, "half_B": XF.HALF_B, "episode_use": chk, "chunks": res,
                       "cfg_A": XF.CFG_A, "cfg_B": XF.CFG_B, "status": STATUS})


def task_crossfit_eval(a) -> None:
    """Gotham-real case study with the cross-fitted scorer S_GOTXA (trained on train-pool half A) and the half-B bank
    CF-GOTHAM-XB, next to the confounded in-domain run (S_GOT2 + CF-GOTHAM-T v2 bank) and the S_TON transfer run."""
    from certichain.eval.gates import bank_within_auc
    from certichain.eval import revision_crossfit as XF
    settings = (("S_GOTXA cross-fit: scorer half A, bank half B (CF-GOTHAM-XB)", ("cf_gotham_xb", "S_GOTXA.stat"),
                 ("gotham_real", "S_GOTXA.stat"), "xf"),) + tuple(GOTHAM_SETTINGS)
    rows, wauc = [], {}
    for name, cal, test, tag in settings:
        if not (X.has_bank(*cal) and X.has_bank(*test)):
            log(f"crossfit eval: missing bank for {tag}")
            continue
        for r in RT.gotham_rows(cal, test, GOTHAM_METHODS):
            r["setting"], r["tag"] = name, tag
            rows.append(r)
        for ds_, sid_ in (cal, test):
            w = bank_within_auc(X.load_bank(ds_, sid_))
            wauc[f"{ds_}|{sid_}"] = {"mean": w["mean"], "n_incidents": w["n_incidents"]}
    ts = {}
    try:
        Xa, _ = X.phi_matrix("cf_gotham_xb", "caltest")
        Xb = _phi("gotham_real", None)
        perm = RV.permutation_auc(Xa, Xb, P=2000 if a.B >= 1000 else 50, seed=0)
        ts = {k: v for k, v in perm.items() if k != "null"}
    except Exception as e:  # pragma: no cover
        ts = {"error": str(e)}
    gen = json.loads((RT.x2_root() / "crossfit" / "generation.json").read_text()) \
        if (RT.x2_root() / "crossfit" / "generation.json").exists() else {}
    meta = json.loads((X.results_root() / "models" / "S_GOTXA" / "meta.json").read_text()) \
        if (X.results_root() / "models" / "S_GOTXA" / "meta.json").exists() else {}
    descr = X.descriptors(X.load_bank("cf_gotham_xb", "S_GOTXA.stat")) if X.has_bank("cf_gotham_xb", "S_GOTXA.stat") else {}
    write_summary("X2_gotham_crossfit", {
        "definition": "task 6 (review R3-M4): Gotham TRAIN-pool devices split by the library episode's victim device into "
                      f"half A {list(XF.HALF_A)} (scorer S_GOTXA trained on 100 CF-GOTHAM-XA incidents) and half B "
                      f"{list(XF.HALF_B)} (bank CF-GOTHAM-XB, 150 incidents, case-study templates); P2-E generator, "
                      "P2-B 07, P2-F 08 / 11 code unchanged; lambda-hat from all 150 bank incidents; Gotham-real "
                      "counts (n = 9 incidents); shared background capture remains (stated limitation)",
        "rows": rows, "within_incident_auc": wauc, "two_sample_auc_xb_vs_gotham_real": ts,
        "episode_use": gen.get("episode_use"), "scorer_meta": {k: meta.get(k) for k in
                                                              ("oof_within_auc_strict", "oof_within_auc_y0", "n_s0",
                                                               "final_rounds", "n_train_incidents") if k in meta},
        "bank_descriptors": descr})


TASKS_DEF = {"iot23_audit": task_iot23_audit, "chains": task_chains, "stats": task_stats, "easiness": task_easiness,
             "tables": task_tables, "relabel": task_relabel, "inject": task_inject, "crossfit_gen": task_crossfit_gen,
             "crossfit_eval": task_crossfit_eval}


# =========================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--B", type=int, default=RT.BOOT_B)
    ap.add_argument("--B-b8", dest="B_b8", type=int, default=500, help="bootstrap replicates for B8 (slow Platt refits)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--numbers-only", action="store_true")
    a = ap.parse_args()
    a.workers = min(8, a.workers or workers(default=8))
    a.only = [s for s in a.only.split(",") if s]
    X.load_prereg()                    # verifies the frozen BLAKE3 (read-only)
    tasks = a.task.split(",")
    try:
        for t in tasks:
            log(f"task {t} (workers {a.workers}, B {a.B})")
            if t == "conditional":
                numbers_conditional(a) if a.numbers_only else task_conditional(a)
            elif t in TASKS:
                TASKS[t](a)
            else:
                raise SystemExit(f"unknown task {t}")
    except MemoryBudgetExceeded as e:
        log(f"memory budget: {e}")
        return EXIT_MEMORY
    return 0


TASKS: dict = dict(TASKS_DEF)

if __name__ == "__main__":
    sys.exit(main())
