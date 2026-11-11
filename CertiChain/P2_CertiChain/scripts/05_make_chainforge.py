#!/usr/bin/env python
"""05 — ChainForge-IoT incidents (DESIGN §5.4.3 Alg. 7). Owner: P2-E.

Run from the project root through memguard (P2 budget 90 GB, ≤ 8 workers):

    .venv/bin/python memguard/memguard.py run --name p2_cf_ton --limit-gb 90 --workers 8 -- \
        .venv/bin/python -I -u P2_CertiChain/scripts/05_make_chainforge.py --variant cf_ton

    --variant   cf_ton | cf_ton_h | cf_gotham | cf_gotham_t     (config: configs/<variant>.yaml)
    --splits    comma list (default: every split of the config)   --n N   override every split size
    --only      comma list of <split>/<idx> (debug)   --workers W   --force (redo existing incidents)

Requires scripts/04_build_libraries.py --variant <ds>. One directory per incident
(results/interim/<variant>/incidents/<split>/<NNNNNN>/: records, labels [Q], meta, gt [Q], gen [Q]),
written atomically; existing complete incidents are skipped (resumable). At the end:
results/interim/<variant>/index.parquet (per-incident summary, GENERATOR-SIDE) and summary.json
(counts, reuse, source-level disjointness check).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402
import yaml  # noqa: E402

from certichain.io.hashing import config_hash, write_run_manifest  # noqa: E402
from certichain.io.paths import (CONFIG_DIR, atomic_write_json, atomic_write_parquet, clean_tmp, interim, log,  # noqa: E402
                                 peak_rss_gb, run_pool)

FILES = ("records.parquet", "labels.parquet", "meta.json", "gt.json", "gen.json")
_CACHE: dict = {}


def load_cfg(variant: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{variant}.yaml").read_text())


def _libs(ds: str):
    if ds not in _CACHE:
        from certichain.chainforge.background import BackgroundLibrary
        from certichain.chainforge.episodes import EpisodeLibrary
        _CACHE[ds] = (EpisodeLibrary.load(ds), BackgroundLibrary.load(ds))
    return _CACHE[ds]


def is_complete(d: Path) -> bool:
    return all((d / f).exists() for f in FILES)


def check_disjoint(out: dict, lib, bglib) -> None:
    """Source-level disjointness: every spliced source belongs to the incident's pool."""
    spec = out["spec"]
    for c in spec.campaigns + spec.decoys:
        for e in c.episodes.values():
            if e.lib_eid and lib.row(e.lib_eid)["pool"] != spec.pool:
                raise AssertionError(f"{out['meta']['inc_id']}: episode {e.lib_eid} not from pool {spec.pool}")
            if e.la_id:
                la = bglib.lookalikes(spec.pool).filter(pl.col("la_id") == e.la_id)
                if la.height != 1:
                    raise AssertionError(f"{out['meta']['inc_id']}: look-alike {e.la_id} not in pool {spec.pool}")
    if out["meta"]["pool"] != spec.pool:
        raise AssertionError("meta pool mismatch")


def work(task: dict) -> dict:
    from certichain.chainforge import generator as G
    from certichain.utils.mem import worker_check

    variant, split, idx, cfg, force = task["variant"], task["split"], task["idx"], task["cfg"], task["force"]
    d = G.incident_dir(variant, split, idx)
    if is_complete(d) and not force:
        try:
            return {"inc_id": G.inc_id(variant, split, idx), "skipped": True,
                    **json.loads((d / "gen.json").read_text()).get("summary", {})}
        except (OSError, json.JSONDecodeError):
            pass
    t = time.time()
    lib, bglib = _libs(cfg["dataset"])
    worker_check(every=1, default_workers=8)
    out = G.generate_incident(variant, split, idx, cfg, lib, bglib)
    check_disjoint(out, lib, bglib)
    summ = G.incident_summary(out)
    summ["gen_s"] = round(time.time() - t, 2)
    out["gen"]["summary"] = summ
    G.write_incident(out, d)
    worker_check(every=1, default_workers=8)
    return {**summ, "skipped": False}


def summarize(variant: str, cfg: dict, rows: list[dict]) -> dict:
    df = pl.DataFrame([{k: v for k, v in r.items() if k not in ("log", "window")} for r in rows],
                      infer_schema_length=None)
    out = {"variant": variant, "config_hash": config_hash(cfg), "n_incidents": df.height,
           "per_split": {}, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    for split in sorted(df["split_role"].unique().to_list()):
        s = df.filter(pl.col("split_role") == split)
        tmpl = s.select("templates").explode("templates").drop_nulls()["templates"].value_counts().sort("templates")
        out["per_split"][split] = {
            "n": s.height,
            "records_per_incident": {q: float(s["n_records"].quantile(q)) for q in (0.0, 0.1, 0.5, 0.9, 1.0)},
            "campaigns": {str(k): int(v) for k, v in s["n_campaigns"].value_counts().sort("n_campaigns").iter_rows()},
            "templates": {k: int(v) for k, v in tmpl.iter_rows()},
            "decoys": {"total": int(s["n_decoys"].sum()), "D1": int(s["n_D1"].sum()), "D2": int(s["n_D2"].sum()),
                       "D3": int(s["n_D3"].sum())},
            "T_star": {"total": int(s["n_T"].sum()), "mean": float(s["n_T"].mean()),
                       "hidden_endpoint": int(s["n_T_hidden_endpoint"].sum())},
            "hidden_episodes": int(s["n_hidden"].sum()),
            "window_min": {q: float(s["L_s"].quantile(q)) / 60 for q in (0.0, 0.5, 1.0)},
        }
    # reuse: incidents per library episode / look-alike, per split role
    reuse = {}
    for split in sorted(df["split_role"].unique().to_list()):
        s = df.filter(pl.col("split_role") == split)
        ex = s.select("inc_id", "lib_eids").explode("lib_eids").drop_nulls()
        reuse[split] = {"n_distinct_episodes": int(ex["lib_eids"].n_unique()) if ex.height else 0,
                        "mean_incidents_per_used_episode": (float(ex.group_by("lib_eids").len()["len"].mean())
                                                            if ex.height else 0.0)}
    out["reuse"] = reuse
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["cf_ton", "cf_ton_h", "cf_gotham", "cf_gotham_t", "cf_gotham_v2", "cf_gotham_t_v2"])
    ap.add_argument("--splits", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    from certichain.utils.mem import workers as mg_workers

    t0 = time.time()
    cfg = load_cfg(a.variant)
    root = interim(a.variant, "incidents")
    root.mkdir(parents=True, exist_ok=True)
    n_tmp = clean_tmp(root)
    if n_tmp:
        log(f"removed {n_tmp} stale tmp files")
    for p in root.glob("*/*.tmp-*"):
        if p.is_dir():
            import shutil
            shutil.rmtree(p, ignore_errors=True)
    splits = a.splits.split(",") if a.splits else list(cfg["splits"])
    tasks = []
    for sp in splits:
        n = a.n if a.n is not None else int(cfg["splits"][sp])
        for i in range(n):
            if a.only and f"{sp}/{i}" not in a.only.split(","):
                continue
            tasks.append({"variant": a.variant, "split": sp, "idx": i, "cfg": cfg, "force": a.force})
    W = min(8, a.workers or mg_workers(default=8))
    log(f"{a.variant}: {len(tasks)} incidents ({', '.join(splits)}) with {W} workers")
    done = {"n": 0}

    def on_result(r):
        done["n"] += 1
        if done["n"] % 25 == 0 or done["n"] == len(tasks):
            log(f"  {done['n']}/{len(tasks)} incidents ({time.time() - t0:.0f}s)")

    if W <= 1:
        results, errors = [], []
        from certichain.utils.mem import check_memory
        for tk in tasks:
            check_memory()
            try:
                results.append(work(tk))
                on_result(results[-1])
            except Exception as e:  # noqa: BLE001
                import traceback
                errors.append({"task": f"{tk['split']}/{tk['idx']}", "error": repr(e), "traceback": traceback.format_exc()})
                log(f"FAILED {tk['split']}/{tk['idx']}: {e!r}")
    else:
        results, errors = run_pool(work, tasks, W, on_result=on_result, label=lambda t: f"{t['split']}/{t['idx']}")
    if errors:
        log(f"{len(errors)} incidents FAILED (first: {errors[0]['task']}: {errors[0]['error']})")
        atomic_write_json(interim(a.variant, "errors.json"), errors)
    # full index from disk (covers skipped incidents)
    rows = []
    for gp in sorted(root.glob("*/*/gen.json")):
        try:
            g = json.loads(gp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "summary" in g:
            rows.append(g["summary"])
    if rows:
        idx = pl.DataFrame([{k: (json.dumps(v) if k in ("log",) else v) for k, v in r.items()} for r in rows],
                           infer_schema_length=None)
        atomic_write_parquet(idx, interim(a.variant, "index.parquet"))
        summ = summarize(a.variant, cfg, rows)
        # source-level disjointness across splits (episodes and look-alikes)
        df = idx.select("split_role", "lib_eids", "la_ids")
        from certichain.chainforge.generator import source_pool
        pools = {}
        for sp in df["split_role"].unique().to_list():
            s = df.filter(pl.col("split_role") == sp)
            pools.setdefault(source_pool(sp, cfg), set()).update(s.select("lib_eids").explode("lib_eids").drop_nulls()["lib_eids"].to_list())
        overlap = sorted(pools.get("train", set()) & pools.get("caltest", set()))
        summ["source_disjoint"] = {"train_caltest_episode_overlap": len(overlap), "ok": not overlap}
        if overlap:
            raise SystemExit(f"source-level disjointness violated: {overlap[:5]}")
        summ["n_errors"] = len(errors)
        summ["runtime_s"] = round(time.time() - t0, 1)
        atomic_write_json(interim(a.variant, "summary.json"), summ)
        log(f"summary: {json.dumps(summ['per_split'])[:600]}")
    write_run_manifest(f"p2_cf_{a.variant}", {"variant": a.variant, "cfg": cfg, "splits": splits, "n": a.n},
                       extra={"n_tasks": len(tasks), "n_errors": len(errors), "runtime_s": round(time.time() - t0, 1),
                              "peak_rss_gb_driver": peak_rss_gb()})
    log(f"done: {len(results)} ok, {len(errors)} failed in {time.time() - t0:.0f}s")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
