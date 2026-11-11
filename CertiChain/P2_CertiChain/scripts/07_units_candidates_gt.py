#!/usr/bin/env python
"""07 — per incident: records → scope → event units → candidates → GT pair labels.

DESIGN §3.2 (Alg. 1), §3.3 (Alg. 2), §2.3 (one GT semantics), §5.2 (per-source GT), §5.5.
Owner: P2-B.

Run from the project root under memguard (P2 budget 90 GB, ≤ 8 workers):

  .venv/bin/python memguard/memguard.py run --name p2_units_iot23 --limit-gb 90 --workers 8 -- \
      .venv/bin/python -I -u P2_CertiChain/scripts/07_units_candidates_gt.py --dataset iot23 --gate

  --dataset  iot23 | gotham_real | cf_ton | cf_ton_h | cf_gotham | cf_gotham_t | chainforge-<variant>
             (gotham = the older config-driven packaging of the case-study windows; superseded by gotham_real)
  --workers  pool size (default min(8, MEMGUARD_WORKERS));  --big-records N: incidents with more
             records are processed one at a time in the driver (check_memory between stages)
  --only a,b --limit K --overwrite;  --gate (iot23 only): write results/gates/G0_sealed.json
  --summary-only: rebuild the dataset summary / gate from existing stats files

Outputs under results/interim/<ds>/ (atomic writes; stats/<inc>.json is written last and marks
completion, so re-running skips finished incidents):
  units/<inc>.parquet           UNIT_SCHEMA
  units_side/<inc>.parquet      unit_id + label-free extras (n_conns, touches_seed, hop, seed_dist, attck, …)
  scope/<inc>.json              seeds, INFRA tokens, S with hops, counts
  candidates/<inc>.parquet      CANDIDATE_SCHEMA
  candidates_cont/<inc>.parquet self-continuation pairs C_cont (u, v, rel_mask, lags)
  labels_pairs/<inc>.parquet    PAIR_LABEL_SCHEMA            ── QUARANTINED (GT side)
  gt_units/<inc>.parquet        unit_id → episode, campaign  ── QUARANTINED
  gt/<inc>.json                 T*, hidden, amb pairs        ── QUARANTINED
  transitions/<inc>.parquet     per-transition enab/first/in_C ── QUARANTINED
  stats/<inc>.json              sizes, runtimes, unit/candidate info, transition stats
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "16")   # driver; pool workers get 4 (set before spawning)

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import yaml  # noqa: E402

from certichain.candidates.generate import CandParams, generate_candidates  # noqa: E402
from certichain.gt import semantics as S  # noqa: E402
from certichain.gt.observability import pooled  # noqa: E402
from certichain.io.paths import atomic_write_json, atomic_write_parquet  # noqa: E402
from certichain.io.schema import RECORD_SCHEMA  # noqa: E402
from certichain.units.sessionize import NEEDED_COLS, UnitParams, event_units  # noqa: E402
from certichain.utils.mem import (EXIT_MEMORY, MemoryBudgetExceeded, check_memory, usage_gb,  # noqa: E402
                                  worker_check, workers as mg_workers)

P2 = Path(__file__).resolve().parents[1]
INTERIM = P2 / "results" / "interim"          # inputs (records, labels, windows.json, incidents)
OUT = INTERIM                                 # outputs (redirect with --out-root for smoke runs)
GATES = P2 / "results" / "gates"
CFG_PATH = P2 / "configs" / "units.yaml"
CF_VARIANTS = ("cf_ton", "cf_ton_h", "cf_gotham", "cf_gotham_t")
OUT_DIRS = ("units", "units_side", "scope", "candidates", "candidates_cont", "labels_pairs", "gt_units", "gt",
            "transitions", "stats")
READ_COLS = [c for c in NEEDED_COLS]          # never src_file / src_sha256 / src_locator (pointers stay on disk)


def log(msg: str) -> None:
    print(f"[07 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cfg() -> dict:
    return yaml.safe_load(CFG_PATH.read_text())


def ds_name(dataset: str) -> str:
    d = dataset.lower()
    if d.startswith("chainforge-"):
        d = d[len("chainforge-"):]
    return d.replace("-", "_") if d.startswith("cf") else d


# --------------------------------------------------------------------------- incidents
def _utc(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def incidents_iot23() -> list[dict]:
    w = json.loads((INTERIM / "iot23" / "windows.json").read_text())
    out = []
    for x in w["windows"]:                          # case metadata only: window id, seed, time span
        rec = INTERIM / "iot23" / "records" / x["scenario"] / f"{x['window_id']}.parquet"
        lab = INTERIM / "iot23" / "labels" / x["scenario"] / f"{x['window_id']}.parquet"
        out.append({"ds": "iot23", "inc_id": x["window_id"], "scenario": x["scenario"], "seeds": [x["seed"]],
                    "t0": x["t_start"], "t1": x["t_end"], "records": str(rec), "labels": str(lab),
                    "n_records": int(x["n_records"]), "kind": x.get("kind")})
    return out


def incidents_chainforge(variant: str) -> list[dict]:
    out = []
    for meta_p in sorted(glob.glob(str(INTERIM / variant / "incidents" / "**" / "meta.json"), recursive=True)):
        d = Path(meta_p).parent
        meta = json.loads(Path(meta_p).read_text())
        seeds = meta.get("seed_device")
        seeds = seeds if isinstance(seeds, list) else [seeds]
        out.append({"ds": variant, "inc_id": meta.get("inc_id") or str(d.relative_to(INTERIM / variant / "incidents")),
                    "seeds": seeds, "t0": meta.get("t0"), "t1": meta.get("t1"), "records": str(d / "records.parquet"),
                    "incident_dir": str(d), "internal_prefixes": meta.get("internal_prefixes") or [],
                    "n_records": None})
    return out


def incidents_gotham(cfg: dict) -> list[dict]:
    out = []
    for inc in cfg.get("gotham", {}).get("incidents", []):
        t0, t1 = _utc(inc["window"][0]), _utc(inc["window"][1])
        seeds = inc["seeds"]
        groups = [[s] for s in seeds] if inc.get("one_incident_per_seed", True) else [seeds]
        for g in groups:
            out.append({"ds": "gotham", "inc_id": f"{inc['name']}__{'+'.join(g)}", "seeds": g, "t0": t0, "t1": t1,
                        "session": inc["session"], "n_records": None})
    return out


def incidents_gotham_real() -> list[dict]:
    """P2-E's packaging: results/interim/gotham_real/incidents/<id>/{records,labels}.parquet + meta.json."""
    out = []
    for meta_p in sorted(glob.glob(str(INTERIM / "gotham_real" / "incidents" / "*" / "meta.json"))):
        d = Path(meta_p).parent
        meta = json.loads(Path(meta_p).read_text())
        seeds = meta.get("seed_device")
        seeds = seeds if isinstance(seeds, list) else [seeds]
        out.append({"ds": "gotham_real", "inc_id": meta["inc_id"], "seeds": seeds, "t0": meta.get("t0"),
                    "t1": meta.get("t1"), "session": meta["session"], "records": str(d / "records.parquet"),
                    "labels": str(d / "labels.parquet"), "incident_dir": str(d),
                    "internal_prefixes": meta.get("internal_prefixes") or [], "n_records": None})
    return out


def list_incidents(ds: str, cfg: dict) -> list[dict]:
    if ds == "iot23":
        return incidents_iot23()
    if ds == "gotham":
        return incidents_gotham(cfg)
    if ds == "gotham_real":
        return incidents_gotham_real()
    if ds in CF_VARIANTS:
        return incidents_chainforge(ds)
    raise SystemExit(f"unknown dataset {ds}")


# --------------------------------------------------------------------------- record loading
def _all_null_columns(path: str) -> set[str]:
    """Columns whose parquet statistics say every value is null (skipped at read time)."""
    import pyarrow.parquet as pq
    md = pq.ParquetFile(path).metadata
    out = None
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        nulls = set()
        for j in range(rg.num_columns):
            c = rg.column(j)
            st = c.statistics
            if st is not None and st.has_null_count and st.null_count == rg.num_rows:
                nulls.add(c.path_in_schema)
        out = nulls if out is None else (out & nulls)
    return out or set()


def _read_records(path: str) -> pl.DataFrame:
    cols = pl.read_parquet_schema(path)
    skip = _all_null_columns(path) & {"dns_q", "dns_rcode", "http_method", "http_uri", "http_host", "http_ua",
                                      "http_status", "http_resp_mime", "tls_sni"}
    return pl.read_parquet(path, columns=[c for c in READ_COLS if c in cols and c not in skip])


def load_gotham_records(inc: dict) -> tuple[pl.DataFrame, list[str]]:
    """All device captures inside the window, flows seen by two captures de-duplicated
    (same 5-tuple, starts < 1 s apart → keep the copy whose capture file sorts first)."""
    rec_files = sorted(glob.glob(str(INTERIM / "gotham" / "records" / "*.parquet")))
    parts = []
    for f in rec_files:
        cols = [c for c in READ_COLS if c in pl.read_parquet_schema(f)]
        df = pl.scan_parquet(f).filter((pl.col("ts") >= inc["t0"]) & (pl.col("ts") <= inc["t1"])).select(cols).collect()
        if df.height:
            parts.append(df.with_columns(pl.lit(Path(f).stem).alias("_cap")))
    if not parts:
        return pl.DataFrame(schema={c: RECORD_SCHEMA[c] for c in READ_COLS}), []
    df = pl.concat(parts, how="diagonal_relaxed")
    k = ["src", "sport", "dst", "dport", "proto"]
    df = df.sort(k + ["ts", "_cap"], nulls_last=True)
    same = pl.all_horizontal([(pl.col(c) == pl.col(c).shift(1)).fill_null(pl.col(c).is_null() & pl.col(c).shift(1).is_null())
                              for c in k])
    dup = same & ((pl.col("ts") - pl.col("ts").shift(1)) < 1.0) & (pl.col("_cap") != pl.col("_cap").shift(1))
    df = df.filter(~dup.fill_null(False)).drop("_cap")
    lab_files = [str(INTERIM / "gotham" / "labels" / Path(f).name) for f in rec_files]
    return df, [p for p in lab_files if Path(p).exists()]


def load_records(inc: dict) -> tuple[pl.DataFrame, dict]:
    extra = {}
    if inc["ds"] == "gotham":
        df, extra["label_paths"] = load_gotham_records(inc)
    else:
        df = _read_records(inc["records"])
    return df, extra


# --------------------------------------------------------------------------- GT
def build_gt(inc: dict, units: pl.DataFrame, records: pl.DataFrame, extra: dict, cfg: dict,
             seed_priority: bool | None = None) -> S.EpisodeGT:
    ds = inc["ds"]
    if ds == "gotham_real":
        from certichain.gt.gotham_episodes import GothamRoles, gotham_gt
        roles = GothamRoles.from_config(cfg.get("gotham", {}).get("roles"))
        sp = True if seed_priority is None else seed_priority
        return gotham_gt(records, units, inc["session"], [inc["labels"]], roles, seeds=inc["seeds"], seed_priority=sp)
    if ds == "iot23":
        from certichain.gt.iot23_episodes import iot23_gt
        return iot23_gt(inc["scenario"], inc["inc_id"], units, inc["seeds"], labels_path=inc["labels"])
    if ds in CF_VARIANTS:
        from certichain.gt.chainforge_episodes import chainforge_gt
        return chainforge_gt(inc["incident_dir"], units)
    if ds == "gotham":
        from certichain.gt.gotham_episodes import GothamRoles, gotham_gt
        roles = GothamRoles.from_config(cfg.get("gotham", {}).get("roles"))
        return gotham_gt(records, units, inc["session"], extra.get("label_paths", []), roles)
    raise ValueError(ds)


# --------------------------------------------------------------------------- one incident
def out_path(ds: str, sub: str, inc_id: str, ext: str) -> Path:
    return OUT / ds / sub / f"{inc_id}.{ext}"


def process_incident(inc: dict, cfg: dict, in_worker: bool) -> dict:
    check = (lambda: worker_check(every=1)) if in_worker else (lambda: check_memory())
    ds, inc_id = inc["ds"], inc["inc_id"]
    t_all = time.perf_counter()
    times = {}
    t = time.perf_counter()
    recs, extra = load_records(inc)
    times["load_s"] = time.perf_counter() - t
    check()
    up = UnitParams.from_config(cfg.get("units"))
    cp = CandParams.from_config(cfg.get("candidates"))
    prefixes = inc.get("internal_prefixes") or (cfg.get("internal_prefixes", {}) or {}).get(ds) or []
    t = time.perf_counter()
    n_records = recs.height
    if ds in ("gotham", "gotham_real"):     # only the Gotham GT reads record fields afterwards
        eu = event_units(recs, inc["seeds"], up, prefixes)
    else:                                   # hand the frame over (no reference kept) so it is freed early
        box = [recs]
        del recs
        recs = None
        eu = event_units(box, inc["seeds"], up, prefixes)
    times["units_s"] = time.perf_counter() - t
    check()
    t = time.perf_counter()
    cr = generate_candidates(eu.units, eu.scope, cp, side=eu.side)
    times["candidates_s"] = time.perf_counter() - t
    check()
    t = time.perf_counter()
    gt = build_gt(inc, eu.units, recs, extra, cfg)
    labels, reasons = S.label_pairs(cr.cands, eu.units, gt, scope=eu.scope, return_reasons=True)
    stats_t = S.transition_stats(cr.cands, eu.units, gt, labels, scope=eu.scope)
    stats_t["label_reasons"] = reasons
    trans = S.transition_table(cr.cands, eu.units, gt, scope=eu.scope)
    stats_default = None
    if ds == "gotham_real":                 # default unit→episode rule, reported side by side
        gt_d = build_gt(inc, eu.units, recs, extra, cfg, seed_priority=False)
        stats_default = S.transition_stats(cr.cands, eu.units, gt_d,
                                           S.label_pairs(cr.cands, eu.units, gt_d, scope=eu.scope), scope=eu.scope)
    times["gt_s"] = time.perf_counter() - t
    del recs
    check()
    t = time.perf_counter()
    atomic_write_parquet(eu.units, out_path(ds, "units", inc_id, "parquet"))
    atomic_write_parquet(eu.side, out_path(ds, "units_side", inc_id, "parquet"))
    sc = eu.scope.to_json()
    sc["info"] = {**sc.get("info", {}), **{k: v for k, v in eu.info.items()}}
    atomic_write_json(out_path(ds, "scope", inc_id, "json"), sc)
    atomic_write_parquet(cr.cands, out_path(ds, "candidates", inc_id, "parquet"))
    atomic_write_parquet(cr.cont, out_path(ds, "candidates_cont", inc_id, "parquet"))
    atomic_write_parquet(labels, out_path(ds, "labels_pairs", inc_id, "parquet"))
    atomic_write_parquet(gt.units_frame(), out_path(ds, "gt_units", inc_id, "parquet"))
    atomic_write_json(out_path(ds, "gt", inc_id, "json"), gt.to_json())
    atomic_write_parquet(trans, out_path(ds, "transitions", inc_id, "parquet"))
    times["write_s"] = time.perf_counter() - t
    times["total_s"] = time.perf_counter() - t_all
    stats = {"ds": ds, "inc_id": inc_id, "scenario": inc.get("scenario"), "session": inc.get("session"),
             "seeds": inc["seeds"], "n_records": n_records, "n_units": eu.units.height,
             "N": cr.cands.height, "N_max_hit": bool(cr.info.get("N_max_hit")),
             "U_max_hit": bool(eu.info.get("U_max_hit")), "unit_info": eu.info, "cand_info": cr.info,
             "transition_stats": stats_t, "gt_meta": gt.meta, "times": times,
             **({"transition_stats_default_rule": stats_default} if stats_default is not None else {}),
             "peak_rss_gb_proc": _peak_rss_gb(), "pid": os.getpid(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "code_fp": code_fingerprint(cfg)}
    atomic_write_json(out_path(ds, "stats", inc_id, "json"), stats)
    return {"inc_id": inc_id, "ok": True, "N": stats["N"], "n_units": stats["n_units"], "n_records": n_records,
            "total_s": times["total_s"]}


def _peak_rss_gb() -> float:
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3) if sys.platform == "darwin" else r / (1024 ** 2)


def _worker(args):
    inc, cfg, out_root = args
    global OUT
    OUT = Path(out_root)
    try:
        return process_incident(inc, cfg, in_worker=True)
    except MemoryBudgetExceeded as e:
        return {"inc_id": inc["inc_id"], "ok": False, "memory": True, "error": str(e)}
    except Exception as e:  # noqa: BLE001 — logged, the run continues
        return {"inc_id": inc["inc_id"], "ok": False, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}


# --------------------------------------------------------------------------- summaries / gate
def _q(xs, qs=(0, 10, 25, 50, 75, 90, 100)) -> dict:
    if not xs:
        return {}
    a = np.asarray(xs, dtype=float)
    return {f"p{q}": float(np.percentile(a, q)) for q in qs} | {"mean": float(a.mean()), "n": int(a.size)}


def summarize(ds: str) -> dict:
    files = sorted(glob.glob(str(OUT / ds / "stats" / "**" / "*.json"), recursive=True))
    files = [f for f in files if not Path(f).name.startswith("_")]
    st = [json.loads(Path(f).read_text()) for f in files]
    if not st:
        return {"ds": ds, "n_incidents": 0}
    N = [s["N"] for s in st]
    ts = [s["transition_stats"] for s in st]
    recalled = [t.get("n_T_obs_in_C", 0) for t in ts]
    per_inc_nr = [s["N"] / t["n_T_obs_in_C"] for s, t in zip(st, ts) if t.get("n_T_obs_in_C")]
    pos = [t.get("n_pos", 0) for t in ts]
    out = {
        "ds": ds, "n_incidents": len(st),
        "N": _q(N), "N_total": int(sum(N)), "n_units": _q([s["n_units"] for s in st]),
        "n_records": _q([s["n_records"] for s in st]),
        "N_max_hit": sum(s["N_max_hit"] for s in st), "U_max_hit": sum(s["U_max_hit"] for s in st),
        "observability": pooled(ts),
        "N_over_r": {"definition": "candidates per observable transition whose enabling pair is in C",
                     "pooled": (sum(N) / sum(recalled)) if sum(recalled) else None,
                     "per_incident": _q(per_inc_nr),
                     "n_incidents_with_recalled_transition": len(per_inc_nr)},
        "N_over_pos": {"definition": "candidates per Y=1 candidate (positives incl. bridges)",
                       "pooled": (sum(N) / sum(pos)) if sum(pos) else None},
        "labels": {"n_pos": int(sum(pos)), "n_null": int(sum(t.get("n_null", 0) for t in ts)),
                   "n_amb": int(sum(t.get("n_amb", 0) for t in ts)),
                   "n_pos_bridge": int(sum(t.get("n_pos_bridge", 0) for t in ts))},
        "runtime_s": {k: _q([s["times"].get(k, 0.0) for s in st])
                      for k in ("total_s", "load_s", "units_s", "candidates_s", "gt_s")},
        "peak_rss_gb_proc": _q([s.get("peak_rss_gb_proc", 0.0) for s in st]),
        "rel_counts_final": {r: int(sum(s["cand_info"].get("rel_counts_final", {}).get(r, 0) for s in st))
                             for r in ("PIVOT", "REVERSE", "SAME_ACTOR", "SAME_VICTIM", "SHARED_EXT")},
        "n_cont_total": int(sum(s["cand_info"].get("n_cont", 0) for s in st)),
        "n_pruned_generic_total": int(sum(s["cand_info"].get("n_pruned_generic", 0) for s in st)),
        "n_dropped_P_cap_total": int(sum(s["cand_info"].get("n_dropped_P_cap", 0) for s in st)),
    }
    out["code_fp"] = sorted({s.get("code_fp", "?") for s in st})
    reasons: dict = {}
    for t in ts:
        for k, v in (t.get("label_reasons") or {}).items():
            reasons[k] = reasons.get(k, 0) + v
    nn = max(sum(N), 1)
    out["label_reasons"] = reasons
    out["AMB_share"] = sum(v for k, v in reasons.items() if k.startswith("AMB")) / nn if reasons else None
    out["same_episode_share"] = reasons.get("AMB:same_episode", 0) / nn if reasons else None
    out["enab_rule"] = {"relation": int(sum(t.get("n_T_obs_enab_relation", 0) for t in ts)),
                        "fallback": int(sum(t.get("n_T_obs_enab_fallback", 0) for t in ts))}
    dflt = [s["transition_stats_default_rule"] for s in st if s.get("transition_stats_default_rule")]
    if dflt:
        out["observability_default_rule"] = pooled(dflt)
    by = {}
    for s in st:
        parts = str(s.get("inc_id", "")).split("/")
        k = s.get("scenario") or s.get("session") or (parts[1] if len(parts) >= 3 else "all")
        by.setdefault(k, []).append(s)
    def _grp(v):
        g = {"n_incidents": len(v), "N": _q([s["N"] for s in v], (0, 50, 100)),
             "observability": pooled([s["transition_stats"] for s in v]),
             "runtime_total_s_mean": float(np.mean([s["times"]["total_s"] for s in v]))}
        rr: dict = {}
        for s in v:
            for k2, c in (s["transition_stats"].get("label_reasons") or {}).items():
                rr[k2] = rr.get(k2, 0) + c
        nv = max(sum(s["N"] for s in v), 1)
        g["AMB_share"] = sum(c for k2, c in rr.items() if k2.startswith("AMB")) / nv if rr else None
        g["same_episode_share"] = rr.get("AMB:same_episode", 0) / nv if rr else None
        rec = sum(s["transition_stats"].get("n_T_obs_in_C", 0) for s in v)
        g["N_over_r"] = sum(s["N"] for s in v) / rec if rec else None
        d = [s["transition_stats_default_rule"] for s in v if s.get("transition_stats_default_rule")]
        if d:
            g["observability_default_rule"] = pooled(d)
        return g
    out["per_group"] = {k: _grp(v) for k, v in sorted(by.items())}
    return out


_FP: dict = {}


def code_fingerprint(cfg: dict | None = None) -> str:
    """BLAKE3 over the Alg. 1/2 + GT-semantics sources and the units/candidates/internal-prefix config."""
    if "fp" not in _FP:
        import blake3
        from certichain.io.hashing import tree_blake3
        cfg = cfg if cfg is not None else load_cfg()
        src = P2 / "src" / "certichain"
        files = [src / "units" / f for f in ("sessionize.py", "unit_types.py", "scope.py")] + \
                [src / "candidates" / "generate.py"] + \
                [src / "gt" / f for f in ("semantics.py", "observability.py", "iot23_episodes.py")]
        sub = {k: cfg.get(k) for k in ("units", "candidates", "internal_prefixes")}
        _FP["fp"] = blake3.blake3((tree_blake3(files) + json.dumps(sub, sort_keys=True)).encode()).hexdigest()[:16]
    return _FP["fp"]


def write_gate(summary: dict, cfg: dict) -> Path:
    gate = {
        "gate": "G0", "dataset": "iot23", "sealed": True,
        "note": ("IoT-23 is a validity set: these numbers were computed once with the frozen Alg. 1/Alg. 2 "
                 "defaults and are NOT used to change W, P, the G1 levers or any design choice."),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "params": {"units": cfg.get("units"), "candidates": cfg.get("candidates")},
        "code_fp_of_incidents": summary.get("code_fp", []),
        "n_windows": summary["n_incidents"],
        "C_size_distribution": summary["N"], "N_total": summary["N_total"],
        "units_per_window": summary["n_units"], "records_per_window": summary["n_records"],
        "N_over_r": summary["N_over_r"], "N_over_pos": summary["N_over_pos"],
        "r_CT": {"pooled": summary["observability"]["r_CT_pooled"], "mean": summary["observability"]["r_CT_mean"]},
        "rho_T": {"pooled": summary["observability"]["rho_T_pooled"], "mean": summary["observability"]["rho_T_mean"]},
        "ceiling": {"pooled": summary["observability"]["ceiling_pooled"],
                    "mean": summary["observability"]["ceiling_mean"]},
        "transitions": {k: summary["observability"][k] for k in ("n_T", "n_T_obs", "n_T_hid", "n_T_obs_in_C")},
        "labels": summary["labels"],
        "N_max_binds": {"n_windows_hit": summary["N_max_hit"], "N_max": cfg["candidates"]["N_max"]},
        "U_max_binds": {"n_windows_hit": summary["U_max_hit"], "U_max": cfg["units"]["U_max"]},
        "runtime_s_per_incident": summary["runtime_s"], "peak_rss_gb_proc": summary["peak_rss_gb_proc"],
        "per_scenario": summary["per_group"],
    }
    p = GATES / "G0_sealed.json"
    if p.exists():                                   # keep the recorded provenance of the sealed run
        prev = json.loads(p.read_text())
        if "provenance" in prev:
            gate["provenance"] = prev["provenance"]
    atomic_write_json(p, gate)
    return p


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--big-records", type=int, default=6_000_000)
    ap.add_argument("--only", default=None)
    ap.add_argument("--exclude", default=None, help="comma-separated inc_ids to skip")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--out-root", default=None, help="write outputs under this root instead of results/interim")
    a = ap.parse_args()
    global OUT, GATES
    if a.out_root:
        OUT = Path(a.out_root)
        GATES = OUT / "gates"
    cfg = load_cfg()
    ds = ds_name(a.dataset)
    for sub in OUT_DIRS:
        (OUT / ds / sub).mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        incs = list_incidents(ds, cfg)
        if a.only:
            keep = set(a.only.split(","))
            incs = [i for i in incs if i["inc_id"] in keep]
        if a.exclude:
            drop = set(a.exclude.split(","))
            incs = [i for i in incs if i["inc_id"] not in drop]
        if not a.overwrite:
            incs = [i for i in incs if not out_path(ds, "stats", i["inc_id"], "json").exists()]
        incs = sorted(incs, key=lambda i: -(i.get("n_records") or 0))
        if a.limit:
            incs = incs[: a.limit]
        W = max(1, min(8, a.workers or mg_workers(default=8)))
        big = [i for i in incs if (i.get("n_records") or 0) > a.big_records]
        small = [i for i in incs if i not in big]
        log(f"{ds}: {len(incs)} pending incidents ({len(big)} big, sequential in driver; {len(small)} in a pool of {W})")
        failures = []
        try:
            for inc in big:
                check_memory()
                r = process_incident(inc, cfg, in_worker=False)
                log(f"big {r['inc_id']}: records={r['n_records']} units={r['n_units']} N={r['N']} "
                    f"t={r['total_s']:.1f}s mem={usage_gb():.1f}GB")
            if small:
                os.environ["POLARS_MAX_THREADS"] = str(cfg.get("runtime", {}).get("polars_threads_per_worker", 4))
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=W, maxtasksperchild=8) as pool:
                    done = 0
                    for r in pool.imap_unordered(_worker, [(i, cfg, str(OUT)) for i in small]):
                        done += 1
                        check_memory()
                        if not r.get("ok"):
                            if r.get("memory"):
                                raise MemoryBudgetExceeded(f"{r['inc_id']}: {r['error']}")
                            failures.append(r)
                            log(f"FAILED {r['inc_id']}: {r['error']}")
                            continue
                        if done % 10 == 0 or done == len(small):
                            log(f"{done}/{len(small)} {r['inc_id']}: units={r['n_units']} N={r['N']} "
                                f"t={r['total_s']:.1f}s mem={usage_gb():.1f}GB")
        except MemoryBudgetExceeded as e:
            log(f"[mem] soft limit reached: {e}; finished incidents are on disk; exiting {EXIT_MEMORY}")
            return EXIT_MEMORY
        if failures:
            atomic_write_json(OUT / ds / "stats" / "_failures.json", failures)
            log(f"{len(failures)} failures written to stats/_failures.json")
    summ = summarize(ds)
    atomic_write_json(OUT / ds / "summary_07.json", summ)
    log(f"summary: {summ.get('n_incidents')} incidents; N p50={summ.get('N', {}).get('p50')}")
    if a.gate:
        if ds != "iot23":
            raise SystemExit("--gate is defined for iot23 (G0) only")
        p = write_gate(summ, cfg)
        log(f"gate written: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
