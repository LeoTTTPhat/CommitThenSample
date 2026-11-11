#!/usr/bin/env python
"""01 -- IoT-23 conn.log.labeled -> selected 2-h windows (records + quarantined labels).

DESIGN §5.1.1. Run from the project root through memguard:

    .venv/bin/python memguard/memguard.py run --name p2_ingest_iot23 --limit-gb 90 --workers 8 -- \
        .venv/bin/python -I -u P2_CertiChain/scripts/01_ingest_iot23.py [--only 34-1,43-1] [--force]

Outputs (results/interim/iot23/):
    records/<scenario>/<window_id>.parquet   RECORD_SCHEMA (conn.log order; sort by ts, tiehash)
    labels/<scenario>/<window_id>.parquet    LABEL_SCHEMA (quarantined)
    meta/<scenario>.json                     per-scenario pass-1 summary (resume unit)
    windows.json                             all windows: bounds, seed, counts, label counts
Resumable: a scenario is skipped when its meta json exists with the same config hash and all
of its window files exist (--force redoes it). Exit 75 on memory stop (memguard restarts).
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402
import yaml  # noqa: E402

from certichain.ingest import zeek_iot23 as Z  # noqa: E402
from certichain.io.hashing import cached_sha256, config_hash, record_sha256, write_run_manifest  # noqa: E402
from certichain.io.paths import (CONFIG_DIR, IOT23_PCAP, IOT23_SMALL, atomic_write_json, clean_tmp,  # noqa: E402
                                 interim, log, peak_rss_gb, rel, run_pool)

DS = "iot23"


def load_config() -> dict:
    base = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text())
    cfg = yaml.safe_load((CONFIG_DIR / "iot23.yaml").read_text())
    seeds = yaml.safe_load((CONFIG_DIR / cfg["seeds_file"]).read_text())["scenarios"]
    return {"base": base, "iot23": cfg, "seeds": seeds}


def conn_log_path(ent: dict) -> Path:
    d = IOT23_SMALL / ent["dir"]
    if ent.get("conn_subdir"):
        d = d / ent["conn_subdir"]
    return d / "bro" / "conn.log.labeled"


_IP = r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.(?:\d{1,3}|xx))"


def check_seed_sources(sid: str, ent: dict) -> dict:
    """Re-read the local README / pcap file names (data, read-only) and compare with the YAML."""
    out = {"readme_found": None, "filename_found": None, "consistent_with_yaml": None}
    d = IOT23_PCAP / ent["dir"]
    if not d.exists():
        out["note"] = "no local pcap dir (remote README facts from notes/refs/iot23_remote_readmes.md)"
        return out
    rd = d / "README.md"
    if rd.exists():
        txt = rd.read_text(errors="replace")
        m = re.search(r"Infected (?:device|host):\s*" + _IP, txt) or re.search(r"IP address:\s*" + _IP, txt)
        out["readme_found"] = m.group(1) if m else None
    names = sorted(p.name for p in d.rglob("*.pcap"))
    ips = sorted({m.group(1) for n in names for m in [re.search(r"-" + _IP + r"(?:\.only\d+)?\.pcap$", n)] if m})
    out["filename_found"] = ips[0] if len(ips) == 1 else (ips or None)
    out["consistent_with_yaml"] = (out["readme_found"] in (ent.get("readme_ip"), None)
                                   and (out["filename_found"] in (ent.get("filename_ip"), None)
                                        or isinstance(out["filename_found"], list)))
    return out


def window_id(sid: str, b: int) -> str:
    return f"{sid}_b{b:03d}"


def outputs_for(sid: str, wids: list[str]) -> list[Path]:
    out = []
    for w in wids:
        out += [interim(DS, "records", sid, f"{w}.parquet"), interim(DS, "labels", sid, f"{w}.parquet")]
    return out


def process_scenario(task: dict) -> dict:
    """Worker: pass 1 + selection + pass 2 for one scenario (resumable unit)."""
    from certichain.utils.mem import worker_check

    sid, ent, cfg, force, workers = task["sid"], task["ent"], task["cfg"], task["force"], task["workers"]
    chash = task["chash"]
    meta_path = interim(DS, "meta", f"{sid}.json")
    if meta_path.exists() and not force:
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("config_hash") == chash and all(p.exists() for p in outputs_for(sid, [w["window_id"] for w in meta["windows"]])):
                meta["skipped"] = True
                return meta
        except (json.JSONDecodeError, KeyError):
            pass
    t_start = time.time()
    path = conn_log_path(ent)
    log(f"{sid}: start ({path.stat().st_size / 1e9:.2f} GB)")
    st = Z.scan_structure(path)
    record_sha256(path, st.sha256)
    if not st.ok:
        raise RuntimeError(f"{sid}: unexpected log structure: {st.reason}")
    n_rows = Z.n_data_rows(st)
    t_struct = time.time() - t_start
    worker_check(every=1, default_workers=workers)

    candidates = ent.get("seed_candidates") or [ent["seed"]]
    hist = Z.pass1_histogram(path, candidates)
    t_p1 = time.time() - t_start - t_struct
    worker_check(every=1, default_workers=workers)
    v4 = hist.filter(~pl.col("ipv6"))
    presence = {ip: int(v4.filter(pl.col(f"s{i}"))["n"].sum() or 0) for i, ip in enumerate(candidates)}
    seed_i = next((i for i, ip in enumerate(candidates) if presence[ip] > 0), None)
    if seed_i is None:
        raise RuntimeError(f"{sid}: no seed candidate {candidates} occurs as an endpoint")
    seed = candidates[seed_i]
    seed_note = "" if seed_i == 0 else f"first candidate {candidates[0]} absent; used {seed}"
    if int(hist["n"].sum()) != n_rows:
        raise RuntimeError(f"{sid}: histogram rows {int(hist['n'].sum())} != data rows {n_rows}")
    bins = Z.make_bins(hist, f"s{seed_i}", cfg["bin_s"])
    if bins.n_null_label:
        raise RuntimeError(f"{sid}: {bins.n_null_label} rows without a parsable label tail")
    sel_cfg = Z.SelectionConfig(cfg["max_mal_windows"], cfg["max_benign_windows"], cfg["honeypot_max_windows"],
                                cfg["benign_min_seed_records"], cfg["rng_seed"])
    chosen, sel_info = Z.select_windows(bins.table, ent["kind"], sel_cfg)

    # pass 2
    windows = []
    if chosen:
        tmp = interim(DS, "tmp", f"{sid}.pass2.parquet")
        all_lf = Z.extract_windows(path, bins, chosen, int(ent["code"]), st.sha256, tmp)
        worker_check(every=1, default_workers=workers)
        tab = {r["bin"]: r for r in bins.table.iter_rows(named=True)}
        for w in chosen:
            wid = window_id(sid, w.bin)
            rp, lp = interim(DS, "records", sid, f"{wid}.parquet"), interim(DS, "labels", sid, f"{wid}.parquet")
            Z.write_window(all_lf, w.bin, rp, lp)
            n_written = pl.scan_parquet(rp).select(pl.len()).collect().item()
            r = tab[w.bin]
            if n_written != r["n"]:
                raise RuntimeError(f"{sid}/{wid}: wrote {n_written} records, histogram says {r['n']}")
            lc = bins.labels.filter(pl.col("bin") == w.bin).drop("bin").to_dicts()
            slc = bins.seed_labels.filter(pl.col("bin") == w.bin).drop("bin").to_dicts()
            windows.append({
                "window_id": wid, "scenario": sid, "kind": w.kind, "bin": w.bin,
                "t_start": bins.t0 + w.bin * bins.bin_s, "t_end": bins.t0 + (w.bin + 1) * bins.bin_s,
                "seed": seed, "n_records": r["n"], "n_seed_records": r["n_seed"], "n_malicious": r["n_mal"],
                "n_seed_malicious": r["n_seed_mal"], "n_ipv6_dropped": r["n_ipv6"],
                "first_ts_sec": r["sec_min"], "last_ts_sec": r["sec_max"],
                "label_counts": lc, "seed_label_counts": slc,
                "records": rel(rp), "labels": rel(lp),
                "records_bytes": rp.stat().st_size, "labels_bytes": lp.stat().st_size,
            })
            worker_check(every=1, default_workers=workers)
        tmp.unlink(missing_ok=True)
    overall = (v4.with_columns(pl.col("label")).group_by(["label", "detailed_label"]).agg(pl.col("n").sum())
               .sort("n", descending=True).to_dicts())
    meta = {
        "scenario": sid, "dir": ent["dir"], "kind": ent["kind"], "code": ent["code"], "malware": ent.get("malware"),
        "device": ent.get("device"), "seed": seed, "seed_candidates": candidates, "seed_presence": presence,
        "seed_note": seed_note, "seed_source": ent.get("source"), "seed_check": task["seed_check"],
        "src_file": rel(path), "src_sha256": st.sha256, "src_bytes": path.stat().st_size,
        "n_data_rows": n_rows, "n_ipv6": int(hist.filter(pl.col("ipv6"))["n"].sum() or 0),
        "t0": bins.t0, "bin_s": bins.bin_s, "n_bins": int(bins.table.height),
        "bins": bins.table.to_dicts(), "selection": sel_info, "overall_label_counts": overall,
        "windows": windows, "config_hash": chash,
        "runtime_s": {"structure": round(t_struct, 1), "pass1": round(t_p1, 1), "total": round(time.time() - t_start, 1)},
        "worker_peak_rss_gb": round(peak_rss_gb(), 2),
    }
    atomic_write_json(meta_path, meta)
    log(f"{sid}: done {len(windows)} windows, {sum(w['n_records'] for w in windows):,} records, "
        f"{meta['runtime_s']['total']} s, peak RSS {meta['worker_peak_rss_gb']} GB")
    return meta


def assemble(cfg_all: dict, sids: list[str]) -> dict:
    metas = []
    for sid in sids:
        p = interim(DS, "meta", f"{sid}.json")
        if p.exists():
            metas.append(json.loads(p.read_text()))
    wins = [w for m in metas for w in m["windows"]]
    out = {
        "dataset": "iot23", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": cfg_all["iot23"],
        "note": ("Window-inclusion rule uses labels (defines the investigated population; identical for every "
                 "scenario). label_counts/n_malicious are LABEL-DERIVED metadata: GT/audit use only, never features."),
        "n_scenarios": len(metas), "n_windows": len(wins),
        "scenarios": {m["scenario"]: {k: m[k] for k in ("dir", "kind", "code", "malware", "device", "seed", "seed_source",
                                                       "seed_note", "seed_presence", "seed_check", "src_file", "src_sha256",
                                                       "n_data_rows", "n_ipv6", "t0", "bin_s", "n_bins", "selection",
                                                       "overall_label_counts", "runtime_s", "worker_peak_rss_gb")}
                      for m in metas},
        "windows": wins,
    }
    atomic_write_json(interim(DS, "windows.json"), out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated scenario ids (e.g. 34-1,HP-4-1)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    from certichain.utils.mem import workers as mg_workers

    cfg_all = load_config()
    cfg = cfg_all["iot23"]
    seeds = cfg_all["seeds"]
    sids = list(seeds)
    if a.only:
        want = [s.strip() for s in a.only.split(",") if s.strip()]
        bad = [s for s in want if s not in seeds]
        if bad:
            raise SystemExit(f"unknown scenario ids {bad}")
        sids = want
    nw = a.workers or mg_workers(default=cfg_all["base"]["memory"]["default_workers"])
    nw = min(nw, 8)
    chash = config_hash({"iot23": cfg, "seeds": seeds, "code": "zeek_iot23.v1"})
    for sub in ("tmp",):
        clean_tmp(interim(DS, sub))
    for sub in ("records", "labels", "meta"):
        clean_tmp(interim(DS, sub))
    tasks = []
    for sid in sids:
        ent = seeds[sid]
        p = conn_log_path(ent)
        if not p.exists():
            raise SystemExit(f"{sid}: missing {p}")
        tasks.append({"sid": sid, "ent": ent, "cfg": cfg, "force": a.force, "workers": nw, "chash": chash,
                      "seed_check": check_seed_sources(sid, ent), "size": p.stat().st_size})
    for t in tasks:
        sc = t["seed_check"]
        if sc.get("consistent_with_yaml") is False:
            log(f"WARNING {t['sid']}: local README/file name {sc} disagree with configs/iot23_seeds.yaml")
    tasks.sort(key=lambda t: -t["size"])  # biggest first
    log(f"IoT-23 ingest: {len(tasks)} scenarios, {nw} workers, config {chash}")
    t0 = time.time()
    results, errors = run_pool(process_scenario, tasks, nw, label=lambda t: t["sid"])
    out = assemble(cfg_all, list(seeds))
    data = {m["src_file"]: m["src_sha256"] for m in out["scenarios"].values()}
    write_run_manifest("p2_ingest_iot23", {"iot23": cfg, "seeds_file": cfg["seeds_file"]}, data,
                       extra={"n_windows": out["n_windows"], "errors": errors, "wall_s": round(time.time() - t0, 1)})
    log(f"windows.json: {out['n_scenarios']} scenarios, {out['n_windows']} windows; errors={len(errors)}; "
        f"wall {time.time() - t0:.0f} s; skipped={sum(1 for r in results if r.get('skipped'))}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
