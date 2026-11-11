#!/usr/bin/env python
"""02 -- TON_IoT processed CSVs + normal Bro logs -> records, quarantined labels, episodes, backgrounds.

DESIGN §5.1.2. Run from the project root through memguard:

    .venv/bin/python memguard/memguard.py run --name p2_ingest_ton --limit-gb 90 --workers 8 -- \
        .venv/bin/python -I -u P2_CertiChain/scripts/02_ingest_ton.py [--force]

Outputs (results/interim/ton/):
    records/Network_dataset_<N>.parquet    RECORD_SCHEMA (ts rounded to 1 s, ts_res = 1.0; IPv6 rows dropped)
    labels/Network_dataset_<N>.parquet     LABEL_SCHEMA: label ("0"/"1" as shipped), detailed_label (= type)
    backgrounds/train.parquet              normal_Bro/normal_IoT_1 (train/dev pool), RECORD_SCHEMA
    backgrounds/caltest.parquet            normal_Bro/normal_IoT_2 (cal/test pool), RECORD_SCHEMA
    episodes.parquet                       ChainForge library episodes (LABEL-DERIVED: generator/GT side only)
    meta/<file>.json, meta.json            per-file stats, counts per type per pool, runtimes
Resumable per file (meta json + config hash); episodes/backgrounds are recomputed unless present.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import polars as pl  # noqa: E402
import yaml  # noqa: E402

from certichain.ingest import ton_iot as T  # noqa: E402
from certichain.io.hashing import config_hash, manifest_sha256, sha256_file, write_run_manifest  # noqa: E402
from certichain.io.labels import write_records  # noqa: E402
from certichain.io.paths import (CONFIG_DIR, TON_BRO, TON_MANIFEST, TON_PROCESSED, atomic_write_json,  # noqa: E402
                                 atomic_write_parquet, clean_tmp, interim, log, peak_rss_gb, rel, run_pool)

DS = "ton"
CFG = {"gap_s": 300.0, "max_dur_s": 600.0, "max_rows": 5000, "seed": 0, "train_frac": 0.5,
       "episode_types": T.EPISODE_TYPES, "attackers": "192.168.1.30-39", "background_parity": True,
       "backgrounds": {"train": "normal_IoT_1", "caltest": "normal_IoT_2"}, "drop_ipv6": True, "code": "ton_iot.v2"}


def process_file(task: dict) -> dict:
    from certichain.utils.mem import worker_check

    path, force, chash = task["path"], task["force"], task["chash"]
    stem = os.path.splitext(os.path.basename(path))[0]
    meta_p = interim(DS, "meta", f"{stem}.json")
    rec_p, lab_p = interim(DS, "records", f"{stem}.parquet"), interim(DS, "labels", f"{stem}.parquet")
    if meta_p.exists() and rec_p.exists() and lab_p.exists() and not force:
        try:
            m = json.loads(meta_p.read_text())
            if m.get("config_hash") == chash:
                m["skipped"] = True
                return m
        except json.JSONDecodeError:
            pass
    t0 = time.time()
    sha = sha256_file(path)
    exp = manifest_sha256(TON_MANIFEST, f"Processed_datasets/Processed_Network_dataset/{os.path.basename(path)}")
    worker_check(every=1, default_workers=task["workers"])
    # newline count = rows + 1 (header) <=> no multi-line quoted fields, so line = row + 2 holds
    with open(path, "rb") as f:
        n_nl = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(64 << 20), b""))
    st = T.ingest_processed_file(path, sha, rec_p, lab_p)
    worker_check(every=1, default_workers=task["workers"])
    if n_nl != st["n_rows"] + 1:
        raise RuntimeError(f"{stem}: {n_nl} newlines vs {st['n_rows']} rows (+1 header): multi-line fields?")
    if st["n_null_ts"] or st["n_null_label"]:
        raise RuntimeError(f"{stem}: null ts/label rows {st['n_null_ts']}/{st['n_null_label']}")
    st.update({"manifest_sha256": exp, "sha256_matches_manifest": (exp == sha) if exp else None,
               "config_hash": chash, "runtime_s": round(time.time() - t0, 1), "worker_peak_rss_gb": round(peak_rss_gb(), 2),
               "records": rel(rec_p), "labels": rel(lab_p)})
    atomic_write_json(meta_p, st)
    log(f"{stem}: {st['n_written']:,} records ({st['n_ipv6']} IPv6 dropped), {st['runtime_s']} s, sha_ok={st['sha256_matches_manifest']}")
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    from certichain.utils.mem import check_memory, workers as mg_workers

    base = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text())
    nw = min(a.workers or mg_workers(default=base["memory"]["default_workers"]), 8)
    chash = config_hash(CFG)
    for sub in ("records", "labels", "meta", "backgrounds"):
        clean_tmp(interim(DS, sub))
    files = T.processed_files(TON_PROCESSED)
    if len(files) != 23:
        raise SystemExit(f"expected 23 processed CSVs, found {len(files)}")
    t0 = time.time()
    log(f"TON ingest: {len(files)} processed CSVs, {nw} workers, config {chash}")
    tasks = [{"path": str(p), "force": a.force, "chash": chash, "workers": nw} for p in files]
    results, errors = run_pool(process_file, tasks, nw, label=lambda t: os.path.basename(t["path"]))
    if errors:
        log(f"{len(errors)} file(s) failed; not building episodes")
        return 1
    t_files = time.time() - t0
    check_memory()

    # ---------------------------------------------------------------- backgrounds
    bg_stats = {}
    for pool, name in CFG["backgrounds"].items():
        out = interim(DS, "backgrounds", f"{pool}.parquet")
        bmeta = interim(DS, "meta", f"background_{pool}.json")
        if out.exists() and bmeta.exists() and not a.force:
            m = json.loads(bmeta.read_text())
            if m.get("config_hash") == chash:
                bg_stats[pool] = m
                continue
        shas = {}
        for lg in ("conn", "dns", "http", "ssl"):
            p = TON_BRO / "normal_Bro" / name / f"{lg}.log"
            if p.exists():
                shas[rel(p)] = sha256_file(p)
        df, st = T.bro_background(TON_BRO / "normal_Bro", name, shas, parity=CFG["background_parity"])
        write_records(df, out)
        st.update({"config_hash": chash, "sha256": shas, "records": rel(out),
                   "manifest_sha256_ok": {k: (manifest_sha256(TON_MANIFEST, k.split("Network_dataset_Bro/")[-1]) == v)
                                          for k, v in shas.items()}})
        atomic_write_json(bmeta, st)
        bg_stats[pool] = st
        log(f"background {pool} <- {name}: {st['n_records']:,} records (dns {st['n_with_dns']:,}, http {st['n_with_http']})")
        check_memory()

    # ---------------------------------------------------------------- episodes
    ep_path = interim(DS, "episodes.parquet")
    t_ep = time.time()
    rec_files = sorted(str(p) for p in interim(DS, "records").glob("Network_dataset_*.parquet"))
    lab_files = sorted(str(p) for p in interim(DS, "labels").glob("Network_dataset_*.parquet"))
    att = T.attack_rows(rec_files, lab_files)
    check_memory()
    ep, ep_stats = T.extract_episodes(att, CFG["gap_s"], CFG["max_dur_s"], CFG["max_rows"], CFG["seed"], CFG["train_frac"])
    del att
    atomic_write_parquet(ep, ep_path)
    check_memory()
    per = (ep.group_by(["type", "pool"]).agg(pl.len().alias("n_episodes"), pl.col("n_rows").sum().alias("n_rows"),
                                              pl.col("n_rows_full").sum().alias("n_rows_full"),
                                              pl.col("burst_id").n_unique().alias("n_bursts"),
                                              pl.col("attacker").n_unique().alias("n_attackers"),
                                              pl.col("t0").min().alias("t0_min"), pl.col("t1").max().alias("t1_max"))
           .sort(["type", "pool"]))
    lab_tot = (pl.scan_parquet(lab_files).group_by(["label", "detailed_label"]).agg(pl.len().alias("n"))
               .sort("n", descending=True).collect().to_dicts())
    meta = {
        "dataset": "ton", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": CFG, "config_hash": chash,
        "files": sorted([{k: r[k] for k in ("file", "code", "sha256", "sha256_matches_manifest", "n_rows", "n_ipv6",
                                             "n_written", "label_counts", "runtime_s", "worker_peak_rss_gb")} for r in results],
                        key=lambda r: r["code"]),
        "n_records": int(sum(r["n_written"] for r in results)), "label_counts": lab_tot,
        "backgrounds": bg_stats, "episodes": {"path": rel(ep_path), **ep_stats, "n_episodes": ep.height,
                                              "per_type_pool": per.to_dicts()},
        "note": "episodes.parquet and labels/ are LABEL-DERIVED: generator/GT side only, never feature inputs.",
        "runtime_s": {"files": round(t_files, 1), "episodes": round(time.time() - t_ep, 1), "total": round(time.time() - t0, 1)},
    }
    atomic_write_json(interim(DS, "meta.json"), meta)
    write_run_manifest("p2_ingest_ton", CFG, {r["file"]: r["sha256"] for r in results} |
                       {k: v for st in bg_stats.values() for k, v in st.get("sha256", {}).items()},
                       extra={"n_records": meta["n_records"], "n_episodes": ep.height})
    pl.Config.set_tbl_rows(40)
    print(per)
    log(f"TON done: {meta['n_records']:,} records, {ep.height} episodes, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
