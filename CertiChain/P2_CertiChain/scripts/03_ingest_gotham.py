#!/usr/bin/env python
"""03 -- Gotham 2025 per-packet CSVs -> bidirectional flows (records) + quarantined flow labels.

DESIGN §3.1.1 / §5.1.3. Run from the project root through memguard:

    .venv/bin/python memguard/memguard.py run --name p2_ingest_gotham --limit-gb 90 --workers 8 -- \
        .venv/bin/python -I -u P2_CertiChain/scripts/03_ingest_gotham.py [--only iotsim-air-quality-1] [--force]

Outputs (results/interim/gotham/):
    records/<device>.parquet   RECORD_SCHEMA flows of the device's own packets (bulk records for floods)
    labels/<device>.parquet    LABEL_SCHEMA: label Benign/Unknown/Malicious, detailed_label = majority Gotham label
    meta/<device>.json         per-device stats (superset filter, sessions, flows, bulk, label purity, runtime)
    sessions.json              capture sessions + attack windows (MANIFEST), device IPs, infrastructure roles
    meta.json                  summary over devices
A flow between two capture devices appears in BOTH devices' files (each file = one device's view).
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

from certichain.ingest import gotham_csv as G  # noqa: E402
from certichain.io.hashing import config_hash, manifest_sha256, sha256_file, write_run_manifest  # noqa: E402
from certichain.io.labels import write_labels, write_records  # noqa: E402
from certichain.io.paths import (CONFIG_DIR, GOTHAM_MANIFEST, GOTHAM_METADATA, GOTHAM_PROCESSED,  # noqa: E402
                                 atomic_write_json, clean_tmp, interim, log, peak_rss_gb, rel, run_pool)
from certichain.io.schema import RECORD_SCHEMA, conform  # noqa: E402

DS = "gotham"
CFG = {"idle_s": 60.0, "active_s": 600.0, "flood_threshold": 200_000, "bulk_bin_s": 10.0, "session_pad_s": 1,
       "own_ip_filter": "all files (caveat 3)", "label": "majority packet label, ties -> lexicographic",
       "code": "gotham_csv.v1"}


def process_device(task: dict) -> dict:
    from certichain.utils.mem import worker_check

    dev, path, ip, code, chash, force = task["dev"], task["path"], task["ip"], task["code"], task["chash"], task["force"]
    meta_p = interim(DS, "meta", f"{dev}.json")
    rec_p, lab_p = interim(DS, "records", f"{dev}.parquet"), interim(DS, "labels", f"{dev}.parquet")
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
    exp = manifest_sha256(GOTHAM_MANIFEST, os.path.basename(path))
    pk, st = G.parse_packets(path, ip)
    worker_check(every=1, default_workers=task["workers"])
    if st["n_bad_time"] or st["n_null_ts"]:
        raise RuntimeError(f"{dev}: {st['n_bad_time']} rows with unparsable frame.time")
    pk = G.assign_sessions(pk)
    sess_names = [s["name"] for s in G.SESSIONS]
    per_sess_pk = pk.group_by("session").agg(pl.len().alias("n")).sort("session").to_dicts()
    n_unassigned = int(pk.filter(pl.col("session") < 0).height)
    recs, labs, fstats = [], [], []
    for s in sorted(pk["session"].unique().to_list()):
        sub = pk.filter(pl.col("session") == s)
        rec, lab, fs = G.flows_from_packets(sub, CFG["idle_s"], CFG["active_s"], CFG["flood_threshold"], CFG["bulk_bin_s"])
        off = (int(s) + 1) * 10**12  # keep internal keys unique across sessions
        recs.append(rec.with_columns(pl.col("_key") + off))
        labs.append(lab.with_columns(pl.col("_key") + off))
        fs["session"] = sess_names[s] if s >= 0 else "unassigned"
        fstats.append(fs)
        del sub
        worker_check(every=1, default_workers=task["workers"])
    del pk
    rec = pl.concat(recs)
    lab = pl.concat(labs)
    out = G.to_record_schema(rec, code, rel(path), sha)
    maj = G.majority_labels(lab)
    out = out.join(maj, on="_key", how="left", maintain_order="left")
    if out["detailed_label"].null_count():
        raise RuntimeError(f"{dev}: {out['detailed_label'].null_count()} flows without a label")
    labels = out.select("rec_id", G.binary_label(pl.col("detailed_label")).alias("label"), "detailed_label")
    sess_col = out["_session"]
    lab_counts = (out.with_columns(pl.col("_session").replace_strict(dict(enumerate(sess_names)) | {-1: "unassigned"},
                                                                       return_dtype=pl.Utf8).alias("session"))
                  .group_by(["session", "detailed_label", "bulk"]).agg(pl.len().alias("n_records"),
                                                                       pl.col("n_merged").sum().alias("n_flows"),
                                                                       (pl.col("opkts") + pl.col("rpkts")).sum().alias("n_packets"))
                  .sort(["session", "n_packets"], descending=[False, True]).to_dicts())
    purity = {"n_mixed": int((out["purity"] < 1).sum()), "min_purity": float(out["purity"].min()),
              "n_mixed_non_bulk": int(out.filter(~pl.col("bulk") & (pl.col("purity") < 1)).height)}
    write_records(conform(out.select([c for c in RECORD_SCHEMA if c in out.columns]), RECORD_SCHEMA), rec_p)
    write_labels(labels, lab_p)
    del sess_col
    meta = {"device": dev, "ip": ip, "code": code, "src_file": rel(path), "sha256": sha, "manifest_sha256": exp,
            "sha256_matches_manifest": (exp == sha) if exp else None,
            "n_rows": st["n_rows"], "n_own": st["n_own"], "n_dropped_not_own": st["n_rows"] - st["n_own"],
            "packets_per_session": [{"session": sess_names[d["session"]] if d["session"] >= 0 else "unassigned", "n": d["n"]}
                                    for d in per_sess_pk],
            "n_unassigned_packets": n_unassigned, "flow_stats": fstats, "n_records": out.height,
            "n_bulk_records": int(out["bulk"].sum()), "label_counts": lab_counts, "label_purity": purity,
            "records": rel(rec_p), "labels": rel(lab_p), "config_hash": chash,
            "runtime_s": round(time.time() - t0, 1), "worker_peak_rss_gb": round(peak_rss_gb(), 2)}
    atomic_write_json(meta_p, meta)
    log(f"{dev}: {st['n_own']:,}/{st['n_rows']:,} own packets -> {out.height:,} records "
        f"({meta['n_bulk_records']} bulk), {meta['runtime_s']} s, peak RSS {meta['worker_peak_rss_gb']} GB")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    from certichain.utils.mem import workers as mg_workers

    base = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text())
    nw = min(a.workers or mg_workers(default=base["memory"]["default_workers"]), 8)
    chash = config_hash(CFG)
    for sub in ("records", "labels", "meta"):
        clean_tmp(interim(DS, sub))
    ipmap = G.device_ip_map(GOTHAM_METADATA / "metadata-benign.json")
    files = sorted(GOTHAM_PROCESSED.glob("iotsim-*.csv"))
    devs = [G.device_name(p) for p in files]
    if len(files) != 78:
        raise SystemExit(f"expected 78 processed CSVs, found {len(files)}")
    missing = [d for d in devs if d not in ipmap]
    if missing:
        raise SystemExit(f"no device IP for {missing}")
    for d, ip in G.ATTACK_DEVICES.items():
        if ipmap[d] != ip:
            raise SystemExit(f"{d}: metadata IP {ipmap[d]} != MANIFEST {ip}")
    codes = {d: i + 1 for i, d in enumerate(sorted(devs))}
    want = set(s.strip() for s in a.only.split(",") if s.strip())
    tasks = [{"dev": d, "path": str(p), "ip": ipmap[d], "code": codes[d], "chash": chash, "force": a.force,
              "workers": nw, "size": p.stat().st_size} for d, p in zip(devs, files) if not want or d in want]
    tasks.sort(key=lambda t: -t["size"])
    atomic_write_json(interim(DS, "sessions.json"), G.sessions_json() | {"device_ips": ipmap, "device_codes": codes})
    t0 = time.time()
    log(f"Gotham ingest: {len(tasks)} devices, {nw} workers, config {chash}")
    results, errors = run_pool(process_device, tasks, nw, label=lambda t: t["dev"])
    metas = []
    for d in devs:
        p = interim(DS, "meta", f"{d}.json")
        if p.exists():
            metas.append(json.loads(p.read_text()))
    summ = {"dataset": "gotham", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": CFG, "config_hash": chash,
            "n_devices": len(metas), "n_records": sum(m["n_records"] for m in metas),
            "n_bulk_records": sum(m["n_bulk_records"] for m in metas),
            "n_rows": sum(m["n_rows"] for m in metas), "n_own": sum(m["n_own"] for m in metas),
            "devices": {m["device"]: {k: m[k] for k in ("ip", "code", "n_rows", "n_own", "n_records", "n_bulk_records",
                                                        "sha256_matches_manifest", "runtime_s", "worker_peak_rss_gb",
                                                        "n_unassigned_packets", "label_purity")} for m in metas},
            "errors": errors, "wall_s": round(time.time() - t0, 1)}
    atomic_write_json(interim(DS, "meta.json"), summ)
    write_run_manifest("p2_ingest_gotham", CFG, {m["src_file"]: m["sha256"] for m in metas},
                       extra={"n_records": summ["n_records"], "errors": errors})
    log(f"Gotham done: {summ['n_devices']} devices, {summ['n_records']:,} records ({summ['n_bulk_records']:,} bulk), "
        f"errors={len(errors)}, wall {time.time() - t0:.0f} s")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
