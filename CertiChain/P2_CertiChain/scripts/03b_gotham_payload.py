#!/usr/bin/env python
"""03b (SHOULD) -- decode the 48101/TCP report records in the Gotham mirai-infection pcaps.

DESIGN §3.1.2 / §5.2.3. Forensic ground truth ("which bot resolved victim y") for the Gotham
case study. Needs ``raw/malicious/mirai-infection/*.pcap`` (all 8 present on 18 Oct). tshark is
light here (only 48101/TCP, tcp.len>0), so it runs single process; still under memguard.

    .venv/bin/python memguard/memguard.py run --name p2_gotham_payload --limit-gb 90 --workers 2 -- \
        .venv/bin/python -I -u P2_CertiChain/scripts/03b_gotham_payload.py [--force]

Outputs (results/interim/gotham/):
    reports.parquet   t, reporter, target, port, user, pass, src_file, frame (+ printable flags)
    meta/reports.json per-pcap stats + the §3.1.2 layout verification (also summarised in notes/P2-A.md)
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import polars as pl  # noqa: E402

from certichain.gotham import report_payload as RP  # noqa: E402
from certichain.io.hashing import config_hash, sha256_file, write_run_manifest  # noqa: E402
from certichain.io.paths import (GOTHAM_MIRAI_PCAPS, atomic_write_json, atomic_write_parquet, clean_tmp,  # noqa: E402
                                 interim, log, peak_rss_gb, rel)

DS = "gotham"
CFG = {"report_port": RP.REPORT_PORT, "tshark": RP.TSHARK, "code": "report_payload.v1"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    from certichain.utils.mem import check_memory

    tshark = shutil.which("tshark") or RP.TSHARK
    if not os.path.exists(tshark):
        raise SystemExit(f"tshark not found at {tshark}")
    chash = config_hash(CFG | {"tshark": tshark})
    clean_tmp(interim(DS))
    pcaps = sorted(GOTHAM_MIRAI_PCAPS.glob("*.pcap"))
    if not pcaps:
        raise SystemExit(f"no mirai-infection pcaps under {GOTHAM_MIRAI_PCAPS}")
    out_p = interim(DS, "reports.parquet")
    meta_p = interim(DS, "meta", "reports.json")
    if out_p.exists() and meta_p.exists() and not a.force:
        m = json.loads(meta_p.read_text())
        if m.get("config_hash") == chash:
            log(f"reports.parquet up to date ({m['n_reports']} reports); use --force to redo")
            print(json.dumps(m["verification"], indent=1))
            return 0
    t0 = time.time()
    log(f"Gotham payload: {len(pcaps)} mirai-infection pcaps, config {chash}")
    frames, per = [], []
    data_sha = {}
    for p in pcaps:
        tp = time.time()
        sha = sha256_file(p)
        data_sha[rel(p)] = sha
        rec, st = RP.parse_pcap(p, rel(p), tshark=tshark)
        frames.append(rec)
        st.update({"pcap": rel(p), "sha256": sha, "runtime_s": round(time.time() - tp, 1)})
        per.append(st)
        log(f"{p.name}: {st['n_segments']} segments, {st['n_streams']} streams, {st['n_decoded']} decoded "
            f"({st['n_undecoded']} undecoded)")
        check_memory()
    rec = pl.concat(frames).sort(["t", "reporter", "frame"]) if frames else frames
    atomic_write_parquet(rec, out_p)
    ver = RP.verify_layout(rec, n=10)
    by_reporter = (rec.group_by("reporter").agg(pl.len().alias("n_reports"), pl.col("target").n_unique().alias("n_targets"),
                                                pl.col("t").min().alias("t0"), pl.col("t").max().alias("t1"))
                   .sort("reporter").to_dicts())
    meta = {"dataset": "gotham", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": CFG | {"tshark": tshark},
            "config_hash": chash, "n_pcaps": len(pcaps), "n_reports": rec.height, "per_pcap": per,
            "by_reporter": by_reporter, "verification": ver,
            "note": ("Forensic GT for §5.2.3 cracker resolution (REP(b) names target y). LABEL-DERIVED metadata: "
                     "GT side only, never a feature input."),
            "runtime_s": round(time.time() - t0, 1), "peak_rss_gb": round(peak_rss_gb(), 2)}
    atomic_write_json(meta_p, meta)
    write_run_manifest("p2_gotham_payload", CFG | {"tshark": tshark}, data_sha,
                       extra={"n_reports": rec.height, "layout_ok": ver["ok"]})
    log(f"reports.parquet: {rec.height} reports from {len(by_reporter)} reporters; layout_ok={ver['ok']}; "
        f"{time.time() - t0:.0f} s")
    print(json.dumps(ver, indent=1, default=str))
    return 0 if ver["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
