#!/usr/bin/env python
"""04 — ChainForge episode + background libraries (DESIGN §5.4.1). Owner: P2-E.

Run from the project root (small job: < 2 min, < 4 GB for TON; Gotham similar):

    .venv/bin/python -I -u P2_CertiChain/scripts/04_build_libraries.py --variant ton [--force]
    .venv/bin/python -I -u P2_CertiChain/scripts/04_build_libraries.py --variant gotham [--force]

Inputs (P2-A): results/interim/<ds>/{episodes.parquet | records,labels}, backgrounds/<pool>.parquet.
Outputs (GENERATOR-SIDE, label-derived; never feature inputs): results/interim/<ds>/library/
    episodes.parquet, records_<pool>.parquet, labels_<pool>.parquet, library.json (Table III)
    background.json (devices, port-profile roles, spans), lookalikes_<pool>.parquet (D3 catalogue)
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

from certichain.chainforge import background as BG  # noqa: E402
from certichain.chainforge import episodes as EP  # noqa: E402
from certichain.io.hashing import write_run_manifest  # noqa: E402
from certichain.io.paths import clean_tmp, interim, log, peak_rss_gb  # noqa: E402


def build_ton(force: bool) -> dict:
    info = EP.build_ton_library(force=force)
    meta = json.loads(interim("ton", "meta.json").read_text())
    pools = {p: {"source": meta["backgrounds"][p]["name"], "path": BG.background_path("ton", p)}
             for p in ("train", "caltest")}
    bg = BG.build_background_library("ton", pools, force=force)
    return {"episodes": info, "background": {p: {k: v for k, v in d.items() if k != "devices"}
                                             for p, d in bg["pools"].items()}}


def build_gotham(force: bool) -> dict:
    info = EP.build_gotham_library(force=force)
    pools = EP.gotham_background_pools()
    bg = BG.build_background_library("gotham", pools, force=force, tile_s=900.0)   # 63-min pools: 15-min tiles
    return {"episodes": info, "background": {p: {k: v for k, v in d.items() if k != "devices"}
                                             for p, d in bg["pools"].items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["ton", "gotham"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t = time.time()
    clean_tmp(interim(a.variant, "library"))
    out = build_ton(a.force) if a.variant == "ton" else build_gotham(a.force)
    write_run_manifest(f"p2_libraries_{a.variant}", {"variant": a.variant, "force": a.force},
                       extra={"summary": out, "runtime_s": round(time.time() - t, 1), "peak_rss_gb": peak_rss_gb()})
    log(f"done in {time.time() - t:.1f}s; peak RSS {peak_rss_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
