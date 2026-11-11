#!/usr/bin/env python
"""06 — Gotham-real case-study incidents (DESIGN §5.1.3). Owner: P2-E.

    .venv/bin/python -I -u P2_CertiChain/scripts/06_make_real_incidents.py --dataset gotham [--force]

Small job (< 1 min, < 2 GB). Reads P2-A's Gotham flows (results/interim/gotham/{records,labels}) and
configs/gotham_real.yaml. Per incident → results/interim/gotham_real/incidents/<id>/:
    records.parquet  RECORD_SCHEMA: every device capture's flows inside [t0, t1], flows seen by two captures
                     de-duplicated exactly as P2-B's Gotham loader (same 5-tuple, starts < 1 s apart → keep the
                     copy of the capture file sorting first); real IPs/times; P2-A's original rec_id (unique in
                     the dataset) so P2-B's gt/gotham_episodes.py can read labels by rec_id; rows sorted by
                     (ts, tiehash)
    labels.parquet   QUARANTINED, LABEL_SCHEMA (rec_id, label, detailed_label)
    meta.json        case metadata: inc_id, variant, pool, split_role, seed_device, t0, t1, scope_rule,
                     internal_prefixes, generator_seed (null), session
    gen.json         QUARANTINED: case kind (attack | benign_only), label counts, the benign-only check, and
                     the record overlap with the CF-GOTHAM-T bank (train-pool records reused as context)
gt.json is NOT written here (P2-B builds it from labels + reports.parquet).
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import polars as pl  # noqa: E402
import yaml  # noqa: E402

from certichain.io.hashing import tiehash_series, write_run_manifest  # noqa: E402
from certichain.io.labels import assert_label_free, read_labels_QUARANTINED  # noqa: E402
from certichain.io.paths import CONFIG_DIR, atomic_write_json, atomic_write_parquet, interim, log  # noqa: E402
from certichain.io.schema import LABEL_SCHEMA, RECORD_SCHEMA, conform  # noqa: E402

FILES = ("records.parquet", "labels.parquet", "meta.json", "gen.json")


def utc(s: str) -> float:
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()


def window_records(t0: float, t1: float) -> pl.DataFrame:
    files = sorted(interim("gotham", "records").glob("*.parquet"))
    parts = []
    for f in files:
        df = pl.scan_parquet(f).filter((pl.col("ts") >= t0) & (pl.col("ts") <= t1)).collect()
        if df.height:
            parts.append(df.with_columns(pl.lit(f.stem).alias("_cap")))
    if not parts:
        return pl.DataFrame(schema=RECORD_SCHEMA)
    df = pl.concat(parts, how="diagonal_relaxed")
    k = ["src", "sport", "dst", "dport", "proto"]
    df = df.sort(k + ["ts", "_cap"], nulls_last=True)
    same = pl.all_horizontal([(pl.col(c) == pl.col(c).shift(1)).fill_null(pl.col(c).is_null() & pl.col(c).shift(1).is_null())
                              for c in k])
    dup = same & ((pl.col("ts") - pl.col("ts").shift(1)) < 1.0) & (pl.col("_cap") != pl.col("_cap").shift(1))
    df = df.filter(~dup.fill_null(False)).drop("_cap")
    df = conform(df, RECORD_SCHEMA)
    return df.with_columns(tiehash_series(df).alias("_th")).sort(["ts", "_th", "rec_id"]).drop("_th")


def bank_record_ids() -> set[int]:
    """Original Gotham rec_ids spliced into CF-GOTHAM-T bank incidents (from their src_locator pointers)."""
    out: set[int] = set()
    for p in interim("cf_gotham_t", "incidents").glob("*/*/records.parquet"):
        s = pl.read_parquet(p, columns=["src_locator"])["src_locator"].str.extract(r"rec=(\d+)$").cast(pl.Int64)
        out.update(s.drop_nulls().to_list())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gotham", choices=["gotham"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    cfg = yaml.safe_load((CONFIG_DIR / "gotham_real.yaml").read_text())
    root = interim("gotham_real", "incidents")
    lab_paths = sorted(str(p) for p in interim("gotham", "labels").glob("*.parquet"))
    bank = bank_record_ids()
    log(f"CF-GOTHAM-T bank: {len(bank)} distinct Gotham source records")
    summary = []
    for inc in cfg["incidents"]:
        d = root / inc["id"]
        if all((d / f).exists() for f in FILES) and not a.force:
            summary.append(json.loads((d / "gen.json").read_text()))
            continue
        t0, t1 = utc(inc["window"][0]), utc(inc["window"][1])
        rec = window_records(t0, t1)
        assert_label_free(rec.columns)
        lab = read_labels_QUARANTINED(lab_paths, rec_ids=rec["rec_id"].to_list())
        if lab.height != rec.height:
            raise SystemExit(f"{inc['id']}: {rec.height - lab.height} records without labels")
        seed = inc["seed"]
        j = rec.select("rec_id", "src", "dst").join(lab, on="rec_id")
        touch = j.filter((pl.col("src") == seed) | (pl.col("dst") == seed))
        mal_seed = touch.filter(pl.col("label") != "Benign").height
        mal_all = j.filter(pl.col("label") != "Benign").height
        if not touch.height:
            raise SystemExit(f"{inc['id']}: the seed has no record in the window (degenerate incident)")
        if inc["kind"] == "benign_only" and mal_seed:
            raise SystemExit(f"{inc['id']}: {mal_seed} non-benign records touch the seed (not benign-only)")
        if inc["kind"] == "attack" and not mal_seed:
            raise SystemExit(f"{inc['id']}: no attack record touches the seed")
        ov = rec.filter(pl.col("rec_id").is_in(pl.Series(sorted(bank), dtype=pl.Int64).implode())).height if bank else 0
        meta = {"inc_id": f"gotham_real/{inc['id']}", "variant": "gotham_real", "pool": "caltest", "split_role": "test",
                "seed_device": seed, "t0": t0, "t1": t1, "scope_rule": cfg["scope_rule"], "internal_prefixes": [],
                "generator_seed": None, "session": inc["session"]}
        gen = {"inc_id": meta["inc_id"], "kind": inc["kind"], "n_records": rec.height,
               "n_seed_records": touch.height, "n_seed_nonbenign": mal_seed, "n_nonbenign_window": mal_all,
               "label_counts": j.group_by("label", "detailed_label").len().sort("len", descending=True).to_dicts(),
               "n_records_in_cf_gotham_t_bank": ov, "share_records_in_bank": (ov / rec.height) if rec.height else 0.0,
               "window_utc": inc["window"]}
        d.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(rec, d / "records.parquet")
        atomic_write_parquet(conform(lab, LABEL_SCHEMA).sort("rec_id"), d / "labels.parquet")
        atomic_write_json(d / "meta.json", meta)
        atomic_write_json(d / "gen.json", gen)
        summary.append(gen)
        log(f"{inc['id']}: {rec.height} records, seed non-benign {mal_seed}, window non-benign {mal_all}, "
            f"bank overlap {ov}")
    atomic_write_json(interim("gotham_real", "summary.json"), {"incidents": summary,
                                                               "runtime_s": round(time.time() - t_start, 1)})
    write_run_manifest("p2_gotham_real", {"config": cfg, "force": a.force}, extra={"n": len(summary)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
