"""Label quarantine (DESIGN §3.1): label columns never travel with records.

At ingestion every source is split into
    records/...  RECORD_SCHEMA  (no label-like column; rec_id is the only key)
    labels/...   LABEL_SCHEMA   (rec_id, label, detailed_label)  -- QUARANTINED

Rules
-----
* ``write_records`` refuses frames that carry label-like columns; ``write_labels`` only
  writes LABEL_SCHEMA columns.
* ``read_labels_QUARANTINED`` is the ONLY reader of label tables. **Only ground-truth code
  (``certichain.gt.*``) may import it** (plus ingest/ChainForge generator code and
  scripts/tests that audit label distributions). Feature, scoring, LLM, unit, candidate,
  certification, baseline and claim code must never import it; tests/test_leakage*.py
  check this by import name, and the reader also refuses at runtime when called from one
  of those packages (``_FORBIDDEN_CALLERS``).
* Other label-derived tables written by ingest (TON ``episodes.parquet`` carries the
  attack ``type``; IoT-23 ``windows.json`` carries malicious-record counts) are
  generator/GT-side metadata with the same rule: never an input to features or scorers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from certichain.io.paths import atomic_write_parquet
from certichain.io.schema import LABEL_SCHEMA, RECORD_SCHEMA, conform

# substrings that make a column "label-like" (records must contain none of them)
LABEL_LIKE_SUBSTRINGS = ("label", "type", "attack", "malicious", "benign", "category", "class")
# RECORD_SCHEMA itself is label-free; these names are allowed despite matching nothing above
_RECORD_COLS = set(RECORD_SCHEMA)

_FORBIDDEN_CALLERS = ("certichain.features", "certichain.scoring", "certichain.llm", "certichain.units",
                      "certichain.candidates", "certichain.certify", "certichain.baselines",
                      "certichain.claims")


def conform_lazy(lf: pl.LazyFrame, schema: dict) -> pl.LazyFrame:
    """LazyFrame version of ``schema.conform``: add missing columns as nulls, cast, order."""
    have = set(lf.collect_schema().names())
    return lf.select([pl.col(n).cast(t) if n in have else pl.lit(None, dtype=t).alias(n) for n, t in schema.items()])


def label_like_columns(columns: Iterable[str]) -> list[str]:
    """Columns whose (lower-cased) name contains a label-like substring."""
    out = []
    for c in columns:
        lc = c.lower()
        if any(s in lc for s in LABEL_LIKE_SUBSTRINGS):
            out.append(c)
    return out


def assert_label_free(df_or_schema, where: str = "records") -> None:
    cols = list(df_or_schema.columns if hasattr(df_or_schema, "columns") else df_or_schema)
    bad = label_like_columns(cols)
    if bad:
        raise ValueError(f"label-like columns {bad} must not appear in {where} (label quarantine)")


def split_quarantine(df: pl.DataFrame, label_col: str = "label",
                     detailed_col: str | None = "detailed_label") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a frame carrying ``rec_id`` + record columns + label columns into
    (records conformed to RECORD_SCHEMA, labels conformed to LABEL_SCHEMA)."""
    if "rec_id" not in df.columns:
        raise ValueError("split_quarantine needs rec_id")
    lab = df.select(
        pl.col("rec_id"),
        pl.col(label_col).cast(pl.Utf8).alias("label"),
        (pl.col(detailed_col).cast(pl.Utf8) if detailed_col and detailed_col in df.columns
         else pl.lit(None, dtype=pl.Utf8)).alias("detailed_label"),
    )
    rec = conform(df, RECORD_SCHEMA)
    return rec, conform(lab, LABEL_SCHEMA)


def write_records(df: pl.DataFrame, path: str | os.PathLike) -> Path:
    """Conform to RECORD_SCHEMA (drops nothing silently: extra columns raise) and write."""
    extra = [c for c in df.columns if c not in _RECORD_COLS]
    if extra:
        raise ValueError(f"write_records: columns not in RECORD_SCHEMA: {extra}")
    assert_label_free(df.columns)
    return atomic_write_parquet(conform(df, RECORD_SCHEMA), path)


def write_labels(df: pl.DataFrame, path: str | os.PathLike) -> Path:
    extra = [c for c in df.columns if c not in LABEL_SCHEMA]
    if extra:
        raise ValueError(f"write_labels: columns not in LABEL_SCHEMA: {extra}")
    out = conform(df, LABEL_SCHEMA)
    if out["rec_id"].null_count() or out["rec_id"].n_unique() != out.height:
        raise ValueError("write_labels: rec_id must be non-null and unique")
    return atomic_write_parquet(out, path)


def _check_caller(depth: int = 2) -> None:
    try:
        mod = sys._getframe(depth).f_globals.get("__name__", "")
    except ValueError:  # pragma: no cover
        return
    if any(mod == p or mod.startswith(p + ".") for p in _FORBIDDEN_CALLERS):
        raise PermissionError(f"{mod} may not read quarantined labels (only certichain.gt.* may)")


def read_labels_QUARANTINED(paths: str | os.PathLike | Sequence[str | os.PathLike],
                            rec_ids: Iterable[int] | None = None) -> pl.DataFrame:
    """Read quarantined label table(s) -> LABEL_SCHEMA frame.

    ONLY ``certichain.gt.*`` (and ingest/generator/audit code) may import this function.
    """
    _check_caller()
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    paths = [str(p) for p in paths]
    lf = pl.scan_parquet(paths)
    if rec_ids is not None:
        lf = lf.filter(pl.col("rec_id").is_in(pl.Series(list(rec_ids), dtype=pl.Int64)))
    return conform(lf.collect(), LABEL_SCHEMA)


__all__ = ["LABEL_LIKE_SUBSTRINGS", "label_like_columns", "assert_label_free", "split_quarantine",
           "write_records", "write_labels", "read_labels_QUARANTINED"]
