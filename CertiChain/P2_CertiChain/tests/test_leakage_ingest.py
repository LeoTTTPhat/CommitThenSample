"""Leakage guards for ingest outputs (P2-A): records never carry labels; the quarantine
reader is import-restricted to gt/*."""
from __future__ import annotations

import ast
from pathlib import Path

import polars as pl
import pytest

from certichain.io import labels as L
from certichain.io.paths import P2_ROOT, interim, records_dir
from certichain.io.schema import LABEL_SCHEMA, RECORD_SCHEMA


def test_record_schema_has_no_label_columns():
    assert L.label_like_columns(RECORD_SCHEMA) == []
    # the only label-like names live in LABEL_SCHEMA
    assert set(L.label_like_columns(LABEL_SCHEMA)) == {"label", "detailed_label"}


def test_write_records_rejects_label_columns(tmp_path):
    df = pl.DataFrame({"rec_id": [1], "label": ["Malicious"]})
    with pytest.raises(ValueError):
        L.write_records(df, tmp_path / "r.parquet")


def test_write_labels_requires_unique_rec_id(tmp_path):
    good = pl.DataFrame({"rec_id": [1, 2], "label": ["a", "b"], "detailed_label": [None, None]})
    L.write_labels(good, tmp_path / "ok.parquet")
    dup = pl.DataFrame({"rec_id": [1, 1], "label": ["a", "b"], "detailed_label": [None, None]})
    with pytest.raises(ValueError):
        L.write_labels(dup, tmp_path / "dup.parquet")


def test_split_quarantine_separates_labels(tmp_path):
    df = pl.DataFrame({"rec_id": [1, 2], "ts": [1.0, 2.0], "src": ["a", "b"], "label": ["0", "1"],
                       "detailed_label": ["normal", "dos"]})
    rec, lab = L.split_quarantine(df)
    assert L.label_like_columns(rec.columns) == []
    assert list(rec.columns) == list(RECORD_SCHEMA)
    assert lab["label"].to_list() == ["0", "1"] and lab["detailed_label"].to_list() == ["normal", "dos"]


def test_quarantine_reader_blocks_forbidden_callers():
    # read_labels_QUARANTINED refuses when the calling module is feature/scoring/etc.
    import types

    ns = types.ModuleType("certichain.features.sneaky")
    code = "from certichain.io.labels import read_labels_QUARANTINED\n" \
           "def go(p):\n    return read_labels_QUARANTINED(p)\n"
    exec(compile(code, "<features>", "exec"), ns.__dict__)
    import sys
    sys.modules["certichain.features.sneaky"] = ns
    try:
        with pytest.raises(PermissionError):
            ns.go("/nonexistent.parquet")
    finally:
        del sys.modules["certichain.features.sneaky"]


SRC = P2_ROOT / "src" / "certichain"
# Packages that must NOT import the quarantine reader (feature/scoring path). ingest/gotham are
# allowed (they write the split); gt/* is the sanctioned reader.
FORBIDDEN_PKGS = ["features", "scoring", "llm", "units", "candidates", "certify", "baselines", "claims"]


def _imports_quarantine_reader(py: Path) -> bool:
    tree = ast.parse(py.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("io.labels"):
            if any(a.name == "read_labels_QUARANTINED" for a in node.names):
                return True
        if isinstance(node, ast.Attribute) and node.attr == "read_labels_QUARANTINED":
            return True
    return False


def test_no_forbidden_package_imports_quarantine_reader():
    for pkg in FORBIDDEN_PKGS:
        d = SRC / pkg
        for py in d.rglob("*.py"):
            assert not _imports_quarantine_reader(py), f"{py} imports read_labels_QUARANTINED (quarantine breach)"


def test_ingested_records_are_label_free_if_present():
    """If an ingest has run (records exist under results/interim), check a sample is label-free."""
    checked = 0
    for ds in ("iot23", "ton", "gotham"):
        d = records_dir(ds)
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.parquet"))[:3]:
            cols = pl.scan_parquet(p).collect_schema().names()
            assert L.label_like_columns(cols) == [], f"{p} has label-like columns {cols}"
            assert set(cols) == set(RECORD_SCHEMA), f"{p} columns != RECORD_SCHEMA"
            checked += 1
    if checked == 0:
        pytest.skip("no ingested records present yet")
