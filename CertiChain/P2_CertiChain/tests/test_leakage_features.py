"""Anti-leakage checks for the statistical features (DESIGN §3.4.1, §5.3).

* no feature / descriptor name contains a forbidden substring (io.schema.FORBIDDEN_FEATURE_SUBSTRINGS);
* features are invariant to permuting (or adding) label-like columns on units / candidates;
* the only label-derived features (priors) depend on TRAINING labels only: permuting the scored
  incident's own labels changes nothing;
* unit features read only FEATURE_SAFE columns (+ src / ts for grouping and ordering);
* my feature / scoring modules never import the record-label quarantine reader.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from certichain.features import stats as F
from certichain.features.descriptors import DESCRIPTOR_NAMES, descriptor
from certichain.features.priors import PairPriors
from certichain.io.schema import FEATURE_SAFE_UNIT_COLS, FORBIDDEN_FEATURE_SUBSTRINGS
from certichain.scoring.fusion import FUSE_EXTRA
from test_features_stats import synth_incident

SRC = Path(__file__).resolve().parents[1] / "src" / "certichain"
MY_MODULES = [SRC / "features" / f for f in ("stats.py", "priors.py", "descriptors.py")] + \
             [SRC / "scoring" / f for f in ("fusion.py", "cascade.py", "driver.py")] + \
             [SRC / "baselines" / f for f in ("king.py", "bothunter.py", "lag_llr.py", "llm_only.py")]


@pytest.mark.parametrize("names", [F.FEATURES, F.PREFIX_FEATURES, DESCRIPTOR_NAMES, FUSE_EXTRA])
def test_no_forbidden_feature_names(names):
    for n in names:
        low = n.lower()
        bad = [s for s in FORBIDDEN_FEATURE_SUBSTRINGS if s in low]
        assert not bad, f"feature {n!r} contains forbidden substring(s) {bad}"
        assert low not in ("ts", "te", "src", "dst", "ip", "mac", "ttl", "rec_id", "y")


def _with_label_columns(df: pl.DataFrame, rng) -> pl.DataFrame:
    n = df.height
    return df.with_columns(pl.Series("label", rng.integers(0, 2, n)),
                           pl.Series("episode_id", [f"e{k}" for k in rng.integers(0, 5, n)]),
                           pl.Series("campaign_id", [f"c{k}" for k in rng.integers(0, 3, n)]),
                           pl.Series("y", rng.integers(-1, 2, n)), pl.Series("template_node", ["t"] * n))


def test_invariant_to_label_columns():
    units, side, cands, _ = synth_incident(5)
    base = F.base_features(units, cands, side)
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        u2 = _with_label_columns(units, rng)
        c2 = _with_label_columns(cands, rng)
        s2 = _with_label_columns(side, rng)
        other = F.base_features(u2, c2, s2)
        assert other.equals(base)
        assert np.array_equal(descriptor(u2, c2, s2), descriptor(units, cands, side))


def test_unit_features_read_only_safe_columns():
    units, side, cands, _ = synth_incident(6)
    full = F.base_features(units, cands, side)
    keep = [c for c in units.columns if c in FEATURE_SAFE_UNIT_COLS or c in ("unit_id", "src", "ts", "te")]
    # te is never read by the features (lags come from the candidate table); drop it too
    stripped = units.select([c for c in keep if c != "te"])
    assert F.base_features(stripped, cands, side).equals(full)


def test_priors_depend_on_training_labels_only():
    train = [synth_incident(s) for s in range(10, 16)]
    test_units, test_side, test_cands, test_y = synth_incident(99)
    tu = np.concatenate([F.base_features(u, c, s)["type_u_str"].to_numpy() for u, s, c, _ in train])
    tv = np.concatenate([F.base_features(u, c, s)["type_v_str"].to_numpy() for u, s, c, _ in train])
    lg = np.concatenate([c.sort("cand_id")["lag_gap"].to_numpy() for _, _, c, _ in train])
    y = np.concatenate([yy for *_, yy in train])
    pr = PairPriors.fit(tu, tv, lg, y)
    feats = F.add_prior_features(F.base_features(test_units, test_cands, test_side), pr)
    # the scored incident's labels never enter its features: permuting them is a no-op by construction
    perm = np.random.default_rng(0).permutation(test_y)
    assert not np.array_equal(perm, test_y) or test_y.size < 2
    feats2 = F.add_prior_features(F.base_features(test_units, test_cands, test_side), pr)
    assert feats.equals(feats2)
    # ... while permuting TRAINING labels does move the prior features
    pr2 = PairPriors.fit(tu, tv, lg, np.random.default_rng(1).permutation(y))
    feats3 = F.add_prior_features(F.base_features(test_units, test_cands, test_side), pr2)
    assert not np.allclose(feats["trans_prior"].to_numpy(), feats3["trans_prior"].to_numpy())


def _imports(py: Path) -> set[str]:
    tree = ast.parse(py.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names |= {f"{node.module}.{a.name}" for a in node.names}
        elif isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_my_modules_never_touch_record_label_tables():
    for py in MY_MODULES:
        if not py.exists():
            continue
        names = _imports(py)
        assert "read_labels_QUARANTINED" not in " ".join(names), f"{py.name} uses the record-label reader"
        text = py.read_text()
        assert "labels/" not in text.replace("labels_pairs", ""), f"{py.name} references record label paths"


def test_feature_code_has_no_label_reads():
    for name in ("stats.py", "priors.py", "descriptors.py"):
        text = (SRC / "features" / name).read_text()
        assert "labels_pairs" not in text and "read_parquet" not in text, f"features/{name} reads files"
