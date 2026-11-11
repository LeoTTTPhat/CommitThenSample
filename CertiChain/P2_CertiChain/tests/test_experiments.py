"""Tests for the P2-G experiment layer (eval/experiments.py) on synthetic banks only (no real data).

Run: .venv/bin/python -I -m pytest -q P2_CertiChain/tests/test_experiments.py
"""
from __future__ import annotations

import json
import math

import numpy as np
import polars as pl
import pytest

from certichain.certify.losses import build_loss_tables
from certichain.eval import experiments as X
from certichain.eval import runner as RN
from certichain.eval import synth
from certichain.eval.bootstrap import cluster_bootstrap, method_stat_fn, paired_stat_fn
from certichain.eval.splits import random_splits


@pytest.fixture(scope="module")
def chain_bank():
    cfg = synth.SynthConfig(chains=True, n_median=60, n_sigma=0.3, mu=3.0, tau=0.5, decoy_rate=1.0)
    s0 = synth.synth_null_pool(40, 11, cfg)
    pool = synth.synth_pool(80, 12, cfg)
    return build_loss_tables(pool, s0)


@pytest.fixture(scope="module")
def flat_bank():
    cfg = synth.SynthConfig(n_median=80, n_sigma=0.3, mu=3.0, tau=0.5)
    s0 = synth.synth_null_pool(40, 21, cfg)
    pool = synth.synth_pool(60, 22, cfg)
    return build_loss_tables(pool, s0)


ALL_METHODS = ["crc_bh", "crc_bh_raw", "crc_fdx020", "crc_fdx030", "ltt", "crc_st", "crc_st_raw", "chain_composite",
               "chain_naive", "chain_rt", "chain_rt_bh", "bh_ref", "by_ref", "bh_iid", "bh_iid_pooled", "b8",
               "lr_a2", "lr_a2b", "lr_a1", "lr_a1b"]


def test_selection_metrics_equal_runner(chain_bank):
    sp = random_splits(chain_bank.n, 40, range(6), 30)
    for m in ALL_METHODS:
        cg = m == "crc_bh"
        r1 = RN.evaluate(chain_bank, m, sp, 0.2, cert_gap=cg)
        sel = X.select(chain_bank, m, sp, 0.2)
        r2 = X.selection_metrics(chain_bank, sel, 0.2, cert_gap=cg)
        for k, v in r1.items():
            assert k in r2, (m, k)
            assert np.allclose(np.asarray(v, float), np.asarray(r2[k], float), equal_nan=True), (m, k)
        r3 = X.add_sums(r1)
        assert np.allclose(r3["sum_R"], r2["sum_R"]) and np.allclose(r3["sum_TP"], r2["sum_TP"]), m


def test_b7_aux_equal(flat_bank):
    sc, y = synth.synth_training(30, 5, synth.SynthConfig(n_median=80))
    aux = {"b7": X.IsotonicB7().fit(sc, y)}
    sp = random_splits(flat_bank.n, 30, range(4), 20)
    r1 = RN.evaluate(flat_bank, "b7", sp, 0.1, aux=aux)
    r2 = X.selection_metrics(flat_bank, X.select(flat_bank, "b7", sp, 0.1, aux=aux), 0.1)
    assert all(np.allclose(np.asarray(r1[k], float), np.asarray(r2[k], float), equal_nan=True) for k in r1)


def test_bootstrap_matches_p2c(flat_bank):
    ci = cluster_bootstrap(np.arange(flat_bank.n), method_stat_fn(flat_bank, "crc_bh", 30, 0.2, "fdr", n_test=20),
                           B=15, splits=10, seed=3)
    out = X.bootstrap_reps(flat_bank, "crc_bh", 0.2, K=30, n_test=20, metrics=("fdr", "str_mean"), B=15,
                           n_splits=10, seed=3)
    assert np.allclose(ci.replicates, out["reps_a"][:, 0])
    # paired: P2-C's paired_stat_fn on the same bank twice is identically 0; ours too
    pf = cluster_bootstrap(np.arange(flat_bank.n), paired_stat_fn(flat_bank, flat_bank, "crc_bh", 30, 0.2, "fdr",
                                                                  n_test=20), B=5, splits=5, seed=1)
    po = X.bootstrap_reps(flat_bank, "crc_bh", 0.2, K=30, n_test=20, metrics=("fdr",), B=5, n_splits=5, seed=1,
                          bank_b=flat_bank)
    assert np.allclose(pf.replicates, po["reps_b"][:, 0] - po["reps_a"][:, 0])


def test_bootstrap_mask_all_true_equals_unmasked(flat_bank):
    a = X.bootstrap_reps(flat_bank, "crc_bh", 0.2, K=30, n_test=20, metrics=("fdr", "str_mean"), B=4, n_splits=5)
    b = X.bootstrap_reps(flat_bank, "crc_bh", 0.2, K=30, n_test=20, metrics=("fdr", "str_mean"), B=4, n_splits=5,
                         mask_col=np.ones(flat_bank.n, bool))
    assert np.allclose(a["reps_a"], b["reps_a"], equal_nan=True)


def test_cell_write_read_resume(tmp_path, monkeypatch, flat_bank):
    monkeypatch.setenv("CERTICHAIN_RESULTS", str(tmp_path))
    df, meta, path = X.run_cell("Etest", "syn", "S.stat", "crc_bh", 0.2, K=30, n_test=20, seeds=range(7),
                                bank=flat_bank)
    assert (path / "split_006.json").exists() and (path / "summary.parquet").exists()
    rec = json.loads((path / "split_003.json").read_text())
    assert rec["seed"] == 3 and abs(rec["fdr"] - df["fdr"][3]) < 1e-12
    df2, meta2, path2 = X.run_cell("Etest", "syn", "S.stat", "crc_bh", 0.2, K=30, n_test=20, seeds=range(7),
                                   bank=flat_bank)
    assert path2 == path and df2.equals(df)
    idx = X.collect("Etest")
    assert idx.height == 1 and idx["method"][0] == "crc_bh"
    s = X.summarize_frame(df)
    tot_r, tot_tp = float(df["sum_R"].sum()), float(df["sum_TP"].sum())
    assert s["load_ratio"] == pytest.approx(tot_r / tot_tp)


def test_oracle_and_fixed(chain_bank):
    cnt = X.oracle_counts(chain_bank)
    sp = random_splits(chain_bank.n, 40, range(5), 30)
    r = X.fixed_metrics(chain_bank, cnt, sp, 0.1)
    assert np.allclose(r["fdr"], 0) and np.allclose(r["pooled_fdr"], 0)
    # oracle STR pooled = transitions in C / |T*| over the test incidents
    tr, _ = X._rows(np.arange(chain_bank.n), sp.test)
    tc = X.transitions_in_C(chain_bank)[tr].sum(1)
    nt = chain_bank.col("n_trans")[tr].sum(1)
    assert np.allclose(r["str_pooled"], tc / nt)
    d = X.descriptors(chain_bank)
    assert d["ceiling"] == pytest.approx(d["n_T_obs_in_C"] / d["n_T"])


def test_loso_threshold_and_jackknife(flat_bank):
    groups = np.array([f"s{i % 6}" for i in range(flat_bank.n)])
    out = X.loso(flat_bank, 0.2, groups=groups)
    assert out.height == 6
    # manual: scenario loss paths and CRC on the other 5
    T = flat_bank.t("Lmono", "bh")
    scen = sorted(set(groups))
    Ls = np.stack([T[groups == s].mean(0) for s in scen])
    for j, s in enumerate(scen):
        others = [i for i in range(6) if i != j]
        risk = (Ls[others].sum(0) + 1) / 6
        ok = np.flatnonzero(risk <= 0.2 + 1e-12)
        col = int(ok.max()) if ok.size else 0
        assert out["lam_idx"][j] == col
        R = flat_bank.t("R", "bh")[groups == s, col]
        V = flat_bank.counts(np.flatnonzero(groups == s), R)["V"]
        assert out["fdp"][j] == pytest.approx(float(np.mean(V / np.maximum(R, 1))))
    # jackknife: with groups passed implicitly via scenario_of we need real ids -> test the formula directly
    vals = np.array([0.1, 0.3, 0.2, 0.25])
    n = vals.size
    th = np.array([(vals.sum() - v) / (n - 1) for v in vals])
    se = math.sqrt((n - 1) / n * np.sum((th - th.mean()) ** 2))
    assert se == pytest.approx(np.std(vals, ddof=1) / math.sqrt(n))


def test_shift_splits_disjoint():
    g = np.array(["a", "a", "b", "c", "c", "c", "d", "e"])
    sp = X.shift_splits(g, 10, salt=7)
    for _, c, t in sp:
        assert set(g[c]).isdisjoint(set(g[t]))
        assert sorted(np.concatenate([c, t]).tolist()) == list(range(g.size))
        assert len(set(g[c])) == 3


def test_holm_bootstrap_and_cp():
    rng = np.random.default_rng(0)
    out = X.holm_bootstrap({"x": rng.normal(0.05, 0.01, 2000), "y": rng.normal(0.0, 0.01, 2000)})
    assert out["x"]["reject"] and not out["y"]["reject"]
    assert out["x"]["holm_level"] == pytest.approx(0.025) and out["x"]["lcb_holm"] > 0
    lo, hi = X.clopper_pearson(0, 10)
    assert lo == 0 and hi == pytest.approx(1 - 0.025 ** (1 / 10))
    lo, hi = X.clopper_pearson(10, 10)
    assert hi == 1 and lo == pytest.approx(0.025 ** (1 / 10))
    assert all(math.isnan(v) for v in X.clopper_pearson(0, 0))


def test_longest_gt_path():
    assert X.longest_gt_path([]) == 0
    assert X.longest_gt_path([("a", "b"), ("b", "c"), ("a", "d")]) == 2
    assert X.longest_gt_path([("a", "b"), ("b", "c"), ("c", "d"), ("x", "d")]) == 3


def test_r3_bound_terms(flat_bank):
    sp = random_splits(flat_bank.n, 30, range(3), 20)
    sel = X.select(flat_bank, "crc_st", sp, 0.2)
    b = X.r3_bound_terms(flat_bank, sel, 3, 0.2)
    n0 = flat_bank.col("n_null0")[sel.test_rows]
    exp = 0.2 + np.array([np.mean(np.minimum(1, np.minimum(3, n0[s]) / np.maximum(sel.R[s], 1)))
                          for s in range(sp.S)])
    assert np.allclose(b, exp)


def test_injection_selection_uses_clean_calibration(flat_bank):
    """test_bank = the same bank -> identical to the plain selection."""
    sp = random_splits(flat_bank.n, 30, range(3), 20)
    a = X.select(flat_bank, "crc_bh", sp, 0.2)
    b = X.select(flat_bank, "crc_bh", sp, 0.2, test_bank=flat_bank)
    assert np.array_equal(a.R, b.R) and np.array_equal(a.col, b.col)


def test_validity_family_holm():
    cells = {"a": {"alpha": 0.1, "mean": 0.09, "se": 0.004}, "b": {"alpha": 0.2, "mean": 0.23, "se": 0.005},
             "c": {"alpha": 0.1, "mean": None, "se": None}}
    fam = X.validity_family(cells)
    d = {c["cell"]: c for c in fam}
    assert d["c"].get("missing") and not d["a"]["violation"] and d["b"]["violation"]
    assert d["b"]["z"] == pytest.approx(6.0)


def test_numbers_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("CERTICHAIN_RESULTS", str(tmp_path))
    N = X.Numbers("Etest")
    N.add("k", float("inf"), definition="d", ci=(0.1, np.float64(0.2)), status="primary")
    N.write()
    d = json.loads((tmp_path / "summary" / "Etest_numbers.json").read_text())
    assert d["numbers"]["k"]["value"] == "inf" and d["numbers"]["k"]["ci"] == [0.1, 0.2]
    N2 = X.Numbers("Etest")
    assert "k" in N2.d
