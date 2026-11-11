"""Tests for the P2-X revision layer (eval/revision*.py) on synthetic banks / hand-made inputs only.

Run: .venv/bin/python -I -m pytest -q P2_CertiChain/tests/test_revision.py
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from certichain.certify.crc import crc_threshold_from_sums
from certichain.certify.losses import build_loss_tables
from certichain.eval import experiments as X
from certichain.eval import revision as RV
from certichain.eval import runner as RN
from certichain.eval import synth
from certichain.eval.splits import random_splits


@pytest.fixture(scope="module")
def chain_bank():
    cfg = synth.SynthConfig(chains=True, n_median=60, n_sigma=0.3, mu=3.0, tau=0.5, decoy_rate=1.0)
    s0 = synth.synth_null_pool(40, 31, cfg)
    pool = synth.synth_pool(80, 32, cfg)
    return build_loss_tables(pool, s0)


@pytest.fixture(scope="module")
def flat_bank():
    cfg = synth.SynthConfig(n_median=80, n_sigma=0.3, mu=3.0, tau=0.5)
    s0 = synth.synth_null_pool(40, 41, cfg)
    pool = synth.synth_pool(60, 42, cfg)
    return build_loss_tables(pool, s0)


def _counts(R, V, A=None, TP=None):
    R = np.asarray(R)[None, :]
    V = np.asarray(V)[None, :]
    A = np.zeros_like(R) if A is None else np.asarray(A)[None, :]
    TP = (R - V) if TP is None else np.asarray(TP)[None, :]
    z = np.zeros_like(R)
    return {"R": R, "V": V, "A": A, "TP": TP, "TR": TP, "BR": z, "DC": z, "FD": z}


def _meta(M):
    one = np.ones((1, M))
    return {"n_true": one * 2, "n_trans": one * 2, "n_trans_obs": one * 2, "n_trans_hid": one * 0, "n_elink": one * 2,
            "n_bridges": one * 0, "n_noncamp_dev": one * 1, "benign_only": np.zeros((1, M), bool)}


def test_conditional_metrics_hand():
    c = _counts([0, 2, 4], [0, 1, 4])
    out = RV.conditional_metrics(c, _meta(3), np.ones((1, 3), bool), 0.1)
    assert math.isclose(out["fdr"][0], 0.5)
    assert math.isclose(out["cond_fdp"][0], 0.75)
    assert math.isclose(out["pooled_fdr"][0], 5 / 6)
    assert math.isclose(out["frac_empty"][0], 1 / 3)
    assert math.isclose(out["n_false"][0], 5 / 3)
    assert math.isclose(out["n_false_cond"][0], 2.5)
    assert math.isclose(out["cond_fdx_020"][0], 1.0) and math.isclose(out["fdx_020"][0], 2 / 3)
    assert math.isclose(out["p_any_false"][0], 2 / 3) and math.isclose(out["cond_p_any_false"][0], 1.0)
    assert math.isclose(out["sum_fdp"][0], 1.5) and out["n_claimed"][0] == 2
    # no claims anywhere: conditional metrics undefined, guaranteed estimand 0
    out0 = RV.conditional_metrics(_counts([0, 0], [0, 0]), _meta(2), np.ones((1, 2), bool), 0.1)
    assert out0["fdr"][0] == 0 and np.isnan(out0["cond_fdp"][0]) and out0["frac_empty"][0] == 1


def test_chain_exceedance_hand():
    ch = {"chN": np.array([[0, 2, 4]]), "chBad": np.array([[0, 0, 1]])}
    out = RV.chain_exceedance(ch, np.ones((1, 3), bool), 0.2)
    assert math.isclose(out["cond_maxchain_fdp"][0], 0.125)
    assert math.isclose(out["chain_exceed_a"][0], 1 / 3)          # 1/4 > 0.2
    assert math.isclose(out["cond_chain_exceed_a"][0], 0.5)
    assert math.isclose(out["frac_no_chain"][0], 1 / 3)


@pytest.mark.parametrize("method", ["crc_bh", "crc_st", "crc_fdx020", "ltt", "bh_iid", "by_ref", "chain_composite"])
def test_x2_eval_matches_runner(chain_bank, method):
    sp = random_splits(chain_bank.n, 40, range(5), 30)
    r1 = RN.evaluate(chain_bank, method, sp, 0.2, cert_gap=False)
    r2 = RV.x2_eval(chain_bank, method, sp, 0.2)
    for k in ("fdr", "fdr_lenient", "pooled_fdr", "frac_empty", "n_claims", "str_mean", "cer_link_mean",
              "wrongful_benign", "fdx_020", "p_fdp_gt_2a"):
        assert np.allclose(r1[k], r2[k], equal_nan=True), (method, k)
    if method in ("crc_bh", "chain_composite"):
        assert np.allclose(r1["maxchain_fdr"], r2["maxchain_fdr"], equal_nan=True)
    # conditional FDP ratio identity (constant test size): mean FDR / (1 - mean abstention)
    ex = RV.ratio_extras(r2)["cond_fdp_ratio"]["mean"]
    if np.nanmean(r2["frac_empty"]) < 1:
        assert math.isclose(ex, np.mean(r2["fdr"]) / (1 - np.mean(r2["frac_empty"])), rel_tol=1e-9)


def test_x2_bootstrap_same_stream_as_p2g(flat_bank):
    a = X.bootstrap_reps(flat_bank, "crc_bh", 0.2, K=30, n_test=20, metrics=("fdr", "str_mean"), B=12, n_splits=8,
                         seed=3)["reps_a"]
    b = RV.x2_bootstrap(flat_bank, "crc_bh", 0.2, K=30, n_test=20, B=12, n_splits=8, seed=3,
                        metrics=("fdr", "str_mean"))
    assert np.allclose(a, b, equal_nan=True)


def test_fixed_baseline_eval(chain_bank):
    cnt = X.oracle_counts(chain_bank)
    sp = random_splits(chain_bank.n, 30, range(4), 30)
    r = RV.x2_eval(chain_bank, "oracle", sp, 0.1, fixed=cnt)
    assert np.allclose(r["fdr"], 0) and np.allclose(r["pooled_fdr"], 0)
    r2 = X.fixed_metrics(chain_bank, cnt, sp, 0.1)
    assert np.allclose(r["str_mean"], r2["str_mean"], equal_nan=True)


def test_episode_sampler_reduces_and_is_deterministic():
    # every incident in one common cluster: weights equal -> plain incident resampling of the right size
    cl = [[0]] * 7
    rows = list(RV.episode_pool_sampler(cl, 1, 5, seed=1))
    assert len(rows) == 5 and all(r.size == 7 and r.min() >= 0 and r.max() < 7 for r in rows)
    again = list(RV.episode_pool_sampler(cl, 1, 5, seed=1))
    assert all(np.array_equal(a, b) for a, b in zip(rows, again))
    # disjoint singleton clusters: an incident whose episode drew count 0 never appears in that replicate
    cl2 = [[i] for i in range(50)]
    rng_rows = next(iter(RV.episode_pool_sampler(cl2, 50, 1, seed=2)))
    assert rng_rows.size == 50
    # incidents without clusters keep weight 1
    cl3 = [[], [], [0]]
    r3 = list(RV.episode_pool_sampler(cl3, 1, 20, seed=0))
    assert all(r.size == 3 for r in r3)


def test_mondrian_single_stratum_equals_marginal(flat_bank):
    sp = random_splits(flat_bank.n, 30, range(6), 25)
    sel_m = RV.mondrian_select(flat_bank, "crc_bh", sp, 0.2, np.zeros(flat_bank.n, np.int64))
    sel_c = X.select(flat_bank, "crc_bh", sp, 0.2)
    assert np.array_equal(sel_m.R, sel_c.R)


def test_mondrian_per_stratum_threshold(flat_bank):
    strata = RV.label_free_strata(flat_bank)
    sp = random_splits(flat_bank.n, 45, range(4), 15)
    sel = RV.mondrian_select(flat_bank, "crc_bh", sp, 0.2, strata)
    L = flat_bank.t("Lmono", "bh")
    for s in range(sp.S):
        for k in np.unique(strata):
            cal = sp.cal[s][strata[sp.cal[s]] == k]
            col = int(crc_threshold_from_sums(L[cal].sum(axis=0)[None, :], np.array([cal.size]), 0.2)[0])
            assert sel.stratum_cols[s, k] == col
            te = sp.test[s]
            m = strata[te] == k
            assert np.array_equal(sel.R[s][m], flat_bank.t("R", "bh")[te[m], col])


def test_loso_matches_p2g(flat_bank):
    groups = np.array([f"g{i % 6}" for i in range(flat_bank.n)])
    for m in ("crc_bh", "crc_st", "crc_fdx020"):
        ref = X.loso(flat_bank, 0.2, method=m, groups=groups)
        res, cols = RV.loso_eval(flat_bank, m, 0.2, groups=groups)
        assert math.isclose(res["fdr_scen"], float(ref["fdp"].mean()), rel_tol=1e-12)
        assert [int(c) for c in cols] == [int(c) for c in ref["lam_idx"].to_list()]
        assert math.isclose(res["sum_R"], float(ref["sum_R"].sum()))


def test_loso_jackknife_matches_p2g(flat_bank, monkeypatch):
    groups = np.array([f"g{i % 6}" for i in range(flat_bank.n)])
    monkeypatch.setattr(X, "scenario_groups", lambda bank: groups)
    jk = X.jackknife_loso(flat_bank, 0.2, method="crc_bh")
    mine = RV.loso_jackknife(flat_bank, "crc_bh", 0.2, groups=groups, metrics=("fdr_scen",))["fdr_scen"]
    assert math.isclose(mine["estimate"], jk["estimate"], rel_tol=1e-12)
    assert math.isclose(mine["se_jack"], jk["se_jack"], rel_tol=1e-9)


def test_loso_bootstrap_excludes_own_copies(flat_bank):
    groups = np.array([f"g{i % 4}" for i in range(flat_bank.n)])
    copies = ["g0", "g0", "g1", "g2"]
    rows_l, R_l, cols = RV.loso_window_R(flat_bank, "crc_bh", 0.2, groups=groups, copies=copies)
    L = flat_bank.t("Lmono", "bh")
    Ls = {g: L[groups == g].mean(axis=0) for g in ("g0", "g1", "g2", "g3")}
    # copy 0 (g0) is calibrated on g1 + g2 only (not on the other g0 copy)
    col0 = int(crc_threshold_from_sums((Ls["g1"] + Ls["g2"])[None, :], np.array([2]), 0.2)[0])
    assert cols[0] == col0 == cols[1]
    col2 = int(crc_threshold_from_sums((2 * Ls["g0"] + Ls["g2"])[None, :], np.array([3]), 0.2)[0])
    assert cols[2] == col2
    reps = RV.loso_bootstrap(flat_bank, "crc_bh", 0.2, groups=groups, B=5, seed=0, metrics=("fdr_scen", "pooled_fdr"))
    assert reps.shape == (5, 2) and np.all((reps[np.isfinite(reps)] >= 0) & (reps[np.isfinite(reps)] <= 1))


def test_mde_and_power():
    assert math.isclose(RV.mde(1.0, 0.025, 0.8), 1.959963984540054 + 0.8416212335729143, rel_tol=1e-9)
    assert math.isclose(RV.power_at(RV.mde(0.02, 0.025), 0.02, 0.025), 0.8, rel_tol=1e-6)


def test_two_sample_auc_permutation():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((60, 4))
    same = rng.standard_normal((9, 4))
    shifted = rng.standard_normal((9, 4)) + 3.0
    p_same = RV.permutation_auc(A, same, P=30, seed=1)
    p_shift = RV.permutation_auc(A, shifted, P=30, seed=1)
    assert p_shift["auc_obs"] > 0.95 and p_shift["p_perm"] <= 2 / 31
    assert p_same["p_perm"] > p_shift["p_perm"]
    null = RV.exchangeable_null_auc(A, 40, 9, R=5, seed=0)
    assert null.shape == (5,) and np.all((null >= 0) & (null <= 1))
    b = RV.group_boot_auc(A, shifted, B=4, seed=0)
    assert np.all(b > 0.8)


def test_ci_dict():
    reps = np.linspace(0, 1, 101)
    d = RV.ci_dict(reps, point=0.5)
    assert math.isclose(d["lo"], 0.025) and math.isclose(d["hi"], 0.975) and math.isclose(d["ucb95_pct"], 0.95)
    assert math.isclose(d["ucb95_normal"], 0.5 + 1.6448536269514722 * reps.std(ddof=1))


def test_inject_above_same_nulls_as_e4():
    from certichain.eval import revision_inject as RI
    from certichain.llm import corrupt
    rng = np.random.default_rng(3)
    s = rng.standard_normal(40)
    y = np.where(rng.random(40) < 0.6, 0, np.where(rng.random(40) < 0.5, 1, -1))
    keys = [("inc", i) for i in range(40)]
    _, inj_weak = corrupt.inject_top(s, y, ["inc"] * 40, 3, seed=0, keys=keys)
    s2, inj = RI.inject_above(s, y, 3, 100.0, seed=0, keys=keys)
    assert np.array_equal(inj, inj_weak) and inj.sum() == 3
    assert np.all(s2[inj] >= 100.0) and np.all(y[inj] == 0) and np.array_equal(s2[~inj], s[~inj])


def test_edge_lags_bridge_path():
    from certichain.eval.revision_easiness import edge_lags
    lg = edge_lags({"campaigns": [{"edges": [{"a": "x.A", "b": "x.H", "lag_s": 10.0},
                                             {"a": "x.H", "b": "x.B", "lag_s": 5.0}]}], "decoys": []})
    assert lg.get(("x.A", "x.H")) == 10.0
    assert lg.get(("x.A", "x.B")) == 15.0          # bridge over the hidden x.H
    assert lg.get(("x.B", "x.A")) is None and ("x.A", "x.B") in lg


def test_design_index_uses_fallback_rule():
    from certichain.eval.revision_relabel import DesignIndex
    from certichain.gt import semantics as S
    units = pl.DataFrame({"unit_id": [0, 1, 2, 3], "ts": [0.0, 1.0, 2.0, 3.0]})
    gt = S.EpisodeGT(unit_episode={0: "A", 1: "A", 2: "B", 3: "B"}, unit_campaign={0: "c", 1: "c", 2: "c", 3: "c"},
                     transitions={("A", "B")}, episode_campaign={"A": "c", "B": "c"})
    ix = DesignIndex(units, gt, None)
    assert ix.enab_rule("A", "B") == (1, "fallback")        # last unit of A before first(B)
    cands = pl.DataFrame({"cand_id": [0, 1, 2], "u": [0, 1, 0], "v": [2, 2, 3]})
    from certichain.eval.revision_relabel import _labels
    lab, _ = _labels(cands, gt, ix)
    assert lab["y"].to_list() == [-1, 1, -1]
    # minimal unit frame (no entity columns): the relation-aware index also falls back -> identical labels
    lab2, _ = _labels(cands, gt, S._Index(units, gt, None))
    assert lab2["y"].to_list() == lab["y"].to_list()


def test_truth_class():
    from certichain.eval.revision_audit import truth_class
    b = {"episode": None, "mixed": False, "malicious": False}
    m1 = {"episode": "CC#0@seed", "mixed": False, "malicious": True}
    m2 = {"episode": "SCAN#1@seed", "mixed": False, "malicious": True}
    assert truth_class(1, m1, m2).startswith("TRUE")
    assert truth_class(0, b, b) == "FALSE: benign -> benign"
    assert truth_class(0, b, m1) == "FALSE: benign <-> malicious"
    assert truth_class(-1, m1, m1) == "AMB: same episode"


def test_crossfit_split_partitions_train_pool():
    from certichain.chainforge.episodes import library_dir
    from certichain.eval import revision_crossfit as XF
    if not (library_dir("gotham") / "episodes.parquet").exists():
        pytest.skip("Gotham library not built")
    table, members = XF.split_tables()
    assert not (members["trainA"] & members["trainB"])
    assert set(table["pool"].unique().to_list()) == {"trainA", "trainB", "caltest"}
    vict = dict(zip(table["lib_eid"].to_list(), table["victim"].to_list()))
    assert all(vict[e] in XF.HALF_A for e in members["trainA"])
    assert all(vict[e] in XF.HALF_B for e in members["trainB"])
    assert XF.CFG_A["pool_of_split"] == {"train": "trainA"} and XF.CFG_B["pool_of_split"] == {"caltest": "trainB"}
