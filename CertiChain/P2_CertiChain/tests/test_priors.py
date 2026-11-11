"""features/priors.py (DESIGN §3.4.1): transition prior + lag-LLR, AMB excluded, shrinkage, OOF."""
from __future__ import annotations

import numpy as np

from certichain.features.priors import N_LAG_BINS, PairPriors, lag_bins, oof_prior_features


def toy(seed=0, n=400):
    rng = np.random.default_rng(seed)
    tu = rng.choice(["A", "B", "C"], n)
    tv = rng.choice(["X", "Y"], n)
    lag = rng.uniform(-100, 1800, n)
    p = np.where((tu == "A") & (tv == "X"), 0.4, 0.02)
    y = (rng.random(n) < p).astype(np.int8)
    y[rng.random(n) < 0.2] = -1
    return tu, tv, lag, y


def test_lag_bins():
    b = lag_bins(np.array([-5.0, 0.0, 0.5, 10.0, 1800.0, 5000.0]))
    assert b[0] == 0 and b[1] == 0                      # overlap / zero gap
    assert b[-2] == N_LAG_BINS - 1 and b[-1] == N_LAG_BINS - 1
    assert np.all(np.diff(b[2:]) >= 0) and b.max() < N_LAG_BINS


def test_amb_rows_are_ignored():
    tu, tv, lag, y = toy()
    p1 = PairPriors.fit(tu, tv, lag, y)
    y2 = y.copy()
    amb = y2 == -1
    tu2 = tu.copy()
    tu2[amb] = "C"                                       # AMB rows changed arbitrarily -> same priors
    p2 = PairPriors.fit(tu2, tv, lag, y2)
    assert p1.to_json() == p2.to_json()
    assert p1.n_amb_excluded == int(amb.sum())


def test_prior_values_and_shrinkage():
    tu, tv, lag, y = toy()
    pr = PairPriors.fit(tu, tv, lag, y)
    tp = pr.transition_prior(np.array(["A", "B", "Z"]), np.array(["X", "X", "Q"]))
    assert tp[0] > tp[1]                                 # A->X is the enriched pair
    assert np.isclose(tp[2], pr.global_logodds)          # unseen pair -> global log-odds
    keep = y != -1
    n1 = int(((tu == "A") & (tv == "X") & (y == 1)).sum())
    n0 = int(((tu == "A") & (tv == "X") & (y == 0)).sum())
    exp = np.log((n1 + pr.prior_m * pr.pi) / (n0 + pr.prior_m * (1 - pr.pi)))
    assert np.isclose(tp[0], exp)
    assert np.isclose(pr.pi, ((y[keep] == 1).sum() + 1) / (keep.sum() + 2))
    ll = pr.lag_llr(np.array(["Z", "Z"]), np.array(["Q", "Q"]), np.array([-1.0, 100.0]))
    g = np.asarray(pr.lag_llr_global)
    assert np.allclose(ll, g[lag_bins(np.array([-1.0, 100.0]))])
    assert pr.seen_pair(np.array(["A", "Z"]), np.array(["X", "Q"])).tolist() == [True, False]


def test_lag_llr_prefers_the_positive_lag_range():
    rng = np.random.default_rng(3)
    n = 3000
    tu = np.array(["A"] * n)
    tv = np.array(["X"] * n)
    y = (rng.random(n) < 0.1).astype(np.int8)
    lag = np.where(y == 1, rng.uniform(1, 30, n), rng.uniform(300, 1800, n))
    pr = PairPriors.fit(tu, tv, lag, y)
    ll = pr.lag_llr(np.array(["A", "A"]), np.array(["X", "X"]), np.array([10.0, 1000.0]))
    assert ll[0] > 0 > ll[1]


def test_json_round_trip(tmp_path):
    tu, tv, lag, y = toy(1)
    pr = PairPriors.fit(tu, tv, lag, y)
    pr.save(tmp_path / "p.json")
    q = PairPriors.load(tmp_path / "p.json")
    x = (np.array(["A", "C"]), np.array(["X", "Y"]))
    assert np.allclose(pr.transition_prior(*x), q.transition_prior(*x))
    assert np.allclose(pr.lag_llr(*x, np.array([3.0, 400.0])), q.lag_llr(*x, np.array([3.0, 400.0])))


def test_oof_priors_never_see_own_fold_labels():
    tu, tv, lag, y = toy(2, n=600)
    groups = np.repeat([f"i{k}" for k in range(30)], 20)
    fold = {f"i{k}": k % 5 for k in range(30)}
    tp, ll = oof_prior_features(tu, tv, lag, y, groups, fold)
    f0 = np.array([fold[g] == 0 for g in groups])
    y2 = y.copy()
    y2[f0] = np.random.default_rng(9).permutation(y2[f0])            # scramble fold-0 labels only
    tp2, ll2 = oof_prior_features(tu, tv, lag, y2, groups, fold)
    assert np.allclose(tp[f0], tp2[f0]) and np.allclose(ll[f0], ll2[f0])
    assert not np.allclose(tp[~f0], tp2[~f0])                          # other folds do use fold-0 labels
