"""DESIGN §7.4 test_ht_unbiased: MC mean of X equals f within 4 SE; Vhat unbiased; Var formula."""
import numpy as np
import pytest

from ctsample.estimation.ht import expected_v, ht_increments, ht_total, ht_variance, v_stat, vhat


def _pop(seed=0, n=60):
    rng = np.random.default_rng(seed)
    f = np.where(rng.random(n) < 0.5, rng.integers(1, 6, n), 0).astype(float)
    pi = rng.uniform(0.01, 0.95, n)
    elig = rng.random(n) < 0.9
    return f, pi, elig


def test_unit_means_within_4se():
    f, pi, elig = _pop()
    R = 200_000
    rng = np.random.default_rng(1)
    Z = rng.random((R, len(f))) < pi
    X = ht_increments(Z, f, pi, eligible=elig)
    m = X.mean(axis=0)
    se = X.std(axis=0, ddof=1) / np.sqrt(R)
    target = np.where(elig, f, 0.0)
    ok = np.abs(m - target) <= 4 * np.maximum(se, 1e-12)
    assert ok.all(), np.nonzero(~ok)


def test_total_unbiased_and_variance_formula():
    f, pi, _ = _pop(2)
    R = 400_000
    Z = np.random.default_rng(3).random((R, len(f))) < pi
    Q = ht_total(Z, f, pi)
    var = ht_variance(f, pi)
    assert abs(Q.mean() - f.sum()) <= 4 * np.sqrt(var / R)
    # Var(Qhat) = sum f^2 (1-pi)/pi  (DESIGN prints /pi^2: wrong, see estimation/ht.py)
    assert Q.var() == pytest.approx(var, rel=0.02)
    assert not np.isclose(var, np.sum(f * f * (1 - pi) / pi ** 2), rtol=0.1)


def test_vhat_unbiased():
    f, pi, _ = _pop(4)
    R = 400_000
    Z = np.random.default_rng(5).random((R, len(f))) < pi
    V = vhat(Z, f, pi)
    var = ht_variance(f, pi)
    se = V.std(ddof=1) / np.sqrt(R)
    assert abs(V.mean() - var) <= 4 * se


def test_expected_v_and_withholding_zero():
    f, pi, _ = _pop(6)
    R = 200_000
    Z = np.random.default_rng(7).random((R, len(f))) < pi
    X = ht_increments(Z, f, pi)
    V = v_stat(X)
    assert V.mean() == pytest.approx(expected_v(f, pi), rel=0.02)
    W = np.zeros(len(f), bool)
    W[::3] = True
    Xw = ht_increments(Z, f, pi, W_mask=W)
    assert np.all(Xw[:, W] == 0)
    assert np.array_equal(Xw[:, ~W], X[:, ~W])


def test_pi_zero_with_hit_rejected():
    with pytest.raises(ValueError):
        ht_variance(np.array([1.0, 0.0]), np.array([0.0, 0.5]))
