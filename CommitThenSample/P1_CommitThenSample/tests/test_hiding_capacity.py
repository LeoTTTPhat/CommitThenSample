"""DESIGN Thm. 1 / Cor. 1: hiding capacity and the UCS null slack.

Cor. 1 numbers at phi0 = 0.015, alpha/2 = 0.025: ratio log(80)/log(40) ~ 1.19; null slack ~ 67 log 80 ~ 293.
Knapsack DP (integer b exact; discretised budget brackets) vs brute force on small instances.
"""
import itertools
import math

import numpy as np
import pytest

from ctsample.estimation.capacity import (capacity_greedy_bounds, capacity_weights, hiding_capacity,
                                          hiding_capacity_grid_dp, hiding_capacity_indicator, hiding_capacity_int_dp,
                                          null_slack_ratio)
from ctsample.estimation.cs import null_ucs_radius


def _brute(pi, b, beta):
    w = capacity_weights(pi)
    C = math.log(1 / beta)
    best = 0.0
    n = len(pi)
    for mask in range(1 << n):
        idx = [i for i in range(n) if mask >> i & 1]
        if w[idx].sum() <= C * (1 + 1e-15):
            best = max(best, float(np.sum(np.asarray(b)[idx])))
    return best


def test_cor1_numbers():
    r = null_slack_ratio(0.015, 0.025)
    assert r["ratio_log"] == pytest.approx(math.log(80) / math.log(40)) and r["ratio_log"] == pytest.approx(1.19, abs=0.005)
    assert r["cbar"] == pytest.approx(66.67, abs=0.01)
    assert r["r_plus_bound"] == pytest.approx(66.67 * math.log(80) / 0.999, rel=1e-3)
    assert r["r_plus_bound"] == pytest.approx(293, abs=1.0)
    assert 0.95 * r["r_plus_bound"] <= r["r_plus"] <= r["r_plus_bound"] + 1e-9
    # capacity of m >= log(1/alpha)/phi floor units: floor(log 40 / -log(1 - 0.015)) = 244
    m = 400
    H, J = hiding_capacity_indicator(np.full(m, 0.015), 0.025)
    assert H == r["H_units"] == 244 and len(J) == 244
    assert r["r_plus"] / H == pytest.approx(1.2, abs=0.03)


def test_indicator_exact_vs_brute():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = 12
        pi = rng.uniform(0.01, 0.6, n)
        b = (rng.random(n) < 0.8).astype(float)
        beta = rng.uniform(0.01, 0.3)
        H, J = hiding_capacity_indicator(pi, beta, b)
        assert H == _brute(pi, b, beta)
        assert np.all(b[J] == 1) and capacity_weights(pi[J]).sum() <= math.log(1 / beta) + 1e-12


def test_integer_dp_and_bounds_vs_brute():
    rng = np.random.default_rng(1)
    for _ in range(25):
        n = 11
        pi = rng.uniform(0.02, 0.5, n)
        b = rng.integers(0, 9, n).astype(float)
        beta = rng.uniform(0.01, 0.2)
        exact = _brute(pi, b, beta)
        assert hiding_capacity_int_dp(pi, b, beta) == exact
        lo, hi = hiding_capacity_grid_dp(pi, b, beta, G=2048)
        assert lo <= exact + 1e-9 <= hi + 2e-9
        glo, ghi = capacity_greedy_bounds(pi, b, beta)
        assert glo <= exact + 1e-9 <= ghi + 1e-9
        res = hiding_capacity(pi, b, beta)
        assert res["exact"] and res["H"] == exact
        bf = b + rng.uniform(0, 0.5, n) * (b > 0)          # non-integer b: bracket only
        r2 = hiding_capacity(pi, bf, beta)
        ex2 = _brute(pi, bf, beta)
        assert r2["lower"] <= ex2 * (1 + 1e-12) + 1e-9 and ex2 <= r2["upper"] * (1 + 1e-12) + 1e-9


def test_ucs_pays_capacity_and_boost_does_not_shrink_it():
    """Thm. 1: a valid UCB's null slack is >= H_{alpha+delta}; our grid uses the GLOBAL cbar, so a
    pre-registered UCS's null slack is unchanged when a subpopulation is boosted (Cor. 1, R19)."""
    phi0 = 0.015
    slack = null_ucs_radius(1 / phi0, 0.025)
    H = hiding_capacity_indicator(np.full(1000, phi0), 0.025 + 0.01)[0]
    assert slack >= H
    boosted = np.full(1000, phi0)
    boosted[:200] = 0.2
    # capacity shrinks when pi grows, but the pre-registered null radius depends only on cbar = 1/phi0
    assert hiding_capacity_indicator(boosted, 0.025)[0] <= hiding_capacity_indicator(np.full(1000, phi0), 0.025)[0]
    assert null_ucs_radius(1 / phi0, 0.025) == pytest.approx(slack)
