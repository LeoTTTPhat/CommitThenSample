"""Hiding capacity (DESIGN Thm. 1, Cor. 1).

    H_beta(pi, b) = max { sum_{i in J} b_i :  J subset {b_i > 0},  sum_{i in J} -log(1 - pi_i) <= log(1/beta) }

Any valid UCB satisfies P_null(UCB >= H_{alpha+delta}) >= delta: exculpatory statements must pay the
hiding capacity of the eligible units. Computation (0/1 knapsack with values b_i and weights
w_i = -log(1 - pi_i), capacity C = log(1/beta)):

* indicator b in {0, 1}: EXACT greedy - take the units with the smallest w first (exchange argument).
* integer b: EXACT dynamic programme over values (min weight to reach value v), O(n * sum b).
* general b: DP on a discretised budget (G cells). Rounding weights UP gives a feasible set, i.e. a
  lower bound; rounding DOWN relaxes the constraint, i.e. an upper bound; the bracket closes as G grows.
* greedy bounds: ratio-greedy integral fill (lower) and the LP relaxation (upper; Dantzig).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["capacity_weights", "hiding_capacity_indicator", "hiding_capacity_int_dp", "hiding_capacity_grid_dp",
           "capacity_greedy_bounds", "hiding_capacity", "null_slack_ratio"]


def capacity_weights(pi):
    pi = np.asarray(pi, dtype=float)
    with np.errstate(divide="ignore"):
        return -np.log1p(-np.clip(pi, 0.0, 1.0))


def _eligible(pi, b):
    pi = np.asarray(pi, dtype=float).ravel()
    b = np.ones_like(pi) if b is None else np.asarray(b, dtype=float).ravel()
    m = b > 0
    return pi[m], b[m], np.nonzero(m)[0]


def hiding_capacity_indicator(pi, beta: float, b=None):
    """Exact capacity for b in {0,1}: (H, J) with J the hidden unit indices (into the input arrays)."""
    p, bb, idx = _eligible(pi, b)
    if np.any(np.abs(bb - 1.0) > 1e-12):
        raise ValueError("indicator capacity requires b in {0, 1}")
    w = capacity_weights(p)
    C = math.log(1.0 / beta)
    o = np.argsort(w, kind="stable")
    cw = np.cumsum(w[o])
    n = int(np.searchsorted(cw, C * (1 + 1e-15), side="right"))
    return n, idx[o[:n]]


def hiding_capacity_int_dp(pi, b, beta: float, max_value: int = 50_000_000):
    """Exact capacity for integer b by DP over values: dp[v] = min total weight with value >= v."""
    p, bb, _ = _eligible(pi, b)
    if np.any(np.abs(bb - np.round(bb)) > 1e-9):
        raise ValueError("integer DP requires integer b")
    bi = np.round(bb).astype(np.int64)
    w = capacity_weights(p)
    C = math.log(1.0 / beta)
    keep = w <= C * (1 + 1e-15)
    bi, w = bi[keep], w[keep]
    Vtot = int(bi.sum())
    if Vtot > max_value:
        raise ValueError(f"sum b = {Vtot} too large for the value DP")
    dp = np.full(Vtot + 1, np.inf)
    dp[0] = 0.0
    for v_i, w_i in zip(bi, w):
        # dp_new[v] = min(dp[v], dp[max(v - v_i, 0)] + w_i)  ("value >= v" semantics)
        shifted = np.empty_like(dp)
        shifted[v_i:] = dp[:-v_i] + w_i if v_i > 0 else dp + w_i
        shifted[:v_i] = dp[0] + w_i
        np.minimum(dp, shifted, out=dp)
    feas = np.nonzero(dp <= C * (1 + 1e-12))[0]
    return int(feas.max()) if len(feas) else 0


def hiding_capacity_grid_dp(pi, b, beta: float, G: int = 4096):
    """(lower, upper) from a DP over a discretised budget of G cells (round weights up / down)."""
    p, bb, _ = _eligible(pi, b)
    w = capacity_weights(p)
    C = math.log(1.0 / beta)
    keep = w <= C * (1 + 1e-15)
    w, bb = w[keep], bb[keep]
    x = w * G / C
    out = []
    for wi in (np.ceil(x - 1e-12).astype(np.int64), np.floor(x + 1e-12).astype(np.int64)):
        dp = np.zeros(G + 1)                    # dp[c] = max value with integer weight <= c
        free = 0.0
        for v_i, c_i in zip(bb, wi):
            if c_i <= 0:
                free += v_i
                continue
            if c_i > G:
                continue
            cand = dp[:-c_i] + v_i
            np.maximum(dp[c_i:], cand, out=dp[c_i:])
        out.append(float(dp[G] + free))
    return out[0], out[1]


def capacity_greedy_bounds(pi, b, beta: float):
    """(lower, upper): ratio-greedy integral fill and LP-relaxation bound."""
    p, bb, _ = _eligible(pi, b)
    w = capacity_weights(p)
    C = math.log(1.0 / beta)
    free = bb[w <= 0].sum()
    m = w > 0
    w, bb = w[m], bb[m]
    o = np.argsort(-(bb / w), kind="stable")
    w, bb = w[o], bb[o]
    cw = np.cumsum(w)
    k = int(np.searchsorted(cw, C * (1 + 1e-15), side="right"))
    lp = bb[:k].sum() + (((C - (cw[k - 1] if k else 0.0)) / w[k]) * bb[k] if k < len(w) else 0.0)
    # greedy integral: continue past the first item that does not fit
    rem, val = C, 0.0
    for wi, vi in zip(w, bb):
        if wi <= rem:
            rem -= wi
            val += vi
    return float(val + free), float(lp + free)


def hiding_capacity(pi, b=None, beta: float = 0.025, method: str = "auto", G: int = 4096) -> dict:
    """Hiding capacity with a method choice; returns {H, lower, upper, exact, method}."""
    p, bb, _ = _eligible(pi, b)
    if method == "auto":
        if np.all(np.abs(bb - 1.0) <= 1e-12):
            method = "indicator"
        elif np.all(np.abs(bb - np.round(bb)) <= 1e-9) and bb.sum() <= 5_000_000:
            method = "int_dp"
        else:
            method = "grid_dp"
    if method == "indicator":
        H, _ = hiding_capacity_indicator(p, beta)
        return {"H": float(H), "lower": float(H), "upper": float(H), "exact": True, "method": method}
    if method == "int_dp":
        H = hiding_capacity_int_dp(p, bb, beta)
        return {"H": float(H), "lower": float(H), "upper": float(H), "exact": True, "method": method}
    if method == "grid_dp":
        lo, hi = hiding_capacity_grid_dp(p, bb, beta, G)
        glo, ghi = capacity_greedy_bounds(p, bb, beta)
        lo, hi = max(lo, glo), min(hi, ghi)
        return {"H": lo, "lower": lo, "upper": hi, "exact": bool(hi - lo <= 1e-9), "method": method}
    if method == "greedy":
        lo, hi = capacity_greedy_bounds(p, bb, beta)
        return {"H": lo, "lower": lo, "upper": hi, "exact": bool(hi - lo <= 1e-9), "method": method}
    raise ValueError(method)


def null_slack_ratio(phi0: float, alpha_side: float = 0.025, w0: float = 0.5, eta0_frac: float = 0.999) -> dict:
    """Cor. 1 in the null regime with indicators at floor phi0: UCS radius vs the capacity bound.

    r+ <= cbar log(1/(alpha w0))/eta0_frac with cbar = 1/phi0; H_alpha ~ log(1/alpha)/phi0 (continuous) and
    floor(log(1/alpha)/(-log(1-phi0))) units (exact, m large). Ratio -> log(80)/log(40) = 1.19 at alpha/2 = 0.025."""
    from .cs import null_ucs_radius
    cbar = 1.0 / phi0
    r_bound = cbar * math.log(1.0 / (alpha_side * w0)) / eta0_frac
    r_exact = null_ucs_radius(cbar, alpha_side, w0=w0, eta0_frac=eta0_frac)
    H_cont = math.log(1.0 / alpha_side) / phi0
    H_units = math.floor(math.log(1.0 / alpha_side) / -math.log1p(-phi0) + 1e-12)
    return {"cbar": cbar, "r_plus_bound": r_bound, "r_plus": r_exact, "H_cont": H_cont, "H_units": H_units,
            "ratio_bound": r_bound / H_cont, "ratio_log": math.log(1 / (alpha_side * w0)) / math.log(1 / alpha_side)}
