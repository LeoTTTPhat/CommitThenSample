"""Horvitz-Thompson increments for single-tier Poisson sampling (DESIGN §3.9, Prop. 1(a)).

X_i = Z_i f_i / pi_i for eligible payload units, X_i := 0 for ineligible units (b_i = 0) and
for every unit in the missing set W (withheld / deleted / failed epoch; Prop. 2).

All functions broadcast over leading axes (e.g. a reps axis); unit sums are over the LAST axis.
"""
from __future__ import annotations

import numpy as np

__all__ = ["ht_increments", "ht_total", "v_stat", "vhat", "ht_variance", "expected_v"]


def _mask(Z, eligible=None, W_mask=None):
    m = np.asarray(Z, dtype=bool)
    if eligible is not None:
        m = m & np.asarray(eligible, dtype=bool)
    if W_mask is not None:
        m = m & ~np.asarray(W_mask, dtype=bool)
    return m


def ht_increments(Z, f, pi, *, eligible=None, W_mask=None) -> np.ndarray:
    """X = Z*f/pi, zero where not retained, ineligible, or in W.

    ``pi`` must be > 0 wherever ``Z`` is True and the unit is eligible (true by construction:
    Z=1 is impossible at pi=0).
    """
    f = np.asarray(f, dtype=float)
    pi = np.asarray(pi, dtype=float)
    m = _mask(Z, eligible, W_mask)
    with np.errstate(divide="ignore", invalid="ignore"):
        X = np.where(m, f / np.where(m, pi, 1.0), 0.0)
    return X


def ht_total(Z, f, pi, *, eligible=None, W_mask=None) -> np.ndarray:
    """Q-hat = sum of HT increments over the last axis."""
    return ht_increments(Z, f, pi, eligible=eligible, W_mask=W_mask).sum(axis=-1)


def v_stat(X) -> np.ndarray:
    """V = sum X^2 (the LCS variance process; not an unbiased variance estimate)."""
    X = np.asarray(X, dtype=float)
    return (X * X).sum(axis=-1)


def vhat(Z, f, pi, *, eligible=None, W_mask=None) -> np.ndarray:
    """Unbiased single-tier Poisson variance estimate  sum Z f^2 (1-pi)/pi^2  (Prop. 1(a));
    E[vhat] = sum f^2 (1-pi)/pi = Var(Q-hat)."""
    f = np.asarray(f, dtype=float)
    pi = np.asarray(pi, dtype=float)
    m = _mask(Z, eligible, W_mask)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(m, pi, 1.0)
        v = np.where(m, f * f * (1.0 - p) / (p * p), 0.0)
    return v.sum(axis=-1)


def ht_variance(f, pi, *, eligible=None) -> np.ndarray:
    """True design variance Var(Q-hat) = sum f^2 (1-pi)/pi  (Var(Z f/pi) = (f/pi)^2 pi(1-pi)).

    NOTE: DESIGN §4 Prop. 1(a) / §6 print "f^2(1-pi)/pi^2"; that is E[Vhat]'s integrand, not the
    variance. Vhat = sum Z f^2 (1-pi)/pi^2 is unbiased FOR sum f^2 (1-pi)/pi (checked in
    tests/test_ht_unbiased.py)."""
    f = np.asarray(f, dtype=float)
    pi = np.asarray(pi, dtype=float)
    m = f != 0
    if eligible is not None:
        m = m & np.asarray(eligible, dtype=bool)
    if np.any(m & (pi <= 0)):
        raise ValueError("unit with f != 0 has pi <= 0: HT is not unbiased (floor violated)")
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(m, pi, 1.0)
        v = np.where(m, f * f * (1.0 - p) / p, 0.0)
    return v.sum(axis=-1)


def expected_v(f, pi) -> np.ndarray:
    """E[V] = E[sum X^2] = sum f^2/pi (used to centre the LCS grid when no training V is given)."""
    f = np.asarray(f, dtype=float)
    pi = np.asarray(pi, dtype=float)
    m = f != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(m, f * f / np.where(m, pi, 1.0), 0.0)
    return v.sum(axis=-1)
