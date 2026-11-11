"""Model-assisted difference (PPI) estimator, baseline B12 [SHOULD] (DESIGN §3.9).

    X^PPI_i = g_i + Z_i (f_i - g_i) / pi_i,     g_i = p_q(m_i) from committed metadata (fixed before Z)
    Q^PPI   = sum_i X^PPI_i   (unbiased: E[X^PPI_i] = f_i)
    Var     = sum (f - g)^2 (1 - pi) / pi
    Vhat    = sum Z (f - g)^2 (1 - pi) / pi^2     (unbiased; uses f only on retained units)
Reported: MSE and Wald coverage. Sums run over every eligible unit (g_i > 0 units outside the hit set
matter: a retained non-hit unit contributes -g_i/pi_i). No anytime-valid version is implemented
(cited as future work, DESIGN §3.9).
"""
from __future__ import annotations

import numpy as np

from .wald import wald_interval

__all__ = ["ppi_increments", "ppi_total", "ppi_variance", "ppi_vhat", "ppi_wald", "ppi_mse_ratio"]


def _arr(x):
    return np.asarray(x, dtype=float)


def ppi_increments(Z, f, g, pi):
    Z = np.asarray(Z, dtype=bool)
    f, g, pi = _arr(f), _arr(g), _arr(pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(Z, (f - g) / np.where(Z, pi, 1.0), 0.0)
    return g + corr


def ppi_total(Z, f, g, pi):
    return ppi_increments(Z, f, g, pi).sum(axis=-1)


def ppi_variance(f, g, pi):
    """Design variance of the difference estimator: sum (f-g)^2 (1-pi)/pi."""
    f, g, pi = _arr(f), _arr(g), _arr(pi)
    d = f - g
    m = d != 0
    if np.any(m & (pi <= 0)):
        raise ValueError("pi = 0 on a unit with f != g")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(m, d * d * (1 - pi) / np.where(m, pi, 1.0), 0.0).sum(axis=-1)


def ppi_vhat(Z, f, g, pi):
    Z = np.asarray(Z, dtype=bool)
    f, g, pi = _arr(f), _arr(g), _arr(pi)
    d = f - g
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(Z, pi, 1.0)
        return np.where(Z, d * d * (1 - p) / (p * p), 0.0).sum(axis=-1)


def ppi_wald(Z, f, g, pi, alpha: float = 0.05):
    """(Qhat, lo, hi) of the PPI Wald interval."""
    Qh = ppi_total(Z, f, g, pi)
    lo, hi = wald_interval(Qh, ppi_vhat(Z, f, g, pi), alpha)
    return Qh, lo, hi


def ppi_mse_ratio(f, g, pi):
    """MSE(PPI)/MSE(HT) for the same design (both unbiased; analytic)."""
    from .ht import ht_variance
    v_ht = ht_variance(f, pi)
    return ppi_variance(f, g, pi) / np.where(v_ht > 0, v_ht, np.nan)
