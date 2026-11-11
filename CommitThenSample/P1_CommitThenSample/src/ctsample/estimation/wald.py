"""Reference method (DESIGN §3.9, E2): fixed-time Wald intervals and naive time-uniform monitoring.

Wald:  Qhat +- z_{1-alpha/2} sqrt(Vhat),  Vhat = sum Z f^2 (1-pi)/pi^2 (unbiased, single-tier Poisson).
It is valid only asymptotically at a FIXED time; "naive monitoring" = checking it at every epoch and
calling a miss if any epoch misses (shown only as a secondary, time-uniform comparison).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .ht import ht_total, vhat

__all__ = ["wald_interval", "wald_from_sample", "wald_miss", "NaiveMonitor"]


def wald_interval(Qhat, Vhat, alpha: float = 0.05):
    """(lo, hi) = Qhat -+ z_{1-alpha/2} sqrt(Vhat) (two-sided level alpha)."""
    z = norm.ppf(1.0 - alpha / 2.0)
    sd = np.sqrt(np.maximum(np.asarray(Vhat, dtype=float), 0.0))
    Qhat = np.asarray(Qhat, dtype=float)
    return Qhat - z * sd, Qhat + z * sd


def wald_from_sample(Z, f, pi, alpha: float = 0.05, W_mask=None):
    Qh = ht_total(Z, f, pi, W_mask=W_mask)
    return wald_interval(Qh, vhat(Z, f, pi, W_mask=W_mask), alpha)


def wald_miss(Q, Qhat, Vhat, alpha: float = 0.05):
    """(miss_lo, miss_hi): Q below the lower / above the upper Wald limit."""
    lo, hi = wald_interval(Qhat, Vhat, alpha)
    Q = np.asarray(Q, dtype=float)
    return Q < lo, Q > hi


class NaiveMonitor:
    """Running time-uniform miss flags of Wald intervals checked at every update."""

    def __init__(self, alpha: float = 0.05, shape=()):
        self.alpha = alpha
        self.miss_lo = np.zeros(shape, dtype=bool)
        self.miss_hi = np.zeros(shape, dtype=bool)
        self.last_lo = np.zeros(shape, dtype=bool)
        self.last_hi = np.zeros(shape, dtype=bool)

    def update(self, Q, Qhat, Vhat) -> None:
        lo, hi = wald_miss(Q, Qhat, Vhat, self.alpha)
        self.last_lo, self.last_hi = lo, hi
        self.miss_lo = self.miss_lo | lo
        self.miss_hi = self.miss_hi | hi
