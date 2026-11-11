"""Horizon budget controller (DESIGN §3.5, Prop. 3(c), Prop. 5(iii); contract §7.1).

    D_e = D_{e-1} + Bbar - cost_e            (corrected recursion; revision 1 used B_e -> driftless)
    log lam_{e+1} = log lam_ref + kappa * clip(D_e / (Wc * Bbar), -L, L),  kappa = 1, Wc = 30, L = ln 16
    cost_e = sum pi s            (mode "expected": a function of committed data only -> static design)
           = sum_{Z=1} s         (mode "realised": makes pi depend on past Z -> adaptive design)
    Lam_{e+1} = round(lam_{e+1} * 2^40) is committed in Hdr_{e+1}.

One deficit D drives every component lambda of a design (Alg. 1 line 8: Lam^{A,S} <- controller(D_e));
each component has its own lam_ref, calibrated on training folds so that it alone spends Bbar.
The state is vectorised: ``shape`` = (R,) runs R independent replicates at once.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .probs import component_pis, sqrt_eff

__all__ = ["HorizonController", "Rev1CarryController", "calibrate_lambda", "calibrate_design",
           "mean_expected_cost", "TWO40"]

TWO40 = float(1 << 40)


class HorizonController:
    """Log-integral horizon controller with saturation.

    lam_ref: scalar, or a 1-D array of component references (then lam() has a trailing axis).
    quantise: if True (default), lam() returns the committed fixed-point value Lam * 2^-40."""

    def __init__(self, lam_ref, Bbar: float, kappa: float = 1.0, Wc: float = 30, L: float = math.log(16),
                 mode: str = "expected", shape=(), D0: float = 0.0, quantise: bool = True):
        if mode not in ("expected", "realised"):
            raise ValueError("mode must be 'expected' or 'realised'")
        if not Bbar > 0:
            raise ValueError("Bbar must be > 0")
        self.lam_ref = np.asarray(lam_ref, dtype=float)
        if np.any(~(self.lam_ref > 0)):
            raise ValueError("lam_ref must be > 0")
        self.Bbar = float(Bbar)
        self.kappa, self.Wc, self.L = float(kappa), float(Wc), float(L)
        self.mode = mode
        self.quantise = quantise
        self.shape = tuple(shape)
        self.D = np.full(self.shape, float(D0))
        self.spend = np.zeros(self.shape)
        self.max_abs_D = np.abs(self.D).copy()
        self.n_epochs = 0

    # ---- lambda --------------------------------------------------------------------------- #
    def factor(self):
        """exp(kappa * clip(D/(Wc Bbar), -L, L)) (shape = state shape)."""
        return np.exp(self.kappa * np.clip(self.D / (self.Wc * self.Bbar), -self.L, self.L))

    def lam(self):
        g = self.factor()
        lam = g[..., None] * self.lam_ref if self.lam_ref.ndim == 1 else g * self.lam_ref
        if self.quantise:
            lam = np.rint(lam * TWO40) / TWO40
        return lam.item() if np.ndim(lam) == 0 else lam

    def Lam_fix(self):
        """Committed fixed-point water level round(lam * 2^40) (Python int for a scalar state)."""
        g = self.factor()
        lam = g[..., None] * self.lam_ref if self.lam_ref.ndim == 1 else g * self.lam_ref
        if np.ndim(lam) == 0:
            return int(round(float(lam) * TWO40))
        x = np.rint(np.asarray(lam) * TWO40)
        if np.any(x >= 2.0 ** 63):
            return np.vectorize(lambda v: int(v), otypes=[object])(x)
        return x.astype(np.int64)

    # ---- state ---------------------------------------------------------------------------- #
    def update(self, cost_e) -> None:
        """D <- D + Bbar - cost_e (vectorised)."""
        cost_e = np.asarray(cost_e, dtype=float)
        self.D = self.D + self.Bbar - cost_e
        self.spend = self.spend + cost_e
        self.max_abs_D = np.maximum(self.max_abs_D, np.abs(self.D))
        self.n_epochs += 1

    def spend_ratio(self):
        """Cumulative spend / (Bbar * epochs)."""
        return self.spend / (self.Bbar * max(self.n_epochs, 1))

    def flood_guard_active(self, Dmax_mult: float = 200.0):
        """floor_mode = 1 iff D_{e-1} < -D_max (D_max = 200 Bbar); call before the epoch's update."""
        return self.D < -Dmax_mult * self.Bbar


class Rev1CarryController:
    """WRONG revision-1 recursion, kept ONLY as a regression guard (test_controller).

    Per-epoch budget B_e = Bbar + clip(D_{e-1}, -Bbar/2, Bbar/2) and D_e = D_{e-1} + B_e - cost_e.
    If the epoch spends B_e in expectation, D is a driftless martingale: it never pulls cumulative
    spend back to Bbar*E, so spend/target random-walks (reviewer check: 5-95% = 0.51-1.45)."""

    def __init__(self, Bbar: float, shape=(), corrected: bool = False):
        self.Bbar = float(Bbar)
        self.corrected = corrected
        self.D = np.zeros(tuple(shape))
        self.spend = np.zeros(tuple(shape))
        self.n_epochs = 0

    def budget(self):
        return self.Bbar + np.clip(self.D, -self.Bbar / 2, self.Bbar / 2)

    def update(self, cost_e, B_e=None) -> None:
        B_e = self.budget() if B_e is None else B_e
        self.D = self.D + (self.Bbar if self.corrected else B_e) - cost_e
        self.spend = self.spend + cost_e
        self.n_epochs += 1

    def spend_ratio(self):
        return self.spend / (self.Bbar * max(self.n_epochs, 1))


# ------------------------------------------------------------------------------------------ #
# calibration of lam_ref on training folds
# ------------------------------------------------------------------------------------------ #
def _bisect_log(cost_of_loglam: Callable[[float], float], target: float, lo: float, hi: float, iters: int):
    """Monotone bisection on log(lam) for cost(log lam) = target."""
    c_lo, c_hi = cost_of_loglam(lo), cost_of_loglam(hi)
    if c_lo > target:
        raise ValueError(f"floor alone overspends: min cost {c_lo:.6g} > target {target:.6g}")
    if c_hi < target:
        return hi, False                       # saturated: cannot spend the target
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        if cost_of_loglam(m) > target:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi), True


def mean_expected_cost(kind_component: str, lam: float, s, a, phi, n_epochs: int, boost=None, sqrt_s=None) -> float:
    """Mean per-epoch expected cost sum(pi s)/n_epochs of ONE design component (protocol sqrt rounding)."""
    kind = {"A": "A", "S": "S", "uniform": "uniform"}[kind_component]
    pi = component_pis(kind, (lam,), a if a is not None else np.ones_like(np.asarray(s, float)), s, phi, boost,
                       sqrt_s=sqrt_s)[0]
    return float(np.sum(pi * np.asarray(s, dtype=float)) / n_epochs)


def calibrate_lambda(s, a, Bbar: float, phi, n_epochs: int, component: str = "A",
                     lo: float = -60.0, hi: float = 60.0, iters: int = 200, return_status: bool = False):
    """lam_ref by bisection so that the mean expected cost per epoch over the training stream
    (payload units: s, utility a, floor phi) equals Bbar for one component ("A", "S", "uniform")."""
    s = np.asarray(s, dtype=float)
    a = None if a is None else np.asarray(a, dtype=float)
    sq = sqrt_eff(s)
    f = lambda ll: mean_expected_cost(component, math.exp(ll), s, a, phi, n_epochs, sqrt_s=sq)  # noqa: E731
    ll, ok = _bisect_log(f, float(Bbar), lo, hi, iters)
    lam = math.exp(ll)
    return (lam, ok) if return_status else lam


def calibrate_design(kind: str, s, a, Bbar: float, phi, n_epochs: int, **kw):
    """Component lam_refs of a design: every component alone spends Bbar (DESIGN §3.5)."""
    comps = {"mixture": ("A", "S"), "A": ("A",), "S": ("S",), "uniform": ("uniform",)}[kind]
    return np.array([calibrate_lambda(s, a, Bbar, phi, n_epochs, component=c, **kw) for c in comps])
