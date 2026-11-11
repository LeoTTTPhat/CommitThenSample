"""Multi-estimand weights w* by exponentiated-gradient ascent (DESIGN Alg. 6, Prop. 3(b)) [SHOULD; simple].

    min_pi max_q V_q(pi)/Qt_q^2  =  max_{w in simplex} sum_q w_q V_q(pi(w))/Qt_q^2,
    a_i(w) = sqrt(sum_q w_q m_qi / Qt_q^2),   pi(w) = clip(lam(w) a(w)/sqrt(s), phi, 1),  sum pi s = B,
    V_q(pi) = sum_i m_qi (1/pi_i - 1)       (model-based; m_qi = predicted E[f_qi^2 | m_i])
Danskin: the gradient in w_q is V_q(pi(w))/Qt_q^2; EG step w <- w exp(eta g / max g), normalised.
Computed once per training fold (100 iterations), frozen and committed in H(policy).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["waterfill", "eg_minimax"]


def waterfill(a, s, B: float, phi: float, iters: int = 200):
    """pi = clip(lam a/sqrt(s), phi, 1) with sum pi s = B (bisection on log lam)."""
    from .probs import sqrt_eff
    a = np.asarray(a, dtype=float)
    s = np.asarray(s, dtype=float)
    r = a / sqrt_eff(s)                       # protocol sqrt rounding for integer costs
    cost = lambda ll: float(np.sum(np.clip(math.exp(ll) * r, phi, 1.0) * s))  # noqa: E731
    lo, hi = -80.0, 80.0
    if cost(lo) > B:
        return np.full_like(s, phi), 0.0
    if cost(hi) < B:
        return np.clip(math.exp(hi) * r, phi, 1.0), math.exp(hi)
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        if cost(m) > B:
            hi = m
        else:
            lo = m
    lam = math.exp(0.5 * (lo + hi))
    return np.clip(lam * r, phi, 1.0), lam


def eg_minimax(M, Qt, s, B: float, phi: float, iters: int = 100, step: float = 1.0, w0=None):
    """M: (Q, n) predicted second moments; Qt: (Q,) reference totals. Returns dict(w, pi, objective,
    history of max_q relative variance)."""
    M = np.asarray(M, dtype=float)
    Qt = np.asarray(Qt, dtype=float)
    Q = M.shape[0]
    Mn = M / (Qt[:, None] ** 2)
    w = np.full(Q, 1.0 / Q) if w0 is None else np.asarray(w0, dtype=float) / np.sum(w0)
    hist = []
    best = (math.inf, w.copy(), None)
    for _ in range(iters):
        a = np.sqrt(np.maximum(w @ Mn, 0.0)) + 1e-300
        pi, _lam = waterfill(a, s, B, phi)
        g = Mn @ (1.0 / pi - 1.0)                 # V_q(pi)/Qt_q^2
        worst = float(g.max())
        hist.append(worst)
        if worst < best[0]:
            best = (worst, w.copy(), pi)
        w = w * np.exp(step * g / max(worst, 1e-300))
        w /= w.sum()
    return {"w": best[1], "pi": best[2], "objective": best[0], "history": np.array(hist)}
