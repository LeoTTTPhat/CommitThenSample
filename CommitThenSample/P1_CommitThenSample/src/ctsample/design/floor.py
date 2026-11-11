"""Floor rules (DESIGN §3.5, Prop. 4).

* Constant floor (default): phi0 = gamma * rho * sum_all s / sum_payload s, pooled over the training
  folds; under the horizon budget the floor then spends ~ gamma * Bbar per epoch.
* Per-device floor cap (flood guard, floor_mode = 1): if D_{e-1} < -D_max (D_max = 200 Bbar), devices
  whose epoch floor spend phi0 * C_{d,e} exceeds gamma Bbar / n_active get
  phi_{d,e} = gamma Bbar / (n_active C_{d,e})  (C_{d,e}: device d's payload bytes in epoch e).
  Capped units may have pi < phi0, which breaks cbar = bbar/phi0: UCS statements whose query
  includes capped units are withheld (V8); the LCS is unaffected (range-free).
* Per-epoch equalised floor (ablation E7): phi_e = clip(gamma B_e / C_e, phi0/2, 1).
* Capped two-tier analytics (Prop. 4(ii); analytic metrics only):
  phi_cap(phi_t) = gamma Bbar E / (C_H + phi_t C_T);  worst case g(phi_t) = phi_cap(phi_t)(1 - (1-phi_t)^m).
"""
from __future__ import annotations

import numpy as np

__all__ = ["phi0_from_training", "device_floor_cap", "flood_guard_floor", "equalised_floor",
           "phi_cap_two_tier", "two_tier_retention", "worst_case_retention", "phi_t_star"]


def phi0_from_training(gamma: float, rho: float, s_all_sum: float, s_payload_sum: float) -> float:
    """phi0 = gamma rho sum_all s / sum_payload s (pre-registered; committed as floor(phi0 2^64))."""
    if not s_payload_sum > 0:
        raise ValueError("no payload bytes in the training folds")
    return float(gamma * rho * s_all_sum / s_payload_sum)


def device_floor_cap(phi0: float, Bbar: float, gamma: float, device, s, pay=None):
    """Static part of the flood guard for ONE epoch.

    Returns (phi_capped, capped): per-unit floor if the guard is active, and the mask of units whose
    device is capped (phi_{d,e} < phi0). Only payload units count towards C_{d,e} and n_active."""
    device = np.asarray(device)
    s = np.asarray(s, dtype=float)
    pay = np.ones(s.shape, dtype=bool) if pay is None else np.asarray(pay, dtype=bool)
    phi = np.full(s.shape, float(phi0))
    capped = np.zeros(s.shape, dtype=bool)
    if not pay.any():
        return phi, capped
    d_pay = device[pay]
    uniq, inv = np.unique(d_pay, return_inverse=True)
    C = np.bincount(inv, weights=s[pay], minlength=len(uniq))
    n_active = len(uniq)
    share = gamma * Bbar / n_active
    over = phi0 * C > share
    if over.any():
        phi_d = np.where(over, share / np.where(over, C, 1.0), phi0)
        phi_pay = phi_d[inv]
        phi[pay] = phi_pay
        capped[pay] = over[inv]
    return phi, capped


def flood_guard_floor(phi0: float, D_prev, Bbar: float, gamma: float, device, s, pay=None,
                      Dmax_mult: float = 200.0):
    """Per-unit floor for one epoch given the deficit(s) D_prev (scalar or shape (R,)).

    Returns (phi, capped, active): phi has shape D_prev.shape + (n,) when D_prev is an array."""
    phi_c, capped_c = device_floor_cap(phi0, Bbar, gamma, device, s, pay)
    active = np.asarray(D_prev, dtype=float) < -Dmax_mult * Bbar
    act = active[..., None] if np.ndim(active) else active
    phi = np.where(act, phi_c, phi0)
    capped = np.logical_and(act, capped_c)
    return phi, capped, active


def equalised_floor(gamma: float, B_e, C_e, phi0: float):
    """Per-epoch equalised floor (ablation): clip(gamma B_e / C_e, phi0/2, 1); B_e = rho sum_{U_e} s."""
    B_e = np.asarray(B_e, dtype=float)
    C_e = np.asarray(C_e, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(C_e > 0, gamma * B_e / np.where(C_e > 0, C_e, 1.0), 1.0)
    return np.clip(x, phi0 / 2.0, 1.0)


def phi_cap_two_tier(phi_t, gamma: float, Bbar: float, E: int, C_H: float, C_T: float):
    """Equalised floor of the capped two-tier design: min(1, gamma Bbar E / (C_H + phi_t C_T))."""
    phi_t = np.asarray(phi_t, dtype=float)
    return np.minimum(1.0, gamma * Bbar * E / (C_H + phi_t * C_T))


def two_tier_retention(phi_t, m: int, gamma: float, Bbar: float, E: int, C_H: float, C_T: float):
    """(unpadded, padded) retention: content in the head is retained w.p. phi_cap; content pushed
    into the tail by a prefix L >= H is retained w.p. phi_cap (1 - (1-phi_t)^m)."""
    pc = phi_cap_two_tier(phi_t, gamma, Bbar, E, C_H, C_T)
    return pc, pc * (1.0 - (1.0 - np.asarray(phi_t, dtype=float)) ** m)


def worst_case_retention(phi_t, m: int, gamma: float, Bbar: float, E: int, C_H: float, C_T: float):
    """g(phi_t) = phi_cap(phi_t) (1 - (1-phi_t)^m) (worst case over {pad, no pad})."""
    unpadded, padded = two_tier_retention(phi_t, m, gamma, Bbar, E, C_H, C_T)
    return np.minimum(unpadded, padded)


def phi_t_star(m: int, gamma: float, Bbar: float, E: int, C_H: float, C_T: float, grid: int = 10001):
    """argmax_{phi_t in (0,1]} g(phi_t) on a grid (=1, the single tier, for m = 1)."""
    t = np.linspace(1.0 / grid, 1.0, grid)
    g = worst_case_retention(t, m, gamma, Bbar, E, C_H, C_T)
    i = int(np.argmax(g))
    return float(t[i]), float(g[i])
