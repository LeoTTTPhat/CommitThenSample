"""Float design probabilities (DESIGN §3.5) used by the MC engine, analytic metrics and baselines.

    p^A = min(1, lam^A a / sqrt(s))      pi^A = max(phi, p^A)       (AI water-filling, size-aware)
    p^S = min(1, lam^S s / sqrt(s))      pi^S = max(phi, p^S)       (size-only = general form with a = s; B10)
    pi  = (1 - tau_mix) pi^A + tau_mix pi^S                        (defensive mixture, tau_mix = 0.2)
    B8 uniform = general form with a = sqrt(s):  pi = max(phi, min(1, lam))
    zero-payload units: pi = 0 (nothing to sample).

Consistency with the protocol path (design/intpi.py, P1-A): sqrt(s) is the protocol's
sqrt_eff(s) = isqrt(s << 32) / 2^16 (``intpi.isqrt_s32``) for integer costs 0 < s < 2^32
(sqrt_mode "auto" or "isqrt"; "true" = textbook np.sqrt), and p^S = lam^S * s / sqrt_eff(s) exactly
as int_threshold(Lam^S, s << 16, s). Fed with the COMMITTED values - lam = Lam 2^-40 (the controller
quantises), a = a_fix 2^-16 (``quantise_utility``), phi = tau_phi / 2^64 (``committed_floor``) and
tau_mix = t16 / 2^16 (``committed_mix``) - the twin equals tau/2^64 of intpi.threshold_unit to
<= 2^-40 pi relative (+2^-63) off the clips (tests/test_analytic.py::test_float_twin_matches_intpi).
With float inputs instead, the relative differences are <= ~2e-5 (tau_mix), immaterial for MC.

Design kinds and their lambda vectors (order matters; see ``n_lams``):
    "mixture": (lam^A, lam^S)   "A": (lam^A,)   "S": (lam^S,)   "uniform": (lam^U,)
Hooks: ``phi`` may be a per-unit array (flood-guard cap, equalised floor); ``boost`` multiplies the
utility a (follow-the-evidence boost, kappa_b) and therefore only affects the A component.
``sqrt_s`` passes a precomputed sqrt_eff(s) (the MC engine computes it once per stream).
"""
from __future__ import annotations

import numpy as np

from . import intpi

__all__ = ["TAU_MIX", "KINDS", "n_lams", "sqrt_eff", "pi_A", "pi_S", "pi_uniform", "pi_mixture", "design_pi",
           "component_pis", "expected_cost", "quantise_utility", "committed_mix", "committed_floor"]

TAU_MIX = 0.2
KINDS = ("mixture", "A", "S", "uniform")
_TWO32 = float(1 << 32)


def n_lams(kind: str) -> int:
    if kind not in KINDS:
        raise ValueError(f"unknown design kind {kind!r}")
    return 2 if kind == "mixture" else 1


def _f(x):
    return np.asarray(x, dtype=float)


def sqrt_eff(s, mode: str = "auto") -> np.ndarray:
    """sqrt(s) as the protocol defines it: isqrt(s << 32)/2^16 for integer 0 < s < 2^32.

    mode "auto": protocol rounding where s is such an integer, np.sqrt elsewhere (elementwise);
    "isqrt": protocol rounding, ValueError otherwise; "true": np.sqrt."""
    s = _f(s)
    if mode == "true":
        return np.sqrt(s)
    if mode not in ("auto", "isqrt"):
        raise ValueError("sqrt mode must be 'auto', 'isqrt' or 'true'")
    ok = (s > 0) & (s < _TWO32) & (s == np.floor(s))
    if mode == "isqrt" and not np.all(ok):
        raise ValueError("isqrt mode needs integer costs 0 < s < 2^32")
    out = np.sqrt(np.maximum(s, 0.0))
    if np.any(ok):
        out[ok] = intpi.isqrt_s32(s[ok].astype(np.int64)).astype(np.float64) / 65536.0
    return out


def _sq(s, sqrt_s, mode):
    return _f(sqrt_s) if sqrt_s is not None else sqrt_eff(s, mode)


def pi_A(lam_A, a, s, phi, sqrt_s=None, sqrt_mode: str = "auto"):
    """max(phi, min(1, lam_A a / sqrt_eff(s)))."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.minimum(1.0, _f(lam_A) * _f(a) / _sq(s, sqrt_s, sqrt_mode))
    return np.maximum(_f(phi), np.nan_to_num(p, nan=0.0))


def pi_S(lam_S, s, phi, sqrt_s=None, sqrt_mode: str = "auto"):
    """max(phi, min(1, lam_S s / sqrt_eff(s)))  (= lam_S sqrt(s) with the true root)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.minimum(1.0, _f(lam_S) * _f(s) / _sq(s, sqrt_s, sqrt_mode))
    return np.maximum(_f(phi), np.nan_to_num(p, nan=0.0))


def pi_uniform(lam_U, phi, like=None):
    """B8: max(phi, min(1, lam_U)), broadcast to ``like`` if given."""
    p = np.maximum(_f(phi), np.minimum(1.0, _f(lam_U)))
    if like is not None:
        p = np.broadcast_to(p, np.broadcast_shapes(np.shape(p), np.shape(like))).astype(float)
    return p


def pi_mixture(lam_A, lam_S, a, s, phi, tau_mix: float = TAU_MIX, sqrt_s=None, sqrt_mode: str = "auto"):
    sq = _sq(s, sqrt_s, sqrt_mode)
    return (1.0 - tau_mix) * pi_A(lam_A, a, s, phi, sq) + tau_mix * pi_S(lam_S, s, phi, sq)


def component_pis(kind: str, lams, a, s, phi, boost=None, sqrt_s=None, sqrt_mode: str = "auto"):
    """List of the component probability arrays of a design (one per lambda)."""
    a_eff = _f(a) if boost is None else _f(a) * _f(boost)
    if kind == "uniform":
        p = pi_uniform(lams[0], phi)
        return [p * np.ones(np.broadcast_shapes(np.shape(p), np.shape(s)))]
    sq = _sq(s, sqrt_s, sqrt_mode)
    if kind == "mixture":
        return [pi_A(lams[0], a_eff, s, phi, sq), pi_S(lams[1], s, phi, sq)]
    if kind == "A":
        return [pi_A(lams[0], a_eff, s, phi, sq)]
    if kind == "S":
        return [pi_S(lams[0], s, phi, sq)]
    raise ValueError(kind)


def design_pi(kind: str, lams, a, s, phi, tau_mix: float = TAU_MIX, pay=None, boost=None, sqrt_s=None,
              sqrt_mode: str = "auto"):
    """Inclusion probability of every unit; 0 for zero-payload units (pay == False).

    ``lams`` is a sequence of lambdas (scalars or arrays broadcastable against the unit arrays,
    e.g. shape (R, 1) for a reps axis)."""
    comps = component_pis(kind, lams, a, s, phi, boost, sqrt_s, sqrt_mode)
    if kind == "mixture":
        pi = (1.0 - tau_mix) * comps[0] + tau_mix * comps[1]
    else:
        pi = comps[0]
    if pay is not None:
        pi = np.where(np.asarray(pay, dtype=bool), pi, 0.0)
    return pi


def expected_cost(pi, s, axis=-1):
    """Single-tier expected cost sum pi s."""
    return (_f(pi) * _f(s)).sum(axis=axis)


# ---- committed fixed-point values (exact protocol inputs) --------------------------------------- #
def quantise_utility(a):
    """a -> a_fix 2^-16 with a_fix = round(a 2^16) (intpi.a_fix_of; half-to-even on the double)."""
    return np.rint(_f(a) * 65536.0) / 65536.0


def committed_mix(tau_mix: float) -> float:
    """Effective mixture weight t16/2^16 with t16 = floor(tau_mix 2^16) (0.2 -> 13107/65536)."""
    return intpi.mix_weight(intpi.t16_from_mix(tau_mix))


def committed_floor(phi0: float) -> float:
    """tau_phi / 2^64 with tau_phi = floor(phi0 2^64) (decimal convention of intpi.floor_tau)."""
    return intpi.floor_tau(phi0) / float(intpi.TWO64)
