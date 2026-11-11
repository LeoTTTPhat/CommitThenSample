"""Exact integer selection thresholds (DESIGN §3.6, §7.1) and their float64 twins.

Protocol path (Python ints, bit-exact; used by the gateway AND the verifier):

    TWO64 = 2^64
    int_threshold(Lam, a_fix, s) = min(2^64, (Lam·a_fix << 24) // isqrt(s << 32))      (s > 0)
        = ⌊λ·a/√s·2^64⌋ up to the isqrt rounding (part of the definition), with
          λ = Lam·2^-40 (Q24.40), a = a_fix·2^-16 (Q16.16, arbitrary precision).
    floor_tau(phi0)  = ⌊phi0·2^64⌋                 (exact rational arithmetic, see below)
    mix_tau(tA, tS, t16) = ((2^16 − t16)·tA + t16·tS) >> 16
    threshold_unit(LamA, LamS, a_fix, s, tau_phi, t16)          (Alg. 1 line 3)
        τ^A = max(τ^φ, IntThreshold(Λ^A, â, s))
        τ^S = max(τ^φ, IntThreshold(Λ^S, s << 16, s))            (size-only component, π^S = λ^S·√s)
        τ   = mix_tau(τ^A, τ^S, t16)
    Selection: Z = [u < τ] with u a u64; π := τ/2^64 exactly (pi_exact).

``floor_tau`` exactness: the argument is converted to a ``fractions.Fraction`` first —
ints/Fractions as is, ``Decimal`` exactly, ``str`` parsed as a decimal literal, and a Python
``float`` through its shortest round-trip repr (``repr(0.05) == '0.05'``), i.e. the decimal value
that was written in the config, NOT the binary double nearest to it. So
floor_tau(0.05) == floor_tau("0.05") == ⌊0.05·2^64⌋ = 922337203685477580, whereas the exact binary
double 0.05 would give 922337203685477632. The pre-registered φ_0 is a decimal number, and gateway
and verifier must agree bit-exactly, hence the decimal convention. φ_0 must lie in [0, 1).

Float twins (numpy, for the Monte Carlo; never used in the protocol path):
    pi_A_float(lam, a, s)       = min(1, lam·a/√s)
    pi_S_float(lam_S, s)        = min(1, lam_S·√s)
    pi_float(lam, a, s, phi0, tau_mix, lam_S=None)
        = (1 − w)·max(φ0, π^A) + w·max(φ0, π^S),  w = tau_mix (lam_S defaults to lam)
  By default √s is the protocol's √s_eff = isqrt(s<<32)/2^16 (sqrt_mode="isqrt"), which makes the
  twin agree with τ/2^64 to ≤ 2^-40·π (+2^-63) off the clips; sqrt_mode="true" uses np.sqrt.
``t16_from_mix(tau_mix)`` = ⌊tau_mix·2^16⌋ (same decimal convention), and ``mix_weight(t16)`` is the
exact weight t16/2^16 that the float twin should use for bit-faithful comparisons.
"""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from math import isqrt

import numpy as np

TWO64 = 1 << 64
TWO16 = 1 << 16
LAM_FRAC_BITS = 40          # λ = Λ·2^-40
A_FRAC_BITS = 16            # a = â·2^-16

__all__ = ["TWO64", "TWO16", "int_threshold", "floor_tau", "mix_tau", "threshold_unit",
           "threshold_unit_parts", "thresholds_batch", "pi_exact", "to_fraction", "lam_fix",
           "a_fix_of", "t16_from_mix", "mix_weight", "pi_A_float", "pi_S_float", "pi_float",
           "pi_float_parts", "expected_cost_exact", "expected_cost_float", "select", "isqrt_s32"]


def to_fraction(v) -> Fraction:
    """Exact rational value of v (float → shortest decimal repr; see module docstring)."""
    if isinstance(v, bool):
        raise TypeError("bool is not a number here")
    if isinstance(v, (int, Fraction)):
        return Fraction(v)
    if isinstance(v, Decimal):
        return Fraction(v)
    if isinstance(v, str):
        return Fraction(Decimal(v.strip()))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            raise ValueError("non-finite value")
        return Fraction(Decimal(repr(f)))
    if isinstance(v, np.integer):
        return Fraction(int(v))
    raise TypeError(f"unsupported type {type(v)}")


def int_threshold(Lam: int, a_fix: int, s: int) -> int:
    """⌊λ·a/√s·2^64⌋ with λ = Lam·2^-40, a = a_fix·2^-16, √s via isqrt(s<<32); clipped to [0, 2^64]."""
    Lam, a_fix, s = int(Lam), int(a_fix), int(s)
    if s <= 0:
        raise ValueError("unit cost s must be > 0")
    if Lam <= 0 or a_fix <= 0:
        return 0
    r = isqrt(s << 32)
    return min(TWO64, (Lam * a_fix << 24) // r)


def floor_tau(phi0) -> int:
    """τ^φ = ⌊φ0·2^64⌋ (exact; φ0 ∈ [0, 1))."""
    f = to_fraction(phi0)
    if not 0 <= f < 1:
        raise ValueError("phi0 must be in [0, 1)")
    return (f.numerator << 64) // f.denominator


def t16_from_mix(tau_mix) -> int:
    """t16 = ⌊tau_mix·2^16⌋ (exact, decimal convention); tau_mix ∈ [0, 1]."""
    f = to_fraction(tau_mix)
    if not 0 <= f <= 1:
        raise ValueError("tau_mix must be in [0, 1]")
    return (f.numerator << 16) // f.denominator


def mix_weight(t16: int) -> float:
    return t16 / TWO16


def lam_fix(lam) -> int:
    """Λ = round(λ·2^40) (round half to even on the exact rational)."""
    f = to_fraction(lam) * (1 << LAM_FRAC_BITS)
    if f < 0:
        raise ValueError("lambda must be >= 0")
    return round(f)


def a_fix_of(a) -> int:
    """â = round(a·2^16)."""
    f = to_fraction(a) * (1 << A_FRAC_BITS)
    if f < 0:
        raise ValueError("utility must be >= 0")
    return round(f)


def mix_tau(tA: int, tS: int, t16: int) -> int:
    """((2^16 − t16)·tA + t16·tS) >> 16."""
    if not 0 <= t16 <= TWO16:
        raise ValueError("t16 must be in [0, 2^16]")
    return ((TWO16 - t16) * tA + t16 * tS) >> 16


def threshold_unit_parts(LamA: int, LamS: int, a_fix: int, s: int, tau_phi: int,
                         t16: int) -> tuple[int, int, int]:
    """(τ^A, τ^S, τ) of Alg. 1 line 3."""
    tA = max(tau_phi, int_threshold(LamA, a_fix, s))
    tS = max(tau_phi, int_threshold(LamS, s << 16, s))
    return tA, tS, mix_tau(tA, tS, t16)


def threshold_unit(LamA: int, LamS: int, a_fix: int, s: int, tau_phi: int, t16: int) -> int:
    """τ_i = mix(max(τ^φ, IntThreshold(Λ^A, â, s)), max(τ^φ, IntThreshold(Λ^S, s<<16, s)))."""
    return threshold_unit_parts(LamA, LamS, a_fix, s, tau_phi, t16)[2]


def thresholds_batch(LamA: int, LamS: int, a_fix, s, tau_phi: int, t16: int, pay=None) -> list[int]:
    """τ for many units (Python ints); zero-payload units (pay False) get τ = 0."""
    out = []
    if pay is None:
        pay = [True] * len(s)
    for a, si, p in zip(a_fix, s, pay, strict=True):
        out.append(threshold_unit(LamA, LamS, int(a), int(si), tau_phi, t16) if p else 0)
    return out


def pi_exact(tau: int) -> Fraction:
    """π := τ/2^64 exactly."""
    return Fraction(int(tau), TWO64)


def select(u: int, tau: int) -> bool:
    """Z = [u < τ] (u a u64)."""
    return u < tau


def expected_cost_exact(taus, sizes) -> Fraction:
    """Σ π_i s_i as an exact rational."""
    num = sum(int(t) * int(s) for t, s in zip(taus, sizes, strict=True))
    return Fraction(num, TWO64)


# ------------------------------------------------------------------ float twins (Monte Carlo)
#
# The protocol defines √s through r = isqrt(s << 32) = ⌊√s·2^16⌋. That rounding has relative size
# up to 2^-16/√s (≈ 2^-19 for a 76-byte unit), far above 2^-40, so a twin that uses the true √s
# cannot be bit-faithful. The default twin therefore uses the SAME r (exact, vectorised):
#     √s_eff = r / 2^16,   π^A = λ·a/√s_eff,   π^S = λ^S·s/√s_eff.
# With λ = Λ·2^-40, a = â·2^-16 (the committed values) the twin then agrees with τ/2^64 to
# ≤ 2^-40·π + 2^-63 off the clips (tests/test_intpi.py). ``sqrt_mode="true"`` gives the textbook
# formula for analysis.

def isqrt_s32(s) -> np.ndarray:
    """Exact r = isqrt(s << 32) for an array of integer costs 0 < s < 2^32 (uint64 output)."""
    s = np.asarray(s)
    if s.size and (np.any(s <= 0) or np.any(s >= (1 << 32))):
        raise ValueError("isqrt_s32 needs 0 < s < 2^32")
    S = s.astype(np.uint64) << np.uint64(32)
    rmax = np.uint64((1 << 32) - 1)            # isqrt(S) ≤ 2^32 − 1 since S < 2^64
    r = np.minimum(np.floor(np.sqrt(s.astype(np.float64)) * 65536.0), float(rmax)).astype(np.uint64)
    one = np.uint64(1)
    for _ in range(3):                         # float estimate is within ±1; fix up exactly
        hi = r * r > S                         # r ≤ 2^32 − 1 ⇒ r² < 2^64, no overflow
        r = np.where(hi, r - one, r)
        r1 = np.minimum(r + one, rmax)         # (r+1)² would overflow at r+1 = 2^32 (never ≤ S)
        lo = (r < rmax) & (r1 * r1 <= S)
        r = np.where(lo, r + one, r)
    return r


def _sqrt_eff(s, sqrt_mode: str) -> np.ndarray:
    if sqrt_mode == "isqrt":
        return isqrt_s32(s).astype(np.float64) / 65536.0
    if sqrt_mode == "true":
        return np.sqrt(np.asarray(s, dtype=np.float64))
    raise ValueError("sqrt_mode must be 'isqrt' or 'true'")


def pi_A_float(lam, a, s, sqrt_mode: str = "isqrt"):
    """min(1, λ·a/√s) (float twin of int_threshold(Λ, â, s)/2^64)."""
    return np.minimum(1.0, np.asarray(lam, dtype=np.float64) * np.asarray(a, dtype=np.float64)
                      / _sqrt_eff(s, sqrt_mode))


def pi_S_float(lam_S, s, sqrt_mode: str = "isqrt"):
    """min(1, λ^S·√s) (float twin of int_threshold(Λ^S, s<<16, s)/2^64)."""
    sf = np.asarray(s, dtype=np.float64)
    return np.minimum(1.0, np.asarray(lam_S, dtype=np.float64) * sf / _sqrt_eff(s, sqrt_mode))


def pi_float_parts(lam, a, s, phi0=0.0, tau_mix=0.0, lam_S=None, sqrt_mode: str = "isqrt"):
    """(π^A, π^S, π) in float64; each component is clipped to [φ0, 1] before mixing.

    For bit-faithful comparisons pass lam = Λ·2^-40, a = â·2^-16, phi0 = τ^φ/2^64 and
    tau_mix = t16/2^16 (the committed integers)."""
    if lam_S is None:
        lam_S = lam
    pA = np.maximum(float(phi0), pi_A_float(lam, a, s, sqrt_mode))
    pS = np.maximum(float(phi0), pi_S_float(lam_S, s, sqrt_mode))
    w = float(tau_mix)
    return pA, pS, (1.0 - w) * pA + w * pS


def pi_float(lam, a, s, phi0=0.0, tau_mix=0.0, lam_S=None, sqrt_mode: str = "isqrt"):
    """Float twin of threshold_unit/2^64 (vectorised)."""
    return pi_float_parts(lam, a, s, phi0, tau_mix, lam_S, sqrt_mode)[2]


def expected_cost_float(pi, s) -> float:
    return float(np.dot(np.asarray(pi, dtype=np.float64), np.asarray(s, dtype=np.float64)))
