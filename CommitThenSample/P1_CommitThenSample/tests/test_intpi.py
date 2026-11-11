"""DESIGN §7.4 test_intpi: integer vs float thresholds (relative error ≤ 2^-40 off the clips);
τ = 2^64 ⇒ Z = 1; mixture ≥ floor; Σ budget identity; random fuzz over wide ranges."""
from __future__ import annotations

import random
from decimal import Decimal
from fractions import Fraction
from math import isqrt

import numpy as np
import pytest

from ctsample.design import intpi as ip

TWO64 = ip.TWO64
REL = 2.0 ** -40
ABS = 2.0 ** -63          # floor of the integer path (≤ 2^-64) plus float rounding at tiny π


def test_contract_constants_and_examples():
    assert ip.TWO64 == 1 << 64
    # λ = 1, a = 1, s = 1 → π = 1 (clipped exactly at 2^64)
    assert ip.int_threshold(1 << 40, 1 << 16, 1) == TWO64
    # λ = 1, a = 1, s = 4 → π = 1/2 exactly (isqrt exact)
    assert ip.int_threshold(1 << 40, 1 << 16, 4) == 1 << 63
    # formula as written in §7.1
    for Lam, a, s in [(12345678901, 70000, 1500), (1, 1, 1), (3 << 40, 5 << 16, 77)]:
        assert ip.int_threshold(Lam, a, s) == min(TWO64, (Lam * a << 24) // isqrt(s << 32))
    assert ip.int_threshold(0, 1 << 16, 100) == 0
    assert ip.int_threshold(1 << 40, 0, 100) == 0
    with pytest.raises(ValueError):
        ip.int_threshold(1, 1, 0)
    # never above 2^64
    assert ip.int_threshold(1 << 80, 1 << 60, 1) == TWO64


def test_floor_tau_exact():
    want = (1 << 64) // 20                         # ⌊0.05·2^64⌋ for the decimal 0.05
    assert want == 922337203685477580
    assert ip.floor_tau(0.05) == want
    assert ip.floor_tau("0.05") == want
    assert ip.floor_tau(Fraction(1, 20)) == want
    assert ip.floor_tau(Decimal("0.05")) == want
    assert ip.floor_tau(0) == 0
    assert ip.floor_tau(0.5) == 1 << 63
    assert ip.floor_tau(0.01) == (1 << 64) // 100
    with pytest.raises(ValueError):
        ip.floor_tau(1.0)
    with pytest.raises(ValueError):
        ip.floor_tau(-0.1)
    # the binary double of 0.05 would give a different value: the decimal convention is explicit
    assert (Fraction(0.05) * TWO64).__floor__() != want
    assert ip.t16_from_mix(0.2) == 13107 and ip.t16_from_mix(0) == 0 and ip.t16_from_mix(1) == 1 << 16


def test_sure_selection_at_two64():
    rng = random.Random(1)
    us = [0, 1, TWO64 - 1] + [rng.randrange(TWO64) for _ in range(10000)]
    assert all(ip.select(u, TWO64) for u in us)
    assert not any(ip.select(u, 0) for u in us)
    # τ = 2^64 also through the mixture
    assert ip.mix_tau(TWO64, TWO64, 13107) == TWO64
    assert ip.threshold_unit(1 << 80, 1 << 80, 1 << 40, 100, 0, 13107) == TWO64


def test_mixture_ge_floor_and_between_components():
    rng = random.Random(2)
    for _ in range(20000):
        tphi = rng.randrange(TWO64)
        tA = rng.randrange(tphi, TWO64 + 1)
        tS = rng.randrange(tphi, TWO64 + 1)
        t16 = rng.randrange(0, (1 << 16) + 1)
        m = ip.mix_tau(tA, tS, t16)
        assert m >= tphi
        assert min(tA, tS) <= m <= max(tA, tS)
    # full threshold_unit: τ_i ≥ τ^φ always
    for _ in range(5000):
        tphi = ip.floor_tau(rng.choice([0.0, 0.001, 0.01, 0.05, 0.2]))
        tau = ip.threshold_unit(rng.randrange(1 << 60), rng.randrange(1 << 50), rng.randrange(1 << 40),
                                rng.randrange(1, 1 << 31), tphi, rng.randrange(1 << 16))
        assert tphi <= tau <= TWO64


def _fuzz_inputs(rng: random.Random, n: int):
    rows = []
    for _ in range(n):
        s = int(round(2 ** rng.uniform(0, 31.9)))
        s = max(1, min(s, (1 << 32) - 1))
        # λ over ~2^-30 .. 2^20, â over 2^-16 .. 2^30 (Q16.16 allows arbitrary precision)
        Lam = max(1, int(2 ** rng.uniform(10, 60)))
        a_fix = max(1, int(2 ** rng.uniform(0, 46)))
        rows.append((Lam, a_fix, s))
    return rows


def test_float_twin_matches_int_threshold_off_clips():
    rng = random.Random(3)
    rows = _fuzz_inputs(rng, 20000)
    Lam = np.array([r[0] for r in rows], dtype=np.float64) * 2.0 ** -40
    a = np.array([r[1] for r in rows], dtype=np.float64) * 2.0 ** -16
    s = np.array([r[2] for r in rows], dtype=np.int64)
    pf = ip.pi_A_float(Lam, a, s)
    n_checked = 0
    for k, (L, A, S) in enumerate(rows):
        tau = ip.int_threshold(L, A, S)
        if tau >= TWO64 or tau == 0:
            continue                                    # clip boundaries
        pe = tau / TWO64
        assert abs(pf[k] - pe) <= REL * pe + ABS, (L, A, S, pf[k], pe)
        n_checked += 1
    assert n_checked > 5000
    # size-only component: â = s << 16
    pS = ip.pi_S_float(Lam, s)
    for k, (L, _, S) in enumerate(rows):
        tau = ip.int_threshold(L, S << 16, S)
        if 0 < tau < TWO64:
            assert abs(pS[k] - tau / TWO64) <= REL * tau / TWO64 + ABS


def test_float_twin_full_unit_with_floor_and_mixture():
    rng = random.Random(4)
    n_checked = 0
    for _ in range(5000):
        s = rng.randrange(60, 1 << 24)
        LamA = int(2 ** rng.uniform(20, 56))
        LamS = int(2 ** rng.uniform(10, 40))
        a_fix = int(2 ** rng.uniform(4, 30))
        tphi = ip.floor_tau(rng.choice(["0", "0.001", "0.01", "0.05"]))
        t16 = rng.choice([0, 13107, 1 << 15, 1 << 16])
        tA, tS, tau = ip.threshold_unit_parts(LamA, LamS, a_fix, s, tphi, t16)
        if tA >= TWO64 or tS >= TWO64:
            continue
        pA, pS, pf = ip.pi_float_parts(LamA * 2.0 ** -40, a_fix * 2.0 ** -16, np.array([s]),
                                       tphi / TWO64, t16 / 65536.0, lam_S=LamS * 2.0 ** -40)
        pe = tau / TWO64
        assert abs(float(pf[0]) - pe) <= REL * pe + ABS
        n_checked += 1
    assert n_checked > 2000


def test_true_sqrt_twin_error_is_isqrt_rounding_only():
    # the textbook twin (np.sqrt) differs only by the isqrt rounding: rel. error ≤ 1/r + 2^-40
    rng = random.Random(5)
    for L, A, S in _fuzz_inputs(rng, 3000):
        tau = ip.int_threshold(L, A, S)
        if not 0 < tau < TWO64:
            continue
        pt = float(ip.pi_A_float(L * 2.0 ** -40, A * 2.0 ** -16, np.array([S]), sqrt_mode="true")[0])
        pe = tau / TWO64
        r = isqrt(S << 32)
        assert abs(pt - pe) <= (1.0 / r + REL) * pe + ABS


def test_isqrt_s32_exact():
    rng = random.Random(6)
    s = np.array([1, 2, 3, 4, 76, 1500, (1 << 32) - 1] + [rng.randrange(1, 1 << 32) for _ in range(20000)],
                 dtype=np.int64)
    r = ip.isqrt_s32(s)
    assert all(int(ri) == isqrt(int(si) << 32) for ri, si in zip(r, s))


def test_budget_identity():
    rng = random.Random(7)
    n = 5000
    s = [rng.randrange(60, 200000) for _ in range(n)]
    a_fix = [rng.randrange(1, 1 << 17) for _ in range(n)]
    LamA, LamS = ip.lam_fix(0.8), ip.lam_fix(1e-3)
    tphi, t16 = ip.floor_tau(0.01), ip.t16_from_mix(0.2)
    parts = [ip.threshold_unit_parts(LamA, LamS, a, si, tphi, t16) for a, si in zip(a_fix, s)]
    tA = [p[0] for p in parts]
    tS = [p[1] for p in parts]
    tau = [p[2] for p in parts]
    # (i) Σ s_i π_i computed exactly == Σ s_i τ_i / 2^64
    E = ip.expected_cost_exact(tau, s)
    assert E == sum(Fraction(si) * ip.pi_exact(t) for si, t in zip(s, tau))
    assert E == Fraction(sum(si * t for si, t in zip(s, tau)), TWO64)
    # (ii) mixture identity: E = (1−w)·E^A + w·E^S − δ with 0 ≤ δ < Σ s_i·2^-64 (floor of >>16)
    w = Fraction(t16, 1 << 16)
    EA, ES = ip.expected_cost_exact(tA, s), ip.expected_cost_exact(tS, s)
    delta = (1 - w) * EA + w * ES - E
    assert 0 <= delta < Fraction(sum(s), TWO64)
    # (iii) float expected cost of the twin matches the exact value
    pf = ip.pi_float(LamA * 2.0 ** -40, np.array(a_fix) * 2.0 ** -16, np.array(s), tphi / TWO64,
                     t16 / 65536.0, lam_S=LamS * 2.0 ** -40)
    assert abs(ip.expected_cost_float(pf, s) - float(E)) <= 1e-9 * float(E)
    # (iv) batch helper and zero-payload handling
    pay = [i % 3 != 0 for i in range(n)]
    tb = ip.thresholds_batch(LamA, LamS, a_fix, s, tphi, t16, pay)
    assert all((t == tau[i]) if pay[i] else (t == 0) for i, t in enumerate(tb))


def test_lam_fix_and_a_fix_roundtrip():
    for lam in [0.0, 1e-6, 0.001, 0.5, 1.0, 123.456, 2 ** 20]:
        L = ip.lam_fix(lam)
        assert abs(L * 2.0 ** -40 - lam) <= 2 ** -41 + 1e-15 * lam
    assert ip.a_fix_of(1.0) == 1 << 16
    assert ip.a_fix_of(0.5) == 1 << 15
    with pytest.raises(ValueError):
        ip.lam_fix(-1.0)


def test_selection_frequency_matches_pi_exact():
    # u uniform on [0, 2^64): P(u < τ) = τ/2^64 exactly; check empirically within 4 SE
    rng = np.random.default_rng(8)
    for phi in ["0.01", "0.05", "0.3"]:
        tau = ip.floor_tau(phi)
        u = rng.integers(0, np.iinfo(np.uint64).max, size=400000, dtype=np.uint64, endpoint=True)
        freq = float(np.mean(u < np.uint64(tau)))
        p = tau / TWO64
        se = (p * (1 - p) / u.size) ** 0.5
        assert abs(freq - p) <= 4 * se
