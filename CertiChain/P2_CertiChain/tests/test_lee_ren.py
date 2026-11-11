"""Hand-computed toys for Lee & Ren (arXiv:2501.02514v1) Alg. 2 (Eqs. 16-18) and Alg. 1 (Eqs. 7-8),
M = 2 test groups. V^ convention (smaller = more significant); every indicator is a strict "<".
alpha = 0.5 -> alpha~ = 0.9 alpha = 0.45; e-BH over n = 4 test units rejects k units iff e_(k) >= n/(alpha k) = 8/k.
"""
import numpy as np
import pytest

from certichain.baselines.lee_ren import lee_ren, lee_ren_evalues, lee_ren_R
from certichain.certify.evalues import ebh
from certichain.certify.losses import Incident, build_loss_tables

ALPHA = 0.5

# --------------------------------------------------------------------------- Alg. 2 toy
# K = 4 calibration groups, N_k = 2 (weights 1/N_k = 0.5):
#   k1: V = [1, 5] null [T, T];  k2: V = [2, 6] null [F, T];  k3: V = [3, 7] null [T, T];  k4: V = [4, 8] null [T, T]
#   weighted null count C(t) = 0.5 * #{null V in {1, 3, 4, 5, 6, 7, 8} : V < t}
# test groups: j1 = [0.5, 2.5], j2 = [1.5, 9.0]; scale (Eqs. 17/18) = sum_{l != j} N_{K+l} / (K+1) = 2/5 = 0.4
#
# j1 (denominator counts group-2 values {1.5, 9} below t):
#   FDP+_1(t) = (C+1)/max(1, c2) * 0.4 : t<=1: (0+1)/1*.4 = 0.40 ok | (1,1.5]: 1.5*.4 = 0.6 | (1.5,3]: 1.5/1*.4 = 0.6 |
#               (3,4]: 0.8 | ... | (8,9]: 4.5*.4 = 1.8 | (9,inf): 4.5/2*.4 = 0.9          -> T+_1 = 1
#   FDP-_1(t) = C/max(1, c2) * 0.4   : (1,1.5]: 0.2 | (1.5,3]: 0.2 | (3,4]: 1.0*.4 = 0.4 ok | (4,5]: 0.6 | ... |
#               (9,inf): 3.5/2*.4 = 0.7                                                     -> T-_1 = 4
#   e_1 = 5 * 1{V < 1} / (C(4) + 1) = 5 / (0.5*#{1,3} + 1) = 5/2 = 2.5 for V = 0.5; 0 for V = 2.5
# j2 (denominator counts group-1 values {0.5, 2.5}):
#   FDP+_2: t<=0.5: 0.4 ok | (0.5,1]: 1/1*.4 = 0.4 ok | (1,2.5]: 1.5*.4 = 0.6 | (2.5,3]: 1.5/2*.4 = 0.3 ok |
#           (3,4]: 2/2*.4 = 0.4 ok | (4,5]: 2.5/2*.4 = 0.5 | ... | (8,inf): 4.5/2*.4 = 0.9        -> T+_2 = 4
#   FDP-_2: (4,5]: 1.5/2*.4 = 0.3 ok | (5,6]: 2/2*.4 = 0.4 ok | (6,7]: 2.5/2*.4 = 0.5 | ... -> T-_2 = 6
#   e_2 = 5 * 1{V < 4} / (C(6) + 1) = 5 / (0.5*#{1,3,4,5} + 1) = 5/3 for V = 1.5; 0 for V = 9
# e = [2.5, 0, 5/3, 0]: k=1: 2.5 < 8, k=2: 5/3 < 4 -> no rejection (unboosted).
# boosted, U = 0.25: e/U = [10, 0, 6.67, 0]: k=1: 10 >= 8, k=2: 6.67 >= 4 -> k* = 2 -> reject (j1, 0.5), (j2, 1.5).
CAL2 = [([1.0, 5.0], [True, True]), ([2.0, 6.0], [False, True]), ([3.0, 7.0], [True, True]), ([4.0, 8.0], [True, True])]
TEST2 = [np.array([0.5, 2.5]), np.array([1.5, 9.0])]


def test_alg2_hand_toy_thresholds_and_evalues():
    d = lee_ren_evalues(CAL2, TEST2, "A2", ALPHA)
    assert d["alpha_tilde"] == pytest.approx(0.45)
    np.testing.assert_allclose(d["T_plus"], [1.0, 4.0])
    np.testing.assert_allclose(d["T_minus"], [4.0, 6.0])
    np.testing.assert_allclose(d["e"][0], [2.5, 0.0])
    np.testing.assert_allclose(d["e"][1], [5 / 3, 0.0])
    assert not ebh(np.concatenate(d["e"]), ALPHA).any()


def test_alg2_hand_toy_ebh_boosted_and_unboosted():
    un = lee_ren(CAL2, TEST2, ALPHA, "A2", boost=False, scores_are_v=True)
    assert not any(m.any() for m in un)
    b = lee_ren(CAL2, TEST2, ALPHA, "A2", boost=True, u=0.25, scores_are_v=True)
    assert b[0].tolist() == [True, False] and b[1].tolist() == [True, False]
    b5 = lee_ren(CAL2, TEST2, ALPHA, "A2", boost=True, u=0.5, scores_are_v=True)   # e/U = [5, 0, 3.33, 0]
    assert not any(m.any() for m in b5)


def test_alg2_score_convention_matches_v_convention():
    cal_s = [(-np.asarray(v), np.where(np.asarray(n), 0, 1)) for v, n in CAL2]   # y = 1 for the non-null unit
    test_s = [-t for t in TEST2]
    a = lee_ren(cal_s, test_s, ALPHA, "A2", boost=True, u=0.25)
    b = lee_ren(CAL2, TEST2, ALPHA, "A2", boost=True, u=0.25, scores_are_v=True)
    assert all((x == z).all() for x, z in zip(a, b))


# --------------------------------------------------------------------------- Alg. 1 toys
# K = 4 calibration groups (N_k = 2):
#   k1: V = [10, 3] null [T, T];  k2: V = [2.5, 11] null [F, T];  k3: V = [12, 0.2] null [T, F];  k4: V = [13, 14] null [T, T]
# scale (Eq. 8) = sum_{l=1}^{M} N_{K+l} / (K+1) = 4/5 = 0.8  (ALL test groups, unlike Eqs. 17/18)
#
# Toy A: test j1 = [0.5, 20], j2 = [1.0, 2.0]; subsample i* = (0,0,0,0) -> (10,T), (2.5,F), (12,T), (13,T);
#   subsampled nulls {10, 12, 13}, C(t) = #{those < t}.
#   j1 (c2 counts {1, 2}): t = 0.5, 1: (0+1)/1*.8 = 0.8 | t = 2: c2 = 1 -> 0.8 | t = 10: C = 0, c2 = 2 -> 1/2*.8 = 0.4 ok |
#       t = 12: 2/2*.8 = 0.8 | t = 13: 1.2 | t = 20, inf: 1.6                         -> T_1 = 10, e = 5/(0+1) = 5 for V = 0.5
#   j2 (c1 counts {0.5, 20}): at most c1 = 1 below t <= 20: >= 0.8; inf: (3+1)/2*.8 = 1.6 -> T_2 = -inf, e = 0
#   e = [5, 0, 0, 0]: 5 < 8 -> no rejection; boosted U = 0.5: 10 >= 8 -> reject (j1, 0.5).
#   (With the Alg.-2-style scale (n - N_j)/(K+1) = 0.4, j2 would qualify at t = 10 -> e_2 = [5, 5]: the toy pins Eq. 8.)
CAL1 = [([10.0, 3.0], [True, True]), ([2.5, 11.0], [False, True]), ([12.0, 0.2], [True, False]), ([13.0, 14.0], [True, True])]
TEST1A = [np.array([0.5, 20.0]), np.array([1.0, 2.0])]
# Toy B (averaged, Eq. 12 with two explicit draws): test j1 = [0.5, 5.0], j2 = [1.0, 2.0]
#   draw A = (0,0,0,0), nulls {10,12,13}:
#     j1: t = 5: C = 0, c2 = 2 -> 0.4 ok; t = 10: 0.4 ok; t = 12: 0.8 -> T_1 = 10 -> both V = 0.5, 5.0 eligible, e = 5
#     j2 (c1 counts {0.5, 5}): t = 10: C = 0, c1 = 2 -> 0.4 ok; t = 12: 0.8 -> T_2 = 10 -> both eligible, e = 5
#     e^A = [5, 5, 5, 5] -> k = 4: 5 >= 2 -> reject all four
#   draw B = (1,0,0,0) -> (3,T), (2.5,F), (12,T), (13,T); nulls {3, 12, 13}:
#     j1: t = 3: C = 0, c2 = 2 -> 0.4 ok; t = 5: C = 1 -> 0.8 -> T_1 = 3 -> only V = 0.5 eligible, e = 5
#     j2: t = 3: c1 = 1 -> 0.8; t = 5: C = 1, c1 = 1 -> 1.6; t = 12: C = 1, c1 = 2 -> 0.8; ... -> T_2 = -inf
#     e^B = [5, 0, 0, 0] -> 5 < 8 -> no rejection
#   e^avg = [5, 2.5, 2.5, 2.5]: k=3: 2.5 < 8/3; k = 4: 2.5 >= 2 -> reject all four
TEST1B = [np.array([0.5, 5.0]), np.array([1.0, 2.0])]


def test_alg1_hand_toy_A_scale_and_evalues():
    d = lee_ren_evalues(CAL1, TEST1A, "A1", ALPHA, subsamples=[[0, 0, 0, 0]])
    np.testing.assert_allclose(d["T"][0], [10.0, -np.inf])
    np.testing.assert_allclose(d["e"][0], [5.0, 0.0])
    np.testing.assert_allclose(d["e"][1], [0.0, 0.0])
    assert not ebh(np.concatenate(d["e"]), ALPHA).any()
    e_b = np.concatenate(d["e"]) / 0.5
    assert ebh(e_b, ALPHA).tolist() == [True, False, False, False]


def test_alg1_hand_toy_B_single_draws_and_average():
    dA = lee_ren_evalues(CAL1, TEST1B, "A1", ALPHA, subsamples=[[0, 0, 0, 0]])
    np.testing.assert_allclose(dA["T"][0], [10.0, 10.0])
    np.testing.assert_allclose(np.concatenate(dA["e"]), [5, 5, 5, 5])
    assert ebh(np.concatenate(dA["e"]), ALPHA).all()
    dB = lee_ren_evalues(CAL1, TEST1B, "A1", ALPHA, subsamples=[[1, 0, 0, 0]])
    np.testing.assert_allclose(dB["T"][0], [3.0, -np.inf])
    np.testing.assert_allclose(np.concatenate(dB["e"]), [5, 0, 0, 0])
    assert not ebh(np.concatenate(dB["e"]), ALPHA).any()
    dAB = lee_ren_evalues(CAL1, TEST1B, "A1", ALPHA, subsamples=[[0, 0, 0, 0], [1, 0, 0, 0]])
    np.testing.assert_allclose(np.concatenate(dAB["e"]), [5, 2.5, 2.5, 2.5])
    assert ebh(np.concatenate(dAB["e"]), ALPHA).all()
    assert dAB["r"] == 2


def test_requires_two_test_groups_and_unboosted_bound():
    with pytest.raises(ValueError):
        lee_ren_evalues(CAL2, TEST2[:1], "A2", ALPHA)
    rng = np.random.default_rng(0)
    cal = [(rng.normal(size=20), rng.random(20) < 0.9) for _ in range(30)]
    test = [rng.normal(size=15) - 2 * (rng.random(15) < 0.2) for _ in range(5)]
    for var in ("A1", "A2"):
        d = lee_ren_evalues(cal, test, var, 0.2, r=5)
        assert np.concatenate(d["e"]).max() <= 31 + 1e-9          # unboosted e <= K + 1 (Prop. 2)


def test_bank_path_matches_reference_implementation():
    rng = np.random.default_rng(5)
    incs = []
    for i in range(14):
        n = int(rng.integers(5, 30))
        y = (rng.random(n) < 0.3).astype(np.int8)
        s = rng.normal(size=n) + 4.0 * y
        s[rng.random(n) < 0.1] = 1.0                               # ties
        incs.append(Incident(f"t{i}", s, y))
    bank = build_loss_tables(incs, rng.normal(size=1000))
    cal, test = np.arange(8), np.arange(8, 14)
    cal_groups = [(bank.scores(k), bank.labels(k)) for k in cal]
    test_groups = [bank.scores(j) for j in test]
    for variant, boost, r in (("A2", False, 1), ("A2", True, 1), ("A1", False, 1), ("A1", True, 1), ("A1", True, 7)):
        R = lee_ren_R(bank, cal, test, 0.3, variant, boost, r=r, seed=11)
        masks = lee_ren(cal_groups, test_groups, 0.3, variant, boost, r=r, seed=11)
        assert R.tolist() == [int(m.sum()) for m in masks]
        for m in masks:                                            # rejections are top-score prefixes
            assert not np.any(np.diff(m.astype(int)) > 0)
