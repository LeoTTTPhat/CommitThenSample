"""Edge cases for reference p-values, the nested BH / score-threshold families, CRC, FDX and LTT."""
import numpy as np
import pytest

from certichain.certify.crc import certificate, crc_threshold, crc_threshold_from_sums, table_hash
from certichain.certify.losses import Incident, build_loss_tables, fdp_curve, monotonize
from certichain.certify.ltt import hb_pvalue, ltt_threshold, ltt_threshold_from_sums
from certichain.certify.pvalues import ref_pvalues, sort_null_pool
from certichain.certify.rules import (LAMBDA_GRID, bh_count_sorted, bh_nested, bh_select, st_counts,
                                      st_thresholds)


def brute_bh(p, lam):
    """Textbook BH: R = max{i : p_(i) <= lam i / N}; select the R smallest (ties at the cut included)."""
    n = len(p)
    ps = np.sort(p)
    R = 0
    for i in range(1, n + 1):
        if ps[i - 1] <= lam * i / n + 1e-15:
            R = i
    return set(np.flatnonzero(p <= ps[R - 1])) if R > 0 else set()


def test_ref_pvalues_bruteforce_and_ties():
    s0 = sort_null_pool([0.0, 1.0, 1.0, 2.0, 3.0])
    s = np.array([-1.0, 0.0, 1.0, 1.5, 3.0, 4.0])
    p = ref_pvalues(s, s0)
    expect = [(1 + np.sum(s0 >= x)) / 6 for x in s]
    np.testing.assert_allclose(p, expect)
    assert p[-1] == pytest.approx(1 / 6)          # above max(S_0): the resolution floor
    with pytest.raises(ValueError):
        ref_pvalues([np.nan], s0)
    with pytest.raises(ValueError):
        sort_null_pool([])


def test_bh_nested_matches_bruteforce_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(1, 40))
        p = np.round(rng.uniform(0, 0.2, n), 3)    # rounding creates ties
        p[rng.random(n) < 0.3] = 0.001
        sets = bh_nested(p, LAMBDA_GRID)
        assert len(sets[0]) == 0                   # S(lambda_0) = {}
        prev = set()
        for lam, S in zip(LAMBDA_GRID, sets):
            S = set(S.tolist())
            assert S == brute_bh(p, lam)
            assert prev <= S                       # nested
            prev = S
        assert set(np.flatnonzero(bh_select(p, 0.1))) == brute_bh(p, 0.1)
        assert bh_count_sorted(np.sort(p), 0.1) == len(brute_bh(p, 0.1))


def test_bh_exact_boundary_is_selected():
    # p_(2) = lam * 2 / N exactly (rational p-values): BH must select it despite float rounding
    p = np.array([0.01, 0.02, 0.9, 0.9, 0.9])
    assert set(np.flatnonzero(bh_select(p, 0.05))) == {0, 1}


def test_score_threshold_family():
    s0 = sort_null_pool(np.arange(10000) / 10000.0)
    t = st_thresholds(s0, LAMBDA_GRID)
    assert np.isinf(t[0]) and np.all(np.diff(t[1:]) <= 0)
    s = np.sort(np.array([0.99995, 0.5, 0.95, 0.1]))[::-1]
    R = st_counts(s, t)
    assert R[0] == 0 and np.all(np.diff(R) >= 0) and R[-1] == 3   # 0.1 < 20% quantile


def test_crc_threshold_basic_abstain_and_ties():
    G = 5
    L = np.zeros((9, G))
    assert crc_threshold(L, 0.1) == G - 1         # K = 9: (0 + 1)/10 <= 0.1 everywhere -> largest
    assert crc_threshold(np.zeros((8, G)), 0.1) == 0   # K + 1 < 1/alpha -> abstain
    assert crc_threshold(np.zeros((0, G)), 0.1) == 0
    # exact equality (sum L + 1)/(K+1) == alpha is accepted
    L = np.zeros((19, G))
    L[0, 2:] = 1.0                                 # (1 + 1)/20 = 0.1 at columns 2..4
    assert crc_threshold(L, 0.1) == G - 1
    L[1, 3:] = 0.5                                 # (1.5 + 1)/20 = 0.125 at 3,4
    assert crc_threshold(L, 0.1) == 2
    with pytest.raises(ValueError):
        crc_threshold(np.array([[0.0, 0.5, 0.2]] * 20), 0.1)        # not monotone
    assert crc_threshold(np.array([[0.0, 0.5, 0.0]] * 20), 0.1, check_monotone=False) == 2


def test_crc_from_sums_matches_loop():
    rng = np.random.default_rng(1)
    L = monotonize(rng.uniform(0, 0.3, size=(200, 61)) * (rng.random((200, 61)) < 0.3))
    sums, Ks, idx = [], [], []
    for s in range(30):
        rows = rng.choice(200, size=int(rng.integers(5, 150)), replace=False)
        sums.append(L[rows].sum(0)); Ks.append(rows.size); idx.append(crc_threshold(L[rows], 0.1))
    np.testing.assert_array_equal(crc_threshold_from_sums(np.array(sums), np.array(Ks), 0.1), idx)


def test_certificate_hash_is_stable():
    L = np.linspace(0, 1, 12).reshape(3, 4)
    c = certificate(L, 0.1, 0, LAMBDA_GRID[:4])
    assert c["K"] == 3 and c["abstain"] and c["calibration_table_blake3"] == table_hash(L.copy())
    assert table_hash(L) != table_hash(L[:, ::-1].copy())


def test_hb_pvalue_and_ltt():
    # R^ = 0: p = min((1-alpha)^K, e (1-alpha)^K) = (1-alpha)^K
    assert hb_pvalue(0.0, 100, 0.1) == pytest.approx(0.9 ** 100)
    assert hb_pvalue(0.2, 100, 0.1) == 1.0          # R^ >= alpha -> no evidence
    F = np.zeros((10, 6))
    assert ltt_threshold(F, 0.1, 0.1) == 0           # 0.9^10 = 0.35 > delta: index 1 fails
    F = np.zeros((100, 6))
    F[:, 3:] = 0.5
    assert ltt_threshold(F, 0.1, 0.1) == 2           # first failure at column 3
    F[:, 2] = 0.03                                   # mean 0.03 < 0.1 still rejected at K = 100
    assert hb_pvalue(0.03, 100, 0.1) <= 0.1
    assert ltt_threshold(F, 0.1, 0.1) == 2
    s = np.stack([F.sum(0), F[:50].sum(0)])
    np.testing.assert_array_equal(ltt_threshold_from_sums(s, np.array([100, 50]), 0.1, 0.1),
                                  [ltt_threshold(F, 0.1), ltt_threshold(F[:50], 0.1)])


def _inc(i, scores, y, **kw):
    return Incident(f"i{i}", np.asarray(scores, float), np.asarray(y, np.int8), **kw)


def test_loss_tables_empty_allnull_amb_and_lenient():
    s0 = np.random.default_rng(0).standard_normal(5000)
    incs = [
        _inc(0, [9.0, 8.0, 0.1], [1, 1, 0]),                # clean
        _inc(1, [9.0, 8.5, 8.0], [0, 0, 0]),                # all null, high scores
        _inc(2, [9.0, 8.5], [-1, 1]),                       # AMB on top
        _inc(3, [-5.0, -6.0], [0, 0]),                      # never selected
    ]
    bank = build_loss_tables(incs, s0)
    fs, fl, Lm = bank.t("fdp_strict"), bank.t("fdp_lenient"), bank.t("Lmono")
    R = bank.t("R")
    assert np.all(R[:, 0] == 0) and np.all(fs[:, 0] == 0)
    assert np.all(np.diff(R, axis=1) >= 0) and np.all(np.diff(Lm, axis=1) >= 0)
    assert np.all(fl <= fs + 1e-12)                      # FDP_lenient <= FDP_strict
    assert fs[1].max() == 1.0 and np.all(R[3] == 0)
    j = int(np.argmax(R[2] == 2))
    assert fs[2, j] == 0.5 and fl[2, j] == 0.0          # AMB counted false only in strict
    # the gathered counts agree with brute-force recounts on the sorted candidates
    for k in range(bank.n):
        y = bank.labels(k)
        for j in range(R.shape[1]):
            r = R[k, j]
            assert bank.t("V")[k, j] == np.sum(y[:r] != 1)
            assert bank.t("TP")[k, j] == np.sum(y[:r] == 1)
    # fdp_curve / monotonize helpers
    np.testing.assert_allclose(fdp_curve([np.array([0, 1]), np.array([], int)], np.array([1, -1])), [0.5, 0.0])
    np.testing.assert_allclose(fdp_curve([np.array([0, 1])], np.array([1, -1]), strict=False), [0.0])
    np.testing.assert_allclose(monotonize([0.0, 0.2, 0.1, 0.3]), [0.0, 0.2, 0.2, 0.3])


def test_bank_save_load_resume_and_subset(tmp_path):
    s0 = np.random.default_rng(0).standard_normal(2000)
    incs = [_inc(i, np.random.default_rng(i).standard_normal(20) + 3 * (np.arange(20) < 2),
                 (np.arange(20) < 2).astype(np.int8), strata={"grp": i % 2}) for i in range(6)]
    out = tmp_path / "bank"
    b1 = build_loss_tables(incs, s0, out)
    b2 = build_loss_tables(incs, s0, out)               # resumes (same input hash)
    assert b2.meta["input_hash"] == b1.meta["input_hash"]
    for k in b1.tables:
        np.testing.assert_array_equal(b1.tables[k], b2.tables[k])
    with pytest.raises(FileExistsError):
        build_loss_tables(incs[:3], s0, out)            # different inputs, no overwrite
    b3 = build_loss_tables(incs[:3], s0, out, overwrite=True)
    assert b3.n == 3
    sub = b1.subset([4, 4, 0])
    np.testing.assert_array_equal(sub.t("R")[0], b1.t("R")[4])
    np.testing.assert_array_equal(sub.scores(1), b1.scores(4))
    np.testing.assert_array_equal(sub.counts(np.array([2]), np.array([3]))["TP"], b1.counts(np.array([0]), np.array([3]))["TP"])
