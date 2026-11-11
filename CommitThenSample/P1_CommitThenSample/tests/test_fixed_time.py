"""rev1 (review R1-M3): valid FIXED-TIME comparators (estimation/cs.py ft_*, bernstein_*, exact_pb_bounds and the
engine columns of MCSpec(fixed_time=True)).

* closed forms = cs.py's machinery with a one-point grid (K = 1, w = 1): lcs_radius / LowerCS, ucs_radius / UpperCS;
  eta* = geometric centre of the LCS grid;
* edge cases (V = 0, no retained hit, W = 0, Qhat = 0, empty eligible set) and monotonicity;
* exact Poisson-binomial bounds = binomial closed forms when all pi are equal; valid under the least favourable
  placement of the hits;
* MC coverage >= 1 - alpha_side per side under an adaptive design (boost + realised-cost controller) with
  heavy-tailed counts, and under withholding (X := 0); a sharp single-eta check at alpha/2 = 0.45 (non-vacuous);
* MCSpec(fixed_time=True) leaves every pre-existing output column bit-identical.
"""
import math

import numpy as np
import pytest
from scipy import stats

from ctsample.estimation import cs
from ctsample.sim.montecarlo import Estimand, MCSpec, calibrated_design, prepare, run_mc
from ctsample.sim.synth import QuerySpec, synth_stream


def _se(p, n):
    return math.sqrt(p * (1 - p) / n)


# --------------------------------------------------------------------------------------------- #
# closed forms vs the K = 1 machinery
# --------------------------------------------------------------------------------------------- #
def test_closed_forms_equal_cs_machinery_with_one_point_grid():
    rng = np.random.default_rng(0)
    a = 0.025
    lt = math.log(1 / a)
    n = 500
    V_ref = rng.lognormal(5, 2, n)
    V = rng.lognormal(5, 2, n)
    Qh = rng.lognormal(3, 1, n)
    eta = cs.ft_eta_lower(V_ref, a)
    etas, _ = cs.lcs_grid(V_ref, a, K=16)
    np.testing.assert_allclose(np.exp(np.mean(np.log(etas), axis=-1)), eta, rtol=1e-12)     # grid centre
    np.testing.assert_allclose(cs.ft_lower(Qh, V, eta, a), Qh - (lt / eta + eta * V / 2), rtol=1e-13)
    r = cs.lcs_radius(V, eta[:, None], np.zeros(1), lt)
    np.testing.assert_allclose(Qh - r, cs.ft_lower(Qh, V, eta, a), rtol=1e-12, atol=1e-9)
    lc = cs.LowerCS(eta[:, None], np.ones(1), a, shape=(n,))
    lc.update_sums(Qh, V)
    np.testing.assert_allclose(lc.current(), cs.ft_lower(Qh, V, eta, a), rtol=1e-12, atol=1e-9)
    # the miss flag is the e-process at the truth (Markov threshold 1/alpha)
    Q = Qh - rng.normal(0, 1, n) * np.sqrt(V)
    np.testing.assert_array_equal(cs.ft_lower_miss(Qh, V, Q, eta, a), lc.miss_at(Q))
    # upper: Fan with a single eta < 1/cbar
    cbar = 57.4
    etau = cs.ft_eta_upper(V_ref, cbar, a)
    assert np.all(etau <= 0.999 / cbar + 1e-15) and np.all(etau <= eta + 1e-15)
    small = cs.ft_eta_lower(V_ref, a) < 0.999 / cbar
    np.testing.assert_allclose(etau[small], eta[small])
    W = rng.lognormal(0, 1, n)
    ru = cs.ucs_radius(W[:, None], etau[:, None], np.zeros(1), lt)
    np.testing.assert_allclose(Qh + ru, cs.ft_upper(Qh, W, etau, a), rtol=1e-12)
    # UpperCS with one grid point fed real increments: bound == closed form on the accumulated W
    X = np.where(rng.random((7, 40)) < 0.3, rng.integers(1, 5, (7, 40)) / 0.05, 0.0)
    c = np.full((7, 40), 4.0 / 0.05)
    e1 = np.full((7, 1), 0.5 / 80.0)
    uc = cs.UpperCS(e1, np.ones(1), a, np.full(7, 80.0), shape=(7,))
    uc.update(X, c)
    Wk = cs.ucs_increment(X, c, e1)[:, 0]
    np.testing.assert_allclose(uc.bound(), cs.ft_upper(X.sum(-1), Wk, e1[:, 0], a), rtol=1e-12)
    np.testing.assert_array_equal(cs.ft_upper_miss(X.sum(-1), Wk, X.sum(-1) + 300, e1[:, 0], a),
                                  uc.miss_at(X.sum(-1) + 300))


def test_edge_cases():
    a = 0.025
    lt = math.log(1 / a)
    eta = cs.ft_eta_lower(400.0, a)
    # no retained hit: Qhat = V = 0 -> L = -l/eta < 0, so L+ = max(L, S = 0) = 0; never a miss at Q >= 0
    L0 = cs.ft_lower(0.0, 0.0, eta, a)
    assert L0 == pytest.approx(-lt / eta) and max(L0, 0.0) == 0.0
    assert not cs.ft_lower_miss(0.0, 0.0, 0.0, eta, a) and not cs.ft_lower_miss(0.0, 0.0, 7.0, eta, a)
    # W = 0: U = Qhat + l/eta; the null radius at eta0 = 0.999/cbar is l cbar/0.999, below the UCS null radius by
    # log(1/alpha)/log(1/(alpha w0)) and within 1% of the floor hiding capacity l/(-log(1 - phi0))
    phi0 = 0.0174
    cbar = 1 / phi0
    e0 = cs.ft_eta_upper(400.0, cbar, a)
    assert e0 == pytest.approx(0.999 / cbar)
    r0 = cs.ft_upper(0.0, 0.0, e0, a)
    assert r0 == pytest.approx(lt * cbar / 0.999)
    assert r0 / cs.null_ucs_radius(cbar, a) == pytest.approx(lt / math.log(1 / (a * 0.5)), rel=0.01)
    assert r0 / (lt / -math.log1p(-phi0)) == pytest.approx(1.0, abs=0.012)
    # Bernstein: Qhat = 0 -> L_B = 0; closed-form inversions hit the boundary exactly
    kap = cs.bernstein_floor_kappa(1.0, phi0)
    assert cs.bernstein_lower(0.0, kap, kap, a) == 0.0
    assert cs.bernstein_upper(0.0, kap, 1.0, a) == pytest.approx((math.sqrt(kap * lt / 2) + math.sqrt(kap * lt / 2 + lt / 3)) ** 2)
    Qh = np.array([100.0, 1000.0, 5000.0])
    LB = cs.bernstein_lower(Qh, kap, kap, a)
    np.testing.assert_allclose(Qh - LB, np.sqrt(2 * kap * LB * lt) + kap * lt / 3, rtol=1e-12)
    UB = cs.bernstein_upper(Qh, kap, 1.0, a)
    np.testing.assert_allclose(UB - Qh, np.sqrt(2 * kap * UB * lt) + lt / 3, rtol=1e-12)
    with pytest.raises(ValueError):
        cs.ft_eta_lower(0.0, a)
    with pytest.raises(ValueError):
        cs.ft_eta_upper(10.0, 0.0, a)
    # exact PB: empty eligible set and kmax = 0
    L, U = cs.exact_pb_bounds(np.zeros(0), 0, a)
    assert L.tolist() == [0] and U.tolist() == [0]
    L, U = cs.exact_pb_bounds(np.full(10, 0.5), 0, a)
    assert L.tolist() == [0] and U.tolist() == [5]          # 0.5^5 = 0.031 > 0.025 >= 0.5^6


def test_monotonicity_and_ordering():
    a = 0.025
    eta = cs.ft_eta_lower(300.0, a)
    V = np.linspace(0, 5000, 50)
    assert np.all(np.diff(cs.ft_lower(100.0, V, eta, a)) < 0)
    Qh = np.linspace(0, 500, 50)
    assert np.all(np.diff(cs.ft_lower(Qh, 1000.0, eta, a)) > 0)
    assert np.all(np.diff(cs.ft_upper(100.0, V / 100, 0.01, a)) > 0)
    kap = cs.bernstein_floor_kappa(1.0, 0.02)
    Lb, Ub = cs.bernstein_lower(Qh, kap, kap, a), cs.bernstein_upper(Qh, kap, 1.0, a)
    assert np.all(np.diff(Lb) >= 0) and np.all(np.diff(Ub) > 0) and np.all(Lb <= Qh) and np.all(Ub >= Qh)
    rng = np.random.default_rng(1)
    pi = np.concatenate([np.full(400, 0.02), rng.uniform(0.02, 1.0, 100)])
    L, U = cs.exact_pb_bounds(pi, 60, a)
    assert np.all(np.diff(L) >= 0) and np.all(np.diff(U) >= 0) and np.all(L <= U)
    assert np.all(L[1:] >= np.arange(1, 61))                 # a retained count is a sure lower bound


def test_exact_pb_equals_binomial_when_pi_equal():
    a, p, n = 0.025, 0.0174, 2000
    L, U = cs.exact_pb_bounds(np.full(n, p), 25, a)
    for k in range(26):
        Lb = 0 if k == 0 else next(q for q in range(k, n + 1) if stats.binom.sf(k - 1, q, p) > a)
        Ub = max(q for q in range(n + 1) if stats.binom.cdf(k, q, p) > a)
        assert (L[k], U[k]) == (Lb, Ub), k
    # all eligible units at the floor: one retained hit already proves Q >= 2, two prove Q >= 15
    assert L[1] == 2 and L[2] == 15
    # U(0) = hiding capacity (Thm. 1) of n floor units: largest m with (1 - p)^m > alpha
    assert U[0] == math.floor(math.log(a) / math.log1p(-p) - 1e-12)


@pytest.mark.parametrize("alpha_side", [0.025, 0.3])
def test_exact_pb_valid_under_least_favourable_placement(alpha_side):
    rng = np.random.default_rng(2)
    pi = np.concatenate([np.full(300, 0.05), rng.uniform(0.05, 0.6, 60)])
    order = np.argsort(pi)
    R = 20000
    for Q in (3, 12, 40):
        L, U = cs.exact_pb_bounds(pi, Q, alpha_side)
        for hits, side in ((order[-Q:], "lo"), (order[:Q], "hi"), (rng.choice(len(pi), Q, replace=False), "both")):
            K = (rng.random((R, Q)) < pi[hits]).sum(axis=1)
            lim = alpha_side + 3 * _se(alpha_side, R)
            if side in ("lo", "both"):
                assert (L[K] > Q).mean() <= lim, (Q, side)
            if side in ("hi", "both"):
                assert (U[K] < Q).mean() <= lim, (Q, side)


@pytest.mark.parametrize("alpha_side", [0.025, 0.3])
def test_bernstein_valid_static_poisson_heavy_tailed(alpha_side):
    rng = np.random.default_rng(3)
    n, phi, R = 400, 0.03, 20000
    b = np.ceil(rng.pareto(1.2, n) + 1).clip(max=200)
    f = np.where(rng.random(n) < 0.1, np.ceil(b * rng.random(n)), 0.0)
    pi = np.maximum(phi, np.minimum(1.0, 0.002 * b))
    kap = cs.bernstein_floor_kappa(b.max(), phi)
    Qh = ((rng.random((R, n)) < pi) * (f / pi)).sum(axis=1)
    Q = f.sum()
    lim = alpha_side + 3 * _se(alpha_side, R)
    lo = cs.bernstein_lower_miss(Qh, Q, kap, kap, alpha_side)
    hi = cs.bernstein_upper_miss(Qh, Q, kap, b.max(), alpha_side)
    assert lo.mean() <= lim and hi.mean() <= lim
    np.testing.assert_array_equal(lo, cs.bernstein_lower(Qh, kap, kap, alpha_side) >= Q - 1e-9 * Q)
    np.testing.assert_array_equal(hi, cs.bernstein_upper(Qh, kap, b.max(), alpha_side) <= Q + 1e-9 * Q)


# --------------------------------------------------------------------------------------------- #
# engine: MCSpec(fixed_time=True)
# --------------------------------------------------------------------------------------------- #
FT_LO = ("ft1_lo", "fth_lo", "ftb_lo", "ftx_lo")
FT_HI = ("ft1_hi", "fth_hi", "ftb_hi", "ftx_hi")


def _heavy_stream():
    qs = [QuerySpec("cnt_heavy", "count", elig_frac=0.3, n_hits=300, n_att=3, pareto_alpha=1.1),
          QuerySpec("cnt_sparse", "count", elig_frac=0.3, n_hits=12, n_att=1, pareto_alpha=1.1),
          QuerySpec("ind_common", "indicator", elig_frac=0.3, n_hits=300, n_att=3),
          QuerySpec("ind_sparse", "indicator", elig_frac=0.3, n_hits=8, n_att=2),
          QuerySpec("ind_rare", "indicator", elig_frac=0.02, n_hits=30, n_att=1, rare=True)]
    return synth_stream(E=150, seed=41, base=80, n_devices=30, queries=qs)


def _ests(T):
    # V_ref of a 'training' total 3x too large and horizon-tuned 'fth' centre: both fixed before the run
    return [Estimand(nm, nm, ucs=True, V_ref=None, V_ref_ft=None) for nm in T.query_names]


def test_engine_fixed_time_valid_adaptive_heavy_tailed_and_old_columns_identical():
    T = _heavy_stream()
    d = calibrated_design(T, "ft_adaptive", controller="realised", boost=True)
    ests = _ests(T)
    R = 1500
    mc = MCSpec(reps=R, width_reps=20, stream_id=41, fixed_time=True)
    res = run_mc(T, d, ests, mc, workers=1, chunk=500)
    cov = res["cov"]
    lim = 0.025 + 3 * _se(0.025, R)
    g = cov.groupby("estimand")
    for fl in ("ft1_lo", "fth_lo", "ftb_lo", "ft1_hi", "fth_hi", "ftb_hi"):
        assert (g[fl].mean() <= lim).all(), (fl, g[fl].mean())
    assert not cov["ftx_on"].any()                         # exact PB only for static designs
    assert cov["ftb_on"].all()
    # engine values = closed forms on the reported finals
    P = prepare(T, d, ests, mc)
    k = cov["est"].to_numpy()
    np.testing.assert_allclose(cov["Lft1"], cs.ft_lower(cov["Qhat"], cov["V"], P.eta_ft_l[k, 0], P.alpha_side[k]), rtol=1e-12)
    np.testing.assert_array_equal(cov["ft1_lo"], cs.ft_lower_miss(cov["Qhat"], cov["V"], cov["Q"], P.eta_ft_l[k, 0],
                                                                  P.alpha_side[k]))
    np.testing.assert_allclose(cov["Lftb"], cs.bernstein_lower(cov["Qhat"], P.ftb_kappa[k], P.ftb_kappa[k], P.alpha_side[k]))
    assert (cov["Lft1"] <= cov["Qhat"]).all() and (cov["Uft1"] >= cov["Qhat"]).all()
    # every pre-existing column (cov, width, ctrl) is bit-identical without the fixed-time block
    res0 = run_mc(T, d, ests, MCSpec(reps=R, width_reps=20, stream_id=41), workers=1, chunk=500)
    for nm in ("cov", "width", "ctrl"):
        A, B = res0[nm], res[nm]
        for c in A.columns:
            x, y = A[c].to_numpy(), B[c].to_numpy()
            assert (np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f" else np.array_equal(x, y)), (nm, c)


def test_engine_fixed_time_static_exact_pb_and_withholding():
    T = _heavy_stream()
    d = calibrated_design(T, "ft_static")
    ests = _ests(T)
    R = 1500
    mc = MCSpec(reps=R, width_reps=0, stream_id=42, fixed_time=True, withhold=("void_unretained", "withhold_half"))
    cov = run_mc(T, d, ests, mc, workers=1, chunk=500)["cov"]
    P = prepare(T, d, ests, mc)
    lim = 0.025 + 3 * _se(0.025, R)
    for var in ("honest", "x0:void_unretained", "x0:withhold_half"):
        g = cov[cov["variant"] == var].groupby("estimand")
        for fl in FT_LO + FT_HI:
            assert (g[fl].mean() <= lim).all(), (var, fl)
    h = cov[cov["variant"] == "honest"]
    ind = h[h["estimand"].str.startswith("ind")]
    assert ind["ftx_on"].all() and not h[h["estimand"].str.startswith("cnt")]["ftx_on"].any()
    for nm, gg in ind.groupby("estimand"):
        k = int(gg["est"].iloc[0])
        Lt, Ut = P.ftx[k]
        K = np.rint(gg["S"].to_numpy()).astype(int)
        np.testing.assert_array_equal(gg["Lftx"].to_numpy(), Lt[K])
        np.testing.assert_array_equal(gg["Uftx"].to_numpy(), Ut[K])
        assert (gg["Lftx"] >= gg["S"]).all()
    # upper statements follow the UCS rules: never issued after a withheld eligible unit
    x = cov[cov["variant"] == "x0:withhold_half"]
    assert x.loc[~x["ucs_issued"].astype(bool), ["Uft1", "Ufth", "Uftb", "Uftx"]].isna().all().all()
    assert (x.loc[~x["ucs_issued"].astype(bool), FT_HI] == False).all().all()  # noqa: E712
    # the (invalid) 'compliant only' variants get the fixed-time statistics too (measured in E2 rev1 A11)
    comp = cov[cov["variant"] == "comp:void_unretained"]
    assert comp[["Lft1", "Lfth", "Lftb"]].notna().all().all()


def test_engine_fixed_time_sharp_single_eta():
    """alpha/2 = 0.45: Markov is far from tight, but the fixed-time bounds must miss non-trivially and <= 0.45."""
    qs = [QuerySpec("dense", "indicator", elig_frac=1.0, n_hits=8000, n_att=40),
          QuerySpec("mid", "indicator", elig_frac=0.5, n_hits=1500, n_att=10)]
    T = synth_stream(E=120, seed=4, base=120, n_devices=50, zero_payload_frac=0.0, queries=qs)
    R = 1500
    lim = 0.45 + 3 * _se(0.45, R)
    for kw in (dict(controller="realised", boost=True), dict()):
        d = calibrated_design(T, "sharp_" + str(bool(kw)), rho=0.05, gamma=0.3, **kw)
        mc = MCSpec(reps=R, alpha=0.9, width_reps=0, stream_id=4, fixed_time=True)
        cov = run_mc(T, d, [Estimand(n, n) for n in T.query_names], mc, workers=1, chunk=500)["cov"]
        g = cov.groupby("estimand")
        for fl in ("ft1_lo", "ft1_hi", "fth_lo", "ftb_lo", "ftb_hi") + (() if kw else ("ftx_lo", "ftx_hi")):
            assert (g[fl].mean() <= lim).all(), (kw, fl, g[fl].mean())
        assert g["ft1_lo"].mean()["dense"] > 0.02 and g["ft1_hi"].mean()["dense"] > 0.02
