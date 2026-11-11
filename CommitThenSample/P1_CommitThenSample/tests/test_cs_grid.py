"""DESIGN §7.4 test_cs_grid: UCS grid requires eta < 1/cbar; LCS inversion monotone; grid layout;
inversion <-> e-process consistency; supermartingale increments (scripts/checks/check1.py)."""
import math

import numpy as np
import pytest

from ctsample.estimation.cs import (LowerCS, NormalMixtureLowerCS, UpperCS, lcs_grid, lcs_log_eproc, lcs_radius,
                                    null_ucs_radius, psi, retro_tighten, sure_bounds, ucs_grid, ucs_log_eproc,
                                    ucs_radius)

A2 = 0.025


def test_lcs_grid_layout():
    etas, w = lcs_grid(1e4, A2)
    eta_star = math.sqrt(2 * math.log(40) / 1e4)
    assert etas.shape == (16,) and np.allclose(w, 1 / 16)
    assert etas[0] == pytest.approx(eta_star / 100) and etas[-1] == pytest.approx(eta_star * 100)
    assert np.allclose(np.diff(np.log10(etas)), 4 / 15)
    e2, _ = lcs_grid(np.array([1e2, 1e4]), A2, K=32, decades=4)
    assert e2.shape == (2, 32)


def test_ucs_grid_layout_and_eta_bound():
    cbar = 1 / 0.015
    etas, w = ucs_grid(cbar)
    assert len(etas) == 16 and w[0] == 0.5 and np.allclose(w[1:], 1 / 30) and np.isclose(w.sum(), 1)
    assert etas[0] == pytest.approx(0.999 / cbar)
    assert np.all(etas * cbar < 1)
    assert etas[-1] == pytest.approx(etas[0] * 1e-4)
    with pytest.raises(ValueError):
        UpperCS(np.array([1.0 / cbar]), np.array([1.0]), A2, cbar)       # eta = 1/cbar not allowed
    with pytest.raises(ValueError):
        ucs_grid(cbar, eta0_frac=1.0)


def test_psi_series_and_limits():
    lam = np.array([0.0, 1e-6, 1e-4 * 0.99, 1e-4 * 1.01, 0.5, 0.999, 1.0, 2.0])
    p = psi(lam)
    exact = -np.log1p(-lam[:6]) - lam[:6]
    assert np.allclose(p[:6], exact, rtol=1e-9, atol=1e-300)
    assert np.isinf(p[6]) and np.isinf(p[7])


@pytest.mark.parametrize("V", [0.0, 1.0, 50.0, 1e4, 1e7])
def test_lcs_inversion_is_the_crossing_point(V):
    etas, w = lcs_grid(1e4, A2)
    lw, thr = np.log(w), math.log(1 / A2)
    r = float(lcs_radius(V, etas, lw, thr))
    assert lcs_log_eproc(r, V, etas, lw) >= thr
    assert lcs_log_eproc(r * (1 - 1e-9) - 1e-12, V, etas, lw) < thr


def test_lcs_inversion_monotone():
    etas, w = lcs_grid(1e4, A2)
    Vs = np.geomspace(1e-2, 1e8, 200)
    r = lcs_radius(Vs, etas, np.log(w), math.log(1 / A2))
    assert np.all(np.diff(r) >= -1e-9)
    # e-process monotone in r (so inversion is well defined)
    rs = np.linspace(0, 5 * r[100], 500)
    m = lcs_log_eproc(rs, np.full_like(rs, Vs[100]), etas, np.log(w))
    assert np.all(np.diff(m) > 0)
    # L = Qhat - r^- decreases when V grows (fixed Qhat), and the running max is monotone
    cs = LowerCS(etas, w, A2)
    Ls = []
    for _ in range(50):
        cs.update(np.array([3.0, 0.0, 10.0]))
        Ls.append(cs.bound())
    assert np.all(np.diff(Ls) >= 0)


def test_ucs_inversion_and_null_slack():
    cbar = 1 / 0.015
    etas, w = ucs_grid(cbar)
    lw, thr = np.log(w), math.log(1 / A2)
    W = np.zeros(16)
    r = float(ucs_radius(W, etas, lw, thr))
    assert ucs_log_eproc(r, W, etas, lw) >= thr > ucs_log_eproc(r * (1 - 1e-9), W, etas, lw)
    bound = cbar * math.log(1 / (A2 * 0.5)) / 0.999
    assert r <= bound + 1e-9 and r >= 0.95 * bound
    assert null_ucs_radius(cbar, A2) == pytest.approx(r)
    # W larger -> radius larger (monotone)
    rs = [float(ucs_radius(W + k, etas, lw, thr)) for k in np.linspace(0, 20, 21)]
    assert np.all(np.diff(rs) >= 0)


def test_vectorised_matches_scalar():
    rng = np.random.default_rng(0)
    etas, w = lcs_grid(1e3, A2)
    cs = LowerCS(etas, w, A2, shape=(4, 3))
    single = [[LowerCS(etas, w, A2) for _ in range(3)] for _ in range(4)]
    for _ in range(20):
        X = rng.exponential(5, (4, 3, 7)) * (rng.random((4, 3, 7)) < 0.2)
        cs.update(X)
        for i in range(4):
            for j in range(3):
                single[i][j].update(X[i, j])
    B = cs.bound()
    for i in range(4):
        for j in range(3):
            assert B[i, j] == pytest.approx(single[i][j].bound())


def test_eproc_check_equals_inversion():
    """miss <=> L > Q, checked on random states (DESIGN §6: Q < L <=> M-(Qhat - Q) >= 2/alpha)."""
    rng = np.random.default_rng(1)
    etas, w = lcs_grid(500.0, A2)
    cs = LowerCS(etas, w, A2, shape=(2000,))
    X = rng.exponential(10, (2000, 30)) * (rng.random((2000, 30)) < 0.3)
    cs.update(X)
    L = cs.current()
    Q = cs.Qhat - rng.uniform(0, 2, 2000) * (cs.Qhat - L)
    miss = cs.miss_at(Q)
    agree = miss == (Q < L)
    near = np.abs(Q - L) < 1e-6 * np.maximum(1, np.abs(L))
    assert np.all(agree | near)
    etas_u, wu = ucs_grid(20.0)
    ucs = UpperCS(etas_u, wu, A2, 20.0, shape=(2000,))
    c = np.full((2000, 30), 20.0)
    ucs.update(np.minimum(X, 20.0), c)
    U = ucs.bound()
    Q2 = ucs.Qhat + rng.uniform(0, 2, 2000) * (U - ucs.Qhat)
    m2 = ucs.miss_at(Q2)
    near2 = np.abs(Q2 - U) < 1e-6 * np.maximum(1, U)
    assert np.all((m2 == (Q2 > U)) | near2)


def test_supermartingale_increments_check1():
    """E[exp(eta(X-f) - eta^2 X^2/2)] <= 1 and the Fan bound for the UCS (scripts/checks/check1.py)."""
    pis = np.geomspace(1e-4, 1, 40)[:, None, None]
    fs = np.geomspace(1e-3, 1e4, 40)[None, :, None]
    etas = np.geomspace(1e-6, 1e3, 60)[None, None, :]
    x = fs / pis
    with np.errstate(over="ignore", under="ignore"):
        E = pis * np.exp(etas * (x - fs) - etas ** 2 * x ** 2 / 2) + (1 - pis) * np.exp(-etas * fs)
    assert E.max() <= 1 + 1e-12
    pis = np.geomspace(1e-4, 1, 40)[:, None, None]
    fs = np.linspace(0, 1, 21)[None, :, None]
    lam = np.linspace(1e-4, 0.999, 60)[None, None, :]
    c = 1.0 / pis
    eta = lam / c
    x = fs / pis
    E2 = pis * np.exp(eta * (fs - x) - psi(lam) * x ** 2 / c ** 2) + (1 - pis) * np.exp(eta * fs)
    assert E2.max() <= 1 + 1e-12


def test_ucs_void_on_cbar_violation():
    etas, w = ucs_grid(10.0)
    u = UpperCS(etas, w, A2, 10.0)
    u.update(np.array([0.0, 0.0]), np.array([5.0, 20.0]))     # an eligible unit with c > cbar
    assert u.void and np.isnan(u.bound())


def test_sure_bounds_and_retro():
    Z = np.array([1, 0, 1, 1, 0], bool)
    f = np.array([1.0, 1.0, 0.0, 2.0, 0.0])
    b = np.array([1.0, 1.0, 1.0, 3.0, 2.0])
    W = np.array([0, 0, 0, 1, 0], bool)
    S, D = sure_bounds(Z, f, b, W)
    assert S == 1.0 and D == 1.0 + 1.0 + 0.0 + 3.0 + 2.0
    assert S <= f.sum() <= D
    U = np.array([10.0, 8.0, np.nan, 9.0, 7.0])
    assert np.allclose(retro_tighten(U), [7, 7, 7, 7, 7])


def test_normal_mixture_lcs_valid_quick():
    """[SHOULD] alternative boundary: valid (miss <= alpha_side + 3 SE) on a dense population."""
    rng = np.random.default_rng(3)
    R, E, n, pi = 4000, 50, 20, 0.05
    a_side = 0.25
    cs = NormalMixtureLowerCS(V_ref=E * n / pi, alpha_side=a_side, shape=(R,))
    for _ in range(E):
        Z = rng.random((R, n)) < pi
        cs.update(Z / pi, f_true_epoch_sum=float(n))
    rate = cs.miss.mean()
    assert rate <= a_side + 3 * math.sqrt(a_side * (1 - a_side) / R)
    assert rate > 0.02      # non-vacuous
