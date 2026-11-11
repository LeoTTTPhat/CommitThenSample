"""DESIGN §7.4 test_controller: synthetic diurnal + bursty traffic over 2 days, spend/target within +-1%
for both modes (scripts/checks/ctrl.py: 0.9992 expected, 1.0001 realised, max|D|/Bbar ~ 77);
the revision-1 recursion drifts (regression guard, scripts/checks/reviewer_chk.py)."""
import math
import os

import numpy as np
import pytest

from ctsample.design.controller import HorizonController, Rev1CarryController, calibrate_lambda, calibrate_design
from ctsample.design.probs import pi_A
from ctsample.sim.synth import synth_traffic

RHO, GAMMA = 0.05, 0.3
SLOW = pytest.mark.skipif(os.environ.get("CTS_SLOW") != "1", reason="full-size run: set CTS_SLOW=1")


def _run(seed: int, realised: bool, E: int = 2 * 2880, Wc: float = 30.0):
    """One ctrl.py-style run: lambda_ref calibrated on every 7th epoch, A-component design with floor
    phi0 = gamma*rho (all units payload), log-integral controller with saturation."""
    tr = synth_traffic(E, seed=seed, base=200.0, n_devices=1)
    s, a, ep = tr["s"], tr["a"], tr["epoch"]
    ptr = np.searchsorted(ep, np.arange(E + 1))
    Bbar = RHO * s.sum() / E
    phi0 = GAMMA * RHO
    train = (ep % 7) == 0
    lam_ref = calibrate_lambda(s[train], a[train], Bbar, phi0, n_epochs=int(np.ceil(E / 7)), component="A")
    ctrl = HorizonController(lam_ref, Bbar, kappa=1.0, Wc=Wc, L=math.log(16), mode="realised" if realised else "expected")
    rng = np.random.default_rng(10_000 + seed)
    for e in range(E):
        lo, hi = ptr[e], ptr[e + 1]
        p = pi_A(ctrl.lam(), a[lo:hi], s[lo:hi], phi0)
        c = float(np.sum((rng.random(hi - lo) < p) * s[lo:hi])) if realised else float(np.sum(p * s[lo:hi]))
        ctrl.update(c)
    return float(ctrl.spend_ratio()), float(ctrl.max_abs_D / Bbar), float(abs(ctrl.D) / Bbar)


def _check(R: int):
    out = {}
    for realised in (False, True):
        res = np.array([_run(1 + r, realised) for r in range(R)])
        out[realised] = res
        assert np.all(np.abs(res[:, 0] - 1) <= 0.01), res[:, 0]            # every run within +-1%
        assert abs(res[:, 0].mean() - 1) <= 0.005
        assert 20 <= res[:, 1].mean() <= 150                                # ctrl.py: ~77
    return out


def test_controller_tracks_two_days_fast():
    _check(R=3)


@pytest.mark.slow
@SLOW
def test_controller_tracks_two_days_full():
    out = _check(R=20)
    for realised, res in out.items():
        print(f"realised={realised}: spend/target mean={res[:, 0].mean():.4f} "
              f"[5,95]%={np.percentile(res[:, 0], [5, 95]).round(4)} max|D|/Bbar={res[:, 1].mean():.1f} "
              f"final|D|/Bbar={res[:, 2].mean():.1f}")


def test_rev1_recursion_drifts_regression_guard():
    """B_e = Bbar + clip(D, +-Bbar/2) spent in expectation; D <- D + B_e - cost is driftless."""
    R, E = 400, 3000
    rng = np.random.default_rng(0)
    res = {}
    for corrected in (False, True):
        c = Rev1CarryController(1.0, shape=(R,), corrected=corrected)
        for _ in range(E):
            B = c.budget()
            c.update(B * rng.lognormal(-0.5 * 0.8 ** 2, 0.8, R), B)
        res[corrected] = c.spend_ratio()
    lo, hi = np.percentile(res[False], [5, 95])
    assert hi - lo > 0.3                      # reviewer check: [0.51, 1.45]
    lo2, hi2 = np.percentile(res[True], [5, 95])
    assert lo2 > 0.99 and hi2 < 1.01          # reviewer check: [0.9993, 1.0014]


def test_lambda_fixed_point_and_saturation():
    c = HorizonController(np.array([3.25, 0.0005]), 100.0, shape=(3,))
    c.D = np.array([0.0, -1e9, 1e9])
    lam = c.lam()
    assert lam.shape == (3, 2)
    assert np.allclose(lam[1] / np.array([3.25, 0.0005]), 1 / 16, rtol=1e-9)
    assert np.allclose(lam[2] / np.array([3.25, 0.0005]), 16, rtol=1e-9)
    Lam = c.Lam_fix()
    assert Lam.dtype == np.int64 and np.all(np.abs(Lam / 2.0 ** 40 - lam) <= 2.0 ** -40)
    s = HorizonController(0.37, 10.0)
    assert isinstance(s.Lam_fix(), int) and s.Lam_fix() == round(0.37 * 2 ** 40)
    s.update(10.0 + 2000.0 * 10)      # overspend by 2000 Bbar -> guard
    assert s.flood_guard_active() and s.D == pytest.approx(-20000.0)


def test_calibrate_design_each_component_spends_bbar():
    rng = np.random.default_rng(3)
    s = rng.lognormal(6, 1.6, 20000) + 80
    a = rng.lognormal(0, 1.5, 20000) * (rng.random(20000) < 0.2)
    E = 100
    Bbar = 0.05 * s.sum() / E
    lam = calibrate_design("mixture", s, a, Bbar, 0.015, E)
    from ctsample.design.probs import pi_S
    assert np.sum(pi_A(lam[0], a, s, 0.015) * s) / E == pytest.approx(Bbar, rel=1e-6)
    assert np.sum(pi_S(lam[1], s, 0.015) * s) / E == pytest.approx(Bbar, rel=1e-6)
    with pytest.raises(ValueError):
        calibrate_lambda(s, a, Bbar, 0.9, E)    # floor alone overspends
