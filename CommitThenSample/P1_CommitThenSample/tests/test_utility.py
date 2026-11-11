"""design/utility.py: integer aggregated utility (exact vs rational arithmetic, close to the float twin),
metadata eligibility == P1-B's detector eligibility, shrinkage tables, calibrated class-balanced forests,
bundle round trip."""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from ctsample.design import utility as U
from ctsample.units import encode as enc
from ctsample.units import features as F

from test_features import synth_records


def test_a_fix_is_exact_floor_sqrt():
    rng = np.random.default_rng(1)
    N, Q = 300, 6
    P = rng.integers(0, 65536, (N, Q))
    m = np.array([1.0, 1.0, 37.5, 1.0, 1234.25, 3.0])
    M = np.repeat(U.m_fix(m)[:, None], U.N_CELLS, axis=1)
    cell = rng.integers(0, U.N_CELLS, N)
    G = U.ghat_int(P, M, cell)
    Qt = np.array([0.004, 0.37, 12.5, 1.0, 55.0, 0.0002])
    W = U.equal_weights(Q)
    Q2 = U.q2_fix(Qt)
    a = U.a_fix_int(G, W, Q2)
    for i in range(0, N, 7):
        S = sum((W[j] * int(G[i, j]) << 32) // Q2[j] for j in range(Q))
        assert a[i] == min(2 ** 32 - 1, math.isqrt(S))
    # close to the float twin (per-term floors and fixed-point roundings only)
    af = U.utility_float(P / 65536.0, np.repeat(m[:, None], U.N_CELLS, axis=1), cell, np.array(W) / 65536.0, Qt)
    ai = (U.utility_int(a) / 65536.0)
    ok = af > 1e-3
    assert np.max(np.abs(ai[ok] / af[ok] - 1)) < 1e-3


def test_meta_eligibility_equals_detectors():
    from ctsample.content.detectors import DET_ESTIMANDS, eligibility
    _, metas = synth_records(2500, seed=3)
    c = F.cols_from_meta(metas)
    dicts = [enc.dec_metadata(m) for m in metas]
    for e in DET_ESTIMANDS:
        mine = U.meta_eligible(e.id, c)
        ref = np.array([eligibility(d["proto"], d["port_a"], d["port_b"], d["pay"], d.get("fps", False),
                                    d["pkts"][0], d["pkts"][1])[e.id][0] for d in dicts])
        assert np.array_equal(mine, ref), e.id
    assert np.array_equal(U.meta_eligible("S3", c), c["pay"].astype(bool))


def test_shrinkage_tables():
    cell = np.array([0, 0, 0, 1, 1, 2])
    f = np.array([2, 4, 0, 10, 0, 0])
    t = U.second_moment_table(cell, f, "count")
    mu = (4 + 16 + 100) / 3
    assert t["global"] == pytest.approx(mu)
    assert t["m"][0] == pytest.approx((4 + 16 + 50 * mu) / (2 + 50))
    assert t["m"][1] == pytest.approx((100 + 50 * mu) / 51)
    assert t["m"][5] == pytest.approx(mu)
    assert np.all(U.second_moment_table(cell, f, "indicator")["m"] == 1)
    st = U.stratum_table(np.array([0, 0, 1, 1, 1]), np.array([1, 0, 1, 0, 0]), np.array([1, 1, 1, 1, 0]))
    rate = 2 / 4
    assert st["p"][0] == pytest.approx((1 + 50 * rate) / (2 + 50))
    assert st["p"][1] == pytest.approx((1 + 50 * rate) / (2 + 50))
    assert st["p"][7] == pytest.approx(rate)


@pytest.fixture(scope="module")
def trained():
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(5)
    ep, metas = synth_records(12000, n_dev=6, n_epochs=200, seed=11)
    c = F.cols_from_meta(metas)
    X = F.features_batch(c, ep)
    # synthetic target: depends on the dport bucket and payload size (rate ~ 3%)
    z = -4.0 + 2.0 * (X[:, 1] == 0) + 0.02 * X[:, 6] + 0.5 * (X[:, 0] == 0)
    y = (rng.random(len(z)) < 1 / (1 + np.exp(-z))).astype(np.int8)
    f, bst, info = U.train_forest(X, y, seed=0, threads=4, n_trees=60)
    return c, X, y, f, bst, info


def test_balanced_forest_is_calibrated(trained):
    c, X, y, f, bst, info = trained
    assert info["model"] == "lgbm" and info["n_pos"] == int(y.sum())
    p = f.predict_int(X) / 65536.0
    assert abs(p.mean() - y.mean()) < 0.35 * y.mean() + 0.003        # prior shift -> calibrated scale
    pf = f.predict_float(X)
    assert np.max(np.abs(p - pf)) <= 2 * 0.25 / 256 + 2 ** -15
    g, _, inf2 = U.train_forest(X, np.zeros(len(y), np.int8), seed=0, threads=2)
    assert inf2["model"] == "constant" and g.n_trees == 0


def test_bundle_roundtrip_and_utilities(trained, tmp_path):
    c, X, y, f, bst, info = trained
    qs = ["D5", "D6", "S6", "L1"]
    M = {q: U.m_fix(np.full(U.N_CELLS, 1.0 if U.kind_of(q) == "indicator" else 7.5)) for q in qs}
    P8 = {q: U.p_fix(np.linspace(0, 0.2, U.N_STRATA)) for q in qs}
    b = U.UtilityBundle(queries=qs, hit={q: f for q in qs}, gmal=f, M=M, P8=P8,
                        m_float={q: M[q] / 65536 for q in qs}, p8_float={q: P8[q] / 65536 for q in qs})
    b.save(tmp_path)
    for q in qs:
        (tmp_path / f"hit_{q}.lgb.txt").write_text(bst.model_to_string())
    (tmp_path / "gmal.lgb.txt").write_text(bst.model_to_string())
    b2 = U.UtilityBundle.load(tmp_path)
    b3 = U.UtilityBundle.load(tmp_path, attach_lgbm=True)
    assert b2.theta_digest() == b.theta_digest() == b3.theta_digest()
    W = U.equal_weights(len(qs))
    Q2 = U.q2_fix([3.0, 40.0, 0.05, 0.5])
    rng = np.random.default_rng(0)
    ftrue = {q: np.where(c["pay"], rng.integers(0, 3, len(c["pay"])), 0) for q in qs}
    u1 = b.utilities(c, X, qs, W, Q2, f_true=ftrue, which=("U3", "U5", "U8", "U7"))
    u2 = b2.utilities(c, X, qs, W, Q2, f_true=ftrue, which=("U3", "U5", "U8", "U7"))
    u3 = b3.utilities(c, X, qs, W, Q2, f_true=ftrue, which=("U3", "U5", "U8", "U7"))
    for k in ("U3", "U5", "U8", "U7"):
        assert np.array_equal(u1[k], u2[k]) and np.array_equal(u1[k], u3[k]), k
        assert np.all(u1[k][~c["pay"]] == 0) and np.all(u1[k][c["pay"]] >= 1), k
    # U7 is monotone in the true second moment (same eligibility pattern)
    el = U.eligibility_matrix(qs, c)
    s2 = sum(np.where(el[:, j], ftrue[q] ** 2, 0) * W[j] * 2 ** 32 / Q2[j] for j, q in enumerate(qs))
    pay = c["pay"].astype(bool)
    assert np.all(np.diff(u1["U7"][pay][np.argsort(s2[pay], kind="stable")]) >= 0)


def test_adaptive_lambda_reduces_overspend():
    """Boost replay: the adaptive λ^A spends B̄ with the boost active; the static λ^A overspends."""
    from ctsample.design.controller import calibrate_lambda
    rng = np.random.default_rng(7)
    E, n_e, n_dev = 150, 40, 8
    ep = np.repeat(np.arange(E), n_e)
    s = rng.lognormal(6, 1, E * n_e).round() + 60
    a = rng.gamma(0.5, 1.0, E * n_e) + 1e-3
    dev = rng.integers(0, n_dev, E * n_e)
    trig = rng.random(E * n_e) < 0.03
    cfgs = []
    for rho, tau in ((0.05, 0.2), (0.02, 0.0)):
        phi0 = 0.3 * rho
        Bbar = rho * s.sum() / E
        lam_s = calibrate_lambda(s, a, Bbar, phi0, E, component="A")
        lam_S = calibrate_lambda(s, a, Bbar, phi0, E, component="S")
        cfgs.append({"lam_A": lam_s, "lam_S": lam_S if tau else None, "tau_mix": tau, "phi0": phi0, "Bbar": Bbar})
    res = U.calibrate_lambda_adaptive(s, a, dev, ep, trig, E, cfgs, R=4, seed=1)
    for (lam_a, info), cfg in zip(res, cfgs):
        assert lam_a < cfg["lam_A"]
        assert info["cost_static_c1"] > cfg["Bbar"] * 1.01               # static λ overspends under the boost
        lo, hi = info["cost_bracket"]
        assert lo <= cfg["Bbar"] <= hi and (hi - lo) / cfg["Bbar"] < 0.05


def test_dedup_path_equals_direct(trained):
    c, X, y, f, bst, info = trained
    qs = ["D5", "D6", "S6", "L1"]
    M = {q: U.m_fix(np.full(U.N_CELLS, 1.0 if U.kind_of(q) == "indicator" else 7.5)) for q in qs}
    P8 = {q: U.p_fix(np.linspace(0, 0.2, U.N_STRATA)) for q in qs}
    b = U.UtilityBundle(queries=qs, hit={q: f for q in qs}, gmal=f, M=M, P8=P8)
    pay = c["pay"].astype(bool)
    idx = np.nonzero(pay)[0]
    cp = {k: v[idx] for k, v in c.items()}
    Xp = np.concatenate([X[idx], X[idx][:500]])                 # repeated rows exercise the dedupe
    cp = {k: np.concatenate([v, v[:500]]) for k, v in cp.items()}
    active = ["D6", "S6", "L1"]
    W, Q2 = U.equal_weights(len(active)), U.q2_fix([40.0, 0.05, 0.5])
    rng = np.random.default_rng(1)
    ft = {q: rng.integers(0, 3, len(Xp)) for q in active}
    direct = b.utilities(cp, Xp, active, W, Q2, f_true=ft, which=("U3", "U5", "U8", "U7"))
    P_all, pmal = U.predict_all(b, Xp, U.eligibility_matrix(b.queries, cp))
    fast = U.utilities_from_P(b, cp, P_all, pmal, active, W, Q2, f_true=ft)
    for k in ("U3", "U5", "U8", "U7"):
        assert np.array_equal(direct[k], fast[k]), k
