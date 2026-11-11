"""Analytic metrics, design helpers and reporting utilities (DESIGN §3.5, Prop. 3-5, §6 analytic-first, §6.6)."""
import math

import numpy as np
import pytest

from ctsample.config import DEFAULTS_PATH, cfg_hash, load_yaml, src_hash
from ctsample.design.boost import BoostState, boost_set_from_hits
from ctsample.design.floor import (device_floor_cap, equalised_floor, flood_guard_floor, phi0_from_training,
                                   phi_t_star, two_tier_retention, worst_case_retention)
from ctsample.design.minimax import eg_minimax, waterfill
from ctsample.design.probs import design_pi, pi_A, pi_mixture, pi_S, pi_uniform
from ctsample.estimation.incident import (content_evidence_miss, grinding_evidence, group_miss, k95, miss_bound,
                                          stage_coverage, tte_quantile, tte_survival)
from ctsample.estimation.ppi import ppi_total, ppi_variance, ppi_vhat
from ctsample.estimation.wald import NaiveMonitor, wald_interval
from ctsample.sim.analytic import expected_spend, miss_by_k_strata, miss_per_incident, mse_ratios, pi_full, var_ht_queries
from ctsample.sim.metrics import cluster_bootstrap, geometric_mean, wilson
from ctsample.sim.montecarlo import calibrated_design, static_pass
from ctsample.sim.synth import synth_stream


@pytest.fixture(scope="module")
def stream():
    return synth_stream(E=200, seed=41, base=80)


def test_static_pass_spend_and_variance_vs_mc(stream):
    d = calibrated_design(stream, "static")
    sp = static_pass(stream, d)
    pi = pi_full(stream, sp)
    assert np.all(pi[~np.asarray(stream.pay)] == 0)
    assert np.all(pi[np.asarray(stream.pay)] >= d.phi0 * (1 - 1e-12))
    es = expected_spend(stream, pi, d.Bbar)
    assert es["total"] == pytest.approx(sp["cost"].sum())
    # identity: spend/target = 1 - D_E/(Bbar E); on a short trending stream the integral state D_E is
    # O(Wc Bbar), here ~3% of the 200-epoch budget (2-day runs: ~0.1%, test_controller)
    assert es["spend_ratio"] == pytest.approx(1 - sp["D"][-1] / (d.Bbar * stream.n_epochs))
    assert abs(es["spend_ratio"] - 1) < 0.05
    v = var_ht_queries(stream, pi)
    rng = np.random.default_rng(0)
    for q in range(stream.n_queries):
        u, f = stream.hits(q)
        Z = rng.random((20000, len(u))) < pi[u]
        Qh = (Z * f / pi[u]).sum(axis=1)
        assert Qh.var() == pytest.approx(v["var"][q], rel=0.06)
        assert (Qh == 0).mean() == pytest.approx(v["p_no_hit"][q], abs=4 * math.sqrt(0.25 / 20000) + 1e-3)


def test_incident_miss_products(stream):
    pi = pi_full(stream, static_pass(stream, calibrated_design(stream, "static")))
    m = miss_per_incident(stream, pi)
    inc = np.asarray(stream.incident)
    for i, iid in enumerate(m["incident"][:3]):
        sel = inc == iid
        assert m["presence_miss"][i] == pytest.approx(np.prod(1 - pi[sel]))
    strata = miss_by_k_strata(m)
    assert sum(v["n"] for v in strata.values()) == int((m["k"] >= 1).sum())
    ids, cm, k = content_evidence_miss(np.array([0.1, 0.5, 0.2, 0.9]), np.array([0, 0, 1, 1]), np.array([1, 0, 1, 1], bool))
    assert np.allclose(cm, [0.9, 0.8 * 0.1]) and list(k) == [1, 2]


def test_incident_formulas_prop4_prop5():
    # DESIGN Prop. 4(iii) quotes 200 / 102 epochs; exact min{n: 1-(1-phi)^n >= .95} is 199 / 102
    # (200 is the exp(-n phi) bound log(20)/0.015 = 199.7)
    assert k95(0.015) == 199 and k95(0.029) == 102
    assert grinding_evidence(0.05, 10, 1) == pytest.approx(0.4013, abs=1e-4)
    assert grinding_evidence(0.05, 10, 5) == pytest.approx(0.0104, abs=1e-4)
    assert miss_bound(10, 0.015) <= math.exp(-10 * 0.015)
    surv = tte_survival(np.full(300, 0.015))
    assert surv[198] <= 0.05 < surv[197]
    assert tte_quantile(np.full(300, 0.015), 0.95) == 199
    sc = stage_coverage(np.array([0.5, 0.5, 0.1]), np.array([0, 0, 1]))
    assert np.allclose(sc["detect"], [0.75, 0.1]) and sc["p_all"] == pytest.approx(0.075)


def test_design_probabilities_and_defensive_bound():
    rng = np.random.default_rng(1)
    s = rng.lognormal(6, 1.6, 5000) + 80
    a = rng.lognormal(0, 1.5, 5000) * (rng.random(5000) < 0.2)
    phi = 0.015
    pA, pS = pi_A(3.0, a, s, phi), pi_S(0.0005, s, phi)
    pi = pi_mixture(3.0, 0.0005, a, s, phi, 0.2)
    assert np.all(pi >= phi - 1e-15) and np.all(pi <= 1)
    assert np.allclose(pi, 0.8 * pA + 0.2 * pS)
    # Remark 1 safety bound: V^wr(pi) = sum a^2/pi <= min(V^wr(pi^A)/(1-tau), V^wr(pi^S)/tau)
    w = a ** 2 + 1e-3
    assert np.sum(w / pi) <= min(np.sum(w / pA) / 0.8, np.sum(w / pS) / 0.2) * (1 + 1e-12)
    assert np.all(design_pi("mixture", [3.0, 0.0005], a, s, phi, pay=np.arange(5000) % 2 == 0)[1::2] == 0)
    assert np.allclose(design_pi("uniform", [0.04], a, s, phi), 0.04)
    assert np.allclose(pi_uniform(0.001, phi), phi)
    boost = np.where(np.arange(5000) < 100, 5.0, 1.0)
    pb = design_pi("A", [3.0], a, s, phi, boost=boost)
    assert np.all(pb[:100] >= pA[:100]) and np.allclose(pb[100:], pA[100:])


def test_floor_rules():
    assert phi0_from_training(0.3, 0.05, 2.0e9, 1.0e9) == pytest.approx(0.03)
    dev = np.array([0, 0, 0, 1, 2, 2])
    s = np.array([1000.0, 1000.0, 1000.0, 10.0, 10.0, 10.0])
    phi, capped = device_floor_cap(0.03, Bbar=100.0, gamma=0.3, device=dev, s=s)
    # n_active = 3, share = 10; device 0 floor spend 0.03*3000 = 90 > 10 -> phi = 10/3000
    assert np.allclose(phi[:3], 10 / 3000) and capped[:3].all() and not capped[3:].any()
    ph, cp, act = flood_guard_floor(0.03, np.array([0.0, -1e9]), 100.0, 0.3, dev, s)
    assert ph.shape == (2, 6) and np.allclose(ph[0], 0.03) and np.allclose(ph[1, :3], 10 / 3000) and list(act) == [False, True]
    assert np.allclose(equalised_floor(0.3, [100.0, 1.0, 50.0], [1000.0, 1000.0, 1.0], 0.02), [0.03, 0.01, 1.0])


def test_two_tier_prop4ii_numbers():
    # measured C_H/C0 = 0.10 (CIC) and 0.05 (IoT-23 34-1): padded single-packet retention 0.53x / 0.69x,
    # unpadded 5.3x / 6.9x the single-tier value at phi_t = 0.1 (DESIGN Prop. 4(ii))
    for h, pad, unpad in ((0.10, 0.53, 5.3), (0.05, 0.69, 6.9)):
        C0, gB = 1.0e9, 0.01 * 1.0e9
        CH, CT = h * C0, (1 - h) * C0
        single = gB / C0
        u, p = two_tier_retention(0.1, 1, 0.3, gB / 0.3, 1, CH, CT)
        assert p / single == pytest.approx(pad, abs=0.01) and u / single == pytest.approx(unpad, abs=0.06)
        t, g = phi_t_star(1, 0.3, gB / 0.3, 1, CH, CT)
        assert t == pytest.approx(1.0) and g == pytest.approx(single)
        assert worst_case_retention(1.0, 3, 0.3, gB / 0.3, 1, CH, CT) == pytest.approx(single)


def test_boost_set_matches_definition():
    rng = np.random.default_rng(2)
    E, D = 80, 6
    bs = BoostState(D, shape=(), kappa_b=5.0, W_b=20)
    hits = {}
    for e in range(E):
        bs.begin_epoch(e)
        ref = boost_set_from_hits(hits, e, 20)
        got = set(np.nonzero(bs.boosted())[0].tolist())
        assert got == ref, (e, got, ref)
        m = rng.random(D) < 0.03
        for d in np.nonzero(m)[0]:
            hits.setdefault(int(d), []).append(e)
        bs.end_epoch(m)
    mult = bs.multiplier(np.array([0, 1, 2]))
    assert set(np.unique(mult)) <= {1.0, 5.0}


def test_ppi_unbiased_and_variance():
    rng = np.random.default_rng(3)
    n = 40
    f = (rng.random(n) < 0.3).astype(float)
    g = np.clip(f * 0.7 + rng.random(n) * 0.2, 0, 1)
    pi = rng.uniform(0.05, 0.9, n)
    Z = rng.random((200000, n)) < pi
    Q = ppi_total(Z, f, g, pi)
    v = ppi_variance(f, g, pi)
    assert abs(Q.mean() - f.sum()) <= 4 * math.sqrt(v / 200000)
    assert Q.var() == pytest.approx(v, rel=0.03)
    assert ppi_vhat(Z, f, g, pi).mean() == pytest.approx(v, rel=0.03)


def test_wald_and_monitor():
    lo, hi = wald_interval(10.0, 4.0, 0.05)
    assert lo == pytest.approx(10 - 1.959964 * 2, abs=1e-5) and hi == pytest.approx(10 + 1.959964 * 2, abs=1e-5)
    m = NaiveMonitor(0.05, shape=(2,))
    m.update(np.array([5.0, 5.0]), np.array([0.0, 5.0]), np.array([0.0, 1.0]))
    assert list(m.miss_hi) == [True, False]


def test_metrics_wilson_bootstrap_geomean():
    p, lo, hi = wilson(25, 1000)
    assert p == 0.025 and lo == pytest.approx(0.0170, abs=5e-4) and hi == pytest.approx(0.0367, abs=5e-4)
    _, lo0, hi0 = wilson(0, 1000)
    assert lo0 == pytest.approx(0, abs=1e-15) and hi0 == pytest.approx(0.00383, abs=1e-4)
    vals = {f"s{i}": np.exp(np.random.default_rng(i).normal(np.log(0.8), 0.3, 10)) for i in range(12)}
    bs = cluster_bootstrap(vals, B=500, seed=1)
    assert bs["lo"] < bs["point"] < bs["hi"]
    assert bs["point"] == pytest.approx(geometric_mean(np.concatenate(list(vals.values()))))
    r = mse_ratios([1.0, 4.0, np.nan], [2.0, 2.0, 1.0])
    assert r["geomean"] == pytest.approx(math.sqrt(0.5 * 2.0))


def test_config_defaults_and_hash():
    cfg = load_yaml()
    assert cfg.policy.gamma == 0.3 and cfg.policy.tau_mix == 0.2 and cfg.policy.Wc == 30
    assert cfg.policy.L == pytest.approx(math.log(16)) and cfg.policy.Dmax_mult == 200
    assert cfg.policy.kappa_b == 5.0 and cfg.policy.W_b == 20 and cfg.policy.alpha == 0.05
    assert cfg.budget.rho == 0.05 and cfg.grid.K_lcs == 16 and cfg.grid.ucs_w0 == 0.5 and cfg.grid.ucs_n_low == 15
    h1 = cfg_hash(cfg)
    assert h1 == cfg_hash(load_yaml()) and len(h1) == 16
    assert cfg_hash(load_yaml(overrides={"policy": {"gamma": 0.2}})) != h1
    assert cfg_hash(cfg, include_src=False) != h1 and len(src_hash()) == 64
    mc = cfg.mc_spec(reps=10)
    assert mc.K_lcs == 16 and mc.width_reps == 200 and mc.reps == 10
    assert DEFAULTS_PATH.exists()


def test_minimax_eg_improves_worst_case():
    rng = np.random.default_rng(4)
    n, Q = 3000, 3
    s = rng.lognormal(6, 1, n)
    M = np.zeros((Q, n))
    for q in range(Q):
        idx = rng.choice(n, 100 * (q + 1) ** 2, replace=False)
        M[q, idx] = rng.uniform(0.2, 1.0, len(idx))
    Qt = M.sum(axis=1)
    B = 0.05 * s.sum()
    res = eg_minimax(M, Qt, s, B, 0.01, iters=100)
    pi_u, _ = waterfill(np.sqrt(M.mean(axis=0) / 1) + 1e-12, s, B, 0.01)
    worst_u = float(((M / Qt[:, None] ** 2) @ (1 / pi_u - 1)).max())
    assert res["objective"] <= worst_u * (1 + 1e-9)
    assert np.isclose(np.sum(res["pi"] * s), B, rtol=1e-6) and np.isclose(res["w"].sum(), 1)


def test_float_twin_matches_intpi():
    """probs.design_pi fed with the committed fixed-point values equals intpi.threshold_unit/2^64 to
    <= 2^-40 relative (+2^-63) off the clips (P1-A's exact protocol path)."""
    from ctsample.design import intpi
    from ctsample.design.probs import committed_floor, committed_mix, quantise_utility, sqrt_eff
    rng = np.random.default_rng(5)
    n = 3000
    s = rng.integers(60, 300000, n)
    a = rng.lognormal(0, 1.5, n) * (rng.random(n) < 0.6)
    LamA, LamS = intpi.lam_fix(3.0), intpi.lam_fix(0.0005)
    tau_phi, t16 = intpi.floor_tau(0.015), intpi.t16_from_mix(0.2)
    taus = intpi.thresholds_batch(LamA, LamS, [intpi.a_fix_of(float(x)) for x in a], s.tolist(), tau_phi, t16)
    pi_int = np.array([t / 2 ** 64 for t in taus])
    pi_f = design_pi("mixture", [LamA / 2 ** 40, LamS / 2 ** 40], quantise_utility(a), s, committed_floor(0.015),
                     committed_mix(0.2))
    assert np.all(np.abs(pi_f - pi_int) <= 2.0 ** -40 * pi_int + 2.0 ** -63)
    assert np.allclose(sqrt_eff(s), intpi.isqrt_s32(s).astype(float) / 65536.0, rtol=0, atol=0)
    # the textbook sqrt differs by up to 2^-16/sqrt(s) relative: NOT bit-faithful
    assert np.max(np.abs(sqrt_eff(s, "true") / sqrt_eff(s) - 1)) > 2.0 ** -40
    assert np.allclose(sqrt_eff(np.array([100.5, 0.0])), np.sqrt([100.5, 0.0]))   # non-integer: np.sqrt
