"""Tests of sim/experiments_core.py (P1-E): strata, stream de-duplication, V_ref fallback, incident recall / TTE /
natural-estimator formulas against brute force, and the exact full-unit adaptive simulator against the static
pass and P1-C's MC engine. Synthetic streams only (fast)."""
import math
from dataclasses import replace

import numpy as np
import pytest

from ctsample.sim import experiments_core as X
from ctsample.sim.montecarlo import Estimand, MCSpec, calibrated_design, run_mc, static_pass
from ctsample.sim.synth import QuerySpec, synth_stream


@pytest.fixture(scope="module")
def stream():
    qs = [QuerySpec("ind_common", "indicator", 0.3, 150, 3), QuerySpec("ind_sparse", "indicator", 0.3, 15, 2),
          QuerySpec("cnt_heavy", "count", 0.3, 120, 3), QuerySpec("ind_tiny", "indicator", 0.0005, 3, 1, rare=True)]
    return synth_stream(E=80, seed=5, base=60, n_devices=20, queries=qs, stream_id=77)


def test_strata_bins_classes():
    assert [X.q_stratum(q) for q in (0, 1, 20, 21, 1e4)] == ["Q=0", "Q1-20", "Q1-20", "Q>20", "Q>20"]
    assert [X.k_stratum(k) for k in (1, 20, 21, 100, 101)] == ["k<=20", "k<=20", "k21-100", "k21-100", "k>100"]
    assert [X.prevalence_bin(p) for p in (0.0003, 0.00104, 0.0041, 0.0118, 0.0325, 0.0496)] == \
        ["0.1%", "0.1%", "1%", "1%", "5%", "5%"]
    assert [X.query_class(n) for n in ("D1o", "S6", "L_XSS", "M1")] == ["D", "S", "L", "M"]


def test_distinct_streams_keeps_lowest_target():
    rows = [{"stream_id": f"lp_iot23_k_p{t}", "kind": "lp_iot23", "fold_short": "k", "p": p, "bin": X.prevalence_bin(r)}
            for t, p, r in (("1000", 0.001, 0.001), ("10000", 0.01, 0.0059), ("50000", 0.05, 0.0059))]
    rows.append({"stream_id": "lp_iot23_j_p50000", "kind": "lp_iot23", "fold_short": "j", "p": 0.05, "bin": "5%"})
    kept, dropped = X.distinct_streams(rows)
    assert [r["stream_id"] for r in kept] == ["lp_iot23_j_p50000", "lp_iot23_k_p1000", "lp_iot23_k_p10000"]
    assert dropped == [dict(rows[2], duplicate_of="lp_iot23_k_p10000")]


def test_v_ref_training_and_fallback():
    g = {"V_per_epoch": {"a": 0.5, "b": 0.0}}
    assert X.v_ref_value(g, "a", 0.02) == (0.5 * 2880, "training")
    assert X.v_ref_value(g, "b", 0.02) == (pytest.approx(50.0), "fallback_one_floor_hit")
    assert X.v_ref_value(g, "c", 0.02, bbar=4.0)[0] == pytest.approx(16 / 0.02)


def test_build_estimands_rule(stream):
    ests, rows = X.build_estimands(stream, {"V_per_epoch": {"ind_common": 1.0}}, phi0=0.02, min_elig=50)
    names = [e.name for e in ests]
    assert "ind_tiny" not in names                         # < 50 eligible units: not evaluated
    assert set(names) == {"ind_common", "ind_sparse", "cnt_heavy"}
    r = {x["estimand"]: x for x in rows}
    assert r["ind_common"]["V_ref_source"] == "training" and r["ind_sparse"]["V_ref_source"].startswith("fallback")
    for nm in names:
        u, f = stream.hits(nm)
        assert r[nm]["Q"] == pytest.approx(f.sum())


def _random_incidents(n_pay, rng):
    incs = []
    for j in range(6):
        k = int(rng.integers(1, 30))
        cpos = np.sort(rng.choice(n_pay, size=k, replace=False))
        ppos = np.union1d(cpos, rng.choice(n_pay, size=5, replace=False))
        incs.append({"incident": j, "k": k, "content_pos": cpos, "presence_pos": ppos, "first_epoch": 0})
    return incs


def test_recall_and_tte_against_bruteforce():
    rng = np.random.default_rng(3)
    n = 400
    pi = rng.uniform(0.0, 0.2, n)
    ep = np.sort(rng.integers(0, 40, n))
    incs = _random_incidents(n, rng)
    for d in incs:
        d["first_epoch"] = int(ep[d["content_pos"]].min())
    rec = X.recall_from_pi(pi, incs)
    Z = rng.random((40000, n)) < pi
    for d, r in zip(incs, rec):
        mc = Z[:, d["content_pos"]].any(axis=1).mean()
        assert abs(mc - r["content_recall"]) < 0.012
        mcp = Z[:, d["presence_pos"]].any(axis=1).mean()
        assert abs(mcp - r["presence_recall"]) < 0.012
    tte = X.tte_from_pi(pi, ep, incs)
    for d, t in zip(incs, tte):
        e = ep[d["content_pos"]] - d["first_epoch"]
        first = np.where(Z[:, d["content_pos"]], e[None, :], 10 ** 9).min(axis=1)
        p_by_t = (first <= t).mean() if math.isfinite(t) else None
        if math.isfinite(t):
            assert p_by_t >= 0.5 - 0.01
            if t > 0:
                assert (first <= t - 1).mean() < 0.5 + 0.01
        else:
            assert (first < 10 ** 9).mean() < 0.5 + 0.01
    mf = rng.random(n) < 0.1
    ma = mf | (rng.random(n) < 0.05)
    rm = X.recall_from_masks(mf, ma, incs)
    for d, r in zip(incs, rm):
        assert r["content_recall"] == float(mf[d["content_pos"]].any())
        assert r["presence_recall"] == float(ma[d["presence_pos"]].any() or mf[d["content_pos"]].any())


def test_natural_estimators(stream):
    pidx = X.pay_index(stream)
    rng = np.random.default_rng(11)
    pi = np.clip(rng.uniform(0.02, 0.6, len(pidx)), 0, 1)
    names = ["ind_common", "cnt_heavy"]
    ht = {r["estimand"]: r for r in X.natural_estimator_stats(stream, names, "ht", pi_pay=pi)}
    sm = {r["estimand"]: r for r in X.natural_estimator_stats(stream, names, "sum", pi_pay=pi)}
    upos = np.full(stream.n_units, -1)
    upos[pidx] = np.arange(len(pidx))
    for nm in names:
        u, f = stream.hits(nm)
        p = pi[upos[u]]
        Z = rng.random((20000, len(u))) < p
        est_ht = (Z * (f / p)).sum(axis=1)
        est_sum = (Z * f).sum(axis=1)
        assert ht[nm]["bias"] == 0.0
        assert ht[nm]["var"] == pytest.approx(est_ht.var(), rel=0.05)
        assert sm[nm]["bias"] == pytest.approx(est_sum.mean() - f.sum(), rel=0.02, abs=1.0)
        assert sm[nm]["var"] == pytest.approx(est_sum.var(), rel=0.05)
        assert X.ht_var_by_query(stream, pi, [nm])[nm] == pytest.approx(ht[nm]["var"])


def _design(stream, **kw):
    return calibrated_design(stream, "t", rho=0.05, gamma=0.3, kind="mixture", util="U5", tau_mix=0.2, **kw)


def test_full_sim_static_reproduces_static_pass(stream):
    d = _design(stream)
    st = static_pass(stream, d)
    sim = X.AdaptiveFullSim(stream, d, queries=["ind_common"], incs=[])
    reps = np.arange(400)
    out = sim._chunk(reps, 5)                       # internal: per-unit frequencies via Qhat-free path
    # static design: pi identical across reps -> HT unbiased and spend = expected-cost pass
    assert np.allclose(out["spend_ratio"], st["cost"].sum() / (d.Bbar * stream.n_epochs), rtol=1e-9)
    u, f = stream.hits("ind_common")
    assert abs(out["Qhat"][:, 0].mean() - f.sum()) < 4 * out["Qhat"][:, 0].std() / math.sqrt(len(reps))


def test_full_sim_adaptive_matches_engine(stream):
    names = ["ind_common", "ind_sparse", "cnt_heavy"]
    R = 300
    for ctl, tol in (("expected", 0.01), ("realised", 0.06)):
        d = replace(_design(stream), controller=ctl, boost=True, boost_queries=("ind_common",), name="ad", design_id=99)
        sim = X.AdaptiveFullSim(stream, d, queries=names, incs=X.incident_sets(stream))
        out = sim.run(np.arange(R), 5, chunk=100)
        for k, nm in enumerate(names):
            u, f = stream.hits(nm)
            se = out["Qhat"][:, k].std() / math.sqrt(R)
            assert abs(out["Qhat"][:, k].mean() - f.sum()) < 4.5 * se + 1e-9
        mc = run_mc(stream, d, [Estimand(nm, nm) for nm in names], MCSpec(reps=R, stream_id=5, width_reps=0),
                    workers=1, check_mem=False)
        eng = mc["ctrl"]
        # same design, different randomness. With the expected-cost controller (boost only) the two must agree
        # closely; with the realised-cost controller the engine approximates the non-hit cost by one normal draw
        # clipped at 0 per epoch, which on this small synthetic stream (~60 units/epoch, lognormal sizes) biases
        # the simulated spend upwards by a few percent (the full-unit simulator is exact).
        assert abs(out["spend_ratio"].mean() - eng["spend_ratio"].mean()) < tol * eng["spend_ratio"].mean()
        assert abs(out["boosted_frac"].mean() - eng["boosted_frac"].mean()) < 0.25 * max(eng["boosted_frac"].mean(), 1e-3) + 0.005
        assert out["content"].shape == (R, len(X.incident_sets(stream)))


def test_hiding_capacity_rows(stream):
    pidx = X.pay_index(stream)
    pi = np.full(len(pidx), 0.02)
    H = X.hiding_capacity_rows(stream, pi, ["ind_common", "cnt_heavy"], alpha_side=0.025)
    assert "cnt_heavy" not in H                           # indicator estimands only
    n_el = int((np.asarray(stream.b_col("ind_common"))[pidx] > 0).sum())
    assert H["ind_common"] == min(n_el, math.floor(math.log(40) / -math.log1p(-0.02)))
