"""DESIGN §6.5 / §7.4: MC engine checks.

* Philox: the vectorised Philox4x64-10 reproduces numpy's Generator(Philox(key=SeedSequence([stream,
  design, rep]), counter=[0, e, 0, 0])).random(n) bit-exactly; results do not depend on chunking.
* normal approximation of the non-hit realised cost vs exact draws on 10^4 units (moments, KS) and
  at the controller level (spend/target, max|D|/Bbar);
* the engine reproduces analytic miss probabilities prod(1 - pi) and HT unbiasedness (static and
  adaptive designs); e-process flags agree with the CS inversion; resumable outputs; spawn workers.
"""
import math
import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ctsample.design.controller import HorizonController, calibrate_lambda
from ctsample.design.probs import pi_A, pi_mixture
from ctsample.estimation.ht import ht_variance
from ctsample.sim.analytic import pi_full
from ctsample.sim.montecarlo import (Estimand, MCSpec, ReactiveSilence, calibrated_design, philox_epoch_uniforms,
                                     prepare, rep_keys, run_mc, simulate_chunk, static_pass)
from ctsample.sim.synth import synth_stream, synth_traffic

SLOW = pytest.mark.skipif(os.environ.get("CTS_SLOW") != "1", reason="full-size run: set CTS_SLOW=1")


@pytest.fixture(scope="module")
def stream():
    return synth_stream(E=120, seed=21, base=80)


def test_philox_bit_exact_with_numpy():
    keys = rep_keys(17, 4242, [0, 1, 999])
    for r, k in zip([0, 1, 999], keys):
        assert np.array_equal(k, np.random.SeedSequence([17, 4242, r]).generate_state(2, np.uint64))
        assert np.array_equal(k, np.random.Philox(np.random.SeedSequence([17, 4242, r])).state["state"]["key"])
    n = np.array([0, 1, 5, 4, 9, 0, 13])
    for word2 in (0, 3):
        U = philox_epoch_uniforms(keys, n, word2=word2)
        c = np.concatenate([[0], np.cumsum(n)])
        for i, k in enumerate(keys):
            for e in range(len(n)):
                ref = np.random.Generator(np.random.Philox(key=k, counter=np.array([0, e, word2, 0], np.uint64))).random(n[e])
                assert np.array_equal(ref, U[i, c[e]:c[e + 1]])
    Uo = philox_epoch_uniforms(keys, n, open_interval=True)
    assert np.all((Uo > 0) & (Uo < 1))


def test_results_independent_of_chunking(stream):
    d = calibrated_design(stream, "adapt", controller="realised", boost=True)
    ests = [Estimand(nm, nm) for nm in stream.query_names]
    mc = MCSpec(reps=24, width_reps=8, stream_id=3)
    a = run_mc(stream, d, ests, mc, workers=1, chunk=24)["cov"].sort_values(["variant", "est", "rep"]).reset_index(drop=True)
    b = run_mc(stream, d, ests, mc, workers=1, chunk=8)["cov"].sort_values(["variant", "est", "rep"]).reset_index(drop=True)
    # identical draws; floats may differ in the last bits (BLAS summation order depends on the batch size)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)


def test_normal_cost_approximation_vs_exact_10k_units():
    rng = np.random.default_rng(0)
    n = 10 ** 4
    s = rng.lognormal(6.0, 1.6, n) + 80
    a = rng.lognormal(0, 1.5, n) * (rng.random(n) < 0.2)
    pi = pi_mixture(3.0, 0.0005, a, s, 0.015)
    mu, var = float((pi * s).sum()), float((pi * (1 - pi) * s * s).sum())
    R = 4000
    ex = np.array([((rng.random(n) < pi) * s).sum() for _ in range(R)])
    assert ex.mean() == pytest.approx(mu, rel=3 * math.sqrt(var / R) / mu + 1e-9)
    assert ex.var() == pytest.approx(var, rel=0.1)
    ks = stats.kstest((ex - mu) / math.sqrt(var), "norm").statistic
    assert ks < 0.06                       # measured 0.031 at R = 20,000 (skewness 0.46)


def _ctrl_run(seed, exact, E=2880):
    tr = synth_traffic(E, seed=seed, base=200.0, n_devices=1)
    s, a, ep = tr["s"], tr["a"], tr["epoch"]
    ptr = np.searchsorted(ep, np.arange(E + 1))
    Bbar, phi0 = 0.05 * s.sum() / E, 0.015
    lam = calibrate_lambda(s, a, Bbar, phi0, E)
    c = HorizonController(lam, Bbar, mode="realised")
    rng = np.random.default_rng(seed + 99)
    for e in range(E):
        p = pi_A(c.lam(), a[ptr[e]:ptr[e + 1]], s[ptr[e]:ptr[e + 1]], phi0)
        se = s[ptr[e]:ptr[e + 1]]
        if exact:
            cost = float(((rng.random(len(p)) < p) * se).sum())
        else:
            cost = max(0.0, float((p * se).sum() + math.sqrt((p * (1 - p) * se * se).sum()) * rng.standard_normal()))
        c.update(cost)
    return float(c.spend_ratio()), float(c.max_abs_D / Bbar)


def test_normal_cost_approximation_controller_level():
    ex = np.array([_ctrl_run(s, True) for s in range(3)])
    na = np.array([_ctrl_run(s, False) for s in range(3)])
    assert np.abs(ex[:, 0] - na[:, 0]).max() < 0.01
    assert abs(ex[:, 1].mean() - na[:, 1].mean()) < 0.35 * ex[:, 1].mean()


def test_engine_reproduces_analytic_miss_and_unbiasedness(stream):
    d = calibrated_design(stream, "static")
    ests = [Estimand(nm, nm) for nm in stream.query_names]
    R = 3000
    res = run_mc(stream, d, ests, MCSpec(reps=R, width_reps=0, stream_id=5), workers=1, chunk=500)
    cov = res["cov"]
    pi = pi_full(stream, static_pass(stream, d))
    for k, nm in enumerate(stream.query_names):
        u, f = stream.hits(nm)
        g = cov[cov["estimand"] == nm]
        p_none = float(np.prod(1 - pi[u]))
        freq = float((g["S"] == 0).mean())
        assert abs(freq - p_none) <= 4 * math.sqrt(max(p_none * (1 - p_none), 1e-4) / R) + 1e-3, nm
        var = float(ht_variance(f, pi[u]))
        assert abs(g["Qhat"].mean() - f.sum()) <= 4 * math.sqrt(var / R), nm
        assert g["Vh"].mean() == pytest.approx(var, rel=0.1), nm
        assert np.allclose(g["Q"], f.sum())


def test_engine_unbiased_adaptive_and_reactive():
    T = synth_stream(E=120, seed=22, base=80, cnc_device=7)
    d = calibrated_design(T, "adaptive", controller="realised", boost=True)
    ests = [Estimand(nm, nm) for nm in T.query_names]
    R = 2000
    mc = MCSpec(reps=R, width_reps=0, stream_id=6, reactor=ReactiveSilence(devices=(7,), silence=20, queries=("cnc",)))
    cov = run_mc(T, d, ests, mc, workers=1, chunk=500)["cov"]
    for nm in T.query_names:
        g = cov[cov["estimand"] == nm]
        err = g["Qhat"] - g["Q"]
        assert abs(err.mean()) <= 4 * err.std(ddof=1) / math.sqrt(R) + 1e-9, nm
    cnc = cov[cov["estimand"] == "cnc"]
    assert cnc["Q"].std() > 0 and cnc["Q"].max() <= T.query_total("cnc")   # population reacts per rep


def test_eproc_flags_agree_with_inversion(stream):
    d = calibrated_design(stream, "static")
    ests = [Estimand(nm, nm) for nm in stream.query_names]
    # alpha = 0.9 so that misses actually happen
    res = run_mc(stream, d, ests, MCSpec(reps=400, alpha=0.9, width_reps=400, stream_id=8), workers=1)
    w, cov = res["width"], res["cov"]
    last = w.loc[w.groupby(["rep", "estimand"])["epoch"].idxmax()].set_index(["rep", "estimand"])
    c = cov.set_index(["rep", "estimand"]).loc[last.index]
    tol = 1e-6 * np.maximum(1, np.abs(last["L"]))
    lcs_inv = last["L"] > last["Q"] + tol
    lcs_inv_lo = last["L"] > last["Q"] - tol
    assert np.all((c["lcs_ft"] == lcs_inv) | (lcs_inv_lo & ~lcs_inv))
    issued = last["ucs_issued"].astype(bool)
    U = last["U"]
    ucs_inv = issued & (last["Q"] > U + 1e-6 * np.maximum(1, U))
    assert np.all(c["ucs_ft"][issued] == ucs_inv[issued])
    assert (last["Lplus"] <= last["Q"] + 1e-9).mean() >= 1 - c["lcs_ft"].mean() - 1e-12
    assert np.all(last["Dcap"] >= last["Q"] - 1e-9) and np.all(last["S"] <= last["Q"] + 1e-9)


def test_resumable_outputs_and_spawn_workers(tmp_path, stream):
    tdir = tmp_path / "tables"
    stream.save(tdir)
    from ctsample.sim.tables import StreamTables
    T = StreamTables.load(tdir)
    d = calibrated_design(T, "static")
    ests = [Estimand(nm, nm) for nm in T.query_names]
    mc = MCSpec(reps=32, width_reps=8, stream_id=4)
    out = tmp_path / "run"
    r1 = run_mc(str(tdir), d, ests, mc, out_dir=out, workers=2, chunk=8)
    assert (out / "DONE").exists() and len(list(out.glob("rep*_cov.parquet"))) == 4
    victim = sorted(out.glob("rep*_cov.parquet"))[1]
    stamp = {p.name: p.stat().st_mtime_ns for p in out.glob("rep*_cov.parquet")}
    victim.unlink()
    r2 = run_mc(str(tdir), d, ests, mc, out_dir=out, workers=1, chunk=8)
    new = {p.name: p.stat().st_mtime_ns for p in out.glob("rep*_cov.parquet")}
    assert [n for n in new if new[n] != stamp.get(n)] == [victim.name]
    k = ["variant", "est", "rep"]
    eq = lambda x, y: pd.testing.assert_frame_equal(x.sort_values(k).reset_index(drop=True),  # noqa: E731
                                                    y.sort_values(k).reset_index(drop=True), check_exact=False,
                                                    rtol=1e-9, atol=1e-9)
    eq(r1["cov"], r2["cov"])
    r3 = run_mc(T, d, ests, mc, workers=1, chunk=32)     # in-memory serial == spawn workers
    eq(r1["cov"], r3["cov"])


def test_flood_guard_voids_ucs_and_caps_floor():
    T = synth_stream(E=150, seed=23, base=60)
    d = calibrated_design(T, "guard", flood_guard=True, controller="expected")
    from dataclasses import replace
    d = replace(d, Dmax_mult=-1e9)          # force floor_mode = 1 from the start (D > -Dmax always)
    ests = [Estimand(nm, nm) for nm in T.query_names]
    res = run_mc(T, d, ests, MCSpec(reps=16, width_reps=0, stream_id=2), workers=1)
    assert (res["ctrl"]["guard_epochs"] == T.n_epochs).all()
    cov = res["cov"]
    ind = cov[cov["estimand"].isin([q for q, k in zip(T.query_names, T.query_kinds) if k == "indicator"])]
    assert not ind["ucs_issued"].any()      # every indicator query includes capped units here
    assert not cov["lcs_tu"].any()


@pytest.mark.slow
@SLOW
def test_benchmark_realistic_size():
    """~870 epochs x ~200 payload units/epoch x 20 queries x 2,000 hits: record reps/s (notes/P1-C.md)."""
    import time
    from ctsample.sim.synth import QuerySpec
    qs = [QuerySpec(f"q{i}", "count" if i % 4 == 3 else "indicator", elig_frac=0.5, n_hits=2000, n_att=8) for i in range(20)]
    T = synth_stream(E=870, seed=5, base=171, n_devices=50, zero_payload_frac=0.3, queries=qs)
    ests = [Estimand(nm, nm) for nm in T.query_names]
    for kw in (dict(controller="expected"), dict(controller="realised", boost=True)):
        d = calibrated_design(T, "bench_" + kw["controller"], **kw)
        mc = MCSpec(reps=64, width_reps=0, stream_id=5)
        P = prepare(T, d, ests, mc)
        t0 = time.time()
        simulate_chunk(P, d, mc, np.arange(200, 264))
        dt = time.time() - t0
        print(f"{kw}: {64 / dt:.1f} reps/s per process (64 reps in {dt:.2f}s)")


def test_lcs_variants_in_engine(stream):
    """K in {16, 32, 64} grids and the [SHOULD] normal-mixture boundary run in the engine and stay valid."""
    d = calibrated_design(stream, "static")
    ests = [Estimand(nm, nm) for nm in stream.query_names]
    R = 300
    lim = 0.025 + 3 * math.sqrt(0.025 * 0.975 / R)
    gaps = {}
    for kw in (dict(K_lcs=16, lcs_decades=2.0), dict(K_lcs=32, lcs_decades=4.0), dict(K_lcs=64, lcs_decades=4.0),
               dict(lcs_kind="normal_mixture")):
        res = run_mc(stream, d, ests, MCSpec(reps=R, width_reps=50, stream_id=13, **kw), workers=1)
        assert res["cov"]["lcs_tu"].mean() <= lim
        w = res["width"]
        last = w.loc[w.groupby(["rep", "estimand"])["epoch"].idxmax()]
        assert np.isfinite(last["L"]).all() and (last["Lplus"] <= last["Q"] + 1e-9).mean() >= 0.97
        gaps[str(kw)] = float(np.median(last["Q"] - last["Lstar"]))
    assert len(set(round(g, 6) for g in gaps.values())) > 1          # the variants really differ


def test_budget_modes_and_equalised_floor(stream):
    from dataclasses import replace
    d = calibrated_design(stream, "static")
    pe = static_pass(stream, replace(d, controller="per_epoch"))
    s_all_e = np.bincount(np.asarray(stream.epoch), weights=np.asarray(stream.s, float), minlength=stream.n_epochs)
    # each component spends B_e = rho * sum_{U_e} s, so the mixture does too (unless saturated)
    ok = np.abs(pe["cost"] - d.rho * s_all_e) <= 1e-6 * d.rho * s_all_e
    assert ok.mean() > 0.95
    eq = replace(d, floor_mode="equalised", name="eqfloor")
    sp = static_pass(stream, eq)
    assert sp["pi_pay"].min() >= d.phi0 / 2 * (1 - 1e-12)
    res = run_mc(stream, eq, [Estimand(nm, nm) for nm in stream.query_names],
                 MCSpec(reps=64, width_reps=0, stream_id=14), workers=1)
    assert not res["cov"]["lcs_tu"].any()


def test_dcap_exact_for_small_eligible_sets(stream):
    d = calibrated_design(stream, "static")
    pi = pi_full(stream, static_pass(stream, d))
    b, f = stream.b_col("ind_rare"), stream.f_dense("ind_rare")
    ED = float(np.sum(b - pi * (b - f)))                       # E[Dcap] = sum b - sum pi (b - f)
    R = 400
    for emax, exact in ((5000, True), (0, False)):
        res = run_mc(stream, d, [Estimand("ind_rare", "ind_rare")],
                     MCSpec(reps=R, width_reps=0, stream_id=15, exact_elig_max=emax), workers=1)
        D, Q = res["cov"]["Dcap"].to_numpy(), res["cov"]["Q"].to_numpy()
        assert np.all(D >= Q - 1e-9)
        assert abs(D.mean() - ED) <= 4 * D.std(ddof=1) / math.sqrt(R) + 1e-6
        assert np.allclose(D, np.round(D)) == exact             # exact Bernoulli sums of b in {0, 1}
