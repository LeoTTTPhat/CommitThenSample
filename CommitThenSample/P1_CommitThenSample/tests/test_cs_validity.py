"""DESIGN §7.4 test_cs_validity: 2,000-rep MC on heavy-tailed sparse populations with an ADAPTIVE design
(boost + realised-cost controller) and an ADAPTIVE ADVERSARIAL population (a C&C that goes silent for
20 epochs after each retained hit): time-uniform miscoverage <= alpha/2 + 3 SE per side; Wald
undercovers (regression guard); scripts/checks/check2.py's numbers approximately reproduced.

The default K = 16 mixtures pay ~log K, so at alpha = 0.05 their miscoverage is ~0; the "sharp" test
uses single-eta e-processes at alpha/2 = 0.45, where Ville's inequality is nearly tight, so that an
anti-conservative bug would be detected (measured misses 0.22-0.27 <= 0.45).
"""
import math
import os

import numpy as np
import pytest

from ctsample.sim.metrics import coverage_table, width_summary
from ctsample.sim.montecarlo import Estimand, MCSpec, ReactiveSilence, calibrated_design, run_mc
from ctsample.sim.synth import QuerySpec, check2_stream, synth_stream

SLOW = pytest.mark.skipif(os.environ.get("CTS_SLOW") != "1", reason="full-size run: set CTS_SLOW=1")


def _se(p, n):
    return math.sqrt(p * (1 - p) / n)


def _check2_run(R, kind="mixture"):
    T = check2_stream(E=100, ne=2000, seed=1, sparse_hits=30)
    d = calibrated_design(T, f"adapt_{kind}", rho=0.03, gamma=1 / 3, kind=kind, util="score", controller="realised",
                          boost=True)
    ests = [Estimand("ind", "ind"), Estimand("cnt", "cnt", ucs=True), Estimand("cnc", "cnc"), Estimand("sparse", "sparse")]
    mc = MCSpec(reps=R, width_reps=200, stream_id=2, report_every=10,
                reactor=ReactiveSilence(devices=(200,), silence=20, queries=("cnc",)))
    return run_mc(T, d, ests, mc, workers=1)


def _assert_valid(ct, R, a2=0.025, ucs=True):
    lim = a2 + 3 * _se(a2, R)
    assert (ct["lcs_tu"] <= lim).all(), ct[["estimand", "lcs_tu"]]
    if ucs:
        assert (ct["ucs_tu"] <= lim).all(), ct[["estimand", "ucs_tu"]]


def test_cs_validity_adaptive_check2_population():
    R = 2000
    res = _check2_run(R)
    ct = coverage_table(res["cov"], by=("estimand",)).set_index("estimand")
    _assert_valid(ct.reset_index(), R)
    # Wald regression guards: sparse estimand at fixed time; naive time-uniform monitoring
    assert ct.loc["sparse", "wald_hi_ft"] > 0.025 + 3 * _se(0.025, R)
    assert ct.loc["ind", "wald_hi_tu"] > 0.25                     # check2: 1 - 0.515 = 0.485
    # approximate reproduction of check2.py (rev-1 numbers: rel LCB 0.809, rel UCB 2.285, Wald-unif 0.515)
    w = res["width"]
    last = w.loc[w.groupby(["rep", "estimand"])["epoch"].idxmax()]
    li = last[last["estimand"] == "ind"]
    rel_l = float(np.median(li["Lplus"] / li["Q"]))
    rel_u = float(np.median(li["Uplus"] / li["Q"]))
    assert 0.7 <= rel_l <= 0.95, rel_l
    assert 1.3 <= rel_u <= 3.0, rel_u
    wald_unif_cov = 1 - float((res["cov"].query("estimand == 'ind'")[["wald_lo_tu", "wald_hi_tu"]].any(axis=1)).mean())
    assert 0.3 <= wald_unif_cov <= 0.7, wald_unif_cov
    # the C&C population really reacted (its truth varies across reps) and S never exceeds Q
    cnc = res["cov"].query("estimand == 'cnc'")
    assert cnc["Q"].std() > 0 and cnc["Q"].mean() < 100
    assert (res["cov"]["S"] <= res["cov"]["Q"] + 1e-9).all()


def _dense_adaptive(R, alpha):
    qs = [QuerySpec("dense", "indicator", elig_frac=1.0, n_hits=20000, n_att=50),
          QuerySpec("mid", "indicator", elig_frac=0.5, n_hits=3000, n_att=10)]
    T = synth_stream(E=200, seed=4, base=150, n_devices=50, zero_payload_frac=0.0, queries=qs, cnc_device=49)
    d = calibrated_design(T, "adapt_dense", rho=0.05, gamma=0.3, controller="realised", boost=True)
    mc = MCSpec(reps=R, alpha=alpha, width_reps=0, stream_id=4, K_lcs=1, lcs_decades=0.0, ucs_n_low=0, ucs_w0=1.0,
                ucs_eta0_frac=0.1, reactor=ReactiveSilence(devices=(49,), silence=20, queries=("cnc",)))
    return run_mc(T, d, [Estimand(n, n) for n in T.query_names], mc, workers=1)


def test_cs_validity_sharp_single_eta():
    R = 2000
    res = _dense_adaptive(R, alpha=0.9)
    ct = coverage_table(res["cov"], by=("estimand",)).set_index("estimand")
    _assert_valid(ct.reset_index(), R, a2=0.45)
    assert ct.loc["dense", "lcs_tu"] > 0.1 and ct.loc["dense", "ucs_tu"] > 0.1     # non-vacuous


@pytest.mark.slow
@SLOW
def test_cs_validity_full_realistic_stream():
    """870 epochs, ~200 payload units/epoch, 20 sparse heavy-tailed queries + reactive C&C, 2,000 reps,
    adaptive design (boost + realised-cost controller), defensive mixture and pure-AI (tau_mix = 0)."""
    qs = [QuerySpec(f"q{i}", "count" if i % 4 == 3 else "indicator", elig_frac=0.5 if i % 5 else 0.03,
                    n_hits=[2000, 300, 60, 20][i % 4], n_att=[8, 3, 2, 1][i % 4], rare=(i % 5 == 0)) for i in range(20)]
    T = synth_stream(E=870, seed=31, base=171, n_devices=50, zero_payload_frac=0.3, queries=qs, cnc_device=49)
    R = 2000
    for kind in ("mixture", "A"):
        d = calibrated_design(T, f"adapt_{kind}", kind=kind, controller="realised", boost=True)
        mc = MCSpec(reps=R, width_reps=200, stream_id=31,
                    reactor=ReactiveSilence(devices=(49,), silence=20, queries=("cnc",)))
        res = run_mc(T, d, [Estimand(n, n, ucs=True) for n in T.query_names], mc, workers=8)
        ct = coverage_table(res["cov"], by=("estimand",))
        _assert_valid(ct, R)
        ws = width_summary(res["width"])
        print(f"kind={kind}: max lcs_tu={ct['lcs_tu'].max():.4f} max ucs_tu={ct['ucs_tu'].max():.4f} "
              f"Wald ft miss (lo+hi) max={(ct['wald_lo_ft'] + ct['wald_hi_ft']).max():.3f} "
              f"median={(ct['wald_lo_ft'] + ct['wald_hi_ft']).median():.3f}; naive-monitoring max="
              f"{(ct['wald_lo_tu'] + ct['wald_hi_tu']).max():.3f}; spend={res['ctrl']['spend_ratio'].mean():.4f}; "
              f"median rel LCB gap={ws['rel_lcb_gap_median'].median():.3f}; {res['seconds']:.1f}s")


@pytest.mark.slow
@SLOW
def test_cs_validity_check2_full_both_designs():
    for kind in ("mixture", "A"):
        res = _check2_run(2000, kind)
        ct = coverage_table(res["cov"], by=("estimand",))
        _assert_valid(ct, 2000)
        ws = width_summary(res["width"]).set_index("estimand")
        cov = res["cov"]
        wf = 1 - (cov[["wald_lo_ft", "wald_hi_ft"]].any(axis=1)).groupby(cov["estimand"]).mean()
        wu = 1 - (cov[["wald_lo_tu", "wald_hi_tu"]].any(axis=1)).groupby(cov["estimand"]).mean()
        print(f"kind={kind}\n", ct[["estimand", "lcs_tu", "ucs_tu", "wald_lo_ft", "wald_hi_ft", "wald_hi_tu"]].to_string(index=False))
        print("Wald fixed-time coverage:", wf.round(3).to_dict(), " naive-monitoring coverage:", wu.round(3).to_dict())
        print(ws[["rel_lcb_gap_median", "ucb_slack_median", "wald_rel_halfwidth_median"]].to_string())
