"""DESIGN §7.4 test_abort_robust (Prop. 2): A11 withholding rules.

* "compliant epochs only" miscovers (>= 0.5) in the incriminating example (scripts/checks/abort_check.py:
  one eligible hit per epoch, pi = 0.05, E = 400, void every epoch whose unit was not retained -> 0.936);
* X := 0 keeps the LCS valid (time-uniform miscoverage <= alpha/2 + 3 SE) under 4 withholding rules,
  including adversaries that see every Z;
* the UCS is voided on unclean windows (never issued after a withholding), while "compliant only"
  breaks the UCS in the mirror case (void the epochs whose unit WAS retained).
"""
import math
import os

import numpy as np
import pytest

from ctsample.estimation.cs import LowerCS, UpperCS, lcs_grid, sure_bounds, ucs_grid
from ctsample.estimation.policy import UCS_WITHHELD, Query, QueryRegistry, statements
from ctsample.sim.metrics import coverage_table
from ctsample.sim.montecarlo import WITHHOLD_RULES, DesignSpec, Estimand, MCSpec, run_mc
from ctsample.sim.synth import synth_stream
from ctsample.sim.tables import StreamTables

SLOW = pytest.mark.skipif(os.environ.get("CTS_SLOW") != "1", reason="full-size run: set CTS_SLOW=1")
A2 = 0.025


def _one_hit_stream(E=400):
    return StreamTables.from_arrays(s=np.full(E, 1000.0), pay=np.ones(E, bool), epoch=np.arange(E),
                                    device=np.zeros(E, int), b=np.ones((E, 1)), hits=[(np.arange(E), np.ones(E))],
                                    query_names=["hit"], n_epochs=E)


def _design(pi=0.05):
    return DesignSpec(name=f"unif{pi}", design_id=1, phi0=pi, Bbar=50.0, lam_ref=(pi,), kind="uniform", util=None,
                      controller="none")


def _se(p, n):
    return math.sqrt(p * (1 - p) / n)


def _run(reps, alpha=0.05, **mckw):
    T = _one_hit_stream()
    mc = MCSpec(reps=reps, alpha=alpha, width_reps=0, withhold=WITHHOLD_RULES, stream_id=7, **mckw)
    res = run_mc(T, _design(), [Estimand("hit", "hit", V_ref=400 / 0.05)], mc, workers=1, chunk=1000)
    return coverage_table(res["cov"], by=("variant",)).set_index("variant")


def _assert_abort(ct, n):
    # (c) compliant-only is invalid, incriminating direction (scratch check 0.936)
    assert ct.loc["comp:void_unretained", "lcs_ft"] >= 0.5
    # mirror case: compliant-only breaks the UCS
    assert ct.loc["comp:void_retained", "ucs_ft"] >= 0.5
    # (a) X := 0: LCS valid under every rule, time-uniformly
    for r in WITHHOLD_RULES:
        assert ct.loc[f"x0:{r}", "lcs_tu"] <= A2 + 3 * _se(A2, n), r
        # (b) UCS voided on unclean windows: never issued at the end here, and no extra misses
        assert ct.loc[f"x0:{r}", "ucs_issued_rate"] == 0.0, r
        assert ct.loc[f"x0:{r}", "ucs_tu"] <= ct.loc["honest", "ucs_tu"] + 1e-12, r
    assert ct.loc["honest", "lcs_tu"] <= A2 + 3 * _se(A2, n)
    assert ct.loc["honest", "ucs_tu"] <= A2 + 3 * _se(A2, n)
    assert ct.loc["honest", "ucs_issued_rate"] == 1.0


def test_abort_fast():
    ct = _run(1000)
    _assert_abort(ct, 1000)


@pytest.mark.slow
@SLOW
def test_abort_full_4000():
    ct = _run(4000)
    _assert_abort(ct, 4000)
    print(ct[["lcs_tu", "lcs_ft", "ucs_tu", "ucs_ft", "ucs_issued_rate"]].to_string())


def test_x0_sharp_single_eta():
    """Single-eta e-processes at alpha/2 = 0.45: Ville is nearly tight, so this checks validity of
    X := 0 non-vacuously (honest miss well above 0, every rule <= alpha/2 + 3 SE)."""
    n = 1000
    ct = _run(n, alpha=0.9, K_lcs=1, lcs_decades=0.0, ucs_n_low=0, ucs_w0=1.0, ucs_eta0_frac=0.1)
    a = 0.45
    assert ct.loc["honest", "lcs_tu"] > 0.05 or ct.loc["honest", "ucs_tu"] > 0.1
    for v in ct.index:
        if v.startswith("x0:") or v == "honest":
            assert ct.loc[v, "lcs_tu"] <= a + 3 * _se(a, n), v
            assert ct.loc[v, "ucs_tu"] <= a + 3 * _se(a, n), v


def test_x0_valid_on_multihit_population():
    T = synth_stream(E=150, seed=11, base=60)
    from ctsample.sim.montecarlo import calibrated_design
    d = calibrated_design(T, "static", rho=0.05, gamma=0.3)
    ests = [Estimand(nm, nm) for nm in T.query_names]
    n = 300
    res = run_mc(T, d, ests, MCSpec(reps=n, width_reps=0, withhold=WITHHOLD_RULES, stream_id=9), workers=1)
    ct = coverage_table(res["cov"], by=("estimand", "variant"))
    x0 = ct[ct["variant"].str.startswith("x0:") | (ct["variant"] == "honest")]
    assert (x0["lcs_tu"] <= A2 + 3 * _se(A2, n)).all()
    assert (x0["ucs_tu"] <= A2 + 3 * _se(A2, n)).all()


def test_policy_withholds_ucs_on_unclean_window():
    etas, w = lcs_grid(100.0, A2)
    eu, wu = ucs_grid(20.0)
    lcs, ucs = LowerCS(etas, w, A2), UpperCS(eu, wu, A2, 20.0)
    Z = np.array([1, 0, 0, 1], bool)
    f = np.array([1.0, 0.0, 1.0, 1.0])
    pi = np.full(4, 0.05)
    Wm = np.array([0, 0, 0, 1], bool)
    X = np.where(Z & ~Wm, f / pi, 0.0)
    lcs.update(X)
    ucs.update(X, np.full(4, 20.0))
    reg = QueryRegistry(0.05)
    reg.declare_family("devices_x_windows", 10)
    q = Query("telnet_prompts", "telnet", devices=(3,), window=(0, 4), family="devices_x_windows")
    a = reg.register(q)
    assert a == pytest.approx(0.005)
    st = statements(q, (0, 4), Wm, lcs, ucs, sure_bounds(Z, f, np.ones(4), Wm), alpha=a, n_statements=3)
    assert st["ucs_status"] == UCS_WITHHELD and st["U_plus"] is None
    assert st["L_plus"] >= st["S"] == 1.0 and st["D_cap"] == 4.0      # withheld retained unit counts b
    assert st["family_wise_alpha"] == pytest.approx(0.015)
    st2 = statements(q, (0, 4), np.zeros(4, bool), lcs, ucs, sure_bounds(Z, f, np.ones(4)), alpha=a)
    assert st2["ucs_status"] == "issued" and st2["U_plus"] <= st2["D_cap"]
    cnt = Query("bytes", "cnt", kind="count")
    st3 = statements(cnt, (0, 4), None, lcs, None, (1.0, 10.0))
    assert st3["U_plus"] is None and "count" in st3["ucs_status"]


def test_v8_hook_with_real_verifier():
    """End to end with P1-A's gateway/verifier: honest disclosure -> UCS issued; one retained unit of
    the query's device withheld (MISSING, V6) -> UCS withheld, S drops by f, LCS still reported."""
    import copy

    from ctsample.crypto.signing import KeySet
    from ctsample.estimation.policy import make_statements_hook, statements_from_v8
    from ctsample.protocol import beacon as bc
    from ctsample.protocol import synth as psynth
    from ctsample.protocol.disclosure import build_disclosure
    from ctsample.protocol.epoch import FixedLambdaPolicy, Gateway
    from ctsample.protocol.verifier import TrustAnchors, Verifier
    from ctsample.protocol.witness import Witness

    drand = bc.DrandRounds()
    n_ep = 8
    clock = bc.align_stream(drand, n_ep)
    keys = KeySet.from_master(b"p1c-v8-hook")
    st = psynth.synthetic_stream(clock, n_ep, seed=3)
    pol = FixedLambdaPolicy(lam_A="10", lam_S="0.001", phi0="0.05", tau_mix="0.2")
    gw = Gateway(keys, clock, drand, pol, Witness(keys.sk_W), b"p1c-v8-hook-escrow-master-secret")
    gw.register()
    gw.run(copy.deepcopy(st.batches))
    units = {e: gw.records[e].batch.units for e in range(n_ep)}
    dev0 = min(u.device_id for e in units for u in units[e] if u.pay)
    f = {e: [1.0 if (u.pay and u.device_id == dev0) else 0.0 for u in units[e]] for e in units}
    b = {e: [1.0 if u.pay else 0.0 for u in units[e]] for e in units}
    dev = {e: [u.device_id for u in units[e]] for e in units}
    q = Query("dev_payload", "dev_payload", devices=(dev0,), window=(0, n_ep))
    hook = make_statements_hook([q], {q.name: f}, {q.name: b}, dev, V_ref=50.0, phi0="0.05")
    disc = build_disclosure(gw, range(n_ep))
    trust = TrustAnchors.from_disclosure(disc, keys.pk_W, beacon=drand)
    rep = Verifier(trust, pol, meta_codec=psynth.SyntheticMetaCodec(), statements_hook=hook).verify(disc)
    assert rep.n_fail == 0
    s0 = rep.statements["statements"][0]
    Qtrue = sum(sum(v) for v in f.values())
    assert s0["ucs_status"] == "issued" and s0["S"] <= Qtrue <= s0["D_cap"]
    assert s0["L_plus"] <= Qtrue <= s0["U_plus"] <= s0["D_cap"]
    assert rep.statements["family_wise_alpha"] == pytest.approx(0.05)
    # direct recomputation from the gateway's own tau/Z
    Zs = {e: np.asarray(gw.records[e].Z, bool) for e in units}
    S_direct = sum(float(np.sum(np.asarray(f[e]) * Zs[e])) for e in units)
    assert s0["S"] == S_direct
    # withhold one retained unit of dev0
    hits = [(e, i) for e in units for i, u in enumerate(units[e]) if Zs[e][i] and f[e][i] > 0]
    assert hits, "no retained unit of the query device in this synthetic run"
    e, i = hits[0]
    d2 = disc.copy()
    del d2.epochs[e].units[i]
    rep2 = Verifier(trust, pol, meta_codec=psynth.SyntheticMetaCodec(), statements_hook=hook).verify(d2)
    assert rep2.n_fail >= 1 and i in rep2.v8_inputs()[e]["W"]
    s2 = rep2.statements["statements"][0]
    assert s2["ucs_status"] == UCS_WITHHELD and s2["U_plus"] is None
    assert s2["S"] == s0["S"] - 1.0 and s2["L_plus"] <= s0["L_plus"] and s2["D_cap"] == s0["D_cap"] + 0.0
    # a statement restricted to another device is not affected by that withholding
    other = Query("other", "dev_payload", devices=(dev0 + 1,), window=(0, n_ep))
    s3 = statements_from_v8(rep2.v8_inputs(), other, f, b, dev, V_ref=50.0, phi0="0.05")
    assert s3["ucs_status"] == "issued"
