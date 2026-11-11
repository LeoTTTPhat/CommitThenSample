"""DESIGN §7.4 test_verifier_detects: an honest run gives zero FAIL; A1–A5, A7, A9, A10, A11 are flagged
by the expected check; A12 (equal-size payload substitution before commit) is NOT flagged (documented
limit); A6 grinding succeeds without TR and is refused with TR; A8 (large omission) is flagged by V7.

Synthetic multi-device stream, 24 epochs, real drand rounds, real ECVRF, Ed25519, BLAKE3, AEAD.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import pytest

from ctsample.adversary import abort, grinding, omission, tamper
from ctsample.crypto.signing import KeySet
from ctsample.protocol import beacon as bc
from ctsample.protocol import synth
from ctsample.protocol.counters import CounterSource, LossModel, SkewModel, calibrate_eps, honest_pairs
from ctsample.protocol.disclosure import build_disclosure
from ctsample.protocol.epoch import ConstantPolicy, Gateway
from ctsample.protocol.header import gw_id_from_pk
from ctsample.protocol.verifier import FAIL, PASS, WARN, TrustAnchors, Verifier
from ctsample.protocol.witness import Witness

N = 24
MARK = b"\xde\xad-EVIL-\xbe\xef"
UPD = {0: (10, 20), 1: (1, 5), 2: (1, 5), 3: (1, 5)}
TARGET = 2                       # device whose payload carries the detector marker


def detector(frames) -> bool:
    return any(MARK in f.data for f in frames)


def make_policy() -> ConstantPolicy:
    return ConstantPolicy("0.3", tau_boost_pi="0.9", boost_epochs=3, detector=detector, detector_name="mark")


@dataclass
class World:
    drand: bc.DrandRounds
    clock: bc.EpochClock
    keys: KeySet
    stream: synth.SyntheticStream
    counters: dict
    eps: object
    gw: Gateway
    disc: object
    ver: Verifier
    rep: object


def run_gateway(drand, clock, keys, batches, policy, counters, eps, withheld=None):
    w = Witness(keys.sk_W)
    gw = Gateway(keys, clock, drand, policy, w, b"escrow-master-secret-for-tests!!")
    gw.register()
    if withheld:
        abort.run_with_withheld_epochs(gw, batches, set(withheld))
    else:
        gw.run(batches)
    disc = build_disclosure(gw, sorted(gw.records), counters=counters)
    trust = TrustAnchors.from_disclosure(disc, keys.pk_W, pk_I=keys.pk_I, beacon=drand)
    ver = Verifier(trust, make_policy(), eps_cell=eps, meta_codec=synth.SyntheticMetaCodec())
    return gw, disc, ver


@pytest.fixture(scope="module")
def world() -> World:
    drand = bc.DrandRounds()
    clock = bc.align_stream(drand, N)
    keys = KeySet.from_master(b"P1-A test_verifier_detects master")
    gw_id = gw_id_from_pk(keys.pk_G)
    eof = synth.epoch_of_us_fn(clock)
    cs = CounterSource(keys.sk_I, gw_id, eof, LossModel(p_loss=0.001), SkewModel(), seed=11)
    stream = synth.synthetic_stream(clock, N, units_per_device=UPD, pkts_per_unit=(1, 30), seed=3,
                                    counter_source=cs, marker=MARK, marker_device=TARGET, marker_rate=0.6)
    counters = cs.records(stream.counter_cells, range(N))
    # ε_cell calibrated on an independent honest stream with the same traffic model (4,000 epochs)
    ts, dev, L = synth.synthetic_packets(clock, 4000, units_per_device=UPD, pkts_per_unit=(1, 30), seed=101)
    sim = CounterSource(None, gw_id, eof, LossModel(p_loss=0.001), SkewModel(), seed=102).simulate(ts, dev, L)
    eps = calibrate_eps(honest_pairs(sim["committed_cells"], sim["counter_cells"]), far=1e-3)
    gw, disc, ver = run_gateway(drand, clock, keys, copy.deepcopy(stream.batches), make_policy(), counters, eps)
    rep = ver.verify(disc)
    return World(drand, clock, keys, stream, counters, eps, gw, disc, ver, rep)


def _fails(rep, check, name=None):
    return rep.failures(check, name)


# ----------------------------------------------------------------------------- honest run

def test_honest_run_zero_fail(world):
    rep = world.rep
    assert rep.n_fail == 0, rep.table(only_non_pass=True)
    assert not rep.warnings("V4", "policy_compliance"), rep.table(only_non_pass=True)
    s = rep.summary()
    for chk in ("V1", "V2", "V3", "V4", "V5", "V6", "V7"):
        assert s[chk].get(PASS, 0) > 0, (chk, s)
    # the scenario exercises boosts, retention and evidence
    assert any(r.state.boost_set for r in world.gw.records.values())
    assert sum(len(d.units) for d in world.disc.epochs.values()) > 20
    assert all(not v.W for v in rep.epochs.values())
    # the store cells equal the simulated gateway capture
    for e in range(N):
        cells = world.gw.records[e].store.cells()
        want = {d: v for (ee, d), v in world.stream.committed_cells.items() if ee == e}
        assert cells == want


def test_v8_hook_and_inputs(world):
    seen = {}

    def hook(rep):
        seen["n"] = len(rep.v8_inputs())
        return "statements-placeholder"

    v = Verifier(world.ver.trust, world.ver.policy, world.eps, synth.SyntheticMetaCodec(), statements_hook=hook)
    rep = v.verify(world.disc)
    assert rep.statements == "statements-placeholder" and seen["n"] == N
    x = rep.v8_inputs()[5]
    assert len(x["tau"]) == len(x["Z"]) == len(x["pay"]) and x["W"] == [] and not x["failed"]


# ----------------------------------------------------------------------------- A_retro

def test_A1_drop_selected_is_missing(world):
    d2, info = tamper.a1_drop_selected(world.disc, random.Random(1))
    rep = world.ver.verify(d2)
    f = _fails(rep, "V6", "missing")
    assert f and f[0].epoch == info["epoch"] and info["unit"] in f[0].evidence["units"]
    assert info["unit"] in rep.epochs[info["epoch"]].W
    assert {r.check for r in rep.failures()} == {"V6"}


def test_A2_extra_and_mismatch(world):
    d2, info = tamper.a2_insert_extra(world.disc, world.gw, random.Random(2))
    rep = world.ver.verify(d2)
    assert _fails(rep, "V6", "extra") and info["unit"] in rep.epochs[info["epoch"]].extra
    for fix in (False, True):
        d3, info3 = tamper.a2_replace_frame(world.disc, random.Random(3), fix_hash_list=fix)
        rep3 = world.ver.verify(d3)
        f = _fails(rep3, "V6", "mismatch")
        assert f and info3["unit"] in rep3.epochs[info3["epoch"]].mismatch
        assert info3["unit"] in rep3.epochs[info3["epoch"]].W
        why = f[0].evidence["units"][info3["unit"]]
        assert ("root_pkt" in why) if fix else ("frame hash" in why)


def test_A3_alter_metadata_breaks_store_hash_and_root(world):
    for field in ("meta", "device_id", "s"):
        d2, info = tamper.a3_alter_metadata(world.disc, random.Random(4), field=field)
        rep = world.ver.verify(d2)
        e = info["epoch"]
        assert any(r.epoch == e for r in _fails(rep, "V4", "store_hash"))
        if field == "meta":
            assert any(r.epoch == e for r in _fails(rep, "V4", "root"))
        assert rep.epochs[e].W_all


def test_A4_swap_theta_at_selection(world):
    d2, info = tamper.a4_swap_theta(world.gw, world.disc, ConstantPolicy("0.1"))
    rep = world.ver.verify(d2)
    assert _fails(rep, "V4", "claimed_taus")
    assert _fails(rep, "V5", "claimed_selection")
    assert _fails(rep, "V6", "missing")
    dropped = sum(len(v["dropped"]) for v in info.values())
    assert dropped > 0 and sum(len(v.missing) for v in rep.epochs.values()) == dropped


def test_A5_alter_lambda_after_beta(world):
    d2, _ = tamper.a5_alter_lambda(world.gw, world.disc, scale=0.25, mode="claim")
    rep = world.ver.verify(d2)
    assert _fails(rep, "V4", "claimed_taus") and _fails(rep, "V6", "missing")
    d3, _ = tamper.a5_alter_lambda(world.gw, world.disc, scale=0.25, mode="header", epochs=[6])
    rep3 = world.ver.verify(d3)
    assert any(r.epoch == 6 for r in _fails(rep3, "V1", "signature"))
    assert any(r.epoch == 6 for r in _fails(rep3, "V1", "cosignature"))
    assert any(r.epoch == 6 for r in _fails(rep3, "V4", "floor"))
    assert any(r.epoch == 7 for r in _fails(rep3, "V1", "chain"))     # H(Hdr_6) no longer links
    assert rep3.epochs[6].W_all


def test_A5_reopen_hiding_state_commitment(world):
    # open C^st_e to another deficit / boost set without touching the signed header
    boosted = [e for e, r in world.gw.records.items() if r.opening.boost_set]
    e_b = boosted[0]
    for kw in ({"D_delta": -5000}, {"boost_remove": world.gw.records[e_b].opening.boost_set}, {"boost_add": (1,)}):
        d2, _ = tamper.reopen_state(world.disc, [e_b], **kw)
        rep = world.ver.verify(d2)
        assert [r.epoch for r in _fails(rep, "V4", "state_opening")] == [e_b], kw
        assert rep.epochs[e_b].W_all
    # the opening also binds H_boost: disclosing other justification records fails boost_opening
    d3 = world.disc.copy()
    d3.epochs[e_b].justifications = ()
    rep3 = world.ver.verify(d3)
    assert [r.epoch for r in _fails(rep3, "V4", "boost_opening")] == [e_b]


def test_A9_cherry_picking(world):
    def dev_of(e, i):
        return world.gw.records[e].batch.units[i].device_id

    d2, dropped = tamper.a9_cherry_pick(world.disc, keep=lambda e, i: dev_of(e, i) != TARGET)
    assert dropped
    rep = world.ver.verify(d2)
    assert _fails(rep, "V6", "missing") and _fails(rep, "V5", "claimed_selection")
    for e, idx in dropped.items():
        assert set(idx) <= rep.epochs[e].missing


def test_A11_selective_abort_both_rules(world):
    # hit labels: the first payload unit of (unboosted, π = 0.3) device 1 in every epoch, so both rules bite
    hits = {}
    for e, r in world.gw.records.items():
        h = [i for i, u in enumerate(r.batch.units) if u.device_id == 1 and u.pay][:1]
        if h:
            hits[e] = h
    for rule in abort.RULES:
        d2, voided = abort.a11_selective_abort(world.disc, world.gw, hits, rule, mode="content")
        assert voided, rule
        rep = world.ver.verify(d2)
        for e in voided:
            S = world.gw.records[e].selected & set(world.disc.epochs[e].requested)
            if S:
                assert S <= rep.epochs[e].missing and S <= rep.epochs[e].W
        assert _fails(rep, "V6", "missing")
        d3, voided3 = abort.a11_selective_abort(world.disc, world.gw, hits, rule, mode="proof")
        rep3 = world.ver.verify(d3)
        assert {r.epoch for r in _fails(rep3, "V3", "vrf")} == voided3
        assert all(rep3.epochs[e].W_all for e in voided3)


# ----------------------------------------------------------------------------- A_rt (pre-commit)

def test_A7_withheld_epoch_is_a_chain_gap(world):
    gw, disc, ver = run_gateway(world.drand, world.clock, world.keys, copy.deepcopy(world.stream.batches),
                                make_policy(), world.counters, world.eps, withheld={7})
    assert 7 not in gw.records
    rep = ver.verify(disc, epochs=range(N))
    f = _fails(rep, "V1", "header_present")
    assert [r.epoch for r in f] == [7] and f[0].evidence["counter_volumes"]
    assert rep.epochs[7].W_all
    # rev1: the gap is attributed to the withheld epoch; epoch 8 stands on its own cosigned header
    w = [r for r in rep.warnings("V1", "chain") if r.epoch == 8]
    assert w and w[0].evidence["gap_before"] == 7
    assert not rep.epochs[8].failed and not rep.epochs[8].W
    assert {r.epoch for r in rep.failures()} == {7}
    # pre-revision semantics: the successor of the gap fails too
    v0 = Verifier(ver.trust, make_policy(), eps_cell=world.eps, meta_codec=synth.SyntheticMetaCodec(), semantics="rev0")
    rep0 = v0.verify(disc, epochs=range(N))
    assert any(r.epoch == 8 for r in _fails(rep0, "V1", "chain")) and rep0.epochs[8].W_all


def test_A8_large_omission_flagged_by_counters(world):
    batches = copy.deepcopy(world.stream.batches)
    removed = omission.omit_packets_in_batches(batches, device=0, epochs={9}, m=10_000, rng=random.Random(5),
                                               include_zero_payload=True)
    n0 = world.stream.counter_cells[(9, 0)][0]
    assert removed[9] > world.eps.pkts(n0)
    gw, disc, ver = run_gateway(world.drand, world.clock, world.keys, batches, make_policy(), world.counters,
                                world.eps)
    rep = ver.verify(disc)
    f = _fails(rep, "V7", "cells")
    assert [r.epoch for r in f] == [9] and f[0].evidence["flagged"][0]["device_id"] == 0
    assert {r.check for r in rep.failures()} == {"V7"}


def test_A10_policy_deviation_boost(world):
    honest = world.gw.records
    boosted = [e for e, r in honest.items() if r.state.boost_set]
    assert boosted and all(1 not in r.state.boost_set for r in honest.values())
    e_f = 10                                    # device 1 is never boosted honestly
    cases = [("forge", [e_f], dict(device=1, fake_justification=False), "boost_set_vs_justifications"),
             ("forge", [e_f], dict(device=1, fake_justification=True), "unjustified_boost"),
             ("omit", [boosted[0]], {}, "omitted_boost")]
    for mode, eps_, kw, rule in cases:
        pol = tamper.BoostDeviationPolicy(make_policy(), mode, eps_, **kw)
        gw, disc, ver = run_gateway(world.drand, world.clock, world.keys, copy.deepcopy(world.stream.batches), pol,
                                    world.counters, world.eps)
        rep = ver.verify(disc)
        w = rep.warnings("V4", "policy_compliance")
        assert w and w[0].epoch == eps_[0], (mode, rule, rep.table(only_non_pass=True))
        assert rule in {d["rule"] for d in w[0].evidence["deviations"]}
        assert rep.n_fail == 0, rep.table(only_non_pass=True)     # harmless to validity (Thm. 2(e))


def test_A12_equal_size_substitution_not_detected(world):
    batches = copy.deepcopy(world.stream.batches)
    info = tamper.a12_equal_size_substitution(batches, TARGET, random.Random(6))
    assert info["frames_rewritten"] > 0
    gw, disc, ver = run_gateway(world.drand, world.clock, world.keys, batches, make_policy(), world.counters,
                                world.eps)
    rep = ver.verify(disc)
    assert rep.n_fail == 0          # documented limit: content authenticity needs attested capture
    assert not rep.warnings("V4", "policy_compliance")


# ----------------------------------------------------------------------------- A6 grinding

def test_A6_grinding_formula():
    assert grinding.grinding_probability(0.05, 10, 1) == pytest.approx(0.401, abs=5e-4)
    assert grinding.grinding_probability(0.05, 10, 5) == pytest.approx(0.0104, abs=5e-5)


def test_A6_grinding_succeeds_without_tr_refused_with_tr():
    drand = bc.DrandRounds()
    clock = bc.align_stream(drand, 400)
    keys = KeySet.from_master(b"P1-A grinding master")
    trials, M = 150, 3
    no_tr = grinding.simulate_grinding(keys, drand, clock, pi="0.05", k=10, M=M, trials=trials, with_tr=False,
                                       seed=1, e_start=0)
    tr = grinding.simulate_grinding(keys, drand, clock, pi="0.05", k=10, M=M, trials=trials, with_tr=True,
                                    seed=2, e_start=200)
    for res in (no_tr, tr):
        p = res.p_formula
        se = (p * (1 - p) / trials) ** 0.5
        assert abs(res.p_hat - p) <= 4 * se + 1e-9, res
    assert no_tr.p_formula == pytest.approx(0.401 ** 3, rel=1e-2)
    assert tr.refusals == trials * (M - 1)              # every late/second header refused
    assert no_tr.p_hat < tr.p_hat                       # grinding lowers retention only without TR
