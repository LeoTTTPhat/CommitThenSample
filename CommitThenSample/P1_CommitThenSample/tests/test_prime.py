"""protocol/prime.py: the disclosure "prime" bundle (P1-G).

A policy whose τ depends on the committed metadata of the W − 1 preceding epochs (like the CtS utility's
rolling features) cannot be recomputed by V4 from the disclosed epochs alone; with the prime bundle it can,
and every manipulation of the bundle is evident (V4 prime_store FAIL). rev1 (committed τ_i): without the
bundle, or with a failing entry, only the COMPLIANCE check of the dependent epochs is lost (V4 tau_compliance
NCHK) — validity (V1–V6) is unaffected; the pre-revision semantics (Verifier(semantics="rev0")) FAIL V4
'thresholds' of every dependent epoch instead. Synthetic stream (P1-A's protocol/synth.py), real drand rounds,
ECVRF, Ed25519, BLAKE3.
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from ctsample.crypto.signing import KeySet
from ctsample.design import intpi
from ctsample.protocol import beacon as bc
from ctsample.protocol import synth
from ctsample.protocol.disclosure import build_disclosure
from ctsample.protocol.epoch import Gateway
from ctsample.protocol.policy import PolicyState, ThresholdPolicy, boost_digest
from ctsample.protocol.prime import build_prime, check_prime, prime_epochs, verify_with_prime
from ctsample.protocol.verifier import FAIL, TrustAnchors, Verifier
from ctsample.protocol.witness import Witness

N = 16
WIN = [9, 10, 11]
ESCROW = b"prime-test-escrow-master-secret!"


class RollingPolicy(ThresholdPolicy):
    """τ_i of a payload unit depends on its device's unit count in the previous W − 1 epochs (committed
    metadata only); like CtSPolicy it keeps the history itself and raises on an incomplete window."""
    name = "rolling-test"

    def __init__(self, W: int = 4):
        self.W = W
        self._hist: dict[int, list[int]] = {}
        self._e0 = None
        self.tau_phi = intpi.floor_tau("0.05")

    def theta_bytes(self):
        return b"rolling-test-theta"

    def policy_params(self):
        return {"name": self.name, "W": self.W}

    def expected_tau_phi(self):
        return self.tau_phi

    def initial_state(self, e0):
        self._e0 = e0
        return PolicyState(e=e0)

    def set_chain_start(self, e0):
        if self._e0 is None:
            self._e0 = e0

    def commit_fields(self, state):
        H_theta, H_policy = self.registration()
        return {"Lam_A": 0, "Lam_S": 0, "tau_phi": self.tau_phi, "H_theta": H_theta, "H_policy": H_policy,
                "H_boost": boost_digest(state.boost_set, state.justifications), "floor_mode": state.floor_mode,
                "D_prev": state.D_prev}

    def prime(self, batches):
        for b in sorted(batches, key=lambda x: x.epoch):
            if self._e0 is None:
                self._e0 = b.epoch
            self._hist[b.epoch] = [u.device_id for u in b.units]

    def thresholds(self, batch, state):
        e = batch.epoch
        if self._e0 is None:
            self._e0 = e
        lo = max(self._e0, e - self.W + 1)
        missing = [k for k in range(lo, e) if k not in self._hist]
        if missing:
            raise ValueError(f"rolling window incomplete: {missing}")
        cnt = Counter(d for k in range(lo, e) for d in self._hist[k])
        self._hist[e] = [u.device_id for u in batch.units]
        levels = ("0.05", "0.2", "0.5")
        return [intpi.floor_tau(levels[min(2, cnt[u.device_id] // 6)]) if u.pay else 0 for u in batch.units]

    def update(self, state, record):
        return PolicyState(e=state.e + 1)


@pytest.fixture(scope="module")
def world():
    drand = bc.DrandRounds()
    clock = bc.align_stream(drand, N)
    keys = KeySet.from_master(b"P1-G test_prime")
    stream = synth.synthetic_stream(clock, N, units_per_device=(2, 6), pkts_per_unit=(1, 8), seed=5)
    w = Witness(keys.sk_W)
    gw = Gateway(keys, clock, drand, RollingPolicy(), w, ESCROW)
    gw.register()
    gw.run(stream.batches)
    disc = build_disclosure(gw, WIN)
    trust = TrustAnchors.from_disclosure(disc, keys.pk_W, beacon=drand)
    return {"gw": gw, "disc": disc, "trust": trust, "drand": drand, "clock": clock, "keys": keys, "stream": stream}


def _verifier(world, semantics="rev1"):
    return Verifier(world["trust"], RollingPolicy(), meta_codec=synth.SyntheticMetaCodec(), semantics=semantics)


def _v16_fails(rep):
    return [r for r in rep.results if r.status == FAIL and r.check in ("V1", "V2", "V3", "V4", "V5", "V6")]


def test_prime_epochs():
    assert prime_epochs([9, 10, 11], 0, 4) == [6, 7, 8]
    assert prime_epochs([2], 0, 4) == [0, 1]
    assert prime_epochs([5, 9], 0, 4) == [2, 3, 4, 6, 7, 8]
    assert prime_epochs([5, 6], 3, 120) == [3, 4]


def test_without_prime_validity_holds_compliance_not_checkable(world):
    from ctsample.protocol.prime import PrimeBundle
    # rev1, plain verifier (window anchored at the chain start e0 = 0 by set_chain_start): validity from the
    # committed τ, compliance not checkable for every disclosed epoch (each needs epochs it has not seen)
    rep = _verifier(world).verify(world["disc"])
    assert rep.n_fail == 0, rep.table(only_non_pass=True)
    assert {r.epoch for r in rep.not_checkable("V4", "tau_compliance")} == set(WIN)
    assert all(not rep.epochs[e].W for e in WIN) and all(rep.epochs[e].taus == world["gw"].records[e].taus for e in WIN)
    # same with an empty prime bundle
    rep1 = verify_with_prime(_verifier(world), world["disc"], PrimeBundle(W=4))
    assert not _v16_fails(rep1) and {r.epoch for r in rep1.not_checkable("V4", "tau_compliance")} == set(WIN)
    # pre-revision semantics: the window is reported incomplete and every dependent epoch FAILs
    rep2 = verify_with_prime(_verifier(world, "rev0"), world["disc"], PrimeBundle(W=4))
    assert {r.epoch for r in rep2.failures("V4", "thresholds")} == set(WIN)
    assert all(rep2.epochs[e].W_all for e in WIN)


def test_with_prime_everything_passes(world):
    bundle = build_prime(world["gw"], WIN, W=4)
    assert sorted(bundle.entries) == [6, 7, 8]
    rep = verify_with_prime(_verifier(world), world["disc"], bundle)
    assert rep.n_fail == 0, rep.table(only_non_pass=True)
    assert rep.prime["verified"] == 3 and rep.prime["fed"] == [6, 7, 8]
    assert sum(1 for r in rep.results if r.name == "prime_store" and r.status == "PASS") == 3
    assert [r.status for r in rep.results if r.name == "tau_compliance"] == ["PASS"] * 3
    # prime stores are redacted: the committed thresholds of non-requested epochs are not disclosed
    for k, ent in bundle.entries.items():
        assert ent.store.taus is None and ent.store.tau_digest is not None
        assert ent.store.digest() == world["gw"].records[k].header.H_store
    sz = bundle.size_bytes()
    assert sz["store_bytes"] == sum(len(world["gw"].records[k].store.redacted().serialize()) for k in (6, 7, 8))
    assert sz["store_bytes"] < sum(len(world["gw"].records[k].store.serialize()) for k in (6, 7, 8))
    assert sz["header_bytes"] == 3 * (272 + 64 + 72)
    rep0 = verify_with_prime(_verifier(world, "rev0"), world["disc"], bundle)
    assert rep0.n_fail == 0, rep0.table(only_non_pass=True)


def test_tampered_prime_store_is_evident(world):
    bundle = build_prime(world["gw"], WIN, W=4)
    b2 = bundle.copy()
    st = b2.entries[7].store
    r = st.records[0]
    st.records[0] = replace(r, device_id=r.device_id + 1)
    rep = verify_with_prime(_verifier(world), world["disc"], b2)
    bad = [x for x in rep.results if x.name == "prime_store" and x.status == FAIL]
    assert [x.epoch for x in bad] == [7]
    # rev1: epochs 9, 10 need epoch 7 only for the compliance recomputation
    assert {x.epoch for x in rep.not_checkable("V4", "tau_compliance")} >= {9, 10}
    assert [x for x in _v16_fails(rep) if x.name != "prime_store"] == []
    rep0 = verify_with_prime(_verifier(world, "rev0"), world["disc"], b2)
    assert {x.epoch for x in rep0.failures("V4", "thresholds")} >= {9, 10}


def test_missing_or_foreign_prime_entry_is_evident(world):
    bundle = build_prime(world["gw"], WIN, W=4)
    b2 = bundle.copy()
    del b2.entries[8]
    rep = verify_with_prime(_verifier(world), world["disc"], b2)
    assert rep.not_checkable("V4", "tau_compliance") and not _v16_fails(rep)
    assert verify_with_prime(_verifier(world, "rev0"), world["disc"], b2).failures("V4", "thresholds")
    # a header from another gateway (other keys) for epoch 6: σ and ω fail
    other = KeySet.from_master(b"someone else")
    w2 = Witness(other.sk_W)
    gw2 = Gateway(other, world["clock"], world["drand"], RollingPolicy(), w2, ESCROW)
    gw2.register()
    gw2.run(copy.deepcopy(world["stream"].batches[:8]))
    b3 = bundle.copy()
    b3.entries[6] = replace(b3.entries[6], header=gw2.records[6].header, sigma=gw2.records[6].sigma,
                            cosig=gw2.records[6].cosig)
    res, ok = check_prime(b3, world["trust"], world["disc"])
    assert 6 not in ok and {7, 8} <= set(ok)
    # a store relabelled from another epoch
    b4 = bundle.copy()
    b4.entries[6] = replace(b4.entries[6], store=copy.deepcopy(world["gw"].records[5].store))
    res, ok = check_prime(b4, world["trust"], world["disc"])
    assert 6 not in ok


def test_real_cts_window_with_prime_if_available():
    """Real data: a 3-epoch window of the E5 honest run (CtS-AI policy, 119-epoch prime bundle). The rev1 run
    (committed τ_i) is verified with the rev1 semantics, the original rev0 run with the pre-revision semantics
    (TrialCtx picks them from the run's store format)."""
    res = Path(__file__).resolve().parents[1] / "results"
    run = res / "E5_rev1" / "runs" / "honest_p0.001_s100"
    if not (run / "DONE").exists():
        run = res / "E5" / "runs" / "honest_p0.001_s100"
    if not (run / "DONE").exists():
        pytest.skip("E5 honest run not available")
    import importlib.util
    spec = importlib.util.spec_from_file_location("e5", Path(__file__).resolve().parents[1] / "scripts" / "14_e5.py")
    e5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e5)
    ctx = e5.TrialCtx(e5.load_cfg(), run)
    win = [400, 401, 402]
    view = ctx.view(win)
    disc = ctx.disclose(view, win)
    rep, _ = ctx.verify(disc, ctx.prime(win))
    v16 = [r for r in rep.results if r.status == FAIL and r.check in ("V1", "V2", "V3", "V4", "V5", "V6")]
    assert not v16, [r.line() for r in v16]
    assert rep.prime["verified"] == 119
    if ctx.semantics == "rev1":
        assert {r.status for r in rep.results if r.name == "tau_compliance"} == {"PASS"}
        # without the bundle: validity unchanged, compliance not checkable
        rep2, _ = ctx.verify(disc, None)
        assert not [r for r in rep2.results if r.status == FAIL and r.check in ("V1", "V2", "V3", "V4", "V5", "V6")]
        assert {r.status for r in rep2.results if r.name == "tau_compliance"} == {"NCHK"}
