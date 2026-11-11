"""rev1 — committed per-unit thresholds τ_i (review blocking issue 1).

* the metadata store commits τ_i of every payload unit (τ − 1 as u64; zero-payload τ = 0) through
  H(store_e) = BLAKE3("CTS/v1/store2" ‖ BLAKE3(meta_e) ‖ BLAKE3(tau_e)); redacted copies (prime bundles)
  keep the digest without the column; rev0 stores keep the old digest;
* the verifier's validity checks use only the committed τ_i and the epoch-local floor (V4 'committed_taus');
  recomputing τ through the policy is a compliance check (WARN on deviation, NCHK when its rolling history
  is missing);
* containment: a withheld epoch, a voided epoch or a missed commit window affects only that epoch's units
  (and the compliance checks of the epochs whose rolling window contains it), never the selection
  verification of later epochs; the pre-revision semantics (Verifier(semantics="rev0")) fail all of them;
* commit-time deviations (A4c: θ' swap, A5c: scaled τ) stay valid (Thm. 2(e)) and are reported, a τ below
  the floor fails; tampering with a committed τ after commit breaks H(store).
Synthetic stream (protocol/synth.py), real drand rounds, ECVRF, Ed25519, BLAKE3.
"""
from __future__ import annotations

import copy
import random
from collections import Counter

import pytest

from ctsample.adversary import tamper
from ctsample.crypto.signing import KeySet
from ctsample.design import intpi
from ctsample.protocol import beacon as bc
from ctsample.protocol import synth
from ctsample.protocol.disclosure import build_disclosure
from ctsample.protocol.epoch import ConstantPolicy, FixedLambdaPolicy, Gateway
from ctsample.protocol.policy import PolicyState, ThresholdPolicy, boost_digest
from ctsample.protocol.prime import PrimeBundle, build_prime, verify_with_prime
from ctsample.protocol.store import MetadataStore, UnitRecord
from ctsample.protocol.verifier import FAIL, NCHK, PASS, WARN, TrustAnchors, Verifier
from ctsample.protocol.witness import Witness

N = 20
W = 4
ESCROW = b"committed-tau-escrow-master!!!!!"
TWO64 = 1 << 64


class Rolling(ThresholdPolicy):
    """τ_i depends on the device's unit count in the previous W − 1 epochs (committed metadata only), like the
    CtS utility's rolling features; keeps the history itself, anchors at the chain start, raises on an
    incomplete window but remembers every epoch it has seen (as CtSPolicy does)."""
    name = "rolling-tau-test"

    def __init__(self):
        self._hist: dict[int, list[int]] = {}
        self._e0 = None
        self.tau_phi = intpi.floor_tau("0.05")

    def theta_bytes(self):
        return b"rolling-tau-test"

    def policy_params(self):
        return {"name": self.name, "W": W}

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
            self._hist[b.epoch] = [u.device_id for u in b.units]

    def thresholds(self, batch, state):
        e = batch.epoch
        if self._e0 is None:
            self._e0 = e
        lo = max(self._e0, e - W + 1)
        missing = [k for k in range(lo, e) if k not in self._hist]
        self._hist[e] = [u.device_id for u in batch.units]
        if missing:
            raise ValueError(f"rolling window incomplete: {missing}")
        cnt = Counter(d for k in range(lo, e) for d in self._hist[k])
        levels = ("0.05", "0.2", "0.5")
        return [intpi.floor_tau(levels[min(2, cnt[u.device_id] // 6)]) if u.pay else 0 for u in batch.units]

    def update(self, state, record):
        return PolicyState(e=state.e + 1)


@pytest.fixture(scope="module")
def drand():
    return bc.DrandRounds()


def _run(drand, policy, n=N, seed=7, withheld=None, recv_times=None, master=b"committed-tau"):
    clock = bc.align_stream(drand, n)
    keys = KeySet.from_master(master)
    st = synth.synthetic_stream(clock, n, units_per_device=(2, 6), pkts_per_unit=(1, 8), seed=seed)
    w = Witness(keys.sk_W)
    gw = Gateway(keys, clock, drand, policy, w, ESCROW)
    gw.register()
    if withheld:
        # A7 as in the E5 fork: the withheld epoch is processed by the gateway's policy (its history and state
        # continue) but its header is never published; the next header links to Hdr_{e−1} (the gap is hidden)
        for b in copy.deepcopy(st.batches):
            if b.epoch in withheld:
                prev = gw.prev_hash
                rec = gw.commit(b)
                gw.select(rec)
                gw.finish(rec)
                del gw.records[b.epoch]
                gw.prev_hash = prev
            else:
                gw.run_epoch(b)
    else:
        gw.run(copy.deepcopy(st.batches), recv_times=recv_times)
    return clock, keys, gw


def _ver(gw, keys, drand, policy, semantics="rev1", disc=None, **kw):
    disc = disc or build_disclosure(gw, sorted(gw.records))
    trust = TrustAnchors.from_disclosure(disc, keys.pk_W, beacon=drand)
    return Verifier(trust, policy, meta_codec=synth.SyntheticMetaCodec(), semantics=semantics, **kw), disc


def _v16(rep, epochs=None):
    return [r for r in rep.results if r.status == FAIL and r.check in ("V1", "V2", "V3", "V4", "V5", "V6")
            and (epochs is None or r.epoch in epochs)]


# --------------------------------------------------------------------------------------------- store format
def _recs(n=6):
    out = []
    for i in range(n):
        pay = i % 3 != 2
        out.append(UnitRecord(meta=bytes([i]) * 20, pay=pay, root_pkt=bytes([i]) * 32 if pay else None, device_id=i % 2,
                              pkts=(1 + i, i), frame_bytes=(60 * (1 + i), 60 * i), s=76 * (1 + i) + 76 * i,
                              first_ts_us=1000 + i))
    return out


@pytest.mark.parametrize("e", [0, 5, 100, 0x64646464, 2 ** 32 - 1])
def test_store_tau_column_roundtrip_and_digests(e):
    recs = _recs()
    taus = [1, TWO64, 0, 12345, intpi.floor_tau("0.05"), 0]
    st = MetadataStore(e, recs, taus=taus)
    b = st.serialize()
    st2 = MetadataStore.deserialize(b)
    assert st2.taus == taus and st2.digest() == st.digest() and st2.serialize() == b
    legacy = MetadataStore(e, recs)
    assert legacy.digest() != st.digest() and not legacy.committed and st.committed
    assert MetadataStore.deserialize(legacy.serialize()).taus is None
    red = st.redacted()
    assert red.taus is None and red.tau_digest == st.tau_root() and red.digest() == st.digest()
    r2 = MetadataStore.deserialize(red.serialize())        # e = 100 starts with 'd': no marker ambiguity
    assert r2.taus is None and r2.tau_digest == st.tau_root() and r2.digest() == st.digest()
    assert len(red.serialize()) < len(b)
    # the column binds every payload threshold: any change changes H(store)
    t3 = list(taus)
    t3[3] += 1
    assert MetadataStore(e, recs, taus=t3).digest() != st.digest()


def test_store_tau_ranges_enforced():
    recs = _recs()
    for bad in ([0, 1, 0, 1, 1, 0], [1, TWO64 + 1, 0, 1, 1, 0], [1, 1, 5, 1, 1, 0]):
        with pytest.raises(ValueError):
            MetadataStore(0, recs, taus=bad).serialize()
    with pytest.raises(ValueError):
        MetadataStore(0, recs, taus=[1, 1])


# --------------------------------------------------------------------------------------------- honest + tamper
def test_gateway_commits_tau_verifier_uses_committed(drand):
    pol = FixedLambdaPolicy(lam_A="20", lam_S="0.002", phi0="0.02", tau_mix="0.2")
    clock, keys, gw = _run(drand, pol)
    for e, r in gw.records.items():
        assert r.store.taus == r.taus and r.header.H_store == r.store.digest() and r.store.committed
    ver, disc = _ver(gw, keys, drand, FixedLambdaPolicy(lam_A="20", lam_S="0.002", phi0="0.02", tau_mix="0.2"))
    rep = ver.verify(disc)
    assert rep.n_fail == 0, rep.table(only_non_pass=True)
    assert all(v.taus == gw.records[e].taus for e, v in rep.epochs.items())
    assert {r.status for r in rep.results if r.name == "committed_taus"} == {PASS}
    assert {r.status for r in rep.results if r.name == "tau_compliance"} == {PASS}
    # fast mode: no recomputation at all, same verdicts
    ver2, _ = _ver(gw, keys, drand, FixedLambdaPolicy(lam_A="20", lam_S="0.002", phi0="0.02", tau_mix="0.2"),
                   disc=disc, recompute_compliance=False)
    rep2 = ver2.verify(disc)
    assert rep2.n_fail == 0 and {r.status for r in rep2.results if r.name == "tau_compliance"} == {"SKIP"}
    # A3 'tau': raising a committed threshold after commit breaks H(store)
    d3, info = tamper.a3_alter_metadata(disc, random.Random(3), field="tau")
    rep3 = ver.verify(d3)
    assert [r.epoch for r in rep3.failures("V4", "store_hash")] == [info["epoch"]] and rep3.epochs[info["epoch"]].W_all
    # a disclosure that withholds the threshold column of a requested epoch fails
    d4 = disc.copy()
    e4 = sorted(d4.epochs)[3]
    d4.epochs[e4].store = d4.epochs[e4].store.redacted()
    rep4 = ver.verify(d4)
    assert [r.epoch for r in rep4.failures("V4", "committed_taus")] == [e4]


def test_rev0_store_needs_legacy_semantics(drand):
    pol = ConstantPolicy("0.2")
    clock, keys, gw = _run(drand, pol, n=6)
    disc = build_disclosure(gw, sorted(gw.records))
    # emulate a rev0 gateway: stores without the column, headers re-signed and re-cosigned over the old digest
    w = Witness(keys.sk_W)
    from ctsample.protocol.header import sign_header
    gw0 = Gateway(keys, clock, drand, ConstantPolicy("0.2"), w, ESCROW)
    gw0.register()
    d0 = disc.copy()
    prev = None
    for e in sorted(d0.epochs):
        ed = d0.epochs[e]
        ed.store = MetadataStore(ed.store.e, list(ed.store.records))
        h = ed.header.replace(H_store=ed.store.digest(), **({"H_prev": prev} if prev else {}))
        ed.header, ed.sigma = h, sign_header(keys.sk_G, h)
        ed.cosig = w.cosign(h, ed.sigma, clock.t_e(e + 1) + 1.5)
        prev = h.digest()
    d0.gateway_registration, d0.policy_registrations = w.export_registrations(gw.gw_id)
    trust = TrustAnchors.from_disclosure(d0, keys.pk_W, beacon=drand)
    rep1 = Verifier(trust, ConstantPolicy("0.2")).verify(d0, epochs=[2, 3])
    assert {r.epoch for r in rep1.failures("V4", "committed_taus")} == {2, 3}
    rep0 = Verifier(trust, ConstantPolicy("0.2"), semantics="rev0").verify(d0, epochs=[2, 3])
    assert not [r for r in rep0.failures() if r.check in ("V4", "V5")], rep0.table(only_non_pass=True)


# --------------------------------------------------------------------------------------------- containment
def test_withheld_epoch_affects_only_compliance_of_later_epochs(drand):
    clock, keys, gw = _run(drand, Rolling(), withheld={7})
    assert 7 not in gw.records
    disc = build_disclosure(gw, [8, 9, 10, 11, 12])
    bundle = build_prime(gw, [8, 9, 10, 11, 12], W=W)          # epoch 7 is not available
    assert 7 not in bundle.entries
    ver, _ = _ver(gw, keys, drand, Rolling(), disc=disc)
    rep = verify_with_prime(ver, disc, bundle)
    assert not _v16(rep), rep.table(only_non_pass=True)
    assert all(not rep.epochs[e].W and not rep.epochs[e].failed for e in (8, 9, 10, 11, 12))
    gap = [r for r in rep.warnings("V1", "chain") if r.epoch == 8]
    assert gap and gap[0].evidence["gap_before"] == 7
    st = {r.epoch: r.status for r in rep.results if r.name == "tau_compliance"}
    assert st == {8: NCHK, 9: NCHK, 10: NCHK, 11: PASS, 12: PASS}      # window [e−3, e−1] contains 7 for 8–10
    # pre-revision semantics: every epoch whose window contains 7 FAILs, and the successor of the gap too
    ver0, _ = _ver(gw, keys, drand, Rolling(), semantics="rev0", disc=disc)
    rep0 = verify_with_prime(ver0, disc, bundle)
    failed0 = {e for e, v in rep0.epochs.items() if v.failed}
    assert failed0 == {8, 9, 10} and all(rep0.epochs[e].W_all for e in failed0)


def test_voided_epoch_after_beta_contained(drand):
    clock, keys, gw = _run(drand, Rolling(), seed=8)
    disc = build_disclosure(gw, [6, 7, 8, 9])
    d2 = disc.copy()
    del d2.epochs[7]                                            # A11 'epoch' mode: the whole epoch is withheld
    bundle = build_prime(gw, [6, 8, 9], W=W)
    del bundle.entries[7]
    ver, _ = _ver(gw, keys, drand, Rolling(), disc=d2)
    rep = verify_with_prime(ver, d2, bundle, epochs=[6, 7, 8, 9])
    assert {r.epoch for r in rep.failures("V1", "header_present")} == {7} and rep.epochs[7].W_all
    assert not _v16(rep, {6, 8, 9}), rep.table(only_non_pass=True)
    assert {e for e, v in rep.epochs.items() if v.W or v.W_all} == {7}
    ver0, _ = _ver(gw, keys, drand, Rolling(), semantics="rev0", disc=d2)
    rep0 = verify_with_prime(ver0, d2, bundle, epochs=[6, 7, 8, 9])
    assert {e for e, v in rep0.epochs.items() if v.failed} == {7, 8, 9}


def test_missed_commit_window_contained(drand):
    clock = bc.align_stream(drand, N)
    late = {7: clock.t_rstar(7) + 0.5}                          # header arrives after β: the witness refuses it
    clock, keys, gw = _run(drand, Rolling(), recv_times=late, seed=9)
    assert gw.records[7].cosig is None and gw.records[7].refusal
    disc = build_disclosure(gw, [7, 8, 9, 10, 11])
    bundle = build_prime(gw, [7, 8, 9, 10, 11], W=W)
    ver, _ = _ver(gw, keys, drand, Rolling(), disc=disc)
    rep = verify_with_prime(ver, disc, bundle)
    assert {r.epoch for r in rep.failures("V1", "cosignature")} == {7} and rep.epochs[7].W_all
    assert not _v16(rep, {8, 9, 10, 11}), rep.table(only_non_pass=True)
    w8 = [r for r in rep.warnings("V1", "chain") if r.epoch == 8]
    assert w8 and "not witness-cosigned" in w8[0].evidence["reason"]
    ver0, _ = _ver(gw, keys, drand, Rolling(), semantics="rev0", disc=disc)
    rep0 = verify_with_prime(ver0, disc, bundle)
    assert rep0.epochs[8].failed                               # rev0: the successor of the refused header fails


# --------------------------------------------------------------------------------------------- deviations
def test_commit_time_deviations(drand):
    base = lambda: FixedLambdaPolicy(lam_A="20", lam_S="0.002", phi0="0.02", tau_mix="0.2")   # noqa: E731
    dev_epochs = [5, 6]
    cases = [("A4c", lambda b: tamper.a4c_commit_theta(b, ConstantPolicy("0.6"), dev_epochs), WARN),
             ("A5c_keep", lambda b: tamper.a5c_commit_scaled(b, 0.1, dev_epochs, keep_floor=True), WARN),
             ("A5c_low", lambda b: tamper.a5c_commit_scaled(b, 0.1, dev_epochs, keep_floor=False), FAIL)]
    for name, wrap, want in cases:
        clock, keys, gw = _run(drand, wrap(base()), master=name.encode())
        ver, disc = _ver(gw, keys, drand, base())
        rep = ver.verify(disc)
        if want == WARN:
            assert rep.n_fail == 0, (name, rep.table(only_non_pass=True))       # harmless to validity
            assert {r.epoch for r in rep.warnings("V4", "tau_compliance")} == set(dev_epochs), name
        else:
            assert {r.epoch for r in rep.failures("V4", "committed_taus")} == set(dev_epochs), name
            assert all(rep.epochs[e].W_all for e in dev_epochs)
            assert not _v16(rep, set(gw.records) - set(dev_epochs))
        # the selection is recomputed with the committed (deviating) τ: no V5/V6 alarm in any case
        assert not rep.failures("V5") and not rep.failures("V6")


def test_post_beta_claims_still_detected(drand):
    pol = ConstantPolicy("0.3")
    clock, keys, gw = _run(drand, pol, n=10)
    ver, disc = _ver(gw, keys, drand, ConstantPolicy("0.3"))
    d2, info = tamper.a4_swap_theta(gw, disc, ConstantPolicy("0.1"))
    rep = ver.verify(d2)
    assert rep.failures("V4", "claimed_taus") and rep.failures("V5", "claimed_selection") and rep.failures("V6", "missing")
    d3, _ = tamper.a5_alter_lambda(gw, disc, scale=0.25, mode="claim")
    rep3 = ver.verify(d3)
    assert rep3.failures("V4", "claimed_taus") and rep3.failures("V6", "missing")
