"""Protocol round trips: encodings, disclosure → verifier all PASS, witness/TR behaviour, determinism,
and selection frequencies (ConstantPolicy: P(Z = 1) = π within 4 SE; FixedLambdaPolicy: Σ Z ≈ Σ π)."""
from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from ctsample.crypto.signing import KeySet
from ctsample.design import intpi
from ctsample.protocol import beacon as bc
from ctsample.protocol import synth
from ctsample.protocol.disclosure import build_disclosure
from ctsample.protocol.epoch import ConstantPolicy, FixedLambdaPolicy, Gateway
from ctsample.protocol.header import HDR_LEN, Header, OpenedHeader, StateOpening, sign_header, state_salt
from ctsample.protocol.store import MetadataStore, frames_to_plaintext, plaintext_to_frames
from ctsample.protocol.verifier import FAIL, PASS, TrustAnchors, Verifier
from ctsample.protocol.witness import Witness, WitnessRefusal

ESCROW = b"roundtrip-escrow-master-secret!!"


@pytest.fixture(scope="module")
def drand():
    return bc.DrandRounds()


def _run(drand, policy, n_epochs, seed=0, **kw):
    clock = bc.align_stream(drand, n_epochs)
    keys = KeySet.from_master(b"roundtrip-%d" % seed)
    st = synth.synthetic_stream(clock, n_epochs, seed=seed, **kw)
    w = Witness(keys.sk_W)
    gw = Gateway(keys, clock, drand, policy, w, ESCROW)
    gw.register()
    gw.run(copy.deepcopy(st.batches))
    return clock, keys, st, w, gw


def test_header_store_frame_roundtrips(drand):
    clock, keys, st, w, gw = _run(drand, ConstantPolicy("0.2"), 4, seed=1)
    for rec in gw.records.values():
        h = rec.header
        enc = h.encode()
        assert len(enc) == HDR_LEN == 272
        assert Header.decode(enc) == h and Header.decode(enc).digest() == h.digest()
        ser = rec.store.serialize()
        st2 = MetadataStore.deserialize(ser)
        assert st2.serialize() == ser and st2.digest() == h.H_store and st2.root() == h.R
        for i in rec.retained:
            u = rec.batch.units[i]
            assert plaintext_to_frames(frames_to_plaintext(u.frames)) == u.frames
            assert gw.retention.frames(rec.e, i) == u.frames
    with pytest.raises(ValueError):
        MetadataStore.deserialize(ser + b"x")
    with pytest.raises(ValueError):
        Header.decode(enc[:-1])


def test_disclosure_verifies_all_pass_fixed_lambda(drand):
    pol = FixedLambdaPolicy(lam_A="10", lam_S="0.001", phi0="0.01", tau_mix="0.2")
    clock, keys, st, w, gw = _run(drand, pol, 22, seed=2)
    taus = [t for r in gw.records.values() for t, u in zip(r.taus, r.batch.units) if u.pay]
    assert len(set(taus)) > 50                      # genuinely unit-specific thresholds
    assert min(taus) >= intpi.floor_tau("0.01")
    disc = build_disclosure(gw, range(22))
    trust = TrustAnchors.from_disclosure(disc, keys.pk_W, beacon=drand)
    rep = Verifier(trust, pol, meta_codec=synth.SyntheticMetaCodec()).verify(disc)
    assert rep.n_fail == 0, rep.table(only_non_pass=True)
    assert all(r.status in (PASS, "SKIP") for r in rep.results)
    # V4/V5 recomputation is bit-exact with the gateway
    for e, v in rep.epochs.items():
        assert v.taus == gw.records[e].taus and v.Z == gw.records[e].Z
    # partial, non-contiguous request: still all PASS (context headers carry the chain)
    req = {3: [0, 1, 2], 9: list(range(gw.records[9].batch.n))}
    disc2 = build_disclosure(gw, [3, 9], requested=req)
    assert set(disc2.context) >= {2} | set(range(4, 9))
    rep2 = Verifier(trust, pol, meta_codec=synth.SyntheticMetaCodec()).verify(disc2)
    assert rep2.n_fail == 0, rep2.table(only_non_pass=True)
    # the verifier's own request overrides the discloser's: asking for more units of epoch 3
    # reveals the selected units that were not disclosed
    full3 = set(range(gw.records[3].batch.n))
    rep3 = Verifier(trust, pol).verify(disc2, request={3: full3, 9: req[9]})
    assert rep3.epochs[3].missing == gw.records[3].selected - set(req[3])
    assert bool(rep3.epochs[3].missing) == bool(_fail_names(rep3, "V6"))


def _fail_names(rep, check):
    return [r.name for r in rep.failures(check)]


def test_empty_epochs_get_headers_and_verify(drand):
    clock = bc.align_stream(drand, 6)
    keys = KeySet.from_master(b"empty")
    st = synth.synthetic_stream(clock, 6, seed=5)
    batches = [b for b in copy.deepcopy(st.batches) if b.epoch not in (2, 3)]
    pol = ConstantPolicy("0.5")
    w = Witness(keys.sk_W)
    gw = Gateway(keys, clock, drand, pol, w, ESCROW)
    gw.register()
    gw.run(batches)
    assert gw.records[2].header.n == 0 and gw.records[3].header.n == 0
    disc = build_disclosure(gw, range(6))
    rep = Verifier(TrustAnchors.from_witness(w, gw.gw_id, beacon=drand), pol).verify(disc)
    assert rep.n_fail == 0, rep.table(only_non_pass=True)


def test_witness_timing_rule_and_registry(drand):
    clock = bc.align_stream(drand, 5)
    keys = KeySet.from_master(b"witness")
    pol = ConstantPolicy("0.5")
    w = Witness(keys.sk_W)
    gw = Gateway(keys, clock, drand, pol, w, ESCROW)
    gw.register()
    st = synth.synthetic_stream(clock, 5, seed=6)
    rec = gw.commit(copy.deepcopy(st.batches[0]))
    dl = clock.deadline(0)
    assert dl == clock.t_e(1) + 3.0 - 1.3
    with pytest.raises(WitnessRefusal, match="late"):
        w.cosign(rec.header, rec.sigma, dl + 0.001)
    cos = w.cosign(rec.header, rec.sigma, dl - 0.001)
    assert w.cosign(rec.header, rec.sigma, dl - 0.5) == cos           # identical resubmission
    other = rec.header.replace(n=rec.header.n)                          # same content
    assert w.cosign(other, rec.sigma, dl - 0.4) == cos
    forked = rec.header.replace(C_st=bytes(32))
    with pytest.raises(WitnessRefusal, match="different header"):
        w.cosign(forked, sign_header(keys.sk_G, forked), dl - 0.4)
    with pytest.raises(WitnessRefusal, match="signature"):
        w.cosign(rec.header.replace(e=1), rec.sigma, dl - 0.4)
    bad_r = rec.header.replace(e=1, r_star=rec.header.r_star + 5)
    with pytest.raises(WitnessRefusal, match="r\\*_e"):
        w.cosign(bad_r, sign_header(keys.sk_G, bad_r), clock.t_e(2))
    # policy registration needs ≥ 1 epoch of lead time
    H_t, H_p = pol.registration()
    with pytest.raises(WitnessRefusal, match="too late"):
        w.register_policy(gw.gw_id, H_t, H_p, 3, clock.t_e(3) - clock.period + 0.01)
    reg = w.register_policy(gw.gw_id, H_t, H_p, 3, clock.t_e(3) - clock.period)
    assert reg.verify(keys.pk_W) and w.policy_in_force(gw.gw_id, 3) == reg
    assert w.policy_in_force(gw.gw_id, 2).effective_epoch == 0


def test_trust_anchor_rejects_forged_registration(drand):
    clock, keys, st, w, gw = _run(drand, ConstantPolicy("0.3"), 3, seed=7)
    disc = build_disclosure(gw, range(3))
    other = KeySet.from_master(b"someone else")
    forged = dataclasses.replace(disc.gateway_registration, pk_V=other.pk_V)
    d2 = disc.copy()
    d2.gateway_registration = forged
    with pytest.raises(ValueError):
        TrustAnchors.from_disclosure(d2, keys.pk_W)
    # a verifier holding the wrong VRF key rejects every epoch in V3
    trust = TrustAnchors.from_disclosure(disc, keys.pk_W)
    t2 = dataclasses.replace(trust, pk_V=other.pk_V)
    rep = Verifier(t2, ConstantPolicy("0.3")).verify(disc)
    assert len(rep.failures("V3", "vrf")) == 3


def test_policy_mismatch_and_wrong_beacon_detected(drand):
    clock, keys, st, w, gw = _run(drand, ConstantPolicy("0.3"), 4, seed=8)
    disc = build_disclosure(gw, range(4))
    trust = TrustAnchors.from_disclosure(disc, keys.pk_W, beacon=drand)
    rep = Verifier(trust, ConstantPolicy("0.31")).verify(disc)          # verifier's policy ≠ registered
    assert rep.failures("V1", "policy_registered") and rep.failures("V4", "floor")
    d2 = disc.copy()
    d2.epochs[1].beacon_round = drand.get(d2.epochs[1].header.r_star + 1)  # wrong round
    rep2 = Verifier(trust, ConstantPolicy("0.3")).verify(d2)
    assert [r.epoch for r in rep2.failures("V2", "round")] == [1]


def test_determinism_bit_exact(drand):
    a = _run(drand, ConstantPolicy("0.25"), 5, seed=9)[4]
    b = _run(drand, ConstantPolicy("0.25"), 5, seed=9)[4]
    for e in range(5):
        ra, rb = a.records[e], b.records[e]
        assert ra.header.encode() == rb.header.encode()
        assert ra.vrf_proof == rb.vrf_proof and ra.Z == rb.Z and ra.u == rb.u


def test_selection_frequency_constant_policy_matches_pi(drand):
    pi = "0.05"
    clock, keys, st, w, gw = _run(drand, ConstantPolicy(pi), 120, seed=10, n_devices=10,
                                  units_per_device=(10, 20), pkts_per_unit=(1, 2), payload_len=(1, 40))
    Z = np.array([z for r in gw.records.values() for z, u in zip(r.Z, r.batch.units) if u.pay], dtype=float)
    p = intpi.floor_tau(pi) / intpi.TWO64
    se = (p * (1 - p) / len(Z)) ** 0.5
    assert len(Z) > 10000
    assert abs(Z.mean() - p) <= 4 * se, (Z.mean(), p, se, len(Z))
    # zero-payload units are never selected
    assert not any(z for r in gw.records.values() for z, u in zip(r.Z, r.batch.units) if not u.pay)
    # u words look uniform: KS-type check on the top bits
    u = np.array([x for r in gw.records.values() for x, uu in zip(r.u, r.batch.units) if uu.pay], dtype=np.float64)
    q = np.sort(u / 2.0 ** 64)
    dks = np.max(np.abs(q - (np.arange(1, len(q) + 1) / len(q))))
    assert dks < 1.63 / len(q) ** 0.5                  # KS 1% critical value


def test_selection_sum_matches_sum_pi_fixed_lambda(drand):
    pol = FixedLambdaPolicy(lam_A="10", lam_S="0.001", phi0="0.01", tau_mix="0.2")
    clock, keys, st, w, gw = _run(drand, pol, 80, seed=11, n_devices=10, units_per_device=(10, 20),
                                  pkts_per_unit=(1, 3), payload_len=(1, 400))
    pis = np.array([t / intpi.TWO64 for r in gw.records.values() for t, u in zip(r.taus, r.batch.units) if u.pay])
    Z = np.array([z for r in gw.records.values() for z, u in zip(r.Z, r.batch.units) if u.pay], dtype=float)
    assert pis.std() > 0.01
    sd = float(np.sqrt(np.sum(pis * (1 - pis))))
    assert abs(Z.sum() - pis.sum()) <= 4 * sd, (Z.sum(), pis.sum(), sd)
    # binned calibration: per π-quartile the selection rate matches the mean π
    qs = np.quantile(pis, [0, 0.25, 0.5, 0.75, 1.0])
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (pis >= lo) & (pis <= hi)
        sdm = float(np.sqrt(np.sum(pis[m] * (1 - pis[m]))))
        assert abs(Z[m].sum() - pis[m].sum()) <= 4 * sdm + 1


PCAP = (Path(__file__).resolve().parents[2] / "data" / "iot23" / "pcap" / "CTU-Honeypot-Capture-7-1" / "Somfy-01"
        / "2019-07-03-15-15-47-first_start_somfy_gateway.pcap")


@pytest.mark.skipif(not PCAP.exists(), reason="IoT-23 Honeypot-7-1 pcap not present")
def test_real_pcap_through_flowtable_gateway_verifier(drand):
    """Integration: P1-B's flow table on a real 2 MB IoT-23 pcap → gateway → disclosure → verifier,
    with the real enc(m_i) cross-checked by EncodeMetaCodec (V4 meta_consistency)."""
    ft = pytest.importorskip("ctsample.units.flowtable")
    from ctsample.protocol.store import EncodeMetaCodec
    batches = list(ft.batches_from_pcap(str(PCAP), keep_frames=True))
    assert batches and sum(b.n_pay for b in batches) > 0
    e_last = batches[-1].epoch
    clock = bc.align_stream(drand, e_last + 1)
    keys = KeySet.from_master(b"real-pcap")
    pol = ConstantPolicy("0.05")
    w = Witness(keys.sk_W)
    gw = Gateway(keys, clock, drand, pol, w, ESCROW, e0=batches[0].epoch)
    gw.register()
    gw.run(batches)
    disc = build_disclosure(gw, sorted(gw.records))
    rep = Verifier(TrustAnchors.from_disclosure(disc, keys.pk_W, beacon=drand), pol,
                   meta_codec=EncodeMetaCodec()).verify(disc)
    assert rep.n_fail == 0, rep.table(only_non_pass=True)
    assert rep.summary()["V4"].get(PASS, 0) >= 9 * len(gw.records)        # incl. meta_consistency

    # a policy whose utility reads Unit.payload_bytes / Unit.key directly: the verifier rebuilds those
    # fields from enc(m_i) via the codec, so τ is still recomputed bit-exactly (claimed_taus PASS)
    def util(u):
        return ((sum(u.payload_bytes) % 997) + 1 + (u.key[0] if u.key else 0)) << 8
    pol2 = FixedLambdaPolicy(lam_A="10", lam_S="0.001", phi0="0.01", tau_mix="0.2", utility=util,
                             theta_id=b"payload-bytes-utility")
    batches2 = list(ft.batches_from_pcap(str(PCAP), keep_frames=True))
    keys2 = KeySet.from_master(b"real-pcap-2")
    w2 = Witness(keys2.sk_W)
    gw2 = Gateway(keys2, clock, drand, pol2, w2, ESCROW, e0=batches2[0].epoch)
    gw2.register()
    gw2.run(batches2)
    disc2 = build_disclosure(gw2, sorted(gw2.records))
    ver2 = Verifier(TrustAnchors.from_disclosure(disc2, keys2.pk_W, beacon=drand), pol2, meta_codec=EncodeMetaCodec())
    rep2 = ver2.verify(disc2)
    assert rep2.n_fail == 0, rep2.table(only_non_pass=True)
    assert len({t for r in gw2.records.values() for t in r.taus if t}) > 20
    # without the codec the rebuilt units lack payload_bytes/key and the recomputed τ differ: rev1 reports a
    # compliance deviation (the committed τ stay valid), the pre-revision semantics FAIL the epochs
    rep3 = Verifier(ver2.trust, pol2).verify(disc2)
    assert rep3.warnings("V4", "tau_compliance") and not rep3.failures("V4"), rep3.table(only_non_pass=True)
    rep4 = Verifier(ver2.trust, pol2, semantics="rev0").verify(disc2)
    assert rep4.failures("V4", "thresholds") or rep4.failures("V4", "claimed_taus")


def test_relabelled_epoch_and_unwitnessed_context_rejected(drand):
    clock, keys, st, w, gw = _run(drand, ConstantPolicy("0.3"), 6, seed=12)
    disc = build_disclosure(gw, range(6))
    trust = TrustAnchors.from_disclosure(disc, keys.pk_W, beacon=drand)
    ver = Verifier(trust, ConstantPolicy("0.3"))
    # present epoch 2's (valid) evidence as epoch 4
    d2 = disc.copy()
    d2.epochs[4] = copy.deepcopy(d2.epochs[2])
    rep = ver.verify(d2)
    assert [r.epoch for r in rep.failures("V1", "epoch_label")] == [4] and rep.epochs[4].W_all
    d2.epochs[4].e = 4                                   # relabel the entry too: the header still says 2
    rep = ver.verify(d2)
    assert [r.epoch for r in rep.failures("V1", "epoch_label")] == [4]
    # chain context header signed by G but never cosigned (a fork) is not accepted as the link
    d3 = build_disclosure(gw, [3])
    forked = d3.context[2].header.replace(C_st=b"\x01" * 32)
    d3.context[2] = dataclasses.replace(d3.context[2], header=forked, sigma=sign_header(keys.sk_G, forked))
    rep3 = ver.verify(d3)
    assert any("cosigned" in r.evidence.get("reason", "") or "link" in r.evidence.get("reason", "")
               for r in rep3.failures("V1", "chain"))


def test_prepare_unit_consistency_checks_and_slim_gateway(drand):
    from ctsample.protocol.store import RetentionStore, prepare_unit
    from ctsample.crypto.aead import EpochCipher
    clock = bc.align_stream(drand, 3)
    st = synth.synthetic_stream(clock, 3, seed=13)
    u = copy.deepcopy(next(x for x in st.batches[0].units if x.pay))
    prepare_unit(u)
    bad = copy.deepcopy(u)
    bad.leaf = bytes(32)
    with pytest.raises(ValueError, match="leaf"):
        prepare_unit(bad)
    bad = copy.deepcopy(u)
    bad.root_pkt = bytes(32)
    with pytest.raises(ValueError, match="root_pkt"):
        prepare_unit(bad)
    bad = copy.deepcopy(u)
    f = bad.frames[0]
    bad.frames[0] = type(f)(f.ts_us, f.dir, f.caplen, f.wirelen, bytes(len(f.data)))
    with pytest.raises(ValueError, match="frame-hash list"):
        RetentionStore(EpochCipher(ESCROW)).retain(0, bad)
    # slim gateway: frames dropped after selection, disclosure and verification unaffected
    keys = KeySet.from_master(b"slim")
    pol = ConstantPolicy("0.3")
    w = Witness(keys.sk_W)
    gw = Gateway(keys, clock, drand, pol, w, ESCROW, keep_frames=False)
    gw.register()
    gw.run(copy.deepcopy(st.batches))
    assert all(x.frames is None for r in gw.records.values() for x in r.batch.units)
    disc = build_disclosure(gw, range(3))
    assert sum(len(d.units) for d in disc.epochs.values()) == len(gw.retention.items) > 0
    rep = Verifier(TrustAnchors.from_witness(w, gw.gw_id, beacon=drand), pol).verify(disc)
    assert rep.n_fail == 0, rep.table(only_non_pass=True)


# ----------------------------------------------------------------------------- hiding state commitment C^st_e

def test_state_commitment_layout_binding_and_salt():
    from blake3 import blake3
    import struct
    k_st = bytes(range(32))
    s5 = state_salt(k_st, 5)
    assert s5 == blake3(b"CTS/v1 state salt" + struct.pack("<Q", 5), key=k_st).digest()
    assert len({state_salt(k_st, e) for e in range(1000)}) == 1000          # one fresh salt per epoch
    assert state_salt(k_st, 5) != state_salt(bytes(32), 5)
    op = StateOpening(salt=s5, Lam_A=123, Lam_S=45, D_prev=-678, floor_mode=1, boost_set=(9, 2, 9), H_boost=b"\xaa" * 32)
    assert op.boost_set == (2, 9)                                            # canonical: sorted, unique
    enc = struct.pack("<QQqB", 123, 45, -678, 1) + struct.pack("<III", 2, 2, 9) + b"\xaa" * 32
    assert op.enc() == enc
    assert op.commitment() == blake3(b"CTS/v1/st" + s5 + enc).digest()
    # binding: every field and the salt matter
    for kw in ({"Lam_A": 124}, {"Lam_S": 46}, {"D_prev": -677}, {"floor_mode": 0}, {"boost_set": (2,)},
               {"boost_set": (2, 9, 11)}, {"H_boost": b"\xab" * 32}, {"salt": state_salt(k_st, 6)}):
        assert op.replace(**kw).commitment() != op.commitment(), kw
    # the public header has no state field any more
    for name in ("Lam_A", "Lam_S", "D_prev", "floor_mode", "H_boost", "boost_set"):
        assert name not in Header.__dataclass_fields__


def test_header_hides_state_outsider_cannot_enumerate_boost_set(drand):
    """With the salt (insider) the small boost set is found by enumeration; without it (outsider)
    no candidate matches — an unsalted H(boost set) would have been inverted (paper Thm. 2(d))."""
    import itertools
    pol = ConstantPolicy("0.3", tau_boost_pi="0.9", boost_epochs=3,
                         detector=lambda fr: any(b"MARK" in f.data for f in fr), detector_name="mark")
    clock = bc.align_stream(drand, 14)
    keys = KeySet.from_master(b"hiding")
    st = synth.synthetic_stream(clock, 14, seed=21, marker=b"MARK", marker_device=1, marker_rate=0.9)
    w = Witness(keys.sk_W)
    gw = Gateway(keys, clock, drand, pol, w, ESCROW)
    gw.register()
    gw.run(copy.deepcopy(st.batches))
    rec = next(r for r in gw.records.values() if r.opening.boost_set)
    op, h = rec.opening, rec.header
    devices = range(4)
    cands = [c for k in range(0, 3) for c in itertools.combinations(devices, k)]
    hit_with_salt = [c for c in cands if op.replace(boost_set=c).commitment() == h.C_st]
    assert hit_with_salt == [op.boost_set]
    for guess_salt in (bytes(32), state_salt(b"\x00" * 32, rec.e)):
        assert not [c for c in cands if op.replace(boost_set=c, salt=guess_salt).commitment() == h.C_st]
    # the public header bytes are identical in form for boosted and non-boosted epochs (no state leaks)
    assert all(len(r.header.encode()) == HDR_LEN for r in gw.records.values())
    # OpenedHeader exposes the opened state under the former header attribute names
    oh = OpenedHeader(h, op)
    assert (oh.e, oh.Lam_A, oh.D_prev, oh.floor_mode, oh.boost_set) == (h.e, op.Lam_A, op.D_prev, op.floor_mode,
                                                                         op.boost_set)
    assert oh.digest() == h.digest() and oh.tau_phi == h.tau_phi


def test_reopening_state_commitment_is_detected(drand):
    from ctsample.adversary import tamper
    pol = FixedLambdaPolicy(lam_A="10", lam_S="0.001", phi0="0.01", tau_mix="0.2")
    clock, keys, st, w, gw = _run(drand, pol, 8, seed=22)
    disc = build_disclosure(gw, range(8))
    trust = TrustAnchors.from_disclosure(disc, keys.pk_W, beacon=drand)
    ver = Verifier(trust, pol, meta_codec=synth.SyntheticMetaCodec())
    assert ver.verify(disc).n_fail == 0
    for kw in ({"Lam_scale": 2.0}, {"D_delta": 1000}, {"floor_mode": 1}, {"boost_add": (3,)},
               {"new_salt": bytes(32)}):
        d2, _ = tamper.reopen_state(disc, [4], **kw)
        rep = ver.verify(d2)
        f = rep.failures("V4", "state_opening")
        assert [r.epoch for r in f] == [4], kw
        assert rep.epochs[4].W_all and {r.epoch for r in rep.failures()} == {4}, kw
    # a missing opening is a FAIL too
    d3 = disc.copy()
    d3.epochs[2].opening = None
    rep3 = ver.verify(d3)
    assert [r.epoch for r in rep3.failures("V4", "state_opening")] == [2]
    # A5 via the opening with Λ' and τ' claims: the opening no longer opens C^st_e
    d4, _ = tamper.a5_alter_lambda(gw, disc, scale=0.5, mode="opening", epochs=[5])
    rep4 = ver.verify(d4)
    assert [r.epoch for r in rep4.failures("V4", "state_opening")] == [5]
