"""design/policy_cts.py: the real CtS ThresholdPolicy. Recomputation from the metadata store alone (fresh
policy instance on the verifier side, MetadataStore.to_batch) reproduces every τ_i bit-exactly, through
P1-A's Gateway -> disclosure -> Verifier; the protocol utility equals the batch (experiment) pipeline."""
from __future__ import annotations

import copy
import struct

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from ctsample.crypto.signing import KeySet  # noqa: E402
from ctsample.design import intpi  # noqa: E402
from ctsample.design import utility as U  # noqa: E402
from ctsample.design.policy_cts import CtSPolicy, detect_frames  # noqa: E402
from ctsample.protocol import beacon as bc  # noqa: E402
from ctsample.protocol.disclosure import build_disclosure  # noqa: E402
from ctsample.protocol.epoch import Gateway  # noqa: E402
from ctsample.protocol.policy import PolicyState  # noqa: E402
from ctsample.protocol.store import EncodeMetaCodec, MetadataStore, prepare_batch  # noqa: E402
from ctsample.protocol.verifier import TrustAnchors, Verifier  # noqa: E402
from ctsample.protocol.witness import Witness  # noqa: E402
from ctsample.units import encode as enc  # noqa: E402
from ctsample.units import features as F  # noqa: E402
from ctsample.units.batch import EpochBatch, Frame, Unit  # noqa: E402

from test_features import synth_records  # noqa: E402

ESCROW = b"policy-cts-escrow-master-secret!"
T_US = 30_000_000


# ------------------------------------------------------------------------------------------ #
# synthetic traffic with real enc(m_i) records and real Ethernet/IPv4 frames
# ------------------------------------------------------------------------------------------ #
def _ip4(s):
    return bytes(int(x) for x in s.split("."))


def _frame(smac, dmac, sip, dip, proto, sport, dport, payload, flags=0x18):
    if proto == 6:
        l4 = struct.pack("!HHIIBBHHH", sport, dport, 1, 1, 0x50, flags, 65535, 0, 0)
    else:
        l4 = struct.pack("!HHHH", sport, dport, 8 + len(payload), 0)
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(l4) + len(payload), 0, 0, 64, proto, 0, sip, dip)
    return dmac + smac + b"\x08\x00" + ip + l4 + payload


def synth_batches(n_epochs, seed=0, t0=1_700_000_000_000_000):
    rng = np.random.default_rng(seed)
    devs = [(bytes([2, 0, 0, 0, 1, d]), f"192.168.1.{10 + d}") for d in range(5)]
    gw_mac = bytes([2, 0, 0, 0, 0, 1])
    wan = [f"93.184.{x}.{y}" for x, y in rng.integers(1, 250, (12, 2))]
    services = [(6, 80, b"GET /index.html HTTP/1.1\r\nHost: x\r\n\r\n"), (6, 23, b"login: "), (17, 53, b"\x12\x34" + b"\x00" * 30),
                (6, 443, b"\x16\x03\x01" + bytes(60)), (6, 8080, b"POST /up HTTP/1.1\r\n\r\n/bin/busybox ECCHI"),
                (6, 1883, bytes([0x30, 5, 0, 1, 0x61, 0x7a, 0x7a]))]
    dev_index = {}
    batches = []
    flows_seen = set()
    for e in range(n_epochs):
        units = []
        for _ in range(int(rng.integers(3, 14))):
            d = int(rng.integers(0, len(devs)))
            mac, ip = devs[d]
            dst = wan[int(rng.integers(0, len(wan)))]
            proto, dport, pl = services[int(rng.integers(0, len(services)))]
            scan = rng.random() < 0.25
            sport = int(rng.integers(30000, 60000))
            start = t0 + e * T_US + int(rng.integers(0, 25_000_000))
            frames, ts = [], start
            npk = 1 if scan else int(rng.integers(1, 6))
            for j in range(npk):
                if scan:
                    data = _frame(mac, gw_mac, _ip4(ip), _ip4(dst), 6, sport, dport, b"", flags=0x02)
                    dr = 0
                else:
                    dr = int(j % 2 == 1)
                    body = pl if dr == (1 if dport == 23 else 0) else b"ok" * int(rng.integers(1, 20))
                    if dr == 0:
                        data = _frame(mac, gw_mac, _ip4(ip), _ip4(dst), proto, sport, dport, body)
                    else:
                        data = _frame(gw_mac, mac, _ip4(dst), _ip4(ip), proto, dport, sport, body)
                frames.append(Frame(ts_us=ts, dir=dr, caplen=len(data), wirelen=len(data), data=data))
                ts += int(rng.integers(100, 400_000))
            pay_b = [0, 0]
            for fr in frames:
                pay_b[fr.dir] += len(fr.data) - 14 - 20 - (20 if (proto == 6 or scan) else 8)
            pay = (pay_b[0] + pay_b[1]) > 0 and not scan
            pr = 6 if scan else proto
            key = enc.make_key(pr, b"\x00" * 10 + b"\xff\xff" + _ip4(ip), sport, b"\x00" * 10 + b"\xff\xff" + _ip4(dst), dport)
            dev = dev_index.setdefault(mac, len(dev_index))
            pk = (sum(1 for f in frames if f.dir == 0), sum(1 for f in frames if f.dir == 1))
            fb = (sum(f.caplen for f in frames if f.dir == 0), sum(f.caplen for f in frames if f.dir == 1))
            s = sum(f.caplen + 16 for f in frames)
            fu = bytes(rng.integers(0, 256, 16).tolist())
            fps = pay and fu not in flows_seen
            flows_seen.add(fu)
            sizes = [(len(f.data) - 54) * (1 if f.dir == 0 else -1) for f in frames[:8]] if pay else []
            iats = [enc.iat_code(frames[j + 1].ts_us - frames[j].ts_us) for j in range(min(8, len(frames) - 1))]
            m = {"key": key, "device_id": dev, "first_ts_us": frames[0].ts_us, "last_ts_us": frames[-1].ts_us,
                 "pkts": pk, "payload_bytes": tuple(pay_b) if pay else (0, 0), "frame_bytes": fb,
                 "n_syn": int(scan), "n_synack": 0, "n_fin": 0, "n_rst": 0, "n_psh": 0 if scan else npk,
                 "conn_state": enc.CS_CODE["S0"] if scan else enc.CS_CODE["SF"], "sizes": sizes, "iats": iats, "s": s,
                 "flow_uid": fu, "continuation": False, "fps": fps, "pay": pay}
            units.append(Unit(epoch=e, index=0, key=key, device_id=dev, first_ts_us=frames[0].ts_us,
                              last_ts_us=frames[-1].ts_us, pkts=pk, payload_bytes=tuple(pay_b) if pay else (0, 0),
                              frame_bytes=fb, s=s, pay=pay, fps=fps, flow_uid=fu, meta=enc.enc_metadata(m),
                              frames=frames if pay else None))
        units.sort(key=lambda u: (u.first_ts_us, u.key))
        for i, u in enumerate(units):
            u.index = i
        batches.append(EpochBatch(epoch=e, t_start_us=t0 + e * T_US, units=units))
    return batches


@pytest.fixture(scope="module")
def bundle():
    rng = np.random.default_rng(3)
    ep, metas = synth_records(8000, n_dev=6, n_epochs=150, seed=21)
    c = F.cols_from_meta(metas)
    X = F.features_batch(c, ep)
    qs = ["D5", "D2c", "S6", "L1"]
    hit = {}
    for j, q in enumerate(qs):
        z = -3.0 + 1.5 * (X[:, 1] == (2 if q == "D5" else 0)) + 0.01 * X[:, 6] - 0.2 * j
        y = (rng.random(len(z)) < 1 / (1 + np.exp(-z))).astype(np.int8)
        hit[q], _, _ = U.train_forest(X, y, seed=j, threads=2, n_trees=25)
    gmal, _, _ = U.train_forest(X, (rng.random(len(X)) < 0.1).astype(np.int8), seed=9, threads=2, n_trees=25)
    M = {q: U.m_fix(np.full(U.N_CELLS, 1.0 if U.kind_of(q) == "indicator" else 9.0)) for q in qs}
    P8 = {q: U.p_fix(np.linspace(0.001, 0.3, U.N_STRATA)) for q in qs}
    return U.UtilityBundle(queries=qs, hit=hit, gmal=gmal, M=M, P8=P8)


def _policy(b, **kw):
    qs = b.queries
    args = dict(active=qs, W=U.equal_weights(len(qs)), Q2=U.q2_fix([0.8, 2.5, 0.05, 0.3]), lam_ref=(0.8, 3.0e-4),
                Bbar=6000, phi0="0.02", tau_mix="0.2", controller="realised", boost=True, kappa_b=5, W_b=6)
    args.update(kw)
    return CtSPolicy(b, **args)


@pytest.fixture(scope="module")
def drand():
    return bc.DrandRounds()


def test_gateway_verifier_recompute_bit_exact(bundle, drand):
    n_epochs = 40
    batches = synth_batches(n_epochs, seed=1)
    clock = bc.align_stream(drand, n_epochs)
    keys = KeySet.from_master(b"policy-cts-1")
    pol = _policy(bundle)
    gw = Gateway(keys, clock, drand, pol, Witness(keys.sk_W), ESCROW)
    gw.register()
    for b in batches:                        # re-anchor the synthetic epochs on the beacon clock
        b.t_start_us = clock.t_e_us(b.epoch)
    gw.run(copy.deepcopy(batches))
    taus = [t for r in gw.records.values() for t, u in zip(r.taus, r.batch.units) if u.pay]
    assert len(set(taus)) > 40 and min(taus) >= intpi.floor_tau("0.02")
    boosted = [e for e, r in gw.records.items() if r.state.boost_set]
    assert boosted, "the follow-the-evidence boost never fired"
    disc = build_disclosure(gw, range(n_epochs))
    trust = TrustAnchors.from_disclosure(disc, keys.pk_W, beacon=drand)
    fresh = _policy(bundle)                  # verifier: a new instance, metadata store only
    rep = Verifier(trust, fresh, meta_codec=EncodeMetaCodec()).verify(disc)
    v4 = [r for r in rep.results if r.check == "V4" and r.name in ("thresholds", "claimed_taus")]
    assert v4 and all(r.status == "PASS" for r in v4), rep.table(only_non_pass=True)
    for e, v in rep.epochs.items():
        assert v.taus == gw.records[e].taus, e
    assert rep.n_fail == 0, rep.table(only_non_pass=True)
    warns = [r for r in rep.results if r.status == "WARN" and r.name == "policy_compliance"]
    assert not warns, warns                   # honest + complete disclosure: boost completeness holds
    # registration is computed once per policy instance
    assert fresh.registration() is fresh.registration()


def test_store_only_recompute_with_priming_and_flood_guard(bundle):
    batches = synth_batches(30, seed=2)
    pol = _policy(bundle, controller="expected")
    st = pol.initial_state(0)
    taus, states = {}, {}
    for b in batches:
        states[b.epoch] = st
        taus[b.epoch] = pol.thresholds(b, st)
        rng = np.random.default_rng(b.epoch)
        Z = [bool(rng.random() < t / 2 ** 64) for t in taus[b.epoch]]
        rec = type("R", (), {"e": b.epoch, "batch": b, "taus": taus[b.epoch], "Z": Z,
                             "retained": [i for i, z in enumerate(Z) if z]})()
        st = pol.update(st, rec)
    codec = EncodeMetaCodec()
    store_b = {b.epoch: MetadataStore.from_batch(prepare_batch(b)).to_batch(b.t_start_us, codec=codec)
               for b in batches}
    e = 25
    v = _policy(bundle, controller="expected")
    with pytest.raises(ValueError):
        v.initial_state(0)
        v.thresholds(store_b[e], states[e])            # window (e-W, e) not seen
    v2 = _policy(bundle, controller="expected")
    v2.initial_state(0)
    v2.prime([store_b[k] for k in range(e)])
    assert v2.thresholds(store_b[e], states[e]) == taus[e]
    # flood guard: floor_mode = 1 caps the floor of the heaviest devices, recomputation stays exact
    g = _policy(bundle, controller="expected", Bbar=40)          # small budget: device epoch bytes exceed the cap
    g.initial_state(0)
    g.prime([batches[k] for k in range(e)])
    s_fm = PolicyState(e=e, D_prev=-10 ** 9, floor_mode=1, Lam_A=states[e].Lam_A, Lam_S=states[e].Lam_S)
    t1 = g.thresholds(batches[e], s_fm)
    tf = g.floor_taus(batches[e], s_fm)
    pay = [u.pay for u in batches[e].units]
    assert min(t for t, p in zip(tf, pay) if p) < g.tau_phi and all(t <= g.tau_phi for t in tf)
    assert all(t >= f for t, f, p in zip(t1, tf, pay) if p)
    g2 = _policy(bundle, controller="expected", Bbar=40)
    g2.initial_state(0)
    g2.prime([store_b[k] for k in range(e)])
    assert g2.thresholds(store_b[e], s_fm) == t1


def test_protocol_utility_equals_batch_pipeline(bundle):
    batches = synth_batches(25, seed=4)
    pol = _policy(bundle, boost=False)
    st = pol.initial_state(0)
    a_proto = []
    for b in batches:
        a_proto += pol.utilities(b, st)
    allu = [u for b in batches for u in b.units]
    c = F.cols_from_units(allu)
    ep = np.array([u.epoch for u in allu])
    X = F.features_batch(c, ep)
    u = bundle.utilities(c, X, pol.active, pol.Wq, pol.Q2, which=("U5",))
    assert np.array_equal(np.asarray(a_proto, dtype=np.int64), u["U5"])


def test_detector_on_frames():
    fr = _frame(bytes(6), bytes(6), _ip4("192.168.1.2"), _ip4("1.2.3.4"), 6, 40000, 80, b"GET / HTTP/1.1\r\n\r\n")
    f = [Frame(ts_us=0, dir=0, caplen=len(fr), wirelen=len(fr), data=fr)]
    assert detect_frames(f, 6, 40000, 80) == "D5"
    assert detect_frames(f, 6, 40000, 80, ("D3",)) is None


def _gateway(bundle, drand, n_epochs, seed, policy):
    batches = synth_batches(n_epochs, seed=seed)
    clock = bc.align_stream(drand, n_epochs)
    keys = KeySet.from_master(b"policy-cts-omit-%d" % seed)
    gw = Gateway(keys, clock, drand, policy, Witness(keys.sk_W), ESCROW)
    gw.register()
    for b in batches:
        b.t_start_us = clock.t_e_us(b.epoch)
    gw.run(copy.deepcopy(batches))
    return gw, keys


def test_omitted_boost_full_vs_partial_disclosure(bundle, drand):
    """A10-omit: the gateway commits an empty boost set at an epoch whose honest state boosts a device.
    Full disclosure -> V4 policy_compliance WARN (omitted_boost / boost_set_incomplete); partial disclosure
    without the justifying epochs -> PASS with the note 'boost completeness not checkable (partial disclosure)'."""
    from ctsample.adversary.tamper import BoostDeviationPolicy
    n = 40
    gw_h, _ = _gateway(bundle, drand, n, 3, _policy(bundle))
    target = next(e for e in range(25, n - 3) if gw_h.records[e].state.boost_set)
    gw_a, keys = _gateway(bundle, drand, n, 3, BoostDeviationPolicy(_policy(bundle), "omit", epochs=[target]))
    assert gw_a.records[target].state.boost_set == () and gw_h.records[target].state.boost_set
    # full disclosure of every epoch
    disc = build_disclosure(gw_a, range(n))
    trust = TrustAnchors.from_disclosure(disc, keys.pk_W, beacon=drand)
    rep = Verifier(trust, _policy(bundle), meta_codec=EncodeMetaCodec()).verify(disc)
    w = [r for r in rep.results if r.status == "WARN" and r.name == "policy_compliance" and r.epoch == target]
    assert w, rep.table(only_non_pass=True)
    rules = {d["rule"] for d in w[0].evidence["deviations"]}
    assert rules & {"omitted_boost", "boost_set_incomplete"}
    # partial disclosure: epochs target-1 .. target+1, verifier primed with the earlier stores (metadata only)
    win = [target - 1, target, target + 1]
    disc2 = build_disclosure(gw_a, win)
    v = _policy(bundle)
    v.initial_state(0)
    v.prime([gw_a.records[k].store.to_batch(gw_a.records[k].batch.t_start_us, codec=EncodeMetaCodec())
             for k in range(0, target - 1)])
    rep2 = Verifier(TrustAnchors.from_disclosure(disc2, keys.pk_W, beacon=drand), v,
                    meta_codec=EncodeMetaCodec()).verify(disc2)
    pc = [r for r in rep2.results if r.name == "policy_compliance" and r.epoch == target]
    assert pc and pc[0].status == "NCHK", rep2.table(only_non_pass=True)        # rev1: "not checkable"
    assert rep2.n_fail == 0, rep2.table(only_non_pass=True)
    notes = pc[0].evidence.get("notes", [])
    assert any(n_.get("rule") == "boost_completeness_not_checkable" for n_ in notes), pc[0].evidence


def test_flood_guard_committed_floor_check(bundle, drand):
    """rev1: in floor mode 1 the committed τ of a flooding device may lie below τ^φ, down to the flood-guard cap,
    which V4 recomputes from the epoch's OWN store (no rolling history); τ below the cap FAILs."""
    from ctsample.adversary.tamper import a5c_commit_scaled
    n = 24
    mk = lambda: _policy(bundle, controller="expected", Bbar=40, Dmax_mult=1)      # noqa: E731
    gw, keys = _gateway(bundle, drand, n, 5, mk())
    fm = [e for e, r in gw.records.items() if r.state.floor_mode == 1]
    assert len(fm) > 5
    capped = sum(1 for e in fm for t, u in zip(gw.records[e].taus, gw.records[e].batch.units)
                 if u.pay and t < gw.policy.tau_phi)
    assert capped > 0
    disc = build_disclosure(gw, range(n))
    trust = TrustAnchors.from_disclosure(disc, keys.pk_W, beacon=drand)
    rep = Verifier(trust, mk(), meta_codec=EncodeMetaCodec()).verify(disc)
    assert rep.n_fail == 0, rep.table(only_non_pass=True)
    nc = sum(r.evidence.get("n_capped", 0) for r in rep.results if r.name == "committed_taus")
    assert nc == capped
    # without the prime history (window e ≥ 3 only) validity still holds: the cap needs only the epoch's store
    win = [fm[-3], fm[-2], fm[-1]]
    d2 = build_disclosure(gw, win)
    rep2 = Verifier(TrustAnchors.from_disclosure(d2, keys.pk_W, beacon=drand), mk(), meta_codec=EncodeMetaCodec()).verify(d2)
    assert rep2.n_fail == 0 and {r.status for r in rep2.results if r.name == "tau_compliance"} == {"NCHK"}
    # committing τ below the cap of floor mode 1 FAILs the epoch
    e_bad = fm[2]
    gw3, keys3 = _gateway(bundle, drand, n, 5, a5c_commit_scaled(mk(), 0.01, [e_bad]))
    d3 = build_disclosure(gw3, range(n))
    rep3 = Verifier(TrustAnchors.from_disclosure(d3, keys3.pk_W, beacon=drand), mk(), meta_codec=EncodeMetaCodec()).verify(d3)
    assert e_bad in {r.epoch for r in rep3.failures("V4", "committed_taus")}


def test_utility_cap_and_rolling_impl(bundle):
    """rev1 flood mitigations of the policy: (i) the incremental rolling statistics give the same τ as the original
    re-sorting implementation; (ii) utility_cap = k scores only the first k payload units of each device per epoch
    (the others get â = 0, i.e. floor / size component: τ ≥ τ^φ still), and is part of H(policy)."""
    batches = synth_batches(30, seed=6)
    pa, pb = _policy(bundle, rolling_impl="incremental"), _policy(bundle, rolling_impl="resort")
    sa, sb = pa.initial_state(0), pb.initial_state(0)
    for b in batches:
        assert pa.thresholds(b, sa) == pb.thresholds(b, sb)
    assert pa.registration() == pb.registration()
    k = 1
    pc = _policy(bundle, utility_cap=k)
    assert pc.registration() != pa.registration()
    pd = _policy(bundle)
    sc, sd = pc.initial_state(0), pd.initial_state(0)
    n_capped = 0
    for b in batches:
        ac, ad = pc.utilities(b, sc), pd.utilities(b, sd)
        seen: dict = {}
        for u, x, y in zip(b.units, ac, ad):
            if not u.pay:
                continue
            seen[u.device_id] = seen.get(u.device_id, 0) + 1
            if seen[u.device_id] <= k:
                assert x == y
            else:
                assert x == 0
                n_capped += 1
        tc = pc.thresholds(b, sc)
        assert all(t >= pc.tau_phi for t, u in zip(tc, b.units) if u.pay)
    assert n_capped > 0
