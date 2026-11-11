"""rev1 baselines (baselines/policies.py) and the zero-payload flood cap (protocol/flood.py).

* B4ep / B13ep (online, per-epoch budget with carry): budget-exact, causal (an epoch's choice never depends on later
  epochs), B13's reserve ≥ φ0; B4on / B13on (horizon controller with a given θ_ref); DLT spends B̄E with π ∝ s;
  calibrate_threshold reproduces the pooled budget; byte-level B3 evidence follows before + offset ≤ N.
* zero-payload cap: per-device / total limits, one overflow unit per capped device, per-device packet and byte totals
  (the V7 cells) unchanged, payload units untouched, canonical order, check_cap; the gateway commits a capped epoch
  that verifies.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from ctsample.baselines import policies as B
from ctsample.sim.tables import StreamTables


@pytest.fixture(scope="module")
def stream():
    rng = np.random.default_rng(1)
    E, N = 200, 20000
    epoch = np.sort(rng.integers(0, E, N))
    pay = rng.random(N) < 0.8
    s = np.where(pay, rng.lognormal(6.3, 1.2, N).round() + 60, rng.integers(60, 200, N)).astype(float)
    dev = rng.integers(0, 10, N)
    b = np.where(pay, 1.0, 0.0)[:, None]
    hu = np.nonzero(pay & (rng.random(N) < 0.02))[0]
    T = StreamTables.from_arrays(s=s, pay=pay, epoch=epoch, device=dev, b=b, hits=[(hu, np.ones(len(hu)))],
                                 query_names=["q"], query_kinds=["indicator"], n_epochs=E)
    score = 1 / (1 + np.exp(-(rng.normal(-3, 1.5, N) + 2.5 * np.isin(np.arange(N), hu))))
    return T, score


@pytest.mark.parametrize("rho", [0.02, 0.05])
def test_epoch_budget_baselines(stream, rho):
    T, score = stream
    s, pay = np.asarray(T.s), np.asarray(T.pay)
    Bbar, Btot = B.budget(T, rho)
    r4 = B.b4_epoch(T, rho, score)
    assert r4["cost"] <= Btot * (1 + 1e-12) and r4["spend_ratio"] > 0.97
    assert not np.any(r4["mask"] & ~pay)
    # causality: truncating the stream after epoch 99 does not change the choices of epochs 0..99
    eptr = np.asarray(T.epoch_ptr)
    cut = int(eptr[100])
    T2 = StreamTables.from_arrays(s=s[:cut], pay=pay[:cut], epoch=np.asarray(T.epoch)[:cut],
                                  device=np.asarray(T.device)[:cut], b=np.asarray(T.b)[:cut],
                                  hits=[(np.zeros(0, np.int64), np.zeros(0))], query_names=["q"],
                                  query_kinds=["indicator"], n_epochs=100)
    r4b = B.b4_epoch(T2, rho * T.s.sum() / s[:cut].sum() * 100 / 200, score[:cut])
    assert r4b["mask"].shape[0] == cut                    # same B̄ per epoch → same choices
    assert np.array_equal(r4b["mask"], r4["mask"][:cut])
    r13 = B.b13_epoch(T, rho, score, phi0=0.01, gamma=0.3)
    assert abs(r13["spend_ratio"] - 1) < 0.01
    assert np.all(r13["reserve_pi"][pay] >= 0.01 - 1e-15)
    assert np.all(r13["pi"][r13["mask_top"]] == 1.0)
    rest = pay & ~r13["mask_top"]
    assert np.allclose(r13["pi"][rest], r13["reserve_pi"][rest])
    # controller variants with a given reference threshold
    lg = float(np.log(0.05 / 0.95))
    rc = B.b4_online(T, rho, score, lg)
    assert rc["params"]["calibration"] == "training" and not np.any(rc["mask"] & ~pay)
    rc13 = B.b13_online(T, rho, score, lg, phi0=0.01)
    assert np.all(rc13["pi"][pay] >= 0.01 - 1e-15)


def test_dlt_and_calibration(stream):
    T, score = stream
    s, pay = np.asarray(T.s), np.asarray(T.pay)
    r = B.dlt_threshold(T, 0.05)
    assert abs(r["spend_ratio"] - 1) < 1e-6
    z = r["params"]["z"]
    assert np.allclose(r["pi"][pay], np.minimum(1.0, s[pay] / z))
    # pooled training calibration: the static rule at θ spends (just under) the pooled budget
    sc = [score[pay][:8000], score[pay][8000:]]
    sz = [s[pay][:8000], s[pay][8000:]]
    budgets = [0.05 * s.sum() / 2, 0.05 * s.sum() / 2]
    th = B.calibrate_threshold(sc, sz, budgets)
    spend = sum(float(z_[B._logit(x) >= th].sum()) for x, z_ in zip(sc, sz))
    assert spend <= sum(budgets) + 1e-6 and spend > 0.95 * sum(budgets)


def test_b3_byte_evidence():
    before = np.array([0.0, 100.0, 900.0, 1500.0])
    off = np.array([80.0, 120.0, 120.0, 10.0])
    ev = B.b3_evidence(1000.0, before, off)
    assert ev.tolist() == [True, True, False, False]


# ------------------------------------------------------------------------------------------- flood cap
def _flood_batch(n_scan=50, n_pay=5, dev_scan=3, seed=0, t0=1_700_000_000_000_000):
    from ctsample.bench import flood as FB
    from ctsample.units.batch import EpochBatch
    rng = np.random.default_rng(seed)
    src = bytes(10) + b"\xff\xff" + bytes([192, 168, 1, 9])
    scans = FB.scan_units(FB.scan_records(n_scan, dev_scan, src, t0, rng), 0)
    other = FB.scan_units(FB.scan_records(4, 1, bytes(10) + b"\xff\xff" + bytes([192, 168, 1, 8]), t0, rng), 0)
    pays = FB.payload_flood_units(n_pay, dev_scan, src, t0, rng, 0)
    units = scans + other + pays
    units.sort(key=lambda u: (u.first_ts_us, u.key))
    for i, u in enumerate(units):
        u.index = i
    return EpochBatch(epoch=0, t_start_us=t0, units=units)


def _cells(units):
    out = {}
    for u in units:
        c = out.setdefault(u.device_id, [0, 0])
        c[0] += u.pkts[0] + u.pkts[1]
        c[1] += u.frame_bytes[0] + u.frame_bytes[1]
    return out


def test_zero_payload_cap_aggregates_and_preserves_cells():
    from ctsample.protocol.flood import OVERFLOW_PROTO, cap_level, cap_zero_payload, check_cap, is_overflow_meta
    from ctsample.protocol.store import MetadataStore
    b = _flood_batch()
    capped, info = cap_zero_payload(copy.deepcopy(b), k_dev=10, k_tot=None)
    assert info["capped_devices"] == [3] and info["aggregated"] == 50 - 9
    zero3 = [u for u in capped.units if u.device_id == 3 and not u.pay]
    assert len(zero3) == 10 and sum(is_overflow_meta(u.meta) for u in zero3) == 1
    ov = next(u for u in zero3 if is_overflow_meta(u.meta))
    assert ov.meta[0] == OVERFLOW_PROTO and ov.key[0] == OVERFLOW_PROTO
    assert _cells(capped.units) == _cells(b.units)                     # V7 cells unchanged
    assert [u.meta for u in capped.units if u.pay] == [u.meta for u in b.units if u.pay]
    assert all(u.index == i for i, u in enumerate(capped.units))
    assert [(u.first_ts_us, u.key) for u in capped.units] == sorted((u.first_ts_us, u.key) for u in capped.units)
    st = MetadataStore.from_batch(capped)
    assert check_cap(st, k_dev=10, k_tot=None) == []
    assert check_cap(MetadataStore.from_batch(b), k_dev=10, k_tot=None)[0]["rule"] == "zero_payload_cap"
    # a total cap lowers the level by water filling: the small device (4 units) stays untouched
    assert cap_level({3: 50, 1: 4}, 1000, 20) == 16
    capped2, info2 = cap_zero_payload(copy.deepcopy(b), k_dev=1000, k_tot=20)
    assert info2["level"] == 16 and sum(1 for u in capped2.units if not u.pay) <= 20
    assert sum(1 for u in capped2.units if not u.pay and u.device_id == 1) == 4
    # nothing to cap: the batch is returned unchanged
    same, info3 = cap_zero_payload(b, k_dev=1000, k_tot=None)
    assert same is b and info3["aggregated"] == 0


def test_gateway_commits_capped_epoch_that_verifies():
    from ctsample.crypto.signing import KeySet
    from ctsample.protocol import beacon as bc
    from ctsample.protocol.disclosure import build_disclosure
    from ctsample.protocol.epoch import ConstantPolicy, Gateway
    from ctsample.protocol.store import EncodeMetaCodec
    from ctsample.protocol.verifier import TrustAnchors, Verifier
    from ctsample.protocol.witness import Witness
    drand = bc.DrandRounds()
    clock = bc.align_stream(drand, 2)
    keys = KeySet.from_master(b"flood-cap-test")
    b = _flood_batch(n_scan=40, t0=clock.t_e_us(0))
    gw = Gateway(keys, clock, drand, ConstantPolicy("0.2"), Witness(keys.sk_W), b"flood-cap-escrow-master-secret!!",
                 zero_cap=(8, None))
    gw.register()
    rec = gw.run_epoch(b)
    assert rec.cap and rec.cap["aggregated"] == 40 - 7
    disc = build_disclosure(gw, [0])
    rep = Verifier(TrustAnchors.from_disclosure(disc, keys.pk_W, beacon=drand), ConstantPolicy("0.2"),
                   meta_codec=EncodeMetaCodec()).verify(disc)
    assert rep.n_fail == 0, rep.table(only_non_pass=True)
