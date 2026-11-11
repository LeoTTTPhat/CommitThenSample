"""Counter source I (DESIGN §6.3): capture-loss and clock-skew model, ε_cell(n) calibration at
FAR = 1e-3, signed counter records, A8 detection curves."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ctsample.adversary import omission
from ctsample.crypto.signing import KeySet
from ctsample.protocol import beacon as bc
from ctsample.protocol import synth
from ctsample.protocol.counters import (HOUR_US, CounterRecord, CounterSource, LossModel, SkewModel, calibrate_eps,
                                        compare_cells, honest_pairs)

UPD = {0: (10, 20), 1: (1, 5), 2: (1, 5), 3: (1, 5)}


@pytest.fixture(scope="module")
def clock():
    d = bc.DrandRounds()
    return bc.EpochClock(r0=d.first_round, genesis=d.genesis)


def _sim(clock, n_epochs, seed, p_loss):
    ts, dev, L = synth.synthetic_packets(clock, n_epochs, units_per_device=UPD, pkts_per_unit=(1, 30), seed=seed)
    cs = CounterSource(None, b"g" * 16, synth.epoch_of_us_fn(clock), LossModel(p_loss=p_loss), SkewModel(),
                       seed=seed + 1)
    return cs.simulate(ts, dev, L), (ts, dev, L)


def test_gilbert_elliott_burst_statistics():
    rng = np.random.default_rng(1)
    n = 5_000_000
    keep = LossModel(p_loss=0.0, ge_onset=1e-4, ge_mean_burst=50).keep_mask(n, rng)
    lost = ~keep
    # run lengths of lost packets
    d = np.diff(np.concatenate([[0], lost.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    runs = ends - starts
    nb = len(runs)
    assert 300 < nb < 700                                       # ≈ n·1e-4 / (1 + 50e-4) ≈ 497
    se = 50 / nb ** 0.5
    assert abs(runs.mean() - 50) <= 4 * se                      # runs may merge, negligible at 1e-4
    rate = nb / keep.sum()
    assert abs(rate - 1e-4) <= 4 * (1e-4 / nb ** 0.5)
    # i.i.d. part
    keep2 = LossModel(p_loss=0.01, ge_onset=0.0).keep_mask(n, rng)
    assert abs((~keep2).mean() - 0.01) <= 4 * (0.01 * 0.99 / n) ** 0.5


def test_skew_model_per_device_hour():
    rng = np.random.default_rng(2)
    ts = np.concatenate([np.arange(0, 3 * HOUR_US, 1_000_000, dtype=np.int64)] * 3)
    dev = np.repeat(np.arange(3), len(ts) // 3)
    off = SkewModel(50.0).offsets_us(dev, ts, rng)
    assert np.all(np.abs(off) <= 50_000)
    hours = ts // HOUR_US
    for dv in range(3):
        for hr in range(3):
            m = (dev == dv) & (hours == hr)
            assert len(np.unique(off[m])) == 1
    assert len(np.unique(off)) == 9


def test_skew_moves_only_edge_packets(clock):
    eof = synth.epoch_of_us_fn(clock)
    T = 30_000_000
    t0 = clock.t_e_us(0)
    ts = t0 + np.array([10, 20_000, 60_000, 15_000_000, T - 60_000, T - 20_000, T - 10], dtype=np.int64)
    dev = np.zeros(len(ts), dtype=np.int64)
    cs = CounterSource(None, b"g" * 16, eof, LossModel(p_loss=0.0, ge_onset=0.0), SkewModel(50.0), seed=3)
    sim = cs.simulate(ts, dev, np.full(len(ts), 100))
    moved = sim["gw_epoch"] != sim["ctr_epoch"]
    assert not moved[[2, 3, 4]].any()                          # > 50 ms from an edge never moves
    assert np.all(np.abs(sim["ctr_epoch"] - sim["gw_epoch"]) <= 1)
    assert sim["keep"].all()


def test_honest_far_matches_target(clock):
    for p in (0.001, 0.01):
        sim, _ = _sim(clock, 4000, 10, p)
        eps = calibrate_eps(honest_pairs(sim["committed_cells"], sim["counter_cells"]), far=1e-3)
        assert all(a <= b for a, b in zip(eps.eps_pkts, eps.eps_pkts[1:]))       # non-decreasing in n
        sim2, _ = _sim(clock, 4000, 20, p)                                         # independent honest sample
        rows = honest_pairs(sim2["committed_cells"], sim2["counter_cells"])
        flags = np.array([eps.flag(n, dp, db) for n, dp, db in rows])
        far = flags.mean()
        se = (1e-3 * (1 - 1e-3) / len(rows)) ** 0.5
        assert far <= 1e-3 + 4 * se, (p, far, len(rows))


def test_eps_lookup_and_extrapolation(clock):
    sim, _ = _sim(clock, 1500, 30, 0.001)
    eps = calibrate_eps(honest_pairs(sim["committed_cells"], sim["counter_cells"]), far=1e-3)
    assert eps.pkts(eps.n_hi * 4) == pytest.approx(eps.pkts(eps.n_hi) * 4, rel=1e-9)
    assert eps.pkts(0) >= 0
    d = eps.to_dict()
    assert type(eps).from_dict(d).to_dict() == d


def test_detection_curve_and_m_star(clock):
    sim, (ts, dev, L) = _sim(clock, 3000, 40, 0.001)
    eps = calibrate_eps(honest_pairs(sim["committed_cells"], sim["counter_cells"]), far=1e-3)
    sim2, (ts2, dev2, L2) = _sim(clock, 2000, 50, 0.001)
    keys = sorted(set(sim2["committed_cells"]) & set(sim2["counter_cells"]))
    cells = np.array([(sim2["counter_cells"][k][0], sim2["committed_cells"][k][0], sim2["counter_cells"][k][1],
                       sim2["committed_cells"][k][1]) for k in keys], dtype=float)
    mean_b = float(np.mean(L2))
    ms = [0, 2, 5, 10, 20, 40, 80, 160, 320]
    curve = omission.detection_curve(eps, cells, ms, mean_b)
    assert curve[0] <= 0.01                                    # m = 0 is the honest false-alarm rate
    vals = [curve[m] for m in ms]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))   # P(flag) non-decreasing in m
    busy = cells[:, 0] >= 128
    curve_busy = omission.detection_curve(eps, cells[busy], ms, mean_b)
    assert curve_busy[320] > 0.95                              # omitting most of a busy cell is caught
    mstar = omission.m_star_curve(eps, cells, ms, mean_b, [(1, 32), (32, 128), (128, 512)])
    assert mstar[(128, 512)]["m_star"] is not None and mstar[(128, 512)]["m_star"] >= 5
    # small C&C-like omissions (m ≤ 5) stay inside ε_cell for busy cells (DESIGN §6.3 expectation)
    assert curve_busy[5] < 0.5


def test_signed_records_and_compare(clock):
    keys = KeySet.from_master(b"counters")
    sim, _ = _sim(clock, 30, 60, 0.001)
    cs = CounterSource(keys.sk_I, b"G" * 16, synth.epoch_of_us_fn(clock))
    recs = cs.records(sim["counter_cells"], range(30))
    r = recs[5]
    assert r.verify(keys.pk_I)
    assert not r.verify(keys.pk_G)
    c0 = r.cells[0]
    bad = dataclasses.replace(r, cells=(dataclasses.replace(c0, pkts=c0.pkts + 1),) + r.cells[1:])
    assert not bad.verify(keys.pk_I)
    assert not dataclasses.replace(r, e=6).verify(keys.pk_I)
    # the record message does not depend on cell order
    shuffled = CounterRecord(r.gw_id, r.e, tuple(reversed(r.cells)), r.sig)
    assert shuffled.verify(keys.pk_I)
    # compare: honest within ε, omitted device flagged
    big, _ = _sim(clock, 3000, 70, 0.001)
    eps = calibrate_eps(honest_pairs(big["committed_cells"], big["counter_cells"]), far=1e-3)
    committed = {d: v for (e, d), v in sim["committed_cells"].items() if e == 5}
    counter = r.as_dict()
    res = compare_cells(committed, counter, eps)
    assert len(res) == len(set(committed) | set(counter))
    omitted = dict(committed)
    omitted.pop(0)
    res2 = compare_cells(omitted, counter, eps)
    flagged = {c["device_id"] for c in res2 if c["flagged"]}
    assert 0 in flagged
    # cell-level A8 helper
    cc = omission.omit_from_cells(sim["committed_cells"], 0, {5}, 10 ** 6, 700.0)
    assert cc[(5, 0)][0] == 0
