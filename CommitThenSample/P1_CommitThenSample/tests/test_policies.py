"""baselines/policies.py: every baseline is budget-matched (expected spend = B̄E for randomised policies,
within ±1 % for deterministic ones) and follows its retention rule."""
from __future__ import annotations

import numpy as np
import pytest

from ctsample.baselines import policies as B
from ctsample.sim.tables import StreamTables


@pytest.fixture(scope="module")
def stream():
    rng = np.random.default_rng(0)
    E, N = 300, 30000
    epoch = np.sort(rng.integers(0, E, N))
    pay = rng.random(N) < 0.75
    s = np.where(pay, rng.lognormal(6.5, 1.3, N).round() + 60, rng.integers(60, 200, N)).astype(float)
    dev = rng.integers(0, 12, N)
    b = np.where(pay, 1.0, 0.0)[:, None]
    hu = np.nonzero(pay & (rng.random(N) < 0.02))[0]
    T = StreamTables.from_arrays(s=s, pay=pay, epoch=epoch, device=dev, b=b, hits=[(hu, np.ones(len(hu)))],
                                 query_names=["q"], query_kinds=["indicator"], n_epochs=E)
    flow = rng.integers(0, 4000, N)
    ts = epoch * 30_000_000 + rng.integers(0, 30_000_000, N)
    ts = np.sort(ts)
    score = 1 / (1 + np.exp(-(rng.normal(-3, 1.5, N) + 2.5 * np.isin(np.arange(N), hu))))
    sev = np.where(rng.random(N) < 0.003, rng.integers(1, 4, N), 0)
    return T, flow, ts, score, sev


@pytest.mark.parametrize("rho", [0.01, 0.05, 0.10])
def test_budget_matching(stream, rho):
    T, flow, ts, score, sev = stream
    s = np.asarray(T.s)
    pay = np.asarray(T.pay)
    _, Btot = B.budget(T, rho)
    smax = s[pay].max()
    r2 = B.b2_ring_buffer(T, rho)
    assert abs(r2["cost"] / Btot - 1) < 1e-9 and r2["mask_any"].sum() <= r2["mask"].sum() + 1
    m = np.nonzero(r2["mask"])[0]
    assert np.all(np.diff(m) >= 0) and (not len(m) or np.all(r2["mask"][m[0]:] == pay[m[0]:]))  # a suffix
    r3 = B.b3_time_machine(T, rho, flow)
    assert abs(r3["spend_ratio"] - 1) < 0.01
    N_cut = r3["params"]["N_bytes"]
    kept = r3["retained_bytes"]
    for f in np.unique(flow)[:200]:                       # per connection: min(total bytes, N)
        sel = (flow == f) & pay
        assert abs(kept[sel].sum() - min(s[sel].sum(), N_cut)) < 1e-6 * max(1.0, N_cut)
    r4 = B.b4_topk_pmal(T, rho, score)
    assert abs(r4["spend_ratio"] - 1) <= 0.01
    assert not np.any(r4["mask"] & ~pay)
    r13 = B.b13_topk_reserve(T, rho, score, gamma=0.3)
    rc = B.b4_controller(T, rho, score)
    assert 0.5 < rc["spend_ratio"] < 1.5 and not np.any(rc["mask"] & ~pay)
    top = r4["mask"]
    thr = r4["params"]["threshold"]                       # rank prefix: every unit scoring above it is kept
    assert np.all(top[pay & (score > thr)])
    assert abs(r4["spend_ratio"] - 1) <= 0.01
    assert abs(r13["cost"] / Btot - 1) < 1e-9
    assert set(np.unique(r13["pi"][pay]).round(12)) <= {1.0, round(r13["params"]["pi_reserve"], 12)}
    r6 = B.b6_alert_triggered(T, rho, sev, flow, ts)
    assert abs(r6["cost"] / Btot - 1) < 1e-9 or r6["params"]["pi_leftover"] in (0.0, 1.0)
    trig = r6["mask_triggered"]
    assert np.all(r6["pi"][trig] == 1.0) and not np.any(trig & ~pay)
    for f in (B.b8_uniform(T, rho), B.b10_size(T, rho), B.b15_packet_sampling(T, rho, np.full(T.n_units, 5))):
        assert abs(f["cost"] / Btot - 1) < 1e-6, f["name"]
        assert np.all(f["pi"][~pay] == 0)


def test_alert_trigger_rule(stream):
    T, flow, ts, score, sev = stream
    sev2 = np.zeros(T.n_units, int)
    pay = np.asarray(T.pay)
    i = int(np.nonzero(pay)[0][500])
    sev2[i] = 1
    r = B.b6_alert_triggered(T, 0.05, sev2, flow, ts)
    trig = r["mask_triggered"]
    dev = np.asarray(T.device)
    want = pay & ((flow == flow[i]) | ((dev == dev[i]) & (np.abs(ts - ts[i]) <= 60_000_000)))
    assert np.array_equal(trig, want)


def test_threshold_controller_tracks_budget(stream):
    T, flow, ts, score, sev = stream
    Bbar, Btot = B.budget(T, 0.05)
    mask, tot, info = B.threshold_controller_pass(T, score, Bbar, logit_ref=-1.0)
    assert info["max_abs_D_over_Bbar"] < 150            # the controller keeps the deficit bounded
    assert 0.5 < tot / Btot < 1.5


def test_topk_matches_budget_with_ties(stream):
    """u16-quantised scores create large ties; the tie-break still matches the budget within ±1 %."""
    T, flow, ts, score, sev = stream
    q = np.round(score * 65536) / 65536 + 2 ** -16
    for rho in (0.02, 0.05):
        r = B.b4_topk_pmal(T, rho, q)
        assert abs(r["spend_ratio"] - 1) <= 0.01, r["spend_ratio"]
