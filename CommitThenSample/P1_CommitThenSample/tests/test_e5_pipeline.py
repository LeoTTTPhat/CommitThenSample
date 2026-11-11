"""E5 end-to-end pipeline (P1-G; bench/overhead.py + scripts/14_e5.py).

* the virtual merged capture of LP-IoT23(34-1, 1 %) (P1-B's io/merge.py with P1-D's recipe) reproduces the
  stream's units: same keys and enc(m_i) byte for byte except the device-id field, same device MACs;
* re-timing onto the drand grid puts stream epoch 0 on the gateway clock's epoch 0;
* the A_rt rebuild of a unit from its own (unmodified) frames reproduces its enc(m_i) exactly;
* a gateway fork from the persisted honest state with the honest batches reproduces the honest headers
  bit-exactly (so the pre-commit adversary trials differ from the honest run only by the attack);
* frame (de)serialisation and the buffer-occupancy arithmetic.
The data-dependent tests skip when the pcaps / the E5 honest run are absent.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ctsample.bench import overhead as O
from ctsample.units.batch import Frame, Unit

ROOT = Path(__file__).resolve().parents[1]
STREAM = "lp_iot23_34-1_p10000"
RUN = ROOT / "results" / "E5_rev1" / "runs" / "honest_p0.001_s100"     # rev1 run (committed τ_i); the fork
# re-commits with the current protocol, so it must be compared with a run of the same protocol revision


def _e5():
    spec = importlib.util.spec_from_file_location("e5", ROOT / "scripts" / "14_e5.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _have_stream() -> bool:
    try:
        vc = O.virtual_capture(STREAM)
    except Exception:                                   # noqa: BLE001
        return False
    return all(Path(p.path).exists() for p in vc.parts)


def test_frames_roundtrip():
    fr = [Frame(1_672_531_200_000_001, 0, 5, 5, b"abcde"), Frame(1_672_531_200_000_009, 1, 3, 60, b"xyz")]
    u = Unit(epoch=0, index=3, key=b"k" * 37, device_id=1, first_ts_us=0, last_ts_us=0, pkts=(1, 1),
             payload_bytes=(1, 1), frame_bytes=(5, 3), s=40, pay=True, fps=True, flow_uid=b"", meta=b"", frames=fr)
    z = Unit(epoch=0, index=4, key=b"k" * 37, device_id=1, first_ts_us=0, last_ts_us=0, pkts=(1, 0),
             payload_bytes=(0, 0), frame_bytes=(5, 0), s=21, pay=False, fps=False, flow_uid=b"", meta=b"")
    back = O.read_frames(O.write_frames([u, z]))
    assert list(back) == [3] and back[3] == fr


def test_buffer_occupancy_arithmetic():
    t0 = 1_000_000_000
    T = 30_000_000
    ts = np.array([t0 + 1, t0 + 10_000_000, t0 + T + 5])            # two packets in epoch 0, one in epoch 1
    b = np.array([100.0, 50.0, 7.0])
    r1 = O.buffer_occupancy(ts, b, t0, 2, delta_c=3.0, delta_sel=0.5, ahead=1, grid_s=0.5)
    assert r1["max_bytes"] == 157.0                                # epoch 0 waits until 33.5 s; epoch 1 arrived
    assert abs(r1["residence_max_s"] - 33.5) < 1e-9
    r2 = O.buffer_occupancy(ts, b, t0, 2, ahead=2, grid_s=0.5)
    assert abs(r2["residence_max_s"] - 63.5) < 1e-9 and r2["max_bytes"] == 157.0


@pytest.mark.skipif(not _have_stream(), reason="E5 source pcaps not available")
def test_virtual_capture_reproduces_stream_units():
    e5 = _e5()
    vc = O.virtual_capture(STREAM)
    assert [p.role for p in vc.parts] == ["background", "attack"]
    assert vc.parts[1].ip_map == {"192.168.1.195": "192.168.137.234"}
    cmp_ = e5.StreamCompare(STREAM)
    r = O.unitise(vc, on_batch=cmp_, keep_frames=False, track=True, max_epoch=12)
    assert cmp_.n_units > 2000 and cmp_.n_bad == 0
    dep = O.make_deployment(b"test", 870)
    shift = O.align_shift(dep.clock, vc)
    assert vc.with_shift(shift).t0_gw_us == dep.clock.t_e_us(0)
    # per-packet tracking: every packet maps to a unit of its epoch
    ep = (r.pk_ts - vc.t0_gw_us) // 30_000_000
    assert np.all(r.unit_epoch[r.pk_unit] == ep)


@pytest.mark.skipif(not (RUN / "DONE").exists(), reason="E5 honest run not available")
def test_rebuild_and_fork_reproduce_honest_run():
    e5 = _e5()
    ctx = e5.TrialCtx(e5.load_cfg(), RUN)
    e = 300
    view = ctx.view([e, e + 1, e + 2])
    # rebuild of every payload unit from all of its own frames == the committed enc(m_i)
    n = 0
    for u in view.records[e].batch.units:
        if u.pay and u.frames:
            v = e5.rebuild_real_unit(u, list(u.frames))
            assert v.meta == u.meta and v.s == u.s and v.pkts == tuple(u.pkts)
            n += 1
    assert n > 50
    honest = {k: view.records[k].header.digest() for k in (e, e + 1, e + 2)}
    batches = {k: view.records[k].batch for k in (e, e + 1, e + 2)}
    gw = ctx.fork(view, e, e + 2, batches)
    assert {k: gw.records[k].header.digest() for k in (e, e + 1, e + 2)} == honest
    raw = ctx.records([e, e + 1, e + 2])
    for k in (e, e + 1, e + 2):
        assert gw.records[k].taus == raw[k]["taus"] and gw.records[k].retained == raw[k]["retained"]
