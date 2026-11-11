"""E6 rev1 flood summary (scripts/15_e6.py): the supported-flood interpolation and the per-packet limit source.

* `_max_supported`: log-log interpolation between measured (n, commit s) points is exact for a power law, returns
  None when even the smallest flood misses the budget and a lower bound when the largest still fits;
* `flood_summary`: rows are grouped per (kind, history, impl, cap, ucap); the per-packet pipeline rate of a core comes
  from the rev1 overhead summary when it has that core, else from the rev0 one (labelled);
* scripts/15b_e6_realstream.py: the scanning host's zero-payload count per batch, and the summary uses only the
  epochs with a complete rolling window.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def e6():
    spec = importlib.util.spec_from_file_location("e6_script", ROOT / "scripts" / "15_e6.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_max_supported_interpolation(e6):
    pts = [(1_000, 0.1), (10_000, 1.0), (100_000, 10.0)]          # t ∝ n
    r = e6._max_supported(pts, 0.8)
    assert r["between"] == [1_000, 10_000] and r["n"] == pytest.approx(8_000, rel=1e-9)
    assert e6._max_supported(pts, 30.0) == {"n": 100_000, "note": "largest measured flood still fits (lower bound)"}
    assert e6._max_supported(pts, 0.05)["n"] is None
    assert e6._max_supported([(0, 0.01)], 0.8) is None              # only the no-flood point
    sq = [(1_000, 0.01), (10_000, 1.0)]                              # t ∝ n² → n = 1000 · sqrt(80)
    assert e6._max_supported(sq, 0.8)["n"] == pytest.approx(1_000 * 80 ** 0.5, rel=1e-9)


def _row(kind, history, impl, cap, n, t, ucap=None):
    rep = {"cap_s": 0.0, "store_bytes": 1000, "rolling_window_rows": 10, "cap": ({"aggregated": n} if cap else None)}
    return {"kind": kind, "history": history, "impl": impl, "cap": cap, "ucap": ucap, "n_flood": n, "n_units": n + 238,
            "n_payload": 163, "commit_s_median": t, "commit_plus_cap_s_median": t, "thresholds_s_median": t / 2,
            "reps_rows": [rep], "machine_before": {"loadavg": [1.0, 1.0, 1.0]}}


def test_flood_summary_groups_and_pipeline_source(e6, tmp_path, monkeypatch):
    out, summ = tmp_path / "E6_rev1", tmp_path / "summary"
    d = out / "flood" / "P"
    d.mkdir(parents=True)
    summ.mkdir()
    rows = [_row("zero", "sustained", "incremental", False, n, t) for n, t in ((1_000, 0.02), (10_000, 0.2),
                                                                              (100_000, 2.0))]
    rows += [_row("zero", "sustained", "incremental", True, n, 0.06) for n in (10_000, 100_000, 1_000_000)]
    for i, r in enumerate(rows):
        (d / f"r{i}.json").write_text(json.dumps(r))
    (summ / "E6_overheads.json").write_text(json.dumps({"pipeline": {"P": {"pps_per_core": 50_000.0},
                                                                     "E": {"pps_per_core": 20_000.0}}}))
    (summ / "E6_overheads_rev1.json").write_text(json.dumps({"pipeline": {"P": {"pps_per_core": 80_000.0}}}))
    monkeypatch.setattr(e6, "OUT_REV1", out)
    monkeypatch.setattr(e6, "SUMMARY", summ)
    res = e6.flood_summary()
    sup = res["cores"]["P"]["supported"]
    nocap = sup["zero/sustained/incremental/nocap"]["one_round_ahead"]
    assert nocap["between"] == [10_000, 100_000] and nocap["n"] == pytest.approx(40_000, rel=1e-9)
    # the capped group borrows the uncapped points below 10^4 units and never exceeds the budget
    assert sup["zero/sustained/incremental/cap"]["one_round_ahead"]["n"] == 1_000_000
    lim = res["per_packet_limit"]
    assert lim["P"]["pps_per_core"] == 80_000.0 and lim["P"]["source"].startswith("rev1")
    assert lim["E"]["pps_per_core"] == 20_000.0 and lim["E"]["source"].startswith("rev0")
    assert lim["P"]["cores_needed_at_33k_pps"] == pytest.approx(33_000 / 80_000)
    assert json.loads((summ / "E6_flood.json").read_text())["per_packet_limit"]["P"]["source"].startswith("rev1")


# ------------------------------------------------------------------------------- real-stream commit timing (15b)
@pytest.fixture(scope="module")
def rs():
    spec = importlib.util.spec_from_file_location("e6_realstream", ROOT / "scripts" / "15b_e6_realstream.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_realstream_top_device_and_summary(rs, tmp_path, monkeypatch):
    from types import SimpleNamespace as U
    units = [U(device_id=3, pay=False)] * 5 + [U(device_id=4, pay=False)] * 2 + [U(device_id=3, pay=True)] * 9
    assert rs.top_zero_payload(units) == (3, 5)
    assert rs.top_zero_payload([U(device_id=1, pay=True)]) == (-1, 0)
    out = tmp_path / "realstream"
    (out / "P").mkdir(parents=True)
    rows = [{"e": e, "n": 100, "n_pay": 10, "top_device": 3, "top_device_zero_payload": 90,
             "commit_s": (0.9 if e % 2 else 0.5) if e >= 120 else 5.0, "thresholds_s": 0.1} for e in range(118, 125)]
    m = {"loadavg": [1.0, 1.0, 1.0]}
    (out / "P" / "s_resort.json").write_text(json.dumps({"stream": "s", "impl": "resort", "window": 120, "rows": rows,
                                                         "machine_before": m, "machine_after": m}))
    monkeypatch.setattr(rs, "OUT", out)
    monkeypatch.setattr(rs, "SUMMARY", tmp_path / "E6_realstream_rev1.json")
    r = rs.summary()["cores"]["P"]["s"]["resort"]
    # only the full-window epochs 120..124 count (5.0 s of the partial-window epochs is ignored)
    assert r["epochs"] == 5 and r["commit_s_max"] == 0.9 and r["commit_s_median"] == 0.5
    assert r["share_over_budget"] == pytest.approx(2 / 5) and r["scanning_host_zero_payload_units_per_epoch_median"] == 90
    assert json.loads((tmp_path / "E6_realstream_rev1.json").read_text())["budget_one_round_ahead_s"] == 0.8
