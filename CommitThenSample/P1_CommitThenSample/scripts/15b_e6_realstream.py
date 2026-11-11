#!/usr/bin/env python
"""E6 rev1 — commit time on the REAL scan-heavy LP-IoT23 streams (review R2-M1(c) / R3-M4); complements the
synthetic flood grid of `15_e6.py flood`.

The infected host of LP-IoT23(49-1), (7-1) and (48-1) at p = 1 % scans continuously (median 6,160 / 4,646 / 1,604
zero-payload units per epoch, in > 98 % of the epochs). For each stream the virtual capture is unitised (flow table)
for the first `--epochs` epochs and a gateway — the E5 rev1 deployment (keys, clock, drand rounds, witness) and the
E5 CtS policy (fold 34-1 model: the commit cost depends on the traffic, not on the model weights) — commits,
publishes, selects and finishes every epoch exactly as in the E6 pipeline (bench/overhead.py time_pipeline). The
commit times of the epochs whose 120-epoch rolling window is complete (e ≥ 120) are kept, for the reviewed
re-sorting rolling statistics (impl "resort") and the rev1 incremental ones ("incremental"); the protocol output
of both is bit-identical (tests/test_features.py).

Stages:
  run      one (core, stream, impl): results/E6_rev1/realstream/<core>/<stream>_<impl>.json (resumable: skipped if
           present). The core is chosen by the caller (efficiency core: `taskpolicy -b`).
  summary  results/summary/E6_realstream_rev1.json (median / p90 / max commit over the full-window epochs, share
           of epochs over the one-round-ahead build budget Δ_build = 0.8 s, the scanning host's zero-payload units
           per epoch, machine load).

    .venv/bin/python -I -u P1_CommitThenSample/scripts/15b_e6_realstream.py run --core P --stream lp_iot23_49-1_p10000 --impl resort
    .venv/bin/python -I -u P1_CommitThenSample/scripts/15b_e6_realstream.py summary
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from ctsample.bench import overhead as O
from ctsample.utils.io import P1_ROOT, RESULTS, atomic_write_json

OUT = RESULTS / "E6_rev1" / "realstream"
SUMMARY = RESULTS / "summary" / "E6_realstream_rev1.json"
STREAMS = ("lp_iot23_49-1_p10000", "lp_iot23_7-1_p10000", "lp_iot23_48-1_p10000")
IMPLS = ("resort", "incremental")
BUDGET_ONE_AHEAD_S = 0.8
W = 120


def _e6():
    spec = importlib.util.spec_from_file_location("e6_script", P1_ROOT / "scripts" / "15_e6.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.set_rev("rev1")
    return m


def top_zero_payload(units) -> tuple[int, int]:
    """(device, zero-payload units) of the device with the most zero-payload units in the batch."""
    c = collections.Counter(u.device_id for u in units if not u.pay)
    return c.most_common(1)[0] if c else (-1, 0)


def run(core: str, sid: str, impl: str, n_epochs: int = W + 5) -> dict | None:
    from ctsample.protocol.epoch import Gateway
    from ctsample.units import features as F
    out = OUT / core / f"{sid}_{impl}.json"
    if out.exists():
        return None
    e6 = _e6()
    cfg, cfg5 = e6.cfgs()
    _e5, dep, _meta, pol0, _Bbar = e6.deployment_and_policy(cfg5)
    pol = O.fresh_policy(pol0)
    pol.rolling_impl = impl
    pol._rs_cls = F.IncrementalRollingStats if impl == "incremental" else F.RollingDeviceStats
    vc0 = O.virtual_capture(sid)
    vc = vc0.with_shift(O.align_shift(dep.clock, vc0))
    tp = O.TimedPolicy(pol)
    gw = Gateway(dep.keys, dep.clock, dep.beacon, tp, dep.witness(), dep.escrow, keep_frames=False)
    O.register(gw, dep)
    rows = []
    m0 = O.machine_state()

    def on_batch(b, recs):
        if b.epoch >= n_epochs:
            return
        while gw.next_e < b.epoch:
            gw.run_epoch(gw.empty_batch(gw.next_e))
        dev, nz = top_zero_payload(b.units)
        a = time.perf_counter()
        rec = gw.commit(b)
        c = time.perf_counter()
        gw.publish(rec)
        gw.select(rec)
        gw.finish(rec)
        rows.append({"e": int(b.epoch), "n": int(b.n), "n_pay": int(b.n_pay), "top_device": int(dev),
                     "top_device_zero_payload": int(nz), "commit_s": c - a,
                     "thresholds_s": tp.t_thresholds[-1] if tp.t_thresholds else None})

    t0 = time.time()
    r = O.unitise(vc, registry=None, on_batch=on_batch, keep_frames=True, max_epoch=n_epochs)
    full = [x for x in rows if x["e"] >= W]
    res = {"core": core, "stream": sid, "impl": impl, "epochs_run": n_epochs, "window": W, "rows": rows,
           "full_window_epochs": len(full),
           "commit_s_median_full_window": float(np.median([x["commit_s"] for x in full])) if full else None,
           "wall_s": time.time() - t0, "packets": int(r.n_packets), "machine_before": m0,
           "machine_after": O.machine_state(), "omp_threads": os.environ.get("OMP_NUM_THREADS")}
    atomic_write_json(out, res)
    med = res["commit_s_median_full_window"]
    zp = float(np.median([x["top_device_zero_payload"] for x in (full or rows)])) if rows else float("nan")
    print(time.strftime("%H:%M:%S"), f"realstream {core} {sid} {impl}: full-window commit median "
          f"{'--' if med is None else f'{med:.3f}'} s over {len(full)} epochs (scanning host zero-payload "
          f"units/epoch median {zp:.0f}); wall {res['wall_s']:.0f} s; load {m0['loadavg']}", flush=True)
    return res


def summary() -> dict:
    res = {"definition": "Gateway.commit time per epoch on the real LP-IoT23 p = 1 % streams whose infected host scans "
                         "continuously, epochs with a complete 120-epoch rolling window (e >= 120 of the first 125); "
                         "resort = rolling statistics as reviewed (re-sort the window every epoch), incremental = rev1 "
                         "(bit-identical output). P = performance core, E = efficiency core (taskpolicy -b); Python "
                         "prototype, OMP 1 thread; budget = Delta_build = 0.8 s (one round ahead).",
           "budget_one_round_ahead_s": BUDGET_ONE_AHEAD_S, "cores": {}}
    for d in sorted(OUT.glob("*")):
        if not d.is_dir():
            continue
        blk = {}
        for p in sorted(d.glob("*.json")):
            r = json.loads(p.read_text())
            full = [x for x in r["rows"] if x["e"] >= r["window"]]
            cs = np.array([x["commit_s"] for x in full])
            blk.setdefault(r["stream"], {})[r["impl"]] = {
                "epochs": len(full), "commit_s_median": float(np.median(cs)), "commit_s_p90": float(np.quantile(cs, 0.9)),
                "commit_s_max": float(cs.max()), "share_over_budget": float(np.mean(cs > BUDGET_ONE_AHEAD_S)),
                "units_per_epoch_median": float(np.median([x["n"] for x in full])),
                "scanning_host_zero_payload_units_per_epoch_median": float(np.median([x["top_device_zero_payload"]
                                                                                      for x in full])),
                "loadavg_1min_before": r["machine_before"]["loadavg"][0],
                "loadavg_1min_after": r["machine_after"]["loadavg"][0]}
        res["cores"][d.name] = blk
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(SUMMARY, res)
    print(json.dumps(res["cores"], indent=1))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run", "summary"])
    ap.add_argument("--core", default="P")
    ap.add_argument("--stream", choices=STREAMS)
    ap.add_argument("--impl", choices=IMPLS)
    ap.add_argument("--epochs", type=int, default=W + 5)
    a = ap.parse_args(argv)
    if a.stage == "run":
        if not (a.stream and a.impl):
            ap.error("run needs --stream and --impl")
        run(a.core, a.stream, a.impl, a.epochs)
    else:
        summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
