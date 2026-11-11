#!/usr/bin/env python
"""E5 — end-to-end protocol security on the E5 stream (DESIGN §6.1 row E5, §6.3; paper Table V `tab:e5`).

Stages (each resumable: outputs are atomic, a stage directory is complete iff it has DONE):

  calib   ε_cell(n) on the CIC TRAINING half (BenignTraffic, 9.26 h): lossless flow-table pass = counter
          source I's view (packets attributed to their unit's device), then per loss/skew seed a LOSSY flow-table
          pass with the pre-seeded device registry = the gateway's committed cells; ε = (1 − FAR) quantile of
          honest |committed − counter| per log2(n+1) bin (protocol/counters.calibrate_eps), FAR = 1e-3.
  honest  the real pipeline end to end on LP-IoT23(34-1, 1 %): merged virtual capture (io/merge.py) re-timed
          onto the drand epoch grid → lossless pass (validated against the stream's units, counter view) →
          loss/skew model (CounterSource, seed) → lossy gateway pass → Gateway (real ECVRF/Ed25519/BLAKE3,
          real drand rounds, witness, CtS-AI policy U6τ0.2 of fold 34-1, ρ = 5 %) → artefacts (bench/overhead.py).
  cells   extra honest loss/skew draws on the E5 stream (lossy passes only): committed vs counter cells for the
          V7 FAR and the A8 detection curves.
  trials  A1–A12 (200 trials each where randomised) on the honest run: detection, FAR, checks fired,
          verification time (protocol/prime.py supplies the 119 preceding stores).
  grind   A6 best-of-M grinding with / without TR, real drand + ECVRF + PRF, vs (1 − (1 − π)^k)^M.
  chi2    real PRF selection vs the Philox MC (P1-C) on the honest run's payload units, χ² over π-bins.
  a8      P(flag) vs omitted packets m and m*(n) (cell level, exact for V7) vs packets per real C&C unit.
  summary results/summary/E5_security.json (every number the paper quotes).

Run from the project root under memguard, e.g.
  .venv/bin/python memguard/memguard.py run --name p1_e5_calib --workers 2 -- \
      .venv/bin/python -I -u P1_CommitThenSample/scripts/14_e5.py calib --p-loss 0.001
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from ctsample.bench import overhead as O
from ctsample.protocol.counters import (CounterSource, EpsCell, LossModel, SkewModel, calibrate_eps,
                                        honest_pairs)
from ctsample.protocol.header import gw_id_from_pk
from ctsample.utils.io import P1_ROOT, RESULTS, atomic_write_bytes, atomic_write_json

CFG_PATH = P1_ROOT / "config" / "experiments" / "E5.yaml"
# rev1 (committed per-unit thresholds): runs, trials, grind, chi2, containment go to results/E5_rev1/; the
# rev0 artefacts stay in results/E5/. calib / cells / a8 do not depend on the threshold commitment (they compare
# committed with counter cells of lossy flow-table passes) and are shared (SHARED = results/E5).
REV = "rev1"
OUT = RESULTS / "E5_rev1"
SHARED = RESULTS / "E5"
SUMMARY = RESULTS / "summary"
T_US = 30_000_000


def set_rev(rev: str) -> None:
    """rev1 (default): results/E5_rev1 + E5_security_rev1.json; rev0: the original results/E5 + E5_security.json."""
    global REV, OUT
    if rev not in ("rev0", "rev1"):
        raise SystemExit(f"--rev must be rev0 or rev1, not {rev}")
    REV = rev
    OUT = RESULTS / ("E5" if rev == "rev0" else "E5_rev1")


def summary_name() -> str:
    return "E5_security.json" if REV == "rev0" else f"E5_security_{REV}.json"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_cfg() -> dict:
    return yaml.safe_load(CFG_PATH.read_text())


def done(d: Path) -> bool:
    return (Path(d) / "DONE").exists()


def mark_done(d: Path, info: dict | None = None) -> None:
    atomic_write_json(Path(d) / "DONE", info or {"time": time.strftime("%Y-%m-%dT%H:%M:%S")})


def save_npz(path: Path, **arrays) -> None:
    import io
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    atomic_write_bytes(path, buf.getvalue())


def loss_model(cfg, p_loss: float) -> LossModel:
    c = cfg["counters"]
    return LossModel(float(p_loss), float(c["ge_onset"]), float(c["ge_mean_burst"]))


def merged_keep(n_total: int, pk_k: np.ndarray, keep_pk: np.ndarray) -> np.ndarray:
    keep = np.ones(n_total, dtype=bool)
    keep[pk_k] = keep_pk
    return keep


def cells_pass(vc, keep, registry, n_epochs: int):
    """Lossy flow-table pass → committed cells {(e, dev): (pkts, frame bytes)} (the store's V7 view)."""
    cells: dict = {}

    def on_batch(b, recs):
        if b.epoch >= n_epochs:
            return
        for u in b.units:
            c = cells.get((b.epoch, u.device_id))
            if c is None:
                c = cells[(b.epoch, u.device_id)] = [0, 0]
            c[0] += u.n_pkts
            c[1] += u.frame_bytes[0] + u.frame_bytes[1]

    res = O.unitise(vc, keep=keep, registry=registry, on_batch=on_batch, keep_frames=False, max_epoch=n_epochs)
    return {k: (v[0], v[1]) for k, v in cells.items()}, res


def pairs_in_range(committed: dict, counter: dict, n_epochs: int) -> np.ndarray:
    c1 = {k: v for k, v in committed.items() if 0 <= k[0] < n_epochs}
    c2 = {k: v for k, v in counter.items() if 0 <= k[0] < n_epochs}
    return honest_pairs(c1, c2)


def cell_table(committed: dict, counter: dict, n_epochs: int) -> np.ndarray:
    """Rows (e, dev, counter_pkts, committed_pkts, counter_bytes, committed_bytes) over the union of cells."""
    keys = sorted(k for k in set(committed) | set(counter) if 0 <= k[0] < n_epochs)
    out = np.zeros((len(keys), 6), dtype=np.int64)
    for j, k in enumerate(keys):
        cp, cb = committed.get(k, (0, 0))
        n_p, n_b = counter.get(k, (0, 0))
        out[j] = (k[0], k[1], n_p, cp, n_b, cb)
    return out


# ============================================================================================== calib
def stage_calib(cfg: dict, p_loss: float) -> None:
    out = SHARED / "calib" / f"p{p_loss:g}"
    if done(out):
        log("calib done", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    c = cfg["counters"]
    ds, src = c["calib_source"]
    vc = O.single_source_capture(ds, src)
    E = vc.n_epochs
    log(f"calib {src}: {E} epochs; lossless pass")
    lossless_path = SHARED / "calib" / "lossless.npz"
    res = O.unitise(vc, keep_frames=False, track=True, max_epoch=E)
    dev = res.unit_dev[res.pk_unit]
    log(f"lossless: {res.n_packets} packets, {len(res.unit_epoch)} units, {res.elapsed_s:.1f} s")
    if not lossless_path.exists():
        save_npz(lossless_path, pk_k=res.pk_k, pk_ts=res.pk_ts, pk_len=res.pk_len, pk_dev=dev,
                 unit_epoch=res.unit_epoch, unit_dev=res.unit_dev, unit_pay=res.unit_pay)
    n_total = int(res.pk_k.max()) + 1

    def epoch_of(t):
        return (np.asarray(t, dtype=np.int64) - vc.t0_gw_us) // T_US

    allp, stats = [], []
    for seed in c["calib_seeds"]:
        sp = out / f"pairs_s{seed}.npz"
        if sp.exists():
            z = np.load(sp)
            allp.append(z["pairs"])
            stats.append(json.loads(str(z["stats"])))
            continue
        cs = CounterSource(None, b"\x00" * 16, epoch_of, loss_model(cfg, p_loss), SkewModel(float(c["skew_ms"])),
                           seed=int(seed))
        sim = cs.simulate(res.pk_ts, dev, res.pk_len)
        keep = merged_keep(n_total, res.pk_k, sim["keep"])
        committed, r2 = cells_pass(vc, keep, res.registry, E)
        pairs = pairs_in_range(committed, sim["counter_cells"], E)
        tab = cell_table(committed, sim["counter_cells"], E)
        new_dev = sorted(set(r2.registry.values()) - set(res.registry.values()))
        st = {"seed": int(seed), "loss_fraction": float(1 - sim["keep"].mean()), "n_cells": int(len(pairs)),
              "n_new_devices_in_lossy_pass": len(new_dev), "lossy_pass_s": r2.elapsed_s}
        save_npz(sp, pairs=pairs, cells=tab, stats=np.array(json.dumps(st)))
        allp.append(pairs)
        stats.append(st)
        log(f"calib p={p_loss:g} seed {seed}: {st}")
    H = np.concatenate(allp)
    eps = calibrate_eps(H, far=float(c["far"]))
    # in-sample FAR (by construction ≤ FAR per bin up to ties) for the record
    fl = np.array([eps.flag(n, dp, db) for n, dp, db in H])
    info = {"p_loss": p_loss, "source": src, "n_epochs": E, "n_pairs": int(len(H)), "seeds": stats,
            "eps": eps.to_dict(), "in_sample_far": float(fl.mean()), "far_target": float(c["far"]),
            "ge_onset": c["ge_onset"], "ge_mean_burst": c["ge_mean_burst"], "skew_ms": c["skew_ms"]}
    atomic_write_json(out / "eps.json", info)
    mark_done(out)
    log("calib written", out / "eps.json", "in-sample FAR", info["in_sample_far"])


def load_eps(p_loss: float) -> EpsCell:
    d = json.loads((SHARED / "calib" / f"p{p_loss:g}" / "eps.json").read_text())
    return EpsCell.from_dict(d["eps"])


# ============================================================================================== honest
def deployment(cfg: dict):
    import json as _j
    m = _j.loads((RESULTS / "streams" / cfg["stream"] / "meta.json").read_text())
    E = int(m["n_epochs"])
    return O.make_deployment(cfg["master_key"].encode(), E, ahead=int(cfg["ahead"])), m


def load_policy(cfg: dict, meta: dict):
    from ctsample.design.policy_cts import load_cts_policy
    E = int(meta["n_epochs"])
    Bbar = int(round(float(cfg["rho"]) * float(meta["extra"]["sum_s_all"]) / E))
    pol = load_cts_policy(cfg["fold"], int(cfg["model_seed"]), cfg["p_key"], float(cfg["rho"]), Bbar,
                          design=cfg["design"], controller=cfg["controller"])
    return pol, Bbar


class StreamCompare:
    """Streaming comparison of a shift-0 lossless pass with the stream's units.parquet."""

    def __init__(self, stream_id: str):
        import pyarrow.parquet as pq
        t = pq.read_table(RESULTS / "streams" / stream_id / "units.parquet",
                          columns=["epoch", "index", "meta", "device_mac", "pay", "s"]).to_pydict()
        self.ep = np.asarray(t["epoch"])
        self.meta = t["meta"]
        self.mac = t["device_mac"]
        self.ptr = np.searchsorted(self.ep, np.arange(int(self.ep.max()) + 2))
        self.n_units = 0
        self.n_bad = 0
        self.bad_epochs: list[int] = []
        self.examples: list = []

    def __call__(self, b, recs):
        e = b.epoch
        if e + 1 >= len(self.ptr):
            self.n_bad += b.n
            self.bad_epochs.append(e)
            return
        lo, hi = int(self.ptr[e]), int(self.ptr[e + 1])
        if hi - lo != b.n:
            self.n_bad += abs(hi - lo - b.n) + 1
            self.bad_epochs.append(e)
            return
        for k, r in enumerate(recs):
            i = lo + k
            m1, m2 = bytearray(r.meta), bytearray(self.meta[i])
            m1[37:41] = m2[37:41] = b"\x00" * 4
            mac = r.device_mac
            mt = ":".join(f"{x:02x}" for x in mac) if isinstance(mac, (bytes, bytearray)) else str(mac)
            self.n_units += 1
            if m1 != m2 or mt != self.mac[i]:
                self.n_bad += 1
                if len(self.examples) < 5:
                    self.examples.append({"epoch": e, "index": k, "mac": mt, "mac_stream": self.mac[i]})


def stage_honest(cfg: dict, p_loss: float, seed: int, tag: str | None = None, save_frames: bool = True,
                 n_epochs: int | None = None, out_root: Path | None = None) -> Path:
    """``n_epochs`` < stream length runs a causal prefix (E8); B̄ stays ρ Σ_all s / E of the whole stream."""
    tag = tag or f"honest_p{p_loss:g}_s{seed}"
    out = (Path(out_root) if out_root is not None else OUT / "runs") / tag
    if done(out):
        log("honest run done", out)
        return out
    out.mkdir(parents=True, exist_ok=True)
    c = cfg["counters"]
    dep, meta = deployment(cfg)
    E = int(n_epochs) if n_epochs else int(meta["n_epochs"])
    vc0 = O.virtual_capture(cfg["stream"])
    shift = O.align_shift(dep.clock, vc0)
    vc = vc0.with_shift(shift)
    gw_id = gw_id_from_pk(dep.keys.pk_G)
    log(f"honest {tag}: {E} epochs, shift {shift} µs (stream t0 {vc0.t0_us} → gateway t_0 {vc.t0_gw_us})")
    # ---- lossless pass (shift 0): validation against the stream + counter view
    cmp_ = StreamCompare(cfg["stream"])
    res = O.unitise(vc0, on_batch=cmp_, keep_frames=False, track=True, max_epoch=E)
    log(f"lossless pass: {res.n_packets} packets, {len(res.unit_epoch)} units, {res.elapsed_s:.1f} s; "
        f"stream comparison: {cmp_.n_units} units, {cmp_.n_bad} mismatches")
    dev = res.unit_dev[res.pk_unit]
    ts_gw = res.pk_ts + shift
    n_total = int(res.pk_k.max()) + 1

    def epoch_of(t):
        return (np.asarray(t, dtype=np.int64) - vc.t0_gw_us) // T_US

    cs = CounterSource(dep.keys.sk_I, gw_id, epoch_of, loss_model(cfg, p_loss), SkewModel(float(c["skew_ms"])),
                       seed=int(seed))
    sim = cs.simulate(ts_gw, dev, res.pk_len)
    keep = merged_keep(n_total, res.pk_k, sim["keep"])
    counters = cs.records(sim["counter_cells"], range(E))
    log(f"counter model: loss fraction {1 - sim['keep'].mean():.5f}; {len(sim['counter_cells'])} counter cells")
    # ---- gateway pass
    pol, Bbar = load_policy(cfg, meta)
    run = O.GatewayRun(out, dep, pol, E, save_frames=save_frames, log=log)
    t0 = time.perf_counter()
    r2 = O.unitise(vc, keep=keep, registry=res.registry, on_batch=run.on_batch, keep_frames=True, max_epoch=E)
    summ = run.finish()
    wall = time.perf_counter() - t0
    gw = run.gw
    log(f"gateway pass: {r2.n_packets} packets in {wall:.1f} s; retained {summ['retained_units']} units, "
        f"{summ['retained_s']} B (target {Bbar * E}); refusals {len(summ['refusals'])}")
    # ---- per-unit tables of the committed universe (+ stream match for labels)
    import pyarrow.parquet as pq
    st = pq.read_table(RESULTS / "streams" / cfg["stream"] / "units.parquet", columns=["epoch", "meta"]).to_pydict()
    s_ep = np.asarray(st["epoch"])
    key_index = {(int(e), bytes(m[:37])): i for i, (e, m) in enumerate(zip(s_ep, st["meta"]))}
    rows = {"epoch": [], "index": [], "device": [], "pay": [], "s": [], "pkts": [], "Z": [], "tau": [],
            "stream_index": []}
    for e in range(E):
        rec = gw.records[e]
        for i, r in enumerate(rec.store.records):
            rows["epoch"].append(e)
            rows["index"].append(i)
            rows["device"].append(r.device_id)
            rows["pay"].append(r.pay)
            rows["s"].append(r.s)
            rows["pkts"].append(r.n_pkts)
            rows["Z"].append(bool(rec.Z[i]))
            rows["tau"].append(float(rec.taus[i]) / 2.0 ** 64)
            rows["stream_index"].append(key_index.get((e, bytes(r.meta[:37])), -1))
    units = {k: np.asarray(v) for k, v in rows.items()}
    save_npz(out / "units.npz", **units)
    save_npz(out / "packets.npz", pk_k=res.pk_k, pk_ts_gw=ts_gw, pk_len=res.pk_len, pk_dev=dev, pk_part=res.pk_part,
             pk_keep=sim["keep"], pk_ctr_epoch=sim["ctr_epoch"], pk_gw_epoch=sim["gw_epoch"],
             unit_epoch=res.unit_epoch, unit_dev=res.unit_dev, unit_pay=res.unit_pay, unit_part=res.unit_part)
    atomic_write_bytes(out / "counters.pkl", pickle.dumps(counters, protocol=5))
    atomic_write_bytes(out / "registry.pkl", pickle.dumps(res.registry, protocol=5))
    committed = O.store_cells({e: gw.records[e].store for e in range(E)})
    tab = cell_table(committed, sim["counter_cells"], E)
    save_npz(out / "cells.npz", cells=tab)
    # ---- storage (E6) and run summary
    from ctsample.protocol.store import parquet_sizes
    stores = [gw.records[e].store for e in range(E)]
    psz = parquet_sizes(stores)
    raw_design = sum(sum(len(r.meta) + (32 if r.pay else 0) for r in s_.records) for s_ in stores)
    tm = run.timing
    fg_macs = vc0.foreground_macs()
    inv_reg = {v: k for k, v in res.registry.items()}
    fg_devs = sorted(d for d, mac in inv_reg.items() if isinstance(mac, (bytes, bytearray)) and bytes(mac) in fg_macs)
    seed_macs = vc0.seed_macs()
    seed_devs = sorted(d for d, mac in inv_reg.items() if isinstance(mac, (bytes, bytearray)) and bytes(mac) in seed_macs)
    seed_dev = seed_devs[0] if seed_devs else None
    seed_mac = ":".join(f"{x:02x}" for x in inv_reg[seed_dev]) if seed_dev is not None else None
    lab = np.load(RESULTS / "streams" / cfg["stream"] / "label.npy", mmap_mode="r")
    si = units["stream_index"]
    labels = np.where(si >= 0, np.asarray(lab)[np.maximum(si, 0)], -2)
    info = {
        "tag": tag, "stream": cfg["stream"], "p_loss": p_loss, "seed": seed, "n_epochs": E, "shift_us": int(shift),
        "t0_gateway_us": int(vc.t0_gw_us), "gw_id": gw.gw_id.hex(), "Bbar": Bbar, "rho": cfg["rho"],
        "policy": {"fold": cfg["fold"], "model_seed": cfg["model_seed"], "design": cfg["design"],
                   "controller": cfg["controller"], "H_theta": pol.registration()[0].hex(),
                   "H_policy": pol.registration()[1].hex(), "phi0": pol.phi0, "tau_mix": pol.tau_mix,
                   "lam_ref": pol.lam_ref, "boost_detectors": list(pol.boost_detectors)},
        "validation": {"lossless_units": int(len(res.unit_epoch)), "stream_units": int(len(s_ep)),
                       "compared": cmp_.n_units, "mismatches": cmp_.n_bad, "bad_epochs": cmp_.bad_epochs[:20],
                       "examples": cmp_.examples, "lossless_packets": int(res.n_packets),
                       "flowtable_qa": res.qa},
        "loss": {"fraction": float(1 - sim["keep"].mean()), "n_lost": int((~sim["keep"]).sum())},
        "gateway_pass": {"packets": int(r2.n_packets), "units": int(len(units["epoch"])),
                         "payload_units": int(units["pay"].sum()), "wall_s": wall, "flowtable_qa": r2.qa,
                         "new_devices_vs_lossless": sorted(set(r2.registry.values()) - set(res.registry.values()))},
        "retention": summ, "spend_over_target": summ["retained_s"] / (Bbar * E),
        "unmatched_to_stream": int((si < 0).sum()),
        "devices": {"n": len(res.registry), "foreground": fg_devs, "seed_device": seed_dev, "seed_mac": seed_mac},
        "labels": {"cnc_payload_units": int(((labels == 1) & units["pay"]).sum()),
                   "malicious_payload_units": int(((labels >= 1) & units["pay"]).sum())},
        "storage": {"raw_design_bytes": int(raw_design), **psz, "sum_s_all": int(units["s"].sum()),
                    "retained_plaintext_s": int(summ["retained_s"]), "retention_bytes": int(summ["retention_bytes"])},
        "timing_gateway_s": {k: float(np.sum(tm[k])) for k in ("commit", "publish", "select", "finish")},
        "timing_gateway_median_ms": {k: 1e3 * float(np.median(tm[k])) for k in ("commit", "publish", "select", "finish")},
        "machine": O.machine_state(),
    }
    save_npz(out / "labels.npz", label=labels)
    save_npz(out / "timing.npz", **{k: np.asarray(v) for k, v in tm.items()})
    atomic_write_json(out / "run.json", info)
    mark_done(out)
    log("honest run written", out)
    return out


# ============================================================================================== cells
def stage_cells(cfg: dict, p_loss: float, seeds: list[int]) -> None:
    """Extra honest loss/skew draws on the E5 stream: committed (lossy flow table) vs counter cells."""
    out = SHARED / "cells" / f"p{p_loss:g}"
    out.mkdir(parents=True, exist_ok=True)
    c = cfg["counters"]
    dep, meta = deployment(cfg)
    E = int(meta["n_epochs"])
    vc0 = O.virtual_capture(cfg["stream"])
    shift = O.align_shift(dep.clock, vc0)
    vc = vc0.with_shift(shift)
    res = None
    for seed in seeds:
        sp = out / f"cells_s{seed}.npz"
        if sp.exists():
            continue
        if res is None:
            res = O.unitise(vc, keep_frames=False, track=True, max_epoch=E)
            dev = res.unit_dev[res.pk_unit]
            n_total = int(res.pk_k.max()) + 1

        def epoch_of(t):
            return (np.asarray(t, dtype=np.int64) - vc.t0_gw_us) // T_US

        cs = CounterSource(None, b"\x00" * 16, epoch_of, loss_model(cfg, p_loss), SkewModel(float(c["skew_ms"])),
                           seed=int(seed))
        sim = cs.simulate(res.pk_ts, dev, res.pk_len)
        keep = merged_keep(n_total, res.pk_k, sim["keep"])
        committed, r2 = cells_pass(vc, keep, res.registry, E)
        tab = cell_table(committed, sim["counter_cells"], E)
        save_npz(sp, cells=tab, stats=np.array(json.dumps({"seed": seed, "loss_fraction": float(1 - sim["keep"].mean()),
                                                           "n_cells": int(len(tab))})))
        log(f"cells p={p_loss:g} seed {seed}: {len(tab)} cells")
    mark_done(out)


# ============================================================================================== trials
ADVERSARIES = ["honest", "A1", "A2", "A3", "A4", "A5", "A7", "A8", "A9", "A10", "A11", "A12"]
# rev1 additions: commit-time threshold deviations (A4c θ' swap, A5c scaled τ) and the full-window omitted-boost
# check (25-epoch windows, attacked epoch a + 21, every epoch that can justify its boost disclosed) + its honest FAR
ADVERSARIES_REV1 = ["A4c", "A5c", "A10omit", "A10omitfull", "honestfull"]
W_ROLL = 120
FULL_W = 25                       # full-window disclosure for the boost-completeness check (ex = a + 21)


def rebuild_real_unit(u, frames):
    """A_rt rebuild of a unit from a reduced frame list with the REAL enc(m_i) layout (units/encode.py):
    per-direction packets / payload bytes / frame bytes, s, first-8 sizes and IAT codes, TCP flag counts and
    first/last ts are re-derived from the remaining frames exactly as the flow table does; key, device,
    flow_uid, conn_state, continuation and fps are kept (the compromised capture point commits any
    self-consistent metadata). A unit whose remaining frames carry no transport payload becomes a
    zero-payload unit (compact layout, no frames); a unit without frames is dropped (None)."""
    from ctsample.crypto import hashing as hs_
    from ctsample.io.pcap_reader import K_IP, P_FLAGS, P_FRAG, P_KIND, P_L4OK, P_PAYLEN, P_PROTO, parser_for
    from ctsample.units import encode as enc
    from ctsample.units.batch import Unit
    if not frames:
        return None
    parse = parser_for(1)
    d0 = enc.dec_metadata(u.meta)
    pk, pb, fb = [0, 0], [0, 0], [0, 0]
    syn = synack = fin = rst = psh = 0
    sizes, iats = [], []
    prev = None
    for j, f in enumerate(frames):
        p = parse(f.data, f.wirelen)
        plen, flags = 0, -1
        if p[P_KIND] == K_IP:
            pr, frag = p[P_PROTO], p[P_FRAG]
            if frag is not None and frag[1] != 0:
                plen = p[P_PAYLEN]
            elif pr in (6, 17) and p[P_L4OK]:
                plen = p[P_PAYLEN]
                if pr == 6:
                    flags = p[P_FLAGS]
        d = f.dir
        pk[d] += 1
        pb[d] += plen
        fb[d] += f.caplen
        if j < 8:
            sizes.append(plen if d == 0 else -plen)
        if 0 < j <= 8:
            iats.append(enc.iat_code(f.ts_us - prev))
        prev = f.ts_us
        if flags >= 0:
            if flags & 0x02:
                if flags & 0x10:
                    synack += 1
                else:
                    syn += 1
            if flags & 0x01:
                fin += 1
            if flags & 0x04:
                rst += 1
            if flags & 0x08:
                psh += 1
    s = sum(f.caplen + 16 for f in frames)
    pay = (pb[0] + pb[1]) > 0
    m = {"key": d0["key"], "device_id": d0["device_id"], "first_ts_us": frames[0].ts_us,
         "last_ts_us": frames[-1].ts_us, "pkts": tuple(pk), "payload_bytes": tuple(pb) if pay else (0, 0),
         "frame_bytes": tuple(fb), "n_syn": syn, "n_synack": synack, "n_fin": fin, "n_rst": rst, "n_psh": psh,
         "conn_state": d0["conn_state"], "sizes": sizes if pay else [], "iats": iats, "s": s,
         "flow_uid": d0.get("flow_uid", hs_.flow_uid(d0["key"], frames[0].ts_us)),
         "continuation": d0.get("continuation", False), "fps": bool(d0.get("fps", False)) and pay, "pay": pay}
    meta = enc.enc_metadata(m)
    return Unit(epoch=u.epoch, index=u.index, key=d0["key"], device_id=u.device_id, first_ts_us=frames[0].ts_us,
                last_ts_us=frames[-1].ts_us, pkts=(pk[0], pk[1]), payload_bytes=tuple(pb) if pay else (0, 0),
                frame_bytes=(fb[0], fb[1]), s=s, pay=pay, fps=m["fps"], flow_uid=m["flow_uid"], meta=meta,
                frames=frames if pay else None)


def _clear_leaves(units):
    for u in units:
        u.leaf, u.root_pkt, u.pkt_leaves = b"", None, None


class TrialCtx:
    """Per-worker context over a persisted honest run."""

    def __init__(self, cfg: dict, run_dir: Path):
        from ctsample.protocol.store import EncodeMetaCodec
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.info = json.loads((self.run_dir / "run.json").read_text())
        self.dep, self.meta = deployment(cfg)
        self.E = int(self.meta["n_epochs"])
        self.counters = pickle.loads((self.run_dir / "counters.pkl").read_bytes())
        self.eps = load_eps(float(self.info["p_loss"]))
        self.pristine, self.Bbar = load_policy(cfg, self.meta)
        self.codec = EncodeMetaCodec()
        z = np.load(self.run_dir / "units.npz")
        self.units = {k: z[k] for k in z.files}
        self.labels = np.load(self.run_dir / "labels.npz")["label"]
        self.ptr = np.searchsorted(self.units["epoch"], np.arange(self.E + 1))
        self.target = int(self.info["devices"]["seed_device"])
        self._chunks: dict[int, dict] = {}
        self._fchunks: dict[int, dict] = {}
        # rev1 runs commit τ_i in their stores; rev0 runs (results/E5) can only be verified with the pre-revision
        # semantics
        from ctsample.protocol.store import MetadataStore
        st0 = MetadataStore.deserialize(self.records([0])[0]["store"])
        self.semantics = "rev1" if st0.committed else "rev0"

    # ---------------------------------------------------------------- data access
    def records(self, epochs) -> dict:
        out = {}
        for e in sorted(set(int(x) for x in epochs)):
            k = e // O.CHUNK
            if k not in self._chunks:
                if len(self._chunks) > 12:
                    self._chunks.pop(next(iter(self._chunks)))
                self._chunks[k] = pickle.loads((self.run_dir / "records" / f"r{k:04d}.pkl").read_bytes())
            out[e] = self._chunks[k][e]
        return out

    def epoch_rows(self, e: int) -> slice:
        return slice(int(self.ptr[e]), int(self.ptr[e + 1]))

    def frames(self, epochs) -> dict:
        """Fresh Frame objects for the payload units of ``epochs`` (raw blobs cached per chunk)."""
        out = {}
        for e in sorted(set(int(x) for x in epochs)):
            k = e // O.CHUNK
            if k not in self._fchunks:
                if len(self._fchunks) > 3:
                    self._fchunks.pop(next(iter(self._fchunks)))
                self._fchunks[k] = pickle.loads((self.run_dir / "frames" / f"f{k:04d}.pkl").read_bytes())
            out[e] = O.read_frames(self._fchunks[k][e])
        return out

    def view(self, win, context=()):
        win = sorted(win)
        ctx = sorted(set(context) | {win[0] - 1}) if win[0] > 0 else sorted(context)
        v = O.RunView(self.run_dir, self.dep, win, context=ctx, codec=self.codec, records=self.records(list(win) + ctx),
                      frames=self.frames(win))
        v.e0 = 0
        return v

    def prime(self, disclosed, exclude=()):
        from ctsample.protocol.prime import PrimeBundle, PrimeEntry, prime_epochs
        from ctsample.protocol.store import MetadataStore
        ks = [k for k in prime_epochs(disclosed, 0, W_ROLL) if k not in set(exclude)]
        raw = self.records(ks)
        b = PrimeBundle(W=W_ROLL)
        for k in ks:
            r = raw[k]
            b.entries[k] = PrimeEntry(r["header"], r["sigma"], r["cosig"], MetadataStore.deserialize(r["store"]))
        return b

    def verify(self, disc, bundle, epochs=None, semantics: str | None = None, recompute_compliance: bool = True):
        from ctsample.protocol.prime import verify_with_prime
        from ctsample.protocol.verifier import TrustAnchors, Verifier
        trust = TrustAnchors.from_disclosure(disc, self.dep.keys.pk_W, pk_I=self.dep.keys.pk_I, beacon=self.dep.beacon)
        ver = Verifier(trust, O.fresh_policy(self.pristine), eps_cell=self.eps, meta_codec=self.codec,
                       semantics=semantics or self.semantics, recompute_compliance=recompute_compliance)
        t0 = time.perf_counter()
        rep = verify_with_prime(ver, disc, bundle, epochs=epochs)
        return rep, time.perf_counter() - t0

    def disclose(self, view, win, requested=None):
        from ctsample.protocol.disclosure import build_disclosure
        return build_disclosure(view, win, requested=requested, counters=self.counters)

    # ---------------------------------------------------------------- forks (pre-commit attacks)
    def fork(self, view, e_start: int, e_end: int, batches: dict, wrap=None, prev_hash: bytes | None = None,
             skip: set | None = None, late: set | None = None):
        """Re-commit epochs e_start..e_end from the honest state at e_start with (possibly attacked) batches;
        epochs in ``skip`` are processed by the policy but never published (A7); epochs in ``late`` reach the
        witness after β of their designated round (a missed commit window: the witness refuses, no ω_e).
        Updates ``view``."""
        import copy as _c
        from ctsample.protocol.epoch import Gateway
        from ctsample.protocol.store import MetadataStore
        pol = O.fresh_policy(self.pristine)
        gwp = wrap(pol) if wrap is not None else pol
        gw = Gateway(self.dep.keys, self.dep.clock, self.dep.beacon, gwp, self.dep.witness(), self.dep.escrow, e0=0,
                     keep_frames=True)
        O.register(gw, self.dep)
        raw = self.records(range(max(0, e_start - W_ROLL + 1), e_start + 1))
        pol.prime([MetadataStore.deserialize(raw[k]["store"]).to_batch(codec=self.codec)
                   for k in range(max(0, e_start - W_ROLL + 1), e_start)])
        gw.state = _c.deepcopy(raw[e_start]["state"])
        gw.prev_hash = prev_hash if prev_hash is not None else self.records([e_start - 1])[e_start - 1]["header"].digest()
        gw.next_e = e_start
        skip = skip or set()
        late = late or set()
        for e in range(e_start, e_end + 1):
            if e in skip:          # A7: processed by the policy (history, state) but never published; the
                prev_before = gw.prev_hash          # next header links to Hdr_{e−1} (the gap is hidden)
                rec = gw.commit(batches[e])
                gw.select(rec)
                gw.finish(rec)
                del gw.records[e]
                gw.prev_hash = prev_before
                continue
            if e in late:          # missed commit window: received after t(r*_e) (TR refuses; evidence void)
                gw.run_epoch(batches[e], recv_time=float(self.dep.clock.t_rstar(e)) + 0.5)
                continue
            gw.run_epoch(batches[e])
        for (ee, i) in [k for k in view.retention.items if e_start <= k[0] <= e_end]:
            del view.retention.items[(ee, i)]
        for k, it in gw.retention.items.items():
            view.retention.items[k] = it
        for e in range(e_start, e_end + 1):
            if e in gw.records:
                view.records[e] = gw.records[e]
            else:
                view.records.pop(e, None)
        return gw

    # ---------------------------------------------------------------- helpers
    def retained_in(self, e: int) -> list[int]:
        return list(self.records([e])[e]["retained"])

    def dev_of(self, e: int):
        return self.units["device"][self.epoch_rows(e)]

    def hits_of_target(self, e: int) -> list[int]:
        sl = self.epoch_rows(e)
        m = (self.units["device"][sl] == self.target) & self.units["pay"][sl] & (self.labels[sl] == 1)
        return [int(i) for i in np.nonzero(m)[0]]

    def target_pkts(self, e: int) -> int:
        sl = self.epoch_rows(e)
        return int(self.units["pkts"][sl][self.units["device"][sl] == self.target].sum())


def _fired(rep) -> list[dict]:
    return [{"check": r.check, "name": r.name, "status": r.status, "e": r.epoch}
            for r in rep.results if r.status in ("FAIL", "WARN")]


def _names(rep, statuses=("FAIL",)) -> set:
    return {f"{r.check}.{r.name}" for r in rep.results if r.status in statuses}


EXPECTED = {
    "A1": {"V6.missing"}, "A2": {"V6.extra", "V6.mismatch"}, "A3": {"V4.store_hash", "V4.root"},
    "A4": {"V4.claimed_taus", "V5.claimed_selection", "V6.missing", "V6.extra"},
    "A5": {"V4.claimed_taus", "V4.state_opening", "V1.signature", "V1.cosignature", "V4.floor"},
    "A7": {"V1.header_present", "V1.chain"}, "A8": {"V7.cells"}, "A9": {"V6.missing", "V5.claimed_selection"},
    "A10": {"V4.policy_compliance"}, "A10omit": {"V4.policy_compliance"}, "A10omitfull": {"V4.policy_compliance"},
    "A11": {"V6.missing", "V3.vrf", "V1.header_present"}, "A12": set(),
    "A4c": {"V4.tau_compliance"}, "A5c": {"V4.tau_compliance", "V4.committed_taus"},
}


def _window(rng: random.Random, E: int, W: int, need=None, lo: int = 0) -> list[int] | None:
    cands = list(range(max(lo, 0), E - W + 1))
    rng.shuffle(cands)
    for a in cands[:400]:
        win = list(range(a, a + W))
        if need is None or need(win):
            return win
    return None


def run_trial(ctx: TrialCtx, adv: str, t: int) -> dict:
    """One trial of adversary ``adv`` (trial index t); returns a JSON-able result."""
    from ctsample.adversary import abort as AB
    from ctsample.adversary import tamper as TA
    cfg = ctx.cfg
    W = int(cfg["trials"]["window"])
    rng = random.Random(f"{cfg['trials']['seed']}/{adv}/{t}")
    E = ctx.E
    res = {"adv": adv, "t": t}
    has_ret = lambda win: any(ctx.retained_in(e) for e in win)                       # noqa: E731
    has_tgt_ret = lambda win: any(set(ctx.retained_in(e)) & set(ctx.hits_of_target(e)) for e in win)   # noqa: E731
    variant = None
    if adv in ("honest", "honestfull"):
        win = _window(rng, E, W if adv == "honest" else FULL_W)
        view = ctx.view(win)
        disc = ctx.disclose(view, win)
        rep, sec = ctx.verify(disc, ctx.prime(win))
        res.update(win=win, verify_s=sec, fired=_fired(rep), n_fail=rep.n_fail, prime=rep.prime,
                   n_cells=sum(len(v.cells) for v in rep.epochs.values()),
                   v7_flagged_cells=sum(sum(1 for c in v.cells if c.get("flagged")) for v in rep.epochs.values()),
                   disclosure=disc.size_report(), detected=None, status_counts=_status_counts(rep),
                   policy_compliance_warn=bool(rep.warnings("V4", "policy_compliance")),
                   tau_compliance_warn=bool(rep.warnings("V4", "tau_compliance")))
        if adv == "honestfull":
            res["detected"] = res["policy_compliance_warn"]          # a false alarm of the full-window check
        return res
    if adv in ("A1", "A2", "A3", "A4", "A5", "A9"):
        need = has_tgt_ret if adv == "A9" else (has_ret if adv in ("A1", "A2") else None)
        win = _window(rng, E, W, need)
        view = ctx.view(win)
        disc = ctx.disclose(view, win)
        if adv == "A1":
            d2, inf = TA.a1_drop_selected(disc, rng)
        elif adv == "A2":
            variant = ["insert_extra", "replace_frame", "replace_frame_fix_hash"][t % 3]
            if variant == "insert_extra":
                d2, inf = TA.a2_insert_extra(disc, view, rng)
            else:
                d2, inf = TA.a2_replace_frame(disc, rng, fix_hash_list=variant.endswith("hash"))
        elif adv == "A3":
            fields = ["meta", "device_id", "s"] + (["tau"] if ctx.semantics == "rev1" else [])
            variant = fields[t % len(fields)]
            d2, inf = TA.a3_alter_metadata(disc, rng, field=variant)
        elif adv == "A4":
            variant = "theta_prime_fold_3-1"
            from ctsample.design.policy_cts import load_cts_policy
            from ctsample.protocol.prime import PrimedPolicy
            if not hasattr(ctx, "_theta2"):
                ctx._theta2 = load_cts_policy("3-1", int(cfg["model_seed"]), cfg["p_key"], float(cfg["rho"]), ctx.Bbar,
                                              design=cfg["design"], controller=cfg["controller"])
            bnd = ctx.prime(win)
            pp = PrimedPolicy(O.fresh_policy(ctx._theta2), {k: x.store for k, x in bnd.entries.items()},
                              codec=ctx.codec, W=W_ROLL, e0=0)
            d2, inf = TA.a4_swap_theta(view, disc, pp)
        elif adv == "A5":
            variant = ["claim", "header", "opening", "reopen"][t % 4]
            scale = [0.25, 0.5, 0.9][(t // 4) % 3]
            if variant == "reopen":
                e = rng.choice(win)
                kind = ["D", "fm", "boost", "salt"][(t // 4) % 4]
                hb = set(ctx.records([e])[e]["opening"].boost_set)
                if ctx.target in hb:
                    bkw = {"boost_remove": [ctx.target]}
                else:
                    sl = ctx.epoch_rows(e)
                    cand = sorted(set(ctx.units["device"][sl].tolist()) - hb)
                    bkw = {"boost_add": [int(rng.choice(cand))]}
                kw = {"D": {"D_delta": int(ctx.Bbar)}, "fm": {"floor_mode": 1}, "boost": bkw,
                      "salt": {"new_salt": bytes(32)}}[kind]
                d2, inf = TA.reopen_state(disc, [e], **kw)
                variant = f"reopen_{kind}"
            else:
                d2, inf = TA.a5_alter_lambda(view, disc, scale=scale, mode=variant)
                variant = f"{variant}_x{scale}"
        else:  # A9
            tg = ctx.target
            devs = {e: ctx.dev_of(e) for e in win}
            d2, inf = TA.a9_cherry_pick(disc, keep=lambda e, i: int(devs[e][i]) != tg, update_claims=True)
        rep, sec = ctx.verify(d2, ctx.prime(win))
        effective = True
        if adv == "A9":
            effective = bool(inf)
        if adv in ("A4", "A5") and isinstance(inf, dict):
            effective = True
        res.update(win=win, variant=variant, info=_jsonable(inf))
    elif adv == "A11":
        rule = AB.RULES[t % 2]
        mode = ["content", "proof", "epoch"][(t // 2) % 3]
        variant = f"{rule}/{mode}"

        def need(win):                    # the rule must void at least one epoch (else there is no abort)
            for e in win:
                h = ctx.hits_of_target(e)
                if not h:
                    continue
                ret = bool(set(h) & set(ctx.retained_in(e)))
                if (rule == "incriminating" and not ret and (mode != "content" or ctx.retained_in(e))) or \
                        (rule == "exculpatory" and ret):
                    return True
            return False
        win = _window(rng, E, W, need)
        view = ctx.view(win)
        disc = ctx.disclose(view, win)
        hits = {e: ctx.hits_of_target(e) for e in win}
        d2, voided = AB.a11_selective_abort(disc, view, hits, rule, mode=mode)
        if mode == "content":
            effective = any(set(ctx.retained_in(e)) for e in voided)
        else:
            effective = bool(voided)
        rep, sec = ctx.verify(d2, ctx.prime(win), epochs=win)
        res.update(win=win, variant=variant, voided=sorted(voided))
    elif adv in ("A7", "A8", "A10", "A10omit", "A10omitfull", "A12", "A4c", "A5c"):
        if adv == "A8":
            m_grid = list(cfg["trials"]["a8_m"])
            m = int(m_grid[t % len(m_grid)])
            need = (lambda win: ctx.target_pkts(win[2]) > 0)
        elif adv == "A12":
            need = (lambda win: any(ctx.units["pay"][ctx.epoch_rows(win[2])] & (ctx.dev_of(win[2]) == ctx.target)))
        elif adv == "A10omit":
            need = (lambda win: bool(ctx.records([win[2]])[win[2]]["state"].boost_set))
        elif adv == "A10omitfull":
            need = (lambda win: bool(ctx.records([win[21]])[win[21]]["state"].boost_set))
        else:
            need = None
        win = _window(rng, E, FULL_W if adv == "A10omitfull" else W, need, lo=1)
        ex = win[21] if adv == "A10omitfull" else win[2]
        view = ctx.view(win)
        batches = {e: view.records[e].batch for e in range(ex, win[-1] + 1)}
        effective = True
        if adv == "A7":
            variant = "withhold_middle_epoch"
            prev = ctx.records([ex - 1])[ex - 1]["header"].digest()
            ctx.fork(view, ex, win[-1], batches, skip={ex}, prev_hash=prev)
            disc = ctx.disclose(view, [e for e in win if e != ex])
            bundle = ctx.prime([e for e in win if e != ex], exclude={ex})
            rep, sec = ctx.verify(disc, bundle, epochs=win)
            res.update(withheld=ex)
        elif adv == "A8":
            from ctsample.adversary.omission import omit_packets_in_batches
            b = batches[ex]
            _clear_leaves(b.units)
            before = sum(u.n_pkts for u in b.units if u.device_id == ctx.target)
            removed = omit_packets_in_batches([b], ctx.target, {ex}, m, rng, rebuild=rebuild_real_unit,
                                              include_zero_payload=True)
            after = sum(u.n_pkts for u in b.units if u.device_id == ctx.target)
            ctx.fork(view, ex, win[-1], batches)
            disc = ctx.disclose(view, win)
            rep, sec = ctx.verify(disc, ctx.prime(win))
            cells = rep.epochs[ex].cells
            cell = next((c for c in cells if c["device_id"] == ctx.target), None)
            variant = f"m={m}"
            res.update(m=m, removed=int(removed.get(ex, 0)), cell=cell, device_pkts_before=before,
                       device_pkts_after=after, attacked_cell_flagged=bool(cell and cell["flagged"]))
            effective = removed.get(ex, 0) > 0
        elif adv == "A10":
            mode = "forge"
            fake = bool(t % 2)
            boosted = set(ctx.records([ex])[ex]["state"].boost_set)
            sl = ctx.epoch_rows(ex)
            cand = sorted(set(ctx.units["device"][sl][ctx.units["pay"][sl]].tolist()) - boosted)
            if not cand:
                cand = sorted(set(range(int(ctx.units["device"].max()) + 1)) - boosted)
            dev = int(rng.choice(cand))          # a device that is NOT boosted in the honest state
            variant = f"forge{'_fake_justification' if fake else ''}"
            wrap = (lambda p: TA.BoostDeviationPolicy(p, mode, epochs=[ex], device=dev, fake_justification=fake))
            ctx.fork(view, ex, win[-1], batches, wrap=wrap)
            disc = ctx.disclose(view, win)
            rep, sec = ctx.verify(disc, ctx.prime(win))
            res.update(device=dev, mode=mode, fake=fake, honest_boost_set=sorted(boosted))
        elif adv in ("A10omit", "A10omitfull"):
            boosted = sorted(ctx.records([ex])[ex]["state"].boost_set)
            variant = "omit_justified_boost" + ("_full_window" if adv == "A10omitfull" else "")
            wrap = (lambda p: TA.BoostDeviationPolicy(p, "omit", epochs=[ex]))
            ctx.fork(view, ex, win[-1], batches, wrap=wrap)
            disc = ctx.disclose(view, win)
            rep, sec = ctx.verify(disc, ctx.prime(win))
            pc = [r for r in rep.results if r.name == "policy_compliance" and r.epoch == ex]
            res.update(mode="omit", honest_boost_set=boosted, ex=ex,
                       rules_at_ex=sorted({d["rule"] for r in pc for d in r.evidence.get("deviations", [])}),
                       status_at_ex=pc[0].status if pc else None)
        elif adv == "A4c":
            from ctsample.design.policy_cts import load_cts_policy
            from ctsample.protocol.prime import PrimedPolicy
            variant = "commit_theta_prime_fold_3-1"
            if not hasattr(ctx, "_theta2"):
                ctx._theta2 = load_cts_policy("3-1", int(cfg["model_seed"]), cfg["p_key"], float(cfg["rho"]), ctx.Bbar,
                                              design=cfg["design"], controller=cfg["controller"])
            bnd = ctx.prime([ex])
            theta = PrimedPolicy(O.fresh_policy(ctx._theta2), {k: x.store for k, x in bnd.entries.items()},
                                 codec=ctx.codec, W=W_ROLL, e0=0)
            wrap = (lambda p: TA.a4c_commit_theta(p, theta, [ex]))
            ctx.fork(view, ex, win[-1], batches, wrap=wrap)
            disc = ctx.disclose(view, win)
            rep, sec = ctx.verify(disc, ctx.prime(win))
            hon = ctx.records([ex])[ex]["taus"]
            dev = view.records[ex].taus
            res.update(n_tau_changed=int(sum(1 for a, b in zip(hon, dev) if a != b)))
            effective = res["n_tau_changed"] > 0
        elif adv == "A5c":
            scale = [0.25, 0.5][(t // 2) % 2]
            keep = bool(t % 2 == 0)
            variant = f"commit_scale_x{scale:g}" + ("_keep_floor" if keep else "_below_floor")
            wrap = (lambda p: TA.a5c_commit_scaled(p, scale, [ex], keep_floor=keep))
            ctx.fork(view, ex, win[-1], batches, wrap=wrap)
            disc = ctx.disclose(view, win)
            rep, sec = ctx.verify(disc, ctx.prime(win))
            hon = ctx.records([ex])[ex]["taus"]
            dev = view.records[ex].taus
            res.update(n_tau_changed=int(sum(1 for a, b in zip(hon, dev) if a != b)), keep_floor=keep)
            effective = res["n_tau_changed"] > 0
        else:  # A12
            b = batches[ex]
            _clear_leaves(b.units)
            inf = TA.a12_equal_size_substitution([b], ctx.target, rng)
            ctx.fork(view, ex, win[-1], batches)
            disc = ctx.disclose(view, win)
            rep, sec = ctx.verify(disc, ctx.prime(win))
            variant = "equal_size_payload_rewrite"
            res.update(info=inf)
            effective = inf["frames_rewritten"] > 0
        res.update(win=win, variant=variant)
    else:
        raise ValueError(adv)
    fails = _names(rep, ("FAIL",))
    warns = _names(rep, ("WARN",))
    exp = EXPECTED[adv]
    if adv == "A8":
        detected = bool(res.get("attacked_cell_flagged"))
    elif adv in ("A10", "A10omit", "A10omitfull"):
        detected = "V4.policy_compliance" in warns
    elif adv == "A4c":
        detected = "V4.tau_compliance" in warns
    elif adv == "A5c":
        detected = bool(("V4.tau_compliance" in warns) or ("V4.committed_taus" in fails))
    elif adv == "A12":
        detected = bool(fails - {"V7.cells"})
    else:
        detected = bool(fails & exp)
    res.update(verify_s=sec, fired=_fired(rep), fail_names=sorted(fails), warn_names=sorted(warns),
               detected=detected, effective=bool(effective), n_fail=rep.n_fail,
               n_W=sum(len(v.W) for v in rep.epochs.values()), status_counts=_status_counts(rep),
               semantics=ctx.semantics,
               # validity view: V1–V6 FAILs other than the attacked epoch's (rev1 containment)
               v16_fail_epochs=sorted({r.epoch for r in rep.results if r.status == "FAIL"
                                       and r.check in ("V1", "V2", "V3", "V4", "V5", "V6") and r.epoch is not None}))
    return res


def _status_counts(rep) -> dict:
    from collections import Counter
    c = Counter(f"{r.check}.{r.name}:{r.status}" for r in rep.results if r.status != "PASS")
    return dict(sorted(c.items()))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, bytes):
        return x.hex()
    return x


def stage_trials(cfg: dict, run_tag: str, advs: list[str], workers: int, worker_index: int | None) -> None:
    run_dir = OUT / "runs" / run_tag
    out = OUT / "trials" / run_tag
    n = int(cfg["trials"]["per_adversary"])
    jobs = [(a, t) for a in advs for t in range(n)]
    if worker_index is None and workers > 1:
        import subprocess
        procs = []
        for w in range(workers):
            cmd = [sys.executable, "-I", "-u", __file__, "trials", "--rev", REV, "--tag", run_tag, "--workers", str(workers),
                   "--worker-index", str(w), "--adversaries", *advs]
            procs.append(subprocess.Popen(cmd))
        rc = [p.wait() for p in procs]
        if any(rc):
            raise SystemExit(f"trial workers failed: {rc}")
        mark_done(out, {"advs": advs, "n": n})
        return
    ctx = TrialCtx(cfg, run_dir)
    wi = worker_index or 0
    mine = [j for k, j in enumerate(jobs) if k % max(1, workers) == wi]
    for adv, t in mine:
        p = out / adv / f"t{t:04d}.json"
        if p.exists():
            continue
        t0 = time.perf_counter()
        try:
            r = run_trial(ctx, adv, t)
        except Exception as ex:                       # noqa: BLE001 — recorded, never silent
            import traceback
            r = {"adv": adv, "t": t, "error": repr(ex), "traceback": traceback.format_exc()[-3000:]}
        r["wall_s"] = time.perf_counter() - t0
        atomic_write_json(p, _jsonable(r))
        if t % 20 == 0:
            log(f"[w{wi}] {adv} t={t}: detected={r.get('detected')} fails={r.get('fail_names')} "
                f"{r.get('error', '')} {r['wall_s']:.1f}s")
    if workers <= 1:
        mark_done(out, {"advs": advs, "n": n})


# ============================================================================================== contain (rev1)
CONTAIN_SCENARIOS = ("withheld", "voided", "late")
CONTAIN_OFFSETS = (1, 30, 60, 90, 115, 120)        # window [ex + o, ex + o + 4]; the rolling dependency is ≤ 119


def run_contain_trial(ctx: TrialCtx, scenario: str, t: int, offsets=CONTAIN_OFFSETS, win_len: int = 5) -> dict:
    """Containment of one void epoch ex (rev1 vs pre-revision semantics on the same committed data).

    scenario  withheld: the gateway never publishes Hdr_ex (A7, before β; the next header links to Hdr_{ex−1});
              voided:   Hdr_ex is public, but nothing else of ex is ever disclosed (A11 'epoch' mode, after β);
              late:     Hdr_ex reaches the witness after β (missed commit window, e.g. a unit flood or WAN
                        jitter): no ω_ex, the evidence of ex is void, the next header links to the refused one.
    For each offset o the verifier checks the 5-epoch window [ex+o, ex+o+4] (o = 1 from a gateway fork that
    realises the scenario; later windows from the honest run) with the 119-epoch prime bundle WITHOUT the
    store of ex (unavailable; 'late': the un-cosigned header + store are offered and rejected by check_prime),
    once with semantics='rev1' and once with semantics='rev0'. Per window epoch: failed (V1–V6), W_all, |W|,
    requested retained units, τ-compliance status."""
    import copy as _copy
    from ctsample.protocol.prime import PrimeEntry
    E = ctx.E
    rng = random.Random(f"{ctx.cfg['trials']['seed']}/contain/{scenario}/{t}")
    ex = rng.randrange(W_ROLL, E - max(offsets) - win_len)
    res = {"scenario": scenario, "t": t, "ex": ex, "offsets": list(offsets), "windows": []}
    fork_view, late_entry = None, None
    if scenario in ("withheld", "late"):
        win0 = list(range(ex, ex + 1 + win_len))
        fork_view = ctx.view(win0)
        batches = {e: fork_view.records[e].batch for e in win0}
        prev = ctx.records([ex - 1])[ex - 1]["header"].digest()
        if scenario == "withheld":
            ctx.fork(fork_view, ex, win0[-1], batches, skip={ex}, prev_hash=prev)
        else:
            ctx.fork(fork_view, ex, win0[-1], batches, late={ex}, prev_hash=prev)
            rec = fork_view.records[ex]
            res["late_refusal"] = rec.refusal
            late_entry = PrimeEntry(rec.header, rec.sigma, rec.cosig, _copy.deepcopy(rec.store))
    for o in offsets:
        win = list(range(ex + o, ex + o + win_len))
        if o == 1 and fork_view is not None:
            view = fork_view
        else:
            view = ctx.view(win)
        disc = ctx.disclose(view, win)
        if scenario == "voided" and o == 1:
            # the published header of ex stays available as chain context (it is public); nothing else of ex
            pass
        bundle = ctx.prime(win, exclude={ex})
        if late_entry is not None:
            bundle.entries[ex] = late_entry
        row = {"offset": o, "win": win}
        for sem in ("rev1", "rev0"):
            rep, sec = ctx.verify(disc, _copy.deepcopy(bundle), semantics=sem)
            per = []
            for e in win:
                v = rep.epochs[e]
                ret_req = sorted(v.selected & v.requested)
                tc = [r.status for r in rep.results if r.name in ("tau_compliance", "thresholds") and r.epoch == e]
                per.append({"e": e, "failed": bool(v.failed), "W_all": bool(v.W_all), "n_W": len(v.W),
                            "n_retained_requested": len(ret_req), "n_retained_in_W": len(set(ret_req) & v.W),
                            "tau_check": tc[0] if tc else None,
                            "fails": sorted({f"{r.check}.{r.name}" for r in rep.results
                                             if r.epoch == e and r.status == "FAIL"})})
            row[sem] = {"verify_s": sec, "epochs": per, "prime_failed": rep.prime.get("failed"),
                        "prime_verified": rep.prime.get("verified")}
        res["windows"].append(row)
    return res


def stage_contain(cfg: dict, run_tag: str, scenarios, n: int, workers: int, worker_index: int | None) -> None:
    out = OUT / "contain" / run_tag
    jobs = [(sc, t) for sc in scenarios for t in range(n)]
    if worker_index is None and workers > 1:
        import subprocess
        procs = [subprocess.Popen([sys.executable, "-I", "-u", __file__, "contain", "--rev", REV, "--tag", run_tag, "--workers",
                                   str(workers), "--worker-index", str(w), "--n", str(n), "--scenarios", *scenarios])
                 for w in range(workers)]
        rc = [p.wait() for p in procs]
        if any(rc):
            raise SystemExit(f"contain workers failed: {rc}")
        mark_done(out, {"scenarios": list(scenarios), "n": n})
        return
    ctx = TrialCtx(cfg, OUT / "runs" / run_tag)
    wi = worker_index or 0
    for k, (sc, t) in enumerate(jobs):
        if k % max(1, workers) != wi:
            continue
        pth = out / sc / f"t{t:04d}.json"
        if pth.exists():
            continue
        t0 = time.perf_counter()
        try:
            r = run_contain_trial(ctx, sc, t)
        except Exception as ex:                       # noqa: BLE001 — recorded, never silent
            import traceback
            r = {"scenario": sc, "t": t, "error": repr(ex), "traceback": traceback.format_exc()[-3000:]}
        r["wall_s"] = time.perf_counter() - t0
        atomic_write_json(pth, _jsonable(r))
        if t % 10 == 0:
            log(f"[w{wi}] contain {sc} t={t}: {r.get('error', 'ok')} {r['wall_s']:.1f}s")
    if workers <= 1:
        mark_done(out, {"scenarios": list(scenarios), "n": n})


def contain_summary(run_tag: str) -> dict:
    base = OUT / "contain" / run_tag
    if not base.exists():
        return {}
    out = {"definition": "one void epoch ex per trial (withheld before β / voided after β / missed commit window); "
                         "5-epoch windows [ex+o, ex+o+4] verified with the 119-epoch prime bundle lacking ex's store, "
                         "rev1 = committed per-unit τ (validity from the committed τ and the epoch-local floor), rev0 = "
                         "pre-revision verifier (τ recomputed from rolling features; a chain gap fails the successor). "
                         "failed = some V1–V6 FAIL in the epoch (its requested units enter W); tau_check = V4 "
                         "tau_compliance (rev1) / thresholds (rev0) status", "scenarios": {}}
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        ts = [json.loads(p.read_text()) for p in sorted(d.glob("t*.json"))]
        ok = [t for t in ts if "error" not in t]
        blk = {"trials": len(ts), "errors": len(ts) - len(ok), "by_offset": {}}
        for o in (ok[0]["offsets"] if ok else []):
            row = {}
            for sem in ("rev1", "rev0"):
                eps = [ep for t in ok for w in t["windows"] if w["offset"] == o for ep in w[sem]["epochs"]]
                n = len(eps)
                if not n:
                    continue
                ret = sum(ep["n_retained_requested"] for ep in eps)
                row[sem] = {"epochs": n, "failed_share": sum(ep["failed"] for ep in eps) / n,
                            "W_all_share": sum(ep["W_all"] for ep in eps) / n,
                            "W_units_per_epoch": sum(ep["n_W"] for ep in eps) / n,
                            "retained_requested": ret,
                            "retained_in_W_share": (sum(ep["n_retained_in_W"] for ep in eps) / ret) if ret else None,
                            "tau_check": dict(sorted(Counter_(ep["tau_check"] for ep in eps).items())),
                            "first_epoch_failed_share": sum(w[sem]["epochs"][0]["failed"] for t in ok for w in t["windows"]
                                                            if w["offset"] == o) / max(1, len(ok)),
                            "verify_s_median": float(np.median([w[sem]["verify_s"] for t in ok for w in t["windows"]
                                                                if w["offset"] == o]))}
            blk["by_offset"][str(o)] = row
        out["scenarios"][d.name] = blk
    return out


def Counter_(it):
    from collections import Counter
    return Counter(str(x) for x in it)


# ============================================================================================== grind (A6)
def _dummy_record(e: int, t_e_us: int, rng: random.Random):
    """A dummy zero-payload unit (real compact enc(m_i) layout, random WAN key) — the grinding knob."""
    from ctsample.protocol.store import UnitRecord
    from ctsample.units import encode as enc
    ts = t_e_us + rng.randrange(1, T_US - 1)
    ip_a = b"\x00" * 10 + b"\xff\xff" + bytes([192, 168, 137, rng.randrange(2, 250)])
    ip_b = b"\x00" * 10 + b"\xff\xff" + bytes(rng.getrandbits(8) for _ in range(4))
    key = enc.make_key(6, ip_a, rng.randrange(1024, 65535), ip_b, rng.choice([23, 80, 443, 2323, 8080]))
    meta = enc.enc_metadata({"key": key, "device_id": 9999, "first_ts_us": ts, "pkts": (1, 0), "n_syn": 1,
                             "n_synack": 0, "n_fin": 0, "n_rst": 0, "n_psh": 0, "conn_state": enc.CS_CODE["S0"],
                             "pay": False})
    return UnitRecord(meta=meta, pay=False, root_pkt=None, device_id=9999, pkts=(1, 0), frame_bytes=(60, 0), s=76,
                      first_ts_us=ts, fps=False)


def grind_trial(dep, keys, policy, witness, gw_id, raw: dict, e: int, ks, Mmax: int, rng: random.Random) -> dict:
    from ctsample.crypto import hashing as hs_
    from ctsample.protocol.epoch import build_header, make_opening, run_selection
    from ctsample.protocol.header import GENESIS_PREV, sign_header
    from ctsample.protocol.store import MetadataStore
    from ctsample.protocol.witness import WitnessRefusal
    from ctsample.units.batch import EpochBatch
    st = MetadataStore.deserialize(raw["store"])
    base = list(st.records)
    pay_idx = [i for i, r in enumerate(base) if r.pay]
    tgt = rng.sample(pay_idx, max(ks))                       # incident: k payload units, fixed before β
    tkeys = [(base[i].first_ts_us, bytes(base[i].meta[:37])) for i in tgt]
    leaf_of = {id(r): r.leaf() for r in base}
    state = policy.initial_state(e)
    fields = policy.commit_fields(state)
    opening = make_opening(keys.k_st, e, fields, ())
    beta = dep.beacon.get(dep.clock.r_star(e)).randomness
    tau = policy.tau
    ret = np.zeros((Mmax, len(ks)), dtype=np.int16)
    refusals, late_refusals, second_refusals, cosigned_first = 0, 0, 0, False
    for j in range(Mmax):
        recs = base + ([_dummy_record(e, dep.clock.t_e_us(e), rng)] if j else [])
        recs = sorted(recs, key=lambda r: (r.first_ts_us, bytes(r.meta[:37])))
        leaves = [leaf_of.get(id(r)) or r.leaf() for r in recs]
        store = MetadataStore(e, recs, taus=[tau if r.pay else 0 for r in recs])      # rev1: τ committed
        hdr = build_header(gw_id, dep.clock, EpochBatch(e, dep.clock.t_e_us(e), []), store, hs_.merkle_root(leaves),
                           fields, GENESIS_PREV, opening.commitment())
        pay = [r.pay for r in recs]
        _, _, _, Z = run_selection(keys.sk_V, hdr, beta, leaves, [tau if p else 0 for p in pay], pay)
        pos = {(r.first_ts_us, bytes(r.meta[:37])): i for i, r in enumerate(recs)}
        zt = [bool(Z[pos[k]]) for k in tkeys]
        for a, k in enumerate(ks):
            ret[j, a] = sum(zt[:k])
        sig = sign_header(keys.sk_G, hdr)
        if j == 0:                                           # honest header, on time (t_{e+1} + 1.5 s)
            witness.cosign(hdr, sig, dep.clock.t_e(e + 1) + 1.5)
            cosigned_first = True
        else:                                                # ground variant, only possible after β
            try:
                witness.cosign(hdr, sig, dep.clock.t_rstar(e) + 0.5)
            except WitnessRefusal:
                refusals += 1
                late_refusals += 1
            if j == 1:                                       # even before the deadline: second header refused
                try:
                    witness.cosign(hdr, sig, dep.clock.t_e(e + 1) + 1.6)
                except WitnessRefusal:
                    second_refusals += 1
    return {"e": e, "ret": ret, "refusals": refusals, "late_refusals": late_refusals,
            "second_refusals": second_refusals, "cosigned_first": cosigned_first, "n_units": len(base),
            "n_pay": len(pay_idx)}


def stage_grind(cfg: dict, run_tag: str, workers: int = 1, worker_index: int | None = None) -> None:
    from ctsample.crypto.signing import KeySet
    from ctsample.protocol.policy import ConstantPolicy
    from ctsample.protocol.witness import Witness
    g = cfg["grinding"]
    out = OUT / "grind"
    if done(out):
        log("grind done")
        return
    out.mkdir(parents=True, exist_ok=True)
    run_dir = OUT / "runs" / run_tag
    dep, meta = deployment(cfg)
    E = int(meta["n_epochs"])
    keys = KeySet.from_master((cfg["master_key"] + "/grind").encode())
    gw_id = gw_id_from_pk(keys.pk_G)
    policy = ConstantPolicy(g["pi"])
    witness = Witness(keys.sk_W)
    witness.register_gateway(gw_id, keys.pk_G, keys.pk_V, dep.clock, dep.clock.t_e(0) - 60, 0)
    Ms, ks = [int(x) for x in g["M"]], [int(x) for x in g["k"]]
    Mmax = max(Ms)
    n = int(g["trials"])
    rng0 = random.Random(int(g["seed"]))
    epochs = sorted(rng0.sample(range(1, E), n))
    raw_all = None
    wi = worker_index or 0
    for t, e in enumerate(epochs):
        if t % max(1, workers) != wi:
            continue
        p = out / f"t{t:04d}.npz"
        if p.exists():
            continue
        if raw_all is None:
            raw_all = O.load_records(run_dir)
        r = grind_trial(dep, keys, policy, witness, gw_id, raw_all[e], e, ks, Mmax, random.Random(f"{g['seed']}/{t}"))
        save_npz(p, ret=r["ret"], meta=np.array(json.dumps({k: v for k, v in r.items() if k != "ret"})))
        if t % 20 == 0:
            log(f"grind t={t} e={e}: refusals {r['refusals']}/{Mmax - 1}, second {r['second_refusals']}")
    if worker_index is not None:
        return
    # ---- aggregate
    R, metas = [], []
    for t in range(n):
        z = np.load(out / f"t{t:04d}.npz")
        R.append(z["ret"])
        metas.append(json.loads(str(z["meta"])))
    R = np.stack(R)                                          # trials × Mmax × len(ks)
    pi = float(policy.tau) / 2.0 ** 64
    rows = []
    for a, k in enumerate(ks):
        for M in Ms:
            best = R[:, :M, a].min(axis=1)                   # no TR: keep the variant with the fewest retained
            p_no = float(np.mean(best > 0))
            p_tr = float(np.mean(R[:, 0, a] > 0))            # TR: only the on-time header is cosigned (M = 1)
            f_no = (1 - (1 - pi) ** k) ** M
            f_tr = 1 - (1 - pi) ** k
            rows.append({"k": k, "M": M, "p_hat_noTR": p_no, "wilson_noTR": wilson(p_no, n), "formula_noTR": f_no,
                         "p_hat_TR": p_tr, "wilson_TR": wilson(p_tr, n), "formula_TR": f_tr, "trials": n})
    info = {"pi": pi, "pi_str": g["pi"], "trials": n, "Mmax": Mmax, "rows": rows,
            "witness_refusals": int(sum(m["refusals"] for m in metas)), "ground_headers": n * (Mmax - 1),
            "second_header_refusals": int(sum(m["second_refusals"] for m in metas)),
            "epochs": epochs, "mean_payload_units": float(np.mean([m["n_pay"] for m in metas])),
            "note": "real drand rounds, ECVRF, BLAKE3 PRF; background = the honest run's committed units of the "
                    "trial epoch; incident = k payload units fixed before β; candidate j ≥ 1 = one dummy "
                    "zero-payload unit inserted (new R_e, H(Hdr_e), α_e); constant π policy"}
    atomic_write_json(out / "grind.json", info)
    mark_done(out)
    log("grind written")


def wilson(p: float, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        return [float("nan"), float("nan")]
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


# ============================================================================================== chi2
def stage_chi2(cfg: dict, run_tag: str) -> None:
    from scipy import stats as sst
    from ctsample.sim.montecarlo import philox_epoch_uniforms, rep_keys, stable_id
    default = f"honest_p{cfg['counters']['p_loss'][0]:g}_s{cfg['counters']['main_seed']}"
    out = OUT / "chi2" if run_tag == default else OUT / "chi2" / run_tag
    if done(out):
        log("chi2 done")
        return
    out.mkdir(parents=True, exist_ok=True)
    run_dir = OUT / "runs" / run_tag
    z = np.load(run_dir / "units.npz")
    pay = z["pay"].astype(bool)
    pi = z["tau"][pay].astype(np.float64)
    Zr = z["Z"][pay].astype(bool)
    ep = z["epoch"][pay].astype(np.int64)
    E = int(z["epoch"].max()) + 1
    N = len(pi)
    B = int(cfg["chi2"]["n_bins"])
    order = np.argsort(pi, kind="stable")
    b = np.empty(N, dtype=np.int64)
    b[order] = (np.arange(N) * B) // N                       # equal-count π-bins
    Ob = np.bincount(b, weights=Zr, minlength=B)
    Eb = np.bincount(b, weights=pi, minlength=B)
    Vb = np.bincount(b, weights=pi * (1 - pi), minlength=B)
    X2 = float(np.sum((Ob - Eb) ** 2 / Vb))
    p_real = float(sst.chi2.sf(X2, B))
    Rn = int(cfg["chi2"]["mc_reps"])
    keys = rep_keys(stable_id(run_tag), stable_id("E5-chi2-philox"), range(Rn))
    n_e = np.bincount(ep, minlength=E)
    U = philox_epoch_uniforms(keys, n_e)
    Zm = U < pi[None, :]
    Om = np.stack([np.bincount(b, weights=Zm[r], minlength=B) for r in range(Rn)])
    X2m = np.sum((Om - Eb) ** 2 / Vb, axis=1)
    p_mc_each = sst.chi2.sf(X2m, B)
    X2_two = float(np.sum((Ob - Om.mean(0)) ** 2 / (Vb * (1 + 1.0 / Rn))))
    p_two = float(sst.chi2.sf(X2_two, B))
    # uniformity of the real PRF words (all payload units)
    raw = O.load_records(run_dir)
    ptr = np.searchsorted(z["epoch"], np.arange(E + 1))
    u = []
    for e in range(E):
        uu = raw[e]["u"]
        pe = z["pay"][int(ptr[e]):int(ptr[e + 1])]
        u.extend(uu[i] for i in np.nonzero(pe)[0])
    uf = np.asarray(u, dtype=np.float64) / 2.0 ** 64
    ks = sst.kstest(uf, "uniform")
    zsum = float((Zr.sum() - pi.sum()) / np.sqrt(np.sum(pi * (1 - pi))))
    info = {"n_payload_units": int(N), "bins": B, "bin_edges_pi": [float(pi[order][(k * N) // B]) for k in range(B)] + [float(pi.max())],
            "observed": Ob.tolist(), "expected": Eb.tolist(), "chi2_real": X2, "p_real": p_real, "df": B,
            "mc_reps": Rn, "chi2_mc_median": float(np.median(X2m)), "p_mc_median": float(np.median(p_mc_each)),
            "mc_fraction_p_below_0.01": float(np.mean(p_mc_each < 0.01)),
            "mc_tail_prob_of_real": float(np.mean(X2m >= X2)),
            "chi2_two_sample": X2_two, "p_two_sample": p_two,
            "ks_prf_words": {"n": int(len(uf)), "D": float(ks.statistic), "p": float(ks.pvalue)},
            "total": {"retained_real": int(Zr.sum()), "expected": float(pi.sum()), "z": zsum,
                      "retained_mc_mean": float(Zm.sum(1).mean())}}
    atomic_write_json(out / "chi2.json", info)
    mark_done(out)
    log(f"chi2: real χ²={X2:.2f} p={p_real:.3f}; two-sample p={p_two:.3f}; KS p={ks.pvalue:.3f}")


# ============================================================================================== a8 (cell level)
def _bins_log2(n):
    return np.floor(np.log2(np.asarray(n, dtype=np.float64) + 1.0)).astype(np.int64)


def stage_a8(cfg: dict, run_tag: str) -> None:
    out = SHARED / "a8"
    out.mkdir(parents=True, exist_ok=True)
    run_dir = OUT / "runs" / run_tag
    info_run = json.loads((run_dir / "run.json").read_text())
    target = int(info_run["devices"]["seed_device"])
    res = {"target_device": target}
    m_grid = np.unique(np.concatenate([np.arange(1, 21), np.round(np.logspace(np.log10(21), np.log10(20000), 60))])).astype(np.int64)
    for p in cfg["counters"]["p_loss"]:
        eps = load_eps(float(p))
        files = sorted((SHARED / "cells" / f"p{float(p):g}").glob("cells_s*.npz"))
        if not files:
            continue
        tabs = [np.load(f)["cells"] for f in files]
        C = np.concatenate(tabs)
        e_, dv, n_p, c_p, n_b, c_b = (C[:, j].astype(np.float64) for j in range(6))
        tol_p = np.array([eps.pkts(x) for x in n_p])
        tol_b = np.array([eps.bytes(x) for x in n_p])
        fl = (np.abs(c_p - n_p) > tol_p) | (np.abs(c_b - n_b) > tol_b)
        fl_p = np.abs(c_p - n_p) > tol_p
        per_seed = [float(np.mean((np.abs(t[:, 3] - t[:, 2]) > np.array([eps.pkts(x) for x in t[:, 2]])) |
                                  (np.abs(t[:, 5] - t[:, 4]) > np.array([eps.bytes(x) for x in t[:, 2]])))) for t in tabs]
        nb = _bins_log2(n_p)
        bpp = np.where(c_p > 0, c_b / np.maximum(c_p, 1), 0.0)
        curves = {}
        mstar = {}
        groups = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 13), (13, 30)]
        for lo, hi in groups:
            sel = (nb >= lo) & (nb < hi) & (c_p > 0)
            if not np.any(sel):
                continue
            cur = []
            for m in m_grid:
                mm = np.minimum(m, c_p[sel])
                dp = np.abs(c_p[sel] - mm - n_p[sel])
                db = np.abs(c_b[sel] - mm * bpp[sel] - n_b[sel])
                cur.append(float(np.mean((dp > tol_p[sel]) | (db > tol_b[sel]))))
            cur = np.asarray(cur)
            below = np.nonzero(cur >= 0.5)[0]
            ms = int(m_grid[below[0] - 1]) if len(below) and below[0] > 0 else (0 if len(below) else int(m_grid[-1]))
            key = f"n=[{2 ** lo - 1},{2 ** hi - 1})"
            curves[key] = {"n_cells": int(sel.sum()), "median_n": float(np.median(n_p[sel])), "p_flag": cur.tolist()}
            mstar[key] = ms
        # the attacker's own cells and its real C&C traffic
        z = np.load(run_dir / "units.npz")
        lab = np.load(run_dir / "labels.npz")["label"]
        cnc = (z["device"] == target) & z["pay"] & (lab == 1)
        pk_unit = z["pkts"][cnc]
        ep_c = z["epoch"][cnc]
        cnc_ep = np.bincount(ep_c, weights=pk_unit, minlength=int(z["epoch"].max()) + 1)
        # P(flag) when the gateway omits ALL C&C packets of an epoch (its bytes: the units' s − 16·pkts), per seed
        cnc_b = np.bincount(ep_c, weights=z["s"][cnc] - 16 * pk_unit, minlength=len(cnc_ep))
        hidden = []
        for t in tabs:
            m_t = t[:, 1] == target
            tt = t[m_t]
            for row in tt:
                e = int(row[0])
                if e >= len(cnc_ep) or cnc_ep[e] <= 0:
                    continue
                m = cnc_ep[e]
                dp = abs(row[3] - min(m, row[3]) - row[2])
                db = abs(row[5] - cnc_b[e] - row[4])
                hidden.append((e, m, row[2], dp > eps.pkts(row[2]) or db > eps.bytes(row[2])))
        hidden = np.array(hidden, dtype=np.float64).reshape(-1, 4)
        tgt_sel = (dv == target) & (c_p > 0)
        tcur = []
        for m in m_grid:
            mm = np.minimum(m, c_p[tgt_sel])
            dp = np.abs(c_p[tgt_sel] - mm - n_p[tgt_sel])
            db = np.abs(c_b[tgt_sel] - mm * bpp[tgt_sel] - n_b[tgt_sel])
            tcur.append(float(np.mean((dp > tol_p[tgt_sel]) | (db > tol_b[tgt_sel]))))
        tcur = np.asarray(tcur)
        below = np.nonzero(tcur >= 0.5)[0]
        res[f"p{float(p):g}"] = {
            "n_cells": int(len(C)), "seeds": len(tabs), "far_cell": float(fl.mean()), "far_cell_pkts_only": float(fl_p.mean()),
            "far_cell_per_seed": per_seed, "far_cell_wilson": wilson(float(fl.mean()), len(C)),
            "m_grid": m_grid.tolist(), "curves": curves, "m_star": mstar,
            "m_star_note": f"largest m on the grid with P(flag) < 0.5; {int(m_grid[-1])} = never reaches 0.5 (the "
                           "whole cell, ≤ 2 packets, can be omitted unflagged)",
            "target": {"n_cells": int(tgt_sel.sum()), "median_counter_pkts": float(np.median(n_p[tgt_sel])),
                       "p90_counter_pkts": float(np.quantile(n_p[tgt_sel], 0.9)), "p_flag": tcur.tolist(),
                       "m_star": int(m_grid[below[0] - 1]) if len(below) and below[0] > 0 else 0,
                       "far_cell": float(fl[dv == target].mean())},
            "cnc": {"units": int(cnc.sum()), "pkts_per_unit_median": float(np.median(pk_unit)),
                    "pkts_per_unit_p90": float(np.quantile(pk_unit, 0.9)), "pkts_per_unit_max": int(pk_unit.max()),
                    "epochs_with_cnc": int((cnc_ep > 0).sum()),
                    "pkts_per_epoch_median": float(np.median(cnc_ep[cnc_ep > 0])),
                    "omit_all_cnc_of_epoch_flag_rate": float(hidden[:, 3].mean()) if len(hidden) else None,
                    "omit_all_cnc_of_epoch_n": int(len(hidden))},
        }
        log(f"a8 p={p}: FAR/cell {fl.mean():.5f}; m* {mstar}; target m* {res[f'p{float(p):g}']['target']['m_star']}")
    atomic_write_json(out / "a8.json", res)
    mark_done(out)


# ============================================================================================== summary
def _load_trials(run_tag: str) -> dict[str, list[dict]]:
    out = {}
    base = OUT / "trials" / run_tag
    for d in sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []:
        out[d.name] = [json.loads(p.read_text()) for p in sorted(d.glob("t*.json"))]
    return out


def stage_summary(cfg: dict, run_tag: str) -> dict:
    from collections import Counter
    run_dir = OUT / "runs" / run_tag
    run = json.loads((run_dir / "run.json").read_text())
    trials = _load_trials(run_tag)
    honest = [t for t in trials.get("honest", []) if "error" not in t]
    res = {"stream": cfg["stream"], "run_tag": run_tag, "config": cfg,
           "run": {k: run[k] for k in ("n_epochs", "shift_us", "t0_gateway_us", "Bbar", "rho", "policy", "loss",
                                       "retention", "spend_over_target", "devices", "labels", "storage", "machine")},
           "validation": {k: v for k, v in run["validation"].items() if k != "flowtable_qa"},
           "gateway_pass": {k: v for k, v in run["gateway_pass"].items() if k != "flowtable_qa"},
           "bls": "not verified: no BLS12-381 library (py_ecc/blst) is installed; V2 checks randomness = "
                  "SHA-256(signature), the chained link to r*−1 and equality with the stored drand copy, BLS → SKIP"}
    # ---- honest windows: FAR per check
    if honest:
        n = len(honest)
        cnt = Counter()
        for t in honest:
            for nm in {f"{f['check']}.{f['name']}" for f in t["fired"] if f["status"] == "FAIL"}:
                cnt[nm] += 1
        cells = sum(t["n_cells"] for t in honest)
        flagged = sum(t["v7_flagged_cells"] for t in honest)
        res["honest"] = {"windows": n, "window_epochs": int(cfg["trials"]["window"]),
                         "fail_rate_by_check": {k: v / n for k, v in sorted(cnt.items())},
                         "any_fail_V1_V6": sum(1 for t in honest if any(f["status"] == "FAIL" and f["check"] in
                                                                        ("V1", "V2", "V3", "V4", "V5", "V6")
                                                                        for f in t["fired"])) / n,
                         "v7_cells": cells, "v7_flagged_cells": flagged, "v7_far_cell": flagged / max(1, cells),
                         "v7_far_cell_wilson": wilson(flagged / max(1, cells), cells),
                         "v7_window_any_flag": sum(1 for t in honest if t["v7_flagged_cells"]) / n,
                         "verify_s_median": float(np.median([t["verify_s"] for t in honest])),
                         "verify_s_p90": float(np.quantile([t["verify_s"] for t in honest], 0.9)),
                         "prime": honest[0]["prime"] and {k: honest[0]["prime"][k] for k in
                                                          ("epochs", "header_bytes", "store_bytes", "total_bytes",
                                                           "store_records")},
                         "disclosure_bytes_median": float(np.median([t["disclosure"]["total_bytes"] for t in honest]))}
    # ---- adversaries
    adv_out = {}
    for adv, ts in trials.items():
        if adv == "honest":
            continue
        ok = [t for t in ts if "error" not in t]
        eff = [t for t in ok if t.get("effective", True)]
        det = [t for t in eff if t.get("detected")]
        fired = Counter()
        for t in eff:
            for nm in set(t.get("fail_names", [])) | set(t.get("warn_names", [])):
                fired[nm] += 1
        by_var = {}
        for t in eff:
            v = t.get("variant") or "-"
            d = by_var.setdefault(v, [0, 0])
            d[0] += 1
            d[1] += bool(t.get("detected"))
        exp = sorted(EXPECTED.get(adv, set()))
        far = None
        if honest and exp:
            far = sum(1 for t in honest if {f"{f['check']}.{f['name']}" for f in t["fired"] if f["status"] in
                                            ("FAIL", "WARN")} & set(exp)) / len(honest)
            if adv == "A8":
                far = res["honest"]["v7_far_cell"]           # the attacked cell, honest
        rate = len(det) / len(eff) if eff else None
        a = {"trials": len(ts), "errors": len(ts) - len(ok), "effective": len(eff), "detected": len(det),
             "detection_rate": rate, "wilson": wilson(rate, len(eff)) if eff else None, "expected_checks": exp,
             "checks_fired_share": {k: v / len(eff) for k, v in sorted(fired.items())} if eff else {},
             "by_variant": {k: {"n": v[0], "detected": v[1]} for k, v in sorted(by_var.items())},
             "far": far, "verify_s_median": float(np.median([t["verify_s"] for t in ok])) if ok else None}
        if adv == "A8":
            by_m = {}
            for t in eff:
                d = by_m.setdefault(int(t["m"]), [0, 0, []])
                d[0] += 1
                d[1] += bool(t["detected"])
                if t.get("cell"):
                    d[2].append(t["cell"]["counter_pkts"])
            a["by_m"] = {m: {"n": v[0], "detected": v[1], "rate": v[1] / v[0], "median_cell_n": float(np.median(v[2]))
                             if v[2] else None} for m, v in sorted(by_m.items())}
            a["other_checks_fired"] = sorted({nm for t in eff for nm in t.get("fail_names", []) if nm != "V7.cells"})
            # consistency with the cell level: attacked committed packets = honest committed − removed
            cz = np.load(run_dir / "cells.npz")["cells"]
            hon = {(int(r[0]), int(r[1])): (int(r[3]), int(r[2])) for r in cz}
            ok_n, bad = 0, []
            for t in eff:
                c = t.get("cell")
                if not c:
                    continue
                hc, hn = hon.get((t["win"][2], c["device_id"]), (0, 0))
                if c["committed_pkts"] == hc - t["removed"] and c["counter_pkts"] == hn:
                    ok_n += 1
                else:
                    bad.append(t["t"])
            a["cell_consistency"] = {"consistent": ok_n, "inconsistent_trials": bad[:20]}
            rb = {}
            for t in eff:
                k = "1-3" if t["removed"] <= 3 else "4-9" if t["removed"] <= 9 else "10-29" if t["removed"] <= 29 \
                    else "30-99" if t["removed"] <= 99 else ">=100"
                x = rb.setdefault(k, [0, 0])
                x[0] += 1
                x[1] += bool(t["detected"])
            a["by_removed"] = {k: {"n": v[0], "detected": v[1], "rate": v[1] / v[0]} for k, v in rb.items()}
        adv_out[adv] = a
    res["adversaries"] = adv_out
    for name, path in (("eps", None), ("grind", OUT / "grind" / "grind.json"), ("chi2", OUT / "chi2" / "chi2.json"),
                       ("a8_cells", SHARED / "a8" / "a8.json")):
        if name == "eps":
            res["eps_cell"] = {}
            for p in cfg["counters"]["p_loss"]:
                f = SHARED / "calib" / f"p{float(p):g}" / "eps.json"
                if f.exists():
                    d = json.loads(f.read_text())
                    res["eps_cell"][f"p{float(p):g}"] = {k: d[k] for k in ("n_pairs", "in_sample_far", "far_target",
                                                                          "eps", "seeds", "source", "n_epochs")}
        elif path.exists():
            res[name] = json.loads(path.read_text())
    reps = {}
    for f in sorted((OUT / "chi2").glob("*/chi2.json")):
        x = json.loads(f.read_text())
        reps[f.parent.name] = {k: x[k] for k in ("chi2_real", "p_real", "p_two_sample", "ks_prf_words", "total",
                                                 "n_payload_units")}
    res["chi2_replicates"] = reps
    # ---- omitted boost with a FULL-window disclosure (every epoch that can justify the boost at ex disclosed)
    fw = None
    if "A10omitfull" in trials:
        att = [t for t in trials["A10omitfull"] if "error" not in t and t.get("effective", True)]
        hon = [t for t in trials.get("honestfull", []) if "error" not in t]
        fw = {"source": f"{OUT.name}/trials/{run_tag}/A10omitfull + honestfull", "window_epochs": FULL_W,
              "attacked_epoch": "a + 21", "attack": {"n": len(att), "detected": sum(1 for t in att if t["detected"]),
                                                   "rules": sorted({r for t in att for r in t.get("rules_at_ex", [])})},
              "honest": {"n": len(hon), "false_alarms": sum(1 for t in hon if t.get("policy_compliance_warn"))}}
    elif (RESULTS / "checks_p1d" / "a10omit_fullwindow" / "summary.json").exists():
        d = json.loads((RESULTS / "checks_p1d" / "a10omit_fullwindow" / "summary.json").read_text())
        fw = {"source": "results/checks_p1d/a10omit_fullwindow (P1-D harness, 19 Oct; 25-epoch windows, ex = a + 21)",
              "window_epochs": 25, "attacked_epoch": "a + 21",
              "attack": {"n": d["attack"]["n"], "detected": d["attack"]["detected"], "rules": d["attack"]["rules"]},
              "honest": {"n": d["honest"]["n"], "false_alarms": d["honest"]["detected"]}}
    if fw is not None:
        res["a10omit_fullwindow"] = fw
    if "A10omit" in res.get("adversaries", {}):
        res["adversaries"]["A10omit"]["note"] = ("5-epoch windows, attacked epoch win[2]: detected by the per-unit "
                                                "omitted-boost proof whenever a justifying retained unit is in the window; "
                                                "the full-window check is 'a10omit_fullwindow'")
    # ---- containment of a void epoch (rev1) / the rev0 A7 behaviour
    cs = contain_summary(run_tag)
    if cs:
        res["containment"] = cs
    a7 = [t for t in trials.get("A7", []) if "error" not in t]
    if a7:
        after, failed_after, w_after = 0, 0, 0
        for t in a7:
            ex = t.get("withheld")
            later = [e for e in t["win"] if e > ex]
            f_e = {f["e"] for f in t["fired"] if f["status"] == "FAIL" and f["check"] in ("V1", "V2", "V3", "V4", "V5", "V6")}
            after += len(later)
            failed_after += sum(1 for e in later if e in f_e)
        res["a7_later_epochs"] = {"definition": "A7 trials (5-epoch windows, withheld epoch win[2]): share of the window "
                                                "epochs AFTER the withheld one with a V1–V6 FAIL (their requested units "
                                                "enter W)", "epochs": after, "failed": failed_after,
                                  "failed_share": failed_after / max(1, after),
                                  "semantics": sorted({t.get("semantics", "rev0") for t in a7})}
    SUMMARY.mkdir(parents=True, exist_ok=True)
    atomic_write_json(SUMMARY / summary_name(), res)
    log("summary written", SUMMARY / summary_name())
    return res


# ============================================================================================== main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["calib", "honest", "cells", "trials", "grind", "chi2", "a8", "contain", "summary"])
    ap.add_argument("--rev", default="rev1", help="rev1 (results/E5_rev1, default) or rev0 (the original results/E5)")
    ap.add_argument("--n", type=int, default=30, help="contain: trials per scenario")
    ap.add_argument("--scenarios", nargs="*", default=None, help="contain: subset of withheld voided late")
    ap.add_argument("--p-loss", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--adversaries", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker-index", type=int, default=None)
    ap.add_argument("--master", default=None, help="override the deployment master key (independent replicate)")
    ap.add_argument("--no-frames", action="store_true", help="honest run without the payload-frame dump")
    args = ap.parse_args(argv)
    set_rev(args.rev)
    cfg = load_cfg()
    if args.master:
        cfg["master_key"] = args.master
    c = cfg["counters"]
    if args.stage == "calib":
        for p in ([args.p_loss] if args.p_loss is not None else c["p_loss"]):
            stage_calib(cfg, float(p))
    elif args.stage == "honest":
        p = float(args.p_loss if args.p_loss is not None else c["p_loss"][0])
        stage_honest(cfg, p, int(args.seed if args.seed is not None else c["main_seed"]), args.tag,
                     save_frames=not args.no_frames)
    elif args.stage == "cells":
        p = float(args.p_loss if args.p_loss is not None else c["p_loss"][0])
        stage_cells(cfg, p, args.seeds if args.seeds else list(c["test_seeds"]))
    elif args.stage == "trials":
        tag = args.tag or f"honest_p{c['p_loss'][0]:g}_s{c['main_seed']}"
        default_advs = ADVERSARIES + (ADVERSARIES_REV1 if REV == "rev1" else [])
        stage_trials(cfg, tag, args.adversaries or default_advs, args.workers, args.worker_index)
    elif args.stage == "contain":
        tag = args.tag or f"honest_p{c['p_loss'][0]:g}_s{c['main_seed']}"
        stage_contain(cfg, tag, args.scenarios or list(CONTAIN_SCENARIOS), args.n, args.workers, args.worker_index)
    elif args.stage in ("grind", "chi2", "a8"):
        tag = args.tag or f"honest_p{c['p_loss'][0]:g}_s{c['main_seed']}"
        if args.stage == "grind":
            if args.workers > 1 and args.worker_index is None:
                import subprocess
                procs = [subprocess.Popen([sys.executable, "-I", "-u", __file__, "grind", "--rev", REV, "--tag", tag, "--workers",
                                           str(args.workers), "--worker-index", str(w)]) for w in range(args.workers)]
                if any(p.wait() for p in procs):
                    raise SystemExit("grind workers failed")
                stage_grind(cfg, tag)
            else:
                stage_grind(cfg, tag, args.workers, args.worker_index)
        elif args.stage == "chi2":
            stage_chi2(cfg, tag)
        else:
            stage_a8(cfg, tag)
    elif args.stage == "summary":
        stage_summary(cfg, args.tag or f"honest_p{c['p_loss'][0]:g}_s{c['main_seed']}")
    else:
        raise SystemExit(f"stage {args.stage} not implemented yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
