#!/usr/bin/env python
"""E6 — overheads, storage and witness load (DESIGN §3.11, §6.1 row E6; paper Table VI `tab:e6`).

Stages (parents launch single-core children; every timing records the machine state — other jobs may run):
  pipeline   per-packet pipeline cost on the E5 stream (first 100 epochs, real CtS policy, real crypto):
             5 runs on P-cores and 5 on E-cores (`taskpolicy -b`), OMP_NUM_THREADS=1 → µs/packet, pps/core,
             commit latency, per-epoch breakdown.
  epochs     per-epoch costs vs n_e ∈ {10², …, 10⁶}: leaves, Merkle, store hash, header + σ, C^st, ECVRF prove /
             verify, PRF words, inclusion proof (bytes, verify), 5 reps, P and E.
  primitives P1-A's microbenchmarks (bench/primitives.py) on P and E cores.
  verifier   verifier time for 5-epoch windows of the E5 honest run (+ the 119-epoch prime bundle), P and E.
  buffer     buffer occupancy (p50/p99/max) at T = 30 s for selection one and two rounds ahead on LP streams
             (per-packet arrival times from the virtual-capture lossless pass).
  storage    zstd-columnar metadata store (DESIGN columns meta + root^pkt) + retained content vs full pcap.
  summary    results/summary/E6_overheads.json (witness load, anchoring cost model, primitive counts and the
             ARM extrapolation are computed here).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import yaml

from ctsample.bench import overhead as O
from ctsample.utils.io import P1_ROOT, PROJECT_ROOT, RESULTS, atomic_write_json

CFG = P1_ROOT / "config" / "experiments" / "E6.yaml"
CFG5 = P1_ROOT / "config" / "experiments" / "E5.yaml"
OUT = RESULTS / "E6"
OUT_REV0 = RESULTS / "E6"
REV = "rev0"                 # --rev rev1: results/E6_rev1/, the E5 rev1 run, summary E6_overheads_rev1.json
SUMMARY = RESULTS / "summary"
E5RUN = RESULTS / "E5" / "runs" / "honest_p0.001_s100"
T_US = 30_000_000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cfgs():
    return yaml.safe_load(CFG.read_text()), yaml.safe_load(CFG5.read_text())


def set_rev(rev: str) -> None:
    global REV, OUT, E5RUN
    REV = rev
    if rev == "rev1":
        OUT = RESULTS / "E6_rev1"
        E5RUN = RESULTS / "E5_rev1" / "runs" / "honest_p0.001_s100"


def child(stage: str, core: str, extra: list[str]) -> int:
    cfg, _ = cfgs()
    env = dict(os.environ, OMP_NUM_THREADS=str(cfg["omp_threads"]), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1")
    cmd = [sys.executable, "-I", "-u", __file__, stage, "--core", core, "--rev", REV, *extra]
    if core == "E":
        cmd = ["taskpolicy", "-b", *cmd]
    log("run", " ".join(cmd))
    return subprocess.call(cmd, env=env, cwd=str(PROJECT_ROOT))


def deployment_and_policy(cfg5):
    import importlib.util
    spec = importlib.util.spec_from_file_location("e5", P1_ROOT / "scripts" / "14_e5.py")
    e5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e5)
    dep, meta = e5.deployment(cfg5)
    pol, Bbar = e5.load_policy(cfg5, meta)
    return e5, dep, meta, pol, Bbar


# ============================================================================================== pipeline
def cache_registration(pol) -> None:
    """Instance-level memo of (H(θ), H(policy)): both are constants of the registered policy, but CtSPolicy
    recomputes H(θ) over the whole model bundle in every commit_fields()/registration() call (≈ 40–55 ms per
    call on a P-core). The protocol output is unchanged; E6 reports the pipeline with and without the memo."""
    H = pol.registration()
    pol.registration = (lambda H=H: H)


def run_pipeline(core: str, rep: int, cached: bool = False) -> None:
    cfg, cfg5 = cfgs()
    out = OUT / ("pipeline_cached" if cached else "pipeline") / f"{core}_r{rep}.json"
    if out.exists():
        return
    e5, dep, meta, pol, Bbar = deployment_and_policy(cfg5)
    if cached:
        cache_registration(pol)
    vc0 = O.virtual_capture(cfg["stream"])
    vc = vc0.with_shift(O.align_shift(dep.clock, vc0))
    registry = pickle.loads((E5RUN / "registry.pkl").read_bytes())
    m0 = O.machine_state()
    t0 = time.time()
    r = O.time_pipeline(vc, dep, pol, int(cfg["pipeline_epochs"]), registry=registry)
    r.update(core=core, rep=rep, wall_s=time.time() - t0, machine_before=m0, machine_after=O.machine_state(),
             omp_threads=os.environ.get("OMP_NUM_THREADS"), cached_registration=cached)
    atomic_write_json(out, r)
    log(f"pipeline {core} r{rep}: {r['us_per_packet']['total']:.1f} µs/packet, {r['pps_per_core']:.0f} pps; "
        f"load {m0['loadavg']}")


# ============================================================================================== epochs
def run_epochs(core: str) -> None:
    from ctsample.crypto import ecvrf
    from ctsample.crypto import hashing as hs
    from ctsample.crypto.prf import select_units, selection_key
    from ctsample.protocol.epoch import make_opening, vrf_input
    from ctsample.protocol.header import Header, sign_header, verify_header
    from ctsample.protocol.policy import ConstantPolicy
    from ctsample.protocol.store import MetadataStore
    cfg, cfg5 = cfgs()
    out = OUT / "epochs" / f"{core}.json"
    if out.exists():
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("e5", P1_ROOT / "scripts" / "14_e5.py")
    e5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e5)
    dep, meta = e5.deployment(cfg5)
    raw = O.load_records(E5RUN, range(0, 30))
    pool = [r for e in sorted(raw) for r in MetadataStore.deserialize(raw[e]["store"]).records]
    rng = np.random.default_rng(1)
    pol = ConstantPolicy("0.05")
    fields = pol.commit_fields(pol.initial_state(0))
    keys = dep.keys
    beta = dep.beacon.get(dep.clock.r_star(0)).randomness
    m0 = O.machine_state()
    res = {"core": core, "machine_before": m0, "rows": []}
    reps = int(cfg["reps"])
    for n in cfg["n_e"]:
        recs = O.synthetic_records(int(n), pool, rng)
        st = MetadataStore(0, recs, taus=[pol.tau if r.pay else 0 for r in recs] if REV == "rev1" else None)
        row = {"n_e": int(n), "n_pay": int(sum(r.pay for r in recs))}
        T = {k: [] for k in ("leaves", "merkle", "store_hash", "header_sign", "state_commitment", "vrf_prove",
                             "vrf_verify", "prf_select", "proof_verify", "verifier_recompute")}
        for _ in range(reps):
            a = time.perf_counter()
            leaves = [r.leaf() for r in recs]
            b = time.perf_counter()
            R = hs.merkle_root(leaves)
            c = time.perf_counter()
            Hs = st.digest()
            d = time.perf_counter()
            op = make_opening(keys.k_st, 0, fields, ())
            Cst = op.commitment()
            e_ = time.perf_counter()
            tot = st.totals()
            hdr = Header(gw_id=b"\x00" * 16, e=0, t_e_us=dep.clock.t_e_us(0), T_us=30_000_000, n=tot["n"],
                         n_pay=tot["n_pay"], sum_s_pay=tot["sum_s_pay"], sum_s=tot["sum_s"], R=R, H_store=Hs,
                         H_theta=fields["H_theta"], H_policy=fields["H_policy"], tau_phi=int(fields["tau_phi"]),
                         C_st=Cst, r_star=dep.clock.r_star(0), H_prev=b"\x00" * 32)
            sig = sign_header(keys.sk_G, hdr)          # header fields (store totals O(n)), encode, digest, σ
            f = time.perf_counter()
            pi = ecvrf.prove(keys.sk_V, vrf_input(hdr, beta))
            g = time.perf_counter()
            ok, bv = ecvrf.verify(keys.pk_V, vrf_input(hdr, beta), pi)
            h = time.perf_counter()
            taus = [pol.tau if r.pay else 0 for r in recs]
            select_units(selection_key(bv), 0, leaves, taus, [r.pay for r in recs])
            t_sel = time.perf_counter() - h
            j = int(rng.integers(0, n))
            prf = hs.merkle_proof(leaves, j)
            i_ = time.perf_counter()                  # proof generation (from scratch) not timed: O(n)
            okp = hs.verify_proof(leaves[j], j, int(n), prf, R)
            k_ = time.perf_counter()
            st2 = MetadataStore.deserialize(st.serialize())
            st2.digest()
            hs.merkle_root(st2.leaves())
            verify_header(keys.pk_G, hdr, sig)
            l_ = time.perf_counter()
            assert ok and okp
            for key, v in (("leaves", b - a), ("merkle", c - b), ("store_hash", d - c), ("state_commitment", e_ - d),
                           ("header_sign", f - e_), ("vrf_prove", g - f), ("vrf_verify", h - g),
                           ("prf_select", t_sel), ("proof_verify", k_ - i_), ("verifier_recompute", l_ - k_)):
                T[key].append(v)
        row.update({k: 1e3 * float(np.median(v)) for k, v in T.items()})
        row["inclusion_proof_bytes"] = 32 * len(prf)
        row["store_bytes"] = len(st.serialize())
        row["commit_total_ms"] = row["leaves"] + row["merkle"] + row["store_hash"] + row["state_commitment"] + \
            row["header_sign"]
        res["rows"].append(row)
        log(f"epochs {core} n_e={n}: leaves {row['leaves']:.1f} ms, merkle {row['merkle']:.1f} ms, "
            f"store {row['store_hash']:.1f} ms, prove {row['vrf_prove']:.1f} ms")
    res["machine_after"] = O.machine_state()
    atomic_write_json(out, res)


# ============================================================================================== verifier
def run_verifier(core: str, cached: bool = False) -> None:
    cfg, cfg5 = cfgs()
    out = OUT / ("verifier_cached" if cached else "verifier") / f"{core}.json"
    if out.exists():
        return
    import random
    import importlib.util
    spec = importlib.util.spec_from_file_location("e5", P1_ROOT / "scripts" / "14_e5.py")
    e5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e5)
    from ctsample.protocol.prime import check_prime
    from ctsample.protocol.verifier import TrustAnchors
    ctx = e5.TrialCtx(cfg5, E5RUN)
    if cached:
        cache_registration(ctx.pristine)                    # fresh_policy() copies share the instance memo
    rng = random.Random(6)
    rows = []
    m0 = O.machine_state()
    for w in range(int(cfg["verifier_windows"])):
        a = rng.randrange(120, ctx.E - 5)
        win = list(range(a, a + 5))
        view = ctx.view(win)
        disc = ctx.disclose(view, win)
        bundle = ctx.prime(win)
        trust = TrustAnchors.from_disclosure(disc, ctx.dep.keys.pk_W, pk_I=ctx.dep.keys.pk_I, beacon=ctx.dep.beacon)
        t0 = time.perf_counter()
        check_prime(bundle, trust, disc)
        t_prime_check = time.perf_counter() - t0
        rep, sec = ctx.verify(disc, bundle)
        fast = {}
        if getattr(ctx, "semantics", "rev0") == "rev1":
            # rev1 validity without the prime bundle and without the policy recomputation (committed τ only)
            rep_f, sec_f = ctx.verify(disc, None, recompute_compliance=False)
            fast = {"verify_fast_s": sec_f, "n_fail_v1_v6_fast": sum(
                1 for r in rep_f.results if r.status == "FAIL" and r.check in ("V1", "V2", "V3", "V4", "V5", "V6"))}
        sz = disc.size_report()
        rows.append({"win": win, "verify_s": sec, "prime_check_s": t_prime_check, "n_fail_v1_v6":
                     sum(1 for r in rep.results if r.status == "FAIL" and r.check in ("V1", "V2", "V3", "V4", "V5", "V6")),
                     "disclosure_bytes": sz["total_bytes"], "unit_bytes": sz["unit_bytes"], "store_bytes": sz["store_bytes"],
                     "prime_bytes": rep.prime["total_bytes"], "units": sum(len(d.units) for d in disc.epochs.values()),
                     **fast})
    res = {"core": core, "cached_registration": cached, "rows": rows, "machine_before": m0, "machine_after": O.machine_state(),
           "median_verify_s_5_epochs": float(np.median([r["verify_s"] for r in rows])),
           "median_prime_check_s": float(np.median([r["prime_check_s"] for r in rows]))}
    if rows and "verify_fast_s" in rows[0]:
        res["median_verify_fast_s_5_epochs"] = float(np.median([r["verify_fast_s"] for r in rows]))
    atomic_write_json(out, res)
    log(f"verifier {core}: median {res['median_verify_s_5_epochs']:.2f} s per 5-epoch window")


# ============================================================================================== buffer
def run_buffer() -> None:
    cfg, cfg5 = cfgs()
    tm = cfg["timing"]
    for sid in cfg["buffer_streams"]:
        out = OUT / "buffer" / f"{sid}.json"
        if out.exists():
            continue
        vc = O.virtual_capture(sid)
        t0 = time.time()
        r = O.unitise(vc, keep_frames=False, track=True, max_epoch=vc.n_epochs)
        pay = r.unit_pay[r.pk_unit]
        b = r.pk_len.astype(np.float64) + 16.0
        res = {"stream": sid, "packets": int(r.n_packets), "payload_packets": int(pay.sum()),
               "duration_s": vc.n_epochs * 30.0, "pps": r.n_packets / (vc.n_epochs * 30.0), "pass_s": time.time() - t0}
        for ahead in (1, 2):
            res[f"ahead{ahead}"] = O.buffer_occupancy(r.pk_ts[pay], b[pay], vc.t0_gw_us, vc.n_epochs, T_s=tm["T"],
                                                      delta_c=tm["delta_c"], delta_sel=tm["delta_sel"], ahead=ahead)
        res["all_bytes_ahead1"] = O.buffer_occupancy(r.pk_ts, b, vc.t0_gw_us, vc.n_epochs, T_s=tm["T"],
                                                     delta_c=tm["delta_c"], delta_sel=tm["delta_sel"], ahead=1)
        atomic_write_json(out, res)
        log(f"buffer {sid}: pps {res['pps']:.0f}; ahead1 max {res['ahead1']['max_bytes'] / 1e6:.2f} MB, "
            f"ahead2 max {res['ahead2']['max_bytes'] / 1e6:.2f} MB")


# ============================================================================================== storage
def _zstd_design_bytes(metas: list, roots: list) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.table({"meta": pa.array(metas, type=pa.binary()), "root_pkt": pa.array(roots, type=pa.binary(32))})
    buf = io.BytesIO()
    pq.write_table(t, buf, compression="zstd")
    return buf.tell()


def run_storage() -> None:
    import polars as pl
    cfg, cfg5 = cfgs()
    rows = []
    for sid in cfg["storage_streams"]:
        sd = RESULTS / "streams" / sid
        if not (sd / "units.parquet").exists():
            continue
        u = pl.read_parquet(sd / "units.parquet", columns=["source", "unit_gid", "pay", "s", "meta"])
        m = json.loads((sd / "meta.json").read_text())
        rho = 0.05
        pay = u["pay"].to_numpy()
        # root^pkt: 32 incompressible bytes per payload unit (deterministic stand-in: BLAKE3 of the meta)
        from blake3 import blake3
        metas = u["meta"].to_list()
        roots = [blake3(x).digest() if p else None for x, p in zip(metas, pay)]
        zs = _zstd_design_bytes(metas, roots)
        raw = int(sum(len(x) for x in metas) + 32 * int(pay.sum()))
        s_all = float(u["s"].sum())
        content = rho * s_all                                   # CtS spends B̄E = ρ Σ s by design (spend/target ≈ 1)
        rows.append({"stream": sid, "units": int(u.height), "payload_units": int(pay.sum()), "sum_s_all": s_all,
                     "pcap_bytes": s_all + 24, "meta_raw_design_bytes": raw, "meta_zstd_design_bytes": zs,
                     "meta_over_pcap": zs / s_all, "content_rho5_bytes": content,
                     "storage_over_pcap": (zs + content) / s_all, "content_share_of_storage": content / (zs + content),
                     "n_epochs": int(m["n_epochs"])})
        log(f"storage {sid}: meta/pcap {zs / s_all:.4f}, storage/pcap {(zs + content) / s_all:.4f}")
    # the E5 honest run (exact: real committed stores, real retained content)
    r = json.loads((E5RUN / "run.json").read_text())["storage"]
    e5 = {"stream": "E5 honest run (lp_iot23_34-1_p10000, committed)", **r,
          "meta_over_pcap": r["parquet_zstd_design_bytes"] / r["sum_s_all"],
          "storage_over_pcap": (r["parquet_zstd_design_bytes"] + r["retained_plaintext_s"]) / r["sum_s_all"],
          "storage_over_pcap_ciphertext": (r["parquet_zstd_design_bytes"] + r["retention_bytes"]) / r["sum_s_all"]}
    atomic_write_json(OUT / "storage.json", {"rows": rows, "e5_run": e5,
                                              "note": "content = ρ Σ s (ρ = 5 %) for the streams; exact retained "
                                                      "bytes for the E5 run; pcap = Σ (caplen + 16) + 24"})


# ============================================================================================== flood (rev1)
OUT_REV1 = RESULTS / "E6_rev1"
E5RUN_REV1 = RESULTS / "E5_rev1" / "runs" / "honest_p0.001_s100"
FLOOD_EPOCH = 400                               # a mid-stream epoch of the E5 rev1 honest run (≈ 217 units)
BUDGET_ONE_AHEAD_S = 0.8                        # Δ_build of the commit window (Prop. 5(i)), one round ahead
BUDGET_TWO_AHEAD_S = 0.8 + 30.0                 # designated round r0 + e + 2: one more drand period
UTILITY_CAP = 4096                              # payload units scored per device and epoch (never binds on our data)


def _flood_cfgs(quick: bool = False) -> list[dict]:
    """Flood benchmark grid: kind (zero / payload), history (single / sustained), rolling-statistics implementation
    (incremental = rev1 default, resort = the original), zero-payload cap (protocol/flood.py), utility cap, n, reps."""
    out = []

    def add(kind, hist, impl, cap, ucap, ns):
        for n in ns:
            reps = 3 if n <= 30000 else (2 if n <= 100000 else 1)
            out.append({"kind": kind, "history": hist, "impl": impl, "cap": cap, "ucap": ucap, "n": n, "reps": reps})

    if quick:
        add("zero", "single", "incremental", False, None, [0, 1000, 30000])
        add("zero", "sustained", "resort", False, None, [1000, 10000])
        add("zero", "sustained", "incremental", False, None, [1000, 10000])
        add("zero", "sustained", "incremental", True, None, [30000])
        add("payload", "single", "incremental", False, None, [1000])
        add("payload", "single", "incremental", False, UTILITY_CAP, [10000])
        return out
    add("zero", "single", "incremental", False, None, [0, 100, 1000, 10000, 30000, 100000, 300000, 1000000])
    add("zero", "single", "incremental", True, None, [10000, 30000, 100000, 300000, 1000000])
    add("zero", "single", "resort", False, None, [0, 10000, 100000, 1000000])
    add("zero", "sustained", "resort", False, None, [1000, 3000, 10000, 30000, 100000])
    add("zero", "sustained", "incremental", False, None, [1000, 3000, 10000, 30000, 100000, 300000])
    add("zero", "sustained", "incremental", True, None, [10000, 30000, 100000, 1000000])
    add("payload", "single", "incremental", False, None, [100, 1000, 3000, 10000, 30000, 100000])
    add("payload", "single", "incremental", False, UTILITY_CAP, [10000, 30000, 100000])
    return out


def _cfg_name(c: dict) -> str:
    return (f"{c['kind']}_{c['history']}_{c['impl']}_{'cap' if c['cap'] else 'nocap'}"
            f"{'_ucap' + str(c['ucap']) if c['ucap'] else ''}_{c['n']}")


def run_flood(core: str, quick: bool = False) -> None:
    """Commit time of one flood epoch (bench/flood.py) for every configuration of _flood_cfgs; one JSON per config."""
    import copy as _copy
    from ctsample.bench import flood as FB
    from ctsample.protocol.epoch import Gateway
    from ctsample.protocol.flood import K_DEV_DEFAULT, K_TOT_DEFAULT
    from ctsample.protocol.store import MetadataStore
    from ctsample.units import features as F
    cfg, cfg5 = cfgs()
    d = OUT_REV1 / ("flood_quick" if quick else "flood") / core
    d.mkdir(parents=True, exist_ok=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("e5", P1_ROOT / "scripts" / "14_e5.py")
    e5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e5)
    ctx = e5.TrialCtx(cfg5, E5RUN_REV1)
    dep = ctx.dep
    es = FLOOD_EPOCH
    W = 120
    raw = ctx.records(range(es - W + 1, es + 1))
    base = ctx.view([es]).records[es].batch
    hist_cols = {k: F.cols_from_units(MetadataStore.deserialize(raw[k]["store"]).to_batch(codec=ctx.codec).units)
                 for k in range(es - W + 1, es)}
    dev = int(ctx.target)
    src = FB.device_src_ip(base, dev)
    t0 = base.t_start_us
    m0 = O.machine_state()
    for cf in _flood_cfgs(quick):
        kind, hist, capped, n, reps, impl, ucap = (cf["kind"], cf["history"], cf["cap"], cf["n"], cf["reps"],
                                                    cf["impl"], cf["ucap"])
        pth = d / f"{_cfg_name(cf)}.json"
        if pth.exists():
            continue
        rows = []
        for rep in range(reps):
            rng = np.random.default_rng([1000 + rep, n, kind == "zero", hist == "sustained", capped])
            pol = O.fresh_policy(ctx.pristine)
            pol.rolling_impl = impl
            pol._rs_cls = F.IncrementalRollingStats if impl == "incremental" else F.RollingDeviceStats
            pol.utility_cap = ucap
            tp = O.TimedPolicy(pol)
            gw = Gateway(dep.keys, dep.clock, dep.beacon, tp, dep.witness(), dep.escrow, keep_frames=False)
            O.register(gw, dep)
            # ---- rolling history (untimed): the real 119 epochs, plus the flood in every one when sustained
            rs = pol._rs_cls(W)
            t_h = time.perf_counter()
            for k in range(es - W + 1, es):
                c = hist_cols[k]
                if hist == "sustained" and n:
                    tk = dep.clock.t_e_us(k)
                    if capped:
                        c = FB.capped_history_cols(c, n, dev, src, tk, k, rng, K_DEV_DEFAULT, K_TOT_DEFAULT)
                    else:
                        c = F.concat_cols([c, FB.scan_cols(FB.scan_records(n, dev, src, tk, rng))])
                rs._push(k, c)
                del c
            t_hist = time.perf_counter() - t_h
            win_rows = rs.n_rows if hasattr(rs, "n_rows") else int(sum(len(w[1]) for w in rs.win))
            pol._e0 = 0
            pol._hist = {k: None for k in range(es - W + 1, es)}    # only the keys are needed once rs is primed
            pol._rs = rs
            pol._cache = {}
            gw.state = _copy.deepcopy(raw[es]["state"])
            gw.prev_hash = raw[es - 1]["header"].digest()
            gw.next_e = es
            # ---- the flood epoch
            if kind == "zero":
                flood = FB.scan_units(FB.scan_records(n, dev, src, t0, rng), es) if n else []
            else:
                flood = FB.payload_flood_units(n, dev, src, t0, rng, es)
            batch = FB.flood_batch(base, flood)
            del flood
            n_before = batch.n
            t_cap, cap_info = 0.0, None
            if capped:
                (batch, cap_info), t_cap = FB.cap_batch(batch, K_DEV_DEFAULT, K_TOT_DEFAULT)
            a = time.perf_counter()
            rec = gw.commit(batch)
            t_commit = time.perf_counter() - a
            t_thr = tp.t_thresholds[-1] if tp.t_thresholds else float("nan")
            rows.append({"rep": rep, "n_units_before_cap": int(n_before), "n_units": int(batch.n),
                         "n_payload": int(batch.n_pay), "commit_s": t_commit, "thresholds_s": t_thr,
                         "rest_s": t_commit - t_thr, "cap_s": t_cap, "commit_plus_cap_s": t_commit + t_cap,
                         "history_build_s": t_hist, "rolling_window_rows": int(win_rows),
                         "store_bytes": len(rec.store.serialize()), "cap": cap_info,
                         "header_ok": rec.header.H_store == rec.store.digest()})
            del batch, rec, gw, pol, tp, rs
            import gc
            gc.collect()
        res = {"core": core, **cf, "capped": capped, "n_flood": n, "reps_rows": rows,
               "commit_s_median": float(np.median([r["commit_s"] for r in rows])),
               "commit_plus_cap_s_median": float(np.median([r["commit_plus_cap_s"] for r in rows])),
               "thresholds_s_median": float(np.median([r["thresholds_s"] for r in rows])),
               "n_units": rows[0]["n_units"], "n_payload": rows[0]["n_payload"], "machine_before": m0,
               "loadavg_now": list(os.getloadavg())}
        atomic_write_json(pth, res)
        log(f"flood {core} {_cfg_name(cf)}: commit {res['commit_s_median']:.3f} s (τ {res['thresholds_s_median']:.3f} s), "
            f"units {res['n_units']}, window rows {rows[0]['rolling_window_rows']}")


def _max_supported(points, budget):
    """Largest flood size n whose median commit time (+ cap) ≤ budget, by log-log interpolation between measured
    points (None if even the smallest exceeds it; '>= max' if the largest still fits)."""
    pts = sorted((n, t) for n, t in points if n > 0)
    if not pts:
        return None
    if pts[0][1] > budget:
        return {"n": None, "note": f"n = {pts[0][0]} already exceeds {budget:g} s"}
    for (n1, t1), (n2, t2) in zip(pts, pts[1:]):
        if t2 > budget >= t1:
            if t2 <= t1:
                return {"n": n1}
            f = (np.log(budget) - np.log(t1)) / (np.log(t2) - np.log(t1))
            return {"n": float(np.exp(np.log(n1) + f * (np.log(n2) - np.log(n1)))), "between": [n1, n2]}
    return {"n": pts[-1][0], "note": "largest measured flood still fits (lower bound)"}


def flood_summary(quick: bool = False) -> dict:
    cfg, _ = cfgs()
    res = {"definition": "Commit time of one gateway epoch = Gateway.commit (unit leaves, Merkle root, CtS "
                         "thresholds = features + 120-epoch rolling statistics + utility, metadata store with the "
                         "committed τ column and H(store), C^st, header, Ed25519 signature) of a real epoch of the "
                         "E5 rev1 run (lp_iot23_34-1_p10000, epoch 400, 238 units, 163 payload) plus N flood units of "
                         "ONE compromised device; median over reps; single core (P = performance, E = efficiency "
                         "core via taskpolicy -b), Python prototype, OMP 1 thread. kind zero = random-port SYN scan "
                         "(one unit per packet: N/30 SYN per second); payload = one-packet TCP units with 16 B payload. "
                         "history single = the real 119 preceding epochs; sustained = the same flood in every "
                         "preceding epoch (the rolling statistics span 120 flooded epochs). impl resort = the "
                         "original RollingDeviceStats (re-sorts the whole window every epoch: the code as reviewed); "
                         "incremental = the rev1 bit-exact IncrementalRollingStats (O(n_e + n_{e-W})). cap = "
                         "zero-payload cap of protocol/flood.py (k_dev = 8,192 per device, k_tot = 16,384 per epoch; "
                         "never binds on any training or test stream), its own cost included in "
                         "commit_plus_cap; ucap = the policy's utility cap (4,096 scored payload units per device "
                         "and epoch; never binds on our data).",
           "budgets_s": {"one_round_ahead": BUDGET_ONE_AHEAD_S, "two_rounds_ahead": BUDGET_TWO_AHEAD_S,
                         "two_rounds_ahead_sustained": 30.0},
           "notes": ["two rounds ahead (designated round r0+e+2): the build budget grows by one drand period, but "
                     "a sustained flood also needs commit ≤ T = 30 s per epoch on average (one commit per epoch on the "
                     "same core); content then waits ≤ 2T + Δ_c + Δ_sel ≈ 63.5 s (E6 buffer ahead2) and the adaptive "
                     "design of epoch e can only use selections up to e − 2",
                     "supported n = largest flood (units per epoch from one device) whose median commit (+ cap) time "
                     "fits the budget, log-log interpolation between measured points; None = even the smallest "
                     "measured flood exceeds it"],
           "cores": {}}
    sub = "flood_quick" if quick else "flood"
    for core in cfg["cores"]:
        d = OUT_REV1 / sub / core
        if not d.exists():
            continue
        rows = [json.loads(q.read_text()) for q in sorted(d.glob("*.json"))]
        if not rows:
            continue

        def grp(r):
            return (r["kind"], r["history"], r["impl"], bool(r["cap"]), r.get("ucap"))

        blk = {"rows": [], "loadavg_1min": rows[0]["machine_before"]["loadavg"][0]}
        for r in sorted(rows, key=lambda x: (grp(x)[0], grp(x)[1], grp(x)[2], grp(x)[3], grp(x)[4] or 0, x["n_flood"])):
            r0 = r["reps_rows"][0]
            blk["rows"].append({"kind": r["kind"], "history": r["history"], "impl": r["impl"], "cap": bool(r["cap"]),
                                "ucap": r.get("ucap"), "n_flood": r["n_flood"], "n_units": r["n_units"],
                                "n_payload": r["n_payload"], "commit_s": r["commit_s_median"],
                                "commit_plus_cap_s": r["commit_plus_cap_s_median"],
                                "thresholds_s": r["thresholds_s_median"], "cap_s": r0["cap_s"],
                                "store_MB": r0["store_bytes"] / 1e6, "rolling_window_rows": r0["rolling_window_rows"],
                                "cap_aggregated": (r0["cap"] or {}).get("aggregated"), "reps": len(r["reps_rows"])})
        sup = {}
        for g in sorted({grp(r) for r in rows}, key=lambda g: (g[0], g[1], g[2], g[3], g[4] or 0)):
            pts = [(r["n_flood"], r["commit_plus_cap_s_median"]) for r in rows if grp(r) == g]
            # below the cap level (and for small n generally) the capped and uncapped pipelines are identical
            if g[3] or g[4]:
                pts += [(r["n_flood"], r["commit_plus_cap_s_median"]) for r in rows
                        if grp(r) == (g[0], g[1], g[2], False, None) and 0 < r["n_flood"] < 10000]
            key = f"{g[0]}/{g[1]}/{g[2]}/{'cap' if g[3] else 'nocap'}" + (f"/ucap{g[4]}" if g[4] else "")
            one = _max_supported(pts, BUDGET_ONE_AHEAD_S)
            two = _max_supported(pts, 30.0)
            sup[key] = {"one_round_ahead": one, "two_rounds_ahead": two,
                        "max_measured": max(pts) if pts else None,
                        "commit_s_at_max_measured": max(pts)[1] if pts else None}
            for m, v in (("one_round_ahead", one), ("two_rounds_ahead", two)):
                if v and v.get("n"):
                    sup[key][f"{m}_flood_units_per_s"] = v["n"] / 30.0
        blk["supported"] = sup
        res["cores"][core] = blk
    # per-packet pipeline rate: the rev1 measurement of a core when present, else the rev0 one (labelled)
    pipe = {}
    for src_name, tag in (("E6_overheads.json", "rev0"), ("E6_overheads_rev1.json", "rev1")):
        if (SUMMARY / src_name).exists():
            for c, v in json.loads((SUMMARY / src_name).read_text()).get("pipeline", {}).items():
                if isinstance(v, dict) and v.get("pps_per_core"):
                    pipe[c] = dict(v, _source=f"{tag} ({src_name})")
    res["per_packet_limit"] = {c: {"pps_per_core": v.get("pps_per_core"),
                                   "cores_needed_at_33k_pps": 33000 / v["pps_per_core"] if v.get("pps_per_core") else None,
                                   "source": v["_source"],
                                   "note": "per-packet pipeline (reader, parser, flow table, gateway, cached H(θ)): "
                                           "it bounds the sustainable packet rate of any flood independently of the "
                                           "commit window (a 10^6-unit/epoch SYN scan is ≈ 33 k packets/s)"}
                               for c, v in sorted(pipe.items())}
    SUMMARY.mkdir(parents=True, exist_ok=True)
    name = "E6_flood_quick.json" if quick else "E6_flood.json"
    atomic_write_json(SUMMARY / name, res)
    log(f"E6 flood summary written ({name})")
    return res


# ============================================================================================== main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["pipeline", "pipeline-run", "epochs", "epochs-run", "primitives", "verifier",
                                      "verifier-run", "buffer", "storage", "summary", "flood", "flood-run",
                                      "flood-summary"])
    ap.add_argument("--quick", action="store_true", help="flood: small grid (smoke test)")
    ap.add_argument("--core", default="P")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--cached", action="store_true", help="memoise the constant (H(θ), H(policy))")
    ap.add_argument("--rev", default="rev0", choices=["rev0", "rev1"])
    args = ap.parse_args(argv)
    set_rev(args.rev)
    cfg, _ = cfgs()
    if args.stage == "pipeline":
        sub = "pipeline_cached" if args.cached else "pipeline"
        for rep in range(int(cfg["reps"])):
            for core in cfg["cores"]:
                if not (OUT / sub / f"{core}_r{rep}.json").exists():
                    child("pipeline-run", core, ["--rep", str(rep)] + (["--cached"] if args.cached else []))
    elif args.stage == "pipeline-run":
        run_pipeline(args.core, args.rep, args.cached)
    elif args.stage == "epochs":
        for core in cfg["cores"]:
            child("epochs-run", core, [])
    elif args.stage == "epochs-run":
        run_epochs(args.core)
    elif args.stage == "primitives":
        for core in cfg["cores"]:
            outp = OUT / "primitives" / f"{core}.json"
            if outp.exists():
                continue
            outp.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, OMP_NUM_THREADS="1")
            cmd = [sys.executable, "-I", "-m", "ctsample.bench.primitives", "--out", str(outp)]
            if core == "E":
                cmd = ["taskpolicy", "-b", *cmd]
            m0 = O.machine_state()
            subprocess.call(cmd, env=env, cwd=str(P1_ROOT))
            d = json.loads(outp.read_text())
            d["machine_before"] = m0
            atomic_write_json(outp, d)
    elif args.stage == "verifier":
        for core in cfg["cores"]:
            child("verifier-run", core, ["--cached"] if args.cached else [])
    elif args.stage == "verifier-run":
        run_verifier(args.core, args.cached)
    elif args.stage == "buffer":
        run_buffer()
    elif args.stage == "storage":
        run_storage()
    elif args.stage == "summary":
        summary()
    elif args.stage == "flood":
        for core in cfg["cores"]:
            child("flood-run", core, ["--quick"] if args.quick else [])
        flood_summary(args.quick)
    elif args.stage == "flood-run":
        run_flood(args.core, args.quick)
    elif args.stage == "flood-summary":
        flood_summary(args.quick)
    return 0


def _med(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(v)) if v else None


def _tree_evals_per_payload_unit(cfg5) -> dict:
    """Integer tree evaluations per payload unit of the CtS-AI utility on the E5 stream: Σ over the active
    estimand models of (#trees) × [unit eligible for the estimand]."""
    import importlib.util
    from ctsample.design import utility as U
    spec = importlib.util.spec_from_file_location("e5", P1_ROOT / "scripts" / "14_e5.py")
    e5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e5)
    dep, meta = e5.deployment(cfg5)
    pol, _ = e5.load_policy(cfg5, meta)
    cp = dict(np.load(RESULTS / "streams" / cfg5["stream"] / "cols_pay.npz"))
    el = U.eligibility_matrix(pol.active, cp)
    trees = np.array([len(pol.bundle.hit[q].trees) if hasattr(pol.bundle.hit[q], "trees") else 0
                      for q in pol.active])                  # constant models evaluate no tree
    per_unit = el.astype(np.int64) @ trees
    return {"active_estimands": len(pol.active), "trees_per_model": trees.tolist(),
            "mean_tree_evals_per_payload_unit": float(per_unit.mean()),
            "max_tree_evals_per_payload_unit": int(per_unit.max()),
            "mean_eligible_estimands": float(el.sum(1).mean())}


def summary() -> None:
    cfg, cfg5 = cfgs()
    res = {"config": cfg}
    # ---- pipeline (as implemented, and with the constant (H(θ), H(policy)) memoised)
    res["pipeline_variants"] = {}
    for variant, sub in (("as_implemented", "pipeline"), ("cached_registration", "pipeline_cached")):
        res["pipeline_variants"][variant] = _pipeline_rows(cfg, sub)
    pipe = res["pipeline_variants"]["cached_registration"] or res["pipeline_variants"]["as_implemented"]
    res["pipeline"] = pipe
    res["pipeline_headline_variant"] = "cached_registration" if res["pipeline_variants"]["cached_registration"] \
        else "as_implemented"
    _rest_of_summary(cfg, cfg5, res, pipe)


def _pipeline_rows(cfg, sub) -> dict:
    pipe = {}
    for core in cfg["cores"]:
        rows = [json.loads(p.read_text()) for p in sorted((OUT / sub).glob(f"{core}_r*.json"))]
        if not rows:
            continue
        upp = {k: float(np.median([r["us_per_packet"][k] for r in rows])) for k in rows[0]["us_per_packet"]}
        pipe[core] = {"runs": len(rows), "us_per_packet": upp, "pps_per_core": 1e6 / upp["total"],
                      "us_per_packet_runs": [r["us_per_packet"]["total"] for r in rows],
                      "commit_latency_ms_median": _med([r["commit_latency_ms"] for r in rows], "median"),
                      "commit_latency_ms_p99": _med([r["commit_latency_ms"] for r in rows], "p99"),
                      "per_epoch_ms": {k: float(np.median([r["per_epoch_ms"][k] for r in rows]))
                                       for k in rows[0]["per_epoch_ms"]},
                      "policy_thresholds_ms_median": _med(rows, "policy_thresholds_ms_median"),
                      "packets": rows[0]["packets"], "epochs": rows[0]["n_epochs"], "units": rows[0]["units"],
                      "payload_units": rows[0]["payload_units"],
                      "loadavg_1min": [r["machine_before"]["loadavg"][0] for r in rows]}
    return pipe


def _rest_of_summary(cfg, cfg5, res, pipe) -> None:
    # ---- per-epoch scaling
    res["epochs"] = {}
    for core in cfg["cores"]:
        p = OUT / "epochs" / f"{core}.json"
        if p.exists():
            d = json.loads(p.read_text())
            res["epochs"][core] = {"rows": d["rows"], "loadavg_1min": d["machine_before"]["loadavg"][0]}
    # ---- primitives (P1-A bench; protocol-independent: rev1 reuses the rev0 measurements unless re-run)
    res["primitives"] = {}
    for core in cfg["cores"]:
        p = OUT / "primitives" / f"{core}.json"
        if not p.exists():
            p = OUT_REV0 / "primitives" / f"{core}.json"
        if p.exists():
            d = json.loads(p.read_text())
            res["primitives"][core] = {k: v for k, v in d.items() if k != "meta"}
    # ---- verifier
    res["verifier"] = {}
    for core in cfg["cores"]:
        for sub, tag in (("verifier", ""), ("verifier_cached", "_cached")):
            p = OUT / sub / f"{core}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            rows = d["rows"]
            res["verifier"][core + tag] = {
                "median_s_per_5_epoch_window_fast_no_prime": d.get("median_verify_fast_s_5_epochs"),
                "windows": len(rows), "median_s_per_5_epoch_window": d["median_verify_s_5_epochs"],
                "median_s_per_epoch": d["median_verify_s_5_epochs"] / 5.0,
                "median_prime_check_s": d["median_prime_check_s"],
                "median_disclosure_bytes": float(np.median([r["disclosure_bytes"] for r in rows])),
                "median_prime_bytes": float(np.median([r["prime_bytes"] for r in rows])),
                "loadavg_1min": d["machine_before"]["loadavg"][0]}
    # ---- buffer
    buf = {}
    bdir = OUT / "buffer" if (OUT / "buffer").exists() else OUT_REV0 / "buffer"        # traffic-only: shared
    for p in sorted(bdir.glob("*.json")):
        d = json.loads(p.read_text())
        buf[d["stream"]] = {"pps": d["pps"], "payload_rate_Bps": d["ahead1"]["mean_rate_Bps"],
                            **{f"ahead{a}_{k}_MB": d[f"ahead{a}"][f"{k}_bytes"] / 1e6 for a in (1, 2)
                               for k in ("p50", "p99", "max")},
                            "residence_s": [d["ahead1"]["residence_max_s"], d["ahead2"]["residence_max_s"]]}
    res["buffer"] = buf
    if buf:
        res["buffer_range"] = {k: [min(v[k] for v in buf.values()), max(v[k] for v in buf.values())]
                               for k in next(iter(buf.values())) if k.endswith("_MB") or k == "pps"}
    # ---- storage (streams: protocol-independent part shared with rev0; rev1 adds the committed τ column)
    sp = OUT / "storage.json" if (OUT / "storage.json").exists() else OUT_REV0 / "storage.json"
    if sp.exists():
        res["storage"] = json.loads(sp.read_text())
    if REV == "rev1":
        st = json.loads((E5RUN / "run.json").read_text())["storage"]
        res["tau_column"] = {
            "definition": "rev1 committed thresholds: 8 B (u64 tau - 1) per payload unit in the metadata store; zstd sizes "
                          "of the E5 rev1 run's design columns (meta, root_pkt) without and with the tau column",
            "parquet_zstd_design_bytes": st.get("parquet_zstd_design_bytes"),
            "parquet_zstd_design_tau_bytes": st.get("parquet_zstd_design_tau_bytes"),
            "increase": (st["parquet_zstd_design_tau_bytes"] / st["parquet_zstd_design_bytes"] - 1)
            if st.get("parquet_zstd_design_tau_bytes") else None,
            "over_pcap": (st["parquet_zstd_design_tau_bytes"] - st["parquet_zstd_design_bytes"]) / st["sum_s_all"]
            if st.get("parquet_zstd_design_tau_bytes") else None}
    # ---- witness load (by construction + measured sizes)
    from ctsample.protocol.header import HDR_LEN
    per_day = 86400 // int(cfg["timing"]["T"])
    up = HDR_LEN + 64
    down = 64 + 8
    prim = res["primitives"].get("P", {})
    res["witness"] = {"cosignatures_per_day": per_day, "bytes_up_per_epoch": up, "bytes_down_per_epoch": down,
                      "wire_MB_per_day": per_day * (up + down) / 1e6,
                      "log_MB_per_day_hash_only": per_day * (32 + 72 + 16 + 4) / 1e6,
                      "log_MB_per_day_full_headers": per_day * (up + down) / 1e6}
    # ---- anchoring cost model (transactions / day; no currency)
    e5 = json.loads((E5RUN / "run.json").read_text())
    hours = e5["n_epochs"] * 30 / 3600.0
    res["anchoring"] = {"per_record_retained_tx_per_day": e5["retention"]["retained_units"] * 24 / hours,
                        "per_committed_unit_tx_per_day": e5["gateway_pass"]["units"] * 24 / hours,
                        "cts_witness_cosignatures_per_day": per_day,
                        "cts_anchor_every_K": {str(K): {"tx_per_day": per_day / K, "bytes_per_day": 32 * per_day / K}
                                               for K in cfg["anchoring_K"]},
                        "note": "Chain4ensic-style per-record anchoring vs CtS's per-epoch cosignature; an optional "
                                "ledger anchor of H(Hdr_e) every K epochs (security does not depend on it: TR does)"}
    # ---- primitive counts and the ARM extrapolation
    z = np.load(E5RUN / "units.npz")
    pay = z["pay"].astype(bool)
    pk = z["pkts"].astype(np.int64)
    n_pk = int(pk.sum())
    pay_pk = int(pk[pay].sum())
    counts = {"packets": n_pk, "payload_packets": pay_pk, "units": int(len(pay)), "payload_units": int(pay.sum()),
              "retained_units": int(z["Z"].sum()),
              # BLAKE3 per payload-unit packet: frame hash + ℓ_ij + packet-tree leaf node + ≈ 1 inner node = 4
              # (protocol); the prototype's prepare_unit re-derives root^pkt once more (+2, a check, not required)
              "blake3_per_payload_packet_protocol": 4.0, "blake3_per_payload_packet_implementation": 6.0,
              "blake3_per_zero_payload_packet": 0.0,
              "blake3_per_packet_avg_protocol": (4.0 * pay_pk) / n_pk,
              "per_payload_unit": {"unit_leaf": 1, "epoch_merkle_blake3": 2, "prf_word_blake3": 1, "isqrt": 1,
                                   "threshold_unit": 1},
              "per_zero_payload_unit": {"leaf0": 1, "epoch_merkle_blake3": 2},
              "per_flow": {"flow_uid_blake3": 1},
              "per_epoch": {"merkle_build": "O(n_e)", "store_hash": 1, "header_hash": 1, "state_commitment_blake3": 2,
                            "ed25519_sign": 1, "ecvrf_prove": 1, "witness_ed25519_verify_sign": 2},
              "aead_per_retained_unit": 1}
    try:
        counts.update(_tree_evals_per_payload_unit(cfg5))
    except Exception as ex:                                  # noqa: BLE001
        counts["tree_evals_error"] = repr(ex)
    res["primitive_counts"] = counts
    if "E" in pipe:
        e_us = pipe["E"]["us_per_packet"]["total"]
        res["arm_extrapolation"] = {
            "LABEL": "EXTRAPOLATION (not measured): E-core µs/packet × assumed slowdown factor of a gateway-class ARM core",
            "ecore_us_per_packet": e_us,
            "rows": [{"slowdown": f, "us_per_packet": e_us * f, "pps_per_core": 1e6 / (e_us * f),
                      "cpu_share_at_pps": {str(r): r * e_us * f / 1e6 for r in cfg["iot_rates_pps"]}}
                     for f in cfg["arm_slowdown_vs_ecore"]]}
    if "P" in pipe:
        res["headroom_P"] = {str(r): pipe["P"]["pps_per_core"] / r for r in cfg["iot_rates_pps"]}
    if "E" in pipe:
        res["headroom_E"] = {str(r): pipe["E"]["pps_per_core"] / r for r in cfg["iot_rates_pps"]}
    res["proof_sizes"] = {"inclusion_bytes": "32 ceil(log2 n_e)", "vrf_proof_bytes": 80, "header_bytes": HDR_LEN,
                          "sigma_bytes": 64, "omega_bytes": 72}
    SUMMARY.mkdir(parents=True, exist_ok=True)
    name = "E6_overheads.json" if REV == "rev0" else f"E6_overheads_{REV}.json"
    atomic_write_json(SUMMARY / name, res)
    log(f"E6 summary written ({name})")


if __name__ == "__main__":
    sys.exit(main())
