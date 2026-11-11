"""S1: unitise pcaps, run detectors D1–D13, write replay pcaps (DESIGN §5.3).

One worker process per pcap source (largest first). Per source ``<src>`` of dataset ``<ds>``
(all under P1_CommitThenSample/results/):

  units/<ds>/<src>/part-<seq>.parquet   unit rows (schema in notes/P1-B.md); a part closes at
                                        every 100-epoch boundary or after 250k rows
  macs/<ds>/<src>.parquet               device_id -> LAN-side MAC (the committed MAC table)
  units/<ds>/<src>/DONE                 completion marker + summary (written last)
  pktmap/<ds>/<src>.npz                 per pcap record (index = pcap_cnt-1): unit_gid, pkt_idx,
                                        ts_us, wirelen, smac_idx, dmac_idx; macs (str table)
  hits/<ds>/<src>.parquet               unit_gid, query_id, value, n_hit_pkts, first_hit_pkt
  counters/<ds>/<src>.parquet           per (MAC, epoch): pkts/bytes tx and rx (wire bytes)
  counters/<ds>/<src>.ts.npz            per (payload-relevant device MAC, epoch) CSR of packet
                                        time offsets (µs from epoch start) + tx flag
  replay/<ds>/<src>.pcap, .sub10.pcap   QinQ-tagged frames of payload units (10 % subset)
  qa/<ds>/<src>.json, qa/<ds>.json      per-source and aggregated QA incl. the §0.1 statistics

Resumable at source granularity: a source is skipped iff units/<ds>/<src>/DONE exists
(``--force`` redoes it). Run under memguard from the project root::

    .venv/bin/python memguard/memguard.py run --name p1_s1_iot23 --workers 16 -- \
        .venv/bin/python -I -u P1_CommitThenSample/scripts/01_extract_units.py --dataset iot23
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from array import array
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ctsample.content.detectors import estimand_hits
from ctsample.content.replay import MAX_UNITS, ReplayWriter
from ctsample.io.pcap_reader import PcapReader, parser_for
from ctsample.units import _hash_shim as hs
from ctsample.units.encode import PAY_LEN, ZERO_LEN
from ctsample.units.flowtable import T_US, FlowSegmenter, ip_text, mac_text
from ctsample.utils.io import (RESULTS, atomic_save_npz, atomic_write_json, atomic_write_parquet, is_done,
                               mark_done)
from ctsample.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory, workers

PART_EPOCHS = 100
PART_ROWS = 250_000
RHO, GAMMA, PHIT = 0.05, 0.3, 0.1

UNIT_SCHEMA = pa.schema([
    ("unit_gid", pa.int64()), ("source", pa.string()), ("src_idx", pa.int32()), ("epoch", pa.int32()),
    ("index", pa.int32()), ("rid", pa.int32()), ("flow_uid", pa.binary(16)),
    ("proto", pa.int16()), ("ip_a", pa.string()), ("port_a", pa.int32()), ("ip_b", pa.string()),
    ("port_b", pa.int32()), ("mac_a", pa.string()), ("mac_b", pa.string()),
    ("device_id", pa.int32()), ("device_mac", pa.string()), ("lan_side", pa.int8()), ("dir_class", pa.int8()),
    ("first_ts_us", pa.int64()), ("last_ts_us", pa.int64()), ("flow_first_ts_us", pa.int64()),
    ("pkts0", pa.int32()), ("pkts1", pa.int32()), ("payload0", pa.int64()), ("payload1", pa.int64()),
    ("frame0", pa.int64()), ("frame1", pa.int64()),
    ("n_syn", pa.int32()), ("n_synack", pa.int32()), ("n_fin", pa.int32()), ("n_rst", pa.int32()),
    ("n_psh", pa.int32()), ("conn_state", pa.int8()),
    *[(f"size{j}", pa.int16()) for j in range(8)], *[(f"iat{j}", pa.uint8()) for j in range(8)],
    ("s", pa.int64()), ("sH", pa.int64()), ("pay", pa.bool_()), ("fps", pa.bool_()),
    ("continuation", pa.bool_()), ("meta", pa.binary()), ("root_pkt", pa.binary(32)), ("leaf", pa.binary(32)),
])
HIT_SCHEMA = pa.schema([("unit_gid", pa.int64()), ("query_id", pa.string()), ("value", pa.int32()),
                        ("n_hit_pkts", pa.int32()), ("first_hit_pkt", pa.int32())])


def out_paths(ds: str, src: str) -> dict:
    return {
        "units": RESULTS / "units" / ds / src,
        "macs": RESULTS / "macs" / ds / f"{src}.parquet",
        "pktmap": RESULTS / "pktmap" / ds / f"{src}.npz",
        "hits": RESULTS / "hits" / ds / f"{src}.parquet",
        "counters": RESULTS / "counters" / ds / f"{src}.parquet",
        "counters_ts": RESULTS / "counters" / ds / f"{src}.ts.npz",
        "replay": RESULTS / "replay" / ds / f"{src}.pcap",
        "replay_sub": RESULTS / "replay" / ds / f"{src}.sub10.pcap",
        "qa": RESULTS / "qa" / ds / f"{src}.json",
    }


class Sink:
    """Collects the segmenter's epochs into Parquet parts, hit rows, replay frames and stats."""

    def __init__(self, ds: str, src: dict, paths: dict, linktype: int):
        self.ds = ds
        self.src = src["source"]
        self.src_idx = int(src["src_idx"])
        self.base = self.src_idx << 32
        self.paths = paths
        self.batches: list = []
        self.part_k = None
        self.part_rows = 0
        self.part_seq = 0
        self.part_info: list = []
        self.hits = {"unit_gid": [], "query_id": [], "value": [], "n_hit_pkts": [], "first_hit_pkt": []}
        self.replay = ReplayWriter(paths["replay"], paths["replay_sub"], linktype)
        self.local2epoch = array("i")
        self.st = {"n_units": 0, "n_pay": 0, "S_all": 0, "S_pay": 0, "CF_pay": 0.0, "n_fps": 0,
                   "n_cont": 0, "meta_bytes_enc": 0, "n_replay_units": 0, "n_replay_sub_units": 0,
                   "n_replay_frames": 0, "dir_class": [0, 0, 0, 0], "proto": {}, "hit_units": {},
                   "epochs_with_units": 0, "max_units_epoch": 0, "max_pay_epoch": 0, "pay_devices": set()}

    # -------------------------------------------------------------- per epoch
    def on_epoch(self, e: int, t0: int, recs: list) -> None:
        k = e // PART_EPOCHS
        if self.part_k is not None and (k != self.part_k or self.part_rows >= PART_ROWS):
            self.write_part()
        self.part_k = k
        st = self.st
        n = len(recs)
        st["epochs_with_units"] += 1
        st["max_units_epoch"] = max(st["max_units_epoch"], n)
        cols = {f.name: [] for f in UNIT_SCHEMA}
        rep = []
        npay = 0
        base = self.base
        hits = self.hits
        for r in recs:
            gid = base | r.local
            self.local2epoch.append(e)
            c = cols
            c["unit_gid"].append(gid)
            c["epoch"].append(e)
            c["index"].append(r.index)
            c["rid"].append(r.rid)
            c["flow_uid"].append(r.flow_uid)
            c["proto"].append(r.proto)
            c["ip_a"].append(ip_text(r.key[1:17], r.proto))
            c["port_a"].append(r.port_a)
            c["ip_b"].append(ip_text(r.key[19:35], r.proto) if r.proto else "")
            c["port_b"].append(r.port_b)
            c["mac_a"].append(mac_text(r.mac_a))
            c["mac_b"].append(mac_text(r.mac_b))
            c["device_id"].append(r.device_id)
            dm = r.device_mac
            c["device_mac"].append(mac_text(dm) if isinstance(dm, bytes) else "ip:" + ip_text(dm[1], 1))
            c["lan_side"].append(r.lan_side)
            c["dir_class"].append(r.dir_class)
            c["first_ts_us"].append(r.first_ts_us)
            c["last_ts_us"].append(r.last_ts_us)
            c["flow_first_ts_us"].append(r.flow_first_ts_us)
            c["pkts0"].append(r.pkts[0])
            c["pkts1"].append(r.pkts[1])
            c["payload0"].append(r.payload_bytes[0])
            c["payload1"].append(r.payload_bytes[1])
            c["frame0"].append(r.frame_bytes[0])
            c["frame1"].append(r.frame_bytes[1])
            c["n_syn"].append(r.n_syn)
            c["n_synack"].append(r.n_synack)
            c["n_fin"].append(r.n_fin)
            c["n_rst"].append(r.n_rst)
            c["n_psh"].append(r.n_psh)
            c["conn_state"].append(r.conn_state)
            sz = r.sizes
            ia = r.iats
            for j in range(8):
                c[f"size{j}"].append(max(-32767, min(32767, sz[j])) if j < len(sz) else 0)
                c[f"iat{j}"].append(ia[j] if j < len(ia) else 255)
            c["s"].append(r.s)
            c["sH"].append(r.sH)
            c["pay"].append(r.pay)
            c["fps"].append(r.fps)
            c["continuation"].append(r.continuation)
            c["meta"].append(r.meta)
            c["root_pkt"].append(r.root_pkt)
            c["leaf"].append(r.leaf)
            st["S_all"] += r.s
            st["dir_class"][r.dir_class] += 1
            st["proto"][r.proto] = st["proto"].get(r.proto, 0) + 1
            st["n_cont"] += r.continuation
            st["meta_bytes_enc"] += len(r.meta)
            if r.pay:
                npay += 1
                st["S_pay"] += r.s
                st["CF_pay"] += r.sH + PHIT * (r.s - r.sH)
                st["n_fps"] += r.fps
                st["pay_devices"].add(r.device_id)
                insub = hs.sub10_selected(gid)
                st["n_replay_units"] += 1
                st["n_replay_sub_units"] += insub
                if r.rid >= MAX_UNITS:
                    raise RuntimeError(f"replay id {r.rid} exceeds the QinQ capacity")
                rid = r.rid
                for f in r.frames:
                    rep.append((f[0], f[1], f[5], f[4], f[3], f[6], f[7], f[2], rid, insub))
            if r.raw_hits:
                for qid, v, nh, fh in estimand_hits(r.raw_hits, r.fps):
                    hits["unit_gid"].append(gid)
                    hits["query_id"].append(qid)
                    hits["value"].append(v)
                    hits["n_hit_pkts"].append(nh)
                    hits["first_hit_pkt"].append(fh)
                    st["hit_units"][qid] = st["hit_units"].get(qid, 0) + 1
        st["n_units"] += n
        st["n_pay"] += npay
        st["max_pay_epoch"] = max(st["max_pay_epoch"], npay)
        cols["source"] = [self.src] * n
        cols["src_idx"] = [self.src_idx] * n
        self.batches.append(pa.RecordBatch.from_pydict(cols, schema=UNIT_SCHEMA))
        self.part_rows += n
        if rep:
            st["n_replay_frames"] += len(rep)
            self.replay.write_epoch(rep)

    def write_part(self) -> None:
        if not self.batches:
            return
        tab = pa.Table.from_batches(self.batches, schema=UNIT_SCHEMA)
        p = self.paths["units"] / f"part-{self.part_seq:05d}.parquet"
        atomic_write_parquet(p, tab)
        ep = tab.column("epoch")
        self.part_info.append({"part": p.name, "rows": tab.num_rows,
                               "epoch_min": int(pc.min(ep).as_py()),
                               "epoch_max": int(pc.max(ep).as_py())})
        self.part_seq += 1
        self.batches = []
        self.part_rows = 0


def _counters(seg: FlowSegmenter, sink: Sink, paths: dict, src_idx: int) -> dict:
    """pktmap, per-(MAC, epoch) counters and per-device packet time offsets."""
    arr2local = np.frombuffer(seg.arr2local, dtype=np.int64)
    local = arr2local[np.frombuffer(seg.pk_arr, dtype=np.int64)]
    if (local < 0).any():
        raise RuntimeError("packets mapped to unflushed units")
    gid = (np.int64(src_idx) << 32) | local
    pkt_idx = np.frombuffer(seg.pk_idx, dtype=np.int32)
    ts = np.frombuffer(seg.pk_ts, dtype=np.int64)
    wire = np.frombuffer(seg.pk_wire, dtype=np.int32)
    smac = np.frombuffer(seg.pk_smac, dtype=np.int32)
    dmac = np.frombuffer(seg.pk_dmac, dtype=np.int32)
    macs = sorted(seg.mac_all.items(), key=lambda kv: kv[1])
    mac_str = np.array([mac_text(m) for m, _ in macs], dtype="U17")
    atomic_save_npz(paths["pktmap"], unit_gid=gid, pkt_idx=pkt_idx, ts_us=ts, wirelen=wire,
                    smac_idx=smac, dmac_idx=dmac, macs=mac_str)
    epoch = np.frombuffer(sink.local2epoch, dtype=np.int32)[local].astype(np.int64)
    E = int(epoch.max()) + 1 if len(epoch) else 1
    M = max(1, len(macs))
    rows = {}
    for name, mi in (("tx", smac), ("rx", dmac)):
        ok = mi >= 0
        key = mi[ok].astype(np.int64) * E + epoch[ok]
        uk, inv = np.unique(key, return_inverse=True)
        rows[name] = (uk, np.bincount(inv), np.bincount(inv, weights=wire[ok].astype(np.float64)))
    allk = np.union1d(rows["tx"][0], rows["rx"][0])
    out = {"mac_idx": (allk // E).astype(np.int32), "epoch": (allk % E).astype(np.int32)}
    for name in ("tx", "rx"):
        uk, cnt, byt = rows[name]
        pk = np.zeros(len(allk), np.int64)
        by = np.zeros(len(allk), np.int64)
        if len(uk):
            pos = np.clip(np.searchsorted(uk, allk), 0, len(uk) - 1)
            hit = uk[pos] == allk
            pk[hit] = cnt[pos[hit]]
            by[hit] = np.rint(byt[pos[hit]]).astype(np.int64)
        out[f"pkts_{name}"] = pk
        out[f"bytes_{name}"] = by
    dev_of_mac = {m: d for m, d in seg.mac_table.items() if isinstance(m, bytes)}
    mac_dev = np.full(M, -1, np.int32)
    for m, i in macs:
        if m in dev_of_mac:
            mac_dev[i] = dev_of_mac[m]
    tab = pa.table({"source": [sink.src] * len(allk), "mac": mac_str[out["mac_idx"]] if len(macs) else [],
                    "mac_idx": out["mac_idx"], "device_id": mac_dev[out["mac_idx"]] if len(macs) else [],
                    "epoch": out["epoch"], "pkts_tx": out["pkts_tx"], "bytes_tx": out["bytes_tx"],
                    "pkts_rx": out["pkts_rx"], "bytes_rx": out["bytes_rx"]})
    atomic_write_parquet(paths["counters"], tab)
    # per-packet time offsets for payload-relevant devices (devices with >= 1 payload unit)
    rel_dev = sink.st["pay_devices"]
    rel_mac = np.array(sorted(i for m, i in macs if dev_of_mac.get(m, -1) in rel_dev), dtype=np.int32)
    t0 = seg.t0 or 0
    off = (ts - (t0 + epoch * T_US)).astype(np.int64)
    parts_m, parts_e, parts_o, parts_tx = [], [], [], []
    for mi, is_tx in ((smac, 1), (dmac, 0)):
        sel = np.isin(mi, rel_mac)
        parts_m.append(mi[sel])
        parts_e.append(epoch[sel])
        parts_o.append(off[sel])
        parts_tx.append(np.full(int(sel.sum()), is_tx, np.uint8))
    m_all = np.concatenate(parts_m) if parts_m else np.zeros(0, np.int32)
    e_all = np.concatenate(parts_e) if parts_e else np.zeros(0, np.int64)
    o_all = np.concatenate(parts_o) if parts_o else np.zeros(0, np.int64)
    tx_all = np.concatenate(parts_tx) if parts_tx else np.zeros(0, np.uint8)
    order = np.lexsort((o_all, e_all, m_all))
    m_all, e_all, o_all, tx_all = m_all[order], e_all[order], o_all[order], tx_all[order]
    rk = m_all.astype(np.int64) * E + e_all
    urk, start = np.unique(rk, return_index=True)
    ptr = np.append(start, len(rk)).astype(np.int64)
    atomic_save_npz(paths["counters_ts"], row_mac_idx=(urk // E).astype(np.int32),
                    row_device_id=mac_dev[(urk // E)] if len(urk) else np.zeros(0, np.int32),
                    row_epoch=(urk % E).astype(np.int32), row_ptr=ptr,
                    ts_off_us=np.clip(o_all, 0, 2**32 - 1).astype(np.uint32), is_tx=tx_all, macs=mac_str,
                    t0_us=np.int64(t0), T_us=np.int64(T_US))
    return {"n_counter_rows": int(len(allk)), "n_ts_entries": int(len(o_all)),
            "n_payload_relevant_devices": int(len(rel_mac)), "n_macs": len(macs)}


def extract_source(ds: str, src: dict, force: bool = False, max_packets: int | None = None) -> dict:
    """Worker: process one pcap source completely; returns its QA dict."""
    name = src["source"]
    paths = out_paths(ds, name)
    if is_done(paths["units"]) and not force and paths["qa"].exists():
        q = json.loads(paths["qa"].read_text())
        q["skipped"] = True
        return q
    t_start = time.time()
    if paths["units"].exists():
        shutil.rmtree(paths["units"])
    paths["units"].mkdir(parents=True, exist_ok=True)
    rd = PcapReader(src["read_path"], max_packets=max_packets or src.get("truncate_to_packets"))
    parse = parser_for(rd.linktype)
    sink = Sink(ds, src, paths, rd.linktype)
    seg = FlowSegmenter(sink.on_epoch)
    first_ts = last_ts = None
    try:
        for cnt, ts, cl, wl, fr in rd:
            seg.add(cnt, ts, cl, wl, fr, parse(fr, wl))
            if first_ts is None:
                first_ts = ts
            last_ts = ts
        seg.finish()
        sink.write_part()
        sink.replay.close()
    except BaseException:
        sink.replay.abort()
        raise
    t_read = time.time() - t_start
    # hits
    atomic_write_parquet(paths["hits"], pa.table(sink.hits, schema=HIT_SCHEMA))
    cinfo = _counters(seg, sink, paths, int(src["src_idx"]))
    # MAC table
    n_by_dev = {}
    macs = sorted(seg.mac_table.items(), key=lambda kv: kv[1])
    first_ep = {k: e for k, e in seg.mac_first}
    atomic_write_parquet(paths["macs"], pa.table({
        "device_id": pa.array([d for _, d in macs], pa.int32()),
        "device_key": [mac_text(m) if isinstance(m, bytes) else "ip:" + ip_text(m[1], 1) for m, _ in macs],
        "first_epoch": [first_ep.get(m, -1) for m, _ in macs],
        "has_payload_units": [d in sink.st["pay_devices"] for _, d in macs]}))
    st = sink.st
    E = seg.n_epochs
    dur = (last_ts - first_ts) / 1e6 if first_ts is not None and last_ts > first_ts else 0.0
    npk = seg.qa["n_packets"]
    n_zero = st["n_units"] - st["n_pay"]
    phi0 = GAMMA * RHO * st["S_all"] / st["S_pay"] if st["S_pay"] else None
    k95 = math.ceil(math.log(0.05) / math.log(1 - phi0)) if phi0 and phi0 < 1 else None
    qa = {
        "dataset": ds, "source": name, "path": src["path"], "read_path": src["read_path"],
        "src_idx": src["src_idx"], "role": src.get("role"), "class": src.get("class"),
        "reader": rd.qa, "segmenter": seg.qa, "capinfos_packets": src.get("packets"),
        "packets": npk, "packets_match_capinfos": (src.get("packets") == npk),
        "first_ts_us": first_ts, "last_ts_us": last_ts, "t0_us": seg.t0, "duration_s": dur,
        "pps": npk / dur if dur else None, "epochs": E, "epochs_with_units": st["epochs_with_units"],
        "units": st["n_units"], "payload_units": st["n_pay"], "zero_payload_units": n_zero,
        "zero_payload_share": n_zero / st["n_units"] if st["n_units"] else None,
        "payload_unit_share": st["n_pay"] / st["n_units"] if st["n_units"] else None,
        "units_per_epoch_mean": st["n_units"] / E if E else None,
        "payload_units_per_epoch_mean": st["n_pay"] / E if E else None,
        "max_units_epoch": st["max_units_epoch"], "max_payload_units_epoch": st["max_pay_epoch"],
        "non_ip_share": (seg.qa["n_l2"] + seg.qa["n_bad_l2"]) / npk if npk else None,
        "S_all": st["S_all"], "S_pay": st["S_pay"],
        "S_all_over_S_pay": st["S_all"] / st["S_pay"] if st["S_pay"] else None,
        "capped_gain_H1KB": st["S_pay"] / st["CF_pay"] if st["CF_pay"] else None,
        "phi0_rho5_gamma03": phi0, "k95": k95,
        "metadata_over_pcap_s01": (st["n_pay"] * 160 + n_zero * 88) / st["S_all"] if st["S_all"] else None,
        "metadata_over_pcap_encoded": (st["n_pay"] * (PAY_LEN + 32) + n_zero * ZERO_LEN) / st["S_all"]
        if st["S_all"] else None,
        "n_fps": st["n_fps"], "n_continuation": st["n_cont"], "dir_class_counts": st["dir_class"],
        "proto_unit_counts": {str(k): v for k, v in sorted(st["proto"].items())},
        "hit_units": st["hit_units"], "n_devices": len(seg.mac_table),
        "n_payload_devices": len(st["pay_devices"]),
        "replay": {"units": st["n_replay_units"], "sub10_units": st["n_replay_sub_units"],
                   "frames": st["n_replay_frames"], "bytes": sink.replay.main.bytes,
                   "sub10_bytes": sink.replay.sub.bytes if sink.replay.sub else 0},
        "parts": sink.part_info, "counters": cinfo,
        "elapsed_s": time.time() - t_start, "read_elapsed_s": t_read,
        "kpps": npk / t_read / 1e3 if t_read else None, "pid": os.getpid(),
    }
    atomic_write_json(paths["qa"], qa)
    mark_done(paths["units"], {"source": name, "units": st["n_units"], "payload_units": st["n_pay"],
                               "packets": npk, "parts": len(sink.part_info), "t0_us": seg.t0,
                               "T_us": T_US, "src_idx": src["src_idx"]})
    return qa


def _worker(ds, src, force, max_packets):
    try:
        return extract_source(ds, src, force, max_packets)
    except Exception as e:  # noqa: BLE001
        return {"source": src["source"], "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


def aggregate(ds: str, sources: list[dict]) -> dict:
    per = {}
    for s in sources:
        p = out_paths(ds, s["source"])["qa"]
        if p.exists():
            per[s["source"]] = json.loads(p.read_text())
    keys = ["packets", "units", "payload_units", "S_all", "S_pay"]
    tot = {k: sum(q.get(k) or 0 for q in per.values()) for k in keys}
    agg = {"dataset": ds, "n_sources": len(sources), "n_done": len(per),
           "missing": sorted(s["source"] for s in sources if s["source"] not in per),
           "totals": tot,
           "packets_match_capinfos_all": all(q.get("packets_match_capinfos") for q in per.values()),
           "table_s01": {k: {f: q.get(f) for f in ("pps", "payload_units_per_epoch_mean", "zero_payload_share",
                                                   "S_all_over_S_pay", "capped_gain_H1KB", "phi0_rho5_gamma03",
                                                   "k95", "metadata_over_pcap_s01", "metadata_over_pcap_encoded",
                                                   "non_ip_share", "units_per_epoch_mean", "payload_unit_share",
                                                   "kpps", "packets", "capinfos_packets")}
                         for k, q in sorted(per.items())},
           "sources": per}
    prev = RESULTS / "qa" / f"{ds}.json"
    if prev.exists():                         # keep sections added by later stages (S2/S3)
        old = json.loads(prev.read_text())
        for k, v in old.items():
            if k not in agg:
                agg[k] = v
    atomic_write_json(prev, agg)
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["iot23", "ciciot2023_benign", "edgeiiot", "gotham"])
    ap.add_argument("--sources", nargs="*", help="only these source names")
    ap.add_argument("--scenarios", nargs="*", help="only sources of these scenarios (gotham: benign mirai-infection ...)")
    ap.add_argument("--no-floods", action="store_true", help="edgeiiot: skip the DDoS flood captures")
    ap.add_argument("--floods-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-packets", type=int, default=None, help="testing only")
    a = ap.parse_args()
    inv = json.loads((RESULTS / "inventory.json").read_text())
    sources = list(inv["datasets"][a.dataset]["sources"])
    if a.sources:
        sources = [s for s in sources if s["source"] in set(a.sources)]
    if a.scenarios:
        sources = [s for s in sources if s.get("scenario") in set(a.scenarios)]
    if a.no_floods:
        sources = [s for s in sources if not s.get("flood")]
    if a.floods_only:
        sources = [s for s in sources if s.get("flood")]
    sources.sort(key=lambda s: -(s.get("size") or 0))
    nw = min(workers(default=16), max(1, len(sources)))
    print(f"[s1] {a.dataset}: {len(sources)} sources, {nw} workers", flush=True)
    t0 = time.time()
    failed = []
    ex = ProcessPoolExecutor(max_workers=nw, mp_context=get_context("spawn"))
    try:
        futs = {ex.submit(_worker, a.dataset, s, a.force, a.max_packets): s for s in sources}
        pending = set(futs)
        while pending:
            done, pending = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
            for f in done:
                q = f.result()
                if "error" in q:
                    failed.append(q["source"])
                    print(f"[s1] FAILED {q['source']}: {q['error']}\n{q['traceback']}", flush=True)
                elif q.get("skipped"):
                    print(f"[s1] skip {q['source']} (done)", flush=True)
                else:
                    print(f"[s1] done {q['source']}: {q['packets']} pkts {q['units']} units "
                          f"{q['payload_units']} pay  {q['kpps']:.1f} kpps  {q['elapsed_s']:.0f}s "
                          f"(capinfos match: {q['packets_match_capinfos']})", flush=True)
            use = check_memory()
            if done:
                print(f"[s1] {len(pending)} pending, tree memory {use:.1f} GB, {time.time() - t0:.0f}s", flush=True)
    except MemoryBudgetExceeded as e:
        print(f"[s1] memory budget: {e}; stopping workers, exit {EXIT_MEMORY}", flush=True)
        for p in list(getattr(ex, "_processes", {}).values()):
            p.terminate()
        ex.shutdown(wait=False, cancel_futures=True)
        return EXIT_MEMORY
    ex.shutdown(wait=True)
    agg = aggregate(a.dataset, list(inv["datasets"][a.dataset]["sources"]))
    print(f"[s1] {a.dataset}: {agg['n_done']}/{agg['n_sources']} sources done, totals {agg['totals']}, "
          f"failed {failed}, {time.time() - t0:.0f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
