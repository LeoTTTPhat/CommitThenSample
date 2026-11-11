"""End-to-end CtS gateway pipeline on a *virtual merged capture* (E5, E6, E8) and the E6 overhead timers.

Virtual capture (DESIGN §5.2 "E5 stream")
-----------------------------------------
P1-D's stream recipe (``results/streams/<id>/meta.json``: parts with source windows, whole-epoch offsets
``delta_us``, IP/MAC remap tables, thinning) is replayed at PACKET level with P1-B's k-way merge
(``io/merge.py``): every source pcap contributes the records of its window, timestamps shifted by
``delta_us + shift_us``, addresses rewritten on the fly (IPv4/TCP/UDP checksums fixed incrementally), and
records of thinned-out connections dropped (``pktmap`` record → unit_gid). ``shift_us`` is one constant
that re-times the whole virtual capture onto the drand-aligned epoch grid t_e = t(r0 + e) − Δ_c of the
gateway clock (``align_shift``); frames, frame hashes, leaves and headers are those of this re-timed
virtual capture. With ``shift_us = 0`` the flow table reproduces the stream's units (validated by
``compare_with_stream``: same keys and enc(m_i) byte for byte except the device-id field, which the
flow table numbers by first appearance; device MACs agree).

Passes
------
* ``unitise(vc, keep=None, registry=None, track=...)`` runs P1-B's ``FlowSegmenter`` over the merged
  packets. ``keep`` = boolean mask over merged packet indices (gateway capture loss); ``registry`` =
  pre-seeded device table (device MAC → id), so lossy passes number devices like the lossless pass (a
  deployment registers its devices; first-appearance numbering would change when a first packet is lost).
* lossless pass (track=True) → per-packet (ts, caplen, device of its unit) = the counter source I's view;
* gateway pass (keep = loss mask) → EpochBatches → ``Gateway`` (real ECVRF / Ed25519 / BLAKE3, real drand
  rounds, witness simulator, P1-D's CtS policy).

Artefacts of a gateway run (``GatewayRun``; ``results/E5/<run>/``)
-----------------------------------------------------------------
``records/r<k>.pkl``  per-epoch compact records (header, σ, ω, Π_e, β_vrf, store bytes, opening, policy
                      state, τ, u, Z, retained) in chunks of ``CHUNK`` epochs (our own outputs, pickled);
``frames/f<k>.bin``   frames of every payload unit (needed only to rebuild pre-selection content for
                      adversary trials); format in ``write_frames``;
``packets.npz``       per-packet arrays of the lossless pass + the loss mask; ``counters.pkl`` the
                      signed counter records; ``run.json`` summary; ``DONE``.
``RunView`` rebuilds ``EpochRecord`` objects (with frames) for a window of epochs and exposes the
attributes of ``Gateway`` that ``protocol.disclosure.build_disclosure`` and P1-A's adversaries use.
"""
from __future__ import annotations

import copy
import json
import os
import pickle
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from blake3 import blake3

from ..crypto import hashing as hs
from ..crypto.aead import EpochCipher
from ..crypto.signing import KeySet
from ..io.merge import SourceSpec, merged
from ..protocol import beacon as bc
from ..protocol.epoch import EpochRecord, Gateway
from ..protocol.store import EncodeMetaCodec, MetadataStore, RetentionStore
from ..protocol.witness import Witness
from ..units.batch import EpochBatch, Frame, Unit
from ..utils.io import RESULTS, atomic_write_bytes, atomic_write_json

T_US = 30_000_000
CHUNK = 30
STREAMS = RESULTS / "streams"

__all__ = ["Part", "VirtualCapture", "virtual_capture", "single_source_capture", "align_shift", "unitise",
           "UnitiseResult", "Deployment", "make_deployment", "GatewayRun", "RunView", "write_frames",
           "read_frames", "store_cells", "fresh_policy", "machine_state", "Timer"]


# ============================================================================== virtual capture
def _inventory() -> dict:
    inv = json.loads((RESULTS / "inventory.json").read_text())
    out = {}
    for ds, d in inv["datasets"].items():
        for s in d["sources"]:
            out[(ds, s["source"])] = s
    return out


def source_t0_us(ds: str, source: str) -> int:
    return int(json.loads((RESULTS / "units" / ds / source / "DONE").read_text())["t0_us"])


def _ip_bytes(txt: str) -> bytes:
    import ipaddress
    return ipaddress.ip_address(txt).packed


def _mac_bytes(txt: str) -> bytes:
    return bytes(int(x, 16) for x in txt.split(":"))


@dataclass
class Part:
    ds: str
    source: str
    role: str
    path: str
    t0_src_us: int
    src_epoch0: int
    stream_epoch0: int
    n_epochs: int
    delta_us: int
    ip_map: dict = field(default_factory=dict)      # text → text
    mac_map: dict = field(default_factory=dict)     # text → text (incl. the "mac_other" expansions)
    dropped_gids: np.ndarray | None = None          # unit_gids removed by thinning (records filtered)
    seed_mac_map: dict = field(default_factory=dict)  # the recipe's explicit MAC map (seed host / attacker)

    @property
    def window(self) -> tuple[int, int]:
        lo = self.t0_src_us + self.src_epoch0 * T_US
        return lo, lo + self.n_epochs * T_US


@dataclass
class VirtualCapture:
    stream_id: str
    parts: list[Part]
    t0_us: int                   # stream epoch-0 start, virtual (unshifted) time
    n_epochs: int
    shift_us: int = 0

    @property
    def t0_gw_us(self) -> int:
        return self.t0_us + self.shift_us

    def with_shift(self, shift_us: int) -> "VirtualCapture":
        return VirtualCapture(self.stream_id, self.parts, self.t0_us, self.n_epochs, int(shift_us))

    def specs(self) -> list[SourceSpec]:
        out = []
        for p in self.parts:
            lo, hi = p.window
            ipm = {_ip_bytes(o): _ip_bytes(n) for o, n in p.ip_map.items()}
            macm = {_mac_bytes(o): _mac_bytes(n) for o, n in p.mac_map.items()}
            keep = None
            if p.dropped_gids is not None and len(p.dropped_gids):
                pm = np.load(RESULTS / "pktmap" / p.ds / f"{p.source}.npz")
                drop = np.isin(pm["unit_gid"], p.dropped_gids)
                keep = (lambda dm: (lambda cnt: not dm[cnt - 1]))(drop)
            out.append(SourceSpec(path=p.path, offset_us=p.delta_us + self.shift_us, t_start_us=lo, t_end_us=hi,
                                  ip_map=ipm, mac_map=macm, keep=keep, name=p.source))
        return out

    def packets(self):
        """(k, ts_us, caplen, wirelen, frame, part, src_cnt), k = 0-based index in the merged capture."""
        for n, ts, cl, wl, fr, i, cnt in merged(self.specs()):
            yield n - 1, ts, cl, wl, fr, i, cnt

    def foreground_macs(self) -> set[bytes]:
        return {_mac_bytes(n) for p in self.parts if p.role != "background" for n in p.mac_map.values()}

    def seed_macs(self) -> set[bytes]:
        """Remapped MACs of the recipe's seed hosts (the infected device / attacker)."""
        return {_mac_bytes(n) for p in self.parts if p.role != "background" for n in p.seed_mac_map.values()}


def virtual_capture(stream_id: str, shift_us: int = 0) -> VirtualCapture:
    """VirtualCapture from P1-D's stream recipe (results/streams/<id>/meta.json)."""
    sd = STREAMS / stream_id
    m = json.loads((sd / "meta.json").read_text())
    ex = m["extra"]
    E = int(m["n_epochs"])
    inv = _inventory()
    remap = {r["source"]: r for r in ex.get("remap", [])}
    fgs = {fg["source"]: fg for fg in ex["recipe"].get("foregrounds", [])}
    parts = []
    for p in ex["parts"]:
        ds, src = p["ds"], p["source"]
        fg = fgs.get(src)
        n = E - int(p["stream_epoch0"]) if fg is None else min(int(fg["n_epochs"]), E - int(p["stream_epoch0"]))
        rr = remap.get(src, {"ip": {}, "mac": {}})
        ipm = {old: v["new"] for old, v in rr.get("ip", {}).items()}
        dropped = None
        if int(p.get("n_dropped_units", 0)) > 0:
            import polars as pl
            kept = pl.read_parquet(sd / "units.parquet", columns=["source", "unit_gid"]).filter(
                pl.col("source") == src)["unit_gid"].to_numpy()
            lo_e, hi_e = int(p["src_epoch0"]), int(p["src_epoch0"]) + n
            win = pl.read_parquet(RESULTS / "units" / ds / src / "part-*.parquet", columns=["unit_gid", "epoch"]).filter(
                (pl.col("epoch") >= lo_e) & (pl.col("epoch") < hi_e))["unit_gid"].to_numpy()
            dropped = np.setdiff1d(win, kept)
        parts.append(Part(ds=ds, source=src, role=p["role"], path=inv[(ds, src)]["read_path"],
                          t0_src_us=source_t0_us(ds, src), src_epoch0=int(p["src_epoch0"]),
                          stream_epoch0=int(p["stream_epoch0"]), n_epochs=n, delta_us=int(p["delta_us"]),
                          ip_map=ipm, mac_map=dict(rr.get("mac", {})), dropped_gids=dropped,
                          seed_mac_map=dict((fg or {}).get("mac_map", {}))))
    return VirtualCapture(stream_id, parts, int(ex["t0_stream_us"]), E, int(shift_us))


def single_source_capture(ds: str, source: str, n_epochs: int | None = None, shift_us: int = 0) -> VirtualCapture:
    """One source pcap alone (e.g. the CIC training half for the ε_cell calibration)."""
    inv = _inventory()[(ds, source)]
    t0 = source_t0_us(ds, source)
    if n_epochs is None:
        n_epochs = int((int(round(inv["last_ts"] * 1e6)) - t0) // T_US) + 1
    p = Part(ds=ds, source=source, role="background", path=inv["read_path"], t0_src_us=t0, src_epoch0=0,
             stream_epoch0=0, n_epochs=n_epochs, delta_us=0)
    return VirtualCapture(f"{ds}:{source}", [p], t0, n_epochs, shift_us)


def align_shift(clock: bc.EpochClock, vc: VirtualCapture) -> int:
    """Constant shift that puts stream epoch 0 on the gateway clock's epoch 0 (t_0 = t(r0) − Δ_c)."""
    return clock.t_e_us(0) - vc.t0_us


# ============================================================================== unitisation
@dataclass
class UnitiseResult:
    registry: dict
    n_epochs_seen: int
    qa: dict
    pk_k: np.ndarray | None = None          # merged index of every packet given to the flow table
    pk_ts: np.ndarray | None = None
    pk_len: np.ndarray | None = None        # caplen
    pk_part: np.ndarray | None = None
    pk_unit: np.ndarray | None = None       # running unit index (pass-local) of the packet's unit
    unit_epoch: np.ndarray | None = None
    unit_index: np.ndarray | None = None
    unit_dev: np.ndarray | None = None
    unit_pay: np.ndarray | None = None
    unit_part: np.ndarray | None = None
    elapsed_s: float = 0.0
    n_packets: int = 0


def unitise(vc: VirtualCapture, *, keep: np.ndarray | None = None, registry: dict | None = None,
            on_batch=None, keep_frames: bool = True, track: bool = False, max_epoch: int | None = None,
            unit_rows: bool = False) -> UnitiseResult:
    """Run the flow table over the merged packets (optionally a loss mask ``keep`` over merged indices).

    ``on_batch(batch, recs)`` is called at every epoch end with the EpochBatch (frames attached when
    ``keep_frames``) and P1-B's UnitRecords. ``track`` records per-packet arrays and the per-unit
    device/part tables (the counter source's view). Epochs ≥ ``max_epoch`` are not unitised."""
    from ..io.pcap_reader import parser_for
    from ..units.flowtable import FlowSegmenter

    u_epoch, u_index, u_dev, u_pay = [], [], [], []

    def on_epoch(e, t0, recs):
        if track or unit_rows:
            for r in recs:
                u_epoch.append(r.epoch)
                u_index.append(r.index)
                u_dev.append(r.device_id)
                u_pay.append(r.pay)
        if on_batch is not None:
            on_batch(EpochBatch(epoch=e, t_start_us=t0, units=[r.to_unit(keep_frames) for r in recs]), recs)

    seg = FlowSegmenter(on_epoch, keep_frames=keep_frames, run_detectors=False, track_packets=track)
    seg.t0 = vc.t0_gw_us
    seg.cur = 0
    if registry:
        seg.mac_table = dict(registry)
        seg.mac_first = [(k, -1) for k in registry]
    parse = parser_for(1)
    from array import array
    pk_k, pk_part, pk_len = array("q"), array("b"), array("i")
    hi_ts = None if max_epoch is None else vc.t0_gw_us + int(max_epoch) * T_US
    t0 = time.perf_counter()
    n = 0
    for k, ts, cl, wl, fr, i, _cnt in vc.packets():
        if hi_ts is not None and ts >= hi_ts:
            break
        if keep is not None and not keep[k]:
            continue
        seg.add(k + 1, ts, cl, wl, fr, parse(fr, wl))
        n += 1
        if track:
            pk_k.append(k)
            pk_part.append(i)
            pk_len.append(cl)
    seg.finish()
    res = UnitiseResult(registry=dict(seg.mac_table), n_epochs_seen=seg.n_epochs, qa=dict(seg.qa),
                        elapsed_s=time.perf_counter() - t0, n_packets=n)
    if track or unit_rows:
        res.unit_epoch = np.asarray(u_epoch, dtype=np.int32)
        res.unit_index = np.asarray(u_index, dtype=np.int32)
        res.unit_dev = np.asarray(u_dev, dtype=np.int32)
        res.unit_pay = np.asarray(u_pay, dtype=bool)
    if track:
        arr = np.frombuffer(seg.pk_arr, dtype=np.int64) if len(seg.pk_arr) else np.zeros(0, np.int64)
        a2l = np.frombuffer(seg.arr2local, dtype=np.int64)
        res.pk_k = np.frombuffer(pk_k, dtype=np.int64).copy()
        res.pk_part = np.frombuffer(pk_part, dtype=np.int8).copy()
        res.pk_len = np.frombuffer(pk_len, dtype=np.int32).copy()          # caplen (= Σ in frame_bytes)
        res.pk_ts = np.frombuffer(seg.pk_ts, dtype=np.int64).copy()
        res.pk_unit = a2l[arr]
        order = np.argsort(res.pk_unit, kind="stable")
        uu, idx = np.unique(res.pk_unit[order], return_index=True)
        part = np.zeros(len(res.unit_epoch), dtype=np.int8)
        part[uu] = res.pk_part[order][idx]
        res.unit_part = part
    return res


def store_cells(stores: dict[int, MetadataStore]) -> dict[tuple[int, int], tuple[int, int]]:
    """Committed (e, device) → (packets, frame bytes) over the given stores (V7's committed view)."""
    out = {}
    for e, st in stores.items():
        for d, v in st.cells().items():
            out[(e, d)] = v
    return out


# ============================================================================== deployment
@dataclass
class Deployment:
    master: bytes
    keys: KeySet
    beacon: bc.DrandRounds
    clock: bc.EpochClock
    escrow: bytes
    ahead: int = 1

    def witness(self) -> Witness:
        return Witness(self.keys.sk_W)


def make_deployment(master: bytes, n_epochs: int, ahead: int = 1, beacon: bc.DrandRounds | None = None) -> Deployment:
    beacon = beacon or bc.DrandRounds()
    clock = bc.align_stream(beacon, n_epochs, ahead=ahead)
    keys = KeySet.from_master(master)
    escrow = blake3(master, derive_key_context="CTS/v1 E5 escrow master").digest()
    return Deployment(master, keys, beacon, clock, escrow, ahead)


def register(gw: Gateway, dep: Deployment) -> None:
    """Deterministic registration (same messages and signatures in every process)."""
    gw.register(policy_effective_epoch=gw.e0, now=dep.clock.t_e(gw.e0) - 2 * dep.clock.period)


def fresh_policy(policy):
    """A copy of a pristine CtS policy with empty rolling history (shares the read-only model bundle)."""
    p = copy.copy(policy)
    for name, val in (("_hist", {}), ("_cache", {}), ("_rs", None), ("_e0", None)):
        if hasattr(p, name):
            setattr(p, name, copy.copy(val))
    if hasattr(p, "_ctrl"):
        p._ctrl = copy.deepcopy(policy._ctrl)
    return p


# ============================================================================== frames on disk
_FH = struct.Struct("<II")
_FR = struct.Struct("<QBII")


def write_frames(units: list[Unit]) -> bytes:
    """Frames of the payload units of one epoch: n_units × (u32 index, u32 n, n × (u64 ts, u8 dir,
    u32 caplen, u32 wirelen, data))."""
    parts = []
    for u in units:
        if not u.pay or not u.frames:
            continue
        parts.append(_FH.pack(u.index, len(u.frames)))
        for f in u.frames:
            parts.append(_FR.pack(f.ts_us, f.dir, f.caplen, f.wirelen))
            parts.append(f.data)
    return b"".join(parts)


def read_frames(b: bytes) -> dict[int, list[Frame]]:
    out = {}
    off = 0
    mv = memoryview(b)
    while off < len(b):
        i, n = _FH.unpack_from(b, off)
        off += _FH.size
        fr = []
        for _ in range(n):
            ts, d, cl, wl = _FR.unpack_from(b, off)
            off += _FR.size
            fr.append(Frame(ts_us=ts, dir=d, caplen=cl, wirelen=wl, data=bytes(mv[off:off + cl])))
            off += cl
        out[i] = fr
    return out


# ============================================================================== gateway run
def _compact(rec: EpochRecord) -> dict:
    return {"e": rec.e, "header": rec.header, "sigma": rec.sigma, "cosig": rec.cosig, "refusal": rec.refusal,
            "t_send": rec.t_send, "r_star": rec.header.r_star, "vrf_proof": rec.vrf_proof, "beta_vrf": rec.beta_vrf,
            "store": rec.store.serialize(), "opening": rec.opening, "state": rec.state, "taus": list(rec.taus),
            "u": list(rec.u or []), "Z": np.packbits(np.asarray(rec.Z or [], dtype=bool)).tobytes(),
            "n": rec.batch.n, "retained": list(rec.retained)}


class GatewayRun:
    """Drive a Gateway over the batches of a gateway pass and persist the artefacts (atomic, DONE)."""

    def __init__(self, out_dir: Path, dep: Deployment, policy, n_epochs: int, save_frames: bool = True,
                 log=print):
        self.out = Path(out_dir)
        self.dep = dep
        self.n_epochs = int(n_epochs)
        self.save_frames = save_frames
        self.log = log
        self.witness = dep.witness()
        self.gw = Gateway(dep.keys, dep.clock, dep.beacon, policy, self.witness, dep.escrow, keep_frames=False)
        register(self.gw, dep)
        self._buf: dict[int, dict] = {}
        self._fbuf: dict[int, bytes] = {}
        self.timing = {"commit": [], "publish": [], "select": [], "finish": [], "n": [], "n_pay": [], "pkts": []}
        self.retained_s = 0
        self.stores: dict[int, MetadataStore] = {}
        (self.out / "records").mkdir(parents=True, exist_ok=True)
        if save_frames:
            (self.out / "frames").mkdir(parents=True, exist_ok=True)

    def _epoch(self, batch: EpochBatch) -> EpochRecord:
        gw = self.gw
        tm = self.timing
        t0 = time.perf_counter()
        rec = gw.commit(batch)
        t1 = time.perf_counter()
        gw.publish(rec)
        t2 = time.perf_counter()
        gw.select(rec)
        t3 = time.perf_counter()
        gw.finish(rec)
        t4 = time.perf_counter()
        tm["commit"].append(t1 - t0)
        tm["publish"].append(t2 - t1)
        tm["select"].append(t3 - t2)
        tm["finish"].append(t4 - t3)
        tm["n"].append(batch.n)
        tm["n_pay"].append(batch.n_pay)
        tm["pkts"].append(sum(u.n_pkts for u in batch.units))
        self.retained_s += sum(batch.units[i].s for i in rec.retained)
        self._buf[rec.e] = _compact(rec)
        self.stores[rec.e] = rec.store
        if len(self._buf) >= CHUNK:
            self._flush()
        return rec

    def on_batch(self, batch: EpochBatch, recs=None) -> None:
        if batch.epoch >= self.n_epochs:
            return
        while self.gw.next_e < batch.epoch:
            self._emit(self.gw.empty_batch(self.gw.next_e))
        self._emit(batch)

    def _emit(self, batch: EpochBatch) -> None:
        if self.save_frames:
            self._fbuf[batch.epoch] = write_frames(batch.units)
        self._epoch(batch)

    def _flush(self) -> None:
        if not self._buf:
            return
        k = min(self._buf) // CHUNK
        atomic_write_bytes(self.out / "records" / f"r{k:04d}.pkl", pickle.dumps(self._buf, protocol=5))
        if self.save_frames:
            blob = pickle.dumps(self._fbuf, protocol=5)
            atomic_write_bytes(self.out / "frames" / f"f{k:04d}.pkl", blob)
        self._buf, self._fbuf = {}, {}

    def finish(self) -> dict:
        while self.gw.next_e < self.n_epochs:
            self._emit(self.gw.empty_batch(self.gw.next_e))
        self._flush()
        gw = self.gw
        refusals = [e for e, r in gw.records.items() if r.cosig is None]
        return {"n_epochs": len(gw.records), "refusals": refusals, "retained_units": len(gw.retention.items),
                "retained_s": int(self.retained_s), "retention_bytes": int(gw.retention.size_bytes()),
                "gw_id": gw.gw_id.hex()}


def load_records(run_dir: Path, epochs=None) -> dict[int, dict]:
    run_dir = Path(run_dir)
    want = None if epochs is None else set(int(e) for e in epochs)
    out = {}
    for p in sorted((run_dir / "records").glob("r*.pkl")):
        k = int(p.stem[1:])
        if want is not None and not any(k * CHUNK <= e < (k + 1) * CHUNK for e in want):
            continue
        d = pickle.loads(p.read_bytes())
        for e, r in d.items():
            if want is None or e in want:
                out[e] = r
    return out


def load_frames(run_dir: Path, epochs) -> dict[int, dict[int, list[Frame]]]:
    run_dir = Path(run_dir)
    want = set(int(e) for e in epochs)
    out = {}
    for k in sorted({e // CHUNK for e in want}):
        p = run_dir / "frames" / f"f{k:04d}.pkl"
        d = pickle.loads(p.read_bytes())
        for e, blob in d.items():
            if e in want:
                out[e] = read_frames(blob)
    return out


class RunView:
    """Gateway-like view of a persisted run for a set of epochs (build_disclosure / adversaries)."""

    def __init__(self, run_dir: Path, dep: Deployment, epochs, context=(), with_frames: bool = True,
                 codec=None, records: dict | None = None, frames: dict | None = None):
        self.run_dir = Path(run_dir)
        self.dep = dep
        self.beacon = dep.beacon
        self.codec = codec or EncodeMetaCodec()
        self.gw_id = None
        self.witness = dep.witness()
        epochs = sorted(set(int(e) for e in epochs))
        ctx = sorted(set(int(e) for e in context) - set(epochs))
        raw = records if records is not None else load_records(self.run_dir, epochs + ctx)
        if frames is None:
            frames = load_frames(self.run_dir, epochs) if with_frames else {}
        self.retention = RetentionStore(EpochCipher(dep.escrow))
        self.records: dict[int, EpochRecord] = {}
        for e in epochs + ctx:
            r = raw[e]
            self.records[e] = self._record(r, frames.get(e) if e in epochs else None, full=e in epochs)
            if self.gw_id is None:
                self.gw_id = r["header"].gw_id
        for e in epochs:
            rec = self.records[e]
            for i in rec.retained:
                u = rec.batch.units[i]
                if u.frames is None:
                    raise ValueError(f"retained unit ({e}, {i}) has no frames in the dump")
                self.retention.retain(e, u)
        reg_gw = Gateway(dep.keys, dep.clock, dep.beacon, _RegOnly(raw[epochs[0]]["header"]), self.witness,
                         dep.escrow)
        register(reg_gw, dep)

    def _record(self, r: dict, frames, full: bool) -> EpochRecord:
        st = MetadataStore.deserialize(r["store"])
        hdr = r["header"]
        if not full:
            return EpochRecord(e=r["e"], batch=None, header=hdr, sigma=r["sigma"], store=st, state=r["state"],
                               taus=r["taus"], leaves=[], opening=r["opening"], cosig=r["cosig"])
        batch = st.to_batch(hdr.t_e_us, codec=self.codec)
        for u in batch.units:
            if u.pay and frames is not None and u.index in frames:
                u.frames = frames[u.index]
                u.pkt_leaves = hs.frame_leaves([f.ts_us for f in u.frames], [f.dir for f in u.frames],
                                               [f.caplen for f in u.frames], [f.data for f in u.frames])
        Z = np.unpackbits(np.frombuffer(r["Z"], dtype=np.uint8))[:r["n"]].astype(bool).tolist()
        leaves = [u.leaf for u in batch.units]
        rec = EpochRecord(e=r["e"], batch=batch, header=hdr, sigma=r["sigma"], store=st, state=r["state"],
                          taus=r["taus"], leaves=leaves, opening=r["opening"], cosig=r["cosig"], refusal=r["refusal"],
                          t_send=r["t_send"], beacon_round=self.beacon.get(r["r_star"]), vrf_proof=r["vrf_proof"],
                          beta_vrf=r["beta_vrf"], u=r["u"], Z=Z, retained=list(r["retained"]))
        return rec


class _RegOnly:
    """Registration-only stand-in policy: (H(θ), H(policy)) of a committed header."""

    def __init__(self, hdr):
        self.H = (hdr.H_theta, hdr.H_policy)

    def initial_state(self, e0):
        from ..protocol.policy import PolicyState
        return PolicyState(e=e0)

    def registration(self):
        return self.H


# ============================================================================== E6 helpers
@dataclass
class Timer:
    samples: dict = field(default_factory=dict)

    def add(self, name: str, v: float) -> None:
        self.samples.setdefault(name, []).append(float(v))

    def median(self, name: str) -> float:
        return float(np.median(self.samples[name]))


def machine_state() -> dict:
    """Load and core information recorded with every timing (other P1 jobs may run)."""
    import platform
    import subprocess
    out = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "platform": platform.platform(),
           "python": platform.python_version(), "loadavg": list(os.getloadavg())}
    for k in ("hw.ncpu", "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu", "machdep.cpu.brand_string",
              "hw.memsize"):
        try:
            out[k] = subprocess.run(["sysctl", "-n", k], capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:                                   # noqa: BLE001
            pass
    try:
        import psutil
        out["cpu_percent_1s"] = psutil.cpu_percent(interval=1.0)
        out["mem_available_gb"] = psutil.virtual_memory().available / 2 ** 30
    except Exception:                                       # noqa: BLE001
        pass
    return out


def write_json(path: Path, obj) -> None:
    atomic_write_json(Path(path), obj)


# ============================================================================== E6: timed pipeline
class TimedPolicy:
    """Transparent proxy that times ``thresholds`` (utility + τ) and ``update`` (controller, boost detectors)."""

    def __init__(self, base):
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "t_thresholds", [])
        object.__setattr__(self, "t_update", [])

    def thresholds(self, batch, state):
        t0 = time.perf_counter()
        out = self.base.thresholds(batch, state)
        self.t_thresholds.append(time.perf_counter() - t0)
        return out

    def update(self, state, record):
        t0 = time.perf_counter()
        out = self.base.update(state, record)
        self.t_update.append(time.perf_counter() - t0)
        return out

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "base"), name)


def time_pipeline(vc: VirtualCapture, dep: Deployment, policy, n_epochs: int, keep=None, registry=None) -> dict:
    """Wall-clock breakdown of the gateway pipeline over the first ``n_epochs`` epochs of a virtual capture.

    Passes (same process, sequential): (1) merged reader only (k-way merge + on-the-fly rewrite);
    (2) reader + header parser; (3) reader + parser + flow table with frames (frame hashes, ℓ_ij, root^pkt,
    L_i, enc(m_i)); (4) as (3) plus the Gateway per epoch (commit: store, Merkle, CtS τ, C^st, header, σ;
    publish: witness cosign; select: drand β, ECVRF prove, PRF, Z, AEAD retention; finish: controller, boost
    detectors). Per-packet cost = (4) / packets."""
    from ..io.pcap_reader import parser_for
    parse = parser_for(1)
    hi_ts = vc.t0_gw_us + int(n_epochs) * T_US
    out = {"n_epochs": int(n_epochs)}
    t0 = time.perf_counter()
    n = nb = 0
    for k, ts, cl, wl, fr, i, _c in vc.packets():
        if ts >= hi_ts:
            break
        n += 1
        nb += cl
    out["reader_s"] = time.perf_counter() - t0
    out["packets"] = n
    out["bytes"] = nb
    t0 = time.perf_counter()
    for k, ts, cl, wl, fr, i, _c in vc.packets():
        if ts >= hi_ts:
            break
        parse(fr, wl)
    out["reader_parse_s"] = time.perf_counter() - t0
    r1 = unitise(vc, keep=keep, registry=registry, keep_frames=True, max_epoch=n_epochs)
    out["flowtable_s"] = r1.elapsed_s
    tp = TimedPolicy(policy)
    gw = Gateway(dep.keys, dep.clock, dep.beacon, tp, dep.witness(), dep.escrow, keep_frames=False)
    register(gw, dep)
    tm = {"commit": [], "publish": [], "select": [], "finish": [], "n": [], "n_pay": [], "pkts": [], "retained": []}

    def on_batch(b, recs):
        if b.epoch >= n_epochs:
            return
        while gw.next_e < b.epoch:
            gw.run_epoch(gw.empty_batch(gw.next_e))
        a = time.perf_counter()
        rec = gw.commit(b)
        c = time.perf_counter()
        gw.publish(rec)
        d = time.perf_counter()
        gw.select(rec)
        f = time.perf_counter()
        gw.finish(rec)
        g = time.perf_counter()
        tm["commit"].append(c - a)
        tm["publish"].append(d - c)
        tm["select"].append(f - d)
        tm["finish"].append(g - f)
        tm["n"].append(b.n)
        tm["n_pay"].append(b.n_pay)
        tm["pkts"].append(sum(u.n_pkts for u in b.units))
        tm["retained"].append(len(rec.retained))

    r2 = unitise(vc, keep=keep, registry=registry, on_batch=on_batch, keep_frames=True, max_epoch=n_epochs)
    out["total_s"] = r2.elapsed_s
    out["gateway_s"] = float(sum(tm["commit"]) + sum(tm["publish"]) + sum(tm["select"]) + sum(tm["finish"]))
    out["policy_thresholds_s"] = float(sum(tp.t_thresholds))
    out["policy_update_s"] = float(sum(tp.t_update))
    out["per_epoch_ms"] = {k: 1e3 * float(np.median(tm[k])) for k in ("commit", "publish", "select", "finish")}
    out["per_epoch_ms_p99"] = {k: 1e3 * float(np.quantile(tm[k], 0.99)) for k in ("commit", "publish", "select", "finish")}
    out["policy_thresholds_ms_median"] = 1e3 * float(np.median(tp.t_thresholds)) if tp.t_thresholds else None
    out["units"] = int(sum(tm["n"]))
    out["payload_units"] = int(sum(tm["n_pay"]))
    out["retained_units"] = int(sum(tm["retained"]))
    p = max(1, out["packets"])
    out["us_per_packet"] = {"reader": 1e6 * out["reader_s"] / p, "reader_parse": 1e6 * out["reader_parse_s"] / p,
                            "flowtable": 1e6 * out["flowtable_s"] / p, "total": 1e6 * out["total_s"] / p,
                            "gateway_epoch_work": 1e6 * out["gateway_s"] / p,
                            "policy_thresholds": 1e6 * out["policy_thresholds_s"] / p}
    out["pps_per_core"] = 1e6 / out["us_per_packet"]["total"]
    out["commit_latency_ms"] = {"median": out["per_epoch_ms"]["commit"], "p99": out["per_epoch_ms_p99"]["commit"],
                                "max": 1e3 * float(np.max(tm["commit"]))}
    return out


# ============================================================================== E6: buffer occupancy
def buffer_occupancy(ts_us: np.ndarray, nbytes: np.ndarray, t0_us: int, n_epochs: int, T_s: float = 30.0,
                     delta_c: float = 3.0, delta_sel: float = 0.5, ahead: int = 1, grid_s: float = 0.25) -> dict:
    """Bytes of payload content waiting for selection: a packet of epoch e (t_e = t0 + e·T) is buffered from its
    arrival until selection of e at t_{e+1} + Δ_c + Δ_sel + (ahead − 1)·T (DESIGN §2.3, Prop. 5(ii)).
    Returns p50/p99/max over a time grid plus the exact maximum (evaluated just before every selection)."""
    ts = np.asarray(ts_us, dtype=np.int64)
    b = np.asarray(nbytes, dtype=np.float64)
    o = np.argsort(ts, kind="stable")
    ts, b = ts[o], b[o]
    T = int(T_s * 1e6)
    ep = (ts - t0_us) // T
    m = (ep >= 0) & (ep < n_epochs)
    ts, b, ep = ts[m], b[m], ep[m]
    cumA = np.concatenate([[0.0], np.cumsum(b)])
    Be = np.bincount(ep, weights=b, minlength=n_epochs)
    sel = t0_us + (np.arange(n_epochs) + 1) * T + int((delta_c + delta_sel + (ahead - 1) * T_s) * 1e6)
    cumD = np.concatenate([[0.0], np.cumsum(Be)])

    def occ(t):
        a = cumA[np.searchsorted(ts, t, side="right")]
        d = cumD[np.searchsorted(sel, t, side="right")]
        return a - d

    grid = np.arange(t0_us, t0_us + n_epochs * T, int(grid_s * 1e6))
    g = occ(grid)
    exact = occ(sel - 1)
    resid = (sel - (t0_us + np.arange(n_epochs) * T)) / 1e6
    return {"ahead": ahead, "residence_max_s": float(resid.max()), "p50_bytes": float(np.quantile(g, 0.5)),
            "p99_bytes": float(np.quantile(g, 0.99)), "max_bytes": float(max(g.max(), exact.max())),
            "mean_rate_Bps": float(b.sum() / (n_epochs * T_s)), "grid_s": grid_s}


# ============================================================================== E6: synthetic epochs
def synthetic_records(n: int, pool: list, rng: np.random.Generator) -> list:
    """n UnitRecords drawn (with replacement) from real records of the E5 run, with fresh keys/root_pkt so that
    every leaf is distinct (sizes and the payload share follow the real stream)."""
    from ..protocol.store import UnitRecord
    idx = rng.integers(0, len(pool), n)
    out = []
    for j, k in enumerate(idx):
        r = pool[int(k)]
        meta = bytearray(r.meta)
        meta[1:17] = rng.integers(0, 256, 16, dtype=np.uint8).tobytes()        # fresh ip_a (key)
        root = rng.integers(0, 256, 32, dtype=np.uint8).tobytes() if r.pay else None
        out.append(UnitRecord(meta=bytes(meta), pay=r.pay, root_pkt=root, device_id=r.device_id, pkts=r.pkts,
                              frame_bytes=r.frame_bytes, s=r.s, first_ts_us=r.first_ts_us, fps=r.fps))
    return out
