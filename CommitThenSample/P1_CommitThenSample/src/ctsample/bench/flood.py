"""E6 rev1 — gateway commit time under unit floods (review R2-M1, R3-M4; Prop. 5(i)); used by scripts/15_e6.py.

A flood epoch = one real epoch of the E5 rev1 honest run (LP-IoT23(34-1, 1 %), ≈ 217 units, 163 payload) plus N
synthetic units of ONE compromised LAN device (the stream's infected host):
  * zero-payload flood: a random-port SYN scan (one 60-B SYN per unit, random WAN responder and port, S0), i.e.
    N/30 packets per second — the reviewer's 10^6 units/epoch ≈ 33 k SYN/s;
  * payload flood: N one-packet TCP units with 16 B of payload (e.g. a data-carrying SYN flood such as
    Edge-IIoTset's DDoS_TCP_SYN, 1.18 M payload units).
History: the real committed metadata of the 119 preceding epochs ("single"), or the same with N flood units of
that device added to EVERY history epoch ("sustained"); the policy's 120-epoch rolling statistics are primed
(untimed) so that the timed commit pays the steady-state cost of one epoch.
Timed: ``Gateway.commit`` of the flood epoch (unit leaves, Merkle root, CtS thresholds = features + rolling
statistics + utility, store with the committed τ column and H(store), C^st, header, Ed25519), with the policy's
``thresholds`` share; the zero-payload cap (protocol/flood.py) is timed separately when applied.
"""
from __future__ import annotations

import copy
import time

import numpy as np

from ..crypto import hashing as hs
from ..protocol.flood import cap_level, cap_zero_payload
from ..units import encode as enc
from ..units import features as F
from ..units.batch import EpochBatch, Frame, Unit

T_US = 30_000_000

__all__ = ["scan_records", "scan_units", "scan_cols", "payload_flood_units", "device_src_ip", "flood_batch",
           "capped_history_cols"]


def device_src_ip(batch: EpochBatch, device_id: int) -> bytes:
    """A LAN address of the device (initiator side of one of its units; fallback 192.168.137.234)."""
    for u in batch.units:
        if u.device_id == device_id:
            d = enc.dec_metadata(u.meta)
            ip = d["ip_a"]
            if ip[:10] == bytes(10) and ip[10:12] == b"\xff\xff" and ip[12] == 192 and ip[13] == 168:
                return ip
    return bytes(10) + b"\xff\xff" + bytes([192, 168, 137, 234])


def scan_records(n: int, device_id: int, src_ip16: bytes, t0_us: int, rng: np.random.Generator) -> np.ndarray:
    """n compact enc(m_i) records (features.ZERO_DT) of a random-port SYN scan inside [t0, t0 + T)."""
    r = np.zeros(n, dtype=F.ZERO_DT)
    r["proto"] = 6
    r["ip_a"] = np.frombuffer(src_ip16, dtype=np.uint8)[None, :]
    r["port_a"] = rng.integers(1024, 65536, n)
    ipb = np.zeros((n, 16), dtype=np.uint8)
    ipb[:, 10:12] = 0xFF
    ipb[:, 12:16] = rng.integers(1, 255, (n, 4), dtype=np.uint8)
    r["ip_b"] = ipb
    r["port_b"] = rng.integers(1, 65536, n)
    r["device_id"] = device_id
    r["first_ts"] = t0_us + np.sort(rng.integers(0, T_US - 1, n))
    r["pk0"] = 1
    r["syn"] = 1
    r["cs"] = enc.CS_CODE["S0"]
    return r


def scan_units(rec: np.ndarray, epoch: int, frame_len: int = 60) -> list[Unit]:
    """Unit objects of scan records (leaf0 precomputed, as the flow table does at unit close)."""
    raw = rec.tobytes()
    L = F.ZERO_DT.itemsize
    out = []
    for j in range(len(rec)):
        meta = raw[j * L:(j + 1) * L]
        ts = int(rec["first_ts"][j])
        key = meta[:37]
        out.append(Unit(epoch=epoch, index=0, key=key, device_id=int(rec["device_id"][j]), first_ts_us=ts, last_ts_us=ts,
                        pkts=(1, 0), payload_bytes=(0, 0), frame_bytes=(frame_len, 0), s=frame_len + 16, pay=False,
                        fps=False, flow_uid=hs.flow_uid(key, ts), meta=meta, leaf=hs.leaf0(meta)))
    return out


def scan_cols(rec: np.ndarray) -> dict:
    """Feature columns (features.cols_from_meta layout) of scan records, without Unit objects (history epochs)."""
    c = F._empty_cols(len(rec))
    F._fill(c, np.arange(len(rec)), rec, False)
    return c


def payload_flood_units(n: int, device_id: int, src_ip16: bytes, t0_us: int, rng: np.random.Generator,
                        epoch: int, payload_len: int = 16) -> list[Unit]:
    """n one-packet TCP payload units (Ethernet/IPv4/TCP + payload_len bytes), frame leaves and leaves precomputed."""
    out = []
    smac, dmac = bytes([2, 0, 0, 0, 0, 9]), bytes([2, 0, 0, 0, 0, 1])
    sip = src_ip16[12:16]
    ts_all = t0_us + np.sort(rng.integers(0, T_US - 1, n))
    for j in range(n):
        dip = bytes(rng.integers(1, 255, 4, dtype=np.uint8).tolist())
        sport, dport = int(rng.integers(1024, 65536)), int(rng.integers(1, 65536))
        body = rng.integers(0, 256, payload_len, dtype=np.uint8).tobytes()
        import struct
        tcp = struct.pack("!HHIIBBHHH", sport, dport, 1, 0, 0x50, 0x02, 65535, 0, 0)
        ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(tcp) + len(body), 0, 0, 64, 6, 0, sip, dip)
        data = dmac + smac + b"\x08\x00" + ip + tcp + body
        ts = int(ts_all[j])
        fr = Frame(ts_us=ts, dir=0, caplen=len(data), wirelen=len(data), data=data)
        key = enc.make_key(6, src_ip16, sport, bytes(10) + b"\xff\xff" + dip, dport)
        fu = hs.flow_uid(key, ts)
        m = {"key": key, "device_id": device_id, "first_ts_us": ts, "last_ts_us": ts, "pkts": (1, 0),
             "payload_bytes": (payload_len, 0), "frame_bytes": (len(data), 0), "n_syn": 1, "n_synack": 0, "n_fin": 0,
             "n_rst": 0, "n_psh": 0, "conn_state": enc.CS_CODE["S0"], "sizes": [payload_len], "iats": [],
             "s": len(data) + 16, "flow_uid": fu, "continuation": False, "fps": True, "pay": True}
        meta = enc.enc_metadata(m)
        pl = [hs.frame_leaf(ts, 0, len(data), data)]
        root = hs.merkle_root(pl)
        out.append(Unit(epoch=epoch, index=0, key=key, device_id=device_id, first_ts_us=ts, last_ts_us=ts, pkts=(1, 0),
                        payload_bytes=(payload_len, 0), frame_bytes=(len(data), 0), s=len(data) + 16, pay=True, fps=True,
                        flow_uid=fu, meta=meta, leaf=hs.unit_leaf(meta, root), root_pkt=root, pkt_leaves=pl,
                        frames=[fr]))
    return out


def flood_batch(base: EpochBatch, flood: list[Unit]) -> EpochBatch:
    """Base epoch (deep-copied) + flood units in canonical order (first timestamp, key), indices reassigned."""
    units = copy.deepcopy(base.units) + list(flood)
    units.sort(key=lambda u: (u.first_ts_us, u.key))
    for i, u in enumerate(units):
        u.index = i
        u.epoch = base.epoch
    return EpochBatch(epoch=base.epoch, t_start_us=base.t_start_us, units=units)


def capped_history_cols(base_cols: dict, n_flood: int, device_id: int, src_ip16: bytes, t0_us: int, epoch: int,
                        rng: np.random.Generator, k_dev: int, k_tot: int | None) -> dict:
    """History columns of a CAPPED sustained flood epoch: base + (c − 1) scan rows + 1 overflow row (c = cap level
    for this epoch's zero-payload counts). The overflow row is built by protocol/flood.py from a small sample."""
    from ..protocol import flood as FL
    pay = base_cols["pay"].astype(bool)
    dev = base_cols["device_id"]
    counts = {}
    for d in dev[~pay]:
        counts[int(d)] = counts.get(int(d), 0) + 1
    base_dev = counts.get(device_id, 0)
    counts[device_id] = base_dev + n_flood
    c = cap_level(counts, k_dev, k_tot)
    n_keep = max(c - 1 - base_dev, 0) if counts[device_id] > c else n_flood
    parts = [base_cols, scan_cols(scan_records(n_keep, device_id, src_ip16, t0_us, rng))]
    if counts[device_id] > c:
        sample = scan_units(scan_records(min(1000, n_flood - n_keep), device_id, src_ip16, t0_us, rng), epoch)
        ov = FL._overflow_unit(epoch, sample)
        parts.append(F.cols_from_units([ov]))
    return F.concat_cols(parts)


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


def cap_batch(batch: EpochBatch, k_dev: int, k_tot: int | None):
    return timed(cap_zero_payload, batch, k_dev, k_tot)
