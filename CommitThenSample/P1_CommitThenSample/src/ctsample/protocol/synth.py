"""Synthetic multi-device streams for protocol tests and benchmarks (not used on real data).

``synthetic_stream`` draws a "true" packet stream (the one the counter source I sees), applies the
gateway capture-loss model, and builds the gateway's committed units (units.batch.Unit) from the
captured frames, in canonical order (first_ts_us, key). Metadata enc(m_i) uses
``SyntheticMetaCodec``: a fixed 46-byte field block followed by random filler bytes
(128 B for payload units, 56 B for zero-payload units, the DESIGN §3.2 sizes).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from ..crypto import hashing as hs
from ..units.batch import EpochBatch, Frame, Unit
from .beacon import EpochClock
from .counters import CounterSource, cells_from_packets

_META = struct.Struct("<BIIIQQQQB")       # flags, device, pkts0, pkts1, fb0, fb1, s, first_ts_us, fps
PAY_META_LEN = 128
ZERO_META_LEN = 56

__all__ = ["SyntheticMetaCodec", "SyntheticStream", "synthetic_stream", "synthetic_packets", "epoch_of_us_fn",
           "units_from_frames"]


class SyntheticMetaCodec:
    """enc/dec of the synthetic m_i layout; ``decode`` is the verifier's meta-codec hook."""

    @staticmethod
    def encode(device_id: int, pkts, frame_bytes, s: int, first_ts_us: int, pay: bool, fps: bool,
               filler: bytes) -> bytes:
        head = _META.pack(1 if pay else 0, device_id, pkts[0], pkts[1], frame_bytes[0], frame_bytes[1], s,
                          first_ts_us, 1 if fps else 0)
        n = (PAY_META_LEN if pay else ZERO_META_LEN) - len(head)
        return head + filler[:n].ljust(n, b"\x00")

    @staticmethod
    def decode(meta: bytes, pay: bool | None = None) -> dict:
        if len(meta) not in (PAY_META_LEN, ZERO_META_LEN):
            raise ValueError(f"bad meta length {len(meta)}")
        f, dev, p0, p1, b0, b1, s, fts, fps = _META.unpack_from(meta)
        if (len(meta) == PAY_META_LEN) != bool(f & 1):
            raise ValueError("meta length does not match its pay flag")
        return {"pay": bool(f & 1), "device_id": dev, "pkts": (p0, p1), "frame_bytes": (b0, b1), "s": s,
                "first_ts_us": fts, "fps": bool(fps)}



def epoch_of_us_fn(clock: EpochClock):
    t0 = clock.t_e_us(0)
    T = clock.period * 1_000_000

    def f(ts_us):
        return (np.asarray(ts_us, dtype=np.int64) - t0) // T
    return f


@dataclass
class _Spec:
    epoch: int
    device_id: int
    key: bytes
    frames: list[Frame]
    plens: list[int]
    filler: bytes


@dataclass
class SyntheticStream:
    clock: EpochClock
    batches: list[EpochBatch]
    ts_us: np.ndarray
    device_ids: np.ndarray
    lengths: np.ndarray
    keep: np.ndarray
    committed_cells: dict
    counter_cells: dict
    n_true_units: int
    n_epochs: int
    extra: dict = field(default_factory=dict)


def _key(rng: np.random.Generator, device_id: int, proto: int) -> bytes:
    ip_a = b"\x00" * 10 + b"\xff\xff" + bytes([192, 168, 1, 10 + device_id])
    ip_b = b"\x00" * 10 + b"\xff\xff" + bytes(rng.integers(1, 255, size=4, dtype=np.uint8).tolist())
    return (bytes([proto]) + ip_a + struct.pack("<H", int(rng.integers(1024, 65535)))
            + ip_b + struct.pack("<H", int(rng.choice([80, 443, 1883, 502, 53, 8080]))))


def units_from_frames(specs: list[_Spec], epoch: int, t_start_us: int,
                      seen_uids: set | None = None) -> EpochBatch:
    """Build one epoch's committed units from (already loss-filtered) per-unit frame lists."""
    seen_uids = set() if seen_uids is None else seen_uids
    rows = []
    for sp in specs:
        if not sp.frames:
            continue
        fr = sorted(zip(sp.frames, sp.plens), key=lambda x: x[0].ts_us)
        frames = [f for f, _ in fr]
        plens = [p for _, p in fr]
        first_ts = frames[0].ts_us
        rows.append((first_ts, sp.key, sp, frames, plens))
    rows.sort(key=lambda r: (r[0], r[1]))
    units = []
    for idx, (first_ts, key, sp, frames, plens) in enumerate(rows):
        pk = [0, 0]
        pb = [0, 0]
        fb = [0, 0]
        for f, p in zip(frames, plens):
            pk[f.dir] += 1
            pb[f.dir] += p
            fb[f.dir] += f.caplen
        s = sum(f.caplen + 16 for f in frames)
        pay = (pb[0] + pb[1]) > 0
        uid = hs.flow_uid(key, first_ts)
        fps = pay and key not in seen_uids
        if pay:
            seen_uids.add(key)
        meta = SyntheticMetaCodec.encode(sp.device_id, pk, fb, s, first_ts, pay, fps, sp.filler)
        units.append(Unit(epoch=epoch, index=idx, key=key, device_id=sp.device_id, first_ts_us=first_ts,
                          last_ts_us=frames[-1].ts_us, pkts=(pk[0], pk[1]), payload_bytes=(pb[0], pb[1]),
                          frame_bytes=(fb[0], fb[1]), s=s, pay=pay, fps=fps, flow_uid=uid, meta=meta,
                          frames=frames if pay else None))
    return EpochBatch(epoch=epoch, t_start_us=t_start_us, units=units)


def synthetic_packets(clock: EpochClock, n_epochs: int, n_devices: int = 4, units_per_device=(1, 6),
                      pkts_per_unit=(1, 24), payload_len=(1, 1400), seed: int = 0):
    """Packet-only twin of ``synthetic_stream`` (same count/size distributions, no frame bytes):
    (ts_us, device_ids, lengths) arrays, for fast ε_cell calibration on many epochs."""
    rng = np.random.default_rng(seed)
    T = clock.period * 1_000_000
    ts_l, dev_l, len_l = [], [], []
    for e in range(n_epochs):
        t0 = clock.t_e_us(e)
        for dev in range(n_devices):
            lo, hi = units_per_device[dev] if isinstance(units_per_device, dict) else units_per_device
            nu = int(rng.integers(lo, hi + 1))
            if nu == 0:
                continue
            npk = rng.integers(pkts_per_unit[0], pkts_per_unit[1] + 1, size=nu)
            n = int(npk.sum())
            ts_l.append(t0 + rng.integers(0, T, size=n))
            dev_l.append(np.full(n, dev, dtype=np.int64))
            len_l.append(54 + rng.integers(payload_len[0], payload_len[1], size=n))
    if not ts_l:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    return np.concatenate(ts_l), np.concatenate(dev_l), np.concatenate(len_l)


def synthetic_stream(clock: EpochClock, n_epochs: int, n_devices: int = 4, units_per_device=(1, 6),
                     pay_frac: float = 0.7, pkts_per_unit=(1, 24), seed: int = 0,
                     counter_source: CounterSource | None = None, marker: bytes | None = None,
                     marker_device: int | None = None, marker_rate: float = 0.0,
                     payload_len=(1, 1400)) -> SyntheticStream:
    """Synthetic stream over epochs 0..n_epochs−1 in protocol time (t_e = clock.t_e(e)).

    ``units_per_device`` is (lo, hi) for every device or {device: (lo, hi)}.

    ``counter_source`` (optional) supplies the gateway capture-loss mask and the counter cells; without
    it every packet is captured and counter = committed. ``marker`` bytes are planted in the payload of
    a fraction ``marker_rate`` of ``marker_device``'s payload units (detector tests)."""
    rng = np.random.default_rng(seed)
    T = clock.period * 1_000_000
    specs: list[_Spec] = []
    for e in range(n_epochs):
        t0 = clock.t_e_us(e)
        for dev in range(n_devices):
            lo, hi = units_per_device[dev] if isinstance(units_per_device, dict) else units_per_device
            for _ in range(int(rng.integers(lo, hi + 1))):
                pay = rng.random() < pay_frac
                proto = 6 if rng.random() < 0.8 else 17
                key = _key(rng, dev, proto)
                n = int(rng.integers(pkts_per_unit[0], pkts_per_unit[1] + 1))
                ts = np.sort(t0 + rng.integers(0, T, size=n))
                frames, plens = [], []
                plant = marker is not None and pay and dev == marker_device and rng.random() < marker_rate
                for j in range(n):
                    d = int(rng.integers(0, 2))
                    if pay and (j > 0 or n == 1 or rng.random() < 0.5):
                        plen = int(rng.integers(payload_len[0], payload_len[1]))
                    else:
                        plen = 0
                    hdr = rng.bytes(54)
                    body = rng.bytes(plen)
                    if plant and plen >= len(marker):
                        body = marker + body[len(marker):]          # plant once, in the first big frame
                        plant = False
                    data = hdr + body
                    frames.append(Frame(ts_us=int(ts[j]), dir=d, caplen=len(data), wirelen=len(data), data=data))
                    plens.append(plen)
                if pay and sum(plens) == 0:
                    plens[-1] = 1
                    f = frames[-1]
                    frames[-1] = Frame(f.ts_us, f.dir, f.caplen + 1, f.wirelen + 1, f.data + rng.bytes(1))
                specs.append(_Spec(e, dev, key, frames, plens, rng.bytes(PAY_META_LEN)))
    # flatten the true packet stream
    ts_all, dev_all, len_all, ref = [], [], [], []
    for si, sp in enumerate(specs):
        for j, f in enumerate(sp.frames):
            ts_all.append(f.ts_us)
            dev_all.append(sp.device_id)
            len_all.append(f.caplen)
            ref.append((si, j))
    ts_a = np.array(ts_all, dtype=np.int64)
    dev_a = np.array(dev_all, dtype=np.int64)
    len_a = np.array(len_all, dtype=np.int64)
    eof = epoch_of_us_fn(clock)
    if counter_source is not None:
        sim = counter_source.simulate(ts_a, dev_a, len_a)
        keep = sim["keep"]
        counter_cells = sim["counter_cells"]
    else:
        keep = np.ones(len(ts_a), dtype=bool)
        counter_cells = cells_from_packets(eof(ts_a), dev_a, len_a)
    committed_cells = cells_from_packets(eof(ts_a), dev_a, len_a, keep)
    # apply the capture mask
    kept: dict[int, list[int]] = {}
    for k, (si, j) in enumerate(ref):
        if keep[k]:
            kept.setdefault(si, []).append(j)
    by_epoch: dict[int, list[_Spec]] = {}
    for si, sp in enumerate(specs):
        js = kept.get(si, [])
        sp2 = _Spec(sp.epoch, sp.device_id, sp.key, [sp.frames[j] for j in js], [sp.plens[j] for j in js], sp.filler)
        by_epoch.setdefault(sp.epoch, []).append(sp2)
    seen: set = set()
    batches = [units_from_frames(by_epoch.get(e, []), e, clock.t_e_us(e), seen) for e in range(n_epochs)]
    return SyntheticStream(clock=clock, batches=batches, ts_us=ts_a, device_ids=dev_a, lengths=len_a, keep=keep,
                           committed_cells=committed_cells, counter_cells=counter_cells, n_true_units=len(specs),
                           n_epochs=n_epochs)
