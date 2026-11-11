"""Flood mitigation for the commit window (rev1, review R2-M1 (c)): cap on ZERO-PAYLOAD units per epoch.

Every 5-tuple is a unit, so a compromised LAN device that scans random ports creates up to one unit per packet;
at ≈ 33 k SYN/s that is ≈ 10^6 units per epoch, and the commit (unit leaves, Merkle tree, store, the policy's
features and 120-epoch rolling statistics) no longer fits the one-round-ahead build budget Δ_build = 0.8 s
(results/summary/E6_flood.json). The cap bounds the number of committed zero-payload units:

* per device: at most ``k_dev`` zero-payload units per epoch;
* per epoch (optional): at most ``k_tot`` zero-payload units in total; when the per-device caps still exceed it,
  the per-device cap is lowered to the largest c with Σ_d min(n_d, c) ≤ k_tot (water filling, so devices below
  the level are untouched and the heaviest devices are aggregated first).
A capped device keeps its first c − 1 zero-payload units (canonical order: first timestamp, key) and ONE committed
*overflow unit* aggregates the rest: zero-payload (no content, π = 0, b = 0 as for every zero-payload unit),
key = (proto 255, ip_a = 12 zero bytes ‖ u32le device id, port_a 0xFFFF, ip_b = u32le aggregated units ‖ u32le
distinct responder addresses ‖ u32le distinct responder ports ‖ u32le S0/REJ units, port_b 0), first timestamp =
the earliest aggregated one, per-direction packets, frame bytes and s summed (so the per-device counter cells of
V7 are unchanged), TCP flag counts summed (saturating), connection state OTH. Payload units are never aggregated
(each must stay individually sampleable); their volume is bounded by the budget's flood guard, not here.

The cap changes only the granularity of the committed METADATA of flooding devices (aggregate counts instead of
one record per scan flow); it does not touch content, π of any payload unit, or any statement over payload
units. Both caps are part of the registered policy (H(policy)); ``check_cap`` lets the verifier report a
violation as a compliance finding. The thresholds of our streams were chosen so that the cap never binds on any
training or test stream (≤ 6,327 zero-payload units per device-epoch, ≤ 12,061 units per epoch observed).
In a production gateway the flow table would apply the cap at epoch close without materialising the excess units
(here it is a transform of the epoch batch, timed separately in E6).
"""
from __future__ import annotations

import struct
from collections import defaultdict

from ..crypto import hashing as hs
from ..units import encode as enc
from ..units.batch import EpochBatch, Unit

OVERFLOW_PROTO = 255
K_DEV_DEFAULT = 8192
K_TOT_DEFAULT = 16384
_IPB = struct.Struct("<IIII")

__all__ = ["OVERFLOW_PROTO", "K_DEV_DEFAULT", "K_TOT_DEFAULT", "overflow_key", "is_overflow_meta", "cap_level",
           "cap_zero_payload", "check_cap"]


def overflow_key(device_id: int, n_agg: int, n_rdst: int, n_dport: int, n_s0rej: int) -> bytes:
    ip_a = bytes(12) + struct.pack("<I", int(device_id) & 0xFFFFFFFF)
    ip_b = _IPB.pack(*(min(int(x), 0xFFFFFFFF) for x in (n_agg, n_rdst, n_dport, n_s0rej)))
    return enc.make_key(OVERFLOW_PROTO, ip_a, 0xFFFF, ip_b, 0)


def is_overflow_meta(meta: bytes) -> bool:
    return bool(meta) and meta[0] == OVERFLOW_PROTO


def cap_level(counts: dict[int, int], k_dev: int, k_tot: int | None) -> int:
    """Per-device cap c: min(k_dev, largest c with Σ_d min(n_d, c) ≤ k_tot)."""
    c = int(k_dev)
    if k_tot is None or sum(min(n, c) for n in counts.values()) <= k_tot:
        return c
    lo, hi = 1, c                      # Σ min(n_d, c) is non-decreasing in c
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(n, mid) for n in counts.values()) <= k_tot:
            lo = mid
        else:
            hi = mid - 1
    return max(1, lo)


def _overflow_unit(epoch: int, units: list[Unit]) -> Unit:
    import numpy as np
    from ..units import features as F
    dev = units[0].device_id
    c = F.cols_from_meta([u.meta for u in units])                  # vectorised decode of the compact records
    fb0 = int(np.fromiter((u.frame_bytes[0] for u in units), dtype=np.int64, count=len(units)).sum())
    fb1 = int(np.fromiter((u.frame_bytes[1] for u in units), dtype=np.int64, count=len(units)).sum())
    s = int(np.fromiter((u.s for u in units), dtype=np.int64, count=len(units)).sum())
    last = int(np.fromiter((u.last_ts_us for u in units), dtype=np.int64, count=len(units)).max())
    first = int(c["first_ts"].min())
    pk0, pk1 = int(c["pk0"].sum()), int(c["pk1"].sum())
    flags = [int(c[k].sum()) for k in ("syn", "synack", "fin", "rst", "psh")]
    n_rdst = int(len(np.unique(np.ascontiguousarray(c["ip_b"]).view(np.uint64).reshape(-1, 2), axis=0)))
    n_dport = int(len(np.unique((c["proto"].astype(np.int64) << 16) | c["port_b"].astype(np.int64))))
    s0rej = int(np.isin(c["cs"], (enc.CS_CODE["S0"], enc.CS_CODE["REJ"])).sum())
    key = overflow_key(dev, len(units), n_rdst, n_dport, s0rej)
    m = {"key": key, "device_id": dev, "first_ts_us": first, "pkts": (pk0, pk1), "n_syn": flags[0],
         "n_synack": flags[1], "n_fin": flags[2], "n_rst": flags[3], "n_psh": flags[4],
         "conn_state": enc.CS_CODE["OTH"], "pay": False}
    return Unit(epoch=epoch, index=0, key=key, device_id=dev, first_ts_us=first, last_ts_us=last, pkts=(pk0, pk1),
                payload_bytes=(0, 0), frame_bytes=(fb0, fb1), s=s, pay=False, fps=False,
                flow_uid=hs.flow_uid(key, first), meta=enc.enc_metadata(m))


def cap_zero_payload(batch: EpochBatch, k_dev: int = K_DEV_DEFAULT, k_tot: int | None = K_TOT_DEFAULT
                     ) -> tuple[EpochBatch, dict]:
    """The capped batch (canonical order, indices reassigned) and a report {level, capped devices, aggregated}.
    The batch is returned unchanged (same object) when no device exceeds the cap."""
    zero = defaultdict(list)
    for u in batch.units:
        if not u.pay:
            zero[u.device_id].append(u)
    counts = {d: len(v) for d, v in zero.items()}
    c = cap_level(counts, k_dev, k_tot)
    over = {d: v for d, v in zero.items() if len(v) > c}
    info = {"level": c, "k_dev": int(k_dev), "k_tot": None if k_tot is None else int(k_tot),
            "n_zero_before": int(sum(counts.values())), "capped_devices": sorted(over), "aggregated": 0}
    if not over:
        info["n_zero_after"] = info["n_zero_before"]
        return batch, info
    drop = set()
    extra = []
    for d, v in over.items():
        v = sorted(v, key=lambda u: (u.first_ts_us, u.key))
        agg = v[c - 1:]
        drop.update(id(u) for u in agg)
        extra.append(_overflow_unit(batch.epoch, agg))
        info["aggregated"] += len(agg)
    units = [u for u in batch.units if id(u) not in drop] + extra
    units.sort(key=lambda u: (u.first_ts_us, u.key))
    for i, u in enumerate(units):
        u.index = i
    info["n_zero_after"] = sum(1 for u in units if not u.pay)
    return EpochBatch(epoch=batch.epoch, t_start_us=batch.t_start_us, units=units), info


def check_cap(store, k_dev: int = K_DEV_DEFAULT, k_tot: int | None = K_TOT_DEFAULT) -> list[dict]:
    """Compliance of a committed metadata store with the cap: ≤ k_dev zero-payload units per device, ≤ k_tot in
    total, at most one overflow record per device (findings; [] = compliant). A violation affects only metadata
    granularity / commit cost, never a statement (zero-payload units have π = 0)."""
    zero = defaultdict(int)
    ovf = defaultdict(int)
    for r in store.records:
        if not r.pay:
            zero[r.device_id] += 1
            if is_overflow_meta(r.meta):
                ovf[r.device_id] += 1
    out = []
    for d, n in sorted(zero.items()):
        if n > k_dev:
            out.append({"rule": "zero_payload_cap", "device_id": d, "units": n, "cap": int(k_dev)})
        if ovf[d] > 1:
            out.append({"rule": "overflow_records", "device_id": d, "n": ovf[d]})
    if k_tot is not None and sum(zero.values()) > k_tot:
        out.append({"rule": "zero_payload_total_cap", "units": int(sum(zero.values())), "cap": int(k_tot)})
    return out
