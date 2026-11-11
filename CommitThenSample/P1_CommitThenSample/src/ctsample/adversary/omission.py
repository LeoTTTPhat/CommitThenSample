"""A8 — pre-commit omission of attack packets (A_rt; DESIGN §6.3).

The compromised capture point drops m packets of the attacker's device in chosen epochs BEFORE
commit, so everything it commits is self-consistent; only the independent counters (V7, ε_cell(n))
can notice the volume gap.

* ``omit_packets_in_batches`` mutates committed units (payload frames removed, units rebuilt through
  a ``rebuild`` callback that re-encodes enc(m_i); default: the synthetic codec) — end-to-end use.
* ``omit_from_cells`` / ``detection_curve`` work on (device, epoch) cell counts — the E5 curve
  P(flag) vs m and m*(n) = largest m with P(flag) < 0.5.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence

import numpy as np

from ..crypto import hashing as hs
from ..protocol.counters import EpsCell
from ..protocol.synth import SyntheticMetaCodec
from ..units.batch import EpochBatch, Frame, Unit

__all__ = ["omit_packets_in_batches", "rebuild_synthetic_unit", "omit_from_cells", "detection_curve", "m_star_curve"]


def rebuild_synthetic_unit(u: Unit, frames: list[Frame]) -> Unit | None:
    """Re-derive the committed fields of ``u`` from a reduced frame list (synthetic meta layout)."""
    if not frames:
        return None
    pk, fb = [0, 0], [0, 0]
    for f in frames:
        pk[f.dir] += 1
        fb[f.dir] += f.caplen
    s = sum(f.caplen + 16 for f in frames)
    first = frames[0].ts_us
    meta = SyntheticMetaCodec.encode(u.device_id, pk, fb, s, first, u.pay, u.fps, u.meta[46:])
    return Unit(epoch=u.epoch, index=u.index, key=u.key, device_id=u.device_id, first_ts_us=first,
                last_ts_us=frames[-1].ts_us, pkts=(pk[0], pk[1]), payload_bytes=u.payload_bytes,
                frame_bytes=(fb[0], fb[1]), s=s, pay=u.pay, fps=u.fps, flow_uid=hs.flow_uid(u.key, first), meta=meta,
                frames=frames)


def omit_packets_in_batches(batches: Sequence[EpochBatch], device: int, epochs: set[int], m: int,
                            rng: random.Random,
                            rebuild: Callable[[Unit, list[Frame]], Unit | None] = rebuild_synthetic_unit,
                            include_zero_payload: bool = False) -> dict:
    """Drop up to m packets of ``device`` per chosen epoch from its payload units (frames removed,
    units rebuilt; emptied units removed); with ``include_zero_payload`` whole zero-payload units of
    the device are dropped too while the budget m lasts. Re-indexes the epoch canonically.
    Must run before the gateway commits. Returns {epoch: packets removed}."""
    removed = {}
    for b in batches:
        if b.epoch not in epochs:
            continue
        if any(u.leaf for u in b.units):
            raise ValueError("A8 must run before the gateway computes leaves")
        cand = [(k, j) for k, u in enumerate(b.units) if u.device_id == device and u.pay and u.frames
                for j in range(len(u.frames))]
        drop = set(rng.sample(cand, min(m, len(cand))))
        budget = m - len(drop)
        n_removed = len(drop)
        new_units = []
        for k, u in enumerate(b.units):
            if u.device_id == device and u.pay and u.frames:
                fr = [f for j, f in enumerate(u.frames) if (k, j) not in drop]
                if len(fr) != len(u.frames):
                    u = rebuild(u, fr)
                    if u is None:
                        continue
            elif include_zero_payload and u.device_id == device and not u.pay and u.n_pkts <= budget:
                budget -= u.n_pkts
                n_removed += u.n_pkts
                continue
            new_units.append(u)
        new_units.sort(key=lambda x: (x.first_ts_us, x.key))
        for i, u in enumerate(new_units):
            u.index = i
        b.units = new_units
        removed[b.epoch] = n_removed
    return removed


def omit_from_cells(committed: dict, device: int, epochs: set[int], m: int, bytes_per_pkt: float) -> dict:
    """Cell-level A8: subtract m packets (and m·bytes_per_pkt bytes) from the chosen cells."""
    out = dict(committed)
    for e in epochs:
        p, b = out.get((e, device), (0, 0))
        mm = min(m, p)
        out[(e, device)] = (p - mm, max(0, int(round(b - mm * bytes_per_pkt))))
    return out


def detection_curve(eps: EpsCell, cells: np.ndarray, m_values: Sequence[int], mean_pkt_bytes: float) -> dict:
    """P(flag) vs m over honest cells. ``cells`` rows: (counter_pkts, committed_pkts, counter_bytes,
    committed_bytes). Omitting m packets moves the committed counts down by m (and m·mean bytes)."""
    cells = np.asarray(cells, dtype=np.float64).reshape(-1, 4)
    out = {}
    for m in m_values:
        n_p, c_p, n_b, c_b = cells[:, 0], cells[:, 1], cells[:, 2], cells[:, 3]
        mm = np.minimum(m, c_p)
        dp = np.abs(c_p - mm - n_p)
        db = np.abs(c_b - mm * mean_pkt_bytes - n_b)
        tol_p = np.array([eps.pkts(x) for x in n_p])
        tol_b = np.array([eps.bytes(x) for x in n_p])
        out[int(m)] = float(np.mean((dp > tol_p) | (db > tol_b)))
    return out


def m_star_curve(eps: EpsCell, cells: np.ndarray, m_values: Sequence[int], mean_pkt_bytes: float,
                 n_bins: Sequence[tuple[int, int]]) -> dict:
    """m*(n) per counter-size bin [lo, hi): the largest m with P(flag) < 0.5."""
    cells = np.asarray(cells, dtype=np.float64).reshape(-1, 4)
    res = {}
    for lo, hi in n_bins:
        sel = (cells[:, 0] >= lo) & (cells[:, 0] < hi)
        if not np.any(sel):
            continue
        curve = detection_curve(eps, cells[sel], m_values, mean_pkt_bytes)
        m_star = None
        for m in sorted(curve):              # largest m before P(flag) first reaches 0.5
            if curve[m] >= 0.5:
                break
            m_star = m
        res[(lo, hi)] = {"m_star": m_star, "curve": curve, "n_cells": int(sel.sum())}
    return res
