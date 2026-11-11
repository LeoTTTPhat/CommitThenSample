"""Independent counter source I and the ε_cell(n) tolerance (DESIGN §2.1, §6.3 counter model, V7).

Model (§6.3), applied to the per-packet stream (time order, all devices):
* gateway-side capture loss: i.i.d. drops with p_ℓ ∈ {0.001, 0.01} plus Gilbert–Elliott bursts
  (Good→Bad onset 1e-4 per packet; every packet of a burst is lost; burst length ~ Geometric with
  mean 50 packets);
* counter clock skew: offset ~ U(−50, +50) ms per (device, hour); I bins packet j into the epoch of
  ts_j + offset(device_j, hour_j), so packets near an epoch edge move to the adjacent epoch.
The counter sees every packet (it is upstream of the gateway's capture); the gateway commits the
packets it captured, binned by its own clock.

``calibrate_eps(honest)`` → EpsCell: ε_cell(n) = the (1 − FAR) quantile of |committed − counter|
per cell in log₂(n+1) bins of the counter's packet count n (sparse bins are pooled with their
neighbours; the empirical quantile of a bin with < 1/FAR cells is its maximum; the curve is made
non-decreasing in n; beyond the calibrated range it grows linearly in n). Separate curves for
packets and bytes.

Records: one Ed25519-signed (sk_I) record per (gateway, epoch) listing (device_id, packets, bytes).
"""
from __future__ import annotations

import math
import struct
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
from blake3 import blake3

from ..crypto.signing import SigningKey, verify as ed_verify

DOM_CTR = b"CTS/v1/ctr"
HOUR_US = 3_600_000_000

__all__ = ["CounterCell", "CounterRecord", "LossModel", "SkewModel", "CounterSource", "EpsCell",
           "calibrate_eps", "compare_cells", "cells_from_packets", "honest_pairs"]


@dataclass(frozen=True)
class CounterCell:
    device_id: int
    pkts: int
    bytes: int


@dataclass(frozen=True)
class CounterRecord:
    gw_id: bytes
    e: int
    cells: tuple[CounterCell, ...]
    sig: bytes = b""

    def message(self) -> bytes:
        cs = sorted(self.cells, key=lambda c: c.device_id)
        body = b"".join(struct.pack("<IQQ", c.device_id, c.pkts, c.bytes) for c in cs)
        return blake3(DOM_CTR + self.gw_id + struct.pack("<II", self.e, len(cs)) + body).digest()

    def signed(self, sk_I: SigningKey) -> "CounterRecord":
        cs = tuple(sorted(self.cells, key=lambda c: c.device_id))
        r = CounterRecord(self.gw_id, self.e, cs)
        return CounterRecord(self.gw_id, self.e, cs, sk_I.sign(r.message()))

    def verify(self, pk_I: bytes) -> bool:
        return ed_verify(pk_I, self.sig, self.message())

    def as_dict(self) -> dict[int, tuple[int, int]]:
        return {c.device_id: (c.pkts, c.bytes) for c in self.cells}


# ---------------------------------------------------------------------------- models

@dataclass(frozen=True)
class LossModel:
    p_loss: float = 0.001
    ge_onset: float = 1e-4
    ge_mean_burst: float = 50.0

    def keep_mask(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """True = captured by the gateway (packets in global time order)."""
        keep = rng.random(n) >= self.p_loss if self.p_loss > 0 else np.ones(n, dtype=bool)
        if self.ge_onset > 0 and n > 0:
            p_end = 1.0 / self.ge_mean_burst
            pos = 0
            while True:
                pos += int(rng.geometric(self.ge_onset)) - 1      # good packets before the onset
                if pos >= n:
                    break
                L = int(rng.geometric(p_end))                     # burst length ≥ 1, mean 1/p_end
                keep[pos:pos + L] = False
                pos += L
        return keep


@dataclass(frozen=True)
class SkewModel:
    max_skew_ms: float = 50.0

    def offsets_us(self, device_ids: np.ndarray, ts_us: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Per-packet counter clock offset, constant per (device, hour), U(−max, +max) ms."""
        if len(ts_us) == 0:
            return np.zeros(0, dtype=np.int64)
        hours = (np.asarray(ts_us, dtype=np.int64) // HOUR_US)
        keys = np.asarray(device_ids, dtype=np.int64) * (1 << 32) + (hours - hours.min())
        uk, inv = np.unique(keys, return_inverse=True)
        offs = rng.uniform(-self.max_skew_ms, self.max_skew_ms, size=len(uk)) * 1000.0
        return np.rint(offs[inv]).astype(np.int64)


def cells_from_packets(epochs: np.ndarray, device_ids: np.ndarray, lengths: np.ndarray,
                       mask: np.ndarray | None = None) -> dict[tuple[int, int], tuple[int, int]]:
    """{(e, device): (packets, bytes)} over the packets selected by ``mask``."""
    e = np.asarray(epochs, dtype=np.int64)
    d = np.asarray(device_ids, dtype=np.int64)
    L = np.asarray(lengths, dtype=np.int64)
    if mask is not None:
        e, d, L = e[mask], d[mask], L[mask]
    if len(e) == 0:
        return {}
    key = e * (1 << 32) + d
    uk, inv = np.unique(key, return_inverse=True)
    pk = np.bincount(inv)
    by = np.bincount(inv, weights=L).astype(np.int64)
    return {(int(k >> 32), int(k & 0xFFFFFFFF)): (int(p), int(b)) for k, p, b in zip(uk, pk, by)}


@dataclass
class CounterSource:
    """Simulates I and the gateway capture for one gateway from a per-packet stream."""
    sk_I: SigningKey | None
    gw_id: bytes
    epoch_of_us: Callable[[np.ndarray], np.ndarray]       # µs timestamps → epoch index (gateway clock)
    loss: LossModel = field(default_factory=LossModel)
    skew: SkewModel = field(default_factory=SkewModel)
    seed: int = 0

    def simulate(self, ts_us, device_ids, lengths) -> dict:
        ts = np.asarray(ts_us, dtype=np.int64)
        dev = np.asarray(device_ids, dtype=np.int64)
        L = np.asarray(lengths, dtype=np.int64)
        order = np.argsort(ts, kind="stable")
        rng = np.random.default_rng(self.seed)
        keep_sorted = self.loss.keep_mask(len(ts), rng)
        keep = np.empty_like(keep_sorted)
        keep[order] = keep_sorted
        off = self.skew.offsets_us(dev, ts, rng)
        gw_epoch = np.asarray(self.epoch_of_us(ts), dtype=np.int64)
        ctr_epoch = np.asarray(self.epoch_of_us(ts + off), dtype=np.int64)
        return {"keep": keep, "offset_us": off, "gw_epoch": gw_epoch, "ctr_epoch": ctr_epoch,
                "committed_cells": cells_from_packets(gw_epoch, dev, L, keep),
                "counter_cells": cells_from_packets(ctr_epoch, dev, L)}

    def records(self, counter_cells: dict[tuple[int, int], tuple[int, int]],
                epochs: Iterable[int]) -> dict[int, CounterRecord]:
        if self.sk_I is None:
            raise ValueError("no counter signing key")
        by_e: dict[int, list[CounterCell]] = {int(e): [] for e in epochs}
        for (e, d), (p, b) in counter_cells.items():
            if e in by_e:
                by_e[e].append(CounterCell(d, p, b))
        return {e: CounterRecord(self.gw_id, e, tuple(cs)).signed(self.sk_I) for e, cs in by_e.items()}


# ---------------------------------------------------------------------------- ε_cell(n)

def honest_pairs(committed: dict, counter: dict) -> np.ndarray:
    """Rows (n_counter_pkts, |Δpkts|, |Δbytes|) over the union of cells."""
    rows = []
    for k in set(committed) | set(counter):
        cp, cb = committed.get(k, (0, 0))
        np_, nb = counter.get(k, (0, 0))
        rows.append((np_, abs(cp - np_), abs(cb - nb)))
    return np.array(rows, dtype=np.float64).reshape(-1, 3)


def _bin_of(n) -> np.ndarray:
    return np.floor(np.log2(np.asarray(n, dtype=np.float64) + 1.0)).astype(np.int64)


@dataclass
class EpsCell:
    bins: list[int]                 # log₂(n+1) bin index (lower edge of each pooled group)
    eps_pkts: list[float]
    eps_bytes: list[float]
    far: float
    n_hi: float                     # largest calibrated counter packet count
    counts: list[int] = field(default_factory=list)

    def _lookup(self, n: float, arr: list[float]) -> float:
        b = int(_bin_of(n))
        k = int(np.searchsorted(np.asarray(self.bins), b, side="right") - 1)
        k = max(0, k)
        v = arr[k]
        if n > self.n_hi and self.n_hi > 0:
            v = v * (n / self.n_hi)               # linear extrapolation beyond the calibrated range
        return float(v)

    def pkts(self, n: float) -> float:
        return self._lookup(n, self.eps_pkts)

    def bytes(self, n: float) -> float:
        return self._lookup(n, self.eps_bytes)

    def flag(self, n_counter: float, d_pkts: float, d_bytes: float) -> bool:
        return d_pkts > self.pkts(n_counter) or d_bytes > self.bytes(n_counter)

    def to_dict(self) -> dict:
        return {"bins": self.bins, "eps_pkts": self.eps_pkts, "eps_bytes": self.eps_bytes, "far": self.far,
                "n_hi": self.n_hi, "counts": self.counts}

    @classmethod
    def from_dict(cls, d: dict) -> "EpsCell":
        return cls(list(d["bins"]), list(d["eps_pkts"]), list(d["eps_bytes"]), float(d["far"]), float(d["n_hi"]),
                   list(d.get("counts", [])))


def calibrate_eps(honest: np.ndarray, far: float = 1e-3, min_pool: int = 50) -> EpsCell:
    """honest: rows (n_counter, |Δpkts|, |Δbytes|) of honest cells (see ``honest_pairs``)."""
    honest = np.asarray(honest, dtype=np.float64).reshape(-1, 3)
    if len(honest) == 0:
        raise ValueError("no honest cells")
    b = _bin_of(honest[:, 0])
    ub = np.unique(b)
    groups: list[list[int]] = []
    cur: list[int] = []
    cnt = 0
    for k in ub:                                   # pool sparse bins with their upper neighbours
        cur.append(int(k))
        cnt += int(np.sum(b == k))
        if cnt >= min_pool:
            groups.append(cur)
            cur, cnt = [], 0
    if cur:
        if groups:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    q = 1.0 - far
    bins, ep, eb, counts = [], [], [], []
    for g in groups:
        m = np.isin(b, g)
        dp, db = honest[m, 1], honest[m, 2]
        if len(dp) * far < 1.0:                     # fewer than 1/FAR cells: sample maximum
            vp, vb = float(dp.max()), float(db.max())
        else:
            vp = float(np.quantile(dp, q, method="higher"))
            vb = float(np.quantile(db, q, method="higher"))
        bins.append(g[0])
        ep.append(vp)
        eb.append(vb)
        counts.append(int(m.sum()))
    ep = list(np.maximum.accumulate(np.asarray(ep)))
    eb = list(np.maximum.accumulate(np.asarray(eb)))
    return EpsCell(bins=bins, eps_pkts=[float(x) for x in ep], eps_bytes=[float(x) for x in eb], far=far,
                   n_hi=float(honest[:, 0].max()), counts=counts)


def compare_cells(committed: dict[int, tuple[int, int]], counter: dict[int, tuple[int, int]],
                  eps: EpsCell) -> list[dict]:
    """Per-device comparison for one epoch; returns one dict per cell (``flagged`` bool)."""
    out = []
    for d in sorted(set(committed) | set(counter)):
        cp, cb = committed.get(d, (0, 0))
        n_p, n_b = counter.get(d, (0, 0))
        dp, db = abs(cp - n_p), abs(cb - n_b)
        tol_p, tol_b = eps.pkts(n_p), eps.bytes(n_p)
        out.append({"device_id": d, "committed_pkts": cp, "counter_pkts": n_p, "committed_bytes": cb,
                    "counter_bytes": n_b, "d_pkts": dp, "d_bytes": db, "eps_pkts": tol_p, "eps_bytes": tol_b,
                    "flagged": bool(dp > tol_p or db > tol_b)})
    return out


def m_star(eps: EpsCell, n: int) -> int:
    """Largest omission m (packets) that a perfectly honest cell of size n could hide: ⌊ε_pkts(n)⌋."""
    return int(math.floor(eps.pkts(n)))
