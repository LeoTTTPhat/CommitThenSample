"""StreamTables: the on-disk unit table consumed by the MC engine and the analytic metrics (DESIGN §6.5).

Format (one directory per stream; every file written atomically; ``DONE`` marks completeness)::

    meta.json          {"format": "ctsample.streamtables", "version": 1, "name", "stream_id",
                        "n_units", "n_epochs", "n_devices",
                        "queries": [{"name", "kind": "indicator"|"count", "bbar"}],
                        "utilities": [names], "extra": {...}}
    s.npy        f8 (N,)    content cost of the unit, pcap bytes  sum(caplen + 16)
    pay.npy      b1 (N,)    payload flag (>= 1 transport payload byte)
    epoch.npy    i4 (N,)    epoch index, NON-DECREASING (units sorted by epoch, then canonical index)
    device.npy   i4 (N,)    dense LAN-device index in [0, n_devices)
    label.npy    i2 (N,)    class label (0 = benign), optional semantics
    incident.npy i4 (N,)    incident id, -1 = none
    content.npy  b1 (N,)    unit bears attack content (content-evidence miss), default = incident >= 0
    b.npy        f4 (N, Q)  eligibility bound b_iq (0 = ineligible); MUST be 0 for zero-payload units
    hit_ptr.npy  i8 (Q+1,)  CSR over queries of the HIT units (f_iq > 0):
    hit_unit.npy i8 (nnz,)    unit indices, strictly increasing within each query
    hit_f.npy    f8 (nnz,)    values f_iq in (0, b_iq]
    util_<name>.npy f8 (N,) utility column a_i of one utility model (e.g. U5 learned second moment)
    epoch_ptr.npy i8 (E+1,) derived: units of epoch e are [epoch_ptr[e], epoch_ptr[e+1])

Units with f = 0 for a query are not stored in its CSR (sparse); eligibility of non-hit units is
carried by b. Metadata estimands are exact and are not represented here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..utils.io import atomic_save_npy, atomic_write_json, is_done, mark_done

__all__ = ["StreamTables", "write_tables", "FORMAT", "VERSION"]

FORMAT = "ctsample.streamtables"
VERSION = 1
_TOL = 1e-6


@dataclass
class StreamTables:
    s: np.ndarray
    pay: np.ndarray
    epoch: np.ndarray
    device: np.ndarray
    b: np.ndarray                    # (N, Q)
    hit_ptr: np.ndarray
    hit_unit: np.ndarray
    hit_f: np.ndarray
    query_names: list
    query_kinds: list
    label: np.ndarray | None = None
    incident: np.ndarray | None = None
    content: np.ndarray | None = None
    utils: dict = field(default_factory=dict)
    name: str = "stream"
    stream_id: int = 0
    n_devices: int | None = None
    n_epochs: int | None = None
    extra: dict = field(default_factory=dict)
    path: Path | None = None

    def __post_init__(self):
        self.b = np.asarray(self.b)
        if self.b.ndim == 1:
            self.b = self.b[:, None]
        if self.n_epochs is None:
            self.n_epochs = int(self.epoch[-1]) + 1 if len(self.epoch) else 0
        if self.n_devices is None:
            self.n_devices = int(np.max(self.device)) + 1 if len(self.device) else 0
        self._eptr = None

    # ---- shape --------------------------------------------------------------------------- #
    @property
    def n_units(self) -> int:
        return int(len(self.s))

    @property
    def n_queries(self) -> int:
        return int(self.b.shape[1])

    @property
    def epoch_ptr(self) -> np.ndarray:
        if self._eptr is None:
            self._eptr = np.searchsorted(np.asarray(self.epoch), np.arange(self.n_epochs + 1), side="left").astype(np.int64)
        return self._eptr

    def query_index(self, q) -> int:
        if isinstance(q, (int, np.integer)):
            if not 0 <= int(q) < self.n_queries:
                raise IndexError(q)
            return int(q)
        return self.query_names.index(q)

    def hits(self, q):
        """(unit indices, f values) of query q's hit units."""
        q = self.query_index(q)
        a, z = int(self.hit_ptr[q]), int(self.hit_ptr[q + 1])
        return np.asarray(self.hit_unit[a:z]), np.asarray(self.hit_f[a:z], dtype=float)

    def b_col(self, q) -> np.ndarray:
        return np.asarray(self.b[:, self.query_index(q)], dtype=float)

    def f_dense(self, q) -> np.ndarray:
        f = np.zeros(self.n_units)
        u, v = self.hits(q)
        f[u] = v
        return f

    def util(self, name) -> np.ndarray:
        if name is None:
            return np.ones(self.n_units)
        return np.asarray(self.utils[name], dtype=float)

    def query_total(self, q) -> float:
        return float(self.hits(q)[1].sum())

    def bbar(self, q) -> float:
        bq = self.b_col(q)
        return float(bq.max()) if bq.size else 0.0

    def incident_ids(self) -> np.ndarray:
        if self.incident is None:
            return np.zeros(0, dtype=np.int64)
        inc = np.asarray(self.incident)
        return np.unique(inc[inc >= 0])

    # ---- validation ---------------------------------------------------------------------- #
    def validate(self) -> None:
        N = self.n_units
        for nm in ("pay", "epoch", "device"):
            if len(getattr(self, nm)) != N:
                raise ValueError(f"column {nm} has wrong length")
        ep = np.asarray(self.epoch)
        if N and np.any(np.diff(ep) < 0):
            raise ValueError("epoch must be non-decreasing (units sorted by epoch)")
        if N and ep[0] < 0:
            raise ValueError("negative epoch")
        s = np.asarray(self.s)
        if np.any(~(s >= 0)):
            raise ValueError("s must be >= 0")
        if self.b.shape[0] != N:
            raise ValueError("b must have shape (N, Q)")
        b = np.asarray(self.b)
        if np.any(b < 0):
            raise ValueError("b must be >= 0")
        pay = np.asarray(self.pay, dtype=bool)
        if np.any(b[~pay] > 0):
            raise ValueError("zero-payload units must have b = 0 (nothing to sample)")
        if len(self.hit_ptr) != self.n_queries + 1:
            raise ValueError("hit_ptr must have Q+1 entries")
        if len(self.query_names) != self.n_queries or len(self.query_kinds) != self.n_queries:
            raise ValueError("query_names/kinds must have Q entries")
        for q in range(self.n_queries):
            u, f = self.hits(q)
            if len(u) and (np.any(np.diff(u) <= 0) or u[0] < 0 or u[-1] >= N):
                raise ValueError(f"query {q}: hit units must be strictly increasing valid indices")
            if np.any(~(f > 0)):
                raise ValueError(f"query {q}: stored hits must have f > 0")
            if np.any(f > b[u, q] * (1 + _TOL) + _TOL):
                raise ValueError(f"query {q}: f exceeds the eligibility bound b")
            if self.query_kinds[q] == "indicator" and np.any(np.abs(f - 1.0) > _TOL):
                raise ValueError(f"query {q}: indicator query with f != 1")
        for k, v in self.utils.items():
            if len(v) != N:
                raise ValueError(f"utility {k} has wrong length")

    # ---- I/O ----------------------------------------------------------------------------- #
    @classmethod
    def load(cls, path, mmap: bool = True) -> "StreamTables":
        path = Path(path)
        if not is_done(path):
            raise FileNotFoundError(f"{path} is not a complete StreamTables directory (no DONE)")
        meta = json.loads((path / "meta.json").read_text())
        if meta.get("format") != FORMAT:
            raise ValueError("not a StreamTables directory")
        mm = "r" if mmap else None
        ld = lambda n: np.load(path / f"{n}.npy", mmap_mode=mm)  # noqa: E731
        opt = lambda n: ld(n) if (path / f"{n}.npy").exists() else None  # noqa: E731
        utils = {u: ld(f"util_{u}") for u in meta.get("utilities", [])}
        t = cls(s=ld("s"), pay=ld("pay"), epoch=ld("epoch"), device=ld("device"), b=ld("b"),
                hit_ptr=np.load(path / "hit_ptr.npy"), hit_unit=ld("hit_unit"), hit_f=ld("hit_f"),
                query_names=[q["name"] for q in meta["queries"]],
                query_kinds=[q["kind"] for q in meta["queries"]],
                label=opt("label"), incident=opt("incident"), content=opt("content"), utils=utils,
                name=meta.get("name", path.name), stream_id=int(meta.get("stream_id", 0)),
                n_devices=int(meta["n_devices"]), n_epochs=int(meta["n_epochs"]),
                extra=meta.get("extra", {}), path=path)
        if (path / "epoch_ptr.npy").exists():
            t._eptr = np.load(path / "epoch_ptr.npy")
        return t

    def save(self, path) -> Path:
        return write_tables(path, s=self.s, pay=self.pay, epoch=self.epoch, device=self.device, b=self.b,
                            hits=[self.hits(q) for q in range(self.n_queries)],
                            query_names=self.query_names, query_kinds=self.query_kinds, label=self.label,
                            incident=self.incident, content=self.content, utils=self.utils, name=self.name,
                            stream_id=self.stream_id, n_devices=self.n_devices, n_epochs=self.n_epochs,
                            extra=self.extra)

    @classmethod
    def from_arrays(cls, *, s, pay, epoch, device, b, hits, query_names=None, query_kinds=None, **kw):
        """In-memory constructor. ``hits``: list over queries of (unit_idx, f) pairs."""
        b = np.asarray(b, dtype=np.float32)
        if b.ndim == 1:
            b = b[:, None]
        Q = b.shape[1]
        ptr, uu, ff = _csr(hits, Q)
        t = cls(s=np.asarray(s, dtype=float), pay=np.asarray(pay, dtype=bool), epoch=np.asarray(epoch, dtype=np.int32),
                device=np.asarray(device, dtype=np.int32), b=b, hit_ptr=ptr, hit_unit=uu, hit_f=ff,
                query_names=list(query_names or [f"q{i}" for i in range(Q)]),
                query_kinds=list(query_kinds or ["indicator"] * Q), **kw)
        t.validate()
        return t


def _csr(hits, Q):
    if len(hits) != Q:
        raise ValueError("hits must have one (units, f) entry per query")
    ptr = np.zeros(Q + 1, dtype=np.int64)
    us, fs = [], []
    for q, (u, f) in enumerate(hits):
        u = np.asarray(u, dtype=np.int64)
        f = np.asarray(f, dtype=float)
        o = np.argsort(u, kind="stable")
        u, f = u[o], f[o]
        keep = f > 0
        us.append(u[keep])
        fs.append(f[keep])
        ptr[q + 1] = ptr[q] + keep.sum()
    return ptr, (np.concatenate(us) if us else np.zeros(0, np.int64)), (np.concatenate(fs) if fs else np.zeros(0))


def write_tables(path, *, s, pay, epoch, device, b, hits, query_names, query_kinds=None, label=None,
                 incident=None, content=None, utils=None, name=None, stream_id=0, n_devices=None,
                 n_epochs=None, extra=None) -> Path:
    """Validate and write a StreamTables directory atomically (DONE written last)."""
    t = StreamTables.from_arrays(s=s, pay=pay, epoch=epoch, device=device, b=b, hits=hits,
                                 query_names=query_names, query_kinds=query_kinds, label=label,
                                 incident=incident, content=content, utils=dict(utils or {}),
                                 name=name or Path(path).name, stream_id=stream_id, n_devices=n_devices,
                                 n_epochs=n_epochs, extra=dict(extra or {}))
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    done = path / "DONE"
    if done.exists():
        done.unlink()
    N = t.n_units
    atomic_save_npy(path / "s.npy", np.asarray(t.s, dtype=np.float64))
    atomic_save_npy(path / "pay.npy", np.asarray(t.pay, dtype=bool))
    atomic_save_npy(path / "epoch.npy", np.asarray(t.epoch, dtype=np.int32))
    atomic_save_npy(path / "device.npy", np.asarray(t.device, dtype=np.int32))
    atomic_save_npy(path / "b.npy", np.asarray(t.b, dtype=np.float32))
    atomic_save_npy(path / "hit_ptr.npy", t.hit_ptr)
    atomic_save_npy(path / "hit_unit.npy", t.hit_unit)
    atomic_save_npy(path / "hit_f.npy", t.hit_f)
    atomic_save_npy(path / "label.npy", np.zeros(N, np.int16) if t.label is None else np.asarray(t.label, dtype=np.int16))
    inc = np.full(N, -1, np.int32) if t.incident is None else np.asarray(t.incident, dtype=np.int32)
    atomic_save_npy(path / "incident.npy", inc)
    atomic_save_npy(path / "content.npy", (inc >= 0) if t.content is None else np.asarray(t.content, dtype=bool))
    for k, v in t.utils.items():
        atomic_save_npy(path / f"util_{k}.npy", np.asarray(v, dtype=np.float64))
    atomic_save_npy(path / "epoch_ptr.npy", t.epoch_ptr)
    meta = {"format": FORMAT, "version": VERSION, "name": t.name, "stream_id": int(stream_id),
            "n_units": N, "n_epochs": int(t.n_epochs), "n_devices": int(t.n_devices),
            "queries": [{"name": n, "kind": k, "bbar": t.bbar(i)} for i, (n, k) in enumerate(zip(t.query_names, t.query_kinds))],
            "utilities": sorted(t.utils), "extra": t.extra}
    atomic_write_json(path / "meta.json", meta)
    mark_done(path, {"n_units": N})
    return path
