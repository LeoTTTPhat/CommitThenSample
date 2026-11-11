"""Per-epoch metadata store, retention store and size accounting (DESIGN §3.2, §3.7; E6).

Metadata store of epoch e (one record per committed unit, index order):
    enc(m_i) for every unit, root^pkt_i for payload units  (DESIGN §3.2), plus a small block of
    *derived columns* (pay, fps, device_id, pkts/dir, frame bytes/dir, s, first_ts_us) that the
    protocol layer needs and cannot read from enc(m_i) because enc(m_i) is opaque here
    (units/encode.py owns the layout). The derived columns are inside the canonical serialisation,
    so they are committed by H(metadata store_e) in the signed header; a verifier with a meta
    codec additionally checks them against enc(m_i) (verifier V4 'meta_consistency').

Canonical serialisation of the metadata records (all little-endian):
    meta_e = b"CTS/v1/store" ‖ u32 e ‖ u32 n ‖ n × record
    record = u8 flags (bit0 pay, bit1 fps) ‖ u32 device_id ‖ u32 pkts0 ‖ u32 pkts1 ‖
             u64 frame_bytes0 ‖ u64 frame_bytes1 ‖ u64 s ‖ u64 first_ts_us ‖
             u32 len(meta) ‖ meta ‖ [32 B root_pkt iff pay]

Committed per-unit thresholds (rev1, review blocking issue 1): every payload unit's selection
threshold τ_i ∈ [1, 2^64] is part of its committed record, in a separate column so that the column can
be redacted (prime bundles, protocol/prime.py) without breaking the commitment:
    tau_e  = b"CTS/v1/tau" ‖ u32 e ‖ u32 n⁺ ‖ n⁺ × u64 (τ_i − 1)      (payload units, index order;
             zero-payload units have τ_i = 0 and no entry; τ_i − 1 so that τ_i = 2^64, π_i = 1, fits u64)
    H(metadata store_e) = BLAKE3("CTS/v1/store2" ‖ BLAKE3(meta_e) ‖ BLAKE3(tau_e))
The gateway computes τ_i BEFORE the header and commits them through H(store_e) in the signed and
witness-cosigned header, i.e. before β of the designated round: the verifier recomputes Z_i = [u_i < τ_i]
and checks the floor τ_i ≥ τ^φ_i from the epoch's OWN store (no rolling features, no prime bundle);
recomputing τ_i through the policy (rolling features over 120 epochs) is only a policy-compliance check.
A store without a τ column (``taus is None`` and no ``tau_digest``; the rev0 format) keeps the old digest
H(store_e) = BLAKE3(meta_e); the verifier accepts such stores only in its legacy mode.
A *redacted* store (``redacted()``) carries BLAKE3(tau_e) instead of the column (prime bundles: the
thresholds of non-requested epochs would reveal their design state, Thm. 2(d)).
Full serialisation (``serialize()``): meta_e ‖ [tau_e | b"CTS/v1/tdig" ‖ 32 B BLAKE3(tau_e)].

Leaves are recomputed from the store using the explicit pay flag: unit_leaf(meta, root_pkt) if
pay else leaf0(meta).

Retained item (Z_i = 1 units): (e, i, AEAD ciphertext of the frames as pcap records, per-packet
frame-hash list ℓ_i·). Plaintext = b"CTSR" ‖ u32 n ‖ n direction bytes ‖ n pcap records
(u32 ts_sec, u32 ts_usec, u32 incl_len, u32 orig_len, frame). Directions are needed to recompute
ℓ_ij and are not part of a pcap record.

Size accounting (E6): ``MetadataStore.sizes()`` reports the DESIGN definition (Σ|enc(m_i)| + 32 B
per payload unit), the full canonical serialisation, and parquet-zstd sizes of both.
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field

from blake3 import blake3

from ..crypto import hashing as hs
from ..crypto.aead import TAG_LEN, EpochCipher
from ..units.batch import EpochBatch, Frame, Unit

DOM_STORE = b"CTS/v1/store"
DOM_STORE2 = b"CTS/v1/store2"
DOM_TAU = b"CTS/v1/tau"
DOM_TAUD = b"CTS/v1/tdig"          # NOT an extension of DOM_TAU (the u32 e that follows DOM_TAU could mimic it)
TWO64 = 1 << 64
_HDR = struct.Struct("<II")
_REC = struct.Struct("<BIIIQQQQI")
_PCAP = struct.Struct("<IIII")

__all__ = ["UnitRecord", "MetadataStore", "RetainedItem", "RetentionStore", "frames_to_plaintext",
           "plaintext_to_frames", "prepare_unit", "prepare_batch", "EncodeMetaCodec", "parquet_sizes",
           "tau_column_bytes", "DOM_TAU", "DOM_STORE2"]


def tau_column_bytes(e: int, pay, taus) -> bytes:
    """tau_e = "CTS/v1/tau" ‖ u32 e ‖ u32 n⁺ ‖ n⁺ × u64 (τ_i − 1) over the payload units (index order).

    Raises ValueError unless every payload τ_i is an integer in [1, 2^64] and every zero-payload τ_i is 0."""
    if len(pay) != len(taus):
        raise ValueError("one threshold per record required")
    vals = []
    for i, (p, t) in enumerate(zip(pay, taus)):
        t = int(t)
        if p:
            if not 1 <= t <= TWO64:
                raise ValueError(f"payload unit {i}: committed threshold {t} outside [1, 2^64]")
            vals.append(t - 1)
        elif t != 0:
            raise ValueError(f"zero-payload unit {i}: threshold must be 0, got {t}")
    return DOM_TAU + _HDR.pack(e, len(vals)) + struct.pack(f"<{len(vals)}Q", *vals)


# ---------------------------------------------------------------------------- unit preparation

def prepare_unit(u: Unit) -> Unit:
    """Fill root_pkt / pkt_leaves / leaf (crypto.hashing) where missing; check precomputed values.

    * payload unit without pkt_leaves: ℓ_ij from frames, root^pkt = Merkle(ℓ);
    * given pkt_leaves: root^pkt must equal Merkle(pkt_leaves) (cheap check);
    * the leaf is always recomputed from (meta, root^pkt) and must equal a precomputed one.
    Frame hashes of precomputed ℓ_ij are checked at retention time (RetentionStore.retain).
    Raises ValueError on any inconsistency. Returns u (mutated)."""
    if u.pay:
        if u.pkt_leaves is None:
            if u.frames is None:
                if u.root_pkt is None:
                    raise ValueError(f"payload unit {u.index} of epoch {u.epoch} has neither frames nor root_pkt")
            else:
                u.pkt_leaves = hs.frame_leaves([f.ts_us for f in u.frames], [f.dir for f in u.frames],
                                               [f.caplen for f in u.frames], [f.data for f in u.frames])
        if u.pkt_leaves is not None:
            r = hs.merkle_root(u.pkt_leaves)
            if u.root_pkt is not None and u.root_pkt != r:
                raise ValueError(f"unit ({u.epoch}, {u.index}): root_pkt != Merkle(pkt_leaves)")
            u.root_pkt = r
        leaf = hs.unit_leaf(u.meta, u.root_pkt)
    else:
        leaf = hs.leaf0(u.meta)
    if u.leaf and u.leaf != leaf:
        raise ValueError(f"unit ({u.epoch}, {u.index}): precomputed leaf != recomputed leaf")
    u.leaf = leaf
    return u


def prepare_batch(batch: EpochBatch) -> EpochBatch:
    for k, u in enumerate(batch.units):
        if u.index != k:
            raise ValueError(f"unit index {u.index} at position {k}: batch must be in canonical order")
        if u.epoch != batch.epoch:
            raise ValueError("unit epoch differs from batch epoch")
        prepare_unit(u)
    return batch


# ---------------------------------------------------------------------------- metadata store

@dataclass(frozen=True)
class UnitRecord:
    meta: bytes
    pay: bool
    root_pkt: bytes | None
    device_id: int
    pkts: tuple[int, int]
    frame_bytes: tuple[int, int]
    s: int
    first_ts_us: int
    fps: bool = False

    @classmethod
    def from_unit(cls, u: Unit) -> "UnitRecord":
        return cls(meta=u.meta, pay=bool(u.pay), root_pkt=u.root_pkt if u.pay else None, device_id=u.device_id,
                   pkts=(int(u.pkts[0]), int(u.pkts[1])), frame_bytes=(int(u.frame_bytes[0]), int(u.frame_bytes[1])),
                   s=int(u.s), first_ts_us=int(u.first_ts_us), fps=bool(u.fps))

    def leaf(self) -> bytes:
        if self.pay:
            if self.root_pkt is None or len(self.root_pkt) != 32:
                raise ValueError("payload record without a 32-byte root_pkt")
            return hs.unit_leaf(self.meta, self.root_pkt)
        return hs.leaf0(self.meta)

    def serialize(self) -> bytes:
        flags = (1 if self.pay else 0) | (2 if self.fps else 0)
        out = _REC.pack(flags, self.device_id, self.pkts[0], self.pkts[1], self.frame_bytes[0],
                        self.frame_bytes[1], self.s, self.first_ts_us, len(self.meta)) + self.meta
        if self.pay:
            if self.root_pkt is None or len(self.root_pkt) != 32:
                raise ValueError("payload record without a 32-byte root_pkt")
            out += self.root_pkt
        return out

    @property
    def n_pkts(self) -> int:
        return self.pkts[0] + self.pkts[1]

    @property
    def n_bytes(self) -> int:
        return self.frame_bytes[0] + self.frame_bytes[1]


@dataclass
class MetadataStore:
    """Committed metadata of one epoch: the unit records and (rev1) the committed thresholds τ_i.

    ``taus``: one int per record (payload τ_i ∈ [1, 2^64], zero-payload 0) or None; ``tau_digest``:
    BLAKE3(tau_e) of a redacted store (prime bundle) whose column is withheld. Both None = rev0 store."""
    e: int
    records: list[UnitRecord]
    taus: list[int] | None = None
    tau_digest: bytes | None = None

    def __post_init__(self):
        if self.taus is not None:
            self.taus = [int(t) for t in self.taus]
            if len(self.taus) != len(self.records):
                raise ValueError("one committed threshold per record required")
        if self.tau_digest is not None and len(self.tau_digest) != 32:
            raise ValueError("tau_digest must be 32 bytes")

    @classmethod
    def from_batch(cls, batch: EpochBatch, taus=None) -> "MetadataStore":
        st = cls(batch.epoch, [UnitRecord.from_unit(u) for u in batch.units], taus=taus)
        if taus is not None:
            st.tau_bytes()                                  # validate the ranges at commit time
        return st

    @property
    def n(self) -> int:
        return len(self.records)

    @property
    def n_pay(self) -> int:
        return sum(1 for r in self.records if r.pay)

    @property
    def committed(self) -> bool:
        """True iff the store commits per-unit thresholds (rev1 format; full or redacted)."""
        return self.taus is not None or self.tau_digest is not None

    def meta_bytes(self) -> bytes:
        """meta_e: the record serialisation (= the whole rev0 serialisation)."""
        parts = [DOM_STORE, _HDR.pack(self.e, len(self.records))]
        parts.extend(r.serialize() for r in self.records)
        return b"".join(parts)

    def tau_bytes(self) -> bytes:
        if self.taus is None:
            raise ValueError("store has no threshold column (redacted or rev0)")
        return tau_column_bytes(self.e, [r.pay for r in self.records], self.taus)

    def tau_root(self) -> bytes:
        """BLAKE3(tau_e) (from the column, or the redacted store's digest)."""
        if self.taus is not None:
            return blake3(self.tau_bytes()).digest()
        if self.tau_digest is not None:
            return self.tau_digest
        raise ValueError("store has no threshold commitment (rev0 format)")

    def serialize(self) -> bytes:
        """meta_e ‖ tau_e (full), meta_e ‖ "CTS/v1/tdig" ‖ BLAKE3(tau_e) (redacted), or meta_e (rev0)."""
        if self.taus is not None:
            return self.meta_bytes() + self.tau_bytes()
        if self.tau_digest is not None:
            return self.meta_bytes() + DOM_TAUD + self.tau_digest
        return self.meta_bytes()

    @classmethod
    def deserialize(cls, b: bytes) -> "MetadataStore":
        if not b.startswith(DOM_STORE):
            raise ValueError("not a CTS metadata store")
        off = len(DOM_STORE)
        e, n = _HDR.unpack_from(b, off)
        off += _HDR.size
        recs = []
        for _ in range(n):
            flags, dev, p0, p1, b0, b1, s, fts, ml = _REC.unpack_from(b, off)
            off += _REC.size
            meta = bytes(b[off:off + ml])
            off += ml
            root = None
            if flags & 1:
                root = bytes(b[off:off + 32])
                off += 32
            recs.append(UnitRecord(meta, bool(flags & 1), root, dev, (p0, p1), (b0, b1), s, fts, bool(flags & 2)))
        taus, tdig = None, None
        rest = bytes(b[off:])
        if rest.startswith(DOM_TAUD):
            tdig = rest[len(DOM_TAUD):]
            if len(tdig) != 32:
                raise ValueError("bad redacted threshold digest")
            off = len(b)
        elif rest.startswith(DOM_TAU):
            o = off + len(DOM_TAU)
            e2, npay = _HDR.unpack_from(b, o)
            o += _HDR.size
            if e2 != e or npay != sum(1 for r in recs if r.pay):
                raise ValueError("threshold column does not match the records")
            vals = struct.unpack_from(f"<{npay}Q", b, o)
            o += 8 * npay
            it = iter(vals)
            taus = [next(it) + 1 if r.pay else 0 for r in recs]
            off = o
        if off != len(b):
            raise ValueError("trailing bytes in metadata store")
        return cls(e, recs, taus=taus, tau_digest=tdig)

    def digest(self) -> bytes:
        """H(metadata store_e) = BLAKE3("CTS/v1/store2" ‖ BLAKE3(meta_e) ‖ BLAKE3(tau_e)) (rev1);
        BLAKE3(meta_e) for a rev0 store without a threshold commitment."""
        hm = blake3(self.meta_bytes()).digest()
        if not self.committed:
            return hm
        return blake3(DOM_STORE2 + hm + self.tau_root()).digest()

    def redacted(self) -> "MetadataStore":
        """Copy with the threshold column replaced by its digest (prime bundles; same H(store_e))."""
        if not self.committed:
            return MetadataStore(self.e, list(self.records))
        return MetadataStore(self.e, list(self.records), taus=None, tau_digest=self.tau_root())

    def pi(self) -> list[float]:
        """π_i = τ_i 2^-64 of the committed thresholds (floats, for reports only)."""
        if self.taus is None:
            raise ValueError("no threshold column")
        return [t / TWO64 for t in self.taus]

    def leaves(self) -> list[bytes]:
        return [r.leaf() for r in self.records]

    def root(self) -> bytes:
        return hs.merkle_root(self.leaves())

    def totals(self) -> dict:
        return {"n": self.n, "n_pay": self.n_pay, "sum_s": sum(r.s for r in self.records),
                "sum_s_pay": sum(r.s for r in self.records if r.pay)}

    def cells(self) -> dict[int, tuple[int, int]]:
        """Committed per-device (packets, frame bytes) of this epoch (V7)."""
        out: dict[int, list[int]] = {}
        for r in self.records:
            c = out.setdefault(r.device_id, [0, 0])
            c[0] += r.n_pkts
            c[1] += r.n_bytes
        return {k: (v[0], v[1]) for k, v in out.items()}

    def to_batch(self, t_start_us: int = 0, codec=None) -> EpochBatch:
        """Units reconstructed from committed data only (verifier side; frames absent).

        The committed derived columns give device_id, pkts, frame_bytes, s, pay, fps, first_ts_us.
        key, last_ts_us, payload_bytes and flow_uid are not store columns: they are filled from
        ``codec.decode(meta)`` when a codec is given and knows them, else left empty (b"", first_ts,
        (0, 0), b""). A policy should read features from ``meta`` (decode it) to be safe either way."""
        units = []
        for i, r in enumerate(self.records):
            extra = {}
            if codec is not None:
                try:
                    extra = codec.decode(r.meta, r.pay)
                except Exception:                       # noqa: BLE001 — reported by V4 meta_consistency
                    extra = {}
            pb = extra.get("payload_bytes", (0, 0))
            units.append(Unit(epoch=self.e, index=i, key=extra.get("key", b""), device_id=r.device_id,
                              first_ts_us=r.first_ts_us, last_ts_us=extra.get("last_ts_us", r.first_ts_us),
                              pkts=r.pkts, payload_bytes=(int(pb[0]), int(pb[1])), frame_bytes=r.frame_bytes,
                              s=r.s, pay=r.pay, fps=r.fps, flow_uid=extra.get("flow_uid", b""), meta=r.meta,
                              leaf=r.leaf(), root_pkt=r.root_pkt, pkt_leaves=None, frames=None))
        return EpochBatch(epoch=self.e, t_start_us=t_start_us, units=units)

    # ---------------------------------------------------------------- size accounting (E6)
    def sizes(self, parquet: bool = True) -> dict:
        raw_design = sum(len(r.meta) + (32 if r.pay else 0) for r in self.records)
        out = {"n": self.n, "n_pay": self.n_pay, "raw_design_bytes": raw_design,
               "raw_full_bytes": len(self.serialize())}
        if parquet:
            out.update(parquet_sizes([self]))
        return out


def parquet_sizes(stores: list[MetadataStore]) -> dict:
    """parquet-zstd sizes of (a) the DESIGN columns (meta, root_pkt), (b) all store columns and (rev1, when every
    store commits its thresholds) (c) the DESIGN columns + the committed threshold column (u64 τ_i − 1, payload
    units; null for zero-payload units)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = {"e": [], "meta": [], "root_pkt": [], "pay": [], "fps": [], "device_id": [], "pkts0": [], "pkts1": [],
            "fb0": [], "fb1": [], "s": [], "first_ts_us": []}
    with_tau = bool(stores) and all(st.taus is not None for st in stores)
    tau_col = []
    for st in stores:
        if with_tau:
            tau_col.extend((t - 1) if r.pay else None for t, r in zip(st.taus, st.records))
        for r in st.records:
            cols["e"].append(st.e)
            cols["meta"].append(r.meta)
            cols["root_pkt"].append(r.root_pkt)
            cols["pay"].append(r.pay)
            cols["fps"].append(r.fps)
            cols["device_id"].append(r.device_id)
            cols["pkts0"].append(r.pkts[0])
            cols["pkts1"].append(r.pkts[1])
            cols["fb0"].append(r.frame_bytes[0])
            cols["fb1"].append(r.frame_bytes[1])
            cols["s"].append(r.s)
            cols["first_ts_us"].append(r.first_ts_us)
    schema = pa.schema([("e", pa.uint32()), ("meta", pa.binary()), ("root_pkt", pa.binary(32)), ("pay", pa.bool_()),
                        ("fps", pa.bool_()), ("device_id", pa.uint32()), ("pkts0", pa.uint32()), ("pkts1", pa.uint32()),
                        ("fb0", pa.uint64()), ("fb1", pa.uint64()), ("s", pa.uint64()), ("first_ts_us", pa.uint64())])
    t = pa.table(cols, schema=schema)

    def _size(tbl) -> int:
        buf = io.BytesIO()
        pq.write_table(tbl, buf, compression="zstd")
        return buf.tell()

    out = {"parquet_zstd_design_bytes": _size(t.select(["meta", "root_pkt"])),
           "parquet_zstd_full_bytes": _size(t)}
    if with_tau:
        td = t.select(["meta", "root_pkt"]).append_column("tau_m1", pa.array(tau_col, type=pa.uint64()))
        out["parquet_zstd_design_tau_bytes"] = _size(td)
        out["parquet_zstd_full_tau_bytes"] = _size(t.append_column("tau_m1", pa.array(tau_col, type=pa.uint64())))
    return out


# ---------------------------------------------------------------------------- retention

def frames_to_plaintext(frames: list[Frame]) -> bytes:
    parts = [b"CTSR", struct.pack("<I", len(frames)), bytes(f.dir for f in frames)]
    for f in frames:
        if f.caplen != len(f.data):
            raise ValueError("caplen must equal len(frame data)")
        parts.append(_PCAP.pack(f.ts_us // 1_000_000, f.ts_us % 1_000_000, f.caplen, f.wirelen))
        parts.append(f.data)
    return b"".join(parts)


def plaintext_to_frames(pt: bytes) -> list[Frame]:
    if not pt.startswith(b"CTSR"):
        raise ValueError("bad retained plaintext")
    (n,) = struct.unpack_from("<I", pt, 4)
    dirs = pt[8:8 + n]
    off = 8 + n
    out = []
    for j in range(n):
        sec, usec, incl, orig = _PCAP.unpack_from(pt, off)
        off += _PCAP.size
        data = bytes(pt[off:off + incl])
        off += incl
        out.append(Frame(ts_us=sec * 1_000_000 + usec, dir=dirs[j], caplen=incl, wirelen=orig, data=data))
    if off != len(pt):
        raise ValueError("trailing bytes in retained plaintext")
    return out


@dataclass
class RetainedItem:
    e: int
    index: int
    ciphertext: bytes
    pkt_leaves: list[bytes]          # per-packet frame-hash list ℓ_i·

    @property
    def size_bytes(self) -> int:
        return len(self.ciphertext) + 32 * len(self.pkt_leaves) + 12


@dataclass
class RetentionStore:
    """Encrypted evidence store S: retained items keyed by (e, i)."""
    cipher: EpochCipher
    items: dict[tuple[int, int], RetainedItem] = field(default_factory=dict)

    def retain(self, e: int, u: Unit) -> RetainedItem:
        if not u.pay or u.frames is None:
            raise ValueError("only payload units with frames can be retained")
        if (e, u.index) in self.items:
            raise ValueError(f"unit ({e}, {u.index}) already retained (nonce reuse guard)")
        prepare_unit(u)
        lv = hs.frame_leaves([f.ts_us for f in u.frames], [f.dir for f in u.frames], [f.caplen for f in u.frames],
                             [f.data for f in u.frames])
        if lv != list(u.pkt_leaves):
            raise ValueError(f"unit ({e}, {u.index}): frames do not match the committed frame-hash list")
        ct = self.cipher.seal(e, u.index, frames_to_plaintext(u.frames))
        item = RetainedItem(e, u.index, ct, list(u.pkt_leaves))
        self.items[(e, u.index)] = item
        return item

    def frames(self, e: int, i: int) -> list[Frame]:
        it = self.items[(e, i)]
        return plaintext_to_frames(self.cipher.open(e, i, it.ciphertext))

    def has(self, e: int, i: int) -> bool:
        return (e, i) in self.items

    def delete(self, e: int, i: int) -> None:
        self.items.pop((e, i), None)

    def indices(self, e: int) -> list[int]:
        return sorted(i for (ee, i) in self.items if ee == e)

    def size_bytes(self, e: int | None = None) -> int:
        return sum(it.size_bytes for (ee, _), it in self.items.items() if e is None or ee == e)

    def plaintext_bytes(self, e: int | None = None) -> int:
        """Σ pcap-record bytes retained (= Σ s_i of retained units)."""
        tot = 0
        for (ee, _), it in self.items.items():
            if e is None or ee == e:
                tot += len(it.ciphertext) - TAG_LEN - 8 - len(it.pkt_leaves)
        return tot


class EncodeMetaCodec:
    """Verifier meta codec for the real enc(m_i) of units/encode.py (P1-B): returns
    ``dec_metadata(meta)``. V4 meta_consistency compares the keys that are also store columns
    (pay, device_id, first_ts_us, pkts, frame_bytes, s, fps; zero-payload records carry only pay,
    device_id, first_ts_us, pkts); ``MetadataStore.to_batch`` uses key, last_ts_us, payload_bytes and
    flow_uid to rebuild the units the policy sees."""

    def __init__(self):
        from ..units.encode import dec_metadata
        self._dec = dec_metadata

    def decode(self, meta: bytes, pay: bool | None = None) -> dict:
        return self._dec(meta)
