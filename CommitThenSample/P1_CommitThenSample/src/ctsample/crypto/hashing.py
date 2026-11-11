"""BLAKE3-256 domain-separated hashes and the epoch Merkle tree (DESIGN §3.2).

Hashes (all BLAKE3-256, 32-byte outputs, never truncated except ``flow_uid``):

* ``frame_leaf(ts_us, dir, caplen, frame)``  ℓ_ij = H("CTS/v1/frm" ‖ u64le ts_us ‖ u8 dir ‖ u32le caplen ‖ H(frame))
* ``unit_leaf(meta, root_pkt)``             L_i  = H("CTS/v1/leaf"  ‖ meta ‖ root_pkt)     (payload units)
* ``leaf0(meta)``                           L_i  = H("CTS/v1/leaf0" ‖ meta)                (zero-payload units)
* ``flow_uid(key, first_ts_us)``                   H("CTS/v1/flow"  ‖ key ‖ u64le first_ts)[:16]

Merkle tree (used both for root^pkt_i over ℓ_i· and for R_e over L_1..L_n):

* leaf node  = H(0x00 ‖ x), inner node = H(0x01 ‖ left ‖ right);
* bottom-up pairing; the last odd node of a level is promoted unchanged;
* empty tree root = H(b"CTS/v1/empty").

Bottom-up pairing with promotion yields exactly the RFC 6962 ``MTH`` tree (split at the largest
power of two < n); ``tests/test_merkle.py`` checks this against a recursive reference for
n = 1..1,025. Inclusion proofs are lists of sibling hashes (⌈log₂ n⌉ × 32 B at most); the path
shape is implied by (index, n), so no direction bits are stored.

Note on tags: "CTS/v1/leaf" is a byte prefix of "CTS/v1/leaf0". The two leaf kinds are kept apart
by the explicit pay flag of the metadata store plus the fixed-width layouts of enc(m_i) (a payload
input is len(meta)+32 bytes after the tag); the verifier recomputes leaves from the store's
pay flag, never by guessing the kind (see protocol/store.py).
"""
from __future__ import annotations

import struct
from collections.abc import Sequence

from blake3 import blake3 as _B3

HASH_LEN = 32

DOM_FRM = b"CTS/v1/frm"
DOM_LEAF = b"CTS/v1/leaf"
DOM_LEAF0 = b"CTS/v1/leaf0"
DOM_FLOW = b"CTS/v1/flow"
DOM_EMPTY = b"CTS/v1/empty"
NODE_LEAF = b"\x00"
NODE_INNER = b"\x01"

_FRM = struct.Struct("<QBI")
_U64 = struct.Struct("<Q")

EMPTY_ROOT = _B3(DOM_EMPTY).digest()

__all__ = [
    "HASH_LEN", "EMPTY_ROOT", "h", "frame_hash", "frame_leaf", "frame_leaf_from_hash",
    "frame_leaves", "unit_leaf", "leaf0", "unit_leaf0", "flow_uid", "merkle_root", "merkle_levels",
    "merkle_proof", "merkle_proofs", "verify_proof", "leaf_node", "inner_node",
    "packet_root", "packet_roots",
]


def h(data: bytes) -> bytes:
    """Plain BLAKE3-256."""
    return _B3(data).digest()


def frame_hash(frame: bytes) -> bytes:
    """BLAKE3(frame) over the whole captured frame (L2..)."""
    return _B3(frame).digest()


def frame_leaf_from_hash(ts_us: int, dir: int, caplen: int, fh: bytes) -> bytes:
    """ℓ_ij from a precomputed frame hash ``fh`` = BLAKE3(frame)."""
    if len(fh) != HASH_LEN:
        raise ValueError("frame hash must be 32 bytes")
    return _B3(DOM_FRM + _FRM.pack(ts_us, dir, caplen) + fh).digest()


def frame_leaf(ts_us: int, dir: int, caplen: int, frame: bytes) -> bytes:
    """ℓ_ij = BLAKE3("CTS/v1/frm" ‖ u64le ts_us ‖ u8 dir ‖ u32le caplen ‖ BLAKE3(frame))."""
    return _B3(DOM_FRM + _FRM.pack(ts_us, dir, caplen) + _B3(frame).digest()).digest()


def frame_leaves(ts_us: Sequence[int], dirs: Sequence[int], caplens: Sequence[int],
                 frames: Sequence[bytes]) -> list[bytes]:
    """Batch ℓ_ij for one unit (or any packet list)."""
    B, pk, dom = _B3, _FRM.pack, DOM_FRM
    return [B(dom + pk(t, d, c) + B(f).digest()).digest()
            for t, d, c, f in zip(ts_us, dirs, caplens, frames, strict=True)]


def unit_leaf(meta: bytes, root_pkt: bytes) -> bytes:
    """Payload-unit leaf L_i = BLAKE3("CTS/v1/leaf" ‖ enc(m_i) ‖ root^pkt_i)."""
    if len(root_pkt) != HASH_LEN:
        raise ValueError("root_pkt must be 32 bytes")
    return _B3(DOM_LEAF + meta + root_pkt).digest()


def leaf0(meta: bytes) -> bytes:
    """Zero-payload leaf L_i = BLAKE3("CTS/v1/leaf0" ‖ enc(m_i))."""
    return _B3(DOM_LEAF0 + meta).digest()


unit_leaf0 = leaf0          # alias matching units/_hash_shim.py naming


def flow_uid(key: bytes, first_ts_us: int) -> bytes:
    """flow_uid = BLAKE3("CTS/v1/flow" ‖ key ‖ u64le first_ts)[:16]."""
    return _B3(DOM_FLOW + key + _U64.pack(first_ts_us)).digest()[:16]


def leaf_node(x: bytes) -> bytes:
    return _B3(NODE_LEAF + x).digest()


def inner_node(left: bytes, right: bytes) -> bytes:
    return _B3(NODE_INNER + left + right).digest()


def _next_level(level: list[bytes]) -> list[bytes]:
    B, one = _B3, NODE_INNER
    m = len(level)
    nxt = [B(one + level[k] + level[k + 1]).digest() for k in range(0, m - 1, 2)]
    if m & 1:
        nxt.append(level[-1])           # last odd node promoted unchanged
    return nxt


def merkle_root(leaves: Sequence[bytes]) -> bytes:
    """Root over ``leaves`` (each typically 32 B); empty → BLAKE3("CTS/v1/empty")."""
    if not leaves:
        return EMPTY_ROOT
    B, zero = _B3, NODE_LEAF
    level = [B(zero + x).digest() for x in leaves]
    while len(level) > 1:
        level = _next_level(level)
    return level[0]


def merkle_levels(leaves: Sequence[bytes]) -> list[list[bytes]]:
    """All levels, bottom (leaf nodes) first, root level last. Empty input → [[EMPTY_ROOT]]."""
    if not leaves:
        return [[EMPTY_ROOT]]
    B, zero = _B3, NODE_LEAF
    level = [B(zero + x).digest() for x in leaves]
    levels = [level]
    while len(level) > 1:
        level = _next_level(level)
        levels.append(level)
    return levels


def _proof_from_levels(levels: list[list[bytes]], index: int) -> list[bytes]:
    path = []
    for level in levels[:-1]:
        sib = index ^ 1
        if sib < len(level):
            path.append(level[sib])
        index >>= 1
    return path


def merkle_proof(leaves: Sequence[bytes], index: int) -> list[bytes]:
    """Inclusion proof (sibling hashes, bottom-up) of leaf ``index``."""
    n = len(leaves)
    if not 0 <= index < n:
        raise IndexError(f"index {index} out of range for {n} leaves")
    return _proof_from_levels(merkle_levels(leaves), index)


def merkle_proofs(leaves: Sequence[bytes], indices: Sequence[int]) -> dict[int, list[bytes]]:
    """Proofs for many indices with one tree build."""
    n = len(leaves)
    levels = merkle_levels(leaves)
    out = {}
    for i in indices:
        if not 0 <= i < n:
            raise IndexError(f"index {i} out of range for {n} leaves")
        out[i] = _proof_from_levels(levels, i)
    return out


def verify_proof(leaf: bytes, index: int, n: int, proof: Sequence[bytes], root: bytes) -> bool:
    """Check that ``leaf`` is at position ``index`` of an ``n``-leaf tree with root ``root``."""
    if n <= 0 or not 0 <= index < n:
        return False
    node = _B3(NODE_LEAF + leaf).digest()
    k, width = 0, n
    while width > 1:
        if index & 1:                               # right child: sibling on the left
            if k >= len(proof):
                return False
            node = _B3(NODE_INNER + proof[k] + node).digest()
            k += 1
        elif index + 1 < width:                     # left child with a right sibling
            if k >= len(proof):
                return False
            node = _B3(NODE_INNER + node + proof[k]).digest()
            k += 1
        # else: last odd node, promoted unchanged
        index >>= 1
        width = (width + 1) >> 1
    return k == len(proof) and node == root


def packet_root(ts_us: Sequence[int], dirs: Sequence[int], caplens: Sequence[int],
                frames: Sequence[bytes]) -> tuple[bytes, list[bytes]]:
    """(root^pkt, [ℓ_ij]) for one unit's packets in packet order."""
    lv = frame_leaves(ts_us, dirs, caplens, frames)
    return merkle_root(lv), lv


def packet_roots(units_frames: Sequence[Sequence]) -> list[bytes]:
    """root^pkt for many units; each element is a sequence of objects with ts_us/dir/caplen/data."""
    out = []
    B, pk, dom = _B3, _FRM.pack, DOM_FRM
    for frames in units_frames:
        lv = [B(dom + pk(f.ts_us, f.dir, f.caplen) + B(f.data).digest()).digest() for f in frames]
        out.append(merkle_root(lv))
    return out
