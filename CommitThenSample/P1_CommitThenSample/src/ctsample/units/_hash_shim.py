"""BLAKE3 commitment formulas used by the flow table (DESIGN §3.2; batch.py docstring).

This is a local shim so the data pipeline does not wait for ``ctsample.crypto.hashing``
(owned by P1-A). The formulas are exactly the DESIGN ones:

* frame leaf   ℓ_ij = BLAKE3("CTS/v1/frm" ‖ u64le ts_us ‖ u8 dir ‖ u32le caplen ‖ BLAKE3(frame))
* unit leaf    L_i  = BLAKE3("CTS/v1/leaf" ‖ meta ‖ root_pkt)      (payload units)
* leaf0        L_i  = BLAKE3("CTS/v1/leaf0" ‖ meta)                (zero-payload units)
* flow_uid          = BLAKE3("CTS/v1/flow" ‖ key ‖ u64le first_ts_us)[:16]
* Merkle root: leaf node BLAKE3(0x00 ‖ x), inner node BLAKE3(0x01 ‖ l ‖ r), bottom-up
  pairing, the last odd node of a level is promoted unchanged; empty root BLAKE3("CTS/v1/empty").

``tests/test_encode.py`` cross-checks these against crypto/hashing.py when that exists.
"""
from __future__ import annotations

import struct

from blake3 import blake3

D_FRM = b"CTS/v1/frm"
D_LEAF = b"CTS/v1/leaf"
D_LEAF0 = b"CTS/v1/leaf0"
D_FLOW = b"CTS/v1/flow"
D_EMPTY = b"CTS/v1/empty"
D_SUB10 = b"CTS/v1/sub10"          # replay-subset selector (data pipeline only, not committed)

_FRM_HDR = struct.Struct("<QBI")    # u64 ts_us, u8 dir, u32 caplen (13 bytes, no padding)
_U64 = struct.Struct("<Q")
EMPTY_ROOT = blake3(D_EMPTY).digest()


def h(data: bytes) -> bytes:
    return blake3(data).digest()


def frame_leaf(ts_us: int, direction: int, caplen: int, frame: bytes) -> bytes:
    return blake3(D_FRM + _FRM_HDR.pack(ts_us, direction, caplen) + blake3(frame).digest()).digest()


def merkle_root(leaves: list[bytes]) -> bytes:
    if not leaves:
        return EMPTY_ROOT
    level = [blake3(b"\x00" + x).digest() for x in leaves]
    while len(level) > 1:
        nxt = [blake3(b"\x01" + level[i] + level[i + 1]).digest() for i in range(0, len(level) - 1, 2)]
        if len(level) & 1:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def unit_leaf(meta: bytes, root_pkt: bytes) -> bytes:
    return blake3(D_LEAF + meta + root_pkt).digest()


def unit_leaf0(meta: bytes) -> bytes:
    return blake3(D_LEAF0 + meta).digest()


def flow_uid(key: bytes, first_ts_us: int) -> bytes:
    return blake3(D_FLOW + key + _U64.pack(first_ts_us)).digest()[:16]


def sub10_selected(unit_gid: int) -> bool:
    """Fixed-hash 10% subset of units for the Suricata subset-invariance audit."""
    return int.from_bytes(blake3(D_SUB10 + _U64.pack(unit_gid & 0xFFFFFFFFFFFFFFFF)).digest()[:8],
                          "little") < 1844674407370955162     # floor(0.1 * 2**64)
