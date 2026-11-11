"""Per-unit replay pcaps for Suricata (DESIGN §5.3 item 4, §5.4 "per-unit replay").

Every frame of a payload unit is rewritten as an Ethernet frame carrying two stacked 802.1Q
tags (TPID 0x8100 for both) that encode the unit's replay id u:

    outer VLAN = u // 4094 + 1,   inner VLAN = u % 4094 + 1      (u < 4094² = 16,760,836)

With ``vlan.use-for-tracking`` Suricata hashes flows on the VLAN ids, so every unit is its own
flow. Tags already present in the source frame are removed first. Linux SLL / SLL2, raw-IP
and NULL frames are converted to Ethernet with locally administered dummy MACs (initiator
02:00:00:00:00:0a, responder 02:00:00:00:00:0b). Frames keep their original timestamps and
order. ``u`` is the per-source replay id ``rid`` (ordinal of the payload unit in the source;
units/<ds>/<source>/part-*.parquet maps rid → unit_gid).

Fallback (DESIGN; only needed if QinQ tracking failed): ``port_rewrite_frame`` rewrites the
client port per unit instead. It is not used by default (tests/test_replay_vlan.py verifies
the VLAN path with Suricata).
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

from ctsample.io.pcap_reader import LT_ETHERNET

VLAN_N = 4094
MAX_UNITS = VLAN_N * VLAN_N
MAC_A = bytes.fromhex("02000000000a")
MAC_B = bytes.fromhex("02000000000b")
TPID = b"\x81\x00"
SNAPLEN = 262144 + 64

_GHDR = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, SNAPLEN, LT_ETHERNET)
_RHDR = struct.Struct("<IIII")


def vlan_pair(u: int) -> tuple[int, int]:
    if not 0 <= u < MAX_UNITS:
        raise ValueError(f"replay id {u} outside [0, {MAX_UNITS})")
    return u // VLAN_N + 1, u % VLAN_N + 1


def unit_from_vlans(outer: int, inner: int) -> int:
    if not (1 <= outer <= VLAN_N and 1 <= inner <= VLAN_N):
        raise ValueError(f"vlan ids out of range: {outer}, {inner}")
    return (outer - 1) * VLAN_N + (inner - 1)


def vlan_tags(u: int) -> bytes:
    o, i = vlan_pair(u)
    return TPID + o.to_bytes(2, "big") + TPID + i.to_bytes(2, "big")


def tag_frame(frame: bytes, linktype: int, l3off: int, ethertype: int, direction: int, u: int) -> bytes:
    """Return the replay version of one frame (Ethernet + QinQ(u) + original L3..)."""
    tags = vlan_tags(u)
    et = ethertype.to_bytes(2, "big")
    if linktype == LT_ETHERNET:
        return frame[:12] + tags + et + frame[l3off:]
    macs = (MAC_B + MAC_A) if direction == 0 else (MAC_A + MAC_B)          # dst, src
    return macs + tags + et + frame[l3off:]


def port_rewrite_frame(frame: bytes, l3off: int, ip_version: int, client_is_src: bool, new_port: int) -> bytes:
    """Fallback codec: rewrite the client-side TCP/UDP port (checksums are not fixed; run
    Suricata with -k none)."""
    b = bytearray(frame)
    if ip_version == 4:
        ihl = (b[l3off] & 15) * 4
        p = l3off + ihl
    else:
        p = l3off + 40
    off = p if client_is_src else p + 2
    b[off:off + 2] = new_port.to_bytes(2, "big")
    return bytes(b)


class PcapWriter:
    """Minimal classic-pcap (µs, Ethernet) writer; written to <path>.tmp, renamed on close()."""

    def __init__(self, path: str | Path, bufsize: int = 1 << 20):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        self.f = open(self.tmp, "wb", buffering=bufsize)
        self.f.write(_GHDR)
        self.n = 0
        self.bytes = len(_GHDR)

    def write(self, ts_us: int, frame: bytes, wirelen: int) -> None:
        self.f.write(_RHDR.pack(ts_us // 1_000_000, ts_us % 1_000_000, len(frame), wirelen))
        self.f.write(frame)
        self.n += 1
        self.bytes += 16 + len(frame)

    def close(self) -> None:
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()
        os.replace(self.tmp, self.path)

    def abort(self) -> None:
        try:
            self.f.close()
        finally:
            self.tmp.unlink(missing_ok=True)


class ReplayWriter:
    """Full replay + 10 % subset replay of one source."""

    def __init__(self, path: str | Path, sub_path: str | Path | None, linktype: int):
        self.linktype = linktype
        self.main = PcapWriter(path)
        self.sub = PcapWriter(sub_path) if sub_path else None
        self.units_main = 0
        self.units_sub = 0

    def write_epoch(self, items: list[tuple]) -> None:
        """items: (pcap_cnt, ts_us, frame, wirelen, caplen, l3off, ethertype, dir, rid, in_sub), any order;
        written sorted by pcap_cnt (= original capture order)."""
        items.sort(key=lambda x: x[0])
        lt = self.linktype
        for cnt, ts, fr, wl, cl, l3off, et, d, rid, insub in items:
            nf = tag_frame(fr, lt, l3off, et, d, rid)
            nwl = wl + (len(nf) - cl)
            self.main.write(ts, nf, nwl)
            if insub and self.sub is not None:
                self.sub.write(ts, nf, nwl)

    def close(self) -> None:
        self.main.close()
        if self.sub is not None:
            self.sub.close()

    def abort(self) -> None:
        self.main.abort()
        if self.sub is not None:
            self.sub.abort()
