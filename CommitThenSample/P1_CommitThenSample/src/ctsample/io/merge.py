"""k-way merge of source pcaps into one virtual gateway capture (DESIGN §5.2 "E5 stream").

Each source contributes the records of a time window, shifted by a whole-epoch offset, with
on-the-fly address rewriting (LAN IP and MAC remap into the background LAN). Frames, frame
hashes and leaves computed downstream are therefore those of the virtual merged capture.

* ``SourceSpec``: path, window [t_start_us, t_end_us) in source time, offset_us added to every
  timestamp, ip_map (4- or 16-byte address → same-length address), mac_map (6 B → 6 B), and an
  optional ``keep(pcap_cnt) -> bool`` record filter (e.g. thinning: drop the records of removed
  connections, looked up in pktmap).
* ``rewrite_frame`` rewrites Ethernet/SLL MACs, IPv4/IPv6 addresses (IPv4 header checksum and
  TCP/UDP/ICMPv6 pseudo-header checksums are updated incrementally, RFC 1624) and ARP addresses.
* ``merged(specs)`` yields ``(cnt, ts_us, caplen, wirelen, frame, src_idx, src_pcap_cnt)`` in
  (ts, src_idx, src_pcap_cnt) order; ``cnt`` is the 1-based record index of the merged capture.
  All sources must share one link type.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ctsample.io.pcap_reader import (K_IP, LT_ETHERNET, LT_SLL, P_ETYPE, P_FRAG, P_KIND, P_L3OFF,
                                     P_L4OK, P_PROTO, PcapReader, parser_for)


@dataclass
class SourceSpec:
    path: str | Path
    offset_us: int = 0
    t_start_us: int | None = None
    t_end_us: int | None = None
    ip_map: dict = field(default_factory=dict)
    mac_map: dict = field(default_factory=dict)
    keep: Callable[[int], bool] | None = None
    name: str = ""


def _csum_update(cs: int, old: bytes, new: bytes) -> int:
    """RFC 1624 incremental update of a 16-bit one's-complement checksum for a changed field."""
    c = (~cs) & 0xFFFF
    for i in range(0, len(old), 2):
        m = (old[i] << 8) | old[i + 1]
        n = (new[i] << 8) | new[i + 1]
        c += ((~m) & 0xFFFF) + n
    while c >> 16:
        c = (c & 0xFFFF) + (c >> 16)
    return (~c) & 0xFFFF


def _fix(b: bytearray, off: int, old: bytes, new: bytes, zero_means_none: bool = False) -> None:
    if off + 2 > len(b):
        return
    cs = (b[off] << 8) | b[off + 1]
    if zero_means_none and cs == 0:
        return
    v = _csum_update(cs, old, new)
    if zero_means_none and v == 0:
        v = 0xFFFF
    b[off] = v >> 8
    b[off + 1] = v & 0xFF


def rewrite_frame(frame: bytes, linktype: int, p: tuple, ip_map: dict, mac_map: dict) -> bytes:
    if not ip_map and not mac_map:
        return frame
    b = bytearray(frame)
    if mac_map:
        if linktype == LT_ETHERNET and len(b) >= 12:
            d, s = bytes(b[0:6]), bytes(b[6:12])
            b[0:6] = mac_map.get(d, d)
            b[6:12] = mac_map.get(s, s)
        elif linktype == LT_SLL and len(b) >= 12:
            s = bytes(b[6:12])
            b[6:12] = mac_map.get(s, s)
    l3 = p[P_L3OFF]
    if p[P_KIND] == K_IP and ip_map:
        v = b[l3] >> 4 if len(b) > l3 else 0
        if v == 4 and len(b) >= l3 + 20:
            so, do = bytes(b[l3 + 12:l3 + 16]), bytes(b[l3 + 16:l3 + 20])
            sn, dn = ip_map.get(so, so), ip_map.get(do, do)
            if sn != so or dn != do:
                b[l3 + 12:l3 + 16] = sn
                b[l3 + 16:l3 + 20] = dn
                _fix(b, l3 + 10, so + do, sn + dn)
                frag = p[P_FRAG]
                if p[P_L4OK] and (frag is None or frag[1] == 0):
                    ihl = (b[l3] & 15) * 4
                    l4 = l3 + ihl
                    if p[P_PROTO] == 6:
                        _fix(b, l4 + 16, so + do, sn + dn)
                    elif p[P_PROTO] == 17:
                        _fix(b, l4 + 6, so + do, sn + dn, zero_means_none=True)
        elif v == 6 and len(b) >= l3 + 40:
            so, do = bytes(b[l3 + 8:l3 + 24]), bytes(b[l3 + 24:l3 + 40])
            sn, dn = ip_map.get(so, so), ip_map.get(do, do)
            if sn != so or dn != do:
                b[l3 + 8:l3 + 24] = sn
                b[l3 + 24:l3 + 40] = dn
                frag = p[P_FRAG]
                if frag is None or frag[1] == 0:
                    # L4 offset = payload offset minus L4 header; recompute from the parser's view
                    pr = p[P_PROTO]
                    l4 = _l4_offset_v6(b, l3)
                    if l4 is not None:
                        if pr == 6:
                            _fix(b, l4 + 16, so + do, sn + dn)
                        elif pr == 17:
                            _fix(b, l4 + 6, so + do, sn + dn)
                        elif pr == 58:
                            _fix(b, l4 + 2, so + do, sn + dn)
    elif p[P_ETYPE] == 0x0806 and ip_map and len(b) >= l3 + 28:       # ARP (IPv4 over Ethernet)
        for off in (l3 + 14, l3 + 24):
            a = bytes(b[off:off + 4])
            b[off:off + 4] = ip_map.get(a, a)
        if mac_map:
            for off in (l3 + 8, l3 + 18):
                m = bytes(b[off:off + 6])
                b[off:off + 6] = mac_map.get(m, m)
    return bytes(b)


def _l4_offset_v6(b: bytearray, l3: int) -> int | None:
    nxt = b[l3 + 6]
    p = l3 + 40
    for _ in range(10):
        if nxt in (0, 43, 60, 135, 139, 140):
            if len(b) < p + 8:
                return None
            hl = (b[p + 1] + 1) << 3
            nxt = b[p]
            p += hl
        elif nxt == 44:
            if len(b) < p + 8:
                return None
            nxt = b[p]
            p += 8
        elif nxt == 51:
            if len(b) < p + 8:
                return None
            hl = (b[p + 1] + 2) << 2
            nxt = b[p]
            p += hl
        else:
            return p
    return None


def _source_iter(i: int, spec: SourceSpec):
    rd = PcapReader(spec.path)
    parse = parser_for(rd.linktype)
    lo, hi, off, keep = spec.t_start_us, spec.t_end_us, spec.offset_us, spec.keep
    for cnt, ts, cl, wl, fr in rd:
        if lo is not None and ts < lo:
            continue
        if hi is not None and ts >= hi:
            continue
        if keep is not None and not keep(cnt):
            continue
        if spec.ip_map or spec.mac_map:
            fr = rewrite_frame(fr, rd.linktype, parse(fr, wl), spec.ip_map, spec.mac_map)
        yield ts + off, i, cnt, cl, wl, fr, rd.linktype


def merged(specs: list[SourceSpec]):
    its = [_source_iter(i, s) for i, s in enumerate(specs)]
    heap = []
    for it in its:
        x = next(it, None)
        if x is not None:
            heapq.heappush(heap, (x[0], x[1], x[2], x, it))
    lt0 = None
    n = 0
    while heap:
        _, _, _, x, it = heapq.heappop(heap)
        ts, i, cnt, cl, wl, fr, lt = x
        if lt0 is None:
            lt0 = lt
        elif lt != lt0:
            raise ValueError("sources with different link types cannot be merged")
        n += 1
        yield n, ts, len(fr), wl + (len(fr) - cl), fr, i, cnt
        y = next(it, None)
        if y is not None:
            heapq.heappush(heap, (y[0], y[1], y[2], y, it))


def write_merged_pcap(specs: list[SourceSpec], out: str | Path) -> int:
    from ctsample.content.replay import PcapWriter
    w = PcapWriter(out)
    n = 0
    for n, ts, cl, wl, fr, _i, _c in merged(specs):
        w.write(ts, fr, wl)
    w.close()
    return n
