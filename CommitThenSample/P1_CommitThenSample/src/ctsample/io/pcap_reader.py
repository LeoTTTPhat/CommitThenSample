"""Streaming pcap / pcapng reader and fast L2–L4 header parser (DESIGN §5.3 item 1).

* Classic pcap (µs or ns, either byte order) is read with a buffered ``struct`` loop, which
  is several times faster than ``dpkt.pcap.Reader`` and keeps integer timestamps exact.
  pcapng is read with dpkt (``dpkt.pcapng.Reader`` for the SHB/IDB, dpkt's Enhanced/simple
  packet block classes for the records) so that ``pkt_len`` and integer timestamps are kept.
* A truncated record at the end of the file (dpkt ``NeedData`` / ``UnpackError`` for pcapng)
  stops reading with ``qa['stop_reason'] = 'truncated_tail'``; a corrupt record header in
  the middle of a file (caplen above the limit) stops with ``'corrupt_record'``.
* Link types: Ethernet (1), Linux SLL (113), SLL2 (276), raw IP (101, 12, 14, 228, 229),
  BSD NULL/LOOP (0, 108). 802.1Q / 802.1ad tags are skipped. IPv4 and IPv6 (extension
  headers, fragments).

Every record yields ``(pcap_cnt, ts_us, caplen, wirelen, frame)``; ``pcap_cnt`` is the
1-based record index (the same index Suricata reports as ``pcap_cnt``). ``parse_frame``
returns a :data:`Parsed` tuple; indices are the ``P_*`` constants below.
"""
from __future__ import annotations

import struct
from pathlib import Path

LT_ETHERNET = 1
LT_NULL = 0
LT_LOOP = 108
LT_SLL = 113
LT_SLL2 = 276
LT_RAW = (101, 12, 14, 228, 229)
SUPPORTED_LINKTYPES = (LT_ETHERNET, LT_NULL, LT_LOOP, LT_SLL, LT_SLL2) + LT_RAW

MAX_CAPLEN_DEFAULT = 262144
V4PFX = b"\x00" * 10 + b"\xff\xff"

# kinds
K_BAD = -1      # frame too short for its link header
K_L2 = 0        # non-IP frame (ARP, LLDP, ...): L2 unit, zero payload
K_IP = 1        # IPv4 / IPv6

# Parsed tuple indices
P_KIND, P_SMAC, P_DMAC, P_ETYPE, P_L3OFF, P_SRC, P_DST, P_PROTO, P_SPORT, P_DPORT, \
    P_FLAGS, P_PAYOFF, P_PAYLEN, P_CAPEND, P_FRAG, P_IPVER, P_L4OK = range(17)
# P_SPORT/P_DPORT: TCP/UDP ports; for ICMP P_SPORT = type, P_DPORT = code.
# P_PAYLEN: transport payload bytes as declared by the IP/UDP headers (TCP/UDP only; for a
#           non-first fragment: the IP payload bytes of the fragment). 0 otherwise.
# P_CAPEND: end offset of the captured payload (frame[P_PAYOFF:P_CAPEND] is what detectors see).
# P_FRAG:   None, or (frag_id, frag_off, more_fragments) for fragmented datagrams.
# P_L4OK:   False if the TCP/UDP header is truncated / malformed (unit keyed as "other IP").

_HDR_LE = struct.Struct("<IIII")
_HDR_BE = struct.Struct(">IIII")


class PcapError(RuntimeError):
    pass


class PcapReader:
    """Iterate records of a pcap or pcapng file.

    ``qa`` after iteration: n_records, stop_reason ('eof', 'truncated_tail', 'corrupt_record',
    'max_packets', 'pcapng_error:<exc>'), bytes_consumed, n_caplen_gt_wirelen, file_format,
    linktype, snaplen, ts_resolution.
    """

    def __init__(self, path: str | Path, max_packets: int | None = None,
                 max_caplen: int | None = None, chunk: int = 1 << 24):
        self.path = str(path)
        self.max_packets = max_packets
        self.chunk = chunk
        self.qa: dict = {"n_records": 0, "stop_reason": None, "n_caplen_gt_wirelen": 0}
        with open(self.path, "rb") as f:
            magic = f.read(4)
            rest = f.read(20)
        if len(magic) < 4:
            raise PcapError(f"{self.path}: empty file")
        m_le = struct.unpack("<I", magic)[0]
        self._nsec = False
        if m_le in (0xA1B2C3D4, 0xA1B23C4D):
            self.fmt, self._hdr = "pcap", _HDR_LE
            self._nsec = m_le == 0xA1B23C4D
            vmaj, vmin, _tz, _sig, snap, lt = struct.unpack("<HHiIII", rest)
        elif struct.unpack(">I", magic)[0] in (0xA1B2C3D4, 0xA1B23C4D):
            self.fmt, self._hdr = "pcap", _HDR_BE
            self._nsec = struct.unpack(">I", magic)[0] == 0xA1B23C4D
            vmaj, vmin, _tz, _sig, snap, lt = struct.unpack(">HHiIII", rest)
        elif m_le == 0x0A0D0D0A:
            self.fmt = "pcapng"
            snap, lt = self._pcapng_probe()
        else:
            raise PcapError(f"{self.path}: unknown capture format (magic {magic.hex()})")
        self.linktype = lt & 0x0FFFFFFF if self.fmt == "pcap" else lt
        self.snaplen = snap
        self.max_caplen = max_caplen or max(MAX_CAPLEN_DEFAULT, snap or 0)
        self.qa.update(file_format=self.fmt, linktype=self.linktype, snaplen=snap,
                       ts_resolution="ns" if self._nsec else "us")

    # ------------------------------------------------------------------ pcapng
    def _pcapng_probe(self):
        import dpkt
        with open(self.path, "rb") as f:
            r = dpkt.pcapng.Reader(f)
            return r.snaplen, r.datalink()

    def _iter_pcapng(self):
        import dpkt
        from dpkt import pcapng as ng
        f = open(self.path, "rb")
        try:
            r = ng.Reader(f)
            le = r._Reader__le                                   # noqa: SLF001 (dpkt private)
            div = r._divisor
            tsoff = int(r._tsoffset)
            # exact integer conversion to µs when the resolution is a power of ten
            p10 = {10.0 ** k: k for k in range(0, 13)}
            k10 = p10.get(div)
            self.qa["ts_resolution"] = f"1/{int(div)} s"
            cnt = 0
            unpack = struct.Struct("<II" if le else ">II").unpack
            EPB = ng.EnhancedPacketBlockLE if le else ng.EnhancedPacketBlock
            PB = ng.PacketBlockLE if le else ng.PacketBlock
            while True:
                hdr = f.read(8)
                if not hdr:
                    self.qa["stop_reason"] = "eof"
                    return
                if len(hdr) < 8:
                    self.qa["stop_reason"] = "truncated_tail"
                    return
                btype, blen = unpack(hdr)
                if blen < 12 or blen > self.max_caplen + 4096:
                    self.qa["stop_reason"] = "corrupt_record"
                    return
                body = f.read(blen - 8)
                if len(body) < blen - 8:
                    self.qa["stop_reason"] = "truncated_tail"
                    return
                if btype not in (ng.PCAPNG_BT_EPB, ng.PCAPNG_BT_PB):
                    if btype == ng.PCAPNG_BT_IDB:
                        self.qa["n_extra_idb"] = self.qa.get("n_extra_idb", 0) + 1
                    continue
                try:
                    blk = (EPB if btype == ng.PCAPNG_BT_EPB else PB)(hdr + body)
                except (dpkt.NeedData, dpkt.UnpackError) as e:
                    self.qa["stop_reason"] = f"truncated_tail:{type(e).__name__}"
                    return
                raw = (blk.ts_high << 32) | blk.ts_low
                if k10 is not None:
                    ts_us = tsoff * 1_000_000 + (raw * 10 ** (6 - k10) if k10 <= 6 else raw // 10 ** (k10 - 6))
                else:
                    ts_us = tsoff * 1_000_000 + int(round(raw * 1e6 / div))
                cnt += 1
                frame = bytes(blk.pkt_data)
                wl = blk.pkt_len
                if len(frame) > wl:
                    self.qa["n_caplen_gt_wirelen"] += 1
                    wl = len(frame)
                self.qa["n_records"] = cnt
                yield cnt, ts_us, len(frame), wl, frame
                if self.max_packets and cnt >= self.max_packets:
                    self.qa["stop_reason"] = "max_packets"
                    return
        except Exception as e:  # noqa: BLE001
            self.qa["stop_reason"] = f"pcapng_error:{type(e).__name__}:{e}"[:200]
            return
        finally:
            f.close()

    # ------------------------------------------------------------------ classic pcap
    def _iter_pcap(self):
        hdr = self._hdr
        unpack_from = hdr.unpack_from
        nsec = self._nsec
        maxc = self.max_caplen
        maxp = self.max_packets or 0
        CH = self.chunk
        cnt = 0
        consumed = 24
        with open(self.path, "rb") as f:
            f.seek(24)
            buf = f.read(CH)
            off = 0
            blen = len(buf)
            while True:
                if blen - off < 16:
                    more = f.read(CH)
                    if not more:
                        self.qa["stop_reason"] = "eof" if blen == off else "truncated_tail"
                        break
                    buf = buf[off:] + more
                    off = 0
                    blen = len(buf)
                    continue
                sec, frac, incl, orig = unpack_from(buf, off)
                if incl > maxc:
                    self.qa["stop_reason"] = "corrupt_record"
                    self.qa["corrupt_record_offset"] = consumed
                    self.qa["corrupt_record_caplen"] = incl
                    break
                end = off + 16 + incl
                if end > blen:
                    more = f.read(max(CH, incl + 16))
                    if not more:
                        self.qa["stop_reason"] = "truncated_tail"
                        break
                    buf = buf[off:] + more
                    off = 0
                    blen = len(buf)
                    continue
                frame = buf[off + 16:end]
                off = end
                consumed += 16 + incl
                cnt += 1
                if orig < incl:
                    self.qa["n_caplen_gt_wirelen"] += 1
                    orig = incl
                yield cnt, sec * 1_000_000 + (frac // 1000 if nsec else frac), incl, orig, frame
                if maxp and cnt >= maxp:
                    self.qa["stop_reason"] = "max_packets"
                    break
        self.qa["n_records"] = cnt
        self.qa["bytes_consumed"] = consumed

    def __iter__(self):
        if self.fmt == "pcap":
            return self._iter_pcap()
        return self._iter_pcapng()


# ====================================================================== frame parsing
def _l4(buf, n, kind, smac, dmac, et, l3off, src, dst, proto, p, end, frag, ver):
    """Parse TCP/UDP/ICMP at offset p of the IP datagram ending (on the wire) at ``end``."""
    if frag is not None and frag[1] != 0:                       # non-first fragment
        plen = end - p if end > p else 0
        return (kind, smac, dmac, et, l3off, src, dst, proto, 0, 0, 0, p, plen, min(end, n), frag, ver, True)
    if proto == 6:
        if n - p < 20:
            return (kind, smac, dmac, et, l3off, src, dst, proto, 0, 0, 0, p, 0, p, frag, ver, False)
        doff = (buf[p + 12] >> 4) << 2
        if doff < 20:
            return (kind, smac, dmac, et, l3off, src, dst, proto, 0, 0, 0, p, 0, p, frag, ver, False)
        po = p + doff
        plen = end - po if end > po else 0
        ce = end if end < n else n
        return (kind, smac, dmac, et, l3off, src, dst, 6, (buf[p] << 8) | buf[p + 1],
                (buf[p + 2] << 8) | buf[p + 3], buf[p + 13], po, plen, ce if ce > po else po, frag, ver, True)
    if proto == 17:
        if n - p < 8:
            return (kind, smac, dmac, et, l3off, src, dst, proto, 0, 0, 0, p, 0, p, frag, ver, False)
        ulen = (buf[p + 4] << 8) | buf[p + 5]
        po = p + 8
        ue = p + ulen if ulen >= 8 else end
        if ue > end:
            ue = end
        plen = ue - po if ue > po else 0
        ce = ue if ue < n else n
        return (kind, smac, dmac, et, l3off, src, dst, 17, (buf[p] << 8) | buf[p + 1],
                (buf[p + 2] << 8) | buf[p + 3], 0, po, plen, ce if ce > po else po, frag, ver, True)
    if proto == 1 or proto == 58:
        t = buf[p] if n > p else 0
        c = buf[p + 1] if n > p + 1 else 0
        return (kind, smac, dmac, et, l3off, src, dst, proto, t, c, 0, p, 0, p, frag, ver, True)
    return (kind, smac, dmac, et, l3off, src, dst, proto, 0, 0, 0, p, 0, p, frag, ver, True)


def _ip(buf, n, wirelen, smac, dmac, et, off):
    if n - off < 1:
        return (K_L2, smac, dmac, et, off, None, None, 0, 0, 0, 0, off, 0, off, None, 0, False)
    v = buf[off] >> 4
    if v == 4:
        if n - off < 20:
            return (K_L2, smac, dmac, et, off, None, None, 0, 0, 0, 0, off, 0, off, None, 0, False)
        ihl = (buf[off] & 15) << 2
        if ihl < 20:
            return (K_L2, smac, dmac, et, off, None, None, 0, 0, 0, 0, off, 0, off, None, 0, False)
        tot = (buf[off + 2] << 8) | buf[off + 3]
        end = off + tot if tot >= ihl else wirelen
        if end > wirelen:
            end = wirelen
        ff = (buf[off + 6] << 8) | buf[off + 7]
        fo = ff & 0x1FFF
        mf = (ff >> 13) & 1
        frag = ((buf[off + 4] << 8) | buf[off + 5], fo, mf) if (fo or mf) else None
        return _l4(buf, n, K_IP, smac, dmac, et, off, V4PFX + buf[off + 12:off + 16],
                   V4PFX + buf[off + 16:off + 20], buf[off + 9], off + ihl, end, frag, 4)
    if v == 6:
        if n - off < 40:
            return (K_L2, smac, dmac, et, off, None, None, 0, 0, 0, 0, off, 0, off, None, 0, False)
        plen = (buf[off + 4] << 8) | buf[off + 5]
        nxt = buf[off + 6]
        p = off + 40
        end = p + plen if plen else wirelen
        if end > wirelen:
            end = wirelen
        frag = None
        for _ in range(10):
            if nxt == 0 or nxt == 43 or nxt == 60 or nxt == 135 or nxt == 139 or nxt == 140:
                if n - p < 8:
                    break
                hl = (buf[p + 1] + 1) << 3
                nxt = buf[p]
                p += hl
            elif nxt == 44:
                if n - p < 8:
                    break
                ff = (buf[p + 2] << 8) | buf[p + 3]
                frag = (int.from_bytes(buf[p + 4:p + 8], "big"), ff >> 3, ff & 1)
                nxt = buf[p]
                p += 8
                if frag[1] != 0:
                    break
            elif nxt == 51:
                if n - p < 8:
                    break
                hl = (buf[p + 1] + 2) << 2
                nxt = buf[p]
                p += hl
            else:
                break
        return _l4(buf, n, K_IP, smac, dmac, et, off, buf[off + 8:off + 24], buf[off + 24:off + 40],
                   nxt, p, end, frag, 6)
    return (K_L2, smac, dmac, et, off, None, None, 0, 0, 0, 0, off, 0, off, None, 0, False)


def parse_ether(buf: bytes, wirelen: int):
    n = len(buf)
    if n < 14:
        return (K_BAD, buf[6:12] if n >= 12 else b"\x00" * 6, buf[0:6] if n >= 6 else b"\x00" * 6,
                0xFFFF, n, None, None, 0, 0, 0, 0, n, 0, n, None, 0, False)
    et = (buf[12] << 8) | buf[13]
    off = 14
    while (et == 0x8100 or et == 0x88A8 or et == 0x9100) and n >= off + 4:
        et = (buf[off + 2] << 8) | buf[off + 3]
        off += 4
    if et == 0x0800 or et == 0x86DD:
        return _ip(buf, n, wirelen, buf[6:12], buf[0:6], et, off)
    if et < 0x0600:                       # 802.3 length field (LLC/SNAP): one L2 key per MAC
        et = 0
    return (K_L2, buf[6:12], buf[0:6], et, off, None, None, 0, 0, 0, 0, off, 0, off, None, 0, False)


def parse_sll(buf: bytes, wirelen: int):
    n = len(buf)
    if n < 16:
        return (K_BAD, b"\x00" * 6, b"\x00" * 6, 0xFFFF, n, None, None, 0, 0, 0, 0, n, 0, n, None, 0, False)
    halen = (buf[4] << 8) | buf[5]
    smac = buf[6:12] if halen == 6 else (buf[6:6 + min(halen, 8)] + b"\x00" * 6)[:6]
    et = (buf[14] << 8) | buf[15]
    off = 16
    while (et == 0x8100 or et == 0x88A8) and n >= off + 4:
        et = (buf[off + 2] << 8) | buf[off + 3]
        off += 4
    if et == 0x0800 or et == 0x86DD:
        return _ip(buf, n, wirelen, smac, None, et, off)
    return (K_L2, smac, None, et, off, None, None, 0, 0, 0, 0, off, 0, off, None, 0, False)


def parse_sll2(buf: bytes, wirelen: int):
    n = len(buf)
    if n < 20:
        return (K_BAD, b"\x00" * 6, b"\x00" * 6, 0xFFFF, n, None, None, 0, 0, 0, 0, n, 0, n, None, 0, False)
    et = (buf[0] << 8) | buf[1]
    halen = buf[11]
    smac = buf[12:18] if halen == 6 else (buf[12:12 + min(halen, 8)] + b"\x00" * 6)[:6]
    off = 20
    if et == 0x0800 or et == 0x86DD:
        return _ip(buf, n, wirelen, smac, None, et, off)
    return (K_L2, smac, None, et, off, None, None, 0, 0, 0, 0, off, 0, off, None, 0, False)


def parse_raw(buf: bytes, wirelen: int):
    n = len(buf)
    if n < 1:
        return (K_BAD, None, None, 0xFFFF, 0, None, None, 0, 0, 0, 0, 0, 0, 0, None, 0, False)
    v = buf[0] >> 4
    return _ip(buf, n, wirelen, None, None, 0x86DD if v == 6 else 0x0800, 0)


def parse_null(buf: bytes, wirelen: int):
    n = len(buf)
    if n < 5:
        return (K_BAD, None, None, 0xFFFF, n, None, None, 0, 0, 0, 0, n, 0, n, None, 0, False)
    v = buf[4] >> 4
    return _ip(buf, n, wirelen, None, None, 0x86DD if v == 6 else 0x0800, 4)


def parser_for(linktype: int):
    if linktype == LT_ETHERNET:
        return parse_ether
    if linktype == LT_SLL:
        return parse_sll
    if linktype == LT_SLL2:
        return parse_sll2
    if linktype in LT_RAW:
        return parse_raw
    if linktype in (LT_NULL, LT_LOOP):
        return parse_null
    raise PcapError(f"unsupported link type {linktype}")


def parse_frame(linktype: int, buf: bytes, wirelen: int | None = None):
    return parser_for(linktype)(buf, len(buf) if wirelen is None else wirelen)


def iter_parsed(path: str | Path, max_packets: int | None = None):
    """Convenience generator: (pcap_cnt, ts_us, caplen, wirelen, frame, parsed)."""
    r = PcapReader(path, max_packets=max_packets)
    parse = parser_for(r.linktype)
    for cnt, ts, cl, wl, fr in r:
        yield cnt, ts, cl, wl, fr, parse(fr, wl)
