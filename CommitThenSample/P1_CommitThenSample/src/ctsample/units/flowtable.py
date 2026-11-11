"""Gateway flow table and 30-s segmenter (DESIGN §3.2; §5.3 item 2).

``FlowSegmenter`` consumes parsed packets in capture order and emits, at every epoch end,
the epoch's units in canonical order with their metadata record, payload flag, fps,
continuation flag, cost s, per-packet frame leaves, root_pkt and leaf L_i.

Keys (bidirectional lookup; the initiator is the source of the flow's first packet):

* TCP / UDP: (proto, ip_a, port_a, ip_b, port_b)
* ICMP / ICMPv6: (src, dst, type)  → key port_a = type, port_b = 0
* other IP protocols: (src, dst, proto), ports 0
* non-IP frames: L2 units per (src MAC, ethertype), proto 0, always zero-payload. The key's
  ip_a holds the src MAC zero-padded to 16 B, ip_b is zero, port_a = ethertype (0 for 802.3/LLC,
  0xFFFF for frames too short for their link header).
* IP fragments: a non-first fragment joins the flow of its datagram's first fragment
  (matched on src, dst, proto, IP id); unmatched non-first fragments are keyed as "other IP".
  Their bytes count as transport payload (they are TCP/UDP payload).

Flow identity across epochs: a flow continues while it is active; it expires after K
consecutive epochs without packets (K = 10 for TCP, i.e. ≥ 300 s idle; K = 2 otherwise, ≥ 60 s),
and a TCP SYN (no ACK) arriving in a new epoch on a flow that was closed (FIN both ways or
RST) starts a new flow. Within one epoch a key is always exactly one unit.

conn_state is Zeek-like (Zeek's ``conn_state`` function over endpoint states that are
accumulated over the flow up to the end of the segment). device_id is the dense index of the
LAN-side MAC (initiator's if both or neither side is LAN), in order of first appearance; the
table is ``mac_table``. Transport payload counts only TCP/UDP payload bytes (as declared by
the IP/UDP length fields, so Ethernet padding is excluded).
"""
from __future__ import annotations

import ipaddress
from array import array

from ctsample.content import detectors as det
from ctsample.io.pcap_reader import (K_IP, P_CAPEND, P_DMAC, P_DPORT, P_DST, P_ETYPE, P_FLAGS, P_FRAG,
                                     P_KIND, P_L3OFF, P_L4OK, P_PAYLEN, P_PAYOFF, P_PROTO, P_SMAC,
                                     P_SPORT, P_SRC)
try:                                   # P1-A's implementation (tests/test_encode.py checks it equals the shim)
    from ctsample.crypto import hashing as hs          # frame_leaf, merkle_root, unit_leaf, unit_leaf0, flow_uid
except ImportError:                    # pragma: no cover
    from ctsample.units import _hash_shim as hs
from ctsample.units.batch import EpochBatch, Frame, Unit
from ctsample.units.encode import CS_CODE, enc_metadata, iat_code, make_key

T_US = 30_000_000
K_TCP = 10
K_OTHER = 2
ZERO16 = b"\x00" * 16

# endpoint TCP states
INACTIVE, SYN_SENT, SYN_ACK_SENT, PARTIAL, ESTABLISHED, CLOSED, RESET = range(7)
F_FIN, F_SYN, F_RST, F_PSH, F_ACK = 0x01, 0x02, 0x04, 0x08, 0x10

# direction classes
DC_LAN_WAN, DC_WAN_LAN, DC_LAN_LAN, DC_WAN_WAN = 0, 1, 2, 3


def default_is_lan(ip16: bytes) -> bool:
    """LAN side: RFC1918, link-local, CGNAT, loopback, unspecified, broadcast/multicast
    (IPv4-mapped) and IPv6 link-local / ULA / multicast / unspecified."""
    if ip16[:12] == b"\x00" * 10 + b"\xff\xff":
        a, b = ip16[12], ip16[13]
        return (a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168) or (a == 169 and b == 254)
                or (a == 100 and 64 <= b <= 127) or a == 127 or a == 0 or a >= 224)
    if ip16[0] == 0xFF or (ip16[0] & 0xFE) == 0xFC or (ip16[0] == 0xFE and (ip16[1] & 0xC0) == 0x80):
        return True
    return ip16 == ZERO16 or ip16 == b"\x00" * 15 + b"\x01"


def ip_text(ip16: bytes, proto: int) -> str:
    if proto == 0:                                        # L2 key: MAC in the first 6 bytes
        return ":".join(f"{x:02x}" for x in ip16[:6]) if ip16 != ZERO16 else ""
    if ip16[:12] == b"\x00" * 10 + b"\xff\xff":
        return ".".join(str(x) for x in ip16[12:])
    return str(ipaddress.IPv6Address(ip16))


def mac_text(m: bytes | None) -> str:
    return ":".join(f"{x:02x}" for x in m) if m else ""


class _Flow:
    __slots__ = ("key", "proto", "first_ts", "uid", "last_ts", "unit", "unit_epoch", "pay_seen", "nseg",
                 "os", "rs", "opk", "rpk", "opay", "rpay", "o_first_syn", "dets", "lan_a", "lan_b", "tup",
                 "port_a", "port_b")


class _Acc:
    __slots__ = ("flow", "arr", "first_ts", "last_ts", "prev_ts", "n", "pk0", "pk1", "pb0", "pb1", "fb0", "fb1",
                 "s", "syn", "synack", "fin", "rst", "psh", "sizes", "iats", "frames", "hits",
                 "s0", "d0", "s1", "d1", "sH", "payH", "Hdone")


class UnitRecord:
    """A finalized unit (one row of units/<ds>/<source>/part-*.parquet)."""
    __slots__ = ("epoch", "index", "local", "rid", "key", "proto", "ip_a", "port_a", "ip_b", "port_b",
                 "mac_a", "mac_b", "device_id", "device_mac", "lan_side", "dir_class", "first_ts_us",
                 "last_ts_us", "pkts", "payload_bytes", "frame_bytes", "n_syn", "n_synack", "n_fin", "n_rst",
                 "n_psh", "conn_state", "sizes", "iats", "s", "pay", "fps", "continuation", "flow_uid", "meta",
                 "root_pkt", "leaf", "pkt_leaves", "frames", "raw_hits", "flow_first_ts_us", "sH")

    def to_unit(self, keep_frames: bool = True) -> Unit:
        frames = None
        if keep_frames and self.frames is not None:
            frames = [Frame(ts_us=f[1], dir=f[2], caplen=f[3], wirelen=f[4], data=f[5]) for f in self.frames]
        return Unit(epoch=self.epoch, index=self.index, key=self.key, device_id=self.device_id,
                    first_ts_us=self.first_ts_us, last_ts_us=self.last_ts_us, pkts=self.pkts,
                    payload_bytes=self.payload_bytes, frame_bytes=self.frame_bytes, s=self.s, pay=self.pay,
                    fps=self.fps, flow_uid=self.flow_uid, meta=self.meta, leaf=self.leaf, root_pkt=self.root_pkt,
                    pkt_leaves=self.pkt_leaves, frames=frames)


def conn_state(fl: _Flow) -> int:
    p = fl.proto
    if p == 6:
        os_, rs = fl.os, fl.rs
        o_in = os_ == INACTIVE or os_ == PARTIAL
        r_in = rs == INACTIVE or rs == PARTIAL
        if rs == RESET:
            if os_ == SYN_SENT or os_ == SYN_ACK_SENT or (os_ == RESET and fl.opay == 0 and fl.rpay == 0):
                return CS_CODE["REJ"]
            return CS_CODE["RSTRH"] if o_in else CS_CODE["RSTR"]
        if os_ == RESET:
            if r_in:
                return CS_CODE["RSTOS0"] if fl.o_first_syn else CS_CODE["OTH"]
            return CS_CODE["RSTO"]
        if rs == CLOSED and os_ == CLOSED:
            return CS_CODE["SF"]
        if os_ == CLOSED:
            return CS_CODE["SH"] if r_in else CS_CODE["S2"]
        if rs == CLOSED:
            return CS_CODE["SHR"] if o_in else CS_CODE["S3"]
        if os_ == SYN_SENT and rs == INACTIVE:
            return CS_CODE["S0"]
        if os_ == ESTABLISHED and rs == ESTABLISHED:
            return CS_CODE["S1"]
        return CS_CODE["OTH"]
    if p == 17:
        if fl.opk:
            return CS_CODE["SF"] if fl.rpk else CS_CODE["S0"]
        return CS_CODE["SHR"] if fl.rpk else CS_CODE["OTH"]
    return CS_CODE["OTH"]


def _tcp_update(fl: _Flow, d: int, f: int) -> None:
    st = fl.os if d == 0 else fl.rs
    if f & F_RST:
        st = RESET
    elif f & F_FIN:
        if st != RESET:
            st = CLOSED
    elif f & F_SYN:
        if f & F_ACK:
            if st == INACTIVE or st == SYN_SENT:
                st = SYN_ACK_SENT
        elif st == INACTIVE:
            st = SYN_SENT
    elif st == INACTIVE:
        st = PARTIAL
    elif st == SYN_SENT or st == SYN_ACK_SENT:
        st = ESTABLISHED
    if d == 0:
        fl.os = st
        if st == ESTABLISHED and fl.rs == SYN_ACK_SENT:
            fl.rs = ESTABLISHED
    else:
        fl.rs = st
        if st == ESTABLISHED and fl.os == SYN_SENT:
            fl.os = ESTABLISHED


class FlowSegmenter:
    """Unitise a packet stream.

    Parameters
    ----------
    on_epoch : callable(epoch:int, t_start_us:int, records:list[UnitRecord]) called at every
        epoch end with the epoch's units in canonical order (frames attached to payload units
        when ``keep_frames``; they are released after the callback returns).
    keep_frames : keep frames of payload units until the callback (needed by the replay writer
        and the in-memory EpochBatch path).
    run_detectors : run D1–D13 on payload packets (hits in UnitRecord.raw_hits).
    track_packets : keep per-packet arrays (pktmap, counters, per-packet timestamps).
    H_bytes : payload horizon for the §0.1 "capped gain" statistic (s^H per unit).
    """

    def __init__(self, on_epoch=None, *, T_us: int = T_US, is_lan=default_is_lan, keep_frames: bool = True,
                 run_detectors: bool = True, track_packets: bool = True, H_bytes: int = 1024):
        self.on_epoch = on_epoch
        self.T = T_us
        self.is_lan = is_lan
        self.keep_frames = keep_frames
        self.run_detectors = run_detectors
        self.track = track_packets
        self.H = H_bytes
        self.t0 = None
        self.cur = 0
        self.flows: dict = {}
        self.frag: dict = {}
        self.epoch_units: list[_Acc] = []
        self.buckets: dict[int, list[_Flow]] = {}
        self.n_arr = 0
        self.arr2local = array("q")
        self.n_units = 0
        self.n_pay = 0
        self.mac_table: dict = {}          # device key (MAC bytes, or IP for MAC-less captures) -> device_id
        self.mac_first: list = []
        # per-packet arrays (track_packets)
        self.pk_arr = array("q")
        self.pk_idx = array("i")
        self.pk_ts = array("q")
        self.pk_smac = array("i")
        self.pk_dmac = array("i")
        self.pk_wire = array("i")
        self.mac_all: dict = {}            # every MAC seen (src or dst) -> index
        self.qa = {"n_packets": 0, "n_late": 0, "n_late_max_us": 0, "n_bad_l2": 0, "n_l2": 0, "n_ip": 0,
                   "n_l4_malformed": 0, "n_frag": 0, "n_frag_unmatched": 0, "n_tcp_reuse_newflow": 0,
                   "n_flows": 0, "n_expired": 0}

    # ------------------------------------------------------------------ helpers
    def _mac_idx(self, m) -> int:
        if m is None:
            return -1
        i = self.mac_all.get(m)
        if i is None:
            i = self.mac_all[m] = len(self.mac_all)
        return i

    def _new_flow(self, tup, proto, ip_a, port_a, ip_b, port_b, ts):
        fl = _Flow()
        fl.tup = tup
        fl.proto = proto
        fl.key = make_key(proto, ip_a, port_a, ip_b, port_b)
        fl.port_a, fl.port_b = port_a, port_b
        fl.first_ts = ts
        fl.uid = hs.flow_uid(fl.key, ts)
        fl.last_ts = ts
        fl.unit = None
        fl.unit_epoch = -1
        fl.pay_seen = False
        fl.nseg = 0
        fl.os = fl.rs = INACTIVE
        fl.opk = fl.rpk = fl.opay = fl.rpay = 0
        fl.o_first_syn = False
        fl.dets = det.flow_detectors(proto, port_a, port_b) if self.run_detectors else ()
        if proto == 0:
            fl.lan_a, fl.lan_b = True, True
        else:
            fl.lan_a, fl.lan_b = self.is_lan(ip_a), self.is_lan(ip_b)
        self.flows[tup] = fl
        self.qa["n_flows"] += 1
        return fl

    # ------------------------------------------------------------------ main entry
    def add(self, cnt: int, ts: int, caplen: int, wirelen: int, frame: bytes, p: tuple) -> None:
        qa = self.qa
        qa["n_packets"] += 1
        if self.t0 is None:
            self.t0 = ts
            self.cur = 0
        e = (ts - self.t0) // self.T
        if e != self.cur:
            if e > self.cur:
                self._flush()
                self._expire(e)
                self.cur = e
            else:
                qa["n_late"] += 1
                late = self.t0 + self.cur * self.T - ts
                if late > qa["n_late_max_us"]:
                    qa["n_late_max_us"] = late
                e = self.cur
        flows = self.flows
        kind = p[P_KIND]
        plen = 0
        flags = -1
        d = 0
        fl = None
        if kind == K_IP:
            qa["n_ip"] += 1
            proto = p[P_PROTO]
            src = p[P_SRC]
            dst = p[P_DST]
            frag = p[P_FRAG]
            if frag is not None:
                qa["n_frag"] += 1
                fk = (src, dst, proto, frag[0])
                if frag[1] != 0:                                   # non-first fragment
                    m = self.frag.get(fk)
                    plen = p[P_PAYLEN]
                    if m is not None and m[0].tup in flows and flows[m[0].tup] is m[0]:
                        fl, d = m[0], m[1]
                    else:
                        qa["n_frag_unmatched"] += 1
                        tup = (proto, src, 0, dst, 0)
                        fl = flows.get(tup)
                        if fl is None:
                            fl = flows.get((proto, dst, 0, src, 0))
                            if fl is None:
                                fl = self._new_flow(tup, proto, src, 0, dst, 0, ts)
                            else:
                                d = 1
            if fl is None:
                if (proto == 6 or proto == 17) and p[P_L4OK]:
                    sp = p[P_SPORT]
                    dp = p[P_DPORT]
                    tup = (proto, src, sp, dst, dp)
                    fl = flows.get(tup)
                    if fl is None:
                        fl = flows.get((proto, dst, dp, src, sp))
                        if fl is None:
                            fl = self._new_flow(tup, proto, src, sp, dst, dp, ts)
                        else:
                            d = 1
                    plen = p[P_PAYLEN]
                    if proto == 6:
                        flags = p[P_FLAGS]
                        # TCP connection reuse: SYN on a closed flow, first packet in this epoch
                        if (flags & 0x12) == F_SYN and fl.unit_epoch != e and (
                                fl.os == RESET or fl.rs == RESET or (fl.os == CLOSED and fl.rs == CLOSED)):
                            del flows[fl.tup]
                            qa["n_tcp_reuse_newflow"] += 1
                            fl = self._new_flow(tup, proto, src, sp, dst, dp, ts)
                            d = 0
                elif proto == 1 or proto == 58:
                    t = p[P_SPORT]
                    tup = (proto, src, t, dst, 0)
                    fl = flows.get(tup)
                    if fl is None:
                        fl = flows.get((proto, dst, t, src, 0))
                        if fl is None:
                            fl = self._new_flow(tup, proto, src, t, dst, 0, ts)
                        else:
                            d = 1
                else:
                    if (proto == 6 or proto == 17) and not p[P_L4OK]:
                        qa["n_l4_malformed"] += 1
                    tup = (proto, src, 0, dst, 0)
                    fl = flows.get(tup)
                    if fl is None:
                        fl = flows.get((proto, dst, 0, src, 0))
                        if fl is None:
                            fl = self._new_flow(tup, proto, src, 0, dst, 0, ts)
                        else:
                            d = 1
                if frag is not None and frag[1] == 0:            # first fragment: remember the flow
                    self.frag[fk] = (fl, d, e)
        else:
            if kind == 0:
                qa["n_l2"] += 1
            else:
                qa["n_bad_l2"] += 1
            smac = p[P_SMAC] or b"\x00" * 6
            et = p[P_ETYPE]
            tup = (0, smac, et)
            fl = flows.get(tup)
            if fl is None:
                fl = self._new_flow(tup, 0, smac + b"\x00" * 10, et, ZERO16, 0, ts)

        # ---- unit accumulator for (flow, epoch)
        u = fl.unit
        if u is None or fl.unit_epoch != e:
            u = _Acc()
            u.flow = fl
            u.arr = self.n_arr
            self.n_arr += 1
            self.arr2local.append(-1)
            u.first_ts = ts
            u.prev_ts = ts
            u.n = u.pk0 = u.pk1 = u.pb0 = u.pb1 = u.fb0 = u.fb1 = u.s = 0
            u.syn = u.synack = u.fin = u.rst = u.psh = 0
            u.sizes = []
            u.iats = []
            u.frames = [] if self.keep_frames else None
            u.hits = None
            u.s0 = u.d0 = u.s1 = u.d1 = None
            u.sH = 0
            u.payH = 0
            u.Hdone = False
            fl.unit = u
            fl.unit_epoch = e
            self.epoch_units.append(u)
            b = self.buckets.get(e)
            if b is None:
                b = self.buckets[e] = []
            b.append(fl)
        pi = u.n
        u.n = pi + 1
        cost = caplen + 16
        if d == 0:
            u.pk0 += 1
            u.pb0 += plen
            u.fb0 += caplen
            fl.opk += 1
            fl.opay += plen
            if u.s0 is None:
                u.s0, u.d0 = p[P_SMAC], p[P_DMAC]
        else:
            u.pk1 += 1
            u.pb1 += plen
            u.fb1 += caplen
            fl.rpk += 1
            fl.rpay += plen
            if u.s1 is None:
                u.s1, u.d1 = p[P_SMAC], p[P_DMAC]
        u.s += cost
        if not u.Hdone:
            u.sH += cost
            u.payH += plen
            if u.payH >= self.H:
                u.Hdone = True
        if pi < 8:
            u.sizes.append(plen if d == 0 else -plen)
        if 0 < pi <= 8:
            u.iats.append(iat_code(ts - u.prev_ts))
        u.prev_ts = ts
        u.last_ts = ts
        fl.last_ts = ts
        if flags >= 0:
            if flags & F_SYN:
                if flags & F_ACK:
                    u.synack += 1
                else:
                    u.syn += 1
                    if fl.opk + fl.rpk == 1:
                        fl.o_first_syn = True
            if flags & F_FIN:
                u.fin += 1
            if flags & F_RST:
                u.rst += 1
            if flags & F_PSH:
                u.psh += 1
            _tcp_update(fl, d, flags)
        if u.frames is not None:
            u.frames.append((cnt, ts, d, caplen, wirelen, frame, p[P_L3OFF], p[P_ETYPE]))
        if plen and fl.dets:
            pc = frame[p[P_PAYOFF]:p[P_CAPEND]]
            if pc:
                hits = det.run_packet(fl.dets, pc, d, fl.proto == 6)
                if hits:
                    if u.hits is None:
                        u.hits = {}
                    for rid, v in hits:
                        h = u.hits.get(rid)
                        if h is None:
                            u.hits[rid] = [v, 1, pi]
                        else:
                            h[0] += v
                            h[1] += 1
        if self.track:
            self.pk_arr.append(u.arr)
            self.pk_idx.append(pi)
            self.pk_ts.append(ts)
            self.pk_wire.append(wirelen)
            sm = p[P_SMAC]
            dm = p[P_DMAC]
            self.pk_smac.append(self._mac_idx(sm) if sm is not None else -1)
            self.pk_dmac.append(self._mac_idx(dm) if dm is not None else -1)

    # ------------------------------------------------------------------ epoch end
    def _device(self, fl: _Flow, u: _Acc, mac_a, mac_b):
        if fl.proto == 0:
            return 0, DC_LAN_LAN, mac_a
        la, lb = fl.lan_a, fl.lan_b
        if la and lb:
            dc, side = DC_LAN_LAN, 0
        elif la:
            dc, side = DC_LAN_WAN, 0
        elif lb:
            dc, side = DC_WAN_LAN, 1
        else:
            dc, side = DC_WAN_WAN, 0
        return side, dc, (mac_a if side == 0 else mac_b)

    def _flush(self) -> None:
        units = self.epoch_units
        e = self.cur
        t_start = self.t0 + e * self.T
        self.epoch_units = []
        units.sort(key=lambda a: (a.first_ts, a.flow.key))
        recs = []
        for i, u in enumerate(units):
            fl = u.flow
            r = UnitRecord()
            r.epoch = e
            r.index = i
            r.local = self.n_units
            self.arr2local[u.arr] = self.n_units
            self.n_units += 1
            r.key = fl.key
            r.proto = fl.proto
            r.port_a, r.port_b = fl.port_a, fl.port_b
            mac_a = u.s0 if u.s0 is not None else u.d1
            mac_b = u.s1 if u.s1 is not None else u.d0
            r.mac_a, r.mac_b = mac_a, mac_b
            side, dc, dmac = self._device(fl, u, mac_a, mac_b)
            if dmac is None:                       # MAC-less capture: device keyed by the LAN-side IP
                dmac = ("ip", fl.key[1:17] if side == 0 else fl.key[19:35])
            dev = self.mac_table.get(dmac)
            if dev is None:
                dev = self.mac_table[dmac] = len(self.mac_table)
                self.mac_first.append((dmac, e))
            r.device_id, r.device_mac, r.lan_side, r.dir_class = dev, dmac, side, dc
            r.first_ts_us, r.last_ts_us = u.first_ts, u.last_ts
            r.flow_first_ts_us = fl.first_ts
            r.pkts = (u.pk0, u.pk1)
            r.payload_bytes = (u.pb0, u.pb1)
            r.frame_bytes = (u.fb0, u.fb1)
            r.n_syn, r.n_synack, r.n_fin, r.n_rst, r.n_psh = u.syn, u.synack, u.fin, u.rst, u.psh
            r.conn_state = conn_state(fl)
            r.sizes = u.sizes
            r.iats = u.iats
            r.s = u.s
            r.sH = u.sH
            pay = (u.pb0 + u.pb1) > 0
            r.pay = pay
            r.fps = pay and not fl.pay_seen
            if pay:
                fl.pay_seen = True
            r.continuation = fl.nseg > 0
            fl.nseg += 1
            r.flow_uid = fl.uid
            r.meta = enc_metadata({
                "key": fl.key, "device_id": dev, "first_ts_us": u.first_ts, "last_ts_us": u.last_ts,
                "pkts": r.pkts, "payload_bytes": r.payload_bytes, "frame_bytes": r.frame_bytes,
                "n_syn": u.syn, "n_synack": u.synack, "n_fin": u.fin, "n_rst": u.rst, "n_psh": u.psh,
                "conn_state": r.conn_state, "sizes": u.sizes, "iats": u.iats, "s": u.s, "flow_uid": fl.uid,
                "continuation": r.continuation, "fps": r.fps, "pay": pay})
            r.raw_hits = u.hits
            if pay:
                r.rid = self.n_pay
                self.n_pay += 1
                if u.frames is not None:
                    leaves = [hs.frame_leaf(f[1], f[2], f[3], f[5]) for f in u.frames]
                    r.pkt_leaves = leaves
                    r.root_pkt = hs.merkle_root(leaves)
                    r.leaf = hs.unit_leaf(r.meta, r.root_pkt)
                else:
                    r.pkt_leaves = r.root_pkt = None
                    r.leaf = b""
                r.frames = u.frames
            else:
                r.rid = -1
                r.pkt_leaves = r.root_pkt = None
                r.leaf = hs.unit_leaf0(r.meta)
                r.frames = None
            fl.unit = None
            recs.append(r)
        if self.frag:
            self.frag = {k: v for k, v in self.frag.items() if v[2] >= e - 1}
        if self.on_epoch is not None:
            self.on_epoch(e, t_start, recs)
        for r in recs:
            r.frames = None

    def _expire(self, e_new: int) -> None:
        """Before epoch ``e_new`` starts: drop flows idle for K full epochs (bucket = last active epoch)."""
        lim_o = e_new - 1 - K_OTHER
        lim_t = e_new - 1 - K_TCP
        for be in sorted(k for k in self.buckets if k <= lim_o):
            b = self.buckets.pop(be)
            keep = []
            for fl in b:
                if fl.unit_epoch != be:              # active in a later epoch (sits in a later bucket)
                    continue
                if fl.proto == 6 and be > lim_t:
                    keep.append(fl)
                    continue
                if self.flows.get(fl.tup) is fl:
                    del self.flows[fl.tup]
                    self.qa["n_expired"] += 1
            if keep:
                self.buckets[be] = keep

    def finish(self) -> None:
        if self.t0 is not None and self.epoch_units:
            self._flush()

    @property
    def n_epochs(self) -> int:
        return 0 if self.t0 is None else self.cur + 1


def batches_from_pcap(path, *, max_packets=None, keep_frames=True, run_detectors=False, T_us=T_US):
    """In-memory path: yield ``EpochBatch`` objects (with frames for payload units)."""
    from ctsample.io.pcap_reader import PcapReader, parser_for
    out: list = []

    def on_epoch(e, t0, recs):
        out.append(EpochBatch(epoch=e, t_start_us=t0, units=[r.to_unit(keep_frames) for r in recs]))

    seg = FlowSegmenter(on_epoch, keep_frames=keep_frames, run_detectors=run_detectors, track_packets=False,
                        T_us=T_us)
    rd = PcapReader(path, max_packets=max_packets)
    parse = parser_for(rd.linktype)
    for cnt, ts, cl, wl, fr in rd:
        seg.add(cnt, ts, cl, wl, fr, parse(fr, wl))
        while out:
            yield out.pop(0)
    seg.finish()
    while out:
        yield out.pop(0)
