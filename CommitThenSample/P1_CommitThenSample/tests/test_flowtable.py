"""Flow table / segmenter (units/flowtable.py) and the pcap reader (io/pcap_reader.py) on synthetic packets."""
import struct

import dpkt
import pytest

from ctsample.content.replay import PcapWriter
from ctsample.io.pcap_reader import PcapReader, parse_ether, parser_for
from ctsample.units import _hash_shim as hs
from ctsample.units.batch import EpochBatch
from ctsample.units.encode import CONN_STATES, dec_metadata
from ctsample.units.flowtable import DC_LAN_WAN, DC_WAN_LAN, FlowSegmenter, batches_from_pcap

MA, MB, MG = bytes.fromhex("020000000001"), bytes.fromhex("020000000002"), bytes.fromhex("0200000000ff")
A, B, W = bytes([192, 168, 1, 10]), bytes([192, 168, 1, 20]), bytes([93, 184, 216, 34])
T0 = 1_600_000_000_000_000
SYN, ACK, FIN, RST, PSH = 0x02, 0x10, 0x01, 0x04, 0x08


def eth(src, dst, et, payload, pad=True):
    fr = dst + src + et.to_bytes(2, "big") + payload
    return fr + b"\x00" * (60 - len(fr)) if pad and len(fr) < 60 else fr


def ipv4(src, dst, proto, l4, ident=1, ff=0):
    return struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(l4), ident, ff, 64, proto, 0, src, dst) + l4


def tcp(sp, dp, flags, payload=b""):
    return struct.pack("!HHIIBBHHH", sp, dp, 1, 1, 5 << 4, flags, 65535, 0, 0) + payload


def udp(sp, dp, payload):
    return struct.pack("!HHHH", sp, dp, 8 + len(payload), 0) + payload


def tcp_frame(src, dst, smac, dmac, sp, dp, flags, payload=b""):
    return eth(smac, dmac, 0x0800, ipv4(src, dst, 6, tcp(sp, dp, flags, payload)))


def run(pkts, **kw):
    recs, leaves = [], {}

    def on_epoch(e, t0, rs):
        for r in rs:
            if r.frames:
                leaves[r.local] = [hs.frame_leaf(f[1], f[2], f[3], f[5]) for f in r.frames]
        recs.extend(rs)

    seg = FlowSegmenter(on_epoch, **kw)
    for i, (ts, fr) in enumerate(pkts, 1):
        seg.add(i, ts, len(fr), len(fr), fr, parse_ether(fr, len(fr)))
    seg.finish()
    return seg, recs, leaves


def http_session(t, sp=40000, req=b"GET / HTTP/1.1\r\nHost: x\r\n\r\n", resp=b"HTTP/1.1 200 OK\r\n\r\nhello"):
    return [(t, tcp_frame(A, W, MA, MG, sp, 80, SYN)),
            (t + 1000, tcp_frame(W, A, MG, MA, 80, sp, SYN | ACK)),
            (t + 2000, tcp_frame(A, W, MA, MG, sp, 80, ACK)),
            (t + 3000, tcp_frame(A, W, MA, MG, sp, 80, PSH | ACK, req)),
            (t + 4000, tcp_frame(W, A, MG, MA, 80, sp, PSH | ACK, resp)),
            (t + 5000, tcp_frame(A, W, MA, MG, sp, 80, FIN | ACK)),
            (t + 6000, tcp_frame(W, A, MG, MA, 80, sp, FIN | ACK)),
            (t + 7000, tcp_frame(A, W, MA, MG, sp, 80, ACK))]


def test_bidirectional_unit_cost_and_commitment():
    pk = http_session(T0)
    seg, recs, leaves = run(pk)
    assert len(recs) == 1
    r = recs[0]
    assert r.proto == 6 and r.port_a == 40000 and r.port_b == 80 and r.key[1:17][-4:] == A
    assert r.pkts == (5, 3) and r.payload_bytes == (len(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n"), 24)
    assert r.s == sum(len(f) + 16 for _, f in pk)
    assert r.pay and r.fps and not r.continuation
    assert CONN_STATES[r.conn_state] == "SF"
    assert r.n_syn == 1 and r.n_synack == 1 and r.n_fin == 2 and r.n_psh == 2
    assert r.dir_class == DC_LAN_WAN and r.device_mac == MA and r.device_id == 0
    assert r.pkt_leaves == leaves[r.local] and r.root_pkt == hs.merkle_root(leaves[r.local])
    assert r.leaf == hs.unit_leaf(r.meta, r.root_pkt)
    assert r.flow_uid == hs.flow_uid(r.key, T0)
    m = dec_metadata(r.meta)
    assert m["pkts"] == (5, 3) and m["s"] == r.s and m["fps"] and m["sizes"][3] > 0 and m["sizes"][4] == -24
    assert seg.qa["n_packets"] == 8


def test_epoch_split_continuation_and_fps():
    pk = [(T0, tcp_frame(A, W, MA, MG, 40001, 80, SYN)),
          (T0 + 1000, tcp_frame(W, A, MG, MA, 80, 40001, SYN | ACK)),
          (T0 + 29_000_000, tcp_frame(A, W, MA, MG, 40001, 80, ACK)),
          (T0 + 31_000_000, tcp_frame(W, A, MG, MA, 80, 40001, PSH | ACK, b"x" * 10)),   # responder first in e1
          (T0 + 65_000_000, tcp_frame(A, W, MA, MG, 40001, 80, PSH | ACK, b"y" * 5))]
    _, recs, _ = run(pk)
    assert [r.epoch for r in recs] == [0, 1, 2]
    assert len({r.flow_uid for r in recs}) == 1
    assert [r.continuation for r in recs] == [False, True, True]
    assert [r.pay for r in recs] == [False, True, True]
    assert [r.fps for r in recs] == [False, True, False]
    assert recs[1].port_a == 40001 and recs[1].pkts == (0, 1)        # orientation kept from the flow start
    assert recs[1].leaf == hs.unit_leaf(recs[1].meta, recs[1].root_pkt)
    assert recs[0].leaf == hs.unit_leaf0(recs[0].meta) and recs[0].root_pkt is None


def test_zero_payload_padding_and_states():
    pk = [(T0, tcp_frame(A, W, MA, MG, 41000, 23, SYN)),                         # S0 scan, padded to 60 B
          (T0 + 10, tcp_frame(A, B, MA, MB, 41001, 23, SYN)),
          (T0 + 20, tcp_frame(B, A, MB, MA, 23, 41001, RST | ACK))]              # REJ
    _, recs, _ = run(pk)
    st = {r.port_a: CONN_STATES[r.conn_state] for r in recs}
    assert st == {41000: "S0", 41001: "REJ"}
    assert all(not r.pay and r.payload_bytes == (0, 0) for r in recs)
    assert all(len(r.meta) == 68 for r in recs)
    assert [r.first_ts_us for r in recs] == sorted(r.first_ts_us for r in recs)
    assert [r.index for r in recs] == [0, 1]


def test_l2_icmp_other_keys():
    arp = eth(MA, b"\xff" * 6, 0x0806, b"\x00\x01\x08\x00\x06\x04\x00\x01" + MA + A + b"\x00" * 6 + B)
    echo_req = eth(MA, MB, 0x0800, ipv4(A, B, 1, b"\x08\x00\x00\x00\x00\x01\x00\x01" + b"p" * 32))
    echo_rep = eth(MB, MA, 0x0800, ipv4(B, A, 1, b"\x00\x00\x00\x00\x00\x01\x00\x01" + b"p" * 32))
    igmp = eth(MA, MB, 0x0800, ipv4(A, B, 2, b"\x16\x00\x00\x00" + bytes([224, 0, 0, 251])))
    _, recs, _ = run([(T0, arp), (T0 + 1, echo_req), (T0 + 2, echo_rep), (T0 + 3, igmp), (T0 + 4, arp)])
    by = {(r.proto, r.port_a): r for r in recs}
    assert set(by) == {(0, 0x0806), (1, 8), (1, 0), (2, 0)}
    assert by[(0, 0x0806)].pkts == (2, 0) and by[(0, 0x0806)].key[1:7] == MA
    assert all(not r.pay for r in recs)                                           # ICMP data is not transport payload


def test_ip_fragments_join_flow():
    data = b"z" * 300
    full = udp(5000, 6000, data)
    f1 = eth(MA, MG, 0x0800, ipv4(A, W, 17, full[:104], ident=77, ff=0x2000), pad=False)   # MF, offset 0
    f2 = eth(MA, MG, 0x0800, ipv4(A, W, 17, full[104:], ident=77, ff=104 // 8), pad=False)
    seg, recs, _ = run([(T0, f1), (T0 + 5, f2)])
    assert len(recs) == 1 and recs[0].pkts == (2, 0) and recs[0].port_a == 5000
    assert recs[0].payload_bytes[0] == 300 and seg.qa["n_frag"] == 2 and seg.qa["n_frag_unmatched"] == 0


def test_tcp_reuse_and_udp_expiry():
    s1 = http_session(T0, sp=42000)
    s2 = http_session(T0 + 95_000_000, sp=42000)                     # same 5-tuple, epoch 3, after FIN/FIN
    u = [(T0 + 10, eth(MA, MG, 0x0800, ipv4(A, W, 17, udp(123, 123, b"n" * 48)))),
         (T0 + 61_000_000, eth(MA, MG, 0x0800, ipv4(A, W, 17, udp(123, 123, b"n" * 48)))),      # epoch 2: same
         (T0 + 155_000_000, eth(MA, MG, 0x0800, ipv4(A, W, 17, udp(123, 123, b"n" * 48))))]     # epoch 5: new
    _, recs, _ = run(sorted(s1 + s2 + u))
    t = [r for r in recs if r.proto == 6]
    assert len(t) == 2 and t[0].flow_uid != t[1].flow_uid and not t[1].continuation and t[1].fps
    v = [r for r in recs if r.proto == 17]
    assert [r.epoch for r in v] == [0, 2, 5]
    assert v[0].flow_uid == v[1].flow_uid != v[2].flow_uid
    assert [r.continuation for r in v] == [False, True, False]


def test_device_direction_classes_and_late_packets():
    pk = [(T0, tcp_frame(W, A, MG, MA, 50000, 22, SYN)),                       # WAN -> LAN: device = responder
          (T0 + 40_000_000, tcp_frame(A, W, MA, MG, 40000, 443, SYN)),
          (T0 + 25_000_000, tcp_frame(A, W, MA, MG, 40002, 443, SYN))]       # late (previous epoch)
    seg, recs, _ = run(pk)
    assert recs[0].dir_class == DC_WAN_LAN and recs[0].lan_side == 1 and recs[0].device_mac == MA
    assert seg.qa["n_late"] == 1 and recs[-1].epoch == 1 and len(recs) == 3


def _write_pcap(path, pkts, nsec=False):
    with open(path, "wb") as f:
        w = dpkt.pcap.Writer(f, nano=nsec)
        for ts, fr in pkts:
            w.writepkt(fr, ts=ts / 1e6)


def test_reader_formats_roundtrip(tmp_path):
    pk = http_session(T0)
    p1 = tmp_path / "a.pcap"
    w = PcapWriter(p1)
    for ts, fr in pk:
        w.write(ts, fr, len(fr))
    w.close()
    rd = PcapReader(p1)
    got = list(rd)
    assert [(g[1], g[4]) for g in got] == pk and rd.qa["stop_reason"] == "eof" and [g[0] for g in got] == list(range(1, 9))
    p2 = tmp_path / "b.pcapng"
    with open(p2, "wb") as f:
        w2 = dpkt.pcapng.Writer(f)
        for ts, fr in pk:
            w2.writepkt(fr, ts=ts / 1e6)
    got2 = list(PcapReader(p2))
    assert [g[4] for g in got2] == [fr for _, fr in pk]
    assert all(abs(g[1] - ts) <= 1 for g, (ts, _) in zip(got2, pk))
    # truncated tail
    raw = p1.read_bytes()
    p3 = tmp_path / "c.pcap"
    p3.write_bytes(raw[:-10])
    rd3 = PcapReader(p3)
    assert len(list(rd3)) == 7 and rd3.qa["stop_reason"] == "truncated_tail"
    # corrupt record header mid-file
    p4 = tmp_path / "d.pcap"
    p4.write_bytes(raw[:24 + 16 + len(pk[0][1])] + struct.pack("<IIII", 1, 0, 10 ** 8, 10 ** 8) + raw[60:])
    rd4 = PcapReader(p4)
    assert len(list(rd4)) == 1 and rd4.qa["stop_reason"] == "corrupt_record"


def test_parse_sll_and_vlan():
    ip = ipv4(A, W, 6, tcp(1, 2, PSH | ACK, b"abc"))
    sll = struct.pack("!HHH8sH", 4, 1, 6, MA + b"\x00\x00", 0x0800) + ip
    p = parser_for(113)(sll, len(sll))
    assert p[0] == 1 and p[1] == MA and p[12] == 3 and sll[p[11]:p[13]] == b"abc"
    vl = MG + MA + b"\x81\x00\x00\x05" + b"\x08\x00" + ip
    q = parse_ether(vl, len(vl))
    assert q[4] == 18 and q[12] == 3


def test_in_memory_batches(tmp_path):
    p = tmp_path / "s.pcap"
    _write_pcap(p, http_session(T0) + http_session(T0 + 40_000_000, sp=40010))
    bs = list(batches_from_pcap(p))
    assert all(isinstance(b, EpochBatch) for b in bs) and [b.epoch for b in bs] == [0, 1]
    u = bs[0].units[0]
    assert u.pay and u.frames and len(u.frames) == 8 and u.leaf == hs.unit_leaf(u.meta, u.root_pkt)
    assert bs[1].t_start_us == T0 + 30_000_000


def test_payload_capture_slice():
    fr = tcp_frame(A, W, MA, MG, 1, 2, PSH | ACK, b"hello")             # 59 B + 1 B Ethernet padding
    p = parse_ether(fr, len(fr))
    assert len(fr) == 60 and p[12] == 5 and fr[p[11]:p[13]] == b"hello"   # padding is not payload
    fr = tcp_frame(A, W, MA, MG, 1, 2, PSH | ACK, b"hello world!")
    trunc = fr[:-2]                                                     # snaplen-truncated capture
    p2 = parse_ether(trunc, len(fr))
    assert p2[12] == 12 and trunc[p2[11]:p2[13]] == b"hello worl"


@pytest.mark.parametrize("n", [1, 3])
def test_canonical_order(n):
    pk = []
    for i in range(n):
        pk += [(T0 + 100 - i, tcp_frame(A, W, MA, MG, 30000 + i, 80, SYN))]
    _, recs, _ = run(sorted(pk))
    assert [r.first_ts_us for r in recs] == sorted(r.first_ts_us for r in recs)
    assert [r.index for r in recs] == list(range(n))


def _dpkt_frame(src, dst, sp, dp, payload, proto="tcp"):
    l4 = (dpkt.tcp.TCP(sport=sp, dport=dp, flags=PSH | ACK, data=payload) if proto == "tcp"
          else dpkt.udp.UDP(sport=sp, dport=dp, data=payload))
    ip = dpkt.ip.IP(src=src, dst=dst, p=6 if proto == "tcp" else 17, data=l4)
    ip.len = len(bytes(ip))
    return bytes(dpkt.ethernet.Ethernet(src=MA, dst=MG, type=0x0800, data=ip))


def test_merge_rewrite_and_checksums(tmp_path):
    from ctsample.io.merge import SourceSpec, merged, rewrite_frame
    a = [(T0 + 10, _dpkt_frame(A, W, 1000, 80, b"x" * 30)), (T0 + 30, _dpkt_frame(W, A, 80, 1000, b"y" * 5))]
    b = [(T0 + 20 - 5_000_000, _dpkt_frame(B, W, 2000, 53, b"q" * 20, "udp"))]
    _write_pcap(tmp_path / "a.pcap", a)
    _write_pcap(tmp_path / "b.pcap", b)
    NEW, NEWMAC = bytes([192, 168, 137, 241]), bytes.fromhex("020000000001")
    specs = [SourceSpec(tmp_path / "a.pcap", ip_map={A: NEW}, mac_map={MA: bytes.fromhex("02000000000a")}),
             SourceSpec(tmp_path / "b.pcap", offset_us=5_000_000)]
    out = list(merged(specs))
    assert [x[1] for x in out] == [T0 + 10, T0 + 20, T0 + 30] and [x[5] for x in out] == [0, 1, 0]
    assert [x[0] for x in out] == [1, 2, 3]
    for n, ts, cl, wl, fr, i, c in out:
        e = dpkt.ethernet.Ethernet(fr)
        ip = e.data
        if i == 0:
            assert NEW in (bytes(ip.src), bytes(ip.dst)) and A not in (bytes(ip.src), bytes(ip.dst))
            assert bytes(e.src) == bytes.fromhex("02000000000a") or bytes(e.dst) == bytes.fromhex("02000000000a")
        s_ip, s_l4 = ip.sum, ip.data.sum
        ip.sum = 0
        ip.data.sum = 0
        re_ = dpkt.ip.IP(bytes(ip))                       # dpkt recomputes both checksums on serialisation
        assert re_.sum == s_ip and re_.data.sum == s_l4
    fr = a[0][1]
    assert rewrite_frame(fr, 1, parse_ether(fr, len(fr)), {}, {}) is fr
