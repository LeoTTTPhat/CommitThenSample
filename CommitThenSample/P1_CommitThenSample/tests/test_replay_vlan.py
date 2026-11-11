"""QinQ per-unit replay (content/replay.py) end to end through Suricata (DESIGN §5.3 item 4, §5.4).

Six units share two 5-tuples: three separate TCP connections on the same 5-tuple in epochs
0, 1, 2, and one long connection whose three 30-s segments are three units (the later two are
mid-stream). Suricata (project yaml + the DESIGN flags, a one-rule file) must report six flows,
one per VLAN pair, and every alert must map back to the unit that carried the matching bytes.
"""
from pathlib import Path

import numpy as np
import pytest

from ctsample.content import replay as rp
from ctsample.content.mapping import map_vlans
from ctsample.content.suricata import SURICATA, YAML, flow_vlans, parse_eve_alerts, run_suricata
from ctsample.io.pcap_reader import LT_ETHERNET, LT_SLL, PcapReader, parse_ether, parser_for
import struct

MA, MG = bytes.fromhex("020000000001"), bytes.fromhex("0200000000ff")
A, W = bytes([192, 168, 1, 10]), bytes([93, 184, 216, 34])
T0 = 1_600_000_000_000_000
SYN, ACK, FIN, PSH = 0x02, 0x10, 0x01, 0x08


def tcp_frame(src, dst, smac, dmac, sp, dp, flags, payload=b""):
    l4 = struct.pack("!HHIIBBHHH", sp, dp, 1, 1, 5 << 4, flags, 65535, 0, 0) + payload
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(l4), 1, 0, 64, 6, 0, src, dst) + l4
    fr = dmac + smac + b"\x08\x00" + ip
    return fr + b"\x00" * (60 - len(fr)) if len(fr) < 60 else fr


def test_vlan_codec():
    for u in (0, 1, 4093, 4094, 4095, 123456, rp.MAX_UNITS - 1):
        o, i = rp.vlan_pair(u)
        assert 1 <= o <= 4094 and 1 <= i <= 4094 and rp.unit_from_vlans(o, i) == u
    assert rp.vlan_pair(4094) == (2, 1)
    with pytest.raises(ValueError):
        rp.vlan_pair(rp.MAX_UNITS)
    with pytest.raises(ValueError):
        rp.unit_from_vlans(0, 5)
    gbr = np.array([100, 101, 102], np.int64)
    assert list(map_vlans([1, 1, 1, 9], [1, 3, 4, 1], gbr)) == [100, 102, -1, -1]


def test_tag_frame_ethernet_and_sll():
    fr = tcp_frame(A, W, MA, MG, 1, 2, PSH | ACK, b"payload!")
    p = parse_ether(fr, len(fr))
    nf = rp.tag_frame(fr, LT_ETHERNET, p[4], p[3], 0, 5000)
    assert nf[:12] == fr[:12] and nf[12:14] == b"\x81\x00" and nf[16:18] == b"\x81\x00"
    assert int.from_bytes(nf[14:16], "big") == 2 and int.from_bytes(nf[18:20], "big") == 5000 - 4094 + 1
    q = parse_ether(nf, len(nf))
    assert q[4] == 22 and nf[q[11]:q[13]] == b"payload!"
    retag = rp.tag_frame(nf, LT_ETHERNET, q[4], q[3], 0, 7)       # existing tags are replaced, not stacked
    assert len(retag) == len(nf) and parse_ether(retag, len(retag))[4] == 22
    sll = b"\x00\x04\x00\x01\x00\x06" + MA + b"\x00\x00" + b"\x08\x00" + fr[14:]
    s = parser_for(LT_SLL)(sll, len(sll))
    nf2 = rp.tag_frame(sll, LT_SLL, s[4], s[3], 1, 3)
    assert nf2[:6] == rp.MAC_A and nf2[6:12] == rp.MAC_B and parse_ether(nf2, len(nf2))[12] == 8


def _sessions():
    pk = []
    for k in range(3):                                              # same 5-tuple, three connections
        t = T0 + k * 30_000_000 + 1_000_000
        pk += [(t, tcp_frame(A, W, MA, MG, 40000, 8080, SYN)),
               (t + 100, tcp_frame(W, A, MG, MA, 8080, 40000, SYN | ACK)),
               (t + 200, tcp_frame(A, W, MA, MG, 40000, 8080, ACK)),
               (t + 300, tcp_frame(A, W, MA, MG, 40000, 8080, PSH | ACK, b"hello P1BQINQ-%d end\r\n" % k)),
               (t + 400, tcp_frame(W, A, MG, MA, 8080, 40000, ACK)),
               (t + 500, tcp_frame(A, W, MA, MG, 40000, 8080, FIN | ACK)),
               (t + 600, tcp_frame(W, A, MG, MA, 8080, 40000, FIN | ACK)),
               (t + 700, tcp_frame(A, W, MA, MG, 40000, 8080, ACK))]
    t = T0 + 2_000_000                                              # one connection, three segments
    pk += [(t, tcp_frame(A, W, MA, MG, 40001, 8080, SYN)),
           (t + 100, tcp_frame(W, A, MG, MA, 8080, 40001, SYN | ACK)),
           (t + 200, tcp_frame(A, W, MA, MG, 40001, 8080, ACK))]
    for k in range(3):
        tk = T0 + k * 30_000_000 + 5_000_000
        pk += [(tk, tcp_frame(A, W, MA, MG, 40001, 8080, PSH | ACK, b"data P1BQINQ-%d end\r\n" % (10 + k))),
               (tk + 100, tcp_frame(W, A, MG, MA, 8080, 40001, ACK))]
    return sorted(pk)


@pytest.mark.skipif(not (Path(SURICATA).exists() and Path(YAML).exists()), reason="suricata or yaml missing")
def test_suricata_sees_one_flow_per_unit(tmp_path):
    pk = _sessions()
    items, units = [], {}

    def on_epoch(e, t0, recs):
        for r in recs:
            if r.pay:
                units[r.rid] = r
                for f in r.frames:
                    items.append((f[0], f[1], f[5], f[4], f[3], f[6], f[7], f[2], r.rid, False))

    from ctsample.units.flowtable import FlowSegmenter
    seg = FlowSegmenter(on_epoch)
    for i, (ts, fr) in enumerate(pk, 1):
        seg.add(i, ts, len(fr), len(fr), fr, parse_ether(fr, len(fr)))
    seg.finish()
    assert len(units) == 6 and len({(u.port_a, u.port_b) for u in units.values()}) == 2
    w = rp.ReplayWriter(tmp_path / "r.pcap", None, LT_ETHERNET)
    w.write_epoch(items)
    w.close()
    assert sum(1 for _ in PcapReader(tmp_path / "r.pcap")) == len(items)
    rules = tmp_path / "t.rules"
    rules.write_text('alert tcp any any -> any any (msg:"P1B qinq test"; flow:to_server; '
                     'content:"P1BQINQ-"; pcre:"/P1BQINQ-\\d+ end/"; sid:9900001; rev:1;)\n')
    r = run_suricata(tmp_path / "r.pcap", tmp_path / "out", rules)
    assert r["rc"] == 0, (tmp_path / "out" / "stdout.txt").read_text()[-2000:]
    eve = tmp_path / "out" / "eve.json"
    fl = flow_vlans(eve)
    assert sorted(fl) == sorted(rp.vlan_pair(u) for u in units)          # one flow per unit
    cols, _ = parse_eve_alerts(eve)
    assert len(cols["sid"]) == 6 and set(cols["sid"]) == {9900001}
    gbr = np.arange(len(units), dtype=np.int64) + 1000                   # rid -> pseudo gid
    gids = map_vlans(cols["vlan_outer"], cols["vlan_inner"], gbr)
    assert sorted(gids.tolist()) == list(range(1000, 1006))
    # packet alerts: the replay record at pcap_cnt carries the alert's VLAN pair and lies inside the unit;
    # stream alerts raised at flow end (pseudo packets, no pcap_cnt) are mapped by their flow's VLAN pair
    recs = {c: fr for c, ts, cl, wl, fr in PcapReader(tmp_path / "r.pcap")}
    n_pkt_alerts = 0
    for g, c, o, i in zip(gids, cols["pcap_cnt"], cols["vlan_outer"], cols["vlan_inner"]):
        if c >= 1:
            fr = recs[c]
            assert (int.from_bytes(fr[14:16], "big"), int.from_bytes(fr[18:20], "big")) == (o, i)
            n_pkt_alerts += 1
    assert n_pkt_alerts >= 3
