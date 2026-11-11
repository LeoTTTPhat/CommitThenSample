#!/usr/bin/env python3
"""Map Suricata eve.json alerts back to pcap packets and flows (and verify the mapping).

    python tools/eve_alert_map.py EVE.json PCAP [--out alerts_mapped.csv] [--flow-packets flow_packets.csv]

* alert -> packet: ``pcap_cnt`` is the 1-based index of the packet (record) in the pcap on which the alert
  fired. For stream/app-layer matches (``stream: 1``) that is the packet that triggered inspection
  (often the ACK after the payload or a later packet of the flow), not necessarily the packet that
  carried the matching bytes; the matched bytes are in ``payload`` / ``payload_printable``.
* alert -> flow: ``flow_id`` joins to the eve ``flow`` record (start/end/pkts/bytes/state);
  ``community_id`` (Community ID v1, seed 0) is also computed here for every pcap packet, which assigns
  *all* packets of a flow to it (within the flow's [start, end] window).
The script streams both files (constant memory) and prints a verification summary: the packet at
``pcap_cnt`` must have the alert's 5-tuple (either direction), the alert timestamp, and community_id.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import socket
import struct
import sys
from datetime import datetime

import dpkt

ICMP4_PAIR = {8: 0, 0: 8, 13: 14, 14: 13, 15: 16, 16: 15, 10: 9, 9: 10, 17: 18, 18: 17}
ICMP6_PAIR = {128: 129, 129: 128, 133: 134, 134: 133, 135: 136, 136: 135, 130: 131, 131: 130, 144: 145, 145: 144}


def community_id(src: bytes, dst: bytes, proto: int, sport: int, dport: int, seed: int = 0) -> str:
    one_way = False
    if proto in (1, 58):  # ICMP / ICMPv6: ports are (type, code) with request/reply pairing
        pairs = ICMP4_PAIR if proto == 1 else ICMP6_PAIR
        if sport in pairs:
            dport = pairs[sport]
        else:
            one_way = True
    if not one_way and (src, sport) > (dst, dport):
        src, dst, sport, dport = dst, src, dport, sport
    data = struct.pack("!H", seed) + src + dst + struct.pack("!BB", proto, 0)
    if proto in (6, 17, 132, 1, 58):
        data += struct.pack("!HH", sport, dport)
    return "1:" + base64.b64encode(hashlib.sha1(data).digest()).decode()


def parse_ts(s: str) -> float:
    if len(s) > 5 and s[-5] in "+-" and s[-3] != ":":  # +0700 -> +07:00
        s = s[:-2] + ":" + s[-2:]
    return datetime.fromisoformat(s).timestamp()


def decode(buf: bytes, linktype: int):
    """Return (src_ip_str, dst_ip_str, src_b, dst_b, proto, sport, dport) or None."""
    try:
        if linktype == 1:
            ip = dpkt.ethernet.Ethernet(buf).data
        elif linktype == 113:
            ip = dpkt.sll.SLL(buf).data
        elif linktype in (101, 12, 228, 229):
            ip = dpkt.ip.IP(buf) if (buf[0] >> 4) == 4 else dpkt.ip6.IP6(buf)
        elif linktype == 0:  # BSD loopback
            ip = dpkt.loopback.Loopback(buf).data
        else:
            return None
    except (dpkt.UnpackError, IndexError, struct.error):
        return None
    if isinstance(ip, dpkt.ip.IP):
        fam, proto, l4 = socket.AF_INET, ip.p, ip.data
    elif isinstance(ip, dpkt.ip6.IP6):
        fam, proto, l4 = socket.AF_INET6, ip.nxt, ip.data
    else:
        return None
    sport = dport = 0
    if isinstance(l4, (dpkt.tcp.TCP, dpkt.udp.UDP)):
        sport, dport = l4.sport, l4.dport
    elif isinstance(l4, (dpkt.icmp.ICMP, dpkt.icmp6.ICMP6)):
        sport, dport = l4.type, l4.code
    return (socket.inet_ntop(fam, ip.src), socket.inet_ntop(fam, ip.dst), bytes(ip.src), bytes(ip.dst),
            proto, sport, dport)


def open_pcap(f):
    try:
        return dpkt.pcap.Reader(f)
    except ValueError:
        f.seek(0)
        return dpkt.pcapng.Reader(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("eve")
    ap.add_argument("pcap")
    ap.add_argument("--out", help="CSV with one row per alert (+ packet verification columns)")
    ap.add_argument("--flow-packets", help="CSV pcap_cnt,flow_id for every packet of an alerted flow")
    a = ap.parse_args()

    alerts, flows = [], {}
    with open(a.eve) as f:
        for line in f:
            if '"event_type":"alert"' not in line and '"event_type":"flow"' not in line:
                continue
            r = json.loads(line)
            if r["event_type"] == "alert":
                al = r["alert"]
                alerts.append({"pcap_cnt": r.get("pcap_cnt"), "ts": r["timestamp"], "flow_id": r.get("flow_id"),
                               "community_id": r.get("community_id"), "src_ip": r.get("src_ip"),
                               "src_port": r.get("src_port"), "dest_ip": r.get("dest_ip"),
                               "dest_port": r.get("dest_port"), "proto": r.get("proto"),
                               "app_proto": r.get("app_proto"), "stream": r.get("stream"), "tx_id": r.get("tx_id"),
                               "sid": al["signature_id"], "rev": al.get("rev"), "signature": al["signature"],
                               "category": al.get("category"), "severity": al.get("severity"),
                               "mitre_technique_id": ";".join(al.get("metadata", {}).get("mitre_technique_id", []))})
            else:
                fl = r["flow"]
                flows[r["flow_id"]] = {"community_id": r.get("community_id"), "start": parse_ts(fl["start"]),
                                       "end": parse_ts(fl["end"]), "pkts": fl["pkts_toserver"] + fl["pkts_toclient"]}
    want = {}
    for i, x in enumerate(alerts):
        if x["pcap_cnt"]:
            want.setdefault(x["pcap_cnt"], []).append(i)
    alerted = {x["flow_id"]: flows.get(x["flow_id"]) for x in alerts if x["flow_id"] in flows}
    by_cid = {}
    for fid, fl in alerted.items():
        by_cid.setdefault(fl["community_id"], []).append((fid, fl))
    fp_out = open(a.flow_packets, "w", newline="") if a.flow_packets else None
    fpw = csv.writer(fp_out) if fp_out else None
    if fpw:
        fpw.writerow(["pcap_cnt", "ts", "flow_id"])
    flow_pkt_count = {fid: 0 for fid in alerted}

    n = 0
    with open(a.pcap, "rb") as f:
        rd = open_pcap(f)
        lt = rd.datalink()
        for n, (ts, buf) in enumerate(rd, start=1):
            if n not in want and not by_cid:
                continue
            d = decode(buf, lt)
            if d is None:
                continue
            cid = community_id(d[2], d[3], d[4], d[5], d[6])
            for fid, fl in by_cid.get(cid, ()):
                if fl["start"] - 1e-6 <= ts <= fl["end"] + 1e-6:
                    flow_pkt_count[fid] += 1
                    if fpw:
                        fpw.writerow([n, f"{ts:.6f}", fid])
            for i in want.get(n, ()):
                x = alerts[i]
                x["pkt_ts"] = float(ts)
                x["pkt_src"], x["pkt_dst"], x["pkt_sport"], x["pkt_dport"] = d[0], d[1], d[5], d[6]
                x["pkt_community_id"] = cid
                same = {(d[0], d[5]), (d[1], d[6])} == {(x["src_ip"], x["src_port"]), (x["dest_ip"], x["dest_port"])}
                x["tuple_ok"] = same
                x["ts_ok"] = abs(parse_ts(x["ts"]) - float(ts)) < 1e-5
                x["cid_ok"] = (cid == x["community_id"])
    if fp_out:
        fp_out.close()

    for x in alerts:
        fl = flows.get(x["flow_id"])
        if fl:
            x["flow_start"], x["flow_end"], x["flow_pkts_eve"] = fl["start"], fl["end"], fl["pkts"]
            x["flow_pkts_pcap"] = flow_pkt_count.get(x["flow_id"])
    ok = sum(1 for x in alerts if x.get("tuple_ok") and x.get("ts_ok") and x.get("cid_ok"))
    no_cnt = sum(1 for x in alerts if not x["pcap_cnt"])
    print(f"pcap packets: {n}; alerts: {len(alerts)} (without pcap_cnt: {no_cnt}); flows in eve: {len(flows)}; "
          f"alerted flows: {len(alerted)}")
    print(f"alerts whose pcap_cnt packet matches 5-tuple + timestamp + community_id: {ok}/{len(alerts) - no_cnt}")
    mism = [(fid, flows[fid]["pkts"], c) for fid, c in flow_pkt_count.items() if flows[fid]["pkts"] != c]
    print(f"alerted flows whose pcap packet count (community_id + time window) equals eve flow pkts: "
          f"{len(flow_pkt_count) - len(mism)}/{len(flow_pkt_count)}" + (f"; first mismatches {mism[:5]}" if mism else ""))
    if a.out:
        cols = list(dict.fromkeys(k for x in alerts for k in x))
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(alerts)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
