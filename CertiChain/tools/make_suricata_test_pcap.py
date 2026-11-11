#!/usr/bin/env python3
"""Write a tiny deterministic pcap for smoke-testing Suricata + ET Open.

    python tools/make_suricata_test_pcap.py OUT.pcap

Contents (full TCP sessions: handshake, data, ACKs, FIN):
  flow A  HTTP  192.168.1.66:40001 -> 192.168.1.20:80   GET with a UNION SELECT SQL-injection URI (internal attacker,
                                                          only alerts when EXTERNAL_NET is "any")
  flow B  HTTP  203.0.113.50:40002 -> 192.168.1.20:80   same request from an external (TEST-NET-3) address
  flow C  TELNET 203.0.113.50:40003 -> 192.168.1.30:23  banner, "login: root" echo, default password xc3511,
                                                          "Login incorrect"
Base time 2026-10-18T12:00:00Z, 10 ms between packets.
"""
import sys

from scapy.all import IP, TCP, Ether, Raw, wrpcap

BASE = 1792324800.0  # 2026-10-18T12:00:00Z
pkts, t = [], [BASE]


def add(p):
    p.time = t[0]
    t[0] += 0.010
    pkts.append(p)


def session(cmac, smac, cip, sip, cport, sport, exchange):
    """exchange: list of (direction, payload) with direction 'c' (client->server) or 's'."""
    E_c = Ether(src=cmac, dst=smac)
    E_s = Ether(src=smac, dst=cmac)
    cseq, sseq = 1000, 5000
    add(E_c / IP(src=cip, dst=sip) / TCP(sport=cport, dport=sport, flags="S", seq=cseq))
    add(E_s / IP(src=sip, dst=cip) / TCP(sport=sport, dport=cport, flags="SA", seq=sseq, ack=cseq + 1))
    cseq += 1
    sseq += 1
    add(E_c / IP(src=cip, dst=sip) / TCP(sport=cport, dport=sport, flags="A", seq=cseq, ack=sseq))
    for d, data in exchange:
        data = data.encode() if isinstance(data, str) else data
        if d == "c":
            add(E_c / IP(src=cip, dst=sip) / TCP(sport=cport, dport=sport, flags="PA", seq=cseq, ack=sseq) / Raw(data))
            cseq += len(data)
            add(E_s / IP(src=sip, dst=cip) / TCP(sport=sport, dport=cport, flags="A", seq=sseq, ack=cseq))
        else:
            add(E_s / IP(src=sip, dst=cip) / TCP(sport=sport, dport=cport, flags="PA", seq=sseq, ack=cseq) / Raw(data))
            sseq += len(data)
            add(E_c / IP(src=cip, dst=sip) / TCP(sport=cport, dport=sport, flags="A", seq=cseq, ack=sseq))
    add(E_c / IP(src=cip, dst=sip) / TCP(sport=cport, dport=sport, flags="FA", seq=cseq, ack=sseq))
    add(E_s / IP(src=sip, dst=cip) / TCP(sport=sport, dport=cport, flags="FA", seq=sseq, ack=cseq + 1))
    add(E_c / IP(src=cip, dst=sip) / TCP(sport=cport, dport=sport, flags="A", seq=cseq + 1, ack=sseq + 1))


SQLI = ("GET /products.php?id=1%27%20UNION%20SELECT%20username,password%20FROM%20users;--%20 HTTP/1.1\r\n"
        "Host: 192.168.1.20\r\nUser-Agent: Mozilla/5.0\r\nAccept: */*\r\n\r\n")
BODY = "<html><body>no results</body></html>"
RESP = (f"HTTP/1.1 200 OK\r\nServer: Apache/2.4.41 (Ubuntu)\r\nContent-Type: text/html\r\n"
        f"Content-Length: {len(BODY)}\r\n\r\n{BODY}")
srv_mac, att_mac, ext_gw_mac, cam_mac = "02:00:00:00:01:14", "02:00:00:00:01:42", "02:00:00:00:00:01", "02:00:00:00:01:1e"

session(att_mac, srv_mac, "192.168.1.66", "192.168.1.20", 40001, 80, [("c", SQLI), ("s", RESP)])
session(ext_gw_mac, srv_mac, "203.0.113.50", "192.168.1.20", 40002, 80, [("c", SQLI), ("s", RESP)])
session(ext_gw_mac, cam_mac, "203.0.113.50", "192.168.1.30", 40003, 23, [
    ("s", "\r\nIPCAM login: "), ("c", "root\r\n"), ("s", "root\r\nPassword: "), ("c", "xc3511\r\n"),
    ("s", "\r\nLogin incorrect\r\nIPCAM login: ")])

wrpcap(sys.argv[1], pkts)
print(f"wrote {len(pkts)} packets to {sys.argv[1]}")
