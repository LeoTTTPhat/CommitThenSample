"""Appendix C unit-type taxonomy (first matching rule wins; label-free)."""
from __future__ import annotations

import polars as pl
import pytest

from certichain.units import unit_types as UT
from certichain.units.sessionize import event_units, records_frame

SEED = "192.168.1.10"
EXT = "5.5.5.5"


def typed(rows, seeds=(SEED,)):
    u = event_units(records_frame(rows), list(seeds)).units
    return u


def one_type(rows, seeds=(SEED,)):
    u = typed(rows, seeds)
    assert u.height == 1, u.select("type", "n_records", "dsts")
    return u["type"][0]


def fails(k, dport=23, dst=None, t0=0.0, gap=0.1):
    return [dict(ts=t0 + k_ * gap, src=SEED, dst=dst or f"77.1.{k_}.9", dport=dport, conn_state="S0",
                 obytes=0, rbytes=0, opkts=1, rpkts=0, history="S") for k_ in range(k)]


CASES = {
    "SCAN_H": fails(20),
    "SCAN_V": [dict(ts=k * 0.5, src=SEED, dst="192.168.1.50", dport=1000 + k, conn_state="REJ", obytes=0, rbytes=0,
                    opkts=1, rpkts=1) for k in range(12)],
    "SWEEP_ICMP": [dict(ts=k * 0.2, src=SEED, dst=f"192.168.1.{100 + k}", proto="icmp", sport=8, dport=0,
                        conn_state="OTH", obytes=56, rbytes=56) for k in range(10)],
    "BRUTE_TELNET": [dict(ts=k * 6, te=k * 6 + 2, src=SEED, dst="192.168.1.60", dport=23, obytes=30, rbytes=40)
                     for k in range(6)],
    "BRUTE_SSH": [dict(ts=k * 6, te=k * 6 + 3, src=SEED, dst="192.168.1.61", dport=22, obytes=600, rbytes=900)
                  for k in range(6)],
    "BRUTE_HTTP": [dict(ts=k * 3, src=SEED, dst="192.168.1.62", dport=80, obytes=200, rbytes=300,
                        http_method="GET", http_uri="/login", http_status=401) for k in range(6)],
    "WEB_ATTACK": [dict(ts=0, src="1.2.3.4", dst=SEED, dport=80, http_method="GET",
                        http_uri="/index.php?id=1%27%20UNION%20SELECT%20password%20FROM%20users--", http_status=200)],
    "HTTP_DOWNLOAD": [dict(ts=0, src=SEED, dst=EXT, dport=80, obytes=200, rbytes=60_000, http_method="GET",
                           http_uri="/bins/mirai.arm7", http_ua="wget/1.20", http_status=200)],
    "HTTP_UPLOAD": [dict(ts=0, src=SEED, dst=EXT, dport=8080, obytes=50_000, rbytes=300, http_method="POST",
                         http_uri="/upload", http_status=200)],
    "C2_PERIODIC": [dict(ts=60 * k, src=SEED, dst=EXT, dport=6667, obytes=60, rbytes=40) for k in range(10)],
    "INTERACTIVE_SESSION": [dict(ts=0, te=120, src="192.168.33.12", dst=SEED, dport=23, obytes=40_000,
                                 rbytes=3_000)],
    "FLOOD_SYN": [dict(ts=k * 0.01, src=SEED, dst="192.168.1.70", dport=80, conn_state="S0", obytes=0, rbytes=0,
                       opkts=1, rpkts=0, history="S") for k in range(800)],
    "FLOOD_UDP": [dict(ts=10.0 * k, te=10.0 * k + 10, src=SEED, dst="192.168.1.71", dport=None, sport=None,
                       proto="udp", obytes=5_000_000, rbytes=0, opkts=20_000, rpkts=0, conn_state="S0",
                       bulk=True, n_merged=20_000) for k in range(6)],
    "FLOOD_ICMP": [dict(ts=0, te=5, src=SEED, dst="192.168.1.72", proto="icmp", sport=8, dport=0, obytes=500_000,
                        rbytes=0, opkts=5_000, rpkts=0, conn_state="OTH")],
    "AMPLIFICATION": [dict(ts=5.0 * (k // 2) + 0.3 * (k % 2), src="192.168.0.200", dst="192.168.20.10", dport=5683,
                           proto="udp", obytes=20, rbytes=600, opkts=1, rpkts=1, conn_state="SF")
                      for k in range(110)],
    # irregular query times (a regular stream of ≥ 5 queries is C2_PERIODIC first, by rule order)
    "DNS_ANOMALY": [dict(ts=t, src=SEED, dst="192.168.1.1", dport=53, proto="udp", dns_q=f"x{k}qz.example.com",
                         dns_rcode=3) for k, t in enumerate([0, 1, 7, 8, 30, 31])],
    "EXFIL_BULK": [dict(ts=0, te=60, src=SEED, dst=EXT, dport=9100, obytes=3_000_000, rbytes=1_000)],
    "REPORT_SMALL": [dict(ts=0, te=0.5, src=SEED, dst=EXT, dport=48101, obytes=60, rbytes=0)],
    "MQTT_PUBSUB": [dict(ts=0, te=50, src=SEED, dst="192.168.1.5", dport=1883, obytes=600, rbytes=600)],
    "COAP": [dict(ts=0, src=SEED, dst="192.168.1.6", dport=5683, proto="udp", obytes=30, rbytes=40)],
    "RTSP_STREAM": [dict(ts=0, te=20, src="192.168.1.7", dst=SEED, dport=554, obytes=1_000, rbytes=90_000)],
    "DNS_ACTIVITY": [dict(ts=k, src=SEED, dst="192.168.1.1", dport=53, proto="udp", dns_q="time.google.com",
                          dns_rcode=0) for k in range(3)],
    "TLS_SESSION": [dict(ts=0, te=5, src=SEED, dst="52.1.2.3", dport=443, obytes=900, rbytes=5_000,
                         tls_sni="api.vendor.example")],
    "BENIGN_GENERIC_TCP_HIGH": [dict(ts=0, src=SEED, dst="192.168.1.8", dport=9000, obytes=100, rbytes=100)],
    "BENIGN_GENERIC_UDP_HIGH": [dict(ts=0, src=SEED, dst="192.168.1.8", dport=9000, proto="udp", obytes=100,
                                     rbytes=100)],
    "BENIGN_GENERIC_TCP_SYS": [dict(ts=0, src=SEED, dst="192.168.1.8", dport=515, obytes=100, rbytes=100)],
    "BENIGN_GENERIC_UDP_SYS": [dict(ts=0, src=SEED, dst="192.168.1.8", dport=123, proto="udp", obytes=48,
                                    rbytes=48)],
    # v2 (b): failed call-back burst — one dst, one port, REJ answers
    "CONN_FAIL_BURST": [dict(ts=3.0 * k + (k % 2), src=SEED, dst="203.0.113.9", dport=4444, conn_state="REJ",
                             obytes=0, rbytes=0, opkts=1, rpkts=1, history="Sr") for k in range(8)],
    # v2 (c): established SMB session with data both ways
    "SMB_SESSION": [dict(ts=0, te=40, src="10.0.0.20", dst=SEED, dport=445, obytes=50_000, rbytes=8_000,
                         conn_state="SF")],
}


SEEDS = {"AMPLIFICATION": ("192.168.20.10",)}


@pytest.mark.parametrize("expected", sorted(CASES))
def test_type_fixture(expected):
    assert one_type(CASES[expected], SEEDS.get(expected, (SEED,))) == expected


def test_standard_service_polling_is_not_c2():
    # v2 (e): periodic NTP / DNS polling of standard servers is never C2_PERIODIC (nor AMPLIFICATION)
    ntp = [dict(ts=64 * k, src=SEED, dst="192.168.1.1", dport=123, proto="udp", obytes=48, rbytes=48)
           for k in range(8)]
    assert one_type(ntp) == "BENIGN_GENERIC_UDP_SYS"
    dns = [dict(ts=60 * k, src=SEED, dst="192.168.1.1", dport=53, proto="udp", obytes=0, rbytes=4_000, rpkts=20,
                dns_q="time.vendor.example", dns_rcode=0) for k in range(8)]
    assert one_type(dns) == "DNS_ACTIVITY"
    other = [dict(ts=64 * k, src=SEED, dst="192.168.1.1", dport=9999, proto="udp", obytes=48, rbytes=48)
             for k in range(8)]
    assert one_type(other) == "C2_PERIODIC"                        # other services keep the C2 rule


def test_brute_http_without_401():
    # v2 (a): ≥ 5 short logins answered 200/302 with small pages (conn-only works the same)
    rows = [dict(ts=t, te=t + 0.5, src="198.51.100.7", dst=SEED, dport=80, obytes=160, rbytes=1_700,
                 http_method="POST", http_uri="/login.php", http_status=302) for t in (0, 1, 3, 4, 7, 8, 12, 13)]
    assert one_type(rows) == "BRUTE_HTTP"
    for r in rows:
        r.update(http_method=None, http_uri=None, http_status=None)
    assert one_type(rows) == "BRUTE_HTTP"
    big = [dict(r, rbytes=40_000) for r in rows]                     # large answers → not brute force
    assert one_type(big) != "BRUTE_HTTP"
    inj = [dict(r, http_method="GET", http_uri="/i.php?id=1%27%20or%201=1--") for r in rows]
    assert one_type(inj) == "WEB_ATTACK"                             # regex evidence wins


def test_periodic_http_beacon_stays_c2():
    rows = [dict(ts=60 * k, src=SEED, dst="203.0.113.9", dport=80, obytes=150, rbytes=300) for k in range(8)]
    assert one_type(rows) == "C2_PERIODIC"


def test_conn_fail_burst_is_distinct_from_scans_and_floods():
    assert one_type(CASES["CONN_FAIL_BURST"]) == "CONN_FAIL_BURST"
    four = CASES["CONN_FAIL_BURST"][:4]
    assert one_type(four) != "CONN_FAIL_BURST"                     # needs ≥ 5 REJ/S0 conns
    flood = [dict(ts=k * 0.001, src=SEED, dst="192.168.1.70", dport=80, conn_state="S0", obytes=0, rbytes=0,
                  opkts=1, rpkts=0, history="S") for k in range(800)]
    assert one_type(flood) == "FLOOD_SYN"
    assert one_type(CASES["SCAN_V"]) == "SCAN_V"


def test_smb_session_requires_data_and_no_flood():
    probe = [dict(ts=0, src="10.0.0.20", dst=SEED, dport=445, obytes=0, rbytes=0, conn_state="RSTO")]
    assert one_type(probe) != "SMB_SESSION"
    assert one_type(CASES["SMB_SESSION"]) == "SMB_SESSION"


def test_transfer_direction_with_frame_bytes():
    # Gotham-style per-frame byte counts: the client's ACK frames make orig ≥ 1 kB; resp > orig decides
    dl = [dict(ts=0, te=2, src=SEED, dst="192.168.33.13", dport=80, obytes=28_000, rbytes=1_150_000, conn_state="SF")]
    assert one_type(dl) == "HTTP_DOWNLOAD"
    up = [dict(ts=0, te=2, src=SEED, dst="203.0.113.9", dport=443, obytes=1_150_000, rbytes=28_000, conn_state="SF")]
    assert one_type(up) == "HTTP_UPLOAD"
    tls = [dict(ts=0, te=2, src=SEED, dst="203.0.113.9", dport=443, obytes=28_000, rbytes=1_150_000, conn_state="SF")]
    assert one_type(tls) != "HTTP_UPLOAD"


def test_http_download_conn_only_fallback():
    rows = [dict(ts=0, src=SEED, dst=EXT, dport=80, obytes=300, rbytes=500_000, conn_state="SF")]
    assert one_type(rows) == "HTTP_DOWNLOAD"
    rows[0]["obytes"] = 600_000                                   # v2: orig > resp → not a conn-only download
    assert one_type(rows) != "HTTP_DOWNLOAD"
    rows = [dict(ts=0, src=SEED, dst=EXT, dport=80, obytes=300, rbytes=500_000, conn_state="SF",
                 http_method="GET", http_uri="/index.html", http_status=200)]
    assert one_type(rows) != "HTTP_DOWNLOAD"                      # http fields present: needs exec MIME/ext/tool UA


def test_c2_long_lived_low_rate():
    rows = [dict(ts=0, te=3600, src=SEED, dst=EXT, dport=23, obytes=4_000, rbytes=3_000)]
    assert one_type(rows) == "C2_PERIODIC"                       # before INTERACTIVE_SESSION in rule order
    rows[0].update(obytes=4_000_000, rbytes=3_000_000)            # not low-rate → interactive telnet session
    assert one_type(rows) == "INTERACTIVE_SESSION"


def test_first_match_order_scan_before_flood():
    # a random-dport SYN flood against one victim satisfies SCAN_V first (Appendix C order)
    rows = [dict(ts=k * 0.001, src=SEED, dst="192.168.1.70", dport=1000 + k, conn_state="S0", obytes=0, rbytes=0)
            for k in range(600)]
    assert one_type(rows) == "SCAN_V"


def test_flood_guard_on_periodic_bulk():
    # 10-s bulk flood records are perfectly periodic with few bytes per flow, yet are a FLOOD (not C2)
    assert one_type(CASES["FLOOD_UDP"]) == "FLOOD_UDP"


def test_flood_ack_and_http_subtypes():
    ack = [dict(ts=k * 0.01, src=SEED, dst="192.168.1.70", dport=80, conn_state="OTH", history="A", obytes=0,
                rbytes=0, opkts=1, rpkts=0) for k in range(800)]
    assert one_type(ack) == "FLOOD_ACK"
    http = [dict(ts=k * 0.05, src=SEED, dst="192.168.1.70", dport=80, conn_state="SF", obytes=150, rbytes=900,
                 http_method="GET", http_uri="/", http_status=200) for k in range(800)]
    assert one_type(http) == "FLOOD_HTTP"


def test_dns_anomaly_entropy_and_nx():
    hi = [dict(ts=k * 2, src=SEED, dst="192.168.1.1", dport=53, proto="udp", dns_q=q, dns_rcode=0)
          for k, q in enumerate(["xjw8f3kdl2nq.com", "qwhdk3j9s0vbz71m.net", "b3s2k1j4h5g6.target.com"])]
    assert one_type(hi) == "DNS_ANOMALY"
    lo = [dict(ts=k * 2, src=SEED, dst="192.168.1.1", dport=53, proto="udp", dns_q=q, dns_rcode=0)
          for k, q in enumerate(["connectivitycheck.gstatic.com", "time.google.com", "pool.ntp.org"])]
    assert one_type(lo) == "DNS_ACTIVITY"


def test_report_small_requires_infra_or_external_dst():
    rows = [dict(ts=0, te=0.5, src=SEED, dst="192.168.1.9", dport=48101, obytes=60, rbytes=0)]
    assert one_type(rows) != "REPORT_SMALL"                       # internal non-INFRA host
    # the same report to a collapsed INFRA host (contacted by ≥ 3 internal hosts)
    rows += [dict(ts=1000 + i, src=f"192.168.1.{30 + i}", dst="192.168.1.9", dport=48101, obytes=60, rbytes=0)
             for i in range(3)]
    u = typed(rows)
    assert u.filter(pl.col("src") == SEED)["type"].to_list() == ["REPORT_SMALL"]


def test_content_flags():
    u = typed(CASES["WEB_ATTACK"])
    assert u["content_flags"][0] & 2
    u = typed(CASES["HTTP_DOWNLOAD"])
    assert u["content_flags"][0] & 1 and u["content_flags"][0] & 4
    u = typed(CASES["DNS_ANOMALY"])
    assert u["content_flags"][0] & 8


def test_attack_hints_and_sets():
    assert UT.attack_hints("SCAN_H") == ("T1046", "T1595.001")
    assert UT.attack_hints("BRUTE_TELNET") == ("T1110.001",)
    assert UT.attack_hints("FLOOD_GRE") == ("T1498", "T1499", "T0814")
    assert UT.attack_hints("TLS_SESSION") == ()
    assert set(UT.BENIGN_PROTOCOL_TYPES) == {"MQTT_PUBSUB", "COAP", "RTSP_STREAM", "DNS_ACTIVITY", "TLS_SESSION",
                                             "BENIGN_GENERIC_TCP_SYS", "BENIGN_GENERIC_TCP_HIGH",
                                             "BENIGN_GENERIC_UDP_SYS", "BENIGN_GENERIC_UDP_HIGH"}
    assert UT.attack_hints("CONN_FAIL_BURST") and UT.attack_hints("SMB_SESSION")
    assert set(CASES) <= set(UT.ALL_TYPES)


def test_side_table_carries_hints():
    eu = event_units(records_frame(CASES["SCAN_H"]), [SEED])
    assert eu.side["attck"][0].to_list() == ["T1046", "T1595.001"]


def test_rules_read_no_label_columns():
    bad = [c for c in UT.REQUIRED_COLS if any(s in c for s in ("label", "episode", "campaign", "rec_id"))]
    assert bad == []
