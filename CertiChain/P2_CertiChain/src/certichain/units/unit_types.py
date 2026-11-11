"""Label-free unit-type taxonomy (DESIGN Appendix C) — first matching rule wins.

The rules read only per-unit aggregates that ``units/sessionize.py`` computes from
RECORD_SCHEMA columns (no label, no rec_id, no absolute time). Thresholds are the
Appendix C values; the few quantities Appendix C does not pin down are fixed here once
(module constants, never swept) and listed in ``notes/P2-B.md``:

* ``C2_LOW_RATE_BPS`` — "long-lived ≥ 300 s low-rate": a record with dur ≥ 300 s and
  (obytes + rbytes)/dur < 100 B/s.
* ``C2_PERIODIC`` additionally requires *not flood-rate* (pkt/s < 100 and conns/min < 500):
  without it, 10-s bulk flood records (perfectly periodic, few bytes per underlying flow)
  would be typed C2 before the FLOOD rule is reached.
* ``REPORT_SMALL`` "single short conn" is evaluated as ≤ 4 short conns (< 10 s each,
  < 2 kB per conn): two isolated report conns ≤ τ_merge apart are merged by the activity
  merge, and a strict "single" would then re-type the merged report as BENIGN_GENERIC.
* "rare port" = svc_class ∈ {report48101, other_tcp, other_udp} (no well-known service).
* qname entropy = Shannon entropy (bits/char) of the left-most DNS label, unit value =
  median over the unit's distinct qnames (benign names such as connectivitycheck → 3.15,
  random 12-char labels → ≥ 3.58).
* FLOOD rates use the unit span floored at 1 s (pkt/s) and 60 s (conns/min) so that a
  handful of packets in a few ms is not a flood. TCP sub-type: HTTP if svc http/https with
  HTTP requests or mostly established conns; ACK if ≥ 50 % of records never show an
  originator SYN (Zeek OTH / history without 'S'); SYN otherwise.
* Non-TCP generic units (UDP, ICMP, GRE, other) are BENIGN_GENERIC_UDP (Appendix C has only
  the two generic types).

TYPER v2 (coordinator decision 18 Oct 19:52; generic, label-free; evidence from TRAIN incidents
only — cf_ton/train, cf_gotham/train; counts in notes/P2-B.md):
(a) BRUTE_HTTP also without 401s: ≥ 5 short conns, same (src, dst), 80/8080, ≥ 50 % with payload,
    mean response ≤ 2 kB per conn, no injection-regex hit and not flood-rate (logins answered 200/302;
    conn-only works). Evaluated after C2_PERIODIC (periodic HTTP beacons stay C2) — the 401 variant keeps
    its Appendix C position.
(b) CONN_FAIL_BURST (after SWEEP_ICMP): one dst, ≤ 3 dports, ≥ 5 REJ/S0 conns that are ≥ 70 % of the
    unit's records, not flood-rate (failed call-back bursts; SCAN_V needs ≥ 8 dports, SCAN_H ≥ 8 dsts).
(c) SMB_SESSION (before C2_PERIODIC): TCP to 139/445/135 with payload both ways in established conns,
    not flood-rate.
(d) BENIGN_GENERIC_{TCP,UDP} split by service-port class: _SYS (dport_mode ≤ 1023 or portless, e.g.
    ICMP/GRE) / _HIGH (dport_mode > 1023).
(e) C2_PERIODIC never fires for svc dns / mdns / ntp (polling of standard resolvers/servers), and such
    periodic polling (≥ 5 conns, start CV < 0.3) is not AMPLIFICATION either (falls to DNS_*/generic); conn-only
    HTTP_DOWNLOAD needs resp > orig (instead of orig < 1 kB) and conn-only HTTP_UPLOAD orig > resp: the
    direction of the larger byte count relative to the initiator (robust to Gotham's per-frame bytes).
"""
from __future__ import annotations

import polars as pl

# --------------------------------------------------------------------------- thresholds
FAN_DSTS = 8                 # SCAN_H / SWEEP_ICMP "≥ 8 dsts"
SCAN_MAX_DPORTS = 2
SCAN_MIN_FAIL = 0.70
SCAN_MAX_MEAN_BYTES = 100.0
SCANV_MIN_DPORTS = 8
BRUTE_MIN_CONNS = 5
BRUTE_MAX_MED_DUR = 10.0
BRUTE_MIN_PAYLOAD_FRAC = 0.5
BRUTE_PORTS = {"BRUTE_TELNET": (23, 2323), "BRUTE_SSH": (22,), "BRUTE_FTP": (21,),
               "BRUTE_MQTT": (1883,), "BRUTE_HTTP": (80, 8080)}
BRUTE_HTTP_MIN_401_FRAC = 0.5
DL_MIN_RESP_BYTES = 10_000
DL_CONN_MAX_ORIG_BYTES = 1_000
UP_MIN_BYTES = 10_000
UP_PORTS = (80, 8080, 443)
C2_MIN_CONNS = 5
C2_MAX_CV = 0.3
C2_MAX_BYTES_PER_CONN = 2_000.0
C2_LONG_DUR = 300.0
C2_LOW_RATE_BPS = 100.0
INTER_MIN_DUR = 30.0
INTER_MIN_BYTES = 2_000
SHELL_PORTS = (4444, 4445, 1337, 31337, 5555, 6666, 7777, 1234, 12345, 9001)
FLOOD_MIN_PPS = 100.0
FLOOD_MIN_CPM = 500.0
FLOOD_MAX_DSTS = 3
AMP_MIN_RATIO = 10.0
AMP_MIN_RESPONSES = 100
DNS_NX_RATE = 0.5
DNS_ENTROPY = 3.5
EXFIL_MIN_ORIG = 1_000_000
EXFIL_MIN_RATIO = 10.0
BRUTE_HTTP_MAX_RESP_PER_CONN = 2_000.0     # v2 (a): "small responses"
FAIL_BURST_MIN_CONNS = 5                   # v2 (b)
FAIL_BURST_MAX_DPORTS = 3
FAIL_BURST_MIN_FRAC = 0.70
SMB_PORTS = (139, 445, 135)                # v2 (c)
SYS_PORT_MAX = 1023                        # v2 (d)
C2_EXCLUDED_SVCS = ("dns", "mdns", "ntp")  # v2 (e)
REPORT_MAX_CONNS = 4
REPORT_MAX_DUR = 10.0
REPORT_MAX_BYTES_PER_CONN = 2_000.0
RARE_PORT_SVCS = ("report48101", "other_tcp", "other_udp")

# --------------------------------------------------------------------------- type sets
GENERIC_TYPES = ("BENIGN_GENERIC_TCP_SYS", "BENIGN_GENERIC_TCP_HIGH", "BENIGN_GENERIC_UDP_SYS",
                 "BENIGN_GENERIC_UDP_HIGH")
BENIGN_PROTOCOL_TYPES = frozenset({"MQTT_PUBSUB", "COAP", "RTSP_STREAM", "DNS_ACTIVITY", "TLS_SESSION", *GENERIC_TYPES})
BENIGN_GENERIC_TYPES = frozenset(GENERIC_TYPES)
FLOOD_TYPES = ("FLOOD_SYN", "FLOOD_UDP", "FLOOD_ICMP", "FLOOD_HTTP", "FLOOD_GRE", "FLOOD_ACK")

ALL_TYPES = ("SCAN_H", "SCAN_V", "SWEEP_ICMP", "CONN_FAIL_BURST", *BRUTE_PORTS, "WEB_ATTACK", "HTTP_DOWNLOAD",
             "HTTP_UPLOAD", "SMB_SESSION", "C2_PERIODIC", "INTERACTIVE_SESSION", *FLOOD_TYPES, "AMPLIFICATION",
             "DNS_ANOMALY", "EXFIL_BULK", "REPORT_SMALL", "MQTT_PUBSUB", "COAP", "RTSP_STREAM", "DNS_ACTIVITY",
             "TLS_SESSION", *GENERIC_TYPES)

ATTACK_HINTS: dict[str, tuple[str, ...]] = {
    "SCAN_H": ("T1046", "T1595.001"),
    "SCAN_V": ("T1046",),
    "SWEEP_ICMP": ("T1018", "T0846"),
    "CONN_FAIL_BURST": ("T1071", "T1571"),
    "SMB_SESSION": ("T1021.002", "T1570"),
    **{t: ("T1110.001",) for t in BRUTE_PORTS},
    "WEB_ATTACK": ("T1190",),
    "HTTP_DOWNLOAD": ("T1105",),
    "HTTP_UPLOAD": ("T1105", "T1505.003"),
    "C2_PERIODIC": ("T1071", "T1571", "T1573"),
    "INTERACTIVE_SESSION": ("T1059", "T1021", "T1105"),
    **{t: ("T1498", "T1499", "T0814") for t in FLOOD_TYPES},
    "AMPLIFICATION": ("T1498.002",),
    "DNS_ANOMALY": ("T1568", "T1071.004"),
    "EXFIL_BULK": ("T1041", "T1048"),
    "REPORT_SMALL": ("T1041",),
}
for _t in ALL_TYPES:
    ATTACK_HINTS.setdefault(_t, ())

# Aggregate columns the rules read (all produced by sessionize._aggregate_light).
REQUIRED_COLS = ("ts", "te", "proto", "svc_class", "dir_class", "fanout", "n_dst", "n_dports", "dport_mode",
                 "n_records", "n_conns", "obytes", "rbytes", "opkts", "rpkts", "fail_frac", "med_dur", "max_dur",
                 "frac_payload", "start_cv", "n_est", "est_obytes", "est_rbytes", "est_max_dur",
                 "any_longlow", "frac_nosyn", "frac_icmp_echo", "http_any", "frac_est", "n_status", "n_401",
                 "any_inj", "any_dl", "any_up", "has_dns", "n_nx", "n_rcode", "qent", "has_tls",
                 "dst1_infra_or_ext", "n_rejs0")


def _c(name: str) -> pl.Expr:
    return pl.col(name)


def rate_exprs() -> dict[str, pl.Expr]:
    span = (_c("te") - _c("ts"))
    pkt_rate = (_c("opkts") + _c("rpkts")).cast(pl.Float64) / pl.max_horizontal(span, pl.lit(1.0))
    conn_rate = _c("n_conns").cast(pl.Float64) / (pl.max_horizontal(span, pl.lit(60.0)) / 60.0)
    mean_bytes = (_c("obytes") + _c("rbytes")).cast(pl.Float64) / pl.max_horizontal(
        _c("n_conns").cast(pl.Float64), pl.lit(1.0))
    return {"pkt_rate": pkt_rate, "conn_rate": conn_rate, "mean_bytes": mean_bytes}


def rule_list() -> list[tuple[str, pl.Expr]]:
    """Ordered (type, predicate) list; the type expression for FLOOD is resolved separately."""
    r = rate_exprs()
    pkt_rate, conn_rate, mean_bytes = r["pkt_rate"], r["conn_rate"], r["mean_bytes"]
    single = _c("n_dst") == 1
    tcp = _c("proto") == "tcp"
    not_flood = (pkt_rate < FLOOD_MIN_PPS) & (conn_rate < FLOOD_MIN_CPM)
    brute_base = (single & tcp & (_c("n_conns") >= BRUTE_MIN_CONNS) & (_c("med_dur") < BRUTE_MAX_MED_DUR)
                  & (_c("frac_payload") >= BRUTE_MIN_PAYLOAD_FRAC))
    resp_per_conn = _c("rbytes").cast(pl.Float64) / pl.max_horizontal(_c("n_conns").cast(pl.Float64), pl.lit(1.0))
    # v2 (e): periodic polling of standard DNS/mDNS/NTP services (the C2_PERIODIC pattern on those svcs)
    std_poll = (_c("svc_class").is_in(list(C2_EXCLUDED_SVCS)) & (_c("n_conns") >= C2_MIN_CONNS)
                & (_c("start_cv") < C2_MAX_CV))
    rules: list[tuple[str, pl.Expr]] = [
        ("SCAN_H", (_c("n_dst") >= FAN_DSTS) & (_c("n_dports") <= SCAN_MAX_DPORTS)
         & (_c("fail_frac") >= SCAN_MIN_FAIL) & (mean_bytes < SCAN_MAX_MEAN_BYTES)),
        ("SCAN_V", single & (_c("n_dports") >= SCANV_MIN_DPORTS) & (_c("fail_frac") >= SCAN_MIN_FAIL)),
        ("SWEEP_ICMP", (_c("proto") == "icmp") & (_c("n_dst") >= FAN_DSTS) & (_c("frac_icmp_echo") >= 0.5)),
        ("CONN_FAIL_BURST", single & tcp & (_c("n_dports") <= FAIL_BURST_MAX_DPORTS)
         & (_c("n_rejs0") >= FAIL_BURST_MIN_CONNS)
         & (_c("n_rejs0").cast(pl.Float64) >= FAIL_BURST_MIN_FRAC * _c("n_records").cast(pl.Float64)) & not_flood),
    ]
    for t, ports in BRUTE_PORTS.items():
        cond = brute_base & _c("dport_mode").is_in(list(ports))
        if t == "BRUTE_HTTP":
            cond = cond & (_c("n_status") > 0) & (
                _c("n_401").cast(pl.Float64) / pl.max_horizontal(_c("n_status").cast(pl.Float64), pl.lit(1.0))
                >= BRUTE_HTTP_MIN_401_FRAC)
        rules.append((t, cond))
    # v2 (a): BRUTE_HTTP without 401s (200/302 login answers, conn-only) — evaluated AFTER C2_PERIODIC so
    # that periodic HTTP beacons stay C2_PERIODIC; WEB_ATTACK (regex evidence) precedes it anyway
    brute_http_small = (brute_base & _c("dport_mode").is_in(list(BRUTE_PORTS["BRUTE_HTTP"]))
                        & (resp_per_conn <= BRUTE_HTTP_MAX_RESP_PER_CONN) & ~_c("any_inj") & not_flood)
    rules += [
        ("WEB_ATTACK", _c("any_inj")),
        ("HTTP_DOWNLOAD", _c("any_dl")),
        ("HTTP_UPLOAD", _c("any_up")),
        ("SMB_SESSION", tcp & _c("dport_mode").is_in(list(SMB_PORTS)) & (_c("est_obytes") > 0) & (_c("est_rbytes") > 0)
         & not_flood),
        ("C2_PERIODIC", single & ~_c("svc_class").is_in(list(C2_EXCLUDED_SVCS))
         & (((_c("n_conns") >= C2_MIN_CONNS) & (_c("start_cv") < C2_MAX_CV)
             & (mean_bytes < C2_MAX_BYTES_PER_CONN) & not_flood) | _c("any_longlow"))),
        ("BRUTE_HTTP", brute_http_small),
        ("INTERACTIVE_SESSION", tcp & (_c("svc_class").is_in(["telnet", "ssh"])
                                       | _c("dport_mode").is_in(list(SHELL_PORTS)))
         & (_c("n_est") >= 1) & ((_c("est_max_dur") >= INTER_MIN_DUR)
                                 | ((_c("est_obytes") >= INTER_MIN_BYTES) & (_c("est_rbytes") >= INTER_MIN_BYTES)))),
        ("FLOOD", (_c("n_dst") <= FLOOD_MAX_DSTS) & ((pkt_rate >= FLOOD_MIN_PPS) | (conn_rate >= FLOOD_MIN_CPM))),
        ("AMPLIFICATION", (_c("proto") == "udp") & ~std_poll.fill_null(False)
         & (_c("rbytes").cast(pl.Float64) >= AMP_MIN_RATIO * pl.max_horizontal(_c("obytes").cast(pl.Float64), pl.lit(1.0)))
         & (_c("rpkts") >= AMP_MIN_RESPONSES)),
        ("DNS_ANOMALY", _c("has_dns") & (((_c("n_rcode") > 0) & (
            _c("n_nx").cast(pl.Float64) / pl.max_horizontal(_c("n_rcode").cast(pl.Float64), pl.lit(1.0)) >= DNS_NX_RATE))
            | (_c("qent").fill_nan(None).fill_null(0.0) >= DNS_ENTROPY))),
        ("EXFIL_BULK", (_c("dir_class") == "int->ext") & (_c("obytes") >= EXFIL_MIN_ORIG)
         & (_c("obytes").cast(pl.Float64) >= EXFIL_MIN_RATIO * pl.max_horizontal(_c("rbytes").cast(pl.Float64), pl.lit(1.0)))),
        ("REPORT_SMALL", single & (_c("n_conns") <= REPORT_MAX_CONNS) & (_c("max_dur") < REPORT_MAX_DUR)
         & (mean_bytes < REPORT_MAX_BYTES_PER_CONN) & _c("svc_class").is_in(list(RARE_PORT_SVCS))
         & _c("dst1_infra_or_ext")),
        ("MQTT_PUBSUB", _c("svc_class") == "mqtt"),
        ("COAP", _c("svc_class") == "coap"),
        ("RTSP_STREAM", _c("svc_class") == "rtsp"),
        ("DNS_ACTIVITY", _c("svc_class").is_in(["dns", "mdns"]) | _c("has_dns")),
        ("TLS_SESSION", (_c("svc_class") == "https") | _c("has_tls")),
    ]
    return rules


def flood_subtype_expr() -> pl.Expr:
    proto = _c("proto")
    http_like = _c("svc_class").is_in(["http", "https"]) & (_c("http_any") | (_c("frac_est") >= 0.5))
    return (pl.when(proto == "icmp").then(pl.lit("FLOOD_ICMP"))
            .when(proto == "gre").then(pl.lit("FLOOD_GRE"))
            .when(proto != "tcp").then(pl.lit("FLOOD_UDP"))
            .when(http_like).then(pl.lit("FLOOD_HTTP"))
            .when(_c("frac_nosyn") >= 0.5).then(pl.lit("FLOOD_ACK"))
            .otherwise(pl.lit("FLOOD_SYN")))


def type_expr() -> pl.Expr:
    """First matching rule wins (Appendix C order)."""
    rules = rule_list()
    expr = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.Utf8))
    for t, cond in rules:
        val = flood_subtype_expr() if t == "FLOOD" else pl.lit(t)
        expr = expr.when(cond.fill_null(False)).then(val)
    sys_port = _c("dport_mode").is_null() | (_c("dport_mode") <= SYS_PORT_MAX)   # portless (ICMP/GRE) → _SYS
    generic = (pl.when((_c("proto") == "tcp") & sys_port).then(pl.lit("BENIGN_GENERIC_TCP_SYS"))
               .when(_c("proto") == "tcp").then(pl.lit("BENIGN_GENERIC_TCP_HIGH"))
               .when(sys_port).then(pl.lit("BENIGN_GENERIC_UDP_SYS"))
               .otherwise(pl.lit("BENIGN_GENERIC_UDP_HIGH")))
    return expr.otherwise(generic)


def assign_types(agg: pl.DataFrame) -> pl.DataFrame:
    """Add ``type`` (Utf8) to a frame of unit aggregates (see REQUIRED_COLS)."""
    missing = [c for c in REQUIRED_COLS if c not in agg.columns]
    if missing:
        raise KeyError(f"assign_types: missing aggregate columns {missing}")
    return agg.with_columns(type_expr().alias("type"))


def attack_hints(unit_type: str) -> tuple[str, ...]:
    return ATTACK_HINTS.get(unit_type, ())


def is_benign_protocol(unit_type: str) -> bool:
    return unit_type in BENIGN_PROTOCOL_TYPES
