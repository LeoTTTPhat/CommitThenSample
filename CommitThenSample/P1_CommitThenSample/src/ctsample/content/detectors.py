"""Packet-local protocol detectors D1–D13 and the derived estimands (DESIGN §5.4).

Rules (identical in S1 extraction, S3 tables and tests):

* Every detector looks at ONE packet's transport payload (TCP segment / UDP datagram as
  captured). Regexes are case-insensitive over bytes; they are applied to ``payload.lower()``
  (ASCII lowering, which is exactly ``re.IGNORECASE`` for bytes patterns) after a cheap
  literal-token prefilter that is a necessary condition for each pattern.
* ``^`` is the start of the payload for D11; for the line-oriented D5 (HTTP request lines) and
  D12 (IRC commands) it is the start of any line (``re.M``), because pipelined requests /
  several IRC commands share one segment.
* A hit is attributed to its packet. Per packet a detector returns a count capped at
  ``PKT_CAP`` = 4 (indicators: 0/1). The unit value of an indicator estimand is 1[any packet
  hit]; of a count estimand Σ_j min(4, c_j) over packets in the relevant direction.
* Directions use the server port (the endpoint whose port is in the detector's port set; the
  responder if both are): s2c = server→client, c2s = client→server. D5 counts requests sent
  by the initiator (dir 0). Detectors without a direction rule look at both directions.
* Opening-based estimands D1o, D2o, D11o: f = fps · 1[hit], b = 1 iff fps and eligible.
* Count estimands D1c, D2c, D5, D6, D9: b = 4 × packets in the relevant direction.

``RAW`` detectors (D1..D13) produce the per-unit hit rows of S1; ``ESTIMANDS`` maps them to
the pre-registered estimands of ``config/estimands.yaml``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import dpkt

PKT_CAP = 4
TCP, UDP = 6, 17

TELNET = frozenset((23, 2323))
SSH = frozenset((22, 2222))
DNS_P = frozenset((53,))
MQTT = frozenset((1883, 8883))
MODBUS = frozenset((502,))

# ----------------------------------------------------------------------------- regexes
# (compiled for lower-cased payloads; equivalent to (?i) on the original bytes)
RE_D1 = re.compile(rb"password\s*:")
RE_D2 = re.compile(rb"(?:login|username)\s*:")
RE_D3 = re.compile(rb"/bin/busybox|busybox\s+[a-z]{4,}|wget\s+http|tftp\s+-[gr]|chmod\s+(?:\+x|7[0-7]{2})"
                   rb"|\.(?:arm7?|mips|mpsl|x86|sh4|m68k|ppc)\b|ecchi|mirai|gayfgt|okiru|hakai")
TOK_D3 = (b"busybox", b"wget", b"tftp", b"chmod", b".arm", b".mips", b".mpsl", b".x86", b".sh4", b".m68k",
          b".ppc", b"ecchi", b"mirai", b"gayfgt", b"okiru", b"hakai")
RE_D4_SQLI = re.compile(rb"union\s+select|\bor\s+1=1|sleep\(|benchmark\(")
TOK_D4_SQLI = (b"union", b"1=1", b"sleep(", b"benchmark(")
TOK_D4_XSS = (b"<script", b"javascript:", b"onerror=", b"onload=")         # the patterns are literals
RE_D4_CMD = re.compile(rb"(?:;|\||&&)\s*(?:cat|ls|id|wget|curl|nc|sh)\b")
TOK_D4_CMD = (b";", b"|", b"&&")
TOK_D4_TRAV = b"../"
RE_D5 = re.compile(rb"^(?:get|post|put|head|delete|options) \S+ http/1\.[01]", re.M)
RE_D11 = re.compile(rb"ssh-[12]\.")                                        # used with .match (start)
RE_D12 = re.compile(rb"^(?:privmsg|join|nick|ping) ", re.M)
RE_D13 = re.compile(rb"<\?php|system\(|passthru\(|shell_exec\(|/bin/(?:ba)?sh\b|whoami|uname\s+-a|/etc/passwd")
TOK_D13 = (b"<?php", b"system(", b"passthru(", b"shell_exec(", b"/bin/", b"whoami", b"uname", b"/etc/passwd")


def _any(low: bytes, toks) -> bool:
    for t in toks:
        if t in low:
            return True
    return False


# ----------------------------------------------------------------------------- per-packet fns
# Each takes (payload, low, is_tcp) and returns a capped per-packet count (indicators: 0/1).
def d1(pl, low, tcp):
    return min(PKT_CAP, len(RE_D1.findall(low))) if b"password" in low else 0


def d2(pl, low, tcp):
    return min(PKT_CAP, len(RE_D2.findall(low))) if (b"login" in low or b"username" in low) else 0


def d3(pl, low, tcp):
    return 1 if _any(low, TOK_D3) and RE_D3.search(low) else 0


def d4(pl, low, tcp):
    if _any(low, TOK_D4_SQLI) and RE_D4_SQLI.search(low):
        return 1
    if _any(low, TOK_D4_XSS):
        return 1
    if TOK_D4_TRAV in low:
        return 1
    if _any(low, TOK_D4_CMD) and RE_D4_CMD.search(low):
        return 1
    return 0


def d5(pl, low, tcp):
    return min(PKT_CAP, len(RE_D5.findall(low)))


def _dns_msgs(pl: bytes, tcp: bool):
    if not tcp:
        yield pl
        return
    off, n = 0, len(pl)
    while off + 2 <= n:
        ln = (pl[off] << 8) | pl[off + 1]
        if ln < 12:
            return
        yield pl[off + 2:off + 2 + ln]
        off += 2 + ln


def _dns_parse(msg: bytes):
    if len(msg) < 12:
        return None
    try:
        return dpkt.dns.DNS(msg)
    except Exception:  # noqa: BLE001  (dpkt raises several types on malformed input)
        return None


def d6(pl, low, tcp):
    c = 0
    for msg in _dns_msgs(pl, tcp):
        d = _dns_parse(msg)
        if d is not None and d.qr == 0:
            c += max(1, len(d.qd))
    return min(PKT_CAP, c)


def d7(pl, low, tcp):
    for msg in _dns_msgs(pl, tcp):
        d = _dns_parse(msg)
        if d is not None and d.qr == 1 and d.rcode == 3:
            return 1
    return 0


def mqtt_types(pl: bytes) -> list[int]:
    """Control-packet types of the MQTT packets that start at offset 0, 1, ... of a segment.
    Stops at the first invalid fixed header (strict MQTT 3.1.1 / 5 flag rules)."""
    out = []
    off, n = 0, len(pl)
    while off < n and len(out) < 64:
        b0 = pl[off]
        t, fl = b0 >> 4, b0 & 15
        if t == 0:
            break
        if t == 3:
            if (fl >> 1) & 3 == 3:
                break
        elif t in (6, 8, 10):
            if fl != 2:
                break
        elif fl != 0:
            break
        rem, mult, i, ok = 0, 1, off + 1, False
        for _ in range(4):
            if i >= n:
                break
            c = pl[i]
            i += 1
            rem += (c & 127) * mult
            mult <<= 7
            if c < 128:
                ok = True
                break
        if not ok:
            break
        if t == 1:                      # CONNECT: protocol name "MQTT" (v3.1.1/5) or "MQIsdp" (v3.1)
            name = pl[i:i + 8]
            if not (name.startswith(b"\x00\x04MQTT") or name.startswith(b"\x00\x06MQIsdp")):
                break
        elif t == 3:                    # PUBLISH: topic length must fit
            if i + 2 <= n and ((pl[i] << 8) | pl[i + 1]) + 2 > rem:
                break
        out.append(t)
        off = i + rem
    return out


def d8(pl, low, tcp):
    return 1 if 1 in mqtt_types(pl) else 0


def d9(pl, low, tcp):
    return min(PKT_CAP, mqtt_types(pl).count(3))


def d10(pl, low, tcp):
    off, n = 0, len(pl)
    while off + 8 <= n:
        pid = (pl[off + 2] << 8) | pl[off + 3]
        ln = (pl[off + 4] << 8) | pl[off + 5]
        if pid != 0 or ln < 2 or ln > 254:
            return 0
        if pl[off + 7] in (5, 6, 15, 16):
            return 1
        off += 6 + ln
    return 0


def d11(pl, low, tcp):
    return 1 if RE_D11.match(low) else 0


def d12(pl, low, tcp):
    return 1 if RE_D12.search(low) else 0


def d13(pl, low, tcp):
    return 1 if _any(low, TOK_D13) and RE_D13.search(low) else 0


# ----------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class Raw:
    id: str
    fn: object
    kind: str                 # 'ind' | 'cnt'
    protos: frozenset
    ports: frozenset | None   # None = any port
    dir: str                  # 'any' | 's2c' | 'c2s' | 'init'


RAW = (
    Raw("D1", d1, "cnt", frozenset((TCP,)), TELNET, "s2c"),
    Raw("D2", d2, "cnt", frozenset((TCP,)), TELNET, "s2c"),
    Raw("D3", d3, "ind", frozenset((TCP, UDP)), None, "any"),
    Raw("D4", d4, "ind", frozenset((TCP,)), None, "any"),
    Raw("D5", d5, "cnt", frozenset((TCP,)), None, "init"),
    Raw("D6", d6, "cnt", frozenset((TCP, UDP)), DNS_P, "c2s"),
    Raw("D7", d7, "ind", frozenset((TCP, UDP)), DNS_P, "any"),
    Raw("D8", d8, "ind", frozenset((TCP,)), MQTT, "any"),
    Raw("D9", d9, "cnt", frozenset((TCP,)), MQTT, "any"),
    Raw("D10", d10, "ind", frozenset((TCP,)), MODBUS, "any"),
    Raw("D11", d11, "ind", frozenset((TCP,)), SSH, "c2s"),
    Raw("D12", d12, "ind", frozenset((TCP,)), None, "any"),
    Raw("D13", d13, "ind", frozenset((TCP,)), None, "any"),
)
RAW_BY_ID = {r.id: r for r in RAW}


def server_is_responder(ports: frozenset, port_a: int, port_b: int) -> bool | None:
    """True if the server (port in ``ports``) is the responder, False if the initiator,
    None if neither port is in the set."""
    if port_b in ports:
        return True
    if port_a in ports:
        return False
    return None


def raw_eligible(r: Raw, proto: int, port_a: int, port_b: int) -> bool:
    if proto not in r.protos:
        return False
    return r.ports is None or port_b in r.ports or port_a in r.ports


def raw_dir(r: Raw, port_a: int, port_b: int) -> int | None:
    """Packet direction (0 = initiator→responder, 1 = reverse) the detector looks at; None = both."""
    if r.dir == "any":
        return None
    if r.dir == "init":
        return 0
    sr = server_is_responder(r.ports, port_a, port_b)
    if r.dir == "s2c":
        return 1 if sr else 0
    return 0 if sr else 1                       # c2s


def flow_detectors(proto: int, port_a: int, port_b: int) -> tuple:
    """Detectors to run on the payload packets of a flow: tuple of (raw_id, fn, dir|None)."""
    if proto != TCP and proto != UDP:
        return ()
    return tuple((r.id, r.fn, raw_dir(r, port_a, port_b)) for r in RAW if raw_eligible(r, proto, port_a, port_b))


def run_packet(dets: tuple, payload: bytes, direction: int, is_tcp: bool) -> list[tuple[str, int]]:
    """Apply a flow's detectors to one payload packet → [(raw_id, capped count > 0)]."""
    out = []
    low = payload.lower()
    for rid, fn, need in dets:
        if need is not None and need != direction:
            continue
        v = fn(payload, low, is_tcp)
        if v:
            out.append((rid, v))
    return out


# ----------------------------------------------------------------------------- estimands
@dataclass(frozen=True)
class DetEstimand:
    id: str
    raw: str
    cls: str                  # 'indicator_opening' | 'indicator_segment' | 'count'


DET_ESTIMANDS = (
    DetEstimand("D1o", "D1", "indicator_opening"),
    DetEstimand("D1c", "D1", "count"),
    DetEstimand("D2o", "D2", "indicator_opening"),
    DetEstimand("D2c", "D2", "count"),
    DetEstimand("D3", "D3", "indicator_segment"),
    DetEstimand("D4", "D4", "indicator_segment"),
    DetEstimand("D5", "D5", "count"),
    DetEstimand("D6", "D6", "count"),
    DetEstimand("D7", "D7", "indicator_segment"),
    DetEstimand("D8", "D8", "indicator_segment"),
    DetEstimand("D9", "D9", "count"),
    DetEstimand("D10", "D10", "indicator_segment"),
    DetEstimand("D11o", "D11", "indicator_opening"),
    DetEstimand("D12", "D12", "indicator_segment"),
    DetEstimand("D13", "D13", "indicator_segment"),
)
DET_ESTIMAND_IDS = tuple(e.id for e in DET_ESTIMANDS)
AUX_RAW_INDICATORS = ("D1", "D2", "D11")     # per-segment raw indicators also kept in the hits table


def estimand_hits(raw_hits: dict, fps: bool) -> list[tuple[str, int, int, int]]:
    """raw_hits: raw_id -> [value(sum of capped counts), n_hit_pkts, first_hit_pkt].
    Returns hit rows (query_id, value, n_hit_pkts, first_hit_pkt) with value > 0."""
    out = []
    for e in DET_ESTIMANDS:
        h = raw_hits.get(e.raw)
        if not h:
            continue
        if e.cls == "count":
            out.append((e.id, h[0], h[1], h[2]))
        elif e.cls == "indicator_segment":
            out.append((e.id, 1, h[1], h[2]))
        elif fps:
            out.append((e.id, 1, h[1], h[2]))
    for rid in AUX_RAW_INDICATORS:
        h = raw_hits.get(rid)
        if h:
            out.append((rid, 1, h[1], h[2]))
    return out


def eligibility(proto: int, port_a: int, port_b: int, pay: bool, fps: bool,
                pkts0: int, pkts1: int) -> dict[str, tuple[bool, int]]:
    """Metadata eligibility and bound b_i for every detector estimand: id -> (eligible, b)."""
    out = {}
    for e in DET_ESTIMANDS:
        r = RAW_BY_ID[e.raw]
        el = bool(pay) and raw_eligible(r, proto, port_a, port_b)
        if e.cls == "indicator_opening":
            el = el and bool(fps)
            out[e.id] = (el, 1 if el else 0)
        elif e.cls == "indicator_segment":
            out[e.id] = (el, 1 if el else 0)
        else:
            d = raw_dir(r, port_a, port_b)
            npk = pkts0 + pkts1 if d is None else (pkts0 if d == 0 else pkts1)
            out[e.id] = (el, PKT_CAP * npk if el else 0)
    return out


def eligibility_exprs():
    """Vectorised (polars) version of :func:`eligibility` over unit columns
    proto, port_a, port_b, pay, fps, pkts0, pkts1 → {id: (elig_expr, b_expr)}."""
    import polars as pl
    out = {}
    pay = pl.col("pay")
    for e in DET_ESTIMANDS:
        r = RAW_BY_ID[e.raw]
        el = pay & pl.col("proto").is_in(sorted(r.protos))
        if r.ports is not None:
            ports = sorted(r.ports)
            el = el & (pl.col("port_a").is_in(ports) | pl.col("port_b").is_in(ports))
        if e.cls == "indicator_opening":
            el = el & pl.col("fps")
            b = el.cast(pl.Int64)
        elif e.cls == "indicator_segment":
            b = el.cast(pl.Int64)
        else:
            if r.dir == "any":
                npk = pl.col("pkts0") + pl.col("pkts1")
            elif r.dir == "init":
                npk = pl.col("pkts0")
            else:
                srv_resp = pl.col("port_b").is_in(sorted(r.ports))
                if r.dir == "s2c":
                    npk = pl.when(srv_resp).then(pl.col("pkts1")).otherwise(pl.col("pkts0"))
                else:
                    npk = pl.when(srv_resp).then(pl.col("pkts0")).otherwise(pl.col("pkts1"))
            b = pl.when(el).then(PKT_CAP * npk.cast(pl.Int64)).otherwise(0)
        out[e.id] = (el, b)
    return out
