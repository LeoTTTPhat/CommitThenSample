"""Alg. 1 (DESIGN §3.2): label-free event units.

Steps 1–5 are implemented here (``sessionize``); step 6 (scope) is in ``units/scope.py``;
``event_units`` runs the whole of Alg. 1 for one incident.

1. key(r) = (src, svc_class(r), dir_class(r)), dir_class ∈ {int->int, int->ext, ext->int, ext->ext}.
2. Sort by (key, ts, tiehash); split each key group into bursts at gaps > τ_gap. The gap is the
   idle gap ts_i − max_{j<i in burst} te_j (same convention as the merge gap t^s_{u'} − t^e_u).
3. Burst with ≥ F_fan distinct dsts → one fan-out unit, else one unit per (src, dst) sub-burst.
4. Aggregate UNIT_SCHEMA features + rec_id pointers; type via ``unit_types`` (first match wins).
5. ACTIVITY MERGE until stable: u' directly after u among units with the same
   (src, svc_class, type, dst | fan-out) is merged into u if t^s_{u'} − t^e_u ≤ τ_merge
   (te = running max over the chain); features and types are recomputed after every round,
   rounds repeat until no merge happens (each round merges maximal chains).

Determinism. Burst/merge boundaries depend only on records (units) with strictly smaller ts,
and every aggregate is order-free, so the result is invariant to the order of equal-ts
records; the BLAKE3 ``tiehash`` (io/schema.py definition) is therefore evaluated only where
it can matter — the earliest record(s) of each unit, which give ``first_tiehash`` for the
unit order (ts, first_tiehash). ``rec_id`` is never used for ordering; ``rec_ids`` is
stored as a sorted pointer set. Cost O(|R| log |R|) per merge round (polars group-bys);
regexes, entropy, IP classes and svc classes are evaluated once per distinct value.
"""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from dataclasses import dataclass, field, fields
from typing import Iterable, Sequence
from urllib.parse import unquote_plus

import blake3
import numpy as np
import polars as pl

from certichain.io.schema import RECORD_SCHEMA, UNIT_SCHEMA, conform
from certichain.units import unit_types as UT

DIR_CLASSES = ("int->int", "int->ext", "ext->int", "ext->ext")
_FANOUT_KEY = "\x00FANOUT"


# --------------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class UnitParams:
    tau_gap: float = 120.0
    F_fan: int = 8
    tau_merge: float = 600.0
    F_infra: int = 3
    U_max: int = 1000
    max_dsts: int = 256
    max_prefixes: int = 256
    max_dports: int = 64
    max_names: int = 64
    max_merge_rounds: int = 64

    @classmethod
    def from_config(cls, cfg: dict | None) -> "UnitParams":
        cfg = dict(cfg or {})
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in names})


# --------------------------------------------------------------------------- service classes
SVC_TCP_PORTS: dict[int, str] = {
    23: "telnet", 2323: "telnet", 22: "ssh", 2222: "ssh", 21: "ftp",
    25: "smtp", 465: "smtp", 587: "smtp",
    80: "http", 8080: "http", 8000: "http", 8081: "http", 443: "https", 8443: "https",
    53: "dns", 1883: "mqtt", 8883: "mqtt", 554: "rtsp", 8554: "rtsp", 502: "modbus",
    48101: "report48101", 7547: "tr069", 139: "smb", 445: "smb",
    **{p: "irc" for p in range(6660, 6670)}, 6697: "irc",
}
SVC_UDP_PORTS: dict[int, str] = {
    53: "dns", 5353: "mdns", 123: "ntp", 67: "dhcp", 68: "dhcp", 1900: "ssdp",
    5683: "coap", 5684: "coap", 554: "rtsp", 443: "https",
}
SVC_FROM_SERVICE: dict[str, str] = {
    "http": "http", "ssl": "https", "tls": "https", "quic": "https", "dns": "dns", "ssh": "ssh",
    "telnet": "telnet", "ftp": "ftp", "smtp": "smtp", "irc": "irc", "mqtt": "mqtt", "dhcp": "dhcp",
    "ntp": "ntp", "rtsp": "rtsp", "coap": "coap", "modbus": "modbus", "smb": "smb", "mdns": "mdns",
    "ssdp": "ssdp",
}
SVC_CLASSES = tuple(sorted(set(SVC_TCP_PORTS.values()) | set(SVC_UDP_PORTS.values())
                           | set(SVC_FROM_SERVICE.values())
                           | {"icmp", "gre", "other", "other_tcp", "other_udp"}))


def norm_proto(proto: str | None) -> str:
    p = (proto or "").strip().lower()
    if p in ("tcp", "udp", "gre"):
        return p
    if p.startswith("icmp") or p in ("ipv6-icmp", "icmp6"):
        return "icmp"
    return "other"


def _service_token(service: str | None) -> str:
    s = (service or "").strip().lower()
    if s in ("", "-"):
        return ""
    return s.split(",")[0].strip()


def svc_class(proto: str | None, dport: int | None, service: str | None = "") -> str:
    """Coarse service class from (proto, dport, service): port table first, then the
    Zeek/tshark service string, else other_tcp / other_udp (icmp, gre, other by proto)."""
    p = norm_proto(proto)
    if p == "icmp":
        return "icmp"
    if p == "gre":
        return "gre"
    if p not in ("tcp", "udp"):
        return "other"
    table = SVC_TCP_PORTS if p == "tcp" else SVC_UDP_PORTS
    if dport is not None and int(dport) in table:
        return table[int(dport)]
    s = _service_token(service)
    if s in SVC_FROM_SERVICE:
        return SVC_FROM_SERVICE[s]
    return "other_tcp" if p == "tcp" else "other_udp"


# --------------------------------------------------------------------------- IP classes
_RFC1918 = tuple(ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
_LOCAL_V6 = tuple(ipaddress.ip_network(n) for n in ("fc00::/7", "fe80::/10"))


def _nets(prefixes: Sequence[str]) -> tuple:
    return tuple(ipaddress.ip_network(p, strict=False) for p in prefixes or ())


def classify_ip(ip: str, extra_nets: tuple = ()) -> tuple[bool, bool, str]:
    """(internal, unicast, /24 prefix) for one address.

    internal = RFC1918 ∪ per-dataset prefixes ∪ local-scope addresses (loopback, link-local,
    multicast, broadcast, unspecified, IPv6 ULA/link-local). unicast = not multicast, not
    255.255.255.255, not unspecified, not an internal IPv4 x.y.z.255 (subnet broadcast)."""
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False, False, str(ip)
    in_extra = any(a in n for n in extra_nets if n.version == a.version)
    if a.version == 4:
        internal = (in_extra or any(a in n for n in _RFC1918) or a.is_loopback or a.is_link_local
                    or a.is_multicast or a.is_unspecified or str(a) == "255.255.255.255"
                    or a in ipaddress.ip_network("0.0.0.0/8"))
        unicast = not (a.is_multicast or a.is_unspecified or str(a) == "255.255.255.255"
                       or a in ipaddress.ip_network("0.0.0.0/8")
                       or (internal and str(a).endswith(".255")))
        o = str(a).split(".")
        prefix = f"{o[0]}.{o[1]}.{o[2]}.0/24"
    else:
        internal = (in_extra or any(a in n for n in _LOCAL_V6) or a.is_loopback or a.is_multicast
                    or a.is_unspecified or a.is_link_local)
        unicast = not (a.is_multicast or a.is_unspecified)
        prefix = str(ipaddress.ip_network(f"{a}/48", strict=False))
    return bool(internal), bool(unicast), prefix


_IP_SCHEMA = {"ip": pl.Utf8, "internal": pl.Boolean, "unicast": pl.Boolean, "prefix24": pl.Utf8}


def classify_ips(ips, internal_prefixes: Sequence[str] = ()) -> pl.DataFrame:
    """classify_ip for many addresses (sorted, distinct). Canonical dotted-quad IPv4 is classified with
    vectorised polars expressions (identical results, tested); anything else falls back to classify_ip."""
    s = ips if isinstance(ips, pl.Series) else pl.Series("ip", [i for i in ips if i is not None], dtype=pl.Utf8)
    df = pl.DataFrame({"ip": s.cast(pl.Utf8).drop_nulls().unique()})
    parts = pl.col("ip").str.split_exact(".", 3)
    df = df.with_columns([parts.struct.field(f"field_{i}").cast(pl.Int64, strict=False).alias(f"o{i}") for i in range(4)])
    o0, o1, o2, o3 = (pl.col(f"o{i}") for i in range(4))
    canon = pl.concat_str([o0.cast(pl.Utf8), o1.cast(pl.Utf8), o2.cast(pl.Utf8), o3.cast(pl.Utf8)], separator=".")
    v4 = (pl.all_horizontal([pl.col(f"o{i}").is_between(0, 255) for i in range(4)]) & (canon == pl.col("ip"))).fill_null(False)
    df = df.with_columns(v4.alias("_v4"))
    u32 = o0 * 16777216 + o1 * 65536 + o2 * 256 + o3
    extra = pl.lit(False)
    for n in _nets(internal_prefixes):
        if n.version == 4:
            extra = extra | u32.is_between(int(n.network_address), int(n.broadcast_address))
    rfc = (o0 == 10) | ((o0 == 172) & o1.is_between(16, 31)) | ((o0 == 192) & (o1 == 168))
    mcast, zero8 = o0.is_between(224, 239), o0 == 0
    bcast = (o0 == 255) & (o1 == 255) & (o2 == 255) & (o3 == 255)
    internal = extra | rfc | (o0 == 127) | ((o0 == 169) & (o1 == 254)) | mcast | bcast | zero8
    unicast = ~(mcast | bcast | zero8 | (internal & (o3 == 255)))
    prefix = pl.concat_str([o0.cast(pl.Utf8), o1.cast(pl.Utf8), o2.cast(pl.Utf8), pl.lit("0/24")], separator=".")
    a = df.filter(pl.col("_v4")).select("ip", internal.alias("internal"), unicast.alias("unicast"),
                                       prefix.alias("prefix24"))
    rest = df.filter(~pl.col("_v4"))["ip"].to_list()
    nets = _nets(internal_prefixes)
    rows = [classify_ip(i, nets) for i in rest]
    b = pl.DataFrame({"ip": rest, "internal": [r[0] for r in rows], "unicast": [r[1] for r in rows],
                      "prefix24": [r[2] for r in rows]}, schema=_IP_SCHEMA)
    return pl.concat([a.cast(_IP_SCHEMA), b]).sort("ip")


_V4_RE = r"^(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])(\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])){3}$"


def ip_flag_exprs(col: str, internal_prefixes: Sequence[str] = ()) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """(is canonical IPv4, internal, unicast) as vectorised expressions on a string column; equal to
    classify_ip for every canonical dotted-quad IPv4 (callers fall back to classify_ip otherwise)."""
    c = pl.col(col)
    v4 = c.str.contains(_V4_RE).fill_null(False)
    o0 = c.str.extract(r"^(\d+)\.", 1).cast(pl.Int64, strict=False)
    o1 = c.str.extract(r"^\d+\.(\d+)\.", 1).cast(pl.Int64, strict=False)
    extra = pl.lit(False)
    nets = [n for n in _nets(internal_prefixes) if n.version == 4]
    if nets:
        o2 = c.str.extract(r"^\d+\.\d+\.(\d+)\.", 1).cast(pl.Int64, strict=False)
        o3 = c.str.extract(r"\.(\d+)$", 1).cast(pl.Int64, strict=False)
        u32 = o0 * 16777216 + o1 * 65536 + o2 * 256 + o3
        for n in nets:
            extra = extra | u32.is_between(int(n.network_address), int(n.broadcast_address))
    rfc = (o0 == 10) | ((o0 == 172) & o1.is_between(16, 31)) | ((o0 == 192) & (o1 == 168))
    mcast, zero8 = o0.is_between(224, 239), o0 == 0
    bcast = c == "255.255.255.255"
    internal = extra | rfc | (o0 == 127) | ((o0 == 169) & (o1 == 254)) | mcast | bcast | zero8
    unicast = ~(mcast | bcast | zero8 | (internal & c.str.ends_with(".255")))
    return v4, internal.fill_null(False), unicast.fill_null(False)


def prefix24_expr(col: str) -> pl.Expr:
    """/24 prefix of a canonical dotted-quad IPv4 (as classify_ip); null otherwise."""
    c = pl.col(col)
    return pl.when(c.str.contains(_V4_RE).fill_null(False)).then(c.str.replace(r"\.\d+$", ".0/24")).otherwise(
        pl.lit(None, dtype=pl.Utf8))


def is_internal(ip: str, internal_prefixes: Sequence[str] = ()) -> bool:
    return classify_ip(ip, _nets(internal_prefixes))[0]


# --------------------------------------------------------------------------- hashing
try:  # single definition (io/schema.py docstring), shared with ingest
    from certichain.io.hashing import tiehash  # noqa: F401
except ImportError:  # pragma: no cover
    def tiehash(src, sport, dst, dport, proto) -> int:
        """First 8 bytes of BLAKE3(f"{src}|{sport}|{dst}|{dport}|{proto}") as u64 little-endian."""
        return int.from_bytes(blake3.blake3(f"{src}|{sport}|{dst}|{dport}|{proto}".encode()).digest()[:8], "little")


def id_hash(x) -> int:
    """u64 hash of an id (used for 'unit-id hash' tie-breaks)."""
    return int.from_bytes(blake3.blake3(str(x).encode()).digest()[:8], "little")


# --------------------------------------------------------------------------- content regexes
EXEC_MIME_RE = (r"(application/x-(executable|elf|sharedlib|dosexec|msdownload|msdos-program|mach-binary|sh|"
                r"shellscript|object|pie-executable)|text/x-(shellscript|sh|python|perl))")
EXEC_EXT_RE = re.compile(r"\.(sh|bin|elf|exe|dll|so|apk|jar|py|pl|arm|arm[4-7]|armv[4-7]l?|mips|mpsl|mipsel|"
                         r"mips64|x86|x86_64|x64|i[3-6]86|m68k|ppc|ppc440|sh4|spc|sparc|arc|aarch64)$")
TOOL_UA_RE = (r"(wget|curl|busybox|python-requests|python-urllib|libwww-perl|lwp-|go-http-client|"
              r"nmap|masscan|zgrab|sqlmap|nikto|hydra|okhttp)")
INJECTION_RE = re.compile(
    # SQL injection
    r"(union(\s|\+|/\*.*?\*/)+(all(\s|\+)+)?select\b|\bselect\b.+\bfrom\b|'\s*(or|and)\s+'?\w*'?\s*=|"
    r"\bor\b\s+\d+\s*=\s*\d+|sleep\(\s*\d|benchmark\(|information_schema|xp_cmdshell|"
    # XSS
    r"<\s*script|javascript:|\bon(error|load|mouseover|focus)\s*=|alert\s*\(|document\.cookie|<\s*svg|<\s*iframe|"
    # command injection
    r"[;|`&]\s*(wget|curl|tftp|sh|bash|busybox|chmod|cat|nc|rm|echo|cd|ping)\b|\$\(\s*(wget|curl|sh|echo|cat|id)\b|"
    r"/bin/(ba)?sh|\bcd\s+/tmp|"
    # traversal
    r"\.\./|\.\.\\|/etc/passwd|/etc/shadow|/proc/self/environ|boot\.ini|win\.ini)")


def uri_flags(uri: str | None) -> tuple[bool, bool]:
    """(injection regex hit, executable path extension) for one HTTP URI (decoded twice)."""
    if not uri:
        return False, False
    u = uri
    for _ in range(2):
        try:
            u = unquote_plus(u)
        except Exception:  # noqa: BLE001 — malformed escapes are attacker-controlled
            break
    u = u.lower()
    path = u.split("?", 1)[0].split("#", 1)[0]
    return bool(INJECTION_RE.search(u)), bool(EXEC_EXT_RE.search(path))


def shannon(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return float(-sum((c / n) * math.log2(c / n) for c in Counter(s).values()))


def qname_entropy(q: str | None) -> float:
    """Shannon entropy (bits/char) of the left-most label of a DNS name."""
    if not q:
        return float("nan")
    labels = [x for x in q.strip(".").split(".") if x]
    return shannon(labels[0]) if labels else float("nan")


# --------------------------------------------------------------------------- record preparation
_STR_NULLS = ("", "-", "(empty)")
_ESTABLISHED = ("SF", "S1", "S2", "S3", "RSTO", "RSTR")
_FAILED = ("S0", "REJ", "RSTO", "RSTOS0")
# RECORD_SCHEMA columns Alg. 1 reads (src_file / src_sha256 / src_locator / ts_res / bulk are not needed)
NEEDED_COLS = ("rec_id", "ts", "te", "src", "sport", "dst", "dport", "proto", "service", "obytes", "rbytes",
               "opkts", "rpkts", "conn_state", "history", "n_merged", "dns_q", "dns_rcode", "http_method",
               "http_uri", "http_host", "http_ua", "http_status", "http_resp_mime", "tls_sni")
_OPTIONAL = ("dns_q", "dns_rcode", "http_method", "http_uri", "http_host", "http_ua", "http_status",
             "http_resp_mime", "tls_sni")
DIR_NAMES = {0: "ext->ext", 1: "ext->int", 2: "int->ext", 3: "int->int"}    # dir_code = 2*src_int + dst_int


def _clean(e: pl.Expr, lower: bool = False) -> pl.Expr:
    e = e.cast(pl.Utf8).str.strip_chars()
    e = pl.when(e.is_in(list(_STR_NULLS))).then(pl.lit(None, dtype=pl.Utf8)).otherwise(e)
    return e.str.to_lowercase() if lower else e


def _map(col: str, mapping, dtype, default=None) -> pl.Expr:
    """replace_strict from a dict or an (old, new) pair of Series (no Python dict for big maps)."""
    if isinstance(mapping, tuple):
        old, new = mapping
        if len(old) == 0:
            return pl.lit(default, dtype=dtype)
        return pl.col(col).replace_strict(old, new, default=default, return_dtype=dtype)
    if not mapping:
        return pl.lit(default, dtype=dtype)
    return pl.col(col).replace_strict(mapping, default=default, return_dtype=dtype)


def prepare_records(records: pl.DataFrame, internal_prefixes: Sequence[str] = ()) -> tuple[pl.DataFrame, dict]:
    """RECORD_SCHEMA frame → lean per-record frame: svc_class, dir_code, IP classes and the
    per-record predicates the unit aggregates need (optional columns that are entirely null are
    carried as literals). Records without ts/src/dst are dropped (counted)."""
    n0 = records.height
    df = records.select([pl.col(c).cast(RECORD_SCHEMA[c]) if c in records.columns
                         else pl.lit(None, dtype=RECORD_SCHEMA[c]).alias(c) for c in NEEDED_COLS])
    valid = df.select((pl.col("ts").is_not_null() & pl.col("ts").is_finite()
                       & pl.col("src").is_not_null() & (pl.col("src") != "")
                       & pl.col("dst").is_not_null() & (pl.col("dst") != "")).alias("ok"))["ok"]
    if not bool(valid.all()):
        df = df.filter(valid)
    info = {"n_records_in": n0, "n_records_invalid": n0 - df.height}
    have = {c for c in _OPTIONAL if c in records.columns and df[c].null_count() < df.height}

    def opt(c: str, lower: bool = False) -> pl.Expr:
        if c not in have:
            return pl.lit(None, dtype=RECORD_SCHEMA[c])
        return _clean(pl.col(c), lower) if RECORD_SCHEMA[c] == pl.Utf8 else pl.col(c)

    df = df.with_columns(
        pl.col("proto").fill_null("").str.strip_chars().str.to_lowercase(),
        pl.col("service").fill_null("").str.strip_chars().str.to_lowercase(),
        pl.col("conn_state").fill_null("").str.strip_chars().str.to_uppercase(),
        pl.col("history").fill_null(""),
        *[pl.col(c).fill_null(0).clip(lower_bound=0) for c in ("obytes", "rbytes", "opkts", "rpkts")],
        pl.col("n_merged").fill_null(1).cast(pl.Int64).clip(lower_bound=1).alias("n_conn"),
        pl.max_horizontal(pl.col("ts"), pl.col("te").fill_nan(None).fill_null(pl.col("ts"))).alias("te_eff"),
        opt("http_method").str.to_uppercase().alias("http_method"), opt("http_uri").alias("http_uri"),
        opt("http_host", True).alias("http_host"), opt("http_ua", True).alias("http_ua"),
        opt("http_resp_mime", True).alias("http_resp_mime"), opt("tls_sni", True).alias("tls_sni"),
        opt("dns_q", True).str.strip_chars_end(".").alias("q"), opt("dns_rcode").alias("dns_rcode"),
        opt("http_status").alias("http_status"),
    ).drop("te", "n_merged", "dns_q")
    pmap = {p: norm_proto(p) for p in df["proto"].unique().to_list()}
    df = df.with_columns(_map("proto", pmap, pl.Utf8, "other").alias("proto"))
    df = df.with_columns((pl.col("te_eff") - pl.col("ts")).alias("dur"))

    # svc_class per distinct (proto, dport, service)
    key = pl.concat_str([pl.col("proto"), pl.col("dport").cast(pl.Utf8).fill_null("~"), pl.col("service")],
                        separator="|")
    combos = df.select("proto", "dport", "service").unique()
    smap = {f"{p}|{'~' if d is None else d}|{s_}": svc_class(p, d, s_) for p, d, s_ in combos.iter_rows()}
    df = df.with_columns(key.replace_strict(smap, default="other", return_dtype=pl.Utf8).alias("svc_class"))

    # IP classes: vectorised for canonical IPv4, classify_ip for anything else (e.g. IPv6)
    flags = []
    for col, short in (("src", "src"), ("dst", "dst")):
        v4, it, un = ip_flag_exprs(col, internal_prefixes)
        flags += [v4.alias(f"_{short}_v4"), it.alias(f"{short}_int"), un.alias(f"{short}_uni")]
    df = df.with_columns(flags)
    for short in ("src", "dst"):
        odd = df.filter(~pl.col(f"_{short}_v4"))[short].unique()
        if odd.len():
            t = classify_ips(odd, internal_prefixes)
            df = df.with_columns(
                pl.when(pl.col(f"_{short}_v4")).then(pl.col(f"{short}_int"))
                .otherwise(_map(short, (t["ip"], t["internal"]), pl.Boolean, False)).alias(f"{short}_int"),
                pl.when(pl.col(f"_{short}_v4")).then(pl.col(f"{short}_uni"))
                .otherwise(_map(short, (t["ip"], t["unicast"]), pl.Boolean, False)).alias(f"{short}_uni"))
    df = df.drop("_src_v4", "_dst_v4")
    df = df.with_columns((pl.col("src_int").cast(pl.Int8) * 2 + pl.col("dst_int").cast(pl.Int8)).alias("dir_code"))

    # URI flags / qname entropy per distinct value
    if "http_uri" in have:
        uris = df["http_uri"].drop_nulls().unique().to_list()
        uf = {u: uri_flags(u) for u in uris}
        inj = _map("http_uri", {u: a for u, (a, _) in uf.items()}, pl.Boolean, False)
        exe = _map("http_uri", {u: b for u, (_, b) in uf.items()}, pl.Boolean, False)
    else:
        inj = exe = pl.lit(False)
    if "dns_q" in have:
        qs = df["q"].drop_nulls().unique().to_list()
        qent = _map("q", {q: qname_entropy(q) for q in qs}, pl.Float64, None)
    else:
        qent = pl.lit(None, dtype=pl.Float64)

    by = pl.col("obytes") + pl.col("rbytes")
    has_http = pl.col("http_method").is_not_null()
    method = pl.col("http_method")
    exec_mime = (pl.col("http_resp_mime").fill_null("").str.contains(EXEC_MIME_RE)
                 if "http_resp_mime" in have else pl.lit(False))
    tool_ua = pl.col("http_ua").fill_null("").str.contains(TOOL_UA_RE) if "http_ua" in have else pl.lit(False)
    tcp = pl.col("proto") == "tcp"
    df = df.with_columns(
        inj.alias("inj"), exe.alias("exec_ext"), qent.alias("q_ent"),
        (pl.col("conn_state").is_in(list(_FAILED)) | (by == 0)).alias("failed"),
        pl.col("conn_state").is_in(["REJ", "S0"]).alias("rejs0"),
        (pl.col("obytes") > 0).alias("payload"),
        (pl.col("conn_state").is_in(list(_ESTABLISHED))
         | (pl.col("conn_state").is_in(["OTH", ""]) & (pl.col("obytes") > 0) & (pl.col("rbytes") > 0))
         ).alias("established"),
        (tcp & ((pl.col("conn_state") == "OTH")
                | ((pl.col("history") != "") & ~pl.col("history").str.contains("S", literal=True)))
         ).alias("nosyn"),
        ((pl.col("proto") == "icmp") & (pl.col("sport").is_null() | (pl.col("sport") < 0) | (pl.col("sport") == 8))
         ).alias("icmp_echo"),
        has_http.alias("has_http"),
        pl.col("http_status").is_not_null().alias("has_status"),
        (pl.col("http_status") == 401).fill_null(False).alias("s401"),
        exec_mime.alias("exec_mime"),
        tool_ua.alias("tool_ua"),
        (pl.col("q").is_not_null() | pl.col("dns_rcode").is_not_null()).alias("has_dns"),
        (pl.col("dns_rcode") == 3).fill_null(False).alias("nx"),
        pl.col("dns_rcode").is_not_null().alias("has_rcode"),
        (pl.col("tls_sni").is_not_null() | pl.col("service").str.contains("ssl|tls")).alias("has_tls"),
        ((pl.col("dur") >= UT.C2_LONG_DUR)
         & (by.cast(pl.Float64) / pl.max_horizontal(pl.col("dur"), pl.lit(1e-9)) < UT.C2_LOW_RATE_BPS)
         ).alias("longlow"),
        (pl.col("http_host").str.replace(r":\d+$", "").str.strip_chars_end(".") if "http_host" in have
         else pl.lit(None, dtype=pl.Utf8)).alias("name_host"),
        (pl.col("tls_sni").str.strip_chars_end(".") if "tls_sni" in have
         else pl.lit(None, dtype=pl.Utf8)).alias("name_sni"),
    )
    df = df.with_columns(
        (pl.col("exec_mime") | pl.col("exec_ext")).alias("exec_flag"),
        ((has_http & (method == "GET") & (pl.col("rbytes") >= UT.DL_MIN_RESP_BYTES)
          & (pl.col("exec_mime") | pl.col("exec_ext") | pl.col("tool_ua")))
         | (~has_http & tcp & (pl.col("dport") == 80) & (pl.col("conn_state") == "SF")
            & (pl.col("rbytes") >= UT.DL_MIN_RESP_BYTES) & (pl.col("rbytes") > pl.col("obytes")))
         ).fill_null(False).alias("dl_rec"),
        ((method.is_in(["POST", "PUT"]) & (pl.col("obytes") >= UT.UP_MIN_BYTES))
         | (~has_http & (pl.col("obytes") >= UT.UP_MIN_BYTES) & pl.col("dport").is_in(list(UT.UP_PORTS))
            & (pl.col("obytes") > pl.col("rbytes")))
         ).fill_null(False).alias("up_rec"),
    )
    keep = ["rec_id", "ts", "te_eff", "dur", "src", "sport", "dst", "dport", "proto", "svc_class", "dir_code",
            "src_int", "src_uni", "dst_int", "dst_uni", "obytes", "rbytes", "opkts", "rpkts", "n_conn",
            "failed", "rejs0", "payload", "established", "nosyn", "icmp_echo", "has_http", "has_status", "s401", "inj",
            "exec_flag", "tool_ua", "dl_rec", "up_rec", "has_dns", "nx", "has_rcode", "q", "q_ent", "has_tls",
            "longlow", "name_host", "name_sni"]
    info["optional_present"] = sorted(have)
    return df.select(keep), info


# --------------------------------------------------------------------------- scope annotations
def default_annotations(prep: pl.DataFrame) -> pl.DataFrame:
    """Annotations used when sessionising without a scope (tests / whole-window units)."""
    return prep.with_columns(
        pl.lit(False).alias("src_is_seed"), pl.lit(False).alias("dst_is_seed"),
        pl.lit(True).alias("src_in_S"), pl.lit(True).alias("dst_in_S"),
        pl.lit(False).alias("src_infra"), pl.lit(False).alias("dst_infra"),
        pl.when(pl.col("dst_int")).then(pl.lit(2)).otherwise(pl.lit(3)).cast(pl.Int8).alias("dst_prio"),
        (~pl.col("dst_int")).alias("dst_infra_or_ext"),
    )


# --------------------------------------------------------------------------- steps 2–3
KEY = ["src", "svc_class", "dir_code"]


def _bursts(rec: pl.DataFrame, p: UnitParams) -> pl.DataFrame:
    """Steps 2–3 on a narrow (key, ts, te, dst) frame; returns ``rec`` (row order unchanged) with the
    pre-merge unit id ``pu``. The aggregates downstream are order-free, so the wide frame is never sorted."""
    n = rec.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("_i"), *KEY, "ts", "te_eff", "dst").sort(KEY + ["ts"])
    newg = ((pl.col("src") != pl.col("src").shift(1)) | (pl.col("svc_class") != pl.col("svc_class").shift(1))
            | (pl.col("dir_code") != pl.col("dir_code").shift(1)))
    n = n.with_columns(newg.fill_null(True).cast(pl.Int64).cum_sum().alias("_g"))
    n = n.with_columns(pl.col("te_eff").cum_max().shift(1).over("_g").alias("_prev"))
    n = n.with_columns((pl.col("_prev").is_null() | ((pl.col("ts") - pl.col("_prev")) > p.tau_gap)).alias("_nb"))
    n = n.select("_i", "dst", pl.col("_nb").cast(pl.Int64).cum_sum().alias("burst"))
    n = n.with_columns((pl.col("dst").n_unique().over("burst") >= p.F_fan).alias("fanout"))
    dk = pl.when(pl.col("fanout")).then(pl.lit(_FANOUT_KEY)).otherwise(pl.col("dst"))
    n = n.select("_i", "fanout", pl.struct(pl.col("burst"), dk.alias("_dk")).rank("dense").cast(pl.Int64).alias("pu"))
    n = n.sort("_i")
    return rec.with_columns(n["fanout"], n["pu"])


# --------------------------------------------------------------------------- step 4 (aggregates)
def _aggregate_light(rec: pl.DataFrame) -> pl.DataFrame:
    """Per-unit aggregates needed by the type rules and the merge keys."""
    gaps = pl.col("ts").sort().diff().drop_nulls()
    agg = rec.group_by("pu").agg(
        pl.col("ts").min().alias("ts"),
        pl.col("te_eff").max().alias("te"),
        pl.col("src").first().alias("src"),
        pl.col("svc_class").first().alias("svc_class"),
        pl.col("fanout").first().alias("fanout"),
        pl.col("dst").min().alias("dst1"),
        pl.col("dst_infra_or_ext").sort_by("dst").first().alias("dst1_infra_or_ext"),
        pl.col("dir_code").mode().sort().first().alias("dir_code"),
        pl.col("proto").mode().sort().first().alias("proto"),
        pl.len().cast(pl.Int32).alias("n_records"),
        pl.col("n_conn").sum().alias("n_conns"),
        pl.col("obytes").sum().alias("obytes"), pl.col("rbytes").sum().alias("rbytes"),
        pl.col("opkts").sum().alias("opkts"), pl.col("rpkts").sum().alias("rpkts"),
        pl.col("failed").mean().alias("fail_frac"),
        pl.col("rejs0").sum().cast(pl.Int64).alias("n_rejs0"),
        pl.col("dst").n_unique().cast(pl.Int32).alias("n_dst"),
        pl.col("dport").drop_nulls().n_unique().cast(pl.Int32).alias("n_dports"),
        pl.col("dport").drop_nulls().mode().sort().first().alias("dport_mode"),
        pl.col("dur").median().alias("med_dur"),
        pl.col("dur").max().alias("max_dur"),
        pl.col("payload").mean().alias("frac_payload"),
        (gaps.std(ddof=0) / gaps.mean()).alias("start_cv"),
        pl.col("established").sum().cast(pl.Int64).alias("n_est"),
        pl.col("established").mean().alias("frac_est"),
        pl.col("obytes").filter(pl.col("established")).sum().alias("est_obytes"),
        pl.col("rbytes").filter(pl.col("established")).sum().alias("est_rbytes"),
        pl.col("dur").filter(pl.col("established")).max().fill_null(0.0).alias("est_max_dur"),
        pl.col("longlow").any().alias("any_longlow"),
        pl.col("nosyn").mean().alias("frac_nosyn"),
        pl.col("icmp_echo").mean().alias("frac_icmp_echo"),
        pl.col("has_http").any().alias("http_any"),
        pl.col("has_status").sum().cast(pl.Int64).alias("n_status"),
        pl.col("s401").sum().cast(pl.Int64).alias("n_401"),
        pl.col("inj").any().alias("any_inj"),
        pl.col("dl_rec").any().alias("any_dl"),
        pl.col("up_rec").any().alias("any_up"),
        pl.col("exec_flag").any().alias("any_exec"),
        pl.col("tool_ua").any().alias("any_tool_ua"),
        pl.col("has_dns").any().alias("has_dns"),
        pl.col("nx").sum().cast(pl.Int64).alias("n_nx"),
        pl.col("has_rcode").sum().cast(pl.Int64).alias("n_rcode"),
        pl.col("has_tls").any().alias("has_tls"),
    )
    agg = agg.with_columns(pl.col("dir_code").replace_strict(DIR_NAMES, return_dtype=pl.Utf8).alias("dir_class"))
    # CV of inter-start gaps: NaN if < 3 records or undefined (all starts equal)
    agg = agg.with_columns(pl.when((pl.col("n_records") >= 3) & pl.col("start_cv").is_finite())
                           .then(pl.col("start_cv")).otherwise(pl.lit(float("nan"))).alias("start_cv"))
    qd = rec.filter(pl.col("q").is_not_null()).select("pu", "q", "q_ent").unique(["pu", "q"])
    if qd.height:
        qd = qd.group_by("pu").agg(pl.col("q_ent").median().alias("qent"))
        agg = agg.join(qd, on="pu", how="left")
    else:
        agg = agg.with_columns(pl.lit(None, dtype=pl.Float64).alias("qent"))
    return agg


def _ranked(rec: pl.DataFrame, col: str, limit: int, name: str) -> pl.DataFrame:
    g = rec.filter(pl.col(col).is_not_null()).group_by(["pu", col]).agg(pl.len().alias("_n"))
    return g.group_by("pu").agg(pl.col(col).sort_by(["_n", col], descending=[True, False]).head(limit).alias(name))


def _aggregate_lists(rec: pl.DataFrame, p: UnitParams, infra_tokens: dict | None = None) -> pl.DataFrame:
    """List-valued unit columns and scope flags (computed once, after the merge loop)."""
    base = rec.group_by("pu").agg(
        pl.col("rec_id").sort().alias("rec_ids"),
        (pl.col("src_is_seed") | pl.col("dst_is_seed")).any().alias("touches_seed"),
        pl.col("src_is_seed").any().alias("src_is_seed"),
        (pl.col("src_in_S") | pl.col("dst_in_S")).any().alias("touches_S"),
        pl.col("src_in_S").any().alias("src_in_S"),
    )
    # destinations: scope members / seed first, then INFRA, internal, external; then by
    # #records desc, first contact, address (deterministic, label-free)
    g = rec.group_by(["pu", "dst"]).agg(pl.len().alias("_n"), pl.col("ts").min().alias("_t0"),
                                       pl.col("dst_prio").min().alias("_pr"))
    order, desc = ["_pr", "_n", "_t0"], [False, True, False]
    parts = [g.group_by("pu").agg(pl.col("dst").sort_by(order + ["dst"], descending=desc + [False])
                                  .head(p.max_dsts).alias("dsts"))]
    g = g.with_columns(prefix24_expr("dst").alias("dst_prefix"))
    odd = g.filter(pl.col("dst_prefix").is_null())["dst"].unique()
    if odd.len():
        pt = classify_ips(odd)
        g = g.with_columns(pl.coalesce(pl.col("dst_prefix"), pl.col("dst").replace_strict(
            pt["ip"], pt["prefix24"], default=None, return_dtype=pl.Utf8)).alias("dst_prefix"))
    gp = (g
          .group_by(["pu", "dst_prefix"]).agg(pl.col("_n").sum(), pl.col("_t0").min(), pl.col("_pr").min()))
    parts.append(gp.group_by("pu").agg(pl.col("dst_prefix").sort_by(order + ["dst_prefix"], descending=desc + [False])
                                       .head(p.max_prefixes).alias("dst_prefixes")))
    parts += [_ranked(rec, "dport", p.max_dports, "dports"), _ranked(rec, "q", p.max_names, "dns_names"),
              _ranked(rec, "name_host", p.max_names, "http_hosts"), _ranked(rec, "name_sni", p.max_names, "tls_snis")]
    if infra_tokens:
        ips = list(infra_tokens)
        toks = pl.concat([rec.filter(pl.col("src_infra")).select("pu", pl.col("src").alias("ip")),
                          rec.filter(pl.col("dst_infra")).select("pu", pl.col("dst").alias("ip"))]).unique()
        toks = toks.with_columns(
            pl.col("ip").replace_strict(ips, [infra_tokens[i] for i in ips], return_dtype=pl.Utf8).alias("tok"),
            pl.col("ip").replace_strict(ips, [int(infra_tokens[i][5:]) for i in ips], return_dtype=pl.Int64).alias("idx"))
        parts.append(toks.group_by("pu").agg(pl.col("tok").sort_by("idx").alias("infra")))
    for part in parts:
        base = base.join(part, on="pu", how="left")
    if "infra" not in base.columns:
        base = base.with_columns(pl.lit(None, dtype=pl.List(pl.Utf8)).alias("infra"))
    list_cols = {"dsts": pl.Utf8, "dst_prefixes": pl.Utf8, "dports": pl.Int32, "dns_names": pl.Utf8,
                 "http_hosts": pl.Utf8, "tls_snis": pl.Utf8, "infra": pl.Utf8}
    return base.with_columns([pl.col(c).cast(pl.List(t)).fill_null(pl.lit([], dtype=pl.List(t)))
                              for c, t in list_cols.items()])


def _first_tiehash(rec: pl.DataFrame) -> pl.DataFrame:
    """min tiehash over each unit's earliest-starting record(s)."""
    firsts = (rec.select("pu", "ts", "src", "sport", "dst", "dport", "proto")
              .filter(pl.col("ts") == pl.col("ts").min().over("pu"))
              .select("pu", "src", "sport", "dst", "dport", "proto").unique())
    h = [tiehash(s_, sp, d, dp, pr) for _, s_, sp, d, dp, pr in firsts.iter_rows()]
    firsts = firsts.with_columns(pl.Series("first_tiehash", h, dtype=pl.UInt64))
    return firsts.group_by("pu").agg(pl.col("first_tiehash").min())


def content_flags_expr() -> pl.Expr:
    nx_heavy = (pl.col("n_rcode") > 0) & (pl.col("n_nx").cast(pl.Float64)
                                         / pl.max_horizontal(pl.col("n_rcode").cast(pl.Float64), pl.lit(1.0))
                                         >= UT.DNS_NX_RATE)
    hi_ent = pl.col("qent").fill_nan(None).fill_null(0.0) >= UT.DNS_ENTROPY
    return (pl.col("any_exec").cast(pl.Int32) * 1 + pl.col("any_inj").cast(pl.Int32) * 2
            + pl.col("any_tool_ua").cast(pl.Int32) * 4 + nx_heavy.cast(pl.Int32) * 8
            + hi_ent.cast(pl.Int32) * 16).alias("content_flags")


# --------------------------------------------------------------------------- step 5 (activity merge)
def _merge_map(agg: pl.DataFrame, tau_merge: float) -> tuple[pl.DataFrame, int]:
    a = agg.select("pu", "src", "svc_class", "type", "fanout", "dst1", "ts", "te").with_columns(
        pl.when(pl.col("fanout")).then(pl.lit(_FANOUT_KEY)).otherwise(pl.col("dst1")).alias("_mk"))
    mk = ["src", "svc_class", "type", "_mk"]
    a = a.sort(mk + ["ts"])
    a = a.with_columns(pl.col("te").cum_max().shift(1).over(mk).alias("_prev"))
    a = a.with_columns((pl.col("_prev").is_null() | ((pl.col("ts") - pl.col("_prev")) > tau_merge)).alias("_nc"))
    a = a.with_columns(pl.col("_nc").cast(pl.Int64).cum_sum().alias("npu"))
    n_new = int(a["npu"].max()) if a.height else 0
    return a.select("pu", "npu"), a.height - n_new


@dataclass
class SessionizeResult:
    units: pl.DataFrame          # one row per unit: UNIT_SCHEMA columns (no unit_id) + extras
    rec_unit: pl.DataFrame       # rec_id → pu
    info: dict = field(default_factory=dict)


def sessionize(rec, p: UnitParams = UnitParams(), infra_tokens: dict | None = None) -> SessionizeResult:
    """Alg. 1 steps 2–5 on prepared + annotated records (see prepare_records / scope.annotate_records).

    ``rec`` may be a one-element list holding the frame: it is popped, so the caller keeps no
    reference and the unsorted frame is freed as soon as the sorted copy exists (memory)."""
    if isinstance(rec, list):
        rec = rec.pop()
    info = {"n_records": rec.height, "merge_rounds": 0, "merges": 0, "merge_rounds_capped": False}
    if rec.height == 0:
        return SessionizeResult(units=pl.DataFrame(), rec_unit=pl.DataFrame({"rec_id": [], "pu": []}), info=info)
    rec = _bursts(rec, p)
    info["n_units_pre_merge"] = int(rec["pu"].n_unique())
    agg = UT.assign_types(_aggregate_light(rec))
    for rnd in range(p.max_merge_rounds):
        mapping, n_merged = _merge_map(agg, p.tau_merge)
        if n_merged == 0:
            break
        info["merge_rounds"] = rnd + 1
        info["merges"] += n_merged
        rec = rec.with_columns(pl.col("pu").replace_strict(mapping["pu"], mapping["npu"], return_dtype=pl.Int64))
        agg = UT.assign_types(_aggregate_light(rec))
    else:
        info["merge_rounds_capped"] = True
    lists = _aggregate_lists(rec, p, infra_tokens)
    agg = agg.join(lists, on="pu", how="left").join(_first_tiehash(rec), on="pu", how="left")
    rates = UT.rate_exprs()
    agg = agg.with_columns(
        (pl.col("te") - pl.col("ts")).alias("dur"),
        rates["mean_bytes"].alias("mean_bytes"),
        content_flags_expr(),
    )
    info["n_units"] = agg.height
    return SessionizeResult(units=agg, rec_unit=rec.select("rec_id", "pu"), info=info)


# --------------------------------------------------------------------------- full Alg. 1
@dataclass
class EventUnits:
    units: pl.DataFrame          # UNIT_SCHEMA, unit_id dense in (ts, first_tiehash) order
    side: pl.DataFrame           # unit_id + extra label-free columns (see SIDE_SCHEMA)
    scope: "object"              # units.scope.ScopeResult
    info: dict = field(default_factory=dict)


SIDE_SCHEMA: dict[str, pl.DataType] = {
    "unit_id": pl.Int64,
    "n_conns": pl.Int64,             # Σ n_merged (underlying flows; bulk records count > 1)
    "touches_seed": pl.Boolean,      # src or any dst (full set) is a seed
    "src_is_seed": pl.Boolean,
    "src_in_scope": pl.Boolean,      # kept because src ∈ S (else because dst-set ∩ S ≠ ∅)
    "hop": pl.Int8,                  # 0 if touches_seed else 1 (U_max ordering)
    "seed_dist": pl.Float64,         # s to the nearest seed-unit interval (0 if overlapping)
    "first_tiehash": pl.UInt64,      # min tiehash of the unit's earliest record(s)
    "attck": pl.List(pl.Utf8),       # ATT&CK hint ids of the type (Appendix C)
    "frac_est": pl.Float64,
    "med_dur": pl.Float64,
    "dport_mode": pl.Int32,
}


def finalize_units(kept: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Order units by (ts, first_tiehash) and emit (UNIT_SCHEMA frame, side frame)."""
    if kept.height == 0:
        return pl.DataFrame(schema=UNIT_SCHEMA), pl.DataFrame(schema=SIDE_SCHEMA)
    kept = kept.sort(["ts", "first_tiehash", "src", "svc_class", "type"]).with_row_index("unit_id")
    kept = kept.with_columns(pl.col("unit_id").cast(pl.Int64))
    units = conform(kept, UNIT_SCHEMA)
    types = kept["type"].unique().sort().to_list()
    hints = pl.DataFrame({"type": types, "attck": [list(UT.attack_hints(t)) for t in types]},
                         schema={"type": pl.Utf8, "attck": pl.List(pl.Utf8)})
    side = kept.join(hints, on="type", how="left").with_columns(pl.col("src_in_S").alias("src_in_scope"))
    side = conform(side, SIDE_SCHEMA).sort("unit_id")
    return units, side


def event_units(records, seeds: Sequence[str], params: UnitParams = UnitParams(),
                internal_prefixes: Sequence[str] = (), *, use_scope: bool = True) -> EventUnits:
    """Alg. 1 for one incident: records (RECORD_SCHEMA frame, or a one-element list holding it so
    that it can be freed early) + case metadata (seeds) → units.

    Scope is applied at the record level before sessionising (exactly equivalent, cheaper):
    units are formed independently per (src, svc_class) group, so only groups with src ∈ S
    or with ≥ 1 record to a host in S can yield a kept unit; the unit-level rule
    (src ∈ S or dst-set ∩ S ≠ ∅, on the full dst set) and the U_max cap are applied after."""
    from certichain.units import scope as SC

    if isinstance(records, list):               # holder: the caller keeps no reference (memory)
        records = records.pop()
    prep, info = prepare_records(records, internal_prefixes)
    del records
    sc = SC.compute_scope(prep, seeds, params.F_infra, internal_prefixes)
    sub = SC.prefilter_records(prep, sc) if use_scope else prep
    del prep
    info["n_records_scope_prefilter"] = sub.height
    box = [SC.annotate_records(sub, sc)]
    del sub
    res = sessionize(box, params, infra_tokens=sc.infra)
    info.update({f"sess_{k}": v for k, v in res.info.items()})
    if res.units.height == 0:
        units, side = finalize_units(pl.DataFrame())
        info.update({"n_units": 0, "U_max_hit": False, "n_units_pre_scope": 0})
        return EventUnits(units=units, side=side, scope=sc, info=info)
    kept, sinfo = SC.select_units(res.units, sc, params.U_max, use_scope=use_scope)
    info.update(sinfo)
    units, side = finalize_units(kept)
    info["n_units"] = units.height
    return EventUnits(units=units, side=side, scope=sc, info=info)


def unit_record_map(units: pl.DataFrame) -> pl.DataFrame:
    """rec_id → unit_id (from the rec_ids pointer lists)."""
    if units.height == 0:
        return pl.DataFrame(schema={"unit_id": pl.Int64, "rec_id": pl.Int64})
    return units.select("unit_id", "rec_ids").explode("rec_ids").rename({"rec_ids": "rec_id"}).drop_nulls("rec_id")


def merged_intervals(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(starts) == 0:
        return np.zeros(0), np.zeros(0)
    o = np.argsort(starts, kind="stable")
    s, e = starts[o], ends[o]
    out_s, out_e = [s[0]], [e[0]]
    for a, b in zip(s[1:], e[1:]):
        if a <= out_e[-1]:
            out_e[-1] = max(out_e[-1], b)
        else:
            out_s.append(a)
            out_e.append(b)
    return np.asarray(out_s, dtype=float), np.asarray(out_e, dtype=float)


def interval_distance(a: np.ndarray, b: np.ndarray, ms: np.ndarray, me: np.ndarray) -> np.ndarray:
    """Distance of intervals [a_i, b_i] to a union of disjoint sorted intervals [ms_k, me_k]."""
    if len(ms) == 0:
        return np.full(len(a), np.inf)
    idx = np.searchsorted(ms, b, side="right")           # intervals with start ≤ b_i
    out = np.full(len(a), np.inf)
    left = idx - 1
    has_left = left >= 0
    le = np.where(has_left, me[np.clip(left, 0, None)], -np.inf)
    overlap = has_left & (le >= a)
    out[overlap] = 0.0
    dl = np.where(has_left & ~overlap, a - le, np.inf)
    has_right = idx < len(ms)
    dr = np.where(has_right, ms[np.clip(idx, 0, len(ms) - 1)] - b, np.inf)
    rest = ~overlap
    out[rest] = np.minimum(dl[rest], dr[rest])
    return out


# --------------------------------------------------------------------------- test/fixture helper
_REC_DEFAULTS = {"te": None, "ts_res": 1e-6, "sport": 40000, "dport": 80, "proto": "tcp", "service": "",
                 "obytes": 100, "rbytes": 100, "opkts": 3, "rpkts": 3, "conn_state": "SF", "history": "ShADadFf",
                 "bulk": False, "n_merged": 1, "dns_q": None, "dns_rcode": None, "http_method": None,
                 "http_uri": None, "http_host": None, "http_ua": None, "http_status": None,
                 "http_resp_mime": None, "tls_sni": None, "src_file": "", "src_sha256": "", "src_locator": ""}


def records_frame(rows: list[dict], start_rec_id: int = 0) -> pl.DataFrame:
    """Build a RECORD_SCHEMA frame from partial dict rows (ts, src, dst required).
    Missing ``te`` defaults to ts + 1 s; missing ``rec_id`` is assigned sequentially."""
    out = []
    for i, r in enumerate(rows):
        d = dict(_REC_DEFAULTS)
        d.update(r)
        d.setdefault("rec_id", start_rec_id + i)
        if d.get("te") is None:
            d["te"] = d["ts"] + 1.0
        out.append(d)
    return pl.DataFrame(out, schema=RECORD_SCHEMA) if out else pl.DataFrame(schema=RECORD_SCHEMA)
