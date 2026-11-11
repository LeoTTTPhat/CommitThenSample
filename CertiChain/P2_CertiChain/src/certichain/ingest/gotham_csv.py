"""Gotham 2025 per-packet tshark CSVs -> bidirectional flows (RECORD_SCHEMA) + quarantined labels.

DESIGN §3.1.1 (replaces the dpkt extractor). Input ``data/gotham2025/processed/iotsim-*.csv``:
22 tshark fields + ``label`` per packet (column order ``ip.dst, ip.src`` -- columns are read by
name). Per device file:

1. Keep only packets with ip.src or ip.dst == the device IP (MANIFEST caveat 3: the four
   ``-1`` files are supersets that also hold the ``-10..-15`` devices' rows; applied to every
   file, a no-op elsewhere). ``frame.time`` is parsed with a fixed-format regex (GMT) into
   integer ns. Ports are taken from tcp.* for TCP and udp.* for UDP only (ICMP errors and GRE
   carry inner-packet ports, ignored); GRE/ICMP flows have no ports.
2. Sessions = the capture sessions of the MANIFEST (raw pcap spans, +-1 s). Flows never cross
   a session.
3. Bidirectional 5-tuple flows: canonical key = sorted endpoints; orientation = first
   packet's direction. A flow ends at an idle gap > 60 s, at a pure SYN after a FIN/RST on the
   same 5-tuple, or when a packet arrives > 600 s after the flow start (active timeout,
   exact sequential semantics via searchsorted).
4. Aggregates: first/last packet ts, packets and frame bytes (frame.len) per direction,
   conn_state approximation from per-direction TCP flags (SYN, SYN-ACK, FIN, RST, data), a
   Zeek-like history (S H A D F R, upper = originator), service = first specific application
   layer after tcp/udp in frame.protocols, else a well-known-port name, else "data" if payload
   was seen, else "".
5. Flood control: a (session, src, dst, proto) group with > 200 k flows is replaced by bulk
   records, one per 10-s bin (bulk=True, n_merged = #flows, ports NULL).
6. Labels: flow (or bulk record) label = majority packet label (ties -> lexicographically
   smallest); LABEL_SCHEMA label = Benign / Unknown / Malicious, detailed_label = the Gotham
   label as shipped (e.g. "Telnet Brute Force", "Unknown" for the masscan rows of caveat 2).
7. Evidence pointer: ``src_locator = "frames:a-b"`` = 1-based data-row numbers (physical line - 1)
   of the flow's first and last packet in the CSV; the flow's packets are the rows in that range
   matching its 5-tuple (frame.number is not in the CSV, so pcap frame numbers are unavailable).
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

GOTHAM_FIELDS = ["frame.time", "frame.len", "frame.protocols", "eth.src", "eth.dst", "ip.dst", "ip.src", "ip.flags",
                 "ip.ttl", "ip.proto", "ip.checksum", "ip.tos", "tcp.srcport", "tcp.dstport", "tcp.flags",
                 "tcp.window_size_value", "tcp.window_size_scalefactor", "tcp.checksum", "tcp.options", "tcp.pdu.size",
                 "udp.srcport", "udp.dstport", "label"]
TIME_RE = (r"^(?P<mon>[A-Z][a-z]{2}) (?P<day>\d{1,2}), (?P<year>\d{4}) (?P<H>\d{2}):(?P<M>\d{2}):(?P<S>\d{2})"
           r"\.(?P<ns>\d{9}) GMT$")
MONTHS = {m: i + 1 for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
PROTO_NAMES = {6: "tcp", 17: "udp", 1: "icmp", 47: "gre"}
PROTO_CODE = {"tcp": 6, "udp": 17, "icmp": 1, "gre": 47}
APP_RE = r"^eth:ethertype:ip:(?:tcp|udp):([^:]+)"
WELL_KNOWN = {20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet", 2323: "telnet", 25: "smtp", 53: "dns", 67: "dhcp",
              68: "dhcp", 69: "tftp", 80: "http", 8080: "http", 123: "ntp", 161: "snmp", 162: "snmp", 443: "tls",
              554: "rtsp", 1883: "mqtt", 8883: "mqtt-tls", 5683: "coap", 5684: "coaps", 502: "modbus", 1900: "ssdp",
              5353: "mdns", 6667: "irc"}
REC_ID_STRIDE = 1_000_000_000
NS = 1_000_000_000

# Capture sessions (MANIFEST "Raw pcaps" table: first/last packet per scenario folder, UTC) and the
# labelled attack windows (MANIFEST "Attack schedule"). Schedule metadata, not per-record labels.
SESSIONS = [
    {"name": "benign", "capture": ["2025-01-14 18:39:53", "2025-01-14 20:47:32"],
     "attack_window": None, "note": "benign capture of all 78 devices"},
    {"name": "network-scanning", "capture": ["2025-01-14 23:36:11", "2025-01-14 23:52:50"],
     "attack_window": ["2025-01-14 23:39:03", "2025-01-14 23:48:13"], "note": "masscan from 192.168.35.10; rows labelled Unknown (caveat 2)"},
    {"name": "coap-amplificator", "capture": ["2025-01-15 10:26:18", "2025-01-15 11:35:20"],
     "attack_window": ["2025-01-15 10:35:42", "2025-01-15 11:20:39"], "note": "UDP scan 10:35:42-11:14:49; amplification 11:20:24-11:20:39"},
    {"name": "merlin", "capture": ["2025-01-17 15:46:38", "2025-01-17 17:17:14"],
     "attack_window": ["2025-01-17 16:06:55", "2025-01-17 17:16:56"], "note": "Merlin C2 192.168.34.10; agents .20.10, .17.15, .17.10"},
    {"name": "mirai-infection", "capture": ["2025-01-18 20:23:19", "2025-01-18 22:58:17"],
     "attack_window": ["2025-01-18 21:29:24", "2025-01-18 21:58:23"], "note": "Mirai lifecycle; infra 192.168.33.10-13, scanner 192.168.0.100"},
    {"name": "mirai-dos", "capture": ["2025-01-25 15:53:02", "2025-01-25 19:46:06"],
     "attack_window": ["2025-01-25 16:13:30", "2025-01-25 19:26:01"], "note": "bots .17.10, .20.10, .17.15 flood .18.10, .18.18, .18.15"},
]
INFRA = {"192.168.0.100": "Mirai initial scanner / telnet brute-forcer", "192.168.33.10": "Mirai C&C (TCP/23)",
         "192.168.33.11": "Mirai report server (TCP/48101)", "192.168.33.12": "Mirai loader (-> victim:23)",
         "192.168.33.13": "Mirai binary download server (HTTP/80)", "192.168.34.10": "Merlin C2 (HTTPS/443)",
         "192.168.35.10": "scanner (masscan TCP; nmap UDP)", "192.168.0.200": "spoofed source / reflection target (CoAP)"}
ATTACK_DEVICES = {"iotsim-air-quality-1": "192.168.18.17", "iotsim-building-monitor-1": "192.168.17.10",
                  "iotsim-city-power-1": "192.168.18.18", "iotsim-combined-cycle-1": "192.168.20.10",
                  "iotsim-combined-cycle-10": "192.168.20.19", "iotsim-domotic-monitor-1": "192.168.18.10",
                  "iotsim-ip-camera-museum-1": "192.168.17.15", "iotsim-ip-camera-street-1": "192.168.18.15"}


def _epoch_ns(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()) * NS


def session_bounds(pad_s: int = 1) -> list[tuple[str, int, int]]:
    """(name, start_ns, end_ns_exclusive) with +-pad_s around the capture spans (second-truncated)."""
    return [(s["name"], _epoch_ns(s["capture"][0]) - pad_s * NS, _epoch_ns(s["capture"][1]) + (1 + pad_s) * NS)
            for s in SESSIONS]


def device_ip_map(metadata_benign: str | os.PathLike) -> dict[str, str]:
    """``iotsim-<family>-<n>`` -> n-th IP (1-based) of the family's device_ip list."""
    meta = json.loads(Path(metadata_benign).read_text())
    out = {}
    for fam, ent in meta.items():
        for i, ip in enumerate(ent["device_ip"], 1):
            out[f"{fam}-{i}"] = ip
    return out


def device_name(path: str | os.PathLike) -> str:
    return Path(path).stem


# --------------------------------------------------------------------------- parsing
def _ip_u32(col: str) -> pl.Expr:
    s = pl.col(col).str.split_exact(".", 3)
    return (s.struct.field("field_0").cast(pl.Int64) * 16777216 + s.struct.field("field_1").cast(pl.Int64) * 65536
            + s.struct.field("field_2").cast(pl.Int64) * 256 + s.struct.field("field_3").cast(pl.Int64)).cast(pl.UInt32)


def u32_to_ip(col: str) -> pl.Expr:
    x = pl.col(col).cast(pl.Int64)
    return pl.concat_str([(x // 16777216).cast(pl.Utf8), ((x // 65536) % 256).cast(pl.Utf8),
                          ((x // 256) % 256).cast(pl.Utf8), (x % 256).cast(pl.Utf8)], separator=".")


def ip_to_u32(ip: str) -> int:
    a, b, c, d = (int(x) for x in ip.split("."))
    return (a << 24) | (b << 16) | (c << 8) | d


def scan_packets(path: str | os.PathLike) -> pl.LazyFrame:
    return pl.scan_csv(path, schema={c: pl.Utf8 for c in GOTHAM_FIELDS}, has_header=True, row_index_name="_row",
                       raise_if_empty=False)


def parse_packets(path: str | os.PathLike, device_ip: str) -> tuple[pl.DataFrame, dict]:
    """Compact per-packet frame of the device's own packets (+ stats)."""
    lf = scan_packets(path)
    g = pl.col("frame.time").str.extract_groups(TIME_RE)
    ts_ns = (pl.datetime(g.struct.field("year").cast(pl.Int32), g.struct.field("mon").replace_strict(MONTHS, return_dtype=pl.Int8),
                         g.struct.field("day").cast(pl.Int8), g.struct.field("H").cast(pl.Int8),
                         g.struct.field("M").cast(pl.Int8), g.struct.field("S").cast(pl.Int8), time_unit="ns")
             .dt.epoch("ns") + g.struct.field("ns").cast(pl.Int64))
    proto = pl.col("ip.proto").cast(pl.Int16, strict=False)
    is_tcp, is_udp = proto == 6, proto == 17
    port = lambda t, u: (pl.when(is_tcp).then(pl.col(t)).when(is_udp).then(pl.col(u)).otherwise(None)
                         .cast(pl.Float64, strict=False).cast(pl.Int32, strict=False))
    own = (pl.col("ip.src") == device_ip) | (pl.col("ip.dst") == device_ip)
    stats = lf.select(pl.len().alias("n_rows"), own.sum().alias("n_own"),
                      pl.col("frame.time").str.contains(TIME_RE).not_().sum().alias("n_bad_time")
                      ).collect(engine="streaming").to_dicts()[0]
    app = pl.col("frame.protocols").str.extract(APP_RE, 1)
    pk = lf.filter(own).select(
        pl.col("_row").cast(pl.UInt32),
        ts_ns.alias("ts_ns"),
        pl.col("frame.len").cast(pl.Int32, strict=False).fill_null(0).alias("len"),
        _ip_u32("ip.src").alias("src"), _ip_u32("ip.dst").alias("dst"),
        proto.fill_null(0).cast(pl.UInt8).alias("proto"),
        port("tcp.srcport", "udp.srcport").alias("sport"), port("tcp.dstport", "udp.dstport").alias("dport"),
        pl.when(is_tcp).then(pl.col("tcp.flags").str.strip_prefix("0x").str.to_integer(base=16, strict=False))
        .otherwise(None).cast(pl.Int16).alias("flags"),
        pl.when(is_tcp | is_udp).then(pl.when(app.str.starts_with("_ws")).then(pl.lit("data")).otherwise(app))
        .otherwise(None).cast(pl.Categorical).alias("app"),
        pl.col("label").cast(pl.Categorical).alias("label"),
    ).collect(engine="streaming")
    stats["n_null_ts"] = int(pk["ts_ns"].null_count())
    return pk, stats


# --------------------------------------------------------------------------- flows
def assign_sessions(pk: pl.DataFrame, bounds=None) -> pl.DataFrame:
    bounds = bounds or session_bounds()
    expr = pl.lit(-1, dtype=pl.Int8)
    for i, (_, a, b) in reversed(list(enumerate(bounds))):
        expr = pl.when((pl.col("ts_ns") >= a) & (pl.col("ts_ns") < b)).then(pl.lit(i, dtype=pl.Int8)).otherwise(expr)
    return pk.with_columns(expr.alias("session"))


def build_flows(pk: pl.DataFrame, idle_s: float = 60.0, active_s: float = 600.0) -> pl.DataFrame:
    """Packets (one session or more; ``session`` column) -> packets with ``flow`` id and per-packet
    direction ``is_orig``; sorted by (session, canonical key, ts, _row)."""
    if pk.height == 0:
        return pk.with_columns(pl.lit(0, dtype=pl.Int64).alias("flow"), pl.lit(True).alias("is_orig"))
    sp, dp = pl.col("sport").fill_null(-1), pl.col("dport").fill_null(-1)
    a_first = (pl.col("src") < pl.col("dst")) | ((pl.col("src") == pl.col("dst")) & (sp <= dp))
    pk = pk.with_columns(
        pl.when(a_first).then(pl.col("src")).otherwise(pl.col("dst")).alias("_kip0"),
        pl.when(a_first).then(sp).otherwise(dp).alias("_kp0"),
        pl.when(a_first).then(pl.col("dst")).otherwise(pl.col("src")).alias("_kip1"),
        pl.when(a_first).then(dp).otherwise(sp).alias("_kp1"),
    ).sort(["session", "_kip0", "_kp0", "_kip1", "_kp1", "proto", "ts_ns", "_row"])
    keys = ["session", "_kip0", "_kp0", "_kip1", "_kp1", "proto"]
    same = pl.all_horizontal([pl.col(k) == pl.col(k).shift(1) for k in keys]).fill_null(False)
    fl = pl.col("flags").fill_null(0)
    finrst_prev = ((fl & 0x05) != 0).shift(1).fill_null(False)
    pure_syn = (fl & 0x12) == 0x02
    gap = pl.col("ts_ns") - pl.col("ts_ns").shift(1)
    idle_ns, active_ns = int(idle_s * NS), int(active_s * NS)
    new_seg = (~same) | (gap > idle_ns).fill_null(True) | (pure_syn & finrst_prev & (pl.col("proto") == 6))
    ns = pk.select(new_seg.alias("n"))["n"].to_numpy().copy()
    ns[0] = True
    ts = pk["ts_ns"].to_numpy()
    starts = np.flatnonzero(ns)
    ends = np.r_[starts[1:], ts.size]
    long = np.flatnonzero(ts[ends - 1] - ts[starts] > active_ns)
    for s in long:  # exact sequential active-timeout splitting inside long idle-segments
        i0, i1 = starts[s], ends[s]
        t = ts[i0:i1]
        j = 0
        while True:
            k = int(np.searchsorted(t, t[j] + active_ns, side="right"))
            if k >= t.size:
                break
            ns[i0 + k] = True
            j = k
    flow = np.cumsum(ns) - 1
    pk = pk.with_columns(pl.Series("flow", flow, dtype=pl.Int64)).drop(["_kip0", "_kp0", "_kip1", "_kp1"])
    first = pl.col("flow") != pl.col("flow").shift(1).fill_null(-1)
    pk = pk.with_columns(
        pl.when(first).then(pl.col("src")).otherwise(None).forward_fill().alias("_fsrc"),
        pl.when(first).then(sp).otherwise(None).forward_fill().alias("_fsp"),
    )
    is_orig = (pl.col("src") == pl.col("_fsrc")) & (sp == pl.col("_fsp"))
    return pk.with_columns(is_orig.alias("is_orig")).drop(["_fsrc", "_fsp"])


_LETTERS = [("S", True, "syn"), ("H", True, "synack"), ("A", True, "pack"), ("D", True, "data"), ("F", True, "fin"),
            ("R", True, "rst"), ("s", False, "syn"), ("h", False, "synack"), ("a", False, "pack"), ("d", False, "data"),
            ("f", False, "fin"), ("r", False, "rst")]
_LETTER_RANK = {L: i for i, L in enumerate("SHADFRshadfr")}


def flow_table(pk: pl.DataFrame) -> pl.DataFrame:
    """Aggregate flow-annotated packets into one row per flow (no labels)."""
    fl = pl.col("flags").fill_null(0)
    o = pl.col("is_orig")
    cond = {
        "syn": ((fl & 0x02) != 0) & ((fl & 0x10) == 0),
        "synack": ((fl & 0x12) == 0x12),
        "fin": (fl & 0x01) != 0,
        "rst": (fl & 0x04) != 0,
        "data": pl.col("app").is_not_null(),
        "pack": ((fl & 0x17) == 0x10) & pl.col("app").is_null() & (pl.col("proto") == 6),
    }
    pk = pk.with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("_i"), *[c.alias(f"_{k}") for k, c in cond.items()])
    tcp_or_udp = pl.col("proto").is_in([6, 17])
    aggs = [
        pl.col("session").first(), pl.col("proto").first(),
        pl.col("src").first(), pl.col("sport").first(), pl.col("dst").first(), pl.col("dport").first(),
        pl.col("ts_ns").first().alias("t0"), pl.col("ts_ns").last().alias("t1"),
        o.sum().cast(pl.Int64).alias("opkts"), (~o).sum().cast(pl.Int64).alias("rpkts"),
        pl.col("len").filter(o).sum().cast(pl.Int64).alias("obytes"), pl.col("len").filter(~o).sum().cast(pl.Int64).alias("rbytes"),
        pl.col("_row").min().alias("row_min"), pl.col("_row").max().alias("row_max"),
        pl.col("app").filter(pl.col("app") != "data").first().cast(pl.Utf8).alias("app_first"),
        (pl.col("app") == "data").any().alias("any_data"),
    ]
    for k in ("syn", "synack", "fin", "rst", "data"):
        aggs += [(pl.col(f"_{k}") & o).any().alias(f"o_{k}"), (pl.col(f"_{k}") & ~o).any().alias(f"r_{k}")]
    for L, orig, k in _LETTERS:
        d = o if orig else ~o
        aggs.append(pl.col("_i").filter(pl.col(f"_{k}") & d & tcp_or_udp).first().alias(f"h_{L}"))
    ft = pk.group_by("flow", maintain_order=True).agg(aggs)
    return ft


def history_strings(ft: pl.DataFrame) -> pl.Series:
    """Zeek-like history from first-occurrence positions (ties broken by S H A D F R order)."""
    hcols = [f"h_{L}" for L, _, _ in _LETTERS]
    long = (ft.select("flow", *hcols).unpivot(index="flow", on=hcols, variable_name="L", value_name="pos")
            .drop_nulls("pos").with_columns(pl.col("L").str.slice(2)))
    long = long.with_columns(pl.col("L").replace_strict(_LETTER_RANK, return_dtype=pl.Int8).alias("rank"))
    h = long.sort(["flow", "pos", "rank"]).group_by("flow", maintain_order=True).agg(pl.col("L").str.join("").alias("history"))
    return ft.select("flow").join(h, on="flow", how="left", maintain_order="left")["history"].fill_null("")


def conn_state_expr() -> pl.Expr:
    tcp = pl.col("proto") == 6
    has_r = pl.col("rpkts") > 0
    est = pl.col("o_syn") & pl.col("r_synack")
    tcp_state = (
        pl.when(pl.col("o_syn") & ~has_r).then(pl.when(pl.col("o_rst")).then(pl.lit("RSTOS0"))
                                               .when(pl.col("o_fin")).then(pl.lit("SH")).otherwise(pl.lit("S0")))
        .when(est).then(pl.when(pl.col("o_rst")).then(pl.lit("RSTO")).when(pl.col("r_rst")).then(pl.lit("RSTR"))
                        .when(pl.col("o_fin") & pl.col("r_fin")).then(pl.lit("SF")).when(pl.col("o_fin")).then(pl.lit("S2"))
                        .when(pl.col("r_fin")).then(pl.lit("S3")).otherwise(pl.lit("S1")))
        .when(pl.col("o_syn") & pl.col("r_rst")).then(pl.lit("REJ"))
        .when(~pl.col("o_syn") & pl.col("o_synack") & pl.col("o_rst")).then(pl.lit("RSTRH"))
        .when(~pl.col("o_syn") & pl.col("o_synack") & pl.col("o_fin")).then(pl.lit("SHR"))
        .otherwise(pl.lit("OTH")))
    return (pl.when(tcp).then(tcp_state)
            .when(pl.col("proto") == 17).then(pl.when(has_r).then(pl.lit("SF")).otherwise(pl.lit("S0")))
            .otherwise(pl.lit("OTH")))


def service_expr() -> pl.Expr:
    wk = pl.col("dport").replace_strict(WELL_KNOWN, default=None, return_dtype=pl.Utf8)
    wk_s = pl.col("sport").replace_strict(WELL_KNOWN, default=None, return_dtype=pl.Utf8)
    return (pl.when(~pl.col("proto").is_in([6, 17])).then(pl.lit(""))
            .otherwise(pl.coalesce(pl.col("app_first"), wk, wk_s,
                                   pl.when(pl.col("any_data")).then(pl.lit("data")).otherwise(pl.lit("")))))


def flows_from_packets(pk: pl.DataFrame, idle_s=60.0, active_s=600.0, flood_threshold=200_000, bulk_bin_s=10.0
                       ) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Packets -> (flow records with internal columns, per-record label counts, stats).

    Record frame columns: rid (dense record index), session, proto, src, sport, dst, dport (u32 IPs),
    t0, t1 (ns), opkts, rpkts, obytes, rbytes, conn_state, history, service, bulk, n_merged,
    row_min, row_max. Label counts: (rid, label, n).
    """
    pkf = build_flows(pk, idle_s, active_s)
    ft = flow_table(pkf)
    ft = ft.with_columns(conn_state_expr().alias("conn_state"), service_expr().alias("service"))
    # flood control
    grp = ["session", "src", "dst", "proto"]
    ft = ft.with_columns(pl.len().over(grp).alias("_gn"))
    is_bulk = pl.col("_gn") > flood_threshold
    lab = pkf.group_by(["flow", "label"]).agg(pl.len().cast(pl.Int64).alias("n"))
    normal = ft.filter(~is_bulk)
    normal = normal.with_columns(history_strings(normal).alias("history"), pl.lit(False).alias("bulk"),
                                 pl.lit(1, dtype=pl.Int32).alias("n_merged"))
    bulk_src = ft.filter(is_bulk).with_columns((pl.col("t0") // int(bulk_bin_s * NS)).alias("_bin"))
    stats = {"n_packets": pk.height, "n_flows": ft.height, "n_bulk_groups": 0, "n_bulk_flows": bulk_src.height,
             "n_bulk_records": 0}
    cols = ["session", "proto", "src", "sport", "dst", "dport", "t0", "t1", "opkts", "rpkts", "obytes", "rbytes",
            "conn_state", "history", "service", "bulk", "n_merged", "row_min", "row_max"]
    parts = [normal.select("flow", *cols).with_columns(pl.col("flow").alias("_key"))]
    lab_parts = [lab.join(normal.select("flow"), on="flow", how="semi").rename({"flow": "_key"})]
    if bulk_src.height:
        bk = ["session", "src", "dst", "proto", "_bin"]
        mode = lambda c: pl.col(c).mode().sort().first()
        b = bulk_src.group_by(bk, maintain_order=True).agg(
            pl.col("t0").min(), pl.col("t1").max(), pl.col("opkts").sum(), pl.col("rpkts").sum(),
            pl.col("obytes").sum(), pl.col("rbytes").sum(), mode("conn_state").alias("conn_state"),
            mode("service").alias("service"), pl.len().cast(pl.Int32).alias("n_merged"),
            pl.col("row_min").min(), pl.col("row_max").max(), pl.col("flow").min().alias("_key"))
        b = b.with_columns(pl.lit(None, dtype=pl.Int32).alias("sport"), pl.lit(None, dtype=pl.Int32).alias("dport"),
                           pl.lit("").alias("history"), pl.lit(True).alias("bulk"))
        fmap = bulk_src.select("flow", *bk).join(b.select(*bk, "_key"), on=bk, how="left")
        bl = lab.join(fmap.select("flow", "_key"), on="flow", how="inner").group_by(["_key", "label"]).agg(pl.col("n").sum())
        parts.append(b.select(pl.col("_key").alias("flow"), *cols, "_key"))
        lab_parts.append(bl)
        stats["n_bulk_groups"] = int(bulk_src.select(grp).unique().height)
        stats["n_bulk_records"] = b.height
    rec = pl.concat([p.select("_key", *cols) for p in parts], how="vertical")
    labs = pl.concat([lp.select("_key", pl.col("label").cast(pl.Utf8), "n") for lp in lab_parts], how="vertical")
    return rec, labs, stats


def majority_labels(labs: pl.DataFrame) -> pl.DataFrame:
    """(_key, label, n) -> (_key, detailed_label, purity)."""
    tot = labs.group_by("_key").agg(pl.col("n").sum().alias("_tot"), pl.col("label").n_unique().alias("n_kinds"))
    m = (labs.sort(["_key", "n", "label"], descending=[False, True, False]).group_by("_key", maintain_order=True)
         .first().join(tot, on="_key"))
    return m.select("_key", pl.col("label").alias("detailed_label"), (pl.col("n") / pl.col("_tot")).alias("purity"), "n_kinds")


def to_record_schema(rec: pl.DataFrame, dev_code: int, src_file: str, src_sha256: str) -> pl.DataFrame:
    """Internal flow frame -> RECORD_SCHEMA columns (+ _key) sorted by (t0, row_min); rec_id dense."""
    rec = rec.sort(["t0", "row_min", "_key"])
    return rec.select(
        (pl.lit(dev_code, dtype=pl.Int64) * REC_ID_STRIDE + pl.int_range(pl.len(), dtype=pl.Int64)).alias("rec_id"),
        pl.col("_key"),
        (pl.col("t0") / NS).alias("ts"), (pl.col("t1") / NS).alias("te"), pl.lit(1e-6).alias("ts_res"),
        u32_to_ip("src").alias("src"), pl.col("sport").cast(pl.Int32), u32_to_ip("dst").alias("dst"),
        pl.col("dport").cast(pl.Int32),
        pl.col("proto").cast(pl.Int64).replace_strict({k: v for k, v in PROTO_NAMES.items()}, default="other",
                                                      return_dtype=pl.Utf8).alias("proto"),
        pl.col("service"), pl.col("obytes"), pl.col("rbytes"), pl.col("opkts"), pl.col("rpkts"),
        pl.col("conn_state"), pl.col("history"), pl.col("bulk"), pl.col("n_merged"),
        pl.lit(src_file).alias("src_file"), pl.lit(src_sha256).alias("src_sha256"),
        pl.format("frames:{}-{}", pl.col("row_min").cast(pl.Int64) + 1, pl.col("row_max").cast(pl.Int64) + 1).alias("src_locator"),
        pl.col("session").alias("_session"),
    )


def binary_label(detailed: pl.Expr) -> pl.Expr:
    return (pl.when(detailed == "Benign").then(pl.lit("Benign")).when(detailed == "Unknown").then(pl.lit("Unknown"))
            .otherwise(pl.lit("Malicious")))


def sessions_json() -> dict:
    return {"source": "data/gotham2025/MANIFEST.md (Raw pcaps table: capture spans; Attack schedule: attack windows)",
            "time_zone": "UTC (GMT)", "sessions": SESSIONS, "pad_s": 1,
            "bounds_ns": [{"name": n, "start_ns": a, "end_ns_excl": b} for n, a, b in session_bounds()],
            "attack_devices": ATTACK_DEVICES, "infrastructure": INFRA,
            "design_case_study_windows": {
                "source": "DESIGN §5.1.3 (P2-B owns incident construction)",
                "mirai_infection_caltest": {"victims": ["192.168.20.19", "192.168.18.10", "192.168.17.15", "192.168.18.17", "192.168.18.18"],
                                            "window": ["2025-01-18 21:25:00", "2025-01-18 22:00:00"]},
                "mirai_dos_17_15": {"window": ["2025-01-25 18:45:00", "2025-01-25 19:30:00"]},
                "merlin_17_15": {"window": ["2025-01-17 16:05:00", "2025-01-17 17:17:00"]},
            }}


_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
