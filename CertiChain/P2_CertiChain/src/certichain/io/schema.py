"""Canonical schemas shared by every CertiChain stage (DESIGN §3.1, §3.2, §3.3, §7.2).

Three tables, one per stage boundary:

1. RECORDS (ingest → units): one row per Zeek-style connection record. All sources
   (IoT-23 conn.log.labeled, TON processed CSV + normal Bro logs, Gotham per-packet CSV
   → polars flows, ChainForge-rewritten records) are mapped to RECORD_SCHEMA.
   Label columns are NEVER here: they live in a quarantined table keyed by rec_id
   (LABEL_SCHEMA), written by io/labels.py and read only by gt/*.
2. UNITS (Alg. 1, units/sessionize.py + scope.py) → UNIT_SCHEMA.
3. CANDIDATES (Alg. 2, candidates/generate.py) → CANDIDATE_SCHEMA; GT labels for pairs
   are a separate quarantined column table (PAIR_LABEL_SCHEMA).

Rules (enforced by tests/test_leakage.py):
* ``rec_id`` is unique within a dataset; it is never used for ordering or as a feature.
* Ordering ties (1-s timestamps) are broken by ``tiehash`` = first 8 bytes of
  BLAKE3(f"{src}|{sport}|{dst}|{dport}|{proto}") read as u64 little-endian.
* Features may only use FEATURE_SAFE columns; no label, id, rec_id, absolute time, raw IP.
"""
from __future__ import annotations

import polars as pl

# --------------------------------------------------------------------------- records
RECORD_SCHEMA: dict[str, pl.DataType] = {
    "rec_id": pl.Int64,
    "ts": pl.Float64,            # epoch seconds (CF-TON: rounded to 1 s)
    "te": pl.Float64,            # ts + duration
    "ts_res": pl.Float64,        # timestamp resolution in s (1.0 for CF-TON, 1e-6 for Zeek)
    "src": pl.Utf8,              # originator IP (IPv4 string)
    "sport": pl.Int32,
    "dst": pl.Utf8,              # responder IP
    "dport": pl.Int32,
    "proto": pl.Utf8,            # tcp | udp | icmp | gre | other
    "service": pl.Utf8,          # Zeek service / tshark top protocol / port-derived ("" if none)
    "obytes": pl.Int64,          # originator payload bytes (Zeek orig_bytes) or frame bytes (Gotham)
    "rbytes": pl.Int64,
    "opkts": pl.Int64,
    "rpkts": pl.Int64,
    "conn_state": pl.Utf8,       # Zeek conn_state or flag-derived approximation
    "history": pl.Utf8,          # Zeek history ("" if unknown)
    "bulk": pl.Boolean,          # True for 10-s bulk records emitted by the flood path
    "n_merged": pl.Int32,        # number of underlying flows in a bulk record (1 otherwise)
    "dns_q": pl.Utf8,
    "dns_rcode": pl.Int32,
    "http_method": pl.Utf8,
    "http_uri": pl.Utf8,
    "http_host": pl.Utf8,
    "http_ua": pl.Utf8,
    "http_status": pl.Int32,
    "http_resp_mime": pl.Utf8,
    "tls_sni": pl.Utf8,
    "src_file": pl.Utf8,         # evidence pointer: path relative to project root
    "src_sha256": pl.Utf8,       # sha256 of src_file
    "src_locator": pl.Utf8,      # "lines:a-b" (CSV/log) or "frames:a-b" (Gotham CSV row ↔ pcap frame)
}

LABEL_SCHEMA: dict[str, pl.DataType] = {
    "rec_id": pl.Int64,
    "label": pl.Utf8,            # dataset label as shipped (Malicious/Benign, normal/attack, ...)
    "detailed_label": pl.Utf8,   # IoT-23 detailed-label, TON `type`, Gotham attack label
}

# --------------------------------------------------------------------------- units
# svc_class: coarse service class used in Alg. 1 keys, e.g. telnet, ssh, http, https, dns,
#            mqtt, coap, rtsp, modbus, ntp, irc, report48101, icmp, other_tcp, other_udp, gre.
# dir_class: int->int | int->ext | ext->int | ext->ext (internal = RFC1918 / incident plan).
UNIT_SCHEMA: dict[str, pl.DataType] = {
    "unit_id": pl.Int64,                 # dense 0..n-1 within an incident (time order, tiehash)
    "ts": pl.Float64,                    # t^s
    "te": pl.Float64,                    # t^e
    "src": pl.Utf8,
    "dsts": pl.List(pl.Utf8),            # ≤ 256 destination IPs (fan-out units)
    "dst_prefixes": pl.List(pl.Utf8),    # /24 prefixes of dsts
    "n_dst": pl.Int32,
    "fanout": pl.Boolean,
    "svc_class": pl.Utf8,
    "dir_class": pl.Utf8,
    "proto": pl.Utf8,
    "dports": pl.List(pl.Int32),         # distinct destination ports (≤ 64, most frequent first)
    "n_dports": pl.Int32,
    "type": pl.Utf8,                     # label-free unit type (DESIGN Appendix C)
    "n_records": pl.Int32,
    "obytes": pl.Int64,
    "rbytes": pl.Int64,
    "opkts": pl.Int64,
    "rpkts": pl.Int64,
    "fail_frac": pl.Float64,             # share of records with S0/REJ/RSTO*/no data
    "dur": pl.Float64,                   # te - ts
    "start_cv": pl.Float64,              # CV of inter-start gaps (NaN if < 3 records)
    "mean_bytes": pl.Float64,
    "content_flags": pl.Int32,           # bitmask: 1 exec-download, 2 injection regex, 4 tool UA,
                                         #          8 DNS NXDOMAIN-heavy, 16 high-entropy qname
    "dns_names": pl.List(pl.Utf8),       # for SHARED_EXT (may be empty)
    "http_hosts": pl.List(pl.Utf8),
    "tls_snis": pl.List(pl.Utf8),
    "infra": pl.List(pl.Utf8),           # collapsed INFRA hosts touched by the unit
    "rec_ids": pl.List(pl.Int64),        # evidence pointers (records of the unit)
}

# --------------------------------------------------------------------------- candidates
REL_PIVOT, REL_REVERSE, REL_SAME_ACTOR, REL_SAME_VICTIM, REL_SHARED_EXT = 1, 2, 4, 8, 16
REL_NAMES = {REL_PIVOT: "PIVOT", REL_REVERSE: "REVERSE", REL_SAME_ACTOR: "SAME_ACTOR",
             REL_SAME_VICTIM: "SAME_VICTIM", REL_SHARED_EXT: "SHARED_EXT"}

CANDIDATE_SCHEMA: dict[str, pl.DataType] = {
    "cand_id": pl.Int64,                 # dense within an incident
    "u": pl.Int64,                       # earlier unit_id (t^s_u < t^s_v)
    "v": pl.Int64,
    "rel_mask": pl.Int32,                # OR of REL_* bits
    "lag_start": pl.Float64,             # t^s_v - t^s_u  (> 0)
    "lag_gap": pl.Float64,               # t^s_v - t^e_u  (may be < 0 if overlapping)
    "overlap": pl.Boolean,               # t^s_v <= t^e_u
    "shared_entity_kind": pl.Utf8,       # internal | external | infra | none
    "shared_entity_deg": pl.Int32,       # degree of the shared entity in the incident
    "shared_is_seed": pl.Boolean,
}

PAIR_LABEL_SCHEMA: dict[str, pl.DataType] = {
    "cand_id": pl.Int64,
    "y": pl.Int8,                        # 1 true F_link, 0 null, -1 AMB (DESIGN §2.3)
    "bridge": pl.Boolean,                # true link through hidden intermediates only
}

# Columns a feature builder may read (anything else is forbidden).
FEATURE_SAFE_UNIT_COLS = [c for c in UNIT_SCHEMA if c not in ("unit_id", "rec_ids", "src", "dsts",
                                                              "dst_prefixes", "ts", "te", "infra",
                                                              "dns_names", "http_hosts", "tls_snis")]
FORBIDDEN_FEATURE_SUBSTRINGS = ("label", "rec_id", "episode", "campaign", "template", "dataset",
                                "incident_id", "position", "since_t0", "ts_abs")


def empty(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def conform(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Add missing columns as nulls, cast, and order columns as in ``schema``."""
    exprs = []
    for name, dtype in schema.items():
        if name in df.columns:
            exprs.append(pl.col(name).cast(dtype))
        else:
            exprs.append(pl.lit(None, dtype=dtype).alias(name))
    return df.select(exprs)
