"""IoT-23 unit ↔ Zeek conn label match (DESIGN §5.5).

Rule: a unit matches the conn with the same (orig_h, orig_p, resp_h, resp_p, proto) whose
interval [ts − 1 s, ts + duration + 1 s] overlaps the unit's [first_ts, last_ts]; ties go to
the smallest |unit.first_ts − conn.ts|. The unit takes ``label`` and ``detailed-label``;
unmatched units are ``unknown``.

Implementation (exact for conns of one 5-tuple that do not overlap each other, which is how
Zeek splits connections): for each unit the latest conn starting at or before the unit's
first packet and the earliest conn starting after it are found with two as-of joins on the
5-tuple; overlapping candidates are kept and the one with the smaller start difference wins.
Fallbacks, reported separately: the reversed orientation (Zeek may orient a connection
differently from the gateway's first-packet rule) and, for ICMP, a host-pair match (Zeek puts
the ICMP type/code in orig_p/resp_p and pairs echo request and reply).

Conn-log quirk: the last three fields (tunnel_parents, label, detailed-label) are separated by
spaces, not tabs. Labels are case-normalised (honeypot logs say "benign").
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

SLACK_S = 1.0
GAP_S = 600.0
CONN_COLS = ["ts", "uid", "orig_h", "orig_p", "resp_h", "resp_p", "proto", "service", "duration", "orig_bytes",
             "resp_bytes", "conn_state", "local_orig", "local_resp", "missed_bytes", "history", "orig_pkts",
             "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "tail"]
PROTO_NAME = {6: "tcp", 17: "udp", 1: "icmp", 58: "icmp"}


def _num(c: str, dt) -> pl.Expr:
    return pl.when(pl.col(c).str.contains(r"^[0-9.]+$")).then(pl.col(c)).otherwise(pl.lit("0")).cast(dt).alias(c)


def load_connlog(path: str | Path, t_lo: float | None = None, t_hi: float | None = None) -> pl.DataFrame:
    """Read a conn.log.labeled into a DataFrame (seconds as float64)."""
    lf = pl.scan_csv(str(path), separator="\t", has_header=False, comment_prefix="#", quote_char=None,
                     schema={c: pl.String for c in CONN_COLS}, truncate_ragged_lines=True)
    lf = lf.select(
        pl.col("ts").cast(pl.Float64), "uid", "orig_h", pl.col("orig_p").cast(pl.Int32, strict=False),
        "resp_h", pl.col("resp_p").cast(pl.Int32, strict=False), "proto",
        _num("duration", pl.Float64), _num("orig_bytes", pl.Int64), _num("resp_bytes", pl.Int64),
        "conn_state", "history",
        pl.col("tail").str.extract(r"^\s*\S+\s+(\S+)", 1).str.to_lowercase().alias("label"),
        pl.col("tail").str.extract(r"^\s*\S+\s+\S+\s+(\S+)", 1).alias("detailed_label"),
    )
    if t_lo is not None:
        lf = lf.filter(pl.col("ts") >= t_lo)
    if t_hi is not None:
        lf = lf.filter(pl.col("ts") <= t_hi)
    try:
        df = lf.collect(engine="streaming")
    except Exception:  # noqa: BLE001  (older polars without the streaming engine keyword)
        df = lf.collect()
    return df.with_row_index("cidx").with_columns(pl.col("cidx").cast(pl.Int64))


def _asof_candidates(ub: pl.DataFrame, cb: pl.DataFrame, by: list[str], tag: str) -> pl.DataFrame:
    out = []
    ub = ub.sort("first_s")
    cb = cb.sort("ts")
    for strat in ("backward", "forward"):
        j = ub.join_asof(cb, left_on="first_s", right_on="ts", by=by, strategy=strat)
        j = j.filter(pl.col("ts").is_not_null()
                     & (pl.col("ts") - SLACK_S <= pl.col("last_s"))
                     & (pl.col("ts") + pl.col("duration") + SLACK_S >= pl.col("first_s")))
        out.append(j.select("unit_gid", "cidx", (pl.col("first_s") - pl.col("ts")).abs().alias("dstart"),
                            pl.lit(tag).alias("match")))
    return pl.concat(out)


def match_units(units: pl.DataFrame, conns: pl.DataFrame) -> pl.DataFrame:
    """units: unit_gid, proto, ip_a, port_a, ip_b, port_b, first_ts_us, last_ts_us.
    Returns one row per matched unit: unit_gid, cidx, dstart, match ('fwd'|'rev'|'icmp')."""
    u = units.with_columns(
        pl.col("proto").replace_strict(PROTO_NAME, default=None, return_dtype=pl.String).alias("pname"),
        (pl.col("first_ts_us") / 1e6).alias("first_s"), (pl.col("last_ts_us") / 1e6).alias("last_s"),
        pl.col("port_a").cast(pl.Int32), pl.col("port_b").cast(pl.Int32))
    c = conns.select("cidx", "ts", "duration", pl.col("proto").alias("pname"), "orig_h", "orig_p", "resp_h", "resp_p")
    by = ["pname", "h1", "p1", "h2", "p2"]
    tu = u.filter(pl.col("pname").is_in(["tcp", "udp"]))
    tc = c.filter(pl.col("pname").is_in(["tcp", "udp"])).rename(
        {"orig_h": "h1", "orig_p": "p1", "resp_h": "h2", "resp_p": "p2"})
    res = []
    fwd = tu.select("unit_gid", "pname", "first_s", "last_s", pl.col("ip_a").alias("h1"), pl.col("port_a").alias("p1"),
                    pl.col("ip_b").alias("h2"), pl.col("port_b").alias("p2"))
    res.append(_asof_candidates(fwd, tc, by, "fwd"))
    rev = tu.select("unit_gid", "pname", "first_s", "last_s", pl.col("ip_b").alias("h1"), pl.col("port_b").alias("p1"),
                    pl.col("ip_a").alias("h2"), pl.col("port_a").alias("p2"))
    res.append(_asof_candidates(rev, tc, by, "rev"))
    # ICMP: host pair in either orientation
    iu = u.filter(pl.col("pname") == "icmp")
    if iu.height:
        ic = c.filter(pl.col("pname") == "icmp").rename({"orig_h": "h1", "resp_h": "h2"}).drop("orig_p", "resp_p")
        for a, b in (("ip_a", "ip_b"), ("ip_b", "ip_a")):
            x = iu.select("unit_gid", "pname", "first_s", "last_s", pl.col(a).alias("h1"), pl.col(b).alias("h2"))
            res.append(_asof_candidates(x, ic, ["pname", "h1", "h2"], "icmp"))
    cand = pl.concat(res)
    if cand.height == 0:
        return cand
    pri = pl.col("match").replace_strict({"fwd": 0, "rev": 1, "icmp": 2}, return_dtype=pl.Int8)
    return (cand.with_columns(pri.alias("pri")).sort(["unit_gid", "pri", "dstart", "cidx"])
            .unique(subset=["unit_gid"], keep="first", maintain_order=True).drop("pri"))


def l_family(dl: str | None) -> str | None:
    """Group a detailed label into the L-estimand families (C&C*, FileDownload*, Attack/DDoS, Okiru*)."""
    if dl is None or dl in ("-", "", "(empty)"):
        return None
    if dl.startswith("C&C"):
        return "C&C"
    if dl.startswith("FileDownload"):
        return "FileDownload"
    if dl.startswith("Attack") or dl.startswith("DDoS"):
        return "Attack"
    if dl.startswith("Okiru"):
        return "Okiru"
    if dl.startswith("PartOfAHorizontalPortScan"):
        return "PortScan"
    return dl


def label_source(units: pl.DataFrame, conns: pl.DataFrame | None, source: str, default_benign: bool = False
                 ) -> tuple[pl.DataFrame, dict]:
    """Label all units of one IoT-23 source. Returns (labels DataFrame, QA dict)."""
    base = units.select("unit_gid", "pay", "proto", "device_id", "first_ts_us")
    if conns is None or conns.height == 0:
        lab = base.with_columns(
            pl.lit("Benign" if default_benign else "unknown").alias("label"),
            pl.lit("-" if default_benign else None, dtype=pl.String).alias("detailed_label"),
            pl.lit("honeypot_default" if default_benign else "none").alias("label_source"),
            pl.lit(None, dtype=pl.String).alias("conn_uid"), pl.lit(None, dtype=pl.Float64).alias("match_dstart_s"))
    else:
        m = match_units(units, conns)
        cj = conns.select("cidx", pl.col("uid").alias("conn_uid"), "label", "detailed_label")
        lab = (base.join(m, on="unit_gid", how="left").join(cj, on="cidx", how="left")
               .with_columns(
                   pl.when(pl.col("label") == "malicious").then(pl.lit("Malicious"))
                   .when(pl.col("label") == "benign").then(pl.lit("Benign"))
                   .when(pl.col("label").is_null()).then(pl.lit("unknown"))
                   .otherwise(pl.col("label")).alias("label"),
                   pl.when(pl.col("match").is_null()).then(pl.lit("none"))
                   .otherwise(pl.lit("conn_") + pl.col("match")).alias("label_source"),
                   pl.col("dstart").alias("match_dstart_s"))
               .drop("cidx", "match", "dstart"))
    lab = lab.with_columns(
        pl.col("detailed_label").map_elements(l_family, return_dtype=pl.String).alias("family"))
    lab = lab.with_columns(pl.col("family").alias("class"),
                           (pl.col("label") == "Malicious").alias("is_malicious"))
    lab = add_incidents(lab, source)
    pay = lab.filter(pl.col("pay"))
    tcpudp = lab.filter(pl.col("proto").is_in([6, 17]))
    qa = {"units": lab.height, "payload_units": pay.height,
          "match_rate_all": float((lab["label"] != "unknown").mean()) if lab.height else None,
          "match_rate_payload": float((pay["label"] != "unknown").mean()) if pay.height else None,
          "match_rate_tcp_udp": float((tcpudp["label"] != "unknown").mean()) if tcpudp.height else None,
          "label_source_counts": dict(lab.group_by("label_source").len().iter_rows()),
          "label_source_counts_payload": dict(pay.group_by("label_source").len().iter_rows()),
          "label_counts": dict(lab.group_by("label").len().iter_rows()),
          "label_counts_payload": dict(pay.group_by("label").len().iter_rows()),
          "family_counts_mal_payload": dict(pay.filter(pl.col("is_malicious")).group_by("family").len().iter_rows()),
          "n_incidents": int(lab["incident_id"].n_unique() - (1 if lab["incident_id"].null_count() else 0)),
          "n_conns": None if conns is None else conns.height}
    return lab.drop("proto", "first_ts_us"), qa


def add_incidents(lab: pl.DataFrame, source: str, gap_s: float = GAP_S) -> pl.DataFrame:
    """Incident = (source, device, family) run of content-bearing (malicious payload) units, split at
    gaps > 600 s. Only content-bearing units get an incident id (string)."""
    cb = (lab.filter(pl.col("is_malicious") & pl.col("pay"))
          .select("unit_gid", "device_id", pl.col("family").fill_null("-"), "first_ts_us")
          .sort(["device_id", "family", "first_ts_us"]))
    if cb.height == 0:
        return lab.with_columns(pl.lit(None, dtype=pl.String).alias("incident_id"))
    cb = cb.with_columns(
        ((pl.col("first_ts_us").diff().over(["device_id", "family"]).fill_null(0) > gap_s * 1e6)
         .cast(pl.Int32).cum_sum().over(["device_id", "family"])).alias("k"))
    cb = cb.with_columns((pl.lit(source + "|") + pl.col("device_id").cast(pl.String) + "|" + pl.col("family")
                          + "|" + pl.col("k").cast(pl.String)).alias("incident_id"))
    return lab.join(cb.select("unit_gid", "incident_id"), on="unit_gid", how="left")
