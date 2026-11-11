"""TON_IoT -> records + quarantined labels, episodes and backgrounds for ChainForge (DESIGN §5.1.2).

Processed CSVs (``Processed_Network_dataset/Network_dataset_{1..23}.csv``): UTF-8 BOM on the
first header, ``-`` = missing, numeric DNS/HTTP fields use 0, file 6 has an extra ``uid``
column (dropped; columns are selected by name). ``ts`` is already rounded to the nearest
second -> ts_res = 1.0. ``label``/``type`` go to quarantine (label, detailed_label).

Backgrounds: Zeek logs ``normal_Bro/normal_IoT_1`` (-> train/dev pool) and
``normal_Bro/normal_IoT_2`` (-> cal/test pool); conn joined with dns/http/ssl on ``uid`` (first
entry per uid by ts), ts rounded to the nearest second like the processed rows.
**Field parity** (default ``parity=True``): the processed CSV has no Zeek ``history``, no
HTTP ``host`` and no TLS ``server_name``, so those fields are written as ""/NULL for the
backgrounds too -- otherwise field availability alone would separate background from attack
records in CF-TON (a label leak). The joined Bro values are computed but not written.

Episodes: attack rows (label 1, type in EPISODE_TYPES; mitm excluded) grouped by
(attacker in 192.168.1.30-39 as either endpoint, type) into bursts at gaps > 300 s; each burst
is cut into consecutive <= 600-s sub-windows from its first row; a sub-window with more than
5,000 rows keeps a uniform random sample of 5,000 (numpy default_rng(seed), in a fixed
order). Each sub-window is one library episode. Pools per type by start time: the first half
(n // 2 by t0 order) -> train, the rest -> caltest.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl

from certichain.io.labels import conform_lazy
from certichain.io.paths import atomic_sink_parquet, rel
from certichain.io.schema import LABEL_SCHEMA, RECORD_SCHEMA, conform

ATTACKERS = [f"192.168.1.{i}" for i in range(30, 40)]
EPISODE_TYPES = ["scanning", "password", "injection", "xss", "backdoor", "ransomware", "dos", "ddos"]
REC_ID_STRIDE = 1_000_000_000
BG_CODES = {"normal_IoT_1": 101, "normal_IoT_2": 102}
BG_POOL = {"normal_IoT_1": "train", "normal_IoT_2": "caltest"}
PROTOS = ("tcp", "udp", "icmp")


def processed_files(root: str | os.PathLike) -> list[Path]:
    fs = sorted(Path(root).glob("Network_dataset_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    return fs


def file_code(path: str | os.PathLike) -> int:
    return int(Path(path).stem.rsplit("_", 1)[1])


def scan_processed(path: str | os.PathLike) -> pl.LazyFrame:
    """All-Utf8 lazy scan with the BOM stripped from the header and ``uid`` dropped."""
    lf = pl.scan_csv(path, infer_schema=False, has_header=True, row_index_name="_row", raise_if_empty=False)
    names = lf.collect_schema().names()
    lf = lf.rename({c: c.lstrip("﻿") for c in names if c != c.lstrip("﻿")})
    if "uid" in lf.collect_schema().names():
        lf = lf.drop("uid")
    return lf


def _null_if(col: str, *vals: str) -> pl.Expr:
    c = pl.col(col)
    return pl.when(c.is_null() | c.is_in(list(vals))).then(None).otherwise(c)


def _int(col: str, dtype=pl.Int64) -> pl.Expr:
    return pl.col(col).cast(pl.Float64, strict=False).cast(dtype, strict=False)


def _ipv6() -> pl.Expr:
    return pl.col("src").str.contains(":", literal=True) | pl.col("dst").str.contains(":", literal=True)


def processed_to_records(lf: pl.LazyFrame, code: int, src_file: str, src_sha256: str) -> pl.LazyFrame:
    """Processed CSV -> RECORD_SCHEMA columns + (label, detailed_label) for the quarantine split."""
    line = pl.col("_row").cast(pl.Int64) + 2  # header is line 1
    ts = pl.col("ts").cast(pl.Float64, strict=False)
    dur = pl.col("duration").cast(pl.Float64, strict=False).fill_null(0.0)
    proto = pl.col("proto").str.to_lowercase()
    dns_q = _null_if("dns_query", "-", "(empty)", "")
    return lf.select(
        (pl.lit(code, dtype=pl.Int64) * REC_ID_STRIDE + line).alias("rec_id"),
        ts.alias("ts"), (ts + dur).alias("te"), pl.lit(1.0).alias("ts_res"),
        pl.col("src_ip").alias("src"), _int("src_port", pl.Int32).alias("sport"),
        pl.col("dst_ip").alias("dst"), _int("dst_port", pl.Int32).alias("dport"),
        pl.when(proto.is_in(list(PROTOS))).then(proto).otherwise(pl.lit("other")).alias("proto"),
        _null_if("service", "-").fill_null("").alias("service"),
        _int("src_bytes").fill_null(0).alias("obytes"), _int("dst_bytes").fill_null(0).alias("rbytes"),
        _int("src_pkts").fill_null(0).alias("opkts"), _int("dst_pkts").fill_null(0).alias("rpkts"),
        _null_if("conn_state", "-").fill_null("").alias("conn_state"),
        pl.lit("").alias("history"),
        pl.lit(False).alias("bulk"), pl.lit(1, dtype=pl.Int32).alias("n_merged"),
        dns_q.alias("dns_q"),
        pl.when(dns_q.is_not_null()).then(_int("dns_rcode", pl.Int32)).otherwise(None).alias("dns_rcode"),
        _null_if("http_method", "-").alias("http_method"),
        _null_if("http_uri", "-").alias("http_uri"),
        pl.lit(None, dtype=pl.Utf8).alias("http_host"),
        _null_if("http_user_agent", "-").alias("http_ua"),
        pl.when(_int("http_status_code", pl.Int32) > 0).then(_int("http_status_code", pl.Int32)).otherwise(None).alias("http_status"),
        _null_if("http_resp_mime_types", "-").alias("http_resp_mime"),
        pl.lit(None, dtype=pl.Utf8).alias("tls_sni"),
        pl.lit(src_file).alias("src_file"), pl.lit(src_sha256).alias("src_sha256"),
        pl.format("lines:{}-{}", line, line).alias("src_locator"),
        pl.col("label").alias("label"), pl.col("type").alias("detailed_label"),
    )


def ingest_processed_file(path: str | os.PathLike, sha: str, rec_out: Path, lab_out: Path) -> dict:
    """Stream one processed CSV to records/labels parquet; returns per-file stats."""
    code = file_code(path)
    lf = processed_to_records(scan_processed(path), code, rel(path), sha)
    stats = lf.select(
        pl.len().alias("n_rows"),
        _ipv6().sum().alias("n_ipv6"),
        pl.col("ts").is_null().sum().alias("n_null_ts"),
        pl.col("label").is_null().sum().alias("n_null_label"),
    ).collect(engine="streaming").to_dicts()[0]
    keep = lf.filter(~_ipv6())
    atomic_sink_parquet(conform_lazy(keep, RECORD_SCHEMA), rec_out)
    atomic_sink_parquet(conform_lazy(keep.select("rec_id", "label", "detailed_label"), LABEL_SCHEMA), lab_out)
    counts = (pl.scan_parquet(lab_out).group_by(["label", "detailed_label"]).agg(pl.len().alias("n"))
              .sort("n", descending=True).collect().to_dicts())
    stats.update({"file": rel(path), "code": code, "sha256": sha, "label_counts": counts,
                  "n_written": int(pl.scan_parquet(rec_out).select(pl.len()).collect().item())})
    return stats


# --------------------------------------------------------------------------- backgrounds
def _first_per_uid(df: pl.DataFrame, cols: dict[str, str], prefer: str | None = None) -> pl.DataFrame:
    """One log entry per uid: the earliest one, preferring entries whose ``prefer`` field is set
    (most dns.log entries of mDNS conns are query-less responses)."""
    if df.height == 0:
        return pl.DataFrame(schema={"uid": pl.Utf8, **{v: pl.Utf8 for v in cols.values()}})
    keys = ["uid", "_t", "_row"]
    d = df.with_columns(pl.col("ts").cast(pl.Float64, strict=False).alias("_t"))
    if prefer:
        d = d.with_columns((pl.col(prefer).is_null() | pl.col(prefer).is_in(["-", "(empty)", ""])).alias("_miss"))
        keys = ["uid", "_miss", "_t", "_row"]
    return (d.sort(keys).group_by("uid", maintain_order=True).first()
            .select("uid", *[pl.col(k).alias(v) for k, v in cols.items()]))


def bro_background(bro_dir: str | os.PathLike, name: str, shas: dict[str, str], parity: bool = True) -> tuple[pl.DataFrame, dict]:
    """normal_Bro/<name>: conn (+ dns/http/ssl on uid) -> RECORD_SCHEMA frame, stats."""
    from certichain.ingest.zeek_iot23 import read_zeek_log

    d = Path(bro_dir) / name
    conn = read_zeek_log(d / "conn.log")
    dns = read_zeek_log(d / "dns.log") if (d / "dns.log").exists() else pl.DataFrame()
    http = read_zeek_log(d / "http.log") if (d / "http.log").exists() else pl.DataFrame()
    ssl = read_zeek_log(d / "ssl.log") if (d / "ssl.log").exists() else pl.DataFrame()
    j = conn
    if dns.height:
        j = j.join(_first_per_uid(dns, {"query": "_dns_query", "rcode": "_dns_rcode"}, prefer="query"), on="uid", how="left")
    else:
        j = j.with_columns(pl.lit(None, dtype=pl.Utf8).alias("_dns_query"), pl.lit(None, dtype=pl.Utf8).alias("_dns_rcode"))
    hcols = {"method": "_http_method", "uri": "_http_uri", "host": "_http_host", "user_agent": "_http_ua",
             "status_code": "_http_status", "resp_mime_types": "_http_resp_mime"}
    if http.height:
        j = j.join(_first_per_uid(http, hcols, prefer="method"), on="uid", how="left")
    else:
        j = j.with_columns(*[pl.lit(None, dtype=pl.Utf8).alias(v) for v in hcols.values()])
    if ssl.height:
        j = j.join(_first_per_uid(ssl, {"server_name": "_tls_sni"}, prefer="server_name"), on="uid", how="left")
    else:
        j = j.with_columns(pl.lit(None, dtype=pl.Utf8).alias("_tls_sni"))
    code = BG_CODES[name]
    line = pl.col("_line")
    ts_raw = pl.col("ts").cast(pl.Float64, strict=False)
    ts = (ts_raw + 0.5).floor()  # nearest second, like the processed CSV
    dur = pl.col("duration").cast(pl.Float64, strict=False).fill_null(0.0)
    proto = pl.col("proto").str.to_lowercase()
    dq = _null_if("_dns_query", "-", "(empty)", "")
    status = _int("_http_status", pl.Int32)
    conn_file = d / "conn.log"
    out = j.select(
        (pl.lit(code, dtype=pl.Int64) * REC_ID_STRIDE + line).alias("rec_id"),
        ts.alias("ts"), (ts + dur).alias("te"), pl.lit(1.0).alias("ts_res"),
        pl.col("id.orig_h").alias("src"), _int("id.orig_p", pl.Int32).alias("sport"),
        pl.col("id.resp_h").alias("dst"), _int("id.resp_p", pl.Int32).alias("dport"),
        pl.when(proto.is_in(list(PROTOS))).then(proto).otherwise(pl.lit("other")).alias("proto"),
        _null_if("service", "-").fill_null("").alias("service"),
        _int("orig_bytes").fill_null(0).alias("obytes"), _int("resp_bytes").fill_null(0).alias("rbytes"),
        _int("orig_pkts").fill_null(0).alias("opkts"), _int("resp_pkts").fill_null(0).alias("rpkts"),
        _null_if("conn_state", "-").fill_null("").alias("conn_state"),
        (pl.lit("") if parity else _null_if("history", "-").fill_null("")).alias("history"),
        pl.lit(False).alias("bulk"), pl.lit(1, dtype=pl.Int32).alias("n_merged"),
        dq.alias("dns_q"),
        pl.when(dq.is_not_null()).then(_int("_dns_rcode", pl.Int32).fill_null(0)).otherwise(None).alias("dns_rcode"),
        _null_if("_http_method", "-").alias("http_method"),
        _null_if("_http_uri", "-").alias("http_uri"),
        (pl.lit(None, dtype=pl.Utf8) if parity else _null_if("_http_host", "-")).alias("http_host"),
        _null_if("_http_ua", "-").alias("http_ua"),
        pl.when(status > 0).then(status).otherwise(None).alias("http_status"),
        _null_if("_http_resp_mime", "-").alias("http_resp_mime"),
        (pl.lit(None, dtype=pl.Utf8) if parity else _null_if("_tls_sni", "-")).alias("tls_sni"),
        pl.lit(rel(conn_file)).alias("src_file"), pl.lit(shas[rel(conn_file)]).alias("src_sha256"),
        pl.format("lines:{}-{}", line, line).alias("src_locator"),
    )
    n_all = out.height
    out = out.filter(~_ipv6())
    stats = {"name": name, "pool": BG_POOL[name], "src_file": rel(conn_file), "n_conn": conn.height,
             "n_ipv6_dropped": n_all - out.height, "n_records": out.height,
             "n_with_dns": int(out["dns_q"].is_not_null().sum()), "n_with_http": int(out["http_method"].is_not_null().sum()),
             "n_dns_log": dns.height, "n_http_log": http.height, "n_ssl_log": ssl.height,
             "n_ssl_sni_joined": int(j["_tls_sni"].is_not_null().sum()) if "_tls_sni" in j.columns else 0,
             "parity": parity, "ts_min": float(out["ts"].min()), "ts_max": float(out["ts"].max())}
    return conform(out, RECORD_SCHEMA), stats


# --------------------------------------------------------------------------- episodes
def attack_rows(records_glob: list[str], labels_glob: list[str]) -> pl.DataFrame:
    """(rec_id, ts, src, dst, type, file) of attack rows with type in EPISODE_TYPES."""
    lab = (pl.scan_parquet(labels_glob).filter((pl.col("label") == "1") & pl.col("detailed_label").is_in(EPISODE_TYPES))
           .select("rec_id", pl.col("detailed_label").alias("type")))
    rec = pl.scan_parquet(records_glob).select("rec_id", "ts", "src", "dst", "src_file")
    return lab.join(rec, on="rec_id", how="inner").collect()


def extract_episodes(att: pl.DataFrame, gap_s: float = 300.0, max_dur_s: float = 600.0, max_rows: int = 5000,
                     seed: int = 0, train_frac: float = 0.5) -> tuple[pl.DataFrame, dict]:
    """Bursts -> <= max_dur_s sub-windows (<= max_rows by uniform sampling) -> pools per type."""
    A = pl.Series(ATTACKERS)
    att = att.with_columns(
        pl.when(pl.col("src").is_in(A)).then(pl.col("src")).when(pl.col("dst").is_in(A)).then(pl.col("dst"))
        .otherwise(None).alias("attacker"))
    n_no_att = int(att["attacker"].is_null().sum())
    att = att.filter(pl.col("attacker").is_not_null()).sort(["type", "attacker", "ts", "rec_id"])
    g = ["type", "attacker"]
    att = att.with_columns((pl.col("ts").diff().over(g).fill_null(np.inf) > gap_s).cum_sum().over(g).cast(pl.Int32).alias("burst"))
    gb = g + ["burst"]
    att = att.with_columns(pl.col("ts").min().over(gb).alias("burst_t0"), pl.col("ts").max().over(gb).alias("burst_t1"),
                           pl.len().over(gb).alias("burst_n"))
    att = att.with_columns(((pl.col("ts") - pl.col("burst_t0")) // max_dur_s).cast(pl.Int32).alias("sub"))
    gs = gb + ["sub"]
    att = att.with_columns(pl.len().over(gs).alias("n_full"))
    # uniform row sampling inside over-full sub-windows, deterministic: one rng, fixed order
    rng = np.random.default_rng(seed)
    keep = np.ones(att.height, dtype=bool)
    over = att.with_row_index("_i").filter(pl.col("n_full") > max_rows)
    if over.height:
        for _, grp in over.group_by(gs, maintain_order=True):
            idx = grp["_i"].to_numpy()
            chosen = rng.choice(idx.size, max_rows, replace=False)
            m = np.zeros(idx.size, dtype=bool)
            m[chosen] = True
            keep[idx[~m]] = False
    att = att.filter(pl.Series(keep))
    last_octet = pl.col("attacker").str.split(".").list.last()
    ep = (att.group_by(gs, maintain_order=True).agg(
        pl.col("ts").min().alias("t0"), pl.col("ts").max().alias("t1"), pl.len().alias("n_rows"),
        pl.col("n_full").first().alias("n_rows_full"),
        pl.col("burst_t0").first(), pl.col("burst_t1").first(), pl.col("burst_n").first().alias("burst_n_rows"),
        pl.when(pl.col("src") == pl.col("attacker")).then(pl.col("dst")).otherwise(pl.col("src")).unique().sort().alias("victims"),
        pl.col("src_file").unique().sort().alias("files"),
        pl.col("rec_id").alias("rec_ids"),
    ).with_columns(
        pl.format("{}-{}-b{}-s{}", pl.col("type"), last_octet, pl.col("burst").cast(pl.Utf8).str.zfill(3),
                  pl.col("sub").cast(pl.Utf8).str.zfill(3)).alias("episode_id"),
        pl.format("{}-{}-b{}", pl.col("type"), last_octet, pl.col("burst").cast(pl.Utf8).str.zfill(3)).alias("burst_id"),
        (pl.col("n_rows") / pl.col("n_rows_full")).alias("sample_frac"),
        pl.col("victims").list.len().cast(pl.Int32).alias("n_victims"),
    ))
    ep = ep.sort(["type", "t0", "episode_id"]).with_columns(
        pl.int_range(pl.len()).over("type").alias("_rank"), pl.len().over("type").alias("_n"))
    ep = ep.with_columns(pl.when(pl.col("_rank") < (pl.col("_n") * train_frac).floor().cast(pl.Int64))
                         .then(pl.lit("train")).otherwise(pl.lit("caltest")).alias("pool"))
    ep = ep.select("episode_id", "type", "attacker", "victims", "n_victims", "t0", "t1", "n_rows", "n_rows_full",
                   "sample_frac", "pool", "burst_id", "burst_t0", "burst_t1", "burst_n_rows", "sub", "files", "rec_ids")
    stats = {"n_attack_rows": int(att.height + (~pl.Series(keep)).sum()), "n_rows_without_attacker": n_no_att,
             "n_rows_kept": int(att.height)}
    return ep, stats
