"""IoT-23 ``conn.log.labeled`` -> RECORD_SCHEMA windows + quarantined labels (DESIGN §5.1.1).

File format (checked on all 23 logs): 8 ``#`` header lines, tab-separated Zeek conn fields
(20), then ONE tail field ``tunnel_parents   label   detailed-label`` whose three parts are
separated by **three spaces**; a final ``#close`` line. polars reads the 21 tab fields as
strings (explicit schema, ``comment_prefix='#'``, no quoting) and the tail is split with
``str.splitn("   ", 3)``. Data row i (0-based) is physical line i + 9 (verified per file).

Two passes (memory: pass 1 keeps only a per-second histogram; pass 2 streams):

* pass 1 -- stream (ts, id.orig_h, id.resp_h, tail) and count records per
  (second, seed involvement, label, detailed-label, IPv6) -> per-2-h-bin counts. Bins tile
  the capture from floor(min ts).
* select -- malicious scenario: bins with >= 1 malicious record of the seed, capped at
  ``max_mal_windows`` (uniform, numpy default_rng(seed)); plus <= ``max_benign_windows``
  benign-only bin (no malicious record of any host, >= ``benign_min_seed_records`` records
  of the seed; uniform). Honeypot: bins with >= 1 seed record, capped at
  ``honeypot_max_windows`` (uniform). The rule uses labels: it defines the investigated
  population and is applied identically everywhere (documented).
* pass 2 -- stream the records of the selected bins only (IPv4 only), map to RECORD_SCHEMA,
  write ``records/<scenario>/<window_id>.parquet`` and quarantined
  ``labels/<scenario>/<window_id>.parquet``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from certichain.io.labels import conform_lazy
from certichain.io.paths import atomic_sink_parquet, rel
from certichain.io.schema import LABEL_SCHEMA, RECORD_SCHEMA

ZEEK_CONN_FIELDS = ["ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", "service",
                    "duration", "orig_bytes", "resp_bytes", "conn_state", "local_orig", "local_resp",
                    "missed_bytes", "history", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
IOT23_TAIL = "tunnel_parents   label   detailed-label"
IOT23_COLS = ZEEK_CONN_FIELDS + ["tail"]
TAIL_SEP = "   "  # exactly three spaces
HEADER_LINES = 8
ROW_TO_LINE = HEADER_LINES + 1  # physical (1-based) line of data row i is i + 9
REC_ID_STRIDE = 1_000_000_000   # rec_id = scenario_code * stride + physical line number
ZEEK_PROTOS = {"tcp", "udp", "icmp"}


# --------------------------------------------------------------------------- structure
@dataclass
class LogStructure:
    sha256: str
    n_lines: int
    comment_lines: list[int]
    fields_line: str
    ok: bool
    reason: str = ""


def scan_structure(path: str | os.PathLike, chunk: int = 64 << 20) -> LogStructure:
    """One read of the file: sha256, line count, positions of ``#`` lines, header check.

    ``ok`` iff lines 1..8 are the Zeek header, no ``#`` line occurs in the data except a
    final ``#close`` line, and the #fields line matches the IoT-23 layout.
    """
    import hashlib

    h = hashlib.sha256()
    n_nl = 0
    comments: list[int] = []
    head = b""
    prev_last = b"\n"  # virtual newline before the file start
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            if len(head) < 8192:
                head += b[: 8192 - len(head)]
            # line starts with '#': preceded by '\n' (or file start)
            if prev_last == b"\n" and b[:1] == b"#":
                comments.append(n_nl + 1)
            pos = b.find(b"\n#")
            while pos != -1:
                comments.append(n_nl + b.count(b"\n", 0, pos + 1) + 1)
                pos = b.find(b"\n#", pos + 1)
            n_nl += b.count(b"\n")
            prev_last = b[-1:]
    lines = head.split(b"\n")
    fields_line = ""
    for ln in lines[:HEADER_LINES]:
        if ln.startswith(b"#fields"):
            fields_line = ln.decode("utf-8", "replace")
    exp_fields = "#fields\t" + "\t".join(ZEEK_CONN_FIELDS) + "\t" + IOT23_TAIL
    n_lines = n_nl + (0 if prev_last == b"\n" else 1)
    ok, reason = True, ""
    if comments[:HEADER_LINES] != list(range(1, HEADER_LINES + 1)):
        ok, reason = False, f"header comment lines {comments[:HEADER_LINES]}"
    extra = comments[HEADER_LINES:]
    if extra and extra != [n_lines]:
        ok, reason = False, f"comment lines inside data: {extra[:10]}"
    if fields_line != exp_fields:
        ok, reason = False, f"unexpected #fields line: {fields_line[:200]!r}"
    return LogStructure(h.hexdigest(), n_lines, comments, fields_line, ok, reason)


def n_data_rows(st: LogStructure) -> int:
    return st.n_lines - len(st.comment_lines)


# --------------------------------------------------------------------------- scanning
def scan_conn_labeled(path: str | os.PathLike) -> pl.LazyFrame:
    """Lazy scan: 21 Utf8 columns + ``_row`` (0-based data-row index)."""
    return pl.scan_csv(path, separator="\t", comment_prefix="#", has_header=False,
                       schema={c: pl.Utf8 for c in IOT23_COLS}, quote_char=None,
                       row_index_name="_row", raise_if_empty=False)


def split_tail(lf: pl.LazyFrame) -> pl.LazyFrame:
    """tail -> tunnel_parents, label, detailed_label ('-' detailed label -> null)."""
    parts = pl.col("tail").str.splitn(TAIL_SEP, 3)
    return lf.with_columns(
        parts.struct.field("field_0").alias("tunnel_parents"),
        parts.struct.field("field_1").str.strip_chars().alias("label"),
        parts.struct.field("field_2").str.strip_chars().alias("detailed_label"),
    ).with_columns(
        pl.when(pl.col("detailed_label").is_in(["-", ""])).then(None)
        .otherwise(pl.col("detailed_label")).alias("detailed_label"))


def _is_mal() -> pl.Expr:
    return pl.col("label").str.to_lowercase() == "malicious"


def _ipv6() -> pl.Expr:
    return pl.col("id.orig_h").str.contains(":", literal=True) | pl.col("id.resp_h").str.contains(":", literal=True)


def pass1_histogram(path: str | os.PathLike, candidates: list[str]) -> pl.DataFrame:
    """Per-second histogram: (sec, ipv6, s0..sK (candidate involvement), label, detailed_label) -> n.

    Streams only ts, orig_h, resp_h and the tail.
    """
    lf = split_tail(scan_conn_labeled(path).select("ts", "id.orig_h", "id.resp_h", "tail"))
    flags = [((pl.col("id.orig_h") == ip) | (pl.col("id.resp_h") == ip)).alias(f"s{i}")
             for i, ip in enumerate(candidates)]
    lf = lf.select(pl.col("ts").cast(pl.Float64, strict=False).floor().cast(pl.Int64).alias("sec"),
                   _ipv6().alias("ipv6"), *flags, "label", "detailed_label")
    keys = ["sec", "ipv6", *[f"s{i}" for i in range(len(candidates))], "label", "detailed_label"]
    return lf.group_by(keys).agg(pl.len().cast(pl.Int64).alias("n")).collect(engine="streaming")


@dataclass
class Bins:
    t0: int
    bin_s: int
    table: pl.DataFrame  # per bin: bin, n_all, n_ipv6, n, n_seed, n_mal, n_seed_mal
    labels: pl.DataFrame  # per bin x label x detailed (IPv4): n
    seed_labels: pl.DataFrame  # per bin x label x detailed for seed records
    n_null_ts: int
    n_null_label: int


def make_bins(hist: pl.DataFrame, seed_flag: str, bin_s: int = 7200) -> Bins:
    """Aggregate the per-second histogram into bins that tile the capture from floor(min ts)."""
    n_null_ts = int(hist.filter(pl.col("sec").is_null())["n"].sum() or 0)
    n_null_label = int(hist.filter(pl.col("label").is_null())["n"].sum() or 0)
    h = hist.filter(pl.col("sec").is_not_null())
    t0 = int(h["sec"].min())
    h = h.with_columns(((pl.col("sec") - t0) // bin_s).cast(pl.Int32).alias("bin"),
                       _is_mal().fill_null(False).alias("mal"))
    v4 = h.filter(~pl.col("ipv6"))
    tab = h.group_by("bin").agg(
        pl.col("n").sum().alias("n_all"),
        pl.col("n").filter(pl.col("ipv6")).sum().alias("n_ipv6"),
    ).join(v4.group_by("bin").agg(
        pl.col("n").sum().alias("n"),
        pl.col("n").filter(pl.col(seed_flag)).sum().alias("n_seed"),
        pl.col("n").filter(pl.col("mal")).sum().alias("n_mal"),
        pl.col("n").filter(pl.col(seed_flag) & pl.col("mal")).sum().alias("n_seed_mal"),
        pl.col("sec").min().alias("sec_min"), pl.col("sec").max().alias("sec_max"),
    ), on="bin", how="left").with_columns(pl.col(c).fill_null(0) for c in ("n", "n_seed", "n_mal", "n_seed_mal")
                                           ).sort("bin")
    labels = v4.group_by(["bin", "label", "detailed_label"]).agg(pl.col("n").sum()).sort(["bin", "n"], descending=[False, True])
    seed_labels = (v4.filter(pl.col(seed_flag)).group_by(["bin", "label", "detailed_label"])
                   .agg(pl.col("n").sum()).sort(["bin", "n"], descending=[False, True]))
    return Bins(t0, bin_s, tab, labels, seed_labels, n_null_ts, n_null_label)


@dataclass
class WindowChoice:
    bin: int
    kind: str  # malicious | benign_only | honeypot


@dataclass
class SelectionConfig:
    max_mal_windows: int = 12
    max_benign_windows: int = 1
    honeypot_max_windows: int = 6
    benign_min_seed_records: int = 1
    rng_seed: int = 0


def select_windows(tab: pl.DataFrame, kind: str, cfg: SelectionConfig) -> tuple[list[WindowChoice], dict]:
    """Window-inclusion rule (label-based; applied identically to every scenario)."""
    info: dict = {}
    if kind == "honeypot":
        cand = tab.filter(pl.col("n_seed") >= 1)["bin"].to_list()
        info["n_candidate_bins"] = len(cand)
        rng = np.random.default_rng(cfg.rng_seed)
        if len(cand) > cfg.honeypot_max_windows:
            cand = sorted(rng.choice(cand, cfg.honeypot_max_windows, replace=False).tolist())
        return [WindowChoice(int(b), "honeypot") for b in sorted(cand)], info
    mal = tab.filter(pl.col("n_seed_mal") >= 1)["bin"].to_list()
    ben = tab.filter((pl.col("n_mal") == 0) & (pl.col("n_seed") >= cfg.benign_min_seed_records))["bin"].to_list()
    info["n_malicious_bins"] = len(mal)
    info["n_benign_only_bins"] = len(ben)
    rng_m = np.random.default_rng(cfg.rng_seed)
    rng_b = np.random.default_rng(cfg.rng_seed + 1)
    if len(mal) > cfg.max_mal_windows:
        mal = sorted(rng_m.choice(mal, cfg.max_mal_windows, replace=False).tolist())
    if len(ben) > cfg.max_benign_windows:
        ben = sorted(rng_b.choice(ben, cfg.max_benign_windows, replace=False).tolist()) if cfg.max_benign_windows else []
    out = [WindowChoice(int(b), "malicious") for b in mal] + [WindowChoice(int(b), "benign_only") for b in ben]
    return sorted(out, key=lambda w: w.bin), info


# --------------------------------------------------------------------------- mapping
def _num(col: str, dtype) -> pl.Expr:
    """Zeek numeric field: '-' / '(empty)' / unparsable -> null."""
    return pl.col(col).cast(pl.Float64, strict=False).cast(dtype, strict=False)


def to_record_columns(lf: pl.LazyFrame, scen_code: int, src_file: str, src_sha256: str,
                      extra: tuple[str, ...] = ()) -> pl.LazyFrame:
    """Map split IoT-23 columns to RECORD_SCHEMA columns (+ label, detailed_label kept for the
    quarantine split, + ``extra`` pass-through columns).

    Conventions: unset ('-') bytes / duration -> 0 (as TON's processed CSV does);
    service/history '-' -> ""; proto not in {tcp, udp, icmp} -> "other"; ICMP keeps Zeek's
    type/code in sport/dport; ts_res = 1e-6.
    """
    line = pl.col("_row").cast(pl.Int64) + ROW_TO_LINE
    ts = pl.col("ts").cast(pl.Float64, strict=False)
    dur = _num("duration", pl.Float64).fill_null(0.0)
    blank = lambda c: pl.when(pl.col(c).is_in(["-", "(empty)"]) | pl.col(c).is_null()).then(pl.lit("")).otherwise(pl.col(c))
    proto = pl.col("proto").str.to_lowercase()
    return lf.select(
        (pl.lit(scen_code, dtype=pl.Int64) * REC_ID_STRIDE + line).alias("rec_id"),
        ts.alias("ts"),
        (ts + dur).alias("te"),
        pl.lit(1e-6).alias("ts_res"),
        pl.col("id.orig_h").alias("src"),
        _num("id.orig_p", pl.Int32).alias("sport"),
        pl.col("id.resp_h").alias("dst"),
        _num("id.resp_p", pl.Int32).alias("dport"),
        pl.when(proto.is_in(list(ZEEK_PROTOS))).then(proto).otherwise(pl.lit("other")).alias("proto"),
        blank("service").alias("service"),
        _num("orig_bytes", pl.Int64).fill_null(0).alias("obytes"),
        _num("resp_bytes", pl.Int64).fill_null(0).alias("rbytes"),
        _num("orig_pkts", pl.Int64).fill_null(0).alias("opkts"),
        _num("resp_pkts", pl.Int64).fill_null(0).alias("rpkts"),
        blank("conn_state").alias("conn_state"),
        blank("history").alias("history"),
        pl.lit(False).alias("bulk"),
        pl.lit(1, dtype=pl.Int32).alias("n_merged"),
        pl.lit(src_file).alias("src_file"),
        pl.lit(src_sha256).alias("src_sha256"),
        pl.format("lines:{}-{}", line, line).alias("src_locator"),
        pl.col("label"),
        pl.col("detailed_label"),
        *[pl.col(c) for c in extra],
    )


def extract_windows(path: str | os.PathLike, bins: Bins, chosen: list[WindowChoice], scen_code: int,
                    src_sha256: str, tmp_path: Path) -> pl.LazyFrame:
    """Pass 2: stream the IPv4 records of the chosen bins to ``tmp_path`` (one parquet with a
    ``_bin`` column, records + label columns). Returns a lazy scan of it."""
    sel = [int(w.bin) for w in chosen]
    lf = split_tail(scan_conn_labeled(path))
    sec = pl.col("ts").cast(pl.Float64, strict=False).floor().cast(pl.Int64)
    lf = lf.filter(~_ipv6()).with_columns(((sec - bins.t0) // bins.bin_s).cast(pl.Int32).alias("_bin"))
    lf = lf.filter(pl.col("_bin").is_in(sel))
    rec = to_record_columns(lf, scen_code, rel(path), src_sha256, extra=("_bin",))
    atomic_sink_parquet(rec, tmp_path)
    return pl.scan_parquet(tmp_path)


def write_window(all_lf: pl.LazyFrame, b: int, rec_path: Path, lab_path: Path) -> None:
    """Stream one bin of the pass-2 temp parquet to its records / quarantined labels files.

    Records keep conn.log order (Zeek writes at connection END, so this is NOT ts order);
    consumers sort by (ts, tiehash). Streaming keeps memory bounded even for the 67 M-record
    window of 43-1.
    """
    w = all_lf.filter(pl.col("_bin") == b)
    atomic_sink_parquet(conform_lazy(w, RECORD_SCHEMA), rec_path)
    atomic_sink_parquet(conform_lazy(w.select("rec_id", "label", "detailed_label"), LABEL_SCHEMA), lab_path)


# --------------------------------------------------------------------------- generic Zeek logs
def read_zeek_log(path: str | os.PathLike) -> pl.DataFrame:
    """Read any Zeek ASCII log (``#fields`` header) as all-Utf8 columns + ``_row``.

    Used for the TON normal_Bro conn/dns/http/ssl logs (no label tail there).
    """
    fields = None
    n_head = 0
    with open(path, "rb") as f:
        for raw in f:
            if not raw.startswith(b"#"):
                break
            n_head += 1
            if raw.startswith(b"#fields"):
                fields = raw.rstrip(b"\r\n").decode("utf-8").split("\t")[1:]
    if fields is None:
        raise ValueError(f"{path}: no #fields header")
    df = pl.read_csv(path, separator="\t", comment_prefix="#", has_header=False,
                     schema={c: pl.Utf8 for c in fields}, quote_char=None, row_index_name="_row",
                     raise_if_empty=False)
    return df.with_columns((pl.col("_row").cast(pl.Int64) + n_head + 1).alias("_line"))
