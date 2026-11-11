"""Decoder for the 48101/TCP report records in the Gotham mirai-infection pcaps (DESIGN §3.1.2).

These records give the forensic ground truth for "which bot resolved victim y" used by the
Gotham case study (§5.2.3): a report from bot b naming target y that precedes ITT(y) makes
REP(b) -> ITT(y) a direct transition. We only *read* the already-captured packets; nothing is
sent or generated.

Wire format of one record on a fresh 48101/TCP connection (big-endian), confirmed below against
the captured bytes:

    byte 0           0x00            (new-format marker; any other value => legacy, port implied 23)
    bytes 1..4       target IPv4     (4 bytes)
    bytes 5..6       target port     (uint16)
    byte 7           L_user          then L_user bytes of user
    byte 8+L_user    L_pass          then L_pass bytes of pass

The sender emits these fields in several writes, so one record can span multiple TCP segments;
segments are read per frame (tshark with reassembly off), grouped by tcp.stream, de-duplicated by
tcp.seq, concatenated in sequence order, then decoded. ``verify_layout`` checks the first decoded
records against §3.1.2 (target inside the Gotham 192.168.16-20.x plan, port in {23, 2323},
printable user/pass) and is recorded in notes/P2-A.md.

Output (one row per decoded record): t (epoch s), reporter (ip.src), target, port, user, pass,
src_file, frame (the second, payload-bearing frame number).
"""
from __future__ import annotations

import ipaddress
import os
import subprocess
from pathlib import Path

import polars as pl

TSHARK = "/opt/homebrew/bin/tshark"
REPORT_PORT = 48101
TSHARK_FIELDS = ["frame.number", "frame.time_epoch", "ip.src", "ip.dst", "tcp.srcport", "tcp.stream", "tcp.seq",
                 "tcp.len", "tcp.payload"]
# Gotham victim address plan (device subnets 192.168.16.0 .. 192.168.20.255) and expected ports.
PLAN_LOW = int(ipaddress.IPv4Address("192.168.16.0"))
PLAN_HIGH = int(ipaddress.IPv4Address("192.168.20.255"))
EXPECTED_PORTS = (23, 2323)


def tshark_cmd(pcap: str | os.PathLike, tshark: str = TSHARK) -> list[str]:
    cmd = [tshark, "-r", str(pcap), "-n", "-o", "tcp.desegment_tcp_streams:FALSE",
           "-Y", f"tcp.dstport=={REPORT_PORT} && tcp.len>0", "-T", "fields"]
    for f in TSHARK_FIELDS:
        cmd += ["-e", f]
    return cmd


def read_segments(pcap: str | os.PathLike, tshark: str = TSHARK, timeout: float = 600.0) -> pl.DataFrame:
    """Run tshark and return the payload-bearing segments as a frame (hex payload kept as text)."""
    proc = subprocess.run(tshark_cmd(pcap, tshark), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"tshark failed on {pcap}: {proc.stderr.strip()[:300]}")
    rows = [ln.split("\t") for ln in proc.stdout.splitlines() if ln.strip()]
    schema = {f: pl.Utf8 for f in TSHARK_FIELDS}
    if not rows:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame({TSHARK_FIELDS[i]: [r[i] if i < len(r) else None for r in rows] for i in range(len(TSHARK_FIELDS))},
                      schema=schema)
    return df.with_columns(pl.col("tcp.payload").str.replace_all(":", ""))


def assemble_streams(seg: pl.DataFrame) -> pl.DataFrame:
    """Per (ip.src, tcp.stream): de-dup by tcp.seq, concat payloads in seq order -> one hex blob."""
    if seg.height == 0:
        return pl.DataFrame(schema={"src": pl.Utf8, "stream": pl.Utf8, "t": pl.Float64, "frame": pl.Int64, "hex": pl.Utf8})
    s = seg.with_columns(pl.col("tcp.seq").cast(pl.Int64, strict=False).alias("_seq"),
                         pl.col("frame.number").cast(pl.Int64, strict=False).alias("_frame"),
                         pl.col("frame.time_epoch").cast(pl.Float64, strict=False).alias("_t"))
    s = s.sort(["ip.src", "tcp.stream", "_seq", "_frame"]).unique(["ip.src", "tcp.stream", "_seq"], keep="first", maintain_order=True)
    return s.group_by(["ip.src", "tcp.stream"], maintain_order=True).agg(
        pl.col("_t").filter(pl.col("tcp.len").cast(pl.Int64, strict=False) > 1).min().fill_null(pl.col("_t").min()).alias("t"),
        pl.col("_frame").filter(pl.col("tcp.len").cast(pl.Int64, strict=False) > 1).min().fill_null(pl.col("_frame").max()).alias("frame"),
        pl.col("tcp.payload").str.join("").alias("hex"),
    ).rename({"ip.src": "src", "tcp.stream": "stream"})


def decode_record(blob: bytes) -> dict | None:
    """Decode one assembled record per the wire format above; None if it is malformed/truncated."""
    if len(blob) < 8 or blob[0] != 0x00:
        return None
    target = ".".join(str(b) for b in blob[1:5])
    port = int.from_bytes(blob[5:7], "big")
    i = 7
    lu = blob[i]
    i += 1
    if i + lu > len(blob):
        return None
    user = blob[i:i + lu]
    i += lu
    if i >= len(blob):
        return None
    lp = blob[i]
    i += 1
    if i + lp > len(blob):
        return None
    pw = blob[i:i + lp]
    dec = lambda z: z.decode("utf-8", "replace")
    printable = lambda z: all(0x20 <= c < 0x7F for c in z)
    return {"target": target, "port": port, "user": dec(user), "pass": dec(pw),
            "user_printable": printable(user), "pass_printable": printable(pw),
            "n_trailing": len(blob) - (i + lp)}


def decode_streams(streams: pl.DataFrame, src_file: str) -> pl.DataFrame:
    recs = []
    for row in streams.iter_rows(named=True):
        try:
            blob = bytes.fromhex(row["hex"])
        except ValueError:
            continue
        d = decode_record(blob)
        if d is None:
            continue
        d.update({"t": row["t"], "reporter": row["src"], "src_file": src_file, "frame": row["frame"]})
        recs.append(d)
    cols = ["t", "reporter", "target", "port", "user", "pass", "src_file", "frame", "user_printable",
            "pass_printable", "n_trailing"]
    if not recs:
        return pl.DataFrame(schema={c: pl.Float64 if c in ("t",) else (pl.Int64 if c in ("port", "frame", "n_trailing")
                            else (pl.Boolean if "printable" in c else pl.Utf8)) for c in cols})
    return pl.DataFrame(recs).select(cols).sort(["t", "frame"])


def parse_pcap(pcap: str | os.PathLike, src_file: str, tshark: str = TSHARK) -> tuple[pl.DataFrame, dict]:
    seg = read_segments(pcap, tshark)
    streams = assemble_streams(seg)
    rec = decode_streams(streams, src_file)
    stats = {"n_segments": seg.height, "n_streams": streams.height, "n_decoded": rec.height,
             "n_undecoded": streams.height - rec.height}
    return rec, stats


def in_plan(ip: str) -> bool:
    try:
        return PLAN_LOW <= int(ipaddress.IPv4Address(ip)) <= PLAN_HIGH
    except ipaddress.AddressValueError:
        return False


def verify_layout(rec: pl.DataFrame, n: int = 10) -> dict:
    """Check the first ``n`` decoded records against DESIGN §3.1.2 (used in the notes)."""
    head = rec.head(n)
    examples = head.select("t", "reporter", "target", "port", "user", "pass", "frame").to_dicts()
    checks = {
        "n_checked": head.height,
        "all_targets_in_plan": bool(all(in_plan(t) for t in head["target"].to_list())) if head.height else None,
        "all_ports_expected": bool(head["port"].is_in(list(EXPECTED_PORTS)).all()) if head.height else None,
        "all_user_printable": bool(head["user_printable"].all()) if head.height else None,
        "all_pass_printable": bool(head["pass_printable"].all()) if head.height else None,
        "distinct_ports": sorted(set(rec["port"].to_list())),
        "n_targets_out_of_plan": int(sum(1 for t in rec["target"].to_list() if not in_plan(t))),
        "examples": examples,
    }
    checks["ok"] = bool(checks["all_targets_in_plan"] and checks["all_ports_expected"]
                        and checks["all_user_printable"] and checks["all_pass_printable"]) if head.height else False
    return checks
