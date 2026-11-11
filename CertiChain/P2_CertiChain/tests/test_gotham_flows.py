"""Tests for the Gotham per-packet CSV -> flow builder (P2-A)."""
from __future__ import annotations

import polars as pl

from certichain.ingest import gotham_csv as G
from certichain.io.schema import RECORD_SCHEMA

HEADER = ",".join(G.GOTHAM_FIELDS) + "\n"


def _pkt(t, src, dst, proto, sport=None, dport=None, flags=None, applayer="data", length=60, label="Benign"):
    """One CSV row. frame.protocols uses the app layer after tcp/udp; ports are tcp.* or udp.*."""
    protos = f"eth:ethertype:ip:{ {6:'tcp',17:'udp',1:'icmp',47:'gre'}[proto] }"
    if proto in (6, 17) and applayer:
        protos += f":{applayer}"
    tcp_sp = tcp_dp = udp_sp = udp_dp = ""
    if proto == 6:
        tcp_sp, tcp_dp = f"{sport}.0", f"{dport}.0"
    elif proto == 17:
        udp_sp, udp_dp = f"{sport}.0", f"{dport}.0"
    fl = f"0x{flags:04x}" if flags is not None else ""
    hms, _, frac = t.partition(".")
    t9 = f"{hms}.{(frac or '0').ljust(9, '0')[:9]}"  # dataset uses 9 fractional digits (ns-formatted)
    # 22 tshark fields + label (column order per the dataset header: ip.dst then ip.src)
    fields = ["Jan 18, 2025 " + t9 + " GMT", str(length), protos, "02:42:00:00:00:01", "02:42:00:00:00:02",
              dst, src, "0x02", "64", str(proto), "0x0", "", tcp_sp, tcp_dp, fl, "", "", "0x0", "", "",
              udp_sp, udp_dp, label]
    assert len(fields) == len(G.GOTHAM_FIELDS)
    return ",".join(f'"{x}"' if "," in x else x for x in fields) + "\n"


# TCP flags
SYN, SA, ACK, PSHACK, FIN, RST = 0x02, 0x12, 0x10, 0x18, 0x11, 0x04
DEV = "192.168.18.17"


def _write(tmp_path, rows):
    p = tmp_path / "iotsim-air-quality-1.csv"
    p.write_text(HEADER + "".join(rows))
    return p


def _flows(path, ip):
    pk, st = G.parse_packets(path, ip)
    pk = G.assign_sessions(pk)
    rec, labs, fstats = G.flows_from_packets(pk)
    out = G.to_record_schema(rec, 1, "x", "sha")
    maj = G.majority_labels(labs)
    return out.join(maj, on="_key", how="left", maintain_order="left"), fstats


def test_tcp_handshake_conn_state_history(tmp_path):
    # a clean telnet session: SYN, SYN-ACK, data both ways, FIN both ways -> SF, ShADadFf
    rows = [
        _pkt("21:29:24.000000", "192.168.0.100", DEV, 6, 5000, 23, SYN, applayer=None, label="Telnet Brute Force"),
        _pkt("21:29:24.010000", DEV, "192.168.0.100", 6, 23, 5000, SA, applayer=None, label="Telnet Brute Force"),
        _pkt("21:29:24.020000", "192.168.0.100", DEV, 6, 5000, 23, PSHACK, applayer="telnet", label="Telnet Brute Force"),
        _pkt("21:29:24.030000", DEV, "192.168.0.100", 6, 23, 5000, PSHACK, applayer="telnet", label="Telnet Brute Force"),
        _pkt("21:29:24.040000", "192.168.0.100", DEV, 6, 5000, 23, FIN, applayer=None, label="Telnet Brute Force"),
        _pkt("21:29:24.050000", DEV, "192.168.0.100", 6, 23, 5000, FIN, applayer=None, label="Telnet Brute Force"),
    ]
    out, _ = _flows(_write(tmp_path, rows), DEV)
    assert out.height == 1
    r = out.row(0, named=True)
    assert r["conn_state"] == "SF"
    # history is a documented first-occurrence approximation; check the salient letters & order
    assert r["history"].startswith("Sh")          # orig SYN, resp SYN-ACK
    assert "D" in r["history"] and "d" in r["history"]  # data both directions
    assert r["history"].endswith("Ff")            # orig then resp FIN
    assert r["service"] == "telnet"
    assert r["proto"] == "tcp"
    assert (r["src"], r["sport"], r["dst"], r["dport"]) == ("192.168.0.100", 5000, DEV, 23)
    assert (r["opkts"], r["rpkts"]) == (3, 3)
    assert r["detailed_label"] == "Telnet Brute Force"


def test_rst_rejected(tmp_path):
    # SYN then RST from the responder -> REJ, history Sr
    rows = [
        _pkt("21:30:00.000000", "192.168.35.10", DEV, 6, 40000, 23, SYN, applayer=None, label="Unknown"),
        _pkt("21:30:00.001000", DEV, "192.168.35.10", 6, 23, 40000, RST, applayer=None, label="Unknown"),
    ]
    out, _ = _flows(_write(tmp_path, rows), DEV)
    assert out.height == 1
    r = out.row(0, named=True)
    assert r["conn_state"] == "REJ"
    assert r["history"] == "Sr"
    # Unknown -> Unknown binary label (masscan caveat 2 handled on GT side)
    lab = out.select(G.binary_label(pl.col("detailed_label")).alias("b"))["b"][0]
    assert lab == "Unknown"


def test_idle_timeout_splits_flows(tmp_path):
    # same 5-tuple, a gap > 60 s -> two separate flows
    rows = [
        _pkt("21:00:00.000000", DEV, "192.168.4.1", 17, 5000, 5683, applayer="coap"),
        _pkt("21:00:30.000000", DEV, "192.168.4.1", 17, 5000, 5683, applayer="coap"),
        _pkt("21:02:00.000000", DEV, "192.168.4.1", 17, 5000, 5683, applayer="coap"),  # +90 s
    ]
    out, _ = _flows(_write(tmp_path, rows), DEV)
    assert out.height == 2
    assert out["service"].to_list() == ["coap", "coap"]
    assert out["conn_state"].to_list() == ["S0", "S0"]  # UDP, no response


def test_active_timeout_splits_long_flow(tmp_path):
    # UDP packets every 100 s for > 600 s with no idle gap > 60 s? gaps are 100 s > 60 s so idle splits;
    # use 30 s gaps to stay within idle but exceed the 600 s active timeout.
    rows = [_pkt(f"21:{m:02d}:{s:02d}.000000", DEV, "192.168.4.1", 17, 5000, 5683, applayer="coap")
            for (m, s) in [(0, 0), (0, 30), (1, 0), (1, 30), (2, 0), (2, 30), (3, 0), (3, 30), (4, 0), (4, 30),
                           (5, 0), (5, 30), (6, 0), (6, 30), (7, 0), (7, 30), (8, 0), (8, 30), (9, 0), (9, 30),
                           (10, 0), (10, 30), (11, 0)]]
    out, _ = _flows(_write(tmp_path, rows), DEV)
    # 11 min of 30-s-spaced packets, active timeout 600 s -> at least 2 flows
    assert out.height >= 2


def test_flood_bulk_path(tmp_path):
    # > threshold flows in one (src,dst,proto) group -> bulk records, one per 10-s bin
    rows = []
    t = 0
    for i in range(12):  # 12 distinct UDP flows, spaced 20 s so each is its own flow, spanning > 10 s bins
        mm, ss = divmod(i * 20, 60)
        rows.append(_pkt(f"21:{mm:02d}:{ss:02d}.000000", DEV, "192.168.18.18", 17, 5000 + i, 44404,
                         applayer="data", length=500, label="Mirai UDP Flooding"))
    out, fstats = _flows_threshold(_write(tmp_path, rows), DEV, threshold=5, bin_s=10.0)
    bulk = out.filter(pl.col("bulk"))
    assert bulk.height >= 1
    assert bulk["n_merged"].sum() == 12  # every flow folded into bulk records
    # bulk records carry NULL ports and the majority label
    assert bulk["sport"].null_count() == bulk.height
    assert bulk["detailed_label"].unique().to_list() == ["Mirai UDP Flooding"]
    assert fstats["n_bulk_records"] >= 1 and fstats["n_bulk_groups"] == 1


def _flows_threshold(path, ip, threshold, bin_s):
    pk, _ = G.parse_packets(path, ip)
    pk = G.assign_sessions(pk)
    rec, labs, fstats = G.flows_from_packets(pk, flood_threshold=threshold, bulk_bin_s=bin_s)
    out = G.to_record_schema(rec, 1, "x", "sha")
    maj = G.majority_labels(labs)
    return out.join(maj, on="_key", how="left", maintain_order="left"), fstats


def test_superset_filter(tmp_path):
    # caveat 3: a -1 superset file holds another device's rows; only the own-IP rows survive
    rows = [
        _pkt("21:00:00.000000", DEV, "192.168.4.1", 17, 5000, 5683, applayer="coap"),          # own
        _pkt("21:00:01.000000", "192.168.20.19", "192.168.4.1", 17, 5001, 5683, applayer="coap"),  # other device
    ]
    p = _write(tmp_path, rows)
    pk, st = G.parse_packets(p, DEV)
    assert st["n_rows"] == 2 and st["n_own"] == 1
    out, _ = _flows(p, DEV)
    assert out.height == 1
    assert out.row(0, named=True)["src"] == DEV


def test_records_are_label_free(tmp_path):
    from certichain.io.labels import label_like_columns
    rows = [_pkt("21:00:00.000000", DEV, "192.168.4.1", 17, 5000, 5683, applayer="coap")]
    out, _ = _flows(_write(tmp_path, rows), DEV)
    rec_cols = [c for c in RECORD_SCHEMA if c in out.columns]
    assert label_like_columns(rec_cols) == []
