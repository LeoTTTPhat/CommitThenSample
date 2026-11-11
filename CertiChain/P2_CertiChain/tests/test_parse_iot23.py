"""Tests for the IoT-23 Zeek conn.log.labeled parser (P2-A)."""
from __future__ import annotations

import polars as pl
import pytest

from certichain.ingest import zeek_iot23 as Z
from certichain.io.labels import label_like_columns
from certichain.io.schema import RECORD_SCHEMA

HEADER = (
    "#separator \\x09\n#set_separator\t,\n#empty_field\t(empty)\n#unset_field\t-\n#path\tconn\n"
    "#open\t2018-05-21-20-52-40\n"
    "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\tduration\torig_bytes\t"
    "resp_bytes\tconn_state\tlocal_orig\tlocal_resp\tmissed_bytes\thistory\torig_pkts\torig_ip_bytes\t"
    "resp_pkts\tresp_ip_bytes\ttunnel_parents   label   detailed-label\n"
    "#types\ttime\tstring\taddr\tport\taddr\tport\tenum\tstring\tinterval\tcount\tcount\tstring\tbool\tbool\t"
    "count\tstring\tcount\tcount\tcount\tcount\tset[string]   string   string\n"
)


def _row(ts, oh, op, rh, rp, proto, service, dur, ob, rb, cs, hist, tail):
    return (f"{ts}\tCuid{ts.replace('.', '')}\t{oh}\t{op}\t{rh}\t{rp}\t{proto}\t{service}\t{dur}\t{ob}\t{rb}\t{cs}\t-\t-\t0\t"
            f"{hist}\t1\t40\t0\t0\t{tail}")


# tail tokens are separated by exactly three spaces; tunnel_parents may itself be "(empty)" or a uid set
ROWS = [
    _row("1526792995.887377", "192.168.2.5", 41501, "54.39.23.28", 2407, "tcp", "irc", "93280.03", 6303, 9069, "S1", "ShADad", "(empty)   Malicious   C&C"),
    _row("1526792996.100000", "192.168.2.5", 123, "8.8.8.8", 53, "udp", "dns", "0.01", 40, 80, "SF", "Dd", "-   Benign   -"),
    _row("1526792997.500000", "10.0.0.9", 5, "192.168.2.5", 7, "icmp", "-", "-", "-", "-", "OTH", "-", "CAbc,CDef   Malicious   PartOfAHorizontalPortScan"),
    _row("1526799000.000000", "192.168.2.5", 44, "1.2.3.4", 80, "tcp", "http", "0.5", 100, 200, "SF", "ShADadFf", "-   Benign   -"),
]
FIXTURE = HEADER + "\n".join(ROWS) + "\n#close\t2018-05-21-21-00-00   benign   -\n"


@pytest.fixture
def log(tmp_path):
    p = tmp_path / "conn.log.labeled"
    p.write_text(FIXTURE)
    return p


def test_structure_ok(log):
    st = Z.scan_structure(log)
    assert st.ok, st.reason
    assert Z.n_data_rows(st) == 4
    assert st.comment_lines[:8] == list(range(1, 9)) and st.comment_lines[-1] == st.n_lines


def test_three_space_tail_split(log):
    df = Z.split_tail(Z.scan_conn_labeled(log)).collect()
    assert df["label"].to_list() == ["Malicious", "Benign", "Malicious", "Benign"]
    assert df["detailed_label"].to_list() == ["C&C", None, "PartOfAHorizontalPortScan", None]
    # tunnel_parents keeps its own value (incl. a comma-set) and is NOT merged into the label
    assert df["tunnel_parents"].to_list() == ["(empty)", "-", "CAbc,CDef", "-"]


def test_record_mapping_and_pointers(log):
    st = Z.scan_structure(log)
    lf = Z.to_record_columns(Z.split_tail(Z.scan_conn_labeled(log)), scen_code=2, src_file="x", src_sha256="deadbeef")
    df = lf.collect()
    assert df["rec_id"].to_list() == [2 * Z.REC_ID_STRIDE + 9 + i for i in range(4)]  # line = row + 9
    assert df["src_locator"].to_list()[0] == "lines:9-9"
    r0 = df.row(0, named=True)
    assert (r0["proto"], r0["service"], r0["conn_state"], r0["obytes"]) == ("tcp", "irc", "S1", 6303)
    # '-' numeric fields (icmp row) -> 0; proto stays icmp; ts_res = 1e-6
    r2 = df.row(2, named=True)
    assert (r2["proto"], r2["obytes"], r2["rbytes"], r2["service"]) == ("icmp", 0, 0, "")
    assert df["ts_res"].unique().to_list() == [1e-6]


def test_window_selection_and_quarantine(log, tmp_path):
    st = Z.scan_structure(log)
    hist = Z.pass1_histogram(log, ["192.168.2.5"])
    assert int(hist["n"].sum()) == 4
    bins = Z.make_bins(hist, "s0", bin_s=7200)  # rows span ~1.7 h -> 2 bins
    chosen, info = Z.select_windows(bins.table, "malicious", Z.SelectionConfig(max_mal_windows=12, max_benign_windows=1))
    # bin 0 has malicious seed records; bin 1 is benign-only for the seed
    kinds = sorted(w.kind for w in chosen)
    assert "malicious" in kinds
    tmp = tmp_path / "p2.parquet"
    all_lf = Z.extract_windows(log, bins, chosen, 2, st.sha256, tmp)
    rp, lp = tmp_path / "rec.parquet", tmp_path / "lab.parquet"
    Z.write_window(all_lf, chosen[0].bin, rp, lp)
    rec = pl.read_parquet(rp)
    lab = pl.read_parquet(lp)
    # records carry NO label-like columns; labels table carries them, keyed by rec_id
    assert label_like_columns(rec.columns) == []
    assert list(rec.columns) == list(RECORD_SCHEMA)
    assert set(lab.columns) == {"rec_id", "label", "detailed_label"}
    assert rec.height == lab.height and rec["rec_id"].to_list() == lab["rec_id"].to_list()


def test_bad_structure_detected(tmp_path):
    # a stray '#'-comment in the data region must be flagged (would corrupt line numbering)
    bad = FIXTURE.replace(ROWS[1], "#oops mid-stream comment\n" + ROWS[1])
    p = tmp_path / "bad.log"
    p.write_text(bad)
    assert not Z.scan_structure(p).ok
