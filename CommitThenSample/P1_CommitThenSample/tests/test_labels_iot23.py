"""IoT-23 unit ↔ conn label match (labels/iot23.py): parsing quirk, orientation, overlap slack, ties,
ICMP, unknowns and incidents."""
import polars as pl

from ctsample.labels import iot23

HDR = ("#separator \\x09\n#set_separator\t,\n#empty_field\t(empty)\n#unset_field\t-\n#path\tconn\n"
       "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\tduration\torig_bytes\t"
       "resp_bytes\tconn_state\tlocal_orig\tlocal_resp\tmissed_bytes\thistory\torig_pkts\torig_ip_bytes\t"
       "resp_pkts\tresp_ip_bytes\ttunnel_parents   label   detailed-label\n")


def row(ts, uid, oh, op, rh, rp, proto, dur, ob, rb, label, dl):
    return (f"{ts}\t{uid}\t{oh}\t{op}\t{rh}\t{rp}\t{proto}\t-\t{dur}\t{ob}\t{rb}\tSF\t-\t-\t0\tShADadFf\t5\t300\t5\t"
            f"300\t-   {label}   {dl}\n")


def write_log(tmp_path):
    lines = [
        row(1000.0, "C1", "192.168.1.5", 40000, "1.2.3.4", 80, "tcp", 10.0, 100, 2000, "Malicious", "C&C-FileDownload"),
        row(1100.0, "C2", "1.2.3.4", 5555, "192.168.1.5", 23, "tcp", 2.0, 10, 10, "Malicious", "Attack"),
        row(1200.0, "C3", "192.168.1.5", 123, "9.9.9.9", 123, "udp", "-", "-", "-", "Benign", "-"),
        row(1300.0, "C4", "192.168.1.5", 50000, "5.5.5.5", 443, "tcp", 100.0, 50, 50, "Malicious", "Okiru"),
        row(1310.0, "C5", "192.168.1.5", 50000, "5.5.5.5", 443, "tcp", 5.0, 50, 50, "Benign", "-"),
        row(1400.0, "C6", "192.168.1.5", 8, "8.8.8.8", 0, "icmp", 1.0, 64, 64, "benign", "-"),
        row(3000.0, "C7", "192.168.1.5", 40100, "1.2.3.4", 80, "tcp", 1.0, 100, 2000, "Malicious", "C&C-FileDownload"),
    ]
    p = tmp_path / "conn.log.labeled"
    p.write_text(HDR + "".join(lines) + "#close\t2019-01-01-00-00-00\n")
    return p


def units_df():
    s = 1_000_000
    rows = [  # gid, proto, ip_a, port_a, ip_b, port_b, first, last, pay, device
        (1, 6, "192.168.1.5", 40000, "1.2.3.4", 80, 1001 * s, 1005 * s, True, 0),        # fwd, inside C1
        (2, 6, "192.168.1.5", 23, "1.2.3.4", 5555, 1100 * s, 1101 * s, True, 0),          # reversed vs C2
        (3, 17, "192.168.1.5", 123, "9.9.9.9", 123, 1200 * s + 500_000, 1200 * s + 600_000, True, 0),  # dur '-'
        (4, 6, "192.168.1.5", 40000, "1.2.3.4", 80, 1010 * s + 900_000, 1012 * s, True, 0),  # within +1 s slack
        (5, 6, "192.168.1.5", 40000, "1.2.3.4", 80, 1013 * s, 1014 * s, True, 0),         # beyond slack: unknown
        (6, 6, "192.168.1.5", 50000, "5.5.5.5", 443, 1311 * s, 1312 * s, True, 0),        # C4 and C5 overlap: C5
        (7, 1, "8.8.8.8", 0, "192.168.1.5", 0, 1400 * s + 10, 1400 * s + 20, False, 0),    # ICMP echo reply
        (8, 0, "02:00:00:00:00:01", 0x0806, "", 0, 1500 * s, 1500 * s, False, 0),          # L2: unknown
        (9, 6, "192.168.1.5", 40100, "1.2.3.4", 80, 3000 * s, 3000 * s + 10, True, 0),     # C7, gap > 600 s
    ]
    cols = ["unit_gid", "proto", "ip_a", "port_a", "ip_b", "port_b", "first_ts_us", "last_ts_us", "pay", "device_id"]
    return pl.DataFrame(rows, schema=cols, orient="row").with_columns(
        pl.col("unit_gid").cast(pl.Int64), pl.col("proto").cast(pl.Int16), pl.col("port_a").cast(pl.Int32),
        pl.col("port_b").cast(pl.Int32), pl.col("device_id").cast(pl.Int32))


def test_load_connlog_quirk(tmp_path):
    c = iot23.load_connlog(write_log(tmp_path))
    assert c.height == 7
    assert c["label"].to_list()[:3] == ["malicious", "malicious", "benign"] and c["label"][5] == "benign"
    assert c["detailed_label"][0] == "C&C-FileDownload" and c["detailed_label"][2] == "-"
    assert c["duration"][2] == 0.0 and c["orig_bytes"][2] == 0 and c["resp_bytes"][0] == 2000
    w = iot23.load_connlog(write_log(tmp_path), 1150.0, 1350.0)
    assert w["uid"].to_list() == ["C3", "C4", "C5"]


def test_match_rules(tmp_path):
    c = iot23.load_connlog(write_log(tmp_path))
    lab, qa = iot23.label_source(units_df(), c, "S")
    d = {r["unit_gid"]: r for r in lab.to_dicts()}
    assert d[1]["conn_uid"] == "C1" and d[1]["label_source"] == "conn_fwd" and d[1]["label"] == "Malicious"
    assert d[1]["family"] == "C&C" and d[1]["is_malicious"]
    assert d[2]["conn_uid"] == "C2" and d[2]["label_source"] == "conn_rev" and d[2]["family"] == "Attack"
    assert d[3]["conn_uid"] == "C3" and d[3]["label"] == "Benign" and d[3]["detailed_label"] == "-"
    assert d[4]["conn_uid"] == "C1"                               # 0.9 s after the conn end: inside the slack
    assert d[5]["label"] == "unknown" and d[5]["label_source"] == "none" and d[5]["conn_uid"] is None
    assert d[6]["conn_uid"] == "C5"                               # tie → smallest start difference
    assert d[7]["conn_uid"] == "C6" and d[7]["label_source"] == "conn_icmp"
    assert d[8]["label"] == "unknown"
    assert qa["match_rate_payload"] == 6 / 7 and qa["payload_units"] == 7
    inc = {g: d[g]["incident_id"] for g in (1, 2, 4, 9)}
    assert inc[1] == inc[4] and inc[1] != inc[9] and inc[2] is not None and inc[2] != inc[1]
    assert d[3]["incident_id"] is None and d[6]["incident_id"] is None
    assert qa["n_incidents"] == 3


def test_no_conn_log_defaults():
    lab, qa = iot23.label_source(units_df(), None, "H", default_benign=True)
    assert set(lab["label"]) == {"Benign"} and set(lab["label_source"]) == {"honeypot_default"}
    lab, qa = iot23.label_source(units_df(), None, "X", default_benign=False)
    assert set(lab["label"]) == {"unknown"} and qa["match_rate_payload"] == 0.0


def test_l_family():
    f = iot23.l_family
    assert [f(x) for x in ("C&C", "C&C-HeartBeat-FileDownload", "FileDownload", "DDoS", "Attack", "Okiru-Attack",
                           "PartOfAHorizontalPortScan", "-", None)] == \
        ["C&C", "C&C", "FileDownload", "Attack", "Attack", "Okiru", "PortScan", None, None]
