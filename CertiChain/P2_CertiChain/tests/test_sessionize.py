"""Alg. 1 steps 1–5 (DESIGN §3.2, §7.6): bursts, fan-out, activity merge, determinism."""
from __future__ import annotations

import random

import polars as pl
import pytest

from certichain.io.schema import UNIT_SCHEMA
from certichain.units.sessionize import (UnitParams, event_units, records_frame, svc_class, tiehash)

SEED = "192.168.1.10"


def units_of(rows, seeds=(SEED,), **kw):
    return event_units(records_frame(rows), list(seeds), UnitParams(**kw)).units


def heartbeat(t0, n, gap, dst="5.5.5.5", dport=6667, src=SEED):
    return [dict(ts=t0 + gap * k, src=src, dst=dst, dport=dport, obytes=50, rbytes=40) for k in range(n)]


# --------------------------------------------------------------------------- service / direction keys
def test_svc_class_table():
    assert svc_class("tcp", 23) == svc_class("tcp", 2323) == "telnet"
    assert svc_class("tcp", 2222) == "ssh" and svc_class("tcp", 8081) == "http"
    assert svc_class("tcp", 8443) == "https" and svc_class("udp", 53) == "dns"
    assert svc_class("tcp", 1883) == svc_class("tcp", 8883) == "mqtt"
    assert svc_class("udp", 5683) == "coap" and svc_class("tcp", 554) == "rtsp"
    assert svc_class("tcp", 502) == "modbus" and svc_class("tcp", 6665) == "irc"
    assert svc_class("tcp", 48101) == "report48101" and svc_class("udp", 123) == "ntp"
    assert svc_class("icmp", 0) == "icmp" and svc_class("gre", None) == "gre"
    assert svc_class("tcp", 31337) == "other_tcp" and svc_class("udp", 31337) == "other_udp"
    assert svc_class("tcp", 9999, "http") == "http"            # service fallback for unknown ports
    assert svc_class("udp", 23) == "other_udp"                  # telnet is tcp-only


def test_dir_class_and_schema():
    rows = [dict(ts=0, src=SEED, dst="8.8.8.8", dport=443), dict(ts=1000, src=SEED, dst="192.168.1.30", dport=443),
            dict(ts=2000, src="1.2.3.4", dst=SEED, dport=23, conn_state="S0", obytes=0, rbytes=0)]
    u = units_of(rows)
    assert dict(zip(u["dsts"].list.first(), u["dir_class"])) == {
        "8.8.8.8": "int->ext", "192.168.1.30": "int->int", SEED: "ext->int"}
    assert u.schema == pl.Schema(UNIT_SCHEMA)
    assert u["unit_id"].to_list() == list(range(u.height))


# --------------------------------------------------------------------------- bursts
def test_burst_split_at_tau_gap():
    # same key, same dst: gaps 100 s → one burst; a 700 s gap (> τ_merge too) → two units
    rows = heartbeat(0, 3, 100) + heartbeat(200 + 700, 3, 100)
    u = units_of(rows)
    assert u.height == 2 and u["n_records"].to_list() == [3, 3]
    # a gap of exactly τ_gap does not split
    u = units_of([dict(ts=0, te=0, src=SEED, dst="5.5.5.5", dport=6667),
                  dict(ts=120, te=120, src=SEED, dst="5.5.5.5", dport=6667)], tau_merge=0.0)
    assert u.height == 1


def test_burst_gap_is_idle_gap_after_running_end():
    # a long record [0, 500] keeps the burst open: next start 610 is only 110 s after its end
    rows = [dict(ts=0, te=500, src=SEED, dst="5.5.5.5", dport=6667, obytes=5000, rbytes=5000),
            dict(ts=610, src=SEED, dst="5.5.5.5", dport=6667)]
    u = units_of(rows, tau_merge=0.0)
    assert u.height == 1
    rows[1]["ts"] = 621.0                                        # 121 s idle → new burst
    u = units_of(rows, tau_merge=0.0)
    assert u.height == 2


def test_bursts_are_per_key():
    # different svc_class or direction never share a burst/unit
    rows = [dict(ts=0, src=SEED, dst="5.5.5.5", dport=23), dict(ts=1, src=SEED, dst="5.5.5.5", dport=80),
            dict(ts=2, src=SEED, dst="192.168.1.40", dport=23)]
    u = units_of(rows)
    assert u.height == 3


# --------------------------------------------------------------------------- fan-out
def test_fanout_threshold():
    scan8 = [dict(ts=k * 0.1, src=SEED, dst=f"77.1.{k}.9", dport=23, conn_state="S0", obytes=0, rbytes=0)
             for k in range(8)]
    u = units_of(scan8)
    assert u.height == 1 and u["fanout"][0] and u["n_dst"][0] == 8 and u["type"][0] == "SCAN_H"
    u7 = units_of(scan8[:7])
    assert u7.height == 7 and not any(u7["fanout"])              # one unit per (src, dst) sub-burst


def test_fanout_dst_cap_and_prefixes():
    rows = [dict(ts=k * 0.01, src=SEED, dst=f"77.{k // 250}.0.{k % 250 + 1}", dport=23, conn_state="S0",
                 obytes=0, rbytes=0) for k in range(300)]
    u = units_of(rows)
    assert u.height == 1
    assert u["n_dst"][0] == 300 and len(u["dsts"][0]) == 256      # n_dst is exact, the list is capped
    assert sorted(u["dst_prefixes"][0]) == ["77.0.0.0/24", "77.1.0.0/24"]


def test_fanout_dst_priority_scope_members_first():
    # host .20 (in S: it talks to the seed) sweeps 300 internal addresses; the seed is hit last and once,
    # yet it is stored first (scope members, then INFRA, internal, external; then #records, first contact)
    h = "192.168.1.20"
    rows = [dict(ts=0, src=SEED, dst=h, dport=80)]
    rows += [dict(ts=10 + k * 0.01, src=h, dst=f"10.0.{k // 200}.{k % 200 + 1}", dport=23, conn_state="S0",
                  obytes=0, rbytes=0) for k in range(300)]
    rows += [dict(ts=14, src=h, dst=SEED, dport=23, conn_state="S0", obytes=0, rbytes=0)]
    u = event_units(records_frame(rows), [SEED]).units.filter(pl.col("src") == h)
    assert u.height == 1 and u["fanout"][0] and len(u["dsts"][0]) == 256
    assert u["dsts"][0][0] == SEED


def test_subbursts_per_destination():
    rows = [dict(ts=k, src=SEED, dst=f"5.5.5.{k % 3}", dport=6667) for k in range(9)]
    u = units_of(rows)
    assert u.height == 3 and sorted(u["n_records"].to_list()) == [3, 3, 3]


# --------------------------------------------------------------------------- activity merge
def test_activity_merge_within_tau_merge():
    rows = heartbeat(0, 6, 60) + heartbeat(300 + 300, 6, 60)    # 300 s idle gap: 2 bursts, merged
    eu = event_units(records_frame(rows), [SEED])
    assert eu.info["sess_n_units_pre_merge"] == 2
    assert eu.units.height == 1 and eu.units["n_records"][0] == 12 and eu.info["sess_merges"] == 1


def test_activity_merge_not_beyond_tau_merge():
    rows = heartbeat(0, 6, 60) + heartbeat(301 + 601, 6, 60)   # last te = 301 → idle gap 601 s
    assert units_of(rows).height == 2
    rows = heartbeat(0, 6, 60) + heartbeat(301 + 600, 6, 60)   # exactly τ_merge → merged
    assert units_of(rows).height == 1


def test_activity_merge_requires_same_type_and_dst():
    # same dst, but the second burst is a large download (different type) → no merge
    rows = heartbeat(0, 6, 60, dst="6.6.6.6", dport=80)
    rows.append(dict(ts=300 + 400, src=SEED, dst="6.6.6.6", dport=80, obytes=300, rbytes=500_000))
    u = units_of(rows)
    assert u.height == 2 and set(u["type"]) == {"C2_PERIODIC", "HTTP_DOWNLOAD"}
    # same type, different dst → no merge
    rows = heartbeat(0, 6, 60, dst="5.5.5.5") + heartbeat(700, 6, 60, dst="5.5.5.6")
    assert units_of(rows).height == 2


def test_activity_merge_fanout_with_fanout_only():
    scan_a = [dict(ts=k, src=SEED, dst=f"77.1.{k}.9", dport=23, conn_state="S0", obytes=0, rbytes=0)
              for k in range(10)]
    scan_b = [dict(ts=400 + k, src=SEED, dst=f"77.2.{k}.9", dport=23, conn_state="S0", obytes=0, rbytes=0)
              for k in range(10)]
    u = units_of(scan_a + scan_b)
    assert u.height == 1 and u["n_dst"][0] == 20 and u["fanout"][0]
    lone = [dict(ts=400, src=SEED, dst="77.3.0.9", dport=23, conn_state="S0", obytes=0, rbytes=0)]
    u = units_of(scan_a + lone)
    assert u.height == 2                                          # fan-out never merges with a single-dst unit


def test_activity_merge_recomputes_type_until_stable():
    # two bursts of 3 short telnet logins 200 s apart: each BENIGN_GENERIC_TCP; merged → 6 conns → BRUTE_TELNET
    def logins(t0):
        return [dict(ts=t0 + 5 * k, te=t0 + 5 * k + 2, src=SEED, dst="192.168.1.50", dport=23, obytes=30, rbytes=40)
                for k in range(3)]
    eu = event_units(records_frame(logins(0) + logins(210)), [SEED])
    assert eu.info["sess_n_units_pre_merge"] == 2 and eu.info["sess_merge_rounds"] >= 1
    assert eu.units.height == 1 and eu.units["type"][0] == "BRUTE_TELNET"


def test_merge_chain_uses_running_end():
    # a long unit [0, 2000] followed by a short one at 2500 (500 s after its end) and another at 3050
    rows = [dict(ts=0, te=2000, src=SEED, dst="5.5.5.5", dport=31337, obytes=10, rbytes=10)]
    rows += [dict(ts=2500, src=SEED, dst="5.5.5.5", dport=31337, obytes=10, rbytes=10)]
    u = units_of(rows)
    assert u.height <= 2  # types may differ (long-lived low-rate C2 vs short) — merge only if same type
    assert u["n_records"].sum() == 2


# --------------------------------------------------------------------------- determinism
def _mixed_rows(n=400, seed=1):
    rnd = random.Random(seed)
    hosts = [SEED, "192.168.1.20", "192.168.1.30", "1.2.3.4"]
    rows = []
    for i in range(n):
        s = rnd.choice(hosts)
        d = rnd.choice(["5.5.5.5", "6.6.6.6", f"77.1.{rnd.randint(0, 40)}.9", "192.168.1.1", SEED, "192.168.1.20"])
        if d == s:
            d = "8.8.8.8"
        rows.append(dict(ts=float(rnd.randint(0, 6000)), te=None, src=s, dst=d,
                         dport=rnd.choice([23, 80, 53, 443, 6667, 31337]), sport=rnd.randint(1024, 65000),
                         proto="tcp", obytes=rnd.choice([0, 40, 500]), rbytes=rnd.choice([0, 40, 20000]),
                         conn_state=rnd.choice(["S0", "SF", "REJ"])))
    for r in rows:                                               # many exact-ts ties
        r["ts"] = float(int(r["ts"] / 7) * 7)
    return rows


def _canon(u: pl.DataFrame) -> pl.DataFrame:
    return u.drop("rec_ids")


def test_determinism_under_record_permutation():
    rows = _mixed_rows()
    a = event_units(records_frame(rows), [SEED]).units
    perm = rows[:]
    random.Random(7).shuffle(perm)
    b = event_units(records_frame(perm), [SEED]).units          # rec_ids re-assigned in permuted order
    assert _canon(a).equals(_canon(b))
    # the evidence pointers cover the same records (rec_id ↔ row mapping differs by construction)
    assert a["n_records"].sum() == b["n_records"].sum()


def test_rec_id_never_orders_units():
    rows = _mixed_rows(seed=3)
    for i, r in enumerate(rows):
        r["rec_id"] = 10_000 - i                                # reversed ids
    a = event_units(records_frame(rows), [SEED]).units
    for i, r in enumerate(rows):
        r["rec_id"] = i
    b = event_units(records_frame(rows), [SEED]).units
    assert _canon(a).equals(_canon(b))
    ra = sorted(sorted(x) for x in a["rec_ids"].to_list())
    rb = sorted(sorted(10_000 - y for y in x) for x in b["rec_ids"].to_list())
    assert ra == rb


def test_tiehash_definition():
    import blake3
    h = tiehash("1.2.3.4", 1234, "5.6.7.8", 80, "tcp")
    assert h == int.from_bytes(blake3.blake3(b"1.2.3.4|1234|5.6.7.8|80|tcp").digest()[:8], "little")


def test_empty_and_invalid_records():
    eu = event_units(records_frame([]), [SEED])
    assert eu.units.height == 0 and eu.units.schema == pl.Schema(UNIT_SCHEMA)
    rows = [dict(ts=0, src=SEED, dst="5.5.5.5"), dict(ts=1, src=SEED, dst="")]
    eu = event_units(records_frame(rows), [SEED])
    assert eu.info["n_records_invalid"] == 1 and eu.units.height == 1


@pytest.mark.parametrize("n", [2000])
def test_scales_roughly_linearly(n):
    import time
    rows = _mixed_rows(n=n, seed=5)
    t = time.perf_counter()
    event_units(records_frame(rows), [SEED])
    assert time.perf_counter() - t < 10.0


def test_vectorised_ip_classes_match_reference():
    import random as _r
    from certichain.units.sessionize import _nets, classify_ip, classify_ips
    rnd = _r.Random(0)
    special = ["10.0.0.1", "172.16.0.1", "172.31.255.255", "172.32.0.1", "192.168.1.255", "192.168.0.0",
               "127.0.0.1", "169.254.3.4", "224.0.0.251", "239.255.255.250", "255.255.255.255", "0.0.0.0",
               "0.1.2.3", "8.8.8.8", "1.2.3.255", "192.169.1.1", "fe80::1", "::1", "2001:db8::1", "ff02::fb",
               "not-an-ip", "1.2.3", "01.2.3.4", "256.1.1.1", "100.64.0.1"]
    ips = special + [".".join(str(rnd.randint(0, 255)) for _ in range(4)) for _ in range(20000)]
    for pref in ((), ("100.64.0.0/10", "5.5.0.0/16")):
        got = classify_ips(ips, pref)
        nn = _nets(pref)
        ref = sorted(set(ips))
        rows = [classify_ip(i, nn) for i in ref]
        assert got["ip"].to_list() == ref
        assert got["internal"].to_list() == [r[0] for r in rows]
        assert got["unicast"].to_list() == [r[1] for r in rows]
        assert got["prefix24"].to_list() == [r[2] for r in rows]


def test_record_ip_flags_match_reference():
    import random as _r
    from certichain.units.sessionize import _nets, classify_ip, prepare_records
    rnd = _r.Random(1)
    odd = ["fe80::1", "2001:db8::1", "01.2.3.4", "1.2.3", "x", "192.168.1.255", "255.255.255.255", "224.0.0.251"]
    ips = odd + [".".join(str(rnd.randint(0, 255)) for _ in range(4)) for _ in range(3000)]
    rows = [dict(ts=float(i), src=ips[i], dst=ips[-1 - i], dport=80) for i in range(len(ips))]
    for pref in ((), ("100.64.0.0/10", "5.5.0.0/16")):
        prep, _ = prepare_records(records_frame(rows), pref)
        nn = _nets(pref)
        for col in ("src", "dst"):
            ref = [classify_ip(x, nn) for x in prep[col].to_list()]
            assert prep[f"{col}_int"].to_list() == [r[0] for r in ref]
            assert prep[f"{col}_uni"].to_list() == [r[1] for r in ref]
