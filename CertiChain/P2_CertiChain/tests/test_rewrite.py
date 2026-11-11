"""ChainForge rewriting (DESIGN §5.1.2 IP plan, §5.4.4 artifact controls): role consistency,
determinism, structure preservation, no label leakage into records, 1-s rounding."""
from __future__ import annotations

import ipaddress

import numpy as np
import polars as pl
import pytest

from certichain.chainforge.rewrite import (EXT_NET, IPPlan, apply_ip_map, is_local, round_times, shift_times,
                                           visible_mask)
from certichain.io.labels import label_like_columns
from certichain.io.schema import RECORD_SCHEMA, conform


def _plan(seed=0):
    return IPPlan(np.random.default_rng(seed))


def _recs(rows):
    base = {"ts_res": 1.0, "sport": 40000, "dport": 80, "proto": "tcp", "service": "http", "obytes": 10, "rbytes": 20,
            "opkts": 1, "rpkts": 2, "conn_state": "SF", "history": "", "bulk": False, "n_merged": 1,
            "src_file": "f.csv", "src_sha256": "x", "src_locator": "lines:1-1"}
    out = []
    for i, r in enumerate(rows):
        d = dict(base)
        d.update(r)
        d.setdefault("rec_id", i)
        d.setdefault("te", d["ts"] + 0.25)
        out.append(d)
    return conform(pl.DataFrame(out), RECORD_SCHEMA)


def test_base_map_plan_and_determinism():
    p1, p2, p3 = _plan(1), _plan(1), _plan(2)
    ips = ["192.168.1.1", "192.168.1.152", "192.168.1.190", "192.168.1.255", "172.16.4.2", "8.8.8.8",
           "224.0.0.251", "255.255.255.255"]
    m1 = {i: p1.map_base(i) for i in ips}
    m2 = {i: p2.map_base(i) for i in ips}
    assert m1 == m2                                   # same seed → same plan
    assert {i: p3.map_base(i) for i in ips} != m1     # different seed → different plan
    n = p1.net_main
    assert m1["192.168.1.1"] == f"10.0.{n}.1" and m1["192.168.1.255"] == f"10.0.{n}.255"
    assert m1["192.168.1.152"].startswith(f"10.0.{n}.") and m1["192.168.1.190"].startswith(f"10.0.{n}.")
    assert m1["172.16.4.2"].startswith(f"10.0.{p1.net_other}.")
    assert ipaddress.ip_address(m1["8.8.8.8"]) in EXT_NET
    assert m1["224.0.0.251"] == "224.0.0.251" and m1["255.255.255.255"] == "255.255.255.255"
    assert len(set(m1.values())) == len(m1)           # injective
    r = {p1.new_role_ip() for _ in range(50)}
    assert len(r) == 50 and all(a.startswith(("198.51.100.", "203.0.113.")) for a in r)
    assert not r & set(m1.values())


def test_episode_map_bindings_and_swap():
    p = _plan(3)
    x_ton = "192.168.1.152"                     # background device chosen as the victim X
    x_ip = p.device_ip(x_ton)
    c2 = p.new_role_ip()
    # TON episode: attacker .33, primary victim .193, secondary victims include X's testbed IP .152
    ips = ["192.168.1.33", "192.168.1.193", "192.168.1.152", "192.168.1.190", "52.1.2.3"]
    mu = p.episode_map(ips, {"192.168.1.33": c2, "192.168.1.193": x_ip})
    assert mu["192.168.1.33"] == c2 and mu["192.168.1.193"] == x_ip
    assert len(set(mu.values())) == len(mu)      # distinct hosts stay distinct
    assert mu["192.168.1.152"] == p.map_base("192.168.1.193")   # swapped onto the primary's old address
    assert mu["192.168.1.190"] == p.map_base("192.168.1.190")   # untouched hosts keep the base map
    assert mu["52.1.2.3"] == p.map_base("52.1.2.3")
    # pivot binding: attacker slot → the internal device X itself
    y_ip = p.device_ip("192.168.1.195")
    mu2 = p.episode_map(["192.168.1.31", "192.168.1.195", "192.168.1.152"],
                        {"192.168.1.31": x_ip, "192.168.1.152": y_ip})
    assert mu2["192.168.1.31"] == x_ip and mu2["192.168.1.152"] == y_ip
    assert len(set(mu2.values())) == 3
    # deterministic
    q = _plan(3)
    q.device_ip(x_ton)
    c2q = q.new_role_ip()
    assert q.episode_map(ips, {"192.168.1.33": c2q, "192.168.1.193": q.device_ip(x_ton)}) == mu


def test_rewrite_preserves_structure_and_is_label_free():
    p = _plan(5)
    df = _recs([{"ts": 100.0, "src": "192.168.1.33", "dst": "192.168.1.193", "dport": 445, "obytes": 7},
                {"ts": 101.0, "src": "192.168.1.193", "dst": "192.168.1.33", "dport": 4444, "conn_state": "REJ"},
                {"ts": 105.0, "src": "192.168.1.33", "dst": "52.1.2.3", "dport": 123, "proto": "udp"}])
    c2, x_ip = p.new_role_ip(), p.device_ip("192.168.1.6")
    ips = sorted(set(df["src"]) | set(df["dst"]))
    mu = p.episode_map(ips, {"192.168.1.33": c2, "192.168.1.193": x_ip})
    out = shift_times(apply_ip_map(df, mu), 1000.0)
    keep = [c for c in df.columns if c not in ("src", "dst", "ts", "te")]
    assert out.select(keep).equals(df.select(keep))          # bytes, packets, states, content untouched
    assert (out["ts"] - df["ts"]).to_list() == [1000.0] * 3
    assert out["src"].to_list()[:2] == [c2, x_ip] and out["dst"].to_list()[:2] == [x_ip, c2]
    vis = visible_mask(out)
    assert vis.to_list() == [True, True, False]               # C2 → external third party: not observable
    assert not label_like_columns(out.columns)
    assert set(out.columns) == set(RECORD_SCHEMA)


def test_round_times_to_one_second():
    df = _recs([{"ts": 10.4, "te": 10.6, "src": "10.0.0.2", "dst": "10.0.0.3"},
                {"ts": 11.0, "te": 12.49, "src": "10.0.0.2", "dst": "10.0.0.3"},
                {"ts": 12.5, "te": 12.5, "src": "10.0.0.2", "dst": "10.0.0.3"}])
    r = round_times(df, 1.0)
    assert r["ts"].to_list() == [10.0, 11.0, 13.0]
    assert r["te"].to_list() == [11.0, 12.0, 13.0]
    assert (r["te"] >= r["ts"]).all() and set(r["ts_res"].to_list()) == {1.0}


@pytest.mark.parametrize("ip,loc", [("10.0.3.4", True), ("192.168.1.9", True), ("224.0.0.251", True),
                                    ("255.255.255.255", True), ("198.51.100.7", False), ("203.0.113.9", False),
                                    ("198.18.3.4", False), ("8.8.8.8", False)])
def test_is_local(ip, loc):
    assert is_local(ip) is loc


def test_subnet_mode_plan_for_gotham():
    p, q = IPPlan(np.random.default_rng(7), mode="subnets"), IPPlan(np.random.default_rng(7), mode="subnets")
    ips = ["192.168.20.10", "192.168.20.19", "192.168.18.10", "192.168.20.1", "192.168.20.255", "192.168.33.10",
           "1.2.3.4", "224.0.0.251"]
    m = {i: p.map_base(i) for i in ips}
    assert m == {i: q.map_base(i) for i in ips}                       # deterministic
    n20 = m["192.168.20.10"].rsplit(".", 1)[0]
    assert m["192.168.20.19"].startswith(n20 + ".") and m["192.168.20.1"] == n20 + ".1"
    assert m["192.168.20.255"] == n20 + ".255"
    assert not m["192.168.18.10"].startswith(n20 + ".")              # each source /24 → its own /24
    assert ipaddress.ip_address(m["1.2.3.4"]) in EXT_NET and m["224.0.0.251"] == "224.0.0.251"
    roles = [p.fresh_in_subnet("192.168.33.10") for _ in range(4)]
    n33 = m["192.168.33.10"].rsplit(".", 1)[0]
    assert all(r.startswith(n33 + ".") for r in roles)               # infra roles stay in the infra subnet
    assert len(set(roles) | set(m.values())) == len(roles) + len(m)   # injective incl. role IPs
    assert all(is_local(r) for r in roles)                            # internal, as in the Gotham captures
