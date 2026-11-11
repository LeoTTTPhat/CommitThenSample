"""Role tokens: numbering by first appearance, label-free port-profile roles, determinism."""
from __future__ import annotations

import random

from certichain.features.roles import build_roles, role_label


def unit(uid, src, dsts, svc, dport, dc, ff=0.0, rb=1000, fanout=False, infra=(), ts=None):
    return {"unit_id": uid, "ts": float(ts if ts is not None else uid * 10), "src": src, "dsts": list(dsts),
            "n_dst": len(dsts), "fanout": fanout, "svc_class": svc, "dir_class": dc, "proto": "tcp",
            "dports": [dport], "fail_frac": ff, "rbytes": rb, "infra": list(infra)}


UNITS = [
    unit(0, "203.0.113.5", ["192.168.1.30"], "telnet", 23, "ext->int", ff=0.89),          # brute force
    unit(1, "192.168.1.40", ["192.168.1.30"], "rtsp", 554, "int->int"),                  # camera stream
    unit(2, "198.51.100.7", ["192.168.1.30"], "telnet", 23, "ext->int", infra=["198.51.100.7"]),  # loader (infra)
    unit(3, "192.168.1.30", ["203.0.113.9"], "http", 80, "int->ext"),                    # download from X2
    unit(4, "192.168.1.30", [f"10.0.0.{i}" for i in range(20)], "telnet", 23, "int->int", ff=0.99, fanout=True),
    unit(5, "192.168.1.50", ["192.168.1.60"], "mqtt", 1883, "int->int"),
    unit(6, "192.168.1.51", ["192.168.1.60"], "mqtt", 1883, "int->int", ff=1.0, rb=0),
]


def test_numbering_by_first_appearance_and_kinds():
    rm = build_roles(UNITS)
    assert rm.name("203.0.113.5") == "X1"
    assert rm.name("192.168.1.30") == "H1"
    assert rm.name("192.168.1.40") == "H2"
    assert rm.name("198.51.100.7") == "INFRA1" and rm.token("198.51.100.7") == "INFRA1"
    assert rm.name("203.0.113.9") == "X2" and rm.token("203.0.113.9") == "X2[ext]"
    assert rm.name("10.0.0.0") == "H3" and rm.name("10.0.0.19") == "H22"     # fan-out dsts in list order
    assert rm.name("192.168.1.50") == "H23" and rm.name("192.168.1.60") == "H24"


def test_roles_from_served_services():
    rm = build_roles(UNITS)
    assert rm.token("192.168.1.30") == "H1[camera,rtsp,telnet]"      # serves rtsp + telnet (established part)
    assert rm.token("192.168.1.40") == "H2[client]"
    assert rm.token("192.168.1.60") == "H24[mqtt-broker,mqtt]"
    assert rm.token("10.0.0.3") == "H6[client]"                       # fan-out targets get no role
    assert rm.token("203.0.113.5") == "X1[ext]"


def test_failed_units_do_not_create_roles():
    rm = build_roles([unit(0, "192.168.1.2", ["192.168.1.3"], "telnet", 23, "int->int", ff=1.0, rb=0)])
    assert rm.token("192.168.1.3") == "H2[client]"


def test_role_label_rules():
    assert role_label(["telnet", "rtsp", "http"]) == "camera,rtsp,web"
    assert role_label(["dns"]) == "dns"
    assert role_label(["modbus"]) == "plc,modbus"
    assert role_label([]) == "client"
    assert role_label(["rtsp"], kind="external") == "ext"


def test_deterministic_and_order_independent_input():
    a = build_roles(UNITS).tokens
    shuffled = UNITS[:]
    random.Random(7).shuffle(shuffled)
    assert build_roles(shuffled).tokens == a                      # sorted by unit_id internally
    assert build_roles(UNITS).tokens == a


def test_polars_input_and_internal_override():
    import polars as pl
    rows = [dict(u, dsts=u["dsts"], infra=u["infra"]) for u in UNITS]
    df = pl.DataFrame(rows)
    assert build_roles(df).tokens == build_roles(UNITS).tokens
    rm = build_roles(UNITS, is_internal=lambda h: h.startswith("192.168."))
    assert rm.kind["10.0.0.1"] == "external" and rm.kind["192.168.1.30"] == "internal"
