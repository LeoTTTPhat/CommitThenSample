"""Alg. 1 step 6 (DESIGN §3.2): INFRA, S = {seed} ∪ N1 ∪ N2, keep rule, U_max, prefilter equivalence."""
from __future__ import annotations

import random

import polars as pl

from certichain.units import scope as SC
from certichain.units.sessionize import UnitParams, event_units, prepare_records, records_frame, sessionize

SEED = "192.168.1.10"
C2 = "5.5.5.5"


def r(ts, src, dst, dport=80, **kw):
    return dict(ts=float(ts), src=src, dst=dst, dport=dport, **kw)


def scope_of(rows, seeds=(SEED,)):
    prep, _ = prepare_records(records_frame(rows))
    return SC.compute_scope(prep, list(seeds))


def test_infra_threshold_and_tokens():
    rows = [r(i, f"192.168.1.{20 + i}", "192.168.1.1", 53, proto="udp") for i in range(3)]       # 3 internal → INFRA
    rows += [r(10 + i, f"192.168.1.{20 + i}", "192.168.1.2", 1883) for i in range(2)]           # 2 → not INFRA
    rows += [r(20 + i, f"192.168.1.{20 + i}", "8.8.8.8", 53, proto="udp") for i in range(4)]     # external, 4 → INFRA
    rows += [r(30 + i, f"1.1.1.{i}", "192.168.1.3", 23) for i in range(5)]                     # external srcs don't count
    sc = scope_of(rows)
    assert set(sc.infra) == {"192.168.1.1", "8.8.8.8"}
    assert sc.infra["8.8.8.8"] == "INFRA0" and sc.infra["192.168.1.1"] == "INFRA1"  # by #contacts desc
    assert sc.infra_contacts == {"8.8.8.8": 4, "192.168.1.1": 3}


def test_seed_is_never_infra():
    rows = [r(i, f"192.168.1.{20 + i}", SEED, 80) for i in range(5)]
    sc = scope_of(rows)
    assert SEED not in sc.infra and sc.hosts[SEED] == 0


def test_n1_both_directions_internal_only_no_infra():
    rows = [r(0, SEED, "192.168.1.20"), r(1, "192.168.1.30", SEED), r(2, SEED, C2), r(3, "9.9.9.9", SEED)]
    rows += [r(10 + i, f"192.168.1.{40 + i}", "192.168.1.1", 53, proto="udp") for i in range(3)]
    rows += [r(20, SEED, "192.168.1.1", 53, proto="udp")]                                    # seed uses INFRA resolver
    rows += [r(21, SEED, "224.0.0.251", 5353, proto="udp")]                                  # multicast never in S
    sc = scope_of(rows)
    assert set(sc.n1) == {"192.168.1.20", "192.168.1.30"}
    assert "192.168.1.1" not in sc.hosts and C2 not in sc.hosts and "9.9.9.9" not in sc.hosts
    assert "224.0.0.251" not in sc.hosts


def test_n2_shared_external_expansion_and_infra_stop():
    rows = [r(0, SEED, C2, 6667), r(5, "192.168.1.50", C2, 6667)]                           # shares the C2 → N2
    rows += [r(1, SEED, "8.8.8.8", 53, proto="udp")]
    rows += [r(2 + i, f"192.168.1.{60 + i}", "8.8.8.8", 53, proto="udp") for i in range(3)]  # 8.8.8.8 is INFRA
    rows += [r(9, "192.168.1.70", "7.7.7.7", 443)]                                          # unrelated
    rows += [r(10, "1.2.3.4", C2, 6667)]                                                    # external host: never in S
    sc = scope_of(rows)
    assert sc.n2 == ("192.168.1.50",)
    assert not any(h.startswith("192.168.1.6") for h in sc.hosts) and "192.168.1.70" not in sc.hosts
    assert sc.hosts["192.168.1.50"] == 1


def test_n2_needs_seed_initiated_contact():
    rows = [r(0, C2, SEED, 4444), r(5, "192.168.1.50", C2, 6667)]                           # inbound only
    assert scope_of(rows).n2 == ()


def test_keep_rule_on_units():
    rows = [r(0, SEED, "192.168.1.20", 23, obytes=30, rbytes=40)]                           # .20 ∈ N1
    rows += [r(100, "192.168.1.20", "6.6.6.6", 80, obytes=300, rbytes=500_000)]             # src ∈ S → kept
    rows += [r(200, "192.168.33.12", "192.168.1.20", 23, obytes=30_000, rbytes=3_000)]       # dst ∈ S → kept
    rows += [r(300, "192.168.1.90", "192.168.1.91", 80)]                                    # neither → dropped
    rows += [r(400 + i, f"192.168.1.{100 + i}", "192.168.33.12", 23) for i in range(3)]      # loader is INFRA
    eu = event_units(records_frame(rows), [SEED])
    u = eu.units
    assert "192.168.33.12" in eu.scope.infra
    srcs = u["src"].to_list()
    assert "192.168.1.90" not in srcs and not any(s.startswith("192.168.1.10") and s != SEED for s in srcs)
    assert set(srcs) == {SEED, "192.168.1.20", "192.168.33.12"}
    itt = u.filter(pl.col("src") == "192.168.33.12")
    assert itt["infra"][0].to_list() == [eu.scope.infra["192.168.33.12"]]
    side = eu.side.join(u.select("unit_id", "src"), on="unit_id")
    assert side.filter(pl.col("src") == "192.168.33.12")["src_in_scope"].to_list() == [False]


def test_u_max_keeps_closest_by_hop_then_time():
    rows = [r(0, SEED, "192.168.1.20", 23, obytes=30, rbytes=40)]
    # seed activity around t=0; .20 makes 6 units at increasing distance from it
    rows += [r(1000 * (k + 1), "192.168.1.20", f"7.7.7.{k}", 443) for k in range(6)]
    eu = event_units(records_frame(rows), [SEED], UnitParams(U_max=4))
    assert eu.info["U_max_hit"] and eu.info["n_units_dropped_U_max"] == 3
    u = eu.units.join(eu.side, on="unit_id")
    assert u.filter(pl.col("hop") == 0).height == 1                    # the seed unit is always kept
    kept_t = sorted(u.filter(pl.col("hop") == 1)["ts"].to_list())
    assert kept_t == [1000.0, 2000.0, 3000.0]                          # the three closest in time


def _random_rows(seed=0, n=600):
    rnd = random.Random(seed)
    hosts = [SEED] + [f"192.168.1.{i}" for i in (20, 30, 40, 50, 60, 70)]
    ext = ["5.5.5.5", "6.6.6.6", "7.7.7.7", "8.8.8.8"] + [f"77.1.{i}.9" for i in range(30)]
    rows = []
    for _ in range(n):
        s = rnd.choice(hosts + ext[:3])
        d = rnd.choice(hosts + ext)
        if d == s:
            continue
        rows.append(r(rnd.randint(0, 7200), s, d, rnd.choice([23, 80, 443, 53, 6667, 1883]),
                      obytes=rnd.choice([0, 50, 900]), rbytes=rnd.choice([0, 60, 30_000]),
                      conn_state=rnd.choice(["S0", "SF"]), sport=rnd.randint(1024, 60000)))
    return rows


def test_record_level_prefilter_is_exact():
    for seed in range(4):
        rows = _random_rows(seed)
        eu = event_units(records_frame(rows), [SEED], UnitParams(U_max=10**6))
        full = event_units(records_frame(rows), [SEED], UnitParams(U_max=10**6), use_scope=False)
        # unit-level rule applied to whole-window units, using the same S
        prep, _ = prepare_records(records_frame(rows))
        sc = SC.compute_scope(prep, [SEED])
        res = sessionize(SC.annotate_records(prep, sc), UnitParams(), infra_tokens=sc.infra)
        kept = res.units.filter(pl.col("touches_S"))
        a = eu.units.drop("unit_id").sort(["ts", "src", "svc_class", "type"])
        b = kept.select(a.columns).sort(["ts", "src", "svc_class", "type"])
        assert a.height == b.height
        assert a.select(pl.exclude("rec_ids")).equals(b.select(pl.exclude("rec_ids")))
        assert full.units.height >= eu.units.height


def test_scope_json_roundtrip():
    rows = [r(0, SEED, C2, 6667), r(5, "192.168.1.50", C2, 6667)]
    rows += [r(10 + i, f"192.168.1.{60 + i}", "192.168.1.1", 53, proto="udp") for i in range(3)]
    sc = scope_of(rows)
    sc2 = SC.ScopeResult.from_json(sc.to_json())
    assert sc2.infra == sc.infra and sc2.hosts == sc.hosts and sc2.seeds == sc.seeds
