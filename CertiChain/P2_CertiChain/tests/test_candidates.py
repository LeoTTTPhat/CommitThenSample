"""Alg. 2 (DESIGN §3.3, §7.6): relations, window, P cap, self-continuation exclusion, prune, N_max, levers."""
from __future__ import annotations

import random

import polars as pl

from certichain.candidates.generate import CandParams, generate_candidates, rel_mask_names
from certichain.io.schema import (CANDIDATE_SCHEMA, REL_PIVOT, REL_REVERSE, REL_SAME_ACTOR, REL_SAME_VICTIM,
                                  REL_SHARED_EXT, UNIT_SCHEMA, conform)
from certichain.units.scope import ScopeResult
from certichain.units.sessionize import event_units, records_frame

SEED = "192.168.1.10"
V = "192.168.1.20"


def U(rows):
    out = []
    for i, r in enumerate(rows):
        d = {"unit_id": i, "ts": 0.0, "te": None, "src": SEED, "dsts": [], "dst_prefixes": [], "n_dst": None,
             "fanout": False, "svc_class": "other_tcp", "dir_class": "int->ext", "proto": "tcp", "dports": [],
             "n_dports": 0, "type": "BENIGN_GENERIC_TCP_HIGH", "n_records": 1, "obytes": 0, "rbytes": 0, "opkts": 0,
             "rpkts": 0, "fail_frac": 0.0, "dur": 0.0, "start_cv": float("nan"), "mean_bytes": 0.0,
             "content_flags": 0, "dns_names": [], "http_hosts": [], "tls_snis": [], "infra": [], "rec_ids": [i]}
        d.update(r)
        if d["te"] is None:
            d["te"] = d["ts"] + 1.0
        if d["n_dst"] is None:
            d["n_dst"] = len(d["dsts"])
        out.append(d)
    return conform(pl.DataFrame(out, schema=UNIT_SCHEMA), UNIT_SCHEMA)


def scope(infra=(), seeds=(SEED,)):
    return ScopeResult(seeds=tuple(seeds), infra={ip: f"INFRA{i}" for i, ip in enumerate(infra)},
                       infra_contacts={ip: 3 for ip in infra}, hosts={s: 0 for s in seeds})


def pairs(res):
    return {(u, v): m for u, v, m in res.cands.select("u", "v", "rel_mask").iter_rows()}


def gen(units, sc=None, **kw):
    return generate_candidates(units, sc or scope(), CandParams(**kw))


# --------------------------------------------------------------------------- relations
def test_pivot_and_reverse():
    u = U([dict(ts=0, src=SEED, dsts=[V], type="BRUTE_TELNET", svc_class="telnet", dir_class="int->int"),
           dict(ts=100, src=V, dsts=["6.6.6.6"], type="HTTP_DOWNLOAD", svc_class="http"),
           dict(ts=200, src=V, dsts=[SEED], type="INTERACTIVE_SESSION", svc_class="telnet", dir_class="int->int")])
    p = pairs(gen(u))
    assert p[(0, 1)] & REL_PIVOT and not p[(0, 1)] & REL_REVERSE
    assert p[(0, 2)] & REL_PIVOT and p[(0, 2)] & REL_REVERSE          # src_v ∈ dst_u and src_u ∈ dst_v
    assert p[(1, 2)] == REL_SAME_ACTOR


def test_same_victim_and_shared_ext_external():
    u = U([dict(ts=0, src=SEED, dsts=["5.5.5.5"], type="C2_PERIODIC"),
           dict(ts=50, src=V, dsts=["5.5.5.5"], type="C2_PERIODIC")])
    p = pairs(gen(u))
    assert p[(0, 1)] == REL_SAME_VICTIM | REL_SHARED_EXT


def test_shared_ext_by_name_namespace():
    u = U([dict(ts=0, src=SEED, dsts=["192.168.1.1"], type="DNS_ACTIVITY", svc_class="dns", dns_names=["cnc.evil.com"],
                dir_class="int->int", proto="udp"),
           dict(ts=30, src=V, dsts=["9.9.9.9"], type="HTTP_DOWNLOAD", svc_class="http", http_hosts=["cnc.evil.com"])])
    c = gen(u).cands
    assert c.height == 1 and c["rel_mask"][0] == REL_SHARED_EXT and c["shared_entity_kind"][0] == "external"


def test_infra_is_collapsed_to_shared_ext_only():
    # both contact INFRA resolver .1: no SAME_VICTIM, only SHARED_EXT-via-INFRA; benign types → pruned
    u = U([dict(ts=0, src=SEED, dsts=["192.168.1.1"], infra=["INFRA0"], type="DNS_ACTIVITY", svc_class="dns"),
           dict(ts=10, src=V, dsts=["192.168.1.1"], infra=["INFRA0"], type="DNS_ACTIVITY", svc_class="dns")])
    sc = scope(infra=["192.168.1.1"])
    assert gen(u, sc).cands.height == 0                               # pruned (benign-protocol pair)
    c = gen(u, sc, prune_generic=False).cands
    assert c["rel_mask"].to_list() == [REL_SHARED_EXT] and c["shared_entity_kind"].to_list() == ["infra"]
    # an attack-type endpoint is never pruned
    u2 = U([dict(ts=0, src=SEED, dsts=["192.168.1.1"], infra=["INFRA0"], type="DNS_ANOMALY", svc_class="dns"),
            dict(ts=10, src=V, dsts=["192.168.1.1"], infra=["INFRA0"], type="DNS_ACTIVITY", svc_class="dns")])
    assert gen(u2, sc).cands.height == 1


def test_infra_never_pivots():
    u = U([dict(ts=0, src=SEED, dsts=["192.168.33.12"], infra=["INFRA0"], type="BENIGN_GENERIC_TCP_HIGH"),
           dict(ts=10, src="192.168.33.12", dsts=[V], infra=["INFRA0"], type="INTERACTIVE_SESSION",
                svc_class="telnet")])
    c = gen(u, scope(infra=["192.168.33.12"])).cands
    assert c["rel_mask"].to_list() == [REL_SHARED_EXT]


def test_time_window():
    base = dict(src=SEED, dsts=["5.5.5.5"], type="C2_PERIODIC")
    u = U([dict(ts=0, te=100, **base), dict(ts=1900, src=V, dsts=["5.5.5.5"], type="SCAN_H", svc_class="telnet"),
           dict(ts=1901, src=V, dsts=["5.5.5.5"], type="HTTP_DOWNLOAD", svc_class="http")])
    p = pairs(gen(u))
    assert (0, 1) in p                                               # 1900 ≤ te_u + W = 1900
    assert (0, 2) not in p                                           # 1901 > 1900
    u = U([dict(ts=5, **base), dict(ts=5, src=V, dsts=["5.5.5.5"], type="SCAN_H", svc_class="telnet")])
    assert gen(u).cands.height == 0                                  # equal starts: no direction → no candidate


def test_self_continuation_excluded():
    u = U([dict(ts=0, src=SEED, dsts=["5.5.5.5"], type="C2_PERIODIC", svc_class="irc"),
           dict(ts=1000, src=SEED, dsts=["5.5.5.6"], type="C2_PERIODIC", svc_class="irc"),
           dict(ts=1100, src=SEED, dsts=["5.5.5.7"], type="SCAN_H", svc_class="irc")])
    res = gen(u)
    assert (0, 1) not in pairs(res) and res.cont.select("u", "v").rows() == [(0, 1)]
    assert (0, 2) in pairs(res) and (1, 2) in pairs(res)


def test_prune_generic_same_actor_benign():
    u = U([dict(ts=0, src=SEED, dsts=["52.1.1.1"], type="TLS_SESSION", svc_class="https"),
           dict(ts=10, src=SEED, dsts=["192.168.1.1"], type="DNS_ACTIVITY", svc_class="dns"),
           dict(ts=20, src=SEED, dsts=["6.6.6.6"], type="HTTP_DOWNLOAD", svc_class="http")])
    p = pairs(gen(u))
    assert (0, 1) not in p and (0, 2) in p and (1, 2) in p
    assert (0, 1) in pairs(gen(u, prune_generic=False))


def test_p_cap_keeps_smallest_lag_gap():
    rows = [dict(ts=float(k * 10), te=float(k * 10 + 1), src=SEED, dsts=[f"5.5.{k}.1"], type="SCAN_H",
                 svc_class=f"s{k}") for k in range(15)]
    rows.append(dict(ts=1000.0, src=SEED, dsts=["9.9.9.9"], type="HTTP_DOWNLOAD", svc_class="http"))
    u = U(rows)
    res = gen(u)
    preds = sorted(res.cands.filter(pl.col("v") == 15)["u"].to_list())
    assert preds == list(range(5, 15))                               # the 10 most recently ended predecessors
    assert res.info["n_dropped_P_cap"] > 0
    res5 = gen(u, lever_P5=True)
    assert sorted(res5.cands.filter(pl.col("v") == 15)["u"].to_list()) == list(range(10, 15))


def test_p_cap_is_per_relation():
    # 12 SAME_ACTOR predecessors plus 1 older SAME_VICTIM predecessor: the latter survives via its own relation
    rows = [dict(ts=0.0, src="192.168.1.99", dsts=["7.7.7.7"], type="SCAN_H", svc_class="x")]
    rows += [dict(ts=float(100 + k), src=SEED, dsts=[f"5.5.{k}.1"], type="SCAN_H", svc_class=f"s{k}")
             for k in range(12)]
    rows.append(dict(ts=500.0, src=SEED, dsts=["7.7.7.7"], type="HTTP_DOWNLOAD", svc_class="http"))
    res = gen(U(rows))
    p = pairs(res)
    v = len(rows) - 1
    assert (0, v) in p and p[(0, v)] & REL_SAME_VICTIM
    assert sum(1 for (a, b) in p if b == v and p[(a, b)] & REL_SAME_ACTOR) == 10


def test_n_max_drops_largest_lag():
    rows = [dict(ts=float(k * 10), src=SEED, dsts=[f"5.5.{k}.1"], type="SCAN_H", svc_class=f"s{k}") for k in range(8)]
    res = gen(U(rows), N_max=5)
    assert res.info["N_max_hit"] and res.cands.height == 5
    full = gen(U(rows)).cands
    from certichain.units.sessionize import id_hash
    full = full.with_columns(pl.col("u").map_elements(id_hash, return_dtype=pl.UInt64).alias("hu"),
                             pl.col("v").map_elements(id_hash, return_dtype=pl.UInt64).alias("hv"))
    keep = full.sort(["lag_gap", "hu", "hv"]).head(5).select("u", "v").sort(["v", "u"])   # tie: unit-id hash
    assert res.cands.select("u", "v").sort(["v", "u"]).equals(keep)


def test_lever_victim_seed_only():
    u = U([dict(ts=0, src="192.168.1.30", dsts=["7.7.7.7"], type="SCAN_H", svc_class="x"),
           dict(ts=10, src="192.168.1.40", dsts=["7.7.7.7", "192.168.1.41"], type="HTTP_DOWNLOAD", svc_class="http",
                fanout=False),
           dict(ts=20, src=SEED, dsts=["192.168.1.41"], type="SCAN_V", svc_class="y")])
    base = pairs(gen(u, relations=("SAME_VICTIM",)))
    lev = pairs(gen(u, relations=("SAME_VICTIM",), lever_victim_seed=True))
    assert (0, 1) in base and (0, 1) not in lev                      # neither touches the seed
    assert (1, 2) in lev                                             # v is the seed's unit


def test_lever_prune_benign_all():
    u = U([dict(ts=0, src=SEED, dsts=[V], type="TLS_SESSION", svc_class="https", dir_class="int->int"),
           dict(ts=10, src=V, dsts=["5.5.5.5"], type="DNS_ACTIVITY", svc_class="dns")])
    assert gen(u).cands.height == 1                                  # PIVOT survives the default prune
    assert gen(u, lever_prune_all=True).cands.height == 0


def test_schema_lags_and_shared_entity():
    u = U([dict(ts=0, te=50, src=SEED, dsts=[V], type="BRUTE_TELNET", svc_class="telnet", dir_class="int->int"),
           dict(ts=40, src=V, dsts=["6.6.6.6"], type="HTTP_DOWNLOAD", svc_class="http")])
    c = gen(u).cands
    assert c.schema == pl.Schema(CANDIDATE_SCHEMA)
    row = c.row(0, named=True)
    assert row["lag_start"] == 40 and row["lag_gap"] == -10 and row["overlap"]
    assert row["shared_entity_kind"] == "internal" and not row["shared_is_seed"]   # the pivot host is V
    assert rel_mask_names(row["rel_mask"]) == ["PIVOT"]
    u = U([dict(ts=0, src=SEED, dsts=["5.5.5.5"], type="HTTP_DOWNLOAD", svc_class="http"),
           dict(ts=10, src=SEED, dsts=["6.6.6.6"], type="C2_PERIODIC", svc_class="irc")])
    c = gen(u).cands
    assert c["rel_mask"].to_list() == [REL_SAME_ACTOR] and c["shared_is_seed"].to_list() == [True]


def test_candidate_determinism_under_record_permutation():
    rnd = random.Random(11)
    rows = []
    for i in range(300):
        s = rnd.choice([SEED, V, "192.168.1.30", "1.2.3.4"])
        d = rnd.choice(["5.5.5.5", "6.6.6.6", SEED, V, "192.168.1.30", f"77.1.{rnd.randint(0, 20)}.9"])
        if d == s:
            continue
        rows.append(dict(ts=float(rnd.randint(0, 5000) // 5 * 5), src=s, dst=d, dport=rnd.choice([23, 80, 443, 6667]),
                         sport=rnd.randint(1024, 60000), obytes=rnd.choice([0, 50, 600]),
                         rbytes=rnd.choice([0, 60, 20000]), conn_state=rnd.choice(["S0", "SF"])))
    a = event_units(records_frame(rows), [SEED])
    perm = rows[:]
    rnd.shuffle(perm)
    b = event_units(records_frame(perm), [SEED])
    ca = generate_candidates(a.units, a.scope, CandParams()).cands
    cb = generate_candidates(b.units, b.scope, CandParams()).cands
    assert ca.height > 0 and ca.equals(cb)
