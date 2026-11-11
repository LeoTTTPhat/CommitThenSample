"""features/stats.py + features/descriptors.py (DESIGN §3.4.1): feature values on hand-built incidents."""
from __future__ import annotations

import numpy as np
import polars as pl

from certichain.candidates.generate import CandParams, generate_candidates
from certichain.features import stats as F
from certichain.features.descriptors import DESCRIPTOR_NAMES, descriptor
from certichain.io.schema import CANDIDATE_SCHEMA, UNIT_SCHEMA, conform
from certichain.units.scope import ScopeResult

SEED = "192.168.1.10"
TYPES = ("SCAN_H", "BRUTE_TELNET", "INTERACTIVE_SESSION", "HTTP_DOWNLOAD", "C2_PERIODIC", "BENIGN_GENERIC_TCP",
         "DNS_ACTIVITY", "FLOOD_SYN")


def units_frame(rows):
    out = []
    for i, r in enumerate(rows):
        d = {"unit_id": i, "ts": 0.0, "te": None, "src": SEED, "dsts": [], "dst_prefixes": [], "n_dst": None,
             "fanout": False, "svc_class": "other_tcp", "dir_class": "int->ext", "proto": "tcp", "dports": [],
             "n_dports": 0, "type": "BENIGN_GENERIC_TCP", "n_records": 1, "obytes": 0, "rbytes": 0, "opkts": 0,
             "rpkts": 0, "fail_frac": 0.0, "dur": 0.0, "start_cv": float("nan"), "mean_bytes": 0.0,
             "content_flags": 0, "dns_names": [], "http_hosts": [], "tls_snis": [], "infra": [], "rec_ids": [i]}
        d.update(r)
        if d["te"] is None:
            d["te"] = d["ts"] + max(1.0, d["dur"])
        d["dur"] = d["te"] - d["ts"]
        if d["n_dst"] is None:
            d["n_dst"] = len(d["dsts"])
        out.append(d)
    return conform(pl.DataFrame(out, schema=UNIT_SCHEMA), UNIT_SCHEMA)


def side_frame(units):
    return pl.DataFrame({"unit_id": units["unit_id"],
                         "touches_seed": (units["src"] == SEED) | units["dsts"].list.contains(SEED)})


def scope():
    return ScopeResult(seeds=(SEED,), infra={}, infra_contacts={}, hosts={SEED: 0})


def synth_incident(seed: int, n_units: int = 24, plant: bool = True):
    """Random units around a seed host + candidates from P2-B's generator + synthetic labels
    (y = 1: planted BRUTE_TELNET -> INTERACTIVE_SESSION on the same victim within 60 s; AMB: same-type
    planted pairs further apart; else 0). Returns (units, side, cands, y)."""
    rng = np.random.default_rng(seed)
    hosts = [SEED] + [f"192.168.1.{20 + k}" for k in range(5)] + [f"8.8.{k}.{k}" for k in range(3)]
    rows = []
    t = 0.0
    for i in range(n_units):
        t += float(rng.integers(5, 120))
        src = hosts[int(rng.integers(0, len(hosts)))]
        dst = hosts[int(rng.integers(0, len(hosts)))]
        typ = TYPES[int(rng.integers(0, len(TYPES)))]
        rows.append(dict(ts=t, te=t + float(rng.integers(1, 300)), src=src, dsts=[dst], type=typ,
                         svc_class=("telnet" if "TELNET" in typ else "http" if "HTTP" in typ else "other_tcp"),
                         dir_class="int->int" if dst.startswith("192.168") else "int->ext",
                         dports=[int(rng.choice([23, 80, 443, 1883, 5555]))], n_dports=1,
                         n_records=int(rng.integers(1, 50)), obytes=int(rng.integers(0, 10**5)),
                         rbytes=int(rng.integers(0, 10**5)), opkts=int(rng.integers(1, 100)),
                         rpkts=int(rng.integers(0, 100)), fail_frac=float(rng.random()),
                         start_cv=float(rng.random()), content_flags=int(rng.choice([0, 0, 1, 4, 5]))))
    if plant:
        t += 30.0
        victim = "192.168.1.21"
        rows.append(dict(ts=t, te=t + 40, src="8.8.1.1", dsts=[victim], type="BRUTE_TELNET", svc_class="telnet",
                         dir_class="ext->int", dports=[23], n_dports=1, n_records=40, fail_frac=0.9))
        rows.append(dict(ts=t + 55, te=t + 80, src="8.8.2.2", dsts=[victim], type="INTERACTIVE_SESSION",
                         svc_class="telnet", dir_class="ext->int", dports=[23], n_dports=1, n_records=1))
    rows.sort(key=lambda r: r["ts"])
    units = units_frame(rows)
    side = side_frame(units)
    cands = generate_candidates(units, scope(), CandParams(W=1800.0)).cands
    tu = dict(zip(units["unit_id"].to_list(), units["type"].to_list()))
    y = []
    for u, v, lag in zip(cands["u"].to_list(), cands["v"].to_list(), cands["lag_gap"].to_list()):
        if tu[u] == "BRUTE_TELNET" and tu[v] == "INTERACTIVE_SESSION":
            y.append(1 if lag <= 60 else -1)
        else:
            y.append(0)
    return units, side, cands, np.asarray(y, dtype=np.int8)


def test_feature_names_and_shapes():
    units, side, cands, _ = synth_incident(0)
    df = F.base_features(units, cands, side)
    assert df.height == cands.height
    assert list(df.columns) == [*F.KEY_COLS, *F.FEATURES]
    assert len(F.FEATURES) == len(set(F.FEATURES)) == 59
    assert set(F.PREFIX_FEATURES) == set(F.FEATURES) - set(F.RARITY_FEATURES)
    assert df["cand_id"].to_list() == sorted(cands["cand_id"].to_list())
    X = F.feature_matrix(df)
    assert X.shape == (cands.height, len(F.FEATURES)) and X.dtype == np.float32
    for n in ("trans_prior", "lag_llr", "rank_llr_u"):     # only after add_prior_features
        assert df[n].is_nan().all()
    cat = F.categorical_indices(F.FEATURES)
    assert [F.FEATURES[i] for i in cat] == [n for n in F.FEATURES if n in F.CATEGORICAL]
    assert np.all(X[:, cat] >= 0)


def test_hand_values():
    u = units_frame([
        dict(ts=0, te=10, src="8.8.1.1", dsts=["192.168.1.21"], type="BRUTE_TELNET", svc_class="telnet",
             dir_class="ext->int", dports=[23], n_dports=1, n_records=40, obytes=999, fail_frac=0.9,
             content_flags=4),
        dict(ts=5, te=8, src="8.8.1.1", dsts=["192.168.1.22"], type="SCAN_H", svc_class="telnet",
             dir_class="ext->int", dports=[2323], n_dports=1),
        dict(ts=20, te=30, src="8.8.2.2", dsts=["192.168.1.21"], type="INTERACTIVE_SESSION", svc_class="telnet",
             dir_class="ext->int", dports=[23], n_dports=1, content_flags=1 | 16),
        dict(ts=25, te=31, src="8.8.2.2", dsts=["192.168.1.21"], type="INTERACTIVE_SESSION", svc_class="ssh",
             dir_class="ext->int", dports=[22], n_dports=1),
    ])
    c = conform(pl.DataFrame({"cand_id": [0, 1, 2, 3], "u": [0, 1, 0, 2], "v": [2, 2, 3, 3],
                              "rel_mask": [8, 0, 8 | 4, 4], "lag_start": [20.0, 15.0, 25.0, 5.0],
                              "lag_gap": [10.0, 12.0, 15.0, -5.0], "overlap": [False, False, False, True],
                              "shared_entity_kind": ["internal", "none", "internal", "external"],
                              "shared_entity_deg": [3, 0, 3, 7], "shared_is_seed": [False, False, True, False]}),
                CANDIDATE_SCHEMA)
    df = F.base_features(u, c, side_frame(u))
    r = {k: df[k].to_list() for k in df.columns}
    assert np.allclose(r["llag_start"], np.log1p([20, 15, 25, 5]))
    assert np.allclose(r["llag_gap"], [np.log1p(10), np.log1p(12), np.log1p(15), -np.log1p(5)])
    assert r["overlap"] == [0, 0, 0, 1]
    assert r["rel_same_victim"] == [1, 0, 1, 0] and r["rel_same_actor"] == [0, 0, 1, 1]
    assert r["rel_count"] == [1, 0, 2, 1]
    # competition among the predecessors of v: v=2 has u in {0, 1}; lag_gap 10 < 12
    assert r["n_pred_v"] == [2, 2, 2, 2] and r["rank_lag_u"] == [1, 2, 2, 1]
    # activity novelty: unit 3 (ssh) is a new activity; unit 2 is the first of its activity
    assert r["act_before_v"] == [0, 0, 0, 0]
    assert r["act_between_u"][0] == 0.0
    # rarity: (BRUTE_TELNET, INTERACTIVE_SESSION) occurs twice among 4 candidates
    assert np.isclose(r["rar_pair"][0], -np.log(2 / 4)) and np.isclose(r["rar_pair"][1], -np.log(1 / 4))
    assert np.isclose(r["rar_type_v"][0], -np.log(2 / 4))
    assert r["ent_kind"][:2] == [F.ENTITY_VOCAB.index("internal"), F.ENTITY_VOCAB.index("none")]
    assert r["ent_seed"] == [0, 0, 1, 0]
    assert r["port_u"][0] == F.IOT_PORTS.index(23) and r["port_u"][1] == F.IOT_PORTS.index(2323)
    assert r["port_v"][2] == F.IOT_PORTS.index(22)
    assert r["c_toolua_u"][0] == 1 and r["c_exec_v"][0] == 1 and r["c_hient_v"][0] == 1 and r["c_exec_u"][0] == 0
    assert r["type_u"][0] == F.TYPE_VOCAB.index("BRUTE_TELNET")
    assert np.isclose(r["lobytes_u"][0], np.log1p(999)) and np.isclose(r["fail_u"][0], 0.9)


def test_activity_counts():
    u = units_frame([dict(ts=0, te=5, src="A", type="SCAN_H", svc_class="telnet"),
                     dict(ts=10, te=15, src="A", type="SCAN_H", svc_class="telnet"),
                     dict(ts=20, te=25, src="A", type="SCAN_H", svc_class="telnet"),
                     dict(ts=30, te=35, src="B", type="SCAN_H", svc_class="telnet")])
    c = conform(pl.DataFrame({"cand_id": [0, 1], "u": [0, 2], "v": [3, 3], "rel_mask": [16, 16],
                              "lag_start": [30.0, 10.0], "lag_gap": [25.0, 5.0], "overlap": [False, False]}),
                CANDIDATE_SCHEMA)
    df = F.base_features(u, c, None)
    assert np.allclose(df["act_between_u"].to_numpy(), np.log1p([2, 0]))   # units 1, 2 of A lie between 0 and v
    assert np.allclose(df["act_before_v"].to_numpy(), [0, 0])               # B starts a new activity
    assert df["seed_u"].is_nan().all()                                     # no side table


def test_empty_incident():
    units, side, _, _ = synth_incident(1, n_units=3, plant=False)
    df = F.base_features(units, pl.DataFrame(schema=CANDIDATE_SCHEMA), side)
    assert df.height == 0 and list(df.columns) == [*F.KEY_COLS, *F.FEATURES]
    assert F.feature_matrix(df).shape == (0, len(F.FEATURES))


def test_descriptor_is_finite_and_label_free_named():
    units, side, cands, _ = synth_incident(2)
    d = descriptor(units, cands, side)
    assert d.shape == (len(DESCRIPTOR_NAMES),) and len(DESCRIPTOR_NAMES) == 42
    assert np.isfinite(d).all()
    assert 0 <= d[DESCRIPTOR_NAMES.index("d_seed")] <= 1
    e = descriptor(units.head(0), pl.DataFrame(schema=CANDIDATE_SCHEMA), None)
    assert np.isfinite(e).all()
