"""The one GT semantics (DESIGN §2.3, §5.2, §7.6) on hand-made episode DAGs + per-source extractors."""
from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from certichain.gt import observability as OB
from certichain.gt.chainforge_episodes import build_chainforge_gt, chainforge_gt
from certichain.gt.gotham_episodes import build_gotham_gt
from certichain.gt.iot23_episodes import EXEC, build_iot23_gt, iot23_gt
from certichain.gt.semantics import AMB, EpisodeGT, label_pairs, transition_stats, transition_table

# --------------------------------------------------------------------------- helpers


def units(ts, src=None, te=None):
    n = len(ts)
    return pl.DataFrame({"unit_id": list(range(n)), "ts": [float(t) for t in ts],
                         "te": [float(t) + 1 for t in ts] if te is None else [float(t) for t in te],
                         "src": src or ["h"] * n, "rec_ids": [[i] for i in range(n)]},
                        schema={"unit_id": pl.Int64, "ts": pl.Float64, "te": pl.Float64, "src": pl.Utf8,
                                "rec_ids": pl.List(pl.Int64)})


def all_pairs(u):
    ids = sorted(u["unit_id"].to_list(), key=lambda i: u["ts"][i])
    pr = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:] if u["ts"][a] < u["ts"][b]]
    return pl.DataFrame({"cand_id": list(range(len(pr))), "u": [a for a, _ in pr], "v": [b for _, b in pr]},
                        schema={"cand_id": pl.Int64, "u": pl.Int64, "v": pl.Int64})


def lab(u, gt, cands=None):
    c = all_pairs(u) if cands is None else cands
    y = label_pairs(c, u, gt)
    return {(a, b): (yy, br) for a, b, yy, br in zip(c["u"], c["v"], y["y"], y["bridge"])}


def gt_of(ep: dict, camp: dict, T, hidden=(), **kw):
    """ep: unit → episode; camp: episode → campaign."""
    return EpisodeGT(unit_episode=dict(ep), unit_campaign={u: camp[e] for u, e in ep.items()},
                     transitions=set(T), hidden=set(hidden), episode_campaign=dict(camp), **kw)


# --------------------------------------------------------------------------- rules 1–5
def test_enabling_unit_rule():
    # A: units 0 (t=0), 1 (t=10), 3 (t=30); B: units 2 (t=20), 4 (t=40)
    u = units([0, 10, 20, 30, 40])
    g = gt_of({0: "A", 1: "A", 3: "A", 2: "B", 4: "B"}, {"A": "c", "B": "c"}, [("A", "B")])
    L = lab(u, g)
    assert L[(1, 2)] == (1, False)                  # enab(A,B) = last A unit before first(B)=2 → 1
    assert L[(0, 2)][0] == AMB and L[(1, 4)][0] == AMB and L[(3, 4)][0] == AMB
    assert L[(0, 1)][0] == AMB                      # same episode (rule 2)
    assert L[(2, 3)][0] == 0                        # B → A has no directed path (rule 5)


def test_hidden_intermediate_bridges_are_true():
    u = units([0, 10])
    g = gt_of({0: "A", 1: "B"}, {"A": "c", "B": "c", "H": "c"}, [("A", "H"), ("H", "B")], hidden={"H"})
    assert lab(u, g)[(0, 1)] == (1, True)
    g2 = gt_of({0: "A", 1: "B"}, {"A": "c", "B": "c"}, [("A", "H1"), ("H1", "H2"), ("H2", "B")],
               hidden={"H1", "H2"})
    assert lab(u, g2)[(0, 1)] == (1, True)
    g3 = gt_of({0: "A", 1: "B"}, {"A": "c", "B": "c"}, [("A", "H"), ("H", "B"), ("A", "B")], hidden={"H"})
    assert lab(u, g3)[(0, 1)] == (1, False)         # a direct edge wins: not a bridge
    # an episode without units is hidden even if not declared (e.g. dropped by the scope rule)
    g4 = gt_of({0: "A", 1: "B"}, {"A": "c", "B": "c", "M": "c"}, [("A", "M"), ("M", "B")])
    assert lab(u, g4)[(0, 1)] == (1, True)


def test_observed_intermediate_transitive_is_amb():
    u = units([0, 10, 20])
    g = gt_of({0: "A", 1: "B", 2: "C"}, {"A": "c", "B": "c", "C": "c"}, [("A", "B"), ("B", "C")])
    L = lab(u, g)
    assert L[(0, 1)][0] == 1 and L[(1, 2)][0] == 1 and L[(0, 2)][0] == AMB
    g2 = gt_of({0: "A", 1: "B", 2: "C"}, {"A": "c", "B": "c", "C": "c"},
               [("A", "H"), ("H", "B"), ("B", "C")], hidden={"H"})
    assert lab(u, g2)[(0, 2)][0] == AMB             # path A⇝C passes the observed B


def test_decoy_campaigns():
    # main campaign M: X → Y; D2 parallel infection: P → Q; D1 single episode Z
    u = units([0, 5, 10, 15, 20])
    g = gt_of({0: "X", 1: "P", 2: "Y", 3: "Q", 4: "Z"},
              {"X": "M", "Y": "M", "P": "D2", "Q": "D2", "Z": "D1"}, [("X", "Y"), ("P", "Q")])
    L = lab(u, g)
    assert L[(0, 2)][0] == 1 and L[(1, 3)][0] == 1  # D2 internal transitions are true links
    assert L[(0, 3)][0] == 0 and L[(1, 2)][0] == 0  # cross-campaign → 0
    assert all(L[(a, 4)][0] == 0 for a in range(4))  # D1 has no positives


def test_background_siblings_and_mixed():
    u = units([0, 10, 20, 30, 40])
    g = gt_of({0: "A", 1: "B", 2: "C"}, {"A": "c", "B": "c", "C": "c"}, [("A", "B"), ("A", "C")])
    g.mixed_units.add(3)
    g.unit_campaign[3] = "c"
    L = lab(u, g)
    assert L[(1, 2)][0] == 0                        # siblings → 0
    assert L[(0, 4)][0] == 0 and L[(3, 4)][0] == 0  # ⊥ (unit 4) → 0, even against a MIXED unit
    assert L[(2, 3)][0] == AMB                      # MIXED → AMB
    g.unit_campaign[3] = "other"
    assert lab(u, g)[(2, 3)][0] == 0                # MIXED but of another campaign → 0


def test_explicit_amb_pairs_and_unrelated_campaigns():
    u = units([0, 10])
    g = gt_of({0: "A", 1: "B"}, {"A": "c", "B": "c"}, [])
    assert lab(u, g)[(0, 1)][0] == 0
    g.amb_pairs.add(("A", "B"))
    assert lab(u, g)[(0, 1)][0] == AMB
    g = gt_of({0: "A", 1: "B"}, {"A": "c", "B": "c"}, [], amb_unrelated_campaigns={"c"})
    assert lab(u, g)[(0, 1)][0] == AMB


def test_transition_stats_and_observability():
    u = units([0, 10, 20])
    # T*: E(hidden)→A, A→B, B→C, C→D (D has no unit → hidden)
    g = gt_of({0: "A", 1: "B", 2: "C"}, {"A": "c", "B": "c", "C": "c", "D": "c", "E": "c"},
              [("E", "A"), ("A", "B"), ("B", "C"), ("C", "D")], hidden={"E"})
    c = pl.DataFrame({"cand_id": [0, 1], "u": [0, 0], "v": [1, 2]})       # (enab(B,C), first(C)) = (1,2) missing
    st = transition_stats(c, u, g, label_pairs(c, u, g))
    assert (st["n_T"], st["n_T_obs"], st["n_T_hid"], st["n_T_obs_in_C"]) == (4, 2, 2, 1)
    assert st["rho_T"] == 0.5 and st["r_CT"] == 0.5 and st["ceiling"] == 0.25
    tt = transition_table(c, u, g)
    assert OB.rho_T(st) == 0.5 and OB.r_CT(st) == 0.5 and OB.ceiling(st) == 0.25
    m = OB.str_metrics([0], tt)
    assert m["STR"] == 0.25 and m["STR"] <= OB.ceiling(st)
    assert OB.str_metrics([1], tt)["STR"] == 0.0
    assert OB.candidate_transitions(tt).select("cand_id", "src_ep", "dst_ep").rows() == [(0, "A", "B")]


def test_fdp_lenient_le_strict_random():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        y = rng.choice([1, 0, -1], size=int(rng.integers(0, 40)))
        assert OB.fdp_lenient(y) <= OB.fdp_strict(y) + 1e-12


# --------------------------------------------------------------------------- IoT-23 (§5.2.2)
SEEDIP = "192.168.1.195"


def _iot23_fixture():
    #        unit: 0 benign, 1 DL, 2 CC, 3 CC-HB, 4 SCAN, 5 CCDL, 6 CC, 7 CC (>1 h later), 8 ATK, 9 MIXED,
    #              10 non-seed malicious, 11 non-seed benign
    ts = [0, 10, 20, 30, 40, 50, 60, 6000, 6010, 6020, 6030, 6040]
    src = [SEEDIP] * 10 + ["1.2.3.4", "5.6.7.8"]
    u = units(ts, src=src)
    dl = {0: ["-"], 1: ["FileDownload"], 2: ["C&C"], 3: ["C&C-HeartBeat"], 4: ["PartOfAHorizontalPortScan"],
          5: ["C&C-FileDownload"], 6: ["C&C"], 7: ["C&C"], 8: ["DDoS"], 9: ["FileDownload", "C&C"],
          10: ["C&C"], 11: ["-"]}
    rec, rows = [], []
    rid = 0
    for uid, labels in dl.items():
        ids = []
        for d in labels:
            mal = d != "-"
            rows.append({"rec_id": rid, "label": "Malicious" if mal else "Benign", "detailed_label": d})
            ids.append(rid)
            rid += 1
        rec.append(ids)
    u = u.with_columns(pl.Series("rec_ids", rec, dtype=pl.List(pl.Int64)))
    return u, pl.DataFrame(rows)


def test_iot23_r0_r1_r2_and_ccdl_direction():
    u, labels = _iot23_fixture()
    g = build_iot23_gt(u, labels, [SEEDIP], "34-1")
    E = g.unit_episode
    assert E[2] == E[3] and E[6] != E[2] and E[7] != E[6]           # CC run; CC split by other classes / > 1 h
    assert 0 not in E and 11 not in E and 9 in g.mixed_units
    assert E[10].endswith("@1.2.3.4")                               # non-seed malicious host: own episode
    T = g.transitions
    assert (E[1], E[2]) in T                                        # R1 DL → next CC
    assert (E[2], E[4]) in T and (E[2], E[5]) in T                  # R2 CC → SCAN, CC → CCDL
    assert (E[7], E[8]) in T and (E[6], E[8]) not in T              # most recent CC before the action
    assert (E[5], E[6]) not in T                                    # CCDL is an action: no CCDL → CC
    assert (EXEC, E[1]) in T and EXEC in g.hidden                   # R0 hidden Execution → first episode
    L = lab(u, g)
    assert L[(1, 2)][0] == 1 and L[(3, 4)][0] == 1 and L[(3, 5)][0] == 1 and L[(7, 8)][0] == 1
    assert L[(2, 4)][0] == AMB                                      # not the enabling unit
    assert L[(5, 6)][0] == AMB and L[(6, 8)][0] == AMB              # malicious pairs not covered by R1/R2
    assert L[(8, 10)][0] == AMB and L[(8, 9)][0] == AMB             # other-host malicious / MIXED
    assert L[(0, 1)][0] == 0 and L[(9, 11)][0] == 0 and L[(10, 11)][0] == 0
    st = transition_stats(all_pairs(u), u, g)
    assert st["n_T"] == 5 and st["n_T_hid"] == 1 and st["rho_T"] == pytest.approx(0.2)
    assert st["r_CT"] == 1.0                                        # all pairs are candidates here


def test_iot23_reads_labels_through_quarantine(tmp_path):
    u, labels = _iot23_fixture()
    p = tmp_path / "w.parquet"
    labels.write_parquet(p)
    g = iot23_gt("34-1", "34-1_b000", u, [SEEDIP], labels_path=str(p))
    assert g.unit_episode == build_iot23_gt(u, labels, [SEEDIP], "34-1").unit_episode


# --------------------------------------------------------------------------- Gotham (§5.2.3)
B, Y = "192.168.18.15", "192.168.20.10"


def _gotham_fixture():
    rows = [  # (ts, src, dst, dport, label)
        (50, "192.168.0.100", Y, 23, "Telnet Brute Force"),            # BF0(y)
        (100, "192.168.33.12", B, 23, "Ingress Tool Transfer"),        # ITT(b)
        (103, B, "192.168.33.13", 80, "File Download"),                # DL(b)
        (104, B, "192.168.33.10", 23, "C&C Communication"),            # CC(b)
        (105, B, "192.168.17.99", 23, "TCP Scan"),                     # SCAN(b)
        (200, B, "192.168.33.11", 48101, "Reporting"),                 # REP(b)
        (201, "192.168.33.12", Y, 23, "Ingress Tool Transfer"),        # ITT(y)
        (300, "192.168.17.50", "192.168.17.51", 80, "Benign"),         # ⊥
    ]
    rec = pl.DataFrame({"rec_id": list(range(len(rows))), "ts": [float(r[0]) for r in rows],
                        "src": [r[1] for r in rows], "dst": [r[2] for r in rows], "dport": [r[3] for r in rows]},
                       schema={"rec_id": pl.Int64, "ts": pl.Float64, "src": pl.Utf8, "dst": pl.Utf8,
                               "dport": pl.Int32})
    labels = pl.DataFrame({"rec_id": list(range(len(rows))),
                           "label": ["Benign" if r[4] == "Benign" else "Malicious" for r in rows],
                           "detailed_label": [r[4] for r in rows]})
    u = units([r[0] for r in rows], src=[r[1] for r in rows])        # one unit per record
    return rec, u, labels


def test_gotham_episodes_and_lifecycle():
    rec, u, labels = _gotham_fixture()
    g = build_gotham_gt(rec, u, labels, "mirai-infection", reports=None)
    E = g.unit_episode
    assert [E.get(i) for i in range(7)] == [f"BF0({Y})", f"ITT({B})", f"DL({B})", f"CC({B})", f"SCAN({B})",
                                            f"REP({B})", f"ITT({Y})"]
    assert 7 not in E
    T = g.transitions
    assert {(E[1], E[2]), (E[2], E[3]), (E[2], E[4]), (E[4], E[5])} <= T
    L = lab(u, g)
    assert L[(1, 2)][0] == 1 and L[(2, 3)][0] == 1 and L[(2, 4)][0] == 1 and L[(4, 5)][0] == 1
    assert L[(3, 4)][0] == 0                                        # CC and SCAN are siblings (Download → {CC, scan})


def test_gotham_cracker_payload_names_victim():
    rec, u, labels = _gotham_fixture()
    rep = pl.DataFrame({"t": [200.01], "reporter": [B], "target": [Y], "port": [23]})
    g = build_gotham_gt(rec, u, labels, "mirai-infection", reports=rep)
    assert g.meta["cracker_cases"][Y] == "a"
    L = lab(u, g)
    assert L[(5, 6)] == (1, False)                                  # REP(b) → ITT(y) ∈ T*
    assert L[(0, 6)][0] == AMB                                      # BF0(y) ⇝ ITT(y) AMB


def test_gotham_cracker_payload_decoded_not_naming():
    rec, u, labels = _gotham_fixture()
    rep = pl.DataFrame({"t": [200.01], "reporter": [B], "target": ["192.168.17.99"], "port": [23]})
    g = build_gotham_gt(rec, u, labels, "mirai-infection", reports=rep)
    assert g.meta["cracker_cases"][Y] == "b" and f"REP0({Y})" in g.hidden
    L = lab(u, g)
    assert L[(0, 6)] == (1, True)                                   # bridge BF0(y) → REP0(y) → ITT(y) = 1
    assert L[(5, 6)][0] == 0


def test_gotham_cracker_payload_absent():
    rec, u, labels = _gotham_fixture()
    g = build_gotham_gt(rec, u, labels, "mirai-infection", reports=None)
    assert g.meta["cracker_cases"][Y] == "c"
    L = lab(u, g)
    assert L[(0, 6)][0] == AMB and L[(5, 6)][0] == AMB


def test_gotham_dos_transitions():
    rows = [(10, "192.168.17.10", "192.168.33.10", 23, "Mirai C&C Communication"),
            (20, "192.168.17.10", "192.168.18.10", None, "Mirai UDP Flooding"),
            (30, "192.168.17.10", "192.168.18.18", None, "Mirai GRE Flooding")]
    rec = pl.DataFrame({"rec_id": [0, 1, 2], "ts": [float(r[0]) for r in rows], "src": [r[1] for r in rows],
                        "dst": [r[2] for r in rows], "dport": [r[3] for r in rows]},
                       schema={"rec_id": pl.Int64, "ts": pl.Float64, "src": pl.Utf8, "dst": pl.Utf8, "dport": pl.Int32})
    labels = pl.DataFrame({"rec_id": [0, 1, 2], "label": ["Malicious"] * 3, "detailed_label": [r[4] for r in rows]})
    u = units([10, 20, 30], src=[r[1] for r in rows])
    g = build_gotham_gt(rec, u, labels, "mirai-dos")
    L = lab(u, g)
    assert L[(0, 1)][0] == 1 and L[(0, 2)][0] == 1 and L[(1, 2)][0] == 0


# --------------------------------------------------------------------------- ChainForge (§5.2.1)
def _cf_fixture(tmp_path):
    # unit → records' episode ids (None = background)
    spec = {0: ["C0.E0"] * 3, 1: ["C0.E1", "C0.E1", None], 2: ["C0.E1", "C0.E2", None], 3: [None, None],
            4: ["C1.E0"], 5: ["C1.E1", "C1.E1"], 6: ["C0.E3"]}
    camp = {"C0": "C0", "C1": "C1"}
    rows, recs = [], []
    rid = 0
    for uid, eps in spec.items():
        ids = []
        for e in eps:
            rows.append({"rec_id": rid, "label": None if e is None else "attack", "detailed_label": None,
                         "episode_id": e, "campaign_id": None if e is None else camp[e.split(".")[0]],
                         "decoy_kind": "D2" if e and e.startswith("C1") else None, "template_node": e})
            ids.append(rid)
            rid += 1
        recs.append(ids)
    labels = pl.DataFrame(rows, schema={"rec_id": pl.Int64, "label": pl.Utf8, "detailed_label": pl.Utf8,
                                        "episode_id": pl.Utf8, "campaign_id": pl.Utf8, "decoy_kind": pl.Utf8,
                                        "template_node": pl.Utf8})
    gtj = {"campaigns": {"C0": {"template": "T-BOT", "kind": "main"}, "C1": {"template": "T-MIRAI", "kind": "D2"}},
           "episodes": {"C0.E0": {"campaign_id": "C0", "type": "scanning", "template_node": "scanning", "hidden": False},
                        "C0.E1": {"campaign_id": "C0", "type": "password", "template_node": "password", "hidden": False},
                        "C0.E2": {"campaign_id": "C0", "type": "backdoor", "template_node": "backdoor", "hidden": False},
                        "C0.H": {"campaign_id": "C0", "type": "backdoor", "template_node": "x", "hidden": True},
                        "C0.E3": {"campaign_id": "C0", "type": "ddos", "template_node": "ddos", "hidden": False},
                        "C1.E0": {"campaign_id": "C1", "type": "ITT", "template_node": "ITT", "hidden": False},
                        "C1.E1": {"campaign_id": "C1", "type": "DL", "template_node": "DL", "hidden": False}},
           "transitions": [["C0.E0", "C0.E1"], ["C0.E1", "C0.E2"], ["C0.E1", "C0.H"], ["C0.H", "C0.E3"],
                           ["C1.E0", "C1.E1"]],
           "hidden": ["C0.H"]}
    d = tmp_path / "cf_ton" / "train" / "000001"
    d.mkdir(parents=True)
    labels.write_parquet(d / "labels.parquet")
    (d / "gt.json").write_text(json.dumps(gtj))
    u = units([0, 10, 20, 30, 40, 50, 60]).with_columns(pl.Series("rec_ids", recs, dtype=pl.List(pl.Int64)))
    return d, u, labels, gtj


def test_chainforge_adapter(tmp_path):
    d, u, labels, gtj = _cf_fixture(tmp_path)
    g = chainforge_gt(d, u)
    assert g.unit_episode == {0: "C0.E0", 1: "C0.E1", 4: "C1.E0", 5: "C1.E1", 6: "C0.E3"}
    assert g.mixed_units == {2} and g.unit_campaign[2] == "C0" and 3 not in g.unit_episode
    L = lab(u, g)
    assert L[(0, 1)][0] == 1 and L[(4, 5)][0] == 1                   # main and D2-internal transitions
    assert L[(1, 6)] == (1, True)                                    # bridge through the hidden C0.H
    assert L[(0, 4)][0] == 0 and L[(1, 5)][0] == 0                   # cross-campaign
    assert L[(1, 2)][0] == AMB and L[(0, 3)][0] == 0                 # MIXED / background
    st = transition_stats(all_pairs(u), u, g)
    # T*: E0→E1 obs, E1→E2 hid (E2 only in a MIXED unit), E1→H hid, H→E3 hid, C1.E0→C1.E1 obs
    assert (st["n_T"], st["n_T_obs"], st["n_T_hid"]) == (5, 2, 3) and st["n_bridges"] == 1
    assert build_chainforge_gt(u, labels, gtj).unit_episode == g.unit_episode


def test_gotham_cracker_uses_out_of_scope_evidence():
    # the incident's units cover only BF0(y) and ITT(y); bot b's report flow is outside the scope but
    # still decides the cracker case (window evidence, not scope)
    rec, u, labels = _gotham_fixture()
    keep = [0, 6]
    us = u.filter(pl.col("unit_id").is_in(keep)).with_columns(pl.int_range(pl.len()).cast(pl.Int64).alias("unit_id"))
    rep = pl.DataFrame({"t": [200.01], "reporter": [B], "target": [Y], "port": [23]})
    g = build_gotham_gt(rec, us, labels, "mirai-infection", reports=rep)
    assert g.meta["cracker_cases"][Y] == "a"
    assert lab(us, g)[(0, 1)][0] == AMB                             # BF0(y) ⇝ ITT(y) AMB
    st = transition_stats(all_pairs(us), us, g)
    assert st["n_T_obs"] == 0 and st["n_T_hid"] == st["n_T"]       # REP(b) → ITT(y) has an unobserved endpoint


# --------------------------------------------------------------------------- relation-aware enab
def units_full(rows):
    """UNIT_SCHEMA frame with the entity columns the relation-aware enab reads."""
    from certichain.io.schema import UNIT_SCHEMA, conform
    out = []
    for i, r in enumerate(rows):
        d = {"unit_id": i, "te": None, "dsts": [], "dst_prefixes": [], "fanout": False, "svc_class": "other_tcp",
             "dir_class": "int->ext", "proto": "tcp", "dports": [], "n_dports": 0, "type": "BENIGN_GENERIC_TCP_HIGH",
             "n_records": 1, "obytes": 0, "rbytes": 0, "opkts": 0, "rpkts": 0, "fail_frac": 0.0, "dur": 0.0,
             "start_cv": float("nan"), "mean_bytes": 0.0, "content_flags": 0, "dns_names": [], "http_hosts": [],
             "tls_snis": [], "infra": [], "rec_ids": [i]}
        d.update(r)
        d["te"] = d["ts"] + 1.0 if d["te"] is None else d["te"]
        d["n_dst"] = len(d["dsts"])
        out.append(d)
    return conform(pl.DataFrame(out, schema=UNIT_SCHEMA), UNIT_SCHEMA)


ATT, X, V2, C2 = "198.51.100.7", "10.0.0.5", "10.0.0.6", "203.0.113.9"


def test_relation_aware_enab_picks_the_unit_on_the_role_pair():
    # A (password) hits X, then the secondary victim v2 (last before B); B (backdoor) starts on X
    u = units_full([dict(ts=0, src=ATT, dsts=[X]), dict(ts=10, src=ATT, dsts=[V2]), dict(ts=20, src=X, dsts=[C2])])
    g = gt_of({0: "A", 1: "A", 2: "B"}, {"A": "c", "B": "c"}, [("A", "B")])
    L = lab(u, g)
    assert L[(0, 2)] == (1, False)                     # A→X shares X with first(B) (PIVOT): enabling unit
    assert L[(1, 2)][0] == AMB                         # the DESIGN rule would have picked A→v2
    tt = transition_table(all_pairs(u), u, g)
    assert tt.select("enab_u", "first_v", "enab_rule").rows() == [(0, 2, "relation")]
    # the coordinator's literal ordering (A hits v2, then X): both rules agree on A→X
    u2 = units_full([dict(ts=0, src=ATT, dsts=[V2]), dict(ts=10, src=ATT, dsts=[X]), dict(ts=20, src=X, dsts=[C2])])
    assert lab(u2, g)[(1, 2)] == (1, False) and lab(u2, g)[(0, 2)][0] == AMB


def test_relation_aware_enab_falls_back_to_design_rule():
    # no unit of A shares an entity with first(B) → last unit of A before first(B)
    u = units_full([dict(ts=0, src=ATT, dsts=[V2]), dict(ts=10, src="198.51.100.8", dsts=["10.0.0.7"]),
                    dict(ts=20, src=X, dsts=[C2])])
    g = gt_of({0: "A", 1: "A", 2: "B"}, {"A": "c", "B": "c"}, [("A", "B")])
    assert lab(u, g)[(1, 2)] == (1, False) and lab(u, g)[(0, 2)][0] == AMB
    assert transition_table(all_pairs(u), u, g)["enab_rule"].to_list() == ["fallback"]


def test_relation_aware_enab_for_bridges():
    u = units_full([dict(ts=0, src=ATT, dsts=[X]), dict(ts=10, src=ATT, dsts=[V2]), dict(ts=20, src=X, dsts=[C2])])
    g = gt_of({0: "A", 1: "A", 2: "B"}, {"A": "c", "B": "c", "H": "c"}, [("A", "H"), ("H", "B")], hidden={"H"})
    L = lab(u, g)
    assert L[(0, 2)] == (1, True) and L[(1, 2)][0] == AMB


def test_relation_aware_enab_respects_infra_collapse():
    # the only entity unit 0 shares with first(B) is an INFRA host → not a host relation, but SHARED_EXT
    # via the INFRA token still counts (Alg. 2 definitions); without the token it falls back
    from certichain.units.scope import ScopeResult
    R = "10.0.0.1"
    u = units_full([dict(ts=0, src=ATT, dsts=[R], infra=["INFRA0"]), dict(ts=10, src=ATT, dsts=[V2]),
                    dict(ts=20, src=X, dsts=[R], infra=["INFRA0"])])
    g = gt_of({0: "A", 1: "A", 2: "B"}, {"A": "c", "B": "c"}, [("A", "B")])
    sc = ScopeResult(seeds=(X,), infra={R: "INFRA0"}, infra_contacts={R: 3}, hosts={X: 0})
    c = all_pairs(u)
    y = dict(zip(zip(c["u"], c["v"]), label_pairs(c, u, g, scope=sc)["y"]))
    assert y[(0, 2)] == 1                               # SHARED_EXT via INFRA token
    u2 = u.with_columns(pl.lit([], dtype=pl.List(pl.Utf8)).alias("infra"))
    y2 = dict(zip(zip(c["u"], c["v"]), label_pairs(c, u2, g, scope=sc)["y"]))
    assert y2[(1, 2)] == 1                              # R collapsed (INFRA) and no token → fallback


def test_gotham_seed_priority_rule():
    # the loader logs into y (= seed) and z within one fan-out unit (same stage ITT for two victims)
    Z = "192.168.17.99"
    rows = [(50, "192.168.0.100", Y, 23, "Telnet Brute Force"),
            (201, "192.168.33.12", Y, 23, "Ingress Tool Transfer"),
            (202, "192.168.33.12", Z, 23, "Ingress Tool Transfer"),
            (205, Y, "192.168.33.13", 80, "File Download")]
    rec = pl.DataFrame({"rec_id": [0, 1, 2, 3], "ts": [float(r[0]) for r in rows], "src": [r[1] for r in rows],
                        "dst": [r[2] for r in rows], "dport": [r[3] for r in rows]},
                       schema={"rec_id": pl.Int64, "ts": pl.Float64, "src": pl.Utf8, "dst": pl.Utf8, "dport": pl.Int32})
    labels = pl.DataFrame({"rec_id": [0, 1, 2, 3], "label": ["Malicious"] * 4, "detailed_label": [r[4] for r in rows]})
    u = units([50, 201, 205], src=["192.168.0.100", "192.168.33.12", Y]).with_columns(
        pl.Series("rec_ids", [[0], [1, 2], [3]], dtype=pl.List(pl.Int64)))
    d = build_gotham_gt(rec, u, labels, "mirai-infection")
    assert 1 in d.mixed_units                                       # default rule: 50/50 → MIXED
    sp = build_gotham_gt(rec, u, labels, "mirai-infection", seeds=[Y], seed_priority=True)
    assert sp.unit_episode[1] == f"ITT({Y})" and 1 not in sp.mixed_units
    assert lab(u, sp)[(1, 2)][0] == 1                               # ITT(y) → DL(y) now observable
    other = build_gotham_gt(rec, u, labels, "mirai-infection", seeds=["192.168.20.77"], seed_priority=True)
    assert 1 in other.mixed_units                                   # seed not among its victims → MIXED
