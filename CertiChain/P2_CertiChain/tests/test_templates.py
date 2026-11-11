"""ChainForge templates (DESIGN §5.4.2): DAG validity, role declarations, lag caps ≤ 0.8 W."""
from __future__ import annotations

import numpy as np
import pytest

from certichain.chainforge import templates as TP


@pytest.mark.parametrize("tid", sorted(TP.TEMPLATES))
def test_template_is_valid_out_tree(tid):
    t = TP.TEMPLATES[tid]
    assert TP.validate(t) == []
    order = t.topo()
    pos = {n: i for i, n in enumerate(order)}
    for e in t.edges:
        assert pos[e.a] < pos[e.b]


def test_lag_laws_capped_at_08W():
    assert TP.LAG_CAP == pytest.approx(0.8 * 1800.0)
    for law in TP.LAGS.values():
        assert 0 < law.lo < law.hi <= TP.LAG_CAP + 1e-9
    rng = np.random.default_rng(0)
    for law in TP.LAGS.values():
        x = np.array([law.sample(rng) for _ in range(2000)])
        assert x.min() >= law.lo - 1e-9 and x.max() <= min(law.hi, TP.LAG_CAP) + 1e-9
        # log-uniform: the median sits at the geometric mean of the range
        assert np.median(np.log(x)) == pytest.approx(0.5 * (np.log(law.lo) + np.log(law.hi)), abs=0.25)


def test_design_lag_table():
    expect = {"scan_exploit": (5, 900), "exploit_backdoor": (5, 600), "password_backdoor": (10, 900),
              "backdoor_impact": (60, 1440), "pivot": (10, 600), "c2_flood": (30, 1440), "bf_itt": (1, 600),
              "itt_dl": (1, 30), "dl_cc_scan": (0.1, 120), "scan_rep": (10, 300), "rep_itt": (0.5, 60),
              "udpscan_amp": (60, 1440)}
    for k, (lo, hi) in expect.items():
        assert (TP.LAGS[k].lo, TP.LAGS[k].hi) == (lo, hi)


def test_ton_template_structure():
    bot = TP.TEMPLATES["T-BOT"]
    assert [e.a + ">" + e.b for e in bot.edges] == ["scanning>password", "password>backdoor", "backdoor>ddos"]
    piv = TP.TEMPLATES["T-PIVOT"]
    kinds = {(e.a, e.b): e.kind for e in piv.edges}
    assert kinds[("password", "scanning_2")] == "pivot"
    # pivot semantics: the victim X of password is the actor of scanning′, whose victim is Y
    assert piv.node("password").modes[0].as_dict() == {"actor": "A", "victim": "X"}
    assert piv.node("scanning_2").modes[0].as_dict() == {"actor": "X", "victim": "Y"}
    # password → backdoor on the SAME victim; backdoor → floods from the victim or the same C2
    assert TP.TEMPLATES["T-CRED"].node("backdoor").modes[0].as_dict() == {"actor": "C2", "victim": "X"}
    floods = {m.name: m.as_dict() for m in TP.TEMPLATES["T-C2DOS"].node("dos").modes}
    assert floods == {"from_victim": {"actor": "X"}, "from_c2": {"actor": "C2", "victim": "X"}}
    assert set(TP.TEMPLATES["T-WEBI"].types()) == {"scanning", "injection", "backdoor", "ransomware"}
    assert set(TP.TEMPLATES["T-WEBX"].types()) == {"scanning", "xss", "backdoor"}


def test_allowed_templates_per_variant():
    assert TP.allowed_templates("cf_ton", "train") == TP.TON_TEMPLATES
    assert TP.allowed_templates("cf_ton", "caltest") == TP.TON_TEMPLATES
    assert set(TP.allowed_templates("cf_ton_h", "train")) == {"T-BOT", "T-C2DOS", "T-PIVOT", "T-CRED"}
    assert set(TP.allowed_templates("cf_ton_h", "dev")) == {"T-BOT", "T-C2DOS", "T-PIVOT", "T-CRED"}
    assert set(TP.allowed_templates("cf_ton_h", "caltest")) == {"T-WEBI", "T-WEBX"}
    assert "T-AMP" in TP.allowed_templates("cf_gotham", "train")
    assert "T-AMP" not in TP.allowed_templates("cf_gotham", "caltest")
    # held-out types never occur in CF-TON-H train templates
    tr = {t for tid in TP.allowed_templates("cf_ton_h", "train") for t in TP.TEMPLATES[tid].types()}
    assert not tr & {"injection", "xss"}


def test_mirai_roles_and_hidden_report():
    m = TP.TEMPLATES["T-MIRAI"]
    assert m.node("REP0").hidden
    roles = {r for n in m.nodes for mo in n.modes for _, r in mo.binds}
    assert {"SCANNER", "LOADER", "DLSRV", "CNC", "REPSRV", "X", "Y"} <= roles
    assert any(e.kind == "pivot" for e in m.edges)


def test_d2_truncations_contain_infra():
    for tid in TP.TON_TEMPLATES:
        t = TP.TEMPLATES[tid]
        opts = t.subtree_chain_options()
        if not t.infra_nodes():
            assert opts == []
            continue
        assert opts
        for o in opts:
            assert len(o) >= 2 and len(o) < len(t.nodes)
            assert any(t.node(n).infra for n in o)
            # connected: every non-top node's parent is inside
            par = t.parents()
            tops = [n for n in o if not any(e.a in o for e in par[n])]
            assert len(tops) == 1


def test_edge_lag_law_for_decoys():
    assert TP.edge_lag_law("scanning", "password").name == "scan_exploit"
    assert TP.edge_lag_law("backdoor", "ddos").name == "backdoor_impact"
    assert TP.edge_lag_law("xss", "backdoor").name == "exploit_backdoor"
    assert TP.edge_lag_law("dos", "scanning").name == "default"


def test_gotham_d2_truncations_never_infect_the_pivot_victim():
    m = TP.TEMPLATES["T-MIRAI"]
    opts = m.subtree_chain_options()
    assert opts
    for o in opts:
        for n in o:
            for mo in m.node(n).modes:
                for slot, role in mo.binds:
                    assert not (role == "Y" and slot != "target")
    assert any("SCAN" in o for o in opts)          # a parallel bot may scan (the seed), never infect Y
    for tid in ("T-MDOS", "T-MERLIN"):
        assert TP.TEMPLATES[tid].subtree_chain_options()
    assert TP.TEMPLATES["T-AMP"].subtree_chain_options() == []
