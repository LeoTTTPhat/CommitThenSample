"""ATT&CK v20 glossary, unit-type table and BM25 hints."""
from __future__ import annotations

import re

from certichain.features import attack_kb as kb


def test_glossary_has_40_active_entries_covering_appendix_c():
    g = kb.glossary()
    ids = [t.tid for t in g]
    assert len(g) == 40 and len(set(ids)) == 40
    need = {t for v in kb.TYPE_TECHNIQUES.values() for t in v}
    need |= {t for m in kb.LABEL_TECHNIQUES.values() for v in m.values() for t in v}
    assert need <= set(ids)
    active = kb.load_techniques()
    assert all(t in active for t in ids)


def test_glossary_lines_format_and_ascii():
    lines = kb.glossary_lines()
    assert "T1110.001 Brute Force: Password Guessing [credential-access]" in lines
    assert "T0814 Denial of Service (ICS) [inhibit-response-function]" in lines
    for ln in lines:
        assert re.match(r"^T\d{4}(\.\d{3})? [ -~]+ \[[a-z/-]+\]$", ln), ln
    assert kb.glossary_lines() == lines                              # deterministic


def test_type_table_matches_appendix_c():
    t = kb.TYPE_TECHNIQUES
    assert t["SCAN_H"] == ("T1046", "T1595.001") and t["SWEEP_ICMP"] == ("T1018", "T0846")
    assert t["BRUTE_SSH"] == ("T1110.001",) and t["FLOOD_GRE"] == ("T1498", "T1499", "T0814")
    assert t["INTERACTIVE_SESSION"] == ("T1059", "T1021", "T1105")
    assert t["MQTT_PUBSUB"] == () and t["BENIGN_GENERIC_UDP"] == ()


def test_hints_top2_from_table_deterministic():
    for typ, ids in kb.TYPE_TECHNIQUES.items():
        h = kb.hints(typ, "telnet", "tcp", 0)
        assert len(h) <= 2 and set(h) <= set(ids)
        assert h == kb.hints(typ, "telnet", "tcp", 0)
    assert kb.hints("BRUTE_TELNET", "telnet", "tcp") == ["T1110.001"]
    assert set(kb.hints("SCAN_H", "telnet", "tcp")) == {"T1046", "T1595.001"}
    assert kb.hints("BENIGN_GENERIC_TCP", "http", "tcp") == []
    assert kb.hints("UNKNOWN_TYPE") == []


def test_hints_content_flags_fill_from_bm25():
    h = kb.hints("HTTP_DOWNLOAD", "http", "tcp", content_flags=1)
    assert h[0] == "T1105" and len(h) == 2
    assert "T1190" in kb.hints("MQTT_PUBSUB", "mqtt", "tcp", content_flags=2)


def test_bm25_basic():
    bm = kb.BM25([["brute", "forc"], ["network", "servic", "discovery"], ["remot", "servic"]])
    s = bm.scores(["discovery"])
    assert s[1] > 0 and s[0] == 0 and s[2] == 0
    s2 = bm.scores(["servic"])
    assert s2[2] > s2[1] > 0                                       # shorter doc wins at equal tf
    assert bm.scores([]) == [0.0, 0.0, 0.0]
    assert kb.bm25_tokens("Scanning services") == ["scan", "servic"]
