"""ATT&CK v20.0 knowledge base for the LLM prompt (DESIGN §3.4.3 "Retrieval", Appendix C).

Three things, all deterministic and label-free at scoring time:

1. ``glossary()`` - the ≈ 40 network-visible techniques shown in the cached prompt prefix.
   IDs = Appendix C hint column ∪ dataset label → technique map (``LABEL_TECHNIQUES``,
   used ONLY to choose which IDs the glossary explains) ∪ a fixed a-priori list of extra
   network-visible techniques (``EXTRA_GLOSSARY_IDS``). Names/tactics come from the local
   STIX 2.1 bundles ``data/attack/{enterprise,ics}-attack.json`` (read as JSON data only;
   sha256 checked against data/attack/MANIFEST.md).
2. ``TYPE_TECHNIQUES`` - the unit-type → technique-ID table of Appendix C.
3. ``hints()`` - a tiny BM25 over the glossary technique names ranks the table IDs of a unit
   (and, when packet content flags are present, may add glossary IDs); the top-2 IDs go into
   the evidence card ("| hints T1105, T1059").
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ATTACK_DIR = PROJECT_ROOT / "data" / "attack"
BUNDLES = (("enterprise", "enterprise-attack.json"), ("ics", "ics-attack.json"))
# sha256 from data/attack/MANIFEST.md (ATT&CK v20.0, downloaded 2026-10-18)
EXPECTED_SHA256 = {
    "enterprise-attack.json": "81113ba98afa2928f3dcff62022539f7ce30c315c32e14422f11ff95ce534352",
    "ics-attack.json": "c9ca0ce523a1b5e512060a390415e8aefe8f6d32c5f82d7de81264960c1e1183",
}
ATTACK_VERSION = "20.0"

# ----------------------------------------------------------------------------- tables
# Appendix C hint column (base type → IDs, in the order printed in DESIGN).
_BRUTE = ("T1110.001",)
_FLOOD = ("T1498", "T1499", "T0814")
TYPE_TECHNIQUES: dict[str, tuple[str, ...]] = {
    "SCAN_H": ("T1046", "T1595.001"),
    "SCAN_V": ("T1046",),
    "SWEEP_ICMP": ("T1018", "T0846"),
    **{f"BRUTE_{p}": _BRUTE for p in ("TELNET", "SSH", "FTP", "MQTT", "HTTP")},
    "WEB_ATTACK": ("T1190",),
    "HTTP_DOWNLOAD": ("T1105",),
    "HTTP_UPLOAD": ("T1105", "T1505.003"),
    "C2_PERIODIC": ("T1071", "T1571", "T1573"),
    "INTERACTIVE_SESSION": ("T1059", "T1021", "T1105"),
    **{f"FLOOD_{p}": _FLOOD for p in ("SYN", "UDP", "ICMP", "HTTP", "GRE", "ACK")},
    "AMPLIFICATION": ("T1498.002",),
    "DNS_ANOMALY": ("T1568", "T1071.004"),
    "EXFIL_BULK": ("T1041", "T1048"),
    "REPORT_SMALL": ("T1041",),
    # protocol-based / benign types: no hints
    "MQTT_PUBSUB": (), "COAP": (), "RTSP_STREAM": (), "DNS_ACTIVITY": (), "TLS_SESSION": (),
    "BENIGN_GENERIC_TCP": (), "BENIGN_GENERIC_UDP": (),
}
UNIT_TYPES = tuple(TYPE_TECHNIQUES)

# Dataset label → technique map (Appendix C). GT/benchmark side ONLY: here it merely decides
# which technique IDs the glossary explains; it is never consulted per candidate.
LABEL_TECHNIQUES: dict[str, dict[str, tuple[str, ...]]] = {
    "TON_IoT": {"scanning": ("T1046",), "password": ("T1110",), "injection": ("T1190",),
                "xss": ("T1190",), "backdoor": ("T1219", "T1071"), "ransomware": ("T1486",),
                "dos": ("T1499",), "ddos": ("T1498",)},
    "Gotham": {"TCP/UDP Scan": ("T1046",), "Telnet Brute Force": ("T1110.001",),
               "Ingress Tool Transfer": ("T1105",), "File Download": ("T1105",),
               "C&C": ("T1071", "T1571"), "Reporting": ("T1041",), "floods": ("T1498",),
               "CoAP Amplification": ("T1498.002",), "Merlin C&C": ("T1071.001", "T1573")},
    "IoT-23": {"PartOfAHorizontalPortScan": ("T1046",), "C&C": ("T1071",),
               "FileDownload": ("T1105",), "DDoS": ("T1498",), "Attack": ("T1110", "T1190"),
               "Okiru": ("T1071",)},
}

# Fixed a-priori extras (network-visible, IoT-relevant), chosen before any data was seen,
# to bring the glossary to 40 entries. Never tuned on dev/cal/test.
EXTRA_GLOSSARY_IDS: tuple[str, ...] = (
    "T1595", "T1595.002", "T1078.001", "T1133", "T1210", "T1570", "T1059.004", "T1021.004",
    "T1095", "T1572", "T1568.002", "T1498.001", "T1499.002", "T0866", "T0886",
)

# kill-chain order used to sort the glossary (Enterprise v20 then ICS-only tactics)
TACTIC_ORDER: tuple[str, ...] = (
    "reconnaissance", "resource-development", "initial-access", "execution", "persistence",
    "privilege-escalation", "stealth", "defense-impairment", "evasion", "credential-access",
    "discovery", "lateral-movement", "collection", "command-and-control", "exfiltration",
    "inhibit-response-function", "impair-process-control", "impact",
)

_TID_RE = re.compile(r"^T\d{4}(\.\d{3})?$")
_PRINTABLE = re.compile(r"[^\x20-\x7e]")


def glossary_ids() -> tuple[str, ...]:
    """Appendix C hints ∪ label map ∪ extras, de-duplicated in first-seen order."""
    seen: dict[str, None] = {}
    for ids in TYPE_TECHNIQUES.values():
        seen.update(dict.fromkeys(ids))
    for m in LABEL_TECHNIQUES.values():
        for ids in m.values():
            seen.update(dict.fromkeys(ids))
    seen.update(dict.fromkeys(EXTRA_GLOSSARY_IDS))
    return tuple(seen)


@dataclass(frozen=True)
class Technique:
    tid: str
    name: str            # full name; sub-techniques as "Parent: Sub"
    tactics: tuple[str, ...]
    domain: str          # "enterprise" | "ics"

    def line(self) -> str:
        ics = " (ICS)" if self.domain == "ics" else ""
        return f"{self.tid} {self.name}{ics} [{'/'.join(self.tactics)}]"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _clean(s: str, n: int = 90) -> str:
    return re.sub(r"\s+", " ", _PRINTABLE.sub("", s)).strip()[:n]


@functools.lru_cache(maxsize=4)
def load_techniques(attack_dir: str | None = None, verify_sha: bool = True) -> dict[str, Technique]:
    """All active (not revoked, not deprecated) attack-patterns of both bundles, by ID."""
    root = Path(attack_dir) if attack_dir else ATTACK_DIR
    raw: dict[str, tuple[str, tuple[str, ...], str]] = {}
    for domain, fn in BUNDLES:
        path = root / fn
        if verify_sha and fn in EXPECTED_SHA256 and _sha256(path) != EXPECTED_SHA256[fn]:
            raise ValueError(f"{path}: sha256 differs from data/attack/MANIFEST.md")
        with open(path, "rb") as f:
            bundle = json.load(f)                       # data only
        for o in bundle.get("objects", []):
            if o.get("type") != "attack-pattern" or o.get("revoked") or o.get("x_mitre_deprecated"):
                continue
            tid = next((r.get("external_id") for r in o.get("external_references", [])
                        if r.get("source_name") in ("mitre-attack", "mitre-ics-attack")
                        and _TID_RE.match(str(r.get("external_id", "")))), None)
            if tid is None or tid in raw:
                continue
            tactics = tuple(dict.fromkeys(str(k.get("phase_name", "")) for k in o.get("kill_chain_phases", [])))
            raw[tid] = (_clean(str(o.get("name", ""))), tactics, domain)
    out: dict[str, Technique] = {}
    for tid, (name, tactics, domain) in raw.items():
        if "." in tid and tid.split(".")[0] in raw:
            name = f"{raw[tid.split('.')[0]][0]}: {name}"
        out[tid] = Technique(tid, name, tactics, domain)
    return out


def _tactic_rank(t: Technique) -> int:
    ranks = [TACTIC_ORDER.index(x) for x in t.tactics if x in TACTIC_ORDER]
    return min(ranks) if ranks else len(TACTIC_ORDER)


@functools.lru_cache(maxsize=4)
def glossary(attack_dir: str | None = None) -> tuple[Technique, ...]:
    """The prompt glossary, sorted by (first tactic in kill-chain order, ID)."""
    kb = load_techniques(attack_dir)
    missing = [t for t in glossary_ids() if t not in kb]
    if missing:
        raise KeyError(f"glossary IDs not active in ATT&CK v{ATTACK_VERSION}: {missing}")
    return tuple(sorted((kb[t] for t in glossary_ids()), key=lambda t: (_tactic_rank(t), t.tid)))


def glossary_lines(attack_dir: str | None = None) -> list[str]:
    return [t.line() for t in glossary(attack_dir)]


def base_type(unit_type: str) -> str:
    return unit_type if unit_type in TYPE_TECHNIQUES else unit_type.upper()


def techniques_for_type(unit_type: str) -> tuple[str, ...]:
    return TYPE_TECHNIQUES.get(base_type(unit_type), ())


# ----------------------------------------------------------------------------- BM25
_STOP = frozenset("a an and as for from in of on or over the to with via".split())


def _stem(w: str) -> str:
    """Tiny suffix stripper: services/service -> servic, scanning/scan -> scan."""
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    for suf in ("ing", "ion", "ed"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiou":   # scann -> scan
        w = w[:-1]
    return w


def bm25_tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


class BM25:
    """Okapi BM25 (k1 = 1.2, b = 0.75) over a small fixed corpus."""

    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avgdl = sum(self.len) / max(1, self.n)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1.0 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for q in query:
                f = tf.get(q, 0)
                if f:
                    s += self.idf[q] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


# Query vocabulary: type-name words → technique-name words (fixed, label-free).
TYPE_QUERY_TERMS = {
    "scan": "scanning discovery service", "sweep": "remote system discovery", "brute": "brute force password",
    "web": "exploit application", "attack": "exploit", "download": "ingress tool transfer",
    "upload": "transfer web shell", "c2": "command channel application protocol", "periodic": "non standard port",
    "interactive": "command interpreter remote services", "session": "remote services", "flood": "denial service flood",
    "amplification": "reflection amplification", "dns": "dns dynamic resolution", "anomaly": "generation algorithms",
    "exfil": "exfiltration", "bulk": "exfiltration alternative protocol", "report": "exfiltration c2 channel",
    "telnet": "remote services", "ssh": "ssh remote services", "http": "web protocols",
}
# content_flags bits (io/schema.py UNIT_SCHEMA): 1 exec-download, 2 injection regex, 4 tool UA,
# 8 DNS NXDOMAIN-heavy, 16 high-entropy qname
FLAG_QUERY_TERMS = {1: "ingress tool transfer", 2: "exploit public facing application", 4: "ingress tool transfer",
                    8: "dynamic resolution dns", 16: "domain generation algorithms"}


@functools.lru_cache(maxsize=4)
def _bm25(attack_dir: str | None = None) -> tuple[BM25, tuple[str, ...]]:
    g = glossary(attack_dir)
    return BM25([bm25_tokens(t.name) for t in g]), tuple(t.tid for t in g)


def unit_query(unit_type: str, svc_class: str = "", proto: str = "", content_flags: int = 0) -> list[str]:
    words = [w for w in re.split(r"[^a-z0-9]+", base_type(unit_type).lower()) if w]
    words += [w for w in re.split(r"[^a-z0-9]+", (svc_class or "").lower()) if w]
    terms: list[str] = []
    for w in words:
        terms += bm25_tokens(w + " " + TYPE_QUERY_TERMS.get(w, ""))
    for bit, txt in FLAG_QUERY_TERMS.items():
        if int(content_flags or 0) & bit:
            terms += bm25_tokens(txt)
    return terms


def hints(unit_type: str, svc_class: str = "", proto: str = "", content_flags: int = 0, k: int = 2,
          attack_dir: str | None = None) -> list[str]:
    """Top-k technique IDs for a unit (deterministic).

    The unit type's Appendix C IDs are ranked by BM25(query, technique name) (ties: table
    order). Only when packet content flags are set and fewer than k IDs were found, the
    remaining slots are filled with the best-scoring other glossary entries (score > 0, ties
    by ID). Protocol-based/benign types without content flags get no hints.
    """
    bm, ids = _bm25(attack_dir)
    sc = dict(zip(ids, bm.scores(unit_query(unit_type, svc_class, proto, content_flags))))
    table = [t for t in techniques_for_type(unit_type) if t in sc]
    out = sorted(table, key=lambda t: (-sc[t], table.index(t)))[:k]
    flags = int(content_flags or 0)
    if len(out) < k and flags:
        q = []
        for bit, txt in FLAG_QUERY_TERMS.items():
            if flags & bit:
                q += bm25_tokens(txt)
        fsc = dict(zip(ids, bm.scores(q)))
        rest = sorted((t for t in ids if t not in out and fsc[t] > 0), key=lambda t: (-fsc[t], t))
        out += rest[: k - len(out)]
    return out
