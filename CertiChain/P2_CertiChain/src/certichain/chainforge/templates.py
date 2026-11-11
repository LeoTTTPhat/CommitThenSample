"""ChainForge-IoT campaign templates (DESIGN §5.4.2) — DAGs, role bindings, lag laws.

A template is an out-tree over *nodes* (one library episode each). Edges are direct
enabling transitions (→) or pivot transitions (⇒: the victim becomes the actor of a
sub-chain); both are GT transitions T* (DESIGN §5.2.1).

Role binding (how a library episode is spliced into an incident, chainforge/rewrite.py):
every library episode exposes source-IP *slots* (TON: ``actor`` = the 192.168.1.30-39
attacker, ``victim`` = its primary internal victim; Gotham: per-label slots such as
``victim``/``server``). A node binds slots to incident *roles*:

    A    the campaign's external attacker               (TEST-NET address)
    C2   the campaign's external C2 / backdoor endpoint (TEST-NET address; shared by D2)
    X    the campaign victim (first campaign: the seed device)  (background device 10.0.N.y)
    Y    the pivot victim (T-PIVOT / T-MIRAI propagation)       (background device)
    LOADER, DLSRV, CNC, REPSRV, SCANNER   T-MIRAI's distinct role IPs (TEST-NET)

``modes`` lists alternative bindings with probabilities (e.g. backdoor → dos/ddos launched
FROM the compromised victim, or from the same C2 against it). A node may be ``hidden``
(no records: T-MIRAI's REP0).

Lags (gap from the parent's last record start to the child's first record start) are
LogUniform on the §5.4.2 ranges, all capped at 0.8·W = 24 min.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

W_CAND = 1800.0                    # Alg. 2 candidate window W (s)
LAG_CAP = 0.8 * W_CAND             # 1440 s = 24 min


@dataclass(frozen=True)
class LagLaw:
    name: str
    lo: float
    hi: float

    def sample(self, rng) -> float:
        lo, hi = float(self.lo), float(min(self.hi, LAG_CAP))
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


# §5.4.2 lag table (seconds)
LAGS: dict[str, LagLaw] = {l.name: l for l in (
    LagLaw("scan_exploit", 5.0, 900.0),        # scan → exploit/brute
    LagLaw("exploit_backdoor", 5.0, 600.0),    # exploit → backdoor
    LagLaw("password_backdoor", 10.0, 900.0),  # password → backdoor
    LagLaw("backdoor_impact", 60.0, 1440.0),   # backdoor → ransom / dos / ddos
    LagLaw("pivot", 10.0, 600.0),              # password ⇒ scanning′
    LagLaw("c2_flood", 30.0, 1440.0),          # C2 / CC → floods
    LagLaw("bf_itt", 1.0, 600.0),              # BF0 → ITT
    LagLaw("itt_dl", 1.0, 30.0),               # ITT → DL
    LagLaw("dl_cc_scan", 0.1, 120.0),          # DL → CC / SCAN
    LagLaw("scan_rep", 10.0, 300.0),           # SCAN → REP
    LagLaw("rep_itt", 0.5, 60.0),              # REP → ITT′
    LagLaw("udpscan_amp", 60.0, 1440.0),       # UDP-scan → amp
    LagLaw("default", 5.0, 1440.0),            # decoy anchors without a table entry (hull of the TON rows)
)}


@dataclass(frozen=True)
class Mode:
    binds: tuple[tuple[str, str], ...]         # (slot, role)
    p: float = 1.0
    name: str = "default"

    def as_dict(self) -> dict[str, str]:
        return dict(self.binds)


@dataclass(frozen=True)
class Node:
    id: str
    type: str                                  # library episode type
    modes: tuple[Mode, ...]
    hidden: bool = False                       # no records (declared hidden step)
    infra: bool = False                        # contacts the campaign's shared external infra (D2 anchor)


@dataclass(frozen=True)
class Edge:
    a: str
    b: str
    lag: str
    kind: str = "enable"                       # enable (→) | pivot (⇒)


@dataclass(frozen=True)
class Template:
    id: str
    dataset: str                               # ton | gotham
    group: str                                 # train_h | held_out | gotham | gotham_train_only
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    roles: tuple[str, ...] = field(default=())  # incident roles the template needs

    # ------------------------------------------------------------------ structure
    def node(self, nid: str) -> Node:
        for n in self.nodes:
            if n.id == nid:
                return n
        raise KeyError(nid)

    def parents(self) -> dict[str, list[Edge]]:
        out: dict[str, list[Edge]] = {n.id: [] for n in self.nodes}
        for e in self.edges:
            out[e.b].append(e)
        return out

    def children(self) -> dict[str, list[Edge]]:
        out: dict[str, list[Edge]] = {n.id: [] for n in self.nodes}
        for e in self.edges:
            out[e.a].append(e)
        return out

    def roots(self) -> list[str]:
        par = self.parents()
        return [n.id for n in self.nodes if not par[n.id]]

    def topo(self) -> list[str]:
        """Topological order (Kahn; ties in declaration order). Raises on a cycle."""
        par = {n.id: {e.a for e in self.parents()[n.id]} for n in self.nodes}
        order, ready = [], [n.id for n in self.nodes if not par[n.id]]
        done: set[str] = set()
        while ready:
            v = ready.pop(0)
            order.append(v)
            done.add(v)
            for n in self.nodes:
                if n.id not in done and n.id not in ready and n.id not in order and par[n.id] <= done:
                    ready.append(n.id)
        if len(order) != len(self.nodes):
            raise ValueError(f"{self.id}: not a DAG")
        return order

    def types(self, observable_only: bool = True) -> set[str]:
        return {n.type for n in self.nodes if not (observable_only and n.hidden)}

    def infra_nodes(self) -> list[str]:
        return [n.id for n in self.nodes if n.infra and not n.hidden]

    def subtree_chain_options(self, min_nodes: int = 2, exclude_roles: tuple[str, ...] = ("Y",)) -> list[tuple[str, ...]]:
        """Truncations for a D2 parallel infection: connected node sets (closed under the
        parent of every non-top node) that contain an infra node, have ≥ min_nodes nodes, are
        strictly smaller than the template (unless the template itself has min_nodes) and use no
        node binding one of ``exclude_roles`` (a D2 has one device of its own)."""
        par = {n.id: [e.a for e in self.parents()[n.id]] for n in self.nodes}
        # a node is excluded when it binds an excluded role to a non-"target" slot (a D2 may scan the
        # seed — its scan target slot is bound to the seed by the generator — but never infects Y)
        ids = [n.id for n in self.nodes if not self.node(n.id).hidden
               and not any(r in exclude_roles and slot != "target" for m in n.modes for slot, r in m.binds)]
        opts: set[tuple[str, ...]] = set()
        n_all = len(ids)
        # enumerate subsets (templates are tiny: ≤ 11 nodes)
        from itertools import combinations
        for k in range(min_nodes, n_all + 1):
            for comb in combinations(ids, k):
                s = set(comb)
                if not any(self.node(v).infra for v in s):
                    continue
                tops = [v for v in s if not any(p in s for p in par[v])]
                if len(tops) != 1:
                    continue
                if k == len([n for n in self.nodes if not n.hidden]) and k > min_nodes:
                    continue
                opts.add(tuple(v for v in self.topo() if v in s))
        return sorted(opts)


# --------------------------------------------------------------------------- TON
def _m(*binds: tuple[str, str], p: float = 1.0, name: str = "default") -> Mode:
    return Mode(binds=tuple(binds), p=p, name=name)


AV = (_m(("actor", "A"), ("victim", "X")),)                       # external attacker → victim
CV = (_m(("actor", "C2"), ("victim", "X")),)                      # victim ↔ C2 (reverse shell, SMB)
XY = (_m(("actor", "X"), ("victim", "Y")),)                       # pivot: victim attacks Y
FLOOD = (_m(("actor", "X"), p=0.5, name="from_victim"),           # compromised victim floods targets
         _m(("actor", "C2"), ("victim", "X"), p=0.5, name="from_c2"))  # the same C2 floods the victim

T_BOT = Template("T-BOT", "ton", "train_h",
                 nodes=(Node("scanning", "scanning", AV), Node("password", "password", AV),
                        Node("backdoor", "backdoor", CV, infra=True), Node("ddos", "ddos", FLOOD)),
                 edges=(Edge("scanning", "password", "scan_exploit"), Edge("password", "backdoor", "password_backdoor"),
                        Edge("backdoor", "ddos", "backdoor_impact")), roles=("A", "C2", "X"))
T_C2DOS = Template("T-C2DOS", "ton", "train_h",
                   nodes=(Node("backdoor", "backdoor", CV, infra=True), Node("dos", "dos", FLOOD),
                          Node("ddos", "ddos", FLOOD)),
                   edges=(Edge("backdoor", "dos", "backdoor_impact"), Edge("backdoor", "ddos", "backdoor_impact")),
                   roles=("C2", "X"))
T_PIVOT = Template("T-PIVOT", "ton", "train_h",
                   nodes=(Node("scanning", "scanning", AV), Node("password", "password", AV),
                          Node("scanning_2", "scanning", XY), Node("password_2", "password", XY)),
                   edges=(Edge("scanning", "password", "scan_exploit"),
                          Edge("password", "scanning_2", "pivot", kind="pivot"),
                          Edge("scanning_2", "password_2", "scan_exploit")), roles=("A", "X", "Y"))
T_CRED = Template("T-CRED", "ton", "train_h",
                  nodes=(Node("password", "password", AV), Node("backdoor", "backdoor", CV, infra=True),
                         Node("ransomware", "ransomware", CV)),
                  edges=(Edge("password", "backdoor", "password_backdoor"),
                         Edge("backdoor", "ransomware", "backdoor_impact")), roles=("A", "C2", "X"))
T_WEBI = Template("T-WEBI", "ton", "held_out",
                  nodes=(Node("scanning", "scanning", AV), Node("injection", "injection", AV),
                         Node("backdoor", "backdoor", CV, infra=True), Node("ransomware", "ransomware", CV)),
                  edges=(Edge("scanning", "injection", "scan_exploit"), Edge("injection", "backdoor", "exploit_backdoor"),
                         Edge("backdoor", "ransomware", "backdoor_impact")), roles=("A", "C2", "X"))
T_WEBX = Template("T-WEBX", "ton", "held_out",
                  nodes=(Node("scanning", "scanning", AV), Node("xss", "xss", AV),
                         Node("backdoor", "backdoor", CV, infra=True)),
                  edges=(Edge("scanning", "xss", "scan_exploit"), Edge("xss", "backdoor", "exploit_backdoor")),
                  roles=("A", "C2", "X"))

# --------------------------------------------------------------------------- Gotham
# Slots of Gotham library episodes (chainforge/episodes.py, Gotham section):
#   victim  the infected device / bot (the device the activity belongs to)
#   server  the infrastructure endpoint (scanner, loader, download server, CNC, report server, Merlin C2,
#           UDP scanner, reflection peer)
#   target  propagation scans only: one internal host the bot scanned (the host it later reported, when
#           known) -- bound to the pivot victim Y so that "X scanned Y" is visible
_BF0 = (_m(("server", "SCANNER"), ("victim", "X")),)                 # scanner -> victim telnet brute force
_ITT = (_m(("server", "LOADER"), ("victim", "X")),)                  # loader -> victim
_DL = (_m(("victim", "X"), ("server", "DLSRV")),)                    # victim -> download server
_CC = (_m(("victim", "X"), ("server", "CNC")),)                      # victim -> CNC
_SCAN = (_m(("victim", "X"), ("target", "Y")),)                      # victim scans (incl. Y)
_REP = (_m(("victim", "X"), ("server", "REPSRV")),)                  # victim -> report server
_ITT2 = (_m(("server", "LOADER"), ("victim", "Y")),)
_DL2 = (_m(("victim", "Y"), ("server", "DLSRV")),)
_CC2 = (_m(("victim", "Y"), ("server", "CNC")),)
_SCAN2 = (_m(("victim", "Y"),),)
_FLOOD_X = (_m(("victim", "X"),),)                                   # bot floods targets
_C2X = (_m(("victim", "X"), ("server", "C2")),)                      # bot <-> Merlin C2
_USCAN = (_m(("victim", "X"), ("server", "USCANNER")),)              # UDP scanner <-> CoAP device
_AMP = (_m(("victim", "X"), ("server", "AMPPEER")),)                 # spoofed peer -> CoAP device

T_MIRAI = Template("T-MIRAI", "gotham", "gotham",
                   nodes=(Node("BF0", "mirai_bf", _BF0), Node("REP0", "mirai_report", (), hidden=True),
                          Node("ITT", "mirai_itt", _ITT, infra=True), Node("DL", "mirai_dl", _DL, infra=True),
                          Node("CC", "mirai_cc", _CC, infra=True), Node("SCAN", "mirai_scan", _SCAN),
                          Node("REP", "mirai_report", _REP, infra=True), Node("ITT_2", "mirai_itt", _ITT2, infra=True),
                          Node("DL_2", "mirai_dl", _DL2, infra=True), Node("CC_2", "mirai_cc", _CC2, infra=True),
                          Node("SCAN_2", "mirai_scan", _SCAN2)),
                   edges=(Edge("BF0", "REP0", "bf_itt"), Edge("REP0", "ITT", "rep_itt"),
                          Edge("ITT", "DL", "itt_dl"), Edge("DL", "CC", "dl_cc_scan"), Edge("DL", "SCAN", "dl_cc_scan"),
                          Edge("SCAN", "REP", "scan_rep"), Edge("REP", "ITT_2", "rep_itt", kind="pivot"),
                          Edge("ITT_2", "DL_2", "itt_dl"), Edge("DL_2", "CC_2", "dl_cc_scan"),
                          Edge("DL_2", "SCAN_2", "dl_cc_scan")),
                   roles=("SCANNER", "LOADER", "DLSRV", "CNC", "REPSRV", "X", "Y"))
T_MDOS = Template("T-MDOS", "gotham", "gotham",
                  nodes=(Node("CC", "mirai_dos_cc", _CC, infra=True), Node("UDP", "mirai_udpflood", _FLOOD_X),
                         Node("TCP", "mirai_tcpflood", _FLOOD_X), Node("GRE", "mirai_greflood", _FLOOD_X)),
                  edges=(Edge("CC", "UDP", "c2_flood"), Edge("CC", "TCP", "c2_flood"), Edge("CC", "GRE", "c2_flood")),
                  roles=("CNC", "X"))
T_MERLIN = Template("T-MERLIN", "gotham", "gotham",
                    nodes=(Node("C2", "merlin_c2", _C2X, infra=True), Node("ICMP", "merlin_icmpflood", _FLOOD_X),
                           Node("UDP", "merlin_udpflood", _FLOOD_X), Node("TCP", "merlin_tcpflood", _FLOOD_X)),
                    edges=(Edge("C2", "ICMP", "c2_flood"), Edge("C2", "UDP", "c2_flood"), Edge("C2", "TCP", "c2_flood")),
                    roles=("C2", "X"))
T_AMP = Template("T-AMP", "gotham", "gotham_train_only",
                 nodes=(Node("UDPSCAN", "udp_scan", _USCAN), Node("AMP", "amplification", _AMP)),
                 edges=(Edge("UDPSCAN", "AMP", "udpscan_amp"),), roles=("USCANNER", "AMPPEER", "X"))

TEMPLATES: dict[str, Template] = {t.id: t for t in (T_BOT, T_C2DOS, T_PIVOT, T_CRED, T_WEBI, T_WEBX,
                                                     T_MIRAI, T_MDOS, T_MERLIN, T_AMP)}
TON_TEMPLATES = ("T-BOT", "T-C2DOS", "T-PIVOT", "T-CRED", "T-WEBI", "T-WEBX")
TRAIN_H = ("T-BOT", "T-C2DOS", "T-PIVOT", "T-CRED")
HELD_OUT = ("T-WEBI", "T-WEBX")
GOTHAM_TEMPLATES = ("T-MIRAI", "T-MDOS", "T-MERLIN", "T-AMP")


def allowed_templates(variant: str, split_role: str, cfg: dict | None = None) -> tuple[str, ...]:
    """Templates per (variant, split role) (DESIGN §5.3/§5.4.2); a config may override."""
    cfg = cfg or {}
    over = (cfg.get("templates") or {}).get(split_role)
    if over:
        return tuple(over)
    pool = "train" if split_role in ("train", "dev") else "caltest"
    if variant == "cf_ton":
        return TON_TEMPLATES
    if variant == "cf_ton_h":
        return TRAIN_H if pool == "train" else HELD_OUT
    if variant in ("cf_gotham", "cf_gotham_t"):
        base = ("T-MIRAI", "T-MDOS", "T-MERLIN")
        return base + (("T-AMP",) if pool == "train" and variant == "cf_gotham" else ())
    raise ValueError(f"unknown variant {variant}")


def edge_lag_law(parent_type: str, child_type: str, kind: str = "enable") -> LagLaw:
    """Lag law for an arbitrary (parent type → child type) pair (decoy anchoring)."""
    if kind == "pivot":
        return LAGS["pivot"]
    exploit = {"password", "injection", "xss"}
    if parent_type == "scanning" and child_type in exploit:
        return LAGS["scan_exploit"]
    if parent_type in ("injection", "xss") and child_type == "backdoor":
        return LAGS["exploit_backdoor"]
    if parent_type == "password" and child_type == "backdoor":
        return LAGS["password_backdoor"]
    if parent_type == "backdoor" and child_type in ("ransomware", "dos", "ddos"):
        return LAGS["backdoor_impact"]
    if parent_type == "password" and child_type == "scanning":
        return LAGS["pivot"]
    return LAGS["default"]


def validate(t: Template) -> list[str]:
    """Structural checks used by tests/test_templates.py; returns a list of problems."""
    errs = []
    ids = [n.id for n in t.nodes]
    if len(set(ids)) != len(ids):
        errs.append("duplicate node ids")
    try:
        t.topo()
    except ValueError as e:
        errs.append(str(e))
    par = t.parents()
    for nid, es in par.items():
        if len(es) > 1:
            errs.append(f"node {nid} has {len(es)} parents (templates are out-trees)")
    for e in t.edges:
        if e.a not in ids or e.b not in ids:
            errs.append(f"edge {e} references unknown node")
        if e.lag not in LAGS:
            errs.append(f"edge {e} has unknown lag law")
        elif LAGS[e.lag].hi > LAG_CAP + 1e-9:
            errs.append(f"edge {e} lag hi {LAGS[e.lag].hi} > cap {LAG_CAP}")
        if e.kind not in ("enable", "pivot"):
            errs.append(f"edge {e} has unknown kind")
    for n in t.nodes:
        if not n.hidden:
            if not n.modes:
                errs.append(f"node {n.id} has no binding mode")
            elif abs(sum(m.p for m in n.modes) - 1.0) > 1e-9:
                errs.append(f"node {n.id} mode probabilities do not sum to 1")
        roles_used = {r for m in n.modes for _, r in m.binds}
        if t.roles and not roles_used <= set(t.roles):
            errs.append(f"node {n.id} uses roles {roles_used - set(t.roles)} not declared")
    if len(t.roots()) != 1:
        errs.append(f"{t.id} has {len(t.roots())} roots")
    return errs


__all__ = ["W_CAND", "LAG_CAP", "LagLaw", "LAGS", "Mode", "Node", "Edge", "Template", "TEMPLATES", "TON_TEMPLATES",
           "TRAIN_H", "HELD_OUT", "GOTHAM_TEMPLATES", "allowed_templates", "edge_lag_law", "validate"]
