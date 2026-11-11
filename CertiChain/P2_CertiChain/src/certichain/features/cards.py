"""Evidence-card renderer for the LLM suffix (DESIGN §3.4.3, Appendix A suffix format).

    Incident clock starts at t0.
    U: t=+12:03 dur 45s | X1[ext] -> H3[camera,rtsp,telnet] tcp/23 telnet | 312 conns; 89% failed,
       34 established | out 9.1kB in 31kB | type BRUTE_TELNET | hints T1110.001
    V: t=+13:10 dur 2s | INFRA2 -> H3[camera,rtsp,telnet] tcp/23 telnet | 1 conn, established | ...
    Relation: V starts 22s after U ends; same victim H3; different source.

Inputs are a UNIT_SCHEMA row dict for U and for V, a CANDIDATE_SCHEMA row dict, and an
``IncidentContext`` (incident clock t0, role tokens, optional per-unit packet text such as HTTP
URIs/UAs that are not in UNIT_SCHEMA). Times are relative to t0, hosts are role tokens, and
every packet-derived string is sanitised (non-printables stripped, ASCII only, whitespace
collapsed, ⟦ ⟧ removed, IPv4 literals replaced by role tokens or "IP", ≤ 40 chars) and wrapped
in ⟦ ⟧. ``render_card`` enforces a hard budget (default 220 tokens for card + question,
measured with the scoring model's tokenizer) by dropping the least important fields first.
The prompt hash covers this file (llm/prompts.py: renderer_source()).
"""
from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from certichain.features import attack_kb
from certichain.features.roles import RoleMap, build_roles

REL_PIVOT, REL_REVERSE, REL_SAME_ACTOR, REL_SAME_VICTIM, REL_SHARED_EXT = 1, 2, 4, 8, 16   # io/schema.py
CARD_TOKEN_BUDGET = 220
MAX_TEXT_CHARS = 40
LQ, RQ = "⟦", "⟧"
FLAG_NAMES = {1: "exec-download", 2: "injection-pattern", 4: "tool-UA", 8: "NXDOMAIN-heavy", 16: "high-entropy-names"}
_NONPRINT = re.compile(r"[^\x20-\x7e]")
_WS = re.compile(r"\s+")
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")


# ----------------------------------------------------------------------------- formatting
def sanitize_text(s: Any, roles: RoleMap | None = None, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Untrusted packet text -> safe ASCII snippet (no ⟦ ⟧, no control chars, ≤ max_chars)."""
    t = str(s if s is not None else "")
    t = t.replace(LQ, " ").replace(RQ, " ")
    t = _WS.sub(" ", t)                       # newlines/tabs -> space (before stripping non-printables)
    t = _NONPRINT.sub("", t)                  # ASCII printable only
    t = _WS.sub(" ", t).strip()
    if roles is not None:
        t = _IPV4.sub(lambda m: roles.lookup(m.group(0)) or "IP", t)
    else:
        t = _IPV4.sub("IP", t)
    return t[:max_chars].rstrip()


def wrap_text(s: Any, roles: RoleMap | None = None) -> str:
    return f"{LQ}{sanitize_text(s, roles)}{RQ}"


def fmt_clock(sec: float) -> str:
    """Relative time vs t0: +mm:ss, +h:mm:ss (units before t0, if any: -mm:ss)."""
    sec = float(sec)
    sign = "-" if sec < 0 else "+"
    sec = int(math.floor(abs(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{h}:{m:02d}:{s:02d}" if h else f"{sign}{m:02d}:{s:02d}"


def fmt_dur(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 10:
        return f"{sec:.1f}s".replace(".0s", "s")
    if sec < 60:
        return f"{int(round(sec))}s"
    s = int(round(sec))
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def fmt_bytes(n: float) -> str:
    n = max(0.0, float(n or 0))
    if n < 1000:
        return f"{int(n)}B"
    for unit, div in (("kB", 1e3), ("MB", 1e6), ("GB", 1e9)):
        v = n / div
        if v < 1000 or unit == "GB":
            return f"{v:.1f}{unit}" if v < 10 else f"{int(round(v))}{unit}"
    return f"{n:.0f}B"


def fmt_count(n: float) -> str:
    return f"{int(n):,}"


def _g(d: Any, k: str, default=None):
    v = d.get(k, default) if isinstance(d, Mapping) else getattr(d, k, default)
    if v is None:
        return default
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):      # numpy arrays / scalars
        v = v.tolist()
    if isinstance(v, float) and math.isnan(v):
        return default
    return v


# ----------------------------------------------------------------------------- context
@dataclass
class IncidentContext:
    t0: float                                   # incident clock start (case window t_0)
    roles: RoleMap
    unit_text: Mapping[int, Mapping[str, Sequence[str]]] = field(default_factory=dict)
    # unit_text[unit_id] may hold "uri", "ua", "host", "dns", "sni" lists (packet-derived, untrusted)


def build_context(units: Any, t0: float | None = None, unit_text: Mapping | None = None,
                  is_internal: Callable[[str], bool] | None = None) -> IncidentContext:
    if hasattr(units, "iter_rows"):
        units = list(units.iter_rows(named=True))
    units = list(units)
    if t0 is None:
        t0 = min(float(_g(u, "ts", 0.0)) for u in units) if units else 0.0
    return IncidentContext(float(t0), build_roles(units, is_internal), dict(unit_text or {}))


# ----------------------------------------------------------------------------- unit line
def _ports(u: Any) -> str:
    proto = str(_g(u, "proto", "") or "other")
    ports = [int(p) for p in (_g(u, "dports", []) or [])]
    nports = int(_g(u, "n_dports", len(ports)) or len(ports))
    if proto == "icmp" or not ports:
        return proto
    if nports <= 3:
        return f"{proto}/" + ",".join(str(p) for p in ports[:3])
    return f"{proto} {fmt_count(nports)} ports (" + ",".join(str(p) for p in ports[:3]) + ",...)"


def _dst_desc(u: Any, roles: RoleMap) -> str:
    dsts = [str(d) for d in (_g(u, "dsts", []) or [])]
    n = int(_g(u, "n_dst", len(dsts)) or len(dsts))
    if n <= 3 and len(dsts) == n and n > 0:
        return ", ".join(roles.token(d) for d in dsts)
    side = str(_g(u, "dir_class", "") or "").split("->")
    kind = side[1].strip() if len(side) == 2 else "?"
    kind = {"int": "int", "ext": "ext"}.get(kind, "mixed")
    pre = _g(u, "dst_prefixes", []) or []
    npre = len(set(pre)) if pre else len({".".join(d.split(".")[:3]) for d in dsts})
    return f"{fmt_count(n)} {kind} hosts in {npre} /24{'s' if npre != 1 else ''}"


def _conn(u: Any) -> str:
    n = int(_g(u, "n_records", 1) or 1)
    ff = float(_g(u, "fail_frac", 0.0) or 0.0)
    if n == 1:
        return "1 conn, " + ("failed" if ff >= 0.5 else "established")
    est = int(round(n * (1.0 - ff)))
    return f"{fmt_count(n)} conns; {int(round(100 * ff))}% failed, {fmt_count(est)} established"


def _text_fields(u: Any, ctx: IncidentContext, per_field: int, keep: Sequence[str]) -> list[str]:
    uid = int(_g(u, "unit_id", -1))
    extra = ctx.unit_text.get(uid, {}) if ctx.unit_text else {}
    src = {"http": list(extra.get("uri", []) or []), "ua": list(extra.get("ua", []) or []),
           "host": list(_g(u, "http_hosts", []) or []) + list(extra.get("host", []) or []),
           "dns": list(_g(u, "dns_names", []) or []) + list(extra.get("dns", []) or []),
           "sni": list(_g(u, "tls_snis", []) or []) + list(extra.get("sni", []) or [])}
    out = []
    for name in ("http", "ua", "host", "dns", "sni"):
        if name not in keep:
            continue
        vals = list(dict.fromkeys(s for s in (sanitize_text(v, ctx.roles) for v in src[name]) if s))[:per_field]
        if vals:
            out.append(f"{name} " + " ".join(f"{LQ}{v}{RQ}" for v in vals))
    return out


# detail levels: 4 = everything ... 0 = minimal; render_card steps down until the budget fits
_LEVELS = {
    4: dict(per_field=2, keep=("http", "ua", "host", "dns", "sni"), cv=True, hints=True, flags=True, bytes=True),
    3: dict(per_field=1, keep=("http", "ua", "host", "dns", "sni"), cv=True, hints=True, flags=True, bytes=True),
    2: dict(per_field=1, keep=("http", "ua", "dns"), cv=False, hints=True, flags=True, bytes=True),
    1: dict(per_field=1, keep=("http",), cv=False, hints=False, flags=True, bytes=True),
    0: dict(per_field=0, keep=(), cv=False, hints=False, flags=False, bytes=False),
}


def render_unit(tag: str, u: Any, ctx: IncidentContext, level: int = 4) -> str:
    L = _LEVELS[level]
    ts = float(_g(u, "ts", 0.0))
    dur = float(_g(u, "dur", float(_g(u, "te", ts)) - ts) or 0.0)
    parts = [f"{tag}: t={fmt_clock(ts - ctx.t0)} dur {fmt_dur(dur)}",
             f"{ctx.roles.token(str(_g(u, 'src', '')))} -> {_dst_desc(u, ctx.roles)} {_ports(u)} "
             f"{_g(u, 'svc_class', '') or ''}".rstrip()]
    conn = _conn(u)
    cv = _g(u, "start_cv", None)
    if L["cv"] and cv is not None and int(_g(u, "n_records", 1) or 1) >= 5:
        conn += f", start CV {float(cv):.2f}"
    parts.append(conn)
    if L["bytes"]:
        parts.append(f"out {fmt_bytes(_g(u, 'obytes', 0))} in {fmt_bytes(_g(u, 'rbytes', 0))}")
    typ = str(_g(u, "type", "") or "UNKNOWN")
    parts.append(f"type {typ}")
    flags = int(_g(u, "content_flags", 0) or 0)
    if L["hints"]:
        h = attack_kb.hints(typ, str(_g(u, "svc_class", "") or ""), str(_g(u, "proto", "") or ""), flags)
        if h:
            parts.append("hints " + ", ".join(h))
    if L["flags"] and flags:
        parts.append("content " + ",".join(n for b, n in FLAG_NAMES.items() if flags & b))
    text = _text_fields(u, ctx, L["per_field"], L["keep"])
    if text:
        parts.append(" ".join(text))
    return " | ".join(parts)


# ----------------------------------------------------------------------------- relation line
def render_relation(u: Any, v: Any, cand: Mapping | None, ctx: IncidentContext) -> str:
    r = ctx.roles
    us, vs = str(_g(u, "src", "")), str(_g(v, "src", ""))
    ud = [str(d) for d in (_g(u, "dsts", []) or [])]
    vd = [str(d) for d in (_g(v, "dsts", []) or [])]
    cand = cand or {}
    lag_gap = _g(cand, "lag_gap", None)
    if lag_gap is None:
        lag_gap = float(_g(v, "ts", 0.0)) - float(_g(u, "te", _g(u, "ts", 0.0)))
    lag_start = _g(cand, "lag_start", None)
    if lag_start is None:
        lag_start = float(_g(v, "ts", 0.0)) - float(_g(u, "ts", 0.0))
    overlap = bool(_g(cand, "overlap", float(lag_gap) <= 0))
    phrases = [f"V starts {fmt_dur(lag_start)} after U starts, while U is active" if overlap
               else f"V starts {fmt_dur(lag_gap)} after U ends"]
    mask = int(_g(cand, "rel_mask", 0) or 0)
    if not mask:     # no candidate row: derive the relations from the units
        mask |= REL_SAME_ACTOR if us == vs else 0
        mask |= REL_SAME_VICTIM if set(ud) & set(vd) else 0
        mask |= REL_PIVOT if vs in set(ud) else 0
        mask |= REL_REVERSE if us in set(vd) else 0
    rel = []
    if mask & REL_SAME_VICTIM:
        common = [d for d in vd if d in set(ud)]
        if common:
            who = r.name(common[0])
            rel.append(f"same victim {who}" if len(ud) <= 3 else f"U's targets include V's target {who}")
        else:
            rel.append("same victim")
    if mask & REL_PIVOT:
        rel.append(f"pivot: U's target {r.name(vs)} is V's source" if vs in set(ud) else "pivot")
    if mask & REL_REVERSE:
        rel.append(f"V targets U's source {r.name(us)}" if us in set(vd) else "reverse direction")
    if mask & REL_SAME_ACTOR:
        rel.append(f"same source {r.name(us)}")
    if mask & REL_SHARED_EXT:
        shared = [h for h in vd if h in set(ud) and r.kind.get(h) in ("external", "infra")]
        uinf = set(str(h) for h in (_g(u, "infra", []) or []))
        shared += [str(h) for h in (_g(v, "infra", []) or []) if str(h) in uinf and str(h) not in shared]
        if shared:
            rel.append(f"shared {'infrastructure' if r.kind.get(shared[0]) == 'infra' else 'external'} "
                       f"{r.name(shared[0])}")
        else:
            names = set(map(str, (_g(u, "dns_names", []) or []) + (_g(u, "http_hosts", []) or [])
                            + (_g(u, "tls_snis", []) or [])))
            vn = [str(x) for x in (_g(v, "dns_names", []) or []) + (_g(v, "http_hosts", []) or [])
                  + (_g(v, "tls_snis", []) or []) if str(x) in names]
            rel.append(f"shared name {wrap_text(vn[0], r)}" if vn else "shared external entity")
    if not rel:
        kind = str(_g(cand, "shared_entity_kind", "none") or "none")
        if kind != "none":
            rel.append(f"shared {kind} entity")
    if not mask & REL_SAME_ACTOR:
        rel.append("different source")
    return "Relation: " + "; ".join(phrases + rel) + "."


# ----------------------------------------------------------------------------- card
def render_card_at(u: Any, v: Any, cand: Mapping | None, ctx: IncidentContext, level: int = 4) -> str:
    return "\n".join(["Incident clock starts at t0.", render_unit("U", u, ctx, level), render_unit("V", v, ctx, level),
                      render_relation(u, v, cand, ctx)])


def render_card(u: Any, v: Any, cand: Mapping | None, ctx: IncidentContext,
                count_tokens: Callable[[str], int] | None = None, budget: int = CARD_TOKEN_BUDGET) -> str:
    """Card text whose suffix (card + question) has ≤ ``budget`` tokens under ``count_tokens``.

    Without ``count_tokens`` the full card is returned (no budget enforcement)."""
    from certichain.llm.prompts import render_suffix
    card = render_card_at(u, v, cand, ctx, 4)
    if count_tokens is None:
        return card
    for level in (4, 3, 2, 1, 0):
        card = render_card_at(u, v, cand, ctx, level)
        if count_tokens(render_suffix(card)) <= budget:
            return card
    lo, hi = 0, len(card)                      # last resort: cut characters (binary search)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(render_suffix(card[:mid])) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return card[:lo]


def card_tokens(card: str, count_tokens: Callable[[str], int]) -> int:
    from certichain.llm.prompts import render_suffix
    return count_tokens(render_suffix(card))


@functools.lru_cache(maxsize=4)
def token_counter(model_id: str) -> Callable[[str], int]:
    """Token counter of a model's tokenizer from the local HF cache (offline, CPU, no weights)."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)

    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))
    return count
