"""B2 — BotHunter-style dialog rules [BH07] (DESIGN §6.3).

Per internal host h, unit types map to dialog stages:
  E1 inbound scan          SCAN_H / SCAN_V / SWEEP_ICMP targeting h (h an internal dst, not the source)
  E2 inbound exploit/brute BRUTE_* / WEB_ATTACK / SMB_SESSION targeting h
  E3 egg download          HTTP_DOWNLOAD (and INTERACTIVE_SESSION) originated by h
  E4 C&C                   C2_PERIODIC / REPORT_SMALL / CONN_FAIL_BURST (failed call-back bursts) / irc, by h
  E5 outbound scan/attack  SCAN_* / BRUTE_* / FLOOD_* / AMPLIFICATION / WEB_ATTACK / SMB_SESSION originated by h
A unit u of stage a for h is linked to every unit v of the NEXT stage b > a present for h among the units
starting in (t^s_u, t^e_u + W] (consecutive stages of one internal host within W); a pair is claimed iff it is
a candidate. Internal = the unit's dir_class side ("int"). No guarantee (uncalibrated).
"""
from __future__ import annotations

import numpy as np
import polars as pl

INBOUND = {"E1": ("SCAN_H", "SCAN_V", "SWEEP_ICMP"), "E2": ("BRUTE_", "WEB_ATTACK", "SMB_SESSION")}
OUTBOUND = {"E3": ("HTTP_DOWNLOAD", "INTERACTIVE_SESSION"),
            "E4": ("C2_PERIODIC", "REPORT_SMALL", "CONN_FAIL_BURST", "CALLBACK", "C2_"),
            "E5": ("SCAN_", "SWEEP_ICMP", "BRUTE_", "FLOOD_", "AMPLIFICATION", "WEB_ATTACK", "SMB_SESSION")}
STAGE_ORDER = ("E1", "E2", "E3", "E4", "E5")


def _match(t: str, pats) -> bool:
    return any(t == p or (p.endswith("_") and t.startswith(p)) or (p.endswith("BACK") and p in t) for p in pats)


def unit_stages(row: dict) -> list[tuple[str, str]]:
    """(stage, host) pairs of one unit."""
    t = str(row.get("type") or "")
    dc = str(row.get("dir_class") or "")
    src_int = dc.startswith("int")
    dst_int = dc.endswith("int")
    out = []
    for s, pats in INBOUND.items():
        if _match(t, pats) and dst_int:
            for h in row.get("dsts") or []:
                if h != row.get("src"):
                    out.append((s, h))
    if src_int:
        for s, pats in OUTBOUND.items():
            if _match(t, pats) or (s == "E4" and row.get("svc_class") == "irc"):
                out.append((s, row.get("src")))
    return out


def bothunter_claims(units: pl.DataFrame, cands: pl.DataFrame, W: float = 1800.0) -> np.ndarray:
    """Claim mask over ``cands`` (row order)."""
    if cands.height == 0:
        return np.zeros(0, dtype=bool)
    rows = {int(r["unit_id"]): r for r in units.iter_rows(named=True)}
    per_host: dict[str, list[tuple[float, int, int, float]]] = {}
    for uid, r in rows.items():
        for s, h in unit_stages(r):
            per_host.setdefault(h, []).append((float(r["ts"]), STAGE_ORDER.index(s), uid, float(r["te"])))
    links: set[tuple[int, int]] = set()
    for h, lst in per_host.items():
        lst.sort()
        for ts_u, a, uid, te_u in lst:
            later = [(ts_v, b, vid) for ts_v, b, vid, _ in lst if ts_u < ts_v <= te_u + W and b > a and vid != uid]
            if not later:
                continue
            b_next = min(b for _, b, _ in later)
            links |= {(uid, vid) for _, b, vid in later if b == b_next}
    pairs = list(zip(cands["u"].to_list(), cands["v"].to_list()))
    return np.array([(int(a), int(b)) in links for a, b in pairs], dtype=bool)
