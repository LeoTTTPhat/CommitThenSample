"""Statement policy and query-selection rule (DESIGN §3.9 (i)-(vi), Prop. 2; contract §7.1).

(i)   The estimand covers ALL committed units of the query's window; "compliant epochs only" is never used.
(ii)  Missing units (set W: withheld, deleted, failed epoch) count X = 0 in the LCS (valid under any W).
(iii) The UCS is issued only on CLEAN windows: window cap W = empty, no V7-flagged cell of the query,
      no flood-guard-capped unit (pi < phi0 breaks cbar). Otherwise the deterministic cap Dcap is
      reported with "UCS withheld: non-compliance". Count estimands get L+ only (no global range bound).
(iv)  A query must be a function of committed metadata, pre-registered q and case information
      independent of Z (e.g. "device d flagged by the operator's IDS at t, the 2 h before t").
(v)   Queries chosen after inspecting retained content need a declared finite family F; each
      statement is then made at alpha/|F| (union bound).
(vi)  A report with Q statements states the per-statement level and the family-wise level (Bonferroni Q*alpha).

Verifier integration (P1-A, protocol/verifier.py): ``Verifier(..., statements_hook=make_statements_hook(
queries, f, b, device, V_ref=..., phi0=...))``; the hook reads ``report.v8_inputs()`` (tau, Z, pay, W,
W_all, requested, failed, floor_mode, flagged_devices per epoch) and returns ``report(...)`` of
``statements_from_v8`` for every query; the verifier stores it in ``report.statements``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Query", "QueryFamily", "QueryRegistry", "window_is_clean", "statements", "report", "UCS_WITHHELD",
           "statements_from_v8", "make_statements_hook"]

UCS_WITHHELD = "UCS withheld: non-compliance"
UCS_NOT_APPLICABLE = "UCS not applicable: count estimand (L+ only)"


@dataclass(frozen=True)
class Query:
    name: str
    q: str | int
    devices: tuple | None = None
    window: tuple | None = None          # (e_start, e_end) epochs, end exclusive
    kind: str = "indicator"              # indicator | count | label | metadata
    family: str | None = None            # declared family (rule (v)); None = metadata-chosen (rule (iv))
    description: str = ""


@dataclass
class QueryFamily:
    name: str
    size: int
    members: list = field(default_factory=list)

    def __post_init__(self):
        if self.size < 1:
            raise ValueError("family size must be >= 1")


class QueryRegistry:
    """Pre-registered query families; per-statement alpha = alpha / |F| for post-selected queries."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        self.families: dict[str, QueryFamily] = {}
        self.queries: dict[str, Query] = {}

    def declare_family(self, name: str, size: int) -> QueryFamily:
        if name in self.families:
            raise ValueError(f"family {name} already declared (families are fixed in advance)")
        fam = QueryFamily(name, int(size))
        self.families[name] = fam
        return fam

    def register(self, query: Query) -> float:
        if query.family is not None:
            fam = self.families.get(query.family)
            if fam is None:
                raise KeyError(f"query {query.name}: family {query.family} was not declared")
            if len(fam.members) >= fam.size and query.name not in fam.members:
                raise ValueError(f"family {fam.name} is full ({fam.size} members declared)")
            if query.name not in fam.members:
                fam.members.append(query.name)
        self.queries[query.name] = query
        return self.alpha_for(query)

    def alpha_for(self, query: Query) -> float:
        """Two-sided per-statement level (alpha/|F| for a family member)."""
        if query.family is None:
            return self.alpha
        return self.alpha / self.families[query.family].size

    def family_wise(self, n_statements: int, alpha_statement: float | None = None) -> float:
        a = self.alpha if alpha_statement is None else alpha_statement
        return min(1.0, a * n_statements)


def _any(x):
    if x is None:
        return False
    return bool(np.any(np.asarray(x, dtype=bool)))


def window_is_clean(W_mask=None, v7_flags=None, capped_mask=None):
    """(clean, reasons). Inputs are boolean arrays over the query window's committed units / cells."""
    reasons = []
    if _any(W_mask):
        reasons.append("window intersects W (missing/withheld units)")
    if _any(v7_flags):
        reasons.append("V7-flagged cell in the query")
    if _any(capped_mask):
        reasons.append("flood-guard-capped unit (pi < phi0)")
    return (not reasons), reasons


def _val(x):
    if x is None:
        return None
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def statements(query: Query, window, W_mask, lcs, ucs, sure, *, v7_flags=None, capped_mask=None,
               alpha: float | None = None, n_statements: int = 1, registry: QueryRegistry | None = None) -> dict:
    """Statement dict for one query (q, A, window).

    lcs: estimation.cs.LowerCS updated with X := 0 on W over ALL committed units of the window;
    ucs: estimation.cs.UpperCS (or None); sure: (S, Dcap) from cs.sure_bounds with the same W.
    L+ = max(L*, S) is always reported. U+ = min(Qhat + r+, Dcap) only on a clean window of an
    indicator/label query whose UCS is not void; otherwise Dcap with UCS_WITHHELD."""
    if alpha is None:
        alpha = registry.alpha_for(query) if registry is not None else 0.05
    S, Dcap = sure
    L_star = _val(lcs.bound())
    L_plus = np.maximum(L_star, S)
    clean, reasons = window_is_clean(W_mask, v7_flags, capped_mask)
    out = {"query": query.name, "window": window if window is not None else query.window, "devices": query.devices,
           "kind": query.kind, "alpha_statement": alpha, "alpha_side": alpha / 2.0,
           "family_wise_alpha": min(1.0, alpha * n_statements), "n_statements": n_statements,
           "L_star": L_star, "S": _val(S), "L_plus": _val(L_plus), "D_cap": _val(Dcap),
           "Qhat": _val(lcs.Qhat), "clean": clean, "reasons": reasons}
    if query.kind == "count":
        out.update(U_plus=None, ucs_status=UCS_NOT_APPLICABLE)
    elif ucs is None:
        out.update(U_plus=None, ucs_status="UCS not computed")
    elif not clean or _any(ucs.void):
        out.update(U_plus=None, ucs_status=UCS_WITHHELD)
    else:
        out.update(U_plus=_val(ucs.bound(Dcap)), ucs_status="issued")
    lp = out["L_plus"]
    txt = f"{query.name}: >= {np.floor(lp + 1e-9) if np.ndim(lp) == 0 else lp} events"
    if out["U_plus"] is not None:
        txt += f", <= {np.ceil(out['U_plus'] - 1e-9) if np.ndim(out['U_plus']) == 0 else out['U_plus']}"
    else:
        txt += f" ({out['ucs_status']}; deterministic cap {out['D_cap']})"
    txt += f" [per-statement alpha = {alpha:g}, family-wise alpha = {out['family_wise_alpha']:g} over {n_statements}]"
    out["text"] = txt
    return out


def report(stmts: list) -> dict:
    """A report of Q statements: per-statement levels and the Bonferroni family-wise level."""
    Q = len(stmts)
    a = [s["alpha_statement"] for s in stmts]
    return {"n_statements": Q, "alpha_statement_max": max(a) if a else None,
            "family_wise_alpha": min(1.0, float(np.sum(a))) if a else None, "statements": stmts}


# --------------------------------------------------------------------------------------------- #
# Adapter for the verifier's V8 interface (protocol/verifier.py, P1-A)
# --------------------------------------------------------------------------------------------- #
_TWO64 = float(1 << 64)


def _per_unit(x, n, fill=0.0):
    """Array of length n from an array-like or a {unit index: value} dict (missing -> fill)."""
    if x is None:
        return np.full(n, fill, dtype=float)
    if isinstance(x, dict):
        out = np.full(n, fill, dtype=float)
        for i, v in x.items():
            out[int(i)] = float(v)
        return out
    a = np.asarray(x, dtype=float)
    if a.shape != (n,):
        raise ValueError(f"per-unit array has shape {a.shape}, expected ({n},)")
    return a


def statements_from_v8(v8: dict, query: Query, f: dict, b: dict, device: dict | None = None, *, V_ref: float,
                       cbar: float | None = None, bbar: float = 1.0, phi0=None, alpha: float | None = None,
                       registry: QueryRegistry | None = None, n_statements: int = 1, K: int = 16,
                       decades: float = 2.0, ucs_grid_kw: dict | None = None, every_epoch_bound: bool = True) -> dict:
    """One statement from ``VerificationReport.v8_inputs()`` (DESIGN Alg. 2 V8; Prop. 2).

    v8[e] = {tau, Z, pay, W, W_all, requested, failed, floor_mode, flagged_devices, ...} (P1-A).
    f[e]: the analyst's f on the epoch's units (array or {unit: value}; only AVAILABLE units are read);
    b[e]: eligibility bounds from committed metadata (array, length n_e); device[e]: device id per unit
    (needed when query.devices is set). V_ref (pre-registered) centres the LCS grid; the UCS uses
    cbar = bbar/phi0 (pre-registered) unless ``cbar`` is given.

    Per unit: pi = tau/2^64; available = Z & requested & not in W (W_all -> all of the epoch). A
    retained unit that is not available (missing, mismatched, not requested, failed epoch) counts as
    missing for the statement: X := 0 in the LCS, b in Dcap, and the UCS is withheld (rule (iii)).
    Epochs of the window absent from v8 or without tau/Z are treated as failed (Dcap += sum b of the
    query's units if b[e] is given, else +inf). V7 flags = flagged_devices in A; flood-guard capped
    units = eligible payload units with tau < floor_tau(phi0) in a floor_mode = 1 epoch."""
    from ..design import intpi
    from .cs import LowerCS, UpperCS, lcs_grid, ucs_grid
    if alpha is None:
        alpha = registry.alpha_for(query) if registry is not None else 0.05
    a_side = alpha / 2.0
    lcs = LowerCS(*lcs_grid(V_ref, a_side, K, decades), a_side)
    if cbar is None and phi0 is not None:
        cbar = float(bbar) / float(phi0)
    ucs = None
    if query.kind != "count" and cbar is not None:
        eu, wu = ucs_grid(cbar, **(ucs_grid_kw or {}))
        ucs = UpperCS(eu, wu, a_side, cbar)
    tau_phi = intpi.floor_tau(phi0) if phi0 is not None else None
    e0, e1 = query.window if query.window is not None else (min(v8), max(v8) + 1)
    S, Dcap = 0.0, 0.0
    w_any = v7_any = cap_any = False
    info = {"epochs": 0, "failed_epochs": [], "retained_eligible": 0, "unavailable_retained": 0}
    for e in range(int(e0), int(e1)):
        be = None if b.get(e) is None else np.asarray(b[e], dtype=float)
        d = v8.get(e)
        dev = None if device is None or device.get(e) is None else np.asarray(device[e])
        if be is not None and query.devices is not None:
            if dev is None:
                raise ValueError("device ids are required for a device-restricted query")
            inA_b = np.isin(dev, np.asarray(query.devices))
        else:
            inA_b = None
        if d is None or d.get("tau") is None or d.get("Z") is None:
            info["failed_epochs"].append(e)
            w_any = True
            Dcap += float(np.sum(be if inA_b is None else be[inA_b])) if be is not None else math.inf
            if every_epoch_bound:
                lcs.bound()
            continue
        info["epochs"] += 1
        tau = d["tau"]
        n = len(tau)
        pi = np.array([int(t) / _TWO64 for t in tau])
        Z = np.asarray(d["Z"], dtype=bool)
        pay = np.asarray(d["pay"], dtype=bool) if d.get("pay") is not None else np.ones(n, bool)
        if be is None:
            raise ValueError(f"b[{e}] missing for a verified epoch")
        if be.shape != (n,):
            raise ValueError(f"b[{e}] has shape {be.shape}, expected ({n},)")
        inA = np.ones(n, bool) if query.devices is None else inA_b
        Wm = np.zeros(n, bool)
        if d.get("W"):
            Wm[np.asarray(d["W"], dtype=np.int64)] = True
        if d.get("W_all") or d.get("failed"):
            Wm[:] = True
            info["failed_epochs"].append(e)
        req = np.zeros(n, bool)
        if d.get("requested") is not None:
            req[np.asarray(d["requested"], dtype=np.int64)] = True
        else:
            req[:] = True
        avail = Z & req & ~Wm
        missing_stmt = (Wm | (Z & ~req)) & inA
        elig = inA & (be > 0) & pay
        fe = np.nan_to_num(_per_unit(f.get(e), n), nan=0.0)
        if np.any(fe[avail & elig] > be[avail & elig] * (1 + 1e-12) + 1e-12):
            raise ValueError(f"epoch {e}: f > b on an available unit")
        X = np.where(avail & elig, fe / np.where(pi > 0, pi, 1.0), 0.0)
        lcs.update(X[elig])
        if ucs is not None:
            c = np.where(elig, be / np.where(pi > 0, pi, 1.0), 0.0)
            ucs.update(X[elig], c[elig])
        S += float(np.sum(np.where(avail & elig, fe, 0.0)))
        Dcap += float(np.sum(np.where(avail & elig, fe, be)[elig]))
        w_any |= bool(missing_stmt.any())
        flagged = set(int(x) for x in (d.get("flagged_devices") or []))
        if flagged:
            v7_any |= True if query.devices is None else bool(flagged & set(int(x) for x in query.devices))
        if d.get("floor_mode", 0) == 1 and tau_phi is not None:
            cap_any |= bool(np.any(elig & np.array([int(t) < tau_phi for t in tau])))
        elif d.get("floor_mode", 0) == 1 and tau_phi is None:
            cap_any = True                     # cannot tell capped units apart: be conservative
        info["retained_eligible"] += int((Z & elig).sum())
        info["unavailable_retained"] += int((Z & elig & ~avail).sum())
        if every_epoch_bound:
            lcs.bound()
    st = statements(query, (e0, e1), np.array([w_any]), lcs, ucs, (S, Dcap), v7_flags=np.array([v7_any]),
                    capped_mask=np.array([cap_any]), alpha=alpha, n_statements=n_statements, registry=registry)
    st["v8"] = info
    return st


def make_statements_hook(queries, f: dict, b: dict, device: dict | None = None, *, V_ref, registry=None, **kw):
    """statements_hook for protocol.verifier.Verifier: hook(report) -> report dict of all queries.

    f[q.name][e], b[q.name][e]: per-epoch arrays (see statements_from_v8); device[e]: device ids;
    V_ref: float or {q.name: float}. Every statement is made at its per-statement level and the report
    carries the Bonferroni family-wise level over the len(queries) statements (rule (vi))."""
    queries = list(queries)

    def hook(report):
        v8 = report.v8_inputs()
        stmts = []
        for q in queries:
            vr = V_ref[q.name] if isinstance(V_ref, dict) else V_ref
            stmts.append(statements_from_v8(v8, q, f[q.name], b[q.name], device, V_ref=vr, registry=registry,
                                            n_statements=len(queries), **kw))
        return report_fn(stmts)

    return hook


report_fn = report
