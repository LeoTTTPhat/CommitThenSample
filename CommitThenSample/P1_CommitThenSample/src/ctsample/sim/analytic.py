"""Static analytic metrics (DESIGN §6 common settings, "analytic first"): no Monte Carlo.

For a static design (pi a function of committed data only):
* presence / content-evidence miss per incident: prod (1 - pi);
* Var(Qhat_q) = sum f^2 (1-pi)/pi per query (HT is unbiased, so MSE = Var). DESIGN prints
  f^2(1-pi)/pi^2, which is the integrand of E[Vhat]'s estimator, not the variance (see estimation/ht.py);
* relative variance Var/Q^2 and the worst case over queries;
* expected spend per epoch, spend/target, max |D|/Bbar of the expected-cost controller pass;
* MSE ratios between designs (per query) and their geometric means.
"""
from __future__ import annotations

import numpy as np

from ..estimation.incident import group_miss
from .tables import StreamTables

__all__ = ["pi_full", "miss_per_incident", "var_ht_queries", "expected_spend", "mse_ratios", "static_metrics",
           "miss_by_k_strata"]


def pi_full(T: StreamTables, static: dict) -> np.ndarray:
    """Scatter a static_pass result (payload order) back to all units (0 for zero-payload)."""
    pi = np.zeros(T.n_units)
    pi[static["pay_idx"]] = static["pi_pay"]
    return pi


def miss_per_incident(T: StreamTables, pi_units) -> dict:
    """Per incident: presence miss (all units), content-evidence miss (content units), k (content units)."""
    pi_units = np.asarray(pi_units, dtype=float)
    inc = np.asarray(T.incident) if T.incident is not None else np.full(T.n_units, -1)
    content = np.asarray(T.content, dtype=bool) if T.content is not None else inc >= 0
    ids, pres, n_all = group_miss(pi_units, inc)
    ids_c, cont, k = group_miss(pi_units, inc, mask=content)
    cm = dict(zip(ids_c.tolist(), cont.tolist()))
    kk = dict(zip(ids_c.tolist(), k.tolist()))
    return {"incident": ids, "presence_miss": pres, "n_units": n_all,
            "content_miss": np.array([cm.get(i, 1.0) for i in ids.tolist()]),
            "k": np.array([kk.get(i, 0) for i in ids.tolist()])}


def miss_by_k_strata(miss: dict, edges=(20, 100)) -> dict:
    """Mean content-evidence recall (1 - miss) stratified by k (<= 20, 21-100, > 100)."""
    k = miss["k"]
    rec = 1.0 - miss["content_miss"]
    out = {}
    lo = 0
    labels = [f"k<={edges[0]}"] + [f"{edges[i] + 1}-{edges[i + 1]}" for i in range(len(edges) - 1)] + [f"k>{edges[-1]}"]
    bounds = [0] + list(edges) + [np.inf]
    for lab, a, b in zip(labels, bounds[:-1], bounds[1:]):
        m = (k > a) & (k <= b) if a > 0 else (k >= 1) & (k <= b)
        out[lab] = {"n": int(m.sum()), "recall": float(rec[m].mean()) if m.any() else float("nan")}
        lo = b
    return out


def var_ht_queries(T: StreamTables, pi_units) -> dict:
    """Per query: Q, Var(Qhat), relative variance, and the probability that no hit is retained."""
    pi_units = np.asarray(pi_units, dtype=float)
    out = {"query": [], "Q": [], "var": [], "rel_var": [], "p_no_hit": []}
    for q in range(T.n_queries):
        u, f = T.hits(q)
        p = pi_units[u]
        if np.any(p <= 0):
            raise ValueError(f"query {q}: hit unit with pi = 0")
        v = float(np.sum(f * f * (1 - p) / p))
        Q = float(f.sum())
        out["query"].append(T.query_names[q])
        out["Q"].append(Q)
        out["var"].append(v)
        out["rel_var"].append(v / Q ** 2 if Q > 0 else np.nan)
        out["p_no_hit"].append(float(np.exp(np.log1p(-np.clip(p, 0, 1)).sum())))
    return {k: np.asarray(v) if k != "query" else v for k, v in out.items()}


def expected_spend(T: StreamTables, pi_units, Bbar: float) -> dict:
    pi_units = np.asarray(pi_units, dtype=float)
    E = T.n_epochs
    cost = np.bincount(np.asarray(T.epoch), weights=pi_units * np.asarray(T.s, dtype=float), minlength=E)
    D = np.cumsum(Bbar - cost)
    return {"cost": cost, "total": float(cost.sum()), "spend_ratio": float(cost.sum() / (Bbar * E)),
            "max_abs_D_over_B": float(np.max(np.abs(D)) / Bbar) if E else 0.0}


def mse_ratios(var_a, var_b) -> dict:
    """Per-query MSE ratio a/b (unbiased estimators) and its geometric mean over queries with var_b > 0."""
    va, vb = np.asarray(var_a, dtype=float), np.asarray(var_b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(vb > 0, va / vb, np.nan)
    ok = np.isfinite(r) & (r > 0)
    return {"ratio": r, "geomean": float(np.exp(np.mean(np.log(r[ok])))) if ok.any() else float("nan"),
            "worst": float(np.nanmax(r)) if ok.any() else float("nan")}


def static_metrics(T: StreamTables, pi_units, Bbar: float) -> dict:
    v = var_ht_queries(T, pi_units)
    return {"queries": v, "worst_rel_var": float(np.nanmax(v["rel_var"])) if len(v["rel_var"]) else float("nan"),
            "spend": expected_spend(T, pi_units, Bbar), "incidents": miss_per_incident(T, pi_units)}
