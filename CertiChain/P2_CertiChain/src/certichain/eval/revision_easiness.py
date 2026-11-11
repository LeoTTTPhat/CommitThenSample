"""P2-X revision: benchmark-easiness sensitivity (review issue 8; R1-M4, R3-M1, R3-M2, R2-M4). Owner: P2-X.

POST-PREREG, EXPLORATORY. Three analyses on the frozen ChainForge banks:
* long-lag STR: STR restricted to template edges whose generator lag (gen.json edge lag_s: parent's last record
  start -> child's first record start; capped at 0.8 W = 24 min by the generator) exceeds the pool median, vs edges at
  or below the median -- shows whether the lag cap (every edge fits the 30-min window) drives power;
* decoy-type strata: realised FDR / STR / pooled FDR for test incidents with >= 1 D1 / D2 / D3 decoy (calibration
  unchanged), and the endpoint classes of the certified false claims (which kind of unit a false claim touches);
* (the two-way incident x episode bootstrap lives in revision_tasks.twoway_cell).
GT-side reads (transitions, gt_units, gt.json, gen.json) go through the driver's guarded readers / the generator log.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from ..certify.losses import LossBank
from ..io.paths import results_root
from . import experiments as X
from . import revision as RV
from . import revision_tasks as RT
from .metrics import masked_mean
from .splits import random_splits

KINDS = ("main", "D1", "D2", "D3", "bg", "mixed")


def gen_json(inc: str) -> dict:
    parts = inc.split("/")
    p = results_root() / "interim" / parts[0] / "incidents" / Path(*parts[1:]) / "gen.json"
    return json.loads(p.read_text())


class EdgeLags(dict):
    """(a, b) -> generator lag (s). Missing pairs (bridges over hidden episodes, e.g. BF0 -> [REP0] -> ITT) get the
    sum of the hop lags along the shortest generator path from a to b (an approximation: hop lags are measured from
    each parent's last record start)."""

    def __init__(self, edges: dict[tuple[str, str], float]):
        super().__init__(edges)
        self.succ: dict[str, list[str]] = {}
        for (a, b) in edges:
            self.succ.setdefault(a, []).append(b)

    def get(self, key, default=None):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        a, b = key
        from collections import deque
        q = deque([(a, 0.0)])
        seen = {a}
        while q:
            x, acc = q.popleft()
            for y in self.succ.get(x, []):
                if y in seen:
                    continue
                lag = acc + dict.__getitem__(self, (x, y))
                if y == b:
                    return lag
                seen.add(y)
                q.append((y, lag))
        return default

    def __contains__(self, key):
        return self.get(key) is not None


def edge_lags(g: dict) -> "EdgeLags":
    out = {}
    for c in (g.get("campaigns") or []) + (g.get("decoys") or []):
        for e in c.get("edges") or []:
            out[(e["a"], e["b"])] = float(e["lag_s"])
    return EdgeLags(out)


def _flat_cum_unused(bank: LossBank, per_row_marks: list[np.ndarray]) -> np.ndarray:
    """Cumulative counts in the bank's off2 layout from per-incident 0/1 (or count) marks in CSR order."""
    off2 = bank.csr["off2"]
    out = np.zeros(int(off2[-1]), dtype=np.int32)
    for r, m in enumerate(per_row_marks):
        out[off2[r]: off2[r + 1]] = np.concatenate(([0], np.cumsum(m))).astype(np.int32)
    return out


def lag_tables(ds: str, bank: LossBank) -> dict:
    """Per bank row: CSR-order first hits of transitions by generator-lag bin (exactly the bank's transition
    convention: a Y = 1 candidate that is the enabling pair of an observable transition hits that transition; the
    last such transition wins, as in scoring/driver.build_incident), and per-row counts of transitions in T* per bin
    (the STR denominators; hidden and not-in-C transitions included). Bins: lag quartiles of the pool (Q1..Q4; the
    median split is Q1+Q2 vs Q3+Q4) and lag <= / > 600 s."""
    from ..scoring import driver as D
    lags_all, rows_info = [], []
    for iid in bank.index["incident_id"].to_list():
        inc = X.inc_of(iid)
        lg = edge_lags(gen_json(inc))
        tr = D.load_transitions(ds, inc)
        lab = D.load_labels(ds, inc)
        cands = D.load_cands(ds, inc)
        rows_info.append((inc, tr, lab, cands, lg))
        for a_, b_, kd in zip(tr["src_ep"].to_list(), tr["dst_ep"].to_list(), tr["kind"].to_list()) if tr.height else []:
            if kd == "bridge":            # bridges are E*_link links, not transitions of T*
                continue
            la = lg.get((a_, b_))
            if la is not None:
                lags_all.append(la)
    lags = np.asarray(lags_all, float)
    q = np.quantile(lags, [0.25, 0.5, 0.75]) if lags.size else np.array([np.nan] * 3)
    schemes = {"quartile": q, "600s": np.array([600.0])}
    nb = {k: v.size + 1 for k, v in schemes.items()}
    marks = {k: [] for k in schemes}
    dens = {k: [] for k in schemes}
    n_unk = 0
    for r, (inc, tr, lab, cands, lg) in enumerate(rows_info):
        sl = bank.slice(r)
        N = sl.stop - sl.start
        pos_of = {int(c): i for i, c in enumerate(cands["cand_id"].to_list())}
        y = lab["y"].to_numpy()
        tid = np.full(N, -1, dtype=np.int64)
        lag_of: dict[int, float] = {}
        den = {k: np.zeros(nb[k]) for k in schemes}
        if tr.height:
            obs_j = 0
            for a_, b_, kd, c in zip(tr["src_ep"].to_list(), tr["dst_ep"].to_list(), tr["kind"].to_list(),
                                     tr["cand_id"].to_list()):
                if kd == "bridge":
                    continue
                lag = lg.get((a_, b_))
                if lag is None:
                    n_unk += 1
                else:
                    for k, th in schemes.items():
                        den[k][int(np.searchsorted(th, lag, side="left"))] += 1
                if kd == "obs" and c is not None:
                    p = pos_of.get(int(c))
                    if p is not None and y[p] == 1:
                        tid[p] = obs_j
                    if lag is not None:
                        lag_of[obs_j] = lag
                    obs_j += 1
        order = bank.csr["cand"][sl]
        t_sorted = tid[order]
        seen: set = set()
        mk = {k: np.zeros((nb[k], N), dtype=np.int32) for k in schemes}
        for i, t in enumerate(t_sorted.tolist()):
            if t >= 0 and t not in seen:
                seen.add(t)
                if t in lag_of:
                    for k, th in schemes.items():
                        mk[k][int(np.searchsorted(th, lag_of[t], side="left")), i] = 1
        for k in schemes:
            marks[k].append(mk[k])
            dens[k].append(den[k])
    off2 = bank.csr["off2"]
    cum = {}
    for k in schemes:
        c = np.zeros((nb[k], int(off2[-1])), dtype=np.int32)
        for r, m in enumerate(marks[k]):
            c[:, off2[r]: off2[r + 1]] = np.concatenate((np.zeros((nb[k], 1), np.int32), np.cumsum(m, axis=1)),
                                                        axis=1)
        cum[k] = c
    chk = bool(np.array_equal(cum["quartile"].sum(axis=0), bank.csr["c_trans"])) if n_unk == 0 else None
    return {"quartiles_s": q.tolist(), "median_lag_s": float(q[1]), "n_transitions": int(lags.size),
            "n_unknown_lag": n_unk, "cum": cum, "den": {k: np.stack(v) for k, v in dens.items()},
            "consistent_with_bank": chk, "lags": lags}


GROUPS = {"long": ("quartile", (2, 3)), "short": ("quartile", (0, 1)), "Q1": ("quartile", (0,)),
          "Q2": ("quartile", (1,)), "Q3": ("quartile", (2,)), "Q4": ("quartile", (3,)), "gt600": ("600s", (1,)),
          "le600": ("600s", (0,))}


def lag_metrics(bank: LossBank, lt: dict, sel) -> dict[str, np.ndarray]:
    off2 = bank.csr["off2"]
    tr, tm = sel.test_rows, sel.tmask
    R = np.where(tm, sel.R, 0)
    out = {}
    for g, (sch, bins) in GROUPS.items():
        hit = sum(lt["cum"][sch][j][off2[tr] + R].astype(float) for j in bins)
        den = lt["den"][sch][:, list(bins)].sum(axis=1)[tr]
        ok = tm & (den > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"str_{g}_mean"] = masked_mean(np.where(ok, hit / np.where(den > 0, den, 1), 0.0), ok)
            out[f"str_{g}_pooled"] = np.where(ok, hit, 0).sum(1) / np.maximum(np.where(ok, den, 0).sum(1), 1)
    out["delta_long_minus_short"] = out["str_long_mean"] - out["str_short_mean"]
    out["delta_pooled"] = out["str_long_pooled"] - out["str_short_pooled"]
    out["delta_gt600_minus_le600"] = out["str_gt600_mean"] - out["str_le600_mean"]
    return out


LAG_METRICS = tuple([f"str_{g}_{k}" for g in GROUPS for k in ("mean", "pooled")] +
                    ["delta_long_minus_short", "delta_pooled", "delta_gt600_minus_le600"])


def oracle_lag(bank: LossBank, lt: dict) -> dict[str, float]:
    """r_C-limited ceiling by lag group: transitions whose enabling pair is a true candidate of C / transitions."""
    off2 = bank.csr["off2"]
    end = off2[:-1] + bank.sizes()
    out = {}
    for g, (sch, bins) in GROUPS.items():
        hit = sum(lt["cum"][sch][j][end].astype(float) for j in bins)
        den = lt["den"][sch][:, list(bins)].sum(axis=1)
        out[f"ceiling_{g}"] = float(hit.sum() / max(den.sum(), 1))
        out[f"n_{g}"] = float(den.sum())
    return out


def lagstr_cell(task: dict) -> dict:
    ds, sid, method, alpha = task["ds"], task["sid"], task.get("method", "crc_bh"), float(task["alpha"])
    d = RT.X2_DATASETS[ds]
    K, n_test, B = d["K"], d["n_test"], int(task.get("B", RT.BOOT_B))
    path = RT.x2_root() / "lagstr" / ds / f"{method}__{sid}" / f"a{alpha}"
    bank = X.load_bank(ds, sid)
    cfg = {"group": "lagstr", "ds": ds, "sid": sid, "method": method, "alpha": alpha, "K": K, "n_test": n_test, "B": B,
           "code": RT.CODE_VERSION}
    h = RT._hash(cfg, X.bank_hash(bank))
    if RT.cached(path, h, task.get("overwrite", False)) is not None:
        return {"task": task, "path": str(path), "cached": True}
    t0 = time.perf_counter()
    lt = lag_tables(ds, bank)
    sp = random_splits(bank.n, K, range(RT.N_SPLITS), n_test)
    res = lag_metrics(bank, lt, X.select(bank, method, sp, alpha))
    summ = RV.summarize_splits(res)
    reps = np.full((B, len(LAG_METRICS)), np.nan)
    for b, rows in enumerate(X.boot_pools(bank.n, B, 0)):
        spb = random_splits(rows.size, K, np.arange(RT.BOOT_SPLITS), n_test)
        reps[b] = RV.replicate_vector(lag_metrics(bank, lt, X.select(bank, method, spb, alpha, pool_rows=rows)),
                                      LAG_METRICS)
    mets = {m: {"point": (summ.get(m) or {}).get("mean"), **RV.ci_dict(reps[:, j], (summ.get(m) or {}).get("mean"))}
            for j, m in enumerate(LAG_METRICS)}
    lags = lt["lags"]
    cell = {"config": cfg, "config_hash": h, "status": RV.STATUS, "status_note": RV.STATUS_NOTE, "metrics": mets,
            "median_lag_s": lt["median_lag_s"], "lag_quartile_cuts_s": lt["quartiles_s"],
            "n_transitions_with_lag": lt["n_transitions"],
            "n_unknown_lag": lt["n_unknown_lag"], "consistent_with_bank_c_trans": lt["consistent_with_bank"],
            "lag_quantiles_s": {q: float(np.quantile(lags, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)}
            if lags.size else {}, "share_lag_gt_0.8W": float(np.mean(lags > 0.8 * 1800)) if lags.size else None,
            "oracle": oracle_lag(bank, lt), "seconds": time.perf_counter() - t0,
            "ci_kind": f"incident-pool bootstrap B={B} x {RT.BOOT_SPLITS} splits"}
    RT.write_cell_dir(path, cell, RT.frame(res, sp.seeds), {"main": reps})
    return {"task": task, "path": str(path), "seconds": cell["seconds"]}


# =========================================================================== decoy-type strata / false-claim classes
def unit_kinds(ds: str, inc: str, n_units: int) -> np.ndarray:
    from ..scoring import driver as D
    gtu = D.load_gt_units(ds, inc)
    gtj = D.load_gt_json(ds, inc)
    camps = (gtj.get("meta") or {}).get("campaigns") or {}
    kind_of = {c: (v or {}).get("kind", "main") for c, v in camps.items()}
    out = np.array(["bg"] * n_units, dtype=object)
    for uid, camp, mixed in zip(gtu["unit_id"].to_list(), gtu["campaign"].to_list(), gtu["mixed"].to_list()):
        if 0 <= int(uid) < n_units:
            out[int(uid)] = "mixed" if mixed else (kind_of.get(camp, "main") if camp is not None else "bg")
    return out


def claim_class_tables(ds: str, bank: LossBank) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Endpoint-kind class of every candidate (sorted kinds of u and v joined by '+'), as cumulative counts of FALSE
    (Y != 1) candidates per class in the bank's off2 layout. Also the decoy composition per incident (gen.json)."""
    from ..scoring import driver as D
    classes: list[str] = []
    for i, a in enumerate(KINDS):
        for b in KINDS[i:]:
            classes.append(a if a == b else f"{a}+{b}")
    cidx = {c: i for i, c in enumerate(classes)}
    off2 = bank.csr["off2"]
    cum = np.zeros((len(classes), int(off2[-1])), dtype=np.int32)
    comp = {"n_D1": [], "n_D2": [], "n_D3": []}
    for r, iid in enumerate(bank.index["incident_id"].to_list()):
        inc = X.inc_of(iid)
        g = gen_json(inc).get("summary", {})
        for k in comp:
            comp[k].append(int(g.get(k) or 0))
        units = D.load_units(ds, inc)
        cands = D.load_cands(ds, inc)
        kinds = unit_kinds(ds, inc, units.height)
        sl = bank.slice(r)
        order = bank.csr["cand"][sl]
        u = cands["u"].to_numpy()[order]
        v = cands["v"].to_numpy()[order]
        yb = bank.csr["y"][sl]
        cl = np.array([cidx["+".join(sorted((kinds[a], kinds[b]), key=KINDS.index)) if kinds[a] != kinds[b]
                             else kinds[a]] for a, b in zip(u.tolist(), v.tolist())], dtype=np.int64)
        false = yb != 1
        for j in range(len(classes)):
            m = (cl == j) & false
            cum[j, off2[r]: off2[r + 1]] = np.concatenate(([0], np.cumsum(m))).astype(np.int32)
    return classes, cum, {k: np.asarray(v) for k, v in comp.items()}


def decoy_cell(task: dict) -> dict:
    ds, sid, method, alpha = task["ds"], task["sid"], task.get("method", "crc_bh"), float(task["alpha"])
    d = RT.X2_DATASETS[ds]
    K, n_test, B = d["K"], d["n_test"], int(task.get("B", RT.BOOT_B))
    path = RT.x2_root() / "decoy" / ds / f"{method}__{sid}" / f"a{alpha}"
    bank = X.load_bank(ds, sid)
    cfg = {"group": "decoy", "ds": ds, "sid": sid, "method": method, "alpha": alpha, "K": K, "n_test": n_test, "B": B,
           "code": RT.CODE_VERSION}
    h = RT._hash(cfg, X.bank_hash(bank))
    if RT.cached(path, h, task.get("overwrite", False)) is not None:
        return {"task": task, "path": str(path), "cached": True}
    t0 = time.perf_counter()
    classes, cum, comp = claim_class_tables(ds, bank)
    strata = {"no_decoy": (comp["n_D1"] + comp["n_D2"] + comp["n_D3"]) == 0, "has_D1": comp["n_D1"] > 0,
              "has_D2": comp["n_D2"] > 0, "has_D3": comp["n_D3"] > 0,
              "D3_only": (comp["n_D3"] > 0) & (comp["n_D1"] == 0) & (comp["n_D2"] == 0),
              "has_D2_or_D1": (comp["n_D1"] + comp["n_D2"]) > 0}
    met = ("fdr", "str_mean", "pooled_fdr", "frac_empty", "cond_fdp", "n_claims", "n_false", "ucr")
    sp = random_splits(bank.n, K, range(RT.N_SPLITS), n_test)
    sel = X.select(bank, method, sp, alpha)

    def strat_eval(sel_, sp_, rows=None):
        out = {}
        for name, mk in strata.items():
            out[name] = RV.x2_eval(bank, method, sp_, alpha, pool_rows=rows, selection=sel_, chains=False, mask_col=mk)
        return out

    def class_shares(sel_):
        off2 = bank.csr["off2"]
        tr, tm = sel_.test_rows, sel_.tmask
        R = np.where(tm, sel_.R, 0)
        tot = np.array([np.where(tm, cum[j][off2[tr] + R], 0).sum() for j in range(len(classes))], dtype=float)
        return tot

    point = {k: RV.summarize_splits(v) for k, v in strat_eval(sel, sp).items()}
    cls_tot = class_shares(sel)
    reps = {k: np.full((B, len(met)), np.nan) for k in strata}
    cls_reps = np.full((B, len(classes)), np.nan)
    for b, rows in enumerate(X.boot_pools(bank.n, B, 0)):
        spb = random_splits(rows.size, K, np.arange(RT.BOOT_SPLITS), n_test)
        selb = X.select(bank, method, spb, alpha, pool_rows=rows)
        for k, r in strat_eval(selb, spb, rows).items():
            reps[k][b] = RV.replicate_vector(r, met)
        ct = class_shares(selb)
        cls_reps[b] = ct / max(ct.sum(), 1)
    mets = {k: {m: {"point": (point[k].get(m) or {}).get("mean"), **RV.ci_dict(reps[k][:, j], (point[k].get(m) or {}).get("mean"))}
                for j, m in enumerate(met)} for k in strata}
    share = cls_tot / max(cls_tot.sum(), 1)
    cls = {c: {"false_claims_over_200_splits": float(cls_tot[j]), "share": float(share[j]),
               **RV.ci_dict(cls_reps[:, j], float(share[j]))} for j, c in enumerate(classes) if cls_tot[j] > 0}
    cell = {"config": cfg, "config_hash": h, "status": RV.STATUS, "status_note": RV.STATUS_NOTE, "strata": mets,
            "n_pool_per_stratum": {k: int(v.sum()) for k, v in strata.items()},
            "false_claim_endpoint_classes": cls,
            "kinds": "main = main-campaign unit; D1/D2/D3 = decoy-campaign unit of that kind; bg = background; mixed",
            "seconds": time.perf_counter() - t0,
            "ci_kind": f"incident-pool bootstrap B={B} x {RT.BOOT_SPLITS} splits; strata restrict TEST incidents"}
    RT.write_cell_dir(path, cell, None, {**{k: v for k, v in reps.items()}, "classes": cls_reps})
    return {"task": task, "path": str(path), "seconds": cell["seconds"]}
