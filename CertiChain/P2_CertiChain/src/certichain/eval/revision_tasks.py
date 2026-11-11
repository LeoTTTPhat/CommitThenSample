"""P2-X revision task layer (process-pool workers for scripts/21_revision_x2.py). Owner: P2-X.

POST-PREREG, EXPLORATORY. Every worker reads frozen banks / baselines / derived interim files and writes ONE new cell
directory atomically under results/exp/X2/<group>/... (cell.json + per_split.parquet + boot.npy); a finished cell
with the same config hash is returned from disk (resumable). Nothing outside results/exp/X2 is written here.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import blake3
import numpy as np
import polars as pl

from ..certify.losses import LossBank
from ..io.paths import atomic_write_json, atomic_write_parquet, results_root
from . import experiments as X
from . import revision as RV
from . import runner as RN
from .splits import random_splits

CODE_VERSION = "p2x-1"
N_SPLITS = 200
BOOT_B = 1000
BOOT_SPLITS = 50
ALPHAS = (0.1, 0.2)

# dataset registry of the revision tables (K / n_test as PREREG / E1)
X2_DATASETS: dict[str, dict[str, Any]] = {
    "cf_ton": {"K": 150, "n_test": 150, "sids": ("S_TON.stat", "S_TON.fuse_s0"), "label": "CF-TON",
               "status": "pre-registered dataset"},
    "cf_ton_h": {"K": 100, "n_test": 100, "sids": ("S_TON_H.stat", "S_TON_H.fuse_s0"), "label": "CF-TON-H",
                 "status": "pre-registered dataset"},
    "cf_gotham": {"K": 100, "n_test": 100, "sids": ("S_GOT.stat",), "label": "CF-GOTHAM v1",
                  "status": "pre-registered dataset (v1)"},
    "cf_gotham_v2": {"K": 100, "n_test": 100, "sids": ("S_GOT2.stat",), "label": "CF-GOTHAM v2",
                     "status": "post-PREREG deviation P1 (v2)"},
    "iot23": {"K": 83, "n_test": 83, "sids": ("S_CF.stat", "S_CF.fuse_s0"), "label": "IoT-23 P-WIN",
              "status": "not pre-registered protocol (P-WIN)"},
}
SPLIT_METHODS = ("crc_bh", "crc_st", "crc_fdx020", "crc_fdx030", "ltt", "bh_iid", "bh_iid_pooled", "by_ref",
                 "bh_ref", "b7", "b8")
FIXED_METHODS = ("B1", "B2", "B4")
LOSO_METHODS = ("crc_bh", "crc_st", "crc_fdx020", "crc_fdx030", "ltt", "bh_iid", "bh_iid_pooled", "by_ref",
                "bh_ref", "b7", "b8")
CHAIN_METHODS = ("chain_composite", "chain_naive", "chain_rt", "chain_rt_bh")
L_BINS = ((1,), (2,), (3,), (4,), (5, 6, 7, 8, 9, 10))


def x2_root() -> Path:
    return results_root() / "exp" / "X2"


def _hash(cfg: dict, *extra: str) -> str:
    h = blake3.blake3()
    h.update(json.dumps(cfg, sort_keys=True, default=str).encode())
    for e in extra:
        h.update(str(e).encode())
    return h.hexdigest()[:16]


def _clean(x):
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def write_cell_dir(path: Path, cell: dict, per_split: pl.DataFrame | None = None,
                   reps: dict[str, np.ndarray] | None = None, extra_parquet: dict[str, pl.DataFrame] | None = None) -> Path:
    """Atomic directory write: cell.json (+ per_split.parquet, boot_<name>.npy, <name>.parquet)."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    tmp.mkdir(parents=True)
    if per_split is not None:
        per_split.write_parquet(tmp / "per_split.parquet")
    for k, a in (reps or {}).items():
        np.save(tmp / f"boot_{k}.npy", a)
    for k, df in (extra_parquet or {}).items():
        df.write_parquet(tmp / f"{k}.parquet")
    (tmp / "cell.json").write_text(json.dumps(_clean(cell), indent=1, default=str))
    if path.exists():
        old = path.with_name(f".{path.name}.old-{os.getpid()}-{time.time_ns()}")
        os.replace(path, old)
        os.replace(tmp, path)
        shutil.rmtree(old, ignore_errors=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, path)
    return path


def cached(path: Path, h: str, overwrite: bool) -> dict | None:
    p = path / "cell.json"
    if p.exists() and not overwrite:
        c = json.loads(p.read_text())
        if c.get("config_hash") == h:
            return c
    return None


def frame(res: dict[str, np.ndarray], seeds: np.ndarray) -> pl.DataFrame:
    cols = {"seed": np.asarray(seeds, dtype=np.int64)}
    for k, v in res.items():
        v = np.asarray(v)
        if v.ndim == 1 and v.size == cols["seed"].size:
            cols[k] = v.astype(float)
    return pl.DataFrame(cols)


def ci_block(summ: dict, reps: np.ndarray | None, metrics: Sequence[str]) -> dict:
    out = {}
    for j, m in enumerate(metrics):
        pt = (summ.get(m) or {}).get("mean")
        d = {"point": pt, "split_sd": (summ.get(m) or {}).get("sd"), "split_q05": (summ.get(m) or {}).get("q05"),
             "split_q95": (summ.get(m) or {}).get("q95")}
        if reps is not None:
            d.update(RV.ci_dict(reps[:, j], point=pt))
        out[m] = d
    return out


# =========================================================================== conditional cells (task 1 / 3)
def _aux(ds: str, sid: str, method: str):
    if method != "b7":
        return None
    a = X.aux_for(ds, sid)
    return a or None


def cond_cell(task: dict) -> dict:
    """Split-based conditional-reliability cell: 200 PREREG split seeds (point) + incident-pool bootstrap."""
    ds, sid, method, alpha = task["ds"], task["sid"], task["method"], float(task["alpha"])
    K, n_test = int(task["K"]), task.get("n_test")
    B, nb = int(task.get("B", BOOT_B)), int(task.get("boot_splits", BOOT_SPLITS))
    group = task.get("group", "conditional")
    tsid = task.get("test_sid")
    path = x2_root() / group / ds / (f"{method}__{sid}" + (f"~{tsid}" if tsid else "")) / f"a{alpha}"
    bank = LossBank.load(task["bank_path"]) if task.get("bank_path") else X.load_bank(ds, sid)
    tbank = LossBank.load(task["test_bank_path"]) if task.get("test_bank_path") else None
    if tbank is not None and tbank.index["incident_id"].to_list() != bank.index["incident_id"].to_list():
        raise RuntimeError("test bank is not row-aligned with the calibration bank")
    fixed = None
    src = ""
    if method in RV.FIXED:
        fixed = X.oracle_counts(bank) if method == "oracle" else X.baseline_counts(bank, ds, method)
        sp_ = X.baseline_dir(ds, method) / "summary.json"
        src = sp_.read_text() if sp_.exists() else ""
    aux = _aux(ds, sid, method)
    if method == "b7" and aux is None:
        return {"task": task, "skipped": "no OOF scores for B7"}
    cfg = {"group": group, "ds": ds, "sid": sid, "method": method, "alpha": alpha, "K": K, "n_test": n_test,
           "splits": N_SPLITS, "B": B, "boot_splits": nb, "seed": 0, "code": CODE_VERSION,
           "prereg": X.PREREG_BLAKE3, "mask": task.get("mask_name")}
    for k_, v_ in (("test_sid", tsid), ("bank_path", task.get("bank_path")), ("test_bank_path", task.get("test_bank_path")),
                   ("inject_q", task.get("inject_q"))):
        if v_ is not None:              # only when set: keeps the hash of plain cells unchanged
            cfg[k_] = v_
    h = _hash(cfg, X.bank_hash(bank), blake3.blake3(src.encode()).hexdigest(),
              X.bank_hash(tbank) if tbank is not None else "")
    c = cached(path, h, task.get("overwrite", False))
    if c is not None:
        return {"task": task, "path": str(path), "cached": True}
    mask_col = None
    if task.get("mask_rows") is not None:
        mask_col = np.zeros(bank.n, dtype=bool)
        mask_col[np.asarray(task["mask_rows"], dtype=np.int64)] = True
    t0 = time.perf_counter()
    sp = random_splits(bank.n, K, range(N_SPLITS), n_test)
    res = RV.x2_eval(bank, method, sp, alpha, aux=aux, fixed=fixed, mask_col=mask_col, test_bank=tbank)
    if task.get("inject_q") is not None and tbank is not None:
        sel = X.select(bank, method, sp, alpha, test_bank=tbank)
        res["remark2_bound"] = X.r3_bound_terms(tbank, sel, int(task["inject_q"]), alpha)
    summ = RV.summarize_splits(res)
    no_claims = float(np.nansum(res["sum_R"])) == 0
    metrics = list(RV.X2_METRICS) + ([m for m in RV.CHAIN_X2 if m in res] if task.get("chains") else [])
    reps = None
    if B > 0 and not no_claims:
        reps = RV.x2_bootstrap(bank, method, alpha, K=K, n_test=n_test, B=B, n_splits=nb, seed=0, aux=aux,
                               fixed=fixed, metrics=metrics, mask_col=mask_col, test_bank=tbank)
    cell = {"config": cfg, "config_hash": h, "bank_input_hash": X.bank_hash(bank), "status": RV.STATUS,
            "status_note": RV.STATUS_NOTE, "no_claims_in_any_split": no_claims,
            "metrics": ci_block(summ, reps, metrics), "load_ratio": summ.get("load_ratio", {}).get("mean"),
            "remark2_bound": (summ.get("remark2_bound") or {}).get("mean"),
            "descriptors": X.descriptors(bank) if task.get("descriptors") else None,
            "n_pool": int(bank.n), "seconds": time.perf_counter() - t0,
            "ci_kind": (f"incident-pool cluster bootstrap (P2-C stream), B={B} x {nb} splits, percentile 95%; "
                        "ucb95_normal = point + 1.645 SE, ucb95_pct = 95th percentile of replicates") if reps is not None
            else "none (no claims in any split: degenerate)"}
    write_cell_dir(path, cell, frame(res, sp.seeds), {"main": reps} if reps is not None else None)
    return {"task": task, "path": str(path), "seconds": cell["seconds"]}


# =========================================================================== P-LOSO cells
def loso_cell(task: dict) -> dict:
    ds, sid, method, alpha = "iot23", task["sid"], task["method"], float(task["alpha"])
    B = int(task.get("B", BOOT_B))
    path = x2_root() / "conditional_loso" / ds / f"{method}__{sid}" / f"a{alpha}"
    bank = X.load_bank(ds, sid)
    g = X.scenario_groups(bank)
    fixed = None
    if method in RV.FIXED:
        fixed = X.baseline_counts(bank, ds, method)
    aux = _aux(ds, sid, method)
    if method == "b7" and aux is None:
        return {"task": task, "skipped": "no OOF scores for B7"}
    cfg = {"group": "conditional_loso", "ds": ds, "sid": sid, "method": method, "alpha": alpha, "B": B,
           "code": CODE_VERSION, "prereg": X.PREREG_BLAKE3, "protocol": "P-LOSO (K = #scenarios - 1)"}
    h = _hash(cfg, X.bank_hash(bank))
    if cached(path, h, task.get("overwrite", False)) is not None:
        return {"task": task, "path": str(path), "cached": True}
    t0 = time.perf_counter()
    res, cols = RV.loso_eval(bank, method, alpha, groups=g, fixed=fixed, aux=aux)
    reps = RV.loso_bootstrap(bank, method, alpha, groups=g, B=B, fixed=fixed, aux=aux) if B > 0 and res["sum_R"] > 0 \
        else None
    jk = RV.loso_jackknife(bank, method, alpha, groups=g, fixed=fixed, aux=aux,
                           metrics=("fdr_scen", "fdr", "cond_fdp", "pooled_fdr", "frac_empty", "fdx_020", "cond_fdp_scen",
                                    "fdr_lenient"))
    mets = {}
    for j, m in enumerate(RV.LOSO_METRICS):
        d = {"point": res.get(m)}
        if reps is not None:
            d.update(RV.ci_dict(reps[:, j], point=res.get(m)))
        if m in jk:
            d["jackknife"] = jk[m]
        mets[m] = d
    scen = sorted(set(g.tolist()))
    cell = {"config": cfg, "config_hash": h, "bank_input_hash": X.bank_hash(bank), "status": RV.STATUS,
            "status_note": RV.STATUS_NOTE, "metrics": mets, "lam_idx_per_scenario": dict(zip(scen, cols)),
            "n_scenarios": len(scen), "n_windows": int(bank.n), "seconds": time.perf_counter() - t0,
            "ci_kind": f"scenario-cluster bootstrap with LOSO recomputation, B={B}, percentile 95% (bounded); "
                       "jackknife = delete-one-scenario (t_{n-1})"}
    write_cell_dir(path, cell, None, {"main": reps} if reps is not None else None)
    return {"task": task, "path": str(path), "seconds": cell["seconds"]}


# =========================================================================== Gotham-real (fixed calibration bank)
def full_bank_col(bank: LossBank, method: str, alpha: float, delta: float = 0.1) -> int:
    spec = RN.COLUMN_METHODS[method]
    rows = np.arange(bank.n)[None, :]
    return int(RN.choose_columns(bank, spec, rows, np.ones(rows.shape, bool), alpha, delta)[0])


def gotham_rows(cal: tuple[str, str], test: tuple[str, str], methods: Sequence[str], alphas=ALPHAS) -> list[dict]:
    """Per (method, alpha): lambda-hat from ALL incidents of the calibration bank, claims on the 9 Gotham-real
    incidents. Counts only (no CI: n = 9 fixed incidents; R1-M8)."""
    cb, tb = X.load_bank(*cal), X.load_bank(*test)
    benign = np.array(["benign" in i for i in tb.index["incident_id"].to_list()])
    out = []
    for m in methods:
        for a in alphas:
            if m in RN.COLUMN_METHODS:
                col = full_bank_col(cb, m, a)
                fam = RN.COLUMN_METHODS[m]["family"]
                R = tb.t("R", fam)[:, col].astype(np.int64)
                ncal = int((cb.t("R", fam)[:, col] > 0).sum())
            elif m == "by_ref":
                from ..baselines.by_ref import by_ref_R
                R, col, ncal = by_ref_R(tb, a), None, None
            elif m == "bh_ref":
                from ..baselines.naive import bh_ref_plain_R
                R, col, ncal = bh_ref_plain_R(tb, a), None, None
            else:
                continue
            c = tb.counts(np.arange(tb.n), R)
            Rf, V, TP = c["R"].astype(float), c["V"].astype(float), c["TP"].astype(float)
            fdp = V / np.maximum(Rf, 1)
            att = ~benign
            row = {"cal": f"{cal[0]}/{cal[1]}", "test": f"{test[0]}/{test[1]}", "method": m, "alpha": a,
                   "lam_idx": col, "cal_incidents_with_claims": ncal, "K": int(cb.n), "n_incidents": int(tb.n),
                   "n_attack": int(att.sum()), "claimed_all": int((Rf > 0).sum()), "claimed_attack": int((Rf[att] > 0).sum()),
                   "claimed_benign": int((Rf[benign] > 0).sum()), "frac_empty_all": float((Rf == 0).mean()),
                   "frac_empty_attack": float((Rf[att] == 0).mean()), "fdr_all": float(fdp.mean()),
                   "fdr_attack": float(fdp[att].mean()), "cond_fdp": float(fdp[Rf > 0].mean()) if (Rf > 0).any() else None,
                   "pooled_fdr": float(V.sum() / max(Rf.sum(), 1)), "sum_R": float(Rf.sum()), "sum_V": float(V.sum()),
                   "sum_TP": float(TP.sum()), "n_claims": float(Rf.mean()), "n_false": float(V.mean()),
                   "fdx_020": float((fdp > 0.2 + 1e-12).mean()), "fdx_030": float((fdp > 0.3 + 1e-12).mean()),
                   "cond_fdx_020": float((fdp[Rf > 0] > 0.2 + 1e-12).mean()) if (Rf > 0).any() else None,
                   "fdp_min_claimed": float(fdp[Rf > 0].min()) if (Rf > 0).any() else None,
                   "fdp_max_claimed": float(fdp[Rf > 0].max()) if (Rf > 0).any() else None,
                   "per_incident": [{"incident": X.inc_of(i), "R": int(r), "V": int(v), "TP": int(t)}
                                    for i, r, v, t in zip(tb.index["incident_id"].to_list(), Rf, V, TP)]}
            out.append(row)
    return out


# =========================================================================== Mondrian (task 4d)
def mondrian_cell(task: dict) -> dict:
    ds, sid, alpha = task["ds"], task["sid"], float(task["alpha"])
    method = task.get("method", "crc_bh")
    d = X2_DATASETS[ds]
    K, n_test = d["K"], d["n_test"]
    B = int(task.get("B", BOOT_B))
    path = x2_root() / "mondrian" / ds / f"{method}__{sid}" / f"a{alpha}"
    bank = X.load_bank(ds, sid)
    strata = RV.label_free_strata(bank, "N_tercile")
    cfg = {"group": "mondrian", "ds": ds, "sid": sid, "method": method, "alpha": alpha, "K": K, "n_test": n_test,
           "strata": "N_tercile (pool terciles)", "B": B, "code": CODE_VERSION}
    h = _hash(cfg, X.bank_hash(bank))
    if cached(path, h, task.get("overwrite", False)) is not None:
        return {"task": task, "path": str(path), "cached": True}
    t0 = time.perf_counter()
    ks = sorted(int(k) for k in np.unique(strata))
    met = ("fdr", "str_mean", "frac_empty", "cond_fdp", "pooled_fdr", "n_claims", "fdx_020", "abstain_split")

    def selector(bk, sp, rows, cache):
        return RV.mondrian_select(bk, method, sp, alpha, strata, pool_rows=rows)

    def evaluate_all(sp, rows=None):
        selm = RV.mondrian_select(bank, method, sp, alpha, strata, pool_rows=rows)
        selc = X.select(bank, method, sp, alpha, pool_rows=rows)
        out = {}
        for tag, sel in (("mondrian", selm), ("marginal", selc)):
            out[(tag, "all")] = RV.x2_eval(bank, method, sp, alpha, pool_rows=rows, selection=sel, chains=False)
            for k in ks:
                out[(tag, k)] = RV.x2_eval(bank, method, sp, alpha, pool_rows=rows, selection=sel, chains=False,
                                           mask_col=(strata == k))
        return out, selm

    sp = random_splits(bank.n, K, range(N_SPLITS), n_test)
    res, selm = evaluate_all(sp)
    point = {f"{t}|{k}": RV.summarize_splits(r) for (t, k), r in res.items()}
    # abstention per stratum (Mondrian): share of splits with lambda-hat_k = 0
    abst = {int(k): float((selm.stratum_cols[:, int(k)] == 0).mean()) for k in ks}
    reps = {}
    if B > 0:
        keys = list(res)
        arr = np.full((B, len(keys), len(met)), np.nan)
        for b, rows in enumerate(X.boot_pools(bank.n, B, 0)):
            spb = random_splits(rows.size, K, np.arange(BOOT_SPLITS), n_test)
            rb, _ = evaluate_all(spb, rows)
            for i, key in enumerate(keys):
                arr[b, i] = RV.replicate_vector(rb[key], met)
        for i, key in enumerate(keys):
            reps[f"{key[0]}|{key[1]}"] = arr[:, i, :]
    mets = {}
    for key, s in point.items():
        mets[key] = {m: {"point": (s.get(m) or {}).get("mean"), **(RV.ci_dict(reps[key][:, j], (s.get(m) or {}).get("mean"))
                                                                   if key in reps else {})}
                     for j, m in enumerate(met)}
    Ncut = np.quantile(bank.col("N"), [1 / 3, 2 / 3])
    cell = {"config": cfg, "config_hash": h, "status": RV.STATUS, "status_note": RV.STATUS_NOTE, "metrics": mets,
            "stratum_abstain_split_mondrian": abst, "N_cut_points": Ncut.tolist(),
            "n_pool_per_stratum": {int(k): int((strata == k).sum()) for k in ks}, "seconds": time.perf_counter() - t0,
            "ci_kind": f"incident-pool bootstrap B={B} x {BOOT_SPLITS} splits (strata fixed from the original pool)"}
    write_cell_dir(path, cell, None, {k.replace("|", "_"): v for k, v in reps.items()})
    return {"task": task, "path": str(path), "seconds": cell["seconds"]}


# =========================================================================== chains (task 3)
def chain_lengths(ds: str, bank: LossBank) -> np.ndarray:
    p = X.exp_root("E5") / ds / "chain_length.parquet"
    if p.exists():
        df = pl.read_parquet(p)
        m = dict(zip(df["incident_id"].to_list(), df["L"].to_list()))
        ids = bank.index["incident_id"].to_list()
        if all(i in m for i in ids):
            return np.array([m[i] for i in ids], dtype=np.int64)
    return X.chain_length_strata(ds, bank)


def chain_cell(task: dict) -> dict:
    ds, sid, method, alpha = task["ds"], task["sid"], task["method"], float(task["alpha"])
    d = X2_DATASETS[ds]
    K, n_test = d["K"], d["n_test"]
    B = int(task.get("B", BOOT_B))
    path = x2_root() / "chains" / ds / f"{method}__{sid}" / f"a{alpha}"
    bank = X.load_bank(ds, sid)
    L = chain_lengths(ds, bank)
    cfg = {"group": "chains", "ds": ds, "sid": sid, "method": method, "alpha": alpha, "K": K, "n_test": n_test,
           "L_bins": [list(b) for b in L_BINS], "B": B, "code": CODE_VERSION}
    h = _hash(cfg, X.bank_hash(bank))
    if cached(path, h, task.get("overwrite", False)) is not None:
        return {"task": task, "path": str(path), "cached": True}
    t0 = time.perf_counter()
    met = tuple(RV.CHAIN_X2) + ("fdr", "str_mean", "recall_pooled", "frac_empty", "n_claims", "pooled_fdr")
    masks = {"all": None}
    for b in L_BINS:
        name = f"L={b[0]}" if len(b) == 1 else f"L={b[0]}+"
        masks[name] = np.isin(L, b)
    sp = random_splits(bank.n, K, range(N_SPLITS), n_test)
    sel = X.select(bank, method, sp, alpha)
    point = {}
    for name, mk in masks.items():
        point[name] = RV.summarize_splits(RV.x2_eval(bank, method, sp, alpha, selection=sel, chains=True, mask_col=mk))
    reps = {name: np.full((B, len(met)), np.nan) for name in masks}
    for b, rows in enumerate(X.boot_pools(bank.n, B, 0)):
        spb = random_splits(rows.size, K, np.arange(BOOT_SPLITS), n_test)
        selb = X.select(bank, method, spb, alpha, pool_rows=rows)
        for name, mk in masks.items():
            reps[name][b] = RV.replicate_vector(RV.x2_eval(bank, method, spb, alpha, pool_rows=rows, selection=selb,
                                                           chains=True, mask_col=mk), met)
    mets = {name: {m: {"point": (point[name].get(m) or {}).get("mean"),
                       **RV.ci_dict(reps[name][:, j], (point[name].get(m) or {}).get("mean"))}
                   for j, m in enumerate(met)} for name in masks}
    cell = {"config": cfg, "config_hash": h, "status": RV.STATUS, "status_note": RV.STATUS_NOTE, "metrics": mets,
            "n_pool_per_bin": {name: (int(bank.n) if mk is None else int(mk.sum())) for name, mk in masks.items()},
            "seconds": time.perf_counter() - t0,
            "ci_kind": f"incident-pool bootstrap B={B} x {BOOT_SPLITS} splits; stratum metrics restrict the TEST "
                       "incidents to the L bin (calibration on all incidents)"}
    write_cell_dir(path, cell, None, {k.replace("=", "").replace("+", "p"): v for k, v in reps.items()})
    return {"task": task, "path": str(path), "seconds": cell["seconds"]}


# =========================================================================== two-way bootstrap (task 5a)
def episode_clusters(ds: str, bank: LossBank, with_lookalikes: bool = True,
                     bg_tile_s: float | None = None) -> tuple[list[list[int]], dict[str, int], dict]:
    """Per bank row: ids of the source-library episodes (gen.json summary.lib_eids; + D3 look-alike ids la_ids)
    spliced into the incident (generator log; GT-side, used for resampling only). Optional background tiles of
    ``bg_tile_s`` seconds of the pool's background capture covered by the incident window."""
    ids: dict[str, int] = {}
    out = []
    n_with = 0
    for iid in bank.index["incident_id"].to_list():
        inc = X.inc_of(iid)
        parts = inc.split("/")
        gp = results_root() / "interim" / parts[0] / "incidents" / Path(*parts[1:]) / "gen.json"
        g = json.loads(gp.read_text())
        s = g.get("summary", {})
        keys = [f"ep:{e}" for e in (s.get("lib_eids") or [])]
        if with_lookalikes:
            keys += [f"la:{e}" for e in (s.get("la_ids") or [])]
        if bg_tile_s:
            w0, w1 = s.get("window") or [None, None]
            if w0 is not None:
                k0, k1 = int(math.floor(w0 / bg_tile_s)), int(math.floor(w1 / bg_tile_s))
                keys += [f"bg:{g.get('pool')}:{k}" for k in range(k0, k1 + 1)]
        n_with += bool(keys)
        out.append(sorted({ids.setdefault(k, len(ids)) for k in keys}))
    n_ep = sum(1 for k in ids if k.startswith("ep:"))
    n_la = sum(1 for k in ids if k.startswith("la:"))
    uses = np.bincount(np.concatenate([np.asarray(c, dtype=np.int64) for c in out if c]), minlength=len(ids)) \
        if ids else np.zeros(0)
    info = {"n_incidents": int(bank.n), "n_incidents_with_clusters": int(n_with), "n_clusters": len(ids),
            "n_distinct_library_episodes": n_ep, "n_distinct_lookalikes": n_la,
            "mean_incidents_per_cluster": float(uses.mean()) if uses.size else None,
            "max_incidents_per_cluster": int(uses.max()) if uses.size else None,
            "mean_clusters_per_incident": float(np.mean([len(c) for c in out]))}
    return out, ids, info


def twoway_cell(task: dict) -> dict:
    ds, sid, method, alpha = task["ds"], task["sid"], task.get("method", "crc_bh"), float(task["alpha"])
    d = X2_DATASETS[ds]
    K, n_test = d["K"], d["n_test"]
    B = int(task.get("B", BOOT_B))
    variant = task.get("variant", "episodes")
    path = x2_root() / "twoway" / ds / f"{method}__{sid}" / f"a{alpha}" / variant
    bank = X.load_bank(ds, sid)
    clusters, ids, info = episode_clusters(ds, bank, with_lookalikes=True,
                                           bg_tile_s=1800.0 if variant == "episodes+bgtiles" else None)
    cfg = {"group": "twoway", "ds": ds, "sid": sid, "method": method, "alpha": alpha, "K": K, "n_test": n_test,
           "variant": variant, "B": B, "code": CODE_VERSION}
    h = _hash(cfg, X.bank_hash(bank), str(len(ids)))
    if cached(path, h, task.get("overwrite", False)) is not None:
        return {"task": task, "path": str(path), "cached": True}
    t0 = time.perf_counter()
    met = ("fdr", "str_mean", "str_pooled", "pooled_fdr", "cond_fdp", "frac_empty", "n_claims", "cer_link_mean")
    sp = random_splits(bank.n, K, range(N_SPLITS), n_test)
    summ = RV.summarize_splits(RV.x2_eval(bank, method, sp, alpha))
    rep_inc = RV.x2_bootstrap(bank, method, alpha, K=K, n_test=n_test, B=B, metrics=met)
    rep_two = RV.x2_bootstrap(bank, method, alpha, K=K, n_test=n_test, B=B, metrics=met,
                              sampler=RV.episode_pool_sampler(clusters, len(ids), B, seed=0))
    mets = {}
    for j, m in enumerate(met):
        pt = (summ.get(m) or {}).get("mean")
        a, b = RV.ci_dict(rep_inc[:, j], pt), RV.ci_dict(rep_two[:, j], pt)
        mets[m] = {"point": pt, "incident_only": a, "two_way": b,
                   "se_ratio": (b["se"] / a["se"]) if a.get("se") and b.get("se") else None}
    cell = {"config": cfg, "config_hash": h, "status": RV.STATUS, "status_note": RV.STATUS_NOTE, "metrics": mets,
            "clusters": info, "seconds": time.perf_counter() - t0,
            "ci_kind": f"incident-only = P2-C incident-pool bootstrap; two-way = episode-cluster multiple-membership "
                       f"bootstrap (revision.episode_pool_sampler), B={B} x {BOOT_SPLITS} splits"}
    write_cell_dir(path, cell, None, {"incident": rep_inc, "twoway": rep_two})
    return {"task": task, "path": str(path), "seconds": cell["seconds"]}


# =========================================================================== dispatcher
def run_task(task: dict) -> dict:
    kind = task["kind"]
    try:
        if kind == "cond":
            return cond_cell(task)
        if kind == "loso":
            return loso_cell(task)
        if kind == "mondrian":
            return mondrian_cell(task)
        if kind == "chain":
            return chain_cell(task)
        if kind == "twoway":
            return twoway_cell(task)
        if kind == "lagstr":
            from . import revision_easiness as RE
            return RE.lagstr_cell(task)
        if kind == "decoy":
            from . import revision_easiness as RE
            return RE.decoy_cell(task)
    except Exception as e:  # pragma: no cover - surfaced in the driver log
        import traceback
        return {"task": task, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    raise ValueError(kind)
