"""EXPLORATORY diagnostics requested by the coordinator after E15 (owner: P2-G). Nothing here is pre-registered;
every number written from this module carries status 'exploratory'.

(a) timing-artifact diagnosis: per-feature and per-group timing/position probe AUCs for the E15 probe targets;
(b) timing-ablated statistical scorers: S_TON refit on CF-TON TRAIN without lag / timing-derived features, with
    P2-F's exact recipe (scripts/08 lgb_params, scoring/fusion.cross_fit: 5 incident-grouped folds, fold seed 0,
    early stopping on the next fold, final refit at the mean best iteration, S_0 = OOF scores of training strict
    nulls); its CF-TON cal/test bank is built with P2-C's build_loss_tables through P2-F's driver.build_incident
    (the same path as scripts/11). Models and banks live under results/exp/X1/ (never results/models|losses);
(c) Gotham scorer transfer vs bank exchangeability: in-domain S_GOT scores of the Gotham-real incidents (label-free
    scoring with S_GOT's final model), within-incident AUCs, and the same case study with S_GOT + CF-GOTHAM-T.
No calibration/test incident is used for any fitting or choice here.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from ..certify.losses import LossBank, build_loss_tables
from ..features import stats as F
from ..io.paths import P2_ROOT, atomic_write_json, atomic_write_parquet, results_root

LAG_FEATURES = ("llag_start", "llag_gap", "overlap", "lag_llr", "rank_lag_u", "rank_llr_u")
TIMING_EXTRA = ("n_pred_v", "act_before_v", "act_between_u", "ldur_u", "ldur_v", "startcv_u", "startcv_v")
ABLATIONS = {
    "full": (),                                    # reproduction check of P2-F's S_TON
    "nolag": LAG_FEATURES,
    "notiming": LAG_FEATURES + TIMING_EXTRA,
}
PROBE_GROUPS = {
    "time_since_t0": ("t_since_t0", "t_norm"),
    "rank": ("rank_norm",),
    "gaps": ("gap_mean", "gap_median", "gap_std", "gap_min", "gap_max", "gap_cv", "gap_zero", "n_rec_probe"),
    "fractional_seconds": ("frac_ts", "frac_rec_mean", "frac_rec_zero"),
}


def x_root() -> Path:
    return results_root() / "exp" / "X1"


# =========================================================================== (a) probe diagnosis
def probe_target_frame(df: pl.DataFrame, target: str) -> tuple[pl.DataFrame, np.ndarray]:
    if target == "i":
        d = df.filter(pl.col("kind").is_in(["main", "D1", "D2", "D3"]))
        y = (d["kind"] == "main").to_numpy().astype(int)
    elif target == "ii":
        d = df.filter(pl.col("kind").is_in(["D3", "background"]))
        y = (d["kind"] == "D3").to_numpy().astype(int)
    elif target == "i_attack":
        d = df.filter(pl.col("kind").is_in(["main", "D1", "D2"]))
        y = (d["kind"] == "main").to_numpy().astype(int)
    elif target == "ii_typematched":
        types = set(df.filter(pl.col("kind") == "D3")["type"].drop_nulls().to_list())
        d = df.filter(pl.col("kind").is_in(["D3", "background"]) & pl.col("type").is_in(list(types)))
        y = (d["kind"] == "D3").to_numpy().astype(int)
    else:
        raise ValueError(target)
    return d, y


def univariate_auc(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """AUC of a single raw feature (direction-free separability = max(AUC, 1 - AUC))."""
    from .gates import auc
    ok = np.isfinite(x)
    a = auc(x[ok & (y == 1)], x[ok & (y == 0)])
    return {"auc": float(a), "separability": float(max(a, 1 - a)) if np.isfinite(a) else math.nan,
            "median_pos": float(np.median(x[ok & (y == 1)])) if (ok & (y == 1)).any() else math.nan,
            "median_neg": float(np.median(x[ok & (y == 0)])) if (ok & (y == 0)).any() else math.nan}


def timing_diagnosis(df: pl.DataFrame, params: dict, rounds: int = 200, B: int = 100) -> dict[str, Any]:
    """Per target: univariate AUCs of every timing/position feature, grouped LightGBM probe AUCs per feature group
    (incident-grouped 5-fold, as E15), and the driver (group with the largest probe AUC)."""
    from .experiments import probe_auc
    out: dict[str, Any] = {}
    for target in ("i", "ii", "i_attack", "ii_typematched"):
        d, y = probe_target_frame(df, target)
        uni = {f: univariate_auc(d[f].to_numpy().astype(float), y) for g in PROBE_GROUPS.values() for f in g
               if f in d.columns}
        grp = {g: probe_auc(df, target, feats, params=params, rounds=rounds, B=B) for g, feats in PROBE_GROUPS.items()}
        allt = probe_auc(df, target, tuple(f for g in PROBE_GROUPS.values() for f in g), params=params, rounds=rounds, B=B)
        drv = max(grp, key=lambda g: (grp[g].get("auc") if np.isfinite(grp[g].get("auc", np.nan)) else -1))
        out[target] = {"n": int(y.size), "n_pos": int(y.sum()), "univariate": uni,
                       "groups": {g: {k: v for k, v in r.items() if k in ("auc", "lo", "hi", "top_features")}
                                  for g, r in grp.items()},
                       "all_timing": {k: v for k, v in allt.items() if k in ("auc", "lo", "hi", "top_features")},
                       "driver_group": drv}
    return out


# =========================================================================== (b) ablated scorers
def _s08():
    spec = importlib.util.spec_from_file_location("s08_fit_stats", P2_ROOT / "scripts" / "08_fit_stats.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ablation_names(variant: str) -> tuple[str, ...]:
    drop = set(ABLATIONS[variant])
    return tuple(n for n in F.FEATURES if n not in drop)


def fit_ablated(variant: str, scorer: str = "S_TON", log=print, check=lambda: None) -> dict:
    """Refit the statistical scorer on its TRAINING incidents with the ablated feature set (P2-F recipe)."""
    from ..scoring import fusion as FU
    out = x_root() / "models" / f"{scorer}.{variant}"
    meta_p = out / "meta.json"
    names = ablation_names(variant)
    if meta_p.exists() and json.loads(meta_p.read_text()).get("names") == list(names):
        return json.loads(meta_p.read_text())
    m08 = _s08()
    cfg = m08.load_cfg()
    pairs = m08.training_set(scorer, cfg)
    incs = m08.load_training(pairs)
    tr = cfg["training"]
    t0 = time.perf_counter()
    res = FU.cross_fit(incs, names, params=m08.lgb_params(cfg), n_folds=int(tr["n_folds"]),
                       fold_seed=int(tr["fold_seed"]), max_rounds=int(tr["max_rounds"]),
                       es_rounds=int(tr["early_stopping"]), W=float(tr["W"]), log=log, check=check)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lgbm.txt").write_text(res.booster.model_to_string())
    res.priors.save(out / "priors.json")
    np.save(out / "null_pool.npy", res.null_pool())
    oof = res.oof_frame().with_columns(pl.Series("cand_id", np.concatenate(
        [i.feats["cand_id"].to_numpy() for i in incs]).astype(np.int64)))
    atomic_write_parquet(oof, out / "oof.parquet")
    scores = FU.split_by_incident(res.oof, res.sizes)
    ys = [i.y for i in incs]
    meta = {"scorer": scorer, "variant": variant, "names": list(names), "dropped": list(ABLATIONS[variant]),
            "n_features": len(names), "n_train_incidents": len(incs), "best_iters": res.best_iters,
            "final_rounds": res.final_rounds, "s0_size": int(res.null_pool().size),
            "oof_within_auc_strict": FU.within_auc(scores, ys, "strict"),
            "oof_within_auc_zero": FU.within_auc(scores, ys, "zero"), "seconds": time.perf_counter() - t0,
            "params": res.params, "status": "exploratory", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    atomic_write_json(meta_p, meta)
    return meta


def bank_path(ds: str, sid: str) -> Path:
    return x_root() / "banks" / ds / sid / "bank"


def build_ablated_bank(variant: str, ds: str = "cf_ton", scorer: str = "S_TON", n_jobs: int = 8,
                       log=print, check=lambda: None) -> LossBank:
    """Score the cal/test pool with the ablated final model and build its loss bank (same path as scripts/11)."""
    from ..features.priors import PairPriors
    from ..scoring import driver as D
    from ..scoring import fusion as FU
    md = x_root() / "models" / f"{scorer}.{variant}"
    names = ablation_names(variant)
    booster = FU.load_booster(md / "lgbm.txt")
    priors = PairPriors.load(md / "priors.json")
    s0 = np.load(md / "null_pool.npy")
    incs = D.list_incidents(ds, "caltest")
    I = []
    for k, inc in enumerate(incs):
        feats = D.load_features(ds, inc, write=False)
        s = FU.score_incident(booster, priors, FU.IncidentData(inc, feats), names)
        I.append(D.build_incident(ds, inc, s))
        if k % 50 == 0:
            check()
    sid = f"{scorer}.{variant}"
    return build_loss_tables(I, s0, bank_path(ds, sid), n_jobs=n_jobs, overwrite=True,
                             meta={"dataset": ds, "scorer_id": sid, "pool": "caltest", "status": "exploratory",
                                   "features": list(names)})


# =========================================================================== (c) Gotham: S_GOT in-domain on Gotham-real
def score_with(scorer: str, ds: str, incs: Sequence[str], tag: str = "stat") -> dict[str, np.ndarray]:
    """Label-free scores of incidents with a scorer's final statistical model (results/models/<scorer>)."""
    from ..features.priors import PairPriors
    from ..scoring import driver as D
    from ..scoring import fusion as FU
    md = D.models_dir(scorer)
    booster = FU.load_booster(md / "lgbm_stat.txt")
    priors = PairPriors.load(md / "priors.json")
    out = {}
    for inc in incs:
        feats = D.load_features(ds, inc, write=False)
        out[inc] = FU.score_incident(booster, priors, FU.IncidentData(inc, feats), F.FEATURES)
    return out


def build_scored_bank(ds: str, scorer: str, scores: dict[str, np.ndarray], sid: str) -> LossBank:
    from ..scoring import driver as D
    s0 = np.load(D.models_dir(scorer) / "null_pool.npy")
    I = [D.build_incident(ds, inc, s) for inc, s in scores.items()]
    return build_loss_tables(I, s0, bank_path(ds, sid), n_jobs=1, overwrite=True,
                             meta={"dataset": ds, "scorer_id": sid, "status": "exploratory"})
