"""B4 — lag-likelihood ("Hawkes-lite") baseline (DESIGN §6.3): score = the training-only lag log-likelihood
ratio of the candidate's type pair and lag bin (features/priors.py). Used two ways: the uncalibrated rule
"lag-LLR > 0", and as a non-LLM scorer inside CRC (its reference null pool = OOF lag-LLR of the training
strict nulls, priors fitted out of fold with the same incident folds as M_stat)."""
from __future__ import annotations

import numpy as np
import polars as pl

from certichain.features.priors import PairPriors, oof_prior_features


def lag_llr_scores(feats: pl.DataFrame, priors: PairPriors) -> np.ndarray:
    """lag-LLR per candidate of a base-feature frame (stats.base_features)."""
    if feats.height == 0:
        return np.zeros(0)
    return priors.lag_llr(feats["type_u_str"].to_numpy(), feats["type_v_str"].to_numpy(),
                          feats["lag_gap"].to_numpy().astype(np.float64))


def lag_llr_claims(scores: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return np.asarray(scores, dtype=np.float64) > threshold


def oof_lag_llr(feats_list, y_list, inc_ids, folds: dict, W: float = 1800.0) -> list[np.ndarray]:
    """Out-of-fold lag-LLR of training incidents (each incident scored by priors of the other folds)."""
    sizes = [f.height for f in feats_list]
    tu = np.concatenate([f["type_u_str"].to_numpy() for f in feats_list]) if feats_list else np.zeros(0)
    tv = np.concatenate([f["type_v_str"].to_numpy() for f in feats_list]) if feats_list else np.zeros(0)
    lg = np.concatenate([f["lag_gap"].to_numpy() for f in feats_list]).astype(np.float64) if feats_list else np.zeros(0)
    y = np.concatenate([np.asarray(v) for v in y_list]) if y_list else np.zeros(0)
    groups = np.repeat(np.array(inc_ids, dtype=object), sizes)
    _, ll = oof_prior_features(tu, tv, lg, y, groups, folds, W)
    return list(np.split(ll, np.cumsum(sizes)[:-1])) if sizes else []
