"""Reference p-values against a label-free null score pool S_0 (DESIGN §3.5).

    p = (1 + #{s' in S_0 : s' >= s}) / (|S_0| + 1)

S_0 is the cross-fitted pool of null scores from TRAINING data (fixed before any
calibration/test incident is seen, A2). The same formula with S_0 replaced by the pooled
calibration-incident null scores gives the BH-iid conformal p-value (baselines/bh_iid.py).

p is a non-increasing function of the score, so sorting candidates by score descending
sorts them by p ascending (the order every rule family in certify/rules.py uses).
"""
from __future__ import annotations

import numpy as np


def sort_null_pool(s0) -> np.ndarray:
    """Return S_0 as a sorted, finite float64 array (raises on NaN / empty input)."""
    a = np.asarray(s0, dtype=np.float64).ravel()
    if a.size == 0:
        raise ValueError("empty null pool S_0")
    if np.isnan(a).any():
        raise ValueError("NaN in null pool S_0")
    return np.sort(a, kind="stable")


def ref_pvalues(scores, null_pool_sorted: np.ndarray) -> np.ndarray:
    """Vectorised reference p-values; ``null_pool_sorted`` must be ascending (sort_null_pool).

    #{s' >= s} = |S_0| - #{s' < s} = |S_0| - searchsorted(S_0, s, side='left').
    """
    s = np.asarray(scores, dtype=np.float64)
    if np.isnan(s).any():
        raise ValueError("NaN score: a NaN would silently receive the smallest p-value")
    pool = np.asarray(null_pool_sorted, dtype=np.float64)
    n0 = pool.size
    if n0 == 0:
        raise ValueError("empty null pool")
    n_lt = np.searchsorted(pool, s, side="left")
    return (1.0 + (n0 - n_lt)) / (n0 + 1.0)


def pvalue_floor(n0: int) -> float:
    """Smallest attainable reference p-value, 1/(|S_0|+1) (resolution limit)."""
    return 1.0 / (n0 + 1.0)
