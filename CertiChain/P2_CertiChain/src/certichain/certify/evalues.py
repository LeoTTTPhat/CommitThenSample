"""Group-conformal e-values (Lemma A, DESIGN §4.6) and e-BH.

Lemma A. If Q_1..Q_{K+1} in [0, 1] are exchangeable, E = (K+1) Q_{K+1} / (sum_{k<=K} Q_k + 1)
satisfies E[E] <= 1 (since sum_{k<=K} Q_k + 1 >= sum_{k<=K+1} Q_k and
E[Q_{K+1} / sum Q] = P(sum Q > 0) / (K+1)).

In Alg. 6 the per-look e-value multiplies the test batch's FDP implicitly: the certification
rule uses E_l := (K+1) / (sum_k Q_{k,l} + 1), which does NOT depend on the test incident.

e-BH (Wang & Ramdas 2022): with n hypotheses and e-values e_(1) >= ... >= e_(n),
k* = max{k : e_(k) >= n / (alpha k)}; reject every hypothesis with e >= e_(k*).
"""
from __future__ import annotations

import numpy as np

TOL = 1e-12


def group_evalue(Q_cal) -> float:
    """E_l = (K+1) / (sum_k Q_{k,l} + 1) for one look (Q_cal: K calibration batch FDPs)."""
    q = np.asarray(Q_cal, dtype=np.float64)
    if np.any((q < 0) | (q > 1)):
        raise ValueError("Q must lie in [0, 1]")
    return float((q.size + 1) / (q.sum() + 1.0))


def group_evalues(Q_cal_table) -> np.ndarray:
    """Vectorised over looks: Q_cal_table [K, L] -> E [L]."""
    q = np.asarray(Q_cal_table, dtype=np.float64)
    return (q.shape[0] + 1) / (q.sum(axis=0) + 1.0)


def lemma_a_evalue(Q_test, Q_cal) -> np.ndarray:
    """E = (K+1) Q_test / (sum Q_cal + 1); Q_cal [..., K] (last axis = calibration incidents)."""
    qc = np.asarray(Q_cal, dtype=np.float64)
    K = qc.shape[-1]
    return (K + 1) * np.asarray(Q_test, dtype=np.float64) / (qc.sum(axis=-1) + 1.0)


def ebh_k(e_desc: np.ndarray, n: int, alpha: float) -> int:
    """k* for e-values sorted descending (length may be < n when zeros are dropped)."""
    if e_desc.size == 0:
        return 0
    k = np.arange(1, e_desc.size + 1)
    ok = alpha * k * e_desc >= n * (1.0 - TOL)
    idx = np.flatnonzero(ok)
    return int(idx[-1] + 1) if idx.size else 0


def ebh(e, alpha: float) -> np.ndarray:
    """Boolean rejection mask of e-BH at level alpha over all len(e) hypotheses."""
    e = np.asarray(e, dtype=np.float64)
    n = e.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    e_desc = np.sort(e)[::-1]
    k = ebh_k(e_desc[e_desc > 0], n, alpha)
    if k == 0:
        return np.zeros(n, dtype=bool)
    return e >= e_desc[k - 1] * (1.0 - TOL)
