"""BY-ref: Benjamini-Yekutieli on reference p-values, per incident (DESIGN §6.3; [BY01]).

BH at level alpha / H_N, H_N = sum_{i<=N} 1/i, over the incident's N reference p-values.
No guarantee: reference p-values are not conformal for the test incident (incident-level
shifts make them anti-conservative), and BY's arbitrary-dependence protection does not fix that.
"""
from __future__ import annotations

import numpy as np

from ..certify.rules import TOL, bh_select


def harmonic(n: int) -> float:
    return float(np.sum(1.0 / np.arange(1, n + 1))) if n > 0 else 1.0


def by_select(p, alpha: float) -> np.ndarray:
    """Boolean mask of BY(alpha) on one incident's p-values."""
    p = np.asarray(p, dtype=np.float64)
    return bh_select(p, alpha / harmonic(p.size))


def by_ref_R(bank, alpha: float, rows=None) -> np.ndarray:
    """Selection sizes per incident (split-independent: no calibration)."""
    rows = np.arange(bank.n) if rows is None else np.asarray(rows)
    idx, seg, rank = bank.cand_index(rows)
    lens = np.diff(seg)
    out = np.zeros(rows.size, dtype=np.int64)
    if idx.size == 0:
        return out
    H = np.concatenate(([1.0], np.cumsum(1.0 / np.arange(1, int(lens.max()) + 1))))
    N = np.repeat(lens, lens)
    level = alpha / H[N]
    p = bank.csr["p"][idx]
    ok = p <= level * rank / N * (1.0 + TOL)
    val = np.where(ok, rank, 0)
    nz = lens > 0
    out[nz] = np.maximum.reduceat(val, seg[:-1][nz])
    return out
