"""BH-iid: Jin-Candes conformal selection with calibration-incident null scores pooled (DESIGN §6.3).

    p_e = (1 + #{calibration nulls with s >= s_e}) / (n_0 + 1),   null = Y != 1 (strict),

then BH at alpha per test incident (``pooled=False``) or over the union of all test incidents'
candidates of the split (``pooled=True``). Valid (FDR <= alpha) only under UNIT-level
exchangeability of nulls across incidents, which incident-level dependence breaks -- it is a
baseline, not a certifier. Stats and fusion scorers are simply different score arrays.

Deterministic version (no random tie-breaking U), i.e. the conservative conformal p-value.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..certify.pvalues import ref_pvalues, sort_null_pool
from ..certify.rules import TOL, bh_select


def bh_iid(cal_groups: Sequence[tuple[np.ndarray, np.ndarray]], test_groups: Sequence[np.ndarray],
           alpha: float, pooled: bool = False) -> list[np.ndarray]:
    """Reference implementation on raw arrays. cal_groups = [(scores, y)], test_groups = [scores].
    Returns a boolean selection mask per test incident."""
    nulls = np.concatenate([np.asarray(s, float)[np.asarray(y) != 1] for s, y in cal_groups])
    pool = sort_null_pool(nulls)
    ps = [ref_pvalues(s, pool) for s in test_groups]
    if not pooled:
        return [bh_select(p, alpha) for p in ps]
    allp = np.concatenate(ps)
    mask = bh_select(allp, alpha)
    out, pos = [], 0
    for p in ps:
        out.append(mask[pos: pos + p.size])
        pos += p.size
    return out


class NullIndex:
    """Bank-wide sorted null scores, so a split's conformal p-values are one cumsum + one gather."""

    def __init__(self, bank):
        y = bank.csr["y"]
        s = bank.csr["score"]
        rows = np.repeat(np.arange(bank.n), bank.sizes())
        nm = y != 1
        o = np.argsort(s[nm], kind="stable")
        self.null_sorted = s[nm][o]
        self.null_row = rows[nm][o]
        self.pos = np.searchsorted(self.null_sorted, s, side="left")   # #{bank nulls < s} per candidate
        self.n_rows = bank.n

    def pvalues(self, cal_rows: np.ndarray, cand_idx: np.ndarray) -> np.ndarray:
        """Conformal p-values of candidates ``cand_idx`` w.r.t. the nulls of ``cal_rows`` (multiplicity
        counted, as needed by the incident bootstrap)."""
        w = np.bincount(np.asarray(cal_rows, dtype=np.int64), minlength=self.n_rows)
        cw = np.concatenate(([0], np.cumsum(w[self.null_row], dtype=np.int64)))
        n0 = int(cw[-1])
        below = cw[self.pos[cand_idx]]
        return (1.0 + (n0 - below)) / (n0 + 1.0)


def bh_counts_segments(p: np.ndarray, seg: np.ndarray, rank: np.ndarray, alpha: float) -> np.ndarray:
    """Per-segment BH(alpha) selection size for p sorted ascending within each segment."""
    lens = np.diff(seg)
    Nrep = np.repeat(lens, lens).astype(np.float64)
    ok = p <= alpha * rank / np.maximum(Nrep, 1.0) * (1.0 + TOL)
    val = np.where(ok, rank, 0)
    out = np.zeros(lens.size, dtype=np.int64)
    nz = lens > 0
    if nz.any():
        out[nz] = np.maximum.reduceat(val, seg[:-1][nz])
    return out


def bh_iid_R(bank, cal_rows: np.ndarray, test_rows: np.ndarray, alpha: float, pooled: bool = False,
             null_index: NullIndex | None = None) -> np.ndarray:
    """Selection sizes R [M] for one split (top sets in each incident's score order)."""
    ni = null_index or NullIndex(bank)
    idx, seg, rank = bank.cand_index(test_rows)
    p = ni.pvalues(cal_rows, idx)
    if not pooled:
        return bh_counts_segments(p, seg, rank, alpha)
    n = p.size
    if n == 0:
        return np.zeros(len(test_rows), np.int64)
    ps = np.sort(p)
    ok = ps <= alpha * np.arange(1, n + 1) / n * (1.0 + TOL)
    k = np.flatnonzero(ok)
    if k.size == 0:
        return np.zeros(len(test_rows), np.int64)
    pstar = ps[k[-1]]
    sel = (p <= pstar).astype(np.int64)
    cs = np.concatenate(([0], np.cumsum(sel)))
    return (cs[seg[1:]] - cs[seg[:-1]]).astype(np.int64)
