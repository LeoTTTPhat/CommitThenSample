"""Uncalibrated baselines on the same candidate set (DESIGN §4.9, §6.3).

* bh_ref_plain: plain BH at alpha on the incident's reference p-values, no calibration. The
  reviewer's re-simulation found it exceeds alpha (0.120) under incident-level heavy tails,
  so it is reported as a baseline, never used as a certifier.
* threshold rule: {s > t} for a fixed t (e.g. B5 "LLM-only @ s_LLM > 0").
Chain-naive and naive repeated CRC live in certify/chains.py and certify/elond.py.
"""
from __future__ import annotations

import numpy as np

from .bh_iid import bh_counts_segments


def bh_ref_plain_R(bank, alpha: float, rows=None) -> np.ndarray:
    """|BH(alpha)| on reference p-values per incident (split-independent)."""
    rows = np.arange(bank.n) if rows is None else np.asarray(rows)
    idx, seg, rank = bank.cand_index(rows)
    return bh_counts_segments(bank.csr["p"][idx], seg, rank, alpha)


def threshold_R(bank, t: float, strict: bool = True, rows=None) -> np.ndarray:
    """#{s > t} (strict) or #{s >= t} per incident."""
    rows = np.arange(bank.n) if rows is None else np.asarray(rows)
    idx, seg, _ = bank.cand_index(rows)
    s = bank.csr["score"][idx]
    sel = (s > t) if strict else (s >= t)
    cs = np.concatenate(([0], np.cumsum(sel, dtype=np.int64)))
    return (cs[seg[1:]] - cs[seg[:-1]]).astype(np.int64)
