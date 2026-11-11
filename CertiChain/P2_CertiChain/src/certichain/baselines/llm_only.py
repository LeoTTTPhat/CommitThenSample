"""B5 — LLM-only @ s_LLM > 0 (zero-shot logit difference) on the kappa = 1 arm: every candidate of 15 CF-TON
cal/test-pool incidents is scored by the frozen LLM prompt (DESIGN §6.3, §3.4.4). The v1 "5 samples at
T = 0.7" design was degenerate (single-token answer) and is not used. No guarantee (uncalibrated)."""
from __future__ import annotations

import numpy as np


def llm_scores(cand_ids, lookup_inc: dict) -> np.ndarray:
    """s_LLM aligned with ``cand_ids`` (NaN where the candidate was not scored)."""
    return np.array([lookup_inc.get(int(c), np.nan) for c in cand_ids], dtype=np.float64)


def llm_only_claims(s_llm, threshold: float = 0.0) -> np.ndarray:
    s = np.asarray(s_llm, dtype=np.float64)
    return np.where(np.isnan(s), False, s > threshold)


def coverage(s_llm) -> float:
    s = np.asarray(s_llm, dtype=np.float64)
    return float(np.mean(~np.isnan(s))) if s.size else 1.0
