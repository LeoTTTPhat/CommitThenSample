"""G1 gate helpers (DESIGN §4.9 implications, G1 on the dev pool): within-incident AUC, N/r and
STR at alpha with internal random splits of the dev pool (no test data involved)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import rankdata

from ..certify.losses import LossBank


def auc(pos_scores, neg_scores) -> float:
    """Mann-Whitney AUC with ties counted 1/2 (NaN if a class is empty)."""
    p = np.asarray(pos_scores, float)
    n = np.asarray(neg_scores, float)
    if p.size == 0 or n.size == 0:
        return float("nan")
    r = rankdata(np.concatenate([p, n]))
    return float((r[: p.size].sum() - p.size * (p.size + 1) / 2) / (p.size * n.size))


def within_incident_auc(scores_list: Sequence[np.ndarray], y_list: Sequence[np.ndarray],
                        null: str = "zero") -> dict[str, float]:
    """AUC of true links (y = 1) vs nulls within each incident; null = 'zero' (y = 0) or 'strict'
    (y != 1). Mean / median over incidents with both classes."""
    vals = []
    for s, y in zip(scores_list, y_list):
        s = np.asarray(s, float)
        y = np.asarray(y)
        neg = (y == 0) if null == "zero" else (y != 1)
        a = auc(s[y == 1], s[neg])
        if np.isfinite(a):
            vals.append(a)
    v = np.asarray(vals)
    return {"mean": float(v.mean()) if v.size else float("nan"),
            "median": float(np.median(v)) if v.size else float("nan"), "n_incidents": int(v.size)}


def bank_within_auc(bank: LossBank, rows=None, null: str = "zero") -> dict[str, float]:
    rows = range(bank.n) if rows is None else rows
    return within_incident_auc([bank.scores(int(r)) for r in rows], [bank.labels(int(r)) for r in rows], null)


def pooled_auc(scores_list, y_list) -> float:
    """AUC over the pooled candidates (mixes incident-level shifts; for contrast with the within AUC)."""
    s = np.concatenate([np.asarray(x, float) for x in scores_list])
    y = np.concatenate([np.asarray(x) for x in y_list])
    return auc(s[y == 1], s[y == 0])


def n_over_r(N, r) -> dict[str, float]:
    """N/r over attack incidents (r > 0): median of per-incident ratios and pooled sum N / sum r."""
    N = np.asarray(N, float)
    r = np.asarray(r, float)
    a = r > 0
    return {"median": float(np.median(N[a] / r[a])) if a.any() else float("nan"),
            "pooled": float(N[a].sum() / r[a].sum()) if a.any() else float("nan"),
            "n_attack": int(a.sum()), "n_benign": int((~a).sum())}


def str_at_alpha(bank: LossBank, alpha: float, K: int | None = None, n_splits: int = 50, seed: int = 0,
                 method: str = "crc_bh") -> dict[str, float]:
    """STR (and FDR, abstention) of CRC-BH at alpha using internal random splits of a DEV pool."""
    from .runner import evaluate
    from .splits import random_splits
    K = bank.n // 2 if K is None else K
    sp = random_splits(bank.n, K, range(seed, seed + n_splits))
    res = evaluate(bank, method, sp, alpha, cert_gap=False)
    keys = ("str_mean", "str_pooled", "ostr_mean", "cer_link_mean", "recall_pooled", "fdr", "abstain")
    return {k: float(np.nanmean(res[k])) for k in keys if k in res} | {"K": K, "n_splits": n_splits}


def g1_report(bank: LossBank, alphas=(0.1, 0.2), K: int | None = None, n_splits: int = 50) -> dict:
    out = {"within_auc": bank_within_auc(bank), "n_over_r": n_over_r(bank.col("N"), bank.col("n_true"))}
    for a in alphas:
        out[f"alpha={a}"] = str_at_alpha(bank, a, K, n_splits)
    return out
