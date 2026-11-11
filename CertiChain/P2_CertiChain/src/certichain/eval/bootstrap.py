"""Incident-pool cluster bootstrap and the pre-registered validity test (DESIGN §6.4).

* cluster_bootstrap(pool_ids, fn, B=1000, splits=50): resample the pool's incidents WITH
  replacement (the incident is the cluster), re-run ``splits`` random cal/test splits per
  replicate on the cached loss tables, and record fn's statistic. SE_boot = SD of replicates.
  Pooling over overlapping splits is NOT used for SEs.
* validity test (primary endpoint 1): H0: FDR <= alpha, z = (mean - alpha) / SE_boot, one-sided
  p = 1 - Phi(z); Holm across cells at 0.05; a rejection is reported as a violation.
* paired bootstrap (primary endpoint 2): the same resampled incidents and splits for two banks
  (e.g. fusion vs stats scorer on identical incidents), statistic = difference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.stats import norm

from ..certify.losses import LossBank
from .runner import evaluate
from .splits import random_splits


@dataclass
class CI:
    estimate: float
    se: float
    lo: float
    hi: float
    level: float
    B: int
    replicates: np.ndarray = field(repr=False)

    def as_dict(self) -> dict:
        return {"estimate": self.estimate, "se": self.se, "lo": self.lo, "hi": self.hi,
                "level": self.level, "B": self.B}


def cluster_bootstrap(pool_ids, fn: Callable[[np.ndarray, np.ndarray], float], B: int = 1000, splits: int = 50,
                      seed: int = 0, level: float = 0.95, estimate: float | None = None) -> CI:
    """``fn(rows, split_seeds)`` -> statistic for a (resampled) pool ``rows`` (bank rows, duplicates
    allowed) evaluated on random splits with the given seeds. Percentile CI; SE = replicate SD."""
    pool = np.asarray(pool_ids, dtype=np.int64)
    rng = np.random.default_rng((int(seed), 0xB00757))
    seeds = np.arange(splits)
    reps = np.empty(B)
    for b in range(B):
        rows = pool[rng.integers(0, pool.size, pool.size)]
        reps[b] = fn(rows, seeds)
    est = float(fn(pool, seeds)) if estimate is None else float(estimate)
    f = reps[np.isfinite(reps)]
    a = (1 - level) / 2
    return CI(est, float(f.std(ddof=1)) if f.size > 1 else float("nan"),
              float(np.quantile(f, a)) if f.size else float("nan"),
              float(np.quantile(f, 1 - a)) if f.size else float("nan"), level, B, reps)


def method_stat_fn(bank: LossBank, method: str, K: int, alpha: float, metric: str = "fdr", aux=None,
                   n_test: int | None = None, **kw) -> Callable[[np.ndarray, np.ndarray], float]:
    """fn for cluster_bootstrap: mean over splits of ``metric`` for ``method`` on the resampled pool."""
    cache: dict = {}

    def fn(rows: np.ndarray, seeds: np.ndarray) -> float:
        sp = random_splits(rows.size, K, seeds, n_test)
        res = evaluate(bank, method, sp, alpha, pool_rows=rows, aux=aux, cache=cache, cert_gap=False, **kw)
        return float(np.nanmean(res[metric]))

    return fn


def paired_stat_fn(bank_a: LossBank, bank_b: LossBank, method: str, K: int, alpha: float,
                   metric: str = "cer_link_mean", **kw) -> Callable[[np.ndarray, np.ndarray], float]:
    """metric(bank_b) - metric(bank_a) on identical resampled incidents and splits (rows must refer to
    the same incidents in both banks)."""
    if bank_a.n != bank_b.n or bank_a.index["incident_id"].to_list() != bank_b.index["incident_id"].to_list():
        raise ValueError("paired banks must hold the same incidents in the same row order")
    fa = method_stat_fn(bank_a, method, K, alpha, metric, **kw)
    fb = method_stat_fn(bank_b, method, K, alpha, metric, **kw)
    return lambda rows, seeds: fb(rows, seeds) - fa(rows, seeds)


def validity_z(mean: float, se: float, alpha: float) -> tuple[float, float]:
    """z = (mean - alpha)/SE_boot and the one-sided p-value of H0: FDR <= alpha."""
    if not np.isfinite(se) or se <= 0:
        z = np.inf if mean > alpha else -np.inf
    else:
        z = (mean - alpha) / se
    return float(z), float(norm.sf(z))


def holm(pvals, level: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down: (adjusted p-values, reject flags) in the input order."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    o = np.argsort(p, kind="stable")
    adj_sorted = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * p[o]))
    adj = np.empty(m)
    adj[o] = adj_sorted
    return adj, adj <= level


def validity_table(cells: dict[str, tuple[float, float]], alpha: float, level: float = 0.05) -> list[dict]:
    """cells: name -> (mean FDR, SE_boot). Returns per-cell z, raw and Holm-adjusted p, violation flag,
    and the maximum standardised excess across cells."""
    names = list(cells)
    zs, ps = zip(*[validity_z(cells[n][0], cells[n][1], alpha) for n in names]) if names else ((), ())
    adj, rej = holm(ps, level)
    zmax = max(zs) if zs else float("nan")
    return [{"cell": n, "mean": cells[n][0], "se_boot": cells[n][1], "z": z, "p": p, "p_holm": float(a),
             "violation": bool(r), "max_standardised_excess": zmax}
            for n, z, p, a, r in zip(names, zs, ps, adj, rej)]
