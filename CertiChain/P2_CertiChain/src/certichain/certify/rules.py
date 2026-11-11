"""Nested rule families indexed by a scalar (DESIGN §3.5, Alg. 4 step 1).

* BH family: S(lambda) = Benjamini-Hochberg at nominal level lambda over the incident's N
  candidates' reference p-values, lambda on LAMBDA_GRID = {0} U 60 geometric levels in
  [1e-4, 0.8]. One sort per incident gives the whole path: with p_(1) <= ... <= p_(N),

      R(lambda) = max{i : p_(i) <= lambda * i / N}
                = #{i : lam_star_i <= lambda},   lam_star_i = min_{j >= i} N p_(j) / j,

  and S(lambda) = the first R(lambda) candidates in p order. lam_star is non-decreasing, so
  the sets are nested in lambda; tied p-values enter together; S(0) = {} because p > 0.
* Score-threshold family (SHOULD ablation): S(t_j) = {e : s_e >= t_j} with t_0 = +inf and
  t_j = the (1 - lambda_j)-quantile of S_0 (60 quantiles, same lambda levels), so S(t_j) is
  "reference p ~<= lambda_j without the BH step-up". Nested in j because t_j is non-increasing.

Both families select TOP-SCORE sets: for every incident and every grid index the selection
is the first R candidates in score-descending order (ties grouped), which is what lets the
loss tables store only counts (certify/losses.py).

Comparisons use a relative tolerance of 1e-12 so that an exact BH tie p_(i) = lambda i/N
(possible with rational reference p-values) resolves as "<=" despite float rounding.
"""
from __future__ import annotations

import numpy as np

TOL = 1e-12


def lambda_grid(n: int = 60, lo: float = 1e-4, hi: float = 0.8) -> np.ndarray:
    """{0} U n geometric levels in [lo, hi] (DESIGN §3.10 default)."""
    return np.concatenate(([0.0], np.geomspace(lo, hi, n)))


LAMBDA_GRID = lambda_grid()


def score_order(scores) -> np.ndarray:
    """Indices sorting candidates by score descending, ties by original index (deterministic)."""
    s = np.asarray(scores, dtype=np.float64)
    if np.isnan(s).any():
        raise ValueError("NaN score")
    return np.lexsort((np.arange(s.size), -s))


def bh_lam_star(p_sorted: np.ndarray) -> np.ndarray:
    """lam_star_i (i = 1..N) for p sorted ascending: the smallest level at which BH selects >= i."""
    p = np.asarray(p_sorted, dtype=np.float64)
    n = p.size
    if n == 0:
        return np.empty(0)
    c = n * p / np.arange(1, n + 1)
    return np.minimum.accumulate(c[::-1])[::-1]


def bh_counts(lam_star: np.ndarray, levels) -> np.ndarray:
    """R(lambda) for each level (int64). ``lam_star`` from bh_lam_star."""
    lv = np.asarray(levels, dtype=np.float64)
    return np.searchsorted(lam_star, lv * (1.0 + TOL), side="right").astype(np.int64)


def bh_path(p) -> tuple[np.ndarray, np.ndarray]:
    """(order, lam_star): order sorts p ascending (stable); top-i selected iff lambda >= lam_star[i-1]."""
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    return order, bh_lam_star(p[order])


def bh_nested(p, lam_grid=LAMBDA_GRID) -> list[np.ndarray]:
    """Selected index sets S(lambda) for every lambda in the grid (DESIGN §7.2 signature)."""
    order, lam_star = bh_path(p)
    R = bh_counts(lam_star, lam_grid)
    return [order[:r] for r in R]


def bh_select(p, alpha: float) -> np.ndarray:
    """Boolean mask of the BH(alpha) rejection set."""
    p = np.asarray(p, dtype=np.float64)
    order, lam_star = bh_path(p)
    r = int(bh_counts(lam_star, [alpha])[0])
    mask = np.zeros(p.size, dtype=bool)
    mask[order[:r]] = True
    return mask


def bh_count_sorted(p_sorted: np.ndarray, alpha: float) -> int:
    """|BH(alpha)| for p already sorted ascending (= max{i : p_(i) <= alpha i / N})."""
    n = p_sorted.size
    if n == 0:
        return 0
    ok = p_sorted <= alpha * np.arange(1, n + 1) / n * (1.0 + TOL)
    idx = np.flatnonzero(ok)
    return int(idx[-1] + 1) if idx.size else 0


def st_thresholds(s0_sorted: np.ndarray, lam_grid=LAMBDA_GRID) -> np.ndarray:
    """Score-threshold grid: t_0 = +inf, t_j = (1 - lambda_j)-quantile of S_0 ('higher' method)."""
    lam = np.asarray(lam_grid, dtype=np.float64)
    t = np.empty(lam.size)
    pos = lam > 0
    t[~pos] = np.inf
    t[pos] = np.quantile(np.asarray(s0_sorted, dtype=np.float64), 1.0 - lam[pos], method="higher")
    return t


def st_counts(scores_desc: np.ndarray, thresholds) -> np.ndarray:
    """R(t) = #{s >= t} for scores sorted descending."""
    th = np.asarray(thresholds, dtype=np.float64)
    return np.searchsorted(-np.asarray(scores_desc, dtype=np.float64), -th, side="right").astype(np.int64)
