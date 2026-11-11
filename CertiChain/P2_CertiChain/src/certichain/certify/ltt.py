"""Learn-then-Test / PAC certification (DESIGN §3.5 "LTT / PAC"; [LTT25]).

For each lambda the null H_lambda: E[FDP(lambda)] > alpha is tested with the
Hoeffding-Bentkus p-value computed from the K *un-monotonised* calibration FDPs:

    p_HB(lambda) = min( exp(-K h1(min(R^, alpha), alpha)),  e * P(Bin(K, alpha) <= ceil(K R^)) )
    h1(a, b)     = a log(a/b) + (1-a) log((1-a)/(1-b)),

R^ = mean_k FDP_k(lambda). Fixed-sequence testing from small to large lambda (grid order,
starting at index 1; index 0 is the empty set, trivially safe): lambda-hat_LTT is the last
lambda rejected before the first failure (index 0 if index 1 already fails). Guarantee:
P( E[FDP(lambda-hat) | calibration] <= alpha ) >= 1 - delta, delta = 0.1 by default.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom


def _h1(a: np.ndarray, b: float) -> np.ndarray:
    a = np.clip(a, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(a > 0, a * np.log(a / b), 0.0)
        t2 = np.where(a < 1, (1 - a) * np.log((1 - a) / (1 - b)), 0.0)
    return t1 + t2


def hb_pvalue(risk_hat, n, alpha: float) -> np.ndarray:
    """Hoeffding-Bentkus p-value for H: risk > alpha from the mean of n losses in [0, 1]."""
    r = np.asarray(risk_hat, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    p_h = np.exp(-n * _h1(np.minimum(r, alpha), alpha))
    k = np.ceil(n * r - 1e-9)
    p_b = np.e * binom.cdf(k, n, alpha)
    return np.minimum(1.0, np.minimum(p_h, p_b))


def ltt_from_pvalues(p: np.ndarray, delta: float) -> np.ndarray:
    """Fixed-sequence rule on p-values [S, G] (column 0 = lambda_0 ignored). Returns indices [S]."""
    p = np.atleast_2d(p)
    rej = p[:, 1:] <= delta
    fail = ~rej
    first_fail = np.where(fail.any(axis=1), np.argmax(fail, axis=1), rej.shape[1])  # # of leading rejections
    return first_fail.astype(np.int64)  # leading rejections r -> last rejected column index = r


def ltt_threshold(FDP: np.ndarray, alpha: float, delta: float = 0.1) -> int:
    """Grid index of lambda-hat_LTT for the un-monotonised calibration FDP table [K, G]."""
    F = np.asarray(FDP, dtype=np.float64)
    K = F.shape[0]
    p = hb_pvalue(F.mean(axis=0), K, alpha)
    return int(ltt_from_pvalues(p[None, :], delta)[0])


def ltt_threshold_from_sums(sums: np.ndarray, K, alpha: float, delta: float = 0.1) -> np.ndarray:
    """Vectorised LTT across splits: ``sums`` [S, G] of calibration FDPs, ``K`` [S] or int."""
    sums = np.atleast_2d(np.asarray(sums, dtype=np.float64))
    K = np.broadcast_to(np.asarray(K, dtype=np.float64), (sums.shape[0],))
    p = hb_pvalue(sums / K[:, None], K[:, None], alpha)
    return ltt_from_pvalues(p, delta)
