"""Training-only label priors (DESIGN §3.4.1): the ONLY label-derived features.

* Transition prior: smoothed log-odds of Y = 1 vs Y = 0 for each (type_u, type_v) pair, fitted on
  F_link labels of a TRAINING pool with AMB (y = -1) excluded:

      prior(a, b) = log((n1 + m*pi) / (n0 + m*(1 - pi)))

  pi = global positive rate (Laplace: (N1 + 1) / (N1 + N0 + 2)), m = PRIOR_M pseudo-candidates.
  This is Laplace smoothing centred on the global rate instead of 1/2, so an unseen pair gets the
  global log-odds (CF-TON-H: held-out transitions get no help from the prior, by construction).
* Lag log-likelihood ratio per type pair: 12 lag bins (bin 0 = overlap, lag_gap <= 0; bins 1..11 =
  equal-width in log(1 + lag_gap) up to log(1 + W), last bin open), class histograms smoothed by
  Laplace and shrunk to the global class histograms with LAG_M pseudo-candidates:

      p_c(bin | a, b) = (h_c(a, b, bin) + LAG_M * g_c(bin)) / (n_c(a, b) + LAG_M),  c in {1, 0}
      llr(a, b, bin)  = log p_1 - log p_0

  This replaces Hawkes EM (DESIGN §3.4.1). B4 (lag-likelihood baseline) = this llr.

Both priors are fixed functions of the training labels they were fitted on, so they are re-fitted per
training pool (and per cross-fitting fold, see scoring/fusion.py) and never see calibration/test labels.
Labels come in as plain arrays (this module never reads label files).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

N_LAG_BINS = 12
PRIOR_M = 2.0          # pseudo-candidates of the transition-prior shrinkage
LAG_M = 10.0           # pseudo-candidates of the per-pair lag-histogram shrinkage
W_DEFAULT = 1800.0
UNK = "<UNK>"


def lag_bins(lag_gap: np.ndarray, W: float = W_DEFAULT) -> np.ndarray:
    """Bin index 0..11: 0 = overlap (lag_gap <= 0); 1..11 equal width in log1p(lag_gap) on (0, W]."""
    g = np.asarray(lag_gap, dtype=np.float64)
    out = np.zeros(g.shape, dtype=np.int64)
    pos = g > 0
    x = np.log1p(np.where(pos, g, 0.0)) / np.log1p(W)          # (0, 1] for 0 < lag_gap <= W
    b = 1 + np.floor(x * (N_LAG_BINS - 1)).astype(np.int64)
    out[pos] = np.clip(b[pos], 1, N_LAG_BINS - 1)
    return out


def _pair_keys(type_u, type_v) -> np.ndarray:
    tu = np.asarray(type_u, dtype=object)
    tv = np.asarray(type_v, dtype=object)
    return np.array([f"{a}|{b}" for a, b in zip(tu, tv)], dtype=object)


@dataclass
class PairPriors:
    """Transition prior + lag LLR fitted on one training pool (JSON-serialisable)."""
    pi: float
    global_logodds: float
    pair_logodds: dict[str, float]
    lag_llr_global: list[float]
    lag_llr_pair: dict[str, list[float]]
    counts: dict[str, list[int]] = field(default_factory=dict)      # pair -> [n1, n0]
    W: float = W_DEFAULT
    n_pos: int = 0
    n_neg: int = 0
    n_amb_excluded: int = 0
    prior_m: float = PRIOR_M
    lag_m: float = LAG_M

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, type_u, type_v, lag_gap, y, W: float = W_DEFAULT, prior_m: float = PRIOR_M,
            lag_m: float = LAG_M) -> "PairPriors":
        y = np.asarray(y)
        keep = y != -1                                   # AMB excluded (DESIGN §3.4.1)
        keys = _pair_keys(type_u, type_v)[keep]
        yy = (y[keep] == 1).astype(np.int64)
        bins = lag_bins(np.asarray(lag_gap)[keep], W)
        N1, N0 = int(yy.sum()), int((1 - yy).sum())
        pi = (N1 + 1.0) / (N1 + N0 + 2.0)
        glo = float(np.log(pi / (1.0 - pi)))
        # global lag histograms (Laplace)
        g1 = np.bincount(bins[yy == 1], minlength=N_LAG_BINS).astype(np.float64)
        g0 = np.bincount(bins[yy == 0], minlength=N_LAG_BINS).astype(np.float64)
        g1 = (g1 + 1.0) / (g1.sum() + N_LAG_BINS)
        g0 = (g0 + 1.0) / (g0.sum() + N_LAG_BINS)
        llr_g = np.log(g1) - np.log(g0)
        pair_lo: dict[str, float] = {}
        pair_llr: dict[str, list[float]] = {}
        counts: dict[str, list[int]] = {}
        if keys.size:
            uk, inv = np.unique(keys, return_inverse=True)
            n1 = np.bincount(inv, weights=yy, minlength=uk.size)
            n = np.bincount(inv, minlength=uk.size).astype(np.float64)
            n0 = n - n1
            for j, k in enumerate(uk):
                lo = np.log((n1[j] + prior_m * pi) / (n0[j] + prior_m * (1.0 - pi)))
                pair_lo[str(k)] = float(lo)
                counts[str(k)] = [int(n1[j]), int(n0[j])]
                m = inv == j
                h1 = np.bincount(bins[m & (yy == 1)], minlength=N_LAG_BINS).astype(np.float64)
                h0 = np.bincount(bins[m & (yy == 0)], minlength=N_LAG_BINS).astype(np.float64)
                p1 = (h1 + lag_m * g1) / (h1.sum() + lag_m)
                p0 = (h0 + lag_m * g0) / (h0.sum() + lag_m)
                pair_llr[str(k)] = (np.log(p1) - np.log(p0)).tolist()
        return cls(pi=float(pi), global_logodds=glo, pair_logodds=pair_lo, lag_llr_global=llr_g.tolist(),
                   lag_llr_pair=pair_llr, counts=counts, W=float(W), n_pos=N1, n_neg=N0,
                   n_amb_excluded=int((~keep).sum()), prior_m=float(prior_m), lag_m=float(lag_m))

    # ------------------------------------------------------------------ apply
    def transition_prior(self, type_u, type_v) -> np.ndarray:
        keys = _pair_keys(type_u, type_v)
        d = self.pair_logodds
        g = self.global_logodds
        return np.fromiter((d.get(k, g) for k in keys), dtype=np.float64, count=keys.size)

    def lag_llr(self, type_u, type_v, lag_gap) -> np.ndarray:
        keys = _pair_keys(type_u, type_v)
        bins = lag_bins(lag_gap, self.W)
        glob = np.asarray(self.lag_llr_global)
        out = glob[bins].copy()
        if keys.size:
            uk, inv = np.unique(keys, return_inverse=True)
            for j, k in enumerate(uk):
                tab = self.lag_llr_pair.get(str(k))
                if tab is not None:
                    m = inv == j
                    out[m] = np.asarray(tab)[bins[m]]
        return out

    def seen_pair(self, type_u, type_v, min_pos: int = 1) -> np.ndarray:
        """True iff the (type_u, type_v) pair had >= min_pos training positives (E3 unseen-pair stratum)."""
        keys = _pair_keys(type_u, type_v)
        c = self.counts
        return np.fromiter((c.get(k, [0, 0])[0] >= min_pos for k in keys), dtype=bool, count=keys.size)

    # ------------------------------------------------------------------ io
    def to_json(self) -> dict:
        return {"pi": self.pi, "global_logodds": self.global_logodds, "pair_logodds": self.pair_logodds,
                "lag_llr_global": self.lag_llr_global, "lag_llr_pair": self.lag_llr_pair, "counts": self.counts,
                "W": self.W, "n_pos": self.n_pos, "n_neg": self.n_neg, "n_amb_excluded": self.n_amb_excluded,
                "prior_m": self.prior_m, "lag_m": self.lag_m, "n_lag_bins": N_LAG_BINS}

    @classmethod
    def from_json(cls, d: dict) -> "PairPriors":
        d = dict(d)
        d.pop("n_lag_bins", None)
        return cls(**d)

    def save(self, path: str | Path) -> Path:
        from certichain.io.paths import atomic_write_json
        p = Path(path)
        atomic_write_json(p, self.to_json())
        return p

    @classmethod
    def load(cls, path: str | Path) -> "PairPriors":
        return cls.from_json(json.loads(Path(path).read_text()))


def oof_prior_features(type_u, type_v, lag_gap, y, groups, fold_of_group: dict, W: float = W_DEFAULT
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold (transition_prior, lag_llr) for TRAINING rows: rows of fold f get priors fitted on
    the other folds, so a training row never sees its own label through a prior feature."""
    groups = np.asarray(groups, dtype=object)
    folds = np.array([fold_of_group[g] for g in groups])
    tp = np.empty(groups.size)
    ll = np.empty(groups.size)
    tu = np.asarray(type_u, dtype=object)
    tv = np.asarray(type_v, dtype=object)
    lg = np.asarray(lag_gap, dtype=np.float64)
    yy = np.asarray(y)
    for f in np.unique(folds):
        m = folds == f
        pr = PairPriors.fit(tu[~m], tv[~m], lg[~m], yy[~m], W)
        tp[m] = pr.transition_prior(tu[m], tv[m])
        ll[m] = pr.lag_llr(tu[m], tv[m], lg[m])
    return tp, ll
