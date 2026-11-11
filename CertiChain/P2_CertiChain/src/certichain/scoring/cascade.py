"""Cascade band (DESIGN §3.4.4): which candidates get an LLM call.

From OOF s_stat on the TRAINING incidents:
  q_lo = largest threshold with <= 0.5 % of training true F_link edges (y = 1) below it;
  q_hi = smallest threshold whose training FDP (strict: AMB counted false) of {s >= q_hi} is <= 1 %;
  band = [q_lo, q_hi);  kappa = fraction of training candidates in the band;
  if kappa > kappa_target: shrink symmetrically in quantile until kappa <= kappa_target (<= kappa_max).
Quantile spaces (``rule``):
  "cand" (DESIGN literal): quantiles of the training candidate score distribution (kappa is a mass of
         that distribution, so equal mass is removed from both ends);
  "pos":  quantiles of the training TRUE-link score distribution (equal true-link mass removed from both
         ends, bisection on the shrink until kappa <= target). Diagnostic alternative; see notes.
The band is a fixed function of M_stat and training labels, so the fused scorer stays fixed (validity).
``Band.contains(s)`` is the llm_called indicator for any incident scored by that M_stat.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

KAPPA_TARGET = 0.10
KAPPA_MAX = 0.15
POS_BELOW = 0.005
FDP_ABOVE = 0.01


@dataclass
class Band:
    q_lo: float
    q_hi: float
    kappa_train: float
    kappa_target: float
    rule: str
    q_lo0: float
    q_hi0: float
    kappa0: float
    pos_in_band: float                 # fraction of training true links inside the band
    pos_above: float                   # ... above q_hi
    pos_below: float                   # ... below q_lo
    fdp_above_train: float             # strict FDP of {s >= q_hi} on training
    n_train: int
    n_pos: int
    extra: dict = field(default_factory=dict)

    def contains(self, s) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        return (s >= self.q_lo) & (s < self.q_hi)

    def to_json(self) -> dict:
        d = asdict(self)
        for k in ("q_lo", "q_hi", "q_lo0", "q_hi0"):
            if not np.isfinite(d[k]):
                d[k] = "inf" if d[k] > 0 else "-inf"
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Band":
        d = dict(d)
        for k in ("q_lo", "q_hi", "q_lo0", "q_hi0"):
            if isinstance(d[k], str):
                d[k] = float(d[k])
        return cls(**d)

    def save(self, path) -> Path:
        from certichain.io.paths import atomic_write_json
        return atomic_write_json(path, self.to_json())

    @classmethod
    def load(cls, path) -> "Band":
        return cls.from_json(json.loads(Path(path).read_text()))


def initial_thresholds(s, y, pos_below: float = POS_BELOW, fdp_above: float = FDP_ABOVE) -> tuple[float, float]:
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y)
    pos = np.sort(s[y == 1])
    if pos.size == 0:
        return float("-inf"), float("inf")
    k = int(np.floor(pos_below * pos.size))
    q_lo = float(pos[k])                    # #{pos < q_lo} <= k  (largest such threshold)
    order = np.argsort(-s, kind="stable")
    ss = s[order]
    false = np.cumsum(y[order] != 1)
    n = np.arange(1, s.size + 1)
    # evaluate FDP only at the last index of each tied score block (sets {s >= t})
    last = np.r_[ss[1:] != ss[:-1], True]
    fdp = false[last] / n[last]
    t = ss[last]
    ok = np.flatnonzero(fdp <= fdp_above + 1e-12)
    q_hi = float(t[ok].min()) if ok.size else float("inf")
    return q_lo, q_hi


def _kappa(s, lo, hi) -> float:
    s = np.asarray(s, dtype=np.float64)
    return float(np.mean((s >= lo) & (s < hi))) if s.size else 0.0


def _shrink_cand(s, lo, hi, target):
    s = np.asarray(s, dtype=np.float64)
    a = float(np.mean(s < lo))
    b = float(np.mean(s < hi)) if np.isfinite(hi) else 1.0
    k0 = b - a
    lo_, hi_ = lo, hi
    if k0 <= target:
        return lo_, hi_
    d = (k0 - target) / 2.0
    for _ in range(60):
        lo_ = float(np.quantile(s, min(1.0, a + d), method="inverted_cdf"))
        hi_ = float(np.quantile(s, max(0.0, b - d), method="inverted_cdf")) if b - d > 0 else lo_
        if hi_ < lo_:
            hi_ = lo_
        if _kappa(s, lo_, hi_) <= target + 1e-12:
            return lo_, hi_
        d += max(1.0 / s.size, 1e-6)
    return lo_, hi_


def _shrink_pos(s, y, lo, hi, target):
    s = np.asarray(s, dtype=np.float64)
    pos = np.sort(s[np.asarray(y) == 1])
    if _kappa(s, lo, hi) <= target or pos.size == 0:
        return lo, hi
    a = float(np.mean(pos < lo))
    b = float(np.mean(pos < hi)) if np.isfinite(hi) else 1.0

    def band(d):                       # linear interpolation between true-link order statistics
        l_ = float(np.quantile(pos, min(1.0, a + d), method="linear"))
        h_ = float(np.quantile(pos, max(0.0, b - d), method="linear")) if b - d > 0 else l_
        if not np.isfinite(hi) and b - d >= 1.0:
            h_ = hi
        return l_, max(h_, l_)
    lo_d, hi_d = 0.0, (b - a) / 2.0
    for _ in range(60):
        mid = (lo_d + hi_d) / 2
        if _kappa(s, *band(mid)) <= target:
            hi_d = mid
        else:
            lo_d = mid
    return band(hi_d)


def fit_band(s_oof, y, kappa_target: float = KAPPA_TARGET, kappa_max: float = KAPPA_MAX, rule: str = "cand",
             pos_below: float = POS_BELOW, fdp_above: float = FDP_ABOVE) -> Band:
    if kappa_target > kappa_max + 1e-12:
        raise ValueError("kappa_target > kappa_max")
    s = np.asarray(s_oof, dtype=np.float64)
    y = np.asarray(y)
    lo0, hi0 = initial_thresholds(s, y, pos_below, fdp_above)
    k0 = _kappa(s, lo0, hi0)
    if rule == "cand":
        lo, hi = _shrink_cand(s, lo0, hi0, kappa_target)
    elif rule == "pos":
        lo, hi = _shrink_pos(s, y, lo0, hi0, kappa_target)
    else:
        raise ValueError(rule)
    pos = s[y == 1]
    above = s >= hi
    fdp = float(np.mean(y[above] != 1)) if above.any() else 0.0
    return Band(q_lo=lo, q_hi=hi, kappa_train=_kappa(s, lo, hi), kappa_target=kappa_target, rule=rule,
                q_lo0=lo0, q_hi0=hi0, kappa0=k0,
                pos_in_band=float(np.mean((pos >= lo) & (pos < hi))) if pos.size else 0.0,
                pos_above=float(np.mean(pos >= hi)) if pos.size else 0.0,
                pos_below=float(np.mean(pos < lo)) if pos.size else 0.0,
                fdp_above_train=fdp, n_train=int(s.size), n_pos=int(pos.size),
                extra={"kappa_max": kappa_max, "pos_below_rule": pos_below, "fdp_above_rule": fdp_above})


def band_grid(s_oof, y, kappas=(0.05, 0.10, 0.20), rule: str = "cand") -> dict[str, Band]:
    """Nested bands for the kappa sensitivity (E3 / §3.4.4: STR vs kappa)."""
    return {f"{k:.2f}": fit_band(s_oof, y, kappa_target=k, kappa_max=max(k, KAPPA_MAX), rule=rule) for k in kappas}
