"""Incident-level, design-based statements (DESIGN §3.9 "Incident-level statements", Prop. 4(iii), Prop. 5(iv)).

* Presence miss of a metadata-defined incident set A: P(no unit of A retained) = prod_{i in A} (1 - pi_i).
* Content-evidence miss: the same product over A's attack-content-bearing units (single tier: a
  content unit is retained with all its packets; two-tier analytics live in design/floor.py).
* Persistent channel (one content unit per epoch): P(TTE > n) = prod_{e<=n} (1 - pi_e) <= (1 - phi0)^n;
  k95 = min{n: 1 - (1-phi)^n >= 0.95} (200 epochs at phi0 = 0.015, 102 at 0.029 -> DESIGN Prop. 4(iii)).
* Lifecycle stage coverage: per stage P(>= 1 unit retained); mean over stages; P(all stages).
* Grinding (Prop. 5(iv)): best-of-M commitments -> P(evidence) = (1 - (1-pi)^k)^M.
All products are computed in log space (log1p) for accuracy.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["log_miss", "presence_miss", "group_miss", "content_evidence_miss", "miss_bound", "tte_survival",
           "tte_quantile", "k95", "k_for_detection", "stage_coverage", "grinding_evidence"]


def log_miss(pi, axis=-1):
    """sum log(1 - pi) (= log P(none retained)); -inf if some pi = 1."""
    pi = np.asarray(pi, dtype=float)
    with np.errstate(divide="ignore"):
        return np.log1p(-np.clip(pi, 0.0, 1.0)).sum(axis=axis)


def presence_miss(pi, axis=-1):
    """prod (1 - pi) over the incident's units."""
    return np.exp(log_miss(pi, axis=axis))


def group_miss(pi, group, mask=None):
    """Per-group miss products. group: int ids (-1 = none). Returns (ids, miss, size)."""
    pi = np.asarray(pi, dtype=float)
    group = np.asarray(group)
    m = group >= 0
    if mask is not None:
        m &= np.asarray(mask, dtype=bool)
    ids, inv = np.unique(group[m], return_inverse=True)
    with np.errstate(divide="ignore"):
        lm = np.bincount(inv, weights=np.log1p(-np.clip(pi[m], 0, 1)), minlength=len(ids))
    size = np.bincount(inv, minlength=len(ids))
    return ids, np.exp(lm), size


def content_evidence_miss(pi, incident, content):
    """Per-incident content-evidence miss over the content-bearing units (single tier)."""
    return group_miss(pi, incident, mask=content)


def miss_bound(k, phi0: float):
    """(1 - phi0)^k <= exp(-k phi0): worst-case miss of k content units at floor phi0 (Prop. 4(iii))."""
    return (1.0 - phi0) ** np.asarray(k, dtype=float)


def tte_survival(pi_seq):
    """P(TTE > n) for n = 1..len: cumulative product of (1 - pi_e) for a persistent channel."""
    with np.errstate(divide="ignore"):
        return np.exp(np.cumsum(np.log1p(-np.clip(np.asarray(pi_seq, dtype=float), 0, 1)), axis=-1))


def tte_quantile(pi_seq, q: float = 0.5):
    """Smallest n (1-based) with P(TTE <= n) >= q; inf if never reached within the sequence."""
    surv = tte_survival(pi_seq)
    hit = (1.0 - surv) >= q - 1e-15
    if not np.any(hit):
        return math.inf
    return int(np.argmax(hit)) + 1


def k_for_detection(phi: float, p: float = 0.95) -> int:
    """min{n : 1 - (1-phi)^n >= p}."""
    if phi <= 0:
        return math.inf
    if phi >= 1:
        return 1
    return int(math.ceil(math.log(1.0 - p) / math.log1p(-phi) - 1e-12))


def k95(phi: float) -> int:
    return k_for_detection(phi, 0.95)


def stage_coverage(pi, stage, incident=None):
    """Per (incident, stage) detection probabilities. Returns dict with per-stage P(>=1 retained),
    mean stage coverage and P(all stages covered) (independent units)."""
    pi = np.asarray(pi, dtype=float)
    stage = np.asarray(stage)
    inc = np.zeros(len(pi), dtype=np.int64) if incident is None else np.asarray(incident)
    m = (stage >= 0) & (inc >= 0)
    key = inc[m].astype(np.int64) * (int(stage.max()) + 1 if len(stage) else 1) + stage[m].astype(np.int64)
    ids, inv = np.unique(key, return_inverse=True)
    with np.errstate(divide="ignore"):
        lm = np.bincount(inv, weights=np.log1p(-np.clip(pi[m], 0, 1)), minlength=len(ids))
    det = 1.0 - np.exp(lm)
    return {"keys": ids, "detect": det, "mean_coverage": float(det.mean()) if len(det) else float("nan"),
            "p_all": float(np.prod(det)) if len(det) else float("nan")}


def grinding_evidence(pi: float, k: int, M: int) -> float:
    """Best-of-M grinding: P(evidence retained) = (1 - (1-pi)^k)^M (0.401 at M=1, 0.0104 at M=5 for pi=.05, k=10)."""
    return (1.0 - (1.0 - pi) ** k) ** M
