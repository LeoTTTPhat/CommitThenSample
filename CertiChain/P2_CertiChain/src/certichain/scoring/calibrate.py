"""Probability-calibration baselines B7 and B8 (DESIGN §6.3). Neither has a finite-sample guarantee.

* B7 "fusion + naive 0.5": isotonic map (non-decreasing, clipped to [0, 1]) fitted on OUT-OF-FOLD
  TRAINING scores against 1{Y = 1}; claim a candidate iff its calibrated probability >= 0.5.
* B8 "fusion + Platt plug-in": Platt scaling P^ = sigmoid(a s + b) fitted on the CALIBRATION
  incidents' candidates (Platt's smoothed targets (N+ + 1)/(N+ + 2) and 1/(N- + 2), Newton with
  backtracking); per test incident select {P^ >= tau} with tau the smallest value such that the
  plug-in FDR estimate mean(1 - P^) over the selection is <= alpha.

Both selections are top-score sets (the maps are non-decreasing in s; for B8 a <= 0 is degenerate
and selects nothing), so they are evaluated on a LossBank through selection sizes R.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

TOL = 1e-12


# --------------------------------------------------------------------------- B7
class IsotonicB7:
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)
        self.iso: IsotonicRegression | None = None

    def fit(self, train_scores, train_y) -> "IsotonicB7":
        s = np.asarray(train_scores, dtype=np.float64)
        t = (np.asarray(train_y) == 1).astype(np.float64)
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(s, t)
        return self

    def predict(self, scores) -> np.ndarray:
        if self.iso is None:
            raise RuntimeError("fit first")
        return self.iso.predict(np.asarray(scores, dtype=np.float64))

    def claim_mask(self, scores) -> np.ndarray:
        return self.predict(scores) >= self.threshold


def b7_R(bank, model: IsotonicB7, rows=None) -> np.ndarray:
    """B7 selection size per incident (split-independent)."""
    rows = np.arange(bank.n) if rows is None else np.asarray(rows)
    idx, seg, _ = bank.cand_index(rows)
    sel = model.claim_mask(bank.csr["score"][idx]).astype(np.int64)
    cs = np.concatenate(([0], np.cumsum(sel)))
    return (cs[seg[1:]] - cs[seg[:-1]]).astype(np.int64)


# --------------------------------------------------------------------------- B8
def platt_fit(scores, y, max_iter: int = 100, tol: float = 1e-10) -> tuple[float, float]:
    """Platt scaling (a, b) with smoothed targets; Newton-Raphson with backtracking line search."""
    s = np.asarray(scores, dtype=np.float64)
    pos = np.asarray(y) == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if s.size == 0:
        return 0.0, 0.0
    t = np.where(pos, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
    # P = sigmoid(a s + b); start at the intercept-only optimum (Platt's B0 with our sign convention)
    a, b = 0.0, float(-np.log((n_neg + 1.0) / (n_pos + 1.0)))

    def nll(a_, b_):
        f = a_ * s + b_
        return float(np.sum(np.logaddexp(0.0, f) - t * f))

    cur = nll(a, b)
    damp = 1e-12
    for _ in range(max_iter):
        f = a * s + b
        p = 0.5 * (1.0 + np.tanh(0.5 * f))       # stable sigmoid
        d = p - t
        g = np.array([np.dot(s, d), d.sum()])
        if np.abs(g).max() < tol * max(1.0, s.size):
            break
        w = p * (1.0 - p)
        H = np.array([[np.dot(w, s * s), np.dot(w, s)], [np.dot(w, s), w.sum()]])
        step = None
        while damp < 1e6:
            try:
                Hd = H + damp * max(1.0, float(np.trace(H))) * np.eye(2)
                step = np.linalg.solve(Hd, g)
                break
            except np.linalg.LinAlgError:
                damp *= 100.0
        if step is None or not np.all(np.isfinite(step)):
            break
        lr = 1.0
        while lr > 1e-10:
            na, nb = a - lr * step[0], b - lr * step[1]
            new = nll(na, nb)
            if new <= cur + 1e-4 * lr * float(np.dot(g, -step)):
                break
            lr *= 0.5
        if lr <= 1e-10:
            break
        a, b, prev, cur = na, nb, cur, new
        if abs(prev - cur) < tol * max(1.0, abs(prev)):
            break
    return float(a), float(b)


class PlattB8:
    def __init__(self):
        self.a = self.b = None

    def fit(self, cal_scores, cal_y) -> "PlattB8":
        self.a, self.b = platt_fit(cal_scores, cal_y)
        return self

    def predict(self, scores) -> np.ndarray:
        f = self.a * np.asarray(scores, dtype=np.float64) + self.b
        return 0.5 * (1.0 + np.tanh(0.5 * f))

    def select_mask(self, scores, alpha: float) -> np.ndarray:
        """{P^ >= tau}, tau = smallest P^ value with mean(1 - P^) over {P^ >= tau} <= alpha."""
        s = np.asarray(scores, dtype=np.float64)
        mask = np.zeros(s.size, dtype=bool)
        if s.size == 0 or self.a is None or self.a <= 0:
            return mask
        o = np.lexsort((np.arange(s.size), -s))
        R = _plugin_counts(self.predict(s[o]), s[o], np.array([0, s.size]), np.arange(1, s.size + 1), alpha)[0]
        mask[o[:R]] = True
        return mask


def _plugin_counts(P_desc: np.ndarray, s_desc: np.ndarray, seg: np.ndarray, rank: np.ndarray, alpha: float) -> np.ndarray:
    """Per segment: largest top-n (ending at a score-tie boundary) with mean(1 - P^) <= alpha."""
    lens = np.diff(seg)
    out = np.zeros(lens.size, dtype=np.int64)
    if P_desc.size == 0:
        return out
    q = 1.0 - P_desc
    cs = np.cumsum(q)
    start_val = np.repeat(np.concatenate(([0.0], cs))[seg[:-1]], lens)
    run_mean = (cs - start_val) / rank
    nxt_same = np.zeros(P_desc.size, dtype=bool)
    nxt_same[:-1] = s_desc[1:] == s_desc[:-1]
    last_in_seg = np.zeros(P_desc.size, dtype=bool)
    last_in_seg[seg[1:][lens > 0] - 1] = True
    block_end = ~nxt_same | last_in_seg
    ok = (run_mean <= alpha * (1.0 + TOL)) & block_end
    val = np.where(ok, rank, 0)
    nz = lens > 0
    out[nz] = np.maximum.reduceat(val, seg[:-1][nz])
    return out


def b8_R(bank, cal_rows: np.ndarray, test_rows: np.ndarray, alpha: float) -> np.ndarray:
    """B8 selection sizes for one split: Platt fitted on the calibration rows' candidates."""
    ci, _, _ = bank.cand_index(cal_rows)
    model = PlattB8().fit(bank.csr["score"][ci], bank.csr["y"][ci])
    idx, seg, rank = bank.cand_index(test_rows)
    if model.a is None or model.a <= 0:
        return np.zeros(len(test_rows), dtype=np.int64)
    s = bank.csr["score"][idx]
    return _plugin_counts(model.predict(s), s, seg, rank, alpha)
