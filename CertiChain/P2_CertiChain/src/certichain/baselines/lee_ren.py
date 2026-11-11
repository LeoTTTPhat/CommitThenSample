"""Lee & Ren (arXiv:2501.02514v1), "Selection from hierarchical data with conformal e-values".

Faithful implementations of Alg. 1 (subsampling conformal e-values, Eqs. 6-8) and Alg. 2
(hierarchical conformal e-values, Eqs. 16-18), each unboosted or boosted (U-eBH, Eq. 13), as
specified in notes/refs/lee_ren_2501.02514v1_method.md.

Mapping to CertiChain: group = incident, unit = candidate link, score V^ = -s (smaller =
stronger evidence), calibration null = Y != 1 (strict; AMB is null), test batch = ALL M >= 2
test incidents of a split. Guarantee (their Thm 1/2, under their Assumption 1 incl. group sizes
independent of the data): POOLED FDR over the M test groups <= alpha.

Alg. 1 (one draw i*_k ~ Unif{1..N_k} per calibration group):
  FDP^_j(t) = [sum_k 1{V*_k < t, null_k} + 1] / [1 v sum_{l != j} #{V_{K+l,i} < t}] * (sum_{l=1}^M N_{K+l}) / (K+1)
  T_j = sup{t : FDP^_j(t) <= alpha~};  e_{j,i} = (K+1) 1{V_{K+j,i} < T_j} / (sum_k 1{V*_k < T_j, null_k} + 1)
  Derandomised: exact Eq. 11 averages over all prod_k N_k tuples (intractable); we use the
  paper's Monte-Carlo average (Eq. 12) over r draws ("A1avg"; r is a parameter, default 100).
Alg. 2:
  FDP^+_j(t) = [sum_k N_k^-1 #{V_{k,i} < t, null} + 1] / [1 v sum_{l != j} #{V_{K+l,i} < t}] * (sum_{l != j} N_{K+l}) / (K+1)
  FDP^-_j(t) = same without the "+1";  T^+_j, T^-_j = sup{t : FDP^+/-_j(t) <= alpha~}
  e_{j,i} = (K+1) 1{V_{K+j,i} < T^+_j} / (sum_k N_k^-1 #{V_{k,i} < T^-_j, null} + 1)
Then e-BH at level alpha over all n = sum_j N_{K+j} test units. alpha~ = 0.9 alpha (paper's
recommendation). Boosting: e / U with ONE U ~ Unif(0, 1) per split (independent of the data,
drawn from a seeded RNG; Remark 3's internal-randomness construction is not implemented -- a
caller may pass its own super-uniform ``u``). Boosting never changes the T's (Remark 4).

Computation: each FDP^_j is a step function of t that changes only at calibration-null values
and other groups' test values, with strict "<". On an interval (v_a, v_{a+1}] it counts values
<= v_a, so sup{t : FDP^ <= alpha~} is the right end of the highest qualifying interval: a
breakpoint, +inf (the last interval qualifies) or -inf (none qualifies). We evaluate at all
breakpoints (sorted union of calibration-null and test values, plus +inf), pruned by the
necessary condition FDP^_j <= alpha~ => (num + 1) * scale_min <= alpha~ * max(1, #test < t).
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from ..certify.evalues import ebh

TOL = 1e-12
NEG_INF_IDX = -1


# --------------------------------------------------------------------------- step-function sup
def _sup_thresholds(bp: np.ndarray, num: np.ndarray, Tt: np.ndarray, gvals: Sequence[np.ndarray],
                    scale: np.ndarray, alpha_t: float) -> np.ndarray:
    """Per test group j: the largest breakpoint index b with
           num_b * scale_j <= alpha_t * max(1, Tt_b - T_{j,b}),   T_{j,b} = #{group-j values < bp_b},
    or -1 if none (T_j = -inf). ``num`` already includes the '+1' when required."""
    M = len(gvals)
    out = np.full(M, NEG_INF_IDX, dtype=np.int64)
    if M == 0 or bp.size == 0:
        return out
    smin = float(np.min(scale))
    den_all = np.maximum(1.0, Tt.astype(np.float64))
    nec = num * smin <= alpha_t * den_all * (1.0 + TOL)
    if not nec.any():
        return out
    b_nec = int(np.flatnonzero(nec)[-1])
    bpr = bp[: b_nec + 1]
    numr = num[: b_nec + 1]
    Ttr = Tt[: b_nec + 1].astype(np.float64)
    ratio = numr / np.maximum(1.0, Ttr)
    gmin = np.minimum.accumulate(ratio[::-1])[::-1]          # suffix min, non-decreasing in b
    top = bpr[-1]
    for j in range(M):
        g = gvals[j]
        if g.size == 0 or g[0] >= top:                        # T_{j,b} = 0 for all b <= b_nec
            theta = alpha_t / scale[j]
            out[j] = int(np.searchsorted(gmin, theta * (1.0 + TOL), side="right")) - 1
        else:
            Tj = np.searchsorted(g, bpr, side="left").astype(np.float64)
            ok = numr * scale[j] <= alpha_t * np.maximum(1.0, Ttr - Tj) * (1.0 + TOL)
            idx = np.flatnonzero(ok)
            out[j] = int(idx[-1]) if idx.size else NEG_INF_IDX
    return out


class _TestBatch:
    """Sorted test values per group (V ascending) + the global sorted array."""

    def __init__(self, test_vals: Sequence[np.ndarray]):
        self.orders = [np.argsort(np.asarray(v, dtype=np.float64), kind="stable") for v in test_vals]
        self.g = [np.asarray(v, dtype=np.float64)[o] for v, o in zip(test_vals, self.orders)]
        self.N = np.array([g.size for g in self.g], dtype=np.int64)
        self.n = int(self.N.sum())
        self.all_sorted = np.sort(np.concatenate(self.g)) if self.n else np.empty(0)
        if any(np.isnan(g).any() for g in self.g):
            raise ValueError("NaN test score")
        # distinct test values and #{test values < each}: shared by every draw / threshold search
        self.tu = np.unique(self.all_sorted)
        self.tu_lt = np.searchsorted(self.all_sorted, self.tu, side="left")

    def n_below(self, j: int, t: float) -> int:
        return int(np.searchsorted(self.g[j], t, side="left"))


def _breakpoints(cal_sorted: np.ndarray, cal_cumw: np.ndarray, tb: _TestBatch,
                 c_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Breakpoints t (sorted; distinct test values merged with calibration-null values, + inf) with
    Tt_b = #{test values < t} and C_b = cal_cumw[#{cal values < t}].

    Exact pruning: every FDP^_j has denominator <= n - N_j, so a qualifying t needs
    C(t) + p1 <= alpha~ (K+1) =: c_max + p1. Breakpoints with C_b > c_max can never qualify and are
    dropped (this also drops +inf whenever some calibration value is excluded). O(n) merge; duplicate
    breakpoints are harmless (identical step-function values)."""
    m = cal_sorted.size
    mstar = int(np.searchsorted(cal_cumw, c_max * (1.0 + TOL) + 1e-12, side="right")) - 1
    if mstar >= m:                       # no cap
        cal_part, tu_part, tu_lt = cal_sorted, tb.tu, tb.tu_lt
        add_inf = True
    else:
        t_cap = cal_sorted[mstar]
        cal_part = cal_sorted[: mstar + 1]
        k = int(np.searchsorted(tb.tu, t_cap, side="right"))
        tu_part, tu_lt = tb.tu[:k], tb.tu_lt[:k]
        add_inf = False
    pos = np.searchsorted(tu_part, cal_part, side="left")
    bp = np.insert(tu_part, pos, cal_part)
    Tt = np.insert(tu_lt, pos, np.searchsorted(tb.all_sorted, cal_part, side="left"))
    if add_inf:
        bp, Tt = np.append(bp, np.inf), np.append(Tt, tb.n)
    C = cal_cumw[np.searchsorted(cal_sorted, bp, side="left")]
    return bp, Tt, C


def _ebh_grouped(e_j: np.ndarray, n_j: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """e-BH when e-values are constant (e_j) on n_j units of group j and 0 elsewhere.
    Returns a boolean per group: all n_j eligible units rejected (True) or none."""
    keep = (n_j > 0) & (e_j > 0)
    rej = np.zeros(e_j.size, dtype=bool)
    if not keep.any():
        return rej
    idx = np.flatnonzero(keep)
    o = idx[np.argsort(-e_j[idx], kind="stable")]
    ends = np.cumsum(n_j[o])
    ok = alpha * ends * e_j[o] >= n * (1.0 - TOL)
    if not ok.any():
        return rej
    kstar = int(np.flatnonzero(ok)[-1])
    e_star = e_j[o][kstar]
    rej[idx[e_j[idx] >= e_star * (1.0 - TOL)]] = True
    return rej


class _CalBatch:
    """Calibration groups as concatenated arrays (V^ values, null flags, group sizes)."""

    def __init__(self, cal: Sequence[tuple[np.ndarray, np.ndarray]]):
        self.K = len(cal)
        self.sizes = np.array([np.asarray(v).size for v, _ in cal], dtype=np.int64)
        self.offsets = np.concatenate(([0], np.cumsum(self.sizes))).astype(np.int64)
        self.v = np.concatenate([np.asarray(v, dtype=np.float64) for v, _ in cal]) if cal else np.empty(0)
        self.null = np.concatenate([np.asarray(nl, dtype=bool) for _, nl in cal]) if cal else np.empty(0, bool)
        if np.isnan(self.v).any():
            raise ValueError("NaN calibration score")
        grp = np.repeat(np.arange(self.K), self.sizes)
        nm = self.null
        o = np.argsort(self.v[nm], kind="stable")
        self.null_sorted = self.v[nm][o]
        w = 1.0 / self.sizes[grp[nm]][o]
        self.null_cumw = np.concatenate(([0.0], np.cumsum(w)))

    def draw(self, rng) -> np.ndarray:
        """i*_k ~ Unif{0..N_k-1} for every group (Eq. 6); -1 for empty groups."""
        u = rng.random(self.K)
        picks = np.floor(u * self.sizes).astype(np.int64)
        return np.where(self.sizes > 0, np.minimum(picks, np.maximum(self.sizes - 1, 0)), -1)


# --------------------------------------------------------------------------- Alg. 2
def _a2_evalues(cb: _CalBatch, tb: _TestBatch, alpha_t: float):
    """Returns (e_j [M], n_eligible_j [M], diagnostics)."""
    K = cb.K
    bp, Tt, C = _breakpoints(cb.null_sorted, cb.null_cumw, tb, alpha_t * (K + 1.0))   # T- cap (looser)
    scale = (tb.n - tb.N).astype(np.float64) / (K + 1.0)
    ip = _sup_thresholds(bp, C + 1.0, Tt, tb.g, scale, alpha_t)
    im = _sup_thresholds(bp, C, Tt, tb.g, scale, alpha_t)
    M = len(tb.g)
    e = np.zeros(M)
    n_el = np.zeros(M, dtype=np.int64)
    for j in range(M):
        if ip[j] < 0:
            continue
        n_el[j] = tb.n_below(j, bp[ip[j]])
        Cm = C[im[j]] if im[j] >= 0 else 0.0
        e[j] = (K + 1.0) / (Cm + 1.0)
    tp = np.where(ip >= 0, bp[np.maximum(ip, 0)], -np.inf)
    tm = np.where(im >= 0, bp[np.maximum(im, 0)], -np.inf)
    return e, n_el, {"T_plus": tp, "T_minus": tm}


# --------------------------------------------------------------------------- Alg. 1
def _a1_draw(cb: _CalBatch, tb: _TestBatch, alpha_t: float, picks: np.ndarray):
    """One subsample (picks[k] = index i*_k into group k, -1 = empty group).
    Returns (e_j, n_eligible_j, T_j)."""
    K = cb.K
    picks = np.asarray(picks, dtype=np.int64)
    ok = picks >= 0
    gidx = cb.offsets[:-1][ok] + picks[ok]
    sv, sn = cb.v[gidx], cb.null[gidx]
    nv = np.sort(sv[sn])
    bp, Tt, C = _breakpoints(nv, np.arange(nv.size + 1, dtype=np.float64), tb, alpha_t * (K + 1.0) - 1.0)
    M = len(tb.g)
    scale = np.full(M, tb.n / (K + 1.0))                     # Eq. (8): ALL M test groups
    it = _sup_thresholds(bp, C + 1.0, Tt, tb.g, scale, alpha_t)
    e = np.zeros(M)
    n_el = np.zeros(M, dtype=np.int64)
    for j in range(M):
        if it[j] < 0:
            continue
        n_el[j] = tb.n_below(j, bp[it[j]])
        e[j] = (K + 1.0) / (C[it[j]] + 1.0)
    T = np.where(it >= 0, bp[np.maximum(it, 0)], -np.inf)
    return e, n_el, T


# --------------------------------------------------------------------------- public API
def lee_ren_evalues(cal_groups, test_groups, variant: Literal["A1", "A2"], alpha: float, *,
                    alpha_tilde: float | None = None, r: int = 1, seed: int = 0,
                    subsamples: Sequence[Sequence[int]] | None = None) -> dict:
    """Unboosted e-values per test unit (original order within each group) + diagnostics.

    ``cal_groups``: list of (V^ values, null flags); ``test_groups``: list of V^ arrays. NOTE the
    V^ convention here (smaller = more significant); ``lee_ren`` takes scores and negates them.
    ``subsamples`` (A1): explicit tuples (i*_1..i*_K), one per draw (overrides r/seed).
    """
    if len(test_groups) < 2:
        raise ValueError("Lee-Ren needs M >= 2 test groups (sums over l != j)")
    a_t = 0.9 * alpha if alpha_tilde is None else float(alpha_tilde)
    cb = _CalBatch([(np.asarray(v, dtype=np.float64), np.asarray(nl, dtype=bool)) for v, nl in cal_groups])
    tb = _TestBatch(test_groups)
    M = len(tb.g)
    e_sorted = [np.zeros(n) for n in tb.N]
    diag: dict = {}
    if variant == "A2":
        e, n_el, d = _a2_evalues(cb, tb, a_t)
        for j in range(M):
            e_sorted[j][: n_el[j]] = e[j]
        diag.update(d)
        diag["e_group"] = e
    elif variant == "A1":
        rng = np.random.default_rng([seed, 1])
        draws = [np.asarray(s, dtype=np.int64) for s in subsamples] if subsamples is not None else \
            [cb.draw(rng) for _ in range(max(1, int(r)))]
        Ts = []
        for picks in draws:
            e, n_el, T = _a1_draw(cb, tb, a_t, picks)
            for j in range(M):
                e_sorted[j][: n_el[j]] += e[j]
            Ts.append(T)
        for j in range(M):
            e_sorted[j] /= len(draws)
        diag["T"] = np.array(Ts)
        diag["r"] = len(draws)
    else:
        raise ValueError(variant)
    e_orig = []
    for j in range(M):
        out = np.empty(tb.N[j])
        out[tb.orders[j]] = e_sorted[j]
        e_orig.append(out)
    diag["e"] = e_orig
    diag["alpha_tilde"] = a_t
    return diag


def lee_ren(cal_groups, test_groups, alpha: float, variant: Literal["A1", "A2"], boost: bool, *,
            alpha_tilde: float | None = None, r: int = 1, seed: int = 0, u: float | None = None,
            scores_are_v: bool = False) -> list[np.ndarray]:
    """DESIGN §7.2 signature. Inputs are SCORES (higher = more likely a true link) unless
    ``scores_are_v``: cal_groups = [(scores, y)], test_groups = [scores]. Returns one boolean
    rejection mask per test group. A1 with r = 1 is the single-draw Alg. 1; r > 1 is Eq. 12."""
    if scores_are_v:
        cg = [(np.asarray(v, float), np.asarray(nl, bool)) for v, nl in cal_groups]
        tg = [np.asarray(v, float) for v in test_groups]
    else:
        cg = [(-np.asarray(s, float), np.asarray(y) != 1) for s, y in cal_groups]
        tg = [-np.asarray(s, float) for s in test_groups]
    d = lee_ren_evalues(cg, tg, variant, alpha, alpha_tilde=alpha_tilde, r=r, seed=seed)
    e_all = np.concatenate(d["e"]) if d["e"] else np.empty(0)
    if boost:
        if u is None:
            u = float(np.random.default_rng([seed, 2]).uniform())
        e_all = e_all / max(float(u), 1e-300)
    mask = ebh(e_all, alpha)
    out, pos = [], 0
    for e in d["e"]:
        out.append(mask[pos: pos + e.size])
        pos += e.size
    return out


def lee_ren_R(bank, cal_rows: np.ndarray, test_rows: np.ndarray, alpha: float, variant: Literal["A1", "A2"],
              boost: bool, *, r: int = 1, seed: int = 0, alpha_tilde: float | None = None,
              u: float | None = None) -> np.ndarray:
    """Selection sizes R [M] on a LossBank for one split (candidates are already sorted by score
    descending inside each incident, so every rejection set is a prefix)."""
    cb = _CalBatch([(-bank.scores(int(k)), bank.labels(int(k)) != 1) for k in cal_rows])
    test = [-bank.scores(int(j)) for j in test_rows]
    a_t = 0.9 * alpha if alpha_tilde is None else float(alpha_tilde)
    tb = _TestBatch(test)
    M = len(test)
    if M < 2:
        raise ValueError("Lee-Ren needs M >= 2 test incidents")
    if variant == "A2" or (variant == "A1" and r == 1):
        if variant == "A2":
            e, n_el, _ = _a2_evalues(cb, tb, a_t)
        else:
            e, n_el, _ = _a1_draw(cb, tb, a_t, cb.draw(np.random.default_rng([seed, 1])))
        if boost:
            uu = float(np.random.default_rng([seed, 2]).uniform()) if u is None else float(u)
            e = e / max(uu, 1e-300)
        rej = _ebh_grouped(e, n_el, tb.n, alpha)
        return np.where(rej, n_el, 0).astype(np.int64)
    # A1 averaged over r draws (Eq. 12): unit-level e-values via a difference array
    rng = np.random.default_rng([seed, 1])
    starts = np.concatenate(([0], np.cumsum(tb.N)))
    diff = np.zeros(tb.n + 1)
    for _ in range(int(r)):
        e, n_el, _ = _a1_draw(cb, tb, a_t, cb.draw(rng))
        nz = n_el > 0
        np.add.at(diff, starts[:-1][nz], e[nz])
        np.add.at(diff, starts[:-1][nz] + n_el[nz], -e[nz])
    e_unit = np.cumsum(diff)[: tb.n] / int(r)
    if boost:
        uu = float(np.random.default_rng([seed, 2]).uniform()) if u is None else float(u)
        e_unit = e_unit / max(uu, 1e-300)
    mask = ebh(e_unit, alpha)
    cm = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    return (cm[starts[1:]] - cm[starts[:-1]]).astype(np.int64)
