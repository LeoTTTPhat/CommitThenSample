"""Chain certification (DESIGN §3.6, Alg. 5; Cor. 2; Example 1) and the chain baselines.

Alg. 5 on an edge set S_e (one fixed map, applied identically to every incident):
  1. G = (U, S_e).
  2. Widest-path DP in time order: best[v] = max_{(u,v) in S_e} min(best[u], s_uv), with
     best = +inf at roots (units without an in-edge in G) and back-pointers. Ties in the max
     are broken by the larger edge score, then the smaller candidate id (deterministic).
  3. sinks = units with an in-edge and no out-edge in G; keep the J (=20) sinks with the
     largest best[.] (ties: earlier unit first).
  4. F^max = widest root->sink paths (deduplicated); F = all prefixes (reporting only).
  5. composite loss L^cmp = max(FDP_edge, FDP_maxchain), monotonised over the grid, then CRC.

A chain is a tuple of candidate ids (edges) in root->sink order. Edge (u, v) always has
pos(u) < pos(v), where pos is the time order of units (unit_id order by default, which is
the time order in UNIT_SCHEMA, or the rank of (unit_ts, unit_id) if unit times are given).

Baselines (E5):
  * chain-naive: lambda-hat from edge-only CRC; chains read off S_e(lambda-hat).
  * chain-RT (coherent-factuality style [RT25]): on the score-threshold family G_t = {s >= t}
    (t on the S_0-quantile grid), r_k = the smallest grid threshold t such that every widest
    chain of G_t' is fully correct for all t' >= t; t-hat = the ceil((1-alpha)(K+1))-th smallest
    r_k. Guarantee: P(all reported maximal chains correct) >= 1 - alpha (per incident, FWER-type).
    In grid-index form this is exactly CRC with B = 1 on the monotonised 0/1 loss
    1{some reported chain is incorrect} (see rt_index / tests).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .crc import crc_threshold

INF = math.inf
J_DEFAULT = 20


# --------------------------------------------------------------------------- unit order
def unit_positions(n_units: int, unit_ts=None) -> np.ndarray:
    """pos[unit_id] = time-order position. Without times, unit ids are taken as the time order."""
    if unit_ts is None:
        return np.arange(n_units, dtype=np.int64)
    ts = np.asarray(unit_ts, dtype=np.float64)
    order = np.lexsort((np.arange(ts.size), ts))
    pos = np.empty(ts.size, dtype=np.int64)
    pos[order] = np.arange(ts.size)
    return pos


# --------------------------------------------------------------------------- DP core
def widest_chains(eu, ev, es, ecid, upos: np.ndarray, J: int = J_DEFAULT) -> list[tuple[int, ...]]:
    """Alg. 5 steps 2-4 on the edge list (eu[i] -> ev[i], score es[i], candidate id ecid[i]).

    Returns F^max: up to J widest root->sink chains (tuples of candidate ids), sorted by the
    sink's bottleneck value descending.
    """
    eu = np.asarray(eu, dtype=np.int64)
    ev = np.asarray(ev, dtype=np.int64)
    E = eu.size
    if E == 0:
        return []
    es = np.asarray(es, dtype=np.float64)
    ecid = np.asarray(ecid, dtype=np.int64)
    pu, pv = upos[eu], upos[ev]
    if np.any(pu >= pv):
        raise ValueError("edge with pos(u) >= pos(v): candidates must point forward in time")
    order = np.lexsort((ecid, pv))
    eu_l = eu[order].tolist()
    ev_l = ev[order].tolist()
    es_l = es[order].tolist()
    ec_l = ecid[order].tolist()
    best: dict[int, list] = {}  # v -> [value, edge score, cand id, edge index]
    for k in range(E):
        u = eu_l[k]
        v = ev_l[k]
        s = es_l[k]
        c = ec_l[k]
        bu = best.get(u)
        if bu is None:            # u has no in-edge (all of u's in-edges precede v in time) -> root
            val = s
        else:
            val = bu[0] if bu[0] < s else s
        cur = best.get(v)
        if (cur is None or val > cur[0]
                or (val == cur[0] and (s > cur[1] or (s == cur[1] and c < cur[2])))):
            best[v] = [val, s, c, k]
    has_out = set(eu_l)
    sinks = [v for v in best if v not in has_out]
    sinks.sort(key=lambda v: (-best[v][0], int(upos[v]), v))
    chains: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for v in sinks[:J]:
        path = []
        x = v
        while True:
            b = best.get(x)
            if b is None:
                break
            k = b[3]
            path.append(ec_l[k])
            x = eu_l[k]
        ch = tuple(reversed(path))
        if ch not in seen:
            seen.add(ch)
            chains.append(ch)
    return chains


def chain_family(S_e, units, scores, J: int = J_DEFAULT) -> list[tuple[int, ...]]:
    """DESIGN §7.2 signature. ``S_e``: selected candidate indices; ``units``: (u, v) arrays over
    ALL candidates, or a dict with keys 'u', 'v' and optional 'unit_ts'; ``scores``: all scores."""
    if isinstance(units, dict):
        u, v, uts = units["u"], units["v"], units.get("unit_ts")
    else:
        u, v = units
        uts = None
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    sel = np.asarray(S_e, dtype=np.int64)
    n_units = int(max(u.max(initial=-1), v.max(initial=-1)) + 1) if uts is None else len(uts)
    upos = unit_positions(n_units, uts)
    sc = np.asarray(scores, dtype=np.float64)
    return widest_chains(u[sel], v[sel], sc[sel], sel, upos, J)


# --------------------------------------------------------------------------- chain losses
def maxchain_fdp(chains: Sequence[tuple[int, ...]], y, strict: bool = True) -> float:
    """|{c in F^max : some edge has Y != 1}| / max(|F^max|, 1). Lenient: chains are false iff they
    contain a Y = 0 edge, and chains whose only non-true edges are AMB are excluded."""
    y = np.asarray(y)
    if len(chains) == 0:
        return 0.0
    bad = bad0 = 0
    for c in chains:
        yc = y[list(c)]
        bad += bool(np.any(yc != 1))
        bad0 += bool(np.any(yc == 0))
    if strict:
        return bad / max(len(chains), 1)
    denom = len(chains) - (bad - bad0)
    return bad0 / max(denom, 1)


def prefix_family(chains: Iterable[tuple[int, ...]]) -> set[tuple[int, ...]]:
    """F(theta): all (deduplicated) prefixes of the maximal chains."""
    out: set[tuple[int, ...]] = set()
    for c in chains:
        for i in range(1, len(c) + 1):
            out.add(c[:i])
    return out


def chain_stats(chains: Sequence[tuple[int, ...]], y: np.ndarray, gt_paths=None) -> tuple[int, int, int, int, int, int]:
    """(n_chains, n_bad_strict, n_bad0, n_prefixes, n_bad_prefixes, n_gt_paths_covered)."""
    n = len(chains)
    bad = bad0 = 0
    pref: dict[tuple[int, ...], bool] = {}
    sets = []
    for c in chains:
        yc = y[list(c)]
        isbad = yc != 1
        bad += bool(isbad.any())
        bad0 += bool(np.any(yc == 0))
        first_bad = int(np.argmax(isbad)) if isbad.any() else len(c)
        for i in range(1, len(c) + 1):
            pref[c[:i]] = i > first_bad
        sets.append(set(c))
    pf_n = len(pref)
    pf_bad = sum(pref.values())
    cov = 0
    if gt_paths:
        for path in gt_paths:
            pth = [int(x) for x in path]
            if len(pth) == 0 or min(pth) < 0:
                continue
            ps = set(pth)
            if any(ps <= s for s in sets):
                cov += 1
    return n, bad, bad0, pf_n, pf_bad, cov


def ccr(chains: Sequence[tuple[int, ...]], gt_paths) -> float:
    """Fraction of GT root->leaf transition paths (cand-id sequences; -1 = not in C) fully covered
    by one maximal chain. NaN if there are no GT paths."""
    if not gt_paths:
        return float("nan")
    sets = [set(c) for c in chains]
    cov = 0
    for path in gt_paths:
        pth = [int(x) for x in path]
        if pth and min(pth) >= 0 and any(set(pth) <= s for s in sets):
            cov += 1
    return cov / len(gt_paths)


# --------------------------------------------------------------------------- calibration
def running_max(table: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(np.asarray(table, dtype=np.float64), axis=-1)


def composite_loss(fdp_edge: np.ndarray, fdp_maxchain: np.ndarray) -> np.ndarray:
    """L~cmp = running max over the grid of max(FDP_edge, FDP_maxchain) (Alg. 5 steps 5-6)."""
    return running_max(np.maximum(fdp_edge, fdp_maxchain))


def composite_threshold(Lcmp_cal: np.ndarray, alpha: float, B: float = 1.0) -> int:
    """theta-hat (grid index) by CRC on the monotonised composite loss (Cor. 2)."""
    return crc_threshold(Lcmp_cal, alpha, B)


def naive_chain_threshold(Ledge_cal: np.ndarray, alpha: float, B: float = 1.0) -> int:
    """chain-naive: lambda-hat from edge-only CRC; chains are then read off S_e(lambda-hat)."""
    return crc_threshold(Ledge_cal, alpha, B)


def rt_r_index(fwer_mono: np.ndarray) -> np.ndarray:
    """Per incident: the largest grid index j with (monotonised) 1{some chain wrong} == 0 at all
    indices <= j. Index 0 (t = +inf, nothing reported) always qualifies."""
    f = np.asarray(fwer_mono) > 0
    if f.ndim == 1:
        f = f[None, :]
    first_bad = np.where(f.any(1), np.argmax(f, axis=1), f.shape[1])
    return (first_bad - 1).astype(np.int64)


def rt_chain_threshold(r_cal, alpha: float) -> float:
    """t-hat = the ceil((1-alpha)(K+1))-th smallest r_k (DESIGN §3.6 (ii)); +inf if that rank > K.
    ``r_cal``: per-calibration-incident smallest all-correct bottleneck thresholds."""
    r = np.sort(np.asarray(r_cal, dtype=np.float64))
    K = r.size
    m = math.ceil((1.0 - alpha) * (K + 1) - 1e-12)
    if m > K or m < 1:
        return INF
    return float(r[m - 1])


def rt_index(fwer_cal_mono: np.ndarray, alpha: float) -> int:
    """chain-RT in grid-index form: q = floor(alpha (K+1)); j-hat = q-th smallest j_k (0 if q = 0).
    Equal to rt_chain_threshold on r_k = t_grid[j_k] and to crc_threshold on the 0/1 loss."""
    j = np.sort(rt_r_index(fwer_cal_mono))
    K = j.size
    q = math.floor(alpha * (K + 1) + 1e-12)
    if q < 1:
        return 0
    return int(j[min(q, K) - 1])
