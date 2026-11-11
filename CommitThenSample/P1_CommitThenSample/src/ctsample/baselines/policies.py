"""Baseline retention policies at equal budget (DESIGN §6.2) on a StreamTables stream.

Budget: every policy is matched to the same total content bytes B̄·E over the stream, B̄ = ρ Σ_all s / E —
in expectation for randomised policies, within ±1 % exactly for deterministic ones (deterministic
baselines are matched on the evaluation stream itself: this is the §6.2 equal-storage normalisation, not a
tuned hyper-parameter, and it can only favour the baselines). Only payload units carry content
(zero-payload units: π = 0, never retained; the shared metadata store is costed once, E6).

    b2_ring_buffer        FIFO of size B̄E evaluated at the stream end (analytic): the newest units that fit
                          (+ the tail packets of the boundary unit: 'mask_any', counted in the cost)
    b3_time_machine       per-connection cut-off: the first N pcap bytes of every connection (flow) are kept,
                          N by bisection; unit-level masks 'full' (whole unit kept) and 'any' (cut inside it)
    b4_topk_pmal          deterministic top-k by P_mal over the whole stream (budget B̄E ± 1 %, exact matching
                          as §6.2 requires; an integral threshold controller cannot be matched to ±1 % because its
                          total spend is not monotone in θ_ref); b4_controller (B4ctl) = the online horizon
                          threshold controller logit θ_{e+1} = logit θ_ref − κ clip(D_e/(W_c B̄), −L, L), with
                          θ_ref = the global top-k threshold, spend/target reported (not matched)
    b6_alert_triggered    Smart-PCAP-like: units of alerting flows + the same device's payload units within
                          ±60 s of an alert, by severity (then time), until the budget; leftover budget = uniform
                          random units (π_r); scored on D estimands only
    b8_uniform / b10_size CtS designs with a = √s (constant π) / a = s (π ∝ √s): see scripts/05_designs.py
                          (sim.montecarlo.static_pass); provided here as one-epoch-free closed forms too
    b13_topk_reserve      (1 − γ) B̄E to deterministic top-k by P_mal (global) + γ B̄E to a keyed uniform
                          reserve with π_r on every other payload unit (expected spend B̄E exactly)
    b15_packet_sampling   [SHOULD] 1:N systematic packet sampling with a random per-epoch offset over payload
                          units' packets, N = Σ_pay s / (B̄E); analytic per-unit P(≥ 1 packet sampled)
rev1 (review R3-M2, fix list B4 / M6) — online and faithful variants:
    b4_online             B4 with the online horizon threshold controller whose θ_ref is calibrated on the fold's
                          TRAINING streams (no test-stream hindsight); spend/target reported, not matched
    b13_online            B13 with an online/committed threshold: per epoch, the top part (score ≥ θ_e, θ_e from
                          the same controller targeting (1 − γ) B̄, training-calibrated θ_ref) is kept with π = 1 and
                          every other payload unit gets the keyed reserve π_r,e = max(φ0, min(1, γ B̄ / Σ_rest,e s))
                          from the epoch's own committed metadata → π predictable, HT / LCS / UCS (floor φ0) valid
    dlt_threshold         Duffield–Lund–Thorup threshold sampling π_i = min(1, s_i / z), z by bisection so that the
                          expected spend is B̄E (keyed, valid HT; no floor)
    b3_offsets / b3_evidence  byte-level Time-Machine scoring: a content unit's evidence is retained iff its first
                          payload-carrying packet (or, for an estimand, its first hit packet) lies within the first N
                          bytes of its connection (consistent with B15 and Prop. 4's packet-level detection)
    calibrate_threshold   θ_ref of a static score threshold that spends a budget share on pooled training streams
Return value: dict(name, type 'pi' | 'mask', pi or mask (N,), cost (expected or realised), budget, spend_ratio,
params). For masks, pi = mask as float (π ∈ {0, 1}).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["tiebreak", "topk_global", "b4_controller", "budget", "b2_ring_buffer", "b3_time_machine", "b4_topk_pmal", "b6_alert_triggered", "b8_uniform",
           "b10_size", "b13_topk_reserve", "b15_packet_sampling", "threshold_controller_pass", "calibrate_threshold",
           "b4_online", "b13_online", "b4_epoch", "b13_epoch", "dlt_threshold", "conn_bytes_before",
           "b3_evidence"]


def budget(T, rho: float) -> tuple[float, float]:
    """(B̄, B̄E) with B̄ = ρ Σ_all s / E."""
    s = np.asarray(T.s, dtype=np.float64)
    E = T.n_epochs
    Bbar = rho * s.sum() / E
    return Bbar, Bbar * E


def _result(name, kind, arr, cost, B, params):
    out = {"name": name, "type": kind, "cost": float(cost), "budget": float(B), "spend_ratio": float(cost / B),
           "params": params}
    if kind == "mask":
        out["mask"] = arr.astype(bool)
        out["pi"] = arr.astype(np.float64)
    else:
        out["pi"] = arr.astype(np.float64)
    return out


# ------------------------------------------------------------------------------------------ #
def b2_ring_buffer(T, rho: float) -> dict:
    """FIFO of B̄E bytes at the stream end: the newest payload units that fit entirely ('full' mask) plus the
    tail packets of the boundary unit (counted in the cost, flagged in 'mask_any'), so the spend is B̄E."""
    s = np.asarray(T.s, dtype=np.float64)
    pay = np.asarray(T.pay, dtype=bool)
    _, B = budget(T, rho)
    sp = np.where(pay, s, 0.0)
    cum_from_end = np.cumsum(sp[::-1])[::-1]               # bytes of this unit and every later one
    mask = pay & (cum_from_end <= B)
    used = float(sp[mask].sum())
    anyk = mask.copy()
    part = np.nonzero(pay & ~mask)[0]
    kept = np.zeros(T.n_units)
    kept[mask] = sp[mask]
    if len(part) and used < B:
        j = part[-1]                                      # newest unit that does not fit entirely
        kept[j] = B - used
        anyk[j] = True
    r = _result("B2", "mask", mask, float(kept.sum()), B, {"rho": rho, "boundary_partial_bytes": float(B - used)})
    r["mask_any"] = anyk
    r["retained_bytes"] = kept
    return r


# ------------------------------------------------------------------------------------------ #
def conn_bytes_before(T, flow_id: np.ndarray) -> np.ndarray:
    """Per PAYLOAD unit (payload order): pcap bytes of the earlier payload units of its connection (B3)."""
    s = np.asarray(T.s, dtype=np.float64)
    pay = np.asarray(T.pay, dtype=bool)
    idx = np.nonzero(pay)[0]
    fid = np.asarray(flow_id)[idx]
    _, f = np.unique(fid, return_inverse=True)
    f = f.ravel()
    sp = s[idx]
    o = np.lexsort((idx, f))
    so = sp[o]
    fo = f[o]
    cs = np.cumsum(so)
    start = np.r_[True, fo[1:] != fo[:-1]] if len(fo) else np.zeros(0, bool)
    base = np.maximum.accumulate(np.where(start, cs - so, 0.0)) if len(fo) else np.zeros(0)
    before = np.empty(len(o))
    before[o] = cs - so - base
    return before


def b3_evidence(N: float, before: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Byte-level B3: True where the packet at cost offset ``offset`` (bytes of the unit up to and including that
    packet) lies within the connection's first N bytes, i.e. before + offset <= N (payload order arrays)."""
    return (np.asarray(before, dtype=np.float64) + np.asarray(offset, dtype=np.float64)) <= float(N)


def b3_time_machine(T, rho: float, flow_id: np.ndarray, iters: int = 80) -> dict:
    """Per-connection cut-off N (pcap bytes). flow_id: (N,) connection identity of every unit (e.g. the
    units.parquet flow_uid_orig, as int codes). Payload units only; a unit is 'full' if all of it lies within
    its connection's first N bytes, 'any' if the cut-off falls inside it; cost = retained bytes."""
    s = np.asarray(T.s, dtype=np.float64)
    pay = np.asarray(T.pay, dtype=bool)
    _, B = budget(T, rho)
    idx = np.nonzero(pay)[0]
    fid = np.asarray(flow_id)[idx]
    _, f = np.unique(fid, return_inverse=True)
    f = f.ravel()
    sp = s[idx]
    o = np.lexsort((idx, f))                               # by connection, then stream order
    so = sp[o]
    fo = f[o]
    cs = np.cumsum(so)
    start = np.r_[True, fo[1:] != fo[:-1]]
    base = np.maximum.accumulate(np.where(start, cs - so, 0.0))
    before = np.empty(len(o))
    before[o] = cs - so - base                             # connection bytes before the unit

    def cost(N):
        return float(np.minimum(sp, np.maximum(N - before, 0.0)).sum())

    lo, hi = 0.0, float(sp.max() + before.max() + 1.0) if len(sp) else 1.0
    if cost(hi) <= B:
        N = hi
    else:
        for _ in range(iters):
            m = 0.5 * (lo + hi)
            if cost(m) > B:
                hi = m
            else:
                lo = m
        N = lo
    kept = np.minimum(sp, np.maximum(N - before, 0.0))
    full = np.zeros(T.n_units, bool)
    anyk = np.zeros(T.n_units, bool)
    full[idx] = kept >= sp
    anyk[idx] = kept > 0
    r = _result("B3", "mask", full, kept.sum(), B, {"rho": rho, "N_bytes": N})
    r["mask_any"] = anyk
    r["retained_bytes"] = np.zeros(T.n_units)
    r["retained_bytes"][idx] = kept
    return r


# ------------------------------------------------------------------------------------------ #
def tiebreak(n: int) -> np.ndarray:
    """Deterministic, content-free tie-break in [0, 1) per stream position (Knuth multiplicative hash)."""
    i = np.arange(n, dtype=np.uint64)
    return ((i * np.uint64(2654435761) + np.uint64(0x9E3779B9)) & np.uint64(0xFFFFFFFF)).astype(np.float64) / 2.0 ** 32


def threshold_controller_pass(T, score: np.ndarray, Bbar: float, logit_ref: float, kappa: float = 1.0,
                              Wc: float = 30.0, L: float = math.log(16), budget_share: float = 1.0,
                              eligible=None, jitter: float = 1e-7) -> tuple[np.ndarray, float, dict]:
    """Deterministic online threshold policy: retain eligible payload units with score >= θ_e where
    logit θ_e = logit_ref − κ clip(D_{e−1}/(W_c B̄'), −L, L), D_e = D_{e−1} + B̄' − cost_e, B̄' = share B̄.
    Ties of the (u16-quantised) score are broken by ``jitter`` × tiebreak(position): distinct LUT values are
    >= 6e-5 apart in logit, so the jitter only orders exact ties (and lets the budget be matched to ±1 %)."""
    s = np.asarray(T.s, dtype=np.float64)
    pay = np.asarray(T.pay, dtype=bool)
    el = pay if eligible is None else (pay & np.asarray(eligible, dtype=bool))
    sc = np.clip(np.asarray(score, dtype=np.float64), 1e-12, 1 - 1e-12)
    lg = np.log(sc / (1 - sc)) + jitter * tiebreak(len(sc))
    eptr = np.asarray(T.epoch_ptr)
    B1 = budget_share * Bbar
    D = 0.0
    mask = np.zeros(T.n_units, bool)
    total = 0.0
    maxD = 0.0
    for e in range(T.n_epochs):
        a, b = int(eptr[e]), int(eptr[e + 1])
        th = logit_ref - kappa * float(np.clip(D / (Wc * B1), -L, L))
        m = el[a:b] & (lg[a:b] >= th)
        mask[a:b] = m
        c = float(s[a:b][m].sum())
        total += c
        D += B1 - c
        maxD = max(maxD, abs(D))
    return mask, total, {"max_abs_D_over_Bbar": maxD / B1, "final_D_over_Bbar": D / B1}


def _match_threshold(T, score, Bbar, target_total, tol=0.01, eligible=None, share=1.0, iters=60):
    """Bisection on logit_ref so that the controller pass spends target_total ± tol."""
    lo, hi = -30.0, 30.0
    best = None
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        mask, tot, info = threshold_controller_pass(T, score, Bbar, m, budget_share=share, eligible=eligible)
        best = (m, mask, tot, info)
        if abs(tot / target_total - 1) <= tol * 0.5:
            break
        if tot > target_total:
            lo = m
        else:
            hi = m
    return best


def topk_global(T, score: np.ndarray, target: float, eligible=None) -> tuple[np.ndarray, float, float]:
    """Deterministic top-k over the whole stream: payload units by decreasing score (ties: tiebreak), the
    longest prefix whose cost is <= target, then a greedy fill with the next-ranked units that still fit.
    Returns (mask, cost, score threshold of the prefix)."""
    s = np.asarray(T.s, dtype=np.float64)
    pay = np.asarray(T.pay, dtype=bool)
    el = pay if eligible is None else (pay & np.asarray(eligible, dtype=bool))
    idx = np.nonzero(el)[0]
    sc = np.asarray(score, dtype=np.float64)[idx]
    o = idx[np.lexsort((tiebreak(T.n_units)[idx], -sc))]
    so = s[o]
    cum = np.cumsum(so)
    k = int(np.searchsorted(cum, target, side="right"))       # longest prefix that fits
    mask = np.zeros(T.n_units, bool)
    mask[o[:k]] = True
    used = float(cum[k - 1]) if k else 0.0
    thr = float(np.asarray(score)[o[k - 1]]) if k else float("inf")
    # greedy fill in rank order: skip units that no longer fit (a huge unit at the boundary), keep adding the
    # next-ranked units that do (deterministic; reaches the budget within the smallest unit size)
    rem = target - used
    j = k
    while rem > 0 and j < len(o):
        fits = np.nonzero(so[j:] <= rem)[0]
        if not len(fits):
            break
        j = j + int(fits[0])
        mask[o[j]] = True
        used += float(so[j])
        rem -= float(so[j])
        j += 1
    return mask, used, thr


def b4_topk_pmal(T, rho: float, pmal: np.ndarray) -> dict:
    """B4: deterministic top-k by P_mal over the stream, budget B̄E matched (±1 %, whole units)."""
    _, B = budget(T, rho)
    mask, tot, thr = topk_global(T, pmal, B)
    return _result("B4", "mask", mask, tot, B, {"rho": rho, "threshold": thr, "mode": "global top-k"})


def b4_controller(T, rho: float, pmal: np.ndarray, logit_ref: float | None = None) -> dict:
    """B4ctl: online threshold controller variant (spend/target reported, NOT budget-matched); logit_ref
    defaults to the logit of the global top-k threshold."""
    Bbar, B = budget(T, rho)
    if logit_ref is None:
        _, _, thr = topk_global(T, pmal, B)
        thr = min(max(thr, 1e-12), 1 - 1e-12)
        logit_ref = math.log(thr / (1 - thr))
    mask, tot, info = threshold_controller_pass(T, pmal, Bbar, logit_ref)
    return _result("B4ctl", "mask", mask, tot, B, {"rho": rho, "logit_ref": logit_ref, **info})


def b13_topk_reserve(T, rho: float, pmal: np.ndarray, gamma: float = 0.3) -> dict:
    """(1 − γ)B̄E deterministic top-k by P_mal (global) + keyed uniform reserve π_r on every other payload
    unit, π_r = (B̄E − top-k cost) / Σ_rest s (expected spend B̄E exactly). A CtS design with a step utility."""
    s = np.asarray(T.s, dtype=np.float64)
    pay = np.asarray(T.pay, dtype=bool)
    _, B = budget(T, rho)
    top, tot, thr = topk_global(T, pmal, (1 - gamma) * B)
    rest = pay & ~top
    pr = min(1.0, (B - tot) / max(s[rest].sum(), 1.0))
    pi = np.where(top, 1.0, np.where(rest, pr, 0.0))
    return _result("B13", "pi", pi, float((pi * s).sum()), B,
                   {"rho": rho, "gamma": gamma, "threshold": thr, "pi_reserve": pr, "topk_cost": tot})


# ------------------------------------------------------------------------------------------ #
# rev1: online (training-calibrated) B4 / B13 and DLT threshold sampling
# ------------------------------------------------------------------------------------------ #
def _logit(score, jitter: float = 1e-7):
    sc = np.clip(np.asarray(score, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(sc / (1 - sc)) + jitter * tiebreak(len(sc))


def calibrate_threshold(scores: list, sizes: list, budgets: list, iters: int = 80) -> float:
    """logit θ_ref of a STATIC rule 'keep a payload unit iff logit(score) (+ the controller's tie-break jitter)
    >= θ' such that the pooled spend over the training streams equals the pooled budget Σ budgets (bisection;
    spend is monotone in θ). scores/sizes: per training stream, over its payload units (stream order)."""
    lg = np.concatenate([_logit(x) for x in scores]) if scores else np.zeros(0)
    sz = np.concatenate([np.asarray(x, dtype=np.float64) for x in sizes]) if sizes else np.zeros(0)
    target = float(sum(budgets))
    lo, hi = -40.0, 40.0
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        if float(sz[lg >= m].sum()) > target:
            lo = m
        else:
            hi = m
    return hi


def b4_online(T, rho: float, pmal: np.ndarray, logit_ref: float) -> dict:
    """B4 online: horizon threshold controller with the training-calibrated θ_ref (no hindsight)."""
    Bbar, B = budget(T, rho)
    mask, tot, info = threshold_controller_pass(T, pmal, Bbar, logit_ref)
    return _result("B4on", "mask", mask, tot, B, {"rho": rho, "logit_ref": logit_ref, "calibration": "training",
                                                   **info})


def b13_online(T, rho: float, pmal: np.ndarray, logit_ref: float, phi0: float, gamma: float = 0.3) -> dict:
    """B13 online: per epoch top part (controller, share 1 − γ, training θ_ref) with π = 1, keyed reserve
    π_r,e = max(φ0, min(1, γ B̄ / Σ_rest,e s)) on the other payload units of the epoch (predictable)."""
    s = np.asarray(T.s, dtype=np.float64)
    pay = np.asarray(T.pay, dtype=bool)
    Bbar, B = budget(T, rho)
    top, tot, info = threshold_controller_pass(T, pmal, Bbar, logit_ref, budget_share=1.0 - gamma)
    eptr = np.asarray(T.epoch_ptr)
    pi = np.zeros(T.n_units)
    reserve = np.zeros(T.n_units)
    for e in range(T.n_epochs):
        a, b = int(eptr[e]), int(eptr[e + 1])
        rest = pay[a:b] & ~top[a:b]
        srest = float(s[a:b][rest].sum())
        pr = max(phi0, min(1.0, gamma * Bbar / srest)) if srest > 0 else 1.0
        reserve[a:b] = np.where(pay[a:b], pr, 0.0)
        pi[a:b] = np.where(top[a:b], 1.0, np.where(rest, pr, 0.0))
    r = _result("B13on", "pi", pi, float((pi * s).sum()), B,
                {"rho": rho, "gamma": gamma, "logit_ref": logit_ref, "calibration": "training", "phi0": phi0,
                 "topk_cost": tot, "reserve_rate_median": float(np.median(reserve[pay])) if pay.any() else 0.0,
                 "reserve_rate_min": float(reserve[pay].min()) if pay.any() else 0.0, **info})
    r["reserve_pi"] = reserve
    r["mask_top"] = top
    return r


def _epoch_topk(s_e, lg_e, el_e, budget_e) -> np.ndarray:
    """Deterministic greedy top-k inside one epoch: eligible units by decreasing (jittered) logit score, the
    longest prefix that fits budget_e, then the next-ranked units that still fit (as topk_global)."""
    idx = np.nonzero(el_e)[0]
    keep = np.zeros(len(s_e), bool)
    if budget_e <= 0 or not len(idx):
        return keep
    o = idx[np.argsort(-lg_e[idx], kind="stable")]
    so = s_e[o]
    cum = np.cumsum(so)
    k = int(np.searchsorted(cum, budget_e, side="right"))
    keep[o[:k]] = True
    rem = budget_e - (float(cum[k - 1]) if k else 0.0)
    j = k
    while rem > 0 and j < len(o):
        fits = np.nonzero(so[j:] <= rem)[0]
        if not len(fits):
            break
        j = j + int(fits[0])
        keep[o[j]] = True
        rem -= float(so[j])
        j += 1
    return keep


def b4_epoch(T, rho: float, pmal: np.ndarray) -> dict:
    """B4 online, budget-exact: per epoch e the highest-P_mal payload units (greedy) within B_e = B̄ + D_{e−1}
    (unspent budget carried forward, D_e = D_{e−1} + B̄ − cost_e ≥ 0); uses only the epoch's own metadata and
    the past, never later epochs; total spend ≤ B̄E."""
    s = np.asarray(T.s, dtype=np.float64)
    pay = np.asarray(T.pay, dtype=bool)
    Bbar, B = budget(T, rho)
    lg = _logit(pmal)
    eptr = np.asarray(T.epoch_ptr)
    mask = np.zeros(T.n_units, bool)
    D = 0.0
    tot = 0.0
    for e in range(T.n_epochs):
        a, b = int(eptr[e]), int(eptr[e + 1])
        k = _epoch_topk(s[a:b], lg[a:b], pay[a:b], Bbar + D)
        mask[a:b] = k
        c = float(s[a:b][k].sum())
        tot += c
        D += Bbar - c
    return _result("B4ep", "mask", mask, tot, B, {"rho": rho, "mode": "per-epoch top-k, budget carry",
                                                   "final_D_over_Bbar": D / Bbar})


def b13_epoch(T, rho: float, pmal: np.ndarray, phi0: float, gamma: float = 0.3) -> dict:
    """B13 online, budget-exact in expectation: per epoch B_e = B̄ + D_{e−1}; the top part takes the highest-P_mal
    payload units greedily within (1 − γ) B_e (π = 1); every other payload unit gets the keyed reserve
    π_r,e = max(φ0, min(1, (B_e − top cost_e) / Σ_rest,e s)); D_e = D_{e−1} + B̄ − (top cost + Σ_rest π_r,e s)
    (expected cost: π is a deterministic function of committed metadata and the past → predictable, valid HT)."""
    s = np.asarray(T.s, dtype=np.float64)
    pay = np.asarray(T.pay, dtype=bool)
    Bbar, B = budget(T, rho)
    lg = _logit(pmal)
    eptr = np.asarray(T.epoch_ptr)
    pi = np.zeros(T.n_units)
    reserve = np.zeros(T.n_units)
    top = np.zeros(T.n_units, bool)
    D = 0.0
    for e in range(T.n_epochs):
        a, b = int(eptr[e]), int(eptr[e + 1])
        Be = Bbar + D
        k = _epoch_topk(s[a:b], lg[a:b], pay[a:b], (1 - gamma) * Be)
        ct = float(s[a:b][k].sum())
        rest = pay[a:b] & ~k
        srest = float(s[a:b][rest].sum())
        pr = max(phi0, min(1.0, (Be - ct) / srest)) if srest > 0 else 1.0
        top[a:b] = k
        reserve[a:b] = np.where(pay[a:b], pr, 0.0)
        pi[a:b] = np.where(k, 1.0, np.where(rest, pr, 0.0))
        D += Bbar - (ct + pr * srest)
    r = _result("B13ep", "pi", pi, float((pi * s).sum()), B,
                {"rho": rho, "gamma": gamma, "phi0": phi0, "mode": "per-epoch top part + reserve, budget carry",
                 "reserve_rate_median": float(np.median(reserve[pay])) if pay.any() else 0.0,
                 "final_D_over_Bbar": D / Bbar})
    r["reserve_pi"] = reserve
    r["mask_top"] = top
    return r


def dlt_threshold(T, rho: float, iters: int = 100) -> dict:
    """Duffield–Lund–Thorup threshold sampling π_i = min(1, s_i/z); z by bisection: Σ π s = B̄E (payload units)."""
    s = np.asarray(T.s, dtype=np.float64)
    pay = np.asarray(T.pay, dtype=bool)
    _, B = budget(T, rho)
    sp = s[pay]
    lo, hi = math.log(max(sp.min(), 1.0)) - 30.0 if len(sp) else 0.0, math.log(max(sp.sum(), 1.0)) + 30.0
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        if float(np.sum(np.minimum(1.0, sp / math.exp(m)) * sp)) > B:
            lo = m
        else:
            hi = m
    z = math.exp(hi)
    pi = np.zeros(T.n_units)
    pi[pay] = np.minimum(1.0, sp / z)
    return _result("DLT", "pi", pi, float((pi * s).sum()), B, {"rho": rho, "z": z,
                                                                "share_pi1": float(np.mean(sp >= z)) if len(sp) else 0})


# ------------------------------------------------------------------------------------------ #
def b6_alert_triggered(T, rho: float, alert_sev: np.ndarray, flow_id: np.ndarray, first_ts_us: np.ndarray,
                       window_us: int = 60_000_000) -> dict:
    """alert_sev: (N,) min Suricata severity of the unit's alerts (1 = highest; 0/neg = no alert);
    flow_id: connection id per unit; first_ts_us: unit start time. Triggered = payload units of an alerting
    flow + payload units of the alerting unit's device within ±60 s; retained in (severity, time) order until
    the budget; leftover budget -> uniform π_r on the remaining payload units."""
    s = np.asarray(T.s, dtype=np.float64)
    pay = np.asarray(T.pay, dtype=bool)
    dev = np.asarray(T.device)
    _, B = budget(T, rho)
    sev = np.asarray(alert_sev, dtype=np.int64)
    alert = sev > 0
    pri = np.full(T.n_units, 99, dtype=np.int64)            # trigger priority (lower = earlier)
    when = np.full(T.n_units, np.iinfo(np.int64).max, dtype=np.int64)
    fid = np.asarray(flow_id)
    ts = np.asarray(first_ts_us, dtype=np.int64)
    if alert.any():
        ai = np.nonzero(alert)[0]
        # flows with alerts
        fl = {}
        for i in ai:
            k = fid[i]
            v = (sev[i], ts[i])
            fl[k] = min(fl.get(k, v), v)
        for i in np.nonzero(pay)[0]:
            v = fl.get(fid[i])
            if v is not None:
                pri[i], when[i] = min((pri[i], when[i]), v)
        # same device within ±60 s of an alert
        order = np.argsort(ts[ai], kind="stable")
        ai = ai[order]
        for d in np.unique(dev[ai]):
            ad = ai[dev[ai] == d]
            uts = ts[ad]
            usev = sev[ad]
            cand = np.nonzero(pay & (dev == d))[0]
            if not len(cand):
                continue
            lo = np.searchsorted(uts, ts[cand] - window_us, side="left")
            hi = np.searchsorted(uts, ts[cand] + window_us, side="right")
            for c, a, b in zip(cand, lo, hi):
                if b > a:
                    j = a + int(np.argmin(usev[a:b]))
                    v = (int(usev[j]), int(uts[j]))
                    if v < (pri[c], when[c]):
                        pri[c], when[c] = v
    trig = pay & (pri < 99)
    ti = np.nonzero(trig)[0]
    o = ti[np.lexsort((ti, when[ti], pri[ti]))]
    cum = np.cumsum(s[o])
    kept = o[cum <= B]
    mask = np.zeros(T.n_units, bool)
    mask[kept] = True
    used = float(s[kept].sum())
    rest = pay & ~mask
    pr = min(1.0, max(B - used, 0.0) / max(s[rest].sum(), 1.0))
    pi = np.where(mask, 1.0, np.where(rest, pr, 0.0))
    r = _result("B6", "pi", pi, float((pi * s).sum()), B,
                {"rho": rho, "n_alert_units": int(alert.sum()), "n_triggered": int(trig.sum()),
                 "n_triggered_kept": int(len(kept)), "triggered_cost": used, "pi_leftover": pr,
                 "window_s": window_us / 1e6, "scored_on": "D estimands only"})
    r["mask_triggered"] = mask
    return r


# ------------------------------------------------------------------------------------------ #
def b8_uniform(T, rho: float, phi0: float = 0.0) -> dict:
    """Closed form without controller: constant π = max(φ0, B̄E / Σ_pay s)."""
    s = np.asarray(T.s, dtype=np.float64)
    pay = np.asarray(T.pay, dtype=bool)
    _, B = budget(T, rho)
    p = min(1.0, max(phi0, B / s[pay].sum()))
    pi = np.where(pay, p, 0.0)
    return _result("B8", "pi", pi, float((pi * s).sum()), B, {"rho": rho, "pi": p})


def b10_size(T, rho: float, phi0: float = 0.0, iters: int = 100) -> dict:
    """Closed form without controller: π = max(φ0, min(1, λ √s)), λ by bisection (expected spend B̄E)."""
    s = np.asarray(T.s, dtype=np.float64)
    pay = np.asarray(T.pay, dtype=bool)
    _, B = budget(T, rho)
    sp = s[pay]
    f = lambda ll: float(np.sum(np.maximum(phi0, np.minimum(1.0, math.exp(ll) * np.sqrt(sp))) * sp))  # noqa: E731
    lo, hi = -40.0, 10.0
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        if f(m) > B:
            hi = m
        else:
            lo = m
    lam = math.exp(lo)
    pi = np.zeros(T.n_units)
    pi[pay] = np.maximum(phi0, np.minimum(1.0, lam * np.sqrt(sp)))
    return _result("B10", "pi", pi, float((pi * s).sum()), B, {"rho": rho, "lam": lam})


def b15_packet_sampling(T, rho: float, n_pkts: np.ndarray) -> dict:
    """1:N systematic packet sampling over payload units' packets (random per-epoch offset, unkeyed):
    N = Σ_pay s / (B̄E); per packet P = 1/N; per unit P(>= 1 packet) ≈ 1 − (1 − 1/N)^{n_i}; expected cost B̄E."""
    s = np.asarray(T.s, dtype=np.float64)
    pay = np.asarray(T.pay, dtype=bool)
    _, B = budget(T, rho)
    N = max(1.0, s[pay].sum() / B)
    n = np.asarray(n_pkts, dtype=np.float64)
    p_any = np.where(pay, 1.0 - (1.0 - 1.0 / N) ** n, 0.0)
    r = _result("B15", "pi", np.where(pay, 1.0 / N, 0.0), float(s[pay].sum() / N), B,
                {"rho": rho, "N": N, "estimator": "N x retained hit packets", "scored_on": "D estimands only"})
    r["p_unit_any_packet"] = p_any
    return r
