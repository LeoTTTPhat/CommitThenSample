"""Monte-Carlo engine (DESIGN §6.5, Alg. 8).

Static designs: one sequential controller pass (expected cost) per (stream, design) -> pi per unit
(``static_pass``). MC (coverage / widths / adaptive arm / A11): rep chunks run an epoch loop that is
vectorised over the chunk's reps:

  per epoch e:
    adaptive designs: pi_e <- Design(e; controller state, boost set, flood guard) for all payload units
    hit units only:   U <- Philox(rep, e);  Z <- U < pi;  X <- Z f / pi;  update Qhat, V, W[K], S, Dcap, Vhat
    A11 variants:     apply the withholding rule (X' = X 1[not in W]) / the invalid "compliant only" estimand
    e-processes at the truth: miss- |= M-(Qhat - Q) >= 1/alpha_side ; miss+ |= M+(Q - Qhat) >= 1/alpha_side
    adaptive:         realised cost = sum_hit Z s + N(sum_nonhit pi s, sum_nonhit pi(1-pi) s^2); D, Lam, boost
    report epochs (every `report_every` epochs of the window and its last epoch), first `width_reps` reps:
                      invert L (running max L*), L+ = max(L*, S), U+ = min(Qhat + r+, Dcap), Wald -> rows

Randomness: the selection uniforms of epoch e and rep r are EXACTLY numpy's
``Generator(Philox(key=SeedSequence([stream_id, design_id, r]).generate_state(2, uint64),
counter=[0, e, 0, 0])).random(n_hits_e)`` (vectorised re-implementation of Philox4x64-10, verified
bit-exact in tests). Other purposes use counter word 2: 1 = realised-cost normal, 2 = Dcap normals,
3 = unit-level withholding, 4 = epoch-level withholding. Results of a rep do not depend on chunking.

Uniforms are drawn only for hit units H = union of the estimands' hits (plus boost/reactor trigger
queries, plus every eligible unit of estimands with <= MCSpec.exact_elig_max eligible units, so that
Dcap is exact where it binds); everything else enters through exact expectations or normal
approximations (realised cost; the non-hit part of Dcap, clipped so that Dcap >= Q holds surely).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import norm

from ..design.boost import BoostState
from ..design.controller import HorizonController
from ..design.floor import device_floor_cap, equalised_floor
from ..design.probs import committed_floor, committed_mix, design_pi, n_lams, quantise_utility, sqrt_eff
from ..estimation.cs import (bernstein_lower, bernstein_lower_miss, bernstein_upper, bernstein_upper_miss,
                              exact_pb_bounds, lcs_radius, nm_log_eproc, nm_radius, psi, ucs_grid, ucs_radius)
from .tables import StreamTables

__all__ = ["philox4x64", "rep_keys", "philox_epoch_uniforms", "stable_id", "Estimand", "DesignSpec",
           "MCSpec", "ReactiveSilence", "static_pass", "prepare", "simulate_chunk", "run_mc",
           "WITHHOLD_RULES", "calibrated_design", "run_dir"]

# --------------------------------------------------------------------------------------------- #
# Philox4x64-10 (vectorised; bit-exact with numpy.random.Philox)
# --------------------------------------------------------------------------------------------- #
_M32 = np.uint64(0xFFFFFFFF)
_S32 = np.uint64(32)
_S11 = np.uint64(11)
_PM0 = np.uint64(0xD2E7470EE14C6C93)
_PM1 = np.uint64(0xCA5A826395121157)
_W0 = np.uint64(0x9E3779B97F4A7C15)
_W1 = np.uint64(0xBB67AE8584CAA73B)
_TWO_M53 = 1.0 / 9007199254740992.0


def _mulhilo(a, m):
    m_lo, m_hi = m & _M32, m >> _S32
    a_lo, a_hi = a & _M32, a >> _S32
    p0 = a_lo * m_lo
    p1 = a_lo * m_hi
    p2 = a_hi * m_lo
    mid = (p0 >> _S32) + (p1 & _M32) + (p2 & _M32)
    hi = a_hi * m_hi + (p1 >> _S32) + (p2 >> _S32) + (mid >> _S32)
    return hi, a * m


def philox4x64(c0, c1, c2, c3, k0, k1, rounds: int = 10):
    """Philox4x64-R block function on uint64 arrays (broadcasting); returns the 4 output words."""
    with np.errstate(over="ignore"):
        for r in range(rounds):
            if r:
                k0 = k0 + _W0
                k1 = k1 + _W1
            hi0, lo0 = _mulhilo(c0, _PM0)
            hi1, lo1 = _mulhilo(c2, _PM1)
            c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return c0, c1, c2, c3


def rep_keys(stream_id: int, design_id: int, reps) -> np.ndarray:
    """(R, 2) uint64 Philox keys = SeedSequence([stream_id, design_id, rep]).generate_state(2, uint64)."""
    return np.array([np.random.SeedSequence([int(stream_id), int(design_id), int(r)]).generate_state(2, np.uint64)
                     for r in np.asarray(reps).ravel()], dtype=np.uint64).reshape(-1, 2)


def philox_epoch_uniforms(keys, n_per_epoch, word2: int = 0, open_interval: bool = False) -> np.ndarray:
    """Uniforms laid out by epoch: columns [cum_e, cum_e + n_e) of the result are
    Generator(Philox(key=keys[r], counter=[0, e, word2, 0])).random(n_e) (bit-exact for open=False).
    open_interval=True maps to (0, 1) via ((x >> 11) + 0.5) 2^-53 (for ndtri)."""
    keys = np.asarray(keys, dtype=np.uint64)
    n = np.asarray(n_per_epoch, dtype=np.int64)
    E = len(n)
    nb = (n + 3) // 4
    NB = int(nb.sum())
    R = keys.shape[0]
    total = int(n.sum())
    if total == 0:
        return np.zeros((R, 0))
    bstart = np.cumsum(nb) - nb
    blk_epoch = np.repeat(np.arange(E, dtype=np.uint64), nb)
    blk_local = (np.arange(NB) - np.repeat(bstart, nb)).astype(np.uint64)
    c0 = (blk_local + np.uint64(1))[None, :]
    c1 = blk_epoch[None, :]
    c2 = np.full((1, NB), word2, dtype=np.uint64)
    c3 = np.zeros((1, NB), dtype=np.uint64)
    out = philox4x64(c0, c1, c2, c3, keys[:, 0:1], keys[:, 1:2])
    raw = np.stack([np.broadcast_to(o, (R, NB)) for o in out], axis=-1).reshape(R, 4 * NB)
    nstart = np.cumsum(n) - n
    pos = np.repeat(4 * bstart, n) + (np.arange(total) - np.repeat(nstart, n))
    x = (raw[:, pos] >> _S11).astype(np.float64)
    if open_interval:
        x += 0.5
    return x * _TWO_M53


def stable_id(name: str, bits: int = 31) -> int:
    """Stable integer id of a name (BLAKE2b; used for stream/design ids in seeds)."""
    h = hashlib.blake2b(str(name).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & ((1 << bits) - 1)


# --------------------------------------------------------------------------------------------- #
# specs
# --------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Estimand:
    """(q, A, window): query column q restricted to devices A and epochs [e_start, e_end)."""
    name: str
    q: int | str
    devices: tuple | None = None
    e_start: int = 0
    e_end: int | None = None
    kind: str | None = None          # default: the query's kind
    V_ref: float | None = None       # LCS grid centre; default: E[V] at window end under the reference design
    bbar: float | None = None        # default: the query's max b over the stream (pre-registered)
    family_size: int = 1             # alpha/|F| per statement
    ucs: bool | None = None          # compute the UCS (default: indicator estimands only)
    V_ref_ft: float | None = None    # rev1 fixed-time variant 'fth': eta centre tuned to the horizon (default V_ref)


@dataclass(frozen=True)
class DesignSpec:
    name: str
    design_id: int
    phi0: float
    Bbar: float
    lam_ref: tuple                   # component references; mixture = (lam_A, lam_S)
    kind: str = "mixture"            # mixture | A | S | uniform
    util: str | None = "U5"
    tau_mix: float = 0.2
    controller: str = "expected"     # expected | realised | per_epoch | none
    kappa: float = 1.0
    Wc: float = 30.0
    L: float = math.log(16)
    quantise_lam: bool = True
    boost: bool = False
    kappa_b: float = 5.0
    W_b: int = 20
    boost_queries: tuple | None = None   # queries whose retained hits trigger the boost (None = all)
    flood_guard: bool = False
    Dmax_mult: float = 200.0
    gamma: float = 0.3
    rho: float = 0.05
    floor_mode: str = "constant"     # constant | equalised

    @property
    def adaptive(self) -> bool:
        return self.controller == "realised" or self.boost

    def phi_min(self) -> float:
        return self.phi0 / 2.0 if self.floor_mode == "equalised" else self.phi0


@dataclass(frozen=True)
class ReactiveSilence:
    """Population-adaptive adversary: after a retained hit (of `queries`, default any) on one of
    `devices`, that device's hit units vanish (are not committed) for the next `silence` epochs."""
    devices: tuple
    silence: int = 20
    queries: tuple | None = None


WITHHOLD_RULES = ("void_unretained", "void_retained", "withhold_half", "void_random")


@dataclass(frozen=True)
class MCSpec:
    reps: int = 1000
    alpha: float = 0.05
    K_lcs: int = 16
    lcs_decades: float = 2.0
    lcs_kind: str = "grid"           # grid | normal_mixture
    ucs_n_low: int = 15
    ucs_decades: float = 4.0
    ucs_w0: float = 0.5
    ucs_eta0_frac: float = 0.999
    report_every: int = 30
    width_reps: int = 200
    withhold: tuple = ()             # A11 rules (subset of WITHHOLD_RULES)
    compliant_only: bool = True      # also run the invalid "compliant epochs only" estimand per rule
    void_random_p: float = 0.3
    withhold_p: float = 0.5
    stream_id: int = 0
    rep_offset: int = 0
    chunk: int | None = None
    dcap_mode: str = "normal"        # normal | expected  (non-hit part of Dcap)
    exact_elig_max: int = 5000       # estimands with <= this many eligible units: exact Z for all of them
    reactor: ReactiveSilence | None = None
    # rev1 (review R1-M3): valid FIXED-TIME comparators at the final epoch of each window (cov columns
    # Lft1/Uft1/ft1_lo/ft1_hi, fth, ftb, ftx; see _ft_final). Off by default: existing outputs are unchanged.
    fixed_time: bool = False
    ft_exact_kmax: int = 1000        # exact Poisson-binomial bounds for indicator estimands with <= this many hits


# --------------------------------------------------------------------------------------------- #
# static pass (expected-cost controller) and calibration helpers
# --------------------------------------------------------------------------------------------- #
def _epoch_floor(T: StreamTables, design: DesignSpec):
    """Per-payload-unit constant/equalised floor (static part) and the epoch payload pointers."""
    pay_idx = np.nonzero(np.asarray(T.pay, dtype=bool))[0]
    E = T.n_epochs
    ep = np.asarray(T.epoch)[pay_idx]
    pptr = np.searchsorted(ep, np.arange(E + 1)).astype(np.int64)
    if design.floor_mode == "constant":
        phi = np.full(len(pay_idx), committed_floor(design.phi0))      # tau_phi / 2^64
    elif design.floor_mode == "equalised":
        s_all = np.asarray(T.s, dtype=float)
        B_e = design.rho * np.bincount(np.asarray(T.epoch), weights=s_all, minlength=E)
        C_e = np.bincount(ep, weights=s_all[pay_idx], minlength=E)
        phi = equalised_floor(design.gamma, B_e, C_e, design.phi0)[ep]
    else:
        raise ValueError(design.floor_mode)
    return pay_idx, pptr, phi


def _lams_tuple(lam, C):
    lam = np.asarray(lam, dtype=float)
    return [lam[..., c:c + 1] for c in range(C)]


def static_pass(T: StreamTables, design: DesignSpec, cache_path=None):
    """Sequential expected-cost controller pass (no boost, no flood guard) -> dict with
    pi_pay (pi of every payload unit, in payload order), pay_idx, lam (E, C), cost (E,), D (E,)."""
    if cache_path is not None and Path(cache_path).exists():
        z = np.load(cache_path)
        return {k: z[k] for k in z.files}
    pay_idx, pptr, phi = _epoch_floor(T, design)
    s = np.asarray(T.s, dtype=float)[pay_idx]
    sq = sqrt_eff(s)
    a = quantise_utility(np.asarray(T.util(design.util), dtype=float)[pay_idx])
    tmix = committed_mix(design.tau_mix)
    E = T.n_epochs
    C = n_lams(design.kind)
    lam_ref = np.asarray(design.lam_ref, dtype=float)
    ctrl = HorizonController(lam_ref, design.Bbar, design.kappa, design.Wc, design.L, "expected",
                             quantise=design.quantise_lam)
    pi = np.zeros(len(pay_idx))
    lam_hist = np.zeros((E, C))
    cost = np.zeros(E)
    D = np.zeros(E)
    s_all_e = np.bincount(np.asarray(T.epoch), weights=np.asarray(T.s, dtype=float), minlength=E)
    for e in range(E):
        a0, a1 = pptr[e], pptr[e + 1]
        if design.controller == "per_epoch":
            lam = _waterfill_epoch(design, a[a0:a1], s[a0:a1], phi[a0:a1], design.rho * s_all_e[e], lam_ref,
                                   sq[a0:a1])
        elif design.controller == "none":
            lam = lam_ref
        else:
            lam = np.atleast_1d(ctrl.lam())
        p = design_pi(design.kind, list(lam), a[a0:a1], s[a0:a1], phi[a0:a1], tmix, sqrt_s=sq[a0:a1])
        pi[a0:a1] = p
        c = float(np.sum(p * s[a0:a1]))
        lam_hist[e] = lam
        cost[e] = c
        ctrl.update(c)
        D[e] = float(ctrl.D)
    out = dict(pi_pay=pi, pay_idx=pay_idx, lam=lam_hist, cost=cost, D=D)
    if cache_path is not None:
        from ..utils.io import atomic_save_npz
        atomic_save_npz(Path(cache_path), **out)
    return out


def _waterfill_epoch(design, a, s, phi, B_e, lam_ref, sq=None):
    """End-of-epoch water-filling ablation (E7 budget mode): each component spends B_e in epoch e."""
    comps = {"mixture": ("A", "S"), "A": ("A",), "S": ("S",), "uniform": ("uniform",)}[design.kind]
    lams = []
    for c, lr in zip(comps, np.atleast_1d(lam_ref)):
        kind = c
        lo, hi = math.log(lr) - 40, math.log(lr) + 40
        f = lambda ll: float(np.sum(design_pi(kind, [math.exp(ll)], a, s, phi, 0.0, sqrt_s=sq) * s))  # noqa: E731
        if f(lo) >= B_e:
            lams.append(math.exp(lo))
            continue
        if f(hi) <= B_e:
            lams.append(math.exp(hi))
            continue
        for _ in range(80):
            m = 0.5 * (lo + hi)
            if f(m) > B_e:
                hi = m
            else:
                lo = m
        lams.append(math.exp(0.5 * (lo + hi)))
    return np.array(lams)


def calibrated_design(T: StreamTables, name: str, rho: float = 0.05, gamma: float = 0.3, kind: str = "mixture",
                      util: str | None = "U5", train_epochs=None, design_id: int | None = None, **kw) -> DesignSpec:
    """Convenience: Bbar = rho * full-capture bytes / E, phi0 = gamma rho sum_all s / sum_pay s and the
    component lam_refs by bisection, all from the training epochs (default: the whole stream)."""
    from ..design.controller import calibrate_lambda
    from ..design.floor import phi0_from_training
    E = T.n_epochs
    ep = np.asarray(T.epoch)
    s_all = np.asarray(T.s, dtype=float)
    pay = np.asarray(T.pay, dtype=bool)
    m = np.ones(len(ep), bool) if train_epochs is None else np.isin(ep, np.asarray(train_epochs))
    n_tr = E if train_epochs is None else len(np.unique(np.asarray(train_epochs)))
    Bbar = rho * s_all.sum() / E
    phi0 = phi0_from_training(gamma, rho, s_all[m].sum(), s_all[m & pay].sum())
    a = np.asarray(T.util(util), dtype=float)
    comps = {"mixture": ("A", "S"), "A": ("A",), "S": ("S",), "uniform": ("uniform",)}[kind]
    Bbar_tr = rho * s_all[m].sum() / n_tr
    lam_ref = tuple(calibrate_lambda(s_all[m & pay], a[m & pay], Bbar_tr, phi0, n_tr, component=c) for c in comps)
    return DesignSpec(name=name, design_id=stable_id(name) if design_id is None else design_id, phi0=phi0, Bbar=Bbar,
                      lam_ref=lam_ref, kind=kind, util=util, gamma=gamma, rho=rho, **kw)


# --------------------------------------------------------------------------------------------- #
# preparation (per process; cached)
# --------------------------------------------------------------------------------------------- #
class _Prep:
    pass


def prepare(T: StreamTables, design: DesignSpec, estimands, mc: MCSpec) -> _Prep:
    P = _Prep()
    E = T.n_epochs
    N = T.n_units
    P.E = E
    P.T = T
    nE = len(estimands)
    P.nE = nE
    pay_idx, pptr, phi_static = _epoch_floor(T, design)
    P.pay_idx, P.pptr, P.phi_static = pay_idx, pptr, phi_static
    P.s_pay = np.asarray(T.s, dtype=float)[pay_idx]
    P.sqrt_pay = sqrt_eff(P.s_pay)                                   # protocol isqrt rounding (intpi)
    P.a_pay = quantise_utility(np.asarray(T.util(design.util), dtype=float)[pay_idx])   # a_fix 2^-16
    P.tau_mix_eff = committed_mix(design.tau_mix)                    # t16 / 2^16
    P.dev_pay = np.asarray(T.device)[pay_idx].astype(np.int64)
    upos = np.full(N, -1, dtype=np.int64)
    upos[pay_idx] = np.arange(len(pay_idx))
    ep_all = np.asarray(T.epoch).astype(np.int64)
    dev_all = np.asarray(T.device).astype(np.int64)

    # ---- estimand masks and hit pairs ------------------------------------------------------- #
    qidx = [T.query_index(es.q) for es in estimands]
    kinds = [es.kind or T.query_kinds[q] for es, q in zip(estimands, qidx)]
    e0 = np.array([es.e_start for es in estimands], dtype=np.int64)
    e1 = np.array([E if es.e_end is None else min(es.e_end, E) for es in estimands], dtype=np.int64)
    if np.any(e1 <= e0):
        raise ValueError("empty estimand window")
    P.e0, P.e1, P.kinds, P.qidx = e0, e1, kinds, qidx
    P.names = [es.name for es in estimands]

    def unit_mask(k, units):
        es = estimands[k]
        m = (ep_all[units] >= e0[k]) & (ep_all[units] < e1[k])
        if es.devices is not None:
            m &= np.isin(dev_all[units], np.asarray(es.devices))
        return m

    hit_lists = []
    for k in range(nE):
        u, f = T.hits(qidx[k])
        m = unit_mask(k, u)
        hit_lists.append((u[m], f[m]))
    extra_q = set()
    if design.boost:
        extra_q |= set(range(T.n_queries) if design.boost_queries is None else [T.query_index(q) for q in design.boost_queries])
    reactor = mc.reactor
    if reactor is not None:
        extra_q |= set(range(T.n_queries) if reactor.queries is None else [T.query_index(q) for q in reactor.queries])
    extra_units = [T.hits(q)[0] for q in sorted(extra_q)]
    # small eligible sets (rare-protocol estimands, where Dcap binds): exact draws for EVERY eligible unit
    for k in range(nE):
        bq = T.b_col(qidx[k])[pay_idx]
        el = pay_idx[(bq > 0) & unit_mask(k, pay_idx)]
        if 0 < len(el) <= mc.exact_elig_max:
            extra_units.append(el)
    H = np.unique(np.concatenate([h[0] for h in hit_lists] + extra_units + [np.zeros(0, np.int64)])).astype(np.int64)
    if np.any(upos[H] < 0):
        raise ValueError("hit units must be payload units")
    nH = len(H)
    P.H, P.nH = H, nH
    P.hpos = upos[H]                                   # payload position
    P.h_epoch = ep_all[H]
    P.hptr = np.searchsorted(P.h_epoch, np.arange(E + 1)).astype(np.int64)
    P.h_local = P.hpos - pptr[P.h_epoch]               # position within the epoch's payload block
    P.h_dev = dev_all[H]
    P.h_s = np.asarray(T.s, dtype=float)[H]
    F = np.zeros((nH, nE))
    for k, (u, f) in enumerate(hit_lists):
        F[np.searchsorted(H, u), k] = f
    P.F = F
    # eligibility bounds over payload units, masked per estimand
    Bpay = np.zeros((len(pay_idx), nE), dtype=np.float64)
    for k in range(nE):
        bq = T.b_col(qidx[k])[pay_idx]
        m = unit_mask(k, pay_idx)
        Bpay[:, k] = np.where(m, bq, 0.0)
    P.Bpay = Bpay
    P.Bh = Bpay[P.hpos]
    if np.any(F > P.Bh * (1 + 1e-9) + 1e-9):
        raise ValueError("f > b for some hit unit")
    # per-epoch sum of b over all (eligible) payload units of the estimand, and of non-H units
    cs = np.vstack([np.zeros((1, nE)), np.cumsum(Bpay, axis=0)])
    P.Bsum = cs[pptr[1:]] - cs[pptr[:-1]]                         # (E, nE)
    isH = np.zeros(len(pay_idx), bool)
    isH[P.hpos] = True
    P.isH_pay = isH
    Bn = np.where(isH[:, None], 0.0, Bpay)
    P.Bn = Bn
    csn = np.vstack([np.zeros((1, nE)), np.cumsum(Bn, axis=0)])
    P.Bn_sum = csn[pptr[1:]] - csn[pptr[:-1]]                     # (E, nE)
    # trigger flags for boost / reactor
    def trig(qs):
        t = np.zeros(nH, bool)
        for q in qs:
            u = T.hits(q)[0]
            t[np.searchsorted(H, u[np.isin(u, H)])] = True
        return t
    P.boost_trig = trig(range(T.n_queries) if design.boost_queries is None else [T.query_index(q) for q in design.boost_queries]) if design.boost else None
    if reactor is not None:
        rq = range(T.n_queries) if reactor.queries is None else [T.query_index(q) for q in reactor.queries]
        P.react_trig = trig(rq)
        rdev = np.asarray(reactor.devices, dtype=np.int64)
        P.react_dev = rdev
        P.h_react = np.searchsorted(rdev, P.h_dev)                 # index into rdev (valid where member)
        P.h_is_react = np.isin(P.h_dev, rdev)
        if np.any(np.diff(rdev) <= 0):
            raise ValueError("reactor devices must be sorted unique")
    # W pairs (hit unit, estimand) with f > 0, grouped by epoch
    pj, pk = np.nonzero(F > 0)
    o = np.lexsort((pk, pj))
    pj, pk = pj[o], pk[o]
    P.pj, P.pk = pj, pk
    P.pf = F[pj, pk]
    P.pb = P.Bh[pj, pk]
    P.pptr_pairs = np.searchsorted(P.h_epoch[pj], np.arange(E + 1)).astype(np.int64)

    # ---- grids --------------------------------------------------------------------------- #
    alpha_side = np.array([mc.alpha / (2.0 * es.family_size) for es in estimands])
    P.alpha_side = alpha_side
    P.log_thr = np.log(1.0 / alpha_side)
    P.z = norm.ppf(1.0 - alpha_side)
    bbar = np.array([es.bbar if es.bbar is not None else max(T.bbar(q), 1e-300) for es, q in zip(estimands, qidx)])
    P.cbar = bbar / design.phi_min()
    P.ucs_on = np.array([(es.ucs if es.ucs is not None else (kd == "indicator")) for es, kd in zip(estimands, kinds)])
    etas_u, w_u = ucs_grid(P.cbar, n_low=mc.ucs_n_low, decades=mc.ucs_decades, w0=mc.ucs_w0, eta0_frac=mc.ucs_eta0_frac)
    P.etas_u, P.logw_u = etas_u, np.log(w_u)
    # reference design for V_ref and static pi
    ref = static_pass(T, replace(design, boost=False, flood_guard=False,
                                 controller="expected" if design.controller == "realised" else design.controller))
    P.pi_ref_pay = ref["pi_pay"]
    P.static_ref = ref
    pi_h_ref = P.pi_ref_pay[P.hpos]
    V_ref = np.array([es.V_ref if es.V_ref is not None else float(np.sum(F[:, k] ** 2 / pi_h_ref))
                      for k, es in enumerate(estimands)])
    V_ref = np.where(V_ref > 0, V_ref, 1.0)
    P.V_ref = V_ref
    # LCS grid per estimand: eta* = sqrt(2 log(1/alpha_side)/V_ref), K points over +-decades
    eta_star = np.sqrt(2.0 * np.log(1.0 / alpha_side) / V_ref)
    P.etas_l = eta_star[:, None] * 10.0 ** np.linspace(-mc.lcs_decades, mc.lcs_decades, mc.K_lcs)
    P.logw_l = np.log(np.full(mc.K_lcs, 1.0 / mc.K_lcs))
    if mc.lcs_kind == "normal_mixture":
        L_ = np.log(1.0 / alpha_side)
        P.nm_rho = V_ref / (2 * L_ + np.log(1 + 2 * L_))
    # rev1 fixed-time comparators: one pre-registered eta per estimand and variant (column 0 'ft1': the LCS grid
    # centre eta*(V_ref); column 1 'fth': eta*(V_ref_ft), tuned to the horizon); upper eta = min(eta0_frac/cbar, .)
    P.bbar = bbar
    P.ft_on = bool(mc.fixed_time)
    if P.ft_on:
        V_ref_ft = np.array([es.V_ref_ft if (es.V_ref_ft is not None and es.V_ref_ft > 0) else V_ref[k]
                             for k, es in enumerate(estimands)], dtype=float)
        P.V_ref_ft = V_ref_ft
        P.eta_ft_l = np.stack([eta_star, np.sqrt(2.0 * np.log(1.0 / alpha_side) / V_ref_ft)], axis=-1)   # (nE, 2)
        P.eta_ft_u = np.minimum(mc.ucs_eta0_frac / P.cbar[:, None], P.eta_ft_l)                           # (nE, 2)
        # Bernstein floor-variance bound: pi_i >= phi for every eligible unit (F); committed floor <= phi0
        phi_b = min(committed_floor(design.phi_min()), design.phi_min()) * (1.0 - 1e-12)
        P.ftb_phi = phi_b
        P.ftb_kappa = bbar * (1.0 - phi_b) / phi_b
        P.ftb_ok = reactor is None            # needs a population fixed in advance (Q deterministic)

    # ---- design state needs --------------------------------------------------------------- #
    P.per_rep = design.adaptive or design.flood_guard or (reactor is not None)
    P.pi_static_h = P.pi_ref_pay[P.hpos] if not P.per_rep else None
    if design.flood_guard:
        # static per-epoch cap structures
        P.cap_phi = np.zeros(len(pay_idx))
        P.cap_mask = np.zeros(len(pay_idx), bool)
        for e in range(E):
            a0, a1 = pptr[e], pptr[e + 1]
            if a1 > a0:
                ph, cm = device_floor_cap(design.phi0, design.Bbar, design.gamma, P.dev_pay[a0:a1], P.s_pay[a0:a1])
                P.cap_phi[a0:a1] = ph
                P.cap_mask[a0:a1] = cm
        csc = np.vstack([np.zeros((1, nE)), np.cumsum((Bpay > 0) & P.cap_mask[:, None], axis=0)])
        P.elig_capped = (csc[pptr[1:]] - csc[pptr[:-1]]) > 0        # (E, nE)
    # static W values per pair (static designs)
    if not P.per_rep:
        c = P.pb / pi_h_ref[pj]
        lam = P.etas_u[pk] * c[:, None]
        P.Wpair_static = psi(lam) * ((P.pf / P.pb) ** 2)[:, None]       # (npairs, Ku)
        if P.ft_on:
            P.Wpair_ft_static = psi(P.eta_ft_u[pk] * c[:, None]) * ((P.pf / P.pb) ** 2)[:, None]   # (npairs, 2)
    # rev1 exact worst-case Poisson-binomial tables (static design, population fixed in advance, indicators)
    P.ftx = None
    if P.ft_on and (not P.per_rep) and reactor is None:
        P.ftx = {}
        for k in range(nE):
            nh_k = len(hit_lists[k][0])
            if kinds[k] != "indicator" or nh_k > mc.ft_exact_kmax:
                continue
            el = Bpay[:, k] > 0
            P.ftx[k] = exact_pb_bounds(P.pi_ref_pay[el], nh_k, float(alpha_side[k]))
    # report epochs per estimand
    rep_ep = np.zeros((E, nE), bool)
    for k in range(nE):
        es_ = np.arange(e0[k], e1[k])
        rmask = ((es_ - e0[k] + 1) % mc.report_every == 0) | (es_ == e1[k] - 1)
        rep_ep[es_[rmask], k] = True
    P.report = rep_ep
    P.in_win = (np.arange(E)[:, None] >= e0[None, :]) & (np.arange(E)[:, None] < e1[None, :])
    P.is_last = np.arange(E)[:, None] == (e1[None, :] - 1)
    return P


# --------------------------------------------------------------------------------------------- #
# chunk simulation
# --------------------------------------------------------------------------------------------- #
FT_BOUNDS = ("Lft1", "Uft1", "Lfth", "Ufth", "Lftb", "Uftb", "Lftx", "Uftx")
FT_FLAGS = ("ft1_lo", "ft1_hi", "fth_lo", "fth_hi", "ftb_lo", "ftb_hi", "ftx_lo", "ftx_hi", "ftb_on", "ftx_on")


class _Var:
    """Statistics of one estimation variant (honest / X:=0 under a rule / compliant-only)."""

    def __init__(self, name, R, nE, Ku, Rw, ft: bool = False):
        self.name = name
        z = lambda: np.zeros((R, nE))  # noqa: E731
        self.Qhat, self.V, self.Vh, self.S, self.Dcap, self.Q = z(), z(), z(), z(), z(), z()
        self.DcapH, self.nonH, self.BnC = z(), z(), z()     # Dcap = DcapH - clip(nonH, 0, BnC)
        self.W = np.zeros((R, nE, Ku))
        self.ucs_void = np.zeros((R, nE), bool)
        b = lambda: np.zeros((R, nE), bool)  # noqa: E731
        self.lcs_tu, self.ucs_tu, self.wlo_tu, self.whi_tu = b(), b(), b(), b()
        self.lcs_ft, self.ucs_ft, self.wlo_ft, self.whi_ft = b(), b(), b(), b()
        self.final = {k: z() for k in ("Q", "Qhat", "V", "S", "Dcap", "Vh")}
        self.ucs_issued_final = b()
        self.Lstar = np.full((Rw, nE), -np.inf)
        if ft:   # rev1 fixed-time comparators (final epoch of each window)
            self.Wft = np.zeros((R, nE, 2))                 # W_e(eta) at the single upper eta of ft1 / fth
            self.floor_void = b()                           # an eligible unit below the floor (flood-guard cap)
            self.ft = {k: np.full((R, nE), np.nan) for k in FT_BOUNDS}
            self.ft.update({k: b() for k in FT_FLAGS})


def _lse_last(t):
    m = t.max(axis=-1)
    mm = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore", under="ignore", over="ignore", invalid="ignore"):
        return np.log(np.exp(t - mm[..., None]).sum(axis=-1)) + mm


def _ft_final(P, v, is_last, issued) -> None:
    """rev1 fixed-time comparators at the final epoch of each window (estimands with is_last), all reps.

    ft1 / fth  single-eta Chernoff LCB  L = Qhat - l/eta_L - eta_L V/2 (range-free; Prop. 1(b) + Markov) and Fan UCB
               U = Qhat + (l + W(eta_U))/eta_U (Prop. 1(c) + Markov); eta_L = eta*(V_ref) [ft1] or eta*(V_ref_ft)
               [fth, tuned to the horizon]; eta_U = min(eta0_frac/cbar, eta_L).
    ftb        Bernstein with the floor-based design-variance bound kappa Q (cs.bernstein_*); only for populations
               fixed in advance (P.ftb_ok); lower voided where an eligible unit is below the floor (flood guard).
    ftx        exact worst-case Poisson-binomial bounds of K = S (cs.exact_pb_bounds): static designs, indicators.
    Upper bounds follow the UCS issuance rules (``issued``: indicator, no capped/withheld eligible unit).
    Miss flags are evaluated at the truth (closed inequalities, i.e. ties count as misses)."""
    ks = np.nonzero(is_last)[0]
    Qh, V, Q, S = v.Qhat[:, ks], v.V[:, ks], v.Q[:, ks], v.S[:, ks]
    lt = P.log_thr[ks][None, :]
    a_s = P.alpha_side[ks][None, :]
    iss = issued[:, ks]
    o = v.ft
    for j, tag in enumerate(("ft1", "fth")):
        eL = P.eta_ft_l[ks, j][None, :]
        eU = P.eta_ft_u[ks, j][None, :]
        W = v.Wft[:, ks, j]
        o[f"L{tag}"][:, ks] = Qh - lt / eL - 0.5 * eL * V
        o[f"{tag}_lo"][:, ks] = eL * (Qh - Q) - 0.5 * eL * eL * V >= lt
        o[f"U{tag}"][:, ks] = np.where(iss, Qh + (lt + W) / eU, np.nan)
        o[f"{tag}_hi"][:, ks] = iss & (eU * (Q - Qh) - W >= lt)
    if P.ftb_ok:
        kap = P.ftb_kappa[ks][None, :]
        bb = P.bbar[ks][None, :]
        okb = ~v.floor_void[:, ks]
        o["ftb_on"][:, ks] = okb
        o["Lftb"][:, ks] = np.where(okb, bernstein_lower(Qh, kap, kap, a_s), np.nan)
        o["ftb_lo"][:, ks] = okb & bernstein_lower_miss(Qh, Q, kap, kap, a_s)
        o["Uftb"][:, ks] = np.where(iss, bernstein_upper(Qh, kap, bb, a_s), np.nan)
        o["ftb_hi"][:, ks] = iss & bernstein_upper_miss(Qh, Q, kap, bb, a_s)
    if P.ftx:
        for jj, k in enumerate(ks):
            tab = P.ftx.get(int(k))
            if tab is None:
                continue
            Lt, Ut = tab
            K = np.clip(np.rint(S[:, jj]).astype(np.int64), 0, len(Lt) - 1)
            Lx = Lt[K].astype(float)
            Ux = Ut[K].astype(float)
            o["ftx_on"][:, k] = True
            o["Lftx"][:, k] = Lx
            o["ftx_lo"][:, k] = Lx > Q[:, jj]
            o["Uftx"][:, k] = np.where(iss[:, jj], Ux, np.nan)
            o["ftx_hi"][:, k] = iss[:, jj] & (Ux < Q[:, jj])


def simulate_chunk(P: _Prep, design: DesignSpec, mc: MCSpec, reps) -> dict:
    """Simulate the reps (global ids) of one chunk; returns dict of column arrays (cov, width, ctrl)."""
    reps = np.asarray(reps, dtype=np.int64)
    R = len(reps)
    E, nE, nH = P.E, P.nE, P.nH
    Ku = P.etas_u.shape[-1]
    keys = rep_keys(mc.stream_id, design.design_id, reps)
    nH_e = np.diff(P.hptr)
    U = philox_epoch_uniforms(keys, nH_e, word2=0)
    w_mask = reps < mc.width_reps
    Rw = int(w_mask.sum())
    widx = np.nonzero(w_mask)[0]
    realised = design.controller == "realised"
    if realised:
        Ncost = ndtri(philox_epoch_uniforms(keys, np.ones(E, np.int64), word2=1, open_interval=True))
    if mc.dcap_mode == "normal":
        Ncap = ndtri(philox_epoch_uniforms(keys, np.full(E, nE, np.int64), word2=2, open_interval=True)).reshape(R, E, nE)
    rules = tuple(mc.withhold)
    for r_ in rules:
        if r_ not in WITHHOLD_RULES:
            raise ValueError(f"unknown withholding rule {r_}")
    if "withhold_half" in rules:
        Uw = philox_epoch_uniforms(keys, nH_e, word2=3)
    if "void_random" in rules:
        Uv = philox_epoch_uniforms(keys, np.ones(E, np.int64), word2=4)
    ft_on = bool(getattr(P, "ft_on", False))
    variants = [_Var("honest", R, nE, Ku, Rw, ft=ft_on)]
    for r_ in rules:
        variants.append(_Var(f"x0:{r_}", R, nE, Ku, Rw, ft=ft_on))
        if mc.compliant_only:
            variants.append(_Var(f"comp:{r_}", R, nE, Ku, Rw, ft=ft_on))

    # design state
    C = n_lams(design.kind)
    ctrl = HorizonController(np.asarray(design.lam_ref, float), design.Bbar, design.kappa, design.Wc, design.L,
                             "realised" if realised else "expected", shape=(R,), quantise=design.quantise_lam)
    boost = BoostState(P.T.n_devices, (R,), design.kappa_b, design.W_b) if design.boost else None
    reactor = mc.reactor
    if reactor is not None:
        silent_until = np.full((R, len(P.react_dev)), -(10 ** 9), dtype=np.int64)
    n_guard = np.zeros(R)
    boosted_frac = np.zeros(R)
    n_ret_hits = np.zeros(R)
    rows = {k: [] for k in ("rep", "est", "variant", "epoch", "Q", "Qhat", "V", "S", "Dcap", "L", "Lstar",
                            "Lplus", "U", "Uplus", "ucs_issued", "wald_lo", "wald_hi")}
    etas_l, logw_l, etas_u, logw_u = P.etas_l, P.logw_l, P.etas_u, P.logw_u
    log_thr, z = P.log_thr, P.z
    use_nm = mc.lcs_kind == "normal_mixture"

    for e in range(E):
        a0, a1 = P.pptr[e], P.pptr[e + 1]
        h0, h1 = P.hptr[e], P.hptr[e + 1]
        nh = h1 - h0
        hl = P.h_local[h0:h1]
        # ---------------- design probabilities ----------------
        if P.per_rep:
            s_e = P.s_pay[a0:a1]
            phi = P.phi_static[a0:a1][None, :]
            guard = None
            if design.flood_guard:
                guard = ctrl.flood_guard_active(design.Dmax_mult)            # (R,)
                n_guard += guard
                phi = np.where(guard[:, None], P.cap_phi[a0:a1][None, :], phi)
            lam = ctrl.lam()                                                 # (R, C)
            bm = None
            if boost is not None:
                boost.begin_epoch(e)
                bm = boost.multiplier(P.dev_pay[a0:a1])
                if a1 > a0:
                    boosted_frac += (bm > 1).mean(axis=1)
            pi_e = design_pi(design.kind, _lams_tuple(lam, C), P.a_pay[a0:a1][None, :], s_e[None, :], phi,
                             P.tau_mix_eff, boost=bm, sqrt_s=P.sqrt_pay[a0:a1][None, :])   # (R, n_e)
            pi_h = pi_e[:, hl]
        else:
            pi_h = P.pi_static_h[h0:h1][None, :]
        # ---------------- population (reactive adversary) ----------------
        active = None
        if reactor is not None and nh:
            hr = P.h_is_react[h0:h1]
            if hr.any():
                idx = np.clip(P.h_react[h0:h1], 0, len(P.react_dev) - 1)
                act_r = e > silent_until[:, idx]                                  # (R, nh)
                active = np.where(hr[None, :], act_r, True)
        # ---------------- selection ----------------
        Z = U[:, h0:h1] < pi_h
        if active is not None:
            Z &= active
        F_e = P.F[h0:h1]
        B_e = P.Bh[h0:h1]
        with np.errstate(divide="ignore"):
            inv = 1.0 / pi_h
        # truth increment
        if active is not None:
            dQ = active.astype(float) @ F_e
            Bsum_act = P.Bsum[e][None, :] - (~active).astype(float) @ B_e
        else:
            dQ = np.broadcast_to(F_e.sum(axis=0), (R, nE))
            Bsum_act = np.broadcast_to(P.Bsum[e], (R, nE))
        # non-H eligible retained (Dcap reduction): normal increments, UNclipped per epoch (clipping tiny
        # per-epoch counts biases them upwards); only the cumulative sum is clipped to [0, sum b]
        if P.per_rep:
            mu = pi_e @ P.Bn[a0:a1]
            var = (pi_e * (1 - pi_e)) @ (P.Bn[a0:a1] ** 2)
        else:
            pis = P.pi_ref_pay[a0:a1]
            mu = np.broadcast_to(pis @ P.Bn[a0:a1], (R, nE))
            var = np.broadcast_to((pis * (1 - pis)) @ (P.Bn[a0:a1] ** 2), (R, nE))
        if mc.dcap_mode == "normal":
            nonH_inc = mu + np.sqrt(var) * Ncap[:, e, :]
        else:
            nonH_inc = np.broadcast_to(mu, (R, nE))
        dBn = np.broadcast_to(P.Bn_sum[e], (R, nE))
        # W pair values
        q0, q1 = P.pptr_pairs[e], P.pptr_pairs[e + 1]
        pj = P.pj[q0:q1] - h0
        pk = P.pk[q0:q1]
        # UCS penalty increments only for RETAINED (rep, pair) entries (X = 0 contributes exactly 0)
        if q1 > q0:
            w_rr, w_pp = np.nonzero(Z[:, pj])
            if P.per_rep:
                c = P.pb[q0:q1][w_pp] / pi_h[w_rr, pj[w_pp]]
                w_vals = psi(etas_u[pk[w_pp]] * c[:, None]) * ((P.pf[q0:q1][w_pp] / P.pb[q0:q1][w_pp]) ** 2)[:, None]
                if ft_on:   # W_e at the single upper eta of the fixed-time comparators (ft1, fth)
                    w_ft_vals = psi(P.eta_ft_u[pk[w_pp]] * c[:, None]) * \
                        ((P.pf[q0:q1][w_pp] / P.pb[q0:q1][w_pp]) ** 2)[:, None]
            else:
                w_vals = P.Wpair_static[q0 + w_pp]
                if ft_on:
                    w_ft_vals = P.Wpair_ft_static[q0 + w_pp]
            w_pk = pk[w_pp]
            w_pj = pj[w_pp]
        F2 = F_e * F_e
        has_hit_est = (F_e > 0).astype(float)
        in_win = P.in_win[e]
        is_last = P.is_last[e]
        # ---------------- withholding rules ----------------
        rule_out = {}
        if rules:
            nret = Z.astype(float) @ has_hit_est                             # retained hits per est
            nact = (active.astype(float) @ has_hit_est) if active is not None else np.broadcast_to(has_hit_est.sum(0), (R, nE))
            for r_ in rules:
                wu = None; ve = None
                if r_ == "void_unretained":
                    ve = (nact > 0) & (nret == 0)
                elif r_ == "void_retained":
                    ve = nret > 0
                elif r_ == "withhold_half":
                    wu = Z & (Uw[:, h0:h1] < mc.withhold_p)
                elif r_ == "void_random":
                    ve = np.broadcast_to((Uv[:, e] < mc.void_random_p)[:, None], (R, nE))
                rule_out[r_] = (wu, ve)
        # ---------------- per-variant updates ----------------
        for v in variants:
            if v.name == "honest":
                Zv, keep_est, keep_tr, uvoid = Z, None, None, None
            else:
                kindv, r_ = v.name.split(":")
                wu, ve = rule_out[r_]
                Zv = Z & ~wu if wu is not None else Z
                if kindv == "x0":
                    keep_est = None if ve is None else ~ve
                    keep_tr = None
                    uvoid = np.zeros((R, nE), bool)
                    if ve is not None:
                        uvoid |= ve
                    if wu is not None:
                        uvoid |= (wu.astype(float) @ (B_e > 0).astype(float)) > 0
                else:  # comp: estimand restricted to compliant epochs
                    if ve is None:
                        ve = np.broadcast_to(wu.any(axis=1)[:, None], (R, nE))
                    Zv = Z
                    keep_est = ~ve
                    keep_tr = keep_est
                    uvoid = None
            Zf = Zv.astype(float)
            ZI = Zf * inv if nh else Zf
            dQh = ZI @ F_e
            dV = (ZI * inv) @ F2
            dVh = (ZI * inv * (1.0 - pi_h)) @ F2
            dS = Zf @ F_e
            dRed = Zf @ (B_e - F_e)
            dNon, dBnv = nonH_inc, dBn
            dW = None
            dWft = None
            if q1 > q0 and len(w_rr):
                dW = np.zeros((R, nE, Ku))
                if ft_on:
                    dWft = np.zeros((R, nE, 2))
                if Zv is Z:
                    np.add.at(dW, (w_rr, w_pk), w_vals)
                    if ft_on:
                        np.add.at(dWft, (w_rr, w_pk), w_ft_vals)
                else:
                    km = Zv[w_rr, w_pj]
                    np.add.at(dW, (w_rr[km], w_pk[km]), w_vals[km])
                    if ft_on:
                        np.add.at(dWft, (w_rr[km], w_pk[km]), w_ft_vals[km])
            dQt = dQ
            if keep_est is not None:
                ke = keep_est.astype(float)
                dQh, dV, dVh, dS = dQh * ke, dV * ke, dVh * ke, dS * ke
                dRed = dRed * ke
                dNon = dNon * ke
                if dW is not None:
                    dW = dW * ke[..., None]
                if dWft is not None:
                    dWft = dWft * ke[..., None]
            dCap = Bsum_act - dRed
            if keep_tr is not None:
                kt = keep_tr.astype(float)
                dQt = dQt * kt
                dCap = dCap * kt
                dNon = dNon * kt
                dBnv = dBnv * kt
            iw = in_win[None, :]
            v.Qhat += dQh * iw
            v.V += dV * iw
            v.Vh += dVh * iw
            v.S += dS * iw
            v.DcapH += dCap * iw
            v.nonH += dNon * iw
            v.BnC += dBnv * iw
            v.Dcap = v.DcapH - np.clip(v.nonH, 0.0, v.BnC)
            v.Q += dQt * iw
            if dW is not None:
                v.W += dW * iw[..., None]
            if dWft is not None:
                v.Wft += dWft * iw[..., None]
            if uvoid is not None:
                v.ucs_void |= uvoid & iw
            if design.flood_guard and guard is not None:
                v.ucs_void |= guard[:, None] & P.elig_capped[e][None, :] & iw
                if ft_on:
                    v.floor_void |= guard[:, None] & P.elig_capped[e][None, :] & iw
            # ---------------- e-processes at the truth ----------------
            diff = v.Qhat - v.Q
            if use_nm:
                lm = nm_log_eproc(diff, v.V, P.nm_rho[None, :])
            else:
                t = logw_l + etas_l[None] * diff[..., None] - 0.5 * etas_l[None] ** 2 * v.V[..., None]
                lm = _lse_last(t)
            ml = (lm >= log_thr) & iw
            tu = logw_u + etas_u[None] * (-diff)[..., None] - v.W
            mu_ = _lse_last(tu)
            mu_ok = P.ucs_on[None, :] & ~v.ucs_void
            mup = (mu_ >= log_thr) & iw & mu_ok
            sd = np.sqrt(v.Vh)
            wlo = (v.Q < v.Qhat - z * sd) & iw
            whi = (v.Q > v.Qhat + z * sd) & iw
            v.lcs_tu |= ml
            v.ucs_tu |= mup
            v.wlo_tu |= wlo
            v.whi_tu |= whi
            if is_last.any():
                il = is_last[None, :]
                v.lcs_ft |= ml & il
                v.ucs_ft |= mup & il
                v.wlo_ft |= wlo & il
                v.whi_ft |= whi & il
                for nm_, arr in (("Q", v.Q), ("Qhat", v.Qhat), ("V", v.V), ("S", v.S), ("Dcap", v.Dcap), ("Vh", v.Vh)):
                    v.final[nm_] = np.where(il, arr, v.final[nm_])
                v.ucs_issued_final = np.where(il, mu_ok, v.ucs_issued_final)
                if ft_on:
                    _ft_final(P, v, is_last, mu_ok)
            # ---------------- report epochs (widths) ----------------
            rk = P.report[e]
            if Rw and rk.any():
                _report(P, v, e, rk, widx, reps, rows, use_nm)
        # ---------------- design state updates ----------------
        if P.per_rep:
            if realised:
                hs = P.h_s[h0:h1]
                cost_h = Z.astype(float) @ hs                                # exact part (Z = 0 if inactive)
                mu_c = pi_e @ s_e - (pi_h @ hs)                              # non-hit units: normal approx.
                var_c = (pi_e * (1 - pi_e)) @ (s_e * s_e) - (pi_h * (1 - pi_h)) @ (hs * hs)
                cost = cost_h + np.maximum(mu_c + np.sqrt(np.maximum(var_c, 0.0)) * Ncost[:, e], 0.0)
            else:
                cost = pi_e @ s_e
                if active is not None:
                    hs = P.h_s[h0:h1]
                    cost = cost - (pi_h * (~active)) @ hs
            ctrl.update(cost)
            if boost is not None:
                trig = P.boost_trig[h0:h1]
                boost.record_units(Z, trig, P.h_dev[h0:h1])
            if reactor is not None and nh:
                hr = P.h_is_react[h0:h1] & P.react_trig[h0:h1]
                if hr.any():
                    zr = Z[:, hr]
                    ridx = P.h_react[h0:h1][hr]
                    for jj in range(zr.shape[1]):
                        hitr = zr[:, jj]
                        silent_until[hitr, ridx[jj]] = e + reactor.silence
        else:
            ctrl.update(np.full(R, P.static_ref["cost"][e]))
        n_ret_hits += Z[:, (F_e > 0).any(axis=1)].sum(axis=1) if nh else 0

    # ---------------- collect ----------------
    cov = {k: [] for k in ("rep", "est", "variant", "lcs_tu", "lcs_ft", "ucs_tu", "ucs_ft", "ucs_issued",
                           "wald_lo_tu", "wald_lo_ft", "wald_hi_tu", "wald_hi_ft", "Q", "Qhat", "V", "S", "Dcap", "Vh")}
    for v in variants:
        rr = np.repeat(reps, nE)
        kk = np.tile(np.arange(nE), R)
        cov["rep"].append(rr)
        cov["est"].append(kk)
        cov["variant"].append(np.full(R * nE, v.name, dtype=object))
        for nm_, arr in (("lcs_tu", v.lcs_tu), ("lcs_ft", v.lcs_ft), ("ucs_tu", v.ucs_tu), ("ucs_ft", v.ucs_ft),
                         ("ucs_issued", v.ucs_issued_final), ("wald_lo_tu", v.wlo_tu), ("wald_lo_ft", v.wlo_ft),
                         ("wald_hi_tu", v.whi_tu), ("wald_hi_ft", v.whi_ft)):
            cov[nm_].append(arr.ravel())
        for nm_ in ("Q", "Qhat", "V", "S", "Dcap", "Vh"):
            cov[nm_].append(v.final[nm_].ravel())
        if ft_on:
            for nm_ in FT_BOUNDS + FT_FLAGS:
                cov.setdefault(nm_, []).append(v.ft[nm_].ravel())
    cov = {k: np.concatenate(v) for k, v in cov.items()}
    width = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in rows.items()}
    ctrl_rows = {"rep": reps, "spend_ratio": ctrl.spend_ratio(), "max_abs_D_over_B": ctrl.max_abs_D / design.Bbar,
                 "final_D_over_B": ctrl.D / design.Bbar, "guard_epochs": n_guard, "boosted_frac": boosted_frac / max(E, 1),
                 "retained_hits": n_ret_hits}
    return {"cov": cov, "width": width, "ctrl": ctrl_rows, "est_names": P.names}


def _report(P, v, e, rk, widx, reps, rows, use_nm):
    """Invert the CSs at a report epoch for the width reps and the estimands with rk True."""
    ks = np.nonzero(rk)[0]
    sub = np.ix_(widx, ks)
    Qhat, V, W, S, Dcap, Q, Vh = v.Qhat[sub], v.V[sub], v.W[widx][:, ks], v.S[sub], v.Dcap[sub], v.Q[sub], v.Vh[sub]
    if use_nm:
        r_lo = nm_radius(V, P.nm_rho[ks][None], P.log_thr[ks][None])
    else:
        r_lo = lcs_radius(V, P.etas_l[ks][None], P.logw_l, P.log_thr[ks][None])
    L = Qhat - r_lo
    v.Lstar[:, ks] = np.maximum(v.Lstar[:, ks], L)
    Lstar = v.Lstar[:, ks]
    Lplus = np.maximum(Lstar, S)
    r_up = ucs_radius(W, P.etas_u[ks][None], P.logw_u, P.log_thr[ks][None])
    Uu = Qhat + r_up
    issued = P.ucs_on[ks][None, :] & ~v.ucs_void[sub]
    Uplus = np.where(issued, np.minimum(Uu, Dcap), np.nan)
    sd = np.sqrt(Vh)
    zz = P.z[ks][None, :]
    n = Qhat.size
    rows["rep"].append(np.repeat(reps[widx], len(ks)))
    rows["est"].append(np.tile(ks, len(widx)))
    rows["variant"].append(np.full(n, v.name, dtype=object))
    rows["epoch"].append(np.full(n, e))
    for nm_, arr in (("Q", Q), ("Qhat", Qhat), ("V", V), ("S", S), ("Dcap", Dcap), ("L", L), ("Lstar", Lstar),
                     ("Lplus", Lplus), ("U", np.where(P.ucs_on[ks][None, :], Uu, np.nan)), ("Uplus", Uplus),
                     ("ucs_issued", issued), ("wald_lo", Qhat - zz * sd), ("wald_hi", Qhat + zz * sd)):
        rows[nm_].append(np.asarray(arr).ravel())


# --------------------------------------------------------------------------------------------- #
# runner (chunks, workers, resumable Parquet outputs)
# --------------------------------------------------------------------------------------------- #
_PREP_CACHE: dict = {}


def run_dir(exp: str, stream: str, cfg=None, design: DesignSpec | None = None, estimands=None,
            mc: MCSpec | None = None) -> Path:
    """results/<exp>/<stream>/<hash>/ with hash = config.cfg_hash of (cfg, design, estimands, mc)
    (BLAKE3 of canonical JSON + src hash, DESIGN §7.2). Pass it as run_mc(out_dir=...)."""
    from ..config import cfg_hash
    from ..utils.io import RESULTS
    payload = {"cfg": getattr(cfg, "raw", cfg), "design": asdict(design) if design is not None else None,
               "estimands": [asdict(e) for e in estimands] if estimands is not None else None,
               "mc": asdict(replace(mc, rep_offset=0, chunk=None)) if mc is not None else None}
    return RESULTS / exp / stream / cfg_hash(payload)


def _spec_key(T, design, estimands, mc):
    blob = json.dumps({"t": str(getattr(T, "path", None) or id(T)), "d": asdict(design),
                       "es": [asdict(e) for e in estimands], "mc": asdict(replace(mc, reps=0, rep_offset=0, chunk=None))},
                      sort_keys=True, default=str)
    return hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()


def _get_prep(T, design, estimands, mc):
    key = _spec_key(T, design, estimands, mc)
    P = _PREP_CACHE.get(key)
    if P is None:
        _PREP_CACHE.clear()
        P = prepare(T, design, estimands, mc)
        _PREP_CACHE[key] = P
    return P


def _to_table(d: dict, est_names=None):
    import pyarrow as pa
    cols = {}
    for k, v in d.items():
        v = np.asarray(v)
        if v.dtype == object:
            cols[k] = pa.array(v.astype(str))
        else:
            cols[k] = pa.array(v)
    if est_names is not None and "est" in d:
        cols["estimand"] = pa.array(np.asarray(est_names, dtype=object)[np.asarray(d["est"], dtype=np.int64)].astype(str)) \
            if len(d["est"]) else pa.array(np.zeros(0, dtype=str))
    return pa.table(cols)


def _chunk_job(args):
    tables_ref, design, estimands, mc, rep0, rep1, out_dir = args
    T = StreamTables.load(tables_ref) if isinstance(tables_ref, (str, Path)) else tables_ref
    P = _get_prep(T, design, estimands, mc)
    t0 = time.time()
    res = simulate_chunk(P, design, mc, np.arange(rep0, rep1))
    res["seconds"] = time.time() - t0
    if out_dir is not None:
        from ..utils.io import atomic_write_parquet
        out_dir = Path(out_dir)
        tag = f"rep{rep0:06d}_{rep1:06d}"
        atomic_write_parquet(out_dir / f"{tag}_width.parquet", _to_table(res["width"], res["est_names"]))
        atomic_write_parquet(out_dir / f"{tag}_ctrl.parquet", _to_table(res["ctrl"]))
        atomic_write_parquet(out_dir / f"{tag}_cov.parquet", _to_table(res["cov"], res["est_names"]))
        return {"rep0": rep0, "rep1": rep1, "seconds": res["seconds"], "out": str(out_dir / f"{tag}_cov.parquet")}
    return res


def run_mc(tables, design: DesignSpec, estimands, mc: MCSpec, out_dir=None, workers: int | None = None,
           chunk: int | None = None, check_mem: bool = True, verbose: bool = False):
    """Run reps [rep_offset, rep_offset + reps) in chunks.

    tables: a StreamTables (in memory; workers must be 1 unless it has a .path) or a path.
    out_dir: if given, writes <out_dir>/rep<a>_<b>_{cov,width,ctrl}.parquet per chunk (atomic; existing
    chunks are skipped -> resumable) and a DONE marker; returns the concatenated pandas DataFrames.
    Parallel chunks use a spawn-context process pool (<= workers processes)."""
    import pandas as pd
    from ..utils.mem import check_memory, scale
    from ..utils.mem import workers as mem_workers
    estimands = list(estimands)
    if chunk is None:
        chunk = mc.chunk or max(8, int(64 * scale()))
    if workers is None:
        workers = mem_workers(default=8)
    T = tables
    tref = tables
    if not isinstance(tables, (str, Path)):
        if workers > 1 and getattr(tables, "path", None) is not None:
            tref = str(tables.path)
        elif workers > 1:
            workers = 1
    else:
        T = StreamTables.load(tables)
    bounds = [(a, min(a + chunk, mc.rep_offset + mc.reps)) for a in range(mc.rep_offset, mc.rep_offset + mc.reps, chunk)]
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [b for b in bounds if not (out_dir / f"rep{b[0]:06d}_{b[1]:06d}_cov.parquet").exists()]
        from ..utils.io import atomic_write_json
        atomic_write_json(out_dir / "spec.json", {"design": asdict(design), "estimands": [asdict(e) for e in estimands],
                                                  "mc": asdict(mc), "stream": getattr(T, "name", None)})
    else:
        todo = bounds
    results = []
    t0 = time.time()
    last_check = [-1e9]

    def _mem():
        # check_memory() walks the process tree (~0.3 s on a loaded host): at most every 10 s
        if check_mem and time.time() - last_check[0] >= 10.0:
            check_memory()
            last_check[0] = time.time()

    jobs = [(tref, design, estimands, mc, a, b, out_dir) for a, b in todo]
    if workers <= 1 or len(jobs) <= 1:
        for j in jobs:
            results.append(_chunk_job(j))
            _mem()
            if verbose:
                print(f"[mc] chunk {j[4]}-{j[5]} done ({time.time() - t0:.1f}s)", flush=True)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(_chunk_job, j) for j in jobs]
            for f in as_completed(futs):
                results.append(f.result())
                _mem()
                if verbose:
                    print(f"[mc] {len(results)}/{len(jobs)} chunks ({time.time() - t0:.1f}s)", flush=True)
    elapsed = time.time() - t0
    if out_dir is not None:
        from ..utils.io import mark_done
        mark_done(out_dir, {"reps": mc.reps, "chunks": len(bounds), "seconds": elapsed})
        cov = pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("rep*_cov.parquet"))], ignore_index=True)
        width = pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("rep*_width.parquet"))], ignore_index=True)
        ctrl = pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("rep*_ctrl.parquet"))], ignore_index=True)
    else:
        names = [e.name for e in estimands]
        cov = pd.concat([_to_table(r["cov"], names).to_pandas() for r in results], ignore_index=True)
        width = pd.concat([_to_table(r["width"], names).to_pandas() for r in results], ignore_index=True)
        ctrl = pd.concat([_to_table(r["ctrl"]).to_pandas() for r in results], ignore_index=True)
    return {"cov": cov, "width": width, "ctrl": ctrl, "seconds": elapsed}
