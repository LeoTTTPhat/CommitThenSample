"""Synthetic incident generator for tests and Monte-Carlo checks (no real data).

Score model for candidate e of incident k (higher = more likely a true link):

    s_{k,e} = b_k + eps_{k,e} + mu * 1{true} + a * mu * 1{AMB} + delta_c * 1{e in decoy cluster c}
              (+ v-unit random effect in chain mode)

* b_k = tau * T_shift_df            shared incident-level shift (heavy-tailed for finite df):
                                    within-incident dependence that breaks unit exchangeability
* eps ~ N(0, 1) for true/AMB; nulls ~ t_null_df (heavy-tailed nulls) or N(0, 1) (df = inf)
* decoy clusters: Poisson(decoy_rate) clusters per incident, size U{lo..hi}, one shared shift
  delta_c ~ N(decoy_shift, decoy_shift_sd^2) per cluster (correlated attack-like nulls)
* hallucination tail: each null independently, w.p. halluc_frac, is drawn from the TRUE-link law
* benign incidents (w.p. benign_frac) have r = 0; attack incidents r = 1 + Poisson(r_mean - 1)
* N_k = max(n_min, round(n_median * exp(n_sigma Z))) (lognormal, independent of the labels)

With Gaussian noise and no decoys/hallucinations, the within-incident AUC is Phi(mu / sqrt(2))
(0.9993 / 0.983 / 0.921 at mu = 4.5 / 3 / 2); ``mu_for_auc`` inverts that, and
``eval.gates.within_incident_auc`` measures the realised value (decoys lower it).

Chain mode (``chains=True``) builds units in time order with true chains c_0 -> ... -> c_L
(one unit per episode, so each hop is a transition whose enabling pair is (c_i, c_{i+1})),
optional hidden intermediates (-> hidden transitions + bridge candidates with Y = 1), optional
r_{C,T} < 1 (enabling pairs missing from C), transitive AMB pairs (c_i, c_{i+2}), null
candidates = random earlier predecessors (<= preds_per_unit per unit), decoy units whose
out-links (incl. links into attack units) get the decoy shift, devices, and GT paths.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from scipy.stats import norm

from ..certify.elond import LOOK_EPOCHS, StreamIncident
from ..certify.losses import Incident


@dataclass(frozen=True)
class SynthConfig:
    n_median: float = 400.0
    n_sigma: float = 0.5
    n_min: int = 30
    r_mean: float = 4.0
    benign_frac: float = 0.10
    mu: float = 3.0
    sigma_true: float = 1.0
    tau: float = 0.5
    shift_df: float = 3.0
    null_df: float = math.inf
    decoy_rate: float = 1.0
    decoy_size: tuple[int, int] = (5, 20)
    decoy_shift: float = 1.5
    decoy_shift_sd: float = 0.5
    halluc_frac: float = 0.0
    amb_per_true: float = 0.5
    amb_shift_frac: float = 0.6
    # chain mode
    chains: bool = False
    chain_len: tuple[int, int] = (1, 5)
    hidden_frac: float = 0.0
    rc_miss: float = 0.0
    unit_effect_sd: float = 0.0
    preds_per_unit: int = 10

    def with_(self, **kw) -> "SynthConfig":
        return replace(self, **kw)


def mu_for_auc(auc: float, sigma_true: float = 1.0) -> float:
    """mu giving within-incident AUC = Phi(mu / sqrt(1 + sigma_true^2)) in the Gaussian model."""
    return float(norm.ppf(auc) * math.sqrt(1.0 + sigma_true ** 2))


def _t(rng, df: float, size=None):
    return rng.standard_normal(size) if not np.isfinite(df) else rng.standard_t(df, size)


def _n_and_r(rng, cfg: SynthConfig) -> tuple[int, int, bool]:
    N = max(cfg.n_min, int(round(cfg.n_median * math.exp(cfg.n_sigma * rng.standard_normal()))))
    benign = bool(rng.random() < cfg.benign_frac)
    r = 0 if benign else 1 + int(rng.poisson(max(cfg.r_mean - 1.0, 0.0)))
    return N, min(r, max(N // 4, 1)) if not benign else 0, benign


def _decoy_clusters(rng, cfg: SynthConfig, n_null: int) -> tuple[list[np.ndarray], np.ndarray]:
    n_cl = int(rng.poisson(cfg.decoy_rate)) if cfg.decoy_rate > 0 else 0
    clusters, shifts = [], []
    free = rng.permutation(n_null)
    pos = 0
    for _ in range(n_cl):
        size = int(rng.integers(cfg.decoy_size[0], cfg.decoy_size[1] + 1))
        size = min(size, n_null - pos)
        if size <= 0:
            break
        clusters.append(free[pos: pos + size])
        pos += size
        shifts.append(cfg.decoy_shift + cfg.decoy_shift_sd * rng.standard_normal())
    return clusters, np.asarray(shifts)


def synth_incident(rng, cfg: SynthConfig, incident_id: str) -> Incident:
    if cfg.chains:
        return _synth_chain_incident(rng, cfg, incident_id)
    N, r, benign = _n_and_r(rng, cfg)
    if benign:
        n_amb = 0
    elif cfg.amb_per_true <= 1:
        n_amb = int(rng.binomial(r, cfg.amb_per_true))
    else:
        n_amb = int(rng.poisson(r * cfg.amb_per_true))
    n_amb = min(n_amb, max(N - r - 1, 0))
    n_null = N - r - n_amb
    b = cfg.tau * float(_t(rng, cfg.shift_df))
    s_true = b + cfg.mu + cfg.sigma_true * rng.standard_normal(r)
    s_amb = b + cfg.amb_shift_frac * cfg.mu + rng.standard_normal(n_amb)
    s_null = b + _t(rng, cfg.null_df, n_null)
    decoy = np.zeros(n_null, dtype=bool)
    clusters, shifts = _decoy_clusters(rng, cfg, n_null)
    for c, d in zip(clusters, shifts):
        s_null[c] += d
        decoy[c] = True
    if cfg.halluc_frac > 0:
        h = rng.random(n_null) < cfg.halluc_frac
        s_null[h] = b + cfg.mu + cfg.sigma_true * rng.standard_normal(int(h.sum()))
    scores = np.concatenate([s_true, s_amb, s_null])
    y = np.concatenate([np.ones(r, np.int8), -np.ones(n_amb, np.int8), np.zeros(n_null, np.int8)])
    dflag = np.concatenate([np.zeros(r + n_amb, bool), decoy])
    trans = np.concatenate([np.arange(r), -np.ones(n_amb + n_null, np.int64)])
    perm = rng.permutation(N)
    return Incident(incident_id=incident_id, scores=scores[perm], y=y[perm], trans=trans[perm].astype(np.int64),
                    decoy=dflag[perm], n_trans=r, n_trans_obs=r, n_trans_hid=0, n_elink=r, n_bridges=0,
                    strata={"benign_only": benign, "n_decoy_clusters": len(clusters),
                            "decoy_rich": len(clusters) >= 2, "shift": b})


def _synth_chain_incident(rng, cfg: SynthConfig, incident_id: str) -> Incident:
    N_target, r_target, benign = _n_and_r(rng, cfg)
    # ---- chains (episode = one unit)
    chains: list[int] = []
    tot = 0
    while not benign and tot < r_target:
        L = int(rng.integers(cfg.chain_len[0], cfg.chain_len[1] + 1))
        chains.append(L)
        tot += L
    n_chain_units = sum(L + 1 for L in chains)
    P = max(1, cfg.preds_per_unit)
    n_units = max(n_chain_units + 10, int(round(N_target / max(P / 2.0, 1.0))) + 1)
    ts = np.sort(rng.uniform(0.0, 3600.0, n_units))
    ts += np.arange(n_units) * 1e-6                       # strictly increasing
    chain_pos = np.sort(rng.choice(n_units, size=n_chain_units, replace=False)) if n_chain_units else np.empty(0, np.int64)
    rng.shuffle(chain_pos)
    units_of_chain, pos = [], 0
    for L in chains:
        units_of_chain.append(np.sort(chain_pos[pos: pos + L + 1]))
        pos += L + 1
    hidden = np.zeros(n_units, dtype=bool)
    for cu in units_of_chain:
        for i in range(1, len(cu) - 1):
            if rng.random() < cfg.hidden_frac:
                hidden[cu[i]] = True
    attack_unit = np.zeros(n_units, dtype=bool)
    if n_chain_units:
        attack_unit[chain_pos] = True
    # ---- labelled pairs
    pairs: dict[tuple[int, int], dict] = {}
    gt_paths, n_trans, n_hid, n_bridges = [], 0, 0, 0
    tid = 0
    for cu in units_of_chain:
        path = []
        for i in range(len(cu) - 1):
            a, b_ = int(cu[i]), int(cu[i + 1])
            n_trans += 1
            if hidden[a] or hidden[b_]:
                n_hid += 1
                path.append(-1)
                continue
            if rng.random() < cfg.rc_miss:
                path.append(-1)
                tid += 1
                continue
            pairs[(a, b_)] = {"y": 1, "trans": tid, "bridge": False}
            path.append((a, b_))
            tid += 1
        # bridges over hidden runs
        obs = [int(x) for x in cu if not hidden[x]]
        full = [int(x) for x in cu]
        for i in range(len(obs) - 1):
            a, b_ = obs[i], obs[i + 1]
            if full.index(b_) - full.index(a) > 1:
                n_bridges += 1
                pairs[(a, b_)] = {"y": 1, "trans": -1, "bridge": True}
        # transitive AMB pairs through an observed intermediate
        for i in range(len(full) - 2):
            a, m, b_ = full[i], full[i + 1], full[i + 2]
            if hidden[a] or hidden[m] or hidden[b_]:
                continue
            if (a, b_) not in pairs and rng.random() < min(1.0, cfg.amb_per_true):
                pairs[(a, b_)] = {"y": -1, "trans": -1, "bridge": False}
        gt_paths.append(path)
    # ---- decoy units
    observed = np.flatnonzero(~hidden)
    null_units = observed[~attack_unit[observed]]
    n_cl = int(rng.poisson(cfg.decoy_rate)) if cfg.decoy_rate > 0 else 0
    decoy_units = rng.choice(null_units[null_units < n_units - 1], size=min(n_cl, max(len(null_units) - 1, 0)),
                             replace=False) if len(null_units) > 1 else np.empty(0, np.int64)
    decoy_unit = np.zeros(n_units, dtype=bool)
    decoy_unit[decoy_units] = True
    decoy_shift = {int(d): cfg.decoy_shift + cfg.decoy_shift_sd * rng.standard_normal() for d in decoy_units}
    for d in decoy_units:
        later_attack = [int(x) for x in np.flatnonzero(attack_unit & ~hidden) if x > d]
        size = int(rng.integers(cfg.decoy_size[0], cfg.decoy_size[1] + 1))
        for x in rng.permutation(later_attack)[: max(1, size // 4)] if later_attack else []:
            if (int(d), int(x)) not in pairs:
                pairs[(int(d), int(x))] = {"y": 0, "trans": -1, "bridge": False}
    # ---- null candidates: random earlier predecessors
    obs_sorted = observed
    for idx in range(1, len(obs_sorted)):
        v = int(obs_sorted[idx])
        k = int(min(idx, rng.integers(1, P + 1)))
        preds = rng.choice(obs_sorted[:idx], size=k, replace=False)
        for u in preds:
            key = (int(u), v)
            if key not in pairs:
                pairs[key] = {"y": 0, "trans": -1, "bridge": False}
    # ---- arrays
    keys = sorted(pairs)
    u = np.array([k[0] for k in keys], dtype=np.int64)
    v = np.array([k[1] for k in keys], dtype=np.int64)
    y = np.array([pairs[k]["y"] for k in keys], dtype=np.int8)
    trans = np.array([pairs[k]["trans"] for k in keys], dtype=np.int64)
    bridge = np.array([pairs[k]["bridge"] for k in keys], dtype=bool)
    N = len(keys)
    b = cfg.tau * float(_t(rng, cfg.shift_df))
    eff = cfg.unit_effect_sd * rng.standard_normal(n_units) if cfg.unit_effect_sd > 0 else np.zeros(n_units)
    s = np.empty(N)
    tr = y == 1
    am = y == -1
    nu = y == 0
    s[tr] = b + cfg.mu + cfg.sigma_true * rng.standard_normal(int(tr.sum()))
    s[am] = b + cfg.amb_shift_frac * cfg.mu + rng.standard_normal(int(am.sum()))
    s[nu] = b + _t(rng, cfg.null_df, int(nu.sum()))
    if cfg.halluc_frac > 0:
        h = nu & (rng.random(N) < cfg.halluc_frac)
        s[h] = b + cfg.mu + cfg.sigma_true * rng.standard_normal(int(h.sum()))
    for d, sh in decoy_shift.items():
        m = nu & (u == d)
        s[m] += sh
    s += eff[v]
    cid = {k: i for i, k in enumerate(keys)}
    gt = [[cid[x] if isinstance(x, tuple) else -1 for x in path] for path in gt_paths]
    # devices: each chain on its own campaign devices; null units on a shared pool
    dev = rng.integers(0, max(5, n_units // 4), size=n_units) + 1000
    camp = []
    for ci, cu in enumerate(units_of_chain):
        for j, x in enumerate(cu):
            dev[x] = ci * 100 + (j % 3)
            camp.append(int(dev[x]))
    perm = rng.permutation(N)
    inv = np.empty(N, dtype=np.int64)
    inv[perm] = np.arange(N)
    gt = [[int(inv[c]) if c >= 0 else -1 for c in path] for path in gt]
    n_obs = n_trans - n_hid
    return Incident(incident_id=incident_id, scores=s[perm], y=y[perm], trans=trans[perm], bridge=bridge[perm],
                    u=u[perm], v=v[perm], unit_ts=ts, decoy_unit=decoy_unit, unit_device=dev,
                    campaign_devices=sorted(set(camp)), gt_paths=gt, n_trans=n_trans, n_trans_obs=n_obs,
                    n_trans_hid=n_hid, n_elink=n_obs + n_bridges, n_bridges=n_bridges,
                    strata={"benign_only": benign, "n_decoy_clusters": int(len(decoy_units)),
                            "decoy_rich": int(len(decoy_units)) >= 2,
                            "short_chain": bool(chains) and max(chains) <= 2, "shift": b})


def synth_pool(n: int, seed: int, cfg: SynthConfig = SynthConfig(), prefix: str = "syn") -> list[Incident]:
    """n i.i.d. incidents (A1 holds by construction)."""
    rng = np.random.default_rng(seed)
    return [synth_incident(rng, cfg, f"{prefix}{seed}_{i:05d}") for i in range(n)]


def synth_null_pool(n_incidents: int, seed: int, cfg: SynthConfig = SynthConfig()) -> np.ndarray:
    """S_0: null scores (Y != 1) of independent 'training' incidents of the same generator."""
    incs = synth_pool(n_incidents, seed, cfg, prefix="train")
    return np.concatenate([np.asarray(i.scores)[np.asarray(i.y) != 1] for i in incs])


def synth_training(n_incidents: int, seed: int, cfg: SynthConfig = SynthConfig()) -> tuple[np.ndarray, np.ndarray]:
    """(scores, y) of independent training incidents (e.g. to fit the B7 isotonic map)."""
    incs = synth_pool(n_incidents, seed, cfg, prefix="train")
    return np.concatenate([i.scores for i in incs]), np.concatenate([i.y for i in incs])


# --------------------------------------------------------------------------- Example 1 (§4.5)
def example1_f(L: int, n: int, alpha: float, K: int) -> int:
    a1 = alpha - (1 - alpha) / K
    return int(min(n, math.floor(a1 * n * L / (1 - a1) + 1e-12)))


def example1_incident(s0_sorted: np.ndarray, L: int = 5, n: int = 20, f: int = 10, n_null_units: int = 200,
                      n_null_links: int = 400, seed: int = 0, incident_id: str = "ex1") -> Incident:
    """Example 1: n disjoint true chains r_i -> v_{i,2} -> ... -> v_{i,L+1}; in f of them a decoy root d_i
    with a false link d_i -> v_{i,2} scoring above the true first hop; all nL true + f decoy links score
    above max(S_0) (one BH step-up jump); N_0 null links among null units with scores in the S_0 bulk."""
    rng = np.random.default_rng(seed)
    top = float(np.max(s0_sorted))
    units = 0
    u, v, s, y = [], [], [], []
    for i in range(n):
        d = None
        if i < f:
            d = units
            units += 1
        chain = list(range(units, units + L + 1))
        units += L + 1
        for h in range(L):
            u.append(chain[h]); v.append(chain[h + 1]); y.append(1)
            s.append(top + 1.0 + 0.001 * (i * L + h))
        if d is not None:
            first_hop = s[-L]
            u.append(d); v.append(chain[1]); y.append(0); s.append(first_hop + 0.5)
    base = units
    units += n_null_units
    q = rng.uniform(0.05, 0.95, n_null_links)
    null_scores = np.quantile(s0_sorted, q)
    seen = set()
    while len(seen) < n_null_links:
        a, b = sorted(rng.choice(n_null_units, size=2, replace=False))
        seen.add((base + int(a), base + int(b)))
    for (a, b), sc in zip(sorted(seen), null_scores):
        u.append(a); v.append(b); y.append(0); s.append(float(sc))
    y = np.asarray(y, dtype=np.int8)
    gt_paths = []
    pos = 0
    for i in range(n):
        gt_paths.append(list(range(pos, pos + L)))
        pos += L + (1 if i < f else 0)
    trans = np.where(y == 1, np.cumsum(y == 1) - 1, -1).astype(np.int64)
    return Incident(incident_id=incident_id, scores=np.asarray(s), y=y, trans=trans,
                    u=np.asarray(u, np.int64), v=np.asarray(v, np.int64), gt_paths=gt_paths,
                    n_trans=n * L, n_trans_obs=n * L, n_trans_hid=0, n_elink=n * L, n_bridges=0)


# --------------------------------------------------------------------------- streaming
def synth_stream_pool(n: int, seed: int, cfg: SynthConfig = SynthConfig(), true_decay: float = 0.6,
                      n_looks: int = len(LOOK_EPOCHS)) -> list[StreamIncident]:
    """Incidents with a look index per candidate: nulls arrive in proportion to the look interval
    lengths, true links earlier (P(look = l) ~ true_decay^l)."""
    rng = np.random.default_rng(seed)
    ep = np.asarray(LOOK_EPOCHS[:n_looks], dtype=np.float64)
    lens = np.diff(np.concatenate(([0.0], ep)))
    p_null = lens / lens.sum()
    p_true = true_decay ** np.arange(n_looks)
    p_true /= p_true.sum()
    out = []
    for i in range(n):
        inc = synth_incident(rng, cfg.with_(chains=False), f"st{seed}_{i:05d}")
        y = np.asarray(inc.y)
        look = np.where(y == 1, rng.choice(n_looks, size=y.size, p=p_true),
                        rng.choice(n_looks, size=y.size, p=p_null)).astype(np.int64)
        out.append(StreamIncident(inc.incident_id, np.asarray(inc.scores), y, look))
    return out
