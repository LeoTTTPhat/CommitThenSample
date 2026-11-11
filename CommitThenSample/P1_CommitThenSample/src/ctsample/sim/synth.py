"""Synthetic streams for tests and benchmarks (no real data).

* ``diurnal_bursty_counts`` / ``synth_traffic``: gateway-like traffic as in scripts/checks/ctrl.py:
  unit counts ~ Poisson(base (1 + amp sin(2 pi e / 2880)) exp(N(0, 0.3))), 1% burst epochs x20;
  sizes ~ round(lognormal(6.0, 1.6) + 80) bytes (integers, like pcap costs); sparse utility
  a = lognormal(0, 1.5) * Bernoulli(0.2).
* ``synth_stream``: the above plus sparse, heavy-tailed hit populations for a set of queries:
  hits concentrated on a few attacker devices (Zipf), indicator (b = f = 1) or count queries
  (b = packets, f = min(b, Pareto counts)); rare-protocol queries with small eligibility sets;
  an optional persistent C&C channel (one hit unit per epoch on device ``cnc_device``) for the
  population-adaptive adversary (sim.montecarlo.ReactiveSilence).
"""
from __future__ import annotations

import math

import numpy as np

from .tables import StreamTables

__all__ = ["diurnal_bursty_counts", "synth_traffic", "synth_stream", "QuerySpec"]

PERIOD = 2880  # epochs per day at T = 30 s


def diurnal_bursty_counts(E: int, rng, base: float = 200.0, amp: float = 0.6, sigma: float = 0.3,
                          burst_p: float = 0.01, burst_mult: float = 20.0, period: int = PERIOD, phase: float = 0.0):
    mu = base * (1 + amp * np.sin(2 * np.pi * (np.arange(E) + phase) / period)) * np.exp(rng.normal(0, sigma, E))
    burst = rng.random(E) < burst_p
    mu[burst] *= burst_mult
    return rng.poisson(mu)


def synth_traffic(E: int, seed: int = 0, base: float = 200.0, n_devices: int = 50, zero_payload_frac: float = 0.0,
                  size_mu: float = 6.0, size_sigma: float = 1.6, size_min: float = 80.0, util_sparsity: float = 0.2,
                  util_sigma: float = 1.5, **count_kw):
    """Unit-level traffic arrays sorted by epoch: dict(s, pay, epoch, device, a)."""
    rng = np.random.default_rng(seed)
    n = diurnal_bursty_counts(E, rng, base=base, **count_kw)
    N = int(n.sum())
    epoch = np.repeat(np.arange(E, dtype=np.int32), n)
    s = np.round(rng.lognormal(size_mu, size_sigma, N) + size_min)     # integer pcap bytes
    pay = rng.random(N) >= zero_payload_frac
    s = np.where(pay, s, np.minimum(s, 60.0 + rng.integers(0, 40, N)))   # zero-payload: small units
    a = rng.lognormal(0.0, util_sigma, N) * (rng.random(N) < util_sparsity)
    a = np.where(pay, a, 0.0)
    w = 1.0 / np.arange(1, n_devices + 1) ** 0.8                          # Zipf-ish device activity
    device = rng.choice(n_devices, size=N, p=w / w.sum()).astype(np.int32)
    return dict(s=s, pay=pay, epoch=epoch, device=device, a=a, n_per_epoch=n)


class QuerySpec:
    """Synthetic query: kind 'indicator' or 'count'; elig_frac of payload units eligible;
    n_hits target hit units on `n_att` attacker devices; count hits f = min(b, ceil(Pareto(alpha)))."""

    def __init__(self, name, kind="indicator", elig_frac=0.3, n_hits=200, n_att=3, pareto_alpha=1.3,
                 rare=False, burst_epochs=None):
        self.name, self.kind, self.elig_frac, self.n_hits = name, kind, elig_frac, n_hits
        self.n_att, self.pareto_alpha, self.rare, self.burst_epochs = n_att, pareto_alpha, rare, burst_epochs


def synth_stream(E: int = 870, seed: int = 0, base: float = 200.0, n_devices: int = 50, zero_payload_frac: float = 0.3,
                 queries=None, cnc_device: int | None = None, cnc_period: int = 1, util_signal: float = 1.5,
                 util_miss: float = 0.3, name: str = "synth", stream_id: int = 0, **traffic_kw) -> StreamTables:
    """Synthetic StreamTables with sparse heavy-tailed hit populations.

    Every query's hits lie on its attacker devices (drawn per query); hits are eligible payload units.
    If ``cnc_device`` is set, query "cnc" gets one hit unit per `cnc_period` epochs on that device
    (a dedicated payload unit is appended to the epoch: a persistent heartbeat channel)."""
    rng = np.random.default_rng(seed + 7919)
    tr = synth_traffic(E, seed=seed, base=base, n_devices=n_devices, zero_payload_frac=zero_payload_frac, **traffic_kw)
    s, pay, epoch, device, a = tr["s"], tr["pay"], tr["epoch"], tr["device"], tr["a"]
    if cnc_device is not None:
        eps = np.arange(0, E, cnc_period, dtype=np.int32)
        s = np.concatenate([s, np.round(rng.lognormal(5.0, 0.3, len(eps)) + 80)])
        pay = np.concatenate([pay, np.ones(len(eps), bool)])
        epoch = np.concatenate([epoch, eps])
        device = np.concatenate([device, np.full(len(eps), cnc_device, np.int32)])
        a = np.concatenate([a, rng.lognormal(0.0, 1.0, len(eps)) * (rng.random(len(eps)) < 0.5)])
        o = np.argsort(epoch, kind="stable")
        s, pay, epoch, device, a = s[o], pay[o], epoch[o], device[o], a[o]
        cnc_units = np.nonzero(o >= len(tr["s"]))[0]          # new positions of the appended heartbeat units
    N = len(s)
    if queries is None:
        queries = [QuerySpec("ind_common", "indicator", 0.3, 400, 3), QuerySpec("ind_rare", "indicator", 0.02, 60, 1, rare=True),
                   QuerySpec("cnt_heavy", "count", 0.3, 300, 3), QuerySpec("ind_sparse", "indicator", 0.3, 40, 2)]
    npk = np.maximum(1, np.round(s / 200.0)).astype(float)
    Q = len(queries) + (1 if cnc_device is not None else 0)
    b = np.zeros((N, Q), np.float32)
    hits, names, kinds = [], [], []
    is_hit_any = np.zeros(N, bool)
    pay_idx = np.nonzero(pay)[0]
    for qi, qs in enumerate(queries):
        elig = np.zeros(N, bool)
        elig[pay_idx] = rng.random(len(pay_idx)) < qs.elig_frac
        att = rng.choice(n_devices, size=min(qs.n_att, n_devices), replace=False)
        cand = np.nonzero(elig & np.isin(device, att) & pay)[0]
        if cnc_device is not None:
            cand = cand[device[cand] != cnc_device]
        k = min(qs.n_hits, len(cand))
        hu = np.sort(rng.choice(cand, size=k, replace=False)) if k else np.zeros(0, np.int64)
        if qs.kind == "indicator":
            b[elig, qi] = 1.0
            f = np.ones(k)
        else:
            b[elig, qi] = npk[elig]
            f = np.minimum(npk[hu], np.ceil(rng.pareto(qs.pareto_alpha, k) + 1.0))
        hits.append((hu, f))
        names.append(qs.name)
        kinds.append(qs.kind)
        is_hit_any[hu] = True
    if cnc_device is not None:
        b[cnc_units, Q - 1] = 1.0
        hits.append((cnc_units, np.ones(len(cnc_units))))
        names.append("cnc")
        kinds.append("indicator")
        is_hit_any[cnc_units] = True
    # informative but imperfect utility: hit units get a strong signal unless the model misses them
    sig = is_hit_any & (rng.random(N) >= util_miss)
    a5 = np.where(sig, rng.lognormal(util_signal, 1.0, N), a)
    a5 = np.where(pay, a5, 0.0)
    incident = np.full(N, -1, np.int32)
    for qi, (hu, _) in enumerate(hits):
        incident[hu] = np.where(incident[hu] < 0, qi, incident[hu])
    return StreamTables.from_arrays(s=s, pay=pay, epoch=epoch, device=device, b=b, hits=hits, query_names=names,
                                    query_kinds=kinds, label=(incident >= 0).astype(np.int16), incident=incident,
                                    utils={"U5": a5, "U0": np.where(pay, 1.0, 0.0), "sparse": a},
                                    name=name, stream_id=stream_id, n_devices=n_devices, n_epochs=E,
                                    extra={"cnc_device": cnc_device, "seed": seed})


def lognormal_mean(mu: float, sigma: float) -> float:
    return math.exp(mu + sigma * sigma / 2)


def check2_stream(E: int = 100, ne: int = 2000, seed: int = 1, n_hosts: int = 200, n_att: int = 5,
                  att_frac: float = 0.15, cnc: bool = True, sparse_hits: int = 0, name: str = "check2",
                  stream_id: int = 2) -> StreamTables:
    """Population of scripts/checks/check2.py in StreamTables form (+ optional persistent C&C channel).

    ne units per epoch, heavy-tailed sizes s = max(64, round(exp(N(log 300, 2)))), 200 hosts, 5 attack
    hosts; ~15% of attack-host flows are attack flows; query "ind" = rule-hit indicator (70% of attack
    flows, b = 1 on every unit); query "cnt" = count min(npk, Poisson(3)+1) on attack flows with
    b = npk = max(1, round(s/200)); utility "score" = clip(0.6 is_att + 0.25 U + 0.02, 0, 1) x b-scale.
    With cnc=True, host n_hosts (an extra device) sends one heartbeat unit per epoch; query "cnc"
    (indicator, b = 1 on its units only) - used with montecarlo.ReactiveSilence (goes silent 20 epochs
    after a retained hit). With sparse_hits > 0, query "sparse" (indicator, b = 1 everywhere) has that many
    hit flows on host n_att that the utility MISSES (score of a benign flow), so they sit near the floor:
    the regime where fixed-time Wald intervals fail."""
    rng = np.random.default_rng(seed)
    n = E * ne
    s = np.maximum(64, np.exp(rng.normal(np.log(300), 2.0, n))).round()
    host = rng.integers(0, n_hosts, n)
    is_att = np.isin(host, np.arange(n_att)) & (rng.random(n) < att_frac)
    npk = np.maximum(1, (s / 200).round())
    f_ind = (is_att & (rng.random(n) < 0.7)).astype(float)
    f_cnt = np.where(is_att, np.minimum(npk, rng.poisson(3, n) + 1), 0.0).astype(float)
    score = np.clip(0.6 * is_att + 0.25 * rng.random(n) + 0.02, 0, 1)
    epoch = np.repeat(np.arange(E, dtype=np.int32), ne)
    if cnc:
        s = np.concatenate([s, np.full(E, 400.0)])
        host = np.concatenate([host, np.full(E, n_hosts)])
        epoch = np.concatenate([epoch, np.arange(E, dtype=np.int32)])
        npk = np.concatenate([npk, np.full(E, 2.0)])
        f_ind = np.concatenate([f_ind, np.zeros(E)])
        f_cnt = np.concatenate([f_cnt, np.zeros(E)])
        score = np.concatenate([score, np.full(E, 0.05)])
        is_att = np.concatenate([is_att, np.zeros(E, bool)])
        o = np.argsort(epoch, kind="stable")
        is_cnc = np.zeros(len(s), bool)
        is_cnc[n:] = True
        s, host, epoch, npk, f_ind, f_cnt, score, is_att, is_cnc = (x[o] for x in (s, host, epoch, npk, f_ind, f_cnt, score, is_att, is_cnc))
    else:
        is_cnc = np.zeros(n, bool)
    N = len(s)
    b = np.stack([np.ones(N), npk, is_cnc.astype(float)], axis=1)
    hits = [(np.nonzero(f_ind > 0)[0], f_ind[f_ind > 0]), (np.nonzero(f_cnt > 0)[0], f_cnt[f_cnt > 0]),
            (np.nonzero(is_cnc)[0], np.ones(int(is_cnc.sum())))]
    names, kinds = ["ind", "cnt", "cnc"], ["indicator", "count", "indicator"]
    if not cnc:
        b, hits, names, kinds = b[:, :2], hits[:2], names[:2], kinds[:2]
    if sparse_hits:
        cand = np.nonzero((host == n_att) & ~is_att & ~is_cnc)[0]
        hu = np.sort(rng.choice(cand, size=min(sparse_hits, len(cand)), replace=False))
        b = np.concatenate([b, np.ones((N, 1))], axis=1)
        hits.append((hu, np.ones(len(hu))))
        names.append("sparse")
        kinds.append("indicator")
    return StreamTables.from_arrays(s=s, pay=np.ones(N, bool), epoch=epoch, device=host.astype(np.int32), b=b,
                                    hits=hits, query_names=names, query_kinds=kinds, label=is_att.astype(np.int16),
                                    incident=np.where(is_att, 0, -1).astype(np.int32),
                                    utils={"score": score, "U0": np.ones(N)}, name=name, stream_id=stream_id,
                                    n_devices=n_hosts + 1, n_epochs=E, extra={"seed": seed, "cnc_device": n_hosts if cnc else None})
