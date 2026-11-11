"""A6 — best-of-M grinding (DESIGN §6.3, Prop. 5(4)).

Without the timing rule (TR) the gateway can produce M candidate commitments AFTER β is known
(e.g. anchoring every K epochs, or no witness) and keep the one that retains the fewest target
units:  P(evidence retained) = (1 − (1 − π)^k)^M   (0.401 at M = 1, 0.0104 at M = 5 for
π = 0.05, k = 10). Under TR the witness cosigns one header per (G, e) and only before
t(r*_e) − δ_W − ε_β, so M = 1.

``simulate_grinding`` runs the attack with the REAL primitives (drand rounds, ECVRF, BLAKE3 PRF):
variant j of epoch e inserts j dummy zero-payload units at random positions and shifts the dummy
timestamps (re-ordering the canonical indices), which changes every L_i, index i, H(Hdr_e) and α_e,
hence the whole selection. With TR the ground variant is submitted to a TR-enforcing witness after β
and refused; the honest (first, on-time) header stands.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from ..crypto import hashing as hs
from ..crypto.signing import KeySet
from ..protocol.beacon import DrandRounds, EpochClock
from ..protocol.epoch import build_header, make_opening, run_selection
from ..protocol.header import GENESIS_PREV, gw_id_from_pk, sign_header
from ..protocol.policy import ConstantPolicy
from ..protocol.store import MetadataStore, prepare_batch
from ..protocol.synth import SyntheticMetaCodec
from ..protocol.witness import Witness, WitnessRefusal
from ..units.batch import EpochBatch, Frame, Unit

__all__ = ["grinding_probability", "make_target_batch", "variant_batch", "GrindResult", "simulate_grinding"]


def grinding_probability(pi: float, k: int, M: int) -> float:
    """P(evidence retained) = (1 − (1 − π)^k)^M."""
    return (1.0 - (1.0 - pi) ** k) ** M


def _unit(e: int, dev: int, ts: int, pay: bool, rng: random.Random, n_pk: int = 3) -> Unit:
    key = bytes([6]) + bytes(rng.getrandbits(8) for _ in range(36))
    frames = []
    for j in range(n_pk):
        data = bytes(rng.getrandbits(8) for _ in range(60 + (40 if pay else 0)))
        frames.append(Frame(ts + j, j & 1, len(data), len(data), data))
    pk = [sum(1 for f in frames if f.dir == 0), sum(1 for f in frames if f.dir == 1)]
    fb = [sum(f.caplen for f in frames if f.dir == 0), sum(f.caplen for f in frames if f.dir == 1)]
    s = sum(f.caplen + 16 for f in frames)
    meta = SyntheticMetaCodec.encode(dev, pk, fb, s, ts, pay, pay, bytes(128))
    return Unit(epoch=e, index=0, key=key, device_id=dev, first_ts_us=ts, last_ts_us=ts + n_pk - 1,
                pkts=(pk[0], pk[1]), payload_bytes=(40 * pk[0], 40 * pk[1]) if pay else (0, 0),
                frame_bytes=(fb[0], fb[1]), s=s, pay=pay, fps=pay, flow_uid=hs.flow_uid(key, ts), meta=meta,
                frames=frames if pay else None)


def _reindex(e: int, t0: int, units: list[Unit]) -> EpochBatch:
    units = sorted(units, key=lambda u: (u.first_ts_us, u.key))
    out = []
    for i, u in enumerate(units):
        v = copy.copy(u)
        v.index, v.epoch = i, e
        v.leaf, v.root_pkt, v.pkt_leaves = b"", None, None
        out.append(v)
    return EpochBatch(epoch=e, t_start_us=t0, units=out)


def make_target_batch(e: int, t0: int, k: int, n_background: int, rng: random.Random,
                      target_device: int = 7) -> tuple[EpochBatch, set[bytes]]:
    """k target payload units of ``target_device`` + background payload units; returns target keys."""
    units, tkeys = [], set()
    for _ in range(k):
        u = _unit(e, target_device, t0 + rng.randrange(1, 29_000_000), True, rng)
        units.append(u)
        tkeys.add(u.key)
    for _ in range(n_background):
        units.append(_unit(e, rng.randrange(4), t0 + rng.randrange(1, 29_000_000), True, rng))
    return _reindex(e, t0, units), tkeys


def variant_batch(base: EpochBatch, j: int, rng: random.Random) -> EpochBatch:
    """Variant j: j dummy zero-payload units inserted at random times (j = 0: the honest batch)."""
    units = [copy.copy(u) for u in base.units]
    for _ in range(j):
        units.append(_unit(base.epoch, 9, base.t_start_us + rng.randrange(1, 29_000_000), False, rng, n_pk=1))
    return _reindex(base.epoch, base.t_start_us, units)


@dataclass
class GrindResult:
    M: int
    with_tr: bool
    trials: int
    retained_any: int
    refusals: int
    p_hat: float
    p_formula: float


def simulate_grinding(keys: KeySet, beacon: DrandRounds, clock: EpochClock, pi: str = "0.05", k: int = 10,
                      M: int = 5, trials: int = 200, with_tr: bool = False, n_background: int = 20,
                      seed: int = 0, e_start: int = 0) -> GrindResult:
    """Empirical P(at least one target unit retained) under best-of-M grinding (one epoch per trial)."""
    rng = random.Random(seed)
    policy = ConstantPolicy(pi)
    gw_id = gw_id_from_pk(keys.pk_G)
    witness = Witness(keys.sk_W, enforce_tr=True) if with_tr else None
    if witness is not None:
        witness.register_gateway(gw_id, keys.pk_G, keys.pk_V, clock, clock.t_e(e_start) - 60, e_start)
    hit, refusals = 0, 0
    for t in range(trials):
        e = e_start + t
        state = policy.initial_state(e)
        fields = policy.commit_fields(state)
        base, tkeys = make_target_batch(e, clock.t_e_us(e), k, n_background, rng)
        beta = beacon.get(clock.r_star(e)).randomness

        opening = make_opening(keys.k_st, e, fields, state.boost_set)

        def commit(batch: EpochBatch):
            prepare_batch(batch)
            store = MetadataStore.from_batch(batch, taus=policy.thresholds(batch, state))   # rev1: τ committed
            leaves = [u.leaf for u in batch.units]
            hdr = build_header(gw_id, clock, batch, store, hs.merkle_root(leaves), fields, GENESIS_PREV,
                               opening.commitment())
            return hdr, leaves

        def n_target_retained(batch, hdr, leaves) -> int:
            taus = policy.thresholds(batch, state)
            _, _, _, Z = run_selection(keys.sk_V, hdr, beta, leaves, taus, [u.pay for u in batch.units])
            return sum(1 for u, z in zip(batch.units, Z) if z and u.key in tkeys)

        # honest commitment, sent before the deadline
        b0 = variant_batch(base, 0, rng)
        h0, l0 = commit(b0)
        if witness is not None:
            witness.cosign(h0, sign_header(keys.sk_G, h0), clock.t_e(e + 1) + 1.5)
        best = n_target_retained(b0, h0, l0)
        # after β: try M − 1 more variants, keep the one that minimises target retention
        chosen = best
        for j in range(1, M):
            bj = variant_batch(base, j, rng)
            hj, lj = commit(bj)
            r = n_target_retained(bj, hj, lj)
            if witness is not None:
                try:   # the ground header arrives after β (t(r*_e) + 0.5 s): TR refuses it
                    witness.cosign(hj, sign_header(keys.sk_G, hj), clock.t_rstar(e) + 0.5)
                except WitnessRefusal:
                    refusals += 1
                    continue
            chosen = min(chosen, r)
        hit += chosen > 0
    p = float(pi)
    return GrindResult(M=M, with_tr=with_tr, trials=trials, retained_any=hit, refusals=refusals,
                       p_hat=hit / trials, p_formula=grinding_probability(p, k, 1 if with_tr else M))
