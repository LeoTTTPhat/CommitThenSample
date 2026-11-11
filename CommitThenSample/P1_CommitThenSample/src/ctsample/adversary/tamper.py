"""Tampering adversaries of DESIGN §6.3: A1–A5, A9, A10, A12.

Post-commit attacks (A_retro: A1, A2, A3, A4, A5, A9) take an honest ``Disclosure`` (and, where the
attacker needs material it had before selection, the honest ``Gateway``) and return a mutated
deep copy plus an ``info`` dict naming what was changed. Pre-commit attacks (A_rt: A10, A12) are
policy wrappers / batch mutations applied BEFORE the gateway commits.

| ID  | function                          | expected detection                                      |
|-----|-----------------------------------|---------------------------------------------------------|
| A1  | a1_drop_selected                  | V6 MISSING                                              |
| A2  | a2_insert_extra / a2_replace_frame| V6 EXTRA / V6 MISMATCH                                  |
| A3  | a3_alter_metadata (meta/device/s/tau) | V4 store_hash (+ root for meta)                     |
| A4  | a4_swap_theta                     | V4 claimed_taus → V5 claimed_selection → V6 MISSING/EXTRA|
| A5  | a5_alter_lambda (claim / header / opening) | V4 claimed_taus (+V5/V6) / V1 signature + cosignature + V4 floor / V4 state_opening |
| A5' | reopen_state (open C^st_e to another state)| V4 state_opening (binding of the hiding state commitment)|
| A9  | a9_cherry_pick                    | V5 claimed_selection, V6 MISSING                        |
| A10 | BoostDeviationPolicy              | V4 policy_compliance (WARN; harmless to validity)       |
| A4c | ThresholdDeviationPolicy (θ' at commit) | V4 tau_compliance WARN with the prime bundle (NCHK without);|
|     |                                   | harmless to validity (Thm. 2(e)): selection checks PASS  |
| A5c | ThresholdDeviationPolicy (τ scaled below τ^φ at commit) | V4 committed_taus FAIL (floor)    |
| A12 | a12_equal_size_substitution       | NOT detected (stated limit)                             |

rev1 (committed τ_i): A4 / A5 "claim" act AFTER β against thresholds that were committed before β, so the
verifier detects them through V4 claimed_taus (claimed ≠ committed) and V5/V6 (selection with the committed τ);
the pre-commit variants A4c / A5c commit the deviating thresholds themselves.
"""
from __future__ import annotations

import copy
import random
from collections.abc import Callable, Sequence
from dataclasses import replace

from ..crypto import hashing as hs
from ..protocol.disclosure import DisclosedUnit, Disclosure
from ..protocol.policy import Justification, PolicyState, ThresholdPolicy
from ..units.batch import EpochBatch, Frame

__all__ = ["a1_drop_selected", "a2_insert_extra", "a2_replace_frame", "a3_alter_metadata", "a4_swap_theta",
           "a5_alter_lambda", "reopen_state", "a9_cherry_pick", "BoostDeviationPolicy", "ThresholdDeviationPolicy",
           "a4c_commit_theta", "a5c_commit_scaled", "a12_equal_size_substitution"]


def _epochs_with_units(disc: Disclosure) -> list[int]:
    return sorted(e for e, d in disc.epochs.items() if d.units)


def a1_drop_selected(disc: Disclosure, rng: random.Random, e: int | None = None,
                     also_from_claims: bool = False) -> tuple[Disclosure, dict]:
    """Delete one retained (Z = 1) unit from the evidence presented."""
    d2 = disc.copy()
    e = rng.choice(_epochs_with_units(d2)) if e is None else e
    ed = d2.epochs[e]
    i = rng.choice(sorted(ed.units))
    del ed.units[i]
    if also_from_claims and ed.claimed_selected is not None:
        ed.claimed_selected = [x for x in ed.claimed_selected if x != i]
    return d2, {"epoch": e, "unit": i}


def a2_insert_extra(disc: Disclosure, gw, rng: random.Random, e: int | None = None) -> tuple[Disclosure, dict]:
    """Present a payload unit with Z = 0 as retained (its true frames, valid proof): EXTRA."""
    d2 = disc.copy()
    cands = []
    for ee, ed in d2.epochs.items():
        rec = gw.records[ee]
        for i, u in enumerate(rec.batch.units):
            if u.pay and not rec.Z[i] and u.frames is not None and (e is None or ee == e):
                cands.append((ee, i))
    ee, i = rng.choice(sorted(cands))
    rec = gw.records[ee]
    u = rec.batch.units[i]
    d2.epochs[ee].units[i] = DisclosedUnit(index=i, frames=copy.deepcopy(u.frames), pkt_leaves=list(u.pkt_leaves),
                                           proof=hs.merkle_proof(rec.leaves, i))
    return d2, {"epoch": ee, "unit": i}


def a2_replace_frame(disc: Disclosure, rng: random.Random, fix_hash_list: bool = False,
                     e: int | None = None) -> tuple[Disclosure, dict]:
    """Replace the payload bytes of one retained frame (same length). With ``fix_hash_list`` the
    attacker also rewrites ℓ_ij in the frame-hash list (then root^pkt no longer matches)."""
    d2 = disc.copy()
    e = rng.choice(_epochs_with_units(d2)) if e is None else e
    ed = d2.epochs[e]
    i = rng.choice(sorted(ed.units))
    du = ed.units[i]
    j = rng.randrange(len(du.frames))
    f = du.frames[j]
    data = bytearray(f.data)
    k = rng.randrange(len(data))
    data[k] ^= 0xFF
    du.frames[j] = Frame(f.ts_us, f.dir, f.caplen, f.wirelen, bytes(data))
    if fix_hash_list:
        du.pkt_leaves[j] = hs.frame_leaf(f.ts_us, f.dir, f.caplen, bytes(data))
    return d2, {"epoch": e, "unit": i, "packet": j, "fix_hash_list": fix_hash_list}


def a3_alter_metadata(disc: Disclosure, rng: random.Random, field: str = "meta",
                      e: int | None = None) -> tuple[Disclosure, dict]:
    """Change m_i (or a committed derived column) of one unit in the metadata store after commit."""
    d2 = disc.copy()
    e = rng.choice(sorted(x for x, d in d2.epochs.items() if d.store is not None and d.store.n)) if e is None else e
    st = d2.epochs[e].store
    i = rng.randrange(st.n)
    r = st.records[i]
    if field == "meta":
        m = bytearray(r.meta)
        m[rng.randrange(len(m))] ^= 0x01
        st.records[i] = replace(r, meta=bytes(m))
    elif field == "device_id":
        st.records[i] = replace(r, device_id=r.device_id + 1)
    elif field == "s":
        st.records[i] = replace(r, s=r.s + 1)
    elif field == "tau":                         # rev1: raise a committed threshold after commit
        pay = [k for k, x in enumerate(st.records) if x.pay]
        if not pay or st.taus is None:
            raise ValueError("no committed payload threshold to alter")
        i = rng.choice(pay)
        t = list(st.taus)
        t[i] = min(1 << 64, t[i] * 2)
        st.taus = t
    else:
        raise ValueError(field)
    return d2, {"epoch": e, "unit": i, "field": field}


def _reselect_claim(gw, disc: Disclosure, new_taus: dict[int, list[int]]) -> tuple[Disclosure, dict]:
    """The gateway selects with τ' (it knows sk_V, so u_i are its own) and discloses accordingly."""
    d2 = disc.copy()
    changed = {}
    for e, ed in d2.epochs.items():
        if e not in new_taus:
            continue
        rec = gw.records[e]
        t2 = new_taus[e]
        Z2 = [bool(u.pay) and rec.u[i] < t2[i] for i, u in enumerate(rec.batch.units)]
        S2 = {i for i, z in enumerate(Z2) if z}
        ed.claimed_taus = list(t2)
        ed.claimed_selected = sorted(S2)
        units = {}
        sel = [i for i in ed.requested if i in S2]
        proofs = hs.merkle_proofs(rec.leaves, sel) if sel else {}
        for i in sel:
            u = rec.batch.units[i]
            units[i] = DisclosedUnit(index=i, frames=copy.deepcopy(u.frames), pkt_leaves=list(u.pkt_leaves),
                                     proof=proofs[i])
        ed.units = units
        changed[e] = {"dropped": sorted(rec.selected - S2), "added": sorted(S2 - rec.selected)}
    return d2, changed


def a4_swap_theta(gw, disc: Disclosure, theta_policy: ThresholdPolicy,
                  epochs: Sequence[int] | None = None) -> tuple[Disclosure, dict]:
    """A4: at selection time the gateway computes τ with another utility model θ' (``theta_policy``,
    evaluated on the same committed batch and state) and retains/discloses accordingly."""
    epochs = sorted(disc.epochs) if epochs is None else list(epochs)
    new = {e: theta_policy.thresholds(gw.records[e].batch, gw.records[e].state) for e in epochs}
    return _reselect_claim(gw, disc, new)


def a5_alter_lambda(gw, disc: Disclosure, scale: float = 0.25, mode: str = "claim",
                    epochs: Sequence[int] | None = None) -> tuple[Disclosure, dict]:
    """A5: after β the gateway uses Λ' (here: every τ_i scaled by ``scale``; for the constant policy
    Λ ≡ τ^φ) to select, and claims τ'.
    mode="claim":   header and opening untouched;
    mode="header":  also rewrites the public τ^φ of the header (breaks σ_e, ω_e and the V4 floor);
    mode="opening": also opens C^st_e to Λ' = scale·Λ (breaks the V4 state opening)."""
    epochs = sorted(disc.epochs) if epochs is None else list(epochs)
    new = {e: [int(t * scale) for t in gw.records[e].taus] for e in epochs}
    d2, info = _reselect_claim(gw, disc, new)
    if mode == "header":
        for e in epochs:
            ed = d2.epochs[e]
            ed.header = ed.header.replace(tau_phi=int(ed.header.tau_phi * scale))
    elif mode == "opening":
        d2, _ = reopen_state(d2, epochs, Lam_scale=scale)
    elif mode != "claim":
        raise ValueError(mode)
    return d2, info


def reopen_state(disc: Disclosure, epochs: Sequence[int], Lam_scale: float | None = None, D_delta: int = 0,
                 floor_mode: int | None = None, boost_add: Sequence[int] = (), boost_remove: Sequence[int] = (),
                 new_salt: bytes | None = None) -> tuple[Disclosure, dict]:
    """Open the hiding state commitment C^st_e of ``epochs`` to a DIFFERENT design state (water levels,
    deficit, floor mode, boost set) or with another salt, leaving the signed header unchanged.
    Binding of C^st_e (BLAKE3 collision resistance) makes V4 'state_opening' FAIL."""
    d2 = disc.copy()
    changed = {}
    for e in epochs:
        ed = d2.epochs[e]
        op = ed.opening
        kw = {}
        if Lam_scale is not None:
            kw.update(Lam_A=int(op.Lam_A * Lam_scale), Lam_S=int(op.Lam_S * Lam_scale))
        if D_delta:
            kw["D_prev"] = op.D_prev + int(D_delta)
        if floor_mode is not None:
            kw["floor_mode"] = int(floor_mode)
        if boost_add or boost_remove:
            kw["boost_set"] = tuple(sorted((set(op.boost_set) | set(boost_add)) - set(boost_remove)))
        if new_salt is not None:
            kw["salt"] = new_salt
        ed.opening = op.replace(**kw)
        changed[e] = kw
    return d2, changed


def a9_cherry_pick(disc: Disclosure, keep: Callable[[int, int], bool], update_claims: bool = True) -> tuple[Disclosure, dict]:
    """A9: disclose only a favourable subset of the retained requested units (keep(e, i) → bool)."""
    d2 = disc.copy()
    dropped = {}
    for e, ed in d2.epochs.items():
        drop = [i for i in ed.units if not keep(e, i)]
        for i in drop:
            del ed.units[i]
        if drop:
            dropped[e] = sorted(drop)
            if update_claims and ed.claimed_selected is not None:
                ed.claimed_selected = [i for i in ed.claimed_selected if i not in drop]
    return d2, dropped


class BoostDeviationPolicy(ThresholdPolicy):
    """A10 (pre-commit policy deviation): wraps an honest policy and, at commit time, either forges a
    boost for ``device`` without justification (``mode="forge"``; optionally with a fabricated
    justification record ``fake_justification``) or omits every justified boost (``mode="omit"``)
    in the epochs ``epochs``. Thresholds are computed from the deviating committed state, so π stays
    predictable (Thm. 2(e)); the verifier, using the honest policy, flags it in V4 compliance."""

    def __init__(self, base: ThresholdPolicy, mode: str, epochs: Sequence[int], device: int | None = None,
                 fake_justification: bool = False):
        if mode not in ("forge", "omit"):
            raise ValueError(mode)
        self.base = base
        self.mode = mode
        self.epochs = set(epochs)
        self.device = device
        self.fake = fake_justification
        self.name = base.name
        self.Bbar, self.D_max = base.Bbar, base.D_max

    def _deviate(self, state: PolicyState) -> PolicyState:
        if state.e not in self.epochs:
            return state
        if self.mode == "omit":
            return replace(state, boost_set=(), justifications=())
        js = list(state.justifications)
        if self.fake:
            det = getattr(self.base, "detector_name", "det0")
            js.append(Justification(self.device, max(0, state.e - 2), 0, det))
        bs = tuple(sorted(set(state.boost_set) | {self.device}))
        return replace(state, boost_set=bs, justifications=tuple(sorted(js)))

    def theta_bytes(self) -> bytes:
        return self.base.theta_bytes()

    def policy_params(self) -> dict:
        return self.base.policy_params()

    def registration(self):
        return self.base.registration()

    def expected_tau_phi(self) -> int:
        return self.base.expected_tau_phi()

    def initial_state(self, e0: int) -> PolicyState:
        return self.base.initial_state(e0)

    def committed_state(self, state: PolicyState) -> PolicyState:
        return self._deviate(state)

    def commit_fields(self, state: PolicyState) -> dict:
        return self.base.commit_fields(state)

    def thresholds(self, batch: EpochBatch, state: PolicyState) -> list[int]:
        return self.base.thresholds(batch, state)

    def floor_thresholds(self, batch: EpochBatch, state: PolicyState) -> list[int]:
        return self.base.floor_thresholds(batch, state)

    def set_chain_start(self, e0: int) -> None:
        return self.base.set_chain_start(e0)

    def update(self, state: PolicyState, record) -> PolicyState:
        return self.base.update(state, record)

    def state_from_commit(self, hdr, boost_set=(), justifications=()):
        return self.base.state_from_commit(hdr, boost_set, justifications)

    def compliance(self, hdr, state, ctx):
        return self.base.compliance(hdr, state, ctx)


class ThresholdDeviationPolicy(ThresholdPolicy):
    """A4c / A5c (pre-commit, A_rt): the gateway COMMITS thresholds that deviate from the registered policy in
    the epochs ``epochs``: τ'_i = ``fn(batch, state, τ_i)`` (honest τ_i of the base policy). The deviating τ' are
    committed before β, so the design stays predictable and every statement stays valid (Thm. 2(e)); V4
    'tau_compliance' reports the deviation (WARN) when the verifier can recompute τ (prime bundle), NCHK
    otherwise; τ' below the floor τ^φ_i FAILs V4 'committed_taus'."""

    def __init__(self, base: ThresholdPolicy, fn: Callable, epochs: Sequence[int], label: str = "deviation"):
        self.base = base
        self.fn = fn
        self.epochs = set(int(e) for e in epochs)
        self.label = label
        self.name = base.name
        self.Bbar, self.D_max = base.Bbar, base.D_max

    def thresholds(self, batch: EpochBatch, state: PolicyState) -> list[int]:
        t = self.base.thresholds(batch, state)
        if int(batch.epoch) not in self.epochs:
            return t
        return [int(x) for x in self.fn(batch, state, t)]

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "base"), name)

    # explicit delegation of the ThresholdPolicy interface (base-class defaults must not shadow the base policy)
    def theta_bytes(self):
        return self.base.theta_bytes()

    def policy_params(self):
        return self.base.policy_params()

    def registration(self):
        return self.base.registration()

    def expected_tau_phi(self):
        return self.base.expected_tau_phi()

    def initial_state(self, e0):
        return self.base.initial_state(e0)

    def set_chain_start(self, e0):
        return self.base.set_chain_start(e0)

    def committed_state(self, state):
        return self.base.committed_state(state)

    def commit_fields(self, state):
        return self.base.commit_fields(state)

    def floor_thresholds(self, batch, state):
        return self.base.floor_thresholds(batch, state)

    def update(self, state, record):
        return self.base.update(state, record)

    def state_from_commit(self, hdr, boost_set=(), justifications=()):
        return self.base.state_from_commit(hdr, boost_set, justifications)

    def compliance(self, hdr, state, ctx):
        return self.base.compliance(hdr, state, ctx)


def a4c_commit_theta(base: ThresholdPolicy, theta_policy: ThresholdPolicy, epochs: Sequence[int]) -> ThresholdDeviationPolicy:
    """A4c: commit τ computed with another utility model θ' (``theta_policy`` on the same batch and state),
    clipped up to the floor of the registered policy (a rational gateway keeps the floor check passing)."""
    def fn(batch, state, t):
        alt = theta_policy.thresholds(batch, state)
        fl = base.floor_thresholds(batch, state)
        return [max(a, f) if u.pay else 0 for a, f, u in zip(alt, fl, batch.units)]
    return ThresholdDeviationPolicy(base, fn, epochs, label="A4c_theta_swap")


def a5c_commit_scaled(base: ThresholdPolicy, scale: float, epochs: Sequence[int], keep_floor: bool = False
                      ) -> ThresholdDeviationPolicy:
    """A5c: commit τ' = scale·τ (water levels scaled at commit time); ``keep_floor`` clips up to τ^φ_i (a pure
    policy deviation, WARN), otherwise τ' may fall below the floor (FAIL)."""
    def fn(batch, state, t):
        fl = base.floor_thresholds(batch, state)
        out = []
        for x, f, u in zip(t, fl, batch.units):
            if not u.pay:
                out.append(0)
                continue
            y = max(1, int(x * scale))
            out.append(max(y, f) if keep_floor else y)
        return out
    return ThresholdDeviationPolicy(base, fn, epochs, label=f"A5c_scale_{scale:g}{'_keepfloor' if keep_floor else ''}")


def a12_equal_size_substitution(batches: Sequence[EpochBatch], device: int, rng: random.Random,
                                header_len: int = 54) -> dict:
    """A12 (A_rt, BEFORE commit): rewrite the payload bytes of every payload unit of ``device``
    keeping all lengths (frames are mutated in place; leaves must not be computed yet)."""
    n = 0
    for b in batches:
        for u in b.units:
            if u.device_id != device or not u.pay or u.frames is None:
                continue
            if u.leaf or u.root_pkt is not None:
                raise ValueError("A12 must run before the gateway computes leaves")
            new = []
            for f in u.frames:
                if len(f.data) > header_len:
                    body = bytes(rng.getrandbits(8) for _ in range(len(f.data) - header_len))
                    f = Frame(f.ts_us, f.dir, f.caplen, f.wirelen, f.data[:header_len] + body)
                    n += 1
                new.append(f)
            u.frames = new
    return {"device": device, "frames_rewritten": n}
