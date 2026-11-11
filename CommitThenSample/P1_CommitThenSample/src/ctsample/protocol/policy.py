"""Pluggable threshold policies (DESIGN §3.4–3.6 hooks; used by protocol/epoch.py and verifier V4).

Interface contract (the real AI/size policy of another worker plugs in here):

    class ThresholdPolicy:
        def initial_state(self, e0) -> PolicyState
        def committed_state(self, state) -> PolicyState         # state actually committed (identity if honest)
        def registration(self) -> (H_theta, H_policy)          # what the witness registers
        def expected_tau_phi(self) -> int                       # ⌊φ0·2^64⌋ (V4 floor check)
        def commit_fields(self, state) -> dict                  # header fields, see COMMIT_KEYS
        def thresholds(self, batch, state) -> list[int]         # τ_i per unit; 0 for zero-payload
        def update(self, state, record) -> PolicyState          # next epoch's state (controller, boost)
        def state_from_commit(self, hdr, boost_set, justifications) -> PolicyState   # verifier side
        def compliance(self, hdr, state, ctx) -> list[dict]     # policy deviations (V4, reported)

rev1 (committed thresholds): the gateway commits τ_i in the metadata store (protocol/store.py), so the
verifier's VALIDITY checks use the committed τ_i and ``floor_thresholds`` only (epoch-local); recomputing
``thresholds`` (which may need the rolling history, i.e. a prime bundle) is a policy-COMPLIANCE check
(deviation → WARN, harmless to validity by Thm. 2(e); history missing → "not checkable").

RULES that make the V4 compliance recomputation bit-exact:
1. ``thresholds(batch, state)`` may depend ONLY on committed data: the units' committed fields
   (meta, pay, s, device_id, pkts, frame_bytes, first_ts_us, fps — exactly what
   MetadataStore.to_batch() reconstructs; decode ``meta`` yourself for anything else) and the
   committed state fields (e, Lam_A, Lam_S, tau_phi, floor_mode, D_prev, boost_set,
   justifications). Never on frames, Z of the same epoch, wall-clock or private randomness.
   The verifier calls ``thresholds(store.to_batch(), state_from_commit(hdr, …))``.
2. Integer arithmetic only (design/intpi.py). τ_i ∈ [τ^φ, 2^64] for payload units, 0 otherwise.
3. ``commit_fields(state)`` returns COMMIT_KEYS; ``state_from_commit`` must invert it, i.e.
   commit_fields(state_from_commit(hdr, bs, js)) reproduces the header's values.
4. H_boost = boost_digest(boost_set, justifications) (committed before β inside the hiding state
   commitment C^st_e of the header together with Λ^A, Λ^S, D_{e−1}, floor_mode and the boost device
   ids; the opening and the justification records are disclosed so V4 can check them). The header
   itself no longer carries these fields; verifier-side calls receive an ``OpenedHeader`` with the
   same attribute names, so policies are unchanged.
5. ``update`` runs after selection of epoch e (it may use Z_e and retained frames) and returns
   the state for e + 1; boost justifications found in epoch e may act from epoch e + 2 on.
"""
from __future__ import annotations

import json
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from blake3 import blake3

from ..design import intpi
from ..units.batch import EpochBatch, Frame, Unit
from .header import Header

DOM_THETA = b"CTS/v1/theta"
DOM_POLICY = b"CTS/v1/policy"
DOM_BOOST = b"CTS/v1/boost"
COMMIT_KEYS = ("Lam_A", "Lam_S", "tau_phi", "H_theta", "H_policy", "H_boost", "floor_mode", "D_prev")

__all__ = ["Justification", "PolicyState", "ComplianceContext", "ThresholdPolicy", "ConstantPolicy",
           "FixedLambdaPolicy", "boost_digest", "COMMIT_KEYS", "canonical_policy_hash", "theta_hash"]


@dataclass(frozen=True, order=True)
class Justification:
    """A boost justification: detector ``detector`` fired on retained unit (src_epoch, src_index)."""
    device_id: int
    src_epoch: int
    src_index: int
    detector: str

    def encode(self) -> bytes:
        d = self.detector.encode()
        if len(d) > 255:
            raise ValueError("detector name too long")
        return struct.pack("<IIIB", self.device_id, self.src_epoch, self.src_index, len(d)) + d


def boost_digest(boost_set: Sequence[int], justifications: Sequence[Justification]) -> bytes:
    """H(boost set_e) over the sorted device set and the sorted justification records."""
    devs = sorted(set(int(x) for x in boost_set))
    js = sorted(justifications)
    parts = [DOM_BOOST, struct.pack("<I", len(devs))] + [struct.pack("<I", x) for x in devs]
    parts += [struct.pack("<I", len(js))] + [j.encode() for j in js]
    return blake3(b"".join(parts)).digest()


def theta_hash(theta_bytes: bytes) -> bytes:
    return blake3(DOM_THETA + theta_bytes).digest()


def canonical_policy_hash(params: dict) -> bytes:
    return blake3(DOM_POLICY + json.dumps(params, sort_keys=True, separators=(",", ":")).encode()).digest()


@dataclass
class PolicyState:
    e: int
    D_prev: int = 0
    floor_mode: int = 0
    Lam_A: int = 0
    Lam_S: int = 0
    boost_set: tuple[int, ...] = ()
    justifications: tuple[Justification, ...] = ()
    extra: dict = field(default_factory=dict)        # policy-private (controller internals, pending boosts)


@dataclass
class ComplianceContext:
    """What the verifier knows (verified data only) when checking policy compliance of epoch e.

    ``headers[e]`` is an ``OpenedHeader`` (public header + verified opening of C^st_e), so policies
    read hdr.e, hdr.Lam_A, hdr.Lam_S, hdr.D_prev, hdr.floor_mode, hdr.H_boost as before."""
    headers: dict[int, Header]
    stores: dict = field(default_factory=dict)        # e -> MetadataStore (verified)
    taus: dict[int, list[int]] = field(default_factory=dict)
    Z: dict[int, list[bool]] = field(default_factory=dict)
    frames: dict[tuple[int, int], list[Frame]] = field(default_factory=dict)   # verified disclosed frames
    states: dict[int, PolicyState] = field(default_factory=dict)


def _expected_cost_int(taus: Sequence[int], sizes: Sequence[int]) -> int:
    return sum(int(t) * int(s) for t, s in zip(taus, sizes, strict=True)) >> 64


class ThresholdPolicy:
    """Base class. Subclasses implement ``commit_fields`` and ``thresholds``."""
    name = "abstract"

    # defaults for budget accounting (D recursion with expected cost; None disables it)
    Bbar: int | None = None
    D_max: int | None = None

    def initial_state(self, e0: int) -> PolicyState:
        return PolicyState(e=e0)

    def set_chain_start(self, e0: int) -> None:
        """Verifier side: the chain start e0 of the trust anchors. A policy with a rolling history must anchor
        its window there, so that τ of an epoch whose history was not seen raises (→ V4 NCHK / rev0 FAIL)
        instead of being computed on a silently truncated window. No-op for history-free policies."""
        return None

    def committed_state(self, state: PolicyState) -> PolicyState:
        """The state the gateway commits for this epoch (identity for an honest policy). The gateway
        calls it once per epoch and passes the result to commit_fields() and thresholds() and records
        it, so the disclosed boost set opens H(boost set)."""
        return state

    def theta_bytes(self) -> bytes:
        raise NotImplementedError

    def policy_params(self) -> dict:
        raise NotImplementedError

    def registration(self) -> tuple[bytes, bytes]:
        return theta_hash(self.theta_bytes()), canonical_policy_hash(self.policy_params())

    def expected_tau_phi(self) -> int:
        raise NotImplementedError

    def commit_fields(self, state: PolicyState) -> dict:
        raise NotImplementedError

    def thresholds(self, batch: EpochBatch, state: PolicyState) -> list[int]:
        raise NotImplementedError

    def floor_thresholds(self, batch: EpochBatch, state: PolicyState) -> list[int]:
        """Per-unit floor τ^φ_i that the COMMITTED thresholds must respect (rev1 V4 'committed_taus', a
        validity check): τ^φ for payload units, 0 for zero-payload units. A policy with a flood guard overrides
        it with the capped per-device floor of floor mode 1; it may use only the epoch's OWN committed store
        and the opened state (never rolling features), so that the check needs no prime bundle."""
        t = int(self.expected_tau_phi())
        return [t if u.pay else 0 for u in batch.units]

    def update(self, state: PolicyState, record) -> PolicyState:
        """Default: D_e = D_{e−1} + B̄ − ⌊Σ τ_i s_i / 2^64⌋ (expected cost); no boost."""
        D = self._next_D(state, record.taus, [u.s for u in record.batch.units])
        return PolicyState(e=state.e + 1, D_prev=D, floor_mode=self._floor_mode(D),
                           Lam_A=state.Lam_A, Lam_S=state.Lam_S)

    def _next_D(self, state: PolicyState, taus, sizes) -> int:
        if self.Bbar is None:
            return 0
        return int(state.D_prev + self.Bbar - _expected_cost_int(taus, sizes))

    def _floor_mode(self, D: int) -> int:
        return 1 if (self.D_max is not None and D < -self.D_max) else 0

    def state_from_commit(self, hdr: Header, boost_set: Sequence[int] = (),
                          justifications: Sequence[Justification] = ()) -> PolicyState:
        """``hdr`` is an OpenedHeader (public header + opened C^st_e): Lam_A, Lam_S, D_prev, floor_mode
        come from the opening."""
        return PolicyState(e=hdr.e, D_prev=hdr.D_prev, floor_mode=hdr.floor_mode, Lam_A=hdr.Lam_A, Lam_S=hdr.Lam_S,
                           boost_set=tuple(sorted(set(boost_set))), justifications=tuple(sorted(justifications)))

    def compliance(self, hdr: Header, state: PolicyState, ctx: ComplianceContext) -> list[dict]:
        """Generic checks: D recursion over the committed past and the floor_mode rule."""
        dev = []
        e = hdr.e
        if self.Bbar is not None and (e - 1) in ctx.headers and (e - 1) in ctx.taus and (e - 1) in ctx.stores:
            prev = ctx.headers[e - 1]
            sizes = [r.s for r in ctx.stores[e - 1].records]
            want = prev.D_prev + self.Bbar - _expected_cost_int(ctx.taus[e - 1], sizes)
            if want != hdr.D_prev:
                dev.append({"rule": "D_recursion", "committed": hdr.D_prev, "expected": want})
        if hdr.floor_mode not in (0, 1):
            dev.append({"rule": "floor_mode_value", "committed": hdr.floor_mode})
        if hdr.floor_mode == 1 and not (self.D_max is not None and hdr.D_prev < -self.D_max):
            dev.append({"rule": "floor_mode_without_deficit", "D_prev": hdr.D_prev, "D_max": self.D_max})
        if hdr.floor_mode == 0 and self.D_max is not None and hdr.D_prev < -self.D_max:
            dev.append({"rule": "floor_mode_missing", "D_prev": hdr.D_prev, "D_max": self.D_max})
        return dev


class ConstantPolicy(ThresholdPolicy):
    """τ_i = ⌊π·2^64⌋ for every payload unit (the floor IS the constant), 0 for zero-payload units.

    Optional follow-the-evidence boost (for tests of A10): if ``detector(frames)`` fires on a
    retained unit of device d in epoch e, payload units of d get τ_boost in epochs
    e+2 … e+1+boost_epochs; the justification records are committed in H(boost set)."""
    name = "constant"

    def __init__(self, pi="0.05", tau_boost_pi=None, boost_epochs: int = 3,
                 detector: Callable[[list[Frame]], bool] | None = None, detector_name: str = "det0",
                 Bbar: int | None = None, D_max: int | None = None):
        self.pi = str(pi)
        self.tau = intpi.floor_tau(self.pi)
        self.tau_boost = intpi.floor_tau(str(tau_boost_pi)) if tau_boost_pi is not None else None
        if self.tau_boost is not None and self.tau_boost < self.tau:
            raise ValueError("boost threshold below the floor")
        self.boost_epochs = int(boost_epochs)
        self.detector = detector
        self.detector_name = detector_name
        self.Bbar = Bbar
        self.D_max = D_max

    def theta_bytes(self) -> bytes:
        return b"constant-utility"

    def policy_params(self) -> dict:
        return {"name": self.name, "pi": self.pi, "tau_boost": self.tau_boost, "boost_epochs": self.boost_epochs,
                "detector": self.detector_name if self.detector else None, "Bbar": self.Bbar, "D_max": self.D_max}

    def expected_tau_phi(self) -> int:
        return self.tau

    def commit_fields(self, state: PolicyState) -> dict:
        H_theta, H_policy = self.registration()
        return {"Lam_A": 0, "Lam_S": 0, "tau_phi": self.tau, "H_theta": H_theta, "H_policy": H_policy,
                "H_boost": boost_digest(state.boost_set, state.justifications), "floor_mode": state.floor_mode,
                "D_prev": state.D_prev}

    def thresholds(self, batch: EpochBatch, state: PolicyState) -> list[int]:
        boost = set(state.boost_set) if self.tau_boost is not None else set()
        out = []
        for u in batch.units:
            if not u.pay:
                out.append(0)
            elif u.device_id in boost:
                out.append(self.tau_boost)
            else:
                out.append(self.tau)
        return out

    # ------------------------------------------------------------------ boost bookkeeping
    def _active(self, js, e: int) -> tuple[Justification, ...]:
        return tuple(sorted(j for j in js if j.src_epoch + 2 <= e <= j.src_epoch + 1 + self.boost_epochs))

    def detect(self, record) -> list[Justification]:
        """Detector hits on the retained units of an epoch record (gateway side)."""
        if self.detector is None:
            return []
        hits = []
        for i in record.retained:
            u = record.batch.units[i]
            if u.frames is not None and self.detector(u.frames):
                hits.append(Justification(u.device_id, record.e, i, self.detector_name))
        return hits

    def update(self, state: PolicyState, record) -> PolicyState:
        D = self._next_D(state, record.taus, [u.s for u in record.batch.units])
        pending = list(state.extra.get("pending", ())) + self.detect(record)
        e1 = state.e + 1
        pending = [j for j in pending if j.src_epoch + 1 + self.boost_epochs >= e1]
        active = self._active(pending, e1) if self.tau_boost is not None else ()
        return PolicyState(e=e1, D_prev=D, floor_mode=self._floor_mode(D), boost_set=tuple(sorted({j.device_id for j in active})),
                           justifications=active, extra={"pending": pending})

    def compliance(self, hdr: Header, state: PolicyState, ctx: ComplianceContext) -> list[dict]:
        dev = super().compliance(hdr, state, ctx)
        e = hdr.e
        if hdr.Lam_A != 0 or hdr.Lam_S != 0:
            dev.append({"rule": "Lambda", "Lam_A": hdr.Lam_A, "Lam_S": hdr.Lam_S, "expected": 0})
        if self.tau_boost is None:
            if state.boost_set or state.justifications:
                dev.append({"rule": "boost_not_in_policy", "boost_set": list(state.boost_set)})
            return dev
        # (i) the boost set must be exactly the devices of the active justifications
        want_set = tuple(sorted({j.device_id for j in state.justifications}))
        if tuple(state.boost_set) != want_set:
            dev.append({"rule": "boost_set_vs_justifications", "boost_set": list(state.boost_set),
                        "justified": list(want_set)})
        # (ii) every justification must be in its window and point at a retained unit of that device
        for j in state.justifications:
            if not (j.src_epoch + 2 <= e <= j.src_epoch + 1 + self.boost_epochs):
                dev.append({"rule": "justification_window", "justification": j.__dict__})
                continue
            if j.detector != self.detector_name:
                dev.append({"rule": "unknown_detector", "justification": j.__dict__})
                continue
            if j.src_epoch in ctx.Z:
                Zs = ctx.Z[j.src_epoch]
                st = ctx.stores.get(j.src_epoch)
                if not (0 <= j.src_index < len(Zs)) or not Zs[j.src_index]:
                    dev.append({"rule": "unjustified_boost", "reason": "source unit not retained",
                                "justification": j.__dict__})
                    continue
                if st is not None and st.records[j.src_index].device_id != j.device_id:
                    dev.append({"rule": "unjustified_boost", "reason": "device mismatch", "justification": j.__dict__})
                    continue
                fr = ctx.frames.get((j.src_epoch, j.src_index))
                if fr is not None and self.detector is not None and not self.detector(fr):
                    dev.append({"rule": "unjustified_boost", "reason": "detector does not fire on the evidence",
                                "justification": j.__dict__})
            else:
                dev.append({"rule": "justification_unverifiable", "justification": j.__dict__, "severity": "info"})
        # (iii) omitted boosts: verified evidence on which the detector fires must be justified
        if self.detector is not None:
            have = set(state.justifications)
            for (es, i), fr in ctx.frames.items():
                if not (es + 2 <= e <= es + 1 + self.boost_epochs):
                    continue
                if es in ctx.Z and i < len(ctx.Z[es]) and ctx.Z[es][i] and self.detector(fr):
                    dv = ctx.stores[es].records[i].device_id if es in ctx.stores else None
                    j = Justification(dv, es, i, self.detector_name)
                    if dv is not None and j not in have:
                        dev.append({"rule": "omitted_boost", "justification": j.__dict__})
        return dev


class FixedLambdaPolicy(ThresholdPolicy):
    """τ_i = threshold_unit(Λ^A, Λ^S, â_i, s_i, τ^φ, t16) with constant (Λ^A, Λ^S) and a utility
    function â_i = utility(unit) computed from committed data only (for tests and benchmarks)."""
    name = "fixed_lambda"

    def __init__(self, lam_A, lam_S, phi0="0.01", tau_mix="0.2",
                 utility: Callable[[Unit], int] | None = None, theta_id: bytes = b"hash-utility-v1",
                 Bbar: int | None = None, D_max: int | None = None):
        self.lam_A, self.lam_S = str(lam_A), str(lam_S)
        self.Lam_A = intpi.lam_fix(self.lam_A)
        self.Lam_S = intpi.lam_fix(self.lam_S)
        self.phi0 = str(phi0)
        self.tau_phi = intpi.floor_tau(self.phi0)
        self.tau_mix = str(tau_mix)
        self.t16 = intpi.t16_from_mix(self.tau_mix)
        self.utility = utility or default_utility
        self.theta_id = theta_id
        self.Bbar = Bbar
        self.D_max = D_max

    def theta_bytes(self) -> bytes:
        return self.theta_id

    def policy_params(self) -> dict:
        return {"name": self.name, "Lam_A": self.Lam_A, "Lam_S": self.Lam_S, "phi0": self.phi0,
                "tau_mix": self.tau_mix, "t16": self.t16, "Bbar": self.Bbar, "D_max": self.D_max}

    def expected_tau_phi(self) -> int:
        return self.tau_phi

    def initial_state(self, e0: int) -> PolicyState:
        return PolicyState(e=e0, Lam_A=self.Lam_A, Lam_S=self.Lam_S)

    def commit_fields(self, state: PolicyState) -> dict:
        H_theta, H_policy = self.registration()
        return {"Lam_A": state.Lam_A, "Lam_S": state.Lam_S, "tau_phi": self.tau_phi, "H_theta": H_theta,
                "H_policy": H_policy, "H_boost": boost_digest(state.boost_set, state.justifications),
                "floor_mode": state.floor_mode, "D_prev": state.D_prev}

    def thresholds(self, batch: EpochBatch, state: PolicyState) -> list[int]:
        out = []
        for u in batch.units:
            if not u.pay:
                out.append(0)
            else:
                out.append(intpi.threshold_unit(state.Lam_A, state.Lam_S, int(self.utility(u)), u.s,
                                                self.tau_phi, self.t16))
        return out

    def compliance(self, hdr: Header, state: PolicyState, ctx: ComplianceContext) -> list[dict]:
        dev = super().compliance(hdr, state, ctx)
        if hdr.Lam_A != self.Lam_A or hdr.Lam_S != self.Lam_S:
            dev.append({"rule": "Lambda", "Lam_A": hdr.Lam_A, "Lam_S": hdr.Lam_S,
                        "expected": [self.Lam_A, self.Lam_S]})
        if state.boost_set or state.justifications:
            dev.append({"rule": "boost_not_in_policy", "boost_set": list(state.boost_set)})
        return dev


def default_utility(u: Unit) -> int:
    """Deterministic pseudo-utility from committed metadata: â ∈ [2^12, 2^17) (a ∈ [1/16, 2))."""
    h = blake3(b"CTS/v1/test-utility" + u.meta).digest()
    return (1 << 12) + int.from_bytes(h[:4], "little") % ((1 << 17) - (1 << 12))
