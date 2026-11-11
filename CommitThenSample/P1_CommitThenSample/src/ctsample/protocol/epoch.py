"""Gateway G: commit, publish, select, retain (DESIGN §3.1 M2–M7, Alg. 1 steps 1 and 4–7).

Per epoch e (``Gateway.run_epoch``):
1. leaves L_i (crypto.hashing; computed if missing), Merkle root R_e;
2. τ_i = policy.thresholds(batch, state) (rev1: committed per payload unit in the metadata store, so
   H(store) binds them before β; protocol/store.py); fields = policy.commit_fields(state); the design state
   (Λ^A, Λ^S, D_{e−1}, floor_mode, boost set + H_boost) goes into the hiding commitment
   C^st_e = BLAKE3("CTS/v1/st" ‖ salt_e ‖ enc(state_e)), salt_e = PRF_{k_st}(e) (protocol/header.py);
   the public Hdr_e carries τ^φ, H(θ), H(policy), C^st_e, r*_e = r0 + e + ahead, H(Hdr_{e−1});
   σ_e = Ed25519(sk_G, H(Hdr_e)); the opening (salt_e, state_e) stays private until disclosure;
3. publish: witness cosign at the simulated receive time
   t_{e+1} + Δ_close + Δ_build + Δ_net (= t_{e+1} + 1.5 s by default; TR deadline t_{e+1} + 1.7 s);
4. at t(r*_e): β = drand randomness of r*_e;
   α_e = b"CTS/v1/vrf-in" ‖ H(Hdr_e) ‖ u64le(r*_e) ‖ β;  (Π_e, β_vrf) = ECVRF.prove(sk_V, α_e);
   K_sel = KDF(β_vrf); u_i = PRF(K_sel, e, i, L_i); Z_i = [u_i < τ_i] for payload units only;
5. retain AEAD-encrypted frames of Z_i = 1 units; Π_e is kept privately (only in disclosures);
6. state ← policy.update(state, record)  (controller / boost bookkeeping for e + 1).
The committed state of epoch e is ``policy.committed_state(state)`` (identity for honest policies);
it is what commit_fields/thresholds see and what the record (and hence a disclosure) carries.

Thresholds come from a pluggable ``ThresholdPolicy`` (protocol/policy.py, re-exported here).
"""
from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..crypto import ecvrf
from ..crypto import hashing as hs
from ..crypto.aead import EpochCipher
from ..crypto.prf import select_units, selection_key
from ..crypto.signing import KeySet
from ..units.batch import EpochBatch
from .beacon import BeaconRound, DrandRounds, EpochClock
from .header import GENESIS_PREV, Header, StateOpening, gw_id_from_pk, sign_header, state_salt
from .policy import (ConstantPolicy, FixedLambdaPolicy, Justification, PolicyState,  # noqa: F401 (re-export)
                     ThresholdPolicy)
from .store import MetadataStore, RetentionStore, prepare_batch
from .witness import Cosignature, Witness, WitnessRefusal

DOM_VRF_IN = b"CTS/v1/vrf-in"

__all__ = ["Timing", "EpochRecord", "Gateway", "vrf_input", "build_header", "make_opening", "run_selection",
           "ThresholdPolicy", "ConstantPolicy", "FixedLambdaPolicy", "PolicyState", "Justification"]


@dataclass(frozen=True)
class Timing:
    """Commit-slack components of §2.3 (seconds)."""
    delta_close: float = 0.5
    delta_build: float = 0.8
    delta_net: float = 0.2
    delta_sel: float = 0.5

    def send_offset(self) -> float:
        """Header arrival at the witness relative to t_{e+1}."""
        return self.delta_close + self.delta_build + self.delta_net


def vrf_input(hdr: Header, beta: bytes) -> bytes:
    """α_e = "CTS/v1/vrf-in" ‖ H(Hdr_e) ‖ u64le(r*_e) ‖ β_{r*_e}."""
    return DOM_VRF_IN + hdr.digest() + struct.pack("<Q", hdr.r_star) + beta


@dataclass
class EpochRecord:
    e: int
    batch: EpochBatch
    header: Header
    sigma: bytes
    store: MetadataStore
    state: PolicyState
    taus: list[int]
    leaves: list[bytes]
    opening: StateOpening | None = None     # (salt_e, state_e) of C^st_e — private until disclosure
    cosig: Cosignature | None = None
    refusal: str | None = None
    t_send: float | None = None
    beacon_round: BeaconRound | None = None
    vrf_proof: bytes | None = None          # Π_e — private until disclosure
    beta_vrf: bytes | None = None
    u: list[int] | None = None
    Z: list[bool] | None = None
    retained: list[int] = field(default_factory=list)
    cap: dict | None = None                  # zero-payload cap report (protocol/flood.py), if the gateway caps

    @property
    def selected(self) -> set[int]:
        return {i for i, z in enumerate(self.Z or []) if z}


def make_opening(k_st: bytes, e: int, fields: dict, boost_set) -> StateOpening:
    """state_e from policy.commit_fields(state) and the committed boost set; salt_e = PRF_{k_st}(e)."""
    return StateOpening(salt=state_salt(k_st, e), Lam_A=int(fields["Lam_A"]), Lam_S=int(fields["Lam_S"]),
                        D_prev=int(fields["D_prev"]), floor_mode=int(fields["floor_mode"]),
                        boost_set=tuple(boost_set), H_boost=fields["H_boost"])


def build_header(gw_id: bytes, clock: EpochClock, batch: EpochBatch, store: MetadataStore, R: bytes,
                 fields: dict, prev_hash: bytes, C_st: bytes) -> Header:
    tot = store.totals()
    e = batch.epoch
    return Header(gw_id=gw_id, e=e, t_e_us=clock.t_e_us(e), T_us=clock.period * 1_000_000, n=tot["n"],
                  n_pay=tot["n_pay"], sum_s_pay=tot["sum_s_pay"], sum_s=tot["sum_s"], R=R, H_store=store.digest(),
                  H_theta=fields["H_theta"], H_policy=fields["H_policy"], tau_phi=int(fields["tau_phi"]), C_st=C_st,
                  r_star=clock.r_star(e), H_prev=prev_hash)


def run_selection(sk_V: bytes, hdr: Header, beta: bytes, leaves: list[bytes], taus: list[int],
                  pay: list[bool]) -> tuple[bytes, bytes, list[int], list[bool]]:
    """(Π_e, β_vrf, u, Z) for a committed epoch once β = beacon randomness of r*_e is known."""
    pi = ecvrf.prove(sk_V, vrf_input(hdr, beta))
    beta_vrf = ecvrf.proof_to_hash(pi)
    u, Z = select_units(selection_key(beta_vrf), hdr.e, leaves, taus, pay)
    return pi, beta_vrf, u, Z


class Gateway:
    def __init__(self, keys: KeySet, clock: EpochClock, beacon: DrandRounds, policy: ThresholdPolicy,
                 witness: Witness | None, escrow_master: bytes, gw_id: bytes | None = None,
                 timing: Timing = Timing(), e0: int = 0, keep_frames: bool = True,
                 zero_cap: tuple[int, int | None] | None = None):
        """``keep_frames=False`` drops every unit's frames (and the non-retained units' ℓ lists) from
        the in-memory record after selection + policy update, as the real gateway does (retained
        content lives encrypted in ``self.retention``). Use it for long streams; the adversary
        simulators that need pre-selection content (A2 extra, A4, A5) require keep_frames=True.
        ``zero_cap=(k_dev, k_tot)`` applies the zero-payload flood cap of protocol/flood.py before commit."""
        self.keys = keys
        self.clock = clock
        self.beacon = beacon
        self.policy = policy
        self.witness = witness
        self.gw_id = gw_id or gw_id_from_pk(keys.pk_G)
        self.timing = timing
        self.retention = RetentionStore(EpochCipher(escrow_master))
        self.records: dict[int, EpochRecord] = {}
        self.prev_hash = GENESIS_PREV
        self.e0 = e0
        self.next_e = e0
        self.state = policy.initial_state(e0)
        self.keep_frames = keep_frames
        self.zero_cap = zero_cap

    # ------------------------------------------------------------------ registration
    def register(self, policy_effective_epoch: int | None = None, now: float | None = None):
        """Register keys and the policy with the witness (≥ 1 epoch before it takes effect)."""
        if self.witness is None:
            return None
        eff = self.e0 if policy_effective_epoch is None else policy_effective_epoch
        now = self.clock.t_e(eff) - 2 * self.clock.period if now is None else now
        greg = self.witness.register_gateway(self.gw_id, self.keys.pk_G, self.keys.pk_V, self.clock, now, self.e0)
        H_theta, H_policy = self.policy.registration()
        preg = self.witness.register_policy(self.gw_id, H_theta, H_policy, eff, now)
        return greg, preg

    def empty_batch(self, e: int) -> EpochBatch:
        return EpochBatch(epoch=e, t_start_us=self.clock.t_e_us(e), units=[])

    # ------------------------------------------------------------------ Alg. 1
    def commit(self, batch: EpochBatch) -> EpochRecord:
        e = batch.epoch
        if e != self.next_e:
            raise ValueError(f"epochs must be committed in order without gaps: expected {self.next_e}, got {e}")
        if self.state.e != e:
            raise ValueError("policy state epoch mismatch")
        cap = None
        if self.zero_cap is not None:
            from .flood import cap_zero_payload
            batch, cap = cap_zero_payload(batch, *self.zero_cap)
        prepare_batch(batch)
        leaves = [u.leaf for u in batch.units]
        R = hs.merkle_root(leaves)
        cstate = self.policy.committed_state(self.state)
        fields = self.policy.commit_fields(cstate)
        taus = [int(t) for t in self.policy.thresholds(batch, cstate)]
        if len(taus) != batch.n:
            raise ValueError("policy returned the wrong number of thresholds")
        # rev1: the thresholds are part of the committed store (H(store_e) in the signed header, before β)
        store = MetadataStore.from_batch(batch, taus=taus)
        opening = make_opening(self.keys.k_st, e, fields, cstate.boost_set)
        hdr = build_header(self.gw_id, self.clock, batch, store, R, fields, self.prev_hash, opening.commitment())
        sigma = sign_header(self.keys.sk_G, hdr)
        rec = EpochRecord(e=e, batch=batch, header=hdr, sigma=sigma, store=store, state=cstate, taus=taus,
                          leaves=leaves, opening=opening, cap=cap)
        self.records[e] = rec
        self.prev_hash = hdr.digest()
        self.next_e = e + 1
        return rec

    def publish(self, rec: EpochRecord, recv_time: float | None = None) -> Cosignature | None:
        if recv_time is None:
            recv_time = self.clock.t_e(rec.e + 1) + self.timing.send_offset()
        rec.t_send = recv_time
        if self.witness is None:
            return None
        try:
            rec.cosig = self.witness.cosign(rec.header, rec.sigma, recv_time)
            rec.refusal = None
        except WitnessRefusal as r:
            rec.cosig = None
            rec.refusal = str(r)
        return rec.cosig

    def select(self, rec: EpochRecord) -> EpochRecord:
        b = self.beacon.get(rec.header.r_star)
        rec.beacon_round = b
        pay = [u.pay for u in rec.batch.units]
        rec.vrf_proof, rec.beta_vrf, rec.u, rec.Z = run_selection(self.keys.sk_V, rec.header, b.randomness,
                                                                  rec.leaves, rec.taus, pay)
        rec.retained = [i for i, z in enumerate(rec.Z) if z]
        for i in rec.retained:
            self.retention.retain(rec.e, rec.batch.units[i])
        return rec

    def finish(self, rec: EpochRecord) -> None:
        self.state = self.policy.update(self.state, rec)
        if not self.keep_frames:
            keep = set(rec.retained)
            for u in rec.batch.units:
                u.frames = None
                if u.index not in keep:
                    u.pkt_leaves = None

    def run_epoch(self, batch: EpochBatch, recv_time: float | None = None) -> EpochRecord:
        rec = self.commit(batch)
        self.publish(rec, recv_time)
        self.select(rec)
        self.finish(rec)
        return rec

    def run(self, batches: Iterable[EpochBatch], recv_times: dict[int, float] | None = None) -> list[EpochRecord]:
        """Run consecutive epochs; epochs absent from ``batches`` get an empty (n_e = 0) header."""
        recv_times = recv_times or {}
        out = []
        for b in sorted(batches, key=lambda x: x.epoch):
            while self.next_e < b.epoch:
                out.append(self.run_epoch(self.empty_batch(self.next_e), recv_times.get(self.next_e)))
            out.append(self.run_epoch(b, recv_times.get(b.epoch)))
        return out
