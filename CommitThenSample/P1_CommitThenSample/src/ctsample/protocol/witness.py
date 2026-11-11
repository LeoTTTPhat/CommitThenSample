"""Witness W: timing-rule cosigner, append-only log and policy registry (DESIGN §2.1, §2.3, §3.3).

    ω_e = Ed25519(sk_W, BLAKE3("CTS/v1/wit" ‖ H(Hdr_e) ‖ f64le τ_W))
issued iff (TR) τ_W ≤ t(r*_e) − δ_W − ε_β, r*_e = r0 + e + ahead, σ_e verifies under the registered
pk_G, and no OTHER header was cosigned for (G, e) (resubmitting the identical header returns
the same cosignature). τ_W is the witness's own clock reading (``recv_time + clock_skew``).

Registrations (signed by sk_W, appended to the public log):
* gateway: (gw_id, pk_G, pk_V, clock params incl. r0);
* policy:  (gw_id, H(θ), H(policy), effective_epoch, τ_reg), accepted only if
  τ_reg ≤ t_{effective_epoch} − T (at least one epoch of lead time). The policy in force at
  epoch e is the registration with the largest effective_epoch ≤ e.

``enforce_tr=False`` simulates a deployment WITHOUT the timing rule (no deadline, no
once-per-epoch rule; e.g. anchoring every K epochs) — used by the grinding adversary A6.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from blake3 import blake3

from ..crypto.signing import SigningKey, verify as ed_verify
from .beacon import EpochClock
from .header import Header, verify_header

DOM_WIT = b"CTS/v1/wit"
DOM_REG = b"CTS/v1/reg"
DOM_GWREG = b"CTS/v1/gwreg"

__all__ = ["Cosignature", "PolicyRegistration", "GatewayRegistration", "Witness", "WitnessRefusal",
           "cosig_message", "verify_cosignature", "policy_in_force", "DOM_WIT"]


class WitnessRefusal(Exception):
    """The witness refuses to cosign / register (reason in args[0])."""


def cosig_message(h_hdr: bytes, tau_W: float) -> bytes:
    return blake3(DOM_WIT + h_hdr + struct.pack("<d", tau_W)).digest()


@dataclass(frozen=True)
class Cosignature:
    omega: bytes
    tau_W: float


def verify_cosignature(pk_W: bytes, hdr: Header, cos: Cosignature) -> bool:
    return ed_verify(pk_W, cos.omega, cosig_message(hdr.digest(), cos.tau_W))


@dataclass(frozen=True)
class PolicyRegistration:
    gw_id: bytes
    H_theta: bytes
    H_policy: bytes
    effective_epoch: int
    tau_reg: float
    sig: bytes = b""

    def message(self) -> bytes:
        return blake3(DOM_REG + self.gw_id + self.H_theta + self.H_policy
                      + struct.pack("<Id", self.effective_epoch, self.tau_reg)).digest()

    def verify(self, pk_W: bytes) -> bool:
        return ed_verify(pk_W, self.sig, self.message())


@dataclass(frozen=True)
class GatewayRegistration:
    gw_id: bytes
    pk_G: bytes
    pk_V: bytes
    clock: EpochClock
    tau_reg: float
    e0: int = 0                 # first epoch of the header chain (its H_prev is GENESIS_PREV)
    sig: bytes = b""

    def message(self) -> bytes:
        c = self.clock
        return blake3(DOM_GWREG + self.gw_id + self.pk_G + self.pk_V
                      + struct.pack("<QQQdIdI", c.r0, c.genesis, c.period, c.delta_c, c.ahead, self.tau_reg,
                                    self.e0)).digest()

    def verify(self, pk_W: bytes) -> bool:
        return ed_verify(pk_W, self.sig, self.message())


def policy_in_force(regs: list[PolicyRegistration], gw_id: bytes, e: int) -> PolicyRegistration | None:
    best = None
    for r in regs:
        if r.gw_id == gw_id and r.effective_epoch <= e and (best is None or r.effective_epoch >= best.effective_epoch):
            best = r
    return best


@dataclass
class _LogEntry:
    kind: str
    data: object


class Witness:
    def __init__(self, sk_W: SigningKey, delta_W: float = 0.3, eps_beta: float = 1.0,
                 enforce_tr: bool = True, clock_skew: float = 0.0):
        if abs(clock_skew) > delta_W:
            raise ValueError("witness clock skew must be within delta_W")
        self.sk_W = sk_W
        self.delta_W = delta_W
        self.eps_beta = eps_beta
        self.enforce_tr = enforce_tr
        self.clock_skew = clock_skew
        self.gateways: dict[bytes, GatewayRegistration] = {}
        self.policies: list[PolicyRegistration] = []
        self._cosigned: dict[tuple[bytes, int], tuple[bytes, Cosignature]] = {}
        self.log: list[_LogEntry] = []
        self.refusals: list[tuple[bytes, int, str]] = []

    @property
    def pk(self) -> bytes:
        return self.sk_W.pk

    # ------------------------------------------------------------------ registry
    def register_gateway(self, gw_id: bytes, pk_G: bytes, pk_V: bytes, clock: EpochClock,
                         now: float, e0: int = 0) -> GatewayRegistration:
        if gw_id in self.gateways:
            raise WitnessRefusal("gateway already registered")
        reg = GatewayRegistration(gw_id, pk_G, pk_V, clock, float(now), int(e0))
        reg = GatewayRegistration(gw_id, pk_G, pk_V, clock, float(now), int(e0), self.sk_W.sign(reg.message()))
        self.gateways[gw_id] = reg
        self.log.append(_LogEntry("gateway", reg))
        return reg

    def register_policy(self, gw_id: bytes, H_theta: bytes, H_policy: bytes, effective_epoch: int,
                        now: float) -> PolicyRegistration:
        g = self.gateways.get(gw_id)
        if g is None:
            raise WitnessRefusal("unknown gateway")
        latest = g.clock.t_e(effective_epoch) - g.clock.period      # ≥ 1 epoch of lead time
        if now > latest:
            raise WitnessRefusal(f"policy registered too late: now={now} > t_eff − T = {latest}")
        reg = PolicyRegistration(gw_id, H_theta, H_policy, int(effective_epoch), float(now))
        reg = PolicyRegistration(gw_id, H_theta, H_policy, int(effective_epoch), float(now),
                                 self.sk_W.sign(reg.message()))
        self.policies.append(reg)
        self.log.append(_LogEntry("policy", reg))
        return reg

    def policy_in_force(self, gw_id: bytes, e: int) -> PolicyRegistration | None:
        return policy_in_force(self.policies, gw_id, e)

    # ------------------------------------------------------------------ cosigning
    def cosign(self, hdr: Header, sigma: bytes, recv_time: float) -> Cosignature:
        """Cosign Hdr_e received at true time ``recv_time``; raises WitnessRefusal otherwise."""
        g = self.gateways.get(hdr.gw_id)
        key = (hdr.gw_id, hdr.e)

        def refuse(reason: str):
            self.refusals.append((hdr.gw_id, hdr.e, reason))
            raise WitnessRefusal(reason)

        if g is None:
            refuse("unknown gateway")
        if not verify_header(g.pk_G, hdr, sigma):
            refuse("invalid gateway signature")
        tau_W = float(recv_time) + self.clock_skew
        h = hdr.digest()
        if self.enforce_tr:
            if hdr.r_star != g.clock.r_star(hdr.e):
                refuse(f"r*_e = {hdr.r_star} != r0 + e + {g.clock.ahead}")
            deadline = g.clock.t_round(hdr.r_star) - self.delta_W - self.eps_beta
            if tau_W > deadline:
                refuse(f"late header: tau_W={tau_W:.3f} > deadline {deadline:.3f}")
            prev = self._cosigned.get(key)
            if prev is not None:
                if prev[0] == h:
                    return prev[1]
                refuse("a different header was already cosigned for this epoch")
        cos = Cosignature(self.sk_W.sign(cosig_message(h, tau_W)), tau_W)
        if key not in self._cosigned:
            self._cosigned[key] = (h, cos)
        self.log.append(_LogEntry("cosign", (hdr.gw_id, hdr.e, h, cos)))
        return cos

    # ------------------------------------------------------------------ public log
    def export_registrations(self, gw_id: bytes) -> tuple[GatewayRegistration | None, list[PolicyRegistration]]:
        return self.gateways.get(gw_id), [p for p in self.policies if p.gw_id == gw_id]

    def cosigned_header_hash(self, gw_id: bytes, e: int) -> bytes | None:
        v = self._cosigned.get((gw_id, e))
        return None if v is None else v[0]
