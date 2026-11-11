"""Verifier V — Alg. 2 of DESIGN §3.8 (V1–V7; V8 statements are exposed through a hook).

Every check emits a ``CheckResult(check, name, status, epoch, evidence)`` with status
PASS / FAIL / WARN / SKIP:

V1 header: σ_e (pk_G); header fields (gw_id, e, t_e, T, ver); r*_e = r0 + e + ahead; ω_e (pk_W);
   TR deadline τ_W ≤ t(r*_e) − δ_W − ε_β; chain H(Hdr_{e−1}) and no gaps (a missing header is a
   FAIL, listed with the counter volumes of the gap); policy (H(θ), H(policy)) in force at e,
   registered (pk_W-signed) ≥ 1 epoch before it took effect, and equal to the verifier's policy.
V2 beacon: round = r*_e; randomness = SHA-256(signature); chain link to r*_e − 1; equality with a
   trusted local drand copy when given; BLS → SKIP (no py_ecc).
V3 ECVRF.verify(pk_V, α_e, Π_e, validate_key=True) → β_vrf.
V4 H(metadata store) and R_e recomputed from the store; header totals; optional meta-codec
   consistency; opening of the hiding state commitment C^st_e by the disclosed (salt_e, state_e)
   ('state_opening'), then commit_fields(state_from_commit(opened header, boost set, justification
   records)) must reproduce the opened state ('state_fields') and H_boost ('boost_opening');
   floor τ^φ = ⌊φ0·2^64⌋ of the registered policy;
   rev1 (committed thresholds, default): the per-unit τ_i are read from the store, which H(store_e)
   commits before β; 'committed_taus' checks the floor τ_i ≥ τ^φ_i (policy.floor_thresholds: τ^φ, or the
   flood-guard cap of floor mode 1 — computed from the epoch's own store) on every payload unit — this
   is all that selection integrity and the statements need (Thm. 2(b)); 'tau_compliance' recomputes τ_i
   through the SAME ThresholdPolicy when its rolling history is available (prime bundle) and reports a
   deviation as WARN (Thm. 2(e): harmless to validity), or NCHK ("compliance not checkable") when the
   history is missing — a voided or withheld epoch therefore affects only the compliance checks of the
   next ≤ 119 epochs, never their selection verification;
   semantics="rev0" (legacy, and the only mode for rev0 stores without a threshold column): τ_i :=
   the policy recomputation, which must succeed (V4 'thresholds' FAIL otherwise) and equal the committed
   column if any; discloser's claimed τ compared with the τ in force (FAIL on mismatch); policy-
   compliance deviations (WARN); budget report (no FAIL).
V5 u_i and Z_i = [u_i < τ_i] for all payload units → S_e; discloser's claimed retained set compared.
V6 for requested i ∈ S_e: frames present (else MISSING); frame hashes → root^pkt_i → L_i → R_e
   (else MISMATCH; also Σ(caplen+16) = s_i and per-direction packet counts); presented units with
   Z_i = 0 → EXTRA.  𝒲_e ← MISSING ∪ MISMATCH ∪ (all requested units if e failed V1–V4).
V7 per-(device, epoch) committed packets/bytes (store) vs the pk_I-signed counter record with
   tolerance ε_cell(n) → flagged cells (FAIL), counter signature (FAIL), missing record (WARN).
V8 ``report.v8_inputs()`` gives τ, Z, pay, 𝒲, floor_mode and V7 flags per epoch; an optional
   ``statements_hook(report)`` (estimation/policy.py of P1-C) is called at the end.
"""
from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from ..crypto import ecvrf
from ..crypto import hashing as hs
from ..crypto.prf import select_units, selection_key
from ..design.intpi import TWO64
from .beacon import DrandRounds, EpochClock, check_round_link, check_round_randomness
from .counters import EpsCell, compare_cells
from .disclosure import Disclosure, EpochDisclosure
from .epoch import vrf_input
from .header import GENESIS_PREV, VER, OpenedHeader, verify_header
from .policy import ComplianceContext, ThresholdPolicy
from .witness import PolicyRegistration, policy_in_force, verify_cosignature

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
NCHK = "NCHK"          # "not checkable" (rev1): a compliance check whose inputs are not in the disclosure

__all__ = ["CheckResult", "EpochVerdict", "TrustAnchors", "VerificationReport", "Verifier",
           "PASS", "FAIL", "WARN", "SKIP", "NCHK", "SEMANTICS"]

SEMANTICS = ("rev1", "rev0")


@dataclass
class CheckResult:
    check: str
    name: str
    status: str
    epoch: int | None = None
    evidence: dict = field(default_factory=dict)

    def line(self) -> str:
        ev = ", ".join(f"{k}={v}" for k, v in list(self.evidence.items())[:6])
        return f"[{self.status}] {self.check}.{self.name} e={self.epoch} {ev}"


@dataclass
class EpochVerdict:
    e: int
    present: bool = False
    v1_ok: bool = False
    v2_ok: bool = False
    v3_ok: bool = False
    v4_ok: bool = False
    beta_vrf: bytes | None = None
    taus: list[int] | None = None
    pay: list[bool] | None = None
    u: list[int] | None = None
    Z: list[bool] | None = None
    selected: set[int] = field(default_factory=set)
    requested: set[int] = field(default_factory=set)
    missing: set[int] = field(default_factory=set)
    extra: set[int] = field(default_factory=set)
    mismatch: set[int] = field(default_factory=set)
    W: set[int] = field(default_factory=set)
    W_all: bool = False                    # every unit of the epoch is in 𝒲 (epoch failed V1–V4)
    deviations: list[dict] = field(default_factory=list)
    cells: list[dict] = field(default_factory=list)
    floor_mode: int = 0
    expected_cost: float | None = None
    realised_cost: int | None = None

    @property
    def failed(self) -> bool:
        return not (self.present and self.v1_ok and self.v2_ok and self.v3_ok and self.v4_ok)


@dataclass
class TrustAnchors:
    gw_id: bytes
    pk_G: bytes
    pk_V: bytes
    pk_W: bytes
    clock: EpochClock
    e0: int = 0
    pk_I: bytes | None = None
    delta_W: float = 0.3
    eps_beta: float = 1.0
    policy_registrations: list[PolicyRegistration] = field(default_factory=list)
    beacon: DrandRounds | None = None

    @classmethod
    def from_disclosure(cls, disc: Disclosure, pk_W: bytes, pk_I: bytes | None = None,
                        beacon: DrandRounds | None = None, delta_W: float = 0.3,
                        eps_beta: float = 1.0) -> "TrustAnchors":
        """Keys/clock from the pk_W-signed registrations carried in the disclosure."""
        g = disc.gateway_registration
        if g is None or g.gw_id != disc.gw_id or not g.verify(pk_W):
            raise ValueError("gateway registration missing or not signed by the witness")
        regs = [r for r in disc.policy_registrations if r.gw_id == g.gw_id and r.verify(pk_W)]
        return cls(gw_id=g.gw_id, pk_G=g.pk_G, pk_V=g.pk_V, pk_W=pk_W, clock=g.clock, e0=g.e0, pk_I=pk_I,
                   delta_W=delta_W, eps_beta=eps_beta, policy_registrations=regs, beacon=beacon)

    @classmethod
    def from_witness(cls, witness, gw_id: bytes, pk_I: bytes | None = None,
                     beacon: DrandRounds | None = None) -> "TrustAnchors":
        g, regs = witness.export_registrations(gw_id)
        if g is None:
            raise ValueError("gateway not registered")
        return cls(gw_id=gw_id, pk_G=g.pk_G, pk_V=g.pk_V, pk_W=witness.pk, clock=g.clock, e0=g.e0, pk_I=pk_I,
                   delta_W=witness.delta_W, eps_beta=witness.eps_beta, policy_registrations=list(regs), beacon=beacon)


@dataclass
class VerificationReport:
    results: list[CheckResult] = field(default_factory=list)
    epochs: dict[int, EpochVerdict] = field(default_factory=dict)
    budget: dict = field(default_factory=dict)
    statements: object = None

    def add(self, check: str, name: str, status: str, epoch: int | None = None, **evidence) -> CheckResult:
        r = CheckResult(check, name, status, epoch, evidence)
        self.results.append(r)
        return r

    def failures(self, check: str | None = None, name: str | None = None) -> list[CheckResult]:
        return [r for r in self.results if r.status == FAIL and (check is None or r.check == check)
                and (name is None or r.name == name)]

    def warnings(self, check: str | None = None, name: str | None = None) -> list[CheckResult]:
        return [r for r in self.results if r.status == WARN and (check is None or r.check == check)
                and (name is None or r.name == name)]

    def not_checkable(self, check: str | None = None, name: str | None = None) -> list[CheckResult]:
        return [r for r in self.results if r.status == NCHK and (check is None or r.check == check)
                and (name is None or r.name == name)]

    def flagged(self, check: str, name: str | None = None, include_warn: bool = False) -> bool:
        return bool(self.failures(check, name) or (include_warn and self.warnings(check, name)))

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.results if r.status == FAIL)

    def summary(self) -> dict:
        out: dict[str, dict[str, int]] = {}
        for r in self.results:
            out.setdefault(r.check, {}).setdefault(r.status, 0)
            out[r.check][r.status] += 1
        return out

    def W(self) -> dict[int, set[int]]:
        return {e: set(v.W) for e, v in self.epochs.items()}

    def v8_inputs(self) -> dict:
        """Per-epoch data for the statement layer (estimation/policy.py)."""
        out = {}
        for e, v in self.epochs.items():
            out[e] = {"tau": v.taus, "Z": v.Z, "pay": v.pay, "W": sorted(v.W), "W_all": v.W_all,
                      "requested": sorted(v.requested), "failed": v.failed, "floor_mode": v.floor_mode,
                      "flagged_devices": [c["device_id"] for c in v.cells if c.get("flagged")],
                      "missing": sorted(v.missing), "extra": sorted(v.extra), "mismatch": sorted(v.mismatch)}
        return out

    def table(self, only_non_pass: bool = False) -> str:
        return "\n".join(r.line() for r in self.results if not (only_non_pass and r.status == PASS))


class Verifier:
    def __init__(self, trust: TrustAnchors, policy: ThresholdPolicy, eps_cell: EpsCell | None = None,
                 meta_codec=None, statements_hook: Callable[[VerificationReport], object] | None = None,
                 semantics: str = "rev1", recompute_compliance: bool = True):
        """``semantics``: "rev1" (default) — validity from the committed τ_i + epoch-local floor; τ recomputation
        is a compliance check (WARN / NCHK); a chain gap is attributed to the missing epoch (the successor's own
        header stands on σ, ω and TR; V1 'chain' WARN). "rev0" — the pre-revision verifier: τ_i := the policy
        recomputation (FAIL if the rolling history is incomplete or it differs from a committed column) and a
        chain gap FAILs the successor; it is the only mode that accepts rev0 stores (no threshold column).
        ``recompute_compliance=False`` skips the (optional) τ recomputation in rev1 (fast verification)."""
        if semantics not in SEMANTICS:
            raise ValueError(f"semantics must be one of {SEMANTICS}")
        self.trust = trust
        self.policy = policy
        self.eps = eps_cell
        self.codec = meta_codec
        self.hook = statements_hook
        self.semantics = semantics
        self.recompute_compliance = bool(recompute_compliance)

    # ======================================================================== driver
    def verify(self, disc: Disclosure, epochs: Iterable[int] | None = None,
               request: dict[int, Iterable[int]] | None = None,
               counters: dict | None = None) -> VerificationReport:
        """Verify epochs E' (default: those in the disclosure). ``request[e]`` is the investigator's
        own request (default: the request recorded in the disclosure). ``counters`` = counter records
        obtained directly from I (default: the I-signed records carried in the disclosure)."""
        if counters is not None:
            disc = Disclosure(gw_id=disc.gw_id, epochs=disc.epochs, context=disc.context,
                              gateway_registration=disc.gateway_registration,
                              policy_registrations=disc.policy_registrations, counters=dict(counters))
        rep = VerificationReport()
        E = sorted(set(epochs) if epochs is not None else set(disc.epochs))
        ctx = ComplianceContext(headers={})
        if hasattr(self.policy, "set_chain_start"):
            self.policy.set_chain_start(self.trust.e0)      # no silently truncated rolling window
        for e in E:
            d = disc.epochs.get(e)
            v = EpochVerdict(e=e)
            rep.epochs[e] = v
            if request is not None and e in request:
                v.requested = set(int(i) for i in request[e])
            elif d is not None:
                v.requested = set(d.requested)
            if d is not None and (d.e != e or d.header.e != e):
                rep.add("V1", "epoch_label", FAIL, e, reason="disclosure entry is labelled with another epoch",
                        entry_e=d.e, header_e=d.header.e)
                d = None
            if d is None:
                self._missing_epoch(rep, disc, e, v)
                continue
            v.present = True
            self._v1(rep, disc, d, v)
            self._v2(rep, d, v)
            self._v3(rep, d, v)
            self._v4(rep, d, v, ctx)
            if v.v3_ok and v.taus is not None:
                self._v5(rep, d, v, ctx)
                self._v6(rep, d, v, ctx)
            else:
                rep.add("V5", "selection", SKIP, e, reason="V3 or V4 failed")
                rep.add("V6", "evidence", SKIP, e, reason="V3 or V4 failed")
            if v.failed:
                v.W_all = True
                v.W |= v.requested
            self._v7(rep, disc, d, v)
        self._compliance(rep, disc, E, ctx)
        self._budget(rep)
        if self.hook is not None:
            rep.statements = self.hook(rep)
        return rep

    # ======================================================================== helpers
    def _missing_epoch(self, rep: VerificationReport, disc: Disclosure, e: int, v: EpochVerdict) -> None:
        vol = None
        rec = disc.counters.get(e)
        if rec is not None and self.trust.pk_I is not None and rec.verify(self.trust.pk_I):
            vol = {c.device_id: (c.pkts, c.bytes) for c in rec.cells}
        rep.add("V1", "header_present", FAIL, e, reason="no header/disclosure for this epoch (withheld epoch or chain gap)",
                counter_volumes=vol)
        v.W_all = True
        v.W |= v.requested
        if self.eps is not None and vol:
            cells = compare_cells({}, vol, self.eps)
            v.cells = cells
            flagged = [c for c in cells if c["flagged"]]
            rep.add("V7", "cells", FAIL if flagged else PASS, e, flagged=[c["device_id"] for c in flagged],
                    note="epoch without header: committed counts are 0")

    # ------------------------------------------------------------------------ V1
    def _v1(self, rep: VerificationReport, disc: Disclosure, d: EpochDisclosure, v: EpochVerdict) -> None:
        t, e, h = self.trust, d.e, d.header
        ok = True
        if verify_header(t.pk_G, h, d.sigma):
            rep.add("V1", "signature", PASS, e)
        else:
            rep.add("V1", "signature", FAIL, e, reason="σ_e does not verify under pk_G")
            ok = False
        bad = {}
        if h.e != e:
            bad["e"] = h.e
        if h.gw_id != t.gw_id:
            bad["gw_id"] = h.gw_id.hex()
        if h.ver != VER:
            bad["ver"] = h.ver
        if h.T_us != t.clock.period * 1_000_000:
            bad["T_us"] = h.T_us
        if h.t_e_us != t.clock.t_e_us(e):
            bad["t_e_us"] = (h.t_e_us, t.clock.t_e_us(e))
        rep.add("V1", "header_fields", FAIL if bad else PASS, e, **({"mismatch": bad} if bad else {}))
        ok &= not bad
        want_r = t.clock.r_star(e)
        if h.r_star == want_r:
            rep.add("V1", "r_star", PASS, e, r_star=h.r_star)
        else:
            rep.add("V1", "r_star", FAIL, e, committed=h.r_star, expected=want_r)
            ok = False
        if d.cosig is None:
            rep.add("V1", "cosignature", FAIL, e, reason="no witness cosignature ω_e")
            rep.add("V1", "deadline", FAIL, e, reason="no τ_W")
            ok = False
        else:
            if verify_cosignature(t.pk_W, h, d.cosig):
                rep.add("V1", "cosignature", PASS, e)
            else:
                rep.add("V1", "cosignature", FAIL, e, reason="ω_e does not verify under pk_W for this header")
                ok = False
            dl = t.clock.t_round(want_r) - t.delta_W - t.eps_beta
            if d.cosig.tau_W <= dl:
                rep.add("V1", "deadline", PASS, e, tau_W=d.cosig.tau_W, deadline=dl)
            else:
                rep.add("V1", "deadline", FAIL, e, tau_W=d.cosig.tau_W, deadline=dl)
                ok = False
        # chain link / gaps
        if e == t.e0:
            if h.H_prev == GENESIS_PREV:
                rep.add("V1", "chain", PASS, e, note="chain start")
            else:
                rep.add("V1", "chain", FAIL, e, reason="first epoch must link to GENESIS_PREV")
                ok = False
        else:
            prev = disc.header_of(e - 1)
            if prev is None and self.semantics == "rev0":
                rep.add("V1", "chain", FAIL, e, reason=f"header of epoch {e - 1} not available (gap)")
                ok = False
            elif prev is None:
                # rev1: the gap is evidence about epoch e−1 (missing / withheld: its units are not verifiable),
                # not about e, whose own header is authenticated by σ_e, ω_e and the TR deadline; a fork would
                # need a second cosigned header for (G, e) or a link to an un-cosigned predecessor (FAIL below)
                rep.add("V1", "chain", WARN, e, reason=f"header of epoch {e - 1} not available (gap): epoch {e - 1} "
                        "is missing or withheld; this epoch is verified on its own cosigned header", gap_before=e - 1)
            else:
                ph, psig, pcos = prev
                if ph.e != e - 1 or not verify_header(t.pk_G, ph, psig):
                    rep.add("V1", "chain", FAIL, e, reason="previous header is not an authentic header of e−1")
                    ok = False
                elif (pcos is None or not verify_cosignature(t.pk_W, ph, pcos)) and \
                        self.semantics == "rev1" and h.H_prev == ph.digest():
                    # rev1: the predecessor was committed but never cosigned (e.g. a missed commit window): epoch
                    # e−1 is void (V1 FAIL there), the link itself is consistent; e stands on its own ω_e
                    rep.add("V1", "chain", WARN, e, reason=f"previous header (epoch {e - 1}) is not witness-cosigned: "
                            f"epoch {e - 1} is void; this epoch is verified on its own cosigned header", gap_before=e - 1)
                elif pcos is None or not verify_cosignature(t.pk_W, ph, pcos):
                    rep.add("V1", "chain", FAIL, e, reason="previous header is not witness-cosigned")
                    ok = False
                elif h.H_prev != ph.digest():
                    rep.add("V1", "chain", FAIL, e, reason="H(Hdr_{e−1}) link broken (gap or fork)")
                    ok = False
                else:
                    rep.add("V1", "chain", PASS, e)
        # policy registration
        reg = policy_in_force(t.policy_registrations, t.gw_id, e)
        mine = self.policy.registration()
        if reg is None:
            rep.add("V1", "policy_registered", FAIL, e, reason="no policy registered for this epoch")
            ok = False
        else:
            problems = []
            if not reg.verify(t.pk_W):
                problems.append("registration not signed by the witness")
            if (reg.H_theta, reg.H_policy) != (h.H_theta, h.H_policy):
                problems.append("header H(θ)/H(policy) differ from the policy in force")
            if reg.tau_reg > t.clock.t_e(reg.effective_epoch) - t.clock.period:
                problems.append("policy registered with less than one epoch of lead time")
            if (reg.H_theta, reg.H_policy) != mine:
                problems.append("verifier's policy object differs from the registered policy")
            if problems:
                rep.add("V1", "policy_registered", FAIL, e, problems=problems)
                ok = False
            else:
                rep.add("V1", "policy_registered", PASS, e, effective_epoch=reg.effective_epoch)
        v.v1_ok = ok

    # ------------------------------------------------------------------------ V2
    def _v2(self, rep: VerificationReport, d: EpochDisclosure, v: EpochVerdict) -> None:
        e, h, b = d.e, d.header, d.beacon_round
        ok = True
        if b is None or b.round != h.r_star:
            rep.add("V2", "round", FAIL, e, reason="beacon round r*_e missing or wrong",
                    got=None if b is None else b.round, expected=h.r_star)
            v.v2_ok = False
            return
        rep.add("V2", "round", PASS, e, round=b.round)
        if check_round_randomness(b):
            rep.add("V2", "randomness", PASS, e)
        else:
            rep.add("V2", "randomness", FAIL, e, reason="randomness != SHA-256(signature)")
            ok = False
        prev = d.beacon_prev
        tb = self.trust.beacon
        if prev is None and tb is not None and (b.round - 1) in tb:
            prev = tb.get(b.round - 1)
        if prev is None:
            rep.add("V2", "chain_link", WARN, e, reason="round r*_e − 1 not available")
        elif check_round_link(prev, b):
            rep.add("V2", "chain_link", PASS, e)
        else:
            rep.add("V2", "chain_link", FAIL, e, reason="previous_signature(r*) != signature(r* − 1)")
            ok = False
        if tb is not None and b.round in tb:
            tr = tb.get(b.round)
            if (tr.randomness, tr.signature, tr.previous_signature) == (b.randomness, b.signature, b.previous_signature):
                rep.add("V2", "trusted_copy", PASS, e)
            else:
                rep.add("V2", "trusted_copy", FAIL, e, reason="round differs from the trusted drand copy")
                ok = False
        else:
            rep.add("V2", "trusted_copy", SKIP, e)
        rep.add("V2", "bls", SKIP, e, reason="BLS verification unavailable (py_ecc not installed)")
        v.v2_ok = ok

    # ------------------------------------------------------------------------ V3
    def _v3(self, rep: VerificationReport, d: EpochDisclosure, v: EpochVerdict) -> None:
        e = d.e
        if d.vrf_proof is None or d.beacon_round is None:
            rep.add("V3", "vrf", FAIL, e, reason="Π_e or beacon round missing")
            v.v3_ok = False
            return
        alpha = vrf_input(d.header, d.beacon_round.randomness)
        ok, beta_vrf, reason = ecvrf.verify_detailed(self.trust.pk_V, alpha, d.vrf_proof, validate_key=True)
        if ok:
            rep.add("V3", "vrf", PASS, e)
            v.beta_vrf = beta_vrf
        else:
            rep.add("V3", "vrf", FAIL, e, reason=reason)
        v.v3_ok = ok

    # ------------------------------------------------------------------------ V4
    def _v4(self, rep: VerificationReport, d: EpochDisclosure, v: EpochVerdict, ctx: ComplianceContext) -> None:
        e, h, st = d.e, d.header, d.store
        if st is None or st.e != e:
            rep.add("V4", "store_present", FAIL, e, reason="metadata store missing or for another epoch")
            v.v4_ok = False
            return
        ok = True
        hs_ = st.digest()
        if hs_ == h.H_store:
            rep.add("V4", "store_hash", PASS, e)
        else:
            rep.add("V4", "store_hash", FAIL, e, reason="H(metadata store) != committed value")
            ok = False
        try:
            leaves = st.leaves()
            R = hs.merkle_root(leaves)
        except ValueError as ex:
            rep.add("V4", "root", FAIL, e, reason=f"store not parseable: {ex}")
            v.v4_ok = False
            return
        if R == h.R:
            rep.add("V4", "root", PASS, e, n=st.n)
        else:
            rep.add("V4", "root", FAIL, e, reason="R_e recomputed from the store != committed R_e")
            ok = False
        tot = st.totals()
        bad = {k: (tot[k], getattr(h, hk)) for k, hk in (("n", "n"), ("n_pay", "n_pay"), ("sum_s", "sum_s"),
                                                         ("sum_s_pay", "sum_s_pay")) if tot[k] != getattr(h, hk)}
        rep.add("V4", "totals", FAIL if bad else PASS, e, **({"mismatch": bad} if bad else {}))
        ok &= not bad
        if self.codec is not None:
            incons = []
            for i, r in enumerate(st.records):
                try:
                    f = self.codec.decode(r.meta, r.pay)
                except Exception as ex:                 # noqa: BLE001 — any decode error is evidence
                    incons.append((i, f"decode error: {ex}"))
                    continue
                for k, val in f.items():
                    if not hasattr(r, k):
                        continue
                    have = getattr(r, k)
                    if isinstance(have, tuple):
                        if tuple(have) != tuple(val):
                            incons.append((i, k))
                    elif have != val:
                        incons.append((i, k))
            rep.add("V4", "meta_consistency", FAIL if incons else PASS, e, **({"units": incons[:20]} if incons else {}))
            ok &= not incons
        else:
            rep.add("V4", "meta_consistency", SKIP, e, reason="no meta codec")
        # ---- open the hiding state commitment C^st_e (Thm. 2(d)): the design state is not public
        op = d.opening
        try:
            opened = op is not None and op.commitment() == h.C_st
        except (ValueError, TypeError, struct.error):
            opened = False
        if not opened:
            rep.add("V4", "state_opening", FAIL, e,
                    reason="no opening of C^st_e" if op is None else "(salt_e, state_e) does not open C^st_e")
            v.v4_ok = False
            return
        rep.add("V4", "state_opening", PASS, e, boost_set=list(op.boost_set))
        oh = OpenedHeader(h, op)
        ctx.headers[e] = oh
        v.floor_mode = op.floor_mode
        state = self.policy.state_from_commit(oh, op.boost_set, d.justifications)
        ctx.states[e] = state
        try:
            fv = self.policy.commit_fields(state)
        except Exception as ex:                         # noqa: BLE001
            fv = {"error": str(ex)}
        if fv.get("H_boost") == op.H_boost:
            rep.add("V4", "boost_opening", PASS, e)
        else:
            rep.add("V4", "boost_opening", FAIL, e,
                    reason="disclosed boost set / justification records do not reproduce the committed H_boost")
            ok = False
        bad_f = {k: (fv.get(k), want) for k, want in (("Lam_A", op.Lam_A), ("Lam_S", op.Lam_S), ("D_prev", op.D_prev),
                                                      ("floor_mode", op.floor_mode), ("tau_phi", h.tau_phi),
                                                      ("H_theta", h.H_theta), ("H_policy", h.H_policy))
                 if fv.get(k) != want}
        if bad_f:
            rep.add("V4", "state_fields", FAIL, e, reason="commit_fields(state_from_commit(·)) != committed values",
                    mismatch={k: str(x) for k, x in bad_f.items()})
            ok = False
        else:
            rep.add("V4", "state_fields", PASS, e)
        want_phi = self.policy.expected_tau_phi()
        if h.tau_phi == want_phi:
            rep.add("V4", "floor", PASS, e, tau_phi=h.tau_phi)
        else:
            rep.add("V4", "floor", FAIL, e, committed=h.tau_phi, expected=want_phi)
            ok = False
        batch = st.to_batch(h.t_e_us, codec=self.codec)
        legacy = not st.committed
        if legacy and self.semantics != "rev0":
            rep.add("V4", "committed_taus", FAIL, e, reason="the store commits no per-unit thresholds (rev0 format); "
                    "verify it with semantics='rev0'")
            v.v4_ok = False
            return
        if not legacy and st.taus is None:
            rep.add("V4", "committed_taus", FAIL, e, reason="threshold column withheld (redacted store disclosed "
                    "for a requested epoch)")
            v.v4_ok = False
            return
        recomputed, why = None, None
        if self.recompute_compliance or self.semantics == "rev0":
            try:
                recomputed = [int(x) for x in self.policy.thresholds(batch, state)]
                if len(recomputed) != st.n:
                    raise ValueError("policy returned the wrong number of thresholds")
            except Exception as ex:                     # noqa: BLE001 — incomplete rolling history, bad policy, ...
                recomputed, why = None, f"{type(ex).__name__}: {ex}"
        if self.semantics == "rev0":
            # pre-revision semantics: the recomputed τ IS the design (needs the 119-epoch history)
            if recomputed is None:
                rep.add("V4", "thresholds", FAIL, e, reason=f"policy failed: {why}")
                v.v4_ok = False
                return
            taus = recomputed
            bad_t = [i for i, (t_, r) in enumerate(zip(taus, st.records))
                     if (r.pay and not (h.tau_phi <= t_ <= TWO64)) or (not r.pay and t_ != 0)]
            if bad_t:
                rep.add("V4", "thresholds", FAIL, e, reason="τ outside [τ^φ, 2^64] (payload) or ≠ 0 (zero-payload)",
                        units=bad_t[:20])
                ok = False
            elif not legacy and recomputed != list(st.taus):
                diff = [i for i, (a, b) in enumerate(zip(recomputed, st.taus)) if a != b]
                rep.add("V4", "thresholds", FAIL, e, reason="recomputed τ differ from the committed τ (rev0 semantics)",
                        n_diff=len(diff), units=diff[:20])
                ok = False
            else:
                rep.add("V4", "thresholds", PASS, e, n_pay=st.n_pay)
        else:
            taus = list(st.taus)
            try:
                floors = [int(x) for x in self.policy.floor_thresholds(batch, state)]
                if len(floors) != st.n:
                    raise ValueError("floor_thresholds returned the wrong number of floors")
            except Exception as ex:                     # noqa: BLE001
                rep.add("V4", "committed_taus", FAIL, e, reason=f"floor not computable: {type(ex).__name__}: {ex}")
                v.v4_ok = False
                return
            low = [i for i, (t_, f_, r) in enumerate(zip(taus, floors, st.records)) if r.pay and t_ < f_]
            capped = sum(1 for t_, r in zip(taus, st.records) if r.pay and t_ < h.tau_phi)
            if low:
                rep.add("V4", "committed_taus", FAIL, e, reason="committed τ_i below the floor τ^φ_i",
                        n_below=len(low), units=low[:20], floor_mode=op.floor_mode)
                ok = False
            else:
                rep.add("V4", "committed_taus", PASS, e, n_pay=st.n_pay, n_capped=capped, floor_mode=op.floor_mode)
            if recomputed is None and not self.recompute_compliance:
                rep.add("V4", "tau_compliance", SKIP, e, reason="τ recomputation disabled")
            elif recomputed is None:
                rep.add("V4", "tau_compliance", NCHK, e, reason="compliance not checkable: the policy cannot recompute "
                        f"τ from the disclosure (e.g. rolling history incomplete: prime bundle absent) [{why}]")
            else:
                diff = [i for i, (a, b) in enumerate(zip(recomputed, taus)) if a != b]
                if diff:
                    rep.add("V4", "tau_compliance", WARN, e, reason="committed τ deviate from the registered policy "
                            "(harmless to validity, Thm. 2(e))", n_diff=len(diff), units=diff[:20])
                    v.deviations.append({"rule": "tau_recomputation", "n_diff": len(diff), "units": diff[:20]})
                else:
                    rep.add("V4", "tau_compliance", PASS, e)
        if d.claimed_taus is not None:
            diff = [i for i in range(max(len(taus), len(d.claimed_taus)))
                    if i >= len(taus) or i >= len(d.claimed_taus) or taus[i] != int(d.claimed_taus[i])]
            if diff:
                rep.add("V4", "claimed_taus", FAIL, e, reason="discloser's τ differ from the committed τ"
                        if self.semantics == "rev1" else "discloser's τ differ from the recomputed τ",
                        n_diff=len(diff), units=diff[:20])
                ok = False
            else:
                rep.add("V4", "claimed_taus", PASS, e)
        else:
            rep.add("V4", "claimed_taus", SKIP, e, reason="no claimed τ in the disclosure")
        v.taus = taus
        v.pay = [r.pay for r in st.records]
        v.v4_ok = ok
        if ok:
            ctx.stores[e] = st
            ctx.taus[e] = taus
        v.expected_cost = sum(t_ * r.s for t_, r in zip(taus, st.records)) / TWO64

    # ------------------------------------------------------------------------ V5
    def _v5(self, rep: VerificationReport, d: EpochDisclosure, v: EpochVerdict, ctx: ComplianceContext) -> None:
        e, st = d.e, d.store
        K = selection_key(v.beta_vrf)
        leaves = st.leaves()
        v.u, v.Z = select_units(K, e, leaves, v.taus, v.pay)
        v.selected = {i for i, z in enumerate(v.Z) if z}
        v.realised_cost = sum(st.records[i].s for i in v.selected)
        if v.v4_ok and v.v1_ok and v.v2_ok:
            ctx.Z[e] = v.Z
        if d.claimed_selected is not None:
            cs = set(int(i) for i in d.claimed_selected)
            if cs == v.selected:
                rep.add("V5", "claimed_selection", PASS, e, n_selected=len(v.selected))
            else:
                rep.add("V5", "claimed_selection", FAIL, e, reason="claimed retained set != S_e",
                        not_claimed=sorted(v.selected - cs)[:20], wrongly_claimed=sorted(cs - v.selected)[:20])
        else:
            rep.add("V5", "selection", PASS, e, n_selected=len(v.selected))

    # ------------------------------------------------------------------------ V6
    def _v6(self, rep: VerificationReport, d: EpochDisclosure, v: EpochVerdict, ctx: ComplianceContext) -> None:
        e, h, st = d.e, d.header, d.store
        required = v.requested & v.selected
        v.missing = {i for i in required if i not in d.units}
        mism: dict[int, str] = {}
        for i, du in d.units.items():
            if not (0 <= i < st.n) or not v.Z[i]:
                v.extra.add(i)
                continue
            why = self._check_unit(du, st.records[i], i, st.n, h.R)
            if why:
                mism[i] = why
            elif v.v1_ok and v.v4_ok:
                ctx.frames[(e, i)] = du.frames
        v.mismatch = set(mism)
        rep.add("V6", "missing", FAIL if v.missing else PASS, e, n_required=len(required),
                **({"units": sorted(v.missing)[:20], "n_missing": len(v.missing)} if v.missing else {}))
        rep.add("V6", "extra", FAIL if v.extra else PASS, e, **({"units": sorted(v.extra)[:20]} if v.extra else {}))
        rep.add("V6", "mismatch", FAIL if mism else PASS, e,
                **({"units": dict(list(sorted(mism.items()))[:20])} if mism else {}))
        v.W |= v.missing | v.mismatch

    @staticmethod
    def _check_unit(du, rec, i: int, n: int, R: bytes) -> str:
        if not rec.pay or rec.root_pkt is None:
            return "unit is zero-payload"
        if len(du.frames) != len(du.pkt_leaves):
            return "frame count != frame-hash list length"
        for j, f in enumerate(du.frames):
            if f.caplen != len(f.data):
                return f"packet {j}: caplen != len(frame)"
            if hs.frame_leaf(f.ts_us, f.dir, f.caplen, f.data) != du.pkt_leaves[j]:
                return f"packet {j}: frame hash mismatch"
        if hs.merkle_root(du.pkt_leaves) != rec.root_pkt:
            return "frame-hash list does not give root_pkt"
        if sum(f.caplen + 16 for f in du.frames) != rec.s:
            return "Σ(caplen+16) != committed s"
        p0 = sum(1 for f in du.frames if f.dir == 0)
        if (p0, len(du.frames) - p0) != tuple(rec.pkts):
            return "per-direction packet counts differ from the store"
        L = hs.unit_leaf(rec.meta, rec.root_pkt)
        if not hs.verify_proof(L, i, n, du.proof, R):
            return "inclusion proof of L_i in R_e fails"
        return ""

    # ------------------------------------------------------------------------ V7
    def _v7(self, rep: VerificationReport, disc: Disclosure, d: EpochDisclosure, v: EpochVerdict) -> None:
        e = d.e
        if self.eps is None or self.trust.pk_I is None:
            rep.add("V7", "cells", SKIP, e, reason="no ε_cell calibration or no pk_I")
            return
        rec = disc.counters.get(e)
        if rec is None:
            rep.add("V7", "counter_record", WARN, e, reason="no counter record for this epoch")
            return
        if rec.e != e or rec.gw_id != self.trust.gw_id or not rec.verify(self.trust.pk_I):
            rep.add("V7", "counter_record", FAIL, e, reason="counter record not signed by I for (G, e)")
            return
        rep.add("V7", "counter_record", PASS, e)
        committed = d.store.cells() if (d.store is not None and v.v4_ok) else {}
        cells = compare_cells(committed, rec.as_dict(), self.eps)
        v.cells = cells
        flagged = [c for c in cells if c["flagged"]]
        if flagged:
            rep.add("V7", "cells", FAIL, e, flagged=[{k: c[k] for k in ("device_id", "committed_pkts", "counter_pkts",
                                                                         "d_pkts", "eps_pkts", "d_bytes", "eps_bytes")}
                                                      for c in flagged])
        else:
            rep.add("V7", "cells", PASS, e, n_cells=len(cells))

    # ------------------------------------------------------------------------ V4 policy compliance (2nd pass)
    def _compliance(self, rep: VerificationReport, disc: Disclosure, E: list[int], ctx: ComplianceContext) -> None:
        for e in E:
            v = rep.epochs[e]
            d = disc.epochs.get(e)
            if d is None or e not in ctx.states or e not in ctx.headers:
                continue
            try:
                devs = self.policy.compliance(ctx.headers[e], ctx.states[e], ctx)
            except Exception as ex:                     # noqa: BLE001
                devs = [{"rule": "compliance_check_error", "error": str(ex)}]
            v.deviations = list(v.deviations) + list(devs)
            real = [x for x in devs if x.get("severity") != "info"]
            nchk = [x for x in devs if x.get("severity") == "info" and "not_checkable" in str(x.get("rule", ""))]
            if real:
                rep.add("V4", "policy_compliance", WARN, e, deviations=real[:10])
            elif nchk and self.semantics == "rev1":
                rep.add("V4", "policy_compliance", NCHK, e, notes=devs[:5])
            elif devs:
                rep.add("V4", "policy_compliance", PASS, e, notes=devs[:5])
            else:
                rep.add("V4", "policy_compliance", PASS, e)

    def _budget(self, rep: VerificationReport) -> None:
        exp = [v.expected_cost for v in rep.epochs.values() if v.expected_cost is not None]
        real = [v.realised_cost for v in rep.epochs.values() if v.realised_cost is not None]
        rep.budget = {"epochs": len(rep.epochs), "expected_spend": float(sum(exp)), "realised_spend": int(sum(real)),
                      "floor_mode_epochs": [e for e, v in rep.epochs.items() if v.floor_mode == 1]}


def sha256(b: bytes) -> bytes:          # small helper kept for callers that check beacon rounds by hand
    return hashlib.sha256(b).digest()
