"""The real Commit-then-Sample threshold policy (DESIGN Alg. 1 lines 2-3 and 8; §3.4-3.6), implementing
P1-A's ``protocol.policy.ThresholdPolicy`` interface.

Per epoch e (gateway and verifier compute the same integers):
  1. features of every committed unit from enc(m_i) and the per-device rolling statistics over the committed
     metadata of epochs (e − W, e] (units/features.py; W = 120). The policy keeps that history itself; the
     verifier must therefore see the stores of the W − 1 preceding epochs (``prime`` or verify from the start);
     an incomplete window raises (V4 then FAILs: the τ cannot be recomputed);
  2. â_i = IntUtility(θ, m_i, rolling state, boost set_e) (design/utility.py; U5 by default, U3/U8 variants),
     κ_b on the committed boost set;
  3. τ^φ = ⌊φ0 2^64⌋, or the per-device flood-guard cap when floor_mode = 1 (D_{e−1} < −D_max):
     τ^φ_d = min(τ^φ, ⌊γ B̄ 2^64 / (n_active C_{d,e})⌋), C_{d,e} = device d's payload bytes in epoch e;
     τ_i = intpi.threshold_unit(Λ^A_e, Λ^S_e, â_i, s_i, τ^φ_i, t16), 0 for zero-payload units;
  4. update (after selection): D_e = D_{e−1} + B̄ − cost_e (expected ⌊Σ τ s / 2^64⌋ or realised Σ_{Z=1} s),
     Λ^{A,S}_{e+1} = round(λ_ref^{A,S} exp(κ clip(D_e/(W_c B̄), −L, L)) 2^40) (design/controller.py, one deficit
     drives both components), floor_mode_{e+1} = 1[D_e < −D_max]; the attack-indicative detectors
     utility.BOOST_DETECTORS (D1, D2, D3, D4, D12, D13 of content/detectors.py) on the RETAINED frames of epoch e
     add Justification records whose device is boosted for e+2 … e+1+W_b.
rev1: the gateway COMMITS the τ_i of every payload unit in the metadata store (protocol/store.py); the verifier's
validity check uses the committed τ_i and ``floor_thresholds`` (τ^φ, or the flood-guard cap of floor mode 1, from the
epoch's own store), and recomputes ``thresholds`` only as a compliance check (which needs the rolling history: prime
bundle). Rolling statistics: bit-exact ``IncrementalRollingStats`` by default (cost O(n_e + n_{e−W}) per epoch; the
re-sorting original as rolling_impl="resort"); ``set_chain_start`` anchors the verifier's window at the chain start;
optional ``utility_cap`` (flood mitigation: only the first k payload units per device and epoch are scored).
Committed: H(θ) over the utility bundle digest, active estimands, W_q, Q2_q, feature version and LUT hash;
H(policy) over every parameter (decimal strings); both are cached per instance. ``compliance`` re-checks the
controller (Λ within ±1 of the float controller), the D recursion, floor_mode, the boost justifications and the
boost COMPLETENESS (omitted boosts: per disclosed retained trigger unit always; the whole boost set when every
retained unit of [e−1−W_b, e−2] is disclosed, else an info note "not checkable (partial disclosure)").
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from ..protocol.policy import (ComplianceContext, Justification, PolicyState, ThresholdPolicy, boost_digest)
from ..units import features as F
from . import intpi
from . import utility as U
from .controller import HorizonController

__all__ = ["CtSPolicy", "load_cts_policy", "detect_frames", "POLICY_VERSION"]

POLICY_VERSION = "ctsample.policy_cts/v1"


def detect_frames(frames, proto: int, port_a: int, port_b: int, detectors=None) -> str | None:
    """First raw detector (sorted ids) that fires on any payload packet of a unit's frames, else None."""
    from ..content import detectors as det
    from ..io.pcap_reader import P_CAPEND, P_KIND, P_PAYOFF, K_IP, parser_for
    if not frames or proto not in (6, 17):
        return None
    dets = det.flow_detectors(proto, port_a, port_b)
    if detectors is not None:
        dets = tuple(d for d in dets if d[0] in detectors)
    if not dets:
        return None
    parse = parser_for(1)
    fired = set()
    for fr in frames:
        p = parse(fr.data, fr.wirelen)
        if p[P_KIND] != K_IP:
            continue
        pl = fr.data[p[P_PAYOFF]:p[P_CAPEND]]
        if not pl:
            continue
        for rid, _v in det.run_packet(dets, pl, fr.dir, proto == 6):
            fired.add(rid)
    return sorted(fired)[0] if fired else None


class CtSPolicy(ThresholdPolicy):
    name = "cts"

    def __init__(self, bundle: U.UtilityBundle, active, W, Q2, lam_ref, Bbar: int, phi0="0.015", tau_mix="0.2",
                 utility: str = "U5", controller: str = "expected", boost: bool = True, kappa_b: int = 5,
                 W_b: int = 20, kappa="1", Wc: int = 30, L: str = "2.772588722239781", Dmax_mult: int = 200,
                 gamma="0.3", rolling_W: int = F.W_DEFAULT, flood_guard: bool = True,
                 boost_detectors=U.BOOST_DETECTORS, rolling_impl: str = "incremental",
                 utility_cap: int | None = None):
        if utility not in ("U5", "U3", "U8"):
            raise ValueError("utility must be U5, U3 or U8")
        if controller not in ("expected", "realised"):
            raise ValueError("controller must be 'expected' or 'realised'")
        self.bundle = bundle
        self.active = list(active)
        missing = [q for q in self.active if q not in bundle.queries]
        if missing:
            raise ValueError(f"active estimands without a model: {missing}")
        self.Wq = [int(x) for x in W]
        self.Q2 = [int(x) for x in Q2]
        if not (len(self.Wq) == len(self.Q2) == len(self.active)):
            raise ValueError("active / W / Q2 lengths differ")
        self.lam_ref = [repr(float(x)) for x in lam_ref]
        if len(self.lam_ref) != 2:
            raise ValueError("lam_ref = (lam_A, lam_S)")
        self.Bbar = int(Bbar)
        self.D_max = int(Dmax_mult) * self.Bbar
        self.Dmax_mult = int(Dmax_mult)
        self.phi0 = str(phi0)
        self.tau_phi = intpi.floor_tau(self.phi0)
        self.tau_mix = str(tau_mix)
        self.t16 = intpi.t16_from_mix(self.tau_mix)
        self.utility = utility
        self.controller = controller
        self.boost = bool(boost) and utility == "U5"
        self.kappa_b = int(kappa_b)
        self.W_b = int(W_b)
        self.kappa, self.Wc, self.L = str(kappa), int(Wc), str(L)
        self.gamma = str(gamma)
        self._gamma = intpi.to_fraction(self.gamma)
        self.rolling_W = int(rolling_W)
        self.flood_guard = bool(flood_guard)
        self.boost_detectors = tuple(sorted(boost_detectors))
        # rev1: bit-exact incremental rolling statistics (cost O(n_e + n_{e−W}) per epoch instead of re-sorting the
        # 120-epoch window); "resort" = the original RollingDeviceStats (same integers, kept for comparisons)
        if rolling_impl not in ("incremental", "resort"):
            raise ValueError("rolling_impl must be 'incremental' or 'resort'")
        self.rolling_impl = rolling_impl
        self._rs_cls = F.IncrementalRollingStats if rolling_impl == "incremental" else F.RollingDeviceStats
        # rev1 flood mitigation (optional, part of H(policy) when set): per device and epoch only the first
        # utility_cap payload units (canonical order) are scored by the model; the others get â = 0, i.e. the
        # floor / size component only (π ≥ φ0 still holds; commit cost bounded under payload floods)
        self.utility_cap = None if utility_cap is None else int(utility_cap)
        self._ctrl = HorizonController(np.array([float(x) for x in self.lam_ref]), float(self.Bbar),
                                       kappa=float(self.kappa), Wc=float(self.Wc), L=float(self.L), mode="expected")
        self._hist: dict[int, dict] = {}
        self._e0: int | None = None
        self._theta_bytes: bytes | None = None
        self._registration: tuple[bytes, bytes] | None = None
        self._rs: F.RollingDeviceStats | None = None
        self._cache: dict[int, tuple] = {}

    # ---------------------------------------------------------------- commitments
    def theta_bytes(self) -> bytes:
        """θ payload (cached: the bundle digest hashes every forest, ≈ 40 ms; the policy is immutable)."""
        if self._theta_bytes is None:
            self._theta_bytes = json.dumps({"version": POLICY_VERSION, "bundle": self.bundle.theta_digest(),
                                            "utility": self.utility, "active": self.active, "W": self.Wq,
                                            "Q2": self.Q2, "features": F.FEATURE_VERSION, "lut": U.te.lut_hash()},
                                           sort_keys=True, separators=(",", ":")).encode()
        return self._theta_bytes

    def registration(self) -> tuple[bytes, bytes]:
        """(H(θ), H(policy)), computed once per policy instance."""
        if self._registration is None:
            self._registration = super().registration()
        return self._registration

    def policy_params(self) -> dict:
        return {"name": self.name, "version": POLICY_VERSION, "phi0": self.phi0, "tau_mix": self.tau_mix,
                "t16": self.t16, "lam_ref": self.lam_ref, "Bbar": self.Bbar, "D_max": self.D_max,
                "Dmax_mult": self.Dmax_mult, "kappa": self.kappa, "Wc": self.Wc, "L": self.L,
                "controller": self.controller, "boost": self.boost, "kappa_b": self.kappa_b, "W_b": self.W_b,
                "gamma": self.gamma, "rolling_W": self.rolling_W, "flood_guard": self.flood_guard,
                "boost_detectors": list(self.boost_detectors), "lut": U.te.lut_hash(),
                **({"utility_cap": self.utility_cap} if self.utility_cap is not None else {})}

    def expected_tau_phi(self) -> int:
        return self.tau_phi

    def Lam_for_D(self, D: int) -> tuple[int, int]:
        self._ctrl.D = np.asarray(float(D))
        lam = np.asarray(self._ctrl.Lam_fix(), dtype=object).ravel()
        return int(lam[0]), int(lam[1])

    def initial_state(self, e0: int) -> PolicyState:
        self._e0 = int(e0)
        la, ls = self.Lam_for_D(0)
        return PolicyState(e=int(e0), D_prev=0, floor_mode=0, Lam_A=la, Lam_S=ls)

    def set_chain_start(self, e0: int) -> None:
        """Verifier side: anchor the rolling window at the chain start (an unseen history then raises)."""
        if self._e0 is None:
            self._e0 = int(e0)

    def commit_fields(self, state: PolicyState) -> dict:
        H_theta, H_policy = self.registration()
        return {"Lam_A": state.Lam_A, "Lam_S": state.Lam_S, "tau_phi": self.tau_phi, "H_theta": H_theta,
                "H_policy": H_policy, "H_boost": boost_digest(state.boost_set, state.justifications),
                "floor_mode": state.floor_mode, "D_prev": state.D_prev}

    # ---------------------------------------------------------------- rolling history
    def prime(self, batches) -> None:
        """Feed committed batches of earlier epochs (verifier with a partial disclosure)."""
        for b in sorted(batches, key=lambda x: x.epoch):
            if self._e0 is None:
                self._e0 = int(b.epoch)
            self._remember(int(b.epoch), F.cols_from_units(b.units))

    def _remember(self, e: int, c: dict) -> None:
        self._hist[e] = c
        for k in [k for k in self._hist if k <= e - self.rolling_W]:
            del self._hist[k]

    def _rolling(self, e: int, c: dict) -> np.ndarray:
        if self._e0 is None:
            self._e0 = e
        lo = max(self._e0, e - self.rolling_W + 1)
        missing = [k for k in range(lo, e) if k not in self._hist]
        if missing:
            raise ValueError(f"rolling window incomplete for epoch {e}: metadata of epochs {missing[:5]}... "
                             "not seen (prime the policy with the preceding stores)")
        rs = self._rs
        if rs is None or rs.last_e is None or rs.last_e != e - 1 or (rs.last_e < lo - 1):
            rs = self._rs_cls(self.rolling_W)
            rs.prime([(k, self._hist[k]) for k in range(lo, e)])
        R = rs.step(e, c)
        self._rs = rs
        return R

    def features(self, batch) -> tuple[dict, np.ndarray]:
        e = int(batch.epoch)
        hit = self._cache.get(e)
        if hit is not None and hit[0] == tuple(u.meta for u in batch.units):
            return hit[1], hit[2]
        c = F.cols_from_units(batch.units)
        if self._rs is not None and self._rs.last_e is not None and e <= self._rs.last_e:
            self._rs = None                                   # replay: rebuild from the history
        try:
            R = self._rolling(e, c)
        finally:
            # remember the committed metadata even when this epoch's window is incomplete (rev1 verifier: its
            # compliance check is NCHK, but later epochs whose windows are complete stay checkable)
            self._remember(e, c)
        X = np.zeros((len(c["pay"]), F.N_FEATURES), dtype=np.int32)
        pidx = np.nonzero(c["pay"])[0]          # static features are only ever read for payload units (rev1: a
        if len(pidx):                           # zero-payload flood no longer pays for them)
            X[pidx, :33] = F.static_features(F.take_cols(c, pidx), wide=False)
        X[:, 33:] = np.clip(R, -(1 << 31), (1 << 31) - 1)
        self._cache = {e: (tuple(u.meta for u in batch.units), c, X)}
        return c, X

    # ---------------------------------------------------------------- utilities and thresholds
    def utilities(self, batch, state: PolicyState) -> list[int]:
        """â_i (Python ints; 0 for zero-payload units) of the epoch's units."""
        c, X = self.features(batch)
        n = len(c["pay"])
        pay = c["pay"].astype(bool)
        out = [0] * n
        idx = np.nonzero(pay)[0]
        if self.utility_cap is not None and len(idx):
            # rev1 flood mitigation: only the first utility_cap payload units of each device (canonical order) are
            # scored; the rest keep â = 0 (floor / size component only)
            dv = c["device_id"][idx]
            order = np.argsort(dv, kind="stable")
            first = np.r_[True, dv[order][1:] != dv[order][:-1]]
            start = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
            rank = np.empty(len(idx), dtype=np.int64)
            rank[order] = np.arange(len(order)) - start
            idx = idx[rank < self.utility_cap]
        if not len(idx):
            return out
        cp = F.take_cols(c, idx)
        Xp = X[idx]
        if self.utility == "U3":
            a = U.utility_int(self.bundle.pmal_int(Xp))
        else:
            el = U.eligibility_matrix(self.active, cp)
            if self.utility == "U5":
                P = np.zeros((len(idx), len(self.active)), dtype=np.int64)
                for j, q in enumerate(self.active):
                    m = el[:, j]
                    if m.any():
                        P[m, j] = self.bundle.hit[q].predict_int(Xp[m])
            else:
                st = U.stratum_index(cp)
                P = np.stack([self.bundle.P8[q][st] for q in self.active], axis=1) if self.active else \
                    np.zeros((len(idx), 0), np.int64)
                P = np.where(el, P, 0)
            G = U.ghat_int(P, self.bundle.M_cells(self.active), U.cell_index(cp))
            mult = None
            if self.boost and state.boost_set:
                bset = np.array(sorted(state.boost_set), dtype=np.int64)
                mult = np.where(np.isin(cp["device_id"], bset), self.kappa_b, 1)
            a = U.utility_int(U.a_fix_int(G, self.Wq, self.Q2), mult)
        for j, i in enumerate(idx):
            out[i] = int(a[j])
        return out

    def floor_taus(self, batch, state: PolicyState) -> list[int]:
        units = batch.units
        tf = [self.tau_phi] * len(units)
        if not (self.flood_guard and state.floor_mode == 1):
            return tf
        C: dict[int, int] = {}
        for u in units:
            if u.pay:
                C[u.device_id] = C.get(u.device_id, 0) + int(u.s)
        n_active = len(C)
        if n_active == 0:
            return tf
        g = self._gamma
        cap = {d: (g.numerator * self.Bbar << 64) // (g.denominator * n_active * Cd) for d, Cd in C.items()}
        return [min(self.tau_phi, cap[u.device_id]) if u.pay else self.tau_phi for u in units]

    def floor_thresholds(self, batch, state: PolicyState) -> list[int]:
        """Validity floor of the committed τ (rev1 V4): τ^φ, or the flood-guard cap of floor mode 1 — both from
        the epoch's own committed store and the opened floor mode (no rolling features); 0 for zero-payload."""
        return [t if u.pay else 0 for u, t in zip(batch.units, self.floor_taus(batch, state))]

    def thresholds(self, batch, state: PolicyState) -> list[int]:
        a = self.utilities(batch, state)
        tf = self.floor_taus(batch, state)
        out = []
        memo: dict = {}                          # identical (â, s, τ^φ_i) give identical τ (flood units)
        LA, LS, t16 = state.Lam_A, state.Lam_S, self.t16
        for u, ai, ti in zip(batch.units, a, tf):
            if not u.pay:
                out.append(0)
            else:
                k = (ai, int(u.s), ti)
                t = memo.get(k)
                if t is None:
                    t = memo[k] = intpi.threshold_unit(LA, LS, ai, k[1], ti, t16)
                out.append(t)
        return out

    # ---------------------------------------------------------------- after selection
    def detect(self, record) -> list[Justification]:
        if not self.boost:
            return []
        out = []
        for i in record.retained:
            u = record.batch.units[i]
            if not u.frames:
                continue
            from ..units.encode import split_key
            proto, _, pa_, _, pb_ = split_key(u.meta[:37])
            rid = detect_frames(u.frames, proto, pa_, pb_, self.boost_detectors)
            if rid is not None:
                out.append(Justification(int(u.device_id), int(record.e), int(i), rid))
        return out

    def _cost(self, state: PolicyState, record) -> int:
        sizes = [int(u.s) for u in record.batch.units]
        if self.controller == "expected":
            return sum(int(t) * s for t, s in zip(record.taus, sizes)) >> 64
        return sum(sizes[i] for i in record.retained)

    def _active_js(self, js, e: int):
        return tuple(sorted(j for j in js if j.src_epoch + 2 <= e <= j.src_epoch + 1 + self.W_b))

    def update(self, state: PolicyState, record) -> PolicyState:
        D = int(state.D_prev + self.Bbar - self._cost(state, record))
        la, ls = self.Lam_for_D(D)
        fm = 1 if (self.flood_guard and D < -self.D_max) else 0
        e1 = state.e + 1
        pending = list(state.extra.get("pending", ())) + self.detect(record)
        pending = [j for j in pending if j.src_epoch + 1 + self.W_b >= e1]
        act = self._active_js(pending, e1)
        return PolicyState(e=e1, D_prev=D, floor_mode=fm, Lam_A=la, Lam_S=ls,
                           boost_set=tuple(sorted({j.device_id for j in act})), justifications=act,
                           extra={"pending": pending})

    # ---------------------------------------------------------------- verifier side
    def compliance(self, hdr, state: PolicyState, ctx: ComplianceContext) -> list[dict]:
        dev = []
        e = hdr.e
        prev = ctx.headers.get(e - 1)
        if prev is not None and (e - 1) in ctx.stores:
            sizes = [r.s for r in ctx.stores[e - 1].records]
            if self.controller == "expected" and (e - 1) in ctx.taus:
                cost = sum(int(t) * s for t, s in zip(ctx.taus[e - 1], sizes)) >> 64
            elif self.controller == "realised" and (e - 1) in ctx.Z:
                cost = sum(s for s, z in zip(sizes, ctx.Z[e - 1]) if z)
            else:
                cost = None
            if cost is not None and prev.D_prev + self.Bbar - cost != hdr.D_prev:
                dev.append({"rule": "D_recursion", "committed": hdr.D_prev, "expected": prev.D_prev + self.Bbar - cost})
        la, ls = self.Lam_for_D(hdr.D_prev)
        if abs(hdr.Lam_A - la) > 1 or abs(hdr.Lam_S - ls) > 1:
            dev.append({"rule": "Lambda", "Lam_A": hdr.Lam_A, "Lam_S": hdr.Lam_S, "expected": [la, ls]})
        want_fm = 1 if (self.flood_guard and hdr.D_prev < -self.D_max) else 0
        if hdr.floor_mode != want_fm:
            dev.append({"rule": "floor_mode", "committed": hdr.floor_mode, "expected": want_fm})
        if not self.boost:
            if state.boost_set or state.justifications:
                dev.append({"rule": "boost_not_in_policy", "boost_set": list(state.boost_set)})
            return dev
        want_set = tuple(sorted({j.device_id for j in state.justifications}))
        if tuple(state.boost_set) != want_set:
            dev.append({"rule": "boost_set_vs_justifications", "boost_set": list(state.boost_set),
                        "justified": list(want_set)})
        for j in state.justifications:
            if not (j.src_epoch + 2 <= e <= j.src_epoch + 1 + self.W_b):
                dev.append({"rule": "justification_window", "justification": j.__dict__})
                continue
            if j.detector not in self.boost_detectors:
                dev.append({"rule": "unknown_detector", "justification": j.__dict__})
                continue
            if j.src_epoch in ctx.Z:
                Zs = ctx.Z[j.src_epoch]
                if not (0 <= j.src_index < len(Zs)) or not Zs[j.src_index]:
                    dev.append({"rule": "unjustified_boost", "reason": "source unit not retained",
                                "justification": j.__dict__})
                    continue
                st = ctx.stores.get(j.src_epoch)
                if st is not None and st.records[j.src_index].device_id != j.device_id:
                    dev.append({"rule": "unjustified_boost", "reason": "device mismatch", "justification": j.__dict__})
                    continue
                fr = ctx.frames.get((j.src_epoch, j.src_index))
                if fr is not None and st is not None:
                    from ..units.encode import split_key
                    proto, _, pa_, _, pb_ = split_key(st.records[j.src_index].meta[:37])
                    if detect_frames(fr, proto, pa_, pb_, (j.detector,)) is None:
                        dev.append({"rule": "unjustified_boost", "reason": "detector does not fire on the evidence",
                                    "justification": j.__dict__})
            else:
                dev.append({"rule": "justification_unverifiable", "justification": j.__dict__, "severity": "info"})
        comp = self._boost_completeness(e, state, ctx)
        # the verifier keeps only the first few info notes of an epoch: the completeness note goes first
        first = [d for d in comp if d.get("rule") == "boost_completeness_not_checkable"]
        return first + dev + [d for d in comp if d.get("rule") != "boost_completeness_not_checkable"]

    def _boost_completeness(self, e: int, state: PolicyState, ctx: ComplianceContext) -> list[dict]:
        """Omitted boosts (A10-omit). The epochs that can justify a boost at e are e' in [e−1−W_b, e−2]
        (clipped at 0; an epoch the verifier has not verified counts as not disclosed). (a) Per unit, sound on any disclosure: every DISCLOSED retained unit of
        those epochs on which a trigger detector fires must appear as a committed justification. (b) If every
        retained unit of those epochs is disclosed (verified epochs, frames present), the committed boost set must
        equal the recomputed one; otherwise an info note 'boost completeness not checkable (partial disclosure)'.
        Deviations are policy-compliance findings (harmless to validity, Thm. 2(e))."""
        from ..units.encode import split_key
        lo = max(e - 1 - self.W_b, 0)          # epochs not seen by the verifier count as missing (conservative)
        hi = e - 2
        want: set = set()
        missing_epochs, missing_units = [], 0
        for es in range(lo, hi + 1):
            Zs, st = ctx.Z.get(es), ctx.stores.get(es)
            if Zs is None or st is None:
                missing_epochs.append(es)
                continue
            for i, z in enumerate(Zs):
                if not z:
                    continue
                fr = ctx.frames.get((es, i))
                if fr is None:
                    missing_units += 1
                    continue
                r = st.records[i]
                proto, _, pa_, _, pb_ = split_key(r.meta[:37])
                rid = detect_frames(fr, proto, pa_, pb_, self.boost_detectors)
                if rid is not None:
                    want.add(Justification(int(r.device_id), int(es), int(i), rid))
        out = []
        have = set(state.justifications)
        for j in sorted(want - have):
            out.append({"rule": "omitted_boost", "justification": j.__dict__})
        if missing_epochs or missing_units:
            out.append({"rule": "boost_completeness_not_checkable", "severity": "info",
                        "reason": "boost completeness not checkable (partial disclosure)",
                        "window": [lo, hi], "missing_epochs": missing_epochs[:20], "n_missing_epochs": len(missing_epochs),
                        "missing_retained_units": missing_units})
        else:
            rec = tuple(sorted({j.device_id for j in want}))
            if tuple(sorted(set(state.boost_set))) != rec or have != want:
                out.append({"rule": "boost_set_incomplete", "committed": list(state.boost_set), "recomputed": list(rec),
                            "n_committed_justifications": len(have), "n_recomputed_justifications": len(want)})
        return out


# ------------------------------------------------------------------------------------------ #
def load_cts_policy(fold: str, seed: int, p_key: str, rho: float, Bbar_test: int, design: str = "U6tau0.2",
                    controller: str = "realised", models_dir: Path | None = None, attach_lgbm: bool = True,
                    **kw) -> CtSPolicy:
    """CtS policy from the results/models/<fold>/<seed>/ outputs of scripts/04_train_utility.py: bundle θ,
    Q̃ (active estimands, W, Q2) of the training LP stream with prevalence key ``p_key`` (e.g. 'p10000'),
    φ0(ρ) and λ_ref(ρ, design) — the adaptive-arm λ_ref when ``controller='realised'`` and the design boosts."""
    from ..utils.io import RESULTS
    d = Path(models_dir or RESULTS / "models") / fold / str(seed)
    b = U.UtilityBundle.load(d, attach_lgbm=attach_lgbm)
    qn = json.loads((d / "Qnorm.json").read_text())[p_key]
    ph = json.loads((d / "phi0.json").read_text())[p_key][f"{rho:g}"]
    lrs = json.loads((d / "lamref.json").read_text())[p_key][f"{rho:g}"]
    lr = lrs[design]
    lam = lr.get("lamref_adaptive") if (controller == "realised" and "lamref_adaptive" in lr) else lr["lam_ref"]
    kind, tau = lr["kind"], lr["tau_mix"]
    if kind == "A":                    # τ_mix = 0: Λ^S is committed but unused; take the size-only design's λ_ref
        lam = [lam[0], lrs["U1"]["lam_ref"][0]]
    util = {"U5": "U5", "U3": "U3", "U8": "U8"}.get(lr["util"], "U5")
    boost = design.startswith("U6")
    phi0 = repr(float(ph))
    if "boost_detectors" not in kw:
        bj = d / "boost.json"
        if bj.exists():
            kw["boost_detectors"] = tuple(json.loads(bj.read_text())[p_key]["F"])
    return CtSPolicy(b, qn["active"], qn["W"], qn["Q2"], lam, Bbar_test, phi0=phi0, tau_mix=repr(float(tau)),
                     utility=util, controller=controller, boost=boost, **kw)
