"""Disclosure "prime" bundle: the committed metadata context that V4 needs to recompute τ (P1-G).

The CtS utility (design/policy_cts.py) uses per-device rolling features over the committed metadata of
epochs (e − W, e], W = 120. A verifier that recomputes τ_i bit-exactly for a disclosed epoch e (V4)
therefore needs the metadata stores of the up to W − 1 = 119 preceding epochs, also when those epochs are
not part of the investigation (DESIGN §3.7 lists only the stores of the requested epochs; P1-D decision 5).

Bundle (``PrimeBundle``): for every epoch k ∈ ∪_{e∈E'} [max(e0, e − W + 1), e − 1] \\ E' the published
header Hdr_k with σ_k and ω_k and the REDACTED metadata store of k (records + BLAKE3 of the threshold column,
not the committed τ_i, which would reveal k's design state). Nothing else is disclosed for k: no Π_k, no
opening of C^st_k, no frames (the selections of prime epochs stay private, Thm. 2(d)).

rev1: the prime bundle is OPTIONAL. Selection integrity and the statements of a disclosed epoch rest on its
committed τ_i (protocol/store.py, verifier V4 'committed_taus'); the bundle only lets V4 recompute τ_i through
the policy (rolling features) as a compliance check ('tau_compliance'). Without it, or with a missing or
failing prime store, that check reports NCHK ("compliance not checkable") — the selection verification of the
dependent epochs is unaffected. (Under the pre-revision semantics, ``Verifier(semantics="rev0")``, a missing
prime store FAILs V4 'thresholds' of every dependent epoch and puts their requested units in 𝒲.)

Verifier side (``check_prime``), per prime epoch k — reported as V4 ``prime_store`` PASS/FAIL:
  * σ_k verifies under pk_G; header fields (e, gw_id, ver, T, t_e) and r*_k = r0 + k + ahead;
  * ω_k verifies under pk_W and τ_W ≤ t(r*_k) − δ_W − ε_β (one cosigned header per (G, k));
  * chain: H(Hdr_k) equals H_prev of the next header (next prime header, or the disclosed / context
    header of k + 1) whenever that header is present (authenticity itself rests on σ_k and ω_k, since the
    witness cosigns one header per (G, k)), and k = e0 links to GENESIS_PREV;
  * the store reproduces H(store_k), R_k (leaves recomputed) and the header totals (n, n⁺, Σs, Σ_{U⁺}s), and
    (with the verifier's meta codec) its derived columns agree with enc(m_i).
Only stores that pass every check are fed to the policy (``PrimedPolicy`` feeds them just before V4 of the
first disclosed epoch whose window needs them); a missing or failing prime store leaves the window
incomplete, the policy raises, V4 'thresholds' FAILs the dependent disclosed epochs and their requested
units enter 𝒲 (Alg. 2) — prime tampering is evident, never silent.

``verify_with_prime(verifier, disc, bundle)`` = check_prime + Verifier.verify with the primed policy; the
prime checks are prepended to the report (``rep.prime`` holds the summary and sizes).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..crypto import hashing as hs
from .disclosure import Disclosure, PublishedHeader
from .header import GENESIS_PREV, HDR_LEN, VER, Header, verify_header
from .policy import ThresholdPolicy
from .store import MetadataStore
from .verifier import FAIL, PASS, CheckResult, TrustAnchors, VerificationReport, Verifier
from .witness import Cosignature, verify_cosignature

__all__ = ["PrimeEntry", "PrimeBundle", "prime_epochs", "build_prime", "check_prime", "PrimedPolicy",
           "verify_with_prime", "W_DEFAULT"]

W_DEFAULT = 120


@dataclass
class PrimeEntry:
    header: Header
    sigma: bytes
    cosig: Cosignature | None
    store: MetadataStore

    def __post_init__(self):
        # a prime entry never carries the committed thresholds of its (non-requested) epoch: only their digest
        if self.store is not None and self.store.taus is not None:
            self.store = self.store.redacted()

    def size_bytes(self) -> int:
        return HDR_LEN + 64 + (72 if self.cosig is not None else 0) + len(self.store.serialize())


@dataclass
class PrimeBundle:
    W: int
    entries: dict[int, PrimeEntry] = field(default_factory=dict)

    def size_bytes(self) -> dict:
        hdr = sum(HDR_LEN + 64 + (72 if x.cosig is not None else 0) for x in self.entries.values())
        st = sum(len(x.store.serialize()) for x in self.entries.values())
        return {"epochs": len(self.entries), "header_bytes": hdr, "store_bytes": st, "total_bytes": hdr + st,
                "store_records": sum(x.store.n for x in self.entries.values())}

    def copy(self) -> "PrimeBundle":
        return copy.deepcopy(self)


def prime_epochs(disclosed, e0: int = 0, W: int = W_DEFAULT) -> list[int]:
    """Epochs whose stores the rolling features of the disclosed epochs need (excluding disclosed ones)."""
    D = set(int(e) for e in disclosed)
    need = set()
    for e in D:
        need.update(k for k in range(max(int(e0), e - W + 1), e) if k not in D)
    return sorted(need)


def build_prime(gw, disclosed, W: int = W_DEFAULT, e0: int | None = None) -> PrimeBundle:
    """Prime bundle from a Gateway (or a view with ``records`` and ``e0``)."""
    e0 = int(getattr(gw, "e0", 0) if e0 is None else e0)
    b = PrimeBundle(W=W)
    for k in prime_epochs(disclosed, e0, W):
        rec = gw.records.get(k)
        if rec is None:                     # withheld / never committed: not available (V4 compliance → NCHK)
            continue
        b.entries[k] = PrimeEntry(rec.header, rec.sigma, rec.cosig, copy.deepcopy(rec.store).redacted())
    return b


def _next_header(bundle: PrimeBundle, disc: Disclosure, k: int) -> Header | None:
    if (k + 1) in bundle.entries:
        return bundle.entries[k + 1].header
    h = disc.header_of(k + 1)
    return None if h is None else h[0]


def _meta_inconsistent(st: MetadataStore, codec) -> int:
    """Records whose committed derived columns disagree with enc(m_i) (as V4 'meta_consistency')."""
    bad = 0
    for r in st.records:
        try:
            f = codec.decode(r.meta, r.pay)
        except Exception:                                   # noqa: BLE001 — a decode error is evidence
            bad += 1
            continue
        for k, val in f.items():
            if hasattr(r, k):
                have = getattr(r, k)
                if (tuple(have) != tuple(val)) if isinstance(have, tuple) else (have != val):
                    bad += 1
                    break
    return bad


def check_prime(bundle: PrimeBundle, trust: TrustAnchors, disc: Disclosure, codec=None) -> tuple[list[CheckResult], dict]:
    """(results, verified stores {k: MetadataStore}) — see the module docstring. With a meta ``codec`` the
    derived store columns are also checked against enc(m_i), as V4 'meta_consistency' does for E'."""
    results: list[CheckResult] = []
    ok: dict[int, MetadataStore] = {}
    c = trust.clock
    for k in sorted(bundle.entries):
        ent = bundle.entries[k]
        h, st = ent.header, ent.store
        probs: list[str] = []
        if h.e != k or h.gw_id != trust.gw_id or h.ver != VER or h.T_us != c.period * 1_000_000 \
                or h.t_e_us != c.t_e_us(k):
            probs.append("header fields")
        if h.r_star != c.r_star(k):
            probs.append("r*_k")
        if not verify_header(trust.pk_G, h, ent.sigma):
            probs.append("σ_k")
        if ent.cosig is None or not verify_cosignature(trust.pk_W, h, ent.cosig):
            probs.append("ω_k")
        elif ent.cosig.tau_W > c.t_round(c.r_star(k)) - trust.delta_W - trust.eps_beta:
            probs.append("TR deadline")
        if k == trust.e0 and h.H_prev != GENESIS_PREV:
            probs.append("chain start")
        nxt = _next_header(bundle, disc, k)
        linked = None
        if nxt is not None:                 # authenticity rests on σ_k and ω_k (one cosigned header per (G, k));
            linked = nxt.H_prev == h.digest()   # the forward link is checked whenever the successor is present
            if not linked:
                probs.append("chain link H(Hdr_k) → Hdr_{k+1}")
        if st is None or st.e != k:
            probs.append("store missing / wrong epoch")
        else:
            try:
                if st.digest() != h.H_store:
                    probs.append("H(store_k)")
                if hs.merkle_root(st.leaves()) != h.R:
                    probs.append("R_k")
                tot = st.totals()
                if (tot["n"], tot["n_pay"], tot["sum_s"], tot["sum_s_pay"]) != (h.n, h.n_pay, h.sum_s, h.sum_s_pay):
                    probs.append("totals")
                if codec is not None and _meta_inconsistent(st, codec):
                    probs.append("meta consistency")
            except (ValueError, TypeError) as ex:
                probs.append(f"store not parseable: {ex}")
        if probs:
            results.append(CheckResult("V4", "prime_store", FAIL, k, {"problems": probs}))
        else:
            results.append(CheckResult("V4", "prime_store", PASS, k,
                                       {"n": st.n, "linked": linked if linked is not None else "successor not disclosed"}))
            ok[k] = st
    return results, ok


class PrimedPolicy(ThresholdPolicy):
    """Verifier-side wrapper: feeds the verified prime stores to the base policy's rolling history just
    before τ of the first disclosed epoch that needs them is recomputed; delegates everything else."""

    def __init__(self, base: ThresholdPolicy, stores: dict[int, MetadataStore], codec=None,
                 W: int = W_DEFAULT, e0: int = 0):
        self.base = base
        self.stores = dict(stores)
        self.codec = codec
        self.W = int(W)
        self.e0 = int(e0)
        self._fed: set[int] = set()
        self.name = base.name
        self.Bbar = getattr(base, "Bbar", None)
        self.D_max = getattr(base, "D_max", None)

    def _feed(self, e: int) -> None:
        # anchor the base policy's rolling window at the chain start e0 of the trust anchors: a window that is
        # not fully primed then raises (V4 'thresholds' FAIL) instead of being silently truncated at e
        if getattr(self.base, "_e0", 0) is None:
            self.base._e0 = self.e0
        need = [k for k in range(max(self.e0, e - self.W + 1), e) if k in self.stores and k not in self._fed]
        if need and hasattr(self.base, "prime"):
            self.base.prime([self.stores[k].to_batch(codec=self.codec) for k in need])
            self._fed.update(need)

    # ---- delegation
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

    def committed_state(self, state):
        return self.base.committed_state(state)

    def commit_fields(self, state):
        return self.base.commit_fields(state)

    def thresholds(self, batch, state):
        self._feed(int(batch.epoch))
        return self.base.thresholds(batch, state)

    def floor_thresholds(self, batch, state):
        return self.base.floor_thresholds(batch, state)

    def set_chain_start(self, e0):
        if getattr(self.base, "_e0", 0) is None:
            self.base._e0 = int(e0)
        return self.base.set_chain_start(e0) if hasattr(self.base, "set_chain_start") else None

    def update(self, state, record):
        return self.base.update(state, record)

    def state_from_commit(self, hdr, boost_set=(), justifications=()):
        return self.base.state_from_commit(hdr, boost_set, justifications)

    def compliance(self, hdr, state, ctx):
        return self.base.compliance(hdr, state, ctx)

    def __getattr__(self, name):                       # utilities(), Lam_for_D(), ...
        return getattr(object.__getattribute__(self, "base"), name)


def verify_with_prime(verifier: Verifier, disc: Disclosure, bundle: PrimeBundle | None, **kw) -> VerificationReport:
    """check_prime + Verifier.verify(disc, **kw) with a PrimedPolicy over the verified prime stores.

    The verifier's policy object must be fresh (empty rolling history); it is wrapped, not mutated except
    for the history fed to it. Prime headers are added to the chain context of (a shallow copy of) disc."""
    bundle = bundle or PrimeBundle(W=W_DEFAULT)
    res, ok = check_prime(bundle, verifier.trust, disc, codec=verifier.codec)
    pp = PrimedPolicy(verifier.policy, ok, codec=verifier.codec, W=bundle.W, e0=verifier.trust.e0)
    v2 = Verifier(verifier.trust, pp, eps_cell=verifier.eps, meta_codec=verifier.codec,
                  statements_hook=verifier.hook, semantics=verifier.semantics,
                  recompute_compliance=verifier.recompute_compliance)
    ctx = dict(disc.context)
    for k, ent in bundle.entries.items():
        if k not in disc.epochs and k not in ctx:
            ctx[k] = PublishedHeader(ent.header, ent.sigma, ent.cosig)
    d2 = Disclosure(gw_id=disc.gw_id, epochs=disc.epochs, context=ctx, gateway_registration=disc.gateway_registration,
                    policy_registrations=disc.policy_registrations, counters=disc.counters)
    rep = v2.verify(d2, **kw)
    rep.results = res + rep.results
    rep.prime = {"epochs": len(bundle.entries), "verified": len(ok), "failed": sorted(set(bundle.entries) - set(ok)),
                 "fed": sorted(pp._fed), **bundle.size_bytes()}
    return rep
