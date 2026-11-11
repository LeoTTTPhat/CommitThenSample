"""Selective disclosure (DESIGN §3.7) and its size / privacy accounting (E6, E8).

A disclosure for epochs E' and requested units contains, per epoch e ∈ E':
* Hdr_e, σ_e, ω_e (+ τ_W), the beacon round r*_e and its predecessor (chain check), Π_e;
* the opening (salt_e, state_e) of the hiding state commitment C^st_e (Λ^A, Λ^S, D_{e−1},
  floor_mode, boost set, H_boost) and the justification records behind H_boost;
* the metadata store of e with its committed threshold column (rev1: V5 recomputes the selection from the committed
  τ_i; recomputing τ through the policy is an optional compliance check that also needs the prime bundle of
  protocol/prime.py, which carries REDACTED stores of the preceding epochs);
* for each requested unit with Z = 1 that the discloser presents: its frames (decrypted by the
  escrow holder), its frame-hash list ℓ_i· and its Merkle inclusion proof of L_i in R_e;
* optional *claims* of the discloser: τ_i it says it used and the set it says it retained.
Plus the published header chain context (Hdr_{min E' − 1}) for the chain link, the witness-log
registrations (gateway keys, policies) and the signed counter records.

Disclosures are plain dataclasses; the adversary modules mutate deep copies of them.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..crypto import hashing as hs
from ..units.batch import Frame
from .beacon import BeaconRound
from .counters import CounterRecord
from .header import HDR_LEN, Header, StateOpening
from .policy import Justification
from .store import MetadataStore
from .witness import Cosignature, GatewayRegistration, PolicyRegistration

__all__ = ["DisclosedUnit", "EpochDisclosure", "PublishedHeader", "Disclosure", "build_disclosure"]


@dataclass
class DisclosedUnit:
    index: int
    frames: list[Frame]
    pkt_leaves: list[bytes]
    proof: list[bytes]

    def size_bytes(self) -> int:
        return sum(f.caplen + 16 for f in self.frames) + 32 * len(self.pkt_leaves) + 32 * len(self.proof) + 8


@dataclass
class PublishedHeader:
    header: Header
    sigma: bytes
    cosig: Cosignature | None


@dataclass
class EpochDisclosure:
    e: int
    header: Header
    sigma: bytes
    cosig: Cosignature | None
    beacon_round: BeaconRound | None
    beacon_prev: BeaconRound | None
    vrf_proof: bytes | None
    store: MetadataStore | None
    opening: StateOpening | None = None
    justifications: tuple[Justification, ...] = ()
    claimed_taus: list[int] | None = None
    claimed_selected: list[int] | None = None
    requested: list[int] = field(default_factory=list)
    units: dict[int, DisclosedUnit] = field(default_factory=dict)

    @property
    def boost_set(self) -> tuple[int, ...]:
        return self.opening.boost_set if self.opening is not None else ()

    def size_bytes(self) -> dict:
        st = self.store.serialize() if self.store is not None else b""
        units = sum(u.size_bytes() for u in self.units.values())
        op = self.opening.size_bytes if self.opening is not None else 0
        js = sum(len(j.encode()) for j in self.justifications)
        hdr = HDR_LEN + 64 + (72 if self.cosig else 0)
        return {"header": hdr, "beacon": 2 * 232, "vrf": 80, "opening": op + js, "store": len(st), "units": units,
                "total": hdr + 464 + 80 + op + js + len(st) + units}


@dataclass
class Disclosure:
    gw_id: bytes
    epochs: dict[int, EpochDisclosure]
    context: dict[int, PublishedHeader] = field(default_factory=dict)
    gateway_registration: GatewayRegistration | None = None
    policy_registrations: list[PolicyRegistration] = field(default_factory=list)
    counters: dict[int, CounterRecord] = field(default_factory=dict)

    def copy(self) -> "Disclosure":
        return copy.deepcopy(self)

    def header_of(self, e: int) -> tuple[Header, bytes, Cosignature | None] | None:
        if e in self.epochs:
            d = self.epochs[e]
            return d.header, d.sigma, d.cosig
        if e in self.context:
            c = self.context[e]
            return c.header, c.sigma, c.cosig
        return None

    def size_report(self) -> dict:
        per = {e: d.size_bytes() for e, d in self.epochs.items()}
        return {"epochs": len(per), "total_bytes": sum(v["total"] for v in per.values()),
                "store_bytes": sum(v["store"] for v in per.values()),
                "unit_bytes": sum(v["units"] for v in per.values())}

    def privacy_report(self, relevant: dict[int, set[int]] | None = None) -> dict:
        """E8: benign bytes disclosed and third-party metadata records disclosed.

        ``relevant[e]`` = indices that are relevant to the investigation (e.g. hit units); every
        other disclosed content byte is counted as benign, every other metadata record as
        third-party."""
        benign, third = 0, 0
        for e, d in self.epochs.items():
            rel = relevant.get(e, set()) if relevant is not None else set(d.requested)
            for i, u in d.units.items():
                if i not in rel:
                    benign += sum(f.caplen + 16 for f in u.frames)
            if d.store is not None:
                third += sum(1 for i in range(d.store.n) if i not in rel)
        return {"benign_bytes_disclosed": benign, "third_party_records_disclosed": third}


def build_disclosure(gw, epochs: Iterable[int], requested: dict[int, Iterable[int]] | None = None, *,
                     include_claims: bool = True, counters: dict[int, CounterRecord] | None = None,
                     witness=None) -> Disclosure:
    """Honest disclosure from a ``Gateway`` (protocol/epoch.py).

    requested[e]: unit indices requested by the investigator (default: all units of e). Every
    requested unit with Z = 1 is disclosed with frames, frame-hash list and inclusion proof."""
    epochs = sorted(set(int(e) for e in epochs))
    out = Disclosure(gw_id=gw.gw_id, epochs={})
    for e in epochs:
        rec = gw.records[e]
        req = sorted(set(range(rec.batch.n)) if requested is None or e not in requested else set(requested[e]))
        b = rec.beacon_round
        prev = gw.beacon.get(rec.header.r_star - 1) if (rec.header.r_star - 1) in gw.beacon else None
        units = {}
        sel = [i for i in req if rec.Z is not None and 0 <= i < len(rec.Z) and rec.Z[i]]
        if sel:
            proofs = hs.merkle_proofs(rec.leaves, sel)
            for i in sel:
                if not gw.retention.has(e, i):
                    continue
                item = gw.retention.items[(e, i)]
                units[i] = DisclosedUnit(index=i, frames=gw.retention.frames(e, i), pkt_leaves=list(item.pkt_leaves),
                                         proof=proofs[i])
        out.epochs[e] = EpochDisclosure(
            e=e, header=rec.header, sigma=rec.sigma, cosig=rec.cosig, beacon_round=b, beacon_prev=prev,
            vrf_proof=rec.vrf_proof, store=copy.deepcopy(rec.store), opening=rec.opening,
            justifications=tuple(rec.state.justifications),
            claimed_taus=list(rec.taus) if include_claims else None,
            claimed_selected=list(rec.retained) if include_claims else None, requested=req, units=units)
    if epochs:                       # published chain context: every header in [min E' − 1, max E'] not in E'
        for ec in range(epochs[0] - 1, epochs[-1] + 1):
            if ec not in out.epochs and ec in gw.records:
                r = gw.records[ec]
                out.context[ec] = PublishedHeader(r.header, r.sigma, r.cosig)
    w = witness if witness is not None else gw.witness
    if w is not None:
        greg, pregs = w.export_registrations(gw.gw_id)
        out.gateway_registration = greg
        out.policy_registrations = list(pregs)
    if counters:   # signed by I, so the discloser cannot forge them; cover the whole span incl. gaps
        span = range(epochs[0] - 1, epochs[-1] + 2) if epochs else range(0)
        out.counters = {e: counters[e] for e in span if e in counters}
    return out
