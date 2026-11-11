"""Withholding / abort adversaries of DESIGN §6.3: A7 (withhold an epoch) and A11 (selective abort).

A7 — the gateway never publishes Hdr_e (decided before β, so it is Z-independent). The next header
links to Hdr_{e−1}. Detected by V1 (missing header for e, broken H(Hdr_{e−1}) link at e+1); the gap
is listed with its counter volumes.

A11 — selective abort AFTER β: the gateway "voids" (fails to produce evidence for) epochs chosen by
looking at Z:
  rule "incriminating":  void epochs where the target's hit unit was NOT retained
                         (the remaining epochs over-represent hits);
  rule "exculpatory":    void epochs where the target's hit unit WAS retained.
Voiding modes: "content" (frames of the epoch's retained units withheld → V6 MISSING),
"proof" (Π_e withheld → V3 FAIL, whole epoch in 𝒲), "epoch" (whole epoch disclosure withheld →
V1 FAIL). In all modes the voided requested units land in 𝒲 (Prop. 2: X := 0 keeps the LCS valid;
"compliant only" estimates miscover — tested by P1-C in test_abort_robust).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

from ..protocol.disclosure import Disclosure
from ..units.batch import EpochBatch

__all__ = ["run_with_withheld_epochs", "void_epochs", "a11_selective_abort", "RULES"]

RULES = ("incriminating", "exculpatory")


def run_with_withheld_epochs(gw, batches: Iterable[EpochBatch], withheld: set[int]) -> list:
    """Drive ``gw`` over consecutive ``batches`` but never commit/publish the epochs in ``withheld``
    (A7). Returns the records of the published epochs."""
    out = []
    for b in sorted(batches, key=lambda x: x.epoch):
        e = b.epoch
        while gw.next_e < e:                         # honest empty epochs in between
            out.append(gw.run_epoch(gw.empty_batch(gw.next_e)))
        if e in withheld:
            gw.next_e = e + 1                        # skip: no header, prev_hash unchanged
            gw.state = replace(gw.state, e=e + 1)
            continue
        out.append(gw.run_epoch(b))
    return out


def void_epochs(Z: dict[int, Sequence[bool]], hits: dict[int, Iterable[int]], rule: str) -> set[int]:
    """Epochs to void under an A11 rule, from the true selection Z and the target's hit units."""
    if rule not in RULES:
        raise ValueError(rule)
    out = set()
    for e, h in hits.items():
        h = list(h)
        if not h or e not in Z:
            continue
        retained = any(Z[e][i] for i in h)
        if (rule == "incriminating" and not retained) or (rule == "exculpatory" and retained):
            out.add(e)
    return out


def a11_selective_abort(disc: Disclosure, gw, hits: dict[int, Iterable[int]], rule: str,
                        mode: str = "content") -> tuple[Disclosure, set[int]]:
    """Apply an A11 rule to an honest disclosure using the gateway's true Z."""
    Z = {e: gw.records[e].Z for e in disc.epochs}
    voided = void_epochs(Z, hits, rule)
    d2 = disc.copy()
    for e in voided:
        ed = d2.epochs.get(e)
        if ed is None:
            continue
        if mode == "content":
            ed.units = {}
            if ed.claimed_selected is not None:
                ed.claimed_selected = []
        elif mode == "proof":
            ed.vrf_proof = None
        elif mode == "epoch":
            del d2.epochs[e]
        else:
            raise ValueError(mode)
    return d2, voided
