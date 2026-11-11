"""IoT-23 GT extractor (DESIGN §5.2.2) — label-derived episodes for gt/semantics.label_pairs.

* Unit phase: majority ``detailed_label`` of the unit's malicious records if ≥ 50 % of them
  share it (a 50/50 tie is resolved only if both labels map to the same class), else MIXED
  (pairs with a MIXED unit are AMB unless rule 1 already gives 0). Units without malicious
  records are ⊥. Malicious detailed labels outside the six classes (e.g. '-') form class OTHER.
* Phase classes: DL {FileDownload}; CC {C&C, C&C-HeartBeat, C&C-Mirai, C&C-Torii};
  SCAN {PartOfAHorizontalPortScan, C&C-PartOfAHorizontalPortScan};
  ATK {Attack, DDoS, Okiru-Attack, C&C-HeartBeat-Attack}; OKIRU {Okiru};
  CCDL {C&C-FileDownload, C&C-HeartBeat-FileDownload} (download commanded over C&C).
* Episodes: maximal runs (unit start order) of the seed's units (src = seed) of one class with
  no intervening unit of another malicious class (benign and MIXED units do not break runs);
  a CC run is split where the idle gap to the next CC unit exceeds 1 h. Units of other hosts
  are ⊥ unless malicious; then they form their own episodes (same run rule, per host) with
  no transitions. Campaign = the scenario's infection ("iot23:<scenario>").
* T*: R1 DL → the next CC episode; R2 most recent CC episode before an action episode
  (SCAN, ATK, OKIRU, CCDL) → that episode; R0 hidden EXEC → the first malicious seed
  episode (one per incident window: Execution, or the truncated pre-window predecessor).
* Malicious pairs not covered by R1/R2 (transitive, reverse, unexplained) are AMB:
  the campaign is listed in ``amb_unrelated_campaigns``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import polars as pl

from certichain.gt.semantics import EpisodeGT, read_quarantined
from certichain.units.sessionize import unit_record_map

PHASE_CLASSES: dict[str, tuple[str, ...]] = {
    "DL": ("FileDownload",),
    "CC": ("C&C", "C&C-HeartBeat", "C&C-Mirai", "C&C-Torii"),
    "SCAN": ("PartOfAHorizontalPortScan", "C&C-PartOfAHorizontalPortScan"),
    "ATK": ("Attack", "DDoS", "Okiru-Attack", "C&C-HeartBeat-Attack"),
    "OKIRU": ("Okiru",),
    "CCDL": ("C&C-FileDownload", "C&C-HeartBeat-FileDownload"),
}
_TO_CLASS = {lab.lower(): cls for cls, labs in PHASE_CLASSES.items() for lab in labs}
ACTION_CLASSES = ("SCAN", "ATK", "OKIRU", "CCDL")
CC_SPLIT_GAP = 3600.0
EXEC = "EXEC(hidden)"
MIXED = "MIXED"


def phase_class(detailed_label: str | None) -> str:
    d = (detailed_label or "").strip()
    return _TO_CLASS.get(d.lower(), "OTHER")


def _is_mal(label: pl.Expr) -> pl.Expr:
    return label.fill_null("").str.to_lowercase().str.contains("malicious")


def unit_phases(units: pl.DataFrame, labels: pl.DataFrame) -> dict[int, str]:
    """unit_id → phase class | MIXED (units absent ⇒ no malicious record ⇒ ⊥)."""
    m = unit_record_map(units).join(labels.select("rec_id", "label", "detailed_label"), on="rec_id", how="left")
    mal = m.filter(_is_mal(pl.col("label"))).with_columns(
        pl.col("detailed_label").fill_null("").str.strip_chars().alias("dl"))
    mal = mal.with_columns(pl.when(pl.col("dl") == "-").then(pl.lit("")).otherwise(pl.col("dl")).alias("dl"))
    if mal.height == 0:
        return {}
    cnt = mal.group_by(["unit_id", "dl"]).agg(pl.len().alias("c"))
    tot = cnt.group_by("unit_id").agg(pl.col("c").sum().alias("n"))
    cnt = cnt.join(tot, on="unit_id").filter(pl.col("c").cast(pl.Float64) >= 0.5 * pl.col("n").cast(pl.Float64))
    top: dict[int, list[str]] = {}
    for uid, dl in cnt.select("unit_id", "dl").iter_rows():
        top.setdefault(int(uid), []).append(dl)
    out: dict[int, str] = {}
    for uid in tot["unit_id"].to_list():
        labs = top.get(int(uid), [])
        classes = {phase_class(d) for d in labs}
        out[int(uid)] = classes.pop() if len(classes) == 1 else MIXED
    return out


@dataclass
class _Run:
    cls: str
    host: str
    k: int
    units: list[int] = field(default_factory=list)
    max_te: float = float("-inf")

    @property
    def eid(self) -> str:
        return f"{self.cls}#{self.k}@{self.host}"


def _runs(seq: list[tuple[float, float, int]], phases: dict[int, str], host: str,
          cc_gap: float) -> tuple[list[_Run], list[int]]:
    runs: list[_Run] = []
    mixed: list[int] = []
    for ts, te, uid in seq:
        p = phases.get(uid)
        if p is None:
            continue
        if p == MIXED:
            mixed.append(uid)
            continue
        cur = runs[-1] if runs else None
        if cur is not None and cur.cls == p and not (p == "CC" and ts - cur.max_te > cc_gap):
            cur.units.append(uid)
            cur.max_te = max(cur.max_te, te)
        else:
            runs.append(_Run(cls=p, host=host, k=len(runs), units=[uid], max_te=te))
    return runs, mixed


def build_iot23_gt(units: pl.DataFrame, labels: pl.DataFrame, seeds: Sequence[str], scenario: str,
                   window_id: str | None = None, cc_split_gap: float = CC_SPLIT_GAP) -> EpisodeGT:
    camp = f"iot23:{scenario}"
    phases = unit_phases(units, labels)
    seeds = set(seeds)
    rows = units.select("unit_id", "ts", "te", "src").sort(["ts", "unit_id"]).iter_rows()
    by_host: dict[str, list[tuple[float, float, int]]] = {}
    for uid, ts, te, src in rows:
        key = "SEED" if src in seeds else src
        by_host.setdefault(key, []).append((float(ts), float(te), int(uid)))
    gt = EpisodeGT(unit_episode={}, unit_campaign={}, transitions=set(), hidden=set(),
                   episode_campaign={}, mixed_units=set(), amb_unrelated_campaigns={camp},
                   meta={"source": "iot23", "scenario": scenario, "window_id": window_id})
    seed_runs: list[_Run] = []
    for host, seq in sorted(by_host.items()):
        runs, mixed = _runs(seq, phases, "seed" if host == "SEED" else host, cc_split_gap)
        for r in runs:
            gt.episode_campaign[r.eid] = camp
            for uid in r.units:
                gt.unit_episode[uid] = r.eid
                gt.unit_campaign[uid] = camp
        for uid in mixed:
            gt.mixed_units.add(uid)
            gt.unit_campaign[uid] = camp
        if host == "SEED":
            seed_runs = runs
    # R1 / R2 over the seed's run sequence
    for i, r in enumerate(seed_runs):
        if r.cls == "DL":
            nxt = next((q for q in seed_runs[i + 1:] if q.cls == "CC"), None)
            if nxt is not None:
                gt.transitions.add((r.eid, nxt.eid))
        elif r.cls in ACTION_CLASSES:
            prev = next((q for q in reversed(seed_runs[:i]) if q.cls == "CC"), None)
            if prev is not None:
                gt.transitions.add((prev.eid, r.eid))
    # R0: hidden Execution → first observed malicious seed episode
    if seed_runs:
        gt.transitions.add((EXEC, seed_runs[0].eid))
        gt.hidden.add(EXEC)
        gt.episode_campaign[EXEC] = camp
    gt.meta.update({"n_seed_episodes": len(seed_runs),
                    "n_other_episodes": len(gt.episode_campaign) - len(seed_runs) - (1 if seed_runs else 0),
                    "n_mixed_units": len(gt.mixed_units),
                    "seed_classes": [r.cls for r in seed_runs]})
    return gt


def iot23_gt(scenario: str, window_id: str, units: pl.DataFrame, seeds: Sequence[str],
             labels_path: str | None = None) -> EpisodeGT:
    """Read the window's quarantined labels (single reader) and build the EpisodeGT."""
    # the whole window's label table is read; build_iot23_gt joins it to the units' rec_ids
    labels = read_quarantined("iot23", scenario=scenario, window_id=window_id, path=labels_path)
    return build_iot23_gt(units, labels, seeds, scenario, window_id)
