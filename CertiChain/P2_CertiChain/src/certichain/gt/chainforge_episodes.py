"""ChainForge GT adapter (DESIGN §5.2.1, §5.4) — exact, generator-provided episodes.

Interface (written by the generator, P2-E; read here only):

  results/interim/<variant>/incidents/<inc_id>/      variant ∈ cf_ton, cf_ton_h, cf_gotham, cf_gotham_t
    records.parquet  RECORD_SCHEMA (rec_id dense 0..n-1, randomly permuted → carries no order)
    meta.json        case metadata only: {inc_id, variant, pool, split_role, seed_device, t0, t1,
                     scope_rule, internal_prefixes, generator_seed}
    labels.parquet   QUARANTINED: rec_id, label, detailed_label, episode_id (null = background),
                     campaign_id (null = background), decoy_kind (null|D1|D2|D3), template_node
    gt.json          QUARANTINED: {"campaigns": {cid: {"template", "kind": main|D1|D2|D3}},
                     "episodes": {eid: {"campaign_id", "type", "template_node", "hidden"}},
                     "transitions": [[eid_a, eid_b], ...]   (template edges incl. pivot ⇒),
                     "hidden": [eid, ...]}

Unit mapping: ep(u) = the episode_id held by ≥ 50 % of ALL the unit's records (unique
maximum); no episode record ⇒ ⊥ (background); episode records present but no episode with
≥ 50 % ⇒ MIXED (AMB unless rule 1 gives 0; its campaign is kept when all its episode
records share one campaign). camp(u) = campaign of ep(u) from gt.json. Decoys are
campaigns of their own (D1/D3 single episodes ⇒ no positives; D2 internal edges are true).
Hidden = gt.json "hidden" ∪ episodes flagged hidden (and, per incident, any episode left
without a unit by the scope rule — handled in semantics).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from certichain.gt.semantics import EpisodeGT, read_quarantined, unit_majority
from certichain.units.sessionize import unit_record_map


def build_chainforge_gt(units: pl.DataFrame, labels: pl.DataFrame, gtj: dict,
                        min_share: float = 0.5) -> EpisodeGT:
    eps = gtj.get("episodes", {}) or {}
    ep_camp = {eid: d.get("campaign_id") for eid, d in eps.items()}
    hidden = set(gtj.get("hidden", []) or []) | {eid for eid, d in eps.items() if d.get("hidden")}
    transitions = {(str(a), str(b)) for a, b in gtj.get("transitions", []) or []}
    gt = EpisodeGT(unit_episode={}, unit_campaign={}, transitions=transitions, hidden=hidden,
                   episode_campaign={k: v for k, v in ep_camp.items() if v is not None},
                   meta={"source": "chainforge", "campaigns": gtj.get("campaigns", {})})
    if units.height == 0:
        return gt
    lab = labels.select("rec_id", pl.col("episode_id").cast(pl.Utf8), pl.col("campaign_id").cast(pl.Utf8))
    maj = unit_majority(units, lab, "episode_id", min_share=min_share, denominator="all")
    # campaign of a MIXED unit: kept only if all its episode records share one campaign
    m = unit_record_map(units).join(lab, on="rec_id", how="left")
    ncamp = (m.filter(pl.col("campaign_id").is_not_null()).group_by("unit_id")
             .agg(pl.col("campaign_id").n_unique().alias("nc"), pl.col("campaign_id").first().alias("c")))
    one_camp = {int(u): c for u, nc, c in ncamp.iter_rows() if nc == 1}
    for uid, val, mixed, _, _ in maj.iter_rows():
        uid = int(uid)
        if mixed:
            gt.mixed_units.add(uid)
            gt.unit_campaign[uid] = one_camp.get(uid)
        elif val is not None:
            gt.unit_episode[uid] = val
            gt.unit_campaign[uid] = ep_camp.get(val) or one_camp.get(uid)
    gt.meta.update({"n_episode_units": len(gt.unit_episode), "n_mixed_units": len(gt.mixed_units),
                    "n_episodes_declared": len(eps), "n_hidden_declared": len(hidden)})
    return gt


def chainforge_gt(incident_dir: str | Path, units: pl.DataFrame) -> EpisodeGT:
    """EpisodeGT of one ChainForge incident directory (reads labels.parquet + gt.json)."""
    labels = read_quarantined("chainforge", what="labels", incident_dir=incident_dir)
    gtj = read_quarantined("chainforge", what="gt", incident_dir=incident_dir)
    gt = build_chainforge_gt(units, labels, gtj)
    gt.meta["incident_dir"] = str(incident_dir)
    return gt
