"""P2-X revision (coordinator addition A): sensitivity to the GT enabling rule. Owner: P2-X. POST-PREREG, EXPLORATORY.

The banks use the RELATION-AWARE enabling rule (coordinator decision 18 Oct 19:52; gt/semantics.py): enab(A, B) =
the last unit of A starting before first(B) that shares an entity with first(B), else the DESIGN rule. The review
(R2-M4) asks for results under the ORIGINAL DESIGN rule, which does not depend on the candidate generator's relations:
enab(A, B) = the last unit of A starting strictly before first(B) (order (t^s, unit_id)), no entity condition.

This module relabels the cal/test pairs with the DESIGN rule WITHOUT editing P2-B's code: it reconstructs each
incident's EpisodeGT from the stored gt.json + gt_units (verified: the relation-aware relabel reproduces the stored
labels_pairs exactly), runs gt/semantics.label_pair_reason with an index whose entity lookup is switched off (which
is exactly semantics' documented DESIGN-rule path), and rebuilds the loss banks with the UNCHANGED scores (the scorers
were trained under the relation-aware rule; only calibration / test labels change). Output banks live under
results/exp/X2/banks/<ds>/<sid>__X2-design-enab/bank.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from ..gt import semantics as S
from ..io.paths import atomic_write_json, results_root
from . import revision_tasks as RT

RULE_TAG = "X2-design-enab"


class DesignIndex(S._Index):
    """semantics._Index with the entity index switched off -> enab_rule() returns the DESIGN rule ('fallback')."""

    @property
    def ent(self):
        return None


def episode_gt(ds: str, inc: str) -> S.EpisodeGT:
    """EpisodeGT reconstructed from the stored gt.json (EpisodeGT.to_json) and gt_units (EpisodeGT.units_frame)."""
    from ..scoring import driver as D
    gj = D.load_gt_json(ds, inc)
    gu = D.load_gt_units(ds, inc)
    unit_episode = {int(u): e for u, e in zip(gu["unit_id"].to_list(), gu["episode"].to_list()) if e is not None}
    unit_campaign = {int(u): c for u, c in zip(gu["unit_id"].to_list(), gu["campaign"].to_list())}
    mixed = {int(u) for u, m in zip(gu["unit_id"].to_list(), gu["mixed"].to_list()) if m}
    return S.EpisodeGT(unit_episode=unit_episode, unit_campaign=unit_campaign,
                       transitions={(str(a), str(b)) for a, b in gj.get("transitions", [])},
                       hidden=set(gj.get("hidden", [])), episode_campaign=dict(gj.get("episode_campaign", {})),
                       mixed_units=mixed, amb_pairs={(str(a), str(b)) for a, b in gj.get("amb_pairs", [])},
                       amb_unrelated_campaigns=set(gj.get("amb_unrelated_campaigns", [])), meta=gj.get("meta", {}))


def _labels(cands: pl.DataFrame, gt: S.EpisodeGT, ix) -> tuple[pl.DataFrame, dict]:
    ys, bs, rs = [], [], {}
    for u, v in zip(cands["u"].to_list(), cands["v"].to_list()):
        y, b, r = S.label_pair_reason(int(u), int(v), gt, ix)
        ys.append(y)
        bs.append(b)
        rs[r] = rs.get(r, 0) + 1
    df = pl.DataFrame({"cand_id": cands["cand_id"], "y": pl.Series(ys, dtype=pl.Int8),
                       "bridge": pl.Series(bs, dtype=pl.Boolean)})
    return df, rs


def _transition_table(cands: pl.DataFrame, gt: S.EpisodeGT, ix) -> pl.DataFrame:
    """semantics.transition_table with a given index (copy of its logic)."""
    cmap = {(int(u), int(v)): int(c) for c, u, v in zip(cands["cand_id"].to_list(), cands["u"].to_list(),
                                                        cands["v"].to_list())} if cands.height else {}
    rows = []
    for a, b in sorted(gt.transitions):
        if a in ix.hidden or b in ix.hidden or a not in ix.observed or b not in ix.observed:
            rows.append((a, b, "hid", None, None, False, None, None))
            continue
        (e, rule), f = ix.enab_rule(a, b), ix.first(b)
        cid = cmap.get((e, f)) if e is not None else None
        rows.append((a, b, "obs", e, f, cid is not None, cid, rule))
    for a in sorted(ix.observed):
        for b in sorted(ix.bridges(a)):
            if (a, b) in ix.T or (a, b) in gt.amb_pairs:
                continue
            (e, rule), f = ix.enab_rule(a, b), ix.first(b)
            cid = cmap.get((e, f)) if e is not None else None
            rows.append((a, b, "bridge", e, f, cid is not None, cid, rule))
    if not rows:
        return pl.DataFrame(schema=S.TRANSITION_SCHEMA)
    return pl.DataFrame(rows, schema=S.TRANSITION_SCHEMA, orient="row")


def _stats(tt: pl.DataFrame, labels: pl.DataFrame) -> dict:
    obs = tt.filter(pl.col("kind") == "obs")
    hid = tt.filter(pl.col("kind") == "hid")
    br = tt.filter(pl.col("kind") == "bridge")
    n_obs = obs.height
    n_in = int(obs["in_C"].sum()) if n_obs else 0
    y = labels["y"]
    return {"n_T": n_obs + hid.height, "n_T_obs": n_obs, "n_T_hid": hid.height, "n_T_obs_in_C": n_in,
            "r_CT": n_in / n_obs if n_obs else None, "n_bridges": br.height,
            "n_bridges_in_C": int(br["in_C"].sum()) if br.height else 0,
            "n_E_link": int(obs["enab_u"].is_not_null().sum() if n_obs else 0)
            + int(br["enab_u"].is_not_null().sum() if br.height else 0),
            "n_pos": int((y == 1).sum()), "n_null": int((y == 0).sum()), "n_amb": int((y == -1).sum())}


def relabel(ds: str, inc: str, rule: str = "design") -> dict:
    """Pair labels / transition table / stats of one incident under 'design' (DESIGN enab) or 'relation'
    (relation-aware, = the stored labels; used to verify the reconstruction)."""
    from ..scoring import driver as D
    from ..units.scope import ScopeResult
    cands = D.load_cands(ds, inc)
    units = D.load_units(ds, inc)
    scope = ScopeResult.from_json(D.load_scope(ds, inc))
    gt = episode_gt(ds, inc)
    ix = (DesignIndex if rule == "design" else S._Index)(units, gt, scope)
    lab, reasons = _labels(cands, gt, ix)
    tt = _transition_table(cands, gt, ix)
    return {"labels": lab, "transitions": tt, "stats": _stats(tt, lab), "reasons": reasons}


def build_incident_with(ds: str, inc: str, scores: np.ndarray, lab: pl.DataFrame, tr: pl.DataFrame, st: dict):
    """scoring/driver.build_incident with substituted labels / transitions / transition stats (copy of its logic;
    every other input -- candidates, units, gt units, GT paths over T*, devices -- is read as there)."""
    from ..certify.losses import Incident
    from ..scoring import driver as D
    cands = D.load_cands(ds, inc)
    if lab.height != cands.height or (lab.height and not (lab["cand_id"] == cands["cand_id"]).all()):
        raise RuntimeError(f"{ds}:{inc}: labels / candidates mismatch")
    scores = np.asarray(scores, dtype=np.float64)
    pos = {int(c): i for i, c in enumerate(cands["cand_id"].to_list())}
    y = lab["y"].to_numpy().astype(np.int8)
    gtj = D.load_gt_json(ds, inc)
    gtu = D.load_gt_units(ds, inc)
    units = D.load_units(ds, inc)
    N = cands.height
    trans = np.full(N, -1, dtype=np.int64)
    if tr.height:
        obs = tr.filter((pl.col("kind") == "obs") & pl.col("cand_id").is_not_null())
        for j, c in enumerate(obs["cand_id"].to_list()):
            p = pos.get(int(c))
            if p is not None and y[p] == 1:
                trans[p] = j
        edge_pos = {(a, b): pos[int(c)] for a, b, c, k in zip(tr["src_ep"].to_list(), tr["dst_ep"].to_list(),
                                                              tr["cand_id"].to_list(), tr["kind"].to_list())
                    if c is not None and k == "obs" and int(c) in pos}
    else:
        edge_pos = {}
    camps = (gtj.get("meta") or {}).get("campaigns") or {}
    decoy_camps = {c for c, d in camps.items() if (d or {}).get("kind", "main") != "main"}
    main_camps = {c for c, d in camps.items() if (d or {}).get("kind", "main") == "main"}
    ucamp = dict(zip(gtu["unit_id"].to_list(), gtu["campaign"].to_list())) if gtu.height else {}
    n_units = units.height
    decoy_unit = np.array([ucamp.get(i) in decoy_camps for i in range(n_units)], dtype=bool)
    dev, _ = D._unit_devices(units)
    camp_units = ([i for i in range(n_units) if ucamp.get(i) in main_camps] if camps
                  else [i for i in range(n_units) if ucamp.get(i) is not None])
    campaign_devices = sorted({int(dev[i]) for i in camp_units})
    paths = D._gt_paths(gtj.get("transitions", []), edge_pos)
    L = max((len(p) for p in paths), default=0)
    strata = {"decoy_rich": len(decoy_camps) >= 2,
              "benign_only": (len(main_camps) == 0) if camps else (st.get("n_T", 0) == 0),
              "short_chain": L <= 2, "n_decoys": len(decoy_camps)}
    return Incident(incident_id=f"{ds}:{inc}", scores=scores, y=y, trans=trans,
                    bridge=lab["bridge"].fill_null(False).to_numpy().astype(bool), decoy_unit=decoy_unit,
                    unit_device=dev, campaign_devices=campaign_devices, gt_paths=paths,
                    n_trans=st.get("n_T"), n_trans_obs=st.get("n_T_obs"), n_trans_hid=st.get("n_T_hid"),
                    n_elink=st.get("n_E_link"), n_bridges=st.get("n_bridges"), strata=strata,
                    u=cands["u"].to_numpy().astype(np.int64), v=cands["v"].to_numpy().astype(np.int64),
                    unit_ts=units["ts"].to_numpy().astype(np.float64))


def null_pool(sid: str) -> np.ndarray:
    from ..scoring import driver as D
    scorer, variant = sid.split(".", 1)
    name = {"stat": "null_pool.npy"}.get(variant, f"null_pool_{variant}.npy")
    return np.load(D.models_dir(scorer) / name)


def read_scores(ds: str, sid: str, inc: str) -> np.ndarray:
    from ..scoring import driver as D
    df = pl.read_parquet(D.scores_path(ds, sid, inc)).sort("cand_id")
    cids = D.load_cands(ds, inc)["cand_id"].to_numpy()
    if df.height != cids.size or not np.array_equal(df["cand_id"].to_numpy(), cids):
        raise RuntimeError(f"scores of {ds}/{sid}/{inc} do not match the candidate view")
    return df["s"].to_numpy().astype(np.float64)


def bank_path(ds: str, sid: str, tag: str = RULE_TAG) -> Path:
    return RT.x2_root() / "banks" / ds / f"{sid}__{tag}" / "bank"


def build_relabel_banks(ds: str, sids, *, n_jobs: int = 8, verify: bool = True) -> dict:
    """Relabel the cal/test pool of ``ds`` with the DESIGN rule and build one bank per scorer id."""
    from ..certify.losses import build_loss_tables
    from ..scoring import driver as D
    t0 = time.perf_counter()
    incs = D.list_incidents(ds, "caltest")
    rel = {}
    mism = []
    agg = {"design": {}, "relation": {}}
    for inc in incs:
        r = relabel(ds, inc, "design")
        rel[inc] = r
        for k, v in r["stats"].items():
            agg["design"][k] = agg["design"].get(k, 0) + (v or 0 if not isinstance(v, float) else v)
        if verify:
            ref = D.load_labels(ds, inc)
            chk = relabel(ds, inc, "relation")
            if not (chk["labels"]["y"].to_list() == ref["y"].to_list()
                    and chk["labels"]["bridge"].fill_null(False).to_list() == ref["bridge"].fill_null(False).to_list()):
                mism.append(inc)
            for k, v in chk["stats"].items():
                agg["relation"][k] = agg["relation"].get(k, 0) + (v or 0 if not isinstance(v, float) else v)
    for a in agg.values():
        a["r_CT_pooled"] = a.get("n_T_obs_in_C", 0) / max(a.get("n_T_obs", 0), 1)
        a.pop("r_CT", None)
    if mism:
        raise RuntimeError(f"{ds}: EpisodeGT reconstruction does not reproduce the stored labels in {len(mism)} "
                           f"incidents (e.g. {mism[:3]})")
    reasons: dict = {}
    for r in rel.values():
        for k, v in r["reasons"].items():
            reasons[k] = reasons.get(k, 0) + v
    out = {"ds": ds, "n_incidents": len(incs), "verified_relation_rule_reproduces_stored_labels": bool(verify),
           "pooled_stats": agg, "label_reasons_design": reasons, "banks": {}}
    for sid in sids:
        I = [build_incident_with(ds, inc, read_scores(ds, sid, inc), rel[inc]["labels"], rel[inc]["transitions"],
                                 rel[inc]["stats"]) for inc in incs]
        bp = bank_path(ds, sid)
        bank = build_loss_tables(I, null_pool(sid), bp, n_jobs=n_jobs, overwrite=True,
                                 meta={"dataset": ds, "scorer_id": f"{sid}__{RULE_TAG}", "pool": "caltest",
                                       "gt_rule": "DESIGN enab (no entity condition); scores unchanged"})
        out["banks"][sid] = {"path": str(bp), "n": bank.n, "input_hash": bank.meta.get("input_hash")}
    out["seconds"] = time.perf_counter() - t0
    atomic_write_json(RT.x2_root() / "banks" / ds / f"relabel_{RULE_TAG}.json", out)
    return out
