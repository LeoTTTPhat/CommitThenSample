#!/usr/bin/env python
"""19 — few-shot cards, 2,000-candidate dev pilot, main-model choice (DESIGN §10 R1) and prompt freeze
(DESIGN §3.4.3, §8.3). Owner: P2-F.

Run from the project root under memguard WITHOUT -I (loads MLX models from the HF cache; one model at a time
under the GPU lock):

  .venv/bin/python memguard/memguard.py run --name p2_pilot --limit-gb 90 --workers 1 -- \
      .venv/bin/python -u P2_CertiChain/scripts/19_fewshot_and_pilot.py --freeze

  --select-only   only select + render the few-shot cards and the pilot set (no GPU), print them
  --models a,b    pilot models (configs/llm.yaml pilot.candidates)
  --freeze        write results/gates/prompt_freeze.json (+ prompt_prefix_<hash>.txt); immutable afterwards

Few-shot selection rule (deterministic; CF-TON DEV incidents whose every campaign uses a train-H template:
main / D2 template in TRAIN_H, D1 type in the train-H episode types, D3 look-alikes allowed; incidents and
candidates scanned in (incident id, cand_id) order):
  rev 0: Yes-1 = first Y = 1 candidate; Yes-2 = first Y = 1 candidate of a different template AND a different
         (type_u, type_v) unit-type pair; decoy-No = first Y = 0 candidate with one endpoint in a decoy campaign,
         the other in a main campaign, a PIVOT/REVERSE/SAME_VICTIM relation and two attack-like unit types;
         activity-No = first AMB candidate whose endpoints are in the same episode, with different attack-like
         unit types and lag_gap <= 600 s. Prompt order: Yes-1, decoy-No, Yes-2, activity-No.
  rev 1 (only if the chosen model's dev AUROC < 0.6): Yes = first candidate of each of the two most frequent
         Y = 1 unit-type pairs; decoy-No as rev 0; background-No = first Y = 0 candidate with both endpoints
         outside every campaign sharing a host (PIVOT/REVERSE/SAME_VICTIM). Order: decoy-No, Yes-1, bg-No, Yes-2.
Pilot: in-band dev candidates (S_TON band on the S_TON s_stat), card candidates excluded, uniform sample of
n = 2000 (numpy default_rng(0)), sorted by (incident, cand_id); in-band AUROC of s_LLM (Y = 1 vs Y != 1, and
vs Y = 0) per model. Dev labels only (never reported in a paper table).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from certichain.eval.gates import auc
from certichain.io.paths import P2_ROOT, atomic_write_json, atomic_write_text, results_root
from certichain.io.schema import REL_PIVOT, REL_REVERSE, REL_SAME_VICTIM
from certichain.scoring import driver as D
from certichain.units.unit_types import BENIGN_PROTOCOL_TYPES

DS = "cf_ton"
SCORER = "S_TON"
STRONG_REL = REL_PIVOT | REL_REVERSE | REL_SAME_VICTIM


def log(msg: str) -> None:
    print(f"[19 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cfg() -> dict:
    return yaml.safe_load((P2_ROOT / "configs" / "llm.yaml").read_text())


# ----------------------------------------------------------------------------- eligibility
def train_h_sets():
    from certichain.chainforge.templates import TEMPLATES, TRAIN_H
    types = set()
    for t in TRAIN_H:
        types |= set(TEMPLATES[t].types(observable_only=False))
    return set(TRAIN_H), types


def eligible(gtj: dict, train_h: set, train_h_types: set) -> bool:
    camps = (gtj.get("meta") or {}).get("campaigns") or {}
    if not camps:
        return False
    for c in camps.values():
        kind, tmpl = c.get("kind"), str(c.get("template"))
        if kind in ("main", "D2"):
            if tmpl not in train_h:
                return False
        elif kind == "D1":
            if tmpl.split(":", 1)[-1] not in train_h_types:
                return False
        elif kind != "D3":
            return False
    return any(c.get("kind") == "main" for c in camps.values())


def incident_table(inc: str) -> pl.DataFrame:
    """Per candidate of a DEV incident: ids, labels, unit types, episodes/campaign kinds (GT side, dev only)."""
    c = D.load_cands(DS, inc)
    lab = D.load_labels(DS, inc)
    u = D.load_units(DS, inc).select("unit_id", "type")
    g = D.load_gt_units(DS, inc).select("unit_id", "episode", "campaign")
    gtj = D.load_gt_json(DS, inc)
    camps = (gtj.get("meta") or {}).get("campaigns") or {}
    kind = {k: v.get("kind") for k, v in camps.items()}
    tmpl = {k: v.get("template") for k, v in camps.items()}
    ug = u.join(g, on="unit_id", how="left")
    ug = ug.with_columns(pl.col("campaign").replace_strict(kind, default=None, return_dtype=pl.Utf8).alias("kind"),
                         pl.col("campaign").replace_strict(tmpl, default=None, return_dtype=pl.Utf8).alias("template"))
    d = (c.join(lab, on="cand_id")
         .join(ug.rename({k: f"{k}_u" for k in ug.columns if k != "unit_id"}).rename({"unit_id": "u"}), on="u", how="left")
         .join(ug.rename({k: f"{k}_v" for k in ug.columns if k != "unit_id"}).rename({"unit_id": "v"}), on="v", how="left"))
    return d.with_columns(pl.lit(inc).alias("incident")).sort("cand_id")


def attack_like(t) -> bool:
    return t is not None and t not in BENIGN_PROTOCOL_TYPES and not str(t).startswith("BENIGN")


def select_cards(incs: list[str], rev: int) -> list[dict]:
    train_h, th_types = train_h_sets()
    elig = [i for i in incs if eligible(D.load_gt_json(DS, i), train_h, th_types)]
    log(f"few-shot: {len(elig)}/{len(incs)} dev incidents use train-H templates only")
    if not elig:
        raise RuntimeError("no eligible dev incident for few-shot cards")
    T = pl.concat([incident_table(i) for i in elig], how="diagonal_relaxed").sort(["incident", "cand_id"])
    rows = list(T.iter_rows(named=True))
    yes = [r for r in rows if r["y"] == 1]
    out: list[dict] = []

    def pick(r, answer, role):
        out.append({"incident": r["incident"], "cand_id": int(r["cand_id"]), "answer": answer, "role": role,
                     "type_u": r["type_u"], "type_v": r["type_v"], "template": r.get("template_u"),
                     "y": int(r["y"])})

    def decoy_no():
        for r in rows:
            ku, kv = r["kind_u"], r["kind_v"]
            if r["y"] == 0 and {ku, kv} & {"D1", "D2", "D3"} and "main" in (ku, kv) \
                    and (int(r["rel_mask"]) & STRONG_REL) and attack_like(r["type_u"]) and attack_like(r["type_v"]):
                return r
        return None
    if rev == 0:
        y1 = yes[0]
        y2 = next(r for r in yes[1:] if r["template_u"] != y1["template_u"]
                  and (r["type_u"], r["type_v"]) != (y1["type_u"], y1["type_v"]))
        dn = decoy_no()
        an = next(r for r in rows if r["y"] == -1 and r["episode_u"] is not None and r["episode_u"] == r["episode_v"]
                  and r["type_u"] != r["type_v"] and attack_like(r["type_u"]) and attack_like(r["type_v"])
                  and r["lag_gap"] <= 600)
        for r, a, role in ((y1, "Yes", "yes_1"), (dn, "No", "decoy_no"), (y2, "Yes", "yes_2"), (an, "No", "activity_no")):
            if r is None:
                raise RuntimeError(f"few-shot rule rev 0: no candidate for {role}")
            pick(r, a, role)
    else:
        from collections import Counter
        cnt = Counter((r["type_u"], r["type_v"]) for r in yes)
        top = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
        ys = [next(r for r in yes if (r["type_u"], r["type_v"]) == k) for k, _ in top]
        dn = decoy_no()
        bn = next(r for r in rows if r["y"] == 0 and r["campaign_u"] is None and r["campaign_v"] is None
                  and (int(r["rel_mask"]) & STRONG_REL))
        for r, a, role in ((dn, "No", "decoy_no"), (ys[0], "Yes", "yes_1"), (bn, "No", "background_no"),
                           (ys[1], "Yes", "yes_2")):
            if r is None:
                raise RuntimeError(f"few-shot rule rev 1: no candidate for {role}")
            pick(r, a, role)
    return out


def render_fewshot(cards: list[dict], model_id: str):
    from certichain.features.cards import token_counter
    from certichain.llm.prompts import FewShot
    counter = token_counter(model_id)
    shots = []
    for c in cards:
        txt = D.CardBuilder(DS, c["incident"], counter, 220).card(c["cand_id"])
        c["card"] = txt
        shots.append(FewShot(card=txt, answer=c["answer"],
                             source=f"CF-TON dev {c['incident']} cand {c['cand_id']} ({c['role']})"))
    return shots


# ----------------------------------------------------------------------------- pilot
def pilot_items(incs: list[str], exclude: set, n: int, seed: int) -> list[tuple[str, int]]:
    band = D.load_band(SCORER)
    allin = [it for it in D.band_members(DS, SCORER, incs, band) if it not in exclude]
    if len(allin) <= n:
        return sorted(allin)
    idx = np.random.default_rng(seed).choice(len(allin), size=n, replace=False)
    return sorted(allin[int(i)] for i in idx)


def run_pilot(model_id: str, prefix: str, items, group: str) -> dict:
    from certichain.features.cards import token_counter
    from certichain.llm.mlx_scorer import MLXScorer, ScorerConfig, read_scores
    from certichain.utils.gpu_lock import gpu_lock
    counter = token_counter(model_id)
    t0 = time.perf_counter()
    with gpu_lock("P2", f"p2_pilot_{model_id.split('/')[-1].lower()}"):
        sc = MLXScorer(model_id, prefix, ScorerConfig())
        try:
            st: dict = {}
            out = sc.run(D.iter_cards(DS, items, counter, 220, st), D.llm_root(DS) / group)
            meta = sc.metadata()
        finally:
            sc.close()
    wall = time.perf_counter() - t0
    df = read_scores(out)
    return {"df": df, "wall_s": wall, "meta": meta, "shard_dir": str(out), "render": st}


def pilot_auroc(df: pl.DataFrame) -> dict:
    y = []
    for inc, g in df.group_by("incident"):
        lab = D.load_labels(DS, inc[0]).select("cand_id", "y")
        y.append(g.select("incident", "cand_id", "s_llm").join(lab, on="cand_id", how="left"))
    t = pl.concat(y)
    s = t["s_llm"].to_numpy().astype(float)
    yy = t["y"].to_numpy()
    return {"n": int(t.height), "n_pos": int((yy == 1).sum()), "n_amb": int((yy == -1).sum()),
            "n_null0": int((yy == 0).sum()), "auroc_strict": auc(s[yy == 1], s[yy != 1]),
            "auroc_vs_null0": auc(s[yy == 1], s[yy == 0]),
            "s_llm_mean_pos": float(s[yy == 1].mean()) if (yy == 1).any() else None,
            "s_llm_mean_neg": float(s[yy != 1].mean()) if (yy != 1).any() else None,
            "frac_positive_s": float((s > 0).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--models", default=None)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    cfg = load_cfg()
    models = cfg["models"]
    cand_models = (a.models.split(",") if a.models else cfg["pilot"]["candidates"])
    n = a.n or int(cfg["pilot"]["n"])
    freeze_p = D.gates_dir() / "prompt_freeze.json"
    if freeze_p.exists() and a.freeze:
        log(f"{freeze_p} exists: the prompt is frozen; refusing to re-freeze")
        return 3
    from certichain.llm.prompts import build_prefix, prompt_hash, uses_placeholders
    incs = D.list_incidents(DS, "dev")
    history = []
    chosen = None
    for rev in (0, 1):
        cards = select_cards(incs, rev)
        shots = render_fewshot(cards, models[cfg["pilot"]["card_tokenizer"]])
        prefix = build_prefix(fewshot=shots)
        assert not uses_placeholders(shots)
        ph = prompt_hash(prefix)
        excl = {(c["incident"], c["cand_id"]) for c in cards}
        items = pilot_items(incs, excl, n, int(cfg["pilot"]["sample_seed"]))
        rec = {"rev": rev, "prompt_hash": ph, "cards": cards, "n_pilot": len(items),
               "card_input_source_hash": D.card_input_source_hash()}
        atomic_write_json(D.gates_dir() / f"fewshot_rev{rev}.json", rec)
        atomic_write_text(D.gates_dir() / f"prompt_prefix_{ph}.txt", prefix)
        log(f"rev {rev}: prompt hash {ph}; {len(items)} pilot candidates; cards: "
            + "; ".join(f"{c['role']}={c['incident']}#{c['cand_id']}" for c in cards))
        if a.select_only:
            for c in cards:
                print(f"--- {c['role']} ({c['answer']})\n{c['card']}")
            return 0
        res = {}
        for m in cand_models:
            mid = models.get(m, m)
            r = run_pilot(mid, prefix, items, f"dev_pilot_rev{rev}")
            au = pilot_auroc(r["df"])
            au.update({"model_id": mid, "model_revision": r["meta"]["model_revision"], "wall_s": round(r["wall_s"], 1),
                       "cand_per_s_incl_load": len(items) / r["wall_s"] if r["wall_s"] else None,
                       "mean_ms_per_cand": float(r["df"]["ms"].mean()) if r["df"].height else None,
                       "suffix_tokens_mean": float(r["df"]["n_tok"].mean()) if r["df"].height else None,
                       "n_truncated": int(r["df"]["truncated"].sum()) if r["df"].height else 0,
                       "shard_dir": r["shard_dir"], "render": r["render"]})
            res[m] = au
            atomic_write_json(D.gates_dir() / f"pilot_rev{rev}_{m}.json", au)
            log(f"rev {rev} {m}: in-band AUROC strict {au['auroc_strict']:.4f} (vs y=0 {au['auroc_vs_null0']:.4f}); "
                f"{au['cand_per_s_incl_load']:.2f} cand/s incl. load")
        # DESIGN §10 R1
        a30 = res.get("qwen3moe", {}).get("auroc_strict", float("nan"))
        a7 = res.get("qwen7b", {}).get("auroc_strict", float("nan"))
        if "qwen3moe" in res and "qwen7b" in res:
            main_m = "qwen3moe" if a30 >= a7 else "qwen7b"
        else:
            main_m = max(res, key=lambda k: res[k]["auroc_strict"])
        history.append({"rev": rev, "prompt_hash": ph, "results": res, "chosen": main_m})
        chosen = (rev, main_m, ph, prefix, cards, res)
        if res[main_m]["auroc_strict"] >= float(cfg["pilot"]["min_auroc"]):
            break
        log(f"rev {rev}: chosen model AUROC {res[main_m]['auroc_strict']:.3f} < {cfg['pilot']['min_auroc']}"
            + ("; one prompt revision on dev" if rev == 0 else "; freezing rev 1 anyway (one revision allowed)"))
    rev, main_m, ph, prefix, cards, res = chosen
    out = {"frozen": bool(a.freeze), "prompt_hash": ph, "prefix_file": f"results/gates/prompt_prefix_{ph}.txt",
           "prefix_blake3": D.file_blake3(D.gates_dir() / f"prompt_prefix_{ph}.txt"),
           "card_input_source_hash": D.card_input_source_hash(), "fewshot_rev": rev, "fewshot_cards": cards,
           "main_model": main_m, "main_model_id": models.get(main_m, main_m),
           "main_model_revision": res[main_m]["model_revision"], "rule": "DESIGN §10 R1: Qwen3-30B-A3B main iff its "
           "dev in-band AUROC >= Qwen2.5-7B's; AUROC < 0.6 -> one prompt revision on dev, then freeze",
           "history": history, "pilot_n": n, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    target = freeze_p if a.freeze else D.gates_dir() / "prompt_pilot_unfrozen.json"
    atomic_write_json(target, out)
    log(f"{'FROZEN' if a.freeze else 'pilot (not frozen)'}: main model {main_m}, prompt hash {ph} -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
