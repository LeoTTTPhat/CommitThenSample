#!/usr/bin/env python
"""16 — uncalibrated baselines B1 (King), B2 (BotHunter), B4 (lag-LLR) and B5 (LLM-only) -> per-incident claim
sets for the experiment runner (DESIGN §6.3). Owner: P2-F.

Run from the project root under memguard:

  .venv/bin/python memguard/memguard.py run --name p2_B_cf_ton --limit-gb 90 --workers 4 -- \
      .venv/bin/python -I -u P2_CertiChain/scripts/16_baselines.py --dataset cf_ton --baselines B1,B2,B4,B5

  --roles caltest,train,...   incidents to process (default: the evaluation pool: CF cal/test, IoT-23 / Gotham all)
  --metrics                   also per-incident FDP / STR / CER_link (GT side: cal/test needs the PREREG file)

Outputs results/baselines/<ds>/<B>/claims.parquet (incident, cand_id[, score]) + summary.json; B4 also writes
the lag-LLR scorer for CRC: results/scores/<ds>/<S>.lag_llr/<inc>.parquet (training incidents: out-of-fold) and
results/models/<S>/null_pool_lag_llr.npy (OOF lag-LLR of the training strict nulls), S = the dataset's scorer.
B5 runs on the kappa = 1 arm only (scoring/driver.b5_incidents) and needs its LLM shards (scripts/09 b5_arm).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time

import numpy as np
import polars as pl
import yaml

from certichain.baselines.bothunter import bothunter_claims
from certichain.baselines.king import king_claims
from certichain.baselines.lag_llr import lag_llr_claims, lag_llr_scores, oof_lag_llr
from certichain.baselines.llm_only import coverage, llm_only_claims, llm_scores
from certichain.features.priors import PairPriors
from certichain.io.paths import P2_ROOT, atomic_write_json, atomic_write_parquet, results_root
from certichain.scoring import driver as D
from certichain.scoring import fusion as FU
from certichain.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory


def log(msg: str) -> None:
    print(f"[16 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def out_dir(ds: str, b: str):
    return results_root() / "baselines" / ds / b


def per_incident_metrics(ds: str, inc: str, mask: np.ndarray) -> dict:
    lab = D.load_labels(ds, inc)
    y = lab["y"].to_numpy()
    tr = D.load_transitions(ds, inc)
    st = D.load_stats(ds, inc)["transition_stats"]
    sel = set(lab["cand_id"].to_numpy()[mask].tolist())
    obs = tr.filter(pl.col("kind") == "obs") if tr.height else tr
    hit = sum(1 for c in (obs["cand_id"].to_list() if obs.height else []) if c is not None and int(c) in sel)
    R = int(mask.sum())
    V = int((y[mask] != 1).sum())
    A = int((y[mask] == -1).sum())
    TP = int((y[mask] == 1).sum())
    den = (st.get("n_E_link") or 0) + (st.get("n_T_hid") or 0)
    return {"incident": inc, "R": R, "V": V, "A": A, "TP": TP, "fdp_strict": V / max(R, 1),
            "fdp_lenient": (V - A) / max(R - A, 1), "STR": hit / st["n_T"] if st.get("n_T") else None,
            "CER_link": TP / den if den else None, "n_T": st.get("n_T")}


def write_claims(ds: str, b: str, rows: list[dict], meta: dict, metrics: list[dict] | None) -> None:
    d = out_dir(ds, b)
    df = pl.DataFrame(rows, schema={"incident": pl.Utf8, "cand_id": pl.Int64, "score": pl.Float64}) if rows else \
        pl.DataFrame(schema={"incident": pl.Utf8, "cand_id": pl.Int64, "score": pl.Float64})
    atomic_write_parquet(df, d / "claims.parquet")
    summ = dict(meta, n_claims=int(df.height), created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if metrics:
        m = pl.DataFrame(metrics, infer_schema_length=None)
        atomic_write_parquet(m, d / "per_incident.parquet")
        summ["mean_fdp_strict"] = float(m["fdp_strict"].mean())
        summ["pooled_fdr_strict"] = float(m["V"].sum() / max(m["R"].sum(), 1))
        summ["mean_STR"] = float(m["STR"].drop_nulls().mean()) if m["STR"].drop_nulls().len() else None
    atomic_write_json(d / "summary.json", summ)
    log(f"{ds}/{b}: {df.height} claims over {meta.get('n_incidents')} incidents -> {d}")


def b4_scorer(ds: str, scorer: str, incs: list[str]) -> None:
    """lag-LLR scores for CRC (+ the OOF training null pool)."""
    spec = importlib.util.spec_from_file_location("s08", P2_ROOT / "scripts" / "08_fit_stats.py")
    m08 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m08)
    cfg = m08.load_cfg()
    md = D.models_dir(scorer)
    priors = PairPriors.load(md / "priors.json")
    pairs = m08.training_set(scorer, cfg)
    tr = m08.load_training(pairs)
    folds = FU.make_folds([t.inc_id for t in tr], int(cfg["training"]["n_folds"]), int(cfg["training"]["fold_seed"]))
    oof = oof_lag_llr([t.feats for t in tr], [t.y for t in tr], [t.inc_id for t in tr], folds,
                      float(cfg["training"]["W"]))
    ys = np.concatenate([t.y for t in tr]) if tr else np.zeros(0)
    s0 = np.sort(np.concatenate(oof)[ys != 1]) if tr else np.zeros(0)
    D.atomic_save_npy(md / "null_pool_lag_llr.npy", s0)
    oof_map = {t.inc_id: s for t, s in zip(tr, oof)}
    sid = f"{scorer}.lag_llr"
    n = 0
    for ds2 in sorted({d for d, _ in pairs} | {ds}):
        for inc in D.list_incidents(ds2):
            if ds2 != ds and (ds2, inc) not in set(pairs):
                continue
            feats = D.load_features(ds2, inc)
            key = f"{ds2}:{inc}"
            s = oof_map[key] if key in oof_map else lag_llr_scores(feats, priors)
            D.write_scores(ds2, sid, inc, pl.DataFrame({"cand_id": feats["cand_id"], "s": s,
                                                        "oof": pl.Series([key in oof_map] * feats.height,
                                                                         dtype=pl.Boolean)}))
            n += 1
    log(f"B4 scorer {sid}: {n} incidents scored; |S0_lag| = {s0.size}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--baselines", default="B1,B2,B4,B5")
    ap.add_argument("--roles", default=None)
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--model", default="main")
    a = ap.parse_args()
    ds = a.dataset
    bl = [b.strip().upper() for b in a.baselines.split(",")]
    roles = a.roles.split(",") if a.roles else None
    default_roles = ("caltest", "test")             # the evaluation pools (CF cal/test; IoT-23 / Gotham-real)
    incs = [i for i in D.list_incidents(ds) if (roles is None and D.role_of(ds, i) in default_roles)
            or (roles is not None and D.role_of(ds, i) in roles)]
    scorer = D.DATASET_SCORER.get(ds)
    W = float(yaml.safe_load((P2_ROOT / "configs" / "units.yaml").read_text())["candidates"]["W"])
    try:
        for b in bl:
            rows, metrics = [], []
            if b in ("B1", "B2"):
                for k, inc in enumerate(incs):
                    units, cands = D.load_units(ds, inc), D.load_cands(ds, inc)
                    m = king_claims(units, cands) if b == "B1" else bothunter_claims(units, cands, W)
                    rows += [{"incident": inc, "cand_id": int(c), "score": None} for c in cands["cand_id"].to_numpy()[m]]
                    if a.metrics:
                        metrics.append(per_incident_metrics(ds, inc, m))
                    if k % 50 == 0:
                        check_memory()
                write_claims(ds, b, rows, {"baseline": b, "dataset": ds, "n_incidents": len(incs),
                                           "rule": "king_claims(expand_benign=False)" if b == "B1"
                                           else f"bothunter_claims(W={W})"}, metrics)
            elif b == "B4":
                if scorer is None:
                    log(f"B4: no scorer for {ds}; skipped")
                    continue
                b4_scorer(ds, scorer, incs)
                for inc in incs:
                    s = D.read_scores(ds, f"{scorer}.lag_llr", inc)
                    m = lag_llr_claims(s["s"].to_numpy())
                    rows += [{"incident": inc, "cand_id": int(c), "score": float(v)}
                             for c, v in zip(s["cand_id"].to_numpy()[m], s["s"].to_numpy()[m])]
                    if a.metrics:
                        metrics.append(per_incident_metrics(ds, inc, m))
                write_claims(ds, b, rows, {"baseline": b, "dataset": ds, "n_incidents": len(incs),
                                           "rule": "lag-LLR > 0", "scorer": f"{scorer}.lag_llr"}, metrics)
            elif b == "B5":
                fz = json.loads((D.gates_dir() / "prompt_freeze.json").read_text())
                cfg = yaml.safe_load((P2_ROOT / "configs" / "llm.yaml").read_text())
                alias = fz["main_model"] if a.model == "main" else a.model
                tag = cfg["models"].get(alias, alias).split("/")[-1].lower()
                lk = D.llm_lookup(ds, tag, fz["prompt_hash"])
                arm = D.b5_incidents(ds)
                covs = []
                for inc in arm:
                    c = D.load_cands(ds, inc)["cand_id"].to_numpy()
                    s = llm_scores(c, lk.get(inc, {}))
                    covs.append(coverage(s))
                    m = llm_only_claims(s)
                    rows += [{"incident": inc, "cand_id": int(x), "score": float(v)} for x, v in zip(c[m], s[m])]
                    if a.metrics:
                        metrics.append(per_incident_metrics(ds, inc, m))
                write_claims(ds, b, rows, {"baseline": b, "dataset": ds, "n_incidents": len(arm), "incidents": arm,
                                           "rule": "s_LLM > 0 (kappa = 1 arm)", "model_tag": tag,
                                           "prompt_hash": fz["prompt_hash"], "min_coverage": min(covs) if covs else None},
                             metrics)
            else:
                raise SystemExit(f"unknown baseline {b}")
    except MemoryBudgetExceeded as e:
        log(f"[mem] {e}; exiting {EXIT_MEMORY}")
        return EXIT_MEMORY
    return 0


if __name__ == "__main__":
    sys.exit(main())
