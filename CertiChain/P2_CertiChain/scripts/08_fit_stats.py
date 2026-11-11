#!/usr/bin/env python
"""08 — priors, cross-fitted M_stat (+ M_stat^prefix), OOF scores, S_0 and the cascade band (DESIGN §3.4.2,
§3.4.4). Owner: P2-F.

Run from the project root under memguard (P2 budget 90 GB; LightGBM <= 16 threads):

  .venv/bin/python memguard/memguard.py run --name p2_stats_S_TON --limit-gb 90 --workers 8 -- \
      .venv/bin/python -I -u P2_CertiChain/scripts/08_fit_stats.py --scorer S_TON --prefix --score

  --scorer S_TON | S_TON_H | S_GOT | S_CF     (training pools: configs/scorers.yaml)
  --prefix   also fit M_stat^prefix (SHOULD, E6)
  --score    score every non-training incident of the scorer's target datasets with the final model(s)
             (label-free) -> results/scores/<ds>/<scorer>.stat[_prefix]/<inc>.parquet; training incidents
             get their OOF score (column oof = true)
  --force    refit even if results/models/<scorer>/meta.json matches the config hash

Outputs results/models/<scorer>/: priors.json, lgbm_stat.txt, lgbm_stat_fold<k>.txt, oof.parquet,
null_pool.npy (sorted OOF scores of training strict nulls), band.json (+ bands.json kappa grid),
meta.json; *_prefix variants. Every write is atomic; a rerun with the same config hash is a no-op.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import yaml  # noqa: E402

from certichain.features import stats as F  # noqa: E402
from certichain.io.hashing import config_hash  # noqa: E402
from certichain.io.paths import P2_ROOT, atomic_write_bytes, atomic_write_json, atomic_write_parquet  # noqa: E402
from certichain.scoring import cascade as C  # noqa: E402
from certichain.scoring import driver as D  # noqa: E402
from certichain.scoring import fusion as FU  # noqa: E402
from certichain.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory, usage_gb  # noqa: E402

CFG_PATH = P2_ROOT / "configs" / "scorers.yaml"


def log(msg: str) -> None:
    print(f"[08 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cfg() -> dict:
    return yaml.safe_load(CFG_PATH.read_text())


def lgb_params(cfg: dict, seed: int | None = None) -> dict:
    p = dict(FU.LGB_PARAMS)
    p.update(cfg.get("lightgbm", {}))
    p["num_threads"] = min(16, int(p.get("num_threads", 16)))
    if seed is not None:
        p["seed"] = int(seed)
    return p


def training_set(scorer: str, cfg: dict) -> list[tuple[str, str]]:
    out = []
    for ds, role in cfg["scorers"][scorer]["train"]:
        out += [(ds, i) for i in D.list_incidents(ds, role)]
    return out


def load_training(pairs, with_labels: bool = True) -> list[FU.IncidentData]:
    incs = []
    for k, (ds, inc) in enumerate(pairs):
        feats = D.load_features(ds, inc)
        y = D.load_labels(ds, inc)["y"].to_numpy().astype(np.int8) if with_labels else None
        if y is not None and y.size != feats.height:
            raise RuntimeError(f"{ds}:{inc}: {y.size} labels vs {feats.height} candidates")
        incs.append(FU.IncidentData(inc_id=f"{ds}:{inc}", feats=feats, y=y))
        if k % 25 == 0:
            check_memory()
    return incs


def save_booster(b, path: Path) -> None:
    atomic_write_bytes(path, b.model_to_string().encode())


def fit_one(name: str, scorer: str, incs, names, cfg: dict, out: Path, tag: str) -> dict:
    tr = cfg["training"]
    res = FU.cross_fit(incs, names, params=lgb_params(cfg), n_folds=int(tr["n_folds"]),
                       fold_seed=int(tr["fold_seed"]), max_rounds=int(tr["max_rounds"]),
                       es_rounds=int(tr["early_stopping"]), W=float(tr["W"]), log=log, check=check_memory)
    suffix = "" if tag == "stat" else f"_{tag.split('_', 1)[1]}" if "_" in tag else ""
    save_booster(res.booster, out / f"lgbm_{tag}.txt")
    for k, b in enumerate(res.fold_boosters):
        save_booster(b, out / f"lgbm_{tag}_fold{k}.txt")
    oof = res.oof_frame()
    # attach cand ids (feats are sorted by cand_id, row_in_inc = cand_id position)
    cand = np.concatenate([i.feats["cand_id"].to_numpy() for i in incs]) if incs else np.zeros(0, np.int64)
    oof = oof.with_columns(pl.Series("cand_id", cand.astype(np.int64)))
    atomic_write_parquet(oof, out / f"oof{suffix}.parquet")
    s0 = res.null_pool()
    D.atomic_save_npy(out / f"null_pool{suffix}.npy", s0)
    scores = FU.split_by_incident(res.oof, res.sizes)
    ys = [i.y for i in incs]
    auc_strict = FU.within_auc(scores, ys, "strict")
    auc_zero = FU.within_auc(scores, ys, "zero")
    from certichain.eval.gates import pooled_auc
    meta = {"model": name, "scorer": scorer, "tag": tag, "features": list(names), "n_features": len(names),
            "categorical": [n for n in names if n in F.CATEGORICAL], "n_train_incidents": len(incs),
            "n_rows": int(res.y.size), "n_pos": int((res.y == 1).sum()), "n_amb": int((res.y == -1).sum()),
            "n_null0": int((res.y == 0).sum()), "best_iters": res.best_iters, "final_rounds": res.final_rounds,
            "folds": res.folds, "params": res.params, "s0_size": int(s0.size),
            "p_min": 1.0 / (s0.size + 1), "oof_within_auc_strict": auc_strict, "oof_within_auc_zero": auc_zero,
            "oof_pooled_auc_zero": pooled_auc(scores, ys), "timings": res.timings,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    log(f"{tag}: OOF within-incident AUC strict {auc_strict['mean']:.4f} / vs y=0 {auc_zero['mean']:.4f}; "
        f"|S0| = {s0.size}; final rounds {res.final_rounds}")
    return {"res": res, "meta": meta, "oof": oof}


def write_band(out: Path, oof: pl.DataFrame, cfg: dict) -> C.Band:
    """Cascade band from the OOF s_stat of the training incidents (cheap; recomputed every run)."""
    cc = cfg["cascade"]
    s, y = oof["s_oof"].to_numpy(), oof["y"].to_numpy()
    band = C.fit_band(s, y, cc["kappa_target"], cc["kappa_max"], cc["rule"], cc["pos_below"], cc["fdp_above"])
    band.save(out / "band.json")
    grid = {k: b.to_json() for k, b in C.band_grid(s, y, cc["kappa_grid"], cc["rule"]).items()}
    alt = C.fit_band(s, y, cc["kappa_target"], cc["kappa_max"], "pos" if cc["rule"] == "cand" else "cand",
                     cc["pos_below"], cc["fdp_above"])
    atomic_write_json(out / "bands.json", {"rule": cc["rule"], "grid": grid, "alternative_rule": alt.to_json()})
    log(f"band ({band.rule}) [{band.q_lo:.3f}, {band.q_hi:.3f}) kappa_train {band.kappa_train:.3f} (initial "
        f"[{band.q_lo0:.3f}, {band.q_hi0:.3f}) kappa0 {band.kappa0:.3f}); true links in band "
        f"{band.pos_in_band:.3f}, above {band.pos_above:.3f}; alt rule '{alt.rule}': in band "
        f"{alt.pos_in_band:.3f} at kappa {alt.kappa_train:.3f}")
    return band


def score_targets(scorer: str, cfg: dict, tag: str, names, booster, priors, oof: pl.DataFrame,
                  targets: list[str] | None = None) -> dict:
    """Write scores/<ds>/<scorer>.<tag>/<inc>.parquet for all incidents of the target datasets (and the
    training datasets: OOF scores for training incidents). ``targets`` overrides the config's list (e.g. the
    E10 transfer banks S_TON -> CF-GOTHAM(-T)); training datasets are then not re-scored."""
    sid = f"{scorer}.{tag}"
    train_pairs = set(training_set(scorer, cfg))
    oof_map = {k: g for (k,), g in oof.group_by(["inc_id"])}
    dsets = list(targets) if targets else \
        list(dict.fromkeys([d for d, _ in cfg["scorers"][scorer]["train"]] + cfg["scorers"][scorer]["targets"]))
    counts = {}
    for ds in dsets:
        incs = D.list_incidents(ds)
        n = 0
        for k, inc in enumerate(incs):
            if (ds, inc) in train_pairs:
                g = oof_map.get(f"{ds}:{inc}")
                if g is None:                       # incident without candidates
                    g = pl.DataFrame(schema={"cand_id": pl.Int64, "s_oof": pl.Float64})
                g = g.sort("cand_id")
                df = pl.DataFrame({"cand_id": g["cand_id"], "s": g["s_oof"],
                                   "oof": pl.Series([True] * g.height, dtype=pl.Boolean)})
            else:
                feats = D.load_features(ds, inc)
                s = FU.score_incident(booster, priors, FU.IncidentData(inc, feats), names)
                df = pl.DataFrame({"cand_id": feats["cand_id"], "s": s,
                                   "oof": pl.Series([False] * feats.height, dtype=pl.Boolean)})
            D.write_scores(ds, sid, inc, df)
            n += 1
            if k % 50 == 0:
                check_memory()
        counts[ds] = n
        log(f"scored {ds}: {n} incidents -> scores/{ds}/{sid}/")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", required=True, choices=["S_TON", "S_TON_H", "S_GOT", "S_CF", "S_GOT2", "S_CF2", "S_GOTXA"])
    ap.add_argument("--prefix", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--targets", default=None, help="comma list overriding the scored datasets (with --score)")
    a = ap.parse_args()
    cfg = load_cfg()
    scorer = a.scorer
    out = D.models_dir(scorer)
    out.mkdir(parents=True, exist_ok=True)
    pairs = training_set(scorer, cfg)
    if not pairs:
        log(f"no training incidents for {scorer} (scripts/07 outputs missing)")
        return 2
    ufp = {}
    for d in sorted({d for d, _ in pairs}):
        sp = D.interim_dir(d) / "summary_07.json"
        ufp[d] = json.loads(sp.read_text()).get("code_fp") if sp.exists() else None
    chash = config_hash({"cfg": {k: cfg[k] for k in ("lightgbm", "training")},
                         "train": [f"{d}:{i}" for d, i in pairs], "features": D.feature_code_hash(),
                         "code": D.scorer_code_hash(), "units_code_fp": ufp})
    meta_p = out / "meta.json"
    try:
        log(f"{scorer}: {len(pairs)} training incidents; config {chash}")
        t0 = time.perf_counter()
        incs = load_training(pairs)
        log(f"loaded features+labels in {time.perf_counter() - t0:.1f}s; mem {usage_gb():.1f} GB")
        done = meta_p.exists() and json.loads(meta_p.read_text()).get("config_hash") == chash and not a.force
        if done:
            log("M_stat up to date (config hash matches); loading")
            booster = FU.load_booster(out / "lgbm_stat.txt")
            from certichain.features.priors import PairPriors
            priors = PairPriors.load(out / "priors.json")
            oof = pl.read_parquet(out / "oof.parquet")
        else:
            r = fit_one("M_stat", scorer, incs, F.FEATURES, cfg, out, "stat")
            booster, priors, oof = r["res"].booster, r["res"].priors, r["oof"]
            priors.save(out / "priors.json")
            atomic_write_json(meta_p, r["meta"] | {"config_hash": chash, "units_code_fp": ufp})
        write_band(out, oof, cfg)
        if a.prefix:
            mp = out / "meta_prefix.json"
            if not (mp.exists() and json.loads(mp.read_text()).get("config_hash") == chash and not a.force):
                r = fit_one("M_stat_prefix", scorer, incs, F.PREFIX_FEATURES, cfg, out, "stat_prefix")
                atomic_write_json(mp, r["meta"] | {"config_hash": chash})
            else:
                log("M_stat^prefix up to date")
        if a.score:
            from certichain.features.priors import PairPriors
            priors = PairPriors.load(out / "priors.json")
            oof = pl.read_parquet(out / "oof.parquet")
            tg = a.targets.split(",") if a.targets else None
            score_targets(scorer, cfg, "stat", F.FEATURES, FU.load_booster(out / "lgbm_stat.txt"), priors, oof, tg)
            if a.prefix:
                oofp = pl.read_parquet(out / "oof_prefix.parquet")
                score_targets(scorer, cfg, "stat_prefix", F.PREFIX_FEATURES,
                              FU.load_booster(out / "lgbm_stat_prefix.txt"), priors, oofp, tg)
    except MemoryBudgetExceeded as e:
        log(f"[mem] {e}; exiting {EXIT_MEMORY} (resumable)")
        return EXIT_MEMORY
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
