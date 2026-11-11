#!/usr/bin/env python
"""10 — M_fuse (stats + s_LLM + llm_called; s_LLM NaN outside the cascade band), seeds 0-4 (E3), cross-fitted
like M_stat with its own S_0, plus the two pre-registered placebos (DESIGN §3.4.2, §6.1 E3, §6.4). Owner: P2-F.

Run from the project root under memguard (LightGBM <= 16 threads):

  .venv/bin/python memguard/memguard.py run --name p2_fuse_S_TON --limit-gb 90 --workers 8 -- \
      .venv/bin/python -I -u P2_CertiChain/scripts/10_fit_fusion.py --scorer S_TON --seeds 0-4 --placebos --score

  --model main|<alias>   LLM whose shards are used (default: the frozen main model, prompt_freeze.json)
  --placebos             fuse_perm (s_LLM permuted within each incident's band, keyed RNG) and fuse_band
                         (band indicator only, s_LLM all NaN), seed 0, trained and applied the same way
  --score                write results/scores/<ds>/<scorer>.fuse_s<seed>/<inc>.parquet (cand_id, s, s_stat,
                         s_llm, llm_called, oof) for the scorer's datasets (training incidents: OOF fused score)
  --allow-missing        in-band candidates without an LLM score are kept as NaN (default: refuse)

Inputs: models/<scorer>/{oof.parquet, band.json} (scripts/08), LLM shards (scripts/09). The band for training
candidates uses their OOF s_stat, for every other incident the final M_stat score (results/scores/...stat).
Outputs models/<scorer>/: lgbm_fuse_<v>.txt (+ folds), oof_fuse_<v>.parquet, null_pool_fuse_<v>.npy,
meta_fuse_<v>.json with v in {s0..s4, perm_s0, band_s0}.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import blake3  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import yaml  # noqa: E402

from certichain.features import stats as F  # noqa: E402
from certichain.io.hashing import config_hash  # noqa: E402
from certichain.io.paths import P2_ROOT, atomic_write_json, atomic_write_parquet  # noqa: E402
from certichain.scoring import driver as D  # noqa: E402
from certichain.scoring import fusion as FU  # noqa: E402
from certichain.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory  # noqa: E402

NAMES = (*F.FEATURES, *FU.FUSE_EXTRA)


def log(msg: str) -> None:
    print(f"[10 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def s08():
    spec = importlib.util.spec_from_file_location("s08", P2_ROOT / "scripts" / "08_fit_stats.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parse_seeds(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def permute_in_band(inc: str, s_llm: np.ndarray, band: np.ndarray) -> np.ndarray:
    """Placebo: s_LLM values permuted among the incident's in-band candidates (keyed by the incident id)."""
    out = s_llm.copy()
    idx = np.flatnonzero(band)
    if idx.size > 1:
        seed = int.from_bytes(blake3.blake3(f"placebo-perm|{inc}".encode()).digest(8), "little")
        out[idx] = s_llm[idx][np.random.default_rng(seed).permutation(idx.size)]
    return out


def llm_arrays(ds: str, inc: str, cand_ids: np.ndarray, s_stat: np.ndarray, band, lookup: dict, variant: str):
    in_band = band.contains(s_stat)
    m = lookup.get(inc, {})
    s = np.array([m.get(int(c), np.nan) for c in cand_ids], dtype=np.float64)
    s = np.where(in_band, s, np.nan)
    missing = int((in_band & np.isnan(s)).sum())
    if variant == "perm":
        s = permute_in_band(inc, s, in_band)
    elif variant == "band":
        s = np.full(s.size, np.nan)
    return s, in_band, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", required=True, choices=["S_TON", "S_TON_H", "S_GOT", "S_CF", "S_GOT2", "S_CF2"])
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--placebos", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--model", default="main")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--kappa", default="0.10", choices=["0.05", "0.10", "0.20"],
                    help="cascade band (kappa sensitivity, E3 / §3.4.4); non-default kappa -> variant suffix _k005/_k020")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    m08 = s08()
    cfg = m08.load_cfg()
    llm_cfg = yaml.safe_load((P2_ROOT / "configs" / "llm.yaml").read_text())
    fz = json.loads((D.gates_dir() / "prompt_freeze.json").read_text())
    alias = fz["main_model"] if a.model == "main" else a.model
    model_id = llm_cfg["models"].get(alias, alias)
    tag, ph = model_id.split("/")[-1].lower(), fz["prompt_hash"]
    scorer = a.scorer
    out = D.models_dir(scorer)
    if a.kappa == "0.10":
        band = D.load_band(scorer)
        ksuf = ""
    else:
        from certichain.scoring.cascade import Band
        band = Band.from_json(json.loads((D.models_dir(scorer) / "bands.json").read_text())["grid"][a.kappa])
        ksuf = "_k" + a.kappa.replace("0.", "0").replace(".", "")
    oof = pl.read_parquet(out / "oof.parquet")
    pairs = m08.training_set(scorer, cfg)
    lookups = {ds: D.llm_lookup(ds, tag, ph) for ds in sorted({d for d, _ in pairs})}
    variants = [(f"s{s}", s, "llm") for s in parse_seeds(a.seeds)]
    if a.placebos:
        variants += [("perm_s0", 0, "perm"), ("band_s0", 0, "band")]
    variants = [(v + ksuf, sd, k) for v, sd, k in variants]
    try:
        base = m08.load_training(pairs)
        oof_map = {k: g.sort("cand_id") for (k,), g in oof.group_by(["inc_id"])}
        fitted = {}
        for vname, seed, kind in variants:
            mp = out / f"meta_fuse_{vname}.json"
            chash = config_hash({"lgb": m08.lgb_params(cfg, seed), "training": cfg["training"], "variant": kind,
                                 "llm": [tag, ph], "band": D.file_blake3(out / "band.json") if not ksuf
                                 else json.dumps(band.to_json(), sort_keys=True),
                                 "stat_oof": D.file_blake3(out / "oof.parquet"), "code": D.scorer_code_hash()})
            if mp.exists() and json.loads(mp.read_text()).get("config_hash") == chash and not a.force:
                log(f"fuse_{vname}: up to date")
                fitted[vname] = None
                continue
            incs, missing = [], 0
            for (ds, inc), it in zip(pairs, base):
                g = oof_map.get(f"{ds}:{inc}")
                s_stat = g["s_oof"].to_numpy() if g is not None else np.zeros(0)
                s_llm, in_band, miss = llm_arrays(ds, inc, it.feats["cand_id"].to_numpy(), s_stat, band,
                                                  lookups[ds], kind)
                missing += miss
                incs.append(FU.IncidentData(it.inc_id, it.feats, it.y, s_llm=s_llm, in_band=in_band))
            n_band = int(sum(int(i.in_band.sum()) for i in incs))
            if missing and not a.allow_missing and kind != "band":
                raise SystemExit(f"{missing}/{n_band} in-band TRAINING candidates have no LLM score "
                                 f"({tag}/{ph}); run scripts/09 first or pass --allow-missing")
            log(f"fuse_{vname}: {len(incs)} training incidents, {n_band} in-band ({missing} without s_LLM)")
            params = m08.lgb_params(cfg, seed)
            res = FU.cross_fit(incs, NAMES, params=params, n_folds=int(cfg["training"]["n_folds"]),
                               fold_seed=int(cfg["training"]["fold_seed"]), max_rounds=int(cfg["training"]["max_rounds"]),
                               es_rounds=int(cfg["training"]["early_stopping"]), W=float(cfg["training"]["W"]),
                               log=log, check=check_memory)
            m08.save_booster(res.booster, out / f"lgbm_fuse_{vname}.txt")
            for k, b in enumerate(res.fold_boosters):
                m08.save_booster(b, out / f"lgbm_fuse_{vname}_fold{k}.txt")
            fr = res.oof_frame().with_columns(pl.Series("cand_id", np.concatenate(
                [i.feats["cand_id"].to_numpy() for i in incs]).astype(np.int64)))
            atomic_write_parquet(fr, out / f"oof_fuse_{vname}.parquet")
            D.atomic_save_npy(out / f"null_pool_fuse_{vname}.npy", res.null_pool())
            sc = FU.split_by_incident(res.oof, res.sizes)
            meta = {"model": f"M_fuse[{vname}]", "variant": kind, "seed": seed, "features": list(NAMES),
                    "llm": {"model_id": model_id, "model_tag": tag, "prompt_hash": ph},
                    "n_in_band_train": n_band, "n_missing_llm_train": missing, "best_iters": res.best_iters,
                    "final_rounds": res.final_rounds, "s0_size": int(res.null_pool().size),
                    "oof_within_auc_strict": FU.within_auc(sc, [i.y for i in incs], "strict"),
                    "oof_within_auc_zero": FU.within_auc(sc, [i.y for i in incs], "zero"),
                    "config_hash": chash, "timings": res.timings, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            atomic_write_json(mp, meta)
            fitted[vname] = res
            log(f"fuse_{vname}: OOF within-AUC strict {meta['oof_within_auc_strict']['mean']:.4f}")
        if a.score:
            from certichain.features.priors import PairPriors
            priors = PairPriors.load(out / "priors.json")
            train_set = set(pairs)
            dsets = list(dict.fromkeys([d for d, _ in pairs] + cfg["scorers"][scorer]["targets"]))
            for vname, seed, kind in variants:
                booster = FU.load_booster(out / f"lgbm_fuse_{vname}.txt")
                foof = pl.read_parquet(out / f"oof_fuse_{vname}.parquet")
                fmap = {k: g.sort("cand_id") for (k,), g in foof.group_by(["inc_id"])}
                sid = f"{scorer}.fuse_{vname}"
                for ds in dsets:
                    lk = lookups.get(ds) or D.llm_lookup(ds, tag, ph)
                    n_inc = n_miss = 0
                    for k, inc in enumerate(D.list_incidents(ds)):
                        if D.role_of(ds, inc) == "dev":
                            continue                          # dev is never reported; not LLM-scored
                        st = D.read_scores(ds, f"{scorer}.stat", inc)
                        s_stat = st["s"].to_numpy()
                        cid = st["cand_id"].to_numpy()
                        s_llm, in_band, miss = llm_arrays(ds, inc, cid, s_stat, band, lk, kind)
                        if (ds, inc) in train_set:
                            g = fmap.get(f"{ds}:{inc}")
                            s = g["s_oof"].to_numpy() if g is not None else np.zeros(0)
                        else:
                            if miss and not a.allow_missing and kind != "band":
                                n_miss += miss
                                continue
                            feats = D.load_features(ds, inc)
                            s = FU.score_incident(booster, priors, FU.IncidentData(inc, feats, s_llm=s_llm,
                                                                                    in_band=in_band), NAMES)
                        df = pl.DataFrame({"cand_id": cid, "s": s, "s_stat": s_stat, "s_llm": s_llm,
                                           "llm_called": in_band, "oof": pl.Series([(ds, inc) in train_set] * cid.size,
                                                                                  dtype=pl.Boolean)})
                        D.write_scores(ds, sid, inc, df)
                        n_inc += 1
                        if k % 50 == 0:
                            check_memory()
                    log(f"scored {ds} with {sid}: {n_inc} incidents"
                        + (f"; {n_miss} in-band candidates without s_LLM -> those incidents skipped" if n_miss else ""))
    except MemoryBudgetExceeded as e:
        log(f"[mem] {e}; exiting {EXIT_MEMORY}")
        return EXIT_MEMORY
    return 0


if __name__ == "__main__":
    sys.exit(main())
