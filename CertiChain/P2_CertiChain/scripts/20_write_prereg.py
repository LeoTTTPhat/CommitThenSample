#!/usr/bin/env python
"""20 — the pre-registration file results/PREREG_2026-10-21.json (DESIGN §6.4), written with P2-C's
eval/prereg.write_prereg BEFORE any calibration/test evaluation (deadline Wed 21 12:00). Owner: P2-F.

  .venv/bin/python -I P2_CertiChain/scripts/20_write_prereg.py --draft    # results/gates/PREREG_draft.json
  .venv/bin/python -I P2_CertiChain/scripts/20_write_prereg.py --final    # the immutable PREREG + BLAKE3

--draft fills every field that is known and lists the missing ones (scorer / prompt hashes, gate numbers);
--final refuses unless G1 (on the current units), the prompt freeze, the LLM budget gate and the S_TON /
S_TON_H stats models exist, and no calibration/test loss bank or experiment output exists yet. The final
file is immutable (eval/prereg: rewriting different content raises); its BLAKE3 is printed and written to
results/gates/prereg_blake3.txt (log it in plans.md).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from certichain.certify.rules import LAMBDA_GRID
from certichain.eval.prereg import REQUIRED_KEYS, canonical_bytes, write_prereg
from certichain.features import stats as F
from certichain.io.hashing import config_hash, sha256_file, tree_blake3
from certichain.io.paths import DATA_ROOT, P2_ROOT, atomic_write_json, atomic_write_text, results_root
from certichain.scoring import driver as D
from certichain.scoring import fusion as FU

DATASETS = ("cf_ton", "cf_ton_h", "cf_gotham", "cf_gotham_t", "iot23", "gotham_real")
STAT_SCORERS = ("S_TON", "S_TON_H", "S_GOT", "S_CF")
FALLBACK = (
    "Pre-registered fallback framing (DESIGN §6.4), used verbatim if primary endpoint 2 fails: the guarantee is "
    "presented as scorer-agnostic; the LLM's effect is reported per stratum with CIs (unseen-type-pair "
    "transitions, decoy-rich, content-rich); E4 (validity under hallucinating / inverted / memorising / injected "
    "LLM scores) is the LLM-related headline; the abstract carries no LLM-gain claim. A null result is reported "
    "as such.")
LOW_POWER = (
    "Pre-registered low-power framing (DESIGN §8.3, used only if G1 had failed after the levers): alpha = 0.2 "
    "primary in the text; transition / chain claims; the minimal-K practitioner curve (E7).")


def log(msg: str) -> None:
    print(f"[20 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def config_hashes() -> dict:
    out = {}
    for p in sorted((P2_ROOT / "configs").rglob("*.yaml")):
        out[str(p.relative_to(P2_ROOT))] = {"blake3": D.file_blake3(p),
                                            "config_hash": config_hash(yaml.safe_load(p.read_text()))}
    return out


def data_manifests() -> dict:
    out = {"manifest_files": {}, "ingest_runs": {}}
    for name in ("ton_iot", "gotham2025", "attack", "models"):
        p = DATA_ROOT / name / "MANIFEST.md"
        if p.exists():
            out["manifest_files"][f"data/{name}/MANIFEST.md"] = sha256_file(p)
    latest: dict[str, str] = {}
    for f in sorted(glob.glob(str(results_root() / "manifests" / "run_*.json"))):
        job = json.loads(Path(f).read_text()).get("job")
        if job:
            latest[job] = f
    for job, f in sorted(latest.items()):
        d = json.loads(Path(f).read_text())
        out["ingest_runs"][job] = {"manifest": Path(f).name, "src_tree_blake3": d.get("src_tree_blake3"),
                                   "config_hash": d.get("config_hash"), "data_sha256": d.get("data_sha256") or {}}
    return out


def units_fp() -> dict:
    out = {}
    for ds in DATASETS:
        s = jload(D.interim_dir(ds) / "summary_07.json")
        out[ds] = None if s is None else {"code_fp": s.get("code_fp"), "n_incidents": s.get("n_incidents")}
    return out


def scorer_hashes() -> dict:
    out = {}
    for sc in STAT_SCORERS:
        md = D.models_dir(sc)
        if not (md / "lgbm_stat.txt").exists():
            out[sc] = None
            continue
        meta = jload(md / "meta.json") or {}
        out[sc] = {"files_blake3": {f: D.file_blake3(md / f) for f in
                                    ("lgbm_stat.txt", "priors.json", "null_pool.npy", "band.json", "bands.json",
                                     "oof.parquet", "lgbm_stat_prefix.txt", "null_pool_prefix.npy")},
                   "n_train_incidents": meta.get("n_train_incidents"), "n_rows": meta.get("n_rows"),
                   "oof_within_auc_strict": (meta.get("oof_within_auc_strict") or {}).get("mean"),
                   "final_rounds": meta.get("final_rounds"), "band": jload(md / "band.json")}
    out["recipe"] = {"scorer_code_hash": D.scorer_code_hash(), "feature_code_hash": D.feature_code_hash(),
                     "features": list(F.FEATURES), "prefix_features": list(F.PREFIX_FEATURES),
                     "fusion_features": [*F.FEATURES, *FU.FUSE_EXTRA], "lightgbm": FU.LGB_PARAMS,
                     "max_rounds": FU.MAX_ROUNDS, "early_stopping": FU.EARLY_STOP, "n_folds": FU.N_FOLDS,
                     "fold_seed": 0, "stat_seed": 0, "fuse_seeds": [0, 1, 2, 3, 4], "primary_fuse_seed": 0,
                     "null_pool": "sorted OOF scores of the training strict nulls (y != 1)",
                     "placebos": {"fuse_perm_s0": "s_LLM permuted within each incident's band (keyed RNG), trained "
                                  "and applied the same way", "fuse_band_s0": "band indicator only (s_LLM all NaN)"}}
    return out


def prompt_block() -> dict | None:
    fz = jload(D.gates_dir() / "prompt_freeze.json")
    if not fz:
        return None
    return {k: fz.get(k) for k in ("prompt_hash", "prefix_file", "prefix_blake3", "card_input_source_hash",
                                   "main_model", "main_model_id", "main_model_revision", "fewshot_rev", "pilot_n")} | {
        "fewshot_sources": [{k: c.get(k) for k in ("incident", "cand_id", "role", "answer")}
                            for c in fz.get("fewshot_cards", [])],
        "pilot": [{"rev": h["rev"], "auroc": {m: r.get("auroc_strict") for m, r in h["results"].items()},
                   "chosen": h["chosen"]} for h in fz.get("history", [])],
        "freeze_file_blake3": D.file_blake3(D.gates_dir() / "prompt_freeze.json")}


def gates() -> dict:
    g1 = jload(D.gates_dir() / "G1.json")
    bud = jload(D.gates_dir() / "llm_budget.json")
    out = {"G1": None, "llm_budget": None}
    if g1:
        b = g1["default"]
        out["G1"] = {"file_blake3": D.file_blake3(D.gates_dir() / "G1.json"), "pass": g1.get("pass"),
                     "action": g1.get("action"), "best_config": g1.get("best_config"),
                     "units_code_fp": g1.get("units_code_fp"),
                     "dev": {"within_auc": b["within_auc_strict"]["mean"], "r_CT": b["r_CT"]["pooled"],
                             "N_median": b["N"].get("p50"),
                             "STR_alpha0.1": b["crc_bh"]["alpha=0.1"]["str_mean"],
                             "STR_alpha0.2": b["crc_bh"]["alpha=0.2"]["str_mean"]},
                     "levers": [{"tag": r["tag"], "STR_alpha0.2": r["crc_bh"]["alpha=0.2"]["str_mean"],
                                 "r_CT": r["r_CT"]["pooled"]} for r in g1.get("levers", [])]}
    if bud:
        out["llm_budget"] = {"file_blake3": D.file_blake3(D.gates_dir() / "llm_budget.json"),
                             "must_calls": bud.get("must_calls_kappa0.10"), "hours_at": bud.get("hours_at"),
                             "decision": bud.get("decision")}
    return out


def build_cfg() -> tuple[dict, list[str]]:
    fine = [round(x, 2) for x in np.arange(0.01, 0.5001, 0.01)]
    pools = {ds: {"caltest": len(D.list_incidents(ds, "caltest")), "train": len(D.list_incidents(ds, "train"))}
             if ds.startswith("cf_") else {"all": len(D.list_incidents(ds))} for ds in DATASETS}
    cells = [f"{ds}|alpha={a}|crc_bh|{sc}" for ds in ("cf_ton", "cf_ton_h", "cf_gotham", "iot23_loso")
             for a in (0.1, 0.2) for sc in ("stats", "fusion")]
    cfg = {
        "title": "CertiChain (P2) pre-registration: frozen before any calibration/test evaluation",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "writer": "P2-F scripts/20_write_prereg.py",
        "config_hashes": config_hashes(),
        "code": {"src_tree_blake3": tree_blake3([P2_ROOT / "src" / "certichain"]),
                 "scripts_tree_blake3": tree_blake3([P2_ROOT / "scripts"]), "units_code_fp": units_fp()},
        "data_manifest_sha256": data_manifests(),
        "scorer_hashes": scorer_hashes(),
        "prompt_hashes": prompt_block(),
        "alpha_grid": {"co_primary": [0.1, 0.2], "secondary": [0.05], "fine_grid": fine,
                       "fine_grid_use": "precision matching only (DESIGN §6.4)"},
        "lambda_grid": {"definition": "{0} U 60 geometric levels in [1e-4, 0.8] (certify.rules.LAMBDA_GRID)",
                        "values": [float(x) for x in LAMBDA_GRID], "B": 1.0},
        "K_per_dataset": {"cf_ton": 150, "cf_ton_h": 100, "cf_gotham": 100,
                          "iot23": "P-LOSO: calibration = all windows of the other 22 scenarios (K = 22 scenarios)",
                          "E7_K_grid": [5, 10, 20, 50, 100, 200]},
        "test_per_split": {"cf_ton": 150, "cf_ton_h": 100, "cf_gotham": 100},
        "pools": pools,
        "split_seeds": {"seeds": list(range(200)), "rule": "eval.splits.random_splits: perm = "
                        "default_rng((seed, 0)).permutation(pool); first K calibrate, the rest test"},
        "seeds": {"benchmark": D.BENCHMARK_SEED, "lightgbm_stat": 0, "lightgbm_fuse": [0, 1, 2, 3, 4],
                  "bootstrap": 0, "b5_arm_incidents": D.b5_incidents("cf_ton") if D.list_incidents("cf_ton") else None},
        "cascade": {"kappa_target": 0.10, "kappa_max": 0.15, "kappa_grid": [0.05, 0.1, 0.2], "kappa_one_arm": "B5 "
                    "(15 cf_ton cal/test-pool incidents, scoring/driver.b5_incidents)", "band_rule": "pos (deviation 1)"},
        "rules": {"edge": "CRC-BH on reference p-values (Alg. 4), strict FDP (AMB = false) running max",
                  "variants": ["crc_fdx020", "crc_fdx030", "ltt (delta 0.1)", "crc_st (score threshold)",
                               "chain_composite", "chain_naive", "chain_rt"], "gammas": [0.2, 0.3], "J": 20},
        "endpoints": {
            "primary_1_validity": {
                "cells": cells,
                "statistic": "realised FDR (mean per-incident strict FDP over test incidents) averaged over 200 "
                             "splits", "test": "one-sided H0: FDR <= alpha, z = (mean - alpha) / SE_boot, "
                             "SE_boot from the incident-pool cluster bootstrap (B = 1000, 50 splits per replicate)",
                "multiplicity": "Holm across all cells at 0.05; cells not run (R10 drop order) are reported as "
                                "missing and excluded from the family", "report": "max standardised excess; a "
                                "rejection is reported as a violation"},
            "primary_2_llm": {
                "estimand": "Delta CER_link = CER_link(CRC-BH, M_fuse seed 0) - CER_link(CRC-BH, M_stat), metric "
                            "cer_link_mean (mean over test incidents with a positive denominator)",
                "dataset": "cf_ton_h (template-held-out)", "alphas": [0.1, 0.2], "K": 100, "n_test": 100,
                "splits": 200, "test": "one-sided paired incident-pool bootstrap (same resampled incidents and "
                "splits for both banks; B = 1000, 50 splits per replicate), H0: Delta <= 0",
                "multiplicity": "Holm over the two alphas at 0.05",
                "success": "iff the Holm-adjusted lower confidence bound of Delta is > 0 AND the two-sided 95% "
                           "CIs of Delta for BOTH placebos (fuse_perm_s0, fuse_band_s0) contain 0 at both alphas",
                "robustness": "seeds 1-4 of M_fuse and cer_link_pooled reported, not tested"},
            "secondary": ["PRE-REGISTERED STRATUM (coordinator 18 Oct): Delta STR (fusion s0 - stats, CRC-BH, alphas "
                          "0.1 / 0.2) on CF-TON-H restricted to transitions whose unit-type pair is unseen in "
                          "training (strata.unseen_type_pair), paired incident-pool bootstrap CI; also on CF-TON",
                          "in-template Delta CER_link on cf_ton (alphas 0.1, 0.2)", "Delta STR (both datasets)",
                          "decoy-rich stratum (>= 2 decoy campaigns)", "content-rich stratum",
                          "chain endpoints (maximal-chain FDR, CCR)",
                          "STR vs kappa in {0, 0.05, 0.1, 0.2} and the kappa = 1 arm",
                          "lenient FDP reported next to the strict one (AMB share ~50% accepted, coordinator)",
                          "all secondary results descriptive with incident-bootstrap CIs (no multiplicity claim)"],
        },
        "holm_families": {"validity": cells, "llm": ["cf_ton_h|alpha=0.1", "cf_ton_h|alpha=0.2"]},
        "statistics": {"ci": "incident-pool cluster bootstrap, B = 1000, 50 splits per replicate, percentile CIs; "
                       "split-to-split SD and the split-conditional FDR distribution reported",
                       "iot23_loso": "jackknife CI over scenarios", "gotham": "Clopper-Pearson per incident, "
                       "conditional on the incident; no FDR claim", "precision_matching": "fine alpha grid"},
        "cf_ton_h_heldout": {
            "templates_held_out": ["T-WEBI", "T-WEBX"], "train_templates": ["T-BOT", "T-C2DOS", "T-PIVOT", "T-CRED"],
            "definition": "held out at the template / episode-type level (coordinator decision, 18 Oct 23:25): no "
                          "held-out edge is fully unseen at the unit-type level; per-edge unseen shares on CF-TON-H "
                          "train (P2-E/P2-B audit, typer v2): scan->xss 0.91, scan->inj 0.50, inj->bd 0.30, "
                          "xss->bd 0.08, bd->ransom 0.014. CF-TON-H stays the primary-endpoint dataset.",
            "audit_file": "results/interim/cf_ton_h/audit/heldout.json",
            "audit_file_blake3": D.file_blake3(D.interim_dir("cf_ton_h") / "audit" / "heldout.json")},
        "strata": {
            "unseen_type_pair": "pre-registered E3 stratum: observable transitions whose (type_u, type_v) unit-type "
                                "pair has no training positive in the scorer's training pool "
                                "(features/priors.PairPriors.seen_pair on models/<scorer>/priors.json); evaluated "
                                "with the stratum banks losses/<ds>/<scorer_id>__stratum-unseen (STR restricted to "
                                "the stratum; FDR columns unchanged)",
            "decoy_rich": ">= 2 decoy campaigns in the incident", "benign_only": "no main campaign",
            "short_chain": "longest GT root-leaf path <= 2 transitions",
            "content_rich": "incident with >= 1 unit carrying a content flag (descriptive)"},
        "fallback_framing": FALLBACK, "low_power_framing": LOW_POWER,
        "gates": gates(),
        "deviations": [
            "Cascade band shrink in the TRUE-link quantile space ('pos') instead of the all-candidate quantile "
            "space: on training OOF the literal rule kept ~1% of true links in the band (train-only evidence).",
            "AMB candidates are LightGBM training negatives (the strict calibration loss counts them false); "
            "priors exclude AMB as specified.",
            "Early stopping on validation AUC; final refit rounds = mean best iteration of the 5 folds.",
            "Transition prior smoothing centred on the global positive rate (unseen pair -> global log-odds).",
            "SNI rarity not used (raw tls_snis not FEATURE_SAFE; no SNI in TON-derived data).",
            "Extended self-continuation lever (same src and type) evaluated on dev and NOT adopted (it removes "
            "true links: r_CT drop > 0.02).",
        ],
        "coordinator_decisions": [
            "18 Oct 19:52: relation-aware enab (GT, all sources) and typer v2 before G1 (units code fp 23dced72c6a24397)",
            "18 Oct 23:25: CF-TON-H accepted as held out at the template / episode-type level; the unseen-type-pair "
            "stratum is pre-registered (strata.unseen_type_pair)",
            "18 Oct 23:25: AMB share (~50% of candidates, same-episode 30-33%) accepted: strict calibration as designed, "
            "lenient FDP reported too",
            "G1 levers only if STR(0.2) < 0.15 (G1 v2 passed with defaults: no lever adopted)"],
        "integrity": {"cal_test_label_guard": "scoring/driver.check_label_access refuses cal/test labels until this "
                      "file exists", "no_cal_test_outputs_at_write": None},
    }
    missing = [k for k in ("scorer_hashes", "prompt_hashes") if not cfg[k] or
               (k == "scorer_hashes" and not (cfg[k].get("S_TON") and cfg[k].get("S_TON_H")))]
    if not cfg["gates"]["G1"]:
        missing.append("gates.G1")
    if not cfg["gates"]["llm_budget"]:
        missing.append("gates.llm_budget")
    for k in REQUIRED_KEYS:
        if k not in cfg:
            missing.append(k)
    return cfg, missing


def cal_test_outputs_exist() -> list[str]:
    hits = []
    for p in glob.glob(str(results_root() / "losses" / "*")):
        if not Path(p).name.startswith("cf_ton_dev"):
            hits.append(p)
    hits += glob.glob(str(results_root() / "exp" / "*"))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--draft", action="store_true")
    g.add_argument("--final", action="store_true")
    a = ap.parse_args()
    cfg, missing = build_cfg()
    if a.draft:
        p = D.gates_dir() / "PREREG_draft.json"
        atomic_write_text(p, canonical_bytes(cfg).decode())
        log(f"draft -> {p}; missing for --final: {missing or 'none'}")
        return 0
    if missing:
        log(f"REFUSED: missing {missing}")
        return 2
    leaks = cal_test_outputs_exist()
    if leaks:
        log(f"REFUSED: calibration/test outputs already exist: {leaks[:5]}")
        return 3
    g1 = jload(D.gates_dir() / "G1.json") or {}
    cur = (units_fp().get("cf_ton") or {}).get("code_fp")
    cur = cur[0] if isinstance(cur, list) and len(cur) == 1 else cur
    if g1.get("units_code_fp") is None or g1.get("units_code_fp") != cur:
        log(f"REFUSED: G1 was computed on units {g1.get('units_code_fp')} but cf_ton is now {cur}")
        return 5
    cfg["integrity"]["no_cal_test_outputs_at_write"] = True
    h = write_prereg(D.prereg_path(), cfg, strict=True)
    atomic_write_text(D.gates_dir() / "prereg_blake3.txt", h + "\n")
    atomic_write_json(D.gates_dir() / "prereg_written.json", {"path": str(D.prereg_path().relative_to(P2_ROOT)),
                                                              "blake3": h, "written": cfg["created"]})
    log(f"PREREG written: {D.prereg_path()}  BLAKE3 {h}")
    print(h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
