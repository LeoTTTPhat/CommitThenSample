#!/usr/bin/env python
"""17 — G1 power gate on the CF-TON DEV pool + the LLM budget gate (DESIGN §8.3, §8.2). Owner: P2-F.

Run from the project root under memguard:

  .venv/bin/python memguard/memguard.py run --name p2_G1 --limit-gb 90 --workers 4 -- \
      .venv/bin/python -I -u P2_CertiChain/scripts/17_power_gate.py --levers all --budget

G1 (dev pool only: 60 CF-TON dev incidents, never in a paper table): S_TON stats scores
(results/scores/cf_ton/S_TON.stat/), S_0 = models/S_TON/null_pool.npy; loss bank of the dev pool
(P2-C build_loss_tables) -> within-incident AUC of s_stat, N distribution, N/r, r_{C,T}, AMB share, certified
STR (CRC-BH, P2-C eval.runner.evaluate) at alpha in {0.1, 0.2} on 100 internal 40/20 splits.

Levers (dev pool only; each re-measured with S_TON RETRAINED on the lever-modified CF-TON train pool):
  P5 / P5+victim_seed / P5+victim_seed+prune_all   DESIGN §3.3 (i)-(iii), cumulative, via P2-B's scripts/07
      per-incident function (imported, not edited) with configs/units.yaml:candidates.g1_levers switched on,
      run on train + dev into results/interim_levers/<tag>/cf_ton/ (the real interim tree is untouched);
  src_type / src_type_dstgap   coordinator's extra label-free lever (18 Oct): extended self-continuation
      exclusion applied AFTER P2-B's generate_candidates (scoring/driver candidate view).
--levers auto (only if the gate fails, §3.3 order, stop at the first pass) | all | none.
Adoption: a lever is adopted iff STR(0.2) < 0.15 for the default and the lever passes (gate rule), or it
raises dev STR at both alphas while lowering dev r_CT by at most 0.02 (coordinator rule); else defaults kept.

LLM budget (--budget): exact |C| and in-band counts (label-free) of the MUST runs with the trained bands,
the |C| distribution of 20 dev incidents, measured throughput (results/bench, dev pilot) -> calls and hours
(§8.2 rule: > 300 k calls or > 10.5 h at kappa 0.1 -> kappa 0.05 first) -> results/gates/llm_budget.json.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import argparse  # noqa: E402
import copy  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from certichain.certify.losses import build_loss_tables  # noqa: E402
from certichain.eval.gates import auc, n_over_r  # noqa: E402
from certichain.eval.runner import evaluate  # noqa: E402
from certichain.eval.splits import random_splits  # noqa: E402
from certichain.features import stats as F  # noqa: E402
from certichain.io.paths import P2_ROOT, atomic_write_bytes, results_root, run_pool  # noqa: E402
from certichain.scoring import cascade as C  # noqa: E402
from certichain.scoring import driver as D  # noqa: E402
from certichain.scoring import fusion as FU  # noqa: E402
from certichain.scoring.cascade import Band  # noqa: E402
from certichain.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory  # noqa: E402

DS = "cf_ton"
SCORER = "S_TON"
P2B_LEVERS = [("P5", {"P5": True}), ("P5+victim_seed", {"P5": True, "victim_seed_only": True}),
              ("P5+victim_seed+prune_all", {"P5": True, "victim_seed_only": True, "prune_benign_all": True})]
FILTER_LEVERS = [("src_type", "src_type"), ("src_type_dstgap", "src_type_dstgap")]
STR_PASS = 0.15
RCT_TOL = 0.02
_RUN = {"suffix": ""}          # "_smoke" in --smoke mode: every lever output goes to *_smoke directories


def log(msg: str) -> None:
    print(f"[17 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def q(xs) -> dict:
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return {}
    return {f"p{p}": float(np.percentile(a, p)) for p in (0, 10, 25, 50, 75, 90, 100)} | {"mean": float(a.mean()),
                                                                                       "n": int(a.size)}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def m08():
    return load_module("s08", P2_ROOT / "scripts" / "08_fit_stats.py")


def units_code_fp() -> str | None:
    s = D.interim_dir(DS) / "summary_07.json"
    if not s.exists():
        return None
    fp = json.loads(s.read_text()).get("code_fp")
    return fp[0] if isinstance(fp, list) and len(fp) == 1 else fp


# ----------------------------------------------------------------------------- scorer per configuration
def default_scorer():
    md = D.models_dir(SCORER)
    from certichain.features.priors import PairPriors
    return {"booster": FU.load_booster(md / "lgbm_stat.txt"), "priors": PairPriors.load(md / "priors.json"),
            "s0": np.load(md / "null_pool.npy"), "band": Band.load(md / "band.json"), "retrained": False}


def retrain_scorer(tag: str) -> dict:
    """S_TON M_stat refit on the CF-TON train pool of the CURRENT view (lever root / candidate filter)."""
    s = m08()
    cfg = s.load_cfg()
    pairs = s.training_set(SCORER, cfg)
    incs = s.load_training(pairs)
    tr = cfg["training"]
    res = FU.cross_fit(incs, F.FEATURES, params=s.lgb_params(cfg), n_folds=int(tr["n_folds"]),
                       fold_seed=int(tr["fold_seed"]), max_rounds=int(tr["max_rounds"]),
                       es_rounds=int(tr["early_stopping"]), W=float(tr["W"]), log=log, check=check_memory)
    cc = cfg["cascade"]
    band = C.fit_band(res.oof, res.y, cc["kappa_target"], cc["kappa_max"], cc["rule"], cc["pos_below"], cc["fdp_above"])
    out = results_root() / f"models_levers{_RUN['suffix']}" / tag.replace("+", "_") / SCORER
    atomic_write_bytes(out / "lgbm_stat.txt", res.booster.model_to_string().encode())
    res.priors.save(out / "priors.json")
    D.atomic_save_npy(out / "null_pool.npy", res.null_pool())
    band.save(out / "band.json")
    sc = FU.split_by_incident(res.oof, res.sizes)
    return {"booster": res.booster, "priors": res.priors, "s0": res.null_pool(), "band": band, "retrained": True,
            "oof_within_auc_strict": FU.within_auc(sc, [i.y for i in incs], "strict")["mean"],
            "n_train_rows": int(res.y.size), "models_dir": str(out.relative_to(results_root()))}


def dev_scores(incs: list[str], scorer: dict, use_cached: bool) -> dict[str, np.ndarray]:
    out = {}
    for k, inc in enumerate(incs):
        if use_cached and D.scores_path(DS, f"{SCORER}.stat", inc).exists():
            out[inc] = D.read_scores(DS, f"{SCORER}.stat", inc)["s"].to_numpy()
        else:
            feats = D.load_features(DS, inc)
            out[inc] = FU.score_incident(scorer["booster"], scorer["priors"], FU.IncidentData(inc, feats), F.FEATURES) \
                if feats.height else np.zeros(0)
        if k % 20 == 0:
            check_memory()
    return out


def measure(tag: str, incs: list[str], scores: dict[str, np.ndarray], scorer: dict, n_splits: int, K: int,
            n_test: int, bank_dir: Path) -> dict:
    """All G1 numbers for one candidate configuration of the dev pool."""
    I = [D.build_incident(DS, inc, scores[inc]) for inc in incs]
    bank = build_loss_tables(I, scorer["s0"], bank_dir, n_jobs=1, overwrite=True,
                             meta={"pool": "cf_ton dev", "tag": tag, "scorer": f"{SCORER}.stat"})
    sp = random_splits(bank.n, K, range(n_splits), n_test=n_test)
    res = {}
    keep = ("str_mean", "str_pooled", "ostr_mean", "cer_link_mean", "cer_link_pooled", "recall_pooled", "fdr",
            "fdr_lenient", "pooled_fdr", "abstain", "n_claims", "frac_empty", "lam", "bridge_recall", "ucr",
            "analyst_load")
    per_split = {}
    for a in (0.1, 0.2):
        r = evaluate(bank, "crc_bh", sp, a, cert_gap=False)
        res[f"alpha={a}"] = {k: float(np.nanmean(v)) for k, v in r.items() if k in keep} | {
            "str_mean_split_sd": float(np.nanstd(r["str_mean"])), "fdr_split_sd": float(np.nanstd(r["fdr"]))}
        per_split[a] = np.asarray(r["str_mean"], dtype=float)
    aucs_s, aucs_0, Ns, n_obs_in_C, n_obs, n_true, kap, n_amb = [], [], [], [], [], [], [], []
    for inc, it in zip(incs, I):
        y = np.asarray(it.y)
        s = np.asarray(it.scores)
        a1 = auc(s[y == 1], s[y != 1])
        a0 = auc(s[y == 1], s[y == 0])
        if np.isfinite(a1):
            aucs_s.append(a1)
        if np.isfinite(a0):
            aucs_0.append(a0)
        vt = D.view_transition_stats(DS, inc)
        Ns.append(s.size)
        n_obs_in_C.append(vt["n_T_obs_in_C"])
        n_obs.append(vt["n_T_obs"])
        n_true.append(int((y == 1).sum()))
        n_amb.append(int((y == -1).sum()))
        kap.append(int(scorer["band"].contains(s).sum()))
    Ns_a = np.asarray(Ns, float)
    r_ct_per = [a / b for a, b in zip(n_obs_in_C, n_obs) if b]
    out = {
        "tag": tag, "n_incidents": len(incs), "N": q(Ns), "N_total": int(Ns_a.sum()),
        "amb_share": float(sum(n_amb) / max(Ns_a.sum(), 1)),
        "within_auc_strict": {"mean": float(np.mean(aucs_s)) if aucs_s else None,
                              "median": float(np.median(aucs_s)) if aucs_s else None, "n": len(aucs_s)},
        "within_auc_vs_null0": {"mean": float(np.mean(aucs_0)) if aucs_0 else None,
                                "median": float(np.median(aucs_0)) if aucs_0 else None, "n": len(aucs_0)},
        "N_over_r": {"per_transition_in_C": n_over_r(Ns, n_obs_in_C), "per_true_link": n_over_r(Ns, n_true)},
        "r_CT": {"pooled": float(sum(n_obs_in_C) / max(sum(n_obs), 1)),
                 "mean": float(np.mean(r_ct_per)) if r_ct_per else None},
        "n_T_obs": int(sum(n_obs)), "n_T_obs_in_C": int(sum(n_obs_in_C)), "n_true_links": int(sum(n_true)),
        "kappa_dev": float(sum(kap) / max(Ns_a.sum(), 1)), "in_band_dev": int(sum(kap)),
        "splits": {"K": K, "n_test": n_test, "n_splits": n_splits, "seeds": [0, n_splits - 1]},
        "crc_bh": res, "bank": str(bank_dir.relative_to(results_root())), "bank_input_hash": bank.meta.get("input_hash"),
        "scorer": {k: v for k, v in scorer.items() if k not in ("booster", "priors", "s0", "band")},
        "_str_per_split": {str(a): v.tolist() for a, v in per_split.items()},
    }
    log(f"[{tag}] N total {out['N_total']} (p50 {out['N'].get('p50')}); AMB {out['amb_share']:.3f}; within-AUC "
        f"{out['within_auc_strict']['mean']:.4f}; r_CT {out['r_CT']['pooled']:.3f}; STR@0.1 "
        f"{res['alpha=0.1']['str_mean']:.3f} STR@0.2 {res['alpha=0.2']['str_mean']:.3f} "
        f"(FDR {res['alpha=0.1']['fdr']:.3f}/{res['alpha=0.2']['fdr']:.3f})")
    return out


# ----------------------------------------------------------------------------- P2-B levers (07 re-run)
_M07: dict = {}


def _lever_worker(args):
    """Pool worker: P2-B's 07 process_incident for one incident with the lever config (spawn-safe)."""
    inc, cfg, out_root = args
    if "m" not in _M07:
        _M07["m"] = load_module("p2b_07", P2_ROOT / "scripts" / "07_units_candidates_gt.py")
    m07 = _M07["m"]
    m07.OUT = Path(out_root)
    return m07.process_incident(inc, cfg, in_worker=True)


def run_p2b_lever(tag: str, switches: dict, incs: list[str], workers: int) -> Path:
    m07 = load_module("p2b_07", P2_ROOT / "scripts" / "07_units_candidates_gt.py")
    root = results_root() / f"interim_levers{_RUN['suffix']}" / tag.replace("+", "_")
    cfg = copy.deepcopy(m07.load_cfg())
    cfg["candidates"]["g1_levers"] = {**{"P5": False, "victim_seed_only": False, "prune_benign_all": False}, **switches}
    for sub in m07.OUT_DIRS:
        (root / DS / sub).mkdir(parents=True, exist_ok=True)
    all_incs = {i["inc_id"]: i for i in m07.list_incidents(DS, cfg)}
    fp = m07.code_fingerprint(cfg)

    def done(i):
        p = root / DS / "stats" / f"{i}.json"
        return p.exists() and json.loads(p.read_text()).get("code_fp") == fp
    todo = [i for i in incs if not done(i)]
    log(f"[{tag}] 07 with levers {cfg['candidates']['g1_levers']} on {len(todo)} incidents ({workers} workers)")
    _, errors = run_pool(_lever_worker, [(all_incs[i], cfg, str(root)) for i in todo], workers,
                         label=lambda t: t[0]["inc_id"])
    if errors:
        raise RuntimeError(f"[{tag}] 07 failed on {len(errors)} incidents: {errors[:2]}")
    return root / DS


# ----------------------------------------------------------------------------- budget
def budget_gate(dev_incs: list[str]) -> dict:
    """Label-free LLM call counts for the MUST runs + throughput -> hours (DESIGN §8.2)."""
    bench = results_root() / "bench"
    tp = {}
    for tag, f in (("qwen2.5-7b-instruct-4bit", "llm_qwen2.5-7b-instruct-4bit_rendered.json"),
                   ("qwen3-30b-a3b-instruct-2507-4bit", "llm_qwen3-30b-a3b-instruct-2507-4bit.json"),
                   ("llama-3.2-3b-instruct-4bit", "llm_llama-3.2-3b-instruct-4bit_rendered.json")):
        p = bench / f
        if p.exists():
            best = json.loads(p.read_text()).get("best")
            tp[tag] = float(best["cand_per_s"]) if isinstance(best, dict) and best.get("cand_per_s") else None
    for alias, tag in (("qwen7b", "qwen2.5-7b-instruct-4bit"), ("qwen3moe", "qwen3-30b-a3b-instruct-2507-4bit")):
        for rev in (1, 0):
            p = D.gates_dir() / f"pilot_rev{rev}_{alias}.json"
            if p.exists() and json.loads(p.read_text()).get("mean_ms_per_cand"):
                tp[tag + " (dev pilot)"] = 1000.0 / float(json.loads(p.read_text())["mean_ms_per_cand"])
                break
    runs = {}
    for ds, g in D.MUST_GROUPS:
        try:
            items, plan = D.plan_group(ds, g, write=False)
            runs[f"{ds}/{g}"] = {"calls": len(items), "kappa": plan.get("kappa_realised"),
                                 "n_incidents": plan.get("n_incidents")}
        except (FileNotFoundError, KeyError) as e:
            runs[f"{ds}/{g}"] = {"calls": None, "error": str(e)}
    N20 = [D.load_cands(DS, i).height for i in dev_incs[:20]]
    must = sum(v["calls"] or 0 for v in runs.values())
    hours = {k: (must / v / 3600.0 if v else None) for k, v in tp.items()}
    fz = D.gates_dir() / "prompt_freeze.json"
    main_tag = json.loads(fz.read_text())["main_model_id"].split("/")[-1].lower() if fz.exists() else None
    if main_tag and (hours.get(main_tag + " (dev pilot)") or hours.get(main_tag)):
        h_main = hours.get(main_tag + " (dev pilot)") or hours.get(main_tag)
        basis = main_tag
    else:                                   # before the model choice: the slower of the two candidates
        cands = [h for k, h in hours.items() if h and ("qwen2.5-7b" in k or "qwen3-30b" in k)]
        h_main = max(cands) if cands else None
        basis = "slower of qwen2.5-7b / qwen3-30b-a3b"
    ok = must <= 300_000 and (h_main is None or h_main <= 10.5)
    return {"runs": runs, "dev_pilot_calls_excluded": 2 * 2000, "must_calls_kappa0.10": must, "N_dev20": q(N20),
            "N_bar_dev20": float(np.mean(N20)) if N20 else None, "throughput_cand_per_s": tp, "hours_at": hours,
            "hours_basis": basis, "hours_main": h_main,
            "rule": "DESIGN §8.2: > 300 k calls (or > ~10.5 h at the measured throughput) at kappa 0.1 -> kappa 0.05",
            "decision": "kappa_target 0.10 kept" if ok else "kappa_target -> 0.05 (budget rule)"}


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--levers", choices=["auto", "all", "none"], default="auto")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", action="store_true")
    ap.add_argument("--budget-only", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="code-path test: first N train/dev incidents, writes "
                    "G1_smoke.json under results/gates_smoke (never a gate result)")
    a = ap.parse_args()
    try:
        dev = D.list_incidents(DS, "dev")
        if a.smoke:
            _RUN["suffix"] = "_smoke"
            dev = dev[: a.smoke]
            a.K, a.n_test, a.splits = max(2, a.smoke // 2), a.smoke - max(2, a.smoke // 2), 5
        if a.budget_only:
            b = budget_gate(dev) | {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            D.write_gate("llm_budget", b)
            log(f"budget: MUST calls {b['must_calls_kappa0.10']}; hours {b['hours_at']}; {b['decision']}")
            return 0
        sc0 = default_scorer()
        log(f"dev pool: {len(dev)} incidents; |S0| = {sc0['s0'].size}; units code fp {units_code_fp()}")
        base = measure("default", dev, dev_scores(dev, sc0, False), sc0, a.splits, a.K, a.n_test,
                       results_root() / "losses" / f"cf_ton_dev{_RUN['suffix']}" / f"{SCORER}.stat" / "bank")
        g1 = {"gate": "G1", "pool": "CF-TON dev (60 incidents; never reported in a paper table)",
              "scorer": f"{SCORER}.stat", "units_code_fp": units_code_fp(),
              "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pass_rule": f"STR(alpha=0.2) >= {STR_PASS}",
              "adoption_rule": f"lever adopted iff the default fails and the lever passes, or the lever raises STR at "
                               f"both alphas with r_CT drop <= {RCT_TOL}", "default": base, "levers": []}
        passed = base["crc_bh"]["alpha=0.2"]["str_mean"] >= STR_PASS
        g1["pass_default"] = bool(passed)
        plan = []
        if a.levers == "all" or (a.levers == "auto" and not passed):
            plan += [("p2b", t, sw) for t, sw in P2B_LEVERS]
        if a.levers == "all":
            plan += [("filter", t, m) for t, m in FILTER_LEVERS]
        train = D.list_incidents(DS, "train")[: (a.smoke * 2 or None)]
        if a.smoke:
            plan = plan[:1] + [p for p in plan if p[0] == "filter"][:1]
        for kind, tag, arg in plan:
            if kind == "p2b":
                root = run_p2b_lever(tag, arg, train + dev, max(1, min(8, a.workers)))
                D.set_interim_root(DS, root)
            else:
                D.set_candidate_filter(arg)
            try:
                sc = retrain_scorer(tag)
                r = measure(tag, dev, dev_scores(dev, sc, False), sc, a.splits, a.K, a.n_test,
                            results_root() / "losses" / f"cf_ton_dev_levers{_RUN['suffix']}" / tag.replace("+", "_")
                            / "bank")
            finally:
                D.set_interim_root(DS, None)
                D.set_candidate_filter(None)
            r["lever"] = {"kind": kind, "switches": arg}
            for al in ("0.1", "0.2"):
                d = np.asarray(r["_str_per_split"][al]) - np.asarray(base["_str_per_split"][al])
                r[f"delta_str_alpha{al}"] = {"mean": float(np.nanmean(d)), "split_sd": float(np.nanstd(d))}
            r["delta_r_CT"] = r["r_CT"]["pooled"] - base["r_CT"]["pooled"]
            r["qualifies"] = bool((not passed and r["crc_bh"]["alpha=0.2"]["str_mean"] >= STR_PASS) or (
                r["delta_str_alpha0.1"]["mean"] > 0 and r["delta_str_alpha0.2"]["mean"] > 0
                and r["delta_r_CT"] >= -RCT_TOL))
            g1["levers"].append(r)
            if a.levers == "auto" and r["crc_bh"]["alpha=0.2"]["str_mean"] >= STR_PASS:
                break
        quals = [r for r in g1["levers"] if r["qualifies"]]
        best = max(quals, key=lambda r: r["crc_bh"]["alpha=0.2"]["str_mean"]) if quals else None
        g1["adopted_lever"] = best["tag"] if best else None
        g1["pass"] = bool(passed or (best is not None and best["crc_bh"]["alpha=0.2"]["str_mean"] >= STR_PASS))
        g1["action"] = (f"adopt lever '{best['tag']}' (re-run every dataset with it before anything else)" if best
                        else ("default candidate parameters kept" if passed else
                              "continue as planned with the pre-registered low-power framing (DESIGN §8.3)"))
        for r in [base] + g1["levers"]:
            r.pop("_str_per_split", None)
        if a.smoke:
            from certichain.io.paths import atomic_write_json
            atomic_write_json(results_root() / "gates_smoke" / "G1_smoke.json", g1)
            log("smoke run written to results/gates_smoke/G1_smoke.json (not a gate)")
            return 0
        D.write_gate("G1", g1)
        log(f"G1 written: pass={g1['pass']}; action: {g1['action']}")
        if a.budget:
            b = budget_gate(dev) | {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            D.write_gate("llm_budget", b)
            log(f"budget: MUST calls {b['must_calls_kappa0.10']}; hours {b['hours_at']}; {b['decision']}")
    except MemoryBudgetExceeded as e:
        log(f"[mem] {e}; exiting {EXIT_MEMORY}")
        return EXIT_MEMORY
    return 0


if __name__ == "__main__":
    sys.exit(main())
