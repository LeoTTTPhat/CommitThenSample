#!/usr/bin/env python
"""11 — per-incident loss tables (CRC-BH and score-threshold families, FDX, chains, transition/TP tables) for
each (dataset, scorer) via P2-C's certify.losses.build_loss_tables, plus the E4 corrupted-score variants for
CF-TON fusion (DESIGN §3.5, §6.1 E4, §7.3). Owner: P2-F.

Run from the project root under memguard (pool workers <= 8; spawn => this file has a __main__ guard):

  .venv/bin/python memguard/memguard.py run --name p2_loss_cf_ton --limit-gb 90 --workers 8 -- \
      .venv/bin/python -I -u P2_CertiChain/scripts/11_precompute_losses.py --dataset cf_ton \
      --scorers S_TON.stat,S_TON.fuse_s0,S_TON.fuse_s1,S_TON.fuse_s2,S_TON.fuse_s3,S_TON.fuse_s4,S_TON.lag_llr --e4

  --pool caltest|dev|all   incidents of the bank (default: the cal/test pool; IoT-23 / Gotham-real: all windows)
  --e4                     E4 variants of <S>.fuse_s0 on CF-TON: s_LLM noise sigma {1,2,4}, inversion f {0.5,1},
                           memorisation +delta {2,5} on in-band nulls (cal AND test; M_fuse trained clean), and
                           test-only top-score injection q {1,3,10} (banks built with every pool incident injected:
                           the runner takes calibration rows from the clean bank, test rows from the injected one;
                           banks are row-aligned)
  --n-jobs N               build_loss_tables processes (<= 8)

Cal/test labels are sealed until results/PREREG_2026-10-21.json exists (scoring/driver guard). Output:
results/losses/<ds>/<scorer_id>/bank/ (P2-C LossBank: both rule families 'bh' and 'st' = thr, edge / FDX /
chain / TP / transition tables, CSR prefix counts, index.parquet) + provenance.json. A bank with the same input
hash is not rebuilt.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from certichain.certify.losses import build_loss_tables  # noqa: E402
from certichain.io.paths import atomic_write_json, results_root  # noqa: E402
from certichain.scoring import driver as D  # noqa: E402
from certichain.scoring import fusion as FU  # noqa: E402
from certichain.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory  # noqa: E402

E4_INJECT_Q = (1, 3, 10)


def log(msg: str) -> None:
    print(f"[11 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def null_pool_for(scorer_id: str) -> np.ndarray:
    scorer, variant = scorer_id.split(".", 1)
    base = variant.split("__", 1)[0]
    md = D.models_dir(scorer)
    name = {"stat": "null_pool.npy", "stat_prefix": "null_pool_prefix.npy", "lag_llr": "null_pool_lag_llr.npy"}.get(base)
    if name is None and base.startswith("fuse_"):
        name = f"null_pool_{base}.npy"
    if name is None:
        raise SystemExit(f"no null pool rule for scorer id {scorer_id}")
    return np.load(md / name)


def pool_incidents(ds: str, pool: str) -> list[str]:
    if ds in D.CF_VARIANTS:
        if pool == "all":
            return [i for i in D.list_incidents(ds) if D.role_of(ds, i) != "train"]
        return D.list_incidents(ds, pool)
    return D.list_incidents(ds)


def provenance(ds: str, scorer_id: str, incs, bank, extra: dict | None = None) -> dict:
    scorer = scorer_id.split(".", 1)[0]
    md = D.models_dir(scorer)
    fz = D.gates_dir() / "prompt_freeze.json"
    return {"dataset": ds, "scorer_id": scorer_id, "n_incidents": len(incs), "bank_input_hash": bank.meta.get("input_hash"),
            "models": {f.name: D.file_blake3(f) for f in sorted(md.glob("*.txt")) + sorted(md.glob("*.npy"))
                       + sorted(md.glob("*.json"))},
            "prompt_freeze_blake3": D.file_blake3(fz), "prereg_blake3": D.file_blake3(D.prereg_path()),
            "candidate_filter": D.candidate_filter(), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **(extra or {})}


POOL = {"name": "caltest"}


def bank_root(ds: str) -> str:
    """losses/<ds> for the cal/test pool (and the non-CF datasets); losses/<ds>_<pool> otherwise (e.g. dev)."""
    return ds if (POOL["name"] == "caltest" or ds not in D.CF_VARIANTS) else f"{ds}_{POOL['name']}"


def build(ds: str, scorer_id: str, incs: list[str], scores: dict[str, np.ndarray], n_jobs: int, tag: str | None = None,
          extra: dict | None = None):
    I = []
    for k, inc in enumerate(incs):
        I.append(D.build_incident(ds, inc, scores[inc]))
        if k % 50 == 0:
            check_memory()
    sid = scorer_id if tag is None else f"{scorer_id}__{tag}"
    out = results_root() / "losses" / bank_root(ds) / sid / "bank"
    s0 = null_pool_for(scorer_id)
    bank = build_loss_tables(I, s0, out, n_jobs=n_jobs, overwrite=True,
                             meta={"dataset": ds, "scorer_id": sid, "pool": POOL["name"]})
    atomic_write_json(out.parent / "provenance.json", provenance(ds, sid, incs, bank, extra))
    log(f"bank {ds}/{sid}: {bank.n} incidents, {int(bank.sizes().sum())} candidates, |S0| {s0.size} -> {out}")
    return bank


def load_scores(ds: str, scorer_id: str, incs: list[str]) -> tuple[dict, list[str]]:
    out, missing = {}, []
    for inc in incs:
        p = D.scores_path(ds, scorer_id, inc)
        if not p.exists():
            missing.append(inc)
            continue
        df = pl.read_parquet(p).sort("cand_id")
        cids = D.load_cands(ds, inc)["cand_id"].to_numpy()
        if df.height != cids.size or not np.array_equal(df["cand_id"].to_numpy(), cids):
            raise RuntimeError(f"{p}: scores do not match the candidate view of {inc}")
        out[inc] = df["s"].to_numpy().astype(np.float64)
    return out, missing


def e4_variants(ds: str, scorer: str, incs: list[str], n_jobs: int) -> None:
    from certichain.features.priors import PairPriors
    from certichain.llm import corrupt
    md = D.models_dir(scorer)
    sid = f"{scorer}.fuse_s0"
    booster = FU.load_booster(md / "lgbm_fuse_s0.txt")
    priors = PairPriors.load(md / "priors.json")
    names = (*FU.F.FEATURES, *FU.FUSE_EXTRA)
    base = {inc: D.read_scores(ds, sid, inc) for inc in incs}
    feats = {inc: D.load_features(ds, inc) for inc in incs}
    ys = {inc: D.load_labels(ds, inc)["y"].to_numpy() for inc in incs}
    for name, params in corrupt.E4_GRID.items():
        for prm in params:
            sc = {}
            for inc in incs:
                b = base[inc]
                s_llm = b["s_llm"].to_numpy().astype(np.float64)
                band = b["llm_called"].to_numpy().astype(bool)
                keys = [(inc, int(c)) for c in b["cand_id"].to_list()]
                s2 = corrupt.apply(name, s_llm, prm, y=ys[inc], seed=0, keys=keys, in_band=band)
                s2 = np.where(band, s2, np.nan)
                sc[inc] = FU.score_incident(booster, priors, FU.IncidentData(inc, feats[inc], s_llm=s2, in_band=band),
                                            names)
            build(ds, sid, incs, sc, n_jobs, tag=f"E4-{name}-{prm:g}",
                  extra={"e4": {"corruption": name, "param": prm, "applied_to": "calibration and test",
                                "model": "M_fuse s0 trained clean"}})
            check_memory()
    for q in E4_INJECT_Q:
        sc = {}
        for inc in incs:
            b = base[inc]
            s, _ = corrupt.inject_top(b["s"].to_numpy(), ys[inc], [inc] * b.height, q, seed=0,
                                      keys=[(inc, int(c)) for c in b["cand_id"].to_list()])
            sc[inc] = s
        build(ds, sid, incs, sc, n_jobs, tag=f"E4-inject-{q}",
              extra={"e4": {"corruption": "inject_top", "q": q, "applied_to": "TEST ONLY: use test rows of this "
                            "bank with calibration rows of the clean bank (row-aligned)"}})


def unseen_stratum(ds: str, scorer_id: str, incs: list[str], scores: dict[str, np.ndarray], n_jobs: int) -> None:
    """E3 secondary stratum: transitions whose (type_u, type_v) unit-type pair had no TRAINING positive
    (PairPriors.seen_pair of the scorer's training priors). Bank '<sid>__stratum-unseen': y unchanged (so FDR
    columns are the plain ones), trans / n_trans restricted to unseen-pair observable transitions, so its STR
    columns are the stratum STR (hidden-endpoint transitions have no type pair and are excluded)."""
    from certichain.features.priors import PairPriors
    scorer = scorer_id.split(".", 1)[0]
    pr = PairPriors.load(D.models_dir(scorer) / "priors.json")
    I = []
    n_unseen = n_all = 0
    for inc in incs:
        it = D.build_incident(ds, inc, scores[inc])
        feats = D.load_features(ds, inc)
        cand_unseen = ~pr.seen_pair(feats["type_u_str"].to_numpy(), feats["type_v_str"].to_numpy()) \
            if feats.height else np.zeros(0, bool)
        tr = D.load_transitions(ds, inc)
        units = D.load_units(ds, inc)
        ut = dict(zip(units["unit_id"].to_list(), units["type"].to_list()))
        obs = tr.filter(pl.col("kind") == "obs") if tr.height else tr
        un = 0
        for e, f in zip(obs["enab_u"].to_list() if obs.height else [], obs["first_v"].to_list() if obs.height else []):
            if e is not None and f is not None and not pr.seen_pair(np.array([ut.get(e)]), np.array([ut.get(f)]))[0]:
                un += 1
        trans = np.where(cand_unseen, it.trans, -1) if it.trans is not None else None
        it.trans, it.n_trans, it.n_trans_obs, it.n_trans_hid = trans, un, un, 0
        it.strata = dict(it.strata, n_unseen_transitions=un)
        n_unseen += un
        n_all += obs.height
        I.append(it)
    sid = f"{scorer_id}__stratum-unseen"
    out = results_root() / "losses" / bank_root(ds) / sid / "bank"
    bank = build_loss_tables(I, null_pool_for(scorer_id), out, n_jobs=n_jobs, overwrite=True,
                             meta={"dataset": ds, "scorer_id": sid, "pool": POOL["name"], "stratum": "unseen type pair"})
    atomic_write_json(out.parent / "provenance.json", provenance(ds, sid, incs, bank, {
        "stratum": "observable transitions whose unit-type pair had no training positive", "n_unseen": n_unseen,
        "n_obs_transitions": n_all}))
    log(f"stratum bank {ds}/{sid}: {n_unseen}/{n_all} observable transitions with an unseen type pair")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scorers", required=True)
    ap.add_argument("--pool", default="caltest")
    ap.add_argument("--e4", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--strata", action="store_true", help="also the unseen-type-pair stratum banks (E3 secondary)")
    a = ap.parse_args()
    ds = a.dataset
    POOL["name"] = a.pool
    incs = pool_incidents(ds, a.pool)
    if not incs:
        log(f"no incidents for {ds} pool {a.pool}")
        return 2
    if any(D.role_of(ds, i) not in D.TRAINING_ROLES for i in incs) and not D.prereg_exists():
        log(f"REFUSED: {ds} pool {a.pool} contains calibration/test incidents and {D.PREREG_NAME} does not exist")
        return 4
    n_jobs = max(1, min(8, a.n_jobs))
    try:
        for sid in a.scorers.split(","):
            sc, missing = load_scores(ds, sid, incs)
            if missing:
                log(f"{sid}: {len(missing)} incidents without scores (e.g. {missing[:3]}); bank built on the rest")
            build(ds, sid, [i for i in incs if i in sc], sc, n_jobs,
                  extra={"n_missing_scores": len(missing), "missing": missing[:50]})
            if a.strata:
                unseen_stratum(ds, sid, [i for i in incs if i in sc], sc, n_jobs)
        if a.e4:
            scorer = D.DATASET_SCORER[ds]
            e4_variants(ds, scorer, [i for i in incs if D.scores_path(ds, f"{scorer}.fuse_s0", i).exists()], n_jobs)
    except MemoryBudgetExceeded as e:
        log(f"[mem] {e}; exiting {EXIT_MEMORY}")
        return EXIT_MEMORY
    return 0


if __name__ == "__main__":
    sys.exit(main())
