#!/usr/bin/env python
"""09 — LLM scoring of the union of cascade bands (each candidate once per dataset; DESIGN §3.4.3, §5.3, §8.2).
Owner: P2-F.

Run from the project root under memguard WITHOUT -I (loads an MLX model from the HF cache; GPU lock):

  .venv/bin/python memguard/memguard.py run --name p2_llm_main --limit-gb 90 --workers 1 -- \
      .venv/bin/python -u P2_CertiChain/scripts/09_score_llm.py --model main --must

  --model main|qwen7b|qwen3moe|<repo id>   main = results/gates/prompt_freeze.json main_model
  --must                  the MUST groups in priority order: cf_ton train_band -> cf_ton caltest_band ->
                          cf_ton_h train_band -> cf_ton_h caltest_band -> cf_ton b5_arm (kappa = 1, 15 incidents)
  --should                the SHOULD groups whose scorers exist (IoT-23, CF-GOTHAM(-T), Gotham-real, kappa 0.2)
  --dataset D --groups g1,g2   explicit groups (scoring/driver.LLM_GROUPS)
  --plan-only             freeze and print the item lists + estimated hours (no GPU)
  --batch N               batch size override (default: MLXScorer, max(4, int(32 * scale())))

The frozen prompt (prefix text + BLAKE3 prompt hash) comes from results/gates/prompt_freeze.json; the run
refuses if P2-D's prompt_hash(prefix) no longer matches (prompt or card renderer changed after the freeze).
Item lists are frozen per group (results/llm/<ds>/<group>/items.parquet); MLXScorer.run writes resumable
4,096-candidate shards under results/llm/<ds>/<group>/<model_tag>/<prompt_hash>/ and records the model
revision + prompt hash in every shard. Exit 75 on the memory soft limit (memguard restarts; shards resume).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import yaml

from certichain.io.paths import P2_ROOT, atomic_write_json
from certichain.scoring import driver as D
from certichain.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded


def log(msg: str) -> None:
    print(f"[09 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def frozen() -> dict:
    p = D.gates_dir() / "prompt_freeze.json"
    if not p.exists():
        raise SystemExit("no results/gates/prompt_freeze.json: run scripts/19 --freeze first")
    f = json.loads(p.read_text())
    if not f.get("frozen"):
        raise SystemExit("prompt_freeze.json is not a frozen record")
    return f


def must_groups() -> list[tuple[str, str]]:
    return list(D.MUST_GROUPS)


def should_groups() -> list[tuple[str, str]]:
    out = []
    for ds, g, sc, role, kind in D.LLM_GROUPS:
        if (ds, g) in D.MUST_GROUPS:
            continue
        if not D.list_incidents(ds) or not D._group_ready(ds, g):
            continue
        out.append((ds, g))
    return out


def throughput(model_tag: str) -> float | None:
    """Measured cand/s: the dev pilot of this model if present, else P2-D's bench."""
    for rev in (1, 0):
        for alias in ("qwen3moe", "qwen7b", "llama3b"):
            p = D.gates_dir() / f"pilot_rev{rev}_{alias}.json"
            if p.exists():
                d = json.loads(p.read_text())
                if d.get("model_id", "").split("/")[-1].lower() == model_tag and d.get("mean_ms_per_cand"):
                    return 1000.0 / float(d["mean_ms_per_cand"])
    from certichain.io.paths import results_root
    for f in (results_root() / "bench").glob(f"llm_{model_tag}*.json"):
        d = json.loads(f.read_text())
        if isinstance(d.get("best"), dict):
            return float(d["best"]["cand_per_s"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="main")
    ap.add_argument("--must", action="store_true")
    ap.add_argument("--should", action="store_true")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--batch", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load((P2_ROOT / "configs" / "llm.yaml").read_text())
    fz = frozen()
    alias = fz["main_model"] if a.model == "main" else a.model
    model_id = cfg["models"].get(alias, alias)
    tag = model_id.split("/")[-1].lower()
    groups: list[tuple[str, str]] = []
    if a.must:
        groups += must_groups()
    if a.should:
        groups += should_groups()
    if a.dataset and a.groups:
        groups += [(a.dataset, g) for g in a.groups.split(",")]
    if not groups:
        raise SystemExit("nothing to do (use --must / --should / --dataset + --groups)")
    plans = []
    total = 0
    ready = []
    for ds, g in groups:
        try:
            items, plan = D.plan_group(ds, g)
        except FileNotFoundError as e:
            if a.plan_only:
                log(f"{ds}/{g}: not plannable yet ({e})")
                continue
            raise SystemExit(f"{ds}/{g}: not plannable ({e}); fit the scorer with scripts/08 --score first")
        ready.append((ds, g))
        done = D.read_llm(ds, tag, fz["prompt_hash"], groups=[g])
        plans.append({"dataset": ds, "group": g, "n_items": len(items), "n_scored": int(done.height), **{
            k: plan.get(k) for k in ("kappa_realised", "n_incidents", "n_dedup_earlier")}})
        total += len(items) - int(done.height)
        log(f"{ds}/{g}: {len(items)} items ({done.height} already scored); kappa {plan.get('kappa_realised')}")
    tp = throughput(tag)
    est_h = (total / tp / 3600.0) if tp else None
    log(f"{total} candidates to score with {model_id}; measured {tp} cand/s -> "
        f"{est_h if est_h is None else round(est_h, 2)} h")
    summary = {"model_id": model_id, "prompt_hash": fz["prompt_hash"], "groups": plans, "to_score": total,
               "throughput_cand_per_s": tp, "estimated_hours": est_h, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if a.plan_only:
        atomic_write_json(D.llm_root("_runs") / f"plan_{tag}_{fz['prompt_hash']}.json", summary)
        return 0
    prefix = (P2_ROOT / fz["prefix_file"]).read_text()
    try:
        recs = D.run_groups(model_id, prefix, ready, frozen_hash=fz["prompt_hash"], job=f"p2_llm_{tag}",
                            batch_size=a.batch, log=log)
    except MemoryBudgetExceeded as e:
        log(f"[mem] {e}; shards written so far are kept; exiting {EXIT_MEMORY}")
        return EXIT_MEMORY
    summary["runs"] = recs
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary["calls"] = sum(r["n_items"] for r in recs)
    summary["hours"] = sum(r["wall_s"] for r in recs) / 3600.0
    atomic_write_json(D.llm_root("_runs") / f"run_{tag}_{fz['prompt_hash']}_{time.strftime('%Y%m%d-%H%M%S')}.json",
                      summary)
    log(f"done: {summary['calls']} calls in {summary['hours']:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
