#!/usr/bin/env python3
"""LLM budget gate (DESIGN §8.2 "LLM budget arithmetic" + rule) - throughput half.

    calls = Σ_ds κ · Σ_{train + cal/test incidents} |C|  +  B5 arm (15 incidents × all candidates, κ = 1)
    hours = calls / measured cand/s

Rule (§8.2): if the MUST total exceeds the call budget (300 k) - or the S5 time budget (10.5 h)
at the measured throughput - set κ_target 0.10 -> 0.05 first; then CF-TON cal/test 300 -> 240 and
CF-TON-H cal/test 200 -> 160. Train is never reduced below 120 (it is not reduced here at all).

Throughput comes from the 00_bench_llm.py JSON (best batch size) or --cand-per-s. The |C| half
uses a placeholder N̄ = 3,000 candidates per incident until candidate counts exist; pass measured
means with --n-bar CF-TON=2710,CF-TON-H=3120 (or --counts-json {"CF-TON": {"mean": ...}, ...}).
Pure arithmetic: no model, no GPU.

    .venv/bin/python P2_CertiChain/scripts/18_budget_gate.py
    .venv/bin/python P2_CertiChain/scripts/18_budget_gate.py --cand-per-s 6.3 --n-bar CF-TON=2500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "results" / "bench"
# prefer the bench on cards rendered by features/cards.py (realistic token counts), else the template-card one
BENCH = next((p for p in (BENCH_DIR / "llm_qwen2.5-7b-instruct-4bit_rendered.json",
                          BENCH_DIR / "llm_qwen2.5-7b-instruct-4bit.json") if p.exists()),
             BENCH_DIR / "llm_qwen2.5-7b-instruct-4bit_rendered.json")
# §8.2 MUST runs: (train incidents, cal/test incidents)
MUST = {"CF-TON": (150, 300), "CF-TON-H": (150, 200)}
B5 = {"dataset": "CF-TON", "incidents": 15}
SHOULD_CALLS = {"IoT-23": 46_000, "CF-GOTHAM-T + Gotham": 45_000, "E12 (3B + 30B-A3B)": 48_000, "B6": 8_000}
REDUCED_CALTEST = {"CF-TON": 240, "CF-TON-H": 160}
TRAIN_MIN = 120


def plan(n_bar: dict[str, float], kappa: float, caltest: dict[str, int]) -> dict:
    rows, total = [], 0.0
    for ds, (tr, ct) in MUST.items():
        ct = caltest.get(ds, ct)
        c = kappa * (tr + ct) * n_bar[ds]
        rows.append({"run": f"{ds} ({tr + ct} incidents: {tr} train + {ct} cal/test)", "calls": c})
        total += c
    b5 = B5["incidents"] * n_bar[B5["dataset"]]
    rows.append({"run": f"B5 / kappa=1 arm ({B5['incidents']} incidents x all candidates)", "calls": b5})
    total += b5
    return {"kappa": kappa, "caltest": dict(caltest), "rows": rows, "must_calls": total}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-json", default=str(BENCH))
    ap.add_argument("--cand-per-s", type=float, default=None, help="override the measured throughput")
    ap.add_argument("--n-bar", default="", help="mean candidates per incident, e.g. CF-TON=2710,CF-TON-H=3120")
    ap.add_argument("--counts-json", default=None, help='{"CF-TON": {"mean": 2710}, ...} from candidate generation')
    ap.add_argument("--default-n-bar", type=float, default=3000.0, help="§8.2 placeholder")
    ap.add_argument("--kappa", type=float, default=0.10)
    ap.add_argument("--max-calls", type=float, default=300_000)
    ap.add_argument("--max-hours", type=float, default=10.5, help="S5 slot (§8.2 table)")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    src = "--cand-per-s"
    cps = a.cand_per_s
    bench = None
    if cps is None:
        p = Path(a.bench_json)
        if not p.exists():
            print(f"no throughput: {p} missing and --cand-per-s not given", file=sys.stderr)
            return 2
        bench = json.loads(p.read_text())
        cps = float(bench["best"]["cand_per_s"])
        src = f"{p.name} (best B={bench['best']['B']}, suffix tokens mean {bench['suffix_tokens']['mean']}, " \
              f"cards={bench.get('cards', 'template')})"
    n_bar = {ds: a.default_n_bar for ds in MUST}
    n_src = {ds: "placeholder" for ds in MUST}
    if a.counts_json:
        for ds, v in json.loads(Path(a.counts_json).read_text()).items():
            if ds in n_bar:
                n_bar[ds], n_src[ds] = float(v["mean"] if isinstance(v, dict) else v), "measured"
    for kv in filter(None, a.n_bar.split(",")):
        ds, v = kv.split("=")
        n_bar[ds.strip()], n_src[ds.strip()] = float(v), "given"

    def hours(calls):
        return calls / cps / 3600.0

    def show(pl, title):
        print(f"\n{title}  (kappa = {pl['kappa']:.2f})")
        print(f"  {'run':66s} {'calls':>9s} {'hours':>7s}")
        for r in pl["rows"]:
            print(f"  {r['run']:66s} {r['calls']:9,.0f} {hours(r['calls']):7.2f}")
        print(f"  {'MUST total':66s} {pl['must_calls']:9,.0f} {hours(pl['must_calls']):7.2f}")

    print(f"throughput: {cps:.2f} cand/s from {src}")
    print("mean candidates per incident: " + ", ".join(f"{k}={v:,.0f} ({n_src[k]})" for k, v in n_bar.items()))
    if cps < 5:
        print(f"\n!!! measured throughput {cps:.2f} cand/s < 5 cand/s: kappa_target must drop to 0.05 !!!")
    elif cps < 8:
        print(f"\n!   measured throughput {cps:.2f} cand/s is below the planned 8 cand/s")

    base = plan(n_bar, a.kappa, {})
    show(base, "Plan A (DESIGN defaults)")
    steps = [base]
    over = lambda pl: pl["must_calls"] > a.max_calls or hours(pl["must_calls"]) > a.max_hours  # noqa: E731
    decision = "keep defaults"
    why = []
    if base["must_calls"] > a.max_calls:
        why.append(f"calls {base['must_calls']:,.0f} > {a.max_calls:,.0f} (DESIGN rule)")
    if hours(base["must_calls"]) > a.max_hours:
        why.append(f"hours {hours(base['must_calls']):.2f} > {a.max_hours} (S5 slot; --max-hours inf to disable)")
    if why:
        print("\nover budget: " + "; ".join(why))
    if over(base):
        p2 = plan(n_bar, 0.05, {})
        show(p2, "Rule step 1: kappa_target -> 0.05")
        steps.append(p2)
        decision = "kappa_target = 0.05"
        if over(p2):
            p3 = plan(n_bar, 0.05, REDUCED_CALTEST)
            show(p3, "Rule step 2: kappa 0.05 + CF-TON cal/test 300->240, CF-TON-H 200->160 (train kept at 150 >= 120)")
            steps.append(p3)
            decision = "kappa_target = 0.05 and reduced cal/test sizes"
            if over(p3):
                decision += " - STILL OVER BUDGET: escalate (smaller N_max/W levers on dev, or more GPU hours)"
    final = steps[-1]
    should = sum(SHOULD_CALLS.values())
    print(f"\nSHOULD (after MUST shards complete): {should:,} calls = {hours(should):.2f} h  "
          + "; ".join(f"{k} {v:,}" for k, v in SHOULD_CALLS.items()))
    print(f"\nDECISION: {decision}; MUST {final['must_calls']:,.0f} calls = {hours(final['must_calls']):.2f} h "
          f"(limits {a.max_calls:,.0f} calls / {a.max_hours} h)")
    if a.json_out:
        out = {"cand_per_s": cps, "source": src, "n_bar": n_bar, "n_bar_source": n_src, "steps": steps,
               "decision": decision, "must_calls": final["must_calls"], "must_hours": hours(final["must_calls"]),
               "should_calls": should, "should_hours": hours(should)}
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
