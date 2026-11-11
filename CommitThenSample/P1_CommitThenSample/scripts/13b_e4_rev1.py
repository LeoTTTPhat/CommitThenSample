"""E4 rev1 — the baseline rows of E4 with the rev1 baselines (review R3-M2; fix list B4, M6). Analytic, ρ = 5 %.

Per LP stream (the 23 streams of 13_e4.py) and incident (content-bearing units, k = their count):
  no attack     content-evidence miss of B4 online (per-epoch budget, B4ep) and B4 online controller (B4on),
                B13 online (B13ep, B13on), B6 on the 36 strict sids (B6s), B3 scored at byte level, DLT, and the
                rev0 references (B4 / B13 offline, B6 all sids, B3 whole-unit) — seed 0 for the seeded policies;
  under attack  B4*: the Alg. 7 attacker drives P_mal below the threshold / top-k cut → miss 1 (as rev0);
                B13*: the top part is lost, content keeps only its epoch's keyed reserve π_r,e (sidecar
                .reserve_pi) → miss = Π(1 − π_r,e) ≤ (1 − φ0)^k, since π_r,e ≥ φ0 (the committed floor);
                B6s zero-day (attack-family alerts disabled) → the leftover uniform reserve π_leftover;
                B3: a prefix beyond the cut-off N → miss 1 (as rev0).
The CtS-AI rows (no attack, white-box / black-box mimicry) do not depend on the baselines and are taken from the
rev0 E4 run (results/E4/<stream>/<hash>/incidents.parquet) unchanged.
Outputs: results/E4_rev1/<stream>/incidents.parquet + DONE; results/summary/E4_incidents_rev1.parquet and
E4_rev1.json (medians, shares at or below the floor bound, per policy and attack).

    .venv/bin/python -I -u P1_CommitThenSample/scripts/13b_e4_rev1.py
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctsample.utils.io import RESULTS, atomic_write_json, mark_done  # noqa: E402

RHO = 0.05
STREAMS = RESULTS / "streams"
DESIGNS = RESULTS / "designs"
OUT = RESULTS / "E4_rev1"
SUMMARY = RESULTS / "summary"


def stream_list():
    out = [f"lp_iot23_{f}_p{p}" for f in ("1-1", "3-1", "7-1", "34-1", "48-1", "49-1") for p in (1000, 10000, 50000)]
    out += [f"lp_edge_p{p}" for p in (1000, 10000, 50000)] + ["lp_gotham_p10000", "lp_gotham_nat"]
    return out


def _miss(p, pos):
    p = np.clip(np.asarray(p, dtype=np.float64)[pos], 0.0, 1.0)
    with np.errstate(divide="ignore"):
        return float(np.exp(np.log1p(-p).sum()))


def run_stream(sid: str) -> list[dict]:
    import polars as pl
    from ctsample.baselines.policies import b3_evidence
    from ctsample.design.probs import committed_floor
    from ctsample.sim.tables import StreamTables
    T = StreamTables.load(STREAMS / sid)
    d = DESIGNS / sid
    pay = np.asarray(T.pay, dtype=bool)
    pidx = np.nonzero(pay)[0]
    content = (np.asarray(T.content, bool) if T.content is not None else np.asarray(T.incident) >= 0)[pidx]
    inc = np.asarray(T.incident)[pidx]
    meta = json.loads((d / f"U1_bm_{RHO:g}.json").read_text())
    phi0 = float(meta["phi0_committed"] or meta["phi0_stream"])
    cf = float(committed_floor(phi0))
    ids = np.unique(inc[content])
    pos = {int(i): np.nonzero(content & (inc == i))[0] for i in ids}
    load = lambda nm, suf="": np.load(d / f"{nm}_{RHO:g}{suf}.npy")              # noqa: E731
    aux = np.load(d / "aux_b3_offsets.npz")
    N = float(json.loads((d / f"B3_{RHO:g}.json").read_text())["params"]["N_bytes"])
    ev = b3_evidence(N, aux["before"], aux["off_pay"]).astype(float)
    pol = {"b4_online": load("B4ep_s0"), "b4_online_ctl": load("B4on_s0"), "b4_offline": load("B4_s0"),
           "b13_online": load("B13ep_s0"), "b13_online_ctl": load("B13on_s0"), "b13_offline": load("B13_s0"),
           "b6_strict": load("B6s"), "b6_allsids": load("B6"), "b3_bytes": ev, "b3_unit": (load("B3") > 0.5) * 1.0,
           "dlt": load("DLT"), "cts_single": load("U6tau0.2_s0_bm"), "cts_size": load("U1_bm"), "cts_unif": load("U0_bm")}
    res13 = {"b13_online": load("B13ep_s0", ".reserve_pi"), "b13_online_ctl": load("B13on_s0", ".reserve_pi")}
    b6p = json.loads((d / f"B6s_{RHO:g}.json").read_text())["params"]
    b13off = json.loads((d / f"B13_s0_{RHO:g}.json").read_text())["params"]
    rows = []
    for i in ids:
        ps = pos[int(i)]
        k = len(ps)
        bound = (1 - cf) ** k
        base = {"stream": sid, "incident": int(i), "k": int(k), "phi0": cf, "bound": float(bound)}
        for key, p in pol.items():
            rows.append(dict(base, policy=key, attack="none", miss=_miss(p, ps)))
        for key in ("b4_online", "b4_online_ctl", "b4_offline", "b3_bytes", "b3_unit"):
            rows.append(dict(base, policy=key, attack="evasion", miss=1.0))
        for key, r in res13.items():
            rows.append(dict(base, policy=key, attack="mimicry_reserve", miss=_miss(r, ps)))
        rows.append(dict(base, policy="b13_offline", attack="mimicry_reserve",
                         miss=float((1 - float(b13off.get("pi_reserve", 0.0))) ** k)))
        rows.append(dict(base, policy="b6_strict", attack="zero_day",
                         miss=float((1 - float(b6p.get("pi_leftover", 0.0))) ** k)))
    # CtS-AI rows of the rev0 E4 run (mimicry etc.), unchanged by the baselines
    for f in sorted(glob.glob(str(RESULTS / "E4" / sid / "*" / "incidents.parquet"))):
        if (Path(f).parent / "DONE").exists():
            t = pl.read_parquet(f).filter(pl.col("policy") == "cts_single")
            for r in t.iter_rows(named=True):
                if r["attack"] in ("mimicry_wb", "mimicry_bb", "mimicry_base"):
                    rows.append({"stream": sid, "incident": int(r["incident"]), "k": int(r["k"]), "phi0": float(r["phi0"]),
                                 "bound": float(r["bound"]), "policy": "cts_single", "attack": r["attack"],
                                 "miss": float(r["miss"]), "source": "rev0 E4 (CtS-AI, unchanged)"})
            break
    return rows


def main() -> int:
    import polars as pl
    t0 = time.time()
    allrows = []
    for sid in stream_list():
        if not (DESIGNS / sid / "REV1_BASELINES_DONE").exists():
            print(f"[e4r1] skip {sid}: rev1 baselines missing", flush=True)
            continue
        rows = run_stream(sid)
        o = OUT / sid
        o.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_parquet(o / "incidents.parquet")
        mark_done(o, {"n_rows": len(rows)})
        allrows += rows
        print(f"[e4r1] {sid}: {len(rows)} rows ({time.time() - t0:.0f}s)", flush=True)
    df = pl.DataFrame(allrows)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    df.write_parquet(SUMMARY / "E4_incidents_rev1.parquet")
    summ = {"definition": "per incident content-evidence miss at rho = 5% (analytic); median and quantiles over the "
                          "incidents of all LP streams; at_or_below_bound = share with miss <= (1 - phi0)^k (1e-9 slack); "
                          "B4/B13 online = per-epoch budget (budget-exact), *_ctl = horizon threshold controller with a "
                          "training theta_ref (spend not matched), *_offline = global top-k with hindsight (rev0)",
            "n_streams": int(df["stream"].n_unique()), "n_incidents": int(df.filter(pl.col("policy") == "dlt")["incident"].len()),
            "rows": {}}
    for (pol, att), g in df.group_by(["policy", "attack"]):
        m = g["miss"].to_numpy()
        b = g["bound"].to_numpy()
        summ["rows"][f"{pol}/{att}"] = {"n": int(len(m)), "median": float(np.median(m)), "q10": float(np.quantile(m, 0.1)),
                                        "q90": float(np.quantile(m, 0.9)), "mean": float(np.mean(m)),
                                        "at_or_below_bound": float(np.mean(m <= b * (1 + 1e-9) + 1e-300))}
    summ["rows"] = dict(sorted(summ["rows"].items()))
    spend = {}
    for sid in stream_list():
        f = DESIGNS / sid / "REV1_BASELINES_DONE"
        if f.exists():
            for r in json.loads(f.read_text())["summary"]:
                if abs(r["rho"] - RHO) < 1e-12:
                    spend.setdefault(r["design"], []).append(r["spend_over_target"])
    summ["spend_over_target_rho5"] = {k: [float(min(v)), float(np.median(v)), float(max(v))] for k, v in sorted(spend.items())}
    atomic_write_json(SUMMARY / "E4_rev1.json", summ)
    print(f"[e4r1] done {len(allrows)} rows in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
