"""Stream-recipe sensitivity (rev1, review R3-M1(a), fix-list M5; helper P1-X-SV): original vs recipe variants
(zq: the infected host's pure zero-payload connections thinned with the stream's payload q; z0: all removed) for
the E3 in-domain configuration (LP-IoT23, p = 1 %, 6 LOSO folds, 3 seeds, rho 2 % / 5 %).

    .venv/bin/python -I -u P1_CommitThenSample/scripts/checks/sv_recipe_compare.py [--variants zq,z0]

Reads only: results/summary/E3_numbers.json (original), results/summary/E3_numbers_rev1_<v>.json (variant; the zq
file also carries group in_domain_orig = the original streams re-run with the same code), the streams
(results/streams/<id>/), results/models{,_<v>}/<fold>/<seed>/auc.json. Writes
results/summary/E3_recipe_sensitivity_rev1.{json,md} (new files; nothing else is touched).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctsample.utils.io import RESULTS, atomic_write_json, atomic_write_text  # noqa: E402

SUMMARY = RESULTS / "summary"
STREAMS = RESULTS / "streams"
FOLDS = ["1-1", "3-1", "7-1", "34-1", "48-1", "49-1"]
BASE = {f: f"lp_iot23_{f}_p10000" for f in FOLDS}
DESIGNS = ["U0", "U3", "U5", "U5tau0.2", "U5tau0.5", "U6", "U6tau0.2", "U6tau0.5", "U8", "U7", "B12"]
ROLL = ["r_units", "r_rdst", "r_dports", "r_s0rej_q16", "r_rarity_q16"]
SEEDS = (0, 1, 2)


def _sid(f: str, v: str | None) -> str:
    return BASE[f] if not v else f"{BASE[f]}_{v}"


def _binom_tail(k: int, n: int) -> float:
    """P(X >= k), X ~ Bin(n, 1/2) (exact one-sided sign test)."""
    return float(sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n)


def _gm(x) -> float:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v) and v > 0], float)
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


# ------------------------------------------------------------------------------------------ #
def stream_table(variants) -> dict:
    out = {}
    for v in variants:
        rows = {}
        for f in FOLDS:
            d = STREAMS / _sid(f, v)
            if not (d / "DONE").exists():
                continue
            m = json.loads((d / "meta.json").read_text())
            mb = json.loads((STREAMS / BASE[f] / "meta.json").read_text())
            z = m["extra"]["zero_thin"]
            zp = z["parts"][0]
            rows[f] = {"stream_id": _sid(f, v), "q": m["extra"]["q"], "zero_keep_prob": z["keep_prob"],
                       "realised_keep_rate": z["realised_keep_rate"],
                       "seed_zero_units_before": zp["n_seed_zero_units_before"],
                       "seed_zero_units_after": zp["n_seed_zero_units_after"],
                       "pure_zero_conns_before": zp["n_pure_zero_conns_before"],
                       "pure_zero_conns_after": zp["n_pure_zero_conns_after"],
                       "zero_units_in_payload_conns": zp["n_zero_in_mal_payload_conns"] + zp["n_zero_in_other_payload_conns"],
                       "pure_zero_labels_before": zp["pure_zero_labels_before"],
                       "pure_zero_labels_after": zp["pure_zero_labels_after"],
                       "n_units_before": mb["n_units"], "n_units_after": m["n_units"],
                       "n_payload_units": m["extra"]["n_payload_units"],
                       "n_mal_payload_units": m["extra"]["n_mal_payload_units"],
                       "s_all_over_s_pay_before": mb["extra"]["s_all_over_s_pay"],
                       "s_all_over_s_pay_after": m["extra"]["s_all_over_s_pay"],
                       "budget_ratio_after_over_before": m["extra"]["sum_s_all"] / mb["extra"]["sum_s_all"]}
        out[v] = rows
    return out


# ------------------------------------------------------------------------------------------ #
def e3_table(variants) -> dict:
    orig = json.loads((SUMMARY / "E3_numbers.json").read_text())
    nums = {"orig": orig}
    for v in variants:
        p = SUMMARY / f"E3_numbers_rev1_{v}.json"
        if p.exists():
            nums[v] = json.loads(p.read_text())
    out = {}
    for rho in (0.02, 0.05):
        for des in DESIGNS:
            key = f"in_domain/rho{rho:g}/{des}"
            if key not in orig:
                continue
            row = {}

            def take(src, k, v):
                e = nums.get(src, {}).get(k)
                if e is None:
                    return None
                ps = json.loads(e["per_scenario"])
                ls = json.loads(e["lcb_per_scenario"]) if e.get("lcb_per_scenario") else {}
                return {"gm": e["gm"], "lo": e["lo"], "hi": e["hi"],
                        "per_scenario": {f: ps.get(_sid(f, v)) for f in FOLDS},
                        "lcb_gm": e.get("lcb_gm"), "lcb_lo": e.get("lcb_lo"), "lcb_hi": e.get("lcb_hi"),
                        "lcb_per_scenario": {f: ls.get(_sid(f, v)) for f in FOLDS} if ls else None,
                        "spend_ratio_mean": e.get("spend_ratio_mean"), "spend_ratio_max": e.get("spend_ratio_max"),
                        "share_exact": e.get("share_exact"), "n_pairs": e.get("n_pairs")}
            row["orig"] = take("orig", key, None)
            if "zq" in nums:
                row["control"] = take("zq", key.replace("in_domain/", "in_domain_orig/"), None)
            for v in variants:
                if v in nums:
                    row[v] = take(v, key, v)
            # exact per-scenario statements
            st = {}
            for src in ["orig"] + [v for v in variants if row.get(v)]:
                ps = row[src]["per_scenario"]
                vals = [ps[f] for f in FOLDS if ps.get(f) is not None]
                st[src] = {"n_below_U1": int(sum(x < 1 for x in vals)), "n": len(vals),
                           "max": max(vals) if vals else None, "min": min(vals) if vals else None}
                if row[src].get("lcb_per_scenario"):
                    lv = [row[src]["lcb_per_scenario"][f] for f in FOLDS if row[src]["lcb_per_scenario"].get(f) is not None]
                    st[src]["lcb_n_below_U1"] = int(sum(x < 1 for x in lv))
                if src != "orig":
                    po = row["orig"]["per_scenario"]
                    rr = {f: ps[f] / po[f] for f in FOLDS if ps.get(f) is not None and po.get(f)}
                    worse = sum(1 for x in rr.values() if x > 1 + 1e-12)
                    same = sum(1 for x in rr.values() if abs(x - 1) <= 1e-12)
                    st[src].update({"ratio_vs_orig": rr, "gm_ratio_vs_orig": _gm(rr.values()),
                                    "n_worse_than_orig": worse, "n_identical_to_orig": same,
                                    "sign_test_p_worse": _binom_tail(worse, len(rr) - same) if len(rr) - same > 0 else None})
            row["statements"] = st
            if row.get("control") and row["orig"]:
                po, pc = row["orig"]["per_scenario"], row["control"]["per_scenario"]
                row["control_max_abs_rel_diff"] = max(abs(pc[f] / po[f] - 1) for f in FOLDS if pc.get(f) and po.get(f))
            out[key] = row
    return out


def _agg_module():
    import importlib.util
    p = Path(__file__).resolve().parents[1] / "90_aggregate.py"
    spec = importlib.util.spec_from_file_location("sv_agg90", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def e3_vs_uniform(variants) -> dict:
    """Same statistic as the aggregator but with the uniform design U0 as reference (review R1-M6: the size-only U1
    reference moves with the committed floor phi0, which the scans' bytes inflate): per scenario
    sum_q MSE_d/Q^2 / sum_q Var_U0/Q^2 over the design's estimands (classes D+S, Q > 0; seed means), GM over scenarios,
    cluster bootstrap = the aggregator's _ratio_of_sums_boot (2,000 resamples, seed 0)."""
    import pandas as pd
    agg = _agg_module()
    srcs = [("orig", SUMMARY / "E3_estimands.parquet", "in_domain")]
    if (SUMMARY / "E3_estimands_rev1_zq.parquet").exists():
        srcs.append(("control", SUMMARY / "E3_estimands_rev1_zq.parquet", "in_domain_orig"))
    srcs += [(v, SUMMARY / f"E3_estimands_rev1_{v}.parquet", "in_domain") for v in variants
             if (SUMMARY / f"E3_estimands_rev1_{v}.parquet").exists()]
    out = {}
    for name, path, grp in srcs:
        A = pd.read_parquet(path)
        A = A[(A["group"] == grp) & A["cls"].isin(["D", "S"]) & (A["Q"] > 0)]
        for rho in (0.02, 0.05):
            G = A[np.isclose(A["rho"], rho)]
            u0 = G[G["design"] == "U0"].groupby(["stream", "estimand"]).agg(m0=("mse", "mean"), Q=("Q", "first"))
            u1 = G[G["design"] == "U0"].groupby(["stream", "estimand"]).agg(m=("var_U1", "first"))
            for des in ["U1"] + [d for d in DESIGNS if d != "U0"]:
                if des == "U1":
                    gd = u1
                else:
                    g = G[G["design"] == des]
                    if g.empty:
                        continue
                    gd = g.groupby(["stream", "estimand"]).agg(m=("mse", "mean"))
                j = gd.join(u0, how="inner")
                per_s = {}
                for sid, h in j.groupby(level=0):
                    f = sid.split("_")[2]
                    per_s[f] = ((h["m"] / h["Q"] ** 2).to_numpy(), (h["m0"] / h["Q"] ** 2).to_numpy())
                if not per_s:
                    continue
                bs = agg._ratio_of_sums_boot(per_s, B=2000, seed=0)
                ps = {f: float(np.sum(a) / np.sum(b)) for f, (a, b) in per_s.items()}
                out.setdefault(f"in_domain/rho{rho:g}/{des}", {})[name] = {
                    "gm": bs["point"], "lo": bs["lo"], "hi": bs["hi"], "per_scenario": ps,
                    "n_below_U0": int(sum(x < 1 for x in ps.values())), "n": len(ps)}
    return out


# ------------------------------------------------------------------------------------------ #
def _seed_mac_new(sid: str) -> str:
    m = json.loads((STREAMS / sid / "meta.json").read_text())
    return list(m["extra"]["recipe"]["foregrounds"][0]["mac_map"].values())[0]


def predictability(variants) -> dict:
    """AUC (P1-D auc.json: every known payload unit; and restricted to the IoT-23 foreground / the seed host) and the
    seed host's rolling-feature shift (KS original vs variant, same payload units)."""
    import polars as pl
    from scipy.stats import ks_2samp
    from ctsample.design.utility import auc
    from ctsample.sim.tables import StreamTables
    from ctsample.units.features import FEATURE_NAMES
    ji = [FEATURE_NAMES.index(n) for n in ROLL]
    out = {}
    for f in FOLDS:
        rows = {}
        X0 = None
        for v in [None] + list(variants):
            sid = _sid(f, v)
            d = STREAMS / sid
            if not (d / "DONE").exists():
                continue
            mroot = RESULTS / ("models" if not v else f"models_{v}")
            a_json = {}
            for s in SEEDS:
                p = mroot / f / str(s) / "auc.json"
                if p.exists():
                    a = json.loads(p.read_text()).get(sid, {})
                    for q, e in a.items():
                        if e.get("auc") is not None:
                            a_json.setdefault(q, []).append(float(e["auc"]))
            T = StreamTables.load(d)
            pay = np.asarray(T.pay, dtype=bool)
            pidx = np.nonzero(pay)[0]
            u = pl.read_parquet(d / "units.parquet", columns=["pay", "source", "device_mac"]).filter(pl.col("pay"))
            seedm = (u["device_mac"] == _seed_mac_new(sid)).to_numpy()
            fg = (u["source"] != "BenignTraffic1").to_numpy()
            lab = np.asarray(T.label)[pidx]
            known = lab >= 0
            mal = lab > 0
            sub = {}
            for s in SEEDS:
                k = f"U3_s{s}"
                if k not in T.utils:
                    continue
                g = np.asarray(T.util(k))[pidx]
                sub.setdefault("g_mal_fg", []).append(auc(g[known & fg], mal[known & fg]))
                sub.setdefault("g_mal_seed", []).append(auc(g[known & seedm], mal[known & seedm]))
                gm = g[seedm & mal]
                sub.setdefault("pmal_median_seed_mal", []).append(float(np.median(gm)) - 1.0 if len(gm) else None)
            X = np.load(d / "features_pay.npy")
            Xs = X[seedm][:, ji].astype(np.float64)
            bg = ~fg & known
            ident = {n: auc(np.concatenate([X[seedm][:, j], X[bg][:, j]]).astype(np.float64),
                            np.concatenate([np.ones(int(seedm.sum()), bool), np.zeros(int(bg.sum()), bool)]))
                     for n, j in zip(ROLL, ji)}
            r = {"stream_id": sid, "n_seed_payload_units": int(seedm.sum()),
                 "n_seed_mal_payload_units": int((seedm & mal).sum()),
                 "auc_json_mean": {q: float(np.mean(x)) for q, x in a_json.items()},
                 "auc_g_mal_foreground_only": _mean(sub.get("g_mal_fg")),
                 "auc_g_mal_seed_host_only": _mean(sub.get("g_mal_seed")),
                 "median_pmal_seed_mal_units_q16": _mean(sub.get("pmal_median_seed_mal")),
                 "seed_rolling_median": {n: float(np.median(Xs[:, i])) for i, n in enumerate(ROLL)},
                 "seed_rolling_mean": {n: float(np.mean(Xs[:, i])) for i, n in enumerate(ROLL)},
                 "auc_seed_vs_background_by_feature": ident}
            if v is None:
                X0 = Xs
            elif X0 is not None and len(X0) == len(Xs):
                r["ks_seed_rolling_vs_orig"] = {n: float(ks_2samp(X0[:, i], Xs[:, i]).statistic) for i, n in enumerate(ROLL)}
                r["share_seed_units_changed"] = {n: float(np.mean(X0[:, i] != Xs[:, i])) for i, n in enumerate(ROLL)}
            rows[v or "orig"] = r
        out[f] = rows
    return out


def _mean(x):
    x = [v for v in (x or []) if v is not None]
    return float(np.mean(x)) if x else None


# ------------------------------------------------------------------------------------------ #
def markdown(res: dict, variants) -> str:
    L = ["# E3 in-domain: stream-recipe sensitivity (R3-M1(a)) — generated by scripts/checks/sv_recipe_compare.py", ""]
    L.append("## Streams (p = 1 %): payload q, seed host's zero-payload units original -> variant")
    L.append("")
    L.append("| variant | scenario | q | zero keep | seed zero-payload units | pure zero conns | units | S_all/S_pay |")
    L.append("|---|---|---|---|---|---|---|---|")
    for v in variants:
        for f, r in res["streams"].get(v, {}).items():
            L.append(f"| {v} | {f} | {r['q']:.4f} | {r['zero_keep_prob']:.4f} | {r['seed_zero_units_before']:,} -> "
                     f"{r['seed_zero_units_after']:,} | {r['pure_zero_conns_before']:,} -> {r['pure_zero_conns_after']:,} | "
                     f"{r['n_units_before']:,} -> {r['n_units_after']:,} | {r['s_all_over_s_pay_before']:.3f} -> "
                     f"{r['s_all_over_s_pay_after']:.3f} |")
    for rho in (0.05, 0.02):
        L += ["", f"## Relative-MSE ratio vs U1 (classes D+S), rho = {rho:g}: per scenario, GM [95% cluster-bootstrap CI]", ""]
        hdr = "| design | src | " + " | ".join(FOLDS) + " | GM [CI] | n < U1 | worse than orig |"
        L += [hdr, "|" + "---|" * (len(FOLDS) + 5)]
        for des in DESIGNS:
            row = res["e3"].get(f"in_domain/rho{rho:g}/{des}")
            if not row:
                continue
            for src in ["orig", "control"] + list(variants):
                e = row.get(src)
                if not e:
                    continue
                st = row["statements"].get(src, {})
                cells = " | ".join("" if e["per_scenario"].get(f) is None else f"{e['per_scenario'][f]:.4g}" for f in FOLDS)
                w = "" if src in ("orig", "control") else f"{st.get('n_worse_than_orig')}/6"
                nb = st.get("n_below_U1", "")
                L.append(f"| {des} | {src} | {cells} | {e['gm']:.4g} [{e['lo']:.3g}, {e['hi']:.3g}] | "
                         f"{'' if nb == '' else str(nb) + '/6'} | {w} |")
    L += ["", "## Relative-MSE ratio vs U0 (uniform reference), rho = 5 %: per scenario, GM [CI]", ""]
    L += ["| design | src | " + " | ".join(FOLDS) + " | GM [CI] | n < U0 |", "|" + "---|" * (len(FOLDS) + 4)]
    for des in ["U1"] + [d for d in DESIGNS if d != "U0"]:
        row = res.get("e3_vs_U0", {}).get(f"in_domain/rho0.05/{des}")
        if not row:
            continue
        for src in ["orig", "control"] + list(variants):
            e = row.get(src)
            if not e:
                continue
            cells = " | ".join("" if e["per_scenario"].get(f) is None else f"{e['per_scenario'][f]:.4g}" for f in FOLDS)
            L.append(f"| {des} | {src} | {cells} | {e['gm']:.4g} [{e['lo']:.3g}, {e['hi']:.3g}] | {e['n_below_U0']}/{e['n']} |")
    L += ["", "## LCB-gap ratio vs U1 (rho = 5 %): per scenario, GM [CI]", ""]
    L += ["| design | src | " + " | ".join(FOLDS) + " | GM [CI] |", "|" + "---|" * (len(FOLDS) + 3)]
    for des in DESIGNS:
        row = res["e3"].get(f"in_domain/rho0.05/{des}")
        if not row:
            continue
        for src in ["orig"] + list(variants):
            e = row.get(src)
            if not e or not e.get("lcb_per_scenario"):
                continue
            cells = " | ".join("" if e["lcb_per_scenario"].get(f) is None else f"{e['lcb_per_scenario'][f]:.3g}" for f in FOLDS)
            L.append(f"| {des} | {src} | {cells} | {e['lcb_gm']:.3g} [{e['lcb_lo']:.3g}, {e['lcb_hi']:.3g}] |")
    L += ["", "## Metadata predictability and the seed host's rolling features", ""]
    L += ["| scenario | src | AUC g_mal (all) | AUC L1 | AUC g_mal (IoT-23 fg) | AUC g_mal (seed host) | "
          "seed-vs-CIC AUC r_rdst / r_s0rej / r_units | KS vs orig r_units / r_rdst / r_s0rej |",
          "|---|---|---|---|---|---|---|---|"]

    def fm(x, p=3):
        return "" if x is None else f"{x:.{p}f}"
    for f, rows in res["predictability"].items():
        for src, r in rows.items():
            a = r["auc_json_mean"]
            idn = r["auc_seed_vs_background_by_feature"]
            ks = r.get("ks_seed_rolling_vs_orig") or {}
            L.append(f"| {f} | {src} | {fm(a.get('g_mal'), 4)} | {fm(a.get('L1'), 4)} | {fm(r['auc_g_mal_foreground_only'])} | "
                     f"{fm(r['auc_g_mal_seed_host_only'])} | {fm(idn['r_rdst'])} / {fm(idn['r_s0rej_q16'])} / "
                     f"{fm(idn['r_units'])} | {fm(ks.get('r_units'))} / {fm(ks.get('r_rdst'))} / {fm(ks.get('r_s0rej_q16'))} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="zq,z0")
    a = ap.parse_args()
    variants = [v for v in a.variants.split(",") if (STREAMS / f"index_{v}.json").exists()]
    res = {"definitions": {
        "variants": {"zq": "original recipe + the seed (infected) host's pure zero-payload connections kept with the "
                           "stream's payload keep probability q (independent child seed stream; payload decisions "
                           "bit-identical); training LP streams and model training sources by the same rule",
                     "z0": "scan-free bound: every pure zero-payload connection of the seed host removed (training LP "
                           "streams likewise; model training sources: each IoT-23 malware scenario's seed host likewise; "
                           "models retrained)"},
        "relmse_ratio": "as E3_numbers.json: per scenario sum_q Var_d/Q^2 / sum_q Var_U1/Q^2 over classes D+S, Q > 0; GM over "
                        "the 6 scenarios with the aggregator's cluster bootstrap (2,000 resamples, estimands nested)",
        "statements": "n_below_U1 = scenarios with ratio < 1; ratio_vs_orig = variant / original per scenario; "
                      "sign_test_p_worse = exact one-sided binomial P(X >= #worse | n = #non-identical, 1/2)",
        "control": "group in_domain_orig of E3_numbers_rev1_zq.json: the original streams re-run with the code of the variant "
                   "run (must equal the original numbers)",
        "e3_vs_U0": "the relmse statistic with the uniform design U0 as reference instead of U1 (U1 = size-only moves with "
                    "the committed floor phi0, which the scans' bytes inflate in the original recipe): per scenario "
                    "sum_q MSE_d/Q^2 / sum_q Var_U0/Q^2 over the design's estimands, GM + the aggregator's cluster bootstrap",
        "auc": "auc_json_mean: P1-D's auc.json (every known payload unit, mean over 3 seeds); foreground / seed host: AUC of "
               "the U3 utility (= P_mal + 1) among the IoT-23 foreground / the seed host's known payload units",
        "seed_vs_background_auc": "AUC of one rolling feature separating the seed host's payload units from the CIC "
                                  "background's (0.5 = not identifiable by that feature alone)",
        "ks": "two-sample KS statistic of the seed host's payload-unit rolling features, original vs variant stream (same units)"}}
    res["streams"] = stream_table(variants)
    res["e3"] = e3_table(variants)
    res["e3_vs_U0"] = e3_vs_uniform(variants)
    res["predictability"] = predictability(variants)
    atomic_write_json(SUMMARY / "E3_recipe_sensitivity_rev1.json", res)
    atomic_write_text(SUMMARY / "E3_recipe_sensitivity_rev1.md", markdown(res, variants))
    print(f"[sv] wrote {SUMMARY / 'E3_recipe_sensitivity_rev1.json'} (variants {variants})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
