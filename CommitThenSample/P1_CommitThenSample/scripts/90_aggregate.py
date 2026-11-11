"""Aggregate E1/E2/E3 run directories into summaries and the machine-readable numbers files (DESIGN §6.6, §7.2).

    .venv/bin/python -I -u P1_CommitThenSample/scripts/90_aggregate.py --exp E2      # or E1, E3, all

Outputs (results/summary/):
  E2_summary.parquet      one row per (run = part x stream x ρ x arm, variant, estimand): miss counts and Wilson
                          95% intervals of every flag (time-uniform 'tu' / final-epoch 'ft', per side; two-sided
                          unions per rep), widths at the last report epoch (median/IQR over the first 200 reps),
                          UCB slack vs the hiding capacity H_{α/2} (Thm. 1), estimand metadata (kind, class, Q,
                          stratum, eligible units, V_ref source)
  E2_runs.parquet         one row per run: controller statistics (spend/target, max |D|/B̄, boosted share),
                          family-wise miscoverage of the stream's report (any statement wrong, both sides, any time)
  E2_numbers.json         every number the paper quotes, each with its definition
  E2_per_estimand.csv     supplement table (ρ = 5%, honest variant)
If several complete run dirs exist for one job key (code changed between runs: cfg_hash covers src/), the newest
is used and the others are listed in the numbers file.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctsample.sim import experiments_core as X  # noqa: E402
from ctsample.sim.metrics import wilson  # noqa: E402
from ctsample.utils.io import RESULTS, atomic_write_json, atomic_write_parquet  # noqa: E402

SUMMARY = RESULTS / "summary"
ALPHA = 0.05
REV = ""      # --rev rev1: read results/<exp>_rev1/, write results/summary/<name>_rev1.<ext> (old outputs kept)


def _sum(name: str) -> Path:
    """results/summary/<stem><REV><ext> (the summary files of the selected revision)."""
    p = Path(name)
    return SUMMARY / f"{p.stem}{REV}{p.suffix}"
FLAGS = ("lcs_tu", "ucs_tu", "lcs_ft", "ucs_ft", "wald_lo_ft", "wald_hi_ft", "wald_lo_tu", "wald_hi_tu")


def _read_parts(d: Path, kind: str) -> pd.DataFrame:
    fs = sorted(d.glob(f"rep*_{kind}.parquet"))
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in fs], ignore_index=True)


def _runs(exp: str) -> tuple[list[tuple[Path, dict]], list[str]]:
    """Complete run dirs of an experiment, newest per job key."""
    best: dict = {}
    superseded = []
    for d in sorted((RESULTS / (exp + REV)).glob("*/*")):
        if not (d / "DONE").exists() or not (d / "config.json").exists():
            continue
        c = json.loads((d / "config.json").read_text())
        key = json.dumps(c["job"], sort_keys=True)
        if key in best:
            if c["created"] > best[key][1]["created"]:
                superseded.append(str(best[key][0]))
                best[key] = (d, c)
            else:
                superseded.append(str(d))
        else:
            best[key] = (d, c)
    return list(best.values()), superseded


def _wil(k, n):
    p, lo, hi = wilson(k, n)
    return float(p), float(lo), float(hi)


# --------------------------------------------------------------------------------------------- #
# E2
# --------------------------------------------------------------------------------------------- #
def e2_run_rows(d: Path, c: dict) -> tuple[list[dict], dict]:
    job = c["job"]
    cov = _read_parts(d, "cov")
    width = _read_parts(d, "width")
    ctrl = _read_parts(d, "ctrl")
    erows = {r["estimand"]: r for r in c["estimands"]}
    # hiding capacity H_{alpha/2} of indicator estimands under the static reference pi (exact for static arms)
    H = {}
    try:
        T = X.load_tables(job["stream"])
        src = c["design_source"]["file"].split("/")[-1].rsplit("_", 1)[0]
        pi = X.design_pi(job["stream"], src, job["rho"])
        fam = int(c.get("family_size", 1))
        H = X.hiding_capacity_rows(T, pi, list(erows), alpha_side=ALPHA / (2 * fam))
    except Exception as e:  # noqa: BLE001
        print(f"[agg] hiding capacity failed for {d}: {e}", flush=True)
    # last report epoch per (rep, estimand, variant)
    wl = None
    if len(width):
        idx = width.groupby(["rep", "estimand", "variant"])["epoch"].idxmax()
        wl = width.loc[idx]
    rows = []
    base = {"run": str(d.relative_to(RESULTS)), "part": job["part"], "stream": job["stream"], "kind": job["kind"],
            "bin": job["bin"], "fold": job["fold"], "realised_prevalence": job.get("realised_prevalence"),
            "rho": job["rho"], "arm": job["dkey"], "family_size": int(c.get("family_size", 1))}
    for (var, est), g in cov.groupby(["variant", "estimand"], sort=True):
        er = erows.get(est, {})
        n = len(g)
        r = dict(base, variant=var, estimand=est, kind_q=er.get("kind"), cls=er.get("class"), n_elig=er.get("n_elig"),
                 n_hits=er.get("n_hits"), Q_stream=er.get("Q"), stratum=er.get("stratum"),
                 V_ref_source=er.get("V_ref_source"), n=n, Q_mean=float(g["Q"].mean()))
        for fl in FLAGS:
            k = int(g[fl].astype(bool).sum())
            p, lo, hi = _wil(k, n)
            r[f"{fl}_k"], r[fl], r[f"{fl}_lo"], r[f"{fl}_hi"] = k, p, lo, hi
        unions = {"ours_tu": g["lcs_tu"].astype(bool) | g["ucs_tu"].astype(bool),
                  "ours_ft": g["lcs_ft"].astype(bool) | g["ucs_ft"].astype(bool),
                  "wald_ft": g["wald_lo_ft"].astype(bool) | g["wald_hi_ft"].astype(bool),
                  "wald_tu": g["wald_lo_tu"].astype(bool) | g["wald_hi_tu"].astype(bool)}
        for nm, v in unions.items():
            k = int(v.sum())
            p, lo, hi = _wil(k, n)
            r[f"{nm}_k"], r[nm], r[f"{nm}_lo"], r[f"{nm}_hi"] = k, p, lo, hi
        r["ucs_issued_rate"] = float(g["ucs_issued"].astype(bool).mean())
        r["final_S_mean"] = float(g["S"].mean())
        if wl is not None:
            w = wl[(wl["variant"] == var) & (wl["estimand"] == est)]
            if len(w):
                Q = w["Q"].to_numpy(float)
                ok = Q > 0
                gap = (Q[ok] - w["Lplus"].to_numpy(float)[ok]) / Q[ok]
                gain = (w["Lstar"].to_numpy(float)[ok] - w["S"].to_numpy(float)[ok]) / Q[ok]
                gap_S = (Q[ok] - w["S"].to_numpy(float)[ok]) / Q[ok]
                gap_L = np.minimum((Q[ok] - w["Lstar"].to_numpy(float)[ok]) / Q[ok], 1.0)   # L* < 0 -> gap 1
                slack = w["Uplus"].to_numpy(float) - Q
                whalf = ((w["wald_hi"] - w["wald_lo"]).to_numpy(float) / 2)[ok] / Q[ok]
                for nm, arr in (("rel_lcb_gap", gap), ("stat_gain", gain), ("ucb_slack", slack), ("wald_rel_hw", whalf),
                                ("gap_S", gap_S), ("gap_Lstar", gap_L)):
                    arr = arr[np.isfinite(arr)]
                    if len(arr):
                        q = np.quantile(arr, [0.25, 0.5, 0.75])
                        r[f"{nm}_q25"], r[f"{nm}_med"], r[f"{nm}_q75"] = map(float, q)
                    else:
                        r[f"{nm}_q25"] = r[f"{nm}_med"] = r[f"{nm}_q75"] = float("nan")
                r["share_Lstar_gt_S"] = float(np.mean(w["Lstar"].to_numpy(float) > w["S"].to_numpy(float)))
                r["Lplus_med"] = float(np.median(w["Lplus"].to_numpy(float)))
                r["Uplus_med"] = float(np.nanmedian(w["Uplus"].to_numpy(float))) if np.isfinite(w["Uplus"].to_numpy(float)).any() else float("nan")
                r["n_width"] = int(len(w))
        if "ft1_lo" in g:   # rev1: valid fixed-time comparators (MCSpec.fixed_time; helper P1-X-E2)
            wft = wl[(wl["variant"] == var) & (wl["estimand"] == est)] if wl is not None else None
            r.update(_e2_ft_pair(g, wft, er))
        h = H.get(est)
        r["H_cap"] = h
        r["slack_over_H"] = (r.get("ucb_slack_med", float("nan")) / h) if (h and h > 0) else float("nan")
        rows.append(r)
    # run-level: controller and family-wise
    run = dict(base, n_reps=int(cov["rep"].nunique()) if len(cov) else 0, n_estimands=len(erows),
               static_repro_max_abs_diff=c.get("static_repro_max_abs_diff"))
    if len(ctrl):
        for col in ("spend_ratio", "max_abs_D_over_B", "boosted_frac", "retained_hits"):
            v = ctrl[col].to_numpy(float)
            run[f"{col}_mean"] = float(v.mean())
            run[f"{col}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            run[f"{col}_p95"] = float(np.quantile(v, 0.95))
            run[f"{col}_max"] = float(v.max())
    for var, g in cov.groupby("variant"):
        anym = (g["lcs_tu"].astype(bool) | g["ucs_tu"].astype(bool)).groupby(g["rep"]).any()
        p, lo, hi = _wil(int(anym.sum()), len(anym))
        run[f"fw_{var}"], run[f"fw_{var}_lo"], run[f"fw_{var}_hi"] = p, lo, hi
        if "ft1_lo" in g:   # rev1: family-wise miss of the fixed-time statements at the final epoch
            for tag in ("ft1", "fth"):
                anyf = (g[f"{tag}_lo"].astype(bool) | g[f"{tag}_hi"].astype(bool)).groupby(g["rep"]).any()
                p, lo, hi = _wil(int(anyf.sum()), len(anyf))
                run[f"fw_{tag}_{var}"], run[f"fw_{tag}_{var}_lo"], run[f"fw_{tag}_{var}_hi"] = p, lo, hi
    return rows, run


def aggregate_e2() -> dict:
    runs, superseded = _runs("E2")
    rows, rrows = [], []
    t0 = time.time()
    for i, (d, c) in enumerate(runs):
        r, rr = e2_run_rows(d, c)
        rows += r
        rrows.append(rr)
    S = pd.DataFrame(rows)
    R = pd.DataFrame(rrows)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(_sum("E2_summary.parquet"), _to_arrow(S))
    atomic_write_parquet(_sum("E2_runs.parquet"), _to_arrow(R))
    nums = e2_numbers(S, R)
    ft = "ft1_lo_k" in S
    if ft:      # rev1 runs (results/E2_rev1): valid fixed-time comparators
        ft_full = e2_fixed_time_numbers(S, R)
        nums["fixed_time"] = e2_fixed_time_slim(ft_full)
    nums["_meta"] = {"runs": len(runs), "superseded_run_dirs": superseded, "seconds": time.time() - t0,
                     "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    atomic_write_json(_sum("E2_numbers.json"), nums)
    e2_per_estimand_table(S)
    if ft:
        e2_fixed_time_table(ft_full)
    return nums


def _to_arrow(df: pd.DataFrame):
    import pyarrow as pa
    return pa.Table.from_pandas(df, preserve_index=False)


def _num(value, definition: str, **extra) -> dict:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        value = None
    return {"value": value, "definition": definition, **extra}


def _pool(g: pd.DataFrame, flag: str) -> tuple[float, float, float, int, int]:
    k = int(g[f"{flag}_k"].sum())
    n = int(g["n"].sum())
    p, lo, hi = _wil(k, n)
    return p, lo, hi, k, n


def e2_numbers(S: pd.DataFrame, R: pd.DataFrame) -> dict:
    out: dict = {}
    if S.empty:
        return out
    a2 = ALPHA / 2
    H = S[(S["variant"] == "honest") & (S["part"] == "main")]
    out["definitions"] = {
        "pair": "(stream, estimand) with >= 50 eligible units (DESIGN §5.4), whole-stream window, one design arm and rho",
        "tu": "time-uniform: the statement is wrong at SOME epoch (LCS: L_e > Q_e; UCS: U_e < Q_e), checked at the truth "
              "by the K-point mixture e-processes (DESIGN §6.5)",
        "ft": "fixed time: wrong at the final epoch of the window",
        "ours_two_sided": "LCS lower side OR UCS upper side (UCS for indicator estimands only; count estimands: L+ only)",
        "wald": "Qhat +- z_{1-alpha/2} sqrt(Vhat), Vhat = sum Z f^2 (1-pi)/pi^2 (DESIGN §3.9); 'naive monitoring' = tu",
        "wilson": "Wilson 95% interval of the miss proportion",
        "streams_main": "LP-IoT23 (13 distinct streams: 6 folds x realised 0.1%/~1% + 7-1 at 4.96%) and LP-Edge (3)",
        "alpha": ALPHA, "alpha_per_side": a2}
    for kind in ("lp_iot23", "lp_edge", "natural", "lp_gotham"):
        Hk = H[H["kind"] == kind]
        if Hk.empty:
            continue
        blk = {}
        for arm in sorted(Hk["arm"].unique()):
            Ha = Hk[Hk["arm"] == arm]
            ab = {}
            for side, fl in (("lower", "lcs_tu"), ("upper", "ucs_tu")):
                Hs = Ha if side == "lower" else Ha[Ha["kind_q"] == "indicator"]
                i = Hs[fl].idxmax()
                ab[f"max_tu_miss_{side}"] = _num(float(Hs.loc[i, fl]), f"max over pairs and rho of the time-uniform {side}-side "
                                                 f"miscoverage (nominal <= {a2})", wilson_hi=float(Hs.loc[i, f'{fl}_hi']),
                                                 at={"stream": Hs.loc[i, "stream"], "estimand": Hs.loc[i, "estimand"],
                                                     "rho": float(Hs.loc[i, "rho"]), "n": int(Hs.loc[i, "n"])})
                p, lo, hi, k, n = _pool(Hs, fl)
                ab[f"pooled_tu_miss_{side}"] = _num(p, f"pooled time-uniform {side}-side miscoverage over all pairs and rho",
                                                    wilson=[lo, hi], k=k, n=n)
                ab[f"n_pairs_above_nominal_{side}"] = _num(int((Hs[fl] > a2).sum()), f"pairs with {side}-side tu miss > alpha/2")
                ab[f"n_pairs_signif_above_{side}"] = _num(int((Hs[f"{fl}_lo"] > a2).sum()),
                                                          f"pairs whose Wilson lower bound exceeds alpha/2 ({side})")
                ab["n_pairs"] = _num(int(len(Ha)), "evaluated (stream, estimand, rho) pairs")
            for st in ("Q1-20", "Q>20"):
                Hq = Ha[Ha["stratum"] == st]
                if Hq.empty:
                    continue
                for fl, nm in (("wald_ft", "Wald fixed-time two-sided"), ("wald_tu", "Wald naive monitoring two-sided"),
                               ("ours_ft", "CtS fixed-time two-sided"), ("ours_tu", "CtS time-uniform two-sided")):
                    p, lo, hi, k, n = _pool(Hq, fl)
                    ab[f"{fl}_{st}_pooled"] = _num(p, f"{nm} miscoverage pooled over {st} pairs (all rho)", wilson=[lo, hi], k=k, n=n)
                    ab[f"{fl}_{st}_max"] = _num(float(Hq[fl].max()), f"max over {st} pairs of the {nm} miscoverage")
                    ab[f"{fl}_{st}_share_gt_alpha"] = _num(float((Hq[f"{fl}_lo"] > ALPHA).mean()),
                                                           f"share of {st} pairs whose {nm} miscoverage is significantly > alpha "
                                                           f"(Wilson lower bound > {ALPHA})")
                for rho in sorted(Hq["rho"].unique()):
                    Hr = Hq[Hq["rho"] == rho]
                    p, lo, hi, k, n = _pool(Hr, "wald_ft")
                    ab[f"wald_ft_{st}_rho{rho:g}"] = _num(p, f"Wald fixed-time two-sided miscoverage pooled over {st} pairs at rho={rho:g}",
                                                          wilson=[lo, hi], k=k, n=n)
                    p, lo, hi, k, n = _pool(Hr, "ours_ft")
                    ab[f"ours_ft_{st}_rho{rho:g}"] = _num(p, f"CtS fixed-time two-sided miscoverage pooled over {st} pairs at rho={rho:g}",
                                                          wilson=[lo, hi], k=k, n=n)
                    ab[f"rel_lcb_gap_{st}_rho{rho:g}_median"] = _num(float(Hr["rel_lcb_gap_med"].median()),
                                                                    f"median over {st} pairs of the per-pair median (Q - L+)/Q "
                                                                    f"at the final report epoch, rho={rho:g}")
                    ab[f"stat_gain_{st}_rho{rho:g}_median"] = _num(float(Hr["stat_gain_med"].median()),
                                                                   f"median over {st} pairs of the per-pair median (L* - S)/Q, rho={rho:g}")
                    ab[f"share_Lstar_gt_S_{st}_rho{rho:g}"] = _num(float((Hr["share_Lstar_gt_S"] >= 0.5).mean()),
                                                                   f"share of {st} pairs where L* > S in at least half of the "
                                                                   f"width reps (the CS says more than the retained count), rho={rho:g}")
            Hz = Ha[(Ha["stratum"] == "Q=0") & (Ha["kind_q"] == "indicator")]
            if len(Hz):
                ab["null_slack_over_H_median"] = _num(float(Hz["slack_over_H"].median()),
                                                      "median over Q=0 indicator pairs of median(U+ - Q)/H_{alpha/2} "
                                                      "(Thm. 1 hiding capacity of the eligible units; Cor. 1 predicts ~1.19)")
                ab["null_slack_over_H_range"] = _num([float(Hz["slack_over_H"].min()), float(Hz["slack_over_H"].max())],
                                                     "range over Q=0 indicator pairs")
            blk[arm] = ab
        out[kind] = blk
    # controller / spend of the arms
    if not R.empty:
        Rm = R[R["part"] == "main"]
        sp = {}
        for (kind, arm), g in Rm.groupby(["kind", "arm"]):
            sp[f"{kind}/{arm}"] = _num([float(g["spend_ratio_mean"].min()), float(g["spend_ratio_mean"].median()),
                                        float(g["spend_ratio_mean"].max())],
                                       "[min, median, max] over runs of the mean spend/target (realised for adaptive)",
                                       max_single_rep=float(g["spend_ratio_max"].max()),
                                       boosted_frac_mean=float(g["boosted_frac_mean"].mean()))
        out["spend"] = sp
        fw = {}
        for (part, kind, arm), g in R.groupby(["part", "kind", "arm"]):
            if "fw_honest" not in g:
                continue
            fw[f"{part}/{kind}/{arm}"] = _num([float(g["fw_honest"].min()), float(g["fw_honest"].max())],
                                              "[min, max] over runs of the family-wise miscoverage of the stream's report "
                                              "(any statement wrong, either side, at any time); part familywise uses alpha/|F| "
                                              "per statement (Bonferroni target <= alpha)",
                                              n_estimands=[int(g["n_estimands"].min()), int(g["n_estimands"].max())])
        out["familywise"] = fw
    # A11
    A = S[S["part"] == "a11"]
    if len(A):
        a11 = {}
        for (kind, arm), g in A.groupby(["kind", "arm"]):
            blk = {}
            Qp = g[g["Q_stream"] > 0]
            for var in sorted(Qp["variant"].unique()):
                v = Qp[Qp["variant"] == var]
                for side, fl in (("lower", "lcs_tu"), ("upper", "ucs_tu")):
                    vv = v if side == "lower" else v[v["kind_q"] == "indicator"]
                    if vv.empty:
                        continue
                    p, lo, hi, k, n = _pool(vv, fl)
                    blk[f"{var}/{side}_pooled"] = _num(p, f"{side}-side time-uniform miscoverage pooled over Q>0 pairs",
                                                       wilson=[lo, hi], k=k, n=n, max=float(vv[fl].max()))
                blk[f"{var}/ucs_issued_rate"] = _num(float(v[v["kind_q"] == "indicator"]["ucs_issued_rate"].mean()),
                                                     "mean share of reps with a UCS issued at the final epoch (indicator pairs)")
            a11[f"{kind}/{arm}"] = blk
        out["a11"] = a11
    Rc = S[(S["part"] == "reactor") & (S["variant"] == "honest")]
    if len(Rc):
        rc = {}
        for (kind, arm), g in Rc.groupby(["kind", "arm"]):
            rc[f"{kind}/{arm}"] = {"max_lcs_tu": _num(float(g["lcs_tu"].max()), "max lower-side tu miss under the reactive adversary"),
                                   "max_ucs_tu": _num(float(g[g["kind_q"] == "indicator"]["ucs_tu"].max()),
                                                      "max upper-side tu miss under the reactive adversary"),
                                   "wald_ft_pooled_Qpos": _num(_pool(g[g["Q_stream"] > 0], "wald_ft")[0],
                                                               "Wald fixed-time two-sided miscoverage pooled over Q>0 pairs"),
                                   "n_pairs": _num(int(len(g)), "pairs")}
        out["reactor"] = rc
    return out


def e2_per_estimand_table(S: pd.DataFrame) -> None:
    H = S[(S["variant"] == "honest") & (S["part"] == "main") & (np.isclose(S["rho"], 0.05))]
    if H.empty:
        return
    cols = ["kind", "stream", "bin", "arm", "estimand", "cls", "kind_q", "n_elig", "Q_stream", "stratum", "n", "lcs_tu",
            "lcs_tu_hi", "ucs_tu", "ucs_tu_hi", "ours_ft", "wald_ft", "wald_ft_lo", "wald_ft_hi", "wald_tu",
            "rel_lcb_gap_med", "stat_gain_med", "share_Lstar_gt_S", "ucb_slack_med", "H_cap", "slack_over_H", "V_ref_source"]
    t = H[[c for c in cols if c in H]].sort_values(["kind", "stream", "arm", "estimand"])
    SUMMARY.mkdir(parents=True, exist_ok=True)
    t.to_csv(_sum("E2_per_estimand.csv"), index=False, float_format="%.4g")
    # supplement longtable (LP-IoT23 + LP-Edge, rho = 5%): static / adaptive side by side
    M = H[H["kind"].isin(["lp_iot23", "lp_edge"]) & H["arm"].isin(["static", "adaptive"])]
    if M.empty:
        return
    piv = M.pivot_table(index=["kind", "stream", "estimand"], columns="arm",
                        values=["lcs_tu", "ucs_tu", "wald_ft", "rel_lcb_gap_med"], aggfunc="first")
    info = M.drop_duplicates(["stream", "estimand"]).set_index(["kind", "stream", "estimand"])[["Q_stream", "n_elig", "kind_q"]]

    def f3(v):
        return "--" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.3f}"

    def short(sid):
        for a, b in (("_p50000", " 5\\%"), ("_p10000", " 1\\%"), ("_p1000", " 0.1\\%")):
            sid = sid.replace(a, b)
        return sid.replace("lp_iot23_", "IoT23 ").replace("lp_edge", "Edge")

    lines = ["% generated by scripts/90_aggregate.py (E2 per-estimand table, rho = 5%, R = 1,000 LP-IoT23 / 500 LP-Edge)",
             "\\begin{longtable}{@{}l l r r r r r r r r r@{}}",
             "\\caption{E2 per estimand at $\\rho=5\\%$: time-uniform miscoverage of the LCS (lower side) and UCS (upper "
             "side, indicators), fixed-time Wald miscoverage (two-sided) and median relative lower-bound gap $(Q-L^+)/Q$, "
             "static (s) and adaptive (a) designs; nominal $\\alpha/2=0.025$ per side.}\\\\",
             "\\toprule Stream & Estimand & $Q$ & Elig. & LCS s & LCS a & UCS s & UCS a & Wald s & Wald a & Gap s\\\\ \\midrule",
             "\\endhead"]
    for idx, r in piv.iterrows():
        inf = info.loc[idx]
        ucs_s = r[("ucs_tu", "static")] if inf["kind_q"] == "indicator" else float("nan")
        ucs_a = r[("ucs_tu", "adaptive")] if inf["kind_q"] == "indicator" else float("nan")
        lines.append(f"{short(idx[1])} & {idx[2]} & {inf['Q_stream']:.0f} & {inf['n_elig']} & {f3(r[('lcs_tu', 'static')])} & "
                     f"{f3(r[('lcs_tu', 'adaptive')])} & {f3(ucs_s)} & {f3(ucs_a)} & {f3(r[('wald_ft', 'static')])} & "
                     f"{f3(r[('wald_ft', 'adaptive')])} & {f3(r[('rel_lcb_gap_med', 'static')])}\\\\")
    lines += ["\\bottomrule", "\\end{longtable}"]
    (_sum("E2_per_estimand.tex")).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------------- #
# E2 rev1: valid fixed-time comparators (review R1-M2/R1-M3, fix list M2; helper P1-X-E2, notes/P1-X-E2.md)
# --------------------------------------------------------------------------------------------- #
FT_TAGS = ("ft1", "fth", "ftb", "ftx")
FT_LOWER = ("anytime", "mixE") + FT_TAGS
FT_LABEL = {
    "anytime": "anytime-valid statements as reported: L* = running max over report epochs (every 30 epochs + the last) "
               "of the K = 16 mixture LCB, L+ = max(L*, S); U = the UCS mixture at the final epoch, U+ = min(U, Dcap); "
               "miss = wrong at SOME epoch (flags lcs_tu / ucs_tu)",
    "mixE": "the same K = 16 / UCS mixtures evaluated ONCE at the final epoch (L_E without running max): their fixed-time "
            "use (Markov and Ville share the threshold 1/alpha_side); miss flags lcs_ft / ucs_ft",
    "ft1": "fixed-time single-eta Chernoff LCB L = Qhat - l/eta - eta V/2 with eta = eta*(V_ref) = sqrt(2 l/V_ref) (the LCS "
           "grid centre) and Fan UCB U = Qhat + (l + W(eta_U))/eta_U with eta_U = min(0.999/cbar, eta*); l = log(1/alpha_side)",
    "fth": "as ft1 with the centre tuned to the evaluation horizon: V_ref_ft = V_ref x (window epochs)/2880 for "
           "training-sourced V_ref (fallback bbar^2/phi0 unchanged)",
    "ftb": "fixed-time Bernstein with the floor-based design-variance bound kappa Q, kappa = bbar (1-phi)/phi: "
           "L_B = (sqrt(kappa l/2 + Qhat - kappa l/3) - sqrt(kappa l/2))^2 (0 if Qhat <= kappa l/3), "
           "U_B = (sqrt(kappa l/2) + sqrt(kappa l/2 + Qhat + bbar l/3))^2; populations fixed in advance (not part reactor)",
    "ftx": "exact worst-case Poisson-binomial bounds of the retained-hit count K = S under the least favourable placement "
           "of the hits among the eligible units (static designs only, indicator estimands with <= 1000 hits)",
    "wald": "Wald Qhat -+ z_{1-alpha/2} sqrt(Vhat) at the final epoch (not valid; Vhat = 0 without a retained hit)"}
FT_STRATA = (("Q1-5", 1, 5), ("Q6-20", 6, 20), ("Q1-20", 1, 20), ("Q>20", 21, np.inf), ("Q>0", 1, np.inf),
             ("Q=0", 0, 0), ("all", 0, np.inf))
FT_WIDTH_DEFS = {
    "rel_gap": "(Q - max(L, S))/Q at the final epoch = the reported lower statement; per pair the median over the first 200 "
               "reps, then median [IQR] over pairs (Q > 0); 'anytime_same_pairs' = the anytime value on the same pairs when a "
               "comparator exists only for a subset (ftx; ftb in part reactor)",
    "rel_gap_raw": "(Q - L)/Q clipped at 1 (the bound alone, without S)",
    "lower_radius_ratio": "(Qhat - L)/(Qhat - L_E) per rep, L_E = the anytime K = 16 mixture at the final epoch with the same "
                          "V_E: the width price of the K-point mixture vs ONE pre-registered eta (Chernoff-type bounds only); "
                          "'runmax' = (Qhat - L*)/(Qhat - L_E) <= 1, the gain of the running max (only where L_E < 0)",
    "lower_radius_vs_wald": "(Qhat - L)/(z_{1-alpha/2} sqrt(Vhat)) per rep with Vhat > 0",
    "upper_radius_ratio": "(U - Qhat)/(U_UCS - Qhat) per issued rep, indicator pairs (U_UCS = anytime UCS at the final epoch)",
    "upper_radius_vs_wald": "(U - Qhat)/(z sqrt(Vhat)) per issued rep with Vhat > 0, indicator pairs",
    "two_sided_ratio": "(U - L)/(U_UCS - L_E), raw bounds of the Chernoff/Fan type, indicator pairs",
    "two_sided_capped_ratio": "(min(U, Dcap) - max(L, S))/(U+ - L+) with the reported anytime U+ = min(U_UCS, Dcap), "
                              "L+ = max(L*, S); indicator pairs, reps with U+ > L+",
    "share_L_gt_S": "L > S (the bound says more than the count of retained hits): per_rep pooled over (pair, width rep); "
                    "pairs_ge_half = share of pairs with L > S in >= 50% of their width reps (rev0 rule of "
                    "share_Lstar_gt_S_*); pairs_any = in at least one rep",
    "share_U_lt_Dcap": "U < Dcap (the upper statement beats the sure cap): per_rep = mean over indicator pairs of the "
                       "per-pair share of issued reps; pairs_ge_half as above",
    "v_over_vref": "median [IQR] over pairs of the per-pair median V_E/V_ref (centring of eta*; ft1/mixture radius ratio "
                   "is a function of it, see theory)"}


def _e2_ft_pair(g: pd.DataFrame, w: pd.DataFrame | None, er: dict) -> dict:
    """Per (run, variant, estimand): fixed-time miss counts over all reps (availability-specific n) and width
    statistics on the width reps at the final epoch, joined with the anytime bounds of the width rows."""
    out: dict = {}
    n = len(g)
    for tag in FT_TAGS:
        on = np.ones(n, bool) if tag in ("ft1", "fth") else g[f"{tag}_on"].to_numpy(bool)
        nn = int(on.sum())
        lo = g[f"{tag}_lo"].to_numpy(bool) & on
        hi = g[f"{tag}_hi"].to_numpy(bool) & on
        for nm, v in ((f"{tag}_lo", lo), (f"{tag}_hi", hi), (f"{tag}_2s", lo | hi)):
            k = int(v.sum())
            p, plo, phi_ = _wil(k, nn) if nn else (float("nan"),) * 3
            out[f"{nm}_k"], out[f"{nm}_n"], out[nm], out[f"{nm}_lo"], out[f"{nm}_hi"] = k, nn, p, plo, phi_
    V = g["V"].to_numpy(float)
    for key, col in (("V_ref", "v_over_vref_med"), ("V_ref_ft", "v_over_vref_ft_med")):
        if er.get(key):
            out[col] = float(np.median(V / float(er[key])))
    if w is None or not len(w):
        return out
    m = g.set_index("rep").reindex(w["rep"].to_numpy())
    Q, Qh, S, D = (w[c].to_numpy(float) for c in ("Q", "Qhat", "S", "Dcap"))
    lows = {"anytime": w["Lstar"].to_numpy(float), "mixE": w["L"].to_numpy(float)}
    ups = {"anytime": w["U"].to_numpy(float)}
    for tag in FT_TAGS:
        lows[tag] = m[f"L{tag}"].to_numpy(float)
        ups[tag] = m[f"U{tag}"].to_numpy(float)
    issued = w["ucs_issued"].to_numpy(bool)
    hw = (w["wald_hi"] - w["wald_lo"]).to_numpy(float) / 2.0
    rE = Qh - lows["mixE"]                            # radius of the mixture at E: > 0 for every V >= 0
    ok = Q > 0
    okw = hw > 0

    def put(name, arr, iqr=False):
        arr = np.asarray(arr, float)
        arr = arr[np.isfinite(arr)]
        if not len(arr):
            vals = [float("nan")] * (3 if iqr else 1)
        else:
            vals = np.quantile(arr, [0.25, 0.5, 0.75]) if iqr else [np.median(arr)]
        for kk, vv in zip(("q25", "med", "q75") if iqr else ("med",), vals):
            out[f"{name}_{kk}"] = float(vv)

    with np.errstate(divide="ignore", invalid="ignore"):
        for X in FT_LOWER:
            LX = lows[X]
            if not np.isfinite(LX).any():
                continue                              # not computed for this pair (ftb in reactor, ftx)
            put(f"gap_{X}", (Q[ok] - np.maximum(LX[ok], S[ok])) / Q[ok], iqr=True)
            put(f"gapraw_{X}", np.minimum((Q[ok] - LX[ok]) / Q[ok], 1.0))
            out[f"sh_{X}_gt_S"] = float(np.mean(LX[np.isfinite(LX)] > S[np.isfinite(LX)]))
        for X, key in (("anytime", "runmax"), ("ft1", "ft1"), ("fth", "fth")):
            put(f"rr_lo_{key}", (Qh - lows[X]) / rE, iqr=True)
        for X in ("mixE", "ft1", "fth"):
            put(f"rw_lo_{X}", (Qh[okw] - lows[X][okw]) / hw[okw])
        out["n_wald_pos"] = int(okw.sum())
        if er.get("kind") == "indicator" and issued.any():
            Ua = ups["anytime"]
            den_hi, den_2s = Ua - Qh, Ua - lows["mixE"]
            den_2sc = np.minimum(Ua, D) - np.maximum(lows["anytime"], S)
            out["sh_anytime_lt_D"] = float(np.mean((Ua < D)[issued]))
            put("rw_hi_anytime", (Ua - Qh)[issued & okw] / hw[issued & okw])
            for X in FT_TAGS:
                UX, LX = ups[X], lows[X]
                iz = issued & np.isfinite(UX)
                if not iz.any():
                    continue
                put(f"rr_hi_{X}", ((UX - Qh) / den_hi)[iz], iqr=True)
                if X in ("ft1", "fth"):
                    put(f"rr_2s_{X}", ((UX - LX) / den_2s)[iz])
                    put(f"rw_hi_{X}", (UX - Qh)[iz & okw] / hw[iz & okw])
                num = (np.minimum(UX, D) - np.maximum(LX, S))[iz]
                dd = den_2sc[iz]
                put(f"rr_2sc_{X}", np.where(dd > 0, num / np.where(dd > 0, dd, 1.0), np.nan))
                out[f"sh_{X}_lt_D"] = float(np.mean((UX < D)[iz]))
    out["n_width_ft"] = int(len(w))
    return out


def _pool_ft(g: pd.DataFrame, flag: str) -> tuple[float, float, float, int, int]:
    """Pooled miss of a flag over pairs; fixed-time flags carry their own n (availability)."""
    if f"{flag}_n" in g:
        gg = g[g[f"{flag}_n"].fillna(0) > 0]
        k, n = int(gg[f"{flag}_k"].sum()), int(gg[f"{flag}_n"].sum())
    else:
        k, n = int(g[f"{flag}_k"].sum()), int(g["n"].sum())
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0, 0
    p, lo, hi = _wil(k, n)
    return p, lo, hi, k, n


def _ft_miss_block(g: pd.DataFrame, flag: str, side: str, label: str) -> dict:
    p, lo, hi, k, n = _pool_ft(g, flag)
    if n == 0:
        return _num(None, f"{side} miss, {label}: not computed for these pairs")
    gg = g[g[f"{flag}_n"].fillna(0) > 0] if f"{flag}_n" in g else g
    i = gg[flag].idxmax()
    nom = ALPHA / 2 if side in ("lower", "upper") else ALPHA
    return _num(p, f"{side} miss, {label}, pooled over pairs and rho (nominal <= {nom:g})", wilson=[lo, hi], k=k, n=n,
                n_pairs=int(len(gg)), max_pair=float(gg.loc[i, flag]),
                max_at=f"{gg.loc[i, 'stream']}/{gg.loc[i, 'estimand']}/rho={float(gg.loc[i, 'rho']):g}",
                n_pairs_signif_above_nominal=int((gg[f"{flag}_lo"] > nom).sum()))


def _ft_width_block(g: pd.DataFrame, iqr: bool = True) -> dict:
    """Width statistics of a set of pairs (one rho): medians over pairs of the per-pair medians, and shares."""
    out: dict = {"n_pairs": int(len(g))}

    def stat(col, ref=None):
        if col not in g:
            return None
        gg = g[g[col].notna()]
        if not len(gg):
            return None
        q = np.quantile(gg[col], [0.25, 0.5, 0.75])
        d = {"median": float(q[1]), "n_pairs": int(len(gg))}
        if iqr:
            d["iqr"] = [float(q[0]), float(q[2])]
        if ref is not None and ref in gg and len(gg) < len(g[g[ref].notna()]):
            d["anytime_same_pairs"] = float(gg[ref].median())
        return d

    sets = {"rel_gap": ("gap_{}_med", FT_LOWER, "gap_anytime_med"),
            "rel_gap_raw": ("gapraw_{}_med", FT_LOWER, "gapraw_anytime_med"),
            "lower_radius_ratio": ("rr_lo_{}_med", ("runmax", "ft1", "fth"), None),
            "lower_radius_vs_wald": ("rw_lo_{}_med", ("mixE", "ft1", "fth"), None),
            "upper_radius_ratio": ("rr_hi_{}_med", FT_TAGS, None),
            "upper_radius_vs_wald": ("rw_hi_{}_med", ("anytime", "ft1", "fth"), None),
            "two_sided_ratio": ("rr_2s_{}_med", ("ft1", "fth"), None),
            "two_sided_capped_ratio": ("rr_2sc_{}_med", FT_TAGS, None)}
    for key, (pat, tags, ref) in sets.items():
        blk = {X: stat(pat.format(X), ref) for X in tags}
        blk = {k: v for k, v in blk.items() if v}
        if blk:
            out[key] = blk
    for key, tags, suf in (("share_L_gt_S", FT_LOWER, "gt_S"), ("share_U_lt_Dcap", ("anytime",) + FT_TAGS, "lt_D")):
        sh = {}
        for X in tags:
            col = f"sh_{X}_{suf}"
            if col not in g or not g[col].notna().any():
                continue
            gg = g[g[col].notna()]
            wts = gg["n_width_ft"].fillna(1.0).to_numpy(float) if "n_width_ft" in gg else np.ones(len(gg))
            d = {"per_rep": float(np.sum(gg[col] * wts) / np.sum(wts)), "pairs_ge_half": float((gg[col] >= 0.5).mean()),
                 "pairs_any": float((gg[col] > 0).mean()), "n_pairs": int(len(gg))}
            ref = f"sh_anytime_{suf}"
            if X != "anytime" and ref in gg and len(gg) < int(g[ref].notna().sum()):
                d["anytime_same_pairs_per_rep"] = float(np.sum(gg[ref] * wts) / np.sum(wts))
            sh[X] = d
        if sh:
            out[key] = sh
    for col, key in (("v_over_vref_med", "v_over_vref"), ("v_over_vref_ft_med", "v_over_vref_ft")):
        vv = g[col].dropna() if col in g else pd.Series(dtype=float)
        if len(vv):
            out[key] = {"median": float(vv.median()), "iqr": [float(vv.quantile(0.25)), float(vv.quantile(0.75))]}
    return out


def _e2_ft_theory() -> dict:
    """Deterministic lower-radius ratios (alpha_side = 0.025, K = 16 over +-2 decades) as a function of V_E/V_ref:
    single eta* vs the mixture vs the best single eta chosen with hindsight (sqrt(2 l V_E): NOT a valid procedure,
    the reference for the 'log K' price); and the null upper radii."""
    from ctsample.estimation import cs as _cs
    a = ALPHA / 2
    lt = math.log(1 / a)
    etas, w = _cs.lcs_grid(1.0, a, K=16, decades=2.0)
    eta = float(_cs.ft_eta_lower(1.0, a))
    rows = {}
    for v in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
        r_mix = float(_cs.lcs_radius(v, etas, np.log(w), lt))
        r_ft = lt / eta + eta * v / 2
        r_opt = math.sqrt(2 * lt * v)
        rows[f"{v:g}"] = {"ft1_over_mixture": r_ft / r_mix, "mixture_over_hindsight_eta": r_mix / r_opt,
                          "ft1_over_hindsight_eta": r_ft / r_opt}
    phi0 = 0.0174
    cbar = 1 / phi0
    ucs0 = float(_cs.null_ucs_radius(cbar, a))
    return {"definition": "lower radius ratios by V_E/V_ref (keys); alpha_side = 0.025, K = 16 grid eta* x 10^[-2, 2]; "
                          "hindsight eta = sqrt(2 l/V_E) (not valid; reference only). ft1/mixture = (t + 1/t)/2 / "
                          "(mixture/hindsight), t = sqrt(V_E/V_ref): below 1 only for V_E/V_ref in about [0.23, 4.3]",
            "by_V_over_Vref": rows, "naive_logK_prediction": math.sqrt((lt + math.log(16)) / lt),
            "upper_null_radius": {"ft_fan": lt * cbar / 0.999, "ucs": ucs0, "hiding_capacity_floor": lt / -math.log1p(-phi0),
                                  "ratio_ft_over_ucs": lt * cbar / 0.999 / ucs0,
                                  "definition": "no retained hit, cbar = 1/phi0, phi0 = 0.0174 (eligible units at the floor): "
                                                "the fixed-time Fan radius l cbar/0.999 is within 1% of the hiding capacity "
                                                "(Thm. 1), the UCS pays log(1/(alpha_side w0))/l = 1.19"}}


def _flatten_ft(nums: dict) -> pd.DataFrame:
    """Long table of the fixed-time block: one row per (kind, arm, stratum, rho or 'pooled', block, comparator)."""
    rows = []
    for kind in ("lp_iot23", "lp_edge", "natural", "lp_gotham"):
        for arm, ab in (nums.get(kind) or {}).items():
            for st, sb in ab.items():
                for key, blk in sb.items():
                    if key.startswith("miss_"):
                        for comp, e in blk.items():
                            rows.append(dict(kind=kind, arm=arm, stratum=st, rho="pooled", block=key, comparator=comp,
                                             value=e.get("value"), lo=(e.get("wilson") or [None, None])[0],
                                             hi=(e.get("wilson") or [None, None])[1], k=e.get("k"), n=e.get("n"),
                                             n_pairs=e.get("n_pairs"), max_pair=e.get("max_pair"),
                                             n_signif=e.get("n_pairs_signif_above_nominal")))
                    elif key.startswith("width_rho"):
                        rho = key[len("width_rho"):]
                        for bk, bb in blk.items():
                            if not isinstance(bb, dict):
                                continue
                            for comp, e in bb.items():
                                if not isinstance(e, dict):
                                    continue
                                val = e.get("median", e.get("per_rep"))
                                iq = e.get("iqr") or [None, None]
                                rows.append(dict(kind=kind, arm=arm, stratum=st, rho=rho, block=bk, comparator=comp,
                                                 value=val, lo=iq[0], hi=iq[1], n_pairs=e.get("n_pairs"),
                                                 pairs_ge_half=e.get("pairs_ge_half"),
                                                 anytime_same_pairs=e.get("anytime_same_pairs",
                                                                          e.get("anytime_same_pairs_per_rep"))))
    return pd.DataFrame(rows)


def e2_fixed_time_numbers(S: pd.DataFrame, R: pd.DataFrame) -> dict:
    out: dict = {"definitions": dict(FT_LABEL, **FT_WIDTH_DEFS, **{
        "pair": "(stream, estimand, rho), honest variant, part main unless stated",
        "strata": "by the stream total Q: Q1-5, Q6-20, Q1-20, Q>20, Q>0 (pooled), Q=0 (upper side and shares only)",
        "miss": "lower = L > Q at the final epoch (anytime: L_e > Q_e at some epoch); upper = U < Q (indicator pairs, "
                "statement issued); two_sided = either; pooled over pairs and rho in {1, 2, 5}%; nominal 0.025 per side; "
                "Wilson 95% intervals; n_pairs_signif_above_nominal = pairs whose Wilson lower bound exceeds the nominal",
        "widths": "blocks width_rho<rho> (IQR only at rho = 5%)",
        "validity": "ft1/fth: Prop. 1(b)/(c) supermartingale with ONE eta + Markov at the fixed final epoch (Prop. 2(a) "
                    "domination under X := 0); ftb: sub-gamma (Bennett) bound with Var <= kappa Q for any predictable design "
                    "with pi >= phi; ftx: super-uniform worst-case p-values (static designs); no tuning on test streams "
                    "(V_ref from training folds, phi0 and cbar committed, window length = metadata)"}),
           "theory": _e2_ft_theory()}
    H = S[(S["variant"] == "honest") & (S["part"] == "main")]
    lower = (("anytime_tu", "lcs_tu"), ("mixE_ft", "lcs_ft"), ("ft1", "ft1_lo"), ("fth", "fth_lo"), ("ftb", "ftb_lo"),
             ("ftx", "ftx_lo"), ("wald_ft", "wald_lo_ft"))
    upper = (("anytime_tu", "ucs_tu"), ("mixE_ft", "ucs_ft"), ("ft1", "ft1_hi"), ("fth", "fth_hi"), ("ftb", "ftb_hi"),
             ("ftx", "ftx_hi"), ("wald_ft", "wald_hi_ft"))
    two = (("anytime_tu", "ours_tu"), ("mixE_ft", "ours_ft"), ("ft1", "ft1_2s"), ("fth", "fth_2s"), ("ftb", "ftb_2s"),
           ("ftx", "ftx_2s"), ("wald_ft", "wald_ft"))
    for kind in ("lp_iot23", "lp_edge", "natural", "lp_gotham"):
        Hk = H[H["kind"] == kind]
        if Hk.empty:
            continue
        blk = {}
        for arm in sorted(Hk["arm"].unique()):
            Ha = Hk[Hk["arm"] == arm]
            ab = {}
            for st, q0, q1 in FT_STRATA:
                Hq = Ha[(Ha["Q_stream"] >= q0) & (Ha["Q_stream"] <= q1)]
                if Hq.empty:
                    continue
                Hi = Hq[Hq["kind_q"] == "indicator"]
                sb: dict = {"n_pairs": int(len(Hq)), "n_pairs_indicator": int(len(Hi))}
                if st != "Q=0":
                    sb["miss_lower"] = {nm: _ft_miss_block(Hq, fl, "lower", nm) for nm, fl in lower}
                    sb["miss_two_sided"] = {nm: _ft_miss_block(Hq, fl, "two-sided", nm) for nm, fl in two}
                if len(Hi):
                    sb["miss_upper"] = {nm: _ft_miss_block(Hi, fl, "upper", nm) for nm, fl in upper}
                for rho in sorted(Hq["rho"].unique()):
                    Hr = Hq[np.isclose(Hq["rho"], rho)]
                    wb = _ft_width_block(Hr, iqr=True)
                    if st == "Q=0":     # lower bounds are never informative when Q = 0: keep the upper side
                        wb = {k: v for k, v in wb.items() if not k.startswith(("rel_gap", "lower_", "share_L"))}
                    sb[f"width_rho{rho:g}"] = wb
                ab[st] = sb
            blk[arm] = ab
        out[kind] = blk
    # A11: fixed-time statements under withholding (X := 0 valid; 'compliant only' not)
    A = S[(S["part"] == "a11") & (S["Q_stream"] > 0)]
    if len(A):
        a11 = {}
        for (kind, arm), g in A.groupby(["kind", "arm"]):
            b = {}
            for var in sorted(g["variant"].unique()):
                v = g[g["variant"] == var]
                for side, suf in (("lower", "lo"), ("upper", "hi")):
                    vv = v if side == "lower" else v[v["kind_q"] == "indicator"]
                    if vv.empty:
                        continue
                    for tag in FT_TAGS:
                        p, lo, hi, k, n = _pool_ft(vv, f"{tag}_{suf}")
                        if n:
                            b[f"{var}/{side}/{tag}"] = _num(p, f"{tag} {side} miss pooled over Q>0 pairs", wilson=[lo, hi],
                                                            k=k, n=n, max=float(vv[f"{tag}_{suf}"].max()))
            a11[f"{kind}/{arm}"] = b
        out["a11"] = a11
    Rc = S[(S["part"] == "reactor") & (S["variant"] == "honest")]
    if len(Rc):
        out["reactor"] = {f"{kind}/{arm}": {f"max_{tag}_{suf}": _num(float(g[f"{tag}_{suf}"].max()),
                                                                     f"max over pairs of the {tag} {suf} miss (reactive C&C; "
                                                                     f"ftb/ftx not computed)")
                                            for tag in ("ft1", "fth") for suf in ("lo", "hi")}
                          for (kind, arm), g in Rc.groupby(["kind", "arm"])}
    if not R.empty and "fw_ft1_honest" in R:
        out["familywise"] = {f"{part}/{kind}/{arm}": _num([float(g["fw_ft1_honest"].min()), float(g["fw_ft1_honest"].max())],
                                                          "[min, max] over runs of the family-wise miss of the ft1 "
                                                          "statements at the final epoch (part familywise: alpha/|F| each)")
                             for (part, kind, arm), g in R.groupby(["part", "kind", "arm"]) if g["fw_ft1_honest"].notna().any()}
    return out


def _f3(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "--"
    if v == 0:
        return "0"
    if isinstance(v, float) and 0 < abs(v) < 0.5 * 10 ** -nd:
        return f"{v:.0e}".replace("e-0", "e-")
    return f"{v:.{nd}f}"


def e2_fixed_time_slim(ft: dict) -> dict:
    """The numbers-file version of the fixed-time block: width blocks at rho = 5% only (all rho: E2_fixedtime.parquet)."""
    out = {}
    for k, v in ft.items():
        if k in ("lp_iot23", "lp_edge", "natural", "lp_gotham"):
            out[k] = {arm: {st: {kk: vv for kk, vv in sb.items() if not (kk.startswith("width_rho") and kk != "width_rho0.05")}
                            for st, sb in ab.items()} for arm, ab in v.items()}
        else:
            out[k] = v
    out["definitions"] = dict(out["definitions"], widths_in_json="width_rho0.05 only; every rho (and the same blocks) in "
                                                                 "results/summary/E2_fixedtime_rev1.parquet")
    return out


def e2_fixed_time_table(ft: dict) -> None:
    """Small booktabs table for the writer (LP-IoT23 and LP-Edge, static arm): miss pooled over rho in {1,2,5}%,
    widths at rho = 5%. Also writes the long table E2_fixedtime.parquet (every fixed-time number, every rho)."""
    atomic_write_parquet(_sum("E2_fixedtime.parquet"), _to_arrow(_flatten_ft(ft)))
    rows = (("anytime", "anytime_tu", "Anytime CS"),
            ("mixE", "mixE_ft", "Mixture at $E$"),
            ("ft1", "ft1", "One $\\eta^*$ at $E$"),
            ("fth", "fth", "One $\\eta_h$ at $E$"),
            ("ftb", "ftb", "Bernstein$^{a}$"),
            ("ftx", "ftx", "Exact PB$^{b}$"),
            ("wald", "wald_ft", "Wald (invalid)"))

    def g(d, *ks):
        for k in ks:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    lines = ["% generated by scripts/90_aggregate.py --rev rev1 (E2 rev1, helper P1-X-E2); every number with its definition "
             "and the other arms/streams: results/summary/E2_numbers_rev1.json['fixed_time'], E2_fixedtime_rev1.parquet",
             "\\begin{table}[!t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{1.6pt}",
             "\\caption{E2: valid fixed-time bounds at the final epoch $E$ versus the anytime-valid statements (static "
             "design).}",
             "\\label{tab:e2-fixed-time}",
             "\\begin{tabular}{@{}l rrr rr r rr@{}}", "\\toprule",
             " & \\multicolumn{3}{c}{Miscoverage} & \\multicolumn{2}{c}{Gap} & $L{>}S$ & \\multicolumn{2}{c}{Radius ratio$^{c}$}\\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){8-9}",
             "Bound & $L$, $Q{\\le}20$ & $L$, $Q{>}20$ & $U$ & $Q{\\le}20$ & $Q{>}20$ & $Q{>}20$ & lower & upper\\\\",
             "\\midrule"]
    notes = []
    for kind, lab in (("lp_iot23", "LP-IoT23"), ("lp_edge", "LP-Edge")):
        blk = g(ft, kind, "static")
        if blk is None:
            continue
        lines.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{lab}}}}}\\\\")
        for X, mk, name in rows:
            lo20 = g(blk, "Q1-20", "miss_lower", mk, "value")
            lo21 = g(blk, "Q>20", "miss_lower", mk, "value")
            up = g(blk, "Q>0", "miss_upper", mk, "value")
            w20, w21, w0, wall = (g(blk, s, "width_rho0.05") for s in ("Q1-20", "Q>20", "Q>0", "all"))
            gap20 = g(w20, "rel_gap", X, "median") if X != "wald" else None
            gap21 = g(w21, "rel_gap", X, "median") if X != "wald" else None
            sh21 = g(w21, "share_L_gt_S", X, "per_rep") if X != "wald" else None
            if X == "anytime":
                rlo, rhi = g(w0, "lower_radius_ratio", "runmax", "median"), 1.0
            elif X == "mixE":
                rlo, rhi = 1.0, 1.0
            elif X in ("ft1", "fth"):
                rlo, rhi = g(w0, "lower_radius_ratio", X, "median"), g(wall, "upper_radius_ratio", X, "median")
            elif X == "wald":
                rw = g(w0, "lower_radius_vs_wald", "mixE", "median")
                rlo, rhi = (1.0 / rw if rw else None), None
            else:
                rlo, rhi = None, g(wall, "upper_radius_ratio", X, "median")
            if X == "ftx":
                ref = g(w21, "rel_gap", "ftx", "anytime_same_pairs")
                if ref is not None:
                    notes.append(f"{lab}: anytime gap on the same $Q{{>}}20$ pairs {ref:.2f}")
            lines.append(f"{name} & {_f3(lo20)} & {_f3(lo21)} & {_f3(up)} & {_f3(gap20, 2)} & {_f3(gap21, 2)} & "
                         f"{_f3(sh21, 2)} & {_f3(rlo, 2)} & {_f3(rhi, 2)}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\\\[2pt]\\parbox{\\columnwidth}{\\tiny Miscoverage per side pooled over $\\rho\\in\\{1,2,5\\}\\%$ (nominal 0.025; "
              "upper side: indicator estimands with an issued statement); gap $(Q-\\max(L,S))/Q$, share of runs with $L>S$ "
              "and radius ratios at $\\rho=5\\%$, medians over pairs. $\\eta^*=\\sqrt{2\\log(2/\\alpha)/V_{\\mathrm{ref}}}$ "
              "(LCS grid centre); $\\eta_h$: $V_{\\mathrm{ref}}$ scaled to the 870-epoch window. "
              "$^{a}$Variance bound $\\kappa Q$, $\\kappa=\\bar b(1-\\varphi_0)/\\varphi_0$. "
              "$^{b}$Exact worst-case Poisson-binomial bound of the retained count; static designs, indicator estimands "
              "with ${\\le}1{,}000$ events" + (f" ({'; '.join(notes)})" if notes else "") + ". "
              "$^{c}$Per run at $E$: lower $(\\hat Q-L)/(\\hat Q-L_E)$ against the anytime mixture $L_E$ with the "
              "same $V_E$ (anytime row: running max $L^*$; Wald row: $z\\widehat V^{1/2}$, runs with $\\widehat V>0$); upper "
              "$(U-\\hat Q)/(U_{\\mathrm{UCS}}-\\hat Q)$, all indicator estimands incl.\\ $Q=0$.}",
              "\\end{table}"]
    (_sum("E2_fixedtime.tex")).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------------- #
# E1
# --------------------------------------------------------------------------------------------- #
E1_ORDER = ["CtS-AI", "CtS-AI-static", "CtS-Size", "CtS-Unif", "B13", "B2", "B3", "B4", "B6", "B15"]
EPOCH_MIN = 0.5


def _e1_group(kind: str, bin_: str) -> str:
    if kind == "lp_edge":
        return "LP-Edge"
    return {"0.1%": "LP-IoT23 0.1%", "1%": "LP-IoT23 1%", "5%": "LP-IoT23 5%"}.get(bin_, f"LP-IoT23 {bin_}")


def aggregate_e1() -> dict:
    runs, superseded = _runs("E1")
    inc_rows, pol_rows, est_rows = [], [], []
    for d, c in runs:
        job = c["job"]
        base = {"stream": job["stream"], "kind": job["kind"], "bin": job["bin"], "fold": job["fold"], "rho": job["rho"],
                "group": _e1_group(job["kind"], job["bin"])}
        if job["job"] == "analytic":
            for nm, lst in (("inc", inc_rows), ("pol", pol_rows), ("est", est_rows)):
                t = pd.read_parquet(d / f"{nm}.parquet")
                for k, v in base.items():
                    t[k] = v
                lst.append(t)
        else:
            s = job["seed"]
            ri = _read_parts(d, "inc")
            rr = _read_parts(d, "runs")
            re = _read_parts(d, "est")
            if ri.empty or rr.empty:
                continue
            g = ri.groupby("incident")
            ti = pd.DataFrame({"incident": g["k"].first().index, "k": g["k"].first().to_numpy(),
                               "content_recall": g["content"].mean().to_numpy(), "presence_recall": g["presence"].mean().to_numpy(),
                               "tte_median_epochs": g["tte"].median().to_numpy()})
            ti["stratum"] = [X.k_stratum(int(k)) for k in ti["k"]]
            ti["policy"], ti["seed"] = "CtS-AI", s
            for k, v in base.items():
                ti[k] = v
            inc_rows.append(ti)
            tp = pd.DataFrame([{"policy": "CtS-AI", "seed": s, "benign_bytes": rr["benign_bytes"].mean(),
                                "malicious_bytes": rr["malicious_bytes"].mean(), "total_bytes": rr["total_bytes"].mean(),
                                "spend_over_target": rr["spend_ratio"].mean(), "spend_over_target_sd": rr["spend_ratio"].std(),
                                "spend_over_target_max": rr["spend_ratio"].max(),
                                "max_abs_D_over_Bbar": rr["max_abs_D_over_B"].median(),
                                "boosted_frac": rr["boosted_frac"].mean(), "n_reps": len(rr), **base}])
            pol_rows.append(tp)
            if len(re):
                ge = re.groupby("estimand")
                te = pd.DataFrame({"estimand": ge["Q"].first().index, "Q": ge["Q"].first().to_numpy(),
                                   "bias": (ge["Qhat"].mean() - ge["Q"].first()).to_numpy(),
                                   "rmse": np.sqrt(ge.apply(lambda x: float(np.mean((x["Qhat"] - x["Q"]) ** 2)),
                                                            include_groups=False)).to_numpy(),
                                   "bias_se": (ge["Qhat"].std() / np.sqrt(ge["Qhat"].count())).to_numpy()})
                te["rel_bias"] = te["bias"] / te["Q"]
                te["rel_rmse"] = te["rmse"] / te["Q"]
                te["cls"] = [X.query_class(e) for e in te["estimand"]]
                te["policy"], te["seed"], te["variant"] = "CtS-AI", s, "natural"
                for k, v in base.items():
                    te[k] = v
                est_rows.append(te)
    INC = pd.concat(inc_rows, ignore_index=True) if inc_rows else pd.DataFrame()
    POL = pd.concat(pol_rows, ignore_index=True) if pol_rows else pd.DataFrame()
    EST = pd.concat(est_rows, ignore_index=True) if est_rows else pd.DataFrame()
    SUMMARY.mkdir(parents=True, exist_ok=True)
    for nm, df in (("E1_incidents", INC), ("E1_policies", POL), ("E1_estimators", EST)):
        atomic_write_parquet(_sum(f"{nm}.parquet"), _to_arrow(df))
    summ = e1_summary(INC, POL, EST)
    # rev1: cluster-bootstrap CIs of the stratum recalls and per-stream incident counts / recalls (R1-M5, R3-M3)
    ci, per_stream = e1_bootstrap(INC)
    if len(ci):
        summ = summ.merge(ci, on=["group", "rho", "policy"], how="left")
        atomic_write_parquet(_sum("E1_per_stream.parquet"), _to_arrow(per_stream))
    atomic_write_parquet(_sum("E1_summary.parquet"), _to_arrow(summ))
    nums = e1_numbers(summ, INC)
    if len(per_stream):
        nums["per_stream_incidents"] = e1_stream_counts(per_stream)
    if REV:
        try:
            import yaml
            base_rev = "_rev1" if REV.startswith("_rev1") else REV          # variants (_rev1_zq) share the config
            cfgp = Path(__file__).resolve().parents[1] / "config" / "experiments" / f"E1{base_rev}.yaml"
            pols = yaml.safe_load(cfgp.read_text())["policies"]
            nums["policy_labels"] = {k: v.get("label", k) for k, v in pols.items()}
        except Exception as ex:                       # noqa: BLE001
            nums["policy_labels_error"] = repr(ex)
    nums["_meta"] = {"runs": len(runs), "superseded_run_dirs": superseded, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    atomic_write_json(_sum("E1_numbers.json"), nums)
    return nums


E1_BOOT_B = 2000


def _e1_clusters(inc: pd.DataFrame) -> pd.Series:
    """Bootstrap cluster of each incident: the stream for LP-IoT23 groups; the attack class for LP-Edge (the same
    Edge-IIoTset attack sessions appear at three thinnings, so they are one cluster, R1-M5)."""
    fam = inc["family"].astype(str) if "family" in inc else pd.Series("na", index=inc.index)
    return pd.Series(np.where(inc["group"] == "LP-Edge", "edge:" + fam, inc["stream"].astype(str)), index=inc.index)


def _boot_ci(values: np.ndarray, clusters: np.ndarray, rng, B: int = E1_BOOT_B, level: float = 0.95):
    """Two-stage cluster bootstrap of the mean: clusters with replacement, incidents within each drawn cluster with
    replacement; percentile interval. Returns (lo, hi, n_clusters)."""
    uc = np.unique(clusters)
    groups = [values[clusters == c] for c in uc]
    if not len(values):
        return float("nan"), float("nan"), 0
    stats = np.empty(B)
    nc = len(groups)
    for b in range(B):
        pick = rng.integers(0, nc, nc)
        tot, cnt = 0.0, 0
        for j in pick:
            g = groups[j]
            r = g[rng.integers(0, len(g), len(g))]
            tot += float(r.sum())
            cnt += len(r)
        stats[b] = tot / cnt
    a = (1 - level) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a)), nc


def e1_bootstrap(INC: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if INC.empty:
        return pd.DataFrame(), pd.DataFrame()
    inc = INC.groupby(["group", "stream", "rho", "policy", "incident"], as_index=False).agg(
        k=("k", "first"), stratum=("stratum", "first"), content_recall=("content_recall", "mean"),
        presence_recall=("presence_recall", "mean"))
    # incident families from the analytic rows (the adaptive rows carry none)
    if "family" in INC:
        fam = INC.dropna(subset=["family"]).drop_duplicates(["stream", "incident"])[["stream", "incident", "family"]]
        inc = inc.merge(fam, on=["stream", "incident"], how="left")
    inc["cluster"] = _e1_clusters(inc)
    rng = np.random.default_rng(20261019)
    rows = []
    groups = list(inc["group"].unique()) + ["LP-IoT23 all"]
    for grp in groups:
        gi = inc[inc["group"].str.startswith("LP-IoT23")] if grp == "LP-IoT23 all" else inc[inc["group"] == grp]
        for (rho, pol), g in gi.groupby(["rho", "policy"]):
            r = {"group": grp, "rho": rho, "policy": pol}
            for st in list(X.K_STRATA) + ["all"]:
                h = g if st == "all" else g[g["stratum"] == st]
                for m in ("content", "presence"):
                    lo, hi, nc = _boot_ci(h[f"{m}_recall"].to_numpy(float), h["cluster"].to_numpy(str), rng)
                    r[f"{m}_{st}_lo"], r[f"{m}_{st}_hi"] = lo, hi
                r[f"n_clusters_{st}"] = int(h["cluster"].nunique())
            rows.append(r)
    ps = inc.groupby(["group", "stream", "rho", "policy", "stratum"], as_index=False).agg(
        n_incidents=("incident", "count"), content_recall=("content_recall", "mean"),
        presence_recall=("presence_recall", "mean"))
    return pd.DataFrame(rows), ps


def e1_stream_counts(ps: pd.DataFrame) -> dict:
    """Incidents per stream and k stratum (they do not depend on the policy or ρ)."""
    one = ps[(np.isclose(ps["rho"], 0.05)) & (ps["policy"] == ps["policy"].iloc[0])]
    out = {}
    for (grp, sid), g in one.groupby(["group", "stream"]):
        out.setdefault(grp, {})[sid] = {st: int(g[g["stratum"] == st]["n_incidents"].sum()) for st in X.K_STRATA}
    return {"value": out, "definition": "content-bearing incidents per stream and k stratum (k <= 20, 21-100, > 100)"}


def e1_summary(INC: pd.DataFrame, POL: pd.DataFrame, EST: pd.DataFrame) -> pd.DataFrame:
    """One row per (group, rho, policy): recall by k stratum (incidents pooled over the group's streams; seed-averaged
    per incident first), median TTE, bytes, spend, estimator bias/RMSE (median over (stream, estimand) with Q > 0)."""
    rows = []
    if INC.empty:
        return pd.DataFrame()
    inc = INC.groupby(["group", "stream", "rho", "policy", "incident"], as_index=False).agg(
        k=("k", "first"), stratum=("stratum", "first"), content_recall=("content_recall", "mean"),
        presence_recall=("presence_recall", "mean"), tte=("tte_median_epochs", "median"))
    groups = list(inc["group"].unique()) + ["LP-IoT23 all"]
    for grp in groups:
        gi = inc[inc["group"].str.startswith("LP-IoT23")] if grp == "LP-IoT23 all" else inc[inc["group"] == grp]
        gp = POL[POL["group"].str.startswith("LP-IoT23")] if grp == "LP-IoT23 all" else POL[POL["group"] == grp]
        ge = EST[EST["group"].str.startswith("LP-IoT23")] if grp == "LP-IoT23 all" else EST[EST["group"] == grp]
        for (rho, pol), g in gi.groupby(["rho", "policy"]):
            r = {"group": grp, "rho": rho, "policy": pol, "n_streams": int(g["stream"].nunique()), "n_incidents": int(len(g))}
            for st in X.K_STRATA:
                h = g[g["stratum"] == st]
                r[f"content_{st}"] = float(h["content_recall"].mean()) if len(h) else float("nan")
                r[f"presence_{st}"] = float(h["presence_recall"].mean()) if len(h) else float("nan")
                r[f"n_{st}"] = int(len(h))
            r["content_all"] = float(g["content_recall"].mean())
            r["presence_all"] = float(g["presence_recall"].mean())
            t = g["tte"].to_numpy(float)
            r["tte_median_min"] = float(np.median(t)) * EPOCH_MIN if len(t) else float("nan")
            r["tte_finite_share"] = float(np.isfinite(t).mean()) if len(t) else float("nan")
            p = gp[(np.isclose(gp["rho"], rho)) & (gp["policy"] == pol)]
            if len(p):
                ps = p.groupby("stream").mean(numeric_only=True)
                r["benign_MB_per_stream"] = float(ps["benign_bytes"].mean() / 1e6)
                r["malicious_MB_per_stream"] = float(ps["malicious_bytes"].mean() / 1e6)
                r["spend_over_target_mean"] = float(ps["spend_over_target"].mean())
                r["spend_over_target_min"] = float(ps["spend_over_target"].min())
                r["spend_over_target_max"] = float(ps["spend_over_target"].max())
                r["max_abs_D_over_Bbar_median"] = float(ps["max_abs_D_over_Bbar"].median())
                r["max_abs_D_over_Bbar_max"] = float(ps["max_abs_D_over_Bbar"].max())
            e = ge[(np.isclose(ge["rho"], rho)) & (ge["policy"] == pol) & (ge["Q"] > 0)]
            for var in ("natural", "scaled"):
                ev = e[e["variant"] == var] if "variant" in e else e
                if ev.empty:
                    continue
                es = ev.groupby(["stream", "estimand"]).mean(numeric_only=True)
                suf = "" if var == "natural" else "_scaled"
                r[f"rel_bias_median{suf}"] = float(es["rel_bias"].median())
                r[f"abs_rel_bias_median{suf}"] = float(es["rel_bias"].abs().median())
                r[f"rel_rmse_median{suf}"] = float(es["rel_rmse"].median())
                r[f"n_estimands{suf}"] = int(len(es))
            rows.append(r)
    return pd.DataFrame(rows)


def e1_numbers(summ: pd.DataFrame, INC: pd.DataFrame) -> dict:
    out = {"definitions": {
        "groups": "LP-IoT23 1% = the 6 LOSO folds at target p = 1% (realised 0.41-1.05%); LP-IoT23 0.1%; LP-IoT23 5% = "
                  "7-1 (4.96%); LP-Edge = realised 0.10/0.97/3.25%; incidents pooled over the group's streams",
        "content_recall": "P(at least one content-bearing unit of the incident retained whole), averaged over the "
                          "stratum's incidents (expected value for randomised policies; seeds averaged per incident)",
        "presence_recall": "P(any byte of a payload unit of the incident's metadata-defined set A = its device(s) x "
                           "[first, last epoch] retained), same averaging",
        "k": "content-bearing (malicious payload) units of the incident; strata k<=20, 21-100, >100",
        "CtS-AI": "U6tau as deployed (realised-cost controller + boost), 200 reps of the exact full-unit simulation per seed",
        "CtS-AI-static": "budget-matched static U5tau (U6tau0.2_bm)", "CtS-Size": "B10_bm", "CtS-Unif": "B8_bm",
        "tte": "per incident the median time to evidence (first retained content unit) in epochs after its first "
               "epoch; median over incidents; inf if not reached with probability 1/2",
        "rel_bias/rel_rmse": "natural estimator per evaluated estimand (>= 50 eligible units, Q > 0): bias/Q and RMSE/Q, "
                             "median over (stream, estimand); B6 on D estimands only; B15 not scored",
        "ci (rev1)": "<metric>_<stratum>_lo/_hi = 95% two-stage cluster-bootstrap percentile interval of the stratum mean "
                     "(2,000 resamples; clusters = streams for LP-IoT23 groups, attack classes for LP-Edge, which pools "
                     "the same sessions at three thinnings; n_clusters_<stratum> = number of clusters)"}}
    if summ.empty:
        return out
    for grp in sorted(summ["group"].unique()):
        blk = {}
        g = summ[summ["group"] == grp]
        for (rho, pol), r in g.set_index(["rho", "policy"]).iterrows():
            key = f"rho{rho:g}/{pol}"
            blk[key] = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in r.to_dict().items()
                        if k not in ("group",)}
        out[grp] = blk
    return out


# --------------------------------------------------------------------------------------------- #
# E3
# --------------------------------------------------------------------------------------------- #
REL_FLOOR = 1e-3          # variances / gaps below (0.001 Q)^2 / 0.001 Q are floored there (Var = 0 when pi = 1 on all hits)
E3_DESIGNS = ["U0", "U3", "U5", "U5tau0.2", "U5tau0.5", "U6", "U6tau0.2", "U6tau0.5", "U8", "U7", "B12"]


def _floor_ratio(a, b, Q, power=2):
    f = (REL_FLOOR * np.asarray(Q, float)) ** power
    return np.maximum(np.asarray(a, float), f) / np.maximum(np.asarray(b, float), f)


def aggregate_e3() -> dict:
    from ctsample.sim.metrics import cluster_bootstrap
    runs, superseded = _runs("E3")
    var_rows, mc_rows, ctrl_rows = [], [], []
    for d, c in runs:
        job = c["job"]
        base = {"stream": job["stream"], "group": job["group"], "rho": job["rho"]}
        if job["job"] == "analytic":
            v = pd.read_parquet(d / "var.parquet")
            p = pd.read_parquet(d / "ppi.parquet")
            p["vwr"] = np.nan
            t = pd.concat([v, p], ignore_index=True)
            for k, val in base.items():
                t[k] = val
            t["source"] = "analytic"
            var_rows.append(t)
            continue
        cov = _read_parts(d, "cov")
        width = _read_parts(d, "width")
        ctrl = _read_parts(d, "ctrl")
        if cov.empty:
            continue
        seed = -1 if job["seed"] is None else job["seed"]
        h = cov[cov["variant"] == "honest"]
        g = h.groupby("estimand")
        t = pd.DataFrame({"estimand": g["Q"].first().index, "Q": g["Q"].first().to_numpy(),
                          "mse_mc": g.apply(lambda x: float(np.mean((x["Qhat"] - x["Q"]) ** 2)), include_groups=False).to_numpy(),
                          "bias_mc": (g["Qhat"].mean() - g["Q"].first()).to_numpy(), "n_reps": g["rep"].nunique().to_numpy()})
        if len(width):
            wl = width.loc[width.groupby(["rep", "estimand", "variant"])["epoch"].idxmax()]
            wl = wl[wl["variant"] == "honest"]
            gw = wl.groupby("estimand")
            gap = (gw["Q"].first() - gw["Lplus"].mean())
            t = t.merge(pd.DataFrame({"estimand": gap.index, "lcb_gap": gap.to_numpy(),
                                      "lcb_gap_median": (gw["Q"].first() - gw["Lplus"].median()).to_numpy()}), on="estimand",
                        how="left")
        t["design"], t["seed"], t["arm"] = job["design"], seed, job["arm"]
        for k, val in base.items():
            t[k] = val
        mc_rows.append(t)
        if len(ctrl):
            ctrl_rows.append(dict(base, design=job["design"], seed=seed, arm=job["arm"],
                                  spend_ratio=float(ctrl["spend_ratio"].mean()),
                                  spend_ratio_max=float(ctrl["spend_ratio"].max()),
                                  boosted_frac=float(ctrl["boosted_frac"].mean())))
    V = pd.concat(var_rows, ignore_index=True) if var_rows else pd.DataFrame()
    M = pd.concat(mc_rows, ignore_index=True) if mc_rows else pd.DataFrame()
    CT = pd.DataFrame(ctrl_rows)
    # ---- per (stream, rho, design, seed, estimand) MSE ratios vs U1 ------------------------------------------- #
    ref = V[V["design"] == "U1"][["stream", "rho", "estimand", "var", "vwr"]].rename(columns={"var": "var_U1", "vwr": "vwr_U1"})
    A = V.merge(ref, on=["stream", "rho", "estimand"], how="left")
    A["mse"] = A["var"]
    # adaptive arms: empirical MSE (MC, rho = 5% only); U6 static = U5 is not used
    Mad = M[M["arm"] == "adaptive"].copy() if len(M) else pd.DataFrame()
    if len(Mad):
        cls = V.drop_duplicates(["stream", "estimand"])[["stream", "estimand", "cls", "kind_q"]]
        Mad = Mad.merge(cls, on=["stream", "estimand"], how="left").merge(ref, on=["stream", "rho", "estimand"], how="left")
        Mad["mse"] = Mad["mse_mc"]
        Mad["source"] = "mc"
        A = pd.concat([A, Mad[["stream", "group", "rho", "design", "seed", "estimand", "Q", "cls", "kind_q", "mse", "var_U1",
                              "vwr_U1", "source"]]], ignore_index=True)
    A = A[A["Q"] > 0]
    A["mse_ratio"] = _floor_ratio(A["mse"], A["var_U1"], A["Q"])
    A["vwr_ratio"] = A["vwr"] / A["vwr_U1"]
    # LCB gap ratio vs U1 (MC, rho = 5%)
    if len(M) and "lcb_gap" in M:
        g1 = M[(M["design"] == "U1") & (M["arm"] == "static_bm")][["stream", "rho", "estimand", "lcb_gap"]].rename(
            columns={"lcb_gap": "lcb_gap_U1"})
        L = M.merge(g1, on=["stream", "rho", "estimand"], how="left")
        L = L[L["Q"] > 0]
        L["lcb_ratio"] = _floor_ratio(L["lcb_gap"], L["lcb_gap_U1"], L["Q"], power=1)
        cls = V.drop_duplicates(["stream", "estimand"])[["stream", "estimand", "cls", "kind_q"]]
        L = L.merge(cls, on=["stream", "estimand"], how="left")
    else:
        L = pd.DataFrame()
    SUMMARY.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(_sum("E3_estimands.parquet"), _to_arrow(A))
    if len(L):
        atomic_write_parquet(_sum("E3_lcb.parquet"), _to_arrow(L))
    if len(CT):
        atomic_write_parquet(_sum("E3_ctrl.parquet"), _to_arrow(CT))
    summ, nums = e3_summarise(A, L, CT, cluster_bootstrap)
    atomic_write_parquet(_sum("E3_summary.parquet"), _to_arrow(summ))
    nums.update(e3_context(A))
    infl = e3_inflation(sorted(A["stream"].unique()))
    if len(infl):
        atomic_write_parquet(_sum("E3_inflation.parquet"), _to_arrow(infl))
        ig = infl.merge(A[["stream", "group"]].drop_duplicates(), on="stream").groupby(["group", "rho", "design"]).agg(
            measured=("measured_inflation", lambda x: float(np.exp(np.mean(np.log(x))))),
            predicted=("predicted_1pcv2", lambda x: float(np.exp(np.mean(np.log(x)))))).reset_index()
        nums["inflation_vs_1pcv2"] = {"value": ig.to_dict(orient="records"),
                                      "definition": "geometric mean over scenarios (and seeds) of the measured V^wr inflation "
                                                    "vs the oracle U7 (sum a°^2/pi, budget-matched designs) and of the unclipped "
                                                    "prediction 1 + CV^2_w(a°/a), w ~ a sqrt(s) (Remark); per stream in "
                                                    "E3_inflation.parquet"}
    nums["_meta"] = {"runs": len(runs), "superseded_run_dirs": superseded, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    atomic_write_json(_sum("E3_numbers.json"), nums)
    return nums


def _ratio_of_sums_boot(per_s: dict, B: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """per_s: scenario -> (num_q, den_q) arrays over the scenario's estimands. Statistic: geometric mean over
    scenarios of sum(num)/sum(den). Cluster bootstrap: scenarios resampled with replacement, estimands nested."""
    rng = np.random.default_rng(seed)
    keys = [k for k, (a, b) in per_s.items() if len(a) and np.sum(b) > 0]
    if not keys:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "B": B}
    arrs = [(np.asarray(per_s[k][0], float), np.asarray(per_s[k][1], float)) for k in keys]

    def stat(groups):
        r = [np.sum(a) / np.sum(b) for a, b in groups]
        r = np.maximum(np.asarray(r), 1e-300)
        return float(np.exp(np.mean(np.log(r))))

    point = stat(arrs)
    S = len(arrs)
    boots = np.empty(B)
    for i in range(B):
        grp = []
        for j in rng.integers(0, S, S):
            a, b = arrs[j]
            pick = rng.integers(0, len(a), len(a))
            if np.sum(b[pick]) <= 0:
                pick = np.arange(len(a))
            grp.append((a[pick], b[pick]))
        boots[i] = stat(grp)
    al = (1 - level) / 2
    return {"point": point, "lo": float(np.quantile(boots, al)), "hi": float(np.quantile(boots, 1 - al)), "B": B}


def e3_summarise(A: pd.DataFrame, L: pd.DataFrame, CT: pd.DataFrame, cluster_bootstrap) -> tuple[pd.DataFrame, dict]:
    rows = []
    nums: dict = {"definitions": {
        "unit": "test scenario (stream); in-domain = 6 LOSO LP-IoT23 streams at p = 1%; zero-shot = LP-Edge (0.97%), "
                "LP-Gotham thinned (1.09%) and natural (15.3%)",
        "relmse_ratio": "per scenario: sum_q Var_d(q)/Q_q^2 / sum_q Var_U1(q)/Q_q^2 over its estimands (classes D, S with "
                        "Q > 0; the utility's equal-weight objective, Prop. design (b)); Var = analytic sum f^2(1-pi)/pi of the "
                        "budget-matched designs (seed-averaged); adaptive U6/U6tau = empirical MSE over 500 MC reps (rho = 5% "
                        "only, as deployed, spend/target reported); summary = geometric mean over scenarios, 95% CI = cluster "
                        "bootstrap over scenarios with estimands nested (2,000 resamples)",
        "worst_rv_ratio": "per scenario max_q Var_d/Q^2 over max_q Var_U1/Q^2 (minimax objective), geometric mean over scenarios",
        "share_exact": "share of (scenario, estimand) pairs where the design retains every hit unit with probability 1 "
                       "(Var = 0: the total is known exactly)",
        "gm_pos": "geometric mean of Var_d/Var_U1 over pairs where both are > 0",
        "worst_pair_ratio": "max over pairs of Var_d/Var_U1 (inflation); worst_vwr = max of sum f^2/pi ratios (Remark bound "
                            "1/tau_eff: 5.0001 for tau = 0.2, 2 for tau = 0.5)",
        "lcb_ratio": "per scenario sum_q (Q - L+)/Q under the design / the same under U1 (mean over 500 reps, final epoch, "
                     "rho = 5%), geometric mean over scenarios, cluster bootstrap",
        "seed_sd": "s.d. over the 3 utility seeds of the per-scenario relmse_ratio, averaged over scenarios"}}
    for primary, classes in (("DS", ["D", "S"]), ("DSL", ["D", "S", "L"])):
        for (grp, rho), G in A.groupby(["group", "rho"]):
            for des in E3_DESIGNS:
                g = G[(G["design"] == des) & G["cls"].isin(classes)]
                if g.empty:
                    continue
                ge = g.groupby(["stream", "estimand"]).agg(mse=("mse", "mean"), var_U1=("var_U1", "first"), Q=("Q", "first"),
                                                            vwr=("vwr", "mean"), vwr_U1=("vwr_U1", "first"))
                per_s = {}
                worst = []
                for sid, h in ge.groupby(level=0):
                    per_s[sid] = ((h["mse"] / h["Q"] ** 2).to_numpy(), (h["var_U1"] / h["Q"] ** 2).to_numpy())
                    worst.append(float((h["mse"] / h["Q"] ** 2).max() / (h["var_U1"] / h["Q"] ** 2).max()))
                bs = _ratio_of_sums_boot(per_s, B=2000, seed=0)
                sd = []
                for sid, gs in g.groupby("stream"):
                    ps = gs.groupby("seed").apply(lambda x: float((x["mse"] / x["Q"] ** 2).sum() / (x["var_U1"] / x["Q"] ** 2).sum()),
                                                  include_groups=False)
                    if len(ps) > 1:
                        sd.append(float(ps.std(ddof=1)))
                rr = (ge["mse"] / ge["var_U1"]).to_numpy(float)
                pos = (ge["mse"] > 0) & (ge["var_U1"] > 0)
                imax = int(np.nanargmax(rr)) if len(rr) else None
                r = {"classes": primary, "group": grp, "rho": rho, "design": des, "n_scenarios": len(per_s),
                     "n_pairs": int(len(ge)), "gm": bs["point"], "lo": bs["lo"], "hi": bs["hi"],
                     "per_scenario": json.dumps({k: float(np.sum(a) / np.sum(b)) for k, (a, b) in per_s.items()}),
                     "seed_sd": float(np.mean(sd)) if sd else float("nan"),
                     "worst_rv_ratio_gm": float(np.exp(np.mean(np.log(np.maximum(worst, 1e-300))))),
                     "share_exact": float((ge["mse"] <= 0).mean()),
                     "gm_pos": float(np.exp(np.mean(np.log(rr[pos.to_numpy()])))) if pos.any() else float("nan"),
                     "median_pair_ratio": float(np.median(rr)),
                     "share_better": float((rr < 1).mean()),
                     "worst_pair_ratio": float(rr[imax]) if imax is not None else float("nan"),
                     "worst_pair_at": "/".join(map(str, ge.index[imax])) if imax is not None else "",
                     "worst_vwr_ratio": float((ge["vwr"] / ge["vwr_U1"]).max()) if ge["vwr"].notna().any() else float("nan")}
                if len(CT) and des in ("U6", "U6tau0.2", "U6tau0.5"):
                    ct = CT[(CT["group"] == grp) & (np.isclose(CT["rho"], rho)) & (CT["design"] == des) & (CT["arm"] == "adaptive")]
                    if len(ct):
                        r["spend_ratio_mean"] = float(ct["spend_ratio"].mean())
                        r["spend_ratio_max"] = float(ct["spend_ratio_max"].max())
                        r["spend_ratio_by_stream"] = json.dumps(ct.groupby("stream")["spend_ratio"].mean().round(4).to_dict())
                if len(L):
                    gl = L[(L["group"] == grp) & (np.isclose(L["rho"], rho)) & (L["design"] == des) & L["cls"].isin(classes)]
                    if len(gl):
                        gle = gl.groupby(["stream", "estimand"]).agg(gap=("lcb_gap", "mean"), gap1=("lcb_gap_U1", "first"),
                                                                     Q=("Q", "first"))
                        ls = {sid: ((h["gap"] / h["Q"]).to_numpy(), (h["gap1"] / h["Q"]).to_numpy())
                              for sid, h in gle.groupby(level=0)}
                        bl = _ratio_of_sums_boot(ls, B=2000, seed=1)
                        r["lcb_gm"], r["lcb_lo"], r["lcb_hi"] = bl["point"], bl["lo"], bl["hi"]
                        r["lcb_per_scenario"] = json.dumps({k: float(np.sum(a) / np.sum(b)) for k, (a, b) in ls.items()})
                rows.append(r)
    S = pd.DataFrame(rows)
    for _, r in S[S["classes"] == "DS"].iterrows():
        key = f"{r['group']}/rho{r['rho']:g}/{r['design']}"
        nums[key] = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in r.items()
                     if k not in ("classes", "group", "rho", "design")}
    return S, nums


def e3_inflation(streams, rhos=(0.02, 0.05), seeds=(0, 1, 2)) -> pd.DataFrame:
    """Supplement 'inflation vs 1 + CV^2' (Remark, paper Sec. VI): with the oracle utility a° (U7 column: the
    pre-registered objective with true f) and V^wr(π) = Σ_i a°_i²/π_i, the measured inflation of a budget-matched design
    is V^wr(π_design)/V^wr(π_U7); an UNCLIPPED design from utility a would inflate it by 1 + CV²_w(a°/a), w_i ∝ a_i √s_i.
    Designs with a single utility: U0 (a = √s), U1 (a = s), U3, U5, U8 (payload units with a > 0)."""
    from ctsample.design.probs import sqrt_eff
    rows = []
    for sid in streams:
        T = X.load_tables(sid)
        pidx = X.pay_index(T)
        s = np.asarray(T.s, dtype=float)[pidx]
        sq = sqrt_eff(s)
        a0 = np.asarray(T.util("U7"), dtype=float)[pidx]
        for rho in rhos:
            try:
                p7 = X.design_pi(sid, "U7_bm", rho)
            except FileNotFoundError:
                continue
            v7 = float(np.sum(a0 ** 2 / p7))
            for lab, stem, util in (("U0", "U0_bm", None), ("U1", "U1_bm", "s"), ("U3", "U3_s{s}_bm", "U3_s{s}"),
                                    ("U5", "U5_s{s}_bm", "U5_s{s}"), ("U8", "U8_bm", "U8")):
                for sd in (seeds if "{s}" in stem else (None,)):
                    pi = X.design_pi(sid, stem.format(s=sd), rho)
                    if util is None:
                        a = sq
                    elif util == "s":
                        a = s
                    else:
                        a = np.asarray(T.util(util.format(s=sd)), dtype=float)[pidx]
                    m = a > 0
                    w = a[m] * sq[m]
                    w = w / w.sum()
                    ratio = a0[m] / a[m]
                    mu = float(np.sum(w * ratio))
                    cv2 = float(np.sum(w * ratio ** 2) / mu ** 2 - 1.0) if mu > 0 else float("nan")
                    rows.append({"stream": sid, "rho": rho, "design": lab, "seed": -1 if sd is None else sd,
                                 "measured_inflation": float(np.sum(a0 ** 2 / pi)) / v7, "predicted_1pcv2": 1.0 + cv2,
                                 "share_a_zero": float((~m).mean()), "share_at_floor": float(np.mean(pi <= pi.min() * (1 + 1e-9)))})
    return pd.DataFrame(rows)


def e3_context(A: pd.DataFrame) -> dict:
    """AUC of g_q per estimand (P1-D auc.json, mean over seeds), KS shift (zero-shot), mixture bound check."""
    from ctsample.sim import experiments_core as Xc
    out = {}
    auc = {}
    for sid in sorted(A["stream"].unique()):
        fold = "all" if not sid.startswith("lp_iot23") else sid.split("_")[2]
        vals: dict = {}
        for s in (0, 1, 2):
            p = Xc.MODELS / fold / str(s) / "auc.json"
            if not p.exists():
                continue
            a = json.loads(p.read_text()).get(sid, {})
            for q, d in a.items():
                if d.get("auc") is not None:
                    vals.setdefault(q, []).append(float(d["auc"]))
        auc[sid] = {q: float(np.mean(v)) for q, v in vals.items()}
    out["auc_by_stream"] = {"value": auc, "definition": "AUC of g_q (P1-D, eligible test units, mean over 3 seeds); "
                            "in-domain g_mal / L1 AUC 0.99-1.00 partly reflects the CIC-background vs IoT-23-foreground "
                            "dataset contrast (domain shortcut, P1-D notes), not only attack detectability"}
    ks = {}
    for sid in sorted(A["stream"].unique()):
        if sid.startswith("lp_iot23"):
            continue
        p = Xc.MODELS / "all" / "0" / "ks.json"
        if p.exists():
            k = json.loads(p.read_text()).get(sid, {})
            if k:
                top = sorted(k.items(), key=lambda kv: -kv[1])[:5]
                ks[sid] = {"max": top[0][1], "top5": top, "median": float(np.median(list(k.values())))}
    out["ks_shift_zero_shot"] = {"value": ks, "definition": "two-sample KS statistic train (all IoT-23 + CIC training "
                                 "half) vs test payload-unit features, seed-0 model (P1-D ks.json)"}
    mix = A[A["design"].isin(["U5tau0.2", "U5tau0.5"])]
    if len(mix):
        out["mixture_bound_check"] = {
            "value": {d: float(g["vwr_ratio"].max()) for d, g in mix.groupby("design")},
            "definition": "max over (scenario, rho, seed, estimand) of V^wr(design)/V^wr(U1); Remark bound 1/tau_eff = "
                          "5.0001 (tau = 0.2, t16 = 13107) and 2 (tau = 0.5)"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all")
    ap.add_argument("--rev", default="", help="revision suffix, e.g. rev1 (reads results/<exp>_rev1)")
    a = ap.parse_args()
    global REV
    REV = f"_{a.rev}" if a.rev else ""
    exps = ("E1", "E2", "E3") if a.exp == "all" else tuple(a.exp.split(","))
    for e in exps:
        fn = {"E2": aggregate_e2}.get(e) or globals().get(f"aggregate_{e.lower()}")
        if fn is None:
            print(f"[agg] {e}: not implemented yet", flush=True)
            continue
        t0 = time.time()
        fn()
        print(f"[agg] {e} done in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
