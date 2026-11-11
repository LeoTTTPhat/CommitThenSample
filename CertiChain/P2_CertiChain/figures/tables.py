#!/usr/bin/env python
"""LaTeX (booktabs) drop-ins for Tables III-V of the P2 paper (+ supplementary interval versions). Owner: P2-G.

  .venv/bin/python -I P2_CertiChain/figures/tables.py
Writes figures/table3_data.tex, table4_main.tex, table4_main_ci.tex (supplement: intervals and CER_link),
table5_real.tex, table_s_strata.tex, table_s_e3.tex and table_s_v1v2.tex (CF-GOTHAM v1 vs v2, deviation P1). Every number comes from results/summary/E*_numbers.json (no hand-typed numbers); a cell whose
source does not exist yet is printed as \\tbd{pending}.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402

E1, E2, E11, E15 = (S.numbers(e) for e in ("E1", "E2", "E11", "E15"))
PENDING = "\\tbd{pending}"


def val(d: dict, key: str):
    e = d.get(key)
    if not e:
        return None
    v = e.get("value")
    if v == "inf":
        return math.inf
    return v


def ci(d: dict, key: str):
    e = d.get(key)
    return (e or {}).get("ci")


def f(v, nd=3) -> str:
    if v is None:
        return PENDING
    if isinstance(v, str):
        return v
    if math.isinf(v):
        return "$\\infty$"
    if math.isnan(v):
        return "--"
    out = f"{v:.{nd}f}"
    if out.startswith("-"):                      # typographic minus; "-0.000" -> "0.000"
        out = out[1:] if float(out) == 0 else "$-$" + out[1:]
    return out


def fci(d, key, nd=3) -> str:
    v = val(d, key)
    c = ci(d, key)
    if v is None:
        return PENDING
    if c and c[0] is not None and c[1] is not None:
        return f"{f(v, nd)} [{f(c[0], nd)}, {f(c[1], nd)}]"
    return f(v, nd)


# =========================================================================== Table IV
ROWS = [
    ("CRC-BH, statistical score", "per-incident FDR", "E1", "S_TON.stat", "crc_bh"),
    ("CRC-BH, fused score", "per-incident FDR", "E1", "S_TON.fuse_s0", "crc_bh"),
    ("CRC-FDX ($\\gamma=0.2$)", "$\\Prob(\\FDP>\\gamma)\\le\\alpha$", "E1", "S_TON.stat", "crc_fdx020"),
    ("LTT ($\\delta=0.1$)", "PAC, per incident", "E1", "S_TON.stat", "ltt"),
    ("BH-iid, per incident", "only under unit exchangeability", "E1", "S_TON.stat", "bh_iid"),
    ("BH-iid, pooled", "pooled FDR, unit exchangeability", "E1", "S_TON.stat", "bh_iid_pooled"),
    ("BY-ref", "none", "E1", "S_TON.stat", "by_ref"),
    ("LR-A1 / LR-A1b", "pooled FDR", "E1", "S_TON.stat", ("lr_a1avg", "lr_a1avgb")),
    ("LR-A2 / LR-A2b", "pooled FDR", "E1", "S_TON.stat", ("lr_a2", "lr_a2b")),
    ("B1 King-style tracking", "none", "E2", "fixed", "B1"),
    ("B2 BotHunter-style rules", "none", "E2", "fixed", "B2"),
    ("B4 lag likelihood, LLR $>0$", "none", "E2", "fixed", "B4"),
    ("CRC-BH, lag-likelihood score (B4)", "per-incident FDR", "E2", "S_TON.lag_llr", "crc_bh"),
    ("B5 LLM only ($\\sllm>0$; 15 incidents)", "none", "E2", "fixed", "B5"),
    ("B7 fused score at 0.5 (isotonic)", "none", "E1", "S_TON.fuse_s0", "b7"),
    ("B8 Platt plug-in, fused score", "none (asymptotic heuristic)", "E1", "S_TON.fuse_s0", "b8"),
    ("$r_C$-limited oracle (test labels)", "--", "E2", "fixed", "oracle"),
]
NO_LEVEL = {"B1", "B2", "B4", "B5", "b7", "oracle"}


def t4_cells(src: str, sid: str, meth, a: float, with_ci: bool) -> list[str]:
    d = E1 if src == "E1" else E2
    out = []
    methods = meth if isinstance(meth, tuple) else (meth,)
    for metric, nd in (("fdr", 3), ("pooled_fdr", 3), ("str_mean", 3), ("load_ratio", 2)):
        parts = []
        for m in methods:
            aa = 0.1 if m in NO_LEVEL else a
            key = f"cf_ton|{sid}|{m}|a={aa}|{metric}" if m != "B5" else f"cf_ton|fixed|B5|{metric}"
            if m == "B5":
                parts.append(f(val(d, key), nd))
                continue
            if metric == "load_ratio":
                v = val(d, key)
                nclaim = val(d, f"cf_ton|{sid}|{m}|a={aa}|n_claims")
                parts.append("--" if (nclaim == 0 or v is None and val(d, f"cf_ton|{sid}|{m}|a={aa}|fdr") is not None)
                             else (fci(d, key, nd) if with_ci else f(v, nd)))
            else:
                parts.append(fci(d, key, nd) if with_ci else f(val(d, key), nd))
        out.append(" / ".join(parts) if len(set(parts)) > 1 else parts[0])
    return out


def table4(with_ci: bool = False) -> str:
    lines = []
    for label, guar, src, sid, meth in ROWS:
        m0 = meth[0] if isinstance(meth, tuple) else meth
        c1 = t4_cells(src, sid, meth, 0.1, with_ci)
        if m0 in NO_LEVEL:
            c2 = ["\\multicolumn{4}{c}{(no level)}"] if m0 != "oracle" else ["\\multicolumn{4}{c}{(as $\\alpha=0.1$)}"]
        else:
            c2 = t4_cells(src, sid, meth, 0.2, with_ci)
        lines.append(" & ".join([label, guar, *c1, *c2]) + "\\\\")
        if label.startswith("CRC-BH, fused") or label.startswith("LR-A2") or label.startswith("CRC-BH, lag"):
            lines.append("\\addlinespace[1pt]")
    cap = ("Main Results on CF-TON: Realised FDR on Both Estimands, Power (STR) and Analyst Load "
           "$|\\hat S|/|\\hat S\\cap\\{Y=1\\}|$ at $\\alpha\\in\\{0.1,0.2\\}$ (Mean over 200 Splits, $K=150$ Calibration and "
           "150 Test Incidents; " + ("Incident-Pool Cluster-Bootstrap 95\\% Intervals, $B=1000\\times50$ Splits"
                                     if with_ci else "Cluster-Bootstrap Intervals and $\\CER$ in the Supplement") +
           "). The Second Column States What Each Method Guarantees; Rows Without ``Fused'' Use the Statistical Score.")
    head = "\\begin{table*}[t]\n\\centering\n\\caption{" + cap + "}\n\\label{tab:main" + ("_ci" if with_ci else "") + "}\n"
    head += "\\scriptsize\n\\setlength{\\tabcolsep}{" + ("1.6pt" if with_ci else "3.0pt") + "}\n"
    head += ("\\resizebox{\\textwidth}{!}{%\n" if with_ci else "") + "\\begin{tabular}{@{}llcccccccc@{}}\n\\toprule\n"
    head += " & & \\multicolumn{4}{c}{$\\alpha=0.1$} & \\multicolumn{4}{c}{$\\alpha=0.2$}\\\\\n"
    head += "\\cmidrule(lr){3-6}\\cmidrule(l){7-10}\n"
    head += "Method & Guarantee & FDR (incident) & FDR (pooled) & STR & load & FDR (incident) & FDR (pooled) & STR & load\\\\\n\\midrule\n"
    foot = "\\bottomrule\n\\end{tabular}" + ("}" if with_ci else "") + "\n"
    foot += ("\n\\vspace{2pt}\n{\\scriptsize Load = ratio of sums over all splits' test incidents; ``--'' = no claims. "
             "LR-A1 = derandomised Alg.~1 (Eq.~12 average of 100 draws). LR rows certify nothing in any split. "
             "No-level rows are split-independent claim sets evaluated on the same test incidents.}\n")
    foot += "\\end{table*}\n"
    return head + "\n".join(lines) + "\n" + foot


def table4_cer() -> str:
    """Supplement: CER_link and CCR for the leveled CRC rows."""
    rows = [("CRC-BH, statistical", "S_TON.stat", "crc_bh"), ("CRC-BH, fused", "S_TON.fuse_s0", "crc_bh"),
            ("composite CRC (chains)", "S_TON.stat", "chain_composite"), ("CRC-FDX (0.2)", "S_TON.stat", "crc_fdx020"),
            ("LTT", "S_TON.stat", "ltt"), ("BH-iid", "S_TON.stat", "bh_iid"), ("B8, fused", "S_TON.fuse_s0", "b8")]
    lines = []
    for lab, sid, m in rows:
        cells = []
        for a in (0.1, 0.2):
            for metric in ("cer_link_mean", "fdr_lenient", "bridge_recall", "ucr", "wrongful_benign"):
                key = f"cf_ton|{sid}|{m}|a={a}|{metric}"
                # CF-TON has no true bridge candidates (c_bridge = 0 in every incident): bridge recall is undefined
                cells.append("--" if metric == "bridge_recall" and val(E1, key) is None else fci(E1, key))
        lines.append(" & ".join([lab, *cells]) + "\\\\")
    head = ("\\begin{table*}[t]\n\\centering\n\\caption{CF-TON Secondary Metrics (Mean over 200 Splits, 95\\% Cluster-Bootstrap "
            "Intervals): CER$_{link}$, Lenient FDR, Bridge Recall, UCR (Share of Claims with a Decoy Endpoint) and "
            "Wrongful Implication $\\Prob$(any claim $|$ benign-only incident)}\n\\label{tab:main_secondary}\n\\tiny\n"
            "\\setlength{\\tabcolsep}{1.2pt}\n\\begin{tabular}{@{}lcccccccccc@{}}\n\\toprule\n"
            " & \\multicolumn{5}{c}{$\\alpha=0.1$} & \\multicolumn{5}{c}{$\\alpha=0.2$}\\\\\n"
            "\\cmidrule(lr){2-6}\\cmidrule(l){7-11}\n"
            "Method & CER$_{link}$ & FDR (len.) & bridge & UCR & wrongful & CER$_{link}$ & FDR (len.) & bridge & UCR & wrongful\\\\\n\\midrule\n")
    foot = ("\\bottomrule\n\\end{tabular}\n\n\\vspace{2pt}\n{\\scriptsize Bridge recall = certified true bridges / "
            "true bridges (pooled); ``--'' = undefined, because the CF-TON cal/test pool contains no true bridge "
            "candidates.}\n\\end{table*}\n")
    return head + "\n".join(lines) + "\n" + foot


# =========================================================================== Table III
V2NOTE = ("regenerated with fractional-second timestamps after the pre-registered E15 artifact check (post-PREREG "
          "deviation P1); the indented v1 rows are the pre-registered originals")


def table3() -> str:
    D = [("CF-TON (exact)", "main benchmark", "150 / 60 / 300", "150", "cf_ton"),
         ("CF-TON-H (exact)", "held-out templates (E3)", "150 / -- / 200", "100", "cf_ton_h"),
         ("CF-GOTHAM v2 (exact)$^\\ddagger$", "second benchmark", "100 / -- / 200", "100", "cf_gotham_v2"),
         ("\\quad CF-GOTHAM v1 (orig.)", "pre-registered original", "100 / -- / 200", "100", "cf_gotham"),
         ("CF-GOTHAM-T v2 (exact)$^\\ddagger$", "bank for Gotham", "-- / -- / 150", "150", "cf_gotham_t_v2"),
         ("\\quad CF-GOTHAM-T v1 (orig.)", "pre-registered original", "-- / -- / 150", "150", "cf_gotham_t"),
         ("IoT-23 (label-derived)", "real data, P-LOSO", "23 scenarios, 166 windows", "22", "iot23"),
         ("Gotham 2025 (schedule/payload)", "case study", "7 attack + 2 benign", "150", "gotham_real")]
    lines = []
    for name, role, inc, K, ds in D:
        g = lambda k: val(E15, f"descriptor|{ds}|{k}")  # noqa: E731
        o1 = val(E15, f"oracle|{ds}|B1|str_mean")
        o2 = val(E15, f"oracle|{ds}|B2|str_mean")
        oracle = f"{f(o1)} / {f(o2)}" if o1 is not None else "--"
        p1 = val(E15, f"probe|{ds}|caltest|i|all|auc")
        p2 = val(E15, f"probe|{ds}|caltest|ii|all|auc")
        probe = f"{f(p1, 2)} / {f(p2, 2)}" if p1 is not None else "--"
        cells = [name, role, inc, K, f(g("N_median"), 0) if g("N_median") is not None else PENDING,
                 f(g("N_over_r_all"), 0) if g("N_over_r_all") is not None else PENDING,
                 f(g("r_CT")), f(g("rho_T")), f(g("ceiling")), oracle, probe]
        lines.append(" & ".join(cells) + "\\\\")
    reuse = []
    for v, lab in (("cf_ton", "CF-TON and CF-TON-H (pooled)"), ("cf_gotham_v2", "CF-GOTHAM v2"),
                   ("cf_gotham", "CF-GOTHAM and CF-GOTHAM-T v1 (pooled)")):
        parts = []
        for pool in ("train", "caltest"):
            e = E15.get(f"reuse|{v}|{pool}|incidents_per_used_episode")
            if e:
                parts.append(f"{pool} pool {e['value']:.1f} (max {e.get('max_per_episode')})")
        if parts:
            reuse.append(f"{lab}: " + ", ".join(parts))
    t1 = val(E15, "probe|cf_ton|caltest|i|timing|auc")
    t2 = val(E15, "probe|cf_ton|caltest|ii|timing|auc")
    g1 = val(E15, "probe|cf_gotham_v2|caltest|i|timing|auc")
    g2 = val(E15, "probe|cf_gotham_v2|caltest|ii|timing|auc")
    o1_ = val(E15, "probe|cf_gotham|caltest|i|timing|auc")
    o2_ = val(E15, "probe|cf_gotham|caltest|ii|timing|auc")
    X1 = S.numbers("X1")
    fr = lambda ds, t: val(X1, f"probe|{ds}|caltest|{t}|group|fractional_seconds|auc")  # noqa: E731
    gp = lambda ds, t: val(X1, f"probe|{ds}|caltest|{t}|group|gaps|auc")  # noqa: E731
    head = ("\\begin{table*}[t]\n\\centering\n\\caption{Benchmarks and Datasets: Roles, Sizes and Descriptors of the "
            "Calibration/Test Pools (Pooled over Incidents)}\n\\label{tab:data}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.4pt}\n"
            "\\begin{tabular}{@{}llccccccccc@{}}\n\\toprule\n"
            " & & Incidents & & median & & & & ceiling & oracle STR & probe AUC\\\\\n"
            "Dataset (GT) & Role & train / dev / cal-test & $K$ & $N$ & $N/r$ & $\\hat r_{C,T}$ & $\\hat\\rho_T$ & "
            "$(1-\\hat\\rho_T)\\hat r_{C,T}$ & B1 / B2 & (i) / (ii)\\\\\n\\midrule\n")
    foot = ("\\bottomrule\n\\end{tabular}\n\n\\vspace{2pt}\n{\\scriptsize $^\\ddagger$v2 = " + V2NOTE + ". "
            "$N/r$ = all candidates of the pool per "
            "observable transition whose enabling pair is in $C$. Oracle STR = mean per-incident STR of the rule-only oracles on the "
            "whole cal/test pool (no level). Probe AUC = pooled out-of-fold AUC of an incident-grouped 5-fold gradient-"
            "boosted probe on timing/position features plus the scorer's unit features, cal/test pool: (i) campaign vs "
            f"decoy units, (ii) injected D3 look-alikes vs native background units; timing/position features alone: "
            f"CF-TON {f(t1, 2)} / {f(t2, 2)}, CF-GOTHAM v2 {f(g1, 2)} / {f(g2, 2)} (v1 {f(o1_, 2)} / {f(o2_, 2)}); "
            f"fractional-second features alone: v2 {f(fr('cf_gotham_v2', 'i'), 2)} / {f(fr('cf_gotham_v2', 'ii'), 2)} "
            f"(v1 {f(fr('cf_gotham', 'i'), 2)} / {f(fr('cf_gotham', 'ii'), 2)}); inter-record gap features alone: v2 "
            f"{f(gp('cf_gotham_v2', 'i'), 2)} / {f(gp('cf_gotham_v2', 'ii'), 2)}. Episode reuse (incidents per used "
            "library episode): " + "; ".join(reuse) + ".}\n\\end{table*}\n")
    return head + "\n".join(lines) + "\n" + foot


# =========================================================================== Table V
GK5 = "gotham_v2"      # case study on the CF-GOTHAM-T v2 bank (deviation P1); v1 claims are identical (checked below)


def table5() -> str:
    default_diff: list[str] = []

    def loso(sid, m, a, k="fdr"):
        return val(E11, f"loso|{sid}|{m}|a={a}|{k}")

    def pair(fn, nd=3):
        a, b = fn(0.1), fn(0.2)
        return f"{f(a, nd)}/{f(b, nd)}"
    ceil = val(E11, "loso|S_CF.stat|crc_bh|a=0.1|ceiling_pooled")
    rows = [
        ("IoT-23 P-LOSO, CRC-BH (stat.)", "0.1/0.2", pair(lambda a: loso("S_CF.stat", "crc_bh", a)),
         pair(lambda a: loso("S_CF.stat", "crc_bh", a, "str_pooled")), f(ceil)),
        ("IoT-23 P-LOSO, CRC-BH (fused)", "0.1/0.2", pair(lambda a: loso("S_CF.fuse_s0", "crc_bh", a)),
         pair(lambda a: loso("S_CF.fuse_s0", "crc_bh", a, "str_pooled")), f(ceil)),
        ("\\quad S$_{CF2}$ (stat.)$^\\S$", "0.1/0.2", pair(lambda a: loso("S_CF2.stat", "crc_bh", a)),
         pair(lambda a: loso("S_CF2.stat", "crc_bh", a, "str_pooled")), f(ceil)),
        ("\\quad S$_{CF2}$ (fused)$^\\S$", "0.1/0.2", pair(lambda a: loso("S_CF2.fuse_s0", "crc_bh", a)),
         pair(lambda a: loso("S_CF2.fuse_s0", "crc_bh", a, "str_pooled")), f(ceil)),
        ("IoT-23 P-WIN, CRC-BH (stat.)", "0.1/0.2", pair(lambda a: val(E11, f"pwin|S_CF.stat|crc_bh|a={a}|fdr")),
         pair(lambda a: val(E11, f"pwin|S_CF.stat|crc_bh|a={a}|str_mean")), f(ceil)),
        ("IoT-23 P-SHIFT, CRC-BH (stress)", "0.1/0.2", pair(lambda a: val(E11, f"pshift|S_CF.stat|crc_bh|a={a}|fdr")),
         pair(lambda a: val(E11, f"pshift|S_CF.stat|crc_bh|a={a}|str_mean")), f(ceil)),
        ("IoT-23 P-LOSO, BH-iid", "0.1/0.2", pair(lambda a: loso("S_CF.stat", "bh_iid", a)),
         pair(lambda a: loso("S_CF.stat", "bh_iid", a, "str_pooled")), f(ceil)),
        ("IoT-23 P-LOSO, B2 rules", "--", f(loso("S_CF.stat", "B2", 0.1)), f(loso("S_CF.stat", "B2", 0.1, "str_pooled")), f(ceil)),
    ]
    lines = [" & ".join(r) + "\\\\" for r in rows]
    lines.append("\\midrule")
    names = [("mirai_infection_17_15", "\\quad victim .17.15"), ("mirai_infection_18_10", "\\quad victim .18.10"),
             ("mirai_infection_18_17", "\\quad victim .18.17"), ("mirai_infection_18_18", "\\quad victim .18.18"),
             ("mirai_infection_20_19", "\\quad victim .20.19"), ("mirai_dos_17_15", "\\quad mirai-dos .17.15"),
             ("merlin_17_15", "\\quad Merlin .17.15"), ("benign_18_17_mirai_dos", "\\quad benign .18.17"),
             ("benign_20_19_merlin", "\\quad benign .20.19")]
    # v1 and v2 banks: are the Gotham claims identical (same R, V, TP per incident and alpha)?
    same_v1 = all((E11.get(f"{GK5}|gotham_real/{i}|a={a}|fdp") or {}).get(k) == (E11.get(f"gotham|gotham_real/{i}|a={a}|fdp") or {}).get(k)
                  for i, _ in names for a in (0.1, 0.2) for k in ("R", "V", "TP"))
    if E11.get(f"{GK5}|gotham_real/{names[0][0]}|a=0.2|fdp"):
        lines.append("Gotham 2025, CRC-BH (S$_{TON}$) & & FDP [exact CI]; claims & &\\\\")
        same = {E11.get(f"{GK5}|summary|a={a}", {}).get("lam_idx") for a in (0.1, 0.2)}
        alab = "0.1/0.2" if len(same) == 1 else "0.2"
        for inc, lab in names:
            e = E11.get(f"{GK5}|gotham_real/{inc}|a=0.2|fdp")
            if not e:
                continue
            R = e.get("R") or 0
            if R == 0:
                fdp = "no claims"
            else:
                c = e.get("ci") or [None, None]
                fdp = f"{f(e['value'], 2)} [{f(c[0], 2)}, {f(c[1], 2)}]; {R}"
                fd = e.get("fdp_default")
                if fd is not None and abs(fd - e["value"]) > 1e-9:
                    fdp += "$^\\dagger$"
                    default_diff.append(f"{lab.replace(chr(92) + 'quad ', '')} {f(fd, 2)}")
            strv = "--" if "benign" in inc else f(e.get("str"), 2)
            ceil_ = "--" if "benign" in inc else f(e.get("ceiling"), 2)
            lines.append(f"{lab} & {alab} & {fdp} & {strv} & {ceil_}\\\\")
    else:
        lines += ["Gotham victims 1--5, CRC-BH & 0.2 & " + PENDING + " & " + PENDING + " & " + PENDING + "\\\\",
                  "Gotham mirai-dos and Merlin, CRC-BH & 0.2 & " + PENDING + " & " + PENDING + " & " + PENDING + "\\\\",
                  "Gotham benign windows 1--2, CRC-BH & 0.2 & any claim: " + PENDING + " & -- & --\\\\"]
    sec = lambda proto, kind: pair(lambda a: val(E11, f"{proto}|S_CF2.{kind}|crc_bh|a={a}|fdr"))  # noqa: E731
    head = ("\\begin{table}[t]\n\\centering\n\\caption{Real Data: IoT-23 Protocols and Gotham 2025 Case-Study Incidents "
            "(Strict GT; Lenient in the Supplement)}\n\\label{tab:real}\n\\scriptsize\n\\setlength{\\tabcolsep}{1.8pt}\n"
            "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nSetting and method & $\\alpha$ & FDR or FDP & STR & ceiling\\\\\n\\midrule\n")
    foot = ("\\bottomrule\n\\end{tabular}\n\n\\vspace{2pt}\n{\\scriptsize P-LOSO: mean over the 23 held-out scenarios of the "
            "scenario FDP, $K=22$ scenarios (jackknife intervals in the text); STR pooled over windows; ceiling = "
            "$(1-\\hat\\rho_T)\\hat r_{C,T}$ pooled. $^\\S$Sensitivity (post-PREREG deviation P1): S$_{CF2}$ = the IoT-23 "
            "scorer retrained on CF-TON + CF-GOTHAM v2; S$_{CF}$ stays the pre-registered primary; S$_{CF2}$ P-WIN "
            f"{sec('pwin', 'stat')} (stat.) and {sec('pwin', 'fuse_s0')} (fused), P-SHIFT {sec('pshift', 'stat')} and "
            f"{sec('pshift', 'fuse_s0')}. P-WIN: 200 window-level 50/50 splits (valid conditionally on the "
            "dataset); P-SHIFT: 100 splits calibrating on the windows of 12 random scenarios and testing on the other 11 "
            "(no guarantee). Gotham: per-incident strict FDP with exact Clopper--Pearson 95\\% intervals, conditional on "
            "the incident under the seed-priority GT" + ("; $^\\dagger$default unit-episode rule: " + ", ".join(default_diff) if default_diff else "") +
            "; scorer S$_{TON}$ (never trained on Gotham), CRC calibrated on all 150 CF-GOTHAM-T v2 incidents"
            + (" (the original v1 bank gives the same $\\hat\\lambda$ and identical claims)" if same_v1 else "") +
            ", which gives the same $\\hat\\lambda$ at $\\alpha=0.1$ and $0.2$; no FDR claim.}\n\\end{table}\n")
    return head + "\n".join(lines) + "\n" + foot


def table_strata() -> str:
    """Supplement: per-stratum realised FDR of CRC-BH next to the marginal (DESIGN §6.2)."""
    rows = [("CF-TON", "cf_ton", "S_TON.stat", "stat."), ("CF-TON", "cf_ton", "S_TON.fuse_s0", "fused"),
            ("CF-TON-H", "cf_ton_h", "S_TON_H.stat", "stat."), ("CF-TON-H", "cf_ton_h", "S_TON_H.fuse_s0", "fused"),
            ("CF-GOTHAM v2", "cf_gotham_v2", "S_GOT2.stat", "stat."),
            ("CF-GOTHAM v1", "cf_gotham", "S_GOT.stat", "stat."), ("CF-GOTHAM v1", "cf_gotham", "S_GOT.fuse_s0", "fused")]
    strata = [("fdr", "all"), ("fdr|decoy_rich=1", "decoy-rich"), ("fdr|decoy_rich=0", "not decoy-rich"),
              ("fdr|benign_only=1", "benign-only"), ("fdr|short_chain=1", "$L\\le2$"), ("fdr|short_chain=0", "$L>2$"),
              ("fdr|N_tercile=0", "$N$ low"), ("fdr|N_tercile=1", "$N$ mid"), ("fdr|N_tercile=2", "$N$ high")]
    lines = []
    for name, ds, sid, kind in rows:
        for a in (0.1, 0.2):
            vals = [f(val(E1, f"{ds}|{sid}|crc_bh|a={a}|{k}")) if val(E1, f"{ds}|{sid}|crc_bh|a={a}|{k}") is not None
                    else "--" for k, _ in strata]
            if all(v == "--" for v in vals):
                continue
            lines.append(" & ".join([name, kind, str(a), *vals]) + "\\\\")
    head = ("\\begin{table*}[t]\n\\centering\n\\caption{Per-Stratum Realised FDR of CRC-BH (Mean over 200 Splits; "
            "Calibration on All Incidents, FDR over the Test Incidents of the Stratum). The Guarantee Is Marginal; a "
            "Stratum May Exceed $\\alpha$; CF-GOTHAM v2 = Deviation P1, v1 = Pre-Registered Original}\n\\label{tab:strata}\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular}{@{}llc" + "c" * len(strata) + "@{}}\n\\toprule\n"
            "Data & score & $\\alpha$ & " + " & ".join(lab for _, lab in strata) + "\\\\\n\\midrule\n")
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"


def table_e3() -> str:
    """Supplement: E3 deltas (fused - statistical CRC-BH) with paired bootstrap CIs, all strata and comparisons."""
    E3 = S.numbers("E3")
    comps = [("fuse_s0", "fused s0"), ("fuse_perm_s0", "placebo perm."), ("fuse_band_s0", "placebo band"),
             ("fuse_s1", "seed 1"), ("fuse_s2", "seed 2"), ("fuse_s3", "seed 3"), ("fuse_s4", "seed 4")]
    lines = []
    for role, ds, stat, pref, title in (("primary", "cf_ton_h", "S_TON_H.stat", "S_TON_H", "CF-TON-H (primary)"),
                                        ("in_template", "cf_ton", "S_TON.stat", "S_TON", "CF-TON (in-template)")):
        lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{title}}}}}\\\\")
        for c, lab in comps:
            cells = []
            for a in (0.1, 0.2):
                for strat, met in (("all", "cer_link_mean"), ("all", "str_mean"), ("unseen_type_pair", "str_mean")):
                    cells.append(fci(E3, f"{role}|{ds}|{pref}.{c}-{stat}|a={a}|{strat}|delta_{met}"))
            lines.append(" & ".join([lab, *cells]) + "\\\\")
    head = ("\\begin{table*}[t]\n\\centering\n\\caption{E3: Fused minus Statistical CRC-BH (200 Splits; Paired "
            "Incident-Pool Bootstrap 95\\% Intervals, $B=1000\\times50$): $\\Delta\\CER$ (Primary Metric), "
            "$\\Delta\\STR$, and the Pre-Registered Stratum: $\\Delta\\STR$ Restricted to Observable Transitions Whose "
            "Unit-Type Pair Has No Training Positive (STR Is the Only Stratum-Restricted Metric; $\\CER$ and FDR Are "
            "Computed over All Test Incidents and Candidates)}\n\\label{tab:e3}\n"
            "\\tiny\n\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
            " & \\multicolumn{3}{c}{$\\alpha=0.1$} & \\multicolumn{3}{c}{$\\alpha=0.2$}\\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n"
            "Comparison & $\\Delta\\CER$ & $\\Delta\\STR$ & $\\Delta\\STR$ unseen & $\\Delta\\CER$ & $\\Delta\\STR$ & $\\Delta\\STR$ unseen\\\\\n\\midrule\n")
    body = "\n".join(l.replace("multicolumn{8}", "multicolumn{7}") for l in lines)
    return head + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"


def x1_in_domain(X1: dict, a: float) -> str:
    """v1 in-domain case study (X1, exploratory): S_GOT on the original CF-GOTHAM-T S_GOT bank -> Gotham-real."""
    rows = [v for k, v in X1.items() if k.startswith("gotham_in_domain|") and k.endswith(f"|a={a}|fdp")
            and "benign" not in k]
    if not rows:
        return "--"
    wc = [v for v in rows if (v.get("R") or 0) > 0]
    if not wc:
        return f"0/{len(rows)}"
    return f"{len(wc)}/{len(rows)}; {min(v['value'] for v in wc):.2f}--{max(v['value'] for v in wc):.2f}"


def table_v2() -> str:
    """Supplement: CF-GOTHAM original (v1, pre-registered) vs regenerated (v2, deviation P1) side by side."""
    E5, X1 = S.numbers("E5"), S.numbers("X1")
    VF = {v: json.loads((S.RESULTS / "summary" / f).read_text()) for v, f in
          (("v1", "primary1_validity.json"), ("v2", "primary1_validity_v2.json"))
          if (S.RESULTS / "summary" / f).exists()}
    SIDES = (("v1", "cf_gotham", "S_GOT.stat", "gotham", "cf_gotham_t"),
             ("v2", "cf_gotham_v2", "S_GOT2.stat", "gotham_v2", "cf_gotham_t_v2"))

    def zcell(v, a):
        for c in (VF.get(v) or {}).get("cells", []):
            if c["cell"] == f"cf_gotham|alpha={a}|crc_bh|stats" and c.get("z") is not None:
                return f(c["z"], 2)
        return PENDING

    def per_alpha(fn):
        return [fn(side, a) for side in SIDES for a in (0.1, 0.2)]

    def per_side(fn):
        return [f"\\multicolumn{{2}}{{c}}{{{fn(side)}}}" for side in SIDES]

    def gsum(side, a, gk_suffix=""):
        e = E11.get(f"{side[3]}{gk_suffix}|summary|a={a}")
        if not e:
            return "--"
        rng = f"{e['fdp_min']:.2f}--{e['fdp_max']:.2f}" if e.get("fdp_min") is not None else "--"
        return f"{e['value']}/7; {rng}"

    def wauc(side, sid, ds_key):
        e = E11.get(f"{side[3]}|within_auc|{ds_key}|{sid}")
        return f(e["value"], 3) if e else "--"

    sec = []
    sec.append(("Validity and power, CRC-BH (statistical score; 200 splits)", [
        ("realised FDR [95\\% CI]", per_alpha(lambda sd, a: fci(E1, f"{sd[1]}|{sd[2]}|crc_bh|a={a}|fdr"))),
        ("validity $z$ (one-sided, Holm family)", per_alpha(lambda sd, a: zcell(sd[0], a))),
        ("pooled FDR", per_alpha(lambda sd, a: f(val(E1, f"{sd[1]}|{sd[2]}|crc_bh|a={a}|pooled_fdr")))),
        ("STR [95\\% CI]", per_alpha(lambda sd, a: fci(E1, f"{sd[1]}|{sd[2]}|crc_bh|a={a}|str_mean")))]))
    sec.append(("Chains (E5)", [
        ("composite CRC, max.-chain FDR", per_alpha(lambda sd, a: fci(E5, f"{sd[1]}|{sd[2]}|chain_composite|a={a}|maxchain_fdr"))),
        ("\\quad stratum $L\\ge5$", per_alpha(lambda sd, a: fci(E5, f"byL|{sd[1]}|chain_composite|a={a}|L=5+|maxchain_fdr"))),
        ("naive chain, max.-chain FDR", per_alpha(lambda sd, a: fci(E5, f"{sd[1]}|{sd[2]}|chain_naive|a={a}|maxchain_fdr"))),
        ("composite CRC, STR", per_alpha(lambda sd, a: f(val(E5, f"{sd[1]}|{sd[2]}|chain_composite|a={a}|str_mean"))))]))
    d = lambda sd, k, nd=3: f(val(E15, f"descriptor|{sd[1]}|{k}"), nd)  # noqa: E731
    pr = lambda sd, t, fs: f(val(E15, f"probe|{sd[1]}|caltest|{t}|{fs}|auc"), 2)  # noqa: E731
    grp = lambda sd, t, g: f(val(X1, f"probe|{sd[1]}|caltest|{t}|group|{g}|auc"), 2)  # noqa: E731
    sec.append(("Benchmark (E15, cal/test pool; no level)", [
        ("$N/r$; $\\hat r_{C,T}$; $\\hat\\rho_T$; ceiling",
         per_side(lambda sd: f"{d(sd, 'N_over_r_all', 1)}; {d(sd, 'r_CT')}; {d(sd, 'rho_T')}; {d(sd, 'ceiling')}")),
        ("oracle STR, B1 / B2", per_side(lambda sd: f"{f(val(E15, f'oracle|{sd[1]}|B1|str_mean'))} / "
                                              f"{f(val(E15, f'oracle|{sd[1]}|B2|str_mean'))}")),
        ("probe AUC (i) / (ii), all features", per_side(lambda sd: f"{pr(sd, 'i', 'all')} / {pr(sd, 'ii', 'all')}")),
        ("\\quad timing/position only", per_side(lambda sd: f"{pr(sd, 'i', 'timing')} / {pr(sd, 'ii', 'timing')}")),
        ("\\quad fractional seconds only (X1)", per_side(lambda sd: f"{grp(sd, 'i', 'fractional_seconds')} / "
                                                               f"{grp(sd, 'ii', 'fractional_seconds')}")),
        ("\\quad inter-record gaps only (X1)", per_side(lambda sd: f"{grp(sd, 'i', 'gaps')} / {grp(sd, 'ii', 'gaps')}"))]))
    sec.append(("Gotham case study (E11; CRC-BH calibrated on all 150 CF-GOTHAM-T incidents)", [
        ("S$_{TON}$: $\\hat\\lambda$ index; cal.\\ incidents with claims",
         per_alpha(lambda sd, a: f"{E11.get(f'{sd[3]}|summary|a={a}', {}).get('lam_idx', '--')}; "
                                 f"{E11.get(f'{sd[3]}|summary|a={a}', {}).get('cal_incidents_with_claims', '--')}")),
        ("S$_{TON}$: attack incidents with claims; FDP", per_alpha(lambda sd, a: gsum(sd, a))),
        ("S$_{TON}$ within-incident AUC, bank / Gotham",
         per_side(lambda sd: f"{wauc(sd, 'S_TON.stat', sd[4])} / {wauc(sd, 'S_TON.stat', 'gotham_real')}")),
        ("in-domain scorer within-incident AUC, bank / Gotham",
         [f"\\multicolumn{{2}}{{c}}{{S$_{{GOT}}$ {wauc(SIDES[0], 'S_GOT.stat', 'cf_gotham_t')} / "
          f"{f(val(X1, 'gotham|within_auc|gotham_real|S_GOT'), 3)}}}",
          f"\\multicolumn{{2}}{{c}}{{S$_{{GOT2}}$ {f(val(E11, 'gotham_v2_sgot2|within_auc|cf_gotham_t_v2|S_GOT2.stat'), 3)} / "
          f"{f(val(E11, 'gotham_v2_sgot2|within_auc|gotham_real|S_GOT2.stat'), 3)}}}"]),
        ("in-domain CRC-BH: attack incidents with claims; FDP",
         [x1_in_domain(X1, a) for a in (0.1, 0.2)] + [gsum(SIDES[1], a, "_sgot2") for a in (0.1, 0.2)]),
        ("two-sample AUC on $\\phi$, bank vs Gotham",
         per_side(lambda sd: f(val(E11, f"{sd[3]}|two_sample_auc|{sd[4]}->gotham_real"), 3)))]))
    lines = []
    for title, rows in sec:
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{title}}}}}\\\\")
        for lab, cells in rows:
            lines.append(" & ".join([lab, *cells]) + "\\\\")
        lines.append("\\addlinespace[2pt]")
    lines = lines[:-1]
    n1, n2 = VF.get("v1", {}), VF.get("v2", {})
    head = ("\\begin{table*}[t]\n\\centering\n\\caption{CF-GOTHAM Side by Side: Pre-Registered Original (v1, Scorer "
            "S$_{GOT}$) and Regenerated Benchmark (v2, Fractional-Second Timestamps, Retrained Scorer S$_{GOT2}$; "
            "Post-PREREG Deviation P1)}\n\\label{tab:v1v2}\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
            " & \\multicolumn{2}{c}{v1 (original, pre-registered)} & \\multicolumn{2}{c}{v2 (deviation P1)}\\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n"
            "Quantity & $\\alpha=0.1$ & $\\alpha=0.2$ & $\\alpha=0.1$ & $\\alpha=0.2$\\\\\n\\midrule\n")
    foot = ("\\bottomrule\n\\end{tabular}\n\n\\vspace{2pt}\n{\\scriptsize Means over 200 splits ($K=100$ calibration, "
            "100 test incidents) with incident-pool cluster-bootstrap 95\\% intervals ($B=1000\\times50$). Validity "
            "$z=(\\overline{\\FDR}-\\alpha)/\\mathrm{SE}_{boot}$; the validity family (16 cells, Holm at 0.05) has "
            f"{n1.get('n_cells_run', '?')} cells run and {n1.get('n_violations', '?')} violations with v1 "
            f"(max.\\ $z$ {f(n1.get('max_standardised_excess'), 2)}) and {n2.get('n_cells_run', '?')} cells run "
            f"and {n2.get('n_violations', '?')} violations with v2 (max.\\ $z$ "
            f"{f(n2.get('max_standardised_excess'), 2)}); the CF-GOTHAM fused cells are missing in both "
            "(CF-GOTHAM fusion was dropped). Probes: pooled out-of-fold AUC, (i) campaign vs decoy units, (ii) injected "
            "D3 look-alikes vs native background units; X1 rows are exploratory single-group probes. Case study: S$_{TON}$ "
            "claims on the 9 Gotham-real incidents; the in-domain rows are exploratory (S$_{GOT}$ within-incident AUC on "
            "Gotham and CRC with S$_{GOT}$ on the original CF-GOTHAM-T S$_{GOT}$ bank from X1; CRC with S$_{GOT2}$ on the "
            "CF-GOTHAM-T v2 S$_{GOT2}$ bank); an FDP of 0.00 is a victim with a single, true claim.}\n\\end{table*}\n")
    return head + "\n".join(lines) + "\n" + foot


def main() -> None:
    out = S.FIG_DIR
    (out / "table_s_strata.tex").write_text(table_strata())
    (out / "table_s_e3.tex").write_text(table_e3())
    (out / "table4_main.tex").write_text(table4(False))
    (out / "table4_main_ci.tex").write_text(table4(True) + "\n" + table4_cer())
    (out / "table3_data.tex").write_text(table3())
    (out / "table5_real.tex").write_text(table5())
    (out / "table_s_v1v2.tex").write_text(table_v2())
    print("tables written")


if __name__ == "__main__":
    main()
