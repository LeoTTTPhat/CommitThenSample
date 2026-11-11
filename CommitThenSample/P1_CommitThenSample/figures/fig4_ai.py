"""Fig. 4 (E3): value of the AI utility at equal budget — per-scenario relative-MSE ratio vs size-only sampling (U1),
in-domain (LOSO LP-IoT23, p = 1%) and zero-shot (LP-Edge, LP-Gotham), ρ = 5%.

Ratio per scenario = Σ_q Var_d(q)/Q_q² ÷ Σ_q Var_U1(q)/Q_q² over the scenario's content estimands (classes D, S; Q > 0),
budget-matched static designs (analytic) or the adaptive arms as deployed (500 MC reps). In-domain: geometric mean
over the 6 scenarios with a cluster-bootstrap 95% CI (scenarios resampled, estimands nested; 2,000 resamples) and
the scenarios as dots; zero-shot: the 3 scenarios individually (too few and too heterogeneous for a pooled mean).

Input: results/summary/E3_summary.parquet. Output: figures/fig4_ai.pdf (3.5 in) + fig4_ai.caption.txt + fig4_ai.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P1 = HERE.parent
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SUMMARY = P1 / "results" / "summary"
# paper-wide Okabe-Ito palette (as P1-F's figures/_style_p1f.py)
C = {"blue": "#0072B2", "orange": "#D55E00", "aqua": "#009E73", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#d9d8d4",
     "blue_l": "#56B4E9"}
ROWS = [("U0", "U0 uniform"), ("U3", "U3 $P_\\mathrm{mal}$"), ("U5", "U5 2nd moment"),
        ("U5tau0.2", "U5$\\tau$ 0.2"), ("U5tau0.5", "U5$\\tau$ 0.5"), ("U6", "U6 boost$^{a}$"),
        ("U6tau0.2", "U6$\\tau$ 0.2$^{a}$"), ("U8", "U8 Neyman"), ("U7", "U7 oracle"),
        ("B12", "B12 PPI")]
ZS = [("lp_edge_p10000", "LP-Edge", "^"), ("lp_gotham_p10000", "LP-Gotham 1%", "o"), ("lp_gotham_nat", "LP-Gotham nat.", "s")]
XMIN, XMAX = 1e-4, 4.0


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7,
        "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.0, "xtick.major.size": 2,
        "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "xtick.color": C["ink2"], "ytick.color": C["ink"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "legend.frameon": False, "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.4})


def clip(v):
    v = np.asarray(v, float)
    return np.clip(v, XMIN * 1.15, XMAX), v < XMIN * 1.15


def main() -> int:
    style()
    S = pd.read_parquet(SUMMARY / "E3_summary.parquet")
    S = S[(S["classes"] == "DS") & np.isclose(S["rho"], 0.05)]
    fig, ax = plt.subplots(figsize=(3.5, 2.4), layout="constrained")
    y = np.arange(len(ROWS))[::-1]
    facts = {}
    for yi, (des, lab) in zip(y, ROWS):
        r_in = S[(S["group"] == "in_domain") & (S["design"] == des)]
        r_zs = S[(S["group"] == "zero_shot") & (S["design"] == des)]
        if len(r_in):
            r = r_in.iloc[0]
            ps = json.loads(r["per_scenario"])
            xs, low = clip(list(ps.values()))
            ax.scatter(xs, np.full(len(xs), yi + 0.13), s=7, color=C["blue_l"], lw=0, zorder=3)
            g, lo, hi = clip([r["gm"]])[0][0], clip([r["lo"]])[0][0], clip([r["hi"]])[0][0]
            ax.plot([lo, hi], [yi + 0.13, yi + 0.13], color=C["blue"], lw=1.2, zorder=4, solid_capstyle="butt")
            ax.scatter([g], [yi + 0.13], s=16, color=C["blue"], zorder=5, edgecolor="white", lw=0.5)
            facts[f"in_domain/{des}"] = {"gm": float(r["gm"]), "ci": [float(r["lo"]), float(r["hi"])], "per_scenario": ps}
        if len(r_zs):
            ps = json.loads(r_zs.iloc[0]["per_scenario"])
            for sid, slab, mk in ZS:
                if sid in ps:
                    xv, low = clip([ps[sid]])
                    ax.scatter(xv, [yi - 0.17], s=13, marker=mk, facecolor=C["orange"] if mk != "s" else "white",
                               edgecolor=C["orange"], lw=0.8, zorder=5)
            facts[f"zero_shot/{des}"] = ps
    ax.axvline(1.0, color=C["ink2"], lw=0.8, ls="--", zorder=2)
    ax.text(1.0, len(ROWS) - 0.45, " U1", fontsize=7, color=C["ink2"], ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(-0.6, len(ROWS) - 0.2)
    ax.set_yticks(y, [r[1] for r in ROWS])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("relative-MSE ratio vs. size-only ($\\leftarrow$ better)")
    hand = [Line2D([], [], color=C["blue"], marker="o", ms=4, lw=1.2, mec="white", mew=0.5),
            Line2D([], [], color=C["blue_l"], marker="o", ms=2.6, lw=0)]
    hand += [Line2D([], [], color=C["orange"], marker=mk, ms=3.5, lw=0, mfc=C["orange"] if mk != "s" else "white", mew=0.8)
             for _, _, mk in ZS]
    labs = ["in-domain: GM, 95% CI", "in-domain scenarios"] + [z[1] for z in ZS]
    fig.legend(hand, labs, loc="outside lower center", ncol=3, handletextpad=0.3, columnspacing=0.7, borderaxespad=0.0,
               labelspacing=0.2)
    out = HERE / "fig4_ai.pdf"
    fig.savefig(out)
    fig.savefig(HERE / "fig4_ai.png", dpi=250)
    cap = {"figure": "fig4_ai.pdf", "rho": 0.05, "seeds": 3, "bootstrap": "2,000 cluster resamples over the 6 in-domain "
           "scenarios, estimands nested", "estimands": "content estimands (D detectors, S Suricata families) with Q > 0 and "
           ">= 50 eligible units", "ratio": "sum_q Var_d/Q^2 over sum_q Var_U1/Q^2 per scenario (budget-matched designs, "
           "analytic Var = sum f^2(1-pi)/pi; adaptive arms U6/U6tau: empirical MSE over R = 500 MC reps, as deployed)",
           "clipped": f"values below {XMIN:g} drawn at the left edge (exact estimation: Var = 0)",
           "footnote_a": "adaptive arm as deployed (realised-cost controller + boost); spend/target in E3_numbers.json",
           "values": facts,
           "text_for_caption": "E3: relative-MSE ratio vs size-only sampling (U1) at equal budget, rho = 5%, per test "
                               "scenario: in-domain LOSO LP-IoT23 (p = 1%, 6 scenarios; geometric mean with cluster-"
                               "bootstrap 95% CI) and zero-shot LP-Edge and LP-Gotham (individual scenarios); 3 seeds."}
    (HERE / "fig4_ai.caption.txt").write_text(json.dumps(cap, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
