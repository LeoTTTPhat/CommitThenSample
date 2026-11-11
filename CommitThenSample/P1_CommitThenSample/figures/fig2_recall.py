"""Fig. 2 (E1): content-evidence recall vs budget ρ by incident-size stratum, LP-IoT23 (in-domain, p = 1%) and
LP-Edge (zero-shot), at equal storage.

Input: results/summary/E1_summary.parquet (scripts/90_aggregate.py --exp E1). Output: figures/fig2_recall.pdf (IEEE
single column, 3.5 in) + fig2_recall.caption.txt (facts the caption must state) + fig2_recall.png (preview).

    .venv/bin/python -I P1_CommitThenSample/figures/fig2_recall.py
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
# paper-wide policy palette (Okabe-Ito; identical hex values to P1-F's figures/_style_p1f.py, copied not imported)
C = {"cts_ai": "#0072B2", "cts_size": "#56B4E9", "cts_unif": "#009E73", "b3": "#D55E00", "b4": "#CC79A7",
     "b6": "#E69F00", "b13": "#8C510A", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#d9d8d4"}
# policy -> (label, colour, linestyle, marker); CtS variants solid, non-CtS baselines dashed (secondary encoding)
POL = [("CtS-AI", "CtS-AI", C["cts_ai"], "-", "o"),
       ("CtS-AI-static", "CtS-AI matched", C["cts_ai"], ":", "o"),
       ("CtS-Size", "CtS-Size", C["cts_size"], "-", "s"),
       ("CtS-Unif", "CtS-Unif", C["cts_unif"], "-", "D"),
       ("B13", "B13", C["b13"], "-", "v"),
       ("B4", "B4", C["b4"], "--", "^"),
       ("B6", "B6", C["b6"], "--", "P"),
       ("B3", "B3 (TM)", C["b3"], "--", "X")]
STRATA = [("k<=20", "$k\\leq20$"), ("k21-100", "$21\\leq k\\leq100$"), ("k>100", "$k>100$")]
ROWS = [("LP-IoT23 1%", "LP-IoT23"), ("LP-Edge", "LP-Edge")]


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7,
        "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
        "ytick.major.size": 2, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "xtick.color": C["ink2"],
        "ytick.color": C["ink2"], "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "legend.frameon": False, "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.4,
        "lines.linewidth": 1.1, "lines.markersize": 3})


def main() -> int:
    style()
    S = pd.read_parquet(SUMMARY / "E1_summary.parquet")
    rhos = sorted(S["rho"].unique())
    fig, axs = plt.subplots(2, 3, figsize=(3.5, 2.4), sharex=True, sharey=True, layout="constrained")
    nI = {}
    for i, (grp, glab) in enumerate(ROWS):
        G = S[S["group"] == grp]
        for j, (st, slab) in enumerate(STRATA):
            ax = axs[i, j]
            n = int(G[f"n_{st}"].max()) if len(G) else 0
            nI[f"{glab}/{st}"] = n
            for key, lab, col, ls, mk in POL:
                g = G[G["policy"] == key].sort_values("rho")
                if g.empty:
                    continue
                y = g[f"content_{st}"].to_numpy(float)
                ax.plot(np.arange(len(g)), y, color=col, ls=ls, marker=mk, ms=2.6, lw=1.0,
                        mfc="white" if key == "CtS-AI-static" else col, mew=0.7)
            ax.set_ylim(-0.04, 1.04)
            ax.set_xticks(np.arange(len(rhos)), [f"{100 * r:g}" for r in rhos])
            if i == 0:
                ax.set_title(slab, pad=2)
            if j == 0:
                ax.set_ylabel(glab)
            if i == 1:
                ax.set_xlabel(r"$\rho$ (%)", labelpad=1)
    hand = [Line2D([], [], color=c, ls=ls, marker=m, ms=2.8, lw=1.0, mfc="white" if k == "CtS-AI-static" else c, mew=0.7)
            for k, _, c, ls, m in POL]
    fig.legend(hand, [p[1] for p in POL], loc="outside upper center", ncol=4, handlelength=1.9,
               columnspacing=0.6, labelspacing=0.15, borderaxespad=0.0, handletextpad=0.3)
    fig.supylabel("content-evidence recall", fontsize=7)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)
    out = HERE / "fig2_recall.pdf"
    fig.savefig(out)
    fig.savefig(HERE / "fig2_recall.png", dpi=250)
    pol = pd.read_parquet(SUMMARY / "E1_policies.parquet")
    ad = pol[pol["policy"] == "CtS-AI"]
    cap = {"figure": "fig2_recall.pdf", "rho": [float(r) for r in rhos], "incidents_per_panel": nI,
           "streams": {"LP-IoT23": "6 LOSO folds at target p = 1% (realised 0.41-1.05%)",
                       "LP-Edge": "realised prevalence 0.10%, 0.97%, 3.25% (zero-shot models)"},
           "recall": "P(at least one content-bearing unit of the incident retained whole); exact for static designs and "
                     "deterministic baselines; CtS-AI (adaptive arm) = mean over R = 200 reps x 3 seeds of the exact "
                     "full-unit simulation; budget-matched policies spend B-bar E exactly or in expectation",
           "adaptive_spend_over_target": {g: [float(x) for x in ad[ad["group"] == g].groupby("rho")["spend_over_target"].mean()]
                                          for g in ("LP-IoT23 1%", "LP-Edge")},
           "text_for_caption": "E1: content-evidence recall vs budget rho by incident size k (content-bearing units), "
                               "LP-IoT23 (6 LOSO folds, p = 1%, in-domain) and LP-Edge (zero-shot); analytic for static "
                               "designs and baselines, R = 200 reps x 3 seeds for the adaptive CtS-AI arm. On LP-Edge the "
                               "adaptive arm overspends (spend/target in the sidecar), the matched static design does not."}
    (HERE / "fig2_recall.caption.txt").write_text(json.dumps(cap, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
