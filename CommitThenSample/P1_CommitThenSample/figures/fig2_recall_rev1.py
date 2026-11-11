"""Fig. 2 rev1 (E1): content-evidence recall vs REALISED storage, by incident-size stratum, LP-IoT23 (in-domain,
p = 1%) and LP-Edge (zero-shot). Review R3-M2(g) "Fig. 2 fairness":

* the main CtS curve is the budget-matched CtS-AI design (static U5τ0.2_bm: expected spend = B̄E exactly), drawn
  with its 95% cluster-bootstrap band (scripts/90_aggregate.py --rev rev1);
* the adaptive as-deployed arm (U6τ: realised-cost controller + boost) is drawn separately (thin dotted line, open
  markers) at its REALISED spend ρ × spend/target, with the factor annotated where it exceeds 1.1 — so no policy is
  credited with storage it did not use;
* baselines (rev1 versions): B13 and B4 online with a per-epoch budget (budget-exact, no hindsight), B6 on the 36
  strict audit sids, B3 scored at byte level, CtS-Size (B10, π ∝ √s), DLT threshold sampling (π = min(1, s/z)),
  CtS-Unif. Everything except the adaptive arm spends B̄E exactly or in expectation.

Input: results/summary/E1_summary_rev1.parquet, E1_policies_rev1.parquet. Output: figures/fig2_recall_rev1.pdf
(IEEE single column, 3.5 in), .png preview and fig2_recall_rev1.caption.txt (caption facts). The rev0 figure
figures/fig2_recall.pdf is kept.

    .venv/bin/python -I P1_CommitThenSample/figures/fig2_recall_rev1.py
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
# paper-wide Okabe-Ito palette (as fig2_recall.py); CtS variants solid, non-CtS baselines dashed
C = {"cts_ai": "#0072B2", "cts_size": "#56B4E9", "cts_unif": "#009E73", "dlt": "#999999", "b3": "#D55E00",
     "b4": "#CC79A7", "b6": "#E69F00", "b13": "#8C510A", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#d9d8d4"}
POL = [("CtS-AI-static", "CtS-AI (matched)", C["cts_ai"], "-", "o"),
       ("CtS-AI", "CtS-AI adaptive @ realised", C["cts_ai"], ":", "o"),
       ("CtS-Size", "CtS-Size ($\\sqrt{s}$)", C["cts_size"], "-", "s"),
       ("DLT", "DLT ($s/z$)", C["dlt"], "-", "d"),
       ("CtS-Unif", "CtS-Unif", C["cts_unif"], "-", "D"),
       ("B13", "B13 online", C["b13"], "--", "v"),
       ("B4", "B4 online", C["b4"], "--", "^"),
       ("B6", "B6 (36 sids)", C["b6"], "--", "P"),
       ("B3", "B3 TM (bytes)", C["b3"], "--", "X")]
STRATA = [("k<=20", "$k\\leq20$"), ("k21-100", "$21\\leq k\\leq100$"), ("k>100", "$k>100$")]
ROWS = [("LP-IoT23 1%", "LP-IoT23"), ("LP-Edge", "LP-Edge")]


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7,
        "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
        "ytick.major.size": 2, "xtick.minor.size": 0, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False, "axes.grid": True, "grid.color": C["grid"],
        "grid.linewidth": 0.4, "lines.linewidth": 1.0, "lines.markersize": 3})


def main() -> int:
    style()
    S = pd.read_parquet(SUMMARY / "E1_summary_rev1.parquet")
    rhos = sorted(S["rho"].unique())
    fig, axs = plt.subplots(2, 3, figsize=(3.5, 2.75), sharex=True, sharey=True, layout="constrained")
    nI, spend_note = {}, {}
    for i, (grp, glab) in enumerate(ROWS):
        G = S[S["group"] == grp]
        for j, (st, slab) in enumerate(STRATA):
            ax = axs[i, j]
            ax.set_xscale("log")
            n = int(G[f"n_{st}"].max()) if len(G) else 0
            nI[f"{glab}/{st}"] = n
            for key, lab, col, ls, mk in POL:
                g = G[G["policy"] == key].sort_values("rho")
                if g.empty:
                    continue
                x = 100 * g["rho"].to_numpy(float)
                if key == "CtS-AI":
                    sp = g["spend_over_target_mean"].to_numpy(float)
                    x = x * sp
                    spend_note[f"{glab}"] = {f"{r:g}": float(v) for r, v in zip(g["rho"], sp)}
                y = g[f"content_{st}"].to_numpy(float)
                if key == "CtS-AI-static" and f"content_{st}_lo" in g:
                    lo, hi = g[f"content_{st}_lo"].to_numpy(float), g[f"content_{st}_hi"].to_numpy(float)
                    ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
                ax.plot(x, y, color=col, ls=ls, marker=mk, ms=2.6, lw=0.8 if key == "CtS-AI" else 1.0,
                        mfc="white" if key == "CtS-AI" else col, mew=0.7, zorder=3 if key.startswith("CtS-AI") else 2)
                if key == "CtS-AI" and j == 0:
                    offs = [(-9, 1), (-9, 5), (7, -4), (7, 2)]
                    for t, (xx, yy, f) in enumerate(zip(x, y, g["spend_over_target_mean"].to_numpy(float))):
                        if f > 1.1:
                            ax.annotate(f"{f:.1f}×", (xx, yy), textcoords="offset points", xytext=offs[t % 4],
                                        ha="center", va="center", fontsize=5, color=col)
            ax.set_ylim(-0.04, 1.06)
            ax.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
            ax.set_xlim(0.8, 25)
            if i == 0:
                ax.set_title(slab, pad=2)
            if j == 0:
                ax.set_ylabel(glab)

    hand = [Line2D([], [], color=c, ls=ls, marker=m, ms=2.8, lw=1.0, mfc="white" if k == "CtS-AI" else c, mew=0.7)
            for k, _, c, ls, m in POL]
    fig.legend(hand, [p[1] for p in POL], loc="outside upper center", ncol=3, handlelength=1.9,
               columnspacing=0.6, labelspacing=0.15, borderaxespad=0.0, handletextpad=0.3)
    fig.supylabel("content-evidence recall", fontsize=7)
    fig.supxlabel("storage actually spent (% of full-capture bytes)", fontsize=7)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)
    out = HERE / "fig2_recall_rev1.pdf"
    fig.savefig(out)
    fig.savefig(HERE / "fig2_recall_rev1.png", dpi=250)
    g5 = S[np.isclose(S["rho"], 0.05)]
    vals = {}
    for grp, glab in ROWS:
        for key, *_ in POL:
            r = g5[(g5["group"] == grp) & (g5["policy"] == key)]
            if len(r):
                r = r.iloc[0]
                vals[f"{glab}/{key}"] = {st: [float(r[f"content_{st}"]), float(r.get(f"content_{st}_lo", np.nan)),
                                              float(r.get(f"content_{st}_hi", np.nan))] for st, _ in STRATA}
                vals[f"{glab}/{key}"]["spend_over_target"] = float(r.get("spend_over_target_mean", np.nan))
    cap = {"figure": "fig2_recall_rev1.pdf", "rho": [float(r) for r in rhos], "incidents_per_panel": nI,
           "x_axis": "storage actually spent = rho x spend/target (log scale); every policy except the adaptive CtS-AI "
                     "arm spends B-bar E exactly or in expectation, so it sits at its nominal rho",
           "main_cts_curve": "CtS-AI budget-matched static design (U5tau0.2_bm), band = 95% two-stage cluster-bootstrap "
                             "interval (clusters: streams for LP-IoT23, attack classes for LP-Edge)",
           "adaptive_spend_over_target": spend_note,
           "rho5_values_[point,lo,hi]": vals,
           "baselines": "B13/B4 online with a per-epoch budget (greedy top-k within B-bar + carried deficit; B13 keeps a "
                        "keyed reserve max(phi0, .)); B6 on the 36 strict audit sids; B3 Time Machine scored at byte level "
                        "(content evidence iff the unit's first payload packet lies within the first N bytes of its "
                        "connection); CtS-Size = B10 (pi ~ sqrt(s)); DLT = threshold sampling pi = min(1, s/z)",
           "text_for_caption": "E1 (rev1): content-evidence recall vs storage actually spent, by incident size k, "
                               "LP-IoT23 (6 LOSO folds, p = 1%) and LP-Edge (zero-shot). Main CtS curve = budget-matched "
                               "CtS-AI with 95% cluster-bootstrap band; the adaptive arm (dotted, open markers) is plotted "
                               "at its realised spend (factor annotated when > 1.1). Online baselines without hindsight."}
    (HERE / "fig2_recall_rev1.caption.txt").write_text(json.dumps(cap, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
