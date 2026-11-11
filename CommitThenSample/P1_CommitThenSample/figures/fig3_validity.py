"""Fig. 3 (E2): (a) miscoverage vs ρ — CtS time-uniform vs fixed-time Wald, static vs adaptive design;
(b) selective abort A11 — 'compliant epochs only' vs X := 0 (+ UCS voiding); (c) relative lower-bound gap of the
sure count S and of the LCS L* vs Q (L+ = max(L*, S) takes the smaller gap).

Input: results/summary/E2_summary.parquet (scripts/90_aggregate.py --exp E2). Output: figures/fig3_validity.pdf
(IEEE double column, 7.16 in) + fig3_validity.caption.txt (facts the caption must state).

    .venv/bin/python -I P1_CommitThenSample/figures/fig3_validity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P1 = HERE.parent
sys.path.insert(0, str(P1 / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ctsample.sim.metrics import wilson  # noqa: E402

SUMMARY = P1 / "results" / "summary"
# reference categorical palette (dataviz skill, light mode), fixed order: blue, orange, aqua, yellow, magenta, green, violet, red
# paper-wide Okabe-Ito palette (as P1-F's figures/_style_p1f.py): CtS = #0072B2 everywhere
C = {"blue": "#0072B2", "orange": "#D55E00", "aqua": "#CC79A7", "yellow": "#E69F00", "magenta": "#CC79A7",
     "green": "#009E73", "violet": "#8C510A", "red": "#D55E00", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#d9d8d4"}
ALPHA = 0.05


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7.5,
        "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "xtick.color": C["ink2"],
        "ytick.color": C["ink2"], "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "legend.frameon": False, "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.4,
        "lines.linewidth": 1.4, "lines.markersize": 4})


def pooled(g: pd.DataFrame, flag: str):
    k, n = int(g[f"{flag}_k"].sum()), int(g["n"].sum())
    p, lo, hi = wilson(k, n)
    return float(p), float(lo), float(hi), k, n


def panel_a(ax, S):
    H = S[(S["part"] == "main") & (S["variant"] == "honest") & S["kind"].isin(["lp_iot23", "lp_edge"]) & (S["Q_stream"] > 0)]
    rhos = sorted(H["rho"].unique())
    x = np.arange(len(rhos))
    series = [("CtS, time-unif.", "ours_tu", None, C["blue"]),
              ("Wald, $Q\\leq20$", "wald_ft", "Q1-20", C["orange"]),
              ("Wald, $Q>20$", "wald_ft", "Q>20", C["aqua"])]
    off = {"static": -0.07, "adaptive": 0.07}
    for lab, fl, st, col in series:
        for arm, ls, mk in (("static", "-", "o"), ("adaptive", "--", "s")):
            g = H[H["arm"] == arm]
            if st is not None:
                g = g[g["stratum"] == st]
            vals = [pooled(g[np.isclose(g["rho"], r)], fl) for r in rhos]
            p = np.array([v[0] for v in vals])
            lo = np.array([v[1] for v in vals])
            hi = np.array([v[2] for v in vals])
            ax.errorbar(x + off[arm], p, yerr=[np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)], color=col, ls=ls, marker=mk, ms=3.2, lw=1.2,
                        capsize=1.5, elinewidth=0.7, mfc=col if arm == "static" else "white", mew=0.9)
    ax.axhline(ALPHA, color=C["ink2"], lw=0.7, ls=":")
    ax.text(len(rhos) - 1 + 0.25, ALPHA, r"$\alpha$", va="bottom", ha="right", fontsize=7, color=C["ink2"])
    ax.set_xticks(x, [f"{100 * r:g}" for r in rhos])
    ax.set_xlabel(r"budget $\rho$ (%)")
    ax.set_ylabel("miscoverage (two-sided)")
    ax.set_xlim(-0.35, len(rhos) - 0.65)
    top = max(0.3, ax.get_ylim()[1])
    ax.set_ylim(0, top * 1.55)
    hand = [Line2D([], [], color=c, lw=1.2, marker="o", ms=3) for _, _, _, c in series]
    hand += [Line2D([], [], color=C["ink2"], ls="-", marker="o", ms=3), Line2D([], [], color=C["ink2"], ls="--", marker="s",
                                                                              ms=3, mfc="white")]
    labs = [s[0] for s in series] + ["static", "adaptive"]
    ax.legend(hand, labs, loc="upper right", bbox_to_anchor=(1.02, 1.04), ncol=2, handlelength=1.5, borderaxespad=0.1,
              labelspacing=0.15, columnspacing=0.5, handletextpad=0.3, fontsize=7)
    ax.set_title("(a) error statements vs. Wald", loc="left")


RULES = [("void_unretained", "void\nunret.", "lcs_tu"), ("void_retained", "void\nret.", "ucs_tu"),
         ("withhold_half", "with-\nhold", "ucs_tu"), ("void_random", "rand.\nvoid", "ucs_tu")]


def panel_b(ax, S):
    A = S[(S["part"] == "a11") & (S["kind"] == "lp_iot23") & (S["Q_stream"] > 0)]
    x = np.arange(len(RULES))
    w = 0.36
    out = {}
    vals = {}
    for pre in ("comp", "x0"):
        ps, los, his = [], [], []
        for rule, _, side in RULES:
            # the side each rule attacks for 'compliant only'; X:=0 reported on the same side
            g = A[A["variant"] == f"{pre}:{rule}"]
            if side == "ucs_tu":
                g = g[g["kind_q"] == "indicator"]
            p, lo, hi, k, n = pooled(g, side)
            ps.append(p), los.append(lo), his.append(hi)
            out[f"{pre}:{rule}"] = (p, k, n)
        vals[pre] = tuple(map(np.array, (ps, los, his)))
    x0max = float(vals["x0"][0].max())
    for j, (lab, pre, col) in enumerate((("compliant only", "comp", C["red"]),
                                          (f"$X{{:=}}0$ ($\\leq${max(x0max, 0.001):.3f})", "x0", C["blue"]))):
        ps, los, his = vals[pre]
        ax.bar(x + (j - 0.5) * w, np.maximum(ps, 0.0015 if pre == "x0" else 0), w * 0.92, color=col, label=lab, zorder=3)
        ax.errorbar(x + (j - 0.5) * w, ps, yerr=[np.clip(ps - los, 0, None), np.clip(his - ps, 0, None)], fmt="none",
                    ecolor=C["ink"], elinewidth=0.6, capsize=1.2, zorder=4)
    ax.axhline(ALPHA / 2, color=C["ink2"], lw=0.7, ls=":")
    ax.text(len(RULES) - 0.45, ALPHA / 2, r"$\alpha/2$", va="bottom", ha="right", fontsize=7, color=C["ink2"])
    ax.set_xticks(x, [r[1] for r in RULES])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("time-unif. miscoverage")
    ax.set_ylim(0, max(0.1, float(vals["comp"][2].max())) * 1.42)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", borderaxespad=0.1, handlelength=1.2, fontsize=7)
    ax.set_title("(b) selective abort (A11)", loc="left")
    return out


def panel_c(ax, S):
    H = S[(S["part"] == "main") & (S["variant"] == "honest") & (S["arm"] == "static") & np.isclose(S["rho"], 0.05)
          & S["kind"].isin(["lp_iot23", "lp_edge"]) & (S["Q_stream"] > 0)].copy()
    edges = np.array([1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 1e6])
    H["bin"] = np.digitize(H["Q_stream"], edges) - 1
    rows = []
    for b, g in H.groupby("bin"):
        if len(g) < 3:
            continue
        xc = float(np.exp(np.mean(np.log(g["Q_stream"]))))
        rows.append((xc, len(g), *[np.quantile(g[c].dropna(), [0.25, 0.5, 0.75]) for c in
                                   ("gap_S_med", "gap_Lstar_med", "rel_lcb_gap_med")]))
    xs = np.array([r[0] for r in rows])
    qL = np.array([r[4] for r in rows])
    ax.fill_between(xs, qL[:, 0], qL[:, 2], color=C["blue"], alpha=0.13, lw=0)
    ax.plot(xs, qL[:, 1], color=C["blue"], lw=3.2, alpha=0.45, solid_capstyle="round", label="$L^+\\!=\\!\\max(L^*\\!,S)$")
    qS = np.array([r[2] for r in rows])
    qT = np.array([r[3] for r in rows])
    ax.plot(xs, qS[:, 1], color=C["yellow"], lw=1.0, marker="o", ms=3.2, mfc="white", mew=0.9, label="$S$")
    ax.plot(xs, qT[:, 1], color=C["green"], lw=1.0, marker="^", ms=3.2, label="$L^*$")
    ax.set_xscale("log")
    ax.set_xlabel("true total $Q$")
    ax.set_ylabel("relative gap $1-$bound$/Q$")
    ax.set_ylim(-0.02, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.04), ncol=3, borderaxespad=0.1, handlelength=1.2,
              labelspacing=0.2, columnspacing=0.6, handletextpad=0.3, fontsize=7)
    ax.set_title("(c) lower-bound gap ($\\rho=5\\%$)", loc="left")
    return [(float(r[0]), int(r[1]), [list(map(float, r[2 + k])) for k in range(3)]) for r in rows]


def main() -> int:
    style()
    S = pd.read_parquet(SUMMARY / "E2_summary.parquet")
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.05), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]}, layout="constrained")
    panel_a(axs[0], S)
    a11 = panel_b(axs[1], S)
    binsC = panel_c(axs[2], S)
    out = HERE / "fig3_validity.pdf"
    fig.savefig(out)
    fig.savefig(HERE / "fig3_validity.png", dpi=220)
    # caption facts
    H = S[(S["part"] == "main") & (S["variant"] == "honest") & S["kind"].isin(["lp_iot23", "lp_edge"])]
    reps = H.groupby(["kind", "rho"])["n"].max().to_dict()
    A = S[(S["part"] == "a11") & (S["kind"] == "lp_iot23")]
    n_streams = H.groupby("kind")["stream"].nunique().to_dict()
    cap = {
        "figure": "fig3_validity.pdf", "alpha": ALPHA, "alpha_per_side": ALPHA / 2,
        "R": {f"{k}/rho={r:g}": int(v) for (k, r), v in reps.items()},
        "streams": n_streams, "rho_panel_a": sorted(map(float, H["rho"].unique())),
        "panel_a": "pooled over (stream, estimand) pairs with Q > 0 of LP-IoT23 and LP-Edge; CtS = L+ lower side OR U+ "
                   "upper side wrong at any epoch (UCS on indicator estimands); Wald = final epoch; Wilson 95% intervals",
        "panel_b": f"LP-IoT23 ({A['stream'].nunique()} streams), rho = 5%, static and adaptive arms pooled, R = "
                   f"{int(A['n'].max()) if len(A) else 'n/a'} per arm; side attacked by each rule: void unretained -> lower, "
                   "others -> upper (UCS); X := 0 voids the UCS from the first withheld eligible unit",
        "panel_b_values": {k: {"miss": v[0], "k": v[1], "n": v[2]} for k, v in a11.items()},
        "panel_c": "static design, rho = 5%, LP-IoT23 and LP-Edge pairs with Q > 0, binned by Q (log bins); line = median "
                   "over pairs of the per-pair median (first 200 reps, final epoch) of 1 - S/Q, 1 - L*/Q (clipped at 1) and "
                   "1 - L+/Q; band = interquartile range over pairs",
        "panel_c_bins": [{"Q_geo": b[0], "n_pairs": b[1], "S": b[2][0], "Lstar": b[2][1], "Lplus": b[2][2]} for b in binsC],
        "text_for_caption": "E2: (a) two-sided miscoverage vs rho of CtS statements (time-uniform) and of fixed-time Wald "
                            "intervals, static (solid) and adaptive (dashed) designs; (b) time-uniform miscoverage under "
                            "selective abort A11 (rho = 5%); (c) relative gap of the sure count S and of the LCS L* at the "
                            "final epoch vs Q (rho = 5%, static). LP-IoT23 and LP-Edge, R = 1,000 (LP-IoT23, rho = 5%) / 500, "
                            "alpha = 0.05, Wilson 95% intervals."}
    (HERE / "fig3_validity.caption.txt").write_text(json.dumps(cap, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
