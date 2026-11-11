"""Fig. 3 rev1 (E2, review R1-M3): as fig3_validity.py, plus the VALID fixed-time comparator.

(a) two-sided miscoverage vs rho: CtS time-uniform, CtS-type fixed-time bound with one pre-registered eta (Chernoff
    lower / Fan upper at the final epoch), and fixed-time Wald (Q <= 20, Q > 20); static solid, adaptive dashed;
(b) selective abort A11 (unchanged);
(c) relative lower-bound gap vs Q at the final epoch: S, the anytime L* and L+ = max(L*, S), and the fixed-time bounds
    with one eta* (Chernoff) and the Bernstein bound with the floor variance (raw bounds, clipped at 1).

Input: results/summary/E2_summary_rev1.parquet (scripts/90_aggregate.py --exp E2 --rev rev1). Output:
figures/fig3_validity_rev1.pdf (+ .png preview, .caption.txt sidecar). figures/fig3_validity.pdf is not touched.

    .venv/bin/python -I P1_CommitThenSample/figures/fig3_validity_rev1.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P1 = HERE.parent
sys.path.insert(0, str(P1 / "src"))
_spec = importlib.util.spec_from_file_location("fig3_rev0", HERE / "fig3_validity.py")
F0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F0)                      # style(), panel_b(), palette C (Agg backend)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ctsample.sim.metrics import wilson  # noqa: E402

SUMMARY = P1 / "results" / "summary"
C = dict(F0.C, sky="#56B4E9", brown="#8C510A")   # Okabe-Ito sky blue / brown for the new series (validated, dataviz)
ALPHA = F0.ALPHA


def pooled_n(g: pd.DataFrame, flag: str):
    """Pooled miss with the flag's own n (fixed-time flags) or the pair n."""
    if f"{flag}_n" in g:
        gg = g[g[f"{flag}_n"].fillna(0) > 0]
        k, n = int(gg[f"{flag}_k"].sum()), int(gg[f"{flag}_n"].sum())
    else:
        k, n = int(g[f"{flag}_k"].sum()), int(g["n"].sum())
    p, lo, hi = wilson(k, n)
    return float(p), float(lo), float(hi), k, n


def panel_a(ax, S):
    H = S[(S["part"] == "main") & (S["variant"] == "honest") & S["kind"].isin(["lp_iot23", "lp_edge"]) & (S["Q_stream"] > 0)]
    rhos = sorted(H["rho"].unique())
    x = np.arange(len(rhos))
    series = [("CtS, time-unif.", "ours_tu", None, C["blue"]),
              ("one $\\eta^*$, final ep.", "ft1_2s", None, C["sky"]),
              ("Wald, $Q\\leq20$", "wald_ft", "Q1-20", C["orange"]),
              ("Wald, $Q>20$", "wald_ft", "Q>20", C["aqua"])]
    off = {"static": -0.07, "adaptive": 0.07}
    vals_out = {}
    for lab, fl, st, col in series:
        for arm, ls, mk in (("static", "-", "o"), ("adaptive", "--", "s")):
            g = H[H["arm"] == arm]
            if st is not None:
                g = g[g["stratum"] == st]
            vals = [pooled_n(g[np.isclose(g["rho"], r)], fl) for r in rhos]
            p = np.array([v[0] for v in vals])
            lo = np.array([v[1] for v in vals])
            hi = np.array([v[2] for v in vals])
            vals_out[f"{fl}/{st or 'Q>0'}/{arm}"] = [{"rho": float(r), "miss": float(v[0]), "k": v[3], "n": v[4]}
                                                     for r, v in zip(rhos, vals)]
            ax.errorbar(x + off[arm], p, yerr=[np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)], color=col, ls=ls,
                        marker=mk, ms=3.2, lw=1.2, capsize=1.5, elinewidth=0.7, mfc=col if arm == "static" else "white",
                        mew=0.9)
    ax.axhline(ALPHA, color=C["ink2"], lw=0.7, ls=":")
    ax.text(len(rhos) - 1 + 0.25, ALPHA, r"$\alpha$", va="bottom", ha="right", fontsize=7, color=C["ink2"])
    ax.set_xticks(x, [f"{100 * r:g}" for r in rhos])
    ax.set_xlabel(r"budget $\rho$ (%)")
    ax.set_ylabel("miscoverage (two-sided)")
    ax.set_xlim(-0.35, len(rhos) - 0.65)
    top = max(0.3, ax.get_ylim()[1])
    ax.set_ylim(0, top * 1.62)
    hand = [Line2D([], [], color=c, lw=1.2, marker="o", ms=3) for _, _, _, c in series]
    hand += [Line2D([], [], color=C["ink2"], ls="-", marker="o", ms=3),
             Line2D([], [], color=C["ink2"], ls="--", marker="s", ms=3, mfc="white")]
    labs = [s[0] for s in series] + ["static", "adaptive"]
    ax.legend(hand, labs, loc="upper right", bbox_to_anchor=(1.02, 1.04), ncol=2, handlelength=1.5, borderaxespad=0.1,
              labelspacing=0.15, columnspacing=0.5, handletextpad=0.3, fontsize=6.5)
    ax.set_title("(a) error statements vs. Wald", loc="left")
    return vals_out


def panel_c(ax, S):
    H = S[(S["part"] == "main") & (S["variant"] == "honest") & (S["arm"] == "static") & np.isclose(S["rho"], 0.05)
          & S["kind"].isin(["lp_iot23", "lp_edge"]) & (S["Q_stream"] > 0)].copy()
    edges = np.array([1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 1e6])
    H["bin"] = np.digitize(H["Q_stream"], edges) - 1
    cols = ("gap_S_med", "gap_Lstar_med", "rel_lcb_gap_med", "gapraw_ft1_med", "gapraw_ftb_med")
    rows = []
    for b, g in H.groupby("bin"):
        if len(g) < 3:
            continue
        xc = float(np.exp(np.mean(np.log(g["Q_stream"]))))
        rows.append((xc, len(g), *[np.quantile(g[c].dropna(), [0.25, 0.5, 0.75]) for c in cols]))
    xs = np.array([r[0] for r in rows])
    q = {c: np.array([r[2 + i] for r in rows]) for i, c in enumerate(cols)}
    ax.fill_between(xs, q["rel_lcb_gap_med"][:, 0], q["rel_lcb_gap_med"][:, 2], color=C["blue"], alpha=0.13, lw=0)
    ax.plot(xs, q["rel_lcb_gap_med"][:, 1], color=C["blue"], lw=3.2, alpha=0.45, solid_capstyle="round",
            label="$L^+\\!=\\!\\max(L^*\\!,S)$")
    ax.plot(xs, q["gap_S_med"][:, 1], color=C["yellow"], lw=1.0, marker="o", ms=3.2, mfc="white", mew=0.9, label="$S$")
    ax.plot(xs, q["gap_Lstar_med"][:, 1], color=C["green"], lw=1.0, marker="^", ms=3.2, label="$L^*$ (anytime)")
    ax.plot(xs, q["gapraw_ft1_med"][:, 1], color=C["sky"], lw=1.0, ls="--", marker="s", ms=3.0,
            label="one $\\eta^*$ (final ep.)")
    ax.plot(xs, q["gapraw_ftb_med"][:, 1], color=C["brown"], lw=1.0, ls=":", marker="D", ms=2.8, label="Bernstein")
    ax.set_xscale("log")
    ax.set_xlabel("true total $Q$")
    ax.set_ylabel("relative gap $1-$bound$/Q$")
    ax.set_ylim(-0.02, 1.42)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.05), ncol=3, borderaxespad=0.1, handlelength=1.4,
              labelspacing=0.15, columnspacing=0.5, handletextpad=0.3, fontsize=6.3)
    ax.set_title("(c) lower-bound gap ($\\rho=5\\%$)", loc="left")
    return [{"Q_geo": float(r[0]), "n_pairs": int(r[1]), **{c: list(map(float, r[2 + i])) for i, c in enumerate(cols)}}
            for r in rows]


def main() -> int:
    F0.style()
    S = pd.read_parquet(SUMMARY / "E2_summary_rev1.parquet")
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.05), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]}, layout="constrained")
    va = panel_a(axs[0], S)
    a11 = F0.panel_b(axs[1], S)
    binsC = panel_c(axs[2], S)
    out = HERE / "fig3_validity_rev1.pdf"
    fig.savefig(out)
    fig.savefig(HERE / "fig3_validity_rev1.png", dpi=220)
    H = S[(S["part"] == "main") & (S["variant"] == "honest") & S["kind"].isin(["lp_iot23", "lp_edge"])]
    reps = H.groupby(["kind", "rho"])["n"].max().to_dict()
    A = S[(S["part"] == "a11") & (S["kind"] == "lp_iot23")]
    cap = {
        "figure": "fig3_validity_rev1.pdf (rev1 variant of fig3_validity.pdf; source results/summary/E2_summary_rev1.parquet)",
        "alpha": ALPHA, "alpha_per_side": ALPHA / 2,
        "R": {f"{k}/rho={r:g}": int(v) for (k, r), v in reps.items()},
        "streams": H.groupby("kind")["stream"].nunique().to_dict(),
        "panel_a": "pooled over (stream, estimand) pairs with Q > 0 of LP-IoT23 and LP-Edge; CtS time-unif. = L+ lower side "
                   "OR U+ upper side wrong at any epoch (UCS on indicator estimands); one eta* = the same supermartingales "
                   "with ONE pre-registered eta evaluated once at the final epoch (L = Qhat - l/eta* - eta* V/2, "
                   "eta* = sqrt(2 l/V_ref); U = Qhat + (l + W(eta_U))/eta_U, eta_U = min(0.999/cbar, eta*); valid at that "
                   "epoch only); Wald = final epoch; Wilson 95% intervals",
        "panel_a_values": va,
        "panel_b": f"unchanged from rev0: LP-IoT23 ({A['stream'].nunique()} streams), rho = 5%, static and adaptive arms "
                   "pooled; side attacked by each rule: void unretained -> lower, others -> upper (UCS)",
        "panel_b_values": {k: {"miss": v[0], "k": v[1], "n": v[2]} for k, v in a11.items()},
        "panel_c": "static design, rho = 5%, LP-IoT23 and LP-Edge pairs with Q > 0, binned by Q (log bins); line = median over "
                   "pairs of the per-pair median (first 200 reps, final epoch) of 1 - S/Q, 1 - L*/Q (clipped at 1), "
                   "1 - L+/Q (band = IQR over pairs), and of the fixed-time bounds alone 1 - L/Q (clipped at 1): one eta* "
                   "(Chernoff) and Bernstein with the floor-based variance bound kappa Q",
        "panel_c_bins": binsC,
        "text_for_caption": "E2: (a) two-sided miscoverage vs rho of the CtS statements (time-uniform), of the same bounds "
                            "with one pre-registered eta evaluated at the final epoch only (valid at that time) and of "
                            "fixed-time Wald intervals, static (solid) and adaptive (dashed) designs; (b) time-uniform "
                            "miscoverage under selective abort A11 (rho = 5%); (c) relative gap at the final epoch vs Q of "
                            "the sure count S, the anytime L* and L+, and the fixed-time bounds (one eta*, Bernstein) "
                            "(rho = 5%, static). LP-IoT23 and LP-Edge, R = 1,000 (LP-IoT23, rho = 5%) / 500, alpha = 0.05, "
                            "Wilson 95% intervals."}
    (HERE / "fig3_validity_rev1.caption.txt").write_text(json.dumps(cap, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
