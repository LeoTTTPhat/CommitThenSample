#!/usr/bin/env python
"""Fig. 6 -- chains (E5): maximal-chain FDR by GT chain length L for composite CRC (Alg. 5), naive composition of the
edge-CRC set and the coherent-factuality-style rule (chain-RT) on CF-TON and CF-GOTHAM (statistical scorers), and the
Example 1 simulation (naive composition inflates the maximal-chain FDR towards min(1, alpha L / (1 - alpha))).
Owner: P2-G.

  .venv/bin/python -I P2_CertiChain/figures/fig6_chains.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from certichain.eval import experiments as X  # noqa: E402

METHODS = [("chain_composite", "composite CRC (ours)", S.BLUE, "o"), ("chain_naive", "chain-naive", S.ORANGE, "s"),
           ("chain_rt", "chain-RT", S.AQUA, "^")]
ALPHAS = (0.1, 0.2)
SIDS = {"cf_ton": "S_TON.stat", "cf_gotham": "S_GOT.stat", "cf_gotham_v2": "S_GOT2.stat"}


def main() -> None:
    S.setup()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    n5 = S.numbers("E5")
    byl = pl.read_parquet(X.exp_root("E5") / "by_length.parquet")
    ex = pl.read_parquet(X.exp_root("E5") / "example1.parquet")
    fig, axes = plt.subplots(2, 4, figsize=(S.DCOL, 3.2), gridspec_kw={"width_ratios": [1, 0.85, 0.85, 1.05]})
    for r, a in enumerate(ALPHAS):
        for c, (ds, name) in enumerate((("cf_ton", "CF-TON"), ("cf_gotham", "CF-GOTHAM v1 (pre-reg.)"),
                                        ("cf_gotham_v2", "CF-GOTHAM v2 (dev. P1)"))):   # v1 first (P2-X, 19 Oct)
            ax = axes[r, c]
            sub = byl.filter((pl.col("ds") == ds) & ((pl.col("alpha") - a).abs() < 1e-9) & (pl.col("n_incidents_pool") >= 10))
            bins = sorted(set(sub["L_bin"].to_list()), key=lambda b: int(b.rstrip("+")))
            labels = [f"L={b}\nn={int(sub.filter(pl.col('L_bin') == b)['n_incidents_pool'][0])}" for b in bins] + ["all"]
            for j, (m, lab, col, mk) in enumerate(METHODS):
                xs, ys, los, his = [], [], [], []
                for i, b in enumerate(bins):
                    e = n5.get(f"byL|{ds}|{m}|a={a}|L={b}|maxchain_fdr")
                    if not e or e.get("value") is None:
                        continue
                    xs.append(i)
                    ys.append(e["value"])
                    ci = e.get("ci") or [np.nan, np.nan]
                    los.append(ci[0])
                    his.append(ci[1])
                e = n5.get(f"{ds}|{SIDS[ds]}|{m}|a={a}|maxchain_fdr")
                if e and e.get("value") is not None:
                    xs.append(len(bins))
                    ys.append(e["value"])
                    ci = e.get("ci") or [np.nan, np.nan]
                    los.append(ci[0])
                    his.append(ci[1])
                xs = np.asarray(xs, float) + (j - 1) * 0.18
                ys, los, his = map(np.asarray, (ys, los, his))
                yerr = np.vstack([np.where(np.isfinite(los), ys - los, 0), np.where(np.isfinite(his), his - ys, 0)])
                ax.errorbar(xs, ys, yerr=yerr, fmt=mk, color=col, ms=3.2, elinewidth=0.8, capsize=1.5,
                            mfc=col if m == "chain_composite" else "white", mew=0.8)
            S.alpha_line(ax, a, color=S.ALPHA_COLORS[a])
            ax.axvline(len(bins) - 0.5, color=S.GRID, lw=0.8)
            ax.set_xticks(range(len(bins) + 1))
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylim(-0.01, 0.42 if a == 0.2 else 0.3)
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(name, fontsize=7.5)
            if c == 0:
                ax.set_ylabel(f"maximal-chain FDR\n($\\alpha$ = {a})")
        # Example 1
        ax = axes[r, 3]
        e = ex.filter((pl.col("alpha") - a).abs() < 1e-9).sort("L")
        L = e["L"].to_numpy()
        ax.plot(L, e["asymptote"].to_numpy(), ls=":", color=S.INK2, lw=0.9)
        ax.plot(L, e["theory_naive_maxchain_fdr"].to_numpy(), ls="-", color=S.ORANGE, lw=1.0)
        ax.plot(L, e["chain_naive|maxchain_fdr"].to_numpy(), ls="none", marker="s", mfc="white", mec=S.ORANGE, ms=3.2)
        ax.plot(L, e["chain_naive|fdr"].to_numpy(), ls="--", marker="s", ms=2.4, color=S.ORANGE, lw=0.7, alpha=0.6)
        ax.plot(L, e["chain_composite|maxchain_fdr"].to_numpy(), ls="none", marker="o", ms=3.2, color=S.BLUE)
        ax.plot(L, e["chain_rt|chain_fwer"].to_numpy(), ls="none", marker="^", ms=3.0, mfc="white", mec=S.AQUA)
        S.alpha_line(ax, a, color=S.ALPHA_COLORS[a])
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks(L)
        if r == 0:
            ax.set_title("Example 1 (simulation)", fontsize=7.5)
        if r == 1:
            ax.set_xlabel("true chain length L")
    hs = [Line2D([], [], color=col, marker=mk, ls="none", ms=3.5, mfc=col if m == "chain_composite" else "white",
                 label=lab) for m, lab, col, mk in METHODS]
    hs += [Line2D([], [], color=S.ORANGE, ls="-", label="naive: theory f/n"),
           Line2D([], [], color=S.ORANGE, ls="--", alpha=0.6, marker="s", ms=2.4, label="naive edge FDR"),
           Line2D([], [], color=S.INK2, ls=":", label="min(1, $\\alpha$L/(1-$\\alpha$))"),
           Line2D([], [], color=S.INK2, ls="--", lw=0.7, label="$\\alpha$")]
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=4, fontsize=7, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.25, hspace=0.55)
    keys = [k for k in n5 if k.startswith(("cf_ton|S_TON.stat|chain", "cf_gotham|S_GOT.stat|chain",
                                           "cf_gotham_v2|S_GOT2.stat|chain")) and
            k.endswith(("|maxchain_fdr", "|ccr_mean", "|fdr"))]
    lines = []
    for k in sorted(keys):
        v = n5[k]
        if v.get("value") is None:
            continue
        lines.append(f"{k} = {v['value']:.3f}" + (f" [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]" if v.get("ci") else ""))
    cap = """Fig. 6 (chains, E5). Required facts (results/summary/E5_numbers.json; results/exp/E5/by_length.parquet, example1.parquet): statistical scorers, 200 splits (CF-TON K = 150 / 150 test, CF-GOTHAM K = 100 / 100; CF-GOTHAM v1 = the pre-registered benchmark (shown first), v2 = regenerated benchmark, post-PREREG deviation P1, side by side), J = 20 maximal chains per incident; maximal-chain FDR (strict: a chain is false if any edge has Y != 1) of composite CRC (Alg. 5, CRC on max(edge FDP, maximal-chain FDP)), chain-naive (chains read off the edge-CRC set) and chain-RT (coherent-factuality style on the score-threshold family, FWER-type guarantee), by the GT longest root-leaf path length L of the test incident (calibration always on all incidents; strata with < 10 pool incidents omitted; CF-TON has L <= 3 by template construction) and over all test incidents ('all'); bars = 95% incident-pool bootstrap CIs with the stratum mask. The guarantee is marginal: a stratum may exceed alpha. Right: Example 1 (P2-C eval/synth.example1_incident): f = floor(alpha_1 n L / (1 - alpha_1)) decoy-hijacked first hops (alpha_1 = alpha - (1 - alpha)/K); naive composition keeps the edge FDR <= alpha (dashed) while its maximal-chain FDR is f/n (solid = theory, squares = simulation); composite CRC and chain-RT stop before the jump (0 claims).
""" + "\n".join(lines)
    S.save(fig, "fig6_chains", cap)
    print("fig6 written")


if __name__ == "__main__":
    main()
