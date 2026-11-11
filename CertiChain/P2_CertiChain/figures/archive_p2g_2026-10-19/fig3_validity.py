#!/usr/bin/env python
"""Fig. 3 -- validity (E1, E11). (a) realised FDR over 200 calibration/test splits on CF-TON, CF-TON-H, CF-GOTHAM for
every method, on both estimands (mean per-incident FDP / pooled FDR), alpha in {0.1, 0.2}; (b) IoT-23 P-LOSO
per-scenario FDP with the LOSO mean and its delete-one-scenario jackknife interval. Owner: P2-G.

  .venv/bin/python -I P2_CertiChain/figures/fig3_validity.py
Reads results/exp/E1 (cells), results/exp/E11/iot23/loso, results/summary/primary1_validity{,_v2}.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from certichain.eval import experiments as X  # noqa: E402

DSETS = [("cf_ton", "CF-TON", "S_TON"), ("cf_ton_h", "CF-TON-H", "S_TON_H"), ("cf_gotham_v2", "CF-GOTHAM v2", "S_GOT2"),
         ("cf_gotham", "CF-GOTHAM v1 (orig.)", "S_GOT")]
METHODS = [("crc_bh", "stat", "CRC-BH"), ("crc_bh", "fuse_s0", "CRC-BH fused"), ("crc_fdx020", "stat", "CRC-FDX(0.2)"),
           ("ltt", "stat", "LTT"), ("bh_iid", "stat", "BH-iid"), ("by_ref", "stat", "BY-ref"), ("b7", "stat", "B7"),
           ("b8", "stat", "B8"), ("lr_a2b", "stat", "LR-A2b")]
ALPHAS = (0.1, 0.2)


def cell_frame(idx: pl.DataFrame, ds: str, sid: str, method: str, alpha: float) -> pl.DataFrame | None:
    sub = idx.filter((pl.col("ds") == ds) & (pl.col("scorer_id") == sid) & (pl.col("method") == method)
                     & ((pl.col("alpha") - alpha).abs() < 1e-9))
    if not sub.height:
        return None
    return pl.read_parquet(Path(sub["path"][0]) / "per_split.parquet")


def main() -> None:
    S.setup()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    idx = X.collect("E1")
    fig = plt.figure(figsize=(S.DCOL, 4.7))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[2.05, 1.0], hspace=0.78, top=0.955, bottom=0.10)
    gs = outer[0].subgridspec(2, len(DSETS), hspace=0.10, wspace=0.08)
    present = {}
    for r, (metric, ylab) in enumerate((("fdr", "realised FDR\n(per-incident FDP)"), ("pooled_fdr", "pooled FDR"))):
        for c, (ds, name, scorer) in enumerate(DSETS):
            ax = fig.add_subplot(gs[r, c])
            pos = 0
            ticks, labels = [], []
            for m, kind, lab in METHODS:
                sid = f"{scorer}.{kind}"
                for j, a in enumerate(ALPHAS):
                    df = cell_frame(idx, ds, sid, m, a)
                    x = pos + (j - 0.5) * 0.36
                    if df is None:
                        continue
                    present[(ds, m, kind)] = True
                    v = df[metric].to_numpy().astype(float)
                    v = v[np.isfinite(v)]
                    q05, q25, q50, q75, q95 = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
                    col = S.ALPHA_COLORS[a]
                    ax.add_patch(plt.Rectangle((x - 0.15, q25), 0.30, max(q75 - q25, 1e-4), facecolor=col, alpha=0.35,
                                               edgecolor=col, lw=0.6, zorder=3))
                    ax.plot([x, x], [q05, q25], color=col, lw=0.6, zorder=3)
                    ax.plot([x, x], [q75, q95], color=col, lw=0.6, zorder=3)
                    ax.plot([x - 0.15, x + 0.15], [q50, q50], color=col, lw=1.0, zorder=4)
                    ax.plot([x], [v.mean()], marker="o", ms=2.6, mfc="white", mec=col, mew=0.7, zorder=5)
                ticks.append(pos)
                labels.append(lab)
                pos += 1
            for a in ALPHAS:
                S.alpha_line(ax, a, color=S.ALPHA_COLORS[a])
            ax.set_xlim(-0.6, pos - 0.4)
            ax.set_ylim(-0.02, 1.0 if metric == "pooled_fdr" else 0.62)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels if r == 1 else [], rotation=90, fontsize=7)
            ax.tick_params(axis="x", pad=1)
            if r == 0:
                ax.tick_params(axis="x", length=0)
                ax.set_title(name)
            if c == 0:
                ax.set_ylabel(ylab)
            else:
                ax.set_yticklabels([])
            ax.grid(axis="x", visible=False)
            if (ds, "crc_bh", "fuse_s0") not in present and r == 0:
                ax.text(1, 0.58, "no fused\nbank", ha="center", va="top", fontsize=7, color=S.INK2)
    # (b) IoT-23 P-LOSO
    ax = fig.add_subplot(outer[1])
    loso = X.exp_root("E11") / "iot23" / "loso"
    scen = None
    xs = None
    summary_pts = []
    for j, a in enumerate(ALPHAS):
        p = loso / "crc_bh__S_CF.stat" / f"a{a}"
        if not (p / "per_scenario.parquet").exists():
            continue
        per = pl.read_parquet(p / "per_scenario.parquet")
        jk = json.loads((p / "jackknife.json").read_text())
        if scen is None:
            per0 = per.with_columns(pl.col("scenario").str.replace("HP-", "HP").alias("lab"))
            order = sorted(range(per0.height), key=lambda i: (per0["scenario"][i].startswith("HP"),
                                                              float(per0["scenario"][i].split("-")[-2])
                                                              if not per0["scenario"][i].startswith("HP")
                                                              else float(per0["scenario"][i].split("-")[1])))
            scen = [per0["scenario"][i] for i in order]
            xs = np.arange(len(scen))
        pm = dict(zip(per["scenario"].to_list(), per["fdp"].to_list()))
        nw = dict(zip(per["scenario"].to_list(), per["n_windows"].to_list()))
        y = np.array([pm[s] for s in scen])
        col = S.ALPHA_COLORS[a]
        ax.scatter(xs + (j - 0.5) * 0.28, y, s=[6 + 1.6 * nw[s] for s in scen], color=col, alpha=0.85,
                   edgecolor="white", linewidth=0.4, zorder=4)
        summary_pts.append((a, jk))
    fused_pts = []
    for j, a in enumerate(ALPHAS):
        pf = loso / "crc_bh__S_CF.fuse_s0" / f"a{a}"
        if scen is not None and (pf / "per_scenario.parquet").exists():
            perf = pl.read_parquet(pf / "per_scenario.parquet")
            pmf = dict(zip(perf["scenario"].to_list(), perf["fdp"].to_list()))
            ax.scatter(xs + (j - 0.5) * 0.28, [pmf[s_] for s_ in scen], s=10, facecolor="none",
                       edgecolor=S.ALPHA_COLORS[a], linewidth=0.7, zorder=5, marker="D")
            fused_pts.append((a, json.loads((pf / "jackknife.json").read_text())))
    if scen is not None:
        xr = len(scen) + 0.8
        for j, (a, jk) in enumerate(fused_pts):
            col = S.ALPHA_COLORS[a]
            xx = xr + j * 0.7 + 0.25
            ax.errorbar([xx], [jk["estimate"]], yerr=[[jk["estimate"] - max(jk["lo"], -0.05)], [jk["hi"] - jk["estimate"]]],
                        fmt="D", ms=3.6, color=col, mfc="white", elinewidth=0.8, capsize=2, zorder=5)
        for j, (a, jk) in enumerate(summary_pts):
            col = S.ALPHA_COLORS[a]
            xx = xr + j * 0.7
            ax.errorbar([xx], [jk["estimate"]], yerr=[[jk["estimate"] - max(jk["lo"], -0.05)], [jk["hi"] - jk["estimate"]]],
                        fmt="D", ms=3.6, color=col, mfc=col, elinewidth=1.0, capsize=2, zorder=5)
        for a in ALPHAS:
            S.alpha_line(ax, a, color=S.ALPHA_COLORS[a])
        ax.set_xticks(list(xs) + [xr + 0.35])
        ax.set_xticklabels([s for s in scen] + ["LOSO mean"], rotation=90, fontsize=7)
        ax.set_xlim(-0.8, xr + 1.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("scenario FDP\n(IoT-23 P-LOSO)")
        ax.grid(axis="x", visible=False)
        ax.set_title("(b) IoT-23 P-LOSO, CRC-BH, K = 22 scenarios: dots = statistical scorer (area $\\propto$ windows), "
                     "open diamonds = fused; right: LOSO means, jackknife 95% CI", loc="left", fontsize=7)
    fig.text(0.0, 0.995, "(a) ChainForge-IoT, 200 calibration/test splits: box = IQR, whiskers = 5-95% of splits, "
             "line = median, circle = mean", fontsize=7, ha="left", va="bottom")
    handles = [Patch(facecolor=S.ALPHA_COLORS[a], alpha=0.5, edgecolor=S.ALPHA_COLORS[a], label=f"$\\alpha$ = {a}")
               for a in ALPHAS] + [Line2D([], [], ls="--", color=S.INK2, lw=0.7, label="$\\alpha$ (dashed)")]
    ax.legend(handles=handles, loc="upper left", ncol=3, fontsize=7)
    v = json.loads((S.RESULTS / "summary" / "primary1_validity.json").read_text())
    run = [c for c in v["cells"] if not c.get("missing")]
    pv2 = S.RESULTS / "summary" / "primary1_validity_v2.json"
    v2txt = ""
    if pv2.exists():
        v2 = json.loads(pv2.read_text())
        zc = {c["cell"]: c for c in v2["cells"]}
        z1, z2 = (zc.get(f"cf_gotham|alpha={a}|crc_bh|stats", {}).get("z") for a in (0.1, 0.2))
        v2txt = (f" Deviation-P1 family (CF-GOTHAM cells from v2 / S_GOT2, primary1_validity_v2.json): "
                 f"{v2['n_cells_run']}/{len(v2['cells'])} cells run, max z = {v2['max_standardised_excess']:.2f}, "
                 f"violations = {v2['n_violations']} (CF-GOTHAM v2 z = {z1:.2f} / {z2:.2f} at alpha = 0.1 / 0.2). The "
                 f"CF-GOTHAM fused cells are missing in both families because P2-F dropped CF-GOTHAM fusion.")
    cap = f"""Fig. 3 (validity; E1, E11). Required facts for the caption (from results/summary/E1_numbers.json, primary1_validity.json):
(a) Realised FDR over 200 random calibration/test splits (split seeds 0-199) of the cal/test pools: CF-TON K = 150 / 150 test incidents, CF-TON-H K = 100 / 100, CF-GOTHAM K = 100 / 100; statistical scorers S_TON, S_TON_H, S_GOT2 (CF-GOTHAM v2, regenerated without the integer-second splicing artifact: documented post-PREREG deviation P1 triggered by the pre-registered E15 check) and S_GOT (CF-GOTHAM v1, the original benchmark, shown side by side); CRC-BH fused = M_fuse seed 0 (kappa = 0.1) where its bank exists ({', '.join(n for d, n, _ in DSETS if (d, 'crc_bh', 'fuse_s0') in present) or 'none'}). Top row: mean per-incident strict FDP (AMB counted false) = the estimand CRC-BH/FDX/LTT guarantee; bottom row: FDR pooled over each split's test incidents. Boxes = IQR of the split distribution, whiskers 5-95%, line = median, open circle = mean; dashed lines at alpha = 0.1 (blue) and 0.2 (orange). LR-A2b (like all six Lee-Ren variants) certifies nothing in every split (Prop. 2). B7/B8 use the statistical score here.
Primary endpoint 1 (pre-registered): {len(run)}/{len(v['cells'])} cells run, max standardised excess z = {v['max_standardised_excess']:.2f}, Holm violations = {v['n_violations']} (one-sided H0: FDR <= alpha, SE from the incident-pool cluster bootstrap B = 1000 x 50 splits; P-LOSO cells: jackknife SE). Missing cells: {', '.join(c['cell'] for c in v['cells'] if c.get('missing')) or 'none'}.{v2txt}
(b) IoT-23 P-LOSO (primary real-data protocol): each of the 23 scenarios is held out once, CRC calibrated on the other 22 scenarios' mean monotonised window losses; dot = scenario FDP (mean strict window FDP), area proportional to its windows; diamond = LOSO mean with the delete-one-scenario jackknife 95% interval (t_22); open markers = fused scorer S_CF fuse_s0 where available ({'yes' if fused_pts else 'not yet'}). Validity is the point there: rho_T ~ 0.91 caps STR at ~0.08.
"""
    S.save(fig, "fig3_validity", cap)
    print("fig3 written")


if __name__ == "__main__":
    main()
