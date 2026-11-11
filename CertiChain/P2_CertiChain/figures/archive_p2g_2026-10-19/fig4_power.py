#!/usr/bin/env python
"""Fig. 4 -- power. (a) E3 (pre-registered primary endpoint 2): Delta CER_link = CER_link(CRC-BH, fused) -
CER_link(CRC-BH, statistical) on CF-TON-H (template-held-out, primary) and CF-TON (in-template), alpha in {0.1, 0.2},
with the two placebos and M_fuse seeds 1-4; (b) E7 minimal K on CF-TON (FDR, STR vs K; CRC-BH, BH-iid, LR-A2/A2b)
and the synthetic panel (decoy-free reproduction vs decoy-rich). Owner: P2-G.

  .venv/bin/python -I P2_CertiChain/figures/fig4_power.py
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

ALPHAS = (0.1, 0.2)
KGRID = [5, 10, 20, 50, 100, 200]


def kpos(Ks):
    return np.array([KGRID.index(int(k)) for k in Ks], dtype=float)


def e3_rows() -> list[dict]:
    n = S.numbers("E3")
    rows = []
    for role, ds, stat, pref in (("primary", "cf_ton_h", "S_TON_H.stat", "S_TON_H"),
                                 ("in_template", "cf_ton", "S_TON.stat", "S_TON")):
        for a in ALPHAS:
            for comp, lab, kind in ((f"{pref}.fuse_s0", "fused (seed 0)", "main"),
                                    (f"{pref}.fuse_perm_s0", "placebo: permuted", "placebo"),
                                    (f"{pref}.fuse_band_s0", "placebo: band only", "placebo"),
                                    *[(f"{pref}.fuse_s{k}", f"seed {k}", "seed") for k in (1, 2, 3, 4)]):
                key = f"{role}|{ds}|{comp}-{stat}|a={a}|all|delta_cer_link_mean"
                e = n.get(key)
                if not e or e.get("value") is None:
                    continue
                rows.append({"role": role, "ds": ds, "alpha": a, "comp": comp, "label": lab, "kind": kind,
                             "value": e["value"], "lo": (e.get("ci") or [None, None])[0],
                             "hi": (e.get("ci") or [None, None])[1]})
    return rows


def main() -> None:
    S.setup()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(S.DCOL, 2.55))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.35, 1, 1, 1], wspace=0.55, left=0.25)
    # ---------------------------------------------------------------- (a) E3 forest
    ax = fig.add_subplot(gs[0, 0])
    rows = e3_rows()
    p2 = S.RESULTS / "summary" / "primary2_llm.json"
    dec = json.loads(p2.read_text()).get("decision") if p2.exists() else None
    y = 0
    yt, yl = [], []
    for role, ds, title in (("primary", "cf_ton_h", "CF-TON-H, held-out templates (primary)"),
                            ("in_template", "cf_ton", "CF-TON, in-template (secondary)")):
        ax.text(-0.02, y + 0.75, title, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7,
                fontweight="bold")
        y -= 0.5
        for a in ALPHAS:
            sub = [r for r in rows if r["role"] == role and r["alpha"] == a]
            if not sub:
                continue
            for r in [r for r in sub if r["kind"] != "seed"]:
                col = S.ALPHA_COLORS[a] if r["kind"] == "main" else S.NEUTRAL
                ax.errorbar([r["value"]], [y], xerr=[[r["value"] - r["lo"]], [r["hi"] - r["value"]]], fmt="o" if r["kind"] == "main" else "s",
                            ms=3.4, color=col, mfc=col if r["kind"] == "main" else "white", elinewidth=0.9, capsize=1.5, zorder=4)
                if role == "primary" and r["kind"] == "main" and dec:
                    h = dec["holm"].get(f"cf_ton_h|alpha={a}")
                    if h:
                        ax.plot([h["lcb_holm"]], [y], marker="|", ms=7, mew=1.4, color=S.INK, zorder=5)
                yt.append(y)
                yl.append(f"{r['label']}, $\\alpha$={a}")
                y -= 1
            seeds = [r for r in sub if r["kind"] == "seed"]
            if seeds:
                ax.scatter([r["value"] for r in seeds], [y + 3] * len(seeds), marker="x", s=10, color=S.INK2, zorder=5, lw=0.7)
            y -= 0.6
        ax.text(0.0, y + 0.35, "", fontsize=7)
        y -= 0.4
    ax.axvline(0, color=S.INK2, lw=0.7)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=7)
    ax.set_xlabel("$\\Delta$CER$_{link}$ (fused $-$ statistical)")
    ax.grid(axis="y", visible=False)
    ax.set_title("(a) E3: $\\Delta$CER$_{link}$, 95% CI", loc="left", fontsize=7)
    # ---------------------------------------------------------------- (b) E7 minimal K
    idx = X.collect("E7")
    if idx.height:
        real = idx.filter(pl.col("ds") == "cf_ton")
        axf = fig.add_subplot(gs[0, 1])
        axs = fig.add_subplot(gs[0, 2])
        axy = fig.add_subplot(gs[0, 3])
        for a in (0.05, 0.1, 0.2):
            col = S.ALPHA_COLORS[a]
            for m, ls, mk in (("crc_bh", "-", "o"), ("bh_iid", ":", "^")):
                sub = real.filter((pl.col("method") == m) & ((pl.col("alpha") - a).abs() < 1e-9)).sort("K")
                if not sub.height:
                    continue
                Ks = sub["K"].to_numpy()
                for ax_, met in ((axf, "fdr"), (axs, "str_mean")):
                    vals, lo, hi = [], [], []
                    for pth in sub["path"].to_list():
                        df = pl.read_parquet(Path(pth) / "per_split.parquet")
                        vals.append(float(df[met].mean()))
                        b = X.read_bootstrap(Path(pth), "main")
                        if b and met in b["metrics"]:
                            lo.append(b["metrics"][met]["lo"])
                            hi.append(b["metrics"][met]["hi"])
                        else:
                            lo.append(np.nan)
                            hi.append(np.nan)
                    vals = np.array(vals)
                    ax_.plot(kpos(Ks), vals, ls=ls, marker=mk, color=col, ms=2.8, lw=1.0)
                    lo, hi = np.array(lo), np.array(hi)
                    if np.isfinite(lo).any():
                        ax_.fill_between(kpos(Ks), lo, hi, color=col, alpha=0.15, lw=0)
                if m == "crc_bh":
                    ab = []
                    for pth in sub["path"].to_list():
                        df = pl.read_parquet(Path(pth) / "per_split.parquet")
                        ab.append(float(df["abstain"].mean()))
                    for K_, v in zip(Ks, ab):
                        if v > 0.5:
                            axs.plot(kpos([K_]), [0], marker="X", ms=4.5, color=col, mec="white", mew=0.4, zorder=6)
            S.alpha_line(axf, a, color=col)
        lr = real.filter(pl.col("method").is_in(["lr_a2", "lr_a2b"])).sort("K")
        if lr.height:
            kk = kpos(sorted(set(lr["K"].to_list())))
            axs.plot(kk, [0] * len(kk), ls="-.", color=S.INK2, lw=0.9)
            axf.plot(kk, [0] * len(kk), ls="-.", color=S.INK2, lw=0.9)
        for ax_ in (axf, axs):
            ax_.set_xticks(range(len(KGRID)))
            ax_.set_xticklabels([str(k) for k in KGRID], rotation=45)
            ax_.set_xlabel("calibration incidents K")
        axf.set_ylabel("realised FDR")
        axs.set_ylabel("STR")
        axf.set_title("(b) E7, CF-TON: FDR", loc="left", fontsize=7)
        axs.set_title("CF-TON: STR", loc="left", fontsize=7)
        axf.set_ylim(-0.01, 0.42)
        # synthetic panel
        for gen, ls, lab in (("synth_decoy_free", "-", "decoy-free"), ("synth_decoy_rich", "--", "decoy-rich")):
            for a in (0.05, 0.1, 0.2):
                sub = idx.filter((pl.col("ds") == gen) & (pl.col("method") == "crc_bh") & ((pl.col("alpha") - a).abs() < 1e-9)).sort("K")
                if not sub.height:
                    continue
                Ks, mu, se = [], [], []
                for K_, pth in zip(sub["K"].to_list(), sub["path"].to_list()):
                    df = pl.read_parquet(Path(pth) / "per_split.parquet")
                    pm = df.with_columns((pl.col("seed") // 1000).alias("pool")).group_by("pool").agg(pl.col("str_mean").mean())
                    v = pm["str_mean"].to_numpy()
                    Ks.append(K_)
                    mu.append(v.mean())
                    se.append(v.std(ddof=1) / np.sqrt(v.size))
                Ks, mu, se = np.array(Ks), np.array(mu), np.array(se)
                axy.plot(kpos(Ks), mu, ls=ls, marker="o", ms=2.4, color=S.ALPHA_COLORS[a], lw=1.0)
                axy.fill_between(kpos(Ks), mu - 1.96 * se, mu + 1.96 * se, color=S.ALPHA_COLORS[a], alpha=0.15, lw=0)
        axy.set_xticks(range(len(KGRID)))
        axy.set_xticklabels([str(k) for k in KGRID], rotation=45)
        axy.set_xlabel("calibration incidents K")
        axy.set_ylabel("STR")
        axy.set_title("synthetic, CRC-BH", loc="left", fontsize=7)
        hs = [Line2D([], [], color=S.ALPHA_COLORS[a], lw=1.2, label=f"$\\alpha$={a}") for a in (0.05, 0.1, 0.2)]
        hs += [Line2D([], [], color=S.INK2, ls="-", marker="o", ms=2.8, label="CRC-BH"),
               Line2D([], [], color=S.INK2, ls=":", marker="^", ms=2.8, label="BH-iid"),
               Line2D([], [], color=S.INK2, ls="-.", label="LR-A2/A2b (0 claims)"),
               Line2D([], [], color=S.INK2, ls="-", label="synth decoy-free"),
               Line2D([], [], color=S.INK2, ls="--", label="synth decoy-rich")]
        fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.62, -0.30), ncol=4, fontsize=7, columnspacing=1.0)
    e3 = json.loads(p2.read_text()) if p2.exists() else {}
    d = e3.get("decision") or {}
    cap = f"""Fig. 4 (power; E3, E7). Required facts for the caption (results/summary/E3_numbers.json, primary2_llm.json, E7_numbers.json):
(a) Pre-registered primary endpoint 2: Delta CER_link = CER_link(CRC-BH, M_fuse seed 0) - CER_link(CRC-BH, M_stat), metric cer_link_mean, 200 splits (CF-TON-H: K = 100 / 100 test; CF-TON: K = 150 / 150), kappa = 0.1; bars = two-sided 95% paired incident-pool bootstrap CIs (B = 1000 x 50 splits; same resampled incidents and splits for both banks); | = Holm-adjusted one-sided lower confidence bound (Holm over alpha in {{0.1, 0.2}}); grey squares = placebos (s_LLM permuted within the band; band indicator only); x = M_fuse seeds 1-4 (robustness, not tested). Decision: success = {d.get('success')} (rejected: {d.get('rejected_alphas')}; placebo CIs contain 0: {d.get('placebos_null')}; complete: {d.get('complete')}).
(b) E7 on CF-TON (statistical scorer): K in {{5, 10, 20, 50, 100, 200}} calibration incidents, 100 test incidents, 200 splits; colour = alpha (0.05 / 0.1 / 0.2), dashed = alpha; CRC-BH (solid) abstains when (K + 1) alpha < 1 (X); BH-iid (dotted); LR-A2 and LR-A2b certify nothing at every K (dash-dot at 0). Bands = 95% incident-pool bootstrap CIs. Right: P2-C synthetic generator (normal incident shift, mu = 3, tau = 0.5), decoy-free (solid, reproduces the pre-study calculation) vs decoy-rich (dashed), 10 pools x 20 splits, bands = 95% between-pool CI.
"""
    S.save(fig, "fig4_power", cap)
    print("fig4 written")


if __name__ == "__main__":
    main()
