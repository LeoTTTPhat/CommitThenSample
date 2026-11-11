"""Fig. 6 (E7): (a) detection--variance frontier over gamma; (b) worst-case retention g_mu(phi_t) of the
two-tier design for mu in {1,3,10}; (c) upper-bound null slack vs the hiding-capacity bound H_{alpha/2} as a
function of phi0 (Thm. 1 / Cor. 1). Analytic. Double-column (7.16 in). Owner: P1-F.

Reads results/E7/<cfg_hash>/{frontier.json, two_tier.json, capacity.json} and
results/summary/E7_frontier.parquet. Writes figures/fig6_design.pdf + figures/fig6_design.caption.txt.

    .venv/bin/python -I P1_CommitThenSample/figures/fig6_design.py [--hash <cfg_hash>]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

import _style_p1f as S  # noqa: E402


def _dir(cfg_hash):
    ds = sorted(glob.glob(str(S.RESULTS / "E7" / ("*" if cfg_hash is None else cfg_hash))))
    ds = [d for d in ds if Path(d, "DONE").exists()]
    return Path(ds[-1])


def panel_a(ax, fr_rows):
    """Frontier over gamma: detection (floor-only, several k) rises, relative variance ~flat; gamma=0.3 knee."""
    df = pl.DataFrame(fr_rows)
    g = df.group_by("gamma").agg(
        pl.col("detect_k10").median().alias("d10"), pl.col("detect_k10").min().alias("d10lo"),
        pl.col("detect_k10").max().alias("d10hi"),
        pl.col("detect_k50").median().alias("d50"),
        pl.col("worst_rel_var").median().alias("wV")).sort("gamma")
    gam = g["gamma"].to_numpy()
    ax.plot(gam, g["d50"].to_numpy(), "-o", color=S.PALETTE["cts_ai"], label=r"detect, $k{=}50$", ms=3)
    ax.plot(gam, g["d10"].to_numpy(), "-s", color=S.PALETTE["cts_unif"], label=r"detect, $k{=}10$", ms=3)
    ax.fill_between(gam, g["d10lo"].to_numpy(), g["d10hi"].to_numpy(), color=S.PALETTE["cts_unif"],
                    alpha=0.15, lw=0)
    ax.axvline(0.3, color="#888888", lw=0.7, ls=":")
    ax.text(0.3, 0.03, r"$\gamma^\star$", fontsize=6, ha="center", color="#555555")
    ax.set_xlabel(r"floor share $\gamma$")
    ax.set_ylabel("floor-only detection")
    ax.set_ylim(0, 1)
    ax.set_xlim(0.08, 0.52)
    ax2 = ax.twinx()
    ax2.plot(gam, g["wV"].to_numpy(), "-^", color=S.PALETTE["b3"], label="worst rel. var.", ms=3, lw=0.9)
    ax2.set_ylabel("worst rel. variance", color=S.PALETTE["b3"])
    ax2.tick_params(axis="y", labelcolor=S.PALETTE["b3"], labelsize=6)
    ax2.set_ylim(0, max(2.0, float(g["wV"].max()) * 1.3))
    ax.set_title("(a) detection--variance frontier")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=5.3, loc="upper left", borderaxespad=0.2)


def panel_b(ax, tt):
    """g_mu(phi_t) for mu in {1,3,10} at the measured CIC head share; markers at the optimum."""
    cur = tt["curves"].get("CIC", list(tt["curves"].values())[0])
    phi_star = tt.get("phi_star", 0.015)
    colors = {"1": S.PALETTE["cts_ai"], "3": S.PALETTE["cts_unif"], "10": S.PALETTE["b4"]}
    for m in ("1", "3", "10"):
        d = cur["m"][m]
        pt = np.asarray(d["phi_t"])
        g = np.asarray(d["worst"])
        ax.plot(pt, g, "-o", color=colors[m], ms=3, label=rf"$\mu{{=}}{m}$")
        i = int(np.argmax(g))
        ax.plot([pt[i]], [g[i]], marker="*", ms=6, color=colors[m], lw=0)
    ax.axhline(phi_star, color="#888888", lw=0.6, ls="--")
    ax.text(0.12, phi_star * 1.05, r"single tier $\varphi^\star$", fontsize=5.4, color="#555555", va="bottom")
    ax.set_xlabel(r"tail rate $\varphi_t$")
    ax.set_ylabel("worst-case retention $g_\\mu$")
    ax.set_xlim(0.05, 1.03)
    ax.set_title(rf"(b) two-tier, head share $h{{=}}{cur['head_share']:.2f}$")
    ax.legend(fontsize=5.6, loc="upper right", ncol=1, borderaxespad=0.2)


def panel_c(ax, cap):
    """UCS null radius r+ vs the capacity bound H_{alpha/2} vs phi0; ratio ~ 1.19."""
    rows = cap["rows"]
    phi0 = np.array([r["phi0"] for r in rows])
    rplus = np.array([r["r_plus"] for r in rows])
    H = np.array([r["H_cont"] for r in rows])
    ax.plot(phi0, rplus, color=S.PALETTE["cts_ai"], lw=1.1, label=r"UCS slack $r^+$")
    ax.plot(phi0, H, color=S.PALETTE["bound"], lw=1.0, ls="--", label=r"capacity $H_{\alpha/2}$")
    ax.fill_between(phi0, H, rplus, color=S.PALETTE["cts_ai"], alpha=0.12, lw=0)
    # reference phi0 = 0.015
    ref = cap["refs"].get("0.015")
    if ref:
        ax.plot([0.015], [ref["r_plus"]], marker="o", ms=3, color=S.PALETTE["cts_ai"])
        ax.annotate(rf"$r^+/H\approx{ref['r_plus'] / ref['H_cont']:.2f}$", xy=(0.015, ref["r_plus"]),
                    xytext=(0.021, ref["r_plus"] * 1.25), fontsize=5.6,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#555555"))
    ax.set_xlabel(r"floor $\varphi_0$")
    ax.set_ylabel("eligible-unit equivalents")
    ax.set_xlim(phi0.min(), phi0.max())
    ax.set_yscale("log")
    ax.set_title("(c) upper-bound null slack")
    ax.legend(fontsize=5.8, loc="upper right", borderaxespad=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", default=None)
    a = ap.parse_args()
    S.setup()
    import matplotlib.pyplot as plt
    d = _dir(a.hash)
    fr = json.loads((d / "frontier.json").read_text())
    tt = json.loads((d / "two_tier.json").read_text())
    cap = json.loads((d / "capacity.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(S.DBL_W, 1.72))
    panel_a(axes[0], fr["rows"])
    panel_b(axes[1], tt)
    panel_c(axes[2], cap)
    for ax in axes:
        ax.grid(True, which="major", alpha=0.5)
    fig.subplots_adjust(wspace=0.58, left=0.060, right=0.945, bottom=0.235, top=0.85)

    out = S.save(fig, "fig6_design")
    ref = cap["refs"]["0.015"]
    gdf = pl.DataFrame(fr["rows"]).group_by("gamma").agg(pl.col("detect_k50").median()).sort("gamma")
    cic = tt["curves"]["CIC"]["m"]
    S.write_caption_facts("fig6_design", [
        "Fig. 6 facts (E7, analytic, rho=5%).",
        f"(a) Frontier over gamma on {len({r['fold'] for r in fr['rows']})} LP-IoT23 folds; floor phi0=gamma*rho*Sum_all/Sum_pay.",
        "    Floor-only detection rises with gamma while worst relative variance stays ~flat (floor-dominated at",
        "    rho=5%, share_at_floor 0.51->0.70); gamma=0.3 is the operating point (diminishing detection gains).",
        f"    detect(k=50): {[round(x,3) for x in gdf['detect_k50'].to_list()]} at gamma={gdf['gamma'].to_list()}.",
        f"(b) g_mu(phi_t), CIC head share h={tt['curves']['CIC']['head_share']:.2f}: mu=1 maximised at phi_t*="
        f"{cic['1']['phi_t_star']:.2f} (single tier), mu=3 interior, mu=10 at small phi_t.",
        f"    worst g: mu=1 {[round(x,3) for x in cic['1']['worst']]}, mu=10 {[round(x,3) for x in cic['10']['worst']]} over phi_t={cic['1']['phi_t']}.",
        f"(c) At phi0=0.015: cbar={ref['cbar']:.1f}, UCS slack r+={ref['r_plus']:.0f}, capacity H={ref['H_cont']:.0f} "
        f"eligible units, ratio {ref['r_plus'] / ref['H_cont']:.3f} (Cor. 1: log(80)/log(40)={cap['cor1_ratio_log80_40']:.3f}).",
        f"Saved {out.name}.",
    ])
    print(f"[fig6] wrote {out}")


if __name__ == "__main__":
    main()
