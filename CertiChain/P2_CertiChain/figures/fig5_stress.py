#!/usr/bin/env python
"""Fig. 5 -- E4 stress tests on CF-TON (fused CRC-BH, M_fuse seed 0 trained on clean scores): realised FDR (top) and
STR (bottom) under s_LLM corruption applied to calibration AND test incidents (Gaussian noise sigma, inversion of a
fraction f, GT-dependent memorisation +delta on in-band nulls) and under TEST-ONLY injection of q nulls at the top
score (BH family vs score-threshold family, with the Remark R3 bound). Owner: P2-G.

  .venv/bin/python -I P2_CertiChain/figures/fig5_stress.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from certichain.eval import experiments as X  # noqa: E402

BASE = "S_TON.fuse_s0"
ALPHAS = (0.1, 0.2)


def point(path: str, metric: str) -> tuple[float, float, float]:
    df = pl.read_parquet(Path(path) / "per_split.parquet")
    v = float(df[metric].mean())
    b = X.read_bootstrap(Path(path), "main")
    if b and metric in b["metrics"]:
        return v, b["metrics"][metric]["lo"], b["metrics"][metric]["hi"]
    return v, np.nan, np.nan


def main() -> None:
    S.setup()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    idx = X.collect("E4")
    n4 = S.numbers("E4")
    panels = [("noise", [0, 1, 2, 4], "noise $\\sigma$", "noise\ncal. + test"),
              ("invert", [0, 0.5, 1], "inverted fraction f", "inversion\ncal. + test"),
              ("memorise", [0, 2, 5], "shift $\\delta$ on in-band nulls", "memorisation\ncal. + test"),
              ("inject_bh", [0, 1, 3, 10], "injected nulls q", "injection, test only\nBH family"),
              ("inject_st", [0, 1, 3, 10], "injected nulls q", "injection, test only\nthreshold family")]
    fig, axes = plt.subplots(2, 5, figsize=(S.DCOL, 3.0), sharey="row")
    for c, (kind, grid, xlabel, title) in enumerate(panels):
        axes[0, c].set_title(title, fontsize=7)
        for a in ALPHAS:
            col = S.ALPHA_COLORS[a]
            for r, metric in enumerate(("fdr", "str_mean")):
                xs, ys, los, his, bnd = [], [], [], [], []
                for j, v in enumerate(grid):
                    meth = "crc_st" if kind == "inject_st" else "crc_bh"
                    if v == 0:
                        sub = idx.filter((pl.col("scorer_id") == BASE) & pl.col("test_scorer_id").is_null()
                                         & (pl.col("method") == meth) & ((pl.col("alpha") - a).abs() < 1e-9))
                    elif kind.startswith("inject"):
                        sub = idx.filter((pl.col("scorer_id") == BASE) & (pl.col("test_scorer_id") == f"{BASE}__E4-inject-{v}")
                                         & (pl.col("method") == meth) & ((pl.col("alpha") - a).abs() < 1e-9))
                    else:
                        sub = idx.filter((pl.col("scorer_id") == f"{BASE}__E4-{kind}-{v:g}") & (pl.col("method") == meth)
                                         & ((pl.col("alpha") - a).abs() < 1e-9))
                    if not sub.height:
                        continue
                    m, lo, hi = point(sub["path"][0], metric)
                    xs.append(j)
                    ys.append(m)
                    los.append(lo)
                    his.append(hi)
                    if kind == "inject_st" and v > 0 and metric == "fdr":
                        e = n4.get(f"cf_ton|inject-{v}|crc_st|a={a}|r3_bound")
                        bnd.append((j, e["value"] if e else np.nan))
                ax = axes[r, c]
                xs, ys, los, his = map(np.asarray, (xs, ys, los, his))
                off = (-0.07 if a == 0.1 else 0.07)
                ax.errorbar(xs + off, ys, yerr=[ys - los, his - ys], fmt="-o", color=col, ms=3, lw=1.0, elinewidth=0.8,
                            capsize=1.5)
                if bnd:
                    bj, bv = zip(*bnd)
                    ax.plot(np.asarray(bj) + off, np.minimum(bv, 1.0), ls=":", marker="v", ms=3, color=col, lw=0.9)
                if r == 0:
                    S.alpha_line(ax, a, color=col)
        for r in range(2):
            ax = axes[r, c]
            ax.set_xticks(range(len(grid)))
            ax.set_xticklabels(["clean"] + [f"{g:g}" for g in grid[1:]], fontsize=7)
            ax.grid(axis="x", visible=False)
        axes[1, c].set_xlabel(xlabel, fontsize=7)
    axes[0, 0].set_ylabel("realised FDR")
    axes[1, 0].set_ylabel("STR")
    axes[0, 0].set_ylim(-0.01, 1.02)
    hs = [Line2D([], [], color=S.ALPHA_COLORS[a], marker="o", ms=3, label=f"$\\alpha$ = {a}") for a in ALPHAS]
    hs += [Line2D([], [], color=S.INK2, ls="--", lw=0.7, label="$\\alpha$"),
           Line2D([], [], color=S.INK2, ls=":", marker="v", ms=3, label="R3 bound (threshold family; capped at 1)")]
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=4, fontsize=7)
    fig.subplots_adjust(wspace=0.12, hspace=0.32)
    rows = []
    for k, v in n4.items():
        if k.endswith("|fdr") and v.get("value") is not None:
            rows.append(f"{k} = {v['value']:.3f}" + (f" [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]" if v.get("ci") else ""))
    cap = """Fig. 5 (E4 stress). Required facts (results/summary/E4_numbers.json): CF-TON, fused CRC-BH (M_fuse seed 0 trained on clean s_LLM, K = 150 calibration / 150 test incidents, 100 splits), alpha = 0.1 (blue) and 0.2 (orange), bars = 95% incident-pool bootstrap CIs (B = 1000 x 50 splits). Columns 1-3: corruption of s_LLM applied identically to calibration and test incidents (Gaussian noise sigma in {1, 2, 4}; sign inversion of a fraction f in {0.5, 1}; memorisation adversary +delta in {2, 5} on in-band null candidates, which uses the GT) -- validity is preserved (Theorem 1) and STR drops. Columns 4-5: test-only injection of q in {1, 3, 10} null candidates per test incident at the top score, calibration on clean incidents: BH family (CRC-BH) and score-threshold family (CRC-ST) with the Remark R3 bound alpha + E[min(1, q/max(|S'|, 1))] (dotted; it is nearly vacuous here because most incidents carry few claims). Measured cells:
""" + "\n".join(sorted(rows))
    S.save(fig, "fig5_stress", cap)
    print("fig5 written")


if __name__ == "__main__":
    main()
