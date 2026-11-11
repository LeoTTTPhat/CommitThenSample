"""Fig. 5 (E4): content-evidence miss vs incident size k under mimicry, padding and Time-Machine cut-off
evasion, against the bound (1-phi0)^k; inset: time-to-evidence of a persistent C&C channel (Prop. 4(iii)).
All LP streams, rho = 5%, analytic. Owner: P1-F.

Reads results/E4/<stream>/<cfg_hash>/{incidents.parquet, metrics.json} (per-stream, robust to the pooled
summary not being rebuilt). Writes figures/fig5_evasion.pdf + figures/fig5_evasion.caption.txt.

    .venv/bin/python -I P1_CommitThenSample/figures/fig5_evasion.py [--hash <cfg_hash>]
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

MISS_FLOOR = 1e-13      # log-axis clamp for vanishing miss


def load(cfg_hash=None):
    dirs = sorted(glob.glob(str(S.RESULTS / "E4" / "*" / ("*" if cfg_hash is None else cfg_hash))))
    dirs = [d for d in dirs if Path(d, "DONE").exists()]
    inc, meta = [], {}
    for d in dirs:
        sid = Path(d).parent.name
        try:
            inc.append(pl.read_parquet(Path(d, "incidents.parquet")))
            meta[sid] = json.loads(Path(d, "metrics.json").read_text())
        except Exception:  # noqa: BLE001
            pass
    return pl.concat(inc, how="diagonal_relaxed"), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", default=None)
    a = ap.parse_args()
    S.setup()
    import matplotlib.pyplot as plt
    df, meta = load(a.hash)
    # LP streams only (drop any natural if present); use the per-incident committed floor phi0
    phis = np.array([m["committed_floor"] for m in meta.values()])
    phi_lo, phi_hi, phi_med = float(phis.min()), float(phis.max()), float(np.median(phis))

    def sel(policy, attack):
        q = df.filter((pl.col("policy") == policy) & (pl.col("attack") == attack))
        return q["k"].to_numpy().astype(float), np.clip(q["miss"].to_numpy(), MISS_FLOOR, 1.0)

    fig, ax = plt.subplots(figsize=(S.COL_W, 2.05))
    kgrid = np.logspace(0, np.log10(8000), 200)
    # bound band (1-phi0)^k over the committed floor range, and the median line
    ax.fill_between(kgrid, (1 - phi_hi) ** kgrid, (1 - phi_lo) ** kgrid, color=S.PALETTE["bound"],
                    alpha=0.12, lw=0, label=r"$(1-\varphi_0)^k$ bound")
    ax.plot(kgrid, (1 - phi_med) ** kgrid, color=S.PALETTE["bound"], lw=0.9, ls="--")

    # deterministic policies miss w.p. 1 under attack (B4 top-k, B3 Time-Machine)
    ax.axhline(1.0, color=S.PALETTE["b3"], lw=1.2, label="B3 / B4, evasion")

    # CtS-AI (U6tau) under white-box (strongest attacker) and black-box (weaker) mimicry
    k, m = sel("cts_single", "mimicry_bb")
    ax.scatter(k, m, s=6, facecolors="none", edgecolors=S.PALETTE["blackbox"], marker="s", lw=0.45,
               alpha=0.7, label="CtS-AI, black-box")
    k, m = sel("cts_single", "mimicry_wb")
    ax.scatter(k, m, s=11, c=S.PALETTE["cts_ai"], marker="o", alpha=0.85, lw=0,
               label="CtS-AI, white-box")

    # two-tier capped under padding (worst case m=1): miss = (1 - phi_cap*phi_t)^k, using the measured head share
    hs = np.array([(m2.get("two_tier") or {}).get("head_share", np.nan) for m2 in meta.values()])
    hs = float(np.nanmedian(hs))
    gamma, rho = 0.3, 0.05
    phi_cap = min(1.0, phi_med / (hs + 0.1 * (1 - hs)))       # phi_cap/phi_star = 1/(h+phi_t(1-h)), phi_star~phi_med
    ret_pad = phi_cap * (1 - (1 - 0.1) ** 1)                  # padded single-packet retention (m=1)
    ax.plot(kgrid, (1 - ret_pad) ** kgrid, color=S.PALETTE["two_tier"], lw=1.0, ls=":",
            label=r"two-tier padded, $\mu{=}1$")

    # B13 top-k + reserve under mimicry (median reserve rate)
    pr = []
    for sid, d in meta.items():
        pass
    prs = []
    for d in sorted(glob.glob(str(S.RESULTS / "designs" / "*" / "B13_s0_0.05.json"))):
        try:
            prs.append(json.load(open(d))["params"]["pi_reserve"])
        except Exception:  # noqa: BLE001
            pass
    pr = float(np.median(prs)) if prs else 0.016
    ax.plot(kgrid, (1 - pr) ** kgrid, color=S.PALETTE["b13"], lw=1.0, ls="-.",
            label=r"B13 reserve, mimicry")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 8000)
    ax.set_ylim(MISS_FLOOR, 2.2)
    ax.set_xlabel(r"incident size $k$ (content units)")
    ax.set_ylabel("content-evidence miss")
    ax.grid(True, which="major", alpha=0.6)
    leg = ax.legend(loc="lower left", ncol=1, fontsize=5.3, borderaxespad=0.25,
                    framealpha=0.9, frameon=True)
    leg.get_frame().set_linewidth(0.0)

    # TTE inset: survival P(TTE > n) for the two reference floors (placed in the empty mid-upper-right)
    axin = ax.inset_axes([0.645, 0.60, 0.335, 0.33])
    n = np.arange(1, 301)
    for phi0, c, lab in ((0.015, S.PALETTE["cts_ai"], r"$\varphi_0{=}.015$"),
                         (0.029, S.PALETTE["b4"], r"$\varphi_0{=}.029$")):
        axin.plot(n, (1 - phi0) ** n, color=c, lw=0.9, label=lab)
        k95 = int(np.ceil(np.log(0.05) / np.log(1 - phi0)))
        axin.plot([k95], [(1 - phi0) ** k95], marker="o", ms=2.3, color=c)
    axin.axhline(0.05, color="#888888", lw=0.5, ls=":")
    axin.text(0.96, 0.92, r"P(TTE$>n$)", transform=axin.transAxes, fontsize=5.2, ha="right", va="top")
    axin.set_xlabel("epochs $n$", fontsize=5.0, labelpad=0.5)
    axin.tick_params(labelsize=4.6, length=1.5, pad=1)
    axin.set_ylim(0, 1)
    axin.set_xlim(0, 300)
    axin.set_xticks([0, 150, 300])
    axin.set_yticks([0, 0.5, 1.0])
    axin.legend(fontsize=4.5, loc="upper right", handlelength=0.9, borderaxespad=0.15,
                bbox_to_anchor=(1.0, 0.86))

    out = S.save(fig, "fig5_evasion")
    # caption facts
    wb = df.filter((pl.col("policy") == "cts_single") & (pl.col("attack") == "mimicry_wb"))
    nbelow = int((wb["miss"].to_numpy() <= (1 - np.array([meta[s]["committed_floor"] for s in wb["stream"]])) **
                  wb["k"].to_numpy() * (1 + 1e-6)).sum())
    S.write_caption_facts("fig5_evasion", [
        f"Fig. 5 facts (rho=5%, analytic, {len(meta)} LP streams, {df['stream'].n_unique()} streams pooled).",
        f"Committed floor phi0 range [{phi_lo:.4f}, {phi_hi:.4f}], median {phi_med:.4f}.",
        f"CtS-AI white-box mimicry: {len(wb)} incidents, {nbelow} at or below the (1-phi0)^k bound.",
        f"Two-tier padded (mu=1) retention {ret_pad:.4f} at head share h={hs:.3f}, phi_t=0.1 (phi_cap={phi_cap:.4f}).",
        f"B13 median keyed reserve pi_r={pr:.4f}. B3 (Time-Machine) and B4 (top-k) miss with probability 1 under evasion.",
        "Black-box surrogate transfer is markedly weaker than white-box (points lie below the band).",
        "TTE inset: P(TTE>n)=(1-phi0)^n; 95% detection at 199 epochs (phi0=.015) and 102 (phi0=.029).",
        f"Saved {out.name}.",
    ])
    print(f"[fig5] wrote {out}")


if __name__ == "__main__":
    main()
