"""Shared matplotlib style for P1-F figures (Fig. 5 E4, Fig. 6 E7). IEEE column width 3.5 in, vector PDF,
fonts >= 7 pt, one consistent palette. Owner: P1-F (not imported by other workers' figure scripts)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL_W = 3.5          # IEEE single-column width (in)
DBL_W = 7.16         # double-column width (in)

# consistent, color-blind-safe palette (Okabe-Ito based)
PALETTE = {
    "cts_ai": "#0072B2",       # CtS-AI (U6tau)
    "cts_size": "#56B4E9",     # CtS size-only (U1)
    "cts_unif": "#009E73",     # CtS uniform (B8)
    "bound": "#000000",        # the (1-phi0)^k bound
    "b3": "#D55E00",           # Time-Machine
    "b4": "#CC79A7",           # top-k
    "b6": "#E69F00",           # alert-triggered (zero-day)
    "b13": "#8C510A",          # top-k + reserve
    "two_tier": "#7570B3",     # capped two-tier
    "blackbox": "#999999",     # black-box transfer
    "grid": "#DDDDDD",
}
MARKERS = {"wb": "o", "bb": "s", "none": "^"}

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGDIR = Path(__file__).resolve().parent


def setup():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 6,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "figure.dpi": 200, "savefig.dpi": 300,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.0,
        "grid.linewidth": 0.4, "grid.color": PALETTE["grid"], "legend.frameon": False,
        "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.columnspacing": 1.0,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.axisbelow": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    })


def save(fig, name: str):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / f"{name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def write_caption_facts(name: str, lines: list[str]):
    (FIGDIR / f"{name}.caption.txt").write_text("\n".join(lines) + "\n")
