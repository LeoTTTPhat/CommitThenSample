"""Shared matplotlib style for the P2 figures (owner: P2-G).

IEEE layout: single column 3.5 in, double column 7.16 in; vector PDF with embedded TrueType fonts (fonttype 42);
all text >= 7 pt; alpha reference lines dashed; one categorical palette in a fixed order (validated with the
dataviz validator: slots 1-3 pass all-pairs CVD / normal-vision checks on a light surface; slot 3 is below 3:1
contrast, so it is always paired with a legend / direct label); intervals always drawn.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent
P2 = FIG_DIR.parent
RESULTS = P2 / "results"
COL = 3.5
DCOL = 7.16

# fixed categorical order (dataviz reference palette, light mode)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SLOTS = (BLUE, ORANGE, AQUA)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
NEUTRAL = "#8a8985"
ALPHA_COLORS = {0.05: AQUA, 0.1: BLUE, 0.2: ORANGE}


def setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.title_fontsize": 7, "figure.titlesize": 8,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.4, "axes.axisbelow": True, "legend.frameon": False, "legend.handlelength": 1.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.dpi": 150,
    })


def alpha_line(ax, a: float, color=INK2, horizontal: bool = True, lw: float = 0.7) -> None:
    if horizontal:
        ax.axhline(a, ls="--", lw=lw, color=color, zorder=1)
    else:
        ax.axvline(a, ls="--", lw=lw, color=color, zorder=1)


def numbers(exp: str) -> dict:
    p = RESULTS / "summary" / f"{exp}_numbers.json"
    return json.loads(p.read_text())["numbers"] if p.exists() else {}


MIN_PT = 7.0


def font_check(fig, target_width: float = DCOL) -> dict:
    """Smallest font of any visible text artist and its size at final size, i.e. after LaTeX scales the saved
    (tight-bbox) PDF to ``target_width`` (the paper includes every figure at \\textwidth = 7.16 in). P2-X, 19 Oct."""
    from matplotlib.text import Text
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    pad = float(plt.rcParams.get("savefig.pad_inches", 0.02))
    w, h = bb.width + 2 * pad, bb.height + 2 * pad
    sizes = [(float(t.get_fontsize()), t.get_text()[:30]) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    mn = min(sizes) if sizes else (MIN_PT, "")
    scale = target_width / w
    return {"saved_width_in": round(float(w), 3), "saved_height_in": round(float(h), 3),
            "target_width_in": float(target_width), "scale_at_target": round(float(scale), 3),
            "min_font_pt": float(mn[0]), "min_font_text": str(mn[1]),
            "min_font_at_final_size_pt": round(float(mn[0] * scale), 2), "ok": bool(mn[0] * scale >= MIN_PT - 1e-6)}


def save(fig, name: str, caption: str, target_width: float = DCOL) -> Path:
    chk = font_check(fig, target_width)
    if not chk["ok"]:
        raise RuntimeError(f"{name}: smallest font {chk['min_font_pt']} pt becomes {chk['min_font_at_final_size_pt']} "
                           f"pt at {target_width} in (saved width {chk['saved_width_in']} in); fix the layout")
    out = FIG_DIR / f"{name}.pdf"
    fig.savefig(out)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    (FIG_DIR / f"{name}.caption.txt").write_text(caption.strip() + "\n")
    (FIG_DIR / f"{name}.fontcheck.json").write_text(json.dumps(chk, indent=1) + "\n")
    plt.close(fig)
    return out
