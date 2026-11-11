#!/usr/bin/env python
"""Table "security matrix" (paper Table V, `tab:e5`; DESIGN Table IV) from results/summary/E5_security.json.

Writes figures/tab4_e5.tex: the complete table environment (same columns, caption and label as the draft),
with the ε_cell calibration and the grinding curve inline in a note row. Run from the project root:
    .venv/bin/python -I P1_CommitThenSample/figures/tab4_e5.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results" / "summary" / "E5_security.json"
OUT = ROOT / "figures" / "tab4_e5.tex"

ROWS = [
    ("A1", "Drop selected content", "V6 MISSING"),
    ("A2", "Insert or replace content", "V6 EXTRA / MISMATCH"),
    ("A3", "Alter metadata after commit", r"V4 $\hash$(store) / root"),
    ("A4", "Swap model at selection", r"V4 $\tau$ $\to$ V5/V6"),
    ("A5", r"Alter $\Lambda$ or $\varphi$ after $\beta$", r"V4 (signed header, $\Cst$)"),
    ("A6", "Grind (no TR)", "witness refuses (TR)"),
    ("A7", "Withhold an epoch", "V1 chain gap"),
    ("A8", "Omit packets before commit", "V7, size-dependent"),
    ("A9", "Cherry-picked disclosure", "V5/V6 MISSING"),
    ("A10", "Forged / omitted boost", "V4 justification"),
    ("A11", r"Selective abort after $\beta$", "V6 / V3 / V1"),
    ("A12", "Equal-size substitution", "not detected (limit)"),
]


def f2(x):
    return "--" if x is None else f"{x:.2f}"


def sci(x):
    if x is None:
        return "--"
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return rf"${m}{{\times}}10^{{{int(e)}}}$"


def main() -> None:
    d = json.loads(SUM.read_text())
    A = d["adversaries"]
    H = d["honest"]
    g = d.get("grind", {})
    a8 = d.get("a8_cells", {}).get("p0.001", {})
    cells = []
    for aid, name, chk in ROWS:
        if aid == "A6":
            det = r"1.00$^{a}$"
            far = "0"
        elif aid == "A8":
            a = A["A8"]
            det = f2(a["detection_rate"]) + r"$^{b}$"
            far = sci(H["v7_far_cell"]) + r"$^{c}$"
        elif aid == "A10":
            det = f"{f2(A['A10']['detection_rate'])} / {f2(A['A10omit']['detection_rate'])}" + r"$^{d}$"
            far = "0"
        elif aid == "A12":
            det = f2(A["A12"]["detection_rate"])
            far = "--"
        else:
            a = A[aid]
            det = f2(a["detection_rate"])
            far = "0" if (a["far"] == 0) else f2(a["far"])
        cells.append(f"{aid} & {name} & {chk} & {det} & {far}\\\\")
    # ---- notes: ε_cell, A8 curve, grinding, χ², prime bundle
    rows = {(r["k"], r["M"]): r for r in g.get("rows", [])}
    r10 = [rows.get((10, M)) for M in (1, 4, 16)]
    eps = d["eps_cell"]["p0.001"]["eps"]
    by_m = A["A8"].get("by_m", {})
    ms = a8.get("m_star", {})
    chi = d.get("chi2", {})
    note_a = (r"$^{a}$With TR all " + f"{g.get('witness_refusals', 0):,}" + r" ground headers (sent after $\beta$) and "
              + f"{g.get('second_header_refusals', 0)}" + r"/" + f"{g.get('trials', 0)}" +
              r" second headers before the deadline were refused. Without TR grinding is undetectable: "
              r"$\Pr(\text{evidence})$ ($\pi=0.05$, $k=10$) $=" + f"{r10[0]['p_hat_noTR']:.2f}" + r"$ (formula "
              + f"{r10[0]['formula_noTR']:.3f}" + r") at $M=1$, " + f"{r10[1]['p_hat_noTR']:.3f}" + r" ("
              + f"{r10[1]['formula_noTR']:.3f}" + r") at $M=4$, 0 ($\le" + sci(r10[2]['formula_noTR']).strip("$") +
              r"$) for $M\ge16$; with TR " + f"{r10[0]['p_hat_TR']:.2f}" + r" for every $M$.") \
        if r10 and all(r10) else ""
    note_b = (r"$^{b}$Attacked cell flagged for $m=1,3,10,50$ omitted packets: " +
              ", ".join(f2(by_m.get(str(m), {}).get("rate")) for m in (1, 3, 10, 50)) +
              r"; largest undetected omission $m^*(n)=" +
              ", ".join(str(ms.get(k, "--")) for k in ("n=[3,15)", "n=[15,63)", "n=[63,255)", "n=[255,1023)")) +
              r"$ packets for $n\in[3,15),[15,63),[63,255),[255,1023)$; a C\&C unit has a median of " +
              f"{a8.get('cnc', {}).get('pkts_per_unit_median', 0):.0f}" + r" packets.") if a8 else ""
    note_c = (r"$^{c}$Per (device, epoch) cell, honest windows; $\varepsilon_{\mathrm{cell}}(n)$ fitted on the "
              r"CIC training half (FAR $10^{-3}$ per criterion): " +
              ", ".join(f"{int(x)}" for x in eps["eps_pkts"][1:8]) + r" packets for $\lfloor\log_2(n+1)\rfloor=2,\dots,8$.")
    note_d = r"$^{d}$Forged / omitted (the CtS policy has no omitted-boost check; reported)."
    note_e = (r"Real PRF vs.\ Philox MC: $\chi^2_{" + f"{chi.get('df', 10)}" + r"}=" + f"{chi.get('chi2_real', 0):.1f}"
              + r"$, $p=" + f"{chi.get('p_real', 0):.2f}" + r"$ over " + f"{chi.get('n_payload_units', 0):,}" +
              r" units.") if chi else ""
    tex = r"""\begin{table}[!t]
\centering
\caption{E5: attacks, expected check, detection rate and false-alarm rate (200 trials each, LP-IoT23(34-1, 1\%)).}
\label{tab:e5}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}l >{\raggedright\arraybackslash}p{3.0cm} >{\raggedright\arraybackslash}p{2.45cm} c c@{}}
\toprule
ID & Attack & Expected check & Det. & FAR\\
\midrule
""" + "\n".join(cells) + r"""
\bottomrule
\multicolumn{5}{@{}p{\columnwidth}@{}}{\tiny """ + " ".join(x for x in (note_a, note_b, note_c, note_d, note_e) if x) + r"""}
\end{tabular}
\end{table}
"""
    OUT.write_text(tex)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
