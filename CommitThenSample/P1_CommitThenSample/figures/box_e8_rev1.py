#!/usr/bin/env python
"""E8 case-study box, rev1 (committed per-unit thresholds) from results/summary/E8_case_rev1.json → figures/box_e8_rev1.tex
(rev0: figures/box_e8.py / box_e8.tex, kept).
(≤ 12 lines in the column). Run from the project root: .venv/bin/python -I P1_CommitThenSample/figures/box_e8.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results" / "summary" / "E8_case_rev1.json"   # rev1 protocol run
OUT = ROOT / "figures" / "box_e8_rev1.tex"


def n(x) -> str:
    return f"{int(round(x)):,}"


def main() -> None:
    d = json.loads(SUM.read_text())
    r, mc = d["report"], d.get("mc") or {}
    e1, e2 = r["window"]
    st = {s["query"]: s for s in r["statements"]["statements"]}
    by_q = {}
    for s in r["statements"]["statements"]:
        by_q[s["query"]] = s
    names = {q: k for k, q in ((v["name"], q) for q, v in (mc.get("queries") or {}).items())}
    tr = r["transcript"]

    def cnt(check, status):
        return tr.get(check, {}).get(status, 0)

    v16_fail = sum(cnt(c, "FAIL") for c in ("V1", "V2", "V3", "V4", "V5", "V6"))
    v7_fail = cnt("V7", "FAIL")
    w_fail = [x for x in r["window_fail"] if "V7" in x]
    a_stmt = r["statements"]["statements"][0]["alpha_statement"]
    fw = r["statements"]["family_wise_alpha"]

    def lo(name):
        return int(math.floor(by_q[name]["L_plus"] + 1e-9))

    def up(name):
        u = by_q[name]["U_plus"]
        return None if u is None else int(math.ceil(u - 1e-9))

    def tru(q):
        return int(round(r["truth"][q]))

    def band(q):
        v = (mc.get("queries") or {}).get(q)
        return None if not v else v["L_plus_q05_50_95"]

    md = r["metadata"]
    tp = md["top_ports_by_bytes"]
    share23 = 100.0 * tp[0]["bytes"] / md["bytes"] if tp else 0.0
    pv = r["privacy"]
    ds = r["disclosure_size"]
    cov = [v["lcs_cover"] for v in (mc.get("queries") or {}).values()]
    covu = [v["ucs_cover"] for v in (mc.get("queries") or {}).values() if v.get("ucs_cover") is not None]
    tight = r.get("tightening") or {}
    tightened = any(v.get("tightened_U_plus") is not None and v["U_plus_by_end"][0] is not None
                    and v["tightened_U_plus"] < v["U_plus_by_end"][0] for v in tight.values())
    b = lambda q: band(q)                                                           # noqa: E731
    lines = [
        rf"\textbf{{Epochs {e1}--{e2 - 1} ({(e2 - e1) * 0.5:g}\,min), device $d$ = infected host, LP-IoT23(48-1, 1\%).}}",
        rf"V1--V6: PASS in every epoch ({v16_fail} FAIL; V2 BLS skipped); V7: {len(w_fail)} flagged cell of another "
        rf"device in the window; $|\Wset|={r['W_size']}$.",
        rf"\textbf{{Statements}} ($\alpha={a_stmt:.4f}$ each, family-wise $\alpha={fw:.2f}$):",
        rf"$\ge{lo('Telnet password prompts')}$ Telnet password and $\ge{lo('Telnet login prompts')}$ login prompts "
        rf"received by $d$ (truth {tru('D1c')}, {tru('D2c')});",
        rf"Mirai loader strings in $[{lo('Mirai loader strings')},{up('Mirai loader strings')}]$, ELF downloads in "
        rf"$[{lo('ELF download (Suricata policy-violation)')},{up('ELF download (Suricata policy-violation)')}]$ "
        rf"segments (truth {tru('D3')}, {tru('S5')}); C\&C connections $\ge{lo('C&C connections (label)')}$, "
        rf"attack segments $\ge{lo('attack segments (label)')}$ (labels; truth {tru('L2')}, {tru('L4')}).",
        rf"200 replications: $L^+$ 5--95\% {b('D1c')[0]:.0f}--{b('D1c')[2]:.0f} (prompts), "
        rf"{b('L2')[0]:.0f}--{b('L2')[2]:.0f} (C\&C); coverage {min(cov):.2f}/{min(covu):.2f} (lower/upper)."
        if cov and covu else "",
        rf"Retrospective tightening to epoch {((tight and next(iter(tight.values()))['ends'][-1]) or e2) - 1}: "
        + ("tighter $U^+$." if tightened else r"no gain ($U^+$ non-decreasing)."),
        rf"\textbf{{Exact metadata:}} {n(md['units'])} units of $d$ ({n(md['scan_units_M1'])} scan units), "
        rf"{n(md['packets'])} packets, {md['bytes'] / 1e6:.1f}\,MB, {share23:.0f}\% to TCP/{tp[0]['port'] if tp else '--'}.",
        rf"\textbf{{Privacy cost}} (epochs {r['epochs_disclosed'][0]}--{r['epochs_disclosed'][1]}): "
        rf"{pv['disclosed_retained_units']} retained units of $d$ "
        rf"({pv['malicious_bytes_disclosed'] / 1e6:.2f}\,MB malicious, {n(pv['benign_bytes_disclosed'])}\,B benign "
        rf"content); {n(pv['third_party_records_disclosed'])} third-party metadata records, no third-party content; "
        rf"disclosure {ds['total_bytes'] / 1e6:.1f}\,MB.",
    ]
    body = " ".join(x for x in lines if x)
    tex = (r"\smallskip\noindent" + "\n" + r"\fbox{\parbox{\dimexpr\columnwidth-2\fboxsep-2\fboxrule\relax}{\footnotesize"
           + "\n" + body + "}}\n")
    OUT.write_text(tex)
    print(f"wrote {OUT} ({len(body)} characters)")


if __name__ == "__main__":
    main()
