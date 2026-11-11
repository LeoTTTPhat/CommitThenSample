#!/usr/bin/env python
"""Table "overheads" (paper Table VI, `tab:e6`; DESIGN Table V) from results/summary/E6_overheads.json.

Writes figures/tab5_e6.tex (complete table environment with the draft's caption, label and rows).
Run from the project root:  .venv/bin/python -I P1_CommitThenSample/figures/tab5_e6.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results" / "summary" / "E6_overheads.json"
OUT = ROOT / "figures" / "tab5_e6.tex"


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def fmt(x, nd=1):
    if x is None:
        return "--"
    if x >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{nd}f}"


def main() -> None:
    d = json.loads(SUM.read_text())
    P, E = g(d, "pipeline", "P", default={}), g(d, "pipeline", "E", default={})
    rows = []
    star = r"$^\ast$" if d.get("pipeline_headline_variant") == "cached_registration" else ""
    rows.append(r"Per packet (\textmu s) / packets/s per core" + star + " & " +
                f"{fmt(g(P, 'us_per_packet', 'total'))} / {fmt(g(P, 'pps_per_core'), 0)} & "
                f"{fmt(g(E, 'us_per_packet', 'total'))} / {fmt(g(E, 'pps_per_core'), 0)}\\\\")
    rows.append(r"Commit latency, median / p99 (ms) & " +
                f"{fmt(g(P, 'commit_latency_ms_median'))} / {fmt(g(P, 'commit_latency_ms_p99'))} & "
                f"{fmt(g(E, 'commit_latency_ms_median'))} / {fmt(g(E, 'commit_latency_ms_p99'))}\\\\")

    def ep(core, n, key):
        for r in g(d, "epochs", core, "rows", default=[]):
            if r["n_e"] == n:
                return r[key]
        return None

    rows.append(r"Merkle build, $n_e=10^2/10^4/10^6$ (ms) & " +
                "/".join(fmt(ep("P", n, "merkle"), 2 if n == 100 else 1) for n in (100, 10_000, 1_000_000)) + " & " +
                "/".join(fmt(ep("E", n, "merkle"), 2 if n == 100 else 1) for n in (100, 10_000, 1_000_000)) + r"\\")
    vP = g(d, "verifier", "P_cached", default=g(d, "verifier", "P", default={}))
    vE = g(d, "verifier", "E_cached", default=g(d, "verifier", "E", default={}))

    def prim(core):
        return g(d, "primitives", core, "us", default={})

    def pv(core, *names):
        r = prim(core)
        for n in names:
            if isinstance(r, dict) and n in r:
                return r[n]
        return None

    rows.append(r"Ed25519 sign/verify (\textmu s); ECVRF prove/verify (ms) & " +
                f"{fmt(pv('P', 'ed25519_sign'), 0)}/{fmt(pv('P', 'ed25519_verify'), 0)}; "
                f"{fmt((pv('P', 'ecvrf_prove') or 0) / 1e3, 1)}/{fmt((pv('P', 'ecvrf_verify') or 0) / 1e3, 1)} & "
                f"{fmt(pv('E', 'ed25519_sign'), 0)}/{fmt(pv('E', 'ed25519_verify'), 0)}; "
                f"{fmt((pv('E', 'ecvrf_prove') or 0) / 1e3, 1)}/{fmt((pv('E', 'ecvrf_verify') or 0) / 1e3, 1)}\\\\")
    rows.append(r"Verifier per epoch, with prime bundle (ms) & " +
                f"{fmt(1e3 * (vP.get('median_s_per_epoch') or 0), 0)} & {fmt(1e3 * (vE.get('median_s_per_epoch') or 0), 0)}\\\\")
    rows.append(r"\midrule")
    rows.append(r"Inclusion proof$^\dagger$; VRF proof$^\dagger$ & \multicolumn{2}{r}{$32\lceil\log_2 n_e\rceil$\,B; 80\,B}\\")
    w = d.get("witness", {})
    rows.append(r"Witness load per gateway & \multicolumn{2}{r}{" + f"{w.get('cosignatures_per_day', 2880):,}" +
                r"/day$^\dagger$; " + f"{w.get('wire_MB_per_day', 0):.2f}" + r"\,MB/day}\\")
    br = d.get("buffer_range", {})

    def rng(k):
        v = br.get(k)
        return "--" if not v else (f"{v[0]:.1f}" if abs(v[0] - v[1]) < 0.05 else f"{v[0]:.1f}--{v[1]:.1f}")

    rows.append(r"Buffer p50 / p99 / max (one round ahead) & \multicolumn{2}{r}{" +
                f"{rng('ahead1_p50_MB')} / {rng('ahead1_p99_MB')} / {rng('ahead1_max_MB')}" + r"\,MB}\\")
    st = {r["stream"]: r for r in g(d, "storage", "rows", default=[])}
    iot = [r["storage_over_pcap"] for k, r in st.items() if k.startswith("lp_iot23")]
    edge = [r["storage_over_pcap"] for k, r in st.items() if k.startswith("lp_edge")]
    nat = [r["meta_over_pcap"] for k, r in st.items() if k.startswith("nat_")]
    rows.append(r"Storage / pcap: LP-IoT23; LP-Edge & \multicolumn{2}{r}{" +
                (f"{100 * min(iot):.1f}--{100 * max(iot):.1f}\\%" if iot else "--") + "; " +
                (f"{100 * edge[0]:.1f}\\%" if edge else "--") + r"}\\")
    ai = g(d, "pipeline_variants", "as_implemented", default={})
    note = ""
    if star and ai:
        note = (r"\multicolumn{3}{@{}p{\columnwidth}@{}}{\tiny $^\ast$Median of 5 single-core runs over 100 epochs ("
                + f"{g(P, 'packets', default=0):,}" + r" packets) with the constant $(\hash(\theta),\hash(\text{policy}))$ "
                r"memoised; recomputing $\hash(\theta)$ every epoch, as the prototype does, gives "
                + f"{fmt(g(ai, 'P', 'us_per_packet', 'total'))}/{fmt(g(ai, 'E', 'us_per_packet', 'total'))}"
                + r"\,\textmu s per packet and " + f"{fmt(g(ai, 'P', 'commit_latency_ms_median'), 0)}/"
                + f"{fmt(g(ai, 'E', 'commit_latency_ms_median'), 0)}" + r"\,ms commits (P/E). Buffer maximum: one "
                r"196-MB transfer of one CIC device within one epoch.}\\" + "\n")
    tex = r"""\begin{table}[!t]
\centering
\caption{E6: overheads, sizes, storage and witness load ($\dagger$: by construction).}
\label{tab:e6}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}l r r@{}}
\toprule
Quantity & P-core & E-core\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
""" + note + r"""\end{tabular}
\end{table}
"""
    OUT.write_text(tex)
    print(f"wrote {OUT}")
    if nat:
        print(f"note: natural single captures meta/pcap {100 * min(nat):.1f}–{100 * max(nat):.1f} %")


if __name__ == "__main__":
    main()
