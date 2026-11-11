#!/usr/bin/env python
"""Table V rev1 (paper `tab:e6`, E6 overheads) from results/summary/E6_overheads_rev1.json and E6_flood.json.

Same rows as the rev0 table (figures/tab5_e6.tex, kept) measured with the rev1 code (committed τ column, incremental
rolling statistics), plus the rev1 rows: verifier in validity-only mode (committed τ, no prime bundle, no policy
recomputation), the largest sustained zero-payload flood whose commit fits Δ_build = 0.8 s, the commit time with the
zero-payload cap, and the size of the committed τ column. A core whose rev1 run is missing is printed as "--" (never
filled with rev0 values). Primitive timings, buffer and storage rows come from the rev0 measurements where the rev1
summary falls back to them (code unchanged; noted in the caption sidecar).

Writes figures/tab5_e6_rev1.tex (complete table environment) and figures/tab5_e6_rev1.caption.txt (facts).
Run from the project root:  .venv/bin/python -I P1_CommitThenSample/figures/tab5_e6_rev1.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results" / "summary" / "E6_overheads_rev1.json"
FLOOD = ROOT / "results" / "summary" / "E6_flood.json"
REAL = ROOT / "results" / "summary" / "E6_realstream_rev1.json"
OUT = ROOT / "figures" / "tab5_e6_rev1.tex"
CAP = ROOT / "figures" / "tab5_e6_rev1.caption.txt"


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


def sig2(n):
    """A supported flood size rounded DOWN to three significant digits (conservative; e.g. 105,377 -> 105,000)."""
    if n is None:
        return "--"
    from math import floor, log10
    p = 10 ** max(0, int(floor(log10(n))) - 2)
    return f"{int(floor(n / p) * p):,}"


def main() -> None:
    d = json.loads(SUM.read_text())
    fl = json.loads(FLOOD.read_text()) if FLOOD.exists() else {}
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
    vP = g(d, "verifier", "P_cached", default=g(d, "verifier", "P", default={}))
    vE = g(d, "verifier", "E_cached", default=g(d, "verifier", "E", default={}))

    def ms(v, key, div=1.0):
        x = v.get(key)
        return None if x is None else 1e3 * x / div

    rows.append(r"Verifier per epoch, with prime bundle and compliance (ms) & " +
                f"{fmt(ms(vP, 'median_s_per_epoch'), 0)} & {fmt(ms(vE, 'median_s_per_epoch'), 0)}\\\\")
    rows.append(r"Verifier per epoch, validity only$^{\ddagger}$ (ms) & " +
                f"{fmt(ms(vP, 'median_s_per_5_epoch_window_fast_no_prime', 5), 1)} & "
                f"{fmt(ms(vE, 'median_s_per_5_epoch_window_fast_no_prime', 5), 1)}\\\\")
    sup = {c: g(fl, "cores", c, "supported", default={}) for c in ("P", "E")}

    def flood_n(c, key):
        return g(sup[c], key, "one_round_ahead", "n")

    def cap_range(c):
        rs = [r for r in g(fl, "cores", c, "rows", default=[]) if r.get("cap") and r.get("kind") == "zero"]
        if not rs:
            return "--"
        lo, hi = min(r["commit_s"] for r in rs), max(r["commit_s"] for r in rs)
        return f"{1e3 * lo:.0f}--{1e3 * hi:.0f}"

    rows.append(r"Sustained SYN flood within $\Delta_{\mathrm{build}}$ (units/epoch)$^{\S}$ & " +
                f"{sig2(flood_n('P', 'zero/sustained/incremental/nocap'))} & "
                f"{sig2(flood_n('E', 'zero/sustained/incremental/nocap'))}\\\\")
    rows.append(r"Commit with zero-payload cap, $10^4$--$10^6$ units (ms) & " + f"{cap_range('P')} & {cap_range('E')}\\\\")
    rs = json.loads(REAL.read_text()) if REAL.exists() else {}

    def real(c, sid="lp_iot23_49-1_p10000"):
        v = g(rs, "cores", c, sid, default={})
        if not v.get("resort") or not v.get("incremental"):
            return "--"
        return f"{v['resort']['commit_s_median']:.2f} $\\to$ {v['incremental']['commit_s_median']:.2f}"

    if rs:
        rows.append(r"Commit on LP-IoT23(49-1), reviewed $\to$ revised statistics (s)$^{\P}$ & " + f"{real('P')} & {real('E')}\\\\")
    rows.append(r"\midrule")
    rows.append(r"Header; inclusion proof$^\dagger$; VRF proof$^\dagger$ & \multicolumn{2}{r}{272\,B; "
                r"$32\lceil\log_2 n_e\rceil$\,B; 80\,B}\\")
    tc = d.get("tau_column", {})
    if tc:
        rows.append(r"Committed $\tau$ column$^\dagger$ & \multicolumn{2}{r}{8\,B per payload unit; " +
                    f"+{100 * tc['increase']:.1f}\\% metadata, +{100 * tc['over_pcap']:.2f}\\% of pcap" + r"}\\")
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
    rows.append(r"Storage / pcap: LP-IoT23; LP-Edge & \multicolumn{2}{r}{" +
                (f"{100 * min(iot):.1f}--{100 * max(iot):.1f}\\%" if iot else "--") + "; " +
                (f"{100 * edge[0]:.1f}\\%" if edge else "--") + r"}\\")
    ai = g(d, "pipeline_variants", "as_implemented", default={})
    rev0 = ROOT / "results" / "summary" / "E6_overheads.json"
    ai0 = g(json.loads(rev0.read_text()), "pipeline_variants", "as_implemented", default={}) if rev0.exists() else {}
    notes = []
    if star:
        s = (r"$^\ast$Median of 5 single-core runs over 100 epochs (" + f"{g(P, 'packets', default=0):,}" +
             r" packets) with the constant $(\hash(\theta),\hash(\text{policy}))$ memoised")
        if ai:
            # rev1 code memoises (H(θ), H(policy)) per policy instance itself: the as-implemented runs only differ by
            # run-to-run noise; the reviewed (rev0) prototype recomputed H(θ) in every commit
            s += (r", as the revised prototype itself does (without the external memo: "
                  + f"{fmt(g(ai, 'P', 'us_per_packet', 'total'))}/{fmt(g(ai, 'E', 'us_per_packet', 'total'))}"
                  + r"\,\textmu s, P/E); the p99 commit includes Python garbage-collection pauses that fall into a "
                  r"commit (" + f"{fmt(g(ai, 'P', 'commit_latency_ms_p99'), 0)}/{fmt(g(ai, 'E', 'commit_latency_ms_p99'), 0)}"
                  + r"\,ms in the runs without the memo, where they fall elsewhere)")
        if ai0:
            s += (r"; the reviewed prototype recomputed $\hash(\theta)$ in every commit: "
                  + f"{fmt(g(ai0, 'P', 'us_per_packet', 'total'))}/{fmt(g(ai0, 'E', 'us_per_packet', 'total'))}"
                  + r"\,\textmu s per packet and " + f"{fmt(g(ai0, 'P', 'commit_latency_ms_median'), 0)}/"
                  + f"{fmt(g(ai0, 'E', 'commit_latency_ms_median'), 0)}" + r"\,ms commits")
        notes.append(s + ".")
    notes.append(r"$^{\ddagger}$V1--V8 with the committed $\tau_i$ and the floor check only: no prime bundle, no "
                 r"recomputation of $\tau_i$ through the policy (Alg.~\ref{alg:verify}).")
    notes.append(r"$^{\S}$One compromised device opening a new connection per SYN in every epoch of the 120-epoch "
                 r"window; with the zero-payload cap (8,192 units per device and epoch) the commit no longer grows with "
                 r"the flood.")
    if REAL.exists():
        notes.append(r"$^{\P}$Median over the epochs with a complete 120-epoch window; the infected host scans "
                     r"continuously (6,125 zero-payload units per epoch); one round ahead allows 0.8\,s.")
    notes.append(r"Buffer maximum: one 196-MB transfer of one CIC device within one epoch.")
    note = r"\multicolumn{3}{@{}p{\columnwidth}@{}}{\tiny " + " ".join(notes) + r"}\\" + "\n"
    tex = r"""\begin{table}[!t]
\centering
\caption{E6: overheads, sizes, storage and witness load ($\dagger$: by construction).}
\label{tab:e6}
\scriptsize
\setlength{\tabcolsep}{3pt}
% generated by figures/tab5_e6_rev1.py from results/summary/E6_overheads_rev1.json and E6_flood.json
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
    loads = {c: g(d, "pipeline", c, "loadavg_1min") for c in ("P", "E")}
    facts = {
        "table": "Table V rev1 (tab:e6)", "source": [str(SUM.relative_to(ROOT)), str(FLOOD.relative_to(ROOT))],
        "cores_measured_rev1": sorted(g(d, "pipeline", default={}).keys()),
        "pipeline_loadavg_1min_per_run": loads,
        "fallbacks_to_rev0": "primitives (Ed25519/ECVRF microbenchmarks), buffer and storage rows: rev0 measurements "
                             "(code unchanged); storage rows exclude the tau column (+0.02% of pcap)",
        "headroom_P": d.get("headroom_P"), "headroom_E": d.get("headroom_E"),
        "flood_supported_one_round_ahead": {c: {k: g(v, "one_round_ahead", "n") for k, v in sup[c].items()}
                                            for c in sup if sup[c]},
        "flood_supported_two_rounds_ahead": {c: {k: g(v, "two_rounds_ahead", "n") for k, v in sup[c].items()}
                                             for c in sup if sup[c]},
        "verifier_validity_only_ms_per_epoch": {"P": ms(vP, "median_s_per_5_epoch_window_fast_no_prime", 5),
                                                "E": ms(vE, "median_s_per_5_epoch_window_fast_no_prime", 5)},
        "tau_column": tc}
    CAP.write_text(json.dumps(facts, indent=1, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
