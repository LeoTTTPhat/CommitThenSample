"""Table II (paper Table III, label tab:data): datasets, streams and estimands -- realised prevalence, floor
phi0, payload units per epoch, metadata-store/full-capture ratio, number of evaluated estimands, and the range
of metadata-predictability AUC per estimand family (with the CIC-vs-IoT-23 domain-shortcut caveat). Owner: P1-F.

Reads results/streams/<id>/meta.json, results/models/<fold>/<seed>/auc.json, results/designs/<id>/U1_bm_0.05.json.
Writes figures/tab2_data.tex (booktabs, ready to replace the tab:data body) and results/summary/tab2_data.json.

    .venv/bin/python -I P1_CommitThenSample/figures/tab2_data.py
"""
from __future__ import annotations

import glob
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGDIR = Path(__file__).resolve().parent
SUMMARY = RESULTS / "summary"

FOLDS = ["1-1", "3-1", "7-1", "34-1", "48-1", "49-1"]
PPMS = [1000, 10000, 50000]
RHO = 0.05
# estimand families for the AUC ranges
FAMILIES = {"D (detectors)": [f"D{i}" for i in range(1, 14)] + ["D1o", "D1c", "D2o", "D2c", "D11o"],
            "S (Suricata)": [f"S{i}" for i in range(1, 7)],
            "L (labels)": ["L1", "L2", "L3", "L4", "L5"],
            "g_mal": ["g_mal"]}


def _meta(sid):
    p = RESULTS / "streams" / sid / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def _committed_floor(fold, pkey="p10000"):
    """Committed floor phi0 at rho=5% from the budget-matched size-only design json."""
    for seed in (0, 1, 2):
        j = RESULTS / "designs" / f"lp_iot23_{fold}_{pkey}" / f"U1_bm_{RHO:g}.json"
        if j.exists():
            d = json.loads(j.read_text())
            # committed_floor applied
            from ctsample.design.probs import committed_floor
            return committed_floor(d["phi0_committed"] or d["phi0_stream"])
    return None


def _meta_over_pcap(sid):
    """zlib-9 compressed metadata store over full-capture bytes (Sum_all s). zstd (the deployed codec) is a
    little smaller; zlib-9 is a reproducible upper proxy. Full capture = Sum (caplen+16) = meta.extra.sum_s_all."""
    import polars as pl
    sd = RESULTS / "streams" / sid
    m = _meta(sid)
    if m is None:
        return None
    pcap = float(m["extra"].get("sum_s_all", 0.0))
    try:
        u = pl.read_parquet(sd / "units.parquet", columns=["meta"])
        blob = b"".join(bytes(x) for x in u["meta"].to_list())
        comp = len(zlib.compress(blob, 9))
    except Exception:  # noqa: BLE001
        # fall back to the uncommitted encoded size (138 B payload / 68 B zero-payload)
        npay = m["extra"].get("n_payload_units", 0)
        nunit = m["extra"].get("n_units", m.get("n_units", 0))
        comp = 138 * npay + 68 * (nunit - npay)
    return comp / pcap if pcap else None


def _auc_ranges(stream_ids):
    """Per-family AUC range over the given streams (eligible pairs with >= 50 eligible units, n_pos >= 1)."""
    byfam = {k: [] for k in FAMILIES}
    for sid, fold in stream_ids:
        for seed in (0, 1, 2):
            p = RESULTS / "models" / fold / str(seed) / "auc.json"
            if not p.exists():
                continue
            a = json.loads(p.read_text()).get(sid, {})
            for fam, qs in FAMILIES.items():
                for q in qs:
                    e = a.get(q)
                    if e and e.get("auc") is not None and e.get("n_elig", 0) >= 50:
                        byfam[fam].append(float(e["auc"]))
            break
    return {k: (min(v), max(v), len(v)) for k, v in byfam.items() if v}


def _fmt_range(vals, pct=False, nd=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    lo, hi = min(vals), max(vals)
    s = (lambda x: f"{100 * x:.{nd}f}") if pct else (lambda x: f"{x:.{nd}f}")
    return s(lo) if abs(hi - lo) < 10 ** (-nd) else f"{s(lo)}--{s(hi)}"


def group_row(name, stream_ids, label_est_families):
    """One table row for a group of streams (ranges across its streams/folds)."""
    prev = {ppm: [] for ppm in PPMS}
    puep, phi0, mop = [], [], []
    n_est = []
    nat_prev = []
    for sid, fold in stream_ids:
        m = _meta(sid)
        if m is None:
            continue
        tail = sid.rsplit("_p", 1)
        ppm = int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else None
        if ppm in prev:
            prev[ppm].append(m["extra"].get("realised_prevalence"))
        elif ppm is None:
            nat_prev.append(m["extra"].get("realised_prevalence"))
        puep.append(m["extra"].get("payload_units_per_epoch"))
    headline = [(sid, fold) for sid, fold in stream_ids if sid.endswith("_p10000")] or stream_ids
    for sid, fold in headline:
        f0 = _committed_floor(fold) if name.startswith("LP-IoT23") else _committed_floor_generic(sid)
        if f0:
            phi0.append(f0)
        r = _meta_over_pcap(sid)
        if r:
            mop.append(r)
        m = _meta(sid)
        if m:
            n_est.append(len([q for q in m["queries"]]))
    auc = _auc_ranges(headline)
    rp_out = {str(k): _fmt_range(v, pct=True, nd=2) for k, v in prev.items()}
    if nat_prev:
        rp_out["nat"] = _fmt_range(nat_prev, pct=True, nd=1)
    return {"name": name, "realised_prevalence": rp_out,
            "payload_units_per_epoch": _fmt_range(puep, nd=0), "phi0": _fmt_range(phi0, nd=4),
            "meta_over_pcap": _fmt_range(mop, nd=3), "n_estimands": _fmt_range(n_est, nd=0),
            "auc_by_family": {k: {"lo": v[0], "hi": v[1], "n": v[2]} for k, v in auc.items()}}


def _committed_floor_generic(sid):
    j = RESULTS / "designs" / sid / f"U1_bm_{RHO:g}.json"
    if j.exists():
        from ctsample.design.probs import committed_floor
        d = json.loads(j.read_text())
        return committed_floor(d["phi0_committed"] or d["phi0_stream"])
    return None


def latex_table(rows, auc_overall, caveat):
    def auc_cell(r):
        fams = r["auc_by_family"]
        parts = []
        for fam in ("D (detectors)", "S (Suricata)", "L (labels)"):
            if fam in fams:
                parts.append(f"{fams[fam]['lo']:.2f}--{fams[fam]['hi']:.2f}")
        return "/".join(parts) if parts else "--"

    lines = [
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Streams: realised prevalence of malicious payload units (targets 0.1/1/5\%), floor "
        r"$\varphi_0$ at $\rho{=}5\%$, payload units per epoch, zstd metadata store over full capture, "
        r"evaluated estimands, and AUC range of metadata predictability (detector/Suricata/label families).}",
        r"\label{tab:data}", r"\scriptsize", r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{@{}l r r r r r l@{}}", r"\toprule",
        r"Stream & \makecell[r]{Realised\\prev.\ (\%)} & \makecell[r]{Payload\\units/ep.} & $\varphi_0$ & "
        r"\makecell[r]{Meta/\\pcap} & \makecell[r]{Esti-\\mands} & \makecell[l]{AUC\\(D/S/L)}\\",
        r"\midrule",
    ]
    for r in rows:
        rp = r["realised_prevalence"]
        prev = (rp["nat"] + r" (nat.)") if rp.get("nat") and all(rp.get(str(p), "--") == "--" for p in PPMS) \
            else "/".join(rp.get(str(p), "--") for p in PPMS)
        lines.append(f"{r['name']} & {prev} & {r['payload_units_per_epoch']} & {r['phi0']} & "
                     f"{r['meta_over_pcap']} & {r['n_estimands']} & {auc_cell(r)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\\[2pt]\scriptsize $^{\dagger}$The high $g_{\mathrm{mal}}$/\,L1 AUC "
              f"({auc_overall['g_mal'][0]:.2f}--{auc_overall['g_mal'][1]:.2f}) is largely the CIC-vs-IoT-23 "
              r"domain contrast, not within-domain predictability (Section~\ref{sec:e3}).",
              r"\end{table}"]
    return "\n".join(lines)


def main():
    lp = [(f"lp_iot23_{k}_p{p}", k) for k in FOLDS for p in PPMS]
    edge = [(f"lp_edge_p{p}", "all") for p in PPMS]
    gotham = [("lp_gotham_nat", "all"), ("lp_gotham_p10000", "all")]
    rows = [group_row("LP-IoT23 (6 folds)", lp, "iot23"),
            group_row("LP-Edge", edge, "edge"),
            group_row("LP-Gotham (nat.)", [("lp_gotham_nat", "all")], "gotham")]
    # overall AUC incl. g_mal for the caveat footnote
    auc_overall = _auc_ranges([(f"lp_iot23_{k}_p10000", k) for k in FOLDS])
    auc_overall = {k: (v[0], v[1], v[2]) for k, v in auc_overall.items()}
    caveat = "g_mal/L1 AUC is largely the CIC-vs-IoT-23 domain contrast, not within-domain predictability."
    tex = latex_table(rows, auc_overall, caveat)
    (FIGDIR / "tab2_data.tex").write_text(tex + "\n")
    SUMMARY.mkdir(parents=True, exist_ok=True)
    out = {"rows": rows, "auc_overall_incl_gmal": {k: {"lo": v[0], "hi": v[1], "n": v[2]}
                                                   for k, v in auc_overall.items()},
           "caveat": caveat, "rho": RHO, "meta_over_pcap_codec": "zlib-9 proxy for zstd",
           "full_capture_bytes": "Sum (caplen+16) = meta.extra.sum_s_all"}
    (SUMMARY / "tab2_data.json").write_text(json.dumps(out, indent=2))
    print(tex)
    print(f"\n[tab2] wrote {FIGDIR / 'tab2_data.tex'} and {SUMMARY / 'tab2_data.json'}")


if __name__ == "__main__":
    main()
