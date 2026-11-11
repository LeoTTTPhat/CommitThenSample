"""S3: unit labels and the merged per-unit estimand tables (DESIGN §5.4, §5.5).

  results/labels/<ds>/<src>.parquet   unit_gid, label (Benign|Malicious|unknown), detailed_label,
                                      family, class, is_malicious, label_source, conn_uid,
                                      match_dstart_s, incident_id (+ <src>.qa.json)
  results/tables/<ds>/<src>.parquet   unit_gid, epoch, index, pay, fps, s, device_id, labels, and per
                                      estimand f_<id>, elig_<id>, b_<id> for D*, S1–S6 (kept sids of
                                      the subset audit), L1–L5 / L_<class>, metadata indicators
  results/qa/<ds>.json ["labels"], ["tables"]

``--labels-only`` writes only the labels (needs S1 only). Tables need S2 for the S families;
without alerts they are written with null S columns and ``suricata_ready = false``
(rerun with ``--force-tables`` once S2 is done). Run under memguard::

    .venv/bin/python memguard/memguard.py run --name p1_s3_iot23 --workers 8 -- \
        .venv/bin/python -I -u P1_CommitThenSample/scripts/03_build_tables.py --dataset iot23
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

from ctsample.utils.io import DATA, RESULTS, atomic_write_json, atomic_write_parquet, is_done
from ctsample.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory, workers

LABEL_COLS = ["unit_gid", "label", "detailed_label", "family", "class", "is_malicious", "label_source", "conn_uid",
              "match_dstart_s", "incident_id"]
UNIT_LABEL_COLS = ["unit_gid", "pay", "proto", "ip_a", "port_a", "ip_b", "port_b", "mac_a", "first_ts_us",
                   "last_ts_us", "device_id", "pkts0", "pkts1"]
TABLE_UNIT_COLS = ["unit_gid", "source", "epoch", "index", "proto", "port_a", "port_b", "pay", "fps", "continuation",
                   "s", "device_id", "dir_class", "conn_state", "pkts0", "pkts1", "payload0", "payload1",
                   "first_ts_us", "n_syn", "n_synack"]
IOT23_L = {"L2": "C&C", "L3": "FileDownload", "L4": "Attack", "L5": "Okiru"}
EDGE_L_CLASSES = ("Backdoor", "OS_Fingerprinting", "Password", "Port_Scanning", "Ransomware", "SQL_injection",
                  "Uploading", "Vulnerability_scanner", "XSS")
GOTHAM_L_CLASSES = ("Network Scanning", "Brute Force", "Infection", "C&C Communication", "DoS")
SCAN_STATES = (0, 3, 8, 10)          # S0, REJ, RSTOS0, SH  (encode.CONN_STATES)


def lab_paths(ds, name):
    return RESULTS / "labels" / ds / f"{name}.parquet", RESULTS / "labels" / ds / f"{name}.qa.json"


def read_units(ds: str, name: str, cols: list[str]):
    import polars as pl
    return pl.read_parquet(str(RESULTS / "units" / ds / name / "part-*.parquet"), columns=cols)


def build_labels(ds: str, src: dict, force: bool) -> dict:
    import polars as pl
    name = src["source"]
    out, qa_p = lab_paths(ds, name)
    if out.exists() and qa_p.exists() and not force:
        q = json.loads(qa_p.read_text())
        q["skipped"] = True
        return q
    t0 = time.time()
    units = read_units(ds, name, UNIT_LABEL_COLS)
    if ds == "iot23":
        from ctsample.labels import iot23
        log = src.get("conn_log")
        conns = None
        if log:
            t_hi = units["last_ts_us"].max() / 1e6 + iot23.SLACK_S
            conns = iot23.load_connlog(DATA / log, None, t_hi)
        lab, qa = iot23.label_source(units, conns, name, default_benign=(not log and src.get("role") == "honeypot"))
        qa["conn_log"] = log
    elif ds == "edgeiiot":
        from ctsample.labels import edgeiiot
        lab, qa = edgeiiot.label_source(units, src)
    elif ds == "ciciot2023_benign":
        from ctsample.labels import cic_benign
        lab, qa = cic_benign.label_source(units, src)
    elif ds == "gotham":
        from ctsample.labels import gotham
        lab, qa = gotham.label_source(units, src)
    else:
        raise ValueError(ds)
    lab = lab.select(LABEL_COLS).with_columns(pl.lit(name).alias("source"))
    atomic_write_parquet(out, lab)
    qa.update(source=name, dataset=ds, elapsed_s=time.time() - t0)
    atomic_write_json(qa_p, qa)
    return qa


def kept_sids(ds: str):
    """Kept sids of the pooled subset audit (rules/audit_all.json; one list for every stream). Falls back
    to the dataset's own audit if the pooled one does not cover this dataset."""
    pa_ = RESULTS / "rules" / "audit_all.json"
    if pa_.exists():
        a = json.loads(pa_.read_text())
        if any(d["dataset"] == ds for d in a.get("datasets_included", [])):
            return set(int(x) for x in a["kept_sids"])
    p = RESULTS / "rules" / f"audit_{ds}.json"
    if not p.exists():
        return None
    return set(int(x) for x in json.loads(p.read_text())["kept_sids"])


def build_table(ds: str, src: dict, force: bool) -> dict:
    import polars as pl
    from ctsample.content.detectors import AUX_RAW_INDICATORS, DET_ESTIMAND_IDS, eligibility_exprs
    from ctsample.content.suricata import FAMILIES
    name = src["source"]
    out = RESULTS / "tables" / ds / f"{name}.parquet"
    kept = kept_sids(ds)
    alerts_p = RESULTS / "alerts" / ds / f"{name}.parquet"
    ready = kept is not None and alerts_p.exists()
    if out.exists() and not force:
        return {"source": name, "skipped": True}
    t0 = time.time()
    u = read_units(ds, name, TABLE_UNIT_COLS)
    lab = pl.read_parquet(str(lab_paths(ds, name)[0]))
    # detector estimands
    h = pl.read_parquet(str(RESULTS / "hits" / ds / f"{name}.parquet"))
    hv = (h.pivot(on="query_id", index="unit_gid", values="value", aggregate_function="sum")
          if h.height else pl.DataFrame({"unit_gid": pl.Series([], dtype=pl.Int64)}))
    t = u.join(hv, on="unit_gid", how="left")
    exprs = []
    for q in list(DET_ESTIMAND_IDS) + list(AUX_RAW_INDICATORS):
        exprs.append((pl.col(q) if q in t.columns else pl.lit(0)).fill_null(0).cast(pl.Int32).alias(f"f_{q}"))
    t = t.with_columns(exprs).drop([c for c in t.columns if c in set(DET_ESTIMAND_IDS) | set(AUX_RAW_INDICATORS)])
    el = eligibility_exprs()
    t = t.with_columns([v[0].alias(f"elig_{k}") for k, v in el.items()] +
                       [v[1].alias(f"b_{k}") for k, v in el.items()])
    # Suricata families (kept sids of the subset audit)
    fam_q = {}
    if ready:
        al = pl.read_parquet(str(alerts_p), columns=["unit_gid", "sid", "family"]).filter(pl.col("unit_gid") >= 0)
        n_alerts = al.height
        alk = al.filter(pl.col("sid").is_in(sorted(kept)))
        fam = (alk.filter(pl.col("family").is_not_null()).select("unit_gid", "family").unique()
               .with_columns(pl.lit(1, dtype=pl.Int32).alias("v"))
               .pivot(on="family", index="unit_gid", values="v", aggregate_function="max"))
        anyk = alk.group_by("unit_gid").agg(pl.col("sid").n_unique().alias("n_kept_sids"))
        t = t.join(fam, on="unit_gid", how="left").join(anyk, on="unit_gid", how="left")
        t = t.with_columns([(pl.col(f) if f in t.columns else pl.lit(0)).fill_null(0).cast(pl.Int32).alias(f"f_{f}")
                            for f in FAMILIES] + [pl.col("n_kept_sids").fill_null(0).cast(pl.Int32)])
        t = t.drop([f for f in FAMILIES if f in t.columns])
        fam_q = {"n_alerts": n_alerts, "n_alerts_kept_sids": alk.height,
                 "units_with_kept_alert": int((t["n_kept_sids"] > 0).sum())}
    else:
        t = t.with_columns([pl.lit(None, dtype=pl.Int32).alias(f"f_{f}") for f in FAMILIES] +
                           [pl.lit(None, dtype=pl.Int32).alias("n_kept_sids")])
    t = t.with_columns([pl.col("pay").alias(f"elig_{f}") for f in FAMILIES] +
                       [pl.col("pay").cast(pl.Int64).alias(f"b_{f}") for f in FAMILIES])
    # labels and label estimands
    t = t.join(lab.drop("source"), on="unit_gid", how="left")
    known = pl.col("label").is_not_null() & (pl.col("label") != "unknown")
    mal_pay = pl.col("pay") & pl.col("is_malicious").fill_null(False)
    lcols = [(mal_pay & known).cast(pl.Int32).alias("f_L1"), (pl.col("pay") & known).alias("elig_L1"),
             (pl.col("pay") & known).cast(pl.Int64).alias("b_L1")]
    if ds == "iot23":
        for k, fam in IOT23_L.items():
            lcols += [(mal_pay & known & (pl.col("family") == fam)).fill_null(False).cast(pl.Int32).alias(f"f_{k}"),
                      (pl.col("pay") & known).alias(f"elig_{k}"), (pl.col("pay") & known).cast(pl.Int64).alias(f"b_{k}")]
    if ds == "edgeiiot":
        for c in EDGE_L_CLASSES:
            lcols += [(mal_pay & (pl.col("class") == c)).fill_null(False).cast(pl.Int32).alias(f"f_L_{c}"),
                      pl.col("pay").alias(f"elig_L_{c}"), pl.col("pay").cast(pl.Int64).alias(f"b_L_{c}")]
    if ds == "gotham":
        for c in GOTHAM_L_CLASSES:
            k = "L_" + c.replace(" ", "_").replace("&", "")
            lcols += [(mal_pay & known & (pl.col("class") == c)).fill_null(False).cast(pl.Int32).alias(f"f_{k}"),
                      (pl.col("pay") & known).alias(f"elig_{k}"), (pl.col("pay") & known).cast(pl.Int64).alias(f"b_{k}")]
    # metadata (exact) indicators
    lcols += [((~pl.col("pay")) & (pl.col("proto") == 6) & pl.col("conn_state").is_in(SCAN_STATES))
              .cast(pl.Int32).alias("m_scan_unit"),
              (pl.col("detailed_label").fill_null("").str.contains("PartOfAHorizontalPortScan")).cast(pl.Int32)
              .alias("m_hpscan_label")]
    t = t.with_columns(lcols)
    t = t.with_columns(pl.lit(ready).alias("suricata_ready"))
    atomic_write_parquet(out, t)
    return {"source": name, "rows": t.height, "suricata_ready": ready, **fam_q, "elapsed_s": time.time() - t0,
            "f_sums": {c: int(t[c].sum()) for c in t.columns if c.startswith("f_") and t[c].null_count() == 0},
            "elig_counts": {c: int(t[c].sum()) for c in t.columns if c.startswith("elig_")}}


def _worker(kind, ds, src, force):
    try:
        return (build_labels if kind == "labels" else build_table)(ds, src, force)
    except Exception as e:  # noqa: BLE001
        return {"source": src["source"], "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


def run_pool(kind, ds, sources, force, nw):
    res, failed = {}, []
    ex = ProcessPoolExecutor(max_workers=nw, mp_context=get_context("spawn"))
    try:
        pending = {ex.submit(_worker, kind, ds, s, force) for s in sources}
        while pending:
            done, pending = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
            for f in done:
                q = f.result()
                if "error" in q:
                    failed.append(q["source"])
                    print(f"[s3] {kind} FAILED {q['source']}: {q['error']}\n{q['traceback']}", flush=True)
                else:
                    res[q["source"]] = q
                    if kind == "labels" and not q.get("skipped"):
                        print(f"[s3] labels {q['source']}: units {q.get('units')}, match payload "
                              f"{q.get('match_rate_payload')}, all {q.get('match_rate_all')} "
                              f"({q.get('elapsed_s', 0):.0f}s)", flush=True)
                    elif not q.get("skipped"):
                        print(f"[s3] table {q['source']}: rows {q.get('rows')} suricata_ready {q.get('suricata_ready')} "
                              f"({q.get('elapsed_s', 0):.0f}s)", flush=True)
            check_memory()
    except MemoryBudgetExceeded as e:
        print(f"[s3] memory budget: {e}; exit {EXIT_MEMORY}", flush=True)
        for p in list(getattr(ex, "_processes", {}).values()):
            p.terminate()
        ex.shutdown(wait=False, cancel_futures=True)
        sys.exit(EXIT_MEMORY)
    ex.shutdown(wait=True)
    return res, failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["iot23", "ciciot2023_benign", "edgeiiot", "gotham"])
    ap.add_argument("--sources", nargs="*")
    ap.add_argument("--labels-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo labels and tables")
    ap.add_argument("--force-tables", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("POLARS_MAX_THREADS", "4")
    inv = json.loads((RESULTS / "inventory.json").read_text())
    sources = [s for s in inv["datasets"][a.dataset]["sources"] if is_done(RESULTS / "units" / a.dataset / s["source"])]
    if a.sources:
        sources = [s for s in sources if s["source"] in set(a.sources)]
    sources.sort(key=lambda s: -(s.get("size") or 0))
    nw = max(1, min(workers(default=8), 8, len(sources)))
    t0 = time.time()
    labs, f1 = run_pool("labels", a.dataset, sources, a.force, nw)
    qa_p = RESULTS / "qa" / f"{a.dataset}.json"
    qa = json.loads(qa_p.read_text()) if qa_p.exists() else {"dataset": a.dataset}
    pay = sum((q.get("payload_units") or 0) for q in labs.values())
    pay_m = sum((q.get("payload_units") or 0) * (q.get("match_rate_payload") or 0) for q in labs.values())
    qa["labels"] = {"n_sources": len(labs), "failed": f1, "payload_units": pay,
                    "match_rate_payload_pooled": pay_m / pay if pay else None,
                    "per_source": {k: {x: v.get(x) for x in ("units", "payload_units", "match_rate_payload",
                                                             "match_rate_all", "match_rate_tcp_udp",
                                                             "label_source_counts_payload", "label_counts_payload",
                                                             "family_counts_mal_payload", "n_incidents",
                                                             "malicious_payload_units", "attack_window_us")}
                                   for k, v in sorted(labs.items())}}
    atomic_write_json(qa_p, qa)
    print(f"[s3] labels {a.dataset}: {len(labs)} sources, pooled payload match {qa['labels']['match_rate_payload_pooled']}",
          flush=True)
    f2 = []
    if not a.labels_only:
        tabs, f2 = run_pool("tables", a.dataset, sources, a.force or a.force_tables, nw)
        qa = json.loads(qa_p.read_text())
        qa["tables"] = {"n_sources": len(tabs), "failed": f2,
                        "suricata_ready_all": all(v.get("suricata_ready", True) for v in tabs.values()
                                                  if not v.get("skipped")),
                        "per_source": {k: {x: v.get(x) for x in ("rows", "suricata_ready", "n_alerts",
                                                                 "n_alerts_kept_sids", "units_with_kept_alert",
                                                                 "f_sums", "elig_counts")}
                                       for k, v in sorted(tabs.items()) if not v.get("skipped")}}
        atomic_write_json(qa_p, qa)
        kept = kept_sids(a.dataset)
        if kept is not None:
            from ctsample.content.mapping import observability_gap
            qa = json.loads(qa_p.read_text())
            qa["observability_gap"] = observability_gap(a.dataset, kept, [s["source"] for s in sources])
            atomic_write_json(qa_p, qa)
            og = qa["observability_gap"]
            if og.get("available"):
                print(f"[s3] observability gap {a.dataset}: ANY {og['families']['ANY']}, unattributed "
                      f"{og['n_orig_alerts_unattributed_no_pcap_cnt']} of {og['n_orig_alerts_kept_sids']}", flush=True)
    print(f"[s3] {a.dataset} finished in {time.time() - t0:.0f}s", flush=True)
    return 1 if (f1 or f2) else 0


if __name__ == "__main__":
    sys.exit(main())
