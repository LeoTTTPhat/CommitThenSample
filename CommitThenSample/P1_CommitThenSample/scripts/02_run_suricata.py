"""S2: Suricata on the per-unit replay pcaps (+ 10 % subsets; optionally the originals).

For every source of ``--dataset`` whose S1 outputs are complete:
  results/alerts/<ds>/<src>.parquet, <src>.sub10.parquet  (and <src>.orig.parquet with --originals)
      one row per eve alert: unit_gid (from the QinQ VLAN pair, or pcap_cnt→pktmap for originals),
      rid, sid, gid, rev, signature, category, classtype, family, severity, pcap_cnt, timestamp,
      5-tuple, app_proto, flow_id, stream, tx_id
  results/suricata/<ds>/<src>[.sub10|.orig]/{job.json, stats.log, suricata.log, stdout.txt}
      (eve.json is deleted after parsing unless --keep-eve)
  results/rules/audit_<ds>.json   subset-invariance audit pooled over the dataset's sources
  qa/<ds>.json ["suricata"]       alert counts, mapping rate, throughput

≤ 4 Suricata instances × 4 worker threads. Run from the project root under memguard::

    .venv/bin/python memguard/memguard.py run --name p1_s2_iot23 --workers 4 -- \
        .venv/bin/python -I -u P1_CommitThenSample/scripts/02_run_suricata.py --dataset iot23
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ctsample.content.mapping import map_pcap_cnt, map_vlans, rid_table, subset_audit
from ctsample.content.replay import VLAN_N
from ctsample.content.suricata import FAMILY_OF_CLASSTYPE, parse_eve_alerts, run_suricata, stats_counters
from ctsample.io.pcap_reader import PcapReader
from ctsample.units import _hash_shim as hs
from ctsample.utils.io import DATA, RESULTS, atomic_write_json, atomic_write_parquet, is_done
from ctsample.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory, workers

RULES = RESULTS / "rules" / "content.rules"
SIDS = RESULTS / "rules" / "content_sids.json"

ALERT_SCHEMA = pa.schema([
    ("unit_gid", pa.int64()), ("rid", pa.int64()), ("source", pa.string()), ("run", pa.string()),
    ("sid", pa.int64()), ("gid", pa.int32()), ("rev", pa.int32()), ("signature", pa.string()),
    ("category", pa.string()), ("classtype", pa.string()), ("family", pa.string()), ("severity", pa.int32()),
    ("pcap_cnt", pa.int64()), ("timestamp", pa.string()), ("src_ip", pa.string()), ("src_port", pa.int32()),
    ("dest_ip", pa.string()), ("dest_port", pa.int32()), ("proto", pa.string()), ("app_proto", pa.string()),
    ("flow_id", pa.int64()), ("stream", pa.int8()), ("tx_id", pa.int64()),
])


def job_paths(ds: str, src: str, run: str) -> dict:
    suf = {"full": "", "sub10": ".sub10", "orig": ".orig"}[run]
    return {"alerts": RESULTS / "alerts" / ds / f"{src}{suf}.parquet",
            "out": RESULTS / "suricata" / ds / f"{src}{suf}",
            "pcap": (RESULTS / "replay" / ds / f"{src}{suf}.pcap") if run != "orig" else None}


def run_job(ds: str, src: dict, run: str, threads: int, keep_eve: bool, sid_class: dict) -> dict:
    name = src["source"]
    jp = job_paths(ds, name, run)
    pcap = jp["pcap"] if run != "orig" else Path(src["read_path"])
    t0 = time.time()
    yml = str(RESULTS / "rules" / "suricata_alert_only.yaml") if run == "orig" else None
    r = run_suricata(pcap, jp["out"], RULES, threads=threads, yaml_path=yml)
    if r["rc"] != 0:
        return {"source": name, "run": run, "error": f"suricata rc={r['rc']}", "elapsed_s": r["elapsed_s"]}
    eve = jp["out"] / "eve.json"
    ctr0 = stats_counters(jp["out"] / "stats.log")
    n_rec = sum(1 for _ in PcapReader(pcap))
    if (ctr0.get("decoder.pkts") or 0) != n_rec:   # e.g. an interrupted run: never publish partial alerts
        return {"source": name, "run": run, "error": f"suricata decoded {ctr0.get('decoder.pkts')} of {n_rec} records",
                "elapsed_s": r["elapsed_s"]}
    cols, other = parse_eve_alerts(eve)
    n = len(cols["sid"])
    if run == "orig":
        gid = map_pcap_cnt(np.array(cols["pcap_cnt"], np.int64), RESULTS / "pktmap" / ds / f"{name}.npz")
        rid = np.full(n, -1, np.int64)
    else:
        gbr = rid_table(ds, name)
        o = np.array(cols["vlan_outer"], np.int64)
        i = np.array(cols["vlan_inner"], np.int64)
        gid = map_vlans(o, i, gbr)
        ok = (o >= 1) & (i >= 1)
        rid = np.where(ok, (o - 1) * VLAN_N + (i - 1), -1)
    ct = [sid_class.get(int(s), "") for s in cols["sid"]]
    tab = pa.table({
        "unit_gid": gid, "rid": rid, "source": [name] * n, "run": [run] * n,
        "sid": np.array(cols["sid"], np.int64), "gid": np.array(cols["gid"], np.int32),
        "rev": np.array(cols["rev"], np.int32), "signature": cols["signature"], "category": cols["category"],
        "classtype": ct, "family": [FAMILY_OF_CLASSTYPE.get(c) for c in ct],
        "severity": np.array(cols["severity"], np.int32), "pcap_cnt": np.array(cols["pcap_cnt"], np.int64),
        "timestamp": cols["timestamp"], "src_ip": cols["src_ip"],
        "src_port": np.array([x if x is not None else -1 for x in cols["src_port"]], np.int32),
        "dest_ip": cols["dest_ip"],
        "dest_port": np.array([x if x is not None else -1 for x in cols["dest_port"]], np.int32),
        "proto": cols["proto"], "app_proto": cols["app_proto"],
        "flow_id": np.array(cols["flow_id"], np.int64), "stream": np.array(cols["stream"], np.int8),
        "tx_id": np.array(cols["tx_id"], np.int64)}, schema=ALERT_SCHEMA)
    atomic_write_parquet(jp["alerts"], tab)
    ctr = stats_counters(jp["out"] / "stats.log")
    size = pcap.stat().st_size
    info = {"source": name, "run": run, "pcap": str(pcap), "pcap_bytes": size, "rc": r["rc"], "pcap_records": n_rec,
            "suricata_elapsed_s": r["elapsed_s"], "elapsed_s": time.time() - t0, "n_alerts": n,
            "n_alerts_mapped": int((gid >= 0).sum()), "n_alert_units": int(len(np.unique(gid[gid >= 0]))),
            "other_events": other, "stats": ctr, "MB_per_s": size / 1e6 / r["elapsed_s"], "cmd": r["cmd"],
            "eve_bytes": eve.stat().st_size if eve.exists() else 0}
    if not keep_eve:
        for f in ("eve.json", "fast.log"):
            (jp["out"] / f).unlink(missing_ok=True)
    atomic_write_json(jp["out"] / "job.json", info)
    return info


def _worker(*args):
    try:
        return run_job(*args)
    except Exception as e:  # noqa: BLE001
        return {"source": args[1]["source"], "run": args[2], "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}


def _collect(ds: str, sources: list[dict], full: dict, sub: dict, common: set) -> int:
    n_units = 0
    for s in sources:
        jf, js = job_paths(ds, s["source"], "full"), job_paths(ds, s["source"], "sub10")
        if not (jf["alerts"].exists() and js["alerts"].exists()):
            continue
        g = rid_table(ds, s["source"])
        g = g[g >= 0]
        n_units += len(g)
        common.update(int(x) for x in g if hs.sub10_selected(int(x)))
        for path, dst in ((jf["alerts"], full), (js["alerts"], sub)):
            t = pq.read_table(path, columns=["sid", "unit_gid"])
            for sid, u in zip(t.column("sid").to_numpy(), t.column("unit_gid").to_numpy()):
                if u >= 0:
                    dst.setdefault(int(sid), set()).add(int(u))
    return n_units


def _audit_out(res: dict, n_units: int, common: set, full: dict, sub: dict, extra: dict) -> dict:
    kept = sorted(k for k, v in res.items() if v["keep"])
    relaxed = sorted(k for k, v in res.items() if v["n_union"] and v["agreement"] is not None
                     and v["agreement"] >= 0.95 and v["n_full_all"] >= 5)
    return {**extra, "rule": "keep sid iff |A∩B|/|A∪B| >= 0.95 and |A∪B| >= 5 on the 10% subset units",
            "n_payload_units": n_units, "n_common_units": len(common),
            "n_sids_alerting_full": len(full), "n_sids_alerting_sub": len(sub), "n_sids_evaluated": len(res),
            "n_kept": len(kept), "kept_sids": kept,
            "n_relaxed_kept_info_only": len(relaxed), "relaxed_kept_sids_info_only": relaxed,
            "relaxed_rule_info_only": "agreement >= 0.95 and >= 5 alerting units in the full run",
            "per_sid": {str(k): v for k, v in res.items()}}


def audit(ds: str, sources: list[dict]) -> dict:
    full, sub, common = {}, {}, set()
    n_units = _collect(ds, sources, full, sub, common)
    res = subset_audit(full, sub, common)
    out = _audit_out(res, n_units, common, full, sub, {"dataset": ds})
    atomic_write_json(RESULTS / "rules" / f"audit_{ds}.json", out)
    return out


def audit_all(inv: dict) -> dict:
    """Pooled audit over every dataset with S2 outputs (one kept-sid list for all streams)."""
    full, sub, common = {}, {}, set()
    n_units, used = 0, []
    for ds, d in inv["datasets"].items():
        srcs = [s for s in d["sources"] if job_paths(ds, s["source"], "full")["alerts"].exists()
                and job_paths(ds, s["source"], "sub10")["alerts"].exists()]
        if not srcs:
            continue
        # gids are unique per dataset only: prefix with the dataset name
        f2, s2, c2 = {}, {}, set()
        n_units += _collect(ds, srcs, f2, s2, c2)
        for src_d, dst in ((f2, full), (s2, sub)):
            for sid, us in src_d.items():
                dst.setdefault(sid, set()).update((ds, u) for u in us)
        common.update((ds, u) for u in c2)
        used.append({"dataset": ds, "n_sources": len(srcs), "n_sources_total": len(d["sources"])})
    res = subset_audit(full, sub, common)
    out = _audit_out(res, n_units, common, full, sub, {"dataset": "all", "datasets_included": used})
    atomic_write_json(RESULTS / "rules" / "audit_all.json", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["iot23", "ciciot2023_benign", "edgeiiot", "gotham"])
    ap.add_argument("--sources", nargs="*")
    ap.add_argument("--runs", nargs="*", default=["full", "sub10"], choices=["full", "sub10", "orig"])
    ap.add_argument("--originals", action="store_true", help="also run on the original pcaps (SHOULD)")
    ap.add_argument("--instances", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-eve", action="store_true")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--wait-s1", action="store_true", help="poll until the dataset's S1 sources are DONE")
    ap.add_argument("--audit-only", action="store_true", help="only recompute audit_<ds>.json and audit_all.json")
    a = ap.parse_args()
    if a.audit_only:
        inv0 = json.loads((RESULTS / "inventory.json").read_text())
        r = audit(a.dataset, list(inv0["datasets"][a.dataset]["sources"]))
        aa = audit_all(inv0)
        print(f"[s2] audit {a.dataset}: kept {r['n_kept']}; pooled: kept {aa['n_kept']} of {aa['n_sids_evaluated']}",
              flush=True)
        return 0
    inv = json.loads((RESULTS / "inventory.json").read_text())
    sources = list(inv["datasets"][a.dataset]["sources"])
    if a.sources:
        sources = [s for s in sources if s["source"] in set(a.sources)]
    runs = list(a.runs) + (["orig"] if a.originals and "orig" not in a.runs else [])
    if "orig" in runs:
        from ctsample.content.suricata import make_alert_only_yaml
        make_alert_only_yaml(RESULTS / "rules" / "suricata_alert_only.yaml")
    sid_info = json.loads(SIDS.read_text())["sids"]
    sid_class = {int(k): v["classtype"] for k, v in sid_info.items()}
    nproc = max(1, min(a.instances, 4, workers(default=4)))
    t0 = time.time()
    done_jobs, failed, submitted = [], [], set()
    ex = ProcessPoolExecutor(max_workers=nproc, mp_context=get_context("spawn"))
    pending = set()
    try:
        while True:
            # submit every job whose S1 output is complete (largest replay first)
            ready = []
            for s in sources:
                if not is_done(RESULTS / "units" / a.dataset / s["source"]):
                    continue
                for run in runs:
                    key = (s["source"], run)
                    if key in submitted:
                        continue
                    jp = job_paths(a.dataset, s["source"], run)
                    if jp["alerts"].exists() and not a.force:
                        submitted.add(key)
                        jj = jp["out"] / "job.json"
                        if jj.exists():
                            done_jobs.append(json.loads(jj.read_text()))
                        continue
                    pc = jp["pcap"] if run != "orig" else Path(s["read_path"])
                    ready.append((pc.stat().st_size if pc.exists() else 0, s, run))
            for _, s, run in sorted(ready, key=lambda x: -x[0]):
                pending.add(ex.submit(_worker, a.dataset, s, run, a.threads, a.keep_eve, sid_class))
                submitted.add((s["source"], run))
            all_s1 = all(is_done(RESULTS / "units" / a.dataset / s["source"]) for s in sources)
            if not pending and (all_s1 or not a.wait_s1):
                break
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for f in done:
                q = f.result()
                if "error" in q:
                    failed.append((q["source"], q["run"]))
                    print(f"[s2] FAILED {q['source']} {q['run']}: {q['error']}\n{q.get('traceback', '')}", flush=True)
                else:
                    done_jobs.append(q)
                    print(f"[s2] done {q['source']} {q['run']}: {q['n_alerts']} alerts "
                          f"({q['n_alerts_mapped']} mapped, {q['n_alert_units']} units) "
                          f"{q['suricata_elapsed_s']:.0f}s {q['MB_per_s']:.1f} MB/s", flush=True)
            use = check_memory()
            if done:
                print(f"[s2] {len(pending)} running, tree memory {use:.1f} GB, {time.time() - t0:.0f}s", flush=True)
    except MemoryBudgetExceeded as e:
        print(f"[s2] memory budget: {e}; exit {EXIT_MEMORY}", flush=True)
        for p in list(getattr(ex, "_processes", {}).values()):
            p.terminate()
        ex.shutdown(wait=False, cancel_futures=True)
        return EXIT_MEMORY
    ex.shutdown(wait=True)
    au = None if a.no_audit else audit(a.dataset, list(inv["datasets"][a.dataset]["sources"]))
    if not a.no_audit:
        aa = audit_all(inv)
        print(f"[s2] pooled audit over {[d['dataset'] for d in aa['datasets_included']]}: kept {aa['n_kept']} "
              f"of {aa['n_sids_evaluated']} sids", flush=True)
    # QA section
    qa_p = RESULTS / "qa" / f"{a.dataset}.json"
    qa = json.loads(qa_p.read_text()) if qa_p.exists() else {"dataset": a.dataset}
    tot_a = sum(j["n_alerts"] for j in done_jobs if j["run"] == "full")
    tot_m = sum(j["n_alerts_mapped"] for j in done_jobs if j["run"] == "full")
    tb = sum(j["pcap_bytes"] for j in done_jobs)
    ts = sum(j["suricata_elapsed_s"] for j in done_jobs)
    qa["suricata"] = {"n_jobs": len(done_jobs), "failed": failed, "alerts_full": tot_a,
                      "alert_to_unit_mapping_rate_full": tot_m / tot_a if tot_a else None,
                      "pcap_bytes": tb, "suricata_seconds_sum": ts,
                      "MB_per_s_per_instance": tb / 1e6 / ts if ts else None,
                      "audit": {k: au[k] for k in ("n_common_units", "n_sids_evaluated", "n_kept",
                                                   "n_relaxed_kept_info_only")} if au else None,
                      "jobs": {f"{j['source']}|{j['run']}": {k: j.get(k) for k in (
                          "n_alerts", "n_alerts_mapped", "n_alert_units", "suricata_elapsed_s", "MB_per_s",
                          "pcap_bytes", "stats")} for j in done_jobs}}
    atomic_write_json(qa_p, qa)
    print(f"[s2] {a.dataset}: {len(done_jobs)} jobs, failed {failed}, alerts(full) {tot_a}, mapped {tot_m}, "
          f"audit kept {au['n_kept'] if au else '-'} sids, {time.time() - t0:.0f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
