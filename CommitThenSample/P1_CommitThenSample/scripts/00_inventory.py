"""S0 inventory (DESIGN §5, rules box; §5.6 LOSO eligibility).

Builds ``results/inventory.json``: per dataset the list of pcap *sources* (path, size, link
type and packet counts from ``capinfos``), the IoT-23 conn-log paths and per-scenario
conn-log statistics (malicious PAYLOAD connections = label Malicious and
orig_bytes + resp_bytes > 0; LOSO-eligible iff >= 50 inside the pcap's time range), and the
seed / infected IP per IoT-23 scenario from its README. Every later script reads file lists
only from this inventory.

All files under data/ are read as data only. Run (from the project root)::

    .venv/bin/python memguard/memguard.py run --name p1_s0 --workers 16 -- \
        .venv/bin/python -I -u P1_CommitThenSample/scripts/00_inventory.py
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from ctsample.utils.io import DATA, RESULTS, atomic_write_json
from ctsample.utils.mem import MemoryBudgetExceeded, check_memory, workers

CAPINFOS = "/opt/homebrew/bin/capinfos"
EDITCAP = "/opt/homebrew/bin/editcap"
SURICATA = "/opt/homebrew/bin/suricata"
EDGE_ROOT = DATA / "edgeiiot" / "Edge-IIoTset dataset"
EDGE_VULN_TRUNC = 265827            # MANIFEST caveat 6: corrupt after this record

# Edge-IIoTset pcap file name -> class (CSV Attack_type spelling)
EDGE_CLASS = {
    "Backdoor_attack": "Backdoor",
    "DDoS HTTP Flood Attacks": "DDoS_HTTP",
    "DDoS ICMP Flood Attacks": "DDoS_ICMP",
    "DDoS TCP SYN Flood Attacks": "DDoS_TCP",
    "DDoS UDP Flood Attacks": "DDoS_UDP",
    "MITM (ARP spoofing + DNS) Attack": "MITM",
    "OS Fingerprinting attack": "OS_Fingerprinting",
    "Password attacks": "Password",
    "Port Scanning attack": "Port_Scanning",
    "Ransomware attack": "Ransomware",
    "SQL injection attack": "SQL_injection",
    "Uploading attack": "Uploading",
    "Vulnerability scanner attack": "Vulnerability_scanner",
    "XSS attacks": "XSS",
}
IPV4_RE = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(DATA.resolve()))


def sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return re.sub(r"_+", "_", s)


# --------------------------------------------------------------------------- capinfos
def capinfos(path: Path) -> dict:
    cmd = [CAPINFOS, "-M", "-T", "-t", "-E", "-c", "-s", "-u", "-a", "-e", "-S", "-l", str(path)]
    pr = subprocess.run(cmd, capture_output=True, text=True)
    out = {"capinfos_rc": pr.returncode}
    lines = [ln for ln in pr.stdout.splitlines() if ln.strip()]
    if len(lines) >= 2:
        hdr = lines[0].split("\t")
        val = lines[1].split("\t")
        d = dict(zip(hdr, val))

        def num(k, f=float):
            try:
                return f(d.get(k, ""))
            except ValueError:
                return None
        out.update({
            "file_type": d.get("File type"),
            "encapsulation": d.get("File encapsulation"),
            "snaplen": num("Packet size limit", int),
            "packets": num("Number of packets", int),
            "size": num("File size (bytes)", int),
            "duration_s": num("Capture duration (seconds)"),
            "first_ts": num("Start time"),
            "last_ts": num("End time"),
        })
    err = pr.stderr.strip()
    if err:
        out["capinfos_stderr"] = err[-600:]
        out["capinfos_cut_short"] = "cut short" in err
    return out


# --------------------------------------------------------------------------- conn logs
def connlog_stats(path: str, t_lo: float | None, t_hi: float | None) -> dict:
    """Stream one Zeek conn.log.labeled (tab fields; the last three space-separated)."""
    n = n_bad = n_mal = n_pay = n_mal_pay = 0
    n_in = n_mal_pay_in = 0
    first_ts = last_ts = first_mal_pay_ts = first_mal_pay_ts_in = None
    det_mal_pay = Counter()
    det_mal_pay_in = Counter()
    labels = Counter()
    with open(path, "rb") as f:
        for line in f:
            if line[:1] == b"#":
                continue
            parts = line.rstrip(b"\r\n").split(b"\t")
            if len(parts) == 21:
                tail = parts[20].split()
                if len(tail) < 2:
                    n_bad += 1
                    continue
                label = tail[1]
                dlabel = tail[2] if len(tail) > 2 else b"-"
            elif len(parts) >= 23:
                label, dlabel = parts[21].strip(), parts[22].strip()
            else:
                n_bad += 1
                continue
            try:
                ts = float(parts[0])
            except ValueError:
                n_bad += 1
                continue
            n += 1
            if first_ts is None or ts < first_ts:
                first_ts = ts
            if last_ts is None or ts > last_ts:
                last_ts = ts
            ob, rb = parts[9], parts[10]
            pay = (int(ob) if ob.isdigit() else 0) + (int(rb) if rb.isdigit() else 0)
            lab = label.decode("ascii", "replace").lower()
            labels[lab] += 1
            inside = t_lo is None or (t_lo <= ts <= t_hi)
            n_in += inside
            if pay > 0:
                n_pay += 1
            if lab == "malicious":
                n_mal += 1
                if pay > 0:
                    n_mal_pay += 1
                    dl = dlabel.decode("ascii", "replace")
                    det_mal_pay[dl] += 1
                    if first_mal_pay_ts is None or ts < first_mal_pay_ts:
                        first_mal_pay_ts = ts
                    if inside:
                        n_mal_pay_in += 1
                        det_mal_pay_in[dl] += 1
                        if first_mal_pay_ts_in is None or ts < first_mal_pay_ts_in:
                            first_mal_pay_ts_in = ts
    return {"path": path, "n_conn": n, "n_bad_lines": n_bad, "n_malicious": n_mal,
            "n_payload": n_pay, "n_mal_payload": n_mal_pay, "labels": dict(labels),
            "first_ts": first_ts, "last_ts": last_ts, "first_mal_payload_ts": first_mal_pay_ts,
            "mal_payload_by_detailed_label": dict(det_mal_pay),
            "pcap_window": [t_lo, t_hi], "n_conn_in_pcap_window": n_in,
            "n_mal_payload_in_pcap_window": n_mal_pay_in,
            "first_mal_payload_ts_in_pcap_window": first_mal_pay_ts_in,
            "mal_payload_by_detailed_label_in_pcap_window": dict(det_mal_pay_in)}


def readme_ip(readme: Path) -> tuple[str | None, str]:
    if not readme.exists():
        return None, "no README"
    txt = readme.read_text(errors="replace")
    for pat in (r"Infected (?:device|host)\s*:\s*(\S+)", r"IP address\s*:\s*(\S+)"):
        m = re.search(pat, txt, flags=re.I)
        if m:
            cand = m.group(1).strip()
            try:
                ipaddress.ip_address(cand)
                return cand, f"README ({pat.split('(')[0].strip()})"
            except ValueError:
                return None, f"README value not an IP: {cand!r}"
    return None, "README without IP"


def ip_from_name(name: str) -> str | None:
    for m in IPV4_RE.finditer(name):
        try:
            ipaddress.ip_address(m.group(1))
            return m.group(1)
        except ValueError:
            pass
    return None


# --------------------------------------------------------------------------- datasets
def list_iot23() -> tuple[list[dict], dict]:
    root = DATA / "iot23"
    sources, scen = [], {}
    for sdir in sorted((root / "pcap").iterdir()):
        if not sdir.is_dir():
            continue
        name = sdir.name
        pcaps = sorted(p for p in sdir.rglob("*.pcap") if "bro" not in p.parts)
        ip, how = readme_ip(sdir / "README.md")
        if ip is None:
            names = [ip_from_name(p.name) for p in pcaps]
            names = [x for x in names if x]
            if names:
                ip = Counter(names).most_common(1)[0][0]
                how = f"pcap file name ({how})"
        scen[name] = {"has_pcap": True, "seed_ip": ip, "seed_ip_source": how,
                      "role": "honeypot" if "Honeypot" in name else "malware",
                      "readme": rel(sdir / "README.md") if (sdir / "README.md").exists() else None,
                      "alt_conn_logs": [rel(p) for p in sorted(sdir.rglob("*conn.log.labeled"))]}
        for p in pcaps:
            sub = p.parent.relative_to(sdir)
            src = name if str(sub) == "." else f"{name}-{sanitize(str(sub))}"
            sources.append({"source": src, "scenario": name, "path": rel(p),
                            "role": scen[name]["role"], "seed_ip": ip})
    for sdir in sorted((root / "small").iterdir()):
        if not sdir.is_dir():
            continue
        logs = sorted(sdir.rglob("conn.log.labeled"))
        d = scen.setdefault(sdir.name, {"has_pcap": False, "seed_ip": None, "seed_ip_source": None,
                                        "role": "honeypot" if "Honeypot" in sdir.name else "malware"})
        d["conn_logs"] = [rel(p) for p in logs]
        # sub-folder conn logs (honeypot 7-1/Somfy-01) belong to the matching source
        d["conn_log_by_source"] = {}
        for p in logs:
            sub = p.parent.parent.relative_to(sdir)          # <scen>/<sub>/bro/conn.log.labeled
            src = sdir.name if str(sub) == "." else f"{sdir.name}-{sanitize(str(sub))}"
            d["conn_log_by_source"][src] = rel(p)
    for s in sources:
        s["conn_log"] = scen.get(s["scenario"], {}).get("conn_log_by_source", {}).get(s["source"])
    return sources, scen


def list_cic() -> list[dict]:
    b = DATA / "ciciot2023" / "pcap" / "Benign"
    out = []
    for name, role in (("BenignTraffic.pcap", "train"), ("BenignTraffic1.pcap", "test")):
        p = b / name
        if p.exists():
            out.append({"source": p.stem, "path": rel(p), "role": role, "class": "Benign"})
    return out


def list_edge() -> list[dict]:
    out = []
    for p in sorted((EDGE_ROOT / "Attack traffic").glob("*.pcap")):
        cls = EDGE_CLASS.get(p.stem)
        d = {"source": sanitize(p.stem), "path": rel(p), "role": "attack", "class": cls,
             "flood": bool(cls and cls.startswith("DDoS_")),
             "spoofed_flood": cls in ("DDoS_ICMP", "DDoS_TCP", "DDoS_UDP")}
        if cls == "Vulnerability_scanner":
            d["truncate_to_packets"] = EDGE_VULN_TRUNC
        out.append(d)
    for p in sorted((EDGE_ROOT / "Normal traffic").rglob("*.pcap")):
        out.append({"source": sanitize(p.stem), "path": rel(p), "role": "normal", "class": "Normal",
                    "flood": False, "spoofed_flood": False})
    return out


def list_gotham() -> tuple[list[dict], dict]:
    g = DATA / "gotham2025"
    out = []
    for p in sorted((g / "raw" / "benign").glob("*.pcap")):
        out.append({"source": sanitize(p.stem), "path": rel(p), "role": "benign", "scenario": "benign"})
    mal = g / "raw" / "malicious"
    if mal.exists():
        for sdir in sorted(x for x in mal.iterdir() if x.is_dir()):
            for p in sorted(sdir.glob("*.pcap")):
                out.append({"source": f"{sdir.name}__{sanitize(p.stem)}", "path": rel(p),
                            "role": "malicious", "scenario": sdir.name})
    meta = sorted(rel(p) for p in (g / "github_labeller" / "metadata").glob("*.json"))
    return out, {"metadata_json": meta}


def tool_version(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception as e:  # noqa: BLE001
        return f"? ({e})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "inventory.json"))
    ap.add_argument("--skip-connlogs", action="store_true")
    ap.add_argument("--no-gotham", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    nw = workers(default=16)

    iot_src, iot_scen = list_iot23()
    datasets = {
        "iot23": {"sources": iot_src, "scenarios": iot_scen},
        "ciciot2023_benign": {"sources": list_cic()},
        "edgeiiot": {"sources": list_edge()},
    }
    if not a.no_gotham and (DATA / "gotham2025" / "raw").exists():
        gs, gextra = list_gotham()
        datasets["gotham"] = {"sources": gs, **gextra}

    # stable per-dataset source index (unit_gid = src_idx << 32 | local unit index)
    for ds in datasets.values():
        for i, s in enumerate(sorted(ds["sources"], key=lambda x: x["source"])):
            s["src_idx"] = i
        names = [s["source"] for s in ds["sources"]]
        assert len(names) == len(set(names)), "duplicate source names"

    # explicit truncation of corrupt pcaps (never modify data/): editcap -r keeps 1..N
    for dsname, ds in datasets.items():
        for s in ds["sources"]:
            s["read_path"] = str((DATA / s["path"]).resolve())
            n = s.get("truncate_to_packets")
            if n:
                out = RESULTS / "tmp" / dsname / f"{s['source']}.pcap"
                if not out.exists():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_name(out.name + ".tmp")
                    pr = subprocess.run([EDITCAP, "-F", "pcap", "-r", str(DATA / s["path"]), str(tmp),
                                         f"1-{n}"], capture_output=True, text=True)
                    if not tmp.exists():
                        raise RuntimeError(f"editcap failed for {s['path']}: {pr.stderr}")
                    tmp.replace(out)
                s["read_path"] = str(out.resolve())
                s["truncation"] = f"editcap -F pcap -r <in> {out} 1-{n}"

    # capinfos (threads: the work is in the subprocess)
    allsrc = [s for ds in datasets.values() for s in ds["sources"]]
    print(f"[s0] capinfos on {len(allsrc)} pcaps with {min(nw, 8)} threads", flush=True)
    with ThreadPoolExecutor(max_workers=min(nw, 8)) as ex:
        for s, info in zip(allsrc, ex.map(lambda s: capinfos(Path(s["read_path"])), allsrc)):
            s.update(info)
            s["size_original"] = (DATA / s["path"]).stat().st_size
            if s.get("size") is None:
                s["size"] = Path(s["read_path"]).stat().st_size
    check_memory()

    # conn-log statistics (processes; the 23 IoT-23 logs total ~44 GB)
    if not a.skip_connlogs:
        jobs = []
        src_by_scen = {}
        for s in iot_src:
            src_by_scen.setdefault(s["scenario"], []).append(s)
        for scen, d in iot_scen.items():
            for log in d.get("conn_logs", []):
                srcs = src_by_scen.get(scen, [])
                if len(srcs) > 1:            # honeypot 7-1: restrict to the matching source
                    owner = [x for x in srcs if x.get("conn_log") == log]
                    srcs = owner or srcs
                lo = min((x["first_ts"] for x in srcs if x.get("first_ts")), default=None)
                hi = max((x["last_ts"] for x in srcs if x.get("last_ts")), default=None)
                if lo is not None:
                    lo, hi = lo - 1.0, hi + 1.0
                jobs.append((scen, log, lo, hi))
        jobs.sort(key=lambda j: -(DATA / j[1]).stat().st_size)
        print(f"[s0] conn-log stats on {len(jobs)} logs with {min(nw, 8)} processes", flush=True)
        with ProcessPoolExecutor(max_workers=min(nw, 8), mp_context=get_context("spawn")) as ex:
            futs = {ex.submit(connlog_stats, str(DATA / log), lo, hi): (scen, log) for scen, log, lo, hi in jobs}
            for fut in as_completed(futs):
                scen, log = futs[fut]
                r = fut.result()
                r["path"] = log
                iot_scen[scen].setdefault("conn_log_stats", []).append(r)
                print(f"[s0]   {scen}: {r['n_conn']} conns, {r['n_mal_payload']} mal-payload "
                      f"({r['n_mal_payload_in_pcap_window']} in pcap window)", flush=True)
                try:
                    check_memory()
                except MemoryBudgetExceeded:
                    ex.shutdown(cancel_futures=True)
                    return 75
        for scen, d in iot_scen.items():
            st = d.get("conn_log_stats", [])
            d["n_conn"] = sum(x["n_conn"] for x in st)
            d["n_malicious"] = sum(x["n_malicious"] for x in st)
            d["n_mal_payload"] = sum(x["n_mal_payload"] for x in st)
            d["n_mal_payload_in_pcap_window"] = sum(x["n_mal_payload_in_pcap_window"] for x in st)
            fm = [x["first_mal_payload_ts_in_pcap_window"] for x in st if x["first_mal_payload_ts_in_pcap_window"]]
            d["first_mal_payload_ts_in_pcap_window"] = min(fm) if fm else None
            d["loso_eligible"] = bool(d["has_pcap"] and d["role"] == "malware"
                                      and d["n_mal_payload_in_pcap_window"] >= 50)

    inv = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "data_root": str(DATA),
        "rules": {"suricata_yaml": "rules/suricata.yaml",
                  "et_open_rules_dir": "rules/et-open-suricata-8.0/rules",
                  "external_net": "any (set in rules/suricata.yaml; see data/rules/MANIFEST.md)"},
        "tools": {"suricata": tool_version([SURICATA, "-V"]), "capinfos": tool_version([CAPINFOS, "-v"])},
        "loso_rule": "malware scenario with pcap and >= 50 malicious payload connections "
                     "(label Malicious, orig_bytes+resp_bytes > 0) in the pcap time window +-1 s",
        "datasets": datasets,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if "iot23" in datasets and not a.skip_connlogs:
        inv["iot23_loso_eligible"] = sorted(k for k, v in iot_scen.items() if v.get("loso_eligible"))
    atomic_write_json(Path(a.out), inv)
    print(f"[s0] wrote {a.out} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
