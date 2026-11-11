"""Suricata runner and eve.json alert parser for the per-unit replay (DESIGN §5.4).

Command (DESIGN; EXTERNAL_NET=any is set in data/rules/suricata.yaml, see its MANIFEST)::

    suricata -c data/rules/suricata.yaml -r <replay.pcap> -k none --runmode autofp -l <out>
             -S results/rules/content.rules --set stream.midstream=true --set vlan.use-for-tracking=true

plus resource-only overrides: ``threading.detect-thread-ratio`` so that each instance runs 4
worker threads, and larger flow/stream memcaps so that no flow is dropped in emergency mode
(these do not change rule semantics). ``TZ=UTC`` makes eve timestamps UTC.

Families S1–S6 are defined on the rule classtype (ET classification.config):
S1 attempted-admin; S2 trojan-activity; S3 attempted-recon, network-scan; S4
web-application-attack; S5 policy-violation; S6 misc-activity, misc-attack.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from ctsample.utils.io import DATA

SURICATA = "/opt/homebrew/bin/suricata"
YAML = str(DATA / "rules" / "suricata.yaml")

FAMILY_OF_CLASSTYPE = {
    "attempted-admin": "S1",
    "trojan-activity": "S2",
    "attempted-recon": "S3",
    "network-scan": "S3",
    "web-application-attack": "S4",
    "policy-violation": "S5",
    "misc-activity": "S6",
    "misc-attack": "S6",
}
FAMILIES = ("S1", "S2", "S3", "S4", "S5", "S6")


def make_alert_only_yaml(out: str | Path) -> Path:
    """Derived copy of data/rules/suricata.yaml whose eve-log logs only alerts (no flow/app-layer/anomaly
    records, no payload/packet dumps). Used for the original-pcap runs (observability gap), where
    millions of scan flows would otherwise produce >10 GB eve files. Detection settings are unchanged."""
    import re as _re
    txt = Path(YAML).read_text()
    i = txt.index("  - eve-log:")
    j = txt.index("\n      types:\n", i)
    m = _re.compile(r"\n  - [a-z]|\n  # output module").search(txt, j + 1)
    block = ("\n      types:\n        - alert:\n            payload: no\n            packet: no\n"
             "            metadata: no\n            tagged-packets: no\n")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# DERIVED by ctsample.content.suricata.make_alert_only_yaml from " + YAML + "\n"
                   + txt[:j] + block + txt[m.start():])
    return out


def suricata_cmd(pcap: str | Path, out_dir: str | Path, rules: str | Path, threads: int = 4,
                 extra_set: list[str] | None = None, yaml_path: str | None = None) -> list[str]:
    ncpu = os.cpu_count() or 32
    ratio = max(threads, 1) / ncpu
    cmd = [SURICATA, "-c", yaml_path or YAML, "-r", str(pcap), "-k", "none", "--runmode", "autofp", "-l", str(out_dir),
           "-S", str(rules), "--set", "stream.midstream=true", "--set", "vlan.use-for-tracking=true",
           "--set", f"threading.detect-thread-ratio={ratio:.6f}",
           "--set", "flow.memcap=2gb", "--set", "stream.memcap=2gb", "--set", "stream.reassembly.memcap=4gb"]
    for s in extra_set or []:
        cmd += ["--set", s]
    return cmd


def run_suricata(pcap, out_dir, rules, threads: int = 4, extra_set=None, timeout: float | None = None,
                 yaml_path: str | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in ("eve.json", "fast.log", "stats.log", "suricata.log"):
        (out_dir / f).unlink(missing_ok=True)
    cmd = suricata_cmd(pcap, out_dir, rules, threads, extra_set, yaml_path)
    env = dict(os.environ, TZ="UTC")
    t0 = time.time()
    pr = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    el = time.time() - t0
    (out_dir / "stdout.txt").write_text(pr.stdout[-20000:] + "\n--- stderr ---\n" + pr.stderr[-20000:])
    return {"rc": pr.returncode, "elapsed_s": el, "cmd": " ".join(cmd)}


ALERT_FIELDS = ("pcap_cnt", "timestamp", "vlan_outer", "vlan_inner", "src_ip", "src_port", "dest_ip",
                "dest_port", "proto", "app_proto", "flow_id", "sid", "gid", "rev", "signature", "category",
                "severity", "stream", "tx_id")


def _int(x) -> int:
    return -1 if x is None else int(x)


def parse_eve_alerts(eve: str | Path) -> tuple[dict, dict]:
    """Stream eve.json; return (columns of alert records, counts of other event types)."""
    cols = {k: [] for k in ALERT_FIELDS}
    other: dict = {}
    with open(eve, "rb") as f:
        for line in f:
            if b'"event_type":"alert"' not in line:
                i = line.find(b'"event_type":"')
                if i >= 0:
                    j = line.find(b'"', i + 14)
                    et = line[i + 14:j].decode("ascii", "replace")
                    other[et] = other.get(et, 0) + 1
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                other["bad_json"] = other.get("bad_json", 0) + 1
                continue
            a = r.get("alert", {})
            v = r.get("vlan") or []
            cols["pcap_cnt"].append(_int(r.get("pcap_cnt")))
            cols["timestamp"].append(r.get("timestamp"))
            cols["vlan_outer"].append(v[0] if len(v) > 0 else -1)
            cols["vlan_inner"].append(v[1] if len(v) > 1 else -1)
            cols["src_ip"].append(r.get("src_ip"))
            cols["src_port"].append(_int(r.get("src_port")))
            cols["dest_ip"].append(r.get("dest_ip"))
            cols["dest_port"].append(_int(r.get("dest_port")))
            cols["proto"].append(r.get("proto"))
            cols["app_proto"].append(r.get("app_proto"))
            cols["flow_id"].append(_int(r.get("flow_id")))
            cols["sid"].append(_int(a.get("signature_id")))
            cols["gid"].append(_int(a.get("gid")))
            cols["rev"].append(_int(a.get("rev")))
            cols["signature"].append(a.get("signature"))
            cols["category"].append(a.get("category"))
            cols["severity"].append(_int(a.get("severity")))
            cols["stream"].append(_int(r.get("stream")))
            cols["tx_id"].append(_int(r.get("tx_id")))
    return cols, other


def flow_vlans(eve: str | Path) -> list[tuple[int, int]]:
    """(outer, inner) VLAN ids of every eve flow record (used by the replay test)."""
    out = []
    with open(eve, "rb") as f:
        for line in f:
            if b'"event_type":"flow"' in line:
                r = json.loads(line)
                v = r.get("vlan") or []
                out.append((v[0] if v else -1, v[1] if len(v) > 1 else -1))
    return out


def stats_counters(stats_log: str | Path, names=("flow.emerg_mode_entered", "tcp.ssn_memcap_drop",
                                                 "tcp.segment_memcap_drop", "flow.memcap", "decoder.pkts",
                                                 "decoder.vlan", "decoder.vlan_qinq", "flow.total",
                                                 "detect.alert")) -> dict:
    """Last value of selected counters from stats.log."""
    out = {}
    p = Path(stats_log)
    if not p.exists():
        return out
    for line in p.read_text(errors="replace").splitlines():
        parts = [x.strip() for x in line.split("|")]
        if len(parts) == 3 and parts[0] in names:
            try:
                out[parts[0]] = int(parts[2])
            except ValueError:
                pass
    return out
