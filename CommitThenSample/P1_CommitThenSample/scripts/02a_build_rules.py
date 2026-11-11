"""S2a: build the content-rules file for per-unit replay (DESIGN §5.4).

Keeps an enabled ET Open rule (from the ``rule-files`` of data/rules/suricata.yaml) iff it
has a ``content`` or ``pcre`` option and none of ``threshold``, ``detection_filter``,
``xbits``, ``hostbits``, ``iprep``, ``dataset``, ``datarep`` (state shared across flows or
hosts). ``flowbits`` are kept. Options are parsed (quotes/escapes respected), so keywords
that only occur inside content strings or metadata do not count.

Outputs: results/rules/content.rules, results/rules/content_sids.json (kept sids with
classtype, msg and family S1–S6; exclusion counts) and the ``suricata -T`` check log.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

from ctsample.content.suricata import FAMILY_OF_CLASSTYPE, SURICATA, YAML
from ctsample.utils.io import RESULTS, atomic_write_json, atomic_write_text

EXCLUDE = ("threshold", "detection_filter", "xbits", "hostbits", "iprep", "dataset", "datarep")
RULE_RE = re.compile(r"^\s*(alert|drop|reject|pass|rejectsrc|rejectdst|rejectboth)\s+\S+\s+.*?\((.*)\)\s*$")


def split_options(body: str) -> list[tuple[str, str]]:
    """Split a rule's option body on ';' outside double quotes (backslash escapes honoured)."""
    out, cur, inq, esc = [], [], False, False
    for ch in body:
        if esc:
            cur.append(ch)
            esc = False
            continue
        if ch == "\\":
            cur.append(ch)
            esc = True
            continue
        if ch == '"':
            inq = not inq
            cur.append(ch)
            continue
        if ch == ";" and not inq:
            opt = "".join(cur).strip()
            if opt:
                k, _, v = opt.partition(":")
                out.append((k.strip().lower(), v.strip()))
            cur = []
            continue
        cur.append(ch)
    opt = "".join(cur).strip()
    if opt:
        k, _, v = opt.partition(":")
        out.append((k.strip().lower(), v.strip()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(RESULTS / "rules"))
    ap.add_argument("--no-test", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    cfg = yaml.safe_load(Path(YAML).read_text())
    rdir = Path(cfg["default-rule-path"])
    files = cfg["rule-files"]
    kept, sids = [], {}
    excl = Counter()
    n_enabled = 0
    for fn in files:
        p = rdir / fn
        if not p.exists():
            excl["missing_file"] += 1
            continue
        for line in p.read_text(errors="replace").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            m = RULE_RE.match(s)
            if not m:
                excl["unparsed_line"] += 1
                continue
            n_enabled += 1
            opts = split_options(m.group(2))
            keys = {k for k, _ in opts}
            if not ({"content", "pcre"} & keys):
                excl["no_content_or_pcre"] += 1
                continue
            bad = [k for k in EXCLUDE if k in keys]
            if bad:
                for k in bad:
                    excl[f"has_{k}"] += 1
                excl["excluded_cross_flow_state"] += 1
                continue
            od = {}
            for k, v in opts:
                od.setdefault(k, v)
            try:
                sid = int(od.get("sid", "").strip())
            except ValueError:
                excl["no_sid"] += 1
                continue
            ct = od.get("classtype", "").strip()
            msg = od.get("msg", "").strip().strip('"')
            if sid in sids:
                excl["duplicate_sid"] += 1
                continue
            sids[sid] = {"classtype": ct, "family": FAMILY_OF_CLASSTYPE.get(ct), "msg": msg, "file": fn,
                         "flowbits": "flowbits" in keys}
            kept.append(s)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_dir / "content.rules", "\n".join(kept) + "\n")
    fam = Counter(v["family"] or "none" for v in sids.values())
    info = {"created": time.strftime("%Y-%m-%d %H:%M:%S %z"), "yaml": YAML, "rule_dir": str(rdir),
            "n_rule_files": len(files), "n_enabled": n_enabled, "n_kept": len(kept),
            "exclusions": dict(excl), "keep_rule": "content|pcre and none of " + ",".join(EXCLUDE),
            "family_counts": dict(fam), "classtype_counts": dict(Counter(v["classtype"] for v in sids.values())),
            "sids": {str(k): v for k, v in sorted(sids.items())}}
    if not a.no_test:
        tdir = out_dir / "suricata_T"
        tdir.mkdir(exist_ok=True)
        pr = subprocess.run([SURICATA, "-T", "-c", YAML, "-S", str(out_dir / "content.rules"),
                             "-l", str(tdir)], capture_output=True, text=True)
        logtxt = pr.stdout + pr.stderr
        slog = (tdir / "suricata.log").read_text(errors="replace") if (tdir / "suricata.log").exists() else ""
        atomic_write_text(out_dir / "suricata_T.log", logtxt + "\n--- suricata.log ---\n" + slog)
        m = re.search(r"(\d+) rules successfully loaded, (\d+) rules failed", logtxt + slog)
        info["suricata_T"] = {"rc": pr.returncode, "loaded": int(m.group(1)) if m else None,
                              "failed": int(m.group(2)) if m else None,
                              "flowbits_checked_not_set": len(re.findall(r"is checked but not set", logtxt + slog))}
        for f in tdir.iterdir():
            f.unlink()
        tdir.rmdir()
    atomic_write_json(out_dir / "content_sids.json", info)
    print(f"[s2a] enabled {n_enabled}, kept {len(kept)}; exclusions {dict(excl)}; families {dict(fam)}; "
          f"suricata -T {info.get('suricata_T')}", flush=True)
    return 0 if info.get("suricata_T", {}).get("rc", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
