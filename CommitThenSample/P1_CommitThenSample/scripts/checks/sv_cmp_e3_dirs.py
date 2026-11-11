"""Compare two E3 results roots job by job (analytic var/ppi parquet; mc cov/width/ctrl parts), ignoring 'group'.

    .venv/bin/python -I P1_CommitThenSample/scripts/checks/sv_cmp_e3_dirs.py <root_a> <root_b> [stream_substring]   (P1-X-SV, rev1)
"""
import json
import sys
from pathlib import Path

import pandas as pd


def jobs(root: Path, sub: str | None):
    out = {}
    for d in sorted(root.glob("*/*")):
        if not (d / "DONE").exists() and not list(d.glob("rep*_cov.parquet")):
            continue
        c = d / "config.json"
        if not c.exists():
            continue
        j = dict(json.loads(c.read_text())["job"])
        if sub and sub not in j["stream"]:
            continue
        j.pop("group", None)
        k = json.dumps(j, sort_keys=True)
        prev = out.get(k)
        if prev is None or json.loads((d / "config.json").read_text())["created"] > json.loads((prev / "config.json").read_text())["created"]:
            out[k] = d
    return out


def frames(d: Path):
    fs = {}
    for name in ("var", "ppi"):
        p = d / f"{name}.parquet"
        if p.exists():
            fs[name] = pd.read_parquet(p)
    for kind in ("cov", "width", "ctrl"):
        ps = sorted(d.glob(f"rep*_{kind}.parquet"))
        if ps:
            fs[kind] = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True)
    return fs


a, b = Path(sys.argv[1]), Path(sys.argv[2])
sub = sys.argv[3] if len(sys.argv) > 3 else None
ja, jb = jobs(a, sub), jobs(b, sub)
common = sorted(set(ja) & set(jb))
bad = []
for k in common:
    fa, fb = frames(ja[k]), frames(jb[k])
    if set(fa) != set(fb):
        bad.append((k, "file kinds differ"))
        continue
    for n in fa:
        x, y = fa[n], fb[n]
        if list(x.columns) != list(y.columns):
            common_cols = [c for c in x.columns if c in y.columns]
            x, y = x[common_cols], y[common_cols]
            note = " (common columns only)"
        else:
            note = ""
        if not x.reset_index(drop=True).equals(y.reset_index(drop=True)):
            bad.append((k, f"{n} differs{note}"))
print(f"jobs: a {len(ja)}, b {len(jb)}, common {len(common)}, differing {len(bad)}")
for k, m in bad[:20]:
    print("  ", m, k)
