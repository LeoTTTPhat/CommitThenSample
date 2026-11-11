#!/usr/bin/env python3
"""Check the local file inventory and control-file checksums."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hash-all',action='store_true',help='compute SHA-256 for every bundled file and write full_sha256.json')
    ap.add_argument('--inventory-only',action='store_true',help='skip Python import checks')
    args=ap.parse_args()
    errors=[];n=0;size=0
    with (ROOT/'files.jsonl').open() as f:
        for line in f:
            rec=json.loads(line);p=ROOT/rec['path'];n+=1;size+=rec['bytes']
            if not p.is_file():errors.append('missing: '+rec['path'])
            elif not (rec.get('relocatable') or rec.get('mutable')) and p.stat().st_size!=rec['bytes']:errors.append('size mismatch: '+rec['path'])
    sums=json.loads((ROOT/'control_sha256.json').read_text())
    for name,digest in sums.items():
        p=ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:errors.append('checksum mismatch: '+name)
    meta=json.loads((ROOT/'environment.json').read_text())
    if not args.inventory_only:
        import importlib
        module='ctsample' if meta['project']=='P1_CommitThenSample' else 'certichain'
        mod=importlib.import_module(module)
        if not Path(mod.__file__).resolve().is_relative_to(ROOT):
            errors.append('Python imports another checkout; run bootstrap.sh first')
        if module=='ctsample':
            from ctsample.utils.io import DATA,RESULTS
            if DATA.resolve()!=ROOT/'data' or RESULTS.resolve()!=ROOT/meta['project']/'results':errors.append('incorrect data/results roots')
        else:
            from certichain.io.paths import DATA_ROOT,results_root
            if DATA_ROOT.resolve()!=ROOT/'data' or results_root().resolve()!=ROOT/meta['project']/'results':errors.append('incorrect data/results roots')
        for tool in meta['external_tools']:
            if not shutil_which(tool):print('External tool not found on PATH: '+tool)
    if args.hash_all:
        all_sums={}
        with (ROOT/'files.jsonl').open() as f:
            for line in f:
                rec=json.loads(line);p=ROOT/rec['path']
                if not p.is_file():continue
                h=hashlib.sha256()
                with p.open('rb') as source:
                    for chunk in iter(lambda:source.read(8*1024*1024),b''):h.update(chunk)
                all_sums[rec['path']]=h.hexdigest()
        (ROOT/'full_sha256.json').write_text(json.dumps(all_sums,indent=2)+'\n')
    print(f'{n:,} files; {size/2**30:.2f} GiB; {len(sums):,} control checksums; {len(errors)} errors')
    for err in errors[:25]:print(err,file=sys.stderr)
    return bool(errors)
def shutil_which(name):
    import shutil
    return shutil.which(name)
if __name__=='__main__':raise SystemExit(main())
