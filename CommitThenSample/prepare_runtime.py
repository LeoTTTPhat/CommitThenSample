#!/usr/bin/env python3
"""Relocate file pointers and configure native executables for this copy."""
from __future__ import annotations
import json, os, re, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
META=json.loads((ROOT/'environment.json').read_text())
OLD=META['source_root']

def main():
    state=ROOT/'.runtime/location.json'
    previous=json.loads(state.read_text()).get('root') if state.exists() else None
    roots=[str(ROOT),OLD+'/'+META['project']+'/Artifact',OLD]
    if previous:roots.append(previous)
    path_pattern=re.compile('|'.join(re.escape(s) for s in sorted(set(roots),key=len,reverse=True)))
    # The inventory is runtime metadata. Frozen registries, model bytes, and signed records are kept intact.
    inv=ROOT/'P1_CommitThenSample/results/inventory.json'
    if inv.exists():
        data=json.loads(inv.read_text());data['data_root']=str(ROOT/'data')
        for dataset in data.get('datasets',{}).values():
            for src in dataset.get('sources',[]):
                old_path=src.get('read_path','')
                marker='/P1_CommitThenSample/results/'
                if marker in old_path:
                    src['read_path']=str(ROOT/'P1_CommitThenSample/results'/old_path.rsplit(marker,1)[1])
                else:
                    src['read_path']=str(ROOT/'data'/src['path'])
        inv.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    # These patches are limited to native-tool lookup and do not change the algorithms.
    native=[(ROOT/'P1_CommitThenSample/src/ctsample/content/suricata.py','SURICATA','suricata'),
            (ROOT/'P2_CertiChain/src/certichain/gotham/report_payload.py','TSHARK','tshark')]
    inventory_script=ROOT/'P1_CommitThenSample/scripts/00_inventory.py'
    native.extend((inventory_script,var,tool) for var,tool in [('CAPINFOS','capinfos'),('EDITCAP','editcap'),('SURICATA','suricata')])
    for file,var,tool in native:
        if file.exists():
            text=file.read_text()
            old=f'{var} = "/opt/homebrew/bin/{tool}"'
            # Preserve the original file bytes whenever the recorded executable location exists.
            if old in text and not Path('/opt/homebrew/bin',tool).is_file():
                text=text.replace(old,f'{var} = os.environ.get("{var}_BIN", __import__("shutil").which("{tool}") or "{tool}")')
                file.write_text(text)
    config=ROOT/'data/rules/suricata.yaml'
    if config.exists():
        text=path_pattern.sub(str(ROOT),config.read_text())
        (ROOT/'data/rules/logs').mkdir(exist_ok=True)
        config.write_text(text)
    for shell in (ROOT/META['project']/'scripts').glob('*.sh'):
        text=shell.read_text().replace('cd '+OLD,'ROOT="$(cd "$(dirname "$0")/../.." && pwd)"\ncd "$ROOT"')
        shell.write_text(text)
    helper=ROOT/'tools/build_suricata_yaml.py'
    if helper.exists():
        text=helper.read_text().replace('RDIR = "'+OLD+'/data/rules"',
             'RDIR = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "rules")')
        helper.write_text(text)
    state.parent.mkdir(parents=True,exist_ok=True)
    state.write_text(json.dumps({'root':str(ROOT)})+'\n')
    print('Runtime paths prepared for '+str(ROOT))
if __name__=='__main__':main()
