#!/usr/bin/env python3
"""Run the bundled workflows from any working directory."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT/'environment.json').read_text())
PROJECT = ROOT / META['project']

def environment():
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
        env.setdefault(key,'1')
    env.setdefault('POLARS_MAX_THREADS','4')
    env['MPLCONFIGDIR'] = str(ROOT/'.runtime/matplotlib')
    if META['project'] == 'P2_CertiChain':
        env['HF_HUB_CACHE'] = str(ROOT/'data/huggingface/hub')
        env.setdefault('HF_HUB_OFFLINE','1')
        env.setdefault('TRANSFORMERS_OFFLINE','1')
    return env

def call(args, guarded=False):
    cmd = [sys.executable,*map(str,args)]
    if guarded:
        cmd = [sys.executable,str(ROOT/'memguard/memguard.py'),'run','--name','artifact_reproduce',
               '--limit-gb',str(META['memory_limit_gb']),'--workers',os.environ.get('MEMGUARD_WORKERS','2'),'--',*cmd]
    print('+ '+ ' '.join(cmd), flush=True)
    subprocess.run(cmd,cwd=ROOT,env=environment(),check=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['check','smoke','summaries','run'])
    ap.add_argument('args',nargs=argparse.REMAINDER)
    a=ap.parse_args()
    if a.action=='check':
        call([ROOT/'verify_artifact.py',*a.args]);return
    if a.action=='smoke':
        if META['project']=='P1_CommitThenSample':
            tests=['test_ecvrf_vectors.py','test_protocol_roundtrip.py','test_committed_tau.py','test_ht_unbiased.py','test_merkle.py']
        else:
            tests=['test_crc.py','test_cascade.py','test_chainforge.py','test_leakage_features.py','test_cards.py']
        extra=[] if META['project']=='P1_CommitThenSample' else ['-k','not test_run_groups_tiny_model_shards_and_lookup']
        call(['-m','pytest','-q','-p','no:cacheprovider',*[PROJECT/'tests'/t for t in tests],*extra,*a.args]);return
    if a.action=='summaries':
        if META['project']=='P1_CommitThenSample':
            call([PROJECT/'scripts/90_aggregate.py','--exp','all','--rev','rev1'])
        else:
            call([PROJECT/'scripts/12_run_experiment.py','--exp','E1,E7','--numbers-only','--workers','1'])
        return
    args=a.args[1:] if a.args and a.args[0]=='--' else a.args
    if not args:
        ap.error('run needs a bundled script and its arguments')
    script=(ROOT/args[0]).resolve()
    if not script.is_relative_to(ROOT) or not script.is_file():
        ap.error('the script must be an existing file inside this artifact')
    call([script,*args[1:]],guarded=True)
if __name__=='__main__':
    main()
