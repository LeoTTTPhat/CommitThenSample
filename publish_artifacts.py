#!/usr/bin/env python3
"""Publish reproducible sources in Git and complete payloads as bounded release assets."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, tarfile, tempfile, threading
import urllib.parse, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECTS = {'CommitThenSample':'P1_CommitThenSample'}
DEFAULT_REPO = 'LeoTTTPhat/CommitThenSample'
DEFAULT_REMOTE = 'git@github-phat:LeoTTTPhat/CommitThenSample.git'
DEFAULT_TAG = 'committhensample-artifacts-2026-11-11'
PART_BYTES = 1800 * 1024 * 1024
IGNORE_DIRS = {'.git','.venv','.runtime','__pycache__','.pytest_cache','__MACOSX'}
BAD_SUFFIX = {'.tex','.pdf','.bib','.bst','.sty','.cls','.aux','.pyc'}


def run(cmd, *, cwd=HERE, capture=False):
    result = subprocess.run([str(v) for v in cmd], cwd=cwd, text=True, capture_output=capture, check=True)
    return result.stdout if capture else None


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    tmp.replace(path)


def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def eligible_files(root):
    for directory,dirs,files in os.walk(root):
        dirs[:]=sorted(d for d in dirs if d not in IGNORE_DIRS and d!='paper')
        for name in sorted(files):
            path=Path(directory)/name
            if name.startswith('._') or name=='.DS_Store' or path.suffix.lower() in BAD_SUFFIX:continue
            if path.is_symlink():raise RuntimeError('Unexpected symlink: '+str(path))
            yield path


def bulk_name(rel, inner):
    return rel.parts[0]=='data' or rel.parts[:2]==(inner,'results')


def source_artifact(source_root, label):
    source=source_root/PROJECTS[label]/'Artifact'
    if not (source/'files.jsonl').is_file():raise RuntimeError('Full artifact not found: '+str(source))
    return source


def prepare(source_root):
    counts={}
    for label,inner in PROJECTS.items():
        source=source_artifact(source_root,label);n=size=0
        for path in eligible_files(source):
            rel=path.relative_to(source)
            # Bulk bytes are carried by the release payload. Keep source attribution readable in Git.
            if bulk_name(rel,inner) and not (rel.parts[0]=='data' and path.suffix.lower()=='.md'):continue
            if path.stat().st_size>=100*1024*1024:raise RuntimeError('Oversized normal Git file: '+str(path))
            target=HERE/label/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()==path.read_bytes():continue
            shutil.copy2(path,target);n+=1;size+=target.stat().st_size
        counts[label]={'new_files':n,'new_bytes':size}
    save_json(HERE/'publication.json',{'repository':DEFAULT_REPO,'remote':DEFAULT_REMOTE,'release_tag':DEFAULT_TAG,
         'projects':PROJECTS,'part_bytes':PART_BYTES,'payload_format':'tar.zst, split into numbered compressed parts'})
    print(json.dumps(counts,indent=2))


class SplitOutput:
    def __init__(self, directory, prefix, limit=PART_BYTES, on_part=None):
        if not 1 <= limit < 2*1024**3:raise ValueError('Part size must be below 2 GiB')
        self.directory=directory;directory.mkdir(parents=True,exist_ok=True)
        self.prefix=prefix;self.limit=limit;self.on_part=on_part;self.parts=[];self.file=None;self.n=0
    def _open(self):
        if shutil.disk_usage(self.directory).free < self.limit + 512*1024**2:
            raise RuntimeError('Insufficient free space for the next bounded part')
        self.path=self.directory/(self.prefix+f'.tar.zst.part{len(self.parts):04d}')
        self.tmp=self.path.with_name(self.path.name+'.partial')
        self.file=self.tmp.open('wb');self.h=hashlib.sha256();self.n=0
    def write(self, block):
        while block:
            if self.file is None:self._open()
            portion=block[:self.limit-self.n];self.file.write(portion);self.h.update(portion)
            self.n+=len(portion);block=block[len(portion):]
            if self.n==self.limit:self._finish()
    def _finish(self):
        self.file.close();self.file=None
        record={'name':self.path.name,'bytes':self.n,'sha256':self.h.hexdigest()}
        if self.path.exists():
            if self.path.stat().st_size!=self.n or hash_file(self.path)!=record['sha256']:
                raise RuntimeError('An existing archive part differs: '+str(self.path))
            self.tmp.unlink()
        else:self.tmp.replace(self.path)
        self.parts.append(record)
        print(f"Ready: {record['name']} ({self.n/1024**2:.1f} MiB)",flush=True)
        if self.on_part:self.on_part(self.path,record)
    def close(self):
        if self.file is not None:self._finish()


def make_payload(source, inner, label, directory, part_bytes=PART_BYTES, on_part=None):
    files=[p for p in eligible_files(source) if bulk_name(p.relative_to(source),inner)]
    count=len(files);size=sum(p.stat().st_size for p in files)
    print(f'{label}: {count:,} payload files, {size/1024**3:.2f} GiB before compression',flush=True)
    compressor=subprocess.Popen(['zstd','-q','-3','-T4','--stdout'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    errors=[]
    def write_tar():
        try:
            def normalized(info):
                info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
                info.pax_headers={};info.mode=0o644
                return info
            with tarfile.open(fileobj=compressor.stdin,mode='w|',format=tarfile.PAX_FORMAT) as tf:
                for p in files:tf.add(p,arcname=p.relative_to(source).as_posix(),recursive=False,filter=normalized)
        except BaseException as exc:errors.append(exc)
        finally:compressor.stdin.close()
    thread=threading.Thread(target=write_tar,daemon=True);thread.start()
    split=SplitOutput(directory,label,part_bytes,on_part)
    try:
        for block in iter(lambda:compressor.stdout.read(4*1024*1024),b''):split.write(block)
        thread.join()
        if compressor.wait():raise RuntimeError('Compression failed')
        compressor.stdout.close()
        if errors:raise errors[0]
        split.close()
    except BaseException:
        compressor.kill();compressor.wait();thread.join(timeout=5);compressor.stdout.close()
        if split.file is not None:split.file.close()
        raise
    manifest={'schema':1,'project':label,'inner_project':inner,'format':'tar.zst','file_count':count,
              'uncompressed_bytes':size,'parts':split.parts}
    target=directory/(label+'-payload.json');save_json(target,manifest)
    return target


def release_info(repo,tag):
    return json.loads(run(['gh','api',f'repos/{repo}/releases/tags/{tag}'],capture=True))


def ensure_draft(repo,tag):
    query=subprocess.run(['gh','api',f'repos/{repo}/releases/tags/{tag}'],cwd=HERE,text=True,capture_output=True)
    if query.returncode==0:
        info=json.loads(query.stdout)
        if not info.get('draft'):raise RuntimeError('Refusing to modify an already-published release')
        return
    if '(HTTP 404)' not in query.stderr:raise RuntimeError('Could not inspect release: '+query.stderr.strip())
    notes=HERE/'RELEASE_NOTES.md'
    run(['gh','release','create',tag,'--repo',repo,'--draft','--target','main','--title','Complete reproduction artifacts',
         '--notes-file',notes])


def upload_part(repo,tag,path,record,delete_after=False):
    info=release_info(repo,tag)
    if not info.get('draft'):raise RuntimeError('Release is no longer a draft')
    match=next((a for a in info.get('assets',[]) if a['name']==path.name),None)
    if match is None:
        run(['gh','release','upload',tag,str(path),'--repo',repo])
        info=release_info(repo,tag)
        match=next((a for a in info.get('assets',[]) if a['name']==path.name),None)
    if not match or match.get('state')!='uploaded' or match.get('size')!=record['bytes']:
        raise RuntimeError('Uploaded asset size/state could not be verified: '+path.name)
    digest=match.get('digest')
    if digest != 'sha256:'+record['sha256']:
        raise RuntimeError('Uploaded asset digest could not be verified: '+path.name)
    print('Verified remote SHA-256: '+path.name,flush=True)
    if delete_after:path.unlink()


def publish(args):
    # Check both transports and repository write permission before changing remote state or compressing data.
    origin=run(['git','remote','get-url','origin'],capture=True).strip()
    if origin!=args.remote:raise RuntimeError('Configured origin differs from --remote')
    if not (args.remote.endswith(':'+args.repo+'.git') or args.remote.endswith('/'+args.repo+'.git')):
        raise RuntimeError('--repo and --remote identify different repositories')
    run(['git','ls-remote',args.remote])
    permission=run(['gh','api',f'repos/{args.repo}','--jq','.permissions.push'],capture=True).strip()
    if permission!='true':raise RuntimeError('The active GitHub CLI account lacks write access to '+args.repo)
    run(['git','push','-u','origin','main'])
    ensure_draft(args.repo,args.tag)
    cache=HERE/'.publication-cache';cache.mkdir(exist_ok=True)
    manifests=[]
    for label in PROJECTS:
        manifest=make_payload(source_artifact(args.source_root,label),PROJECTS[label],label,cache,
                   on_part=lambda p,r:upload_part(args.repo,args.tag,p,r,delete_after=True))
        record={'bytes':manifest.stat().st_size,'sha256':hash_file(manifest)}
        upload_part(args.repo,args.tag,manifest,record)
        manifests.append(json.loads(manifest.read_text()))
    expected=[part['name'] for m in manifests for part in m['parts']]+[m['project']+'-payload.json' for m in manifests]
    actual={a['name'] for a in release_info(args.repo,args.tag).get('assets',[])}
    if not set(expected)<=actual:raise RuntimeError('Some release assets are missing')
    run(['gh','release','edit',args.tag,'--repo',args.repo,'--draft=false'])
    info=release_info(args.repo,args.tag)
    if info.get('draft'):raise RuntimeError('Release still marked as draft')
    print('Published and verified: '+info['html_url'])


def restore_payload(manifest, fetch_part, destination, on_consumed=None):
    destination=destination.resolve();destination.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen(['zstd','-q','-d','--stdout'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    errors=[]
    def feed():
        try:
            for part in manifest['parts']:
                p=fetch_part(part['name'])
                if p.stat().st_size!=part['bytes'] or hash_file(p)!=part['sha256']:
                    raise RuntimeError('Archive part checksum mismatch: '+part['name'])
                with p.open('rb') as f:
                    for block in iter(lambda:f.read(4*1024*1024),b''):proc.stdin.write(block)
                if on_consumed:on_consumed(p)
        except BaseException as exc:errors.append(exc)
        finally:proc.stdin.close()
    thread=threading.Thread(target=feed,daemon=True);thread.start();count=0
    try:
        with tarfile.open(fileobj=proc.stdout,mode='r|') as tf:
            for member in tf:
                rel=Path(member.name)
                allowed=(rel.parts and rel.parts[0]=='data') or rel.parts[:2]==(manifest['inner_project'],'results')
                if not allowed or rel.is_absolute() or '..' in rel.parts or not member.isfile():
                    raise RuntimeError('Unsafe payload member: '+member.name)
                tf.extract(member,destination,filter='data');count+=1
        thread.join()
        if errors:raise errors[0]
        if proc.wait():raise RuntimeError('Decompression failed')
        proc.stdout.close()
    except BaseException:
        proc.kill();proc.wait();thread.join(timeout=5);proc.stdout.close();raise
    if count!=manifest['file_count']:raise RuntimeError('Extracted file count differs from the manifest')
    print(f'Restored {count:,} payload files into {destination}')


def restore(args):
    for label in args.projects or list(PROJECTS):
        target=HERE/label
        with tempfile.TemporaryDirectory(prefix='restore-',dir=HERE) as tmp:
            cache=Path(tmp)
            def fetch(name):
                if args.assets_dir:return args.assets_dir/name
                path=cache/name
                url='https://github.com/'+args.repo+'/releases/download/'+urllib.parse.quote(args.tag,safe='')+'/'+urllib.parse.quote(name,safe='')
                temporary=path.with_name(path.name+'.partial')
                with urllib.request.urlopen(url,timeout=120) as response, temporary.open('wb') as output:
                    shutil.copyfileobj(response,output,length=1024*1024)
                temporary.replace(path)
                return path
            manifest=json.loads(fetch(label+'-payload.json').read_text())
            if manifest['project']!=label or manifest['inner_project']!=PROJECTS[label]:raise RuntimeError('Unexpected payload project')
            restore_payload(manifest,fetch,target,None if args.assets_dir else lambda p:p.unlink())
        run([sys.executable,str(target/'prepare_runtime.py')])
        run([sys.executable,str(target/'verify_artifact.py'),'--inventory-only'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--source-root',type=Path,default=HERE.parent)
    p=sub.add_parser('pack');p.add_argument('project',choices=PROJECTS);p.add_argument('--source-root',type=Path,default=HERE.parent);p.add_argument('--output',type=Path,required=True)
    for action in ['publish','restore']:
        p=sub.add_parser(action);p.add_argument('--repo',default=DEFAULT_REPO);p.add_argument('--tag',default=DEFAULT_TAG)
        if action=='publish':
            p.add_argument('--remote',default=DEFAULT_REMOTE);p.add_argument('--source-root',type=Path,default=HERE.parent)
        else:p.add_argument('--projects',nargs='+',choices=PROJECTS);p.add_argument('--assets-dir',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.source_root.resolve())
    elif args.action=='pack':make_payload(source_artifact(args.source_root.resolve(),args.project),PROJECTS[args.project],args.project,args.output.resolve())
    elif args.action=='publish':publish(args)
    elif args.action=='restore':restore(args)
if __name__=='__main__':
    try:main()
    except (RuntimeError,subprocess.CalledProcessError,OSError) as exc:
        print('Publication stopped: '+str(exc),file=sys.stderr);raise SystemExit(1)
