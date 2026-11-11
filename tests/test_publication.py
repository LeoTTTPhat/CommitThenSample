import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('publication',Path(__file__).resolve().parents[1]/'publish_artifacts.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)


class PublicationTests(unittest.TestCase):
    def test_roundtrip_split_payload_and_repeatability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';assets=root/'assets';dest=root/'restored'
            (source/'data/example').mkdir(parents=True)
            (source/'P1_CommitThenSample/results').mkdir(parents=True)
            (source/'P1_CommitThenSample/src').mkdir(parents=True)
            payload=os.urandom(2*1024*1024+4096)
            (source/'data/example/input.bin').write_bytes(payload)
            (source/'P1_CommitThenSample/results/checkpoint.json').write_text('{"score":42}')
            (source/'P1_CommitThenSample/src/algorithm.py').write_text('value = 42\n')
            (source/'data/example/unused.tex').write_text('excluded')
            path=P.make_payload(source,'P1_CommitThenSample','CommitThenSample',assets,1024*1024)
            manifest=json.loads(path.read_text())
            self.assertEqual(manifest['file_count'],2)
            self.assertGreater(len(manifest['parts']),1)
            self.assertTrue(all(0 < p['bytes'] <= 1024*1024 for p in manifest['parts']))
            repeated=P.make_payload(source,'P1_CommitThenSample','CommitThenSample',assets,1024*1024)
            self.assertEqual(manifest,json.loads(repeated.read_text()))
            P.restore_payload(manifest,lambda name:assets/name,dest)
            self.assertEqual((dest/'data/example/input.bin').read_bytes(),payload)
            self.assertEqual((dest/'P1_CommitThenSample/results/checkpoint.json').read_text(),'{"score":42}')
            self.assertFalse((dest/'P1_CommitThenSample/src').exists())
            self.assertFalse((dest/'data/example/unused.tex').exists())

    def test_rejects_unsafe_archive_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tar=root/'unsafe.tar';packed=root/'unsafe.zst'
            with tarfile.open(tar,'w') as tf:
                info=tarfile.TarInfo('../escape');info.size=4;tf.addfile(info,io.BytesIO(b'nope'))
            subprocess.run(['zstd','-q','-f',str(tar),'-o',str(packed)],check=True)
            manifest={'inner_project':'P1_CommitThenSample','file_count':1,'parts':[{
                'name':packed.name,'bytes':packed.stat().st_size,'sha256':P.hash_file(packed)}]}
            with self.assertRaisesRegex(RuntimeError,'Unsafe payload member'):
                P.restore_payload(manifest,lambda name:root/name,root/'restored')
            self.assertFalse((root/'escape').exists())

    def test_rejects_changed_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);part=root/'bad.part';part.write_bytes(b'changed data')
            manifest={'inner_project':'P1_CommitThenSample','file_count':1,'parts':[{
                'name':part.name,'bytes':part.stat().st_size,'sha256':hashlib.sha256(b'original').hexdigest()}]}
            with self.assertRaises(Exception):
                P.restore_payload(manifest,lambda name:root/name,root/'restored')

if __name__=='__main__':unittest.main()
