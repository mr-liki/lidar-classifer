"""Offline tests with an emulated LASzip binary; not a codec test."""
import csv
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from convert_las14_to_laz12 import read_header, verify_output


def fake_header(path, minor, pointformat, pointcount, vlrs=()):
    n = 375 if minor == 4 else 227
    b = bytearray(n)
    b[:4] = b'LASF'
    b[24] = 1
    b[25] = minor
    struct.pack_into('<H',b,94,n)
    struct.pack_into('<I',b,96,n+54*len(vlrs))
    struct.pack_into('<I',b,100,len(vlrs))
    b[104] = pointformat
    struct.pack_into('<I',b,107,pointcount if minor == 2 else 0)
    if minor == 4:
        struct.pack_into('<Q',b,247,pointcount)
    struct.pack_into('<3d',b,131,0.01,0.01,0.01)
    struct.pack_into('<3d',b,155,100000.0,200000.0,10.0)
    for user, rid in vlrs:
        b.extend(struct.pack('<H16sHH32s',0,user.encode('ascii'),rid,0,b''))
    b.extend(b'payload' * 8)
    path.write_bytes(b)


class BatchTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.base=Path(self.temp.name)
        self.input=self.base/'LAS'
        self.output=self.base/'LAZ_1_2'
        self.input.mkdir()
        self.tool=self.base/'fake_laszip.py'
        self.tool.write_text('''#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, %(path)r)
from test_header_and_batch import fake_header
args=sys.argv
source=Path(args[args.index('-i')+1])
out=Path(args[args.index('-o')+1])
assert '-compatible' in args
f=source.read_bytes()
assert f[24:26] == bytes((1,4))
n=int.from_bytes(f[247:255],'little')
p=f[104]&63
fake_header(out,2,{6:1,7:3,8:3}[p],n,[('laszip encoded',22204),('lascompatible',22204)])
''' % {'path':str(Path(__file__).resolve().parent)})
        os.chmod(self.tool,0o755)
        self.script=Path(__file__).resolve().parent/'convert_las14_to_laz12.py'

    def tearDown(self):
        self.temp.cleanup()

    def run_converter(self, *extra):
        p=subprocess.run([sys.executable,str(self.script),'--input',str(self.input),'--output',str(self.output),'--laszip',str(self.tool),*extra],capture_output=True,text=True)
        return p

    def test_good_format7(self):
        source=self.input/'source.las'
        fake_header(source,4,7,123)
        p=self.run_converter()
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        out=self.output/'source.laz'
        self.assertTrue(out.exists())
        h=verify_output(read_header(source),out)
        self.assertEqual((h.version,h.point_format,h.points),('1.2',3,123))
        p2=self.run_converter()
        self.assertEqual(p2.returncode,0)
        self.assertIn('SKIP_EXISTING',p2.stdout)

    def test_unsupported(self):
        fake_header(self.input/'wave.las',4,9,20)
        p=self.run_converter()
        self.assertEqual(p.returncode,1)
        self.assertIn('UNSUPPORTED_POINT_FORMAT',p.stdout)
        self.assertFalse((self.output/'wave.laz').exists())

    def test_header_only(self):
        fake_header(self.input/'s.las',4,7,20)
        p=self.run_converter('--scan-only')
        self.assertEqual(p.returncode,0)
        self.assertIn('ELIGIBLE',p.stdout)
        self.assertFalse(self.output.exists())

    def test_existing_unverified_never_overwritten(self):
        fake_header(self.input/'s.las',4,7,20)
        self.output.mkdir()
        out=self.output/'s.laz'
        out.write_bytes(b'wrong')
        p=self.run_converter()
        self.assertEqual(p.returncode,1)
        self.assertIn('EXISTING_UNVERIFIED',p.stdout)
        self.assertEqual(out.read_bytes(),b'wrong')

if __name__ == '__main__':
    unittest.main()
