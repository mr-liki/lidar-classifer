"""Small native-only cache acceptance test; never changes the user's data."""
import ctypes as C
import os
from pathlib import Path
import shutil
import struct
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gui import lnt_native as nat
import numpy as np
# The cut-based path's whole-file mesh cache; the tiled archive has its own tests.
os.environ['NAKSHA_LNT_SURFACE_LOD']='0'
nat.LNT_DLL=os.environ.get('NAKSHA_TEST_LNT_DLL',str(ROOT/'native/lnt/main_shading_embedded.dll'))
if os.environ.get('NAKSHA_TEST_VULKAN_DLL'):nat.VULKAN_DLL=os.environ['NAKSHA_TEST_VULKAN_DLL']
path=ROOT/'audit/embedded_reopen_test.lnt'
shutil.copyfile(ROOT/'native/lnt/test_shading.lnt',path)
with path.open('rb') as f:
    h=struct.unpack('<4sHHQ6d',f.read(64));f.seek(h[3]);points,nodes=struct.unpack('<QQ',f.read(16))
base=h[3]+16+nodes*52
with path.open('r+b') as f:f.truncate(base)
dll=nat.lib()
dll.lnt_view_surface_revision.argtypes=[C.c_void_p]
dll.lnt_view_surface_revision.restype=C.c_uint64
palette=np.zeros((256,4),dtype=np.uint8);palette[:,:3]=(200,100,50);palette[:,3]=1
try:
    previous=None
    for attempt in range(3):
        if attempt==2:
            # Same settings but changed source bytes must reject the cached mesh.
            with path.open('r+b') as f:
                f.seek(64+6);value=f.read(1);f.seek(64+6);f.write(bytes([value[0]^1]))
        err=C.create_string_buffer(512)
        nv=nat.NativeView.__new__(nat.NativeView)
        nv.h=dll.lnt_view_open(None,640,480,os.fsencode(path),os.fsencode(nat.VULKAN_DLL),512,8.0,err,512)
        assert nv.h,err.value
        nv.stats=nat.LntViewStats()
        try:
            assert nv.set_classification_lut(palette)
            assert nv.set_shading(nat.SHADE_NORMAL)
            deadline=time.monotonic()+20
            while time.monotonic()<deadline:
                assert nv.frame(.016)
                assert not nv.surface_error(),nv.surface_error()
                if nv.surface_method()==1 and not nv.surface_building():break
                time.sleep(.016)
            else:raise AssertionError('surface timeout')
            counts=nv.surface_point_counts();assert counts[0]==counts[1]==points,counts
            if previous is not None:assert counts==previous
            previous=counts
            revision=dll.lnt_view_surface_revision(nv.h)
            side=h[7]-h[4]
            nv.frame_region(h[4]+side*.2,h[5]+side*.2,side*.05)
            for _ in range(35):
                assert nv.frame(.016)
                assert not nv.surface_building(),'camera change retriangulated a complete file mesh'
                assert dll.lnt_view_surface_revision(nv.h)==revision
                assert nv.surface_point_counts()==counts
                time.sleep(.016)
            print('PASS zoom keeps the complete resident file surface without rebuilding',flush=True)
        finally:nv.close() # joins the background save before reopen
        assert path.stat().st_size>base
        print(f'PASS open {attempt+1}: all {points} source records accounted for',flush=True)
finally:
    path.unlink(missing_ok=True)
