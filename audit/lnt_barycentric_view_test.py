"""Focused native viewer acceptance test, independent of Qt/VTK teardown.
Run: venv/Scripts/python.exe audit/lnt_barycentric_view_test.py
"""
import ctypes as C
import os
from pathlib import Path
import sys
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gui import lnt_native as nat
import numpy as np

def main():
    dll = nat.lib()
    dll.lnt_view_surface_triangles.argtypes = [C.c_void_p]
    dll.lnt_view_surface_triangles.restype = C.c_uint64
    dll.lnt_view_surface_building.argtypes = [C.c_void_p]
    dll.lnt_view_surface_building.restype = C.c_int
    dll.lnt_view_surface_revision.argtypes = [C.c_void_p]
    dll.lnt_view_surface_revision.restype = C.c_uint64
    error = C.create_string_buffer(512)
    nv = nat.NativeView.__new__(nat.NativeView)
    nv.h = dll.lnt_view_open(None, 640, 480,
        os.fsencode(ROOT / 'native/lnt/test_shading.lnt'), os.fsencode(nat.VULKAN_DLL),
        512, 8.0, error, 512)
    assert nv.h, error.value
    nv.stats = nat.LntViewStats()
    try:
        palette = np.zeros((256,4),dtype=np.uint8)
        palette[:,:3] = (200,100,50)
        palette[:,3] = 1
        # Point rendering is deliberately coarse and budget limited. Shading
        # must independently read every visible leaf, without either limit.
        nv.set_sse(1000000)
        nv.set_point_budget(1000)
        assert nv.set_classification_lut(palette)
        assert nv.set_shading(nat.SHADE_NORMAL)
        assert nv.frame(0.016)
        assert dll.lnt_view_surface_building(nv.h), 'initial Shading build did not start on the first frame'
        print('PASS full-detail build starts on the first Shading frame',flush=True)
        def wait(method):
            end = time.monotonic()+15
            while time.monotonic()<end:
                assert nv.frame(0.016)
                if nv.surface_method()==method and dll.lnt_view_surface_triangles(nv.h)>0:
                    print(f'PASS method {method}, {dll.lnt_view_surface_triangles(nv.h)} triangles',flush=True)
                    return
                time.sleep(0.016)
            raise AssertionError(f'timeout waiting for method {method}')
        wait(1)
        source, visible, coverage = nv.surface_point_counts()
        dataset_points = nv.info()[0]
        assert source == visible == dataset_points, (source, visible, dataset_points)
        print(f'PASS full-detail shading uses all {visible} records despite coarse LOD and point budget; coverage {coverage}',flush=True)
        assert nv.set_shading(nat.SHADE_CLASSIFICATION)
        assert nv.set_shading(nat.SHADE_NORMAL)
        assert nv.frame(0.016)
        assert nv.surface_method()==1 and nv.surface_point_counts()==(source,visible,coverage)
        print('PASS returning to Shading reuses the complete resident surface immediately',flush=True)
        revision=dll.lnt_view_surface_revision(nv.h)
        assert nv.set_classification_lut(palette) # identical Apply must not invalidate the mesh
        nv.set_point_budget(200000)
        nv.set_sse(0)
        nv.zoom(.05) # a camera change small enough to retain the same full-detail leaves
        for _ in range(45):
            assert nv.frame(.016)
            assert dll.lnt_view_surface_revision(nv.h)==revision
            assert not nv.surface_building()
            time.sleep(.016)
        assert nv.surface_point_counts()==(source,visible,coverage)
        print('PASS identical palette, point LOD and camera changes reuse an unchanged full-detail cut',flush=True)
        with (ROOT/'native/lnt/test_shading.lnt').open('rb') as f:
            header=struct.unpack('<4sHHQ6d',f.read(64))
        side=header[7]-header[4]
        nv.frame_region(header[4]+side*.2,header[5]+side*.2,side*.05)
        def wait_source(predicate):
            end=time.monotonic()+15
            while time.monotonic()<end:
                assert nv.frame(.016)
                counts=nv.surface_point_counts()
                if not nv.surface_building() and predicate(counts): return counts
                time.sleep(.016)
            raise AssertionError(f'timeout waiting for full-detail cut: {nv.surface_point_counts()}')
        smaller=wait_source(lambda counts:0<counts[0]<dataset_points and counts[0]==counts[1])
        assert dll.lnt_view_surface_revision(nv.h)>revision
        print(f'PASS a changed cut rebuilds and accounts for all {smaller[0]} source records',flush=True)
        nv.reset()
        wait_source(lambda counts:counts[0]==counts[1]==dataset_points)
        nv.set_point_budget(1000)
        nv.set_sse(1000000)
        palette[:,3]=0
        assert nv.set_classification_lut(palette)
        for _ in range(80): assert nv.frame(0.016); time.sleep(0.016)
        assert nv.surface_point_counts()==(dataset_points,0,0)
        print('PASS hiding every class leaves no visible surface records',flush=True)
        palette[:,3]=1
        assert nv.set_classification_lut(palette)
        wait(1)
        palette[:,3]=0;palette[2,3]=1
        assert nv.set_classification_lut(palette)
        # Old mesh/method stay visible immediately after invalidation.
        assert nv.surface_method()==1 and dll.lnt_view_surface_triangles(nv.h)>0
        wait(0)
        source, visible, coverage = nv.surface_point_counts()
        assert source == dataset_points and 0 < visible < source
        print(f'PASS full source read; {visible} visible class records retained',flush=True)
        palette[5,3]=1
        assert nv.set_classification_lut(palette)
        assert nv.surface_method()==0 and dll.lnt_view_surface_triangles(nv.h)>0
        wait(1)
        assert nv.set_surface_elevation(True,nat.RAMP_HUE,(0,100))
        # Same method still needs a completed rebuild; pump a full second.
        for _ in range(65): assert nv.frame(0.016); time.sleep(0.016)
        wait(1)
        palette[:,3]=0;palette[2,3]=1
        assert nv.set_classification_lut(palette)
        wait(0)
        # Rapid invalidations exercise stale-job rejection.
        for _ in range(5):
            palette[5,3]=1; assert nv.set_classification_lut(palette); assert nv.frame(0.016)
            palette[5,3]=0; assert nv.set_classification_lut(palette); assert nv.frame(0.016)
        for _ in range(65): assert nv.frame(0.016); time.sleep(0.016)
        wait(0)
        print('PASS visibility, Surface rule, retained mesh, rapid palette invalidation',flush=True)
    finally:
        nv.close()

if __name__=='__main__':
    main()
