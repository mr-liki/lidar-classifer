"""Exercise the actual streamed LNT root/refinement on a deliberately hollow scan."""
import ctypes as C
import os
from pathlib import Path
import sys
import tempfile
import time
import laspy
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gui import lnt_native as nat
from audit.lnt_surface_lod_view_test import View
root = Path(__file__).resolve().parents[1]
nat.LNT_DLL = os.environ.get('NAKSHA_TEST_LNT_DLL',nat.LNT_DLL)
nat.VULKAN_DLL = str(root / 'native/naksha_vulkan/build_msvc/naksha_vulkan.dll')
with tempfile.TemporaryDirectory(prefix='lnt_void_') as folder:
    path = Path(folder) / 'hollow.las'
    x, y = np.meshgrid(np.linspace(0,100,701), np.linspace(0,100,701))
    keep = ~((x > 25) & (x < 75) & (y > 25) & (y < 75))
    h = laspy.LasHeader(point_format=3, version='1.2')
    h.scales = [.001]*3; h.offsets = [500000,4000000,0]
    data = laspy.LasData(h)
    data.x = 500000+x[keep]; data.y = 4000000+y[keep]; data.z = 20+.02*x[keep]+.01*y[keep]
    data.classification = np.full(keep.sum(),2,np.uint8)
    data.write(path)
    print('fixture',nat.convert_las(str(path),str(path.with_suffix('.lnt'))),flush=True)
    v = View(nat,path.with_suffix('.lnt'))
    print('view opened',flush=True)
    try:
        v.nv.plan_view()
        print('plan view set',flush=True)
        assert v.nv.set_shading(nat.SHADE_NORMAL)
        print('settling',flush=True)
        s = v.settle()
        assert s['drawn_tiles'] > 0
        holes, lit, frame = v.holes()
        assert holes == 0, (holes,s)
        assert lit > 10000
        assert v.dll.lnt_view_set_surface_lod_px(v.nv.h,.25)
        print('settling',flush=True)
        s = v.settle()
        holes, lit, frame = v.holes()
        assert s['drawn_tiles'] > 1, s
        assert holes == 0, (holes,s)
        assert s['source_exact'] == s['source_covered'], s
        assert v.nv.set_surface_elevation(True, z_range=(20.,23.))
        surface_stats = v.settle()
        surface_holes, _, surface_frame = v.holes()
        assert surface_holes == 0, (surface_holes, surface_stats)
        assert not np.array_equal(surface_frame,frame), 'Surface did not switch to elevation colours'
        from PIL import Image
        Image.fromarray(frame).save(root/'audit/native_lnt_void_regression.png')
        print(f"PASS streamed LNT: {s['drawn_tiles']} exact tiles, {s['triangles_drawn']} triangles, zero enclosed void pixels",flush=True)
    finally:
        v.close()
