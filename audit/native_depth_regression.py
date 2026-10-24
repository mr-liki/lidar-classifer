"""Real native LNT Depth pixels, independent of source RGB and class colour."""
import os
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import laspy
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gui import lnt_native as nat
from audit.lnt_surface_lod_view_test import View
nat.LNT_DLL = os.environ.get('NAKSHA_TEST_LNT_DLL', nat.LNT_DLL)

with tempfile.TemporaryDirectory(dir=ROOT/'audit') as scratch:
    path = Path(scratch)/'depth.las'
    h = laspy.LasHeader(point_format=3, version='1.2')
    h.offsets = [500000., 4000000., 0.]
    las = laspy.LasData(h)
    x, y = np.meshgrid(np.linspace(-40,40,161), np.linspace(-30,30,121))
    las.x = x.ravel()+h.offsets[0]
    las.y = y.ravel()+h.offsets[1]
    las.z = (x*.4+y*.2).ravel()
    las.red = np.full(x.size,65535,np.uint16)
    las.classification = np.full(x.size,6,np.uint8)
    las.write(path)
    lnt = path.with_suffix('.lnt')
    nat.convert_las(str(path),str(lnt))
    view = View(nat,str(lnt))
    try:
        # An intensity LUT ending in red must not tint Depth.
        red = np.zeros((256,3),np.uint8)
        red[:,0] = 255
        assert view.nv.set_luts(None,None,red)
        for mode in ('top','zoom','orbit','back_to_depth'):
            if mode == 'zoom': view.nv.zoom(2)
            if mode == 'orbit':
                view.nv.set_2d(False)
                view.nv.orbit(15,10)
            if mode == 'back_to_depth': view.nv.set_mode(nat.MODE_CLASS)
            view.nv.set_mode(nat.MODE_DEPTH)
            for _ in range(80):
                assert view.nv.frame(.016)
                time.sleep(.005)
            img = view.nv.capture()[:,:,:3]
            lit = img[np.any(img>0,axis=2)].astype(int)
            print(mode,'Depth mean',lit.mean(axis=0),'levels',len(np.unique(lit[:,0])),flush=True)
            assert len(lit)>1000
            assert np.max(np.ptp(lit,axis=1))<=2, 'Depth is coloured instead of grayscale'
            assert np.ptp(lit[:,0])>70, 'Depth range is saturated or flat'
    finally:
        view.close()
print('PASS native Depth is grayscale with a useful range')
