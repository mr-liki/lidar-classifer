"""Real Vulkan coverage and foreground-preservation test for the TIN underlay."""
import ctypes as C
import os
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication, QWidget
from gui.render_backend import VulkanRenderBackend

class Tile(C.Structure):
    _fields_ = [('key', C.c_uint64), ('positions', C.c_void_p), ('cls', C.c_void_p),
                ('faces', C.c_void_p), ('coverage', C.c_void_p), ('coverage_cls', C.c_void_p),
                ('vertices', C.c_uint32), ('triangles', C.c_uint32), ('points', C.c_uint32),
                ('result', C.c_int32)]
class Draw(C.Structure):
    _fields_ = [('key', C.c_uint64), ('first', C.c_uint32), ('count', C.c_uint32),
                ('coverage', C.c_uint32), ('flags', C.c_uint32)]

qt = QApplication([])
w = QWidget(); w.resize(500, 500); w.show(); qt.processEvents()
b = VulkanRenderBackend()
assert b.initialize(int(w.winId()), 500, 500)
d = b._dll; h = C.c_uint64(b._handle)
d.nkv_surface_tiles_configure.argtypes = [C.c_uint64] * 4
d.nkv_surface_tiles_upload.argtypes = [C.c_uint64, C.POINTER(Tile), C.c_uint32]
d.nkv_surface_tiles_set_draw_list.argtypes = [C.c_uint64, C.POINTER(Draw), C.c_uint32]
d.nkv_surface_tiles_set_palette.argtypes = [C.c_uint64, C.c_void_p, C.c_float]
# Root deliberately lies ABOVE the detail. An ordinary coarse overlay would
# incorrectly hide the detail; the underlay must leave it byte-identical.
root = np.array([[-4,-4,100],[4,-4,100],[4,4,100],[-4,4,100]], np.float64)
rf = np.array([[0,1,2],[0,2,3]], np.int32)
rc = np.full(4, 2, np.uint8)
pts = np.array([[-4,-4,0],[-1,-4,0],[-1,4,0],[-4,4,0],
                [1,-4,0],[4,-4,0],[4,4,0],[1,4,0]], np.float64)
f = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]], np.int32)
c = np.full(8, 5, np.uint8)
palette = np.full((256,3), 100, np.uint8); palette[2] = [0,100,255]; palette[5] = [255,50,0]
ptr = lambda a: C.c_void_p(a.ctypes.data)
assert d.nkv_surface_tiles_configure(h, 1024, 1024, 64)
assert d.nkv_surface_tiles_set_palette(h, ptr(palette), C.c_float(.5))
tiles = (Tile * 2)(Tile(1,ptr(root),ptr(rc),ptr(rf),None,None,4,2,0,0),
                   Tile(2,ptr(pts),ptr(c),ptr(f),None,None,8,4,0,0))
assert d.nkv_surface_tiles_upload(h, tiles, 2) == 2
assert all(t.result == 1 for t in tiles)
b.set_camera_view_up(0.,1.,0.)
b.set_camera_ortho((0.,0.,0.), (0.,0.,-1.), 5., .001, 10000.)
b.set_point_cloud_visible(False)

def frame(items):
    array = (Draw * len(items))(*items)
    assert d.nkv_surface_tiles_set_draw_list(h, array, len(items))
    for _ in range(3):
        b.request_render(); qt.processEvents()
    image = b.capture_frame()
    assert image is not None
    return image
try:
    detail = frame([Draw(2,0,4,0,0)])
    filled = frame([Draw(2,0,4,0,0), Draw(1,0,2,0,1)])
    # Entire 400x400 interior (away from raster edge rounding) is covered.
    assert np.all(filled[60:440,60:440,:3].max(axis=2) > 20), 'background contains uncovered pixels'
    occupied = detail[:,:,:3].max(axis=2) > 20
    assert occupied.sum() > 10000
    np.testing.assert_array_equal(filled[occupied], detail[occupied])
    assert filled[250,250,:3].max() > 20 and detail[250,250,:3].max() <= 20
    from PIL import Image
    Image.fromarray(filled).save(Path(__file__).with_name('native_surface_underlay.png'))
    print('PASS: no interior voids; underlay above terrain preserves every detailed pixel')
finally:
    b.shutdown(); w.close()
