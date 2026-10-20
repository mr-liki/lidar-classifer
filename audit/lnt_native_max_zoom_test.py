"""Native zoom reaches an individual return and retains cursor anchoring at its limit."""
from pathlib import Path
import builtins
import ctypes
import shutil
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
real_import=builtins.__import__
def no_vtk(name,*a,**kw):
    if name=='vtk' or name.startswith(('vtkmodules','pyvista')):
        raise AssertionError('Native zoom imported VTK: '+name)
    return real_import(name,*a,**kw)
builtins.__import__=no_vtk
import laspy
import numpy as np
from PySide6.QtCore import QCoreApplication,QEvent,Qt
from PySide6.QtWidgets import QApplication,QMainWindow,QWidget
from gui import lnt_native
from gui.render_backend import VulkanRenderBackend
qapp=QApplication([])
folder=ROOT/'scratch'/('native_max_zoom_'+uuid.uuid4().hex)
folder.mkdir()
app=QMainWindow()
app.resize(640,480)
widget=QWidget(app)
widget.setAttribute(Qt.WA_NativeWindow)
app.setCentralWidget(widget)
app.show()
qapp.processEvents()
native=None
backend=VulkanRenderBackend()
try:
    # UTM-sized coordinates, dense 2 cm returns, and kilometre-wide bounds.
    x,y=np.meshgrid(np.arange(-5,6)*.02,np.arange(-5,6)*.02)
    xyz=np.column_stack((680500.+x.ravel(),5000500.+y.ravel(),np.full(x.size,100.)))
    xyz=np.vstack((xyz,[[680000.,5000000.,100.],[681000.,5001000.,100.]]))
    header=laspy.LasHeader(point_format=7,version='1.4')
    header.offsets=[680000.,5000000.,0.]
    header.scales=[.0001,.0001,.0001]
    las=laspy.LasData(header)
    las.x,las.y,las.z=xyz.T
    las.classification=np.full(len(xyz),2,np.uint8)
    las.red=las.green=las.blue=np.full(len(xyz),65535,np.uint16)
    source=folder/'dense.las'
    path=folder/'dense.lnt'
    las.write(source)
    lnt_native.convert_las(str(source),str(path))
    native=lnt_native.NativeView(int(widget.winId()),widget.width(),widget.height(),str(path),64,0.)
    native.set_sse(0.)
    backend._dll=ctypes.CDLL(lnt_native.VULKAN_DLL)
    backend._bind_signatures()
    backend._handle=native.renderer_handle()
    backend._last_extent=(widget.width(),widget.height())
    backend.set_point_size(5.)
    native.set_classification_lut(np.full((256,4),255,np.uint8))
    records=native.query_points((680499.8,5000499.8,99.),(680500.2,5000500.2,101.))
    points=records['xyz']
    assert len(points)==121
    centre=points[np.argmin(np.linalg.norm(points[:,:2]-[680500.,5000500.],axis=1))]
    def pump(n=10):
        for _ in range(n):
            native.frame(.1)
            qapp.processEvents()
            time.sleep(.005)
    native.frame_region(centre[0],centre[1],.005)
    pump(30)
    projected=np.array([native.world_to_screen(*p[:2]) for p in points])
    inside=(projected[:,0]>=0)&(projected[:,0]<widget.width())&(projected[:,1]>=0)&(projected[:,1]<widget.height())
    assert np.count_nonzero(inside)==1
    image=backend.capture_frame()
    assert image is not None and np.any(image[...,:3]>150),'isolated point was not drawn'
    print('PASS 5 mm viewport isolates and renders one native point at UTM coordinates',flush=True)
    native.frame_region(centre[0],centre[1],2.)
    pump()
    px,py=123.,111.
    anchor=np.array(native.screen_to_world(px,py))
    for _ in range(120):
        assert native.zoom_at(1.,px,py)
    pump()
    assert native.stats.distance < .001,native.stats.distance
    after=np.array(native.screen_to_world(px,py))
    assert np.max(np.abs(after-anchor))<1e-7,(after,anchor)
    before_floor=np.array(native.screen_to_world(px,py))
    for _ in range(20):
        assert native.zoom_at(1.,px,py)
    assert np.max(np.abs(np.array(native.screen_to_world(px,py))-before_floor))<1e-9
    print('PASS wheel zoom reaches sub-millimetre scales and preserves the cursor anchor at its limit',flush=True)
    native.frame_region(centre[0],centre[1],1e-9)
    pump()
    assert abs(native.stats.distance-1e-4)<1e-10
    image=backend.capture_frame()
    assert image is not None and np.any(image[...,:3]>150),'point disappeared at maximum zoom'
    print('PASS maximum zoom remains finite and renders the isolated return',flush=True)
    from PIL import Image
    Image.fromarray(image).save(ROOT/'scratch'/'native_single_point_max_zoom.png')
finally:
    backend._handle=0
    if native is not None:native.close()
    app.close()
    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)
    qapp.quit()
    shutil.rmtree(folder)
    builtins.__import__=real_import
