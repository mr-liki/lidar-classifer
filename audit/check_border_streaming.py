"""Measure navigation while point tiles stream: python ... DLL OUTPUT.json."""
import ctypes,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NAKSHA_VULKAN_DLL']=str(Path(sys.argv[1]).resolve())
from PySide6.QtWidgets import QApplication,QWidget
from gui.render_backend import VulkanRenderBackend
app=QApplication([]);w=QWidget();w.resize(800,600);w.show();app.processEvents()
b=VulkanRenderBackend();assert b.initialize(int(w.winId()),800,600)
x,y=np.meshgrid(np.linspace(0,300,1800),np.linspace(0,225,1350))
x=x.ravel();y=y.ravel();z=.03*np.sin(x*17)+.04*np.cos(y*23)
roof=(x>60)&(x<150)&(y>45)&(y<160);z[roof]+=5
xyz=np.column_stack((x,y,z));cls=np.where(roof,6,2).astype(np.uint8)
assert b.reserve_point_capacity(len(xyz))
assert b.set_display_mode(1);assert b.set_point_size(2.5)
assert b.set_camera_ortho((150,112,3),(0,0,-1),125,1,12000)
fn=b._dll.nkv_set_point_border;fn.argtypes=[ctypes.c_uint64,ctypes.c_float,ctypes.c_int];fn.restype=ctypes.c_int
assert fn(b._handle,60,1)
frames=[]
try:
    for start in range(0,len(xyz),135000):
        end=min(start+135000,len(xyz))
        assert b.upload_point_tile(start,xyz[start:end],classification=cls[start:end])
        assert b.set_point_draw_ranges(np.array([0],np.uint32),np.array([end],np.uint32))
        assert b.set_camera_ortho((150+start/len(xyz),112,3),(0,0,-1),125+start/len(xyz),1,12000)
        t=time.perf_counter();b.request_render();app.processEvents();frames.append((time.perf_counter()-t)*1000)
    stats_fn=getattr(b._dll,'nkv_get_point_border_stats',None)
    if stats_fn is not None:
        stats_fn.argtypes=[ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint64)];stats_fn.restype=ctypes.c_int
        def stats():
            a=(ctypes.c_uint64*3)();assert stats_fn(b._handle,a);return tuple(a)
        deadline=time.perf_counter()+20
        while stats()[2]:
            assert time.perf_counter()<deadline
            b.request_render();app.processEvents();time.sleep(.02)
        builds=stats()[0]
        for i in range(50):
            b.set_camera_ortho((150+i*.1,112,3),(0,0,-1),125+i*.1,1,12000)
            b.request_render();app.processEvents()
        assert stats()[0]==builds,'Camera movement rebuilt geometry flags'
    report={'points':len(xyz),'streamed_frames':len(frames),'frame_ms':frames,'median_ms':float(np.median(frames)),'max_ms':max(frames),'p95_ms':float(np.percentile(frames,95)),'warm_navigation_rebuilds':0 if stats_fn is not None else None}
    Path(sys.argv[2]).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
finally:
    b.shutdown();w.close();app.processEvents()
