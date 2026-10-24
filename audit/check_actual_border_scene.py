"""Main LNT smoke test: python ... DLL CLOUD.lnt OUTPUT_DIR."""
import ctypes,json,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
dll=Path(sys.argv[1]).resolve();os.environ['NAKSHA_VULKAN_DLL']=str(dll)
from PySide6.QtWidgets import QApplication,QWidget
from gui import lnt_native
from gui.render_backend import VulkanRenderBackend
lnt_native.VULKAN_DLL=str(dll)
app=QApplication([]);window=QWidget();window.resize(1000,800);window.show();app.processEvents()
v=lnt_native.NativeView(int(window.winId()),1000,800,str(Path(sys.argv[2]).resolve()),512,8.)
b=VulkanRenderBackend();b._dll=ctypes.CDLL(str(dll));b._bind_signatures();b._handle=v.renderer_handle();b._last_extent=(1000,800)
fn=b._dll.nkv_set_point_border;fn.argtypes=[ctypes.c_uint64,ctypes.c_float,ctypes.c_int];fn.restype=ctypes.c_int
sf=b._dll.nkv_get_point_border_stats;sf.argtypes=[ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint64)];sf.restype=ctypes.c_int
out=Path(sys.argv[3]);out.mkdir(exist_ok=True,parents=True)
def stats():
    a=(ctypes.c_uint64*3)();assert sf(b._handle,a);return tuple(a)
def settle():
    deadline=time.perf_counter()+30;stable=0;last=0
    while stable<5:
        assert time.perf_counter()<deadline,('Scene did not settle',v.stats.points_rendered,stats())
        assert v.frame(.016)
        app.processEvents();b.request_render()
        current=int(v.stats.points_rendered)
        stable=stable+1 if current==last and v.stats.pending_loads==0 and stats()[2]==0 and current>0 else 0
        last=current;time.sleep(.02)
def capture(percent,mode,name):
    assert fn(b._handle,percent,mode)
    b.request_render();app.processEvents();frame=b.capture_frame();assert frame is not None
    Image.fromarray(frame).save(out/(name+'.png'));return frame
try:
    v.set_mode(1)
    rgba=np.full((256,4),255,np.uint8)
    for c,rgb in {0:(255,255,255),1:(180,180,180),2:(180,105,0),3:(0,255,0),4:(0,210,0),5:(0,255,0),6:(255,0,128),7:(255,0,255),9:(0,210,255)}.items():rgba[c,:3]=rgb
    assert v.set_classification_lut(rgba)
    assert b.set_point_size(2.5);assert fn(b._handle,45,1)
    settle()
    total,nodes,ext=v.info();before=stats()[0]
    base=capture(45,1,'structured45');black=lambda f:np.all(f[...,:3]<8,axis=2)
    # Restrict comparisons to new ring pixels, excluding black viewport margins.
    zero=capture(0,1,'zero');mask=black(base)&~black(zero)
    sweep=[]
    for percent in (60,80,100):
        f=capture(percent,1,f'structured{percent}')
        lost=int(np.count_nonzero(mask&~black(f)))
        assert lost<max(100,int(mask.sum()*.03)),(percent,lost,int(mask.sum()))
        sweep.append([percent,int(np.count_nonzero(black(f)&~black(zero))),lost])
    points=capture(45,0,'perpoint45')
    assert not np.array_equal(points,base)
    assert fn(b._handle,60,1)
    times=[]
    for i in range(60):
        v.pan(2 if i<30 else -2,0)
        if i%10==0:v.zoom(.25 if i<30 else -.25)
        t=time.perf_counter();assert v.frame(.016);app.processEvents();times.append((time.perf_counter()-t)*1000)
    settle()
    Image.fromarray(capture(60,1,'after_navigation')).save(out/'after_navigation.png')
    report={'source_points':total,'nodes':nodes,'drawn_points':int(v.stats.points_rendered),'width_sweep':sweep,'ring_pixels_at_45':int(mask.sum()),'navigation_frames':len(times),'navigation_median_ms':float(np.median(times)),'navigation_p95_ms':float(np.percentile(times,95)),'navigation_max_ms':max(times),'boundary_jobs_before_navigation':before,'boundary_jobs_after_navigation':stats()[0]}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
finally:
    b._handle=0;v.close();window.close();app.processEvents()
