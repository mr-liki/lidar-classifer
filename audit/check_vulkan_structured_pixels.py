"""Real GPU regression: python audit/check_vulkan_structured_pixels.py DLL OUTPUT_DIR.
Uses a generated noisy roof/tree scene; requires a visible Windows Vulkan device.
"""
from pathlib import Path
import ctypes
import json
import os
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['NAKSHA_VULKAN_DLL']=str(Path(sys.argv[1]).resolve())
os.environ['NAKSHA_VULKAN_VALIDATION']='1'
import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication,QWidget
from gui.render_backend import VulkanRenderBackend
out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
app=QApplication([]);window=QWidget();window.resize(800,600);window.show();app.processEvents()
b=VulkanRenderBackend()
assert b.initialize(int(window.winId()),800,600)
rng=np.random.default_rng(71)
x,y=np.meshgrid(np.linspace(0,40,601),np.linspace(0,30,451))
x=x.ravel();y=y.ravel();z=rng.normal(0,.035,len(x));cls=np.ones(len(x),np.uint8)
roof=(x>5)&(x<20)&(y>6)&(y<22)
z[roof]=6+rng.normal(0,.12,int(roof.sum()));cls[roof]=5
step=roof&(x>13);z[step]+=.7
r2=((x-29)/5)**2+((y-18)/7)**2
tree=r2<1;z[tree]=3+5*np.sqrt(1-r2[tree])+rng.normal(0,.13,int(tree.sum()));cls[tree]=6
xyz=np.column_stack([x,y,z])
colors=np.zeros((256,3),np.uint8);colors[1]=(0,255,0);colors[5]=(0,255,255);colors[6]=(125,0,255)
# RGB is also supplied so missing optional streams cannot affect this check.
rgb=np.ascontiguousarray(colors[cls]);cls=np.ascontiguousarray(cls)
assert b.set_point_cloud(xyz,rgb=rgb,classification=cls)
assert b.set_display_mode(1);assert b.set_point_luts(class_rgb=colors)
assert b.set_point_size(2.5);assert b.set_clear_color(.18,.18,.18)
assert b.set_camera_ortho((20,15,4),(0,0,-1),17,near_clip=1,far_clip=1600)
fn=b._dll.nkv_set_point_border;fn.argtypes=[ctypes.c_uint64,ctypes.c_float,ctypes.c_int];fn.restype=ctypes.c_int
stats_fn=getattr(b._dll,'nkv_get_point_border_stats',None)
if stats_fn is not None:
    stats_fn.argtypes=[ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint64)]
    stats_fn.restype=ctypes.c_int

def stats():
    values=(ctypes.c_uint64*3)()
    assert stats_fn(b._handle,values)
    return tuple(values)

def settle():
    if stats_fn is None: return
    deadline=time.perf_counter()+15
    while stats()[2]:
        assert time.perf_counter()<deadline,'Background borders did not settle'
        b.request_render();app.processEvents();time.sleep(.01)


def capture(percent,mode,name):
    assert fn(b._handle,percent,mode)
    b.request_render();app.processEvents()
    if mode==1 and percent>0: settle()
    frame=b.capture_frame();assert frame is not None,b.last_error()
    Image.fromarray(frame).save(out/(name+'.png'))
    return frame
try:
    baseline=capture(0,1,'zero')
    uploads=b.get_point_position_upload_count()
    perpoint=capture(45,0,'perpoint45')
    started=time.perf_counter();structured=capture(45,1,'structured45');cold=time.perf_counter()-started
    hybrid=capture(45,2,'hybrid45')
    if '--baseline' in sys.argv:
        print('Captured previous renderer for comparison');sys.exit(0)
    assert np.array_equal(capture(0,1,'zero_after'),baseline)
    assert b.get_point_position_upload_count()==uploads
    assert not np.array_equal(structured,baseline),'Structured border vanished'
    assert not np.array_equal(structured,perpoint)
    assert not np.array_equal(structured,hybrid)
    # The nearest roof core can never be shifted sideways or overwritten by
    # a recessed outline from the same or a lower return.
    cyan=np.all(baseline[...,:3]==(0,255,255),axis=2)
    assert np.all(structured[...,:3][cyan]==(0,255,255))
    black=lambda a: np.all(a[...,:3]<8,axis=2)
    assert black(structured).sum()>20
    # Width must be monotone: the 45% outline must not recede behind a roof
    # when the user asks for 60/100%. Check visible edge pixels, not just totals.
    high=[]
    for percent in (60,80,100):
        wide=capture(percent,1,f'structured{percent}')
        lost=int(np.count_nonzero(black(structured)&~black(wide)))
        assert lost < max(20,int(black(structured).sum()*.02)),(percent,lost)
        high.append((percent,int(black(wide).sum()),lost))
    assert black(perpoint).sum()>black(structured).sum()*2
    # All modes use projection-correct depth, including Per-Point.
    assert b.set_camera_ortho((20,15,4),(0,0,-1),17,near_clip=1,far_clip=100)
    narrow_points=capture(45,0,'perpoint45_narrow')
    assert np.mean(np.any(narrow_points!=perpoint,axis=2))<.002
    assert b.set_camera_ortho((20,15,4),(0,0,-1),17,near_clip=1,far_clip=1600)

    # A cold streamed arena with reverse tile order must match a full upload.
    assert b.reserve_point_capacity(len(xyz))
    assert b.upload_point_tile(0,xyz,classification=cls)
    assert b.set_point_draw_ranges(np.array([0],np.uint32),np.array([len(xyz)],np.uint32))
    arena_reference=capture(45,1,'arena_single_tile45')
    split=len(xyz)//2
    assert b.upload_point_tile(split,xyz[split:],classification=cls[split:])
    assert b.upload_point_tile(0,xyz[:split],classification=cls[:split])
    assert b.set_point_draw_ranges(np.array([0],np.uint32),np.array([len(xyz)],np.uint32))
    tiled=capture(45,1,'tiled45')
    assert np.array_equal(tiled,arena_reference),'Tile seams changed rendered border'
    # A class-only patch must invalidate the flags without uploading positions.
    start=len(xyz)//3;patch=np.full(3000,6,np.uint8)
    before=b.get_point_position_upload_count()
    # Force an in-flight snapshot, then edit one of its retained tiles. A
    # stale worker result must not replace flags for the edited class stream.
    assert b.upload_point_tile(0,xyz[:split],classification=cls[:split])
    b.request_render()
    assert b.upload_point_tile(start,None,classification=patch)
    edited=capture(45,1,'class_edit45')
    assert b.get_point_position_upload_count()==before+1
    assert not np.array_equal(edited,tiled)
    expected_cls=cls.copy();expected_cls[start:start+len(patch)]=patch
    assert b.upload_point_tile(0,xyz,classification=expected_cls)
    assert np.array_equal(capture(45,1,'class_edit_full_reference'),edited), 'A stale background result overwrote class edits' 
    assert b.set_camera_lookat((20,-30,50),(20,15,3),45,.1,2000)
    perspective=capture(45,1,'perspective45')
    assert np.count_nonzero(np.any(perspective[...,:3]!=(118,118,118),axis=2))>1000
    # A narrow camera range must give the same orthographic outline pixels.
    assert b.set_camera_ortho((20,15,4),(0,0,-1),17,near_clip=1,far_clip=100)
    narrow=capture(45,1,'narrow_clip45')
    assert np.mean(np.any(narrow!=edited,axis=2))<.002
    if stats_fn is not None:
        before_builds=stats()[0]
        for i in range(30):
            assert b.set_camera_ortho((20+i*.01,15,4),(0,0,-1),17+i*.01,near_clip=1,far_clip=1600)
            b.request_render();app.processEvents()
        assert stats()[0]==before_builds,'Navigation rebuilt structural flags'
    report={'width_sweep':high,'points':len(xyz),'structured_black_pixels':int(black(structured).sum()),'perpoint_black_pixels':int(black(perpoint).sum()),'hybrid_black_pixels':int(black(hybrid).sum()),'cold_capture_seconds':cold,'position_uploads_unchanged_by_controls':True,'tile_pixels_identical':True,'class_edit_invalidates_flags':True,'clip_range_pixel_difference':float(np.mean(np.any(narrow!=edited,axis=2)))}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
finally:
    b.shutdown();window.close();app.processEvents()
