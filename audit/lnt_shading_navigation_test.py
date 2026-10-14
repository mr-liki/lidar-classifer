"""Native navigation: pixel equivalence to an uncullled full-detail draw."""
import ctypes as C
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gui import lnt_native as nat
# The cut-based full-detail path (still used for one visible class / elevation colour).
# The persistent viewport LOD path has its own tests (lnt_surface_lod_*).
os.environ['NAKSHA_LNT_SURFACE_LOD']='0'
nat.LNT_DLL=os.environ.get('NAKSHA_TEST_LNT_DLL',str(ROOT/'native/lnt/main_shading_navigation.dll'))
nat.VULKAN_DLL=os.environ.get('NAKSHA_TEST_VULKAN_DLL',str(ROOT/'native/naksha_vulkan/build_refresh/naksha_vulkan.dll'))
path=ROOT/'audit/navigation_test.lnt'
shutil.copyfile(ROOT/'native/lnt/test_shading.lnt',path)
with path.open('rb') as f:
    h=struct.unpack('<4sHHQ6d',f.read(64));f.seek(h[3]);points,nodes=struct.unpack('<QQ',f.read(16))
with path.open('r+b') as f:f.truncate(h[3]+16+nodes*52)
dll=nat.lib();vk=C.CDLL(nat.VULKAN_DLL)
dll.lnt_view_renderer_handle.argtypes=[C.c_void_p];dll.lnt_view_renderer_handle.restype=C.c_uint64
dll.lnt_view_surface_revision.argtypes=[C.c_void_p];dll.lnt_view_surface_revision.restype=C.c_uint64
for name,result in [('nkv_get_surface_total_tile_count',C.c_uint32),('nkv_get_surface_visible_tile_count',C.c_uint32),('nkv_get_surface_visible_triangle_count',C.c_uint64),('nkv_get_surface_total_triangle_count',C.c_uint64),('nkv_get_point_position_upload_count',C.c_uint64)]:
    fn=getattr(vk,name);fn.argtypes=[C.c_uint64];fn.restype=result
vk.nkv_set_shaded_class_surface_blocks.argtypes=[C.c_uint64,C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p,C.c_uint32]
vk.nkv_set_shaded_class_surface_blocks.restype=C.c_int
error=C.create_string_buffer(512);nv=nat.NativeView.__new__(nat.NativeView)
nv.h=dll.lnt_view_open(None,640,480,os.fsencode(path),os.fsencode(nat.VULKAN_DLL),512,8.0,error,512)
assert nv.h,error.value;nv.stats=nat.LntViewStats()
try:
    palette=np.empty((256,4),dtype=np.uint8)
    for i in range(256):palette[i]=((i*37)%255,(i*71)%255,(i*113)%255,1)
    assert nv.set_classification_lut(palette);assert nv.set_shading(nat.SHADE_NORMAL)
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        assert nv.frame(.016);assert not nv.surface_error(),nv.surface_error()
        if nv.surface_method()==1 and not nv.surface_building():break
        time.sleep(.016)
    else:raise AssertionError('surface timeout')
    handle=dll.lnt_view_renderer_handle(nv.h)
    revision=dll.lnt_view_surface_revision(nv.h);counts=nv.surface_point_counts()
    assert counts[0]==counts[1]==points
    tiles=vk.nkv_get_surface_total_tile_count(handle);assert tiles>1,tiles
    uploaded=vk.nkv_get_point_position_upload_count(handle)
    side=h[7]-h[4]
    for x,y in [(.49,.51),(.2,.2),(.8,.8)]:
        nv.frame_region(h[4]+side*x,h[5]+side*y,side*.06)
        for _ in range(8):
            nv.pan(1,-1);nv.zoom(.02);assert nv.frame(.016)
            assert nv.stats.update_ms==0,'hidden point streamer ran during Shading'
            assert not nv.surface_building();assert dll.lnt_view_surface_revision(nv.h)==revision
            assert nv.surface_point_counts()==counts
        assert vk.nkv_get_point_position_upload_count(handle)==uploaded
        visible=vk.nkv_get_surface_visible_tile_count(handle)
        assert 0<visible<=tiles,(visible,tiles)
        print(f'PASS pan/zoom uses {visible}/{tiles} conservative tiles, all {points} source records retained',flush=True)
    culled=nv.capture();assert culled is not None and np.any(culled[:,:,:3])
    submitted=vk.nkv_get_surface_visible_triangle_count(handle)
    total=vk.nkv_get_surface_total_triangle_count(handle);assert submitted<total
    assert vk.nkv_set_shaded_class_surface_blocks(handle,None,0,None,None,0)
    complete=nv.capture();assert np.array_equal(culled,complete),'tile culling changed visible pixels or triangle identity'
    print(f'PASS exact pixel equivalence: {submitted}/{total} triangles submitted; no on-screen geometry lost',flush=True)
    assert nv.set_shading(nat.SHADE_CLASSIFICATION)
    for _ in range(40):assert nv.frame(.016);time.sleep(.005)
    assert nv.stats.points_rendered>0,'point streaming did not resume after leaving Shading'
    print('PASS point streaming resumes on mode exit',flush=True)
    # Whole-file style GPU saturation: deliberately repeat every triangle of a
    # synthetic mesh, so culling cannot help. Verify input callbacks do not wait.
    assert nv.set_shading(nat.SHADE_NORMAL);assert nv.frame(.016)
    vk.nkv_set_shaded_class_surface_indexed.argtypes=[C.c_uint64,C.c_void_p,C.c_uint64,C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_uint64]
    vk.nkv_set_shaded_class_surface_indexed.restype=C.c_int
    vk.nkv_get_frame_stats.argtypes=[C.c_uint64,C.POINTER(C.c_uint64),C.POINTER(C.c_uint64),C.POINTER(C.c_uint64)]
    triangles=10000000
    positions=np.array([[h[4]+side*.3,h[5]+side*.3,h[6]], [h[4]+side*.7,h[5]+side*.3,h[6]], [h[4]+side*.5,h[5]+side*.7,h[6]]],dtype=np.float64)
    faces=np.tile(np.array([0,1,2],dtype=np.int32),triangles)
    classes=np.array([1,2,3],dtype=np.uint8);lut=np.ascontiguousarray(palette[:,:3])
    assert vk.nkv_set_shaded_class_surface_indexed(handle,positions.ctypes.data,3,faces.ctypes.data,triangles,classes.ctypes.data,lut.ctypes.data,None,None,0)
    assert vk.nkv_get_surface_total_triangle_count(handle)==triangles
    rendered,skipped,recreates=C.c_uint64(),C.c_uint64(),C.c_uint64()
    assert vk.nkv_get_frame_stats(handle,C.byref(rendered),C.byref(skipped),C.byref(recreates))
    skipped_before=skipped.value
    nv.frame_region(h[4]+side*.5,h[5]+side*.5,side*2)
    durations=[]
    for _ in range(80):
        nv.pan(.2,0);nv.zoom(-.001)
        start=time.perf_counter();assert nv.frame(.008);durations.append((time.perf_counter()-start)*1000)
    assert vk.nkv_get_frame_stats(handle,C.byref(rendered),C.byref(skipped),C.byref(recreates))
    assert skipped.value>skipped_before,'GPU saturation did not exercise nonblocking deferral'
    assert np.percentile(durations,95)<25,sorted(durations)[-8:]
    assert max(durations)<100, max(durations)
    assert vk.nkv_get_surface_total_triangle_count(handle)==triangles
    print(f'PASS full-view GPU saturation: {triangles} triangles retained, input p95 {np.percentile(durations,95):.2f} ms, max {max(durations):.2f} ms, {skipped.value-skipped_before} obsolete frame attempts deferred',flush=True)
    submitted_before=rendered.value;deadline=time.monotonic()+2
    while time.monotonic()<deadline:
        assert nv.frame(.008)
        assert vk.nkv_get_frame_stats(handle,C.byref(rendered),C.byref(skipped),C.byref(recreates))
        if rendered.value>submitted_before:break
        time.sleep(.002)
    else:raise AssertionError('latest camera was not retried after GPU deferral')
    print('PASS pending camera presents after GPU becomes available',flush=True)
finally:
    nv.close();path.unlink(missing_ok=True)

