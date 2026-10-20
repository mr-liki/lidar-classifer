"""Real native multi-LNT rendering, queries, edits and independent cache reuse."""
import ctypes as C
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
import laspy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gui import lnt_native as nat
from gui.lnt_batch import build_batch
from gui.lnt_collection import read_collection,payload_bytes
from gui.lnt_class_counts import scan_class_histogram
from audit.lnt_surface_lod_view_test import View
ROOT=Path(__file__).resolve().parents[1]
nat.LNT_DLL=os.environ.get('NAKSHA_TEST_LNT_DLL',str(ROOT/'native/lnt/main_display_modes.dll'))
nat.VULKAN_DLL=str(ROOT/'native/naksha_vulkan/build_msvc/naksha_vulkan.dll')

with tempfile.TemporaryDirectory(prefix='individual_lnt_',dir=ROOT/'audit') as scratch:
    folder=Path(scratch)
    paths=[]
    for file,(count,shift,code) in enumerate([(90000,0.,2),(130000,53.123,5)]):
        rng=np.random.default_rng(file)
        h=laspy.LasHeader(point_format=3,version='1.2')
        h.offsets=[500000.,4000000.,0.];h.scales=[.001]*3
        las=laspy.LasData(h)
        x=rng.uniform(0,70,count)+shift;y=rng.uniform(0,60,count)+file*11.234
        las.x=500000.+x;las.y=4000000.+y;las.z=20.+.02*x+.01*y
        las.classification=np.full(count,code,np.uint8)
        las.point_source_id=np.full(count,file+1,np.uint16)
        path=folder/f'tile_{file}.laz';las.write(path);paths.append(path)
    _,index=build_batch(paths)
    g=read_collection(index)
    assert all(p.with_suffix('.lnt').is_file() for p in paths)
    assert not list(folder.rglob('combined.*'))
    assert Path(index).stat().st_size<20000
    assert scan_class_histogram(index).sum()==220000
    print('PASS per-source LNT caches and metadata-only collection',flush=True)
    v=View(nat,index)
    try:
        v.nv.plan_view()
        result=v.nv.query_points([499999.,3999999.,-1000.],[500200.,4000200.,1000.])
        assert result is not None,v.nv.last_error
        assert len(result['xyz'])==220000,len(result['xyz'])
        np.testing.assert_array_equal(np.bincount(result['classification'],minlength=256)[[2,5]],[90000,130000])
        owners=g['routes']['file'][result['node']]
        np.testing.assert_array_equal(np.bincount(owners),[90000,130000])
        print('PASS exact point queries retain every source record and owning file',flush=True)
        from gui.native_section_display import source_line_ids
        picks=np.random.default_rng(41).choice(len(owners),200,replace=False)
        expected_ids=owners[picks].astype(np.uint16)+1
        actual_ids=source_line_ids(index,result['xyz'][picks],.002)
        np.testing.assert_array_equal(actual_ids,expected_ids)
        print('PASS flight-line attributes resolve from individual sources without a merged LAS',flush=True)

        assert v.nv.set_shading(nat.SHADE_NORMAL)
        overview=v.settle()
        assert v.dll.lnt_view_set_surface_lod_px(v.nv.h,.25)
        detailed=v.settle()
        assert detailed['source_exact']==220000,detailed
        assert detailed['represented_drawn']==220000,detailed
        holes,lit,image=v.holes()
        assert holes==0 and lit>10000,(holes,lit,detailed)
        print('PASS overlapping independently partitioned LNTs: exact surface coverage, zero void pixels',flush=True)
        rows=np.r_[np.flatnonzero(owners==0)[:10],np.flatnonzero(owners==1)[:10]]
        old=result['classification'][rows].copy()
        # A read-only member must reject the entire cross-file edit before
        # another source is modified.
        import ctypes
        blocked=str(paths[1].with_suffix('.lnt'))
        attrs=ctypes.windll.kernel32.GetFileAttributesW(blocked)
        assert ctypes.windll.kernel32.SetFileAttributesW(blocked,attrs|1)
        try:
            assert not v.nv.set_classes(result['node'][rows],result['local'][rows],np.full(len(rows),9,np.uint8))
            assert scan_class_histogram(index)[9]==0
        finally:
            ctypes.windll.kernel32.SetFileAttributesW(blocked,attrs)
        print('PASS read-only source rejects cross-file edits without partial changes',flush=True)

        assert v.nv.set_classes(result['node'][rows],result['local'][rows],np.full(len(rows),9,np.uint8)),v.nv.last_error
        for path in paths:
            assert scan_class_histogram(path.with_suffix('.lnt'))[9]==10
        assert scan_class_histogram(index)[9]==20
        offsets=g['entries']['byte_offset'][result['node'][rows]].astype(np.int64)+result['local'][rows]*11+10
        np.testing.assert_array_equal(payload_bytes(index,offsets),np.full(20,9,np.uint8))
        updated=v.settle()
        assert v.nv.set_classes(result['node'][rows],result['local'][rows],old),v.nv.last_error
        assert scan_class_histogram(index)[9]==0
        print('PASS classification and undo write to the correct individual LNTs',flush=True)
    finally:v.close()
    from unittest.mock import patch
    from gui.lnt_open import build_lnt
    with patch('gui.lnt_native.convert_las',side_effect=AssertionError('reconverted cached source')):
        for path in paths:assert build_lnt(str(path),lambda *a:None,lambda:False)==str(path.with_suffix('.lnt'))
        build_batch(paths[::-1])
    print('PASS each source reopens independently and in a different selection without conversion',flush=True)
print('ALL INDIVIDUAL LNT COLLECTION CHECKS PASSED',flush=True)
