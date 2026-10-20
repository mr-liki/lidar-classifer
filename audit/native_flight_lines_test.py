"""Real Line rendering, visibility and Fields on individual/collected LNTs."""
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import laspy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gui import lnt_native as nat
nat.LNT_DLL=os.environ.get('NAKSHA_TEST_LNT_DLL',nat.LNT_DLL)
from gui.lnt_batch import build_batch
from gui.native_fields import NativeFieldsReader, FieldsTask, K_INT0
from gui.native_flight_lines import FlightLines, ensure_sidecars
from gui.native_section_display import line_colors
from audit.lnt_surface_lod_view_test import View

with tempfile.TemporaryDirectory(dir=ROOT/'audit') as work:
    folder=Path(work); paths=[]
    for index in range(2):
        p=folder/f'flight{index}.laz'; paths.append(p)
        h=laspy.LasHeader(point_format=3,version='1.2')
        h.offsets=[500000.,4000000.,0.]
        a=laspy.LasData(h)
        x,y=np.meshgrid(np.linspace(0,100,101),np.linspace(0,50,51))
        a.x=x.ravel()+500000.+index*130; a.y=y.ravel()+4000000.; a.z=x.ravel()*.1
        a.point_source_id=np.where(x.ravel()<50,11+index*10,12+index*10).astype(np.uint16)
        a.classification=np.full(x.size,2,np.uint8)
        a.write(p)
    _,index=build_batch(paths)
    flight=FlightLines(index)
    assert set(flight.unique())=={11,12,21,22}
    reader=NativeFieldsReader(index)
    try:
        data=reader.read(np.arange(reader.count))
        assert reader.count==10302
        for file in range(2):
            mask=np.array([str(paths[file])==p for p in data['source_file']])
            expected=np.where(data['xyz'][mask,0]<500049.99+file*130,11+file*10,12+file*10)
            np.testing.assert_array_equal(data['point_source_id'][mask],expected)
        assert len(reader.rows_for_slots([2]))==reader.count
        task=FieldsTask(index,K_INT0,'point_source_id')
        order=task._sort(reader)
        assert np.all(np.diff(data['point_source_id'][order].astype(int))>=0)
    finally: reader.close()
    print('PASS Fields: every collection row has exact flight line, owning source, class filtering and sorting',flush=True)
    v=View(nat,index)
    try:
        colors=line_colors(np.arange(65536,dtype=np.uint32))
        visible=np.ones(65536,np.uint8)
        assert v.nv.set_lines(colors,visible)
        for _ in range(60): assert v.nv.frame(.016); time.sleep(.005)
        img=v.nv.capture()[:,:,:3]
        lit=img[np.any(img>0,axis=2)].astype(int)
        palette=colors[[11,12,21,22]].astype(int)
        assert len(lit)>1000
        assert np.all(np.min(np.max(np.abs(lit[:,None]-palette),axis=2),axis=1)<=2)
        visible[:]=0; assert v.nv.set_lines(colors,visible)
        for _ in range(30): assert v.nv.frame(.016); time.sleep(.005)
        assert not np.any(v.nv.capture()[:,:,:3])
        v.nv.set_mode(nat.MODE_CLASS)
        for _ in range(30): assert v.nv.frame(.016); time.sleep(.005)
        assert np.any(v.nv.capture()[:,:,:3])
    finally: v.close()
    print('PASS native Line: correct colours, hide-all clears points, Class restores original rendering',flush=True)
    # Simulate a pre-existing classified cache with no attribute sidecar.
    old=folder/'old.lnt'
    old.write_bytes(paths[0].with_suffix('.lnt').read_bytes())
    from gui.lnt_collection import read_plain_directory
    _,_,nodes=read_plain_directory(old)
    leaf=nodes[nodes['children_mask']==0][0]
    with open(old,'r+b') as f:
        f.seek(int(leaf['byte_offset'])+10); f.write(bytes([9]))
    before=old.read_bytes()
    view=SimpleNamespace(lnt_path=str(old),app=SimpleNamespace(_lnt_source_path=str(paths[0])))
    recovered=ensure_sidecars(view)
    assert set(recovered.unique())=={11,12} and old.read_bytes()==before
    recovered.close(); flight.close()
    print('PASS older cache backfill keeps the complete classified LNT byte-for-byte intact',flush=True)
    duplicate=folder/'coincident.las'
    h=laspy.LasHeader(point_format=3,version='1.2')
    a=laspy.LasData(h)
    a.x=[1.,1.]; a.y=[2.,2.]; a.z=[3.,3.]; a.point_source_id=[101,202]
    a.write(duplicate)
    from gui.lnt_open import build_lnt
    cache=build_lnt(str(duplicate),lambda *args:None,lambda:False)
    os.remove(cache+'.lines')
    recovered=ensure_sidecars(SimpleNamespace(lnt_path=cache,app=SimpleNamespace(_lnt_source_path=str(duplicate))))
    assert set(recovered.unique())=={101,202}
    recovered.close()
    print('PASS coincident returns retain distinct original flight lines during old-cache recovery',flush=True)
print('ALL FLIGHT LINE CHECKS PASSED',flush=True)
