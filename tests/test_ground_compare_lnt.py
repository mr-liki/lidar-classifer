import importlib.util
from pathlib import Path

import numpy as np
import pytest

POINT=np.dtype([('x','<u2'),('y','<u2'),('z','<u2'),
                ('classification','u1')])
SRC=Path(__file__).resolve().parents[1] / 'scripts/compare_ground_lnt_classes.py'
spec=importlib.util.spec_from_file_location('naksha_class_diff',SRC)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture_files(tmp_path):
    rec=np.zeros(550,POINT)
    rec['x']=np.arange(len(rec),dtype=np.uint16)
    rec['y']=np.arange(len(rec),dtype=np.uint16)*2
    rec['z']=1200
    rec['classification']=0
    a,b=tmp_path/'cpu.lnt',tmp_path/'gpu.lnt'
    a.write_bytes(rec.tobytes())
    b.write_bytes(rec.tobytes())
    def info(f,max_points):
        return {'total':550,'origin':np.zeros(3),
                'leaves':[(101,0,250,np.zeros(3),100),
                          (102,250*POINT.itemsize,300,np.zeros(3),100)]}
    return a,b,rec,info


def _compare(a,b,info):
    return mod.compare_lnt(a,b,inspector=info,point_dtype=POINT,
                           exactly=lambda f,n,name:f.read(n),chunk=67)


def test_exact_equal_cpu_and_gpu_lnt(tmp_path):
    a,b,rec,info=fixture_files(tmp_path)
    report=_compare(a,b,info)
    assert report['identical'] and report['points']==550
    assert report['different_classifications']==0


def test_reports_exact_node_local_class_mismatch(tmp_path):
    a,b,rec,info=fixture_files(tmp_path)
    rec['classification'][[3,300,549]]=[1,2,7]
    b.write_bytes(rec.tobytes())
    report=_compare(a,b,info)
    assert not report['identical']
    assert report['different_classifications']==3
    assert [row[:2] for row in report['examples']] == [(101,3),(102,50),(102,299)]


def test_refuses_geometry_change_or_same_file(tmp_path):
    a,b,rec,info=fixture_files(tmp_path)
    rec['x'][100]+=1
    b.write_bytes(rec.tobytes())
    with pytest.raises(mod.LNTComparisonError,match='Non-classification'):
        _compare(a,b,info)
    with pytest.raises(mod.LNTComparisonError,match='Two different'):
        _compare(a,a,info)
