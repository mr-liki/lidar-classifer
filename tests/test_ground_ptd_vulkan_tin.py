"""CPU-only tests: Vulkan TIN facet bridge must NEVER change PTD results.

All tests exercise the actual bridge and source AST; a Windows hardware smoke
is separately invoked after the new GPU DLL/shader build.
"""
import ast
import ctypes
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay

from gui import ground_ptd_vulkan_tin as tin


class FakeTinDLL:
    def __init__(self, mode='correct', capacity=16384):
        self.mode, self.capacity = mode, capacity
        self.calls = 0
        self.uploaded = []

    def nk_ground_vk_tin_max_batch(self):
        return self.capacity

    def nk_ground_vk_tin_build(self, ctx, xyz_ptr, n, out_ptr, message, capacity):
        self.calls += 1
        if self.mode == 'fail':
            return -1
        xyz = np.ctypeslib.as_array(xyz_ptr, shape=(n*9,)).reshape(n,9)
        self.uploaded.append(xyz.copy())
        a, b, c = xyz[:, :3], xyz[:, 3:6], xyz[:, 6:9]
        me, nor, _, length_n = tin.cpu_facets(a,b,c)
        out = np.ctypeslib.as_array(out_ptr, shape=(n*5,)).reshape(n,5)
        out[:,:3], out[:,3], out[:,4] = nor,me,length_n
        if self.mode == 'wrong':
            out[:,0] = 1.0  # deliberately wrong normal; cannot leak to PTD
            out[:,3] *= 20.
        if self.mode == 'rounding':
            out[::2,3] = np.nextafter(out[::2,3], np.inf)
        return 0


def triangles(n=100,seed=418):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-100,100,(n,3)).astype(np.float64)
    b = a + rng.normal(size=(n,3))
    c = a + rng.normal(size=(n,3))
    return a,b,c


def assert_cpu_equal(reference,actual):
    for before,after in zip(reference,actual):
        np.testing.assert_array_equal(np.ascontiguousarray(before).view(np.uint8),
                                      np.ascontiguousarray(after).view(np.uint8))


def test_cpu_path_is_exact_original_numpy_expression(monkeypatch):
    monkeypatch.setenv('NAKSHA_GROUND_VK_TIN','0')
    a,b,c=triangles()
    me = np.maximum(np.maximum(np.linalg.norm(b-a,axis=1),np.linalg.norm(c-b,axis=1)),np.linalg.norm(a-c,axis=1))
    nor=np.cross(b-a,c-a);nl=np.linalg.norm(nor,axis=1,keepdims=True)
    good=nl[:,0]>1e-12;nor=nor/np.where(nl>0,nl,1.0);nor[nor[:,2]<0]*=-1
    assert_cpu_equal((me,nor,good),tin.exact_facet_geometry(a,b,c))


def test_correct_gpu_facets_are_actually_selected_and_batched(monkeypatch):
    fake=FakeTinDLL(capacity=64)
    monkeypatch.setenv('NAKSHA_GROUND_VK_TIN','1')
    monkeypatch.setattr(tin,'_get_context',lambda:(fake,ctypes.c_void_p(1)))
    a,b,c=triangles(210)
    before=(a.copy(),b.copy(),c.copy())
    expected=tin.cpu_facets(a,b,c)[:3]
    got=tin.exact_facet_geometry(a,b,c)
    assert_cpu_equal(expected,got)
    assert fake.calls==4
    assert np.array_equal(fake.uploaded[0][:,0:3],a[:64])
    assert np.array_equal(fake.uploaded[0][:,3:6],b[:64])
    assert np.array_equal(fake.uploaded[0][:,6:9],c[:64])
    for x, y in zip((a,b,c),before):np.testing.assert_array_equal(x,y)


@pytest.mark.parametrize('mode',['wrong','rounding','fail'])
def test_malicious_or_numerically_different_gpu_is_fully_corrected(monkeypatch,mode):
    fake=FakeTinDLL(mode=mode)
    monkeypatch.setenv('NAKSHA_GROUND_VK_TIN','1')
    monkeypatch.setattr(tin,'_get_context',lambda:(fake,ctypes.c_void_p(1)))
    a,b,c=triangles()
    expected=tin.cpu_facets(a,b,c)[:3]
    got=tin.exact_facet_geometry(a,b,c)
    assert_cpu_equal(expected,got)
    assert fake.calls==1


def test_nonfinite_data_never_submitted(monkeypatch):
    fake=FakeTinDLL()
    monkeypatch.setenv('NAKSHA_GROUND_VK_TIN','1')
    monkeypatch.setattr(tin,'_get_context',lambda:(fake,ctypes.c_void_p(1)))
    a,b,c=triangles()
    a[1,2]=np.nan
    with np.errstate(invalid='ignore'):
        result=tin.exact_facet_geometry(a,b,c)
    assert result[0].shape==(100,)
    assert fake.calls == 0


@pytest.fixture(scope='module')
def classify():
    path = Path(__file__).resolve().parents[1] / 'gui/lidar_classification_tools.py'
    tree = ast.parse(path.read_text(encoding='utf8'), filename=str(path))
    names={'_select_ground_class0_holdout','_subsample_ground_tin_points',
           '_bounded_ground_seed_indices','_spatial_cell_low_indices',
           '_estimate_adaptive_ground_thresholds','classify_ground_ptd'}
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {n.name for n in selected}==names
    scope={'np':np,'Delaunay':Delaunay}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(path),'exec'),scope)
    return scope['classify_ground_ptd']


@pytest.mark.parametrize('fenced',[True,False])
@pytest.mark.parametrize('upward',[False,True])
def test_full_classification_unchanged_by_false_gpu_facets(monkeypatch,classify,fenced,upward):
    rng=np.random.default_rng(413)
    xy=rng.uniform(0,48,(6600,2))+[256000,4837000]
    z=120+.035*(xy[:,0]-256000)+rng.uniform(0,6,len(xy))
    xyz=np.c_[xy,z]
    initial=np.zeros(len(xyz),dtype=np.uint8);initial[::23]=2
    fence=(xy[:,0]<256036) if fenced else None
    args=dict(from_classes=[0],current_ground=[2],to_class=2,
              max_building_size=25., add_only_upward=upward,
              terrascan_compatible_densification=True)
    monkeypatch.setenv('NAKSHA_GROUND_ENGINE_V2','0')
    monkeypatch.setenv('NAKSHA_GROUND_CPP','0')
    monkeypatch.setenv('NAKSHA_GROUND_GPU','0')
    monkeypatch.setenv('NAKSHA_GROUND_EARLY_REJECT','1')
    monkeypatch.setenv('NAKSHA_GROUND_PROFILE','0')
    monkeypatch.setenv('NAKSHA_GROUND_VK_TIN','0')
    original=initial.copy()
    res=classify(xyz,original,fence_mask=fence,**args)
    fake=FakeTinDLL(mode='wrong')
    monkeypatch.setenv('NAKSHA_GROUND_VK_TIN','1')
    monkeypatch.setattr(tin,'_get_context',lambda:(fake,ctypes.c_void_p(1)))
    test=initial.copy()
    out=classify(xyz,test,fence_mask=fence,**args)
    assert fake.calls>0
    np.testing.assert_array_equal(original,test)
    for k in res:
        np.testing.assert_equal(res[k],out[k],err_msg=k)


def test_existing_vulkan_distance_gate_not_replaced_by_tin():
    path=Path(__file__).resolve().parents[1]/'gui/lidar_classification_tools.py'
    src=path.read_text(encoding='utf8')
    assert src.count('me, nor, good = _tin_geometry(A3, B3, C3)')==2
    assert 'from gui.ground_ptd_gpu import conservative_cuda_prefilter' in src
