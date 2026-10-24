"""GPU locator source-level protection. Hardware dispatch is a separate test."""
import os
import numpy as np
import pytest
from scipy.spatial import Delaunay

from gui import ground_ptd_vulkan_locator as loc


def _tri(seed=20261024,n=900):
    r=np.random.default_rng(seed)
    xy=r.uniform([256000.,4837000.],[256250.,4837220.],size=(n,2))
    return Delaunay(xy)


def _queries(tri, seed=92,n=12000):
    r=np.random.default_rng(seed)
    # Point centroids, random exterior and near-boundary positions.
    tri_xyz=tri.points[tri.simplices]
    inside=tri_xyz[r.integers(len(tri_xyz),size=n//2)].mean(axis=1)
    outside=r.uniform([255900.,4836900.],[256350.,4837320.],size=(n-n//2,2))
    return np.ascontiguousarray(np.vstack([inside,outside]),dtype=np.float64)


def test_grid_csr_contains_vertices_and_centroids():
    tri=_tri(n=900)
    transform,offsets,ids,x0,y0,dx,dy=loc.build_index(tri)
    assert transform.shape==(len(tri.simplices),3,2)
    assert offsets.shape==(4097,)
    assert offsets[0]==0 and offsets[-1]==len(ids)
    assert np.all(offsets[1:]>=offsets[:-1])
    # Every triangle belongs to the bin containing its centroid.
    for sid,vertices in enumerate(tri.simplices):
        center=tri.points[vertices].mean(axis=0)
        bx=int(np.floor((center[0]-x0)*dx))
        by=int(np.floor((center[1]-y0)*dy))
        assert 0<=bx<64 and 0<=by<64
        cell=by*64+bx
        assert sid in ids[offsets[cell]:offsets[cell+1]]


def test_grid_duplicate_and_degenerate_rejected():
    tri=_tri()
    tri.transform[:] = np.nan  # only affects test-local triangulation
    with pytest.raises(ValueError):
        loc.build_index(tri)


def test_disabled_never_dispatches(monkeypatch):
    tri=_tri()
    query=_queries(tri)
    monkeypatch.delenv('NAKSHA_GROUND_VK_LOCATOR',raising=False)
    monkeypatch.setattr(loc,'_gpu_propose',lambda *a: (_ for _ in ()).throw(AssertionError('unexpected GPU')))
    assert loc.locate(tri,query,tri.find_simplex) is None


def test_exact_audit_restores_malicious_id(monkeypatch):
    tri=_tri()
    query=_queries(tri)
    refs=tri.find_simplex(query).astype(np.int32)
    bad=refs.copy(); bad[:300]=0
    monkeypatch.setenv('NAKSHA_GROUND_VK_LOCATOR','1')
    monkeypatch.setenv('NAKSHA_GROUND_VK_LOCATOR_AUDIT','1')
    monkeypatch.setattr(loc,'_gpu_propose',lambda *a:(bad,0.,0.))
    actual=loc.locate(tri,query,tri.find_simplex)
    np.testing.assert_array_equal(actual,refs)


def test_exact_audit_reference_when_gpu_ambiguous(monkeypatch):
    tri=_tri()
    query=_queries(tri)
    refs=tri.find_simplex(query).astype(np.int32)
    gpu=refs.copy()
    gpu[::3]=-2
    monkeypatch.setenv('NAKSHA_GROUND_VK_LOCATOR','1')
    monkeypatch.setenv('NAKSHA_GROUND_VK_LOCATOR_AUDIT','1')
    monkeypatch.setattr(loc,'_gpu_propose',lambda *a:(gpu,0.,0.))
    actual=loc.locate(tri,query,tri.find_simplex)
    np.testing.assert_array_equal(actual,refs)


def test_fast_explicit_dual_optin_cpu_only_for_uncertain(monkeypatch):
    tri=_tri()
    query=_queries(tri)
    refs=tri.find_simplex(query).astype(np.int32)
    gpu=refs.copy()
    gpu[::4]=-2
    processed=[]
    def cpu_ref(q):
        processed.append(len(q))
        return tri.find_simplex(q)
    monkeypatch.setenv('NAKSHA_GROUND_VK_LOCATOR','1')
    monkeypatch.setenv('NAKSHA_GROUND_VK_LOCATOR_AUDIT','0')
    monkeypatch.setenv('NAKSHA_GROUND_VK_LOCATOR_EXPERIMENTAL_FAST','1')
    monkeypatch.setattr(loc,'_gpu_propose',lambda *a:(gpu,0.,0.))
    actual=loc.locate(tri,query,cpu_ref)
    np.testing.assert_array_equal(actual,refs)
    assert processed==[int(np.count_nonzero(gpu < 0))]


def test_audit_cannot_be_bypassed_without_both_flags(monkeypatch):
    tri=_tri()
    query=_queries(tri)
    ref=tri.find_simplex(query).astype(np.int32)
    gpu=ref.copy();gpu[::2]=-2
    calls=[]
    def cpu_ref(q):
        calls.append(len(q));return tri.find_simplex(q)
    monkeypatch.setenv('NAKSHA_GROUND_VK_LOCATOR','1')
    monkeypatch.setenv('NAKSHA_GROUND_VK_LOCATOR_AUDIT','0')
    monkeypatch.delenv('NAKSHA_GROUND_VK_LOCATOR_EXPERIMENTAL_FAST',raising=False)
    monkeypatch.setattr(loc,'_gpu_propose',lambda *a:(gpu,0.,0.))
    np.testing.assert_array_equal(loc.locate(tri,query,cpu_ref),ref)
    assert calls==[len(query)]


def test_gpu_failure_does_not_change_ground(monkeypatch):
    tri=_tri()
    query=_queries(tri)
    monkeypatch.setenv('NAKSHA_GROUND_VK_LOCATOR','1')
    monkeypatch.setattr(loc,'_gpu_propose',lambda *a: (_ for _ in ()).throw(OSError('Vulkan error')))
    assert loc.locate(tri,query,tri.find_simplex) is None


def test_small_queries_keep_original_path(monkeypatch):
    tri=_tri()
    query=_queries(tri)[:200]
    monkeypatch.setenv('NAKSHA_GROUND_VK_LOCATOR','1')
    monkeypatch.setattr(loc,'_gpu_propose',lambda *a: (_ for _ in ()).throw(AssertionError('unexpected GPU')))
    assert loc.locate(tri,query,tri.find_simplex) is None


def test_source_engine_calls_optional_locator(monkeypatch):
    from gui import ground_ptd_engine_v2 as engine
    tri=_tri(n=200)
    query=_queries(tri)[:500]
    reference=tri.find_simplex(query)
    monkeypatch.setenv('NAKSHA_GROUND_VK_LOCATOR','1')
    monkeypatch.setattr(engine.scipy,'__version__','1.15.3')
    called=[]
    def fake_locate(t, q, cpu, stats):
        called.append(len(q))
        return cpu(q)
    monkeypatch.setattr(loc,'locate',fake_locate)
    got=engine.find_simplex(tri,query,verify=False)
    np.testing.assert_array_equal(got,reference)
    assert called==[len(query)]


def _simulate_shader_grid(tri, queries):
    """Independent Python simulation of the *strict interior* shader contract.

    Verifies bounding-index coverage and excludes near-edge ties before the
    actual Vulkan hardware smoke test runs on the user's T400.
    """
    tr, offsets, entries, x0, y0, dx, dy = loc.build_index(tri)
    out = np.full(len(queries), -2, dtype=np.int32)
    for i, (x,y) in enumerate(queries):
        gx=(x-x0)*dx;gy=(y-y0)*dy
        if gx < 0 or gy < 0 or gx >= 64 or gy >= 64:
            continue
        cell = 64*int(np.floor(gy))+int(np.floor(gx))
        ids = entries[offsets[cell]:offsets[cell+1]]
        if len(ids)>512:
            continue
        possible=0
        winner=-2
        for sid in ids:
            a,b,c,d,bx,by=tr[sid].ravel()
            xx=x-bx;yy=y-by
            z0=xx*a+yy*b
            z1=xx*c+yy*d
            z2=1-z0-z1
            bound0=1e-12+128*np.finfo(float).eps*(abs(xx*a)+abs(yy*b)+1)
            bound1=1e-12+128*np.finfo(float).eps*(abs(xx*c)+abs(yy*d)+1)
            tol=max(bound0,bound1,1e-12+bound0+bound1)
            if -tol<=z0<=1+tol and -tol<=z1<=1+tol and -tol<=z2<=1+tol:
                possible+=1
                if possible>1:winner=-2;break
                if tol<z0<1-tol and tol<z1<1-tol and tol<z2<1-tol:
                    winner=int(sid)
        if possible==1:
            out[i]=winner
    return out


@pytest.mark.parametrize('seed', [19, 52, 20261024])
def test_cpu_shader_simulation_strict_interior_reference(seed):
    rng=np.random.default_rng(seed)
    verts=rng.uniform([255900,4836900],[256250,4837220],(850,2))
    tri=Delaunay(verts)
    q=rng.uniform(verts.min(0)-5,verts.max(0)+5,(1600,2))
    q[:200]=tri.points[tri.simplices[:200]].mean(axis=1)
    q[200:400]=tri.points[tri.simplices[:200,:2]].mean(axis=1)
    proposals=_simulate_shader_grid(tri,q)
    certified=proposals>=0
    assert certified.sum()>100
    np.testing.assert_array_equal(proposals[certified],tri.find_simplex(q)[certified])


def test_shader_simulation_close_to_grid_boundaries():
    tri=_tri(seed=21,n=650)
    verts=tri.points[tri.simplices]
    q=np.r_[verts[:200].mean(axis=1),
             verts[:200,:2].mean(axis=1),
             np.nextafter(verts[:200,:2].mean(axis=1),np.inf),
             np.nextafter(verts[:200,:2].mean(axis=1),-np.inf)]
    proposals=_simulate_shader_grid(tri,q)
    certified=proposals>=0
    np.testing.assert_array_equal(proposals[certified],tri.find_simplex(q)[certified])


def test_complete_ground_classes_unchanged_after_malicious_gpu_ids(monkeypatch):
    """End-to-end 12k-point Ground audit keeps every class and result field."""
    import ast
    from pathlib import Path
    from gui import ground_ptd_engine_v2 as engine
    source = Path(__file__).resolve().parents[1] / 'gui' / 'lidar_classification_tools.py'
    names={'_select_ground_class0_holdout','_subsample_ground_tin_points',
           '_bounded_ground_seed_indices','_spatial_cell_low_indices',
           '_estimate_adaptive_ground_thresholds','classify_ground_ptd'}
    tree=ast.parse(source.read_text(encoding='utf-8'))
    functions=[v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name in names]
    assert {v.name for v in functions}==names
    ns={'np':np,'Delaunay':Delaunay}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(source),'exec'),ns)
    classify=ns['classify_ground_ptd']
    rng=np.random.default_rng(20261024)
    xy=rng.uniform([256000,4837000],[256080,4837080],(12000,2))
    xyz=np.c_[xy, 100+.01*(xy[:,0]-256000)+rng.choice([0.,.05,.8,3.],len(xy))]
    original=np.zeros(len(xy),np.uint8)
    original[::23]=2
    kwargs=dict(from_classes=[0],current_ground=[2],to_class=2,
                max_building_size=25.,terrascan_compatible_densification=True)
    monkeypatch.setattr(engine.scipy,'__version__','1.15.3')
    for env in ('NAKSHA_GROUND_GPU','NAKSHA_GROUND_VK_TIN','NAKSHA_GROUND_CPP'):
        monkeypatch.setenv(env,'0')
    monkeypatch.setenv('NAKSHA_GROUND_ENGINE_V2','1')
    monkeypatch.setenv('NAKSHA_GROUND_V2_AUDIT','0')
    monkeypatch.setenv('NAKSHA_GROUND_VK_LOCATOR','0')
    a=original.copy()
    ref=classify(xyz,a,**kwargs)
    monkeypatch.setenv('NAKSHA_GROUND_VK_LOCATOR','1')
    monkeypatch.setenv('NAKSHA_GROUND_VK_LOCATOR_AUDIT','1')
    calls=[]
    def corrupt(tri,q):
        calls.append(len(q))
        return np.zeros(len(q),np.int32),0.,0.
    monkeypatch.setattr(loc,'_gpu_propose',corrupt)
    b=original.copy()
    got=classify(xyz,b,**kwargs)
    assert calls, 'test must invoke attempted GPU locator'
    np.testing.assert_array_equal(b,a)
    for key in ref:
        np.testing.assert_equal(got[key],ref[key],err_msg=key)
