"""Peak geometry and collection-reader resource regressions."""
from pathlib import Path
import os
import shutil
import subprocess
import numpy as np
import pytest
from gui import section_surface_math as surface
from gui import section_shading_math as shading

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def native_probe(tmp_path_factory):
    exe=tmp_path_factory.mktemp('surface_peak')/'probe.exe'
    subprocess.run([r'H:/tools/mingw1310_64/bin/g++.exe','-O2','-std=c++17','-static',
        '-I',str(ROOT/'native/lnt/include'),str(ROOT/'audit/surface_peak_reader_test.cpp'),'-o',str(exe)],check=True,
        env={**os.environ,'PATH':r'H:\tools\mingw1310_64\bin;'+os.environ.get('PATH','')})
    return exe

def test_native_peak_geometry(native_probe):
    subprocess.run([str(native_probe)],check=True)

def test_many_collection_readers(native_probe,tmp_path):
    from gui.lnt_collection import write_collection
    source=ROOT/'native/lnt/test_shading.lnt'
    paths=[]
    for i in range(16):
        p=tmp_path/f'flight{i}.lnt'; shutil.copyfile(source,p); paths.append(str(p))
    index=tmp_path/'flights.lntset'; write_collection(index,paths)
    subprocess.run([str(native_probe),str(index)],check=True)

@pytest.mark.parametrize('quality',['fast','normal','slow'])
@pytest.mark.parametrize('reverse',[False,True])
def test_surface_peak_vertex(quality,reverse):
    xyz=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],[0,0,0],[0,0,10]],float)
    if reverse: xyz=xyz[::-1].copy()
    result=surface._compute_surface_geometry_backend(xyz,np.arange(len(xyz)),0.,100,0.,45.,45.,.25,None,quality_mode=quality)
    tips=result['unique_global_idx'][result['faces']]
    tri=xyz[tips]
    centre=(tri[:,:,:2]==0).all(axis=2)
    assert centre.sum()==4
    np.testing.assert_array_equal(tri[:,:,2][centre],np.full(4,10.))

@pytest.mark.parametrize('accelerated',[False,True])
def test_grid_representative_preserves_upper_return(monkeypatch,accelerated):
    monkeypatch.setattr(surface,'_SURFACE_HAS_NUMBA',accelerated)
    xyz=np.array([[0,0,0],[.01,.01,8],[1,0,2],[1.01,.01,9],[0,1,3],[.01,1.01,7]],float)
    selected=surface._surface_grid_select(xyz,.1)
    np.testing.assert_array_equal(selected,[1,3,5])
    # Production's XY-only fast path must still use authoritative heights.
    selected=surface._surface_grid_select(xyz[:,:2],.1,heights=xyz[:,2])
    np.testing.assert_array_equal(selected,[1,3,5])

def test_shading_already_uses_upper_return():
    xyz=np.array([[0,0,0],[0,0,8],[1,0,2],[1,0,9],[0,1,3],[0,1,7]],float)
    chosen=shading._grid_dedup_at_precision(xyz,.1)
    assert set(chosen)=={1,3,5}
