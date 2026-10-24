"""Dense-existing-ground 1/1 tile: opt-in V2 must not alter classifications.

Source is extracted from the installed BRUSH PTD module by AST, so this test
executes the actual algorithm without importing Qt/Vulkan UI. Uses a real
100,000+ trusted-ground point workload to enter the production dense path.
"""
import ast
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from gui import ground_ptd_engine_v2 as v2


def _extract():
    file=Path(__file__).resolve().parents[1]/'gui/lidar_classification_tools.py'
    names={'_select_ground_class0_holdout','_subsample_ground_tin_points',
           '_bounded_ground_seed_indices','_spatial_cell_low_indices',
           '_estimate_adaptive_ground_thresholds','classify_ground_ptd'}
    funcs=[node for node in ast.parse(file.read_text(encoding='utf-8')).body
           if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in funcs} == names
    scope={'np':np,'Delaunay':Delaunay}
    exec(compile(ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[])),
                 str(file),'exec'),scope)
    return scope['classify_ground_ptd']


def test_dense_existing_ground_native_optin_same_class_ids_and_stop(monkeypatch):
    classify=_extract()
    rng=np.random.default_rng(240105)
    ntrusted, nsrc = 100_120, 3_900
    n=ntrusted+nsrc
    xy=rng.uniform([256000.,4837000.],[256035.,4837035.],(n,2))
    z=110.+.002*(xy[:,0]-256000.) + rng.uniform(-.08,.07,n)
    z[ntrusted:]+=rng.choice([0.,.03,1.5,5.],nsrc,p=[.3,.4,.2,.1])
    xyz=np.column_stack((xy,z))
    original=np.zeros(n,np.uint8)
    original[:ntrusted]=2
    kwargs=dict(from_classes=[0],current_ground=[2],to_class=2,
                max_building_size=12.,terrascan_compatible_densification=True)
    monkeypatch.setenv('NAKSHA_GROUND_PROFILE','0')
    monkeypatch.setenv('NAKSHA_GROUND_GPU','0')
    monkeypatch.setenv('NAKSHA_GROUND_VK_TIN','0')
    monkeypatch.setenv('NAKSHA_GROUND_VK_LOCATOR','0')
    monkeypatch.setenv('NAKSHA_GROUND_CPP','0')
    monkeypatch.setenv('NAKSHA_GROUND_ENGINE_V2','1')
    monkeypatch.setenv('NAKSHA_GROUND_EARLY_REJECT','1')
    monkeypatch.setenv('NAKSHA_GROUND_V2_AUDIT','1')
    monkeypatch.setenv('NAKSHA_GROUND_V2_DENSE','0')
    baseline=original.copy()
    r0=classify(xyz,baseline,**kwargs)
    assert r0['dense_ground_fast_path'] is True
    monkeypatch.setenv('NAKSHA_GROUND_V2_DENSE','1')
    init_calls=[]
    real_init=v2.GroundEngineV2.__init__
    def track(self,*a,**kw):
        init_calls.append(True)
        real_init(self,*a,**kw)
    monkeypatch.setattr(v2.GroundEngineV2,'__init__',track)
    actual=original.copy()
    r1=classify(xyz,actual,**kwargs)
    assert init_calls, 'V2 optimizer must run in the one-iteration dense path'
    np.testing.assert_array_equal(actual,baseline)
    for key in r0:
        np.testing.assert_equal(r0[key],r1[key],err_msg=key)
