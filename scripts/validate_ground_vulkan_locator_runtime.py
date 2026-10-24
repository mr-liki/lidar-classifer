"""Hardware Vulkan FP64 point-to-simplex dispatch; read-only.

Requires new DLL/shader. Runs both GPU-CPU differential queries and
an end-to-end PTD comparison on the identical synthetic input state.
Does NOT validate full 47M-cloud speed or GPU Delaunay connectivity.
"""
import ast
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial import Delaunay
from gui import ground_ptd_vulkan_locator as vk
from gui import ground_ptd_engine_v2 as engine


def _load_ptd():
    path = Path(__file__).resolve().parents[1]/'gui/lidar_classification_tools.py'
    names={'_select_ground_class0_holdout','_subsample_ground_tin_points',
           '_bounded_ground_seed_indices','_spatial_cell_low_indices',
           '_estimate_adaptive_ground_thresholds','classify_ground_ptd'}
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body
           if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise RuntimeError('Ground PTD reference extraction changed; abort runtime audit')
    scope={'np':np,'Delaunay':Delaunay}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])), str(path), 'exec'),scope)
    return scope['classify_ground_ptd']


def main():
    rng=np.random.default_rng(20261024)
    xy=rng.uniform([256000.,4837000.],[256250.,4837300.],(3500,2))
    tri=Delaunay(xy)
    query=rng.uniform([255990.,4836990.],[256260.,4837310.],(70000,2))
    # Include points precisely on triangle edges and vertices.
    query[:1000]=tri.points[tri.simplices[:1000,:2]].mean(axis=1)
    query[1000:1500]=tri.points[:500]
    query=np.ascontiguousarray(query,dtype=np.float64)
    t0=perf_counter()
    reference=tri.find_simplex(query).astype(np.int32)
    cpu_sec=perf_counter()-t0
    t0=perf_counter()
    proposed, index_sec, gpu_sec=vk._gpu_propose(tri,query)
    wall=perf_counter()-t0
    certified=proposed>=0
    mismatch=np.flatnonzero(certified & (proposed!=reference))
    print(f'[GROUND VULKAN GPU LOCATOR] candidates={len(query)} certified={int(certified.sum())} '
          f'uncertain={int((~certified).sum())} mismatches={len(mismatch)} '
          f'CPU_SciPy={cpu_sec:.4f}s gpu_index={index_sec:.4f}s gpu_dispatch={gpu_sec:.4f}s total={wall:.4f}s',flush=True)
    if len(mismatch):
        raise AssertionError(f'GPU proposal misidentified simplex: first row {int(mismatch[0])}. Keep GPU locator disabled.')
    if certified.sum()<100:
        raise AssertionError('Real GPU simplex shader returned too few certified interior IDs')
    os.environ['NAKSHA_GROUND_VK_LOCATOR']='1'
    os.environ['NAKSHA_GROUND_VK_LOCATOR_AUDIT']='1'
    os.environ['NAKSHA_GROUND_VK_LOCATOR_EXPERIMENTAL_FAST']='0'
    result=vk.locate(tri,query,tri.find_simplex)
    np.testing.assert_array_equal(result,reference)
    if vk.diagnostics()['audited']<len(query):
        raise AssertionError('Complete CPU reference audit not recorded')
    print('[GROUND VULKAN GPU LOCATOR] full-query exact equivalence PASS',flush=True)

    classify=_load_ptd()
    ptsxy=rng.uniform([256000.,4837000.],[256065.,4837065.],(14000,2))
    z=110.+.014*(ptsxy[:,0]-256000.)+rng.choice([0.,.02,.4,2.,8.],len(ptsxy),p=[.35,.25,.2,.15,.05])
    pts=np.c_[ptsxy,z]
    original=np.zeros(len(pts),np.uint8)
    original[::31]=2
    params=dict(from_classes=[0],current_ground=[2],to_class=2,
                max_building_size=25.,terrascan_compatible_densification=True)
    env_keys=['NAKSHA_GROUND_ENGINE_V2','NAKSHA_GROUND_CPP','NAKSHA_GROUND_GPU','NAKSHA_GROUND_VK_TIN',
              'NAKSHA_GROUND_VK_LOCATOR','NAKSHA_GROUND_V2_AUDIT']
    saved={k:os.environ.get(k) for k in env_keys}
    original_version=engine.scipy.__version__
    try:
        # Check whether this SciPy wheel supports the fast CPU path.
        if original_version != '1.15.3':
            raise RuntimeError('PTD full parity requires project SciPy 1.15.3; got '+original_version)
        for k in ['NAKSHA_GROUND_CPP','NAKSHA_GROUND_GPU','NAKSHA_GROUND_VK_TIN']:
            os.environ[k]='0'
        os.environ['NAKSHA_GROUND_ENGINE_V2']='1'
        os.environ['NAKSHA_GROUND_V2_AUDIT']='0'
        os.environ['NAKSHA_GROUND_VK_LOCATOR']='0'
        t0=perf_counter()
        c0=original.copy(); r0=classify(pts,c0,**params)
        cpu_total=perf_counter()-t0
        os.environ['NAKSHA_GROUND_VK_LOCATOR']='1'
        t0=perf_counter()
        c1=original.copy(); r1=classify(pts,c1,**params)
        gpu_total=perf_counter()-t0
        np.testing.assert_array_equal(c0,c1)
        for k in r0:
            np.testing.assert_equal(r0[k],r1[k],err_msg=k)
        print(f'[GROUND VULKAN GPU LOCATOR PTD PARITY] 14000 points same IDs/classes/results '
              f'cpu={cpu_total:.3f}s gpu_audited={gpu_total:.3f}s GPU_calls={vk.diagnostics()["calls"]} '
              'Delaunay_topology=CPU_SciPy',flush=True)
    finally:
        for k,val in saved.items():
            if val is None: os.environ.pop(k,None)
            else: os.environ[k]=val
    print('GROUND VULKAN LOCATOR HARDWARE/CLASS PARITY PASSED; remains experimental.',flush=True)

if __name__=='__main__': main()
