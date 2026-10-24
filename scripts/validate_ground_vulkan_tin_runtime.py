"""Validate actual T400 Vulkan TIN facet dispatch + full PTD class parity.

Run after building DLL+SPIR-V on Windows. No LNT/LAZ writes. Unlike a
GPU-renderer check, readiness sends and receives actual shader data.
This is GPU FACET evaluation only. Delaunay topology remains SciPy on CPU.
"""
from __future__ import annotations
import ast
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial import Delaunay
from gui import ground_ptd_vulkan_tin as tin


def _load_ptd():
    file = Path(__file__).resolve().parents[1]/'gui/lidar_classification_tools.py'
    names={'_select_ground_class0_holdout','_subsample_ground_tin_points',
           '_bounded_ground_seed_indices','_spatial_cell_low_indices',
           '_estimate_adaptive_ground_thresholds','classify_ground_ptd'}
    nodes=[n for n in ast.parse(file.read_text(encoding='utf-8')).body
           if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=names:
        raise RuntimeError('Source changed: cannot extract reference Ground PTD')
    ns={'np':np,'Delaunay':Delaunay}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(file),'exec'),ns)
    return ns['classify_ground_ptd']


def main():
    os.environ['NAKSHA_GROUND_VK_TIN']='1'
    valid,why=tin.readiness()
    print('[GROUND VULKAN TIN HARDWARE]',why,flush=True)
    if not valid:
        raise RuntimeError('Actual Vulkan TIN dispatch FAILED. Do not enable the optional TIN shader.')
    # Use realistic LAS-scale X/Y coordinates, negative triangle orientation,
    # nearly horizontal triangles, and deterministic large-batch geometry.
    rng=np.random.default_rng(20261024)
    n=45000
    a=rng.uniform([256000,4837000,90],[257000,4838000,120],(n,3))
    b=a+rng.normal(0,1,(n,3))
    c=a+rng.normal(0,1,(n,3))
    c[::7], b[::7] = b[::7].copy(),c[::7].copy()
    baseline=tin.cpu_facets(a,b,c)[:3]
    started=perf_counter()
    observed=tin.exact_facet_geometry(a,b,c)
    gpu_time=perf_counter()-started
    for r,g in zip(baseline,observed):
        np.testing.assert_array_equal(np.ascontiguousarray(r).view('u1'),
                                      np.ascontiguousarray(g).view('u1'))
    summary=tin.stats()
    if summary['calls']<1 or summary['facets']<n:
        raise AssertionError('The shader was not dispatched for batch triangles')
    print('[GROUND VULKAN TIN FACET PARITY] {} facets exact CPU bytes, GPU rows exact={} '
          'CPU fallback={} wall={:.3f}s'.format(
              n,summary['gpu_exact_facets'],summary['fallback_facets'],gpu_time),flush=True)

    classify=_load_ptd()
    xy=rng.uniform([256000,4837000],[256065,4837065],(9000,2))
    z=110+.014*(xy[:,0]-256000)+rng.choice([0.,.02,.4,2.,8.],len(xy),p=[.35,.25,.2,.15,.05])
    xyz=np.c_[xy,z]
    classes=np.zeros(len(xyz),np.uint8);classes[::31]=2
    params=dict(from_classes=[0],current_ground=[2],to_class=2,
                max_building_size=25.,terrascan_compatible_densification=True)
    before_v2=os.environ.get('NAKSHA_GROUND_ENGINE_V2')
    before_gpu=os.environ.get('NAKSHA_GROUND_GPU')
    try:
        os.environ['NAKSHA_GROUND_ENGINE_V2']='0'
        os.environ['NAKSHA_GROUND_GPU']='0'
        os.environ['NAKSHA_GROUND_VK_TIN']='0'
        start=perf_counter()
        reference=classes.copy(); ref_result=classify(xyz,reference,**params)
        cpu_elapsed=perf_counter()-start
        os.environ['NAKSHA_GROUND_VK_TIN']='1'
        start=perf_counter()
        observed=classes.copy(); out_result=classify(xyz,observed,**params)
        gpu_elapsed=perf_counter()-start
        np.testing.assert_array_equal(reference,observed)
        for k in ref_result:
            np.testing.assert_equal(ref_result[k],out_result[k],err_msg=k)
        final=tin.stats()
        print('[GROUND VULKAN TIN CLASS PARITY] point IDs/classes/result fields identical '
              'for {} points CPU={:.3f}s Vulkan_facets={:.3f}s TIN_GPU_calls={} '
              'TIN_GPU_facets={} Delaunay_topology=CPU_SciPy'.format(
                  len(classes),cpu_elapsed,gpu_elapsed,final['calls'],final['facets']),flush=True)
    finally:
        for key,value in [('NAKSHA_GROUND_ENGINE_V2',before_v2),('NAKSHA_GROUND_GPU',before_gpu)]:
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        os.environ['NAKSHA_GROUND_VK_TIN']='0'
    print('GROUND VULKAN TIN HARDWARE FACET AND PTD PARITY PASSED. Still experimental.',flush=True)


if __name__=='__main__':main()
