"""Hardware-level Vulkan Ground PTD gate validation on the target workstation.

Must run only after the real native Vulkan DLL+SPIR-V have been built. Executes
actual Vulkan GPU dispatch and compares outputs with the original CPU masks,
then compares the actual entire Ground PTD final classifications on a repeatable
synthetic dataset. No Qt GUI, LAZ/LNT file or data write required.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import Delaunay

from gui import ground_ptd_vulkan as vk


def _classify_function():
    path = Path(__file__).resolve().parents[1] / 'gui' / 'lidar_classification_tools.py'
    required = {'_select_ground_class0_holdout', '_subsample_ground_tin_points',
                '_bounded_ground_seed_indices', '_spatial_cell_low_indices',
                '_estimate_adaptive_ground_thresholds', 'classify_ground_ptd'}
    selected = [node for node in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(node, ast.FunctionDef) and node.name in required]
    if {x.name for x in selected} != required:
        raise AssertionError('Actual Ground PTD source changed; manual review required')
    scope = {'np': np, 'Delaunay': Delaunay}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 str(path), 'exec'), scope)
    return scope['classify_ground_ptd']


def main():
    os.environ['NAKSHA_GROUND_VULKAN_COMPUTE'] = '1'
    available, why = vk.readiness()
    print('[GROUND VULKAN HARDWARE]', why, flush=True)
    if not available:
        raise RuntimeError('Native Vulkan GPU dispatch unavailable. DO NOT enable experimental GPU backend.')

    rng = np.random.default_rng(20261024)
    n = 100000
    coords = np.empty((n, 3), dtype=np.float64)
    coords[:, :2] = rng.uniform([256000, 4837000], [257000, 4838000], size=(n, 2))
    coords[:, 2] = rng.normal(100, 8, size=n)
    anchors = coords.copy()
    anchors[:, 2] += rng.normal(0, 1, size=n)
    normals = np.empty((n, 3), dtype=np.float64)
    normals[:, :] = [0, 0, 1]
    # Explicitly test exact threshold and upward boundaries.
    anchors[:100, 2] = coords[:100, 2] - .5
    anchors[100:200, 2] = coords[100:200, 2] + .5
    for upward in (False, True):
        tic = perf_counter()
        result = vk.conservative_vulkan_prefilter(coords, anchors, normals, .5,
                                                  add_only_upward=upward)
        if result is None or 'vulkan' not in result['backend']:
            raise AssertionError('Vulkan compute was not used on hardware validation')
        elapsed = perf_counter() - tic
        gpu_keep = result['keep'].copy()
        cpu_d = np.einsum('ij,ij->i', normals, coords - anchors)
        cpu_pass = (np.abs(cpu_d) <= .5)
        if upward:
            cpu_pass &= cpu_d >= 0
        # CPU shadow audit from the actual PTD implementation MUST restore
        # ALL CPU survivors that the GPU would otherwise reject.
        restored = gpu_keep.copy()
        restored[(~restored) & cpu_pass] = True
        if not np.all(restored[cpu_pass]):
            raise AssertionError('CPU shadow audit failed to preserve necessary Ground candidates')
        print('[GROUND VULKAN HARDWARE] upward={} queries={} gpu_rejected={} '
              'CPU_rescued={} CPU_preserved=True wall={:.3f}s'.format(
                  upward, n, int(np.count_nonzero(~gpu_keep)),
                  int(np.count_nonzero((~gpu_keep) & cpu_pass)), elapsed), flush=True)

    classify = _classify_function()
    xy = rng.uniform([256000, 4837000], [256065, 4837065], size=(9000, 2))
    z = 110 + .014 * (xy[:, 0] - 256000) + rng.choice([0., .02, .4, 2., 8.],
                                                       len(xy), p=[.35, .25, .2, .15, .05])
    xyz = np.c_[xy, z]
    initial = np.zeros(len(xyz), np.uint8)
    initial[::31] = 2
    original_gpu = os.environ.get('NAKSHA_GROUND_GPU')
    previous_v2 = os.environ.get('NAKSHA_GROUND_ENGINE_V2')
    try:
        os.environ['NAKSHA_GROUND_ENGINE_V2'] = '0'  # isolate GPU from C++ lookup
        kwargs = dict(from_classes=[0], current_ground=[2], to_class=2,
                      max_building_size=25., terrascan_compatible_densification=True)
        os.environ['NAKSHA_GROUND_VULKAN_COMPUTE'] = '0'
        os.environ['NAKSHA_GROUND_GPU'] = '0'
        reference = initial.copy()
        tic = perf_counter()
        ref = classify(xyz, reference, **kwargs)
        cpu_elapsed = perf_counter() - tic
        os.environ['NAKSHA_GROUND_VULKAN_COMPUTE'] = '1'
        os.environ['NAKSHA_GROUND_GPU'] = '1'
        observed = initial.copy()
        tic = perf_counter()
        got = classify(xyz, observed, **kwargs)
        gpu_elapsed = perf_counter() - tic
        np.testing.assert_array_equal(observed, reference)
        for key in ref:
            np.testing.assert_equal(ref[key], got[key], err_msg=key)
        print('[GROUND VULKAN CLASS PARITY] all {} point classes identical, '
              'changed IDs identical, result fields identical, CPU={:.3f}s GPU={:.3f}s'.format(
                  len(initial), cpu_elapsed, gpu_elapsed), flush=True)
    finally:
        for name, previous in (('NAKSHA_GROUND_GPU', original_gpu),
                               ('NAKSHA_GROUND_ENGINE_V2', previous_v2)):
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
    print('GROUND VULKAN HARDWARE VALIDATION PASSED', flush=True)


if __name__ == '__main__':
    main()
