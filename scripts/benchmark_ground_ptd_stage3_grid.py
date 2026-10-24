"""Run a repeatable in-memory benchmark comparing Stage 1 and Stage 3 C++.

Do not extrapolate kernel-only speedups to a complete 47M-point Ground run.
"""
import argparse
import os
from statistics import median
from time import perf_counter

import numpy as np
from scipy.spatial import Delaunay

from gui.ground_ptd_cpp import native_find_simplex


def bench(vertices, queries, runs):
    rng = np.random.default_rng(7424)
    vx = np.column_stack([rng.uniform(0, 1000, vertices),
                          rng.uniform(0, 1000, vertices)])
    tri = Delaunay(vx)
    q = np.ascontiguousarray(rng.uniform(-100, 1100, (queries, 2)), dtype=np.float64)
    expected = tri.find_simplex(q)
    results = {}
    for name, flag in [('original_cpp', '0'), ('grid_cpp', '1')]:
        os.environ['NAKSHA_GROUND_CPP_GRID'] = flag
        times=[]
        for _ in range(runs):
            stats = {}
            started=perf_counter()
            result=native_find_simplex(tri, q, stats)
            times.append(perf_counter()-started)
            if not np.array_equal(result, expected):
                raise AssertionError(f'{name} different from SciPy')
        results[name] = median(times)
        if flag == '1':
            results.update(stats)
    results['speedup'] = results['original_cpp']/results['grid_cpp']
    print(f'TIN verts={vertices:,} triangles={len(tri.simplices):,} '
          f'queries={queries:,} runs={runs}')
    for k,v in results.items():
        print(f'{k}: {v:.6f}' if isinstance(v,float) else f'{k}: {v:,}')
    print('Exact simplex IDs match full SciPy:',True)
    return results


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--vertices',type=int,default=12000)
    parser.add_argument('--queries',type=int,default=80000)
    parser.add_argument('--runs',type=int,default=3)
    a=parser.parse_args()
    bench(a.vertices,a.queries,a.runs)
