"""Stage3-vs-Stage4 native triangle lookup benchmark; not full PTD.

Reports the exact SciPy match count for BOTH engines. Some near-degenerate
triangulations can expose Stage 3 mismatches, so faster but incorrect output
must NEVER be described as an acceptable result.
"""
from __future__ import annotations

import argparse
import os
from statistics import median
from time import perf_counter

import numpy as np
from scipy.spatial import Delaunay

from gui.ground_ptd_cpp import native_find_simplex


def bench(vertices=30000, queries=50000, runs=3, distribution='clustered'):
    rng = np.random.default_rng(7424)
    n = int(vertices)
    qn = int(queries)
    dense = int(n * .97)
    if distribution == 'clustered':
        vx = np.r_[rng.normal(0, 32, (dense, 2)),
                   rng.uniform(-500, 500, (n-dense, 2))]
        query = rng.normal(0, 32, (qn, 2))
    elif distribution == 'corridor':
        vx = np.r_[np.c_[rng.uniform(-500, 500, dense), rng.normal(0, 2, dense)],
                   rng.uniform(-500, 500, (n-dense, 2))]
        query = np.c_[rng.uniform(-500, 500, qn), rng.normal(0, 2, qn)]
    else:
        vx = rng.uniform(0, 1000, (n, 2))
        query = rng.uniform(0, 1000, (qn, 2))
    vertices_xy = np.ascontiguousarray(vx + [256000., 4837000.], dtype=np.float64)
    query_xy = np.ascontiguousarray(query + [256000., 4837000.], dtype=np.float64)
    started = perf_counter()
    tri = Delaunay(vertices_xy)
    build_time = perf_counter() - started
    expected = tri.find_simplex(query_xy)
    print(f'distribution={distribution} verts={n:,} triangles={len(tri.simplices):,} '
          f'queries={qn:,} runs={runs} original_TIN={build_time:.3f}s')
    timings = {}
    for name, flag in (('stage3', '0'), ('stage4', '1')):
        os.environ['NAKSHA_GROUND_CPP_GRID'] = '1'
        os.environ['NAKSHA_GROUND_CPP_STAGE4'] = flag
        samples = []
        stats = {}
        for _ in range(int(runs)):
            started = perf_counter()
            result = native_find_simplex(tri, query_xy, stats)
            samples.append(perf_counter() - started)
            if flag == '1' and not np.array_equal(result, expected):
                mismatch = int(np.count_nonzero(result != expected))
                raise AssertionError(f'Stage 4 {mismatch} simplex IDs DIFFER from full SciPy')
        mismatch = int(np.count_nonzero(result != expected))
        timings[name] = median(samples)
        print(f'{name}: median={timings[name]:.6f}s SciPy_mismatches={mismatch:,}')
        if flag == '1':
            print('Stage4 adaptive index metrics:', stats)
    print('Stage4 lookup speedup over Stage3: {:.3f}x'.format(
        timings['stage3'] / max(timings['stage4'], 1e-9)))
    print('Stage4 exact SciPy simplex IDs: YES')
    print('Note: this is not whole-fence or complete-Ground performance.')
    return timings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vertices', type=int, default=30000)
    parser.add_argument('--queries', type=int, default=50000)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--distribution', choices=('uniform', 'clustered', 'corridor'),
                        default='clustered')
    options = parser.parse_args()
    bench(options.vertices, options.queries, options.runs, options.distribution)
