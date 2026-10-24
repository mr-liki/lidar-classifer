"""Read-only simplex microbenchmark; no GUI, LNT writes or PTD claims."""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gui.ground_ptd_cpp import native_find_simplex
from gui.ground_ptd_engine_v2 import _walk_2d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vertices', type=int, default=12000)
    ap.add_argument('--queries', type=int, default=80000)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    rng = np.random.default_rng(4607)
    vertices = np.ascontiguousarray(rng.uniform(0., 1600., (args.vertices, 2)) +
                                    [256000., 4837000.])
    query = np.ascontiguousarray(rng.uniform(0., 1600., (args.queries, 2)) +
                                 [256000., 4837000.])
    # Place known challenging boundary and outside-hull query cases.
    tri = Delaunay(vertices)
    if len(query) >= 12:
        query[:3] = vertices[:3]
        query[3:6] = vertices[tri.simplices[:3, :2]].mean(axis=1)
        query[6:9] = np.nextafter(query[3:6], np.inf)
        query[9:12] = np.nextafter(query[3:6], -np.inf)

    def scipy_query():
        return tri.find_simplex(query)

    def cpp_query():
        return native_find_simplex(tri, query)

    def numba_query():
        if _walk_2d is None:
            return scipy_query()
        out, success = _walk_2d(query, tri.neighbors, tri.equations,
                                tri.transform, tri.min_bound, tri.max_bound,
                                tri.paraboloid_scale, tri.paraboloid_shift)
        if not success:
            raise RuntimeError('incomplete Numba walk')
        return out

    actions = {'scipy_full_query': scipy_query,
               'numba_ordered_walk': numba_query,
               'cpp_ordered_walk': cpp_query}
    for name, action in actions.items():
        result = action()  # excludes JIT initialization and disk import
        np.testing.assert_array_equal(result, scipy_query(), err_msg=name)
    times = {name: [] for name in actions}
    for trial in range(args.runs):
        # Reverse the execution order on alternating runs to reduce bias.
        keys = list(actions) if trial % 2 == 0 else list(reversed(actions))
        for name in keys:
            started = perf_counter()
            result = actions[name]()
            elapsed = perf_counter() - started
            times[name].append(elapsed)
            np.testing.assert_array_equal(result, scipy_query(), err_msg=name)
    median = {key: float(statistics.median(value)) for key,value in times.items()}
    report = {
        'scope': 'single-triangulation isolated read-only point-location microbenchmark',
        'system': {'python': sys.version.split()[0], 'scipy': scipy.__version__,
                   'numpy': np.__version__},
        'vertices': args.vertices, 'queries': args.queries,
        'native_cpp_audit_parity': 'all query IDs matched complete SciPy',
        'median_seconds': median,
        'native_relative_to_scipy_speedup': median['scipy_full_query'] / median['cpp_ordered_walk'],
        'native_relative_to_numba_speedup': median['numba_ordered_walk'] / median['cpp_ordered_walk'],
        'raw_times': times,
        'full_cloud_speedup_proven': False,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
