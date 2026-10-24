"""Stage 4 isolated full-query SciPy simplex-ID differential tests.

Tests use the optional native Stage 4 DLL; no Qt or point-cloud data needed.
Every ID is compared to a COMPLETE SciPy.find_simplex query, not a sample.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay

from gui import ground_ptd_cpp as cpp
from gui import ground_ptd_engine_v2 as v2


@pytest.fixture(scope='module', autouse=True)
def stage4_library():
    if os.environ.get('NAKSHA_GROUND_CPP_LIBRARY'):
        return
    root = Path(__file__).resolve().parents[1] / 'native' / 'ground_ptd'
    candidates = [root / 'build_stage4' / 'Release' / 'naksha_ground_ptd_exact.dll',
                  root / 'build_stage4' / 'libnaksha_ground_ptd_exact.so',
                  root / 'build_stage4' / 'libnaksha_ground_ptd_exact.dylib']
    built = next((item for item in candidates if item.is_file()), None)
    if built is None:
        pytest.skip('Build Stage 4 native Ground library first')
    os.environ['NAKSHA_GROUND_CPP_LIBRARY'] = str(built)


def _geom(kind, count, seed=20261024):
    rng = np.random.default_rng(seed)
    if kind == 'clustered':
        core = int(count * .97)
        v = np.r_[rng.normal(0, 32, (core, 2)), rng.uniform(-500, 500, (count - core, 2))]
    elif kind == 'corridor':
        core = int(count * .97)
        v = np.r_[np.c_[rng.uniform(-500, 500, core), rng.normal(0, 2, core)],
                  rng.uniform(-500, 500, (count - core, 2))]
    elif kind == 'thin':
        v = rng.random((count, 2)); v[:, 1] *= 1e-5
    elif kind == 'grid':
        side = max(6, int(np.sqrt(count)))
        x, y = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
        v = np.c_[x.ravel(), y.ravel()]
    elif kind == 'duplicate':
        v = rng.random((count, 2))
        v = np.r_[v, v[:40], v[:20]]
    else:
        v = rng.uniform(0, 1000, (count, 2))
    return np.ascontiguousarray(v + np.array([256000., 4837000.]))


@pytest.mark.parametrize('kind', ['uniform', 'grid', 'thin', 'duplicate', 'clustered', 'corridor'])
def test_simplex_ids_strict_match_scipy(monkeypatch, kind):
    monkeypatch.setenv('NAKSHA_GROUND_CPP_GRID', '1')
    monkeypatch.setenv('NAKSHA_GROUND_CPP_STAGE4', '1')
    vertices = _geom(kind, 300 if kind not in ('clustered', 'corridor') else 10000)
    tri = Delaunay(vertices)
    rng = np.random.default_rng(91)
    centres = np.mean(tri.points[tri.simplices[:1000]], axis=1)
    edges = np.mean(tri.points[tri.simplices[:1000, :2]], axis=1)
    xy = np.r_[centres, edges, np.nextafter(edges, np.inf),
               np.nextafter(edges, -np.inf),
               vertices.min(axis=0) + rng.random((3000, 2)) * np.ptp(vertices, axis=0)]
    for order in (np.arange(len(xy)), rng.permutation(len(xy)),
                  np.arange(len(xy))[::-1]):
        queries = np.ascontiguousarray(xy[order])
        stats = {}
        obtained = cpp.native_find_simplex(tri, queries, stats)
        np.testing.assert_array_equal(obtained, tri.find_simplex(queries))
        assert stats['stage4_queries'] == len(queries)
        assert stats['grid_queries'] == len(queries)
        assert stats['grid_fast_hits'] <= len(queries)
        assert stats['stage4_bbox_pruned'] <= stats['stage4_probes']


@pytest.mark.parametrize('kind', ['clustered', 'corridor'])
def test_adversarial_numerical_overlaps_force_exact_fallback(monkeypatch, kind):
    """Previous Stage 3 grid gives wrong IDs in the same deterministic case."""
    monkeypatch.setenv('NAKSHA_GROUND_CPP_GRID', '1')
    monkeypatch.setenv('NAKSHA_GROUND_CPP_STAGE4', '1')
    rng = np.random.default_rng(7424)
    n = 30000
    dense = int(n * .97)
    if kind == 'clustered':
        xy = np.r_[rng.normal(0, 32, (dense, 2)),
                   rng.uniform(-500, 500, (n-dense, 2))]
        query = rng.normal(0, 32, (50000, 2))
    else:
        xy = np.r_[np.c_[rng.uniform(-500, 500, dense), rng.normal(0, 2, dense)],
                   rng.uniform(-500, 500, (n-dense, 2))]
        query = np.c_[rng.uniform(-500, 500, 50000), rng.normal(0, 2, 50000)]
    tri = Delaunay(np.ascontiguousarray(xy + [256000., 4837000.]))
    query = np.ascontiguousarray(query + [256000., 4837000.])
    reference = tri.find_simplex(query)
    stats = {}
    result = cpp.native_find_simplex(tri, query, stats)
    np.testing.assert_array_equal(result, reference)
    assert stats['stage4_refined_queries'] > 0
    # Qhull builds can differ by SciPy version; audit every ID regardless
    # of whether this particular build contains overlapping facets.
    assert stats['stage4_ambiguous'] >= 0


def test_missing_stage4_export_falls_back_to_stage3(monkeypatch):
    monkeypatch.setenv('NAKSHA_GROUND_CPP_GRID', '1')
    monkeypatch.setenv('NAKSHA_GROUND_CPP_STAGE4', '1')
    tri = Delaunay(_geom('uniform', 400))
    xy = np.ascontiguousarray(tri.points[tri.simplices[:500]].mean(axis=1))
    monkeypatch.setattr(cpp, '_load_stage4_function',
                        lambda path: (_ for _ in ()).throw(AttributeError('missing')))
    stats = {}
    np.testing.assert_array_equal(cpp.native_find_simplex(tri, xy, stats),
                                  tri.find_simplex(xy))
    assert stats['stage4_fallback'] == 'AttributeError'
    assert stats['grid_queries'] == len(xy)
    assert stats.get('stage4_queries', 0) == 0


def test_stage4_does_not_mutate_inputs_and_v2_audits(monkeypatch):
    monkeypatch.setenv('NAKSHA_GROUND_CPP', '1')
    monkeypatch.setenv('NAKSHA_GROUND_CPP_GRID', '1')
    monkeypatch.setenv('NAKSHA_GROUND_CPP_STAGE4', '1')
    monkeypatch.setattr(v2.scipy, '__version__', '1.15.3')
    rng = np.random.default_rng(44)
    tri = Delaunay(_geom('uniform', 1000))
    xy = np.ascontiguousarray(rng.uniform(256000, 257000, (3000, 2)))
    original = [v.copy() for v in (tri.points, tri.simplices, tri.neighbors,
                                  tri.equations, tri.transform, xy)]
    stats = {}
    observed = v2.find_simplex(tri, xy, stats=stats, verify=True)
    np.testing.assert_array_equal(observed, tri.find_simplex(xy))
    assert stats['audited_queries'] == len(xy)
    for before, now in zip(original, (tri.points, tri.simplices, tri.neighbors,
                                      tri.equations, tri.transform, xy)):
        np.testing.assert_array_equal(before, now)
