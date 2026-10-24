"""Stage 3 proposed grid: original-order, full-ID/classification parity first.

Run with a freshly compiled native Ground DLL. The tests do not need Qt.
"""
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from scipy.spatial import Delaunay

from gui import ground_ptd_cpp as cpp
from gui import ground_ptd_engine_v2 as v2


@pytest.fixture(scope='module', autouse=True)
def native_binary():
    if os.environ.get('NAKSHA_GROUND_CPP_LIBRARY'):
        return
    root = Path(__file__).resolve().parents[1] / 'native' / 'ground_ptd' / 'build'
    choices = [root / 'libnaksha_ground_ptd_exact.so',
               root / 'libnaksha_ground_ptd_exact.dylib',
               root / 'Release' / 'naksha_ground_ptd_exact.dll']
    dll = next((p for p in choices if p.is_file()), None)
    if dll is None:
        pytest.skip('Build Stage 3 native Ground library first')
    os.environ['NAKSHA_GROUND_CPP_LIBRARY'] = str(dll)


@pytest.mark.parametrize('kind', ['random', 'grid', 'thin', 'near_collinear', 'duplicate'])
@pytest.mark.parametrize('offset', [(0., 0.), (256000., 4837000.)])
def test_grid_matches_full_scipy_all_orderings(kind, offset, monkeypatch):
    monkeypatch.setenv('NAKSHA_GROUND_CPP_GRID', '1')
    rng = np.random.default_rng(1024)
    xy = rng.random((250, 2))
    if kind == 'grid':
        xx, yy = np.meshgrid(np.arange(17.) / 16, np.arange(17.) / 16)
        xy = np.c_[xx.ravel(), yy.ravel()]
    elif kind == 'thin':
        xy[:, 1] *= 1e-4
    elif kind == 'near_collinear':
        xy[:, 1] = xy[:, 0] / 4 + xy[:, 1] * 1e-7
    elif kind == 'duplicate':
        xy = np.r_[xy, xy[:40], xy[:5]]
    xy += offset
    tri = Delaunay(xy)
    centers = tri.points[tri.simplices].mean(axis=1)
    edge = tri.points[tri.simplices[:, :2]].mean(axis=1)
    query = np.r_[xy, centers, edge, np.nextafter(edge, np.inf),
                  np.nextafter(edge, -np.inf),
                  xy.min(axis=0) + rng.uniform(-0.15, 1.15, (1500, 2)) * np.ptp(xy, axis=0)]
    for idx in (np.arange(len(query)), rng.permutation(len(query)),
                np.arange(len(query))[::-1]):
        ordered = np.ascontiguousarray(query[idx], dtype=np.float64)
        expected = tri.find_simplex(ordered)
        stats = {}
        result = cpp.native_find_simplex(tri, ordered, stats)
        np.testing.assert_array_equal(result, expected)
        assert stats['grid_queries'] == len(ordered)
        # Exact fallback is expected when near-degenerate/boundary geometry
        # cannot be certified sufficiently far from a facet edge.
        assert stats['grid_fast_hits'] >= 0
        assert 0 <= stats['grid_ordered_fallbacks'] <= len(ordered)


def test_legacy_kernel_still_matches_after_grid_build(monkeypatch):
    rng = np.random.default_rng(94)
    tri = Delaunay(rng.random((350, 2)))
    query = np.ascontiguousarray(rng.random((1200, 2)))
    monkeypatch.setenv('NAKSHA_GROUND_CPP_GRID', '0')
    legacy = cpp.native_find_simplex(tri, query)
    monkeypatch.setenv('NAKSHA_GROUND_CPP_GRID', '1')
    updated = cpp.native_find_simplex(tri, query)
    np.testing.assert_array_equal(legacy, updated)
    np.testing.assert_array_equal(updated, tri.find_simplex(query))


def test_missing_grid_symbol_runs_original_cpp(monkeypatch):
    rng = np.random.default_rng(943)
    tri = Delaunay(rng.random((120, 2)))
    query = np.ascontiguousarray(rng.random((1000, 2)))
    monkeypatch.setenv('NAKSHA_GROUND_CPP_GRID', '1')
    monkeypatch.setattr(cpp, '_load_grid_function',
                        lambda path: (_ for _ in ()).throw(AttributeError('old DLL')))
    stats = {}
    result = cpp.native_find_simplex(tri, query, stats)
    np.testing.assert_array_equal(result, tri.find_simplex(query))
    assert stats['grid_fallback'] == 'AttributeError'
    assert stats.get('grid_queries', 0) == 0


def test_v2_audit_exact_full_query_and_reports_grid(monkeypatch):
    rng = np.random.default_rng(9348)
    tri = Delaunay(rng.random((200, 2)))
    query = np.ascontiguousarray(rng.random((1200, 2)))
    monkeypatch.setenv('NAKSHA_GROUND_CPP', '1')
    monkeypatch.setenv('NAKSHA_GROUND_CPP_GRID', '1')
    monkeypatch.setattr(v2.scipy, '__version__', '1.15.3')
    stats = {}
    result = v2.find_simplex(tri, query, stats=stats, verify=True)
    np.testing.assert_array_equal(result, tri.find_simplex(query))
    assert stats['audited_queries'] == len(query)
    assert stats['native_cpp_queries'] == len(query)
    assert stats['grid_queries'] == len(query)


def test_stage3_grid_preserves_input_buffers(monkeypatch):
    rng = np.random.default_rng(2026)
    tri = Delaunay(rng.random((150, 2)))
    query = np.ascontiguousarray(rng.random((1500, 2)))
    original = [a.copy() for a in (tri.simplices, tri.points, tri.neighbors,
                                    tri.equations, tri.transform, query)]
    monkeypatch.setenv('NAKSHA_GROUND_CPP_GRID', '1')
    cpp.native_find_simplex(tri, query)
    for before, now in zip(original,(tri.simplices, tri.points, tri.neighbors,
                                      tri.equations, tri.transform, query)):
        np.testing.assert_array_equal(before, now)
