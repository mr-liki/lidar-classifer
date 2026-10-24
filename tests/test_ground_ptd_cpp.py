"""Standalone native PTD parity tests. No Qt/viewer or real input data needed."""
from pathlib import Path
import os

import numpy as np
import pytest
from scipy.spatial import Delaunay

from gui import ground_ptd_cpp
from gui import ground_ptd_engine_v2 as v2


@pytest.fixture(scope='module', autouse=True)
def native_library():
    if os.environ.get('NAKSHA_GROUND_CPP_LIBRARY'):
        return
    path = Path(__file__).resolve().parents[1] / 'native' / 'ground_ptd' / 'build'
    candidates = [path / 'libnaksha_ground_ptd_exact.so',
                  path / 'libnaksha_ground_ptd_exact.dylib',
                  path / 'Release' / 'naksha_ground_ptd_exact.dll',
                  path / 'naksha_ground_ptd_exact.dll']
    built = next((candidate for candidate in candidates if candidate.exists()), None)
    if built is None:
        pytest.skip('native PTD library not yet built')
    os.environ['NAKSHA_GROUND_CPP_LIBRARY'] = str(built)


@pytest.mark.parametrize('geometry', ['random', 'grid', 'thin', 'collinear', 'duplicate'])
@pytest.mark.parametrize('offset', [(0., 0.), (256000., 4837000.)])
def test_exact_simplex_identity_on_adversarial_geometry(geometry, offset):
    rng = np.random.default_rng(841)
    vertices = rng.random((120, 2))
    if geometry == 'grid':
        x, y = np.meshgrid(np.arange(10.) / 9, np.arange(10.) / 9)
        vertices = np.c_[x.ravel(), y.ravel()]
    elif geometry == 'thin':
        vertices[:, 1] *= 1e-6
    elif geometry == 'collinear':
        vertices[:, 1] = vertices[:, 0] * .25 + vertices[:, 1] * 1e-7
    elif geometry == 'duplicate':
        vertices = np.r_[vertices, vertices[:15], vertices[:5]]
    vertices += offset
    tri = Delaunay(vertices)
    edges = vertices[tri.simplices[:, :2]].mean(axis=1)
    centers = vertices[tri.simplices].mean(axis=1)
    random = vertices.min(0) + rng.uniform(-.1, 1.1, (1500, 2)) * np.ptp(vertices, axis=0)
    queries = np.r_[vertices, edges, np.nextafter(edges, np.inf),
                    np.nextafter(edges, -np.inf), centers, random]
    for order in (np.arange(len(queries)), rng.permutation(len(queries)),
                  np.arange(len(queries))[::-1]):
        xy = np.ascontiguousarray(queries[order], dtype=np.float64)
        expected = tri.find_simplex(xy)
        actual = ground_ptd_cpp.native_find_simplex(tri, xy)
        np.testing.assert_array_equal(actual, expected)


def test_cpp_is_opt_in_and_full_scipy_audit_can_reject_mismatch(monkeypatch):
    tri = Delaunay(np.array([[0., 0.], [1., 0.], [0., 1.]]))
    xy = np.ascontiguousarray([[0.2, 0.2], [0.7, 0.7], [-0.1, 0.3]])
    # Exercise integration independent of our Linux SciPy development version;
    # the shipped integration is restricted to the approved 1.15.3 runtime.
    monkeypatch.setattr(v2.scipy, '__version__', '1.15.3')
    monkeypatch.setenv('NAKSHA_GROUND_CPP', '1')
    stats = {}
    np.testing.assert_array_equal(v2.find_simplex(tri, xy, verify=True, stats=stats),
                                  tri.find_simplex(xy))
    assert stats['native_cpp_queries'] == len(xy)
    assert stats['audited_queries'] == len(xy)
    monkeypatch.setattr(ground_ptd_cpp, 'native_find_simplex',
                        lambda *args: np.zeros(len(xy), dtype=np.int32))
    stats = {}
    np.testing.assert_array_equal(v2.find_simplex(tri, xy, verify=True, stats=stats),
                                  tri.find_simplex(xy))
    assert stats['fallback'] == 'complete-query equivalence failure'


def test_bad_binary_uses_existing_compiled_fallback(monkeypatch):
    rng = np.random.default_rng(100)
    tri = Delaunay(rng.random((100, 2)))
    xy = np.ascontiguousarray(rng.random((200, 2)))
    monkeypatch.setattr(v2.scipy, '__version__', '1.15.3')
    monkeypatch.setenv('NAKSHA_GROUND_CPP', '1')
    monkeypatch.setenv('NAKSHA_GROUND_CPP_LIBRARY', '/does/not/exist.so')
    stats = {}
    np.testing.assert_array_equal(v2.find_simplex(tri, xy, verify=True, stats=stats),
                                  tri.find_simplex(xy))
    assert stats.get('native_cpp_fallback') == 'FileNotFoundError'


def test_no_native_writes_and_reject_unsupported_layout():
    tri = Delaunay(np.array([[0., 0.], [1., 0.], [0., 1.]]))
    xy = np.ascontiguousarray([[0.2, 0.2], [0.4, 0.1]])
    neighbors, transform, equations = tri.neighbors.copy(), tri.transform.copy(), tri.equations.copy()
    result = ground_ptd_cpp.native_find_simplex(tri, xy)
    np.testing.assert_array_equal(result, tri.find_simplex(xy))
    np.testing.assert_array_equal(tri.neighbors, neighbors)
    np.testing.assert_array_equal(tri.transform, transform)
    np.testing.assert_array_equal(tri.equations, equations)
    with pytest.raises(ValueError):
        ground_ptd_cpp.native_find_simplex(tri, xy.astype(np.float32))
