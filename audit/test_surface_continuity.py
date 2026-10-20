"""Exercise the real geometry backends without importing the Qt/VTK UI."""
import ast
from collections import OrderedDict
from pathlib import Path
import os
import time

import numpy as np
import pytest
from scipy.spatial import Delaunay


def load_functions(filename, names, extra=None):
    path = Path(__file__).resolve().parents[1] / "gui" / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    scope = dict(np=np, time=time, os=os, OrderedDict=OrderedDict,
                 HAS_NUMBA=False, HAS_TRIANGLE=False, Delaunay=Delaunay)
    scope.update(extra or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def sparse_scan():
    x, y = np.meshgrid(np.arange(33), np.arange(33))
    keep = ~((x > 7) & (x < 25) & (y > 7) & (y < 25))
    xy = np.column_stack((x[keep], y[keep])).astype(float)
    # Include a valid thin triangle that a shape-based filter would remove.
    xy = np.vstack((xy, [0.0001, 0.00001]))
    return np.column_stack((xy, .1 * xy[:, 0] + .2 * xy[:, 1]))


def area(points, faces):
    tri = points[faces, :2]
    a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    return np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() / 2


@pytest.mark.parametrize("quality", ["fast", "normal", "slow"])
@pytest.mark.parametrize("visible", [{2}, {2, 5}])
def test_shading_covers_sparse_interior(monkeypatch, quality, visible):
    monkeypatch.delenv("NAKSHA_SHADING_FEATURE_AWARE", raising=False)
    xyz = sparse_scan()
    scope = load_functions("shading_display.py", [
        "_compute_shading_geometry_backend", "_do_triangulate",
        "triangulate_scipy_direct", "_filter_edges_by_absolute",
        "_feature_aware_filter_faces"], {
        "_ADAPTIVE_SHADING_VERSION": 1, "_FEATURE_AWARE_VERSION": 2,
        "_SHADING_LINE_FILTER_VERSION": 2,
        "_compute_face_normals": lambda pts, faces: np.zeros((len(faces), 3)),
        "_compute_face_shade": lambda pts, faces, *args, **kwargs: np.ones(len(faces)),
    })
    seed = dict(offset=np.zeros(3), unique_indices=np.arange(len(xyz)),
                xyz_unique=xyz, spacing=1.)
    build = scope["_compute_shading_geometry_backend"]
    def run(edge):
        return build(xyz, np.full(len(xyz), 2), visible, 315, 45, .25, 3.,
                     edge, "fixture", representative_seed=seed, quality_mode=quality)
    result = run(None)
    assert area(result["xyz_unique"], result["faces"]) == pytest.approx(1024.)
    assert len(result["faces"]) == len(Delaunay(xyz[:, :2]).simplices)
    # An explicit limit still permits intentional gaps.
    limited = run(2.)
    assert area(limited["xyz_unique"], limited["faces"]) < 1024.


@pytest.mark.parametrize("quality", ["fast", "normal", "slow"])
def test_surface_covers_sparse_interior(quality):
    xyz = sparse_scan()
    scope = load_functions("surface_mode.py", [
        "_compute_surface_geometry_backend", "_filter_long_edges"], {
        "_SURFACE_HAS_NUMBA": False,
        "normalize_surface_quality": lambda q: q,
        "_surface_quality_target": lambda q, n: n,
        "_triangulate_surface_xy": lambda xy: (Delaunay(xy).simplices, "scipy"),
        "_compute_surface_face_colors": lambda pts, faces, *args, **kw: np.zeros((len(faces), 3)),
    })
    build = scope["_compute_surface_geometry_backend"]
    result = build(xyz, np.arange(len(xyz)), 0, len(xyz), 0, 315, 45, .25,
                   None, quality_mode=quality)
    assert area(result["points"], result["faces"]) == pytest.approx(1024.)


@pytest.mark.parametrize("quality", ["fast", "normal", "slow"])
def test_native_section_backends_cover_sparse_interior(monkeypatch, quality):
    from gui import section_shading_math as shading
    from gui import section_surface_math as surface
    monkeypatch.delenv("NAKSHA_SHADING_FEATURE_AWARE", raising=False)
    xyz = sparse_scan()
    seed = dict(offset=np.zeros(3), unique_indices=np.arange(len(xyz)),
                xyz_unique=xyz, spacing=1.)
    result = shading._compute_shading_geometry_backend(
        xyz, np.full(len(xyz), 2), {2}, 315, 45, .25, 3., None,
        "fixture", representative_seed=seed, quality_mode=quality)
    assert area(result["xyz_unique"], result["faces"]) == pytest.approx(1024.)
    result = surface._compute_surface_geometry_backend(
        xyz, np.arange(len(xyz)), 0, len(xyz), 0, 315, 45, .25,
        None, quality_mode=quality)
    assert area(result["points"], result["faces"]) == pytest.approx(1024.)
