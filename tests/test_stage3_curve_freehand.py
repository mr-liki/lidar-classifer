"""Stage 3 Curve/Freehand Vulkan input and geometry regression tests.

No C++ Vulkan DLL, PySide6 or VTK is required: the original VTK Catmull-Rom
method is parsed from the actual supplied source, and NativeDraw input
handlers are executed using the same lightweight Qt stubs as Stages 1 and 2.
This is NOT a Windows GPU interaction test.
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_stage1_parity import runtime as stage1_runtime, App, Canvas, Pos
from gui.native_stroke_geometry import (
    VTK_CURVE_SAMPLES_PER_SEGMENT,
    catmull_rom_vtk,
    drop_consecutive_duplicates,
    snap_endpoint_near_start,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def runtime():
    if 'gui.native_draw' not in sys.modules:
        return stage1_runtime.__wrapped__()
    draw = sys.modules['gui.native_draw']
    bridge = sys.modules['gui.native_selection_bridge']
    return bridge.Qt, bridge.QApplication, bridge, draw


@pytest.fixture
def draw(runtime):
    Qt, _, _, module = runtime
    d = module.NativeDraw.__new__(module.NativeDraw)
    d.canvas = Canvas()
    d.app = App()
    d.app.digitizer = SimpleNamespace(snap_enabled=False, snap_mode=None)
    d.app.snt_attachments = []
    d.drawings = []
    d.redo = []
    d.point_redo = []
    d._z_redo = []
    d._z_points = []
    d._object_actions = []
    d._object_redo = []
    d._freehand_before_drag = None
    d._freehand_undo_strokes = []
    d._freehand_redo_strokes = []
    d._freehand_last_snap_at = 0.0
    d._line_session_refs = []
    d._line_session_undone = []
    d._last_pointer_pos = None
    d._legacy_tool_dialog = None
    d._selection_bridge = None
    d._parallel_source = None
    d._selected_imported = None
    d._angle_sticky = False
    d._last_z = 0.0
    d._snap_target = None
    d._text_options = None
    d._hatch_dialog = None
    d._precision_dialog = None
    d._precision_adapter = None
    d.points = []
    d.hover = None
    d.dragging = False
    d.vertex_target = None
    d.tool = 'freehand'
    d.paints = 0
    d._paint = lambda: setattr(d, 'paints', d.paints + 1)
    d.active = lambda: True
    return d


def _original_vtk_curve_sampler():
    """Execute ONLY the two small geometry methods from original VTK source."""
    src = (ROOT / 'gui' / 'curve_tools.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    curve = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CurveTool')
    methods = [n for n in curve.body if isinstance(n, ast.FunctionDef)
               and n.name in ('_interpolate_catmull_rom', '_catmull_rom_point')]
    dummy = ast.ClassDef(name='VtkGeometry', bases=[], keywords=[],
                         body=methods, decorator_list=[])
    mod = ast.fix_missing_locations(ast.Module(body=[dummy], type_ignores=[]))
    scope = {'np': np}
    exec(compile(mod, 'original_vtk_curve_geometry', 'exec'), scope)
    return scope['VtkGeometry']()


def test_exact_vtk_curve_interpolation_for_open_three_point_curve():
    pts = [(0., 0.), (4., 8.), (8., 0.)]
    reference = _original_vtk_curve_sampler()._interpolate_catmull_rom(pts)
    sampled = catmull_rom_vtk(pts)
    assert len(sampled) == 2 * 20 + 1
    assert np.allclose(sampled, reference, rtol=0, atol=1e-12)
    assert sampled[0] == pts[0] and sampled[-1] == pts[-1]


def test_two_point_curve_has_vtk_interpolation_samples():
    pts = [(1., 4.), (15., 9.)]
    reference = _original_vtk_curve_sampler()._interpolate_catmull_rom(pts)
    assert len(catmull_rom_vtk(pts)) == 21
    assert np.allclose(catmull_rom_vtk(pts), reference, rtol=0, atol=1e-12)


def test_xyz_curve_matches_vtk_and_keeps_control_z():
    pts = [(4., 6., 18.), (7., 10., 21.), (11., 4., 19.), (14., 7., 18.)]
    reference = _original_vtk_curve_sampler()._interpolate_catmull_rom(pts)
    sampled = catmull_rom_vtk(pts)
    assert np.allclose(sampled, reference, rtol=0, atol=1e-12)
    assert sampled[0] == pts[0] and sampled[-1] == pts[-1]


def test_curve_loop_crossing_and_immutable_control_points():
    pts = [(0., 0.), (8., 9.), (0., 9.), (8., 0.), (0., 0.)]
    copy = list(pts)
    output = catmull_rom_vtk(pts)
    assert len(output) == 4 * VTK_CURVE_SAMPLES_PER_SEGMENT + 1
    assert output[0] == output[-1]
    assert pts == copy
    assert max(p[1] for p in output) > 0


def test_interpolation_does_not_expose_tessellation_as_control_points(runtime):
    _, _, _, module = runtime
    controls = [(0., 0.), (4., 7.), (8., 0.)]
    stored = {'type': 'curve', 'coords': list(controls)}
    assert module.NativeDraw._native_coords(stored) == catmull_rom_vtk(controls)
    assert stored['coords'] == controls


def test_curve_render_cache_is_bounded_and_reused(runtime):
    _, _, _, module = runtime
    module._catmull_rom_cached.cache_clear()
    coords = [(0., 0.), (1., 9.), (5., 0.)]
    assert module._catmull_rom(coords) == module._catmull_rom(coords)
    assert module._catmull_rom_cached.cache_info().hits >= 1
    assert module._catmull_rom_cached.cache_info().maxsize <= 16
    # Large curves must not become dozens of resident cached vertex arrays.
    before = module._catmull_rom_cached.cache_info().misses
    assert len(module._catmull_rom([(float(i), 0.) for i in range(129)])) == 2561
    assert module._catmull_rom_cached.cache_info().misses == before


def test_drop_only_consecutive_duplicate_controls():
    assert drop_consecutive_duplicates([(0, 0), (1, 1), (1, 1), (0, 0)]) == [
        (0, 0), (1, 1), (0, 0)]
    assert drop_consecutive_duplicates([(0., 0.), (0.000000001, 0.)]) == [
        (0., 0.), (0.000000001, 0.)]


def test_curve_closure_outranks_snt_and_is_exact(draw):
    draw.tool = 'curve'
    draw.points = [(0., 0.), (9., 0.), (9., 9.)]
    draw._near_scene = lambda *a: pytest.fail('Closure should not query SNT/native scene')
    assert draw._point(Pos(.1, .2)) == draw.points[0]
    assert draw._snap_target == draw.points[0]


def test_curve_does_not_snap_to_its_immediately_previous_draft_vertex(draw):
    draw.tool = 'curve'
    draw.points = [(0., 0.), (9., 0.)]
    assert draw._point(Pos(9.3, 0.2)) == pytest.approx((9.3, 0.2))
    assert draw._snap_target is None


def test_curve_can_snap_to_actual_line_endpoint(draw):
    draw.tool = 'curve'
    draw.drawings = [{'type': 'line', 'coords': [(8., 3.), (12., 4.)]}]
    draw.points = [(0., 0.)]
    assert draw._point(Pos(8.2, 3.2)) == (8., 3.)


def test_curve_samples_not_misidentified_as_keypoints(draw):
    draw.tool = 'curve'
    controls = [(0., 0.), (5., 8.), (10., 0.)]
    draw.drawings = [{'type': 'curve', 'coords': controls}]
    samples = catmull_rom_vtk(controls)
    middle = samples[7]
    assert all(math.dist(middle, c) > .05 for c in controls)
    assert draw._snap_at(middle, 'keypoint', .04) is None


def test_curve_preview_rubberband_does_not_move_committed_spline(draw):
    draw.tool = 'curve'
    draw.points = [(0., 0.), (5., 8.), (10., 0.)]
    draw._style = lambda tool: {'color': '#ee3322', 'width': 2}
    draw._sync_block_vectors = lambda: False
    draw._shutting_down = False
    draw.aux_shapes = []
    draw.overlay = SimpleNamespace(set_shapes=lambda shapes: setattr(draw, 'visible', shapes))
    draw.hover = (14., 9.)
    type(draw)._paint(draw)
    spline1 = next(x for x in draw.visible if x[0] == 'drawing_preview')
    guide1 = next(x for x in draw.visible if x[0] == 'drawing')
    assert len(spline1[1]) == 41
    assert guide1[1] == [(10., 0.), (14., 9.)]
    draw.hover = (20., -3.)
    type(draw)._paint(draw)
    spline2 = next(x for x in draw.visible if x[0] == 'drawing_preview')
    assert spline1[1] == spline2[1]
    assert next(x for x in draw.visible if x[0] == 'drawing')[1] == [
        (10., 0.), (20., -3.)]


def test_curve_commit_preserves_control_points_and_undo(draw):
    draw.tool = 'curve'
    draw.points = [(0., 0.), (7., 2.), (6., 6.), (.1, .1)]
    assert draw._finish()
    assert len(draw.drawings) == 1
    assert draw.drawings[-1]['coords'] == [(0., 0.), (7., 2.), (6., 6.), (0., 0.)]
    assert draw.drawings[-1]['type'] == 'curve'
    assert draw.history(redo=False)
    assert draw.drawings == []
    assert draw.history(redo=True)
    assert len(draw.drawings) == 1


def test_curve_completion_separates_successive_drafts(draw):
    draw.tool = 'curve'
    draw.points = [(0., 0.), (3., 5.), (5., 1.)]
    assert draw._finish()
    draw.points = [(20., 20.), (23., 25.)]
    assert draw._finish()
    assert len(draw.drawings) == 2
    assert draw.drawings[0]['coords'][-1] != draw.drawings[1]['coords'][0]


def test_freehand_release_pauses_and_resume_only_rightclick_commits(draw, runtime):
    Qt, _, _, _ = runtime
    draw.press(Pos(0., 0.), Qt.LeftButton)
    draw.move(Pos(3., 2.), Qt.LeftButton)
    draw.move(Pos(5., 6.), Qt.LeftButton)
    assert draw.release(Pos(8., 5.), Qt.LeftButton)
    assert draw.dragging is False and draw.drawings == []
    saved = list(draw.points)
    draw.move(Pos(500., 500.), Qt.NoButton)
    assert draw.hover is None and draw.points == saved
    draw.press(Pos(8., 5.), Qt.LeftButton)
    draw.move(Pos(2., 7.), Qt.LeftButton)
    draw.release(Pos(.2, .1), Qt.LeftButton)
    assert draw.drawings == []
    assert draw.press(Pos(1000., 1000.), Qt.RightButton)
    assert len(draw.drawings) == 1
    assert draw.drawings[0]['coords'][0] == draw.drawings[0]['coords'][-1]
    assert draw.points == []
    assert draw.tool == 'freehand'


def test_freehand_samples_are_never_snapped_on_mousemove(draw, runtime):
    Qt, _, _, _ = runtime
    raw = [(0., 0.), (5., 2.), (10., 5.), (15., 3.)]
    seen = []
    draw.press(Pos(*raw[0]), Qt.LeftButton)
    draw._snap_at = lambda p, mode, tol: seen.append(tuple(p)) or (20., 20.)
    for p in raw[1:]:
        draw.move(Pos(*p), Qt.LeftButton)
    assert draw.points == raw
    assert (5., 2.) in draw.points and (10., 5.) in draw.points
    draw.release(Pos(*raw[-1]), Qt.LeftButton)
    assert draw.points == raw
    assert draw._snap_target == (20., 20.)
    draw.press(Pos(100., 100.), Qt.RightButton)
    assert draw.drawings[-1]['coords'][-2] == (20., 20.)
    assert draw.drawings[-1]['coords'][1:-2] == raw[1:-1]


def test_freehand_snap_preview_is_throttled_for_large_stroke(draw, runtime, monkeypatch):
    Qt, _, _, module = runtime
    clock = [100.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    calls = []
    draw._snap_at = lambda *args: calls.append(args) or None
    draw.press(Pos(0., 0.), Qt.LeftButton)
    for i in range(2, 201):
        clock[0] += .001
        draw.move(Pos(i, math.sin(i)*5 + 20), Qt.LeftButton)
    # 200 raw samples but under 6 bounded scene-index queries.
    assert len(calls) <= 6
    assert len(draw.points) > 150
    draw.release(Pos(201, 20), Qt.LeftButton)
    draw.press(Pos(50, 50), Qt.RightButton)
    assert len(draw.drawings) == 1


def test_freehand_exact_snap_to_line_endpoint(draw, runtime):
    Qt, _, _, _ = runtime
    draw.drawings = [{'type':'line', 'coords':[(50., 10.), (70., 10.)]}]
    draw.press(Pos(0., 0.), Qt.LeftButton)
    for p in [(5., 5.), (25., 20.), (50.4, 10.2)]:
        draw.move(Pos(*p), Qt.LeftButton)
    draw.release(Pos(50.4, 10.2), Qt.LeftButton)
    draw.press(Pos(1000, 1000), Qt.RightButton)
    last = draw.drawings[-1]['coords']
    assert last[-2] == (50., 10.)
    assert last[-1] == last[0]
    assert last[1:-2] == [(5., 5.), (25., 20.)]


def test_freehand_self_closure_wins_over_external_target(draw, runtime):
    Qt, _, _, _ = runtime
    draw.press(Pos(0., 0.), Qt.LeftButton)
    for p in [(7., 0.), (7., 6.), (.2, .3)]:
        draw.move(Pos(*p), Qt.LeftButton)
    draw.release(Pos(.2, .3), Qt.LeftButton)
    draw._snap_at = lambda *args: pytest.fail('Own closure outranks scene geometry')
    draw.press(Pos(1000, 1000), Qt.RightButton)
    assert draw.drawings[-1]['coords'][-1] == draw.drawings[-1]['coords'][0]
    assert draw.drawings[-1]['coords'] == [
        (0., 0.), (7., 0.), (7., 6.), (0., 0.)]


def test_freehand_undo_and_redo_paused_stroke_then_committed_object(draw, runtime):
    Qt, _, _, _ = runtime
    draw.press(Pos(0., 0.), Qt.LeftButton)
    draw.move(Pos(5., 0.), Qt.LeftButton)
    draw.release(Pos(7., 0.), Qt.LeftButton)
    first = list(draw.points)
    draw.press(Pos(7., 0.), Qt.LeftButton)
    draw.move(Pos(8., 8.), Qt.LeftButton)
    draw.release(Pos(9., 8.), Qt.LeftButton)
    second = list(draw.points)
    assert draw.history(redo=False)
    assert draw.points == first
    assert draw.drawings == []
    assert draw.history(redo=True)
    assert draw.points == second
    draw.press(Pos(300, 300), Qt.RightButton)
    assert len(draw.drawings) == 1
    assert draw.history(redo=False)
    assert draw.drawings == []
    assert draw.history(redo=True)
    assert len(draw.drawings) == 1


def test_freehand_escape_cancels_draft_without_mutating_old(draw, runtime):
    Qt, _, _, _ = runtime
    existing = {'type': 'line', 'coords': [(100., 0.), (101., 0.)]}
    draw.drawings = [existing]
    draw._close_palettes = lambda: None
    draw._store_selection = lambda _: None
    draw.press(Pos(0., 0.), Qt.LeftButton)
    draw.move(Pos(3., 4.), Qt.LeftButton)
    draw.release(Pos(4., 5.), Qt.LeftButton)
    assert draw.key(Qt.Key_Escape)
    assert draw.drawings == [existing]
    assert draw.tool is None


def test_short_freehand_remains_open_and_does_not_make_degenerate_polygon(draw, runtime):
    Qt, _, _, _ = runtime
    draw.press(Pos(0., 0.), Qt.LeftButton)
    draw.move(Pos(3., 5.), Qt.LeftButton)
    draw.release(Pos(3., 5.), Qt.LeftButton)
    draw.press(Pos(100, 100), Qt.RightButton)
    coords = draw.drawings[-1]['coords']
    assert coords == [(0., 0.), (3., 5.)]


def test_original_vtk_curve_source_restored_verbatim():
    original_zip = ROOT.parent / 'baseline' / 'gui' / 'curve_tools.py'
    if original_zip.exists():
        assert (ROOT / 'gui' / 'curve_tools.py').read_bytes() == original_zip.read_bytes()
    # This guards the original VTK handler even on user's Windows install.
    source = (ROOT / 'gui' / 'curve_tools.py').read_text(encoding='utf-8')
    assert 'from gui.native_selection_targets import snap_draft_closure' not in source
    assert 'points_per_segment = 20' in source
