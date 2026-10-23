"""Center Snap selects a closed Draw figure by its interior, not pixel distance."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPointF

from gui.native_draw import NativeDraw
from gui.native_vector_pick import polygon_centroid


@pytest.fixture
def draw():
    draw = NativeDraw.__new__(NativeDraw)
    draw.app = SimpleNamespace(
        digitizer=SimpleNamespace(snap_enabled=True, snap_mode='center'),
        snt_attachments=[])
    draw.canvas = SimpleNamespace(section_at=lambda x, y: (x, y), world_per_pixel=lambda: .01)
    draw.drawings = []
    draw.tool = 'line'
    draw.points = []
    draw._line_session_refs = []
    draw._snap_target = None
    return draw


@pytest.mark.parametrize('kind', ['polyline', 'ortho', 'freehand', 'hatch', 'curve'])
def test_closed_draw_figures_snap_from_far_inside(draw, kind):
    draw.drawings = [dict(type=kind, coords=[(0., 0.), (100., 0.),
                                            (100., 100.), (0., 100.), (0., 0.)])]
    expected = polygon_centroid(draw._native_coords(draw.drawings[0]))
    for click in [(10., 10.), (80., 80.), (50., 50.), (10., 10.)]:
        assert draw._point(QPointF(*click)) == pytest.approx(expected)
        assert draw._snap_target == pytest.approx(expected)


@pytest.mark.parametrize('tool', ['line', 'smartline', 'polyline', 'rectangle',
    'circle', 'freehand', 'curve', 'text', 'point', 'vertex', 'movevertex', 'ortho', 'accudraw'])
def test_explicit_center_snap_applies_to_each_point_placement_tool(draw, tool):
    draw.drawings = [dict(type='rectangle', coords=[(0., 0.), (100., 100.)])]
    draw.tool = tool
    # Exercise a non-axis-aligned target after placement has already started.
    draw.points = [(-20., -30.), (-10., -40.), (5., 5.)]
    assert draw._point(QPointF(5.01, 5.01)) == (50., 50.)
    assert draw._snap_target == (50., 50.)


def test_circle_uses_exact_interior_and_center(draw):
    draw.drawings = [dict(type='circle', coords=[(100., 100.), (200., 100.)])]
    for click in [(190., 100.), (100., 190.), (30., 80.)]:
        assert draw._point(QPointF(*click)) == (100., 100.)


def test_closed_line_chain_snaps_from_interior_far_from_every_edge(draw):
    corners = [(0., 0.), (100., 0.), (100., 100.), (0., 100.), (0., 0.)]
    draw.drawings = [dict(type='line', coords=[a, b], line_chain='figure')
                     for a, b in zip(corners, corners[1:])]
    assert draw._point(QPointF(20., 20.)) == (50., 50.)
    draw.drawings.pop()  # An unfinished/open chain has no center snap target.
    assert draw._point(QPointF(20., 20.)) == (20., 20.)


def test_bbox_interior_outside_concave_polygon_does_not_snap(draw):
    draw.drawings = [dict(type='polyline', coords=[(0., 0.), (20., 0.),
        (20., 5.), (5., 5.), (5., 20.), (0., 20.), (0., 0.)])]
    assert draw._point(QPointF(18., 18.)) == (18., 18.)
    assert draw._snap_target is None


def test_open_line_is_not_a_closed_center_target(draw):
    draw.drawings = [dict(type='polyline', coords=[(0., 0.), (100., 0.), (100., 100.)])]
    assert draw._point(QPointF(80., 10.)) == (80., 10.)
    assert draw._snap_target is None


def test_disabling_center_snap_or_using_another_mode_keeps_raw_interior(draw):
    draw.drawings = [dict(type='rectangle', coords=[(0., 0.), (100., 100.)])]
    draw.app.digitizer.snap_enabled = False
    assert draw._point(QPointF(10., 10.)) == (10., 10.)
    draw.app.digitizer.snap_enabled = True
    draw.app.digitizer.snap_mode = 'nearby'
    assert draw._point(QPointF(10., 10.)) == (10., 10.)


def test_overlapping_figures_choose_nearest_containing_center(draw):
    draw.drawings = [dict(type='rectangle', coords=[(0., 0.), (100., 100.)]),
                     dict(type='rectangle', coords=[(0., 0.), (20., 20.)])]
    assert draw._point(QPointF(2., 2.)) == (10., 10.)


def test_existing_near_center_snap_outside_concave_shape_is_preserved(draw):
    # A concave polygon centroid can be outside the polygon. Retain the
    # ordinary small-radius center pick as well as the new interior pick.
    draw.drawings = [dict(type='polyline', coords=[(0., 0.), (20., 0.),
        (20., 5.), (5., 5.), (5., 20.), (0., 20.), (0., 0.)])]
    center = polygon_centroid(draw.drawings[0]['coords'])
    assert draw._point(QPointF(center[0] + .01, center[1])) == pytest.approx(center)
