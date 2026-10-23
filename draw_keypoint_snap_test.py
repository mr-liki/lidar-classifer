"""Key Point Snap picks the hit line first, then its closest real vertex."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPointF

from gui.native_draw import NativeDraw


@pytest.fixture
def draw():
    draw = NativeDraw.__new__(NativeDraw)
    draw.app = SimpleNamespace(digitizer=SimpleNamespace(snap_enabled=True, snap_mode='keypoint'),
                               snt_attachments=[])
    draw.canvas = SimpleNamespace(section_at=lambda x, y: (x, y), world_per_pixel=lambda: .01)
    draw.drawings = [dict(type='line', coords=[(0., 0.), (100., 0.)])]
    draw.tool = 'line'
    draw.points = []
    draw._line_session_refs = []
    draw._snap_target = None
    return draw


@pytest.mark.parametrize(('click', 'expected'), [((20., .05), (0., 0.)),
    ((80., -.05), (100., 0.)), ((20., 0.), (0., 0.))])
def test_click_near_long_line_snaps_to_nearest_endpoint(draw, click, expected):
    assert draw._point(QPointF(*click)) == expected
    assert draw._snap_target == expected


def test_polyline_segment_snaps_to_nearest_vertex(draw):
    draw.drawings = [dict(type='polyline', coords=[(0., 0.), (100., 0.), (100., 100.)])]
    assert draw._point(QPointF(100.05, 20.)) == (100., 0.)
    assert draw._point(QPointF(100.05, 80.)) == (100., 100.)


@pytest.mark.parametrize('tool', ['line', 'smartline', 'polyline', 'rectangle',
                                  'circle', 'curve', 'text', 'ortho', 'accudraw'])
def test_point_placement_keeps_exact_keypoint_target(draw, tool):
    draw.tool = tool
    draw.points = [(-20., -30.), (-10., -40.), (5., 5.)]
    assert draw._point(QPointF(80., .05)) == (100., 0.)


def test_closest_stroke_wins_over_another_lines_closer_vertex(draw):
    draw.drawings.append(dict(type='line', coords=[(40., .1), (50., .1)]))
    assert draw._point(QPointF(40., .01)) == (0., 0.)


def test_outside_line_pick_tolerance_does_not_snap(draw):
    assert draw._point(QPointF(40., 5.)) == (40., 5.)
    assert draw._snap_target is None


def test_near_vertex_snap_remains_available(draw):
    assert draw._point(QPointF(-.05, .05)) == (0., 0.)


def test_nearby_snap_retains_projection_onto_stroke(draw):
    draw.app.digitizer.snap_mode = 'nearby'
    assert draw._point(QPointF(40., .05)) == (40., 0.)


def test_center_snap_does_not_treat_open_line_as_closed_figure(draw):
    draw.app.digitizer.snap_mode = 'center'
    assert draw._point(QPointF(40., .05)) == (40., .05)


def test_curve_snaps_to_control_endpoint_not_tessellated_sample(draw):
    controls = [(0., 0.), (50., 50.), (100., 0.)]
    draw.drawings = [dict(type='curve', coords=controls)]
    rendered = draw._native_coords(draw.drawings[0])
    click = rendered[len(rendered) // 3]
    expected = min((controls[0], controls[-1]),
                   key=lambda p: (p[0]-click[0])**2 + (p[1]-click[1])**2)
    assert draw._point(QPointF(*click)) == expected
