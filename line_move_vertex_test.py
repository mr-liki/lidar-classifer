"""Line-chain vertex gestures keep neighboring edges connected, like Smart Line."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

from gui.native_draw import NativeDraw


@pytest.fixture
def draw():
    qapp = QApplication.instance() or QApplication([])
    app = QMainWindow()
    widget = QWidget(app)
    app.setCentralWidget(widget)
    view = SimpleNamespace(app=app, native=Mock(), mode_2d=True,
                           owner=SimpleNamespace(vulkan_widget=widget))
    drawing = NativeDraw(view)
    view.drawing = drawing
    app._lnt_main_view = view
    drawing._paint = Mock()
    drawing._point = lambda p: (p.x(), p.y())
    drawing.canvas.world_per_pixel = lambda: .01
    drawing.tool = 'line'
    for point in [(0, 0), (10, 0), (20, 0)]:
        drawing.press(QPointF(*point), Qt.LeftButton)
    drawing._finish_line()
    drawing.tool = 'movevertex'
    yield drawing
    drawing.shutdown()
    app.close()


def assert_corner(draw, point):
    assert draw.drawings[0]['coords'] == [(0., 0.), point]
    assert draw.drawings[1]['coords'] == [point, (20., 0.)]


@pytest.mark.parametrize('drag', [True, False])
def test_shared_corner_preview_commit_and_undo_redo(draw, drag):
    originals = list(draw.drawings)
    history = len(draw._object_actions)
    assert originals[0]['line_chain'] == originals[1]['line_chain']
    draw.press(QPointF(10, 0), Qt.LeftButton)
    if not drag:
        draw.release(QPointF(10, 0), Qt.LeftButton)
    draw.move(QPointF(12, 5), Qt.LeftButton if drag else Qt.NoButton)
    assert_corner(draw, (12., 5.))
    assert originals[0]['coords'] == [(0., 0.), (10., 0.)]
    assert originals[1]['coords'] == [(10., 0.), (20., 0.)]
    assert len(draw._object_actions) == history
    if drag:
        draw.release(QPointF(12, 5), Qt.LeftButton)
    else:
        draw.press(QPointF(12, 5), Qt.LeftButton)
    assert_corner(draw, (12., 5.))
    assert len(draw._object_actions) == history + 1
    assert draw.history()
    assert_corner(draw, (10., 0.))
    assert draw.history(redo=True)
    assert_corner(draw, (12., 5.))


def test_legacy_line_edges_move_together(draw):
    for record in draw.drawings:
        record.pop('line_chain')
    draw.press(QPointF(10, 0), Qt.LeftButton)
    draw.move(QPointF(12, 5), Qt.LeftButton)
    draw.release(QPointF(12, 5), Qt.LeftButton)
    assert_corner(draw, (12., 5.))


@pytest.mark.parametrize('cancel', ['escape', 'undo', 'return_to_origin'])
def test_cancel_or_noop_restores_entire_chain(draw, cancel):
    originals = list(draw.drawings)
    history = len(draw._object_actions)
    draw.press(QPointF(10, 0), Qt.LeftButton)
    draw.move(QPointF(12, 5), Qt.LeftButton)
    if cancel == 'escape':
        draw.cancel()
    elif cancel == 'undo':
        draw.history()
    else:
        draw.release(QPointF(10, 0), Qt.LeftButton)
    assert all(old is new for old, new in zip(originals, draw.drawings))
    assert len(draw._object_actions) == history


def test_unrelated_touching_chain_and_other_figures_are_not_moved(draw):
    other = dict(type='line', coords=[(10., 0.), (10., 20.)], line_chain='other')
    smart = dict(type='smartline', coords=[(10., 0.), (30., 20.)])
    draw.drawings.extend([other, smart])
    draw.press(QPointF(10, 0), Qt.LeftButton)
    draw.move(QPointF(12, 5), Qt.LeftButton)
    draw.release(QPointF(12, 5), Qt.LeftButton)
    assert_corner(draw, (12., 5.))
    assert draw.drawings[2] is other
    assert draw.drawings[3] is smart


def test_line_endpoint_only_moves_its_incident_edge(draw):
    second = draw.drawings[1]
    draw.press(QPointF(0, 0), Qt.LeftButton)
    draw.move(QPointF(1, 5), Qt.LeftButton)
    draw.release(QPointF(1, 5), Qt.LeftButton)
    assert draw.drawings[0]['coords'] == [(1., 5.), (10., 0.)]
    assert draw.drawings[1] is second


def test_smart_line_vertex_behavior_is_preserved(draw):
    smart = dict(type='smartline', coords=[(0., 0.), (10., 0.), (20., 0.)])
    draw.drawings[:] = [smart]
    draw.press(QPointF(10, 0), Qt.LeftButton)
    draw.move(QPointF(12, 5), Qt.LeftButton)
    draw.release(QPointF(12, 5), Qt.LeftButton)
    assert draw.drawings[0]['coords'] == [(0., 0.), (12., 5.), (20., 0.)]
    assert smart['coords'] == [(0., 0.), (10., 0.), (20., 0.)]
    draw.history()
    assert draw.drawings[0] is smart
