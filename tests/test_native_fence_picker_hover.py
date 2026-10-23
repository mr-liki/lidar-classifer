"""Fence row hover is a transient native cyan stroke, independent of selection."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF
from PySide6.QtGui import QEnterEvent
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QMainWindow, QWidget

from gui.native_draw import NativeDraw
from gui.menu_sidebar_system import _show_fence_selection_dialog


class ForbiddenVTK:
    def __getattribute__(self, name):
        raise AssertionError('Legacy VTK accessed: ' + name)


@pytest.fixture
def picker_scene():
    qt = QApplication.instance() or QApplication([])
    app = QMainWindow()
    canvas = QWidget(app)
    canvas.resize(640, 480)
    app.setCentralWidget(canvas)
    app.vtk_widget = ForbiddenVTK()
    app.data = {'classification': np.array([2, 3, 4], dtype=np.uint8)}
    native = Mock()
    native.screen_to_world.side_effect = lambda x, y: (x, 480-y, 0.)
    native.world_to_screen.side_effect = lambda x, y: (x, 480-y)
    view = SimpleNamespace(app=app, native=native, mode_2d=True,
                           owner=SimpleNamespace(vulkan_widget=canvas))
    app._lnt_main_view = view
    draw = NativeDraw(view)
    view.drawing = draw
    draw.drawings = [
        {'type': 'polyline', 'coords': [(10., 10.), (60., 10.), (40., 50.), (10., 10.)]},
        {'type': 'rectangle', 'coords': [(90., 10.), (120., 50.)]},
    ]
    draw._paint()
    owner = QDialog(app)
    owner.app = app
    owner.selected_fences = []
    owner.permanent_fence_mode = False
    owner.update_fence_display = Mock()
    app.show()
    owner.show()
    _show_fence_selection_dialog(owner, 'Select Fence(s)', 'Select fences')
    picker = owner._fence_selection_dialog
    qt.processEvents()
    rows = [w for w in picker.findChildren(QWidget) if w.objectName() == 'fencePickerRow']
    assert len(rows) == 2
    # Begin each test with no pointer-dependent preview.
    for row in rows:
        qt.sendEvent(row, QEvent(QEvent.Leave))
    yield qt, app, owner, picker, rows, draw
    if owner._fence_selection_dialog is not None:
        owner._fence_selection_dialog.close()
    draw.shutdown()
    owner.close()
    app.close()
    qt.processEvents()


def enter(qt, row):
    qt.sendEvent(row, QEnterEvent(QPointF(5., 5.), QPointF(5., 5.), QPointF(5., 5.)))


def hovered(draw):
    return [s for s in draw.overlay.shapes if s[2].get('source') == 'fence_picker_hover']


def test_hover_only_matching_row_and_restore_on_leave(picker_scene):
    qt, app, owner, _, rows, draw = picker_scene
    originals = deepcopy(draw.drawings)
    classes = app.data['classification'].copy()
    revision = draw._fence_selection_revision
    assert not hovered(draw)
    for index, row in enumerate(rows):
        enter(qt, row)
        shapes = hovered(draw)
        assert len(shapes) == 1
        _, ring, style = shapes[0]
        assert style['color'] == '#00ffff' and style['width'] == 6
        assert style['show_vertices'] is False and style['style'] == 'solid'
        expected = draw._native_coords(draw.drawings[index])
        assert list(ring) == list(expected)
        qt.sendEvent(row, QEvent(QEvent.Leave))
        assert not hovered(draw)
    assert draw.drawings == originals and owner.selected_fences == []
    assert draw._fence_selection_revision == revision
    np.testing.assert_array_equal(app.data['classification'], classes)


def test_checkbox_preview_survives_hover_leave_without_sticking_hover(picker_scene):
    qt, _, owner, _, rows, draw = picker_scene
    check = rows[0].findChild(QCheckBox)
    check.setChecked(True)
    before = deepcopy(draw._fence_picker_preview_rings)
    enter(qt, rows[1])
    assert len(hovered(draw)) == 1
    qt.sendEvent(rows[1], QEvent(QEvent.Leave))
    assert not hovered(draw)
    assert draw._fence_picker_preview_rings == before
    assert check.isChecked() and owner.selected_fences == []


@pytest.mark.parametrize('action', ['hide', 'close', 'deactivate'])
def test_picker_lifecycle_clears_hover(picker_scene, action):
    qt, _, _, picker, rows, draw = picker_scene
    enter(qt, rows[0])
    assert hovered(draw)
    if action == 'deactivate':
        qt.sendEvent(picker, QEvent(QEvent.WindowDeactivate))
    else:
        getattr(picker, action)()
    assert not hovered(draw)


def test_stale_picker_cleanup_cannot_clear_new_picker_hover(picker_scene):
    _, _, _, _, _, draw = picker_scene
    first, second = object(), object()
    ring = {'coords': draw._native_coords(draw.drawings[0])}
    draw.set_classification_fence_hover(first, ring)
    draw.set_classification_fence_hover(second, ring)
    draw.set_classification_fence_hover(first)
    assert len(hovered(draw)) == 1
    draw.set_classification_fence_hover(second)
    assert not hovered(draw)
