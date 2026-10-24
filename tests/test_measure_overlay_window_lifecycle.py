"""Measurement overlays follow the owning native viewport window."""
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from gui.native_main_classify import MainCanvas
from gui.native_section_classify import CanvasOverlay


@pytest.fixture
def scene():
    qt = QApplication.instance() or QApplication([])
    owner = QWidget()
    owner.resize(640, 480)
    widget = QWidget(owner)
    widget.resize(640, 480)
    view = SimpleNamespace(owner=SimpleNamespace(vulkan_widget=widget),
                           app=SimpleNamespace())
    canvas = MainCanvas(view)
    canvas.screen_from_section = lambda x, y: (x, y)
    canvas.projection_snapshot = lambda: ((1,), canvas.screen_from_section)
    overlay = CanvasOverlay(canvas)
    owner.show()
    qt.processEvents()
    yield qt, owner, widget, canvas, overlay
    overlay._timer.stop()
    overlay.close()
    owner.close()
    qt.processEvents()


@pytest.mark.parametrize("tool", ["line", "path", "block", "grid", "cross_section", "marks"])
def test_minimize_hides_measurements_until_restore(scene, tool):
    qt, owner, widget, _, overlay = scene
    points = [(20., 20.), (100., 100.), (100., 20.)]
    options = {"style": {}, "complete": True, "path": tool == "path"}
    if tool in ("grid", "block"):
        options.update(area=3200., grid_label="sample.laz")
    if tool == "marks":
        overlay.set_marks([points[:2]])
    else:
        overlay.set_shapes([("measurement", points, options)])
    qt.processEvents()
    assert overlay.isVisible()
    before = (list(overlay.shapes), list(overlay.marks))
    # No polling: the owner event must hide the overlay immediately.
    overlay._timer.stop()
    owner.showMinimized()
    assert owner.isMinimized()
    assert widget.isVisible()  # Child visibility alone misses minimization.
    assert not overlay.isVisible()
    # Polling and frame-present notifications must not re-show it.
    for immediate in (False, True, False):
        overlay._follow(immediate=immediate)
        qt.processEvents()
        assert not overlay.isVisible()
    owner.showNormal()
    qt.processEvents()
    overlay._follow()
    assert overlay.isVisible()
    assert (overlay.shapes, overlay.marks) == before


@pytest.mark.parametrize("action", ["hide", "close"])
def test_owner_hide_or_close_preserves_hidden_measurements(scene, action):
    qt, owner, _, _, overlay = scene
    overlay.set_shapes([("measurement", [(20., 20.), (100., 100.)], {})])
    qt.processEvents()
    overlay._timer.stop()
    getattr(owner, action)()
    assert not overlay.isVisible()
    overlay._follow(immediate=True)
    assert not overlay.isVisible()
    assert len(overlay.shapes) == 1
    owner.show()
    qt.processEvents()
    overlay._follow()
    assert overlay.isVisible()


def test_overlay_tracks_floating_canvas_owner(scene):
    qt, owner, widget, _, overlay = scene
    floating = QWidget()
    floating.resize(640, 480)
    try:
        overlay.set_shapes([("measurement", [(20., 20.), (100., 100.)], {})])
        widget.setParent(floating)
        widget.show()
        floating.show()
        qt.processEvents()
        overlay._follow()
        assert overlay.parentWidget() is floating
        assert overlay.isVisible()
        owner.showMinimized()
        overlay._follow()
        assert overlay.isVisible()
        overlay._timer.stop()
        floating.showMinimized()
        assert not overlay.isVisible()
        overlay._follow(immediate=True)
        assert not overlay.isVisible()
        floating.showNormal()
        qt.processEvents()
        overlay._follow()
        assert overlay.isVisible()
    finally:
        widget.setParent(owner)
        widget.show()
        overlay._follow()
        floating.close()
