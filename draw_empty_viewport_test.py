"""Fresh-launch Draw uses the same native viewport promotion as Measure."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication, QWidget

from gui.native_draw import NativeDraw, activate_native_draw, get_native_draw
from gui.render_backend import _VulkanSurfaceWidget


class Owner:
    def __init__(self, widget):
        self.vulkan_widget = widget
        self.vulkan_backend = SimpleNamespace(supports_vector_overlay=lambda: True,
                                               request_render=Mock())
        self.active = True
        self._viewport_installed = False
        self.install_count = 0

    def vulkan_viewer_host_is_ready(self):
        return True

    def install_as_main_viewport(self):
        self._viewport_installed = True
        self.install_count += 1
        return True

    def vulkan_viewport_is_visible(self):
        return self._viewport_installed

    def owns_main_camera(self):
        return self._viewport_installed


@pytest.fixture
def app(monkeypatch):
    qapp = QApplication.instance() or QApplication([])
    widget = QWidget()
    owner = Owner(widget)
    app = SimpleNamespace(render_backend=owner, _lnt_main_view=None,
                           data=None, class_palette={}, statusBar=Mock())
    owner._app = app
    # Keep viewport/input logic real; no GPU is needed for shape assertions.
    monkeypatch.setattr(NativeDraw, '_paint', lambda self: None)
    yield app
    view = getattr(app, '_snt_viewport_view', None)
    drawing = getattr(view, 'drawing', None)
    if drawing is not None:
        drawing.shutdown()
    widget.close()


@pytest.mark.parametrize('tool', ['Line', 'SmartLine', 'Polyline', 'Rectangle',
                                  'Circle', 'Freehand', 'Curve', 'Move Vertex',
                                  'Delete Vertex', 'Vertex', 'Select', 'Parallel', 'Centerline'])
def test_empty_viewport_draw_activation_promotes_and_routes_native_input(app, tool):
    assert activate_native_draw(app, tool)
    owner = app.render_backend
    assert owner.install_count == 1
    assert app._snt_viewport_promoted
    drawing = get_native_draw(app)
    assert isinstance(drawing, NativeDraw)
    assert owner._plan_draw_view is app._snt_viewport_view
    assert drawing.canvas.widget is owner.vulkan_widget
    assert _VulkanSurfaceWidget._active_plan_draw(SimpleNamespace(_owner=owner)) is drawing
    assert app.data is None


def test_native_line_can_be_drawn_before_any_data_load(app):
    assert activate_native_draw(app, 'Line')
    drawing = get_native_draw(app)
    drawing._point = lambda pos: (pos.x(), pos.y())
    drawing.press(QPointF(1, 2), Qt.LeftButton)
    drawing.press(QPointF(3, 4), Qt.LeftButton)
    assert drawing.drawings[0]['coords'] == [(1., 2.), (3., 4.)]
    assert activate_native_draw(app, 'Move Vertex')
    assert get_native_draw(app) is drawing
    assert app.render_backend.install_count == 1
    assert len(drawing.drawings) == 1


def test_read_only_lookup_does_not_promote_empty_viewport(app):
    assert get_native_draw(app, create=False) is None
    assert app.render_backend.install_count == 0


def test_loaded_legacy_cloud_is_not_covered_by_empty_native_canvas(app):
    app.data = {'xyz': np.zeros((3, 3))}
    assert not activate_native_draw(app, 'Line')
    assert app.render_backend.install_count == 0


def test_engine_without_vector_support_is_not_promoted(app):
    app.render_backend.vulkan_backend.supports_vector_overlay = lambda: False
    assert not activate_native_draw(app, 'Line')
    assert app.render_backend.install_count == 0


def test_unknown_tool_does_not_promote_canvas(app):
    assert not activate_native_draw(app, 'unknown-tool')
    assert app.render_backend.install_count == 0


def test_draw_after_3d_orbit_returns_empty_canvas_to_plan(app):
    # A 3D flag kept across Clear Project made owns_main_camera() refuse the
    # native canvas, so Draw fell to the VTK digitizer hidden under Vulkan.
    def to_top(mode):
        app.is_3d_mode = False
        app._main_view_3d_user_enabled = False
    app.is_3d_mode = True
    app.handle_view_change = Mock(side_effect=to_top)
    assert activate_native_draw(app, 'SmartLine')
    app.handle_view_change.assert_called_once_with('top')


def test_draw_in_3d_lnt_view_returns_to_plan_view():
    from gui.native_draw import _return_to_plan_view_for_draw
    native = SimpleNamespace(plan_view=Mock())
    view = SimpleNamespace(native=native, mode_2d=False)
    view.set_2d = lambda on: setattr(view, 'mode_2d', on)
    app = SimpleNamespace(_lnt_main_view=view, is_3d_mode=True,
                          _main_view_3d_user_enabled=True,
                          handle_view_change=Mock(return_value=False))
    _return_to_plan_view_for_draw(app)
    app.handle_view_change.assert_called_once_with('top')
    assert view.mode_2d and native.plan_view.called
    assert not app.is_3d_mode and not app._main_view_3d_user_enabled


def test_draw_keeps_plan_lnt_view_and_vtk_cloud_camera_untouched():
    from gui.native_draw import _return_to_plan_view_for_draw
    plan = SimpleNamespace(_lnt_main_view=SimpleNamespace(native=object(), mode_2d=True),
                           handle_view_change=Mock())
    _return_to_plan_view_for_draw(plan)
    vtk_cloud = SimpleNamespace(_lnt_main_view=None, data={'xyz': 1}, is_3d_mode=True,
                                handle_view_change=Mock())
    _return_to_plan_view_for_draw(vtk_cloud)
    plan.handle_view_change.assert_not_called()
    vtk_cloud.handle_view_change.assert_not_called()
