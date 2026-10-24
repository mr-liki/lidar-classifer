"""Exercise Vulkan-only Zoom through the production Qt input filter."""
import gc
import os
from types import SimpleNamespace
import pytest

from PySide6.QtCore import QEvent, QObject, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

from gui.native_zoom_rect import NativeZoomRect


@pytest.mark.parametrize('use_native', [False, True])
@pytest.mark.parametrize('operation', ['zoom', 'block', 'shape'])
def test_zoom_drag_with_real_qt_event_dispatch(tmp_path, use_native, monkeypatch, operation):
    if use_native and os.environ.get('NAKSHA_TEST_NATIVE_ZOOM') != '1':
        pytest.skip('Set NAKSHA_TEST_NATIVE_ZOOM=1 for Windows Vulkan gesture test')
    qt = QApplication.instance() or QApplication([])
    window = QMainWindow()
    viewport = QWidget(window)
    window.setCentralWidget(viewport)
    window.resize(640, 480)
    frames = []
    native = SimpleNamespace(
        screen_to_world=lambda x, y: (float(x), float(y), 0.),
        world_to_screen=lambda x, y: (x, y),
        frame_region=lambda *args: frames.append(args),
    )
    app = SimpleNamespace(statusBar=window.statusBar)
    view = QObject()
    view.app = app
    view.native = native
    view.owner = SimpleNamespace(vulkan_widget=viewport)
    view.mode_2d = True
    window.show()
    qt.processEvents()
    if use_native:
        import laspy
        import numpy as np
        from gui.lnt_native import NativeView, convert_las
        las = laspy.LasData(laspy.LasHeader(point_format=7, version='1.4'))
        las.x = np.linspace(0, 500, 1000)
        las.y = np.linspace(0, 500, 1000)
        las.z = np.ones(1000)
        source, document = str(tmp_path / 'zoom.las'), str(tmp_path / 'zoom.lnt')
        las.write(source)
        assert convert_las(source, document)
        native = view.native = NativeView(int(viewport.winId()), viewport.width(),
                                          viewport.height(), document, 64, 8.)
        from gui.render_backend import VulkanRenderBackend
        backend = VulkanRenderBackend()
        assert backend.is_available()
        backend._bind_signatures()
        view.owner.vulkan_backend = backend
        view.lnt_path = document
        app._lnt_main_view = view
        app.render_backend = view.owner
        native.frame(.016)
    else:
        from unittest.mock import Mock
        monkeypatch.setattr('gui.native_zoom_preview.VulkanZoomPreview', lambda view: Mock())
        monkeypatch.setattr('gui.native_zoom_preview.VulkanSelectionPreview', lambda view: Mock())
    if operation == 'zoom':
        tool = NativeZoomRect(view)
        view.native_zoom = tool
    else:
        import numpy as np
        from unittest.mock import Mock
        from gui.native_select_rect import NativeSelectRect
        tool = NativeSelectRect(view)
        view.native_select = tool
        tool.pick_method = 'Shape' if operation == 'shape' else 'Block'
        world = [tool._canvas().section_at(x, y) for x, y in
                 [(60, 60), (180, 200), (300, 300)]]
        store = SimpleNamespace(
            data={'xyz': np.array([(x, y, 1.) for x, y in world]),
                  'classification': np.array([2, 2, 2], dtype=np.uint8)},
            ensure_region=lambda *args: True, last_region_indices=np.arange(3),
            show=Mock(), flush=Mock())
        monkeypatch.setattr('gui.lnt_section_bridge.get_store', lambda app: store)
        app.undo_stack, app.redo_stack = [], []

    from gui.lnt_open import _InputFilter

    class Input(_InputFilter):
        dispatching = False

        def eventFilter(self, obj, event):
            self.dispatching = True
            try:
                return super().eventFilter(obj, event)
            finally:
                self.dispatching = False

    input_filter = Input(view)
    viewport.installEventFilter(input_filter)
    from gui.crash_reporter import CrashReporter
    reporter = CrashReporter()
    original_notify = qt.notify
    reporter.install_qt_notify(qt)
    window.show()
    qt.processEvents()
    windows_before = set(qt.topLevelWidgets())
    try:
        for _ in range(20):
            tool.activate()
            QTest.mousePress(viewport, Qt.LeftButton, pos=QPoint(20, 30))
            if operation == 'shape':
                QTest.mouseRelease(viewport, Qt.LeftButton, pos=QPoint(20, 30))
                QTest.mouseClick(viewport, Qt.LeftButton, pos=QPoint(220, 30))
                QTest.mouseClick(viewport, Qt.LeftButton, pos=QPoint(20, 230))
            QTest.mouseMove(viewport, QPoint(220, 230))
            if use_native:
                native.frame(.016)
                document = tool._overlay.layer.documents[tool._overlay.key]
                assert document['handles'][tool._overlay.LAYER]._visible
            if operation == 'shape':
                QTest.mouseClick(viewport, Qt.RightButton, pos=QPoint(20, 230))
            else:
                QTest.mouseRelease(viewport, Qt.LeftButton, pos=QPoint(220, 230))
            if operation != 'zoom':
                assert tool.selected_count == (1 if operation == 'shape' else 2)
            qt.processEvents()
            gc.collect()
            tool.deactivate()
            if use_native:
                native.frame(.016)
        if not use_native and operation == 'zoom':
            assert len(frames) == 20
            assert frames[-1][:2] == (120., 130.)
        assert set(qt.topLevelWidgets()) == windows_before
        if use_native:
            layer = tool._overlay.layer
            assert layer.overlay is None
            document = layer.documents[tool._overlay.key]
            chunk = document['gpu'][tool._overlay.LAYER][0]
            capacity = 8 if operation == 'shape' else 5
            assert chunk['v_count'] == capacity
            assert layer._v_used == capacity
            assert not document['handles'][tool._overlay.LAYER]._visible
            # Sharing and growing the arena must retain unrelated linework and
            # update the rectangle's current offsets after compaction.
            other = [{'type': 'polyline', 'points': [(10., 10., 10.), (20., 20., 10.)],
                      'layer': 'existing', 'color': (0, 255, 255)}]
            assert layer.put_entities('snt', 'existing.snt', other)
            tool._overlay.set_rectangle((100., 100.), (200., 200.))
            native.frame(.016)
            existing = layer.documents[('snt', 'existing.snt')]
            assert existing['handles']['existing']._visible
            assert existing['gpu']['existing'][0]['v_count'] == 2
            tool._overlay.clear()
            assert existing['handles']['existing']._visible
        if operation != 'zoom':
            from PySide6.QtWidgets import QMessageBox
            monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.Yes)
            tool.delete_selected_points()
            expected = [7, 2, 2] if operation == 'shape' else [7, 7, 2]
            assert store.data['classification'].tolist() == expected
            assert len(app.undo_stack) == 1
            entry = app.undo_stack[-1]
            store.data['classification'][entry['indices']] = entry['old_classes']
            assert store.data['classification'].tolist() == [2, 2, 2]
    finally:
        tool.shutdown()
        viewport.removeEventFilter(input_filter)
        qt.notify = original_notify
        qt._crash_reporter_notify_installed = False
        if use_native:
            native.close()
        window.close()
        qt.processEvents()


def test_zoom_activation_releases_native_draw_and_measure():
    from unittest.mock import Mock
    view = SimpleNamespace(app=SimpleNamespace(statusBar=lambda: Mock()),
                           drawing=Mock(), measurement=Mock())
    view.drawing.active.return_value = True
    view.measurement.active = True
    tool = NativeZoomRect(view)
    tool._canvas = Mock(return_value=Mock())
    tool._overlay_for = Mock()
    assert tool.activate()
    view.drawing.cancel.assert_called_once()
    view.measurement.cancel.assert_called_once()


def test_generated_elevation_mesh_owns_real_lnt_vulkan_surface(tmp_path):
    if os.environ.get('NAKSHA_TEST_NATIVE_ZOOM') != '1':
        pytest.skip('Set NAKSHA_TEST_NATIVE_ZOOM=1 for Windows Vulkan test')
    import laspy
    import numpy as np
    from gui.elevation_models import ElevationModel, display_elevation_model_vulkan
    from gui.lnt_native import NativeView, convert_las, MODE_CLASS
    from gui.render_backend import VulkanRenderBackend

    qt = QApplication.instance() or QApplication([])
    window = QMainWindow()
    viewport = QWidget(window)
    window.setCentralWidget(viewport)
    window.resize(640, 480)
    window.show()
    qt.processEvents()
    las = laspy.LasData(laspy.LasHeader(point_format=7, version='1.4'))
    las.x = np.linspace(0, 20, 400)
    las.y = np.linspace(0, 20, 400)
    las.z = np.linspace(1, 3, 400)
    source, document = str(tmp_path / 'dem.las'), str(tmp_path / 'dem.lnt')
    las.write(source)
    assert convert_las(source, document)
    native = NativeView(int(viewport.winId()), viewport.width(), viewport.height(),
                        document, 64, 8.)
    backend = VulkanRenderBackend()
    assert backend.is_available()
    backend._bind_signatures()
    owner = SimpleNamespace(vulkan_widget=viewport, vulkan_backend=backend)
    app = SimpleNamespace(render_backend=owner)
    view = SimpleNamespace(native=native, owner=owner, app=app)
    app._lnt_main_view = view
    try:
        grid = np.arange(25, dtype=np.float32).reshape(5, 5)
        model = ElevationModel('DSM', grid, left=0., top=20., resolution=4.)
        assert display_elevation_model_vulkan(app, model)
        assert app._elevation_vulkan_surface['triangles'] == 32
        for _ in range(5):
            assert native.frame(.016) is not None
        native.set_mode(MODE_CLASS)
        native.frame(.016)
    finally:
        native.close()
        window.close()
        qt.processEvents()


def test_temp_fence_gesture_uses_real_lnt_vulkan_overlay(tmp_path, monkeypatch):
    if os.environ.get('NAKSHA_TEST_NATIVE_ZOOM') != '1':
        pytest.skip('Set NAKSHA_TEST_NATIVE_ZOOM=1 for Windows Vulkan test')
    import laspy
    import numpy as np
    from gui.lnt_native import NativeView, convert_las
    from gui.render_backend import VulkanRenderBackend
    from gui.temp_fence_tool import TempFenceTool
    from gui.lnt_open import _InputFilter

    qt = QApplication.instance() or QApplication([])
    window = QMainWindow()
    viewport = QWidget(window)
    window.setCentralWidget(viewport)
    window.resize(640, 480)
    window.show()
    qt.processEvents()
    las = laspy.LasData(laspy.LasHeader(point_format=7, version='1.4'))
    las.x = np.linspace(0, 500, 1000)
    las.y = np.linspace(0, 500, 1000)
    las.z = np.ones(1000)
    source, document = str(tmp_path / 'fence.las'), str(tmp_path / 'fence.lnt')
    las.write(source)
    assert convert_las(source, document)
    native = NativeView(int(viewport.winId()), viewport.width(), viewport.height(),
                        document, 64, 8.)
    backend = VulkanRenderBackend()
    assert backend.is_available()
    backend._bind_signatures()
    owner = SimpleNamespace(vulkan_widget=viewport, vulkan_backend=backend)
    window.render_backend = owner
    view = QObject()
    view.app = window
    view.native = native
    view.owner = owner
    view.mode_2d = True
    view.picker = None
    view.drawing = view.measurement = view.classifier = None
    window._lnt_main_view = view
    popup_calls = []
    tool = TempFenceTool(window)
    monkeypatch.setattr(tool, '_show_popup', lambda centroid: popup_calls.append(centroid))
    input_filter = _InputFilter(view)
    viewport.installEventFilter(input_filter)
    try:
        tool.activate()
        assert view.native_temp_fence is tool
        QTest.mouseClick(viewport, Qt.LeftButton, pos=QPoint(100, 100))
        QTest.mouseMove(viewport, QPoint(300, 100))
        QTest.mouseClick(viewport, Qt.LeftButton, pos=QPoint(300, 100))
        QTest.mouseClick(viewport, Qt.LeftButton, pos=QPoint(200, 300))
        native.frame(.016)
        preview_doc = tool._native_preview.layer.documents[tool._native_preview.key]
        assert preview_doc['handles'][tool._native_preview.LAYER]._visible
        QTest.mouseClick(viewport, Qt.RightButton, pos=QPoint(200, 300))
        native.frame(.016)
        assert window.temp_fence['_temp_fence'] is True
        assert len(window.temp_fence['coords']) == 3
        assert len(popup_calls) == 1
        final_doc = tool._native_fence_overlay.layer.documents[
            tool._native_fence_overlay.key]
        assert final_doc['handles'][tool._native_fence_overlay.LAYER]._visible
        assert not preview_doc['handles'][tool._native_preview.LAYER]._visible
    finally:
        tool.shutdown()
        viewport.removeEventFilter(input_filter)
        native.close()
        window.close()
        qt.processEvents()
