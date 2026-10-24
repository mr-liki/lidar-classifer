import ast
import builtins
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QPushButton, QWidget

from gui.global_shortcuts import GlobalShortcutFilter
from gui.native_section_classify import SectionClassifier


def _guard_with_focus(widget_type):
    qt_app = QApplication.instance() or QApplication([])
    dialog = QDialog()
    focused = widget_type(dialog)
    dialog.show()
    focused.show()
    dialog.activateWindow()
    focused.setFocus()
    qt_app.processEvents()

    owner = SimpleNamespace(display_mode_dialog=dialog)
    guard = GlobalShortcutFilter(owner)
    result = guard._is_display_mode_dialog_active()
    dialog.close()
    return result


def test_real_section_escape_shortcut_disarms_classification():
    from PySide6.QtGui import QKeySequence, QShortcut
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QWidget
    from gui.native_sections import SectionCanvas
    from gui.native_section_classify import SectionClassifier

    qt_app = QApplication.instance() or QApplication([])
    canvas = QWidget()
    canvas.setFocusPolicy(Qt.StrongFocus)
    calls = []
    app = SimpleNamespace(active_classify_tool="above_line")

    def deactivate(**kwargs):
        calls.append(kwargs)
        app.active_classify_tool = None

    app.deactivate_classification_tool = deactivate
    canvas.owner = SimpleNamespace(app=app, cancel_cut=lambda: None)
    canvas.pan_gesture = SimpleNamespace(finish=lambda: None)
    classifier = SectionClassifier.__new__(SectionClassifier)
    classifier.app = app
    classifier.canvas = canvas
    classifier._brush_tx_active = False
    classifier._paint = lambda: None
    canvas.classifier = classifier
    shortcut = QShortcut(QKeySequence("Escape"), canvas)
    shortcut.activated.connect(lambda: SectionCanvas.escape(canvas))
    canvas.show()
    canvas.activateWindow()
    canvas.setFocus()
    qt_app.processEvents()
    try:
        # Exercise the Qt shortcut dispatcher, which consumes Escape before
        # KeyPress handlers. Check both idle and unfinished classification.
        for points in ([], [(1., 2.)]):
            app.active_classify_tool = "above_line"
            classifier.points = list(points)
            QTest.keyClick(canvas, Qt.Key_Escape)
            qt_app.processEvents()
            assert app.active_classify_tool is None
            assert classifier.points == []
        assert len(calls) == 2
    finally:
        canvas.close()


def test_display_mode_button_focus_allows_global_shortcuts():
    assert not _guard_with_focus(QPushButton)


def test_display_mode_text_focus_keeps_typed_keys_local():
    assert _guard_with_focus(QLineEdit)


def test_escape_deactivates_classification_before_display_dialog_guard():
    qt_app = QApplication.instance() or QApplication([])
    dialog = QDialog()
    field = QLineEdit(dialog)
    dialog.show()
    field.show()
    field.setFocus()
    qt_app.processEvents()

    calls = []
    owner = SimpleNamespace(
        display_mode_dialog=dialog,
        active_classify_tool="above_line",
        shortcuts={},
        _left_pan_shortcut_active=False,
    )

    def deactivate_classification_tool(**kwargs):
        calls.append(kwargs)
        owner.active_classify_tool = None

    owner.deactivate_classification_tool = deactivate_classification_tool
    guard = GlobalShortcutFilter(owner)
    guard._is_runtime_shortcut_suspended = lambda: False
    event = QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)

    assert guard.eventFilter(field, event)
    assert owner.active_classify_tool is None
    assert calls == [{
        "preserve_cross_section": True,
        "cancel_pending_brush": True,
    }]
    dialog.close()


def _native_deactivate_method():
    # Load the actual shared teardown without starting the legacy app/VTK host.
    path = Path(__file__).resolve().parents[1] / "gui" / "app_window.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    app_class = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                     and n.name == "NakshaApp")
    method = next(n for n in app_class.body if isinstance(n, ast.FunctionDef)
                  and n.name == "deactivate_classification_tool")
    namespace = {"__package__": "gui"}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]


@pytest.mark.parametrize("route", ["global", "canvas", "picker"])
@pytest.mark.parametrize("stage", ["tap", "drag"])
def test_escape_clears_native_rectangle_preview_without_vtk(monkeypatch, route, stage):
    qt = QApplication.instance() or QApplication([])
    status = Mock()
    app = SimpleNamespace(
        active_classify_tool="rectangle", shortcuts={},
        _left_pan_shortcut_active=False, skip_main_view_refresh=True,
        _gpu_sync_done=True, _section_visibility_refresh_required=True,
        _get_live_class_picker=lambda: object(),
        _hide_class_picker_safely=Mock(), statusBar=lambda: status,
        vtk_widget=SimpleNamespace(render=Mock(side_effect=AssertionError("VTK render"))),
        section_controller=SimpleNamespace(
            unlock_after_classification=Mock(side_effect=AssertionError("VTK controller"))),
    )
    app.deactivate_classification_tool = MethodType(_native_deactivate_method(), app)
    monkeypatch.setitem(sys.modules, "gui.session_manager", SimpleNamespace(APP_BUSY=True))
    canvases = []
    try:
        for line_tools in (False, True, True):
            canvas = QWidget()
            canvas.owner = SimpleNamespace(app=app)
            canvas._closed = False
            canvas.line_tools = line_tools
            canvas.section_at = lambda x, y: (x, y)
            canvas.screen_from_section = lambda x, y: (x, y)
            canvas.classifier = SectionClassifier(canvas)
            canvas.classifier._commit = Mock(side_effect=AssertionError("Incomplete gesture committed"))
            canvases.append(canvas)
            canvas.show()
            classifier = canvas.classifier
            assert classifier.press(QPointF(10, 10), Qt.LeftButton)
            if stage == "tap":
                assert classifier.release(QPointF(10, 10), Qt.LeftButton)
            classifier.move(QPointF(80, 70))
            assert classifier.overlay.shapes
            assert classifier.overlay.isVisible()
        main, cross, cut = canvases
        app._lnt_main_view = SimpleNamespace(native=object(), classifier=main.classifier)
        app._native_sections = SimpleNamespace(views={0: cross, 1: cut})

        original_import = builtins.__import__
        def no_vtk(name, *args, **kwargs):
            if name == "vtk" or name.startswith("vtkmodules"):
                raise AssertionError("Native cancellation imported VTK")
            return original_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", no_vtk)
        if route == "global":
            guard = GlobalShortcutFilter(app)
            guard._is_runtime_shortcut_suspended = lambda: False
            event = QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
            assert guard.eventFilter(main, event)
        elif route == "canvas":
            assert main.classifier.key(Qt.Key_Escape)
            qt.processEvents()  # native key route defers the shared teardown
        else:
            app.deactivate_classification_tool()

        assert app.active_classify_tool is None
        assert not app.skip_main_view_refresh
        assert not app._gpu_sync_done
        assert not app._section_visibility_refresh_required
        app._hide_class_picker_safely.assert_called_once()
        for canvas in canvases:
            classifier = canvas.classifier
            assert classifier.points == []
            assert classifier.hover is None
            assert classifier.press_px is None
            assert not classifier.dragging
            assert classifier.overlay.shapes == []
            assert not classifier.overlay.isVisible()
            assert not classifier.overlay._timer.isActive()
            # The physical release arriving after Esc cannot finish the old shape.
            assert not classifier.release(QPointF(80, 70), Qt.LeftButton)
            classifier._commit.assert_not_called()
        app.vtk_widget.render.assert_not_called()
        app.section_controller.unlock_after_classification.assert_not_called()
    finally:
        for canvas in canvases:
            canvas.classifier.overlay.shutdown()
            canvas.close()
