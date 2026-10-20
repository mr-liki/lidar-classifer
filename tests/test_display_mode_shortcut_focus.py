from types import SimpleNamespace

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QPushButton

from gui.global_shortcuts import GlobalShortcutFilter


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
