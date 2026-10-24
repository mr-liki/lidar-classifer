from types import SimpleNamespace

from PySide6.QtCore import Qt
from gui.display_mode import DisplayModeDialog
from gui import native_flight_lines


class _LinesPopup:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


def test_view_switch_closes_lines_popup_and_clears_its_slot_context():
    dialog = type("DialogStub", (), {})()
    popup = _LinesPopup()
    dialog._lines_popup = popup

    DisplayModeDialog._close_lines_dialog(dialog)

    assert popup.close_count == 1
    assert dialog._lines_popup is None


def test_view_switch_close_is_safe_when_lines_popup_is_not_open():
    dialog = type("DialogStub", (), {})()

    DisplayModeDialog._close_lines_dialog(dialog)

    assert dialog._lines_popup is None


class _Window:
    def __init__(self, visible=True, children=None):
        self._visible = visible
        self._children = list(children or [])
        self.raise_count = 0
        self.activation_count = 0

    def isVisible(self):
        return self._visible

    def windowState(self):
        return Qt.WindowNoState

    def findChildren(self, _widget_type):
        return self._children

    def isWindow(self):
        return True

    def windowType(self):
        return Qt.Window

    def raise_(self):
        self.raise_count += 1

    def activateWindow(self):
        self.activation_count += 1


def test_owner_activation_raises_display_mode_and_visible_child_popups():
    popup = _Window()
    hidden_popup = _Window(visible=False)
    dialog = _Window(children=[popup, hidden_popup])

    DisplayModeDialog._raise_visible_window_family(dialog)

    assert dialog.raise_count == 1
    assert popup.raise_count == 1
    assert hidden_popup.raise_count == 0
    assert popup.activation_count == 1


def test_existing_flight_line_sidecar_is_attached_without_worker(monkeypatch):
    calls = []

    class _ExistingLines:
        def __init__(self, path):
            calls.append(path)

        def unique(self):
            return [7, 19, 42]

    monkeypatch.setattr(native_flight_lines, "FlightLines", _ExistingLines)
    view = SimpleNamespace(lnt_path="project.lnt", flight_lines=None)

    result = native_flight_lines.open_existing(view)

    assert isinstance(result, _ExistingLines)
    assert view.flight_lines is result
    assert calls == ["project.lnt"]

    # A second Lines click reuses the attached metadata.
    assert native_flight_lines.open_existing(view) is result
    assert calls == ["project.lnt"]
