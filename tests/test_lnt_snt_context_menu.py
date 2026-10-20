from types import SimpleNamespace

import pytest
from PySide6.QtCore import QEvent, QObject, QPointF, Qt

from gui.lnt_open import _InputFilter


class View(QObject):
    def __init__(self):
        super().__init__()
        self.native = object()
        self.mode_2d = True
        self.app = SimpleNamespace()


class Mouse:
    def __init__(self, kind, x=10, y=20):
        self.kind = kind
        self.pos = QPointF(x, y)

    def type(self):
        return self.kind

    def position(self):
        return self.pos

    def button(self):
        return Qt.RightButton

    def buttons(self):
        return Qt.RightButton

    def modifiers(self):
        return Qt.NoModifier


@pytest.mark.parametrize("drag, tool_owns_click, expected", [
    (False, False, [(10.0, 20.0)]),
    (True, False, []),
    (False, True, []),
])
def test_native_right_click_routes_free_clicks_only(monkeypatch, drag, tool_owns_click, expected):
    view = View()
    filt = _InputFilter(view)
    calls = []
    monkeypatch.setattr(filt, "_show_snt_context_menu", lambda x, y: calls.append((x, y)))
    monkeypatch.setattr("gui.lnt_open.QTimer.singleShot", lambda delay, callback: callback())
    monkeypatch.setattr(filt._pan, "handle", lambda *args: False)
    widget = SimpleNamespace(setFocus=lambda: None)
    if tool_owns_click:
        view.creation = SimpleNamespace(active=True, press=lambda *args: True,
                                        release=lambda *args: None)
    filt.eventFilter(widget, Mouse(QEvent.MouseButtonPress))
    if drag:
        filt.eventFilter(widget, Mouse(QEvent.MouseMove, 30, 20))
    filt.eventFilter(widget, Mouse(QEvent.MouseButtonRelease))
    assert calls == expected


def test_native_menu_uses_native_world_coordinates(monkeypatch):
    view = View()
    calls = []
    view.app.grid_label_manager = SimpleNamespace(
        on_native_right_click=lambda xy, upp: calls.append((xy, upp)))
    canvas = SimpleNamespace(section_at=lambda x, y: (1000 + x, 2000 + y),
                             world_per_pixel=lambda: 0.25)
    monkeypatch.setattr("gui.native_main_classify.MainCanvas", lambda view: canvas)
    filt = _InputFilter(view)
    filt._show_snt_context_menu(10, 20)
    assert calls == [((1010, 2020), 0.25)]
