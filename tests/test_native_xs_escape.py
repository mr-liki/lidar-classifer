"""Esc on a native cross-section (XS) measurement stands the whole XS tool down."""
from types import SimpleNamespace
from unittest.mock import Mock

from PySide6.QtCore import Qt

import gui.native_shortcuts as native_shortcuts
from gui.native_measure import NativeMeasure


def make_measure(style_key):
    measure = object.__new__(NativeMeasure)
    measure.active = True
    measure.style_key = style_key
    measure.canvas = SimpleNamespace(owner=SimpleNamespace(app=object()))
    measure.cancel = Mock(side_effect=lambda: setattr(measure, 'active', False))
    measure._status = Mock()
    return measure


def test_escape_on_xs_turns_the_xs_tool_off(monkeypatch):
    set_xs = Mock()
    monkeypatch.setattr(native_shortcuts, 'set_xs', set_xs)
    measure = make_measure('cross_section')
    assert measure.key(Qt.Key_Escape)
    set_xs.assert_called_once_with(measure.app, False)
    measure.cancel.assert_called()
    assert not measure.active


def test_escape_on_line_measure_only_cancels_it(monkeypatch):
    set_xs = Mock()
    monkeypatch.setattr(native_shortcuts, 'set_xs', set_xs)
    measure = make_measure(None)
    assert measure.key(Qt.Key_Escape)
    set_xs.assert_not_called()
    measure.cancel.assert_called_once()


def test_section_window_escape_shortcut_reaches_the_xs_measurement(monkeypatch):
    # The section canvas's Escape QShortcut fires before keyPressEvent, so
    # escape() itself must hand Esc to an armed measurement.
    import gui.native_sections as native_sections
    from gui.native_sections import SectionCanvas
    monkeypatch.setattr(native_sections, 'disarm_pan', lambda app: None)
    measurement = SimpleNamespace(key=Mock(return_value=True))
    classifier = SimpleNamespace(active=Mock(return_value=False), busy=Mock(return_value=False),
                                 key=Mock(), cancel=Mock())
    owner = SimpleNamespace(app=object(), cancel_cut=Mock())
    canvas = SimpleNamespace(pan_gesture=SimpleNamespace(finish=Mock()), owner=owner,
                             measurement=measurement, classifier=classifier)
    SectionCanvas.escape(canvas)
    measurement.key.assert_called_once_with(Qt.Key_Escape)
    owner.cancel_cut.assert_not_called()
