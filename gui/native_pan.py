"""Pan-shortcut navigation for the native Vulkan viewports (no VTK).

The Shortcut Manager's Pan tool carries a required mode:

* ``"left"`` - Left - Click Pan: hold the left button and drag.
* ``"tap"``  - Tap - Tap Pan (MicroStation dynamic pan, same model as
  naksha-lidar's section_zoom): the first left tap starts a button-free pan,
  the view follows the cursor, and a second tap / right-click / any key ends
  it. The release of each tap is swallowed.

The gesture only reports pixel deltas; each viewport applies them to its own
native camera (``LntView.pan`` for the main view, the section centre for
cross/cut canvases).
"""

import weakref

from PySide6.QtCore import QEvent, Qt

PAN_MODES = ('left', 'tap')

# Canvases that have shown the pan cursor, so arming / disarming the Pan tool
# can refresh them at once instead of on the next mouse move.
_PAN_CURSOR_WIDGETS = weakref.WeakSet()


def pan_cursor_shape(app, grabbing=False):
    """Open hand while the Pan tool is armed, closed hand while it pans; else None."""
    if armed_pan_mode(app) is None:
        return None
    return Qt.ClosedHandCursor if grabbing else Qt.OpenHandCursor


def apply_pan_cursor(widget, app, grabbing=False):
    """Show the pan hand on ``widget``; restore its normal cursor when disarmed."""
    if widget is None:
        return
    try:
        shape = pan_cursor_shape(app, grabbing)
        applied = widget.property('panCursorShape')
        # Record the shape BEFORE touching the cursor: setCursor/unsetCursor
        # send a synchronous CursorChange event to the canvas's filters, and a
        # re-entrant call must see the new state and do nothing.
        if shape is None:
            if applied:
                widget.setProperty('panCursorShape', '')
                widget.unsetCursor()
            return
        _PAN_CURSOR_WIDGETS.add(widget)
        if applied != shape.name:
            widget.setProperty('panCursorShape', shape.name)
            widget.setCursor(shape)
            print(f'[PAN] {type(widget).__name__} cursor -> {shape.name}')
    except RuntimeError:                # C++ widget already deleted
        pass
    except Exception as exc:            # noqa: BLE001 - cursor is cosmetic, but say why
        print(f'[PAN] cursor update failed on {type(widget).__name__}: {exc}')


def _native_canvases(app):
    """The live native Vulkan canvases: .lnt main viewport + cross/cut windows."""
    widgets = list(_PAN_CURSOR_WIDGETS)
    view = getattr(app, '_lnt_main_view', None)
    main = getattr(getattr(view, 'owner', None), 'vulkan_widget', None)
    if main is not None:
        widgets.append(main)
    try:
        from gui.native_sections import manager
        sections = manager(app, create=False)
        if sections is not None:
            widgets.extend(sections.views.values())
    except Exception:                   # noqa: BLE001 - no native sections yet
        pass
    unique = []
    for widget in widgets:
        if widget is not None and all(widget is not seen for seen in unique):
            unique.append(widget)
    return unique


def refresh_pan_cursors(app):
    """Show / remove the pan hand on every native canvas when Pan is (dis)armed."""
    print(f'[PAN] tool {"armed (" + armed_pan_mode(app) + ")" if armed_pan_mode(app) else "disarmed"}')
    for widget in _native_canvases(app):
        gesture = getattr(widget, 'pan_gesture', None)
        apply_pan_cursor(widget, app, bool(gesture is not None and gesture.busy()))


def armed_pan_mode(app):
    """The armed Pan shortcut's mode, or None when the Pan tool is not armed."""
    if app is None or not getattr(app, '_left_pan_shortcut_active', False):
        return None
    mode = getattr(app, '_pan_shortcut_mode', None)
    return mode if mode in PAN_MODES else 'left'


def disarm_pan(app):
    """Esc turns the Pan tool off like any other tool (native only, no VTK).

    Needed because Esc pressed in a native viewport does not always reach the
    main-window shortcut filter: section canvases own an Escape QShortcut and
    floating docks are separate windows.
    """
    if app is None or not getattr(app, '_left_pan_shortcut_active', False):
        return False
    app._left_pan_shortcut_active = False
    saved = getattr(app, '_pan_nav_suspended_button', None)
    if saved is not None:                 # undo a Pan-shortcut mode override
        app._pan_nav_suspended_button = None
        app.panning_button = saved
    status_bar = getattr(app, 'statusBar', None)
    if callable(status_bar):
        try:
            status_bar().showMessage('Pan deactivated (Esc)', 2500)
        except Exception:   # noqa: BLE001 - status bar is cosmetic
            pass
    return True


# Input that can change the pan state, and so the hand cursor.
_CURSOR_EVENTS = (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick,
                  QEvent.MouseButtonRelease, QEvent.MouseMove, QEvent.KeyPress)


class NativePanGesture:
    def __init__(self, pan):
        self._pan = pan                 # pan(dx, dy) in widget pixels
        self.tap_session = False
        self._dragging = False
        self._last = None
        self._swallow_release = False
        self._cursor_target = None      # (app, weakref(widget)) for the hand cursor

    def finish(self):
        self.tap_session = False
        self._dragging = False
        self._last = None
        self._sync_cursor()

    def _sync_cursor(self):
        """Closed hand while panning, open hand while armed, normal otherwise."""
        if self._cursor_target is None:
            return
        app, widget_ref = self._cursor_target
        apply_pan_cursor(widget_ref(), app, self.busy())

    def busy(self):
        return self.tap_session or self._dragging

    def _move(self, pos):
        if self._last is None:
            self._last = pos
            return
        dx, dy = pos.x() - self._last.x(), pos.y() - self._last.y()
        self._last = pos
        if dx or dy:
            self._pan(dx, dy)

    def handle(self, app, ev, widget=None):
        """Consume ``ev`` when it belongs to the armed Pan tool.

        ``widget`` is the canvas receiving ``ev``; it shows the pan hand cursor.
        """
        if widget is not None:
            try:
                self._cursor_target = (app, weakref.ref(widget))
            except TypeError:
                self._cursor_target = None
        consumed = self._handle(app, ev)
        if ev.type() in _CURSOR_EVENTS:      # never on CursorChange & co. (re-entrancy)
            self._sync_cursor()
        return consumed

    def _handle(self, app, ev):
        t = ev.type()
        if t == QEvent.MouseButtonRelease and self._swallow_release \
                and ev.button() == Qt.LeftButton:
            self._swallow_release = False
            return True
        mode = armed_pan_mode(app)
        if mode is None:
            if self.busy():             # disarmed by Esc / another tool
                self.finish()
            return False
        if t == QEvent.KeyPress:
            if self.tap_session:        # any key ends a tap session
                self.finish()
            if ev.key() == Qt.Key_Escape:
                disarm_pan(app)
            return False
        if t in (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick):
            button = ev.button()
            if button == Qt.RightButton and self.tap_session:
                self.finish()
                return True
            if button != Qt.LeftButton or ev.modifiers() & Qt.ShiftModifier:
                return False
            if mode == 'tap':
                # Toggle on the press itself, unconditionally (MicroStation).
                if self.tap_session:
                    self.finish()
                else:
                    self.tap_session, self._last = True, ev.position()
                self._swallow_release = True
            else:
                self._dragging, self._last = True, ev.position()
            return True
        if t == QEvent.MouseMove:
            if self.tap_session:
                self._move(ev.position())
                return True
            if self._dragging:
                if ev.buttons() & Qt.LeftButton:
                    self._move(ev.position())
                else:                   # release was lost (focus change)
                    self.finish()
                return True
            return False
        if t == QEvent.MouseButtonRelease and self._dragging \
                and ev.button() == Qt.LeftButton:
            self.finish()
            return True
        return False
