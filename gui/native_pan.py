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

from PySide6.QtCore import QEvent, Qt

PAN_MODES = ('left', 'tap')


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


class NativePanGesture:
    def __init__(self, pan):
        self._pan = pan                 # pan(dx, dy) in widget pixels
        self.tap_session = False
        self._dragging = False
        self._last = None
        self._swallow_release = False

    def finish(self):
        self.tap_session = False
        self._dragging = False
        self._last = None

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

    def handle(self, app, ev):
        """Consume ``ev`` when it belongs to the armed Pan tool."""
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
