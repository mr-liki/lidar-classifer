"""Pan shortcut (Left - Click / Tap - Tap) on the native Vulkan main view and
cross/cut section canvases - no VTK. Drives the real GlobalShortcutFilter,
the main view's _InputFilter and a real SectionCanvas."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PySide6.QtCore import QEvent, QObject, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtTest import QTest
from gui import lnt_native
from gui.global_shortcuts import GlobalShortcutFilter
from gui.lnt_open import _InputFilter
from gui.native_sections import NativeSections


def mouse(kind, pos, button=Qt.NoButton, buttons=Qt.NoButton):
    p = QPointF(*pos)
    return QMouseEvent(kind, p, p, button, buttons, Qt.NoModifier)


def press(target, pos):
    return target(mouse(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton))


def move(target, pos, held=False):
    return target(mouse(QEvent.MouseMove, pos, Qt.NoButton,
                        Qt.LeftButton if held else Qt.NoButton))


def release(target, pos):
    return target(mouse(QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton))


def key(filt, app, k, mods=Qt.NoModifier):
    return filt.eventFilter(app, QKeyEvent(QEvent.KeyPress, k, mods))


qapp = QApplication([])
app = QMainWindow()
app.resize(900, 650)
widget = QWidget(app)
widget.setAttribute(Qt.WA_NativeWindow)
widget.setFocusPolicy(Qt.StrongFocus)
app.setCentralWidget(widget)
app.show()
qapp.processEvents()
with tempfile.TemporaryDirectory(prefix='native_pan_') as scratch:
    path = Path(scratch)/'pan.lnt'
    shutil.copy2(ROOT/'native/lnt/test_shading.lnt', path)
    with path.open('rb') as stream:
        header = struct.unpack('<4sHHQ6d', stream.read(64))
    lo, hi = np.array(header[4:7]), np.array(header[7:10])
    native = lnt_native.NativeView(int(widget.winId()), widget.width(), widget.height(), str(path), 64, 8.)
    pans = []
    real_pan = native.pan
    native.pan = lambda dx, dy: (pans.append((dx, dy)), real_pan(dx, dy))
    app.data = None
    app.undo_stack, app.redo_stack = [], []
    app.view_palettes = {}
    app.class_palette = {c: {'show': True, 'color': (100, 150, 200)} for c in range(256)}
    app.section_controller = SimpleNamespace(active_view=0)
    view = QObject()
    view.native, view.app, view.mode_2d = native, app, True
    view.owner = SimpleNamespace(vulkan_widget=widget)
    view.picker = view.classifier = None
    view.set_2d = lambda on: native.set_2d(on)
    app._lnt_main_view = view
    main = _InputFilter(view)
    main_event = lambda ev: main.eventFilter(widget, ev)
    sections = app._native_sections = NativeSections(app)
    filt = GlobalShortcutFilter(app)
    try:
        app.shortcuts = {('alt', 'F2'): {'tool': 'Pan', 'from': None, 'to': None, 'pan_mode': 'left'},
                         ('alt', 'F3'): {'tool': 'Pan', 'from': None, 'to': None, 'pan_mode': 'tap'}}
        widget.setFocus()
        qapp.processEvents()

        # Not armed: 2D left drag must not move the main camera.
        press(main_event, (100, 100)); move(main_event, (140, 120), held=True); release(main_event, (140, 120))
        assert not pans, pans

        # Left - Click Pan: hold-drag pans, plain hover does not.
        assert key(filt, app, Qt.Key_F2, Qt.AltModifier)
        assert app._left_pan_shortcut_active and app._pan_shortcut_mode == 'left'
        assert press(main_event, (100, 100))
        assert move(main_event, (130, 110), held=True)
        assert release(main_event, (130, 110))
        assert pans == [(30.0, 10.0)], pans
        move(main_event, (200, 200))
        assert len(pans) == 1
        print('PASS main view Left - Click Pan: drag pans the native camera, hover does not', flush=True)

        # Tap - Tap Pan: tap starts, cursor follows with NO button held, tap stops.
        pans.clear()
        assert key(filt, app, Qt.Key_F3, Qt.AltModifier)
        assert app._pan_shortcut_mode == 'tap'
        assert press(main_event, (100, 100)) and release(main_event, (100, 100))
        assert main._pan.tap_session
        assert move(main_event, (120, 100)) and move(main_event, (120, 130))
        assert pans == [(20.0, 0.0), (0.0, 30.0)], pans
        assert press(main_event, (120, 130)) and release(main_event, (120, 130))
        assert not main._pan.tap_session
        move(main_event, (300, 300))
        assert len(pans) == 2
        # Esc ends an open session and disarms Pan.
        press(main_event, (10, 10)); release(main_event, (10, 10))
        assert main._pan.tap_session
        key(filt, app, Qt.Key_Escape)
        assert not app._left_pan_shortcut_active
        move(main_event, (50, 50))
        assert not main._pan.tap_session
        assert len(pans) == 2
        print('PASS main view Tap - Tap Pan: tap/follow/tap, Esc stops and disarms', flush=True)

        # Real Esc key delivered to the focused main viewport (not via the
        # main-window shortcut filter) still disarms Pan.
        widget.installEventFilter(main)
        assert key(filt, app, Qt.Key_F3, Qt.AltModifier) and app._left_pan_shortcut_active
        press(main_event, (10, 10)); release(main_event, (10, 10))
        assert main._pan.tap_session
        widget.setFocus(); qapp.processEvents()
        QTest.keyClick(widget, Qt.Key_Escape)
        assert not app._left_pan_shortcut_active and not main._pan.tap_session
        widget.removeEventFilter(main)
        print('PASS real Esc in main viewport disarms Pan', flush=True)

        # Cross-section canvas (cut canvases are the same SectionCanvas class).
        p1, p2 = (lo[0], (lo[1]+hi[1])*.5, 0.), (hi[0], (lo[1]+hi[1])*.5, 0.)
        sections.finalize(p1, p2, max((hi[1]-lo[1])*.5, .1), 0, 0.)
        canvas = next(iter(sections.views.values()))
        qapp.processEvents()
        send = lambda ev: (canvas.event(ev), True)[1]
        canvas.setFocus()
        # Not armed: left-drag must NOT pan the section; middle-drag still does.
        start = canvas.centre.copy()
        send(mouse(QEvent.MouseButtonPress, (100, 100), Qt.LeftButton, Qt.LeftButton))
        send(mouse(QEvent.MouseMove, (60, 70), Qt.NoButton, Qt.LeftButton))
        send(mouse(QEvent.MouseButtonRelease, (60, 70), Qt.LeftButton, Qt.NoButton))
        assert np.array_equal(canvas.centre, start), (start, canvas.centre)
        send(mouse(QEvent.MouseButtonPress, (100, 100), Qt.MiddleButton, Qt.MiddleButton))
        send(mouse(QEvent.MouseMove, (60, 100), Qt.NoButton, Qt.MiddleButton))
        send(mouse(QEvent.MouseButtonRelease, (60, 100), Qt.MiddleButton, Qt.NoButton))
        assert canvas.centre[0] > start[0]
        print('PASS cross-section canvas unarmed: left-drag does not pan, middle-drag does', flush=True)
        assert key(filt, app, Qt.Key_F3, Qt.AltModifier)
        start = canvas.centre.copy()
        send(mouse(QEvent.MouseButtonPress, (80, 80), Qt.LeftButton, Qt.LeftButton))
        send(mouse(QEvent.MouseButtonRelease, (80, 80), Qt.LeftButton, Qt.NoButton))
        assert canvas.pan_gesture.tap_session
        send(mouse(QEvent.MouseMove, (120, 60)))
        moved = canvas.centre.copy()
        assert moved[0] < start[0] and moved[2] < start[2], (start, moved)
        send(mouse(QEvent.MouseButtonPress, (120, 60), Qt.LeftButton, Qt.LeftButton))
        send(mouse(QEvent.MouseButtonRelease, (120, 60), Qt.LeftButton, Qt.NoButton))
        send(mouse(QEvent.MouseMove, (200, 200)))
        assert np.array_equal(canvas.centre, moved)
        assert key(filt, app, Qt.Key_F2, Qt.AltModifier)
        send(mouse(QEvent.MouseButtonPress, (100, 100), Qt.LeftButton, Qt.LeftButton))
        send(mouse(QEvent.MouseMove, (60, 100), Qt.NoButton, Qt.LeftButton))
        send(mouse(QEvent.MouseButtonRelease, (60, 100), Qt.LeftButton, Qt.NoButton))
        assert canvas.centre[0] > moved[0]

        # Real Esc in the section canvas: its Escape QShortcut consumes the
        # key, and must disarm Pan and end an open tap session.
        assert key(filt, app, Qt.Key_F3, Qt.AltModifier) and app._left_pan_shortcut_active
        send(mouse(QEvent.MouseButtonPress, (80, 80), Qt.LeftButton, Qt.LeftButton))
        send(mouse(QEvent.MouseButtonRelease, (80, 80), Qt.LeftButton, Qt.NoButton))
        assert canvas.pan_gesture.tap_session
        canvas.window().activateWindow(); canvas.setFocus(); qapp.processEvents()
        QTest.keyClick(canvas, Qt.Key_Escape)
        qapp.processEvents()
        assert not app._left_pan_shortcut_active, 'Esc did not disarm Pan in section'
        assert not canvas.pan_gesture.tap_session
        before = canvas.centre.copy()
        send(mouse(QEvent.MouseButtonPress, (100, 100), Qt.LeftButton, Qt.LeftButton))
        send(mouse(QEvent.MouseMove, (60, 70), Qt.NoButton, Qt.LeftButton))
        send(mouse(QEvent.MouseButtonRelease, (60, 70), Qt.LeftButton, Qt.NoButton))
        assert np.array_equal(canvas.centre, before)
        print('PASS real Esc in cross-section canvas disarms Pan; left-drag no longer pans', flush=True)
        print('PASS cross-section canvas: Tap - Tap and Left - Click Pan move the native section camera', flush=True)
    finally:
        sections.close()
        store = getattr(app, '_lnt_section_store', None)
        if store is not None:
            store.release()
        native.close()
        app.close()
        qapp.processEvents()
