"""Native section classification: screen mapping vs real Vulkan pixels, and every
drawable tool's selection/undo/visibility/From-class rules on store indices."""
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from gui.native_sections import NativeSections, CUT_SLOT
from gui import lnt_native

qapp = QApplication([])
app = QMainWindow()
app.resize(1200, 800)
main_canvas = QWidget(app)
main_canvas.setAttribute(Qt.WA_NativeWindow)
app.setCentralWidget(main_canvas)
app.show()
qapp.processEvents()
native = lnt_native.NativeView(int(main_canvas.winId()), 600, 400,
                               str(ROOT / 'native/lnt/test_shading.lnt'), 64, 8.)
app._lnt_main_view = SimpleNamespace(native=native)
app.section_controller = SimpleNamespace(active_view=0)
app.section_buffer = 0.
app.view_palettes = {}
app.undo_stack, app.redo_stack = [], []
app.class_palette = {c: {'color': (255, 255, 255), 'show': True} for c in (1, 2, 5, 6)}

# A grid of records on the section plane y=0: h = 0..10, z = 0..10 (1 m pitch).
hh, zz = np.meshgrid(np.arange(11.), np.arange(11.))
xyz = np.column_stack([hh.ravel(), np.zeros(hh.size), zz.ravel()])
classes = np.full(len(xyz), 2, dtype=np.uint8)
classes[::7] = 5
store = SimpleNamespace(view=app._lnt_main_view, ensure_for_section=Mock(return_value=True),
                        data={'xyz': xyz, 'classification': classes,
                              'rgb': np.ones((len(xyz), 3), dtype=np.float32),
                              'intensity': np.zeros(len(xyz), dtype=np.float32)})
app._lnt_section_store = store
store.show = Mock()
store.flush = Mock(return_value=0)
store.preview_live = Mock()
app.data = store.data
sections = NativeSections(app)
app._native_sections = sections


def reset(cls=2):
    classes[:] = cls
    classes[::7] = 5
    app.undo_stack.clear()
    sections.refresh_edits()


def click(canvas, h, z, button=Qt.LeftButton, release=True):
    x, y = canvas.screen_from_section(h, z)
    canvas.classifier.press(QPointF(x, y), button)
    if release:
        canvas.classifier.release(QPointF(x, y), button)


def changed():
    return set(np.flatnonzero(classes == 6).tolist())


def at(h, z):
    return int(round(z)) * 11 + int(round(h))


try:
    sections.open(0, draw=False)
    sections.finalize((0, 0, 0), (10, 0, 0), .5, 0, 0.)
    canvas = sections.views[1]
    canvas.resize(400, 400)
    qapp.processEvents()
    canvas.fit()
    assert len(canvas.indices) == len(xyz)

    # ---- mapping round trip + real pixels ---------------------------------
    for h, z in [(0, 0), (10, 10), (3.25, 7.5)]:
        x, y = canvas.screen_from_section(h, z)
        back = canvas.section_at(x, y)
        assert np.allclose(back, (h, z)), (back, h, z)
    # One bright class at the left-bottom corner record, all others hidden.
    classes[:] = 2
    classes[at(0, 0)] = 1
    sections.refresh_edits()
    canvas.apply_palette({1: {'color': (255, 0, 0), 'show': True},
                          2: {'color': (0, 0, 0), 'show': False}}, 'class')
    canvas.backend.set_point_size(9.)
    canvas._dirty = True
    for _ in range(6):
        canvas.render()
        qapp.processEvents()
        time.sleep(.05)
    frame = canvas.backend.capture_frame()
    assert frame is not None, 'capture_frame unavailable'
    red = (frame[..., 0] > 200) & (frame[..., 1] < 80) & (frame[..., 2] < 80)
    ys, xs = np.nonzero(red)
    assert xs.size, 'record not rendered'
    ratio = frame.shape[1] / canvas.width()
    expect = np.array(canvas.screen_from_section(0., 0.)) * ratio
    got = np.array([xs.mean(), ys.mean()])
    assert np.abs(got - expect).max() <= 3 * ratio, (got, expect)
    print(f'PASS screen mapping matches rendered pixels (expected {expect.round(1)}, got {got.round(1)})', flush=True)
    canvas.backend.set_point_size(3.)
    canvas.apply_palette(app.class_palette, 'class')

    # Drive the actual cut mouse handlers, including subpixel depth where
    # orange edges overlap the fixed white centre.
    from PySide6.QtGui import QMouseEvent
    def cut_event(kind, x, button=Qt.NoButton):
        pos = QPointF(x, canvas.height() * .5)
        event = QMouseEvent(kind, pos, canvas.mapToGlobal(pos.toPoint()).toPointF(),
                            button, button, Qt.NoModifier)
        QApplication.sendEvent(canvas, event)
        canvas.render()
        qapp.processEvents()

    def white_at(x):
        frame = canvas.backend.capture_frame()
        assert frame is not None
        px = round(x * frame.shape[1] / canvas.width())
        mask = np.all(frame[..., :3] > 245, axis=2)
        return bool(np.any(mask[:, max(0, px-2):px+3].sum(axis=0) > frame.shape[0] * .9))

    for mode in ('class', 'shaded_class'):
        canvas.apply_palette(app.class_palette, mode)
        assert sections.arm_cut(False, 1)
        for x in (51.2, 130.7, 249.3):
            cut_event(QEvent.MouseMove, x)
            assert white_at(x), (mode, 'cursor preview missing', x)
        cut_event(QEvent.MouseButtonPress, 249.3, Qt.LeftButton)
        for dx in (0., .1, .4, 20.):
            cut_event(QEvent.MouseMove, 249.3 + dx)
            assert white_at(249.3), (mode, 'fixed centre obscured', dx)
        sections.cancel_cut()
    canvas.apply_palette(app.class_palette, 'class')
    print('PASS cut mouse preview follows cursor and keeps white centre at overlapping depths', flush=True)

    # ---- rectangle (click-click and drag) ----------------------------------
    app.to_class, app.from_classes = 6, None
    app.active_classify_tool = 'rectangle'
    reset()
    click(canvas, 1.5, 1.5)
    click(canvas, 3.5, 3.5)
    expect = {at(h, z) for h in (2, 3) for z in (2, 3)}
    assert changed() == expect, changed()
    assert len(app.undo_stack) == 1 and set(app.undo_stack[0]['indices'].tolist()) == expect
    reset()
    x1, y1 = canvas.screen_from_section(5.5, 5.5)
    x2, y2 = canvas.screen_from_section(6.5, 8.5)
    canvas.classifier.press(QPointF(x1, y1), Qt.LeftButton)
    canvas.classifier._press_time -= .2  # Deliberate drag beyond BRUSH's tap grace.
    canvas.classifier.move(QPointF(x2, y2))
    canvas.classifier.release(QPointF(x2, y2), Qt.LeftButton)
    assert changed() == {at(6, z) for z in (6, 7, 8)}, changed()
    print('PASS rectangle click-click and drag select exact records, one undo step each', flush=True)

    # ---- From classes and hidden classes are protected ----------------------
    reset()
    app.from_classes = [5]
    click(canvas, -.5, -.5)
    click(canvas, 10.5, 10.5)
    assert changed() == set(range(0, len(xyz), 7)), 'From filter'
    app.from_classes = None
    reset()
    canvas.apply_palette({2: {'color': (1, 1, 1), 'show': True},
                          5: {'color': (1, 1, 1), 'show': False}}, 'class')
    click(canvas, -.5, -.5)
    click(canvas, 10.5, 10.5)
    assert changed() == set(range(len(xyz))) - set(range(0, len(xyz), 7)), 'hidden class protected'
    canvas.apply_palette(app.class_palette, 'class')
    print('PASS From-class filter and hidden-class protection', flush=True)

    # ---- polygon, circle, lines, brush, point --------------------------------
    reset()
    app.active_classify_tool = 'polygon'
    for h, z in [(.5, .5), (4.6, .5), (.5, 4.6)]:
        click(canvas, h, z)
    x, y = canvas.screen_from_section(.5, 4.6)
    assert canvas.classifier.double_click(QPointF(x, y))
    expect = {at(h, z) for h in range(1, 5) for z in range(1, 5) if h + z <= 5}
    assert changed() == expect, changed()
    reset()
    app.active_classify_tool = 'circle'
    click(canvas, 5, 5)
    click(canvas, 6.1, 5)
    assert changed() == {at(5, 5), at(4, 5), at(6, 5), at(5, 4), at(5, 6)}, changed()
    reset()
    app.active_classify_tool = 'above_line'
    click(canvas, 1.5, 7.5)
    click(canvas, 3.5, 7.5)
    assert changed() == {at(h, z) for h in (2, 3) for z in (8, 9, 10)}, changed()
    reset()
    app.active_classify_tool = 'below_line'
    click(canvas, 1.5, 1.5)
    click(canvas, 3.5, 1.5)
    assert changed() == {at(h, z) for h in (2, 3) for z in (0, 1)}, changed()
    reset()
    app.active_classify_tool = 'parallel_line'
    app.parallel_line_total_width = 1.2
    click(canvas, 0, 5)
    click(canvas, 10, 5)
    assert changed() == {at(h, 5) for h in range(11)}, changed()
    reset()
    app.active_classify_tool = 'brush'
    app.brush_preview_px = .4 / canvas.world_per_pixel()       # 0.4 m
    x1, y1 = canvas.screen_from_section(2, 2)
    canvas.classifier.move(QPointF(x1, y1))
    assert not canvas.classifier.overlay.shapes, 'idle Brush must have no hover preview'
    canvas.classifier.press(QPointF(x1, y1), Qt.LeftButton)
    assert all(s[0] == 'brush_cursor_px' for s in canvas.classifier.overlay.shapes)
    for h in np.linspace(2, 6, 40):
        # Simulate moves across BRUSH's 30 Hz query batches rather than
        # compressing the entire stroke into one synchronous test instant.
        canvas.classifier._brush_last_query_time -= .04
        canvas.classifier.move(QPointF(*canvas.screen_from_section(h, 2)))
        assert all(s[0] == 'brush_cursor_px' for s in canvas.classifier.overlay.shapes), 'no painted trail'
    assert changed() == {at(h, 2) for h in range(2, 7)}, 'brush must paint live before release'
    assert not app.undo_stack, 'incomplete stroke must not enter history'
    canvas.classifier.release(QPointF(*canvas.screen_from_section(6, 2)), Qt.LeftButton)
    assert changed() == {at(h, 2) for h in range(2, 7)}, changed()
    assert len(app.undo_stack) == 1, 'brush stroke must be one undo step'
    assert app.active_classify_tool == 'brush' and not canvas.classifier.overlay.shapes
    reset()
    baseline = classes.copy()
    canvas.classifier.press(QPointF(x1, y1), Qt.LeftButton)
    assert changed(), 'next stroke must work without reactivation'
    canvas.classifier.key(Qt.Key_Escape)
    np.testing.assert_array_equal(classes, baseline)
    assert not app.undo_stack and not app._native_brush_transaction_active
    assert app.active_classify_tool is None and not canvas.classifier.overlay.shapes
    print('PASS BRUSH parity: live paint, cursor only during stroke, release commit, persistent tool, Escape rollback', flush=True)
    reset()
    app.active_classify_tool = 'point'
    click(canvas, 7.02, 3.01)
    assert changed() == {at(7, 3)}, changed()
    reset()
    app.active_classify_tool = 'freehand'
    pts = [(.5, .5), (3.5, .5), (3.5, 2.5), (.5, 2.5)]
    canvas.classifier.press(QPointF(*canvas.screen_from_section(*pts[0])), Qt.LeftButton)
    for p in pts[1:]:
        canvas.classifier.move(QPointF(*canvas.screen_from_section(*p)))
    canvas.classifier.release(QPointF(*canvas.screen_from_section(*pts[0])), Qt.LeftButton)
    assert changed() == {at(h, z) for h in (1, 2, 3) for z in (1, 2)}, changed()
    print('PASS polygon, circle, above/below/parallel line, brush stroke, point and freehand', flush=True)

    # ---- viewport limit, right-click cancel, tool off -> pan ------------------
    reset()
    app.active_classify_tool = 'rectangle'
    canvas.scale = 2.
    canvas.centre[:] = (5, 0, 5)
    lo_h, hi_h, lo_z, hi_z = canvas.visible_extent()
    click(canvas, -100, -100)
    click(canvas, 100, 100)
    hz = canvas.local[:, [canvas.axis, 2]]
    inside = (hz[:, 0] >= lo_h) & (hz[:, 0] <= hi_h) & (hz[:, 1] >= lo_z) & (hz[:, 1] <= hi_z)
    assert changed() == set(canvas.indices[inside].tolist()) and 0 < len(changed()) < len(xyz)
    canvas.fit()
    reset()
    click(canvas, 1, 1)
    click(canvas, 1, 1, Qt.RightButton)
    assert not canvas.classifier.busy()
    app.active_classify_tool = None
    assert not canvas.classifier.press(QPointF(10, 10), Qt.LeftButton)
    print('PASS viewport-limited selection, right-click cancel, no tool leaves navigation', flush=True)

    # ---- cut canvas classifies too, and edits reach the other view ------------
    reset()
    canvas.cut_centre = 5.
    sections.finish_cut(canvas, 5.4)
    cut = sections.views[CUT_SLOT]
    cut.resize(300, 300)
    qapp.processEvents()
    cut.fit()
    assert set(cut.indices.tolist()) == {at(5, z) for z in range(11)}
    app.active_classify_tool = 'rectangle'
    click(cut, -1, 2.5)
    click(cut, 1, 4.5)
    assert changed() == {at(5, 3), at(5, 4)}, changed()
    assert np.array_equal(sections._classes[1], classes[canvas.indices])
    print('PASS cut view classification on store indices refreshes the cross view', flush=True)

    # A volume, rather than the flat classification fixture, exercises repeated
    # orthogonal cuts and the sign change from each 90 degree rotation.
    import builtins
    real_import = builtins.__import__
    def native_import(name, *args, **kwargs):
        assert name != 'vtk' and not name.startswith('vtkmodules'), name
        return real_import(name, *args, **kwargs)
    builtins.__import__ = native_import
    try:
        xx, yy, zz = np.meshgrid(np.arange(11.), np.arange(-5., 6.), np.arange(4.))
        volume = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        store.data = {'xyz': volume, 'classification': np.full(len(volume), 2, np.uint8),
                      'rgb': np.ones((len(volume), 3)), 'intensity': np.zeros(len(volume))}
        app.data = store.data
        sections.finalize((0, 0, 0), (10, 0, 0), 5., 0, 0.)
        cross = sections.views[1]
        cross.cut_centre = 5.
        sections.finish_cut(cross, 9.)
        canvas = sections.views[CUT_SLOT]
        original_handle = canvas.backend._handle
        assert sections.arm_cut(True)
        for cycle in range(3):
            before_local, before_ids, axis = canvas.local.copy(), canvas.indices.copy(), canvas.axis
            centre = float(np.median(before_local[:, axis]))
            half = max(1., float(np.ptp(before_local[:, axis])) * .35)
            cx = canvas.screen_from_section(centre, canvas.centre[2])[0]
            cut_event(QEvent.MouseMove, cx)
            assert white_at(cx), 'nested white cursor preview missing'
            cut_event(QEvent.MouseButtonPress, cx, Qt.LeftButton)
            saved_centre = canvas.cut_centre
            assert sections.arm_cut(True) and canvas.cut_centre == saved_centre
            ex = canvas.screen_from_section(centre + half, canvas.centre[2])[0]
            cut_event(QEvent.MouseMove, ex)
            assert white_at(cx), 'nested fixed centre missing'
            cut_event(QEvent.MouseButtonPress, ex, Qt.LeftButton)
            keep = np.abs(before_local[:, axis]-centre) <= half + 1e-10
            expected = np.column_stack((before_local[keep, 1-axis],
                                        -before_local[keep, axis], before_local[keep, 2]))
            assert np.array_equal(canvas.indices, before_ids[keep])
            assert np.allclose(canvas.local, expected)
            assert canvas.axis == 0 and canvas.cut_armed and canvas.cut_centre is None
            assert canvas.backend._handle == original_handle
            # Widening the current result uses its saved parent; it cannot
            # recursively filter an already narrowed result.
            sections.update_cut_depth(half * 2. + .5)
            wider = np.abs(before_local[:, axis]-centre) <= half + .25
            assert np.array_equal(canvas.indices, before_ids[wider])
        kept_ids = canvas.indices.copy()
        canvas.cut_centre = 1e6
        sections.finish_cut(canvas, 1e6 + 1.)
        assert np.array_equal(canvas.indices, kept_ids) and canvas.cut_armed
        canvas.escape()
        assert not canvas.cut_armed and not sections._nested_cut_active
        assert np.array_equal(canvas.indices, kept_ids)
        assert sections.arm_cut(True)
        cut_event(QEvent.MouseButtonPress, 100., Qt.RightButton)
        assert not canvas.cut_armed and np.array_equal(canvas.indices, kept_ids)
        print('PASS repeated native Cut-in-Cut: rotation, point IDs, preview, depth edits, empty range and cancel; no VTK imports', flush=True)
        activations = []
        def activate_native(tool):
            assert app._right_click_reactivating
            activations.append(tool)
            app.active_classify_tool = tool
        app.set_classify_tool = activate_native
        app.active_classify_tool = 'rectangle'
        app.from_classes, app.to_class = [2], 6
        cut_event(QEvent.MouseButtonPress, 100., Qt.LeftButton)
        assert app.last_classify_from_classes == [2] and app.last_classify_to_class == 6
        # Returning to idle must not resurrect a partial selection on any view.
        app.active_classify_tool = None
        cross.classifier.points = [(1., 1.)]
        app.from_classes, app.to_class = [5], 9
        cut_event(QEvent.MouseButtonPress, 100., Qt.RightButton)
        assert activations == ['rectangle'] and app.active_classify_tool == 'rectangle'
        assert app.from_classes == [2] and app.to_class == 6
        assert not canvas.classifier.busy() and not cross.classifier.busy()
        assert not app._right_click_reactivating
        # An active gesture consumes right-click before idle reactivation.
        cut_event(QEvent.MouseButtonPress, 100., Qt.LeftButton)
        cut_event(QEvent.MouseButtonPress, 100., Qt.RightButton)
        assert activations == ['rectangle'] and not canvas.classifier.busy()
        app.active_classify_tool = None
        app.last_classify_tool = None
        cut_event(QEvent.MouseButtonPress, 100., Qt.RightButton)
        assert app.active_classify_tool is None
        app.last_classify_tool = 'brush'
        canvas.measurement = SimpleNamespace(press=lambda *args: True)
        cut_event(QEvent.MouseButtonPress, 100., Qt.RightButton)
        del canvas.measurement
        assert app.active_classify_tool is None and activations == ['rectangle']
        print('PASS native right-click recalls last-used classes, clears old gestures, respects active gestures and measurement, and ignores missing history; no VTK imports', flush=True)
    finally:
        builtins.__import__ = real_import
finally:
    sections.close()
    native.close()
    app.close()
    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qapp.quit()
