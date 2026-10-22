"""Stage 4 Vulkan measurement parity: production geometry/controller/overlay tests.

Runs headless: input and painter are stubbed, but the production NativeMeasure
controller and CanvasOverlay.paintEvent body execute unchanged.
"""
import ast
import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gui.native_measure_geometry import (
    measurement_distance, measurement_labels, measurement_text,
)


class Point:
    def __init__(self, x, y):
        self._x, self._y = x, y

    def x(self):
        return self._x

    def y(self):
        return self._y


class Status:
    def __init__(self):
        self.messages = []

    def showMessage(self, message, timeout=0):
        self.messages.append(str(message))


class App:
    def __init__(self):
        self.status = Status()

    def statusBar(self):
        return self.status


class Canvas:
    def __init__(self):
        self.owner = types.SimpleNamespace(app=App())
        self.picks = []

    def section_at(self, x, y):
        result = (float(x), float(y))
        self.picks.append(result)
        return result


class Overlay:
    def __init__(self, canvas):
        self.canvas = canvas
        self.shapes = []
        self._timer = types.SimpleNamespace(stop=lambda: None)

    def set_shapes(self, shapes):
        self.shapes = shapes

    def close(self):
        pass


@pytest.fixture
def native_module(monkeypatch):
    qt = types.ModuleType('PySide6.QtCore')

    class Qt:
        LeftButton = 1
        RightButton = 2
        MiddleButton = 4
        Key_Escape = 27
        Key_Backspace = 8
        Key_Delete = 127

    qt.QEvent = type('QEvent', (), {'MouseButtonPress': 1,
                                    'MouseButtonRelease': 2,
                                    'MouseButtonDblClick': 3,
                                    'MouseMove': 4,
                                    'KeyPress': 5})

    class QObject:
        def __init__(self, *a, **kw):
            pass

    qt.QObject = QObject
    qt.Qt = Qt
    monkeypatch.setitem(sys.modules, 'PySide6', types.ModuleType('PySide6'))
    monkeypatch.setitem(sys.modules, 'PySide6.QtCore', qt)
    overlay = types.ModuleType('gui.native_section_classify')
    overlay.CanvasOverlay = Overlay
    monkeypatch.setitem(sys.modules, 'gui.native_section_classify', overlay)
    settings = types.ModuleType('gui.measure_settings_dialog')
    settings.load_measure_settings = lambda: {
        'line': {'unit': 'm', 'color': (1., 1., 0.), 'width': 3},
        'path': {'unit': 'm', 'color': (1., .5, 0.), 'width': 3},
        'cross_section': {'unit': 'm', 'color': (1., 1., 0.), 'width': 3},
    }
    monkeypatch.setitem(sys.modules, 'gui.measure_settings_dialog', settings)
    spec = importlib.util.spec_from_file_location(
        'stage4_measure_controller', ROOT / 'gui' / 'native_measure.py'
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, Qt


def click(measure, qt, point, button=None):
    return measure.press(Point(*point), qt.LeftButton if button is None else button)


def test_line_accepts_four_vertices_and_single_completed_measurement(native_module):
    mod, qt = native_module
    m = mod.NativeMeasure(Canvas())
    points = [(0, 0), (3, 4), (6, 4), (6, 9)]
    for n, point in enumerate(points, 1):
        assert click(m, qt, point)
        assert len(m.points) == n
        assert m.completed == []
    kind, live_points, options = m.overlay.shapes[-1]
    assert kind == 'measurement' and options['complete'] is False and options['path'] is False
    assert live_points == points
    assert click(m, qt, (100, 100), qt.RightButton)
    assert m.active and m.points == [] and m.hover is None
    assert m.completed == [points] and m.completed_modes == [False]
    assert m.distance(m.completed[0]) == pytest.approx(13.0)
    kind, saved, options = m.overlay.shapes[0]
    assert saved == points and options['complete'] and not options['path']
    assert any('13.000 m' in s for s in m.app.statusBar().messages)


def test_native_line_can_start_another_stroke_without_rearming(native_module):
    mod, qt = native_module
    m = mod.NativeMeasure(Canvas())
    for a, b in [[(0, 0), (3, 4)], [(100, 0), (100, 8)]]:
        click(m, qt, a)
        click(m, qt, b)
        click(m, qt, (999, 999), qt.RightButton)
    assert m.completed == [[(0., 0.), (3., 4.)], [(100., 0.), (100., 8.)]]
    assert m.active and len(m.overlay.shapes) == 2


def test_path_keeps_multivertex_and_path_mode(native_module):
    mod, qt = native_module
    m = mod.NativeMeasure(Canvas(), path=True)
    for point in [(0, 0), (3, 4), (6, 4), (6, 9)]:
        click(m, qt, point)
    assert m.completed == []
    assert m.overlay.shapes[-1][-1]['path'] is True
    click(m, qt, (900, 900), qt.RightButton)
    assert m.completed_modes == [True]
    assert m.completed[0][-1] == (6., 9.)


def test_double_click_finishes_line_without_duplicating_endpoint(native_module):
    mod, qt = native_module
    m = mod.NativeMeasure(Canvas())
    click(m, qt, (0, 0))
    click(m, qt, (3, 4))
    assert m.double_click(Point(3, 4), qt.LeftButton)
    assert m.completed == [[(0., 0.), (3., 4.)]]


def test_double_click_finishes_path_and_accepts_new_endpoint(native_module):
    mod, qt = native_module
    m = mod.NativeMeasure(Canvas(), path=True)
    click(m, qt, (0, 0))
    click(m, qt, (3, 4))
    assert m.double_click(Point(6, 4), qt.LeftButton)
    assert m.completed == [[(0., 0.), (3., 4.), (6., 4.)]]


def test_duplicate_clicks_create_no_zero_length_legs(native_module):
    mod, qt = native_module
    m = mod.NativeMeasure(Canvas())
    click(m, qt, (0, 0))
    click(m, qt, (0, 0))
    click(m, qt, (3, 4))
    click(m, qt, (3, 4))
    click(m, qt, (500, 0), qt.RightButton)
    assert m.completed == [[(0., 0.), (3., 4.)]]


def test_right_click_without_two_distinct_points_cancels_draft(native_module):
    mod, qt = native_module
    m = mod.NativeMeasure(Canvas())
    click(m, qt, (0, 0))
    click(m, qt, (33, 44), qt.RightButton)
    assert m.points == [] and m.completed == [] and m.active


def test_line_live_cursor_preview_and_label_does_not_mutate_points(native_module):
    mod, qt = native_module
    m = mod.NativeMeasure(Canvas())
    click(m, qt, (0, 0))
    click(m, qt, (3, 4))
    assert m.move(Point(6, 4))
    shape = m.overlay.shapes[-1]
    assert shape[1] == [(0., 0.), (3., 4.), (6., 4.)]
    assert shape[2]['complete'] is False
    assert m.points == [(0., 0.), (3., 4.)]
    click(m, qt, (10, 10), qt.RightButton)
    assert m.completed[0][-1] == (3., 4.)  # right-click never appends cursor XY



def test_hover_on_existing_last_vertex_does_not_duplicate_preview_endpoint(native_module):
    mod, qt = native_module
    m = mod.NativeMeasure(Canvas(), path=True)
    click(m, qt, (0, 0))
    click(m, qt, (3, 4))
    m.move(Point(3, 4))
    assert m.overlay.shapes[-1][1] == [(0., 0.), (3., 4.)]
    assert measurement_labels(m.overlay.shapes[-1][1], path=True) == [(1, 5.0)]

def test_completed_measurements_preserve_original_mode_after_mode_switch(native_module):
    mod, qt = native_module
    m = mod.NativeMeasure(Canvas())
    click(m, qt, (0, 0));click(m, qt, (3, 4));click(m, qt, (0, 0), qt.RightButton)
    m.path = True
    click(m, qt, (10, 0));click(m, qt, (13, 4));click(m, qt, (0, 0), qt.RightButton)
    assert m.completed_modes == [False, True]
    assert [data['path'] for _, _, data in m.overlay.shapes] == [False, True]


def test_cross_section_remains_two_click_measurement(native_module):
    mod, qt = native_module
    m = mod.NativeMeasure(Canvas())
    m.style_key = 'cross_section'
    click(m, qt, (0, 0))
    assert m.completed == []
    click(m, qt, (3, 4))
    assert len(m.completed) == 1 and m.points == []
    assert m.double_click(Point(6, 4), qt.LeftButton) is False
    assert m.completed_modes == [False]


def test_native_measure_point_undo_redo_and_after_finish(native_module):
    mod, qt = native_module
    m = mod.NativeMeasure(Canvas())
    for point in [(0, 0), (3, 4), (6, 4)]:
        click(m, qt, point)
    assert m.history(redo=False)
    assert m.points == [(0., 0.), (3., 4.)]
    assert m.history(redo=True)
    assert m.points == [(0., 0.), (3., 4.), (6., 4.)]
    click(m, qt, (999, 999), qt.RightButton)
    assert m.completed_modes == [False]
    assert not m.undone
    assert m.history(redo=False)
    assert m.completed == [] and m.completed_modes == []
    assert m.history(redo=True)
    assert len(m.completed) == 1


def test_clear_all_measurements_is_undoable_and_redoable(native_module):
    mod, qt = native_module
    m = mod.NativeMeasure(Canvas())
    click(m, qt, (0, 0)); click(m, qt, (3, 4)); click(m, qt, (0, 0), qt.RightButton)
    m.grids.append({
        'points': [(0., 0.), (2., 0.), (2., 2.), (0., 2.)],
        'label': 'G01', 'area': 4., 'kind': 'grid',
    })

    m.clear()
    assert not m.completed and not m.grids
    assert m.can_history(redo=False)

    assert m.history(redo=False)
    assert len(m.completed) == 1 and len(m.grids) == 1
    assert m.can_history(redo=True)

    assert m.history(redo=True)
    assert not m.completed and not m.grids


def test_measure_history_cannot_redo_undone_point_into_new_measurement(native_module):
    mod, qt = native_module
    m = mod.NativeMeasure(Canvas())
    click(m, qt, (0, 0))
    click(m, qt, (3, 4))
    click(m, qt, (6, 4))
    m.history(redo=False)
    click(m, qt, (99, 99), qt.RightButton)
    assert m.undone == []
    assert m.completed[0] == [(0., 0.), (3., 4.)]


def test_grid_mode_unchanged_and_no_line_points(native_module):
    mod, qt = native_module
    m = mod.NativeMeasure(Canvas(), grid_area=True)
    m._pick_area = lambda *_: {'points': [(0, 0), (2, 0), (2, 2), (0, 2)],
                                'label': 'G01', 'area': 4.}
    click(m, qt, (1, 1))
    assert len(m.grids) == 1 and m.points == [] and m.completed == []
    click(m, qt, (1, 1))
    assert m.grids == []


def test_cancel_preserves_completed_measurements(native_module):
    mod, qt = native_module
    m = mod.NativeMeasure(Canvas())
    click(m, qt, (0, 0));click(m, qt, (3, 4));click(m, qt, (1, 0), qt.RightButton)
    click(m, qt, (200, 200))
    m.cancel()
    assert m.active is False and m.points == [] and len(m.completed) == 1


def test_distance_labels_path_are_cumulative_at_each_vertex():
    points = [(0, 0), (3, 4), (6, 4), (6, 9)]
    assert measurement_labels(points, path=True) == [(1, 5.0), (2, 8.0), (3, 13.0)]
    assert measurement_labels(points, path=False) == [(3, 13.0)]


def test_path_two_vertices_has_one_distance_label():
    points = [(0, 0), (3, 4)]
    assert measurement_labels(points, path=True) == [(1, 5.0)]
    assert measurement_labels(points, path=False) == [(1, 5.0)]


def test_zero_length_duplicate_segments_never_create_label_duplicates():
    points = [(0, 0), (3, 4), (3, 4), (6, 4)]
    assert measurement_labels(points, path=True) == [(1, 5.0), (3, 8.0)]
    assert measurement_labels(points, path=False) == [(3, 8.0)]


def test_labels_support_km_with_no_rounding_drift():
    pts = [(0, 0), (300, 400), (900, 400)]
    assert measurement_distance(pts) == pytest.approx(1100.0)
    assert measurement_labels(pts, path=True) == [(1, 500.0), (2, 1100.0)]
    assert measurement_text(1100.0, 'km') == '1.100 km'
    assert measurement_text(1100.0, 'm') == '1100.000 m'


def test_world_coordinate_precision_with_large_offsets():
    pts = [(500_000.0, 5_100_000.0), (500_003.0, 5_100_004.0),
           (500_006.0, 5_100_004.0)]
    assert measurement_distance(pts) == 8.0
    assert measurement_labels(pts, path=True)[-1] == (2, 8.0)


def test_empty_single_and_identical_measurements_have_no_visible_label():
    assert measurement_labels([], path=True) == []
    assert measurement_labels([(0, 0)], path=False) == []
    assert measurement_labels([(0, 0), (0, 0)], path=True) == []


class QPointF:
    def __init__(self, x=0., y=0.):
        self._x, self._y = float(x), float(y)

    def __add__(self, other):
        return QPointF(self._x + other._x, self._y + other._y)

    def __mul__(self, k):
        return QPointF(self._x * k, self._y * k)

    def x(self):
        return self._x

    def y(self):
        return self._y


class FakeQColor:
    def __init__(self, *a):
        pass


class FakePen:
    def __init__(self, *a):
        pass


class FakeFont:
    def setPointSize(self, *a):
        pass


class FakePainter:
    Antialiasing = 1

    def __init__(self, instance):
        self.instance = instance
        self.texts = []
        instance.last_painter = self

    def setRenderHint(self, *a):
        pass

    def setPen(self, *a):
        pass

    def setBrush(self, *a):
        pass

    def pen(self):
        return FakePen()

    def font(self):
        return FakeFont()

    def setFont(self, *a):
        pass

    def drawPolyline(self, *a):
        pass

    def drawEllipse(self, *a):
        pass

    def drawText(self, point, text):
        self.texts.append((point.x(), point.y(), text))

    def end(self):
        pass


class FakeQt:
    DashLine = 1
    DotLine = 2
    DashDotLine = 3
    DashDotDotLine = 4
    SolidLine = 5
    NoBrush = 6


@pytest.fixture(scope='module')
def real_overlay_paint_event():
    """Compile the original paintEvent method body from production source.

    This exercises the actual label rendering loop while faking only Qt's
    graphics surface; it does not copy or reimplement the UI algorithm.
    """
    source = (ROOT / 'gui' / 'native_section_classify.py').read_text(encoding='utf8')
    tree = ast.parse(source)
    overlay = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == 'CanvasOverlay')
    func = next(node for node in overlay.body
                if isinstance(node, ast.FunctionDef) and node.name == 'paintEvent')
    code = compile(ast.Module(body=[func], type_ignores=[]), filename='native_section_classify.py', mode='exec')
    scope = {
        'QPainter': FakePainter, 'Qt': FakeQt, 'QPen': FakePen,
        'QColor': FakeQColor, 'QPointF': QPointF,
        'QPolygonF': list, 'measurement_labels': measurement_labels,
        'measurement_text': measurement_text,
    }
    exec(code, scope)
    return scope['paintEvent']


def render_labels(real_paint, points, path=False, unit='m', complete=True):
    canvas = types.SimpleNamespace(
        screen_from_section=lambda x, y: (x, y),
        world_projection_snapshot=None,
    )
    overlay = types.SimpleNamespace(
        canvas=canvas, _paint_projection=None, marks=[],
        shapes=[('measurement', points, {
            'path': path, 'complete': complete,
            'style': {'color': (1., 1., 0.), 'width': 2,
                      'label_font_size': 16, 'unit': unit},
        })],
    )
    real_paint(overlay, None)
    return overlay.last_painter.texts


def test_actual_paint_event_line_draws_only_one_final_total(real_overlay_paint_event):
    labels = render_labels(real_overlay_paint_event,
                           [(0, 0), (3, 4), (6, 4), (6, 9)], path=False)
    assert labels == [(14.0, 1.0, '13.000 m')]


def test_actual_paint_event_path_draws_cumulative_at_vertices(real_overlay_paint_event):
    labels = render_labels(real_overlay_paint_event,
                           [(0, 0), (3, 4), (6, 4), (6, 9)], path=True)
    assert labels == [(11.0, -4.0, '5.000 m'),
                      (14.0, -4.0, '8.000 m'),
                      (14.0, 1.0, 'Total 13.000 m')]


def test_actual_paint_event_live_preview_displays_floating_sum(real_overlay_paint_event):
    labels = render_labels(real_overlay_paint_event,
                           [(0, 0), (3, 4), (6, 4)], path=False, complete=False)
    assert labels[-1][-1] == '8.000 m'


def test_actual_paint_event_path_km_distances(real_overlay_paint_event):
    labels = render_labels(real_overlay_paint_event,
                           [(0, 0), (300, 400), (900, 400)], path=True, unit='km')
    assert [t[-1] for t in labels] == ['0.500 km', 'Total 1.100 km']


def test_source_does_not_change_vtk_handlers():
    # The original PyVista/VTK Measure functions remain in measurement_tools.py.
    vtk = (ROOT / 'gui' / 'measurement_tools.py').read_text(encoding='utf8')
    assert 'def activate(self, mode="measure_line")' in vtk
    assert 'def _finalize_measurement(self, push_undo=True)' in vtk
    assert 'def _create_distance_line(self):' in vtk


def test_native_tool_hint_distinguishes_line_from_cross_section():
    source = (ROOT / 'gui' / 'native_shortcuts.py').read_text(encoding='utf8')
    assert "'MeasureLine': 'Click connected points; right-click finishes the total length.'" in source
    assert "'CrossSectionMeasure': 'Click 2 points in a cross-section view to measure.'" in source


def test_snt_only_vulkan_input_filter_routes_multiclick_line_and_right_finish(native_module):
    mod, qt = native_module

    class Widget:
        def __init__(self):
            self.filter = None
            self.focus_count = 0

        def installEventFilter(self, installed):
            self.filter = installed

        def setFocus(self):
            self.focus_count += 1

    class Event:
        def __init__(self, kind, pt, button=qt.LeftButton):
            self.kind, self.pt, self.click_button = kind, Point(*pt), button

        def type(self):
            return self.kind

        def position(self):
            return self.pt

        def button(self):
            return self.click_button

        def buttons(self):
            return self.click_button

    widget = Widget()
    m = mod.NativeMeasure(Canvas())
    view = types.SimpleNamespace(measurement=m, creation=None)
    input_filter = mod.NativeMeasureInput(view, widget)
    assert widget.filter is input_filter
    for xy in [(0, 0), (3, 4), (6, 4)]:
        assert input_filter.eventFilter(widget, Event(mod.QEvent.MouseButtonPress, xy))
    assert m.completed == [] and len(m.points) == 3
    assert input_filter.eventFilter(widget, Event(mod.QEvent.MouseButtonPress,
                                                  (999, 999), qt.RightButton))
    assert m.completed == [[(0., 0.), (3., 4.), (6., 4.)]]
    assert m.completed_modes == [False]
    assert widget.focus_count == 4


def test_large_route_labels_do_not_involve_source_geometry_indexes():
    # Long measurement routes operate only on their own clicked vertices,
    # never scanning the imported SNT or LiDAR arrays.
    points = [(float(i), float(i % 7)) for i in range(2500)]
    samples = measurement_labels(points, path=True)
    assert len(samples) == 2499
    assert samples[-1][1] == pytest.approx(measurement_distance(points))
    assert measurement_labels(points, path=False) == [samples[-1]]
