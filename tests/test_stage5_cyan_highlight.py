"""Stage 5: production Vulkan selection/measurement color path tests.

These tests run without PySide6, VTK, or the native Vulkan DLL. They execute
NativeSelectionController, NativeMeasure._paint, NativeDraw._paint and the real
CanvasOverlay.paintEvent body with a minimal recording QPainter substitute.
A Windows GPU/Qt mouse test remains necessary before merging.
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_stage1_parity import runtime as stage1_runtime, App, Canvas, Pos
from test_stage2_parity import prepared_draw
from gui.native_measure_geometry import measurement_labels, measurement_text

ROOT = Path(__file__).resolve().parents[1]
CYAN = '#00ffff'


@pytest.fixture(scope='module')
def runtime():
    if 'gui.native_selection_bridge' not in sys.modules:
        return stage1_runtime.__wrapped__()
    bridge = sys.modules['gui.native_selection_bridge']
    return bridge.Qt, bridge.QApplication, bridge, sys.modules['gui.native_draw']


def _method_from_source(path, cls, method, scope):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    klass = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == cls)
    func = next(x for x in klass.body if isinstance(x, ast.FunctionDef) and x.name == method)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])),
                 path, 'exec'), scope)
    return scope[method]


class QColor:
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], QColor):
            self.rgb = args[0].rgb
        elif len(args) == 1 and isinstance(args[0], str):
            raw = args[0]
            self.rgb = tuple(int(raw[i:i + 2], 16) for i in (1, 3, 5)) if (
                len(raw) == 7 and raw.startswith('#')) else None
        elif len(args) == 3:
            self.rgb = tuple(int(x) for x in args)
        else:
            self.rgb = None

    def isValid(self):
        return self.rgb is not None

    def setAlpha(self, *_):
        pass

    def setAlphaF(self, *_):
        pass


class QPointF:
    def __init__(self, x=0, y=0):
        self._x, self._y = float(x), float(y)

    def x(self):
        return self._x

    def y(self):
        return self._y

    def __add__(self, other):
        return QPointF(self._x + other.x(), self._y + other.y())

    def __truediv__(self, divisor):
        return QPointF(self._x / divisor, self._y / divisor)


class QPen:
    def __init__(self, color=None, width=1, style=None):
        self.color, self.width, self.style = color, width, style


class QFont:
    def setPointSize(self, *_):
        pass


class Metrics:
    def __init__(self, *_):
        pass

    def boundingRect(self, text):
        return SimpleNamespace(width=lambda: len(text) * 8,
                               height=lambda: 12)


class Qt:
    DashLine, DotLine, DashDotLine, DashDotDotLine, SolidLine, NoBrush = range(6)


class QPainter:
    Antialiasing = 1

    def __init__(self, surface):
        self.surface = surface
        surface.last_painter = self
        self.current_pen = QPen()
        self.current_brush = None
        self.strokes = []
        self.ellipses = []
        self.labels = []

    def setRenderHint(self, *_):
        pass

    def setPen(self, pen):
        self.current_pen = pen

    def pen(self):
        return self.current_pen

    def setBrush(self, brush):
        self.current_brush = brush

    def font(self):
        return QFont()

    def setFont(self, *_):
        pass

    def drawPolyline(self, *_):
        self.strokes.append(self.current_pen.color.rgb)

    def drawPolygon(self, *_):
        self.strokes.append(self.current_pen.color.rgb)

    def drawLine(self, *_):
        self.strokes.append(self.current_pen.color.rgb)

    def drawEllipse(self, *_):
        self.ellipses.append(self.current_brush.rgb if isinstance(self.current_brush, QColor) else None)

    def drawText(self, _pos, text):
        self.labels.append((str(text), self.current_pen.color.rgb if isinstance(self.current_pen.color, QColor) else None))

    def end(self):
        pass


@pytest.fixture(scope='module')
def paint_event():
    return _method_from_source('gui/native_section_classify.py', 'CanvasOverlay', 'paintEvent', {
        'QPainter': QPainter, 'QColor': QColor, 'QPen': QPen, 'QPointF': QPointF,
        'QPolygonF': list, 'QFontMetricsF': Metrics, 'Qt': Qt,
        '_drawing_vertex_markers_visible': lambda *_: False,
        'measurement_labels': measurement_labels, 'measurement_text': measurement_text,
    })


def rendered(paint_event, kind, points, options):
    canvas = SimpleNamespace(screen_from_section=lambda x, y: (x, y),
                             world_projection_snapshot=None)
    overlay = SimpleNamespace(canvas=canvas, marks=[], _paint_projection=None,
                              shapes=[(kind, list(points), dict(options))])
    paint_event(overlay, None)
    return overlay.last_painter


@pytest.mark.parametrize('state', ['selected', 'hovered'])
def test_draw_selection_and_hover_are_pure_cyan(paint_event, state):
    data = {'type': 'polyline', 'color': '#b94b22', 'highlight_scope': 'draw', state: True}
    result = rendered(paint_event, 'drawing', [(1, 1), (2, 4)], data)
    assert result.strokes == [(0, 255, 255)]
    assert data['color'] == '#b94b22'


def test_deselect_restores_original_draw_color(paint_event):
    style = {'type': 'line', 'color': '#aabbcc', 'highlight_scope': 'draw'}
    result = rendered(paint_event, 'drawing', [(0, 0), (2, 2)], style)
    assert result.strokes == [(170, 187, 204)]


@pytest.mark.parametrize('origin', ['attachment', 'snt', 'snt_readonly', 'block_creation', 'cell_creation'])
def test_imported_or_protected_geometry_does_not_turn_cyan(paint_event, origin):
    style = {'type': 'polyline', 'source': origin, 'color': '#994422',
             'highlight_scope': 'draw', 'selected': True}
    result = rendered(paint_event, 'drawing', [(0, 0), (3, 3)], style)
    assert (0, 255, 255) not in result.strokes


def test_snt_legacy_selected_still_uses_original_selection_color(paint_event):
    result = rendered(paint_event, 'drawing', [(0, 0), (3, 3)],
                      {'type': 'polyline', 'selected': True})
    assert result.strokes == [(255, 80, 255)]


def test_rubberband_preview_retains_authored_preview_color(paint_event):
    result = rendered(paint_event, 'drawing_preview', [(0, 0), (3, 3)],
                      {'type': 'polyline', 'color': '#ffdf4a'})
    assert result.strokes == [(255, 223, 74)]


@pytest.mark.parametrize('state', ['selected', 'hovered'])
def test_measure_line_highlight_is_cyan_but_distance_label_keeps_style(paint_event, state):
    result = rendered(paint_event, 'measurement', [(0, 0), (3, 4), (6, 4)], {
        'style': {'color': (1., .2, 0.), 'unit': 'm', 'width': 2},
        'highlight_scope': 'measure', 'path': False, state: True,
    })
    assert result.strokes == [(0, 255, 255)]
    assert all(color == (0, 255, 255) for color in result.ellipses)
    assert result.labels == [('8.000 m', (255, 51, 0))]


def test_measure_path_cumulative_labels_and_original_style_unchanged(paint_event):
    style = {'color': (0.1, 0.4, 0.6), 'unit': 'm', 'width': 3}
    result = rendered(paint_event, 'measurement', [(0, 0), (3, 4), (6, 4)], {
        'style': style, 'highlight_scope': 'measure', 'path': True, 'selected': True,
    })
    assert result.strokes == [(0, 255, 255)]
    assert [text for text, _ in result.labels] == ['5.000 m', 'Total 8.000 m']
    assert all(color == (26, 102, 153) for _, color in result.labels)
    assert style['color'] == (0.1, 0.4, 0.6)


def test_measure_unselected_restores_original_color(paint_event):
    result = rendered(paint_event, 'measurement', [(0, 0), (3, 4)], {
        'style': {'color': (0.0, 0.5, 1.), 'unit': 'm'},
        'highlight_scope': 'measure', 'path': False,
    })
    assert result.strokes == [(0, 128, 255)]


def test_measure_grid_area_is_never_cyan_even_if_selected(paint_event):
    result = rendered(paint_event, 'measurement', [(0, 0), (2, 0), (2, 2), (0, 2)], {
        'style': {'color': (1., 0.55, 0.0), 'unit': 'm'},
        'highlight_scope': 'measure', 'area': 4.0, 'selected': True,
    })
    assert result.strokes == [(255, 140, 0)]


@pytest.fixture
def measure_paint():
    return _method_from_source('gui/native_measure.py', 'NativeMeasure', '_paint', {})


def fake_measure(measure_paint):
    measure = SimpleNamespace(
        completed=[[(0., 0.), (3., 4.)], [(20., 0.), (24., 0.)]],
        completed_modes=[False, True], grids=[], points=[], hover=None,
        path=False, _style=lambda path=None: {'color': (1., 0.25, 0.), 'width': 2},
    )
    measure.overlay = SimpleNamespace(set_shapes=lambda shapes: setattr(measure, 'shapes', shapes))
    measure._paint = lambda: measure_paint(measure)
    return measure


def test_measure_selection_and_clear_repaint_measured_strokes(runtime, measure_paint):
    _, _, bridge, _ = runtime
    measure = fake_measure(measure_paint)
    draw = prepared_draw([], measurement=measure)
    measure.canvas = SimpleNamespace(owner=draw.view)
    draw.view.drawing = draw
    controller = bridge.NativeSelectionController(draw)
    assert controller.activate('individual')
    controller.selection_manager.apply([controller._measure_features()[0]])
    assert measure.shapes[0][2]['selected'] is True
    assert measure.shapes[1][2]['selected'] is False
    assert measure.shapes[0][2]['highlight_scope'] == 'measure'
    controller.selection_manager.clear()
    assert measure.shapes[0][2]['selected'] is False
    assert measure.shapes[0][2]['hovered'] is False
    assert measure.completed_modes == [False, True]


def test_hover_measurement_uses_only_completed_line_path(runtime, measure_paint):
    qt, _, bridge, _ = runtime
    measure = fake_measure(measure_paint)
    draw = prepared_draw([], measurement=measure)
    measure.canvas = SimpleNamespace(owner=draw.view)
    draw.view.drawing = draw
    controller = bridge.NativeSelectionController(draw)
    assert controller.activate('individual')
    before = draw.paints
    assert controller.move(Pos(1.5, 2.), qt.NoButton)
    assert controller._hover_feature['source'] == 'measure'
    assert measure.shapes[0][2]['hovered'] is True
    assert draw.paints > before
    # Force a new 20 Hz pick after moving far away.
    controller._hover_pick_at -= 1.
    controller.move(Pos(100., 100.), qt.NoButton)
    assert controller._hover_feature is None
    assert measure.shapes[0][2]['hovered'] is False


def test_selected_draw_hover_block_and_deselect_are_consistent(runtime):
    qt, _, bridge, _ = runtime
    a = {'type': 'line', 'coords': [(1., 1.), (5., 1.)], 'color': '#334455'}
    b = {'type': 'line', 'coords': [(8., 8.), (12., 8.)], 'color': '#cc5500'}
    protected = {'type': 'polyline', 'source': 'block_creation',
                 'coords': [(1, 0), (4, 0)], 'color': '#888888'}
    draw = prepared_draw([a, b, protected])
    controller = bridge.NativeSelectionController(draw)
    assert controller.activate('individual')
    controller.move(Pos(3., 1.), qt.NoButton)
    assert controller._hover_feature['drawing'] is a
    controller.set_method('block')
    controller._select_area('block', [(0, 0), (15, 15)])
    assert {f['drawing']['color'] for f in controller.selection_manager.get_features()} == {
        '#334455', '#cc5500',
    }
    controller.selection_manager.clear()
    assert controller.selection_manager.count() == 0
    assert a['color'] == '#334455' and b['color'] == '#cc5500'


def test_hover_throttle_does_not_touch_snt_pick_index(runtime):
    qt, _, bridge, _ = runtime
    draw = prepared_draw([{'type': 'line', 'coords': [(0, 0), (4, 0)]}])
    controller = bridge.NativeSelectionController(draw)
    assert controller.activate('individual')
    draw._snt_index = lambda: (_ for _ in ()).throw(AssertionError('SNT scan forbidden'))
    count = [0]
    original = controller._pick_editable
    def pick(p):
        count[0] += 1
        return original(p)
    controller._pick_editable = pick
    for _ in range(30):
        controller.move(Pos(1, 0), qt.NoButton)
    assert count[0] == 1
    controller._hover_pick_at -= 1.
    controller.move(Pos(2, 0), qt.NoButton)
    assert count[0] == 2


def test_native_draw_overlay_marks_selected_and_hovered_without_measure_duplicates(runtime):
    qt, _, bridge_module, draw_module = runtime
    drawing = {'type': 'curve', 'coords': [(1., 1.), (2., 3.), (3., 1.)], 'color': '#844422'}
    measure = SimpleNamespace(completed=[[(10., 10.), (15., 10.)]], completed_modes=[False],
                              _paint=lambda: None)
    draw = draw_module.NativeDraw.__new__(draw_module.NativeDraw)
    draw.drawings = [drawing]
    draw.tool = 'select'
    draw.app = App()
    draw.app.snt_attachments = []
    draw.view = SimpleNamespace(measurement=measure)
    draw.canvas = Canvas()
    draw._style = lambda tool: {'color': '#ffaa11'}
    draw._scene_token = lambda: ()
    draw._sync_block_vectors = lambda: False
    draw._shutting_down = False
    draw._selected_imported = None
    draw.selected = None
    draw._selected_native_ref = None
    draw._legacy_tool_dialog = None
    draw._legacy_dialog_highlights = []
    draw.aux_shapes = []
    draw.points = []
    draw.hover = None
    draw._snap_target = None
    draw._hover_selected_native_ref = None
    draw.overlay = SimpleNamespace(set_shapes=lambda shapes: setattr(draw, 'shapes', shapes))
    controller = bridge_module.NativeSelectionController(draw)
    controller.activate('individual')
    controller.selection_manager.apply([controller._native_feature(0, drawing),
                                        controller._measure_features()[0]])
    actual = [item for item in draw.shapes if item[0] == 'drawing']
    assert len(actual) == 1  # no second Draw overlay for selected measurement
    _, _, style = actual[0]
    assert style['selected'] is True and style['highlight_scope'] == 'draw'
    assert style['color'] == '#844422'  # permanent style remains unchanged
    controller._set_hover_feature(controller._native_feature(0, drawing))
    assert draw.shapes[0][2]['hovered'] is True
    controller.selection_manager.clear()
    assert draw.shapes[0][2]['selected'] is False


def test_native_draw_standalone_select_hover_is_scoped_and_throttled(runtime):
    qt, _, _, draw_module = runtime
    a = {'type': 'line', 'coords': [(0., 0.), (4., 0.)]}
    d = draw_module.NativeDraw.__new__(draw_module.NativeDraw)
    d.tool = 'select'; d.drawings = [a]; d.canvas = Canvas()
    d._selection_bridge = None; d._hover_pick_at = 0.
    d._hover_selected_native_ref = None
    d.active = lambda: True
    paints = [0]; picks = [0]
    d._paint = lambda: paints.__setitem__(0, paints[0] + 1)
    def pick(p):
        picks[0] += 1
        return {'drawing': a} if p[0] < 10. else None
    d._pick_editable_draw = pick
    d._snt_index = lambda: (_ for _ in ()).throw(AssertionError('SNT scan forbidden'))
    d.move(Pos(2., 0.), qt.NoButton)
    assert d._hover_selected_native_ref is a
    d.move(Pos(2., 0.), qt.NoButton)
    assert picks[0] == 1 and paints[0] == 1
    d._hover_pick_at -= 1.
    d.move(Pos(20., 0.), qt.NoButton)
    assert d._hover_selected_native_ref is None
    assert paints[0] == 2


def test_switching_out_of_select_clears_measured_cyan(runtime):
    _, _, _, draw_module = runtime
    events = []
    measurement = SimpleNamespace(
        cancel=lambda: events.append('measure_cancel'),
        _paint=lambda: events.append('measure_repaint'),
    )
    d = draw_module.NativeDraw.__new__(draw_module.NativeDraw)
    d.view = SimpleNamespace(measurement=measurement)
    d.app = SimpleNamespace(_native_sections=None,
                            statusBar=lambda: SimpleNamespace(showMessage=lambda *a: None))
    d.canvas = SimpleNamespace(setFocus=lambda: None)
    d.tool = 'select'
    d._legacy_tool_dialog = None
    d._selection_bridge = SimpleNamespace(_active=True, _intersect_mode=False)
    d.cancel = lambda stand_down=False: None
    d._selected_feature = lambda: None
    d._instruction = lambda: 'SmartLine'
    assert d.activate('smartline')
    assert d.tool == 'smartline' and d._selection_bridge is None
    assert events == ['measure_cancel', 'measure_repaint']
