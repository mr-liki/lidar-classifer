"""Parallel/Centerline figure picking excludes overlapping attachments."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gui.native_selection_targets import parallel_line_source, drawn_open_line_source
from gui.native_vector_pick import nearest_segment


def feature(source, kind='polyline', **metadata):
    coords = [(0., 0.), (10., 0.)]
    record = dict(type=kind, coords=coords, **metadata)
    return dict(source=source, type=kind, coords=coords, key=(source, id(record)),
                **{'drawing' if source == 'native' else 'entity': record})


def method(filename, class_name, name):
    # Run the production handler without loading unrelated GUI/renderer imports.
    path = Path(__file__).resolve().parent / 'gui' / filename
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(drawn_open_line_source=drawn_open_line_source)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('kind', ['line', 'polyline', 'lwpolyline'])
@pytest.mark.parametrize('layer', ['', 'Survey', 'Road_Edge', 'BL001', 'Trajectory'])
def test_all_snt_layers_are_ineligible(kind, layer):
    imported = feature('snt', kind, layer=layer)
    assert not parallel_line_source(imported)


@pytest.mark.parametrize('kind', ['line', 'smartline', 'polyline', 'curve', 'parallel', 'freehand'])
def test_draw_lines_remain_eligible(kind):
    assert parallel_line_source(feature('native', kind))


@pytest.mark.parametrize('metadata', [dict(imported=True), dict(source='snt'),
                                      dict(source='block_creation'), dict(source='cell_creation')])
def test_imports_and_creation_boundaries_in_draw_store_are_ineligible(metadata):
    drawing = feature('native')
    drawing['drawing'].update(metadata)
    assert not parallel_line_source(drawing)


@pytest.mark.parametrize('predicate', [parallel_line_source, drawn_open_line_source])
def test_nearer_snt_line_does_not_steal_draw_figure_pick(predicate):
    imported, drawn = feature('snt'), feature('native')
    drawn['coords'] = [(0., 1.), (10., 1.)]
    picker = SimpleNamespace(
        _near_scene=lambda p, tolerance: iter([imported, drawn]),
        _feature_distance=lambda p, f: nearest_segment(p, f['coords'])[0])
    pick = method('native_draw.py', 'NativeDraw', '_pick_feature')
    assert pick(picker, (5., 0.), tolerance=2., predicate=predicate) is drawn
    picker._near_scene = lambda p, tolerance: iter([imported])
    assert pick(picker, (5., 0.), tolerance=2., open_only=True,
                predicate=predicate) is None


def test_dialog_ignores_snt_click_and_accepts_draw_click():
    draw = SimpleNamespace(_valid_source=Mock(return_value=True),
                           set_legacy_dialog_highlights=Mock())
    dialog = SimpleNamespace(_native_draw=draw, _select_mode=True,
                             _selected_drawings=[], _selected_drawing=None,
                             _set_status=Mock(), _update_figure_label=Mock(),
                             apply_btn=Mock(), classify_btn=Mock())
    picked = method('parallel_tool_dialog.py', 'ParallelToolDialog', '_on_native_feature_picked')
    picked(dialog, feature('snt'))
    assert dialog._selected_drawings == []
    draw.set_legacy_dialog_highlights.assert_not_called()
    drawn = feature('native')
    picked(dialog, drawn)
    assert dialog._selected_drawings == [drawn]
    draw.set_legacy_dialog_highlights.assert_called_once_with(dialog, [drawn])


def test_classification_rejects_stale_snt_selection():
    dialog = SimpleNamespace(
        _refresh_class_lists_from_active_palette=Mock(),
        _selected_drawing=feature('snt'), _selected_drawings=[],
        _native_draw=SimpleNamespace(_valid_source=Mock(return_value=True)),
        _set_status=Mock(), _drawing_coords=Mock())
    classify = method('parallel_tool_dialog.py', 'ParallelToolDialog', '_on_classify_band')
    classify(dialog)
    dialog._drawing_coords.assert_not_called()
    dialog._set_status.assert_called_once_with('Select an open line drawn with the Draw tools.')


def test_centerline_ignores_snt_and_selects_two_draw_lines():
    draw = SimpleNamespace(_valid_source=Mock(return_value=True),
                           set_legacy_dialog_highlights=Mock())
    dialog = SimpleNamespace(_native_draw=draw, _select_mode=True,
        _selected_drawings=[], _show_message=Mock(),
        _is_open_line_feature=lambda f: True, _update_lines_label=Mock(),
        apply_btn=Mock(), _on_finish_selection=Mock())
    pick = method('centerline_tool_dialog.py', 'CenterlineToolDialog', '_on_native_feature_picked')
    pick(dialog, feature('snt'))
    assert dialog._selected_drawings == []
    draw.set_legacy_dialog_highlights.assert_not_called()
    first, second = feature('native'), feature('native', 'curve')
    pick(dialog, first)
    dialog.apply_btn.setEnabled.assert_called_with(False)
    pick(dialog, feature('snt'))
    assert dialog._selected_drawings == [first]
    pick(dialog, second)
    assert dialog._selected_drawings == [first, second]
    dialog.apply_btn.setEnabled.assert_called_with(True)
    dialog._on_finish_selection.assert_called_once()


@pytest.mark.parametrize('import_index', [0, 1])
def test_centerline_generate_rejects_stale_snt_selection(import_index):
    sources = [feature('native'), feature('native')]
    sources[import_index] = feature('snt')
    dialog = SimpleNamespace(_selected_drawings=sources,
        _native_draw=SimpleNamespace(_valid_source=Mock(return_value=True)),
        _show_message=Mock(), apply_btn=Mock(), _compute_centerline_coords=Mock())
    apply = method('centerline_tool_dialog.py', 'CenterlineToolDialog', '_on_apply')
    apply(dialog)
    dialog._compute_centerline_coords.assert_not_called()
    dialog.apply_btn.setEnabled.assert_called_once_with(False)
    dialog._show_message.assert_called_once()
