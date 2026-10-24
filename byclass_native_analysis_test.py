"""Exercise analysis handlers with file-backed points and no VTK access."""
import ast
import builtins
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial import cKDTree


@pytest.fixture
def native_cloud(monkeypatch):
    original_import = builtins.__import__

    def without_vtk(name, *args, **kwargs):
        if name == 'vtk' or name.startswith(('vtk.', 'vtkmodules')):
            pytest.fail(f'Native analysis imported {name}')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_vtk)
    from gui import native_by_class, native_byclass_parity

    xyz = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0],
                    [1, 1, 2], [2, 2, 5]], dtype=float)
    classes = np.array([1, 1, 1, 6, 6], dtype=np.uint8)
    snapshot = dict(xyz=xyz, classification=classes,
                    node=np.zeros(5, dtype=np.uint32),
                    local=np.arange(5, dtype=np.uint32))
    native = SimpleNamespace(
        has_section_api=lambda: True,
        query_points=Mock(return_value=snapshot),
        set_classes=Mock(side_effect=AssertionError('Analysis wrote classes')),
    )
    view = SimpleNamespace(native=native, lnt_path='native-analysis.lnt')
    app = SimpleNamespace(data=None, _lnt_main_view=view, class_palette={})
    monkeypatch.setattr(native_byclass_parity, '_available_cloud', lambda *a, **k: view)
    monkeypatch.setattr(native_by_class, 'cloud_bbox', lambda v: (0, 0, 10, 10, 5))
    yield app, snapshot
    native.set_classes.assert_not_called()
    np.testing.assert_array_equal(classes, [1, 1, 1, 6, 6])


def handler(class_name, method_name, messages):
    # Avoid importing the ribbon's unrelated legacy GUI dependencies.
    tree = ast.parse((Path(__file__).resolve().parent /
                      'gui/menu_sidebar_system.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    namespace = dict(np=np, cKDTree=cKDTree, Qt=SimpleNamespace(UserRole=256),
                     QApplication=SimpleNamespace(processEvents=lambda: None),
                     QMessageBox=messages)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(tree), 'exec'), namespace)
    return namespace[method_name]


def selection(code):
    return SimpleNamespace(selectedItems=lambda: [SimpleNamespace(data=lambda role: code)])


def test_distance_analysis_with_none_data_uses_exact_native_points(native_cloud):
    app, _ = native_cloud
    messages = Mock()
    dialog = SimpleNamespace(app=app, from_list=selection(6),
                             to_combo=SimpleNamespace(currentData=lambda: 1),
                             distance_info_label=Mock(), radius_combo=Mock())
    dialog.radius_combo.count.return_value = 6
    handler('ClosedByClassDialog', 'analyze_distances', messages)(dialog)
    messages.warning.assert_not_called()
    messages.information.assert_called_once()
    assert 'Analyzed 2 From points -> 3 To points' in dialog.distance_info_label.setText.call_args.args[0]
    options = dialog.radius_combo.addItem.call_args_list
    assert options[0].args[1] == pytest.approx((np.sqrt(6) + np.sqrt(33)) / 2)
    assert app.data is None
    app._lnt_main_view.native.query_points.assert_called_once()


def test_distance_query_failure_is_reported_without_falling_back(native_cloud):
    app, _ = native_cloud
    app.data = {'xyz': np.ones((1, 3)), 'classification': np.array([6])}
    app._lnt_main_view.native.query_points.side_effect = RuntimeError('native query failed')
    messages = Mock()
    dialog = SimpleNamespace(app=app, from_list=selection(6),
                             to_combo=SimpleNamespace(currentData=lambda: 1),
                             distance_info_label=Mock(), radius_combo=Mock())
    handler('ClosedByClassDialog', 'analyze_distances', messages)(dialog)
    messages.warning.assert_called_once()
    assert 'native query failed' in messages.warning.call_args.args[2]
    dialog.radius_combo.addItem.assert_not_called()


@pytest.mark.parametrize('from_classes', [[6], None])
def test_auto_height_uses_union_of_native_fences(native_cloud, from_classes):
    app, _ = native_cloud
    messages = Mock()
    dialog = SimpleNamespace(
        app=app,
        selected_fences=[dict(coords=[[.5, .5], [1.5, .5], [1.5, 1.5], [.5, 1.5]]),
                         dict(coords=[[1.5, 1.5], [2.5, 1.5], [2.5, 2.5], [1.5, 2.5]])],
        _get_selected_from_classes=lambda: from_classes,
        min_height_spin=Mock(), max_height_spin=Mock(), height_stats_label=Mock())
    handler('InsideFenceDialog', 'auto_analyze_heights', messages)(dialog)
    messages.warning.assert_not_called()
    messages.critical.assert_not_called()
    dialog.min_height_spin.setValue.assert_called_once_with(2.0)
    dialog.max_height_spin.setValue.assert_called_once_with(5.0)
    assert '2 points inside fence' in dialog.height_stats_label.setText.call_args.args[0]
    assert app.data is None


def test_auto_height_query_failure_is_reported(native_cloud):
    app, _ = native_cloud
    app._lnt_main_view.native.query_points.side_effect = RuntimeError('native query failed')
    messages = Mock()
    dialog = SimpleNamespace(app=app,
        selected_fences=[dict(coords=[[0, 0], [3, 0], [3, 3], [0, 3]])],
        _get_selected_from_classes=lambda: [6], height_stats_label=Mock(),
        min_height_spin=Mock(), max_height_spin=Mock())
    handler('InsideFenceDialog', 'auto_analyze_heights', messages)(dialog)
    messages.critical.assert_called_once()
    dialog.min_height_spin.setValue.assert_not_called()
    dialog.max_height_spin.setValue.assert_not_called()


def test_relative_height_analysis_uses_native_tin_with_stale_data(native_cloud, monkeypatch):
    from PySide6 import QtWidgets
    monkeypatch.setattr(QtWidgets, 'QProgressDialog', Mock())
    app, _ = native_cloud
    # Legacy cached data must never override the active native cloud.
    app.data = {'xyz': np.zeros((1, 3)), 'classification': np.array([99])}
    messages = Mock()
    dialog = SimpleNamespace(
        app=app, from_list=selection(6), selected_fences=[],
        ref_class_combo=SimpleNamespace(currentData=lambda: 1, currentText=lambda: 'Ground'),
        height_info_label=Mock(), height_combo=Mock(),
        max_triangle_spin=SimpleNamespace(value=lambda: 20.0),
        min_height_spin=Mock(), max_height_spin=Mock())
    dialog.height_combo.count.return_value = 6
    handler('ByClassHeightDialog', 'analyze_heights', messages)(dialog)
    messages.warning.assert_not_called()
    messages.information.assert_called_once()
    dialog.min_height_spin.setValue.assert_called_once_with(2.0)
    dialog.max_height_spin.setValue.assert_called_once_with(5.0)
    assert 'Analyzed 2 supported points' in dialog.height_info_label.setText.call_args.args[0]


def test_native_analysis_refuses_incomplete_records(native_cloud):
    from gui.native_byclass_height_analysis import height_analysis_points
    app, snapshot = native_cloud
    snapshot['local'] = snapshot['local'][:-1]
    with pytest.raises(RuntimeError, match='inconsistent array lengths'):
        height_analysis_points(app)
