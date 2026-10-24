"""Exercise streamed Ground commit/history/save on GENERATED data only.

Requires the existing Windows native DLL and Qt Windows platform. Never opens
MARCENA or another user point file. The test-only snapshot cap forces the
unchanged large-cloud dispatch on a small disposable native-converted fixture.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['QT_QPA_PLATFORM'] = 'windows'
os.environ.update(NAKSHA_GROUND_ENGINE_V2='1', NAKSHA_GROUND_LOCATOR='0',
                  NAKSHA_GROUND_GPU='0', NAKSHA_GROUND_WORKERS='2')
import laspy
import numpy as np
from scipy.spatial import cKDTree
from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from gui.lnt_native import NativeView, convert_las
from gui import native_byclass_parity as parity
from gui.native_ground_below_stream import plan_streamed_ground, _identity
from gui.native_lnt_directory import read_exact_leaf_snapshot
from gui.native_shortcuts import native_classification_history
from gui.native_history_routing import route_native_history
from gui.native_save import write_pair
from gui.lidar_classification_tools import _BaseClassifyDialog
from benchmark_ground_ptd_v2 import synthetic, environment


def main():
    out = ROOT / 'audit/ground_ptd_v2/native_fixture'
    out.mkdir(parents=True, exist_ok=True)
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(out / 'settings'))
    xyz, classes = synthetic(8000)
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.scales = [.001, .001, .001]
    header.offsets = xyz.min(0)
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz.T
    las.classification = classes
    source, path = out / 'generated.las', out / 'generated.lnt'
    las.write(source)
    convert_las(str(source), str(path))
    before, _ = read_exact_leaf_snapshot(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    params = dict(from_classes=[0], to_class=1, current_ground=[1],
                  max_building_size=25., terrascan_compatible_densification=True)
    polygon = np.array([[256020., 4837020.], [256160., 4837020.],
                        [256160., 4837160.], [256020., 4837160.]])
    plans = []
    for enabled in ('0', '1'):
        with environment(NAKSHA_GROUND_ENGINE_V2=enabled):
            plan, _ = plan_streamed_ground(path, [polygon], params, expected_total=8000)
            plans.append(plan)
    for key in ('node', 'local', 'old_classes', 'new_classes'):
        np.testing.assert_array_equal(plans[0][key], plans[1][key])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    plan = plans[1]
    assert len(plan['node']) > 0
    full_ids = _identity(before['node'], before['local'])
    rows = np.searchsorted(full_ids, _identity(plan['node'], plan['local']))
    expected = before['classification'].copy()
    expected[rows] = plan['new_classes']
    qt = QApplication.instance() or QApplication([])
    app = QMainWindow()
    widget = QWidget(app)
    widget.setAttribute(Qt.WA_NativeWindow)
    app.setCentralWidget(widget)
    app.resize(640, 480)
    # A native HWND is sufficient; the test window does not need showing.
    native = NativeView(int(widget.winId()), 640, 480, str(path), 64, 8.)
    view = SimpleNamespace(app=app, native=native, lnt_path=str(path), mode_2d=True,
                           owner=SimpleNamespace(vulkan_widget=widget, vulkan_backend=None))
    app._lnt_main_view = view
    app.data = None
    app.undo_stack, app.redo_stack = [], []
    app.class_palette = {0: {'name': 'Default', 'show': True},
                         1: {'name': 'Ground', 'show': True}}
    app.undo_classification = lambda: native_classification_history(app)
    app.redo_classification = lambda: native_classification_history(app, redo=True)
    dialog = _BaseClassifyDialog(app, 'Ground V2 generated-fixture verification')
    result = dict(points=8000, planned_changes=len(rows), exact_plan=True,
                  source_unchanged_before_commit=True)
    original_writer = native.set_classes
    def timed_write(*args, **kwargs):
        started = perf_counter()
        success = original_writer(*args, **kwargs)
        result.setdefault('native_commit_seconds', perf_counter() - started)
        return success
    native.set_classes = timed_write
    try:
        started = perf_counter()
        # Force the ACTUAL streamed dispatch, while leaving production's 8M
        # guard unchanged. An accidental snapshot path fails this test.
        with patch.object(parity, 'MAX_EXACT_POINTS', 7999), patch.object(
                parity, 'read_snapshot', side_effect=AssertionError('whole-cloud snapshot entered')):
            count = parity._with_progress(dialog, 'Ground V2', None,
                algorithm_spec=dict(type='ground', params=params, polygons=[polygon]))
        result['planning_and_commit_seconds'] = perf_counter() - started
        assert count == len(rows)
        assert len(app.undo_stack) == 1
        after, _ = read_exact_leaf_snapshot(path)
        np.testing.assert_array_equal(after['classification'], expected)
        result['native_commit_exact'] = True
        assert route_native_history(app)
        undone, _ = read_exact_leaf_snapshot(path)
        np.testing.assert_array_equal(undone['classification'], before['classification'])
        assert route_native_history(app, redo=True)
        redone, _ = read_exact_leaf_snapshot(path)
        np.testing.assert_array_equal(redone['classification'], expected)
        result['undo_exact'] = result['redo_exact'] = True
        native.close()
        saved = write_pair(str(source), str(path), str(out / 'saved.laz'))
        reopened, _ = read_exact_leaf_snapshot(saved['lnt'])
        np.testing.assert_array_equal(reopened['classification'], expected)
        exported = laspy.read(saved['path'])
        assert len(exported.points) == 8000
        # Independently resolve the quantized native rows against the unique
        # source records; rounding into decimal bins is not an identity map.
        original = laspy.read(source)
        source_xyz = np.c_[original.x, original.y, original.z]
        distances, mapped = cKDTree(before['xyz']).query(source_xyz, k=2)
        assert np.all(distances[:, 0] < .005)
        assert np.all(distances[:, 1] > .01)
        assert len(np.unique(mapped[:, 0])) == 8000
        np.testing.assert_array_equal(np.c_[exported.x, exported.y, exported.z], source_xyz)
        np.testing.assert_array_equal(exported.classification, expected[mapped[:, 0]])
        result['save_export_reload_exact'] = True
        result['passed'] = True
    finally:
        native.close()
        dialog.close(); app.close(); qt.processEvents()
        (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
