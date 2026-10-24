"""Real MARCENA fence/commit/history/export verification on a disposable COPY."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import json
import os
import sys
from time import perf_counter
import shutil
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['QT_QPA_PLATFORM'] = 'windows'
os.environ.update(NAKSHA_GROUND_ENGINE_V2='1', NAKSHA_GROUND_LOCATOR='0',
                  NAKSHA_GROUND_GPU='0', NAKSHA_GROUND_WORKERS='2')
import laspy
import numpy as np
from benchmark_ground_ptd_v2 import environment
from benchmark_streamed_ground_v2 import digest
from gui import lidar_classification_tools as tools
from gui import native_ground_below_stream as stream
from gui.native_lnt_directory import inspect_directory, read_exact_polygon_snapshot
from gui.native_save import read_classes, write_pair
from gui.native_history_routing import route_native_history
from gui.native_shortcuts import native_classification_history
from gui import native_byclass_parity as parity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT) or args.source.resolve().is_relative_to(out):
        raise ValueError('Use an output directory inside the repository, separate from source.')
    out.mkdir(parents=True, exist_ok=True)
    info = inspect_directory(args.source, max_points=2**63-1)
    core = stream._regions(info, [], 110.)[81]
    cx, cy = (core[0]+core[2])/2, (core[1]+core[3])/2
    fence = np.array([[cx-12, cy-12], [cx+12, cy-12], [cx+12, cy+12], [cx-12, cy+12]])
    params = dict(from_classes=[0], to_class=1, current_ground=[1],
                  terrascan_compatible_densification=True)
    report = dict(source=str(args.source), source_sha256_before=digest(args.source),
                  total_points=info['total'], fence=fence.tolist(), params=params)
    function = next(n for n in ast.parse(args.baseline_source.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'classify_ground_ptd')
    ns = dict(tools.__dict__)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(args.baseline_source), 'exec'), ns)
    with environment(NAKSHA_GROUND_ENGINE_V2=0), patch.object(
            tools, 'classify_ground_ptd', ns['classify_ground_ptd']):
        baseline, _ = stream.plan_streamed_ground(args.source, [fence], params, expected_total=info['total'])
    original, entries, leaves, starts = read_classes(str(args.source))
    copy = out / 'real_fence_copy.lnt'
    shutil.copyfile(args.source, copy)
    from PySide6.QtCore import Qt, QSettings
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
    from gui.lnt_native import NativeView
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(out / 'settings'))
    qt = QApplication.instance() or QApplication([])
    app = QMainWindow()
    widget = QWidget(app); widget.setAttribute(Qt.WA_NativeWindow)
    app.setCentralWidget(widget)
    native = NativeView(int(widget.winId()), 640, 480, str(copy), 64, 8.)
    view = SimpleNamespace(app=app, native=native, lnt_path=str(copy), mode_2d=True,
        owner=SimpleNamespace(vulkan_widget=widget, vulkan_backend=None))
    app._lnt_main_view = view
    app.data = None; app.undo_stack, app.redo_stack = [], []
    app.class_palette = {0: {'name': 'Default', 'show': True}, 1: {'name': 'Ground', 'show': True}}
    app.undo_classification = lambda: native_classification_history(app)
    app.redo_classification = lambda: native_classification_history(app, redo=True)
    dialog = tools._BaseClassifyDialog(app, 'Real-data Ground V2 validation')
    actual_planner = stream.plan_streamed_ground
    captured = []
    def planner(*a, **kw):
        result = actual_planner(*a, **kw)
        captured.append(result[0])
        return result
    actual_writer = native.set_classes
    def writer(*a, **kw):
        started = perf_counter()
        result = actual_writer(*a, **kw)
        report.setdefault('native_commit_seconds', perf_counter()-started)
        return result
    native.set_classes = writer
    try:
        with patch.object(stream, 'plan_streamed_ground', planner), patch.object(
                parity, 'read_snapshot', side_effect=AssertionError('whole-cloud snapshot entered')):
            changed = parity._with_progress(dialog, 'Ground V2 real fence', None,
                algorithm_spec=dict(type='ground', params=params, polygons=[fence]))
        plan = captured[0]
        assert changed > 0
        for key in ('node', 'local', 'old_classes', 'new_classes'):
            np.testing.assert_array_equal(plan[key], baseline[key])
        ids = stream._identity(plan['node'], plan['local'])
        scope, _ = read_exact_polygon_snapshot(args.source, [fence], expected_total=info['total'])
        assert np.isin(ids, stream._identity(scope['node'], scope['local'])).all()
        node_start = np.full(len(entries), -1, np.int64); node_start[leaves] = starts[:-1]
        rows = node_start[plan['node']] + plan['local']
        expected = original.copy(); expected[rows] = plan['new_classes']
        def check(want):
            actual, _, _, _ = read_classes(str(copy))
            np.testing.assert_array_equal(actual, want)
        check(expected)
        report.update(planned_changes=changed, exact_node_local_class_parity=True,
                      outside_fence_changes=0, commit_class_differences=0)
        assert route_native_history(app)
        check(original)
        assert route_native_history(app, redo=True)
        check(expected)
        report.update(undo_class_differences=0, redo_class_differences=0)
        native.close()
        if args.export:
            cache = {}
            started = perf_counter()
            saved = write_pair(str(args.source.with_suffix('.laz')), str(copy),
                               str(out / 'real_fence_saved.laz'), cache=cache)
            report['save_export_seconds'] = perf_counter()-started
            reloaded, _, _, _ = read_classes(saved['lnt'])
            np.testing.assert_array_equal(reloaded, expected)
            source_order = np.empty_like(expected); source_order[cache['order']] = expected
            at = 0
            with laspy.open(saved['path']) as reader:
                assert reader.header.point_count == len(expected)
                for chunk in reader.chunk_iterator(500000):
                    np.testing.assert_array_equal(chunk.classification, source_order[at:at+len(chunk)])
                    at += len(chunk)
            assert at == len(expected)
            report.update(exported_points=at, export_class_differences=0, reload_class_differences=0)
        report['source_sha256_after'] = digest(args.source)
        assert report['source_sha256_after'] == report['source_sha256_before']
        report['passed'] = True
    finally:
        native.close(); dialog.close(); app.close(); qt.processEvents()
        (out/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
