"""Full original streamed planner, native commit and exact readback on a COPY.

Never writes the supplied source. Checkpoints preserve completed PTD calls if
the validation session stops; resumed timings are explicitly not cold full-run
acceptance timings. No changed worker count, regions, halo or candidate rules.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import sys
import uuid
from importlib.metadata import version, PackageNotFoundError
from threading import Lock, local
from time import perf_counter, monotonic
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['QT_QPA_PLATFORM'] = 'windows'
import numpy as np
from scipy.spatial import Delaunay
from gui import lidar_classification_tools as tools
_import_start = perf_counter()
from gui import ground_ptd_engine_v2 as v2
V2_IMPORT_SECONDS = perf_counter() - _import_start
from benchmark_ground_ptd_v2 import Meter, environment
from benchmark_streamed_ground_v2 import digest
from gui import native_ground_below_stream as stream
from gui.native_save import read_classes


def dump(path, value):
    temporary = path.with_suffix('.temporary.json')
    temporary.write_text(json.dumps(value, indent=2,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()), encoding='utf-8')
    os.replace(temporary, path)


def json_result(result):
    return {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engine', choices=['0', '1'], required=True)
    parser.add_argument('--baseline-source', type=Path)
    parser.add_argument('--source-class', type=int, default=0)
    parser.add_argument('--ground-class', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume-checkpoints', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT) or args.source.resolve().is_relative_to(out):
        raise ValueError('Output must be inside the repository and separate from the source.')
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'checkpoints'
    checkpoint.mkdir(exist_ok=True)
    available = set(checkpoint.glob('*.npz')) if args.resume_checkpoints else set()
    source_hash = digest(args.source)
    try:
        numba_version = version('numba')
    except PackageNotFoundError:
        numba_version = None
    runtime = dict(python=sys.version, numpy=np.__version__, scipy=v2.scipy.__version__,
                   numba=numba_version, architecture=platform.machine())
    report = dict(engine=args.engine, workers=args.workers, source=str(args.source),
        source_sha256_before=source_hash, status='RUNNING', resumed=False, runtime=runtime,
        v2_import_seconds=V2_IMPORT_SECONDS if args.engine == '1' else 0.,
        cells=[], checkpoint_hits=0, cumulative_tin_seconds=0.,
        cumulative_lookup_seconds=0., cumulative_read_seconds=0.)
    destination = out / 'MARCENA_validation_copy.lnt'
    if destination.exists() and (out / 'committed.flag').exists():
        raise ValueError('This output already committed a validation copy; choose a new output directory.')
    copy_start = perf_counter()
    shutil.copyfile(args.source, destination)
    report['preparatory_copy_seconds'] = perf_counter() - copy_start
    original, entries, leaves, starts = read_classes(str(args.source))
    report['total_points'] = len(original)
    params = dict(from_classes=[args.source_class], to_class=args.ground_class,
                  current_ground=[args.ground_class], terrascan_compatible_densification=True)
    report['params'] = params
    function = tools.classify_ground_ptd
    baseline_namespace = None
    if args.engine == '0' and args.baseline_source:
        saved = args.baseline_source.read_bytes()
        node = next(n for n in ast.parse(saved.decode('utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'classify_ground_ptd')
        baseline_namespace = dict(tools.__dict__)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(args.baseline_source), 'exec'), baseline_namespace)
        function = baseline_namespace['classify_ground_ptd']
        report['baseline_source_sha256'] = hashlib.sha256(saved).hexdigest()
    code_hash = hashlib.sha256((ROOT / 'gui/ground_ptd_engine_v2.py').read_bytes()
                              + (ROOT / 'gui/lidar_classification_tools.py').read_bytes()).hexdigest()
    namespace = local()
    lock = Lock()
    raw_read = stream._read_window
    raw_locate = v2.find_simplex
    def read(*a, **kw):
        start = perf_counter()
        try:
            return raw_read(*a, **kw)
        finally:
            namespace.bounds = tuple(a[2])
            with lock:
                report['cumulative_read_seconds'] += perf_counter() - start
    class TimedTri:
        def __init__(self, xy):
            start = perf_counter()
            self.tri = Delaunay(xy)
            namespace.stats['tin_seconds'] += perf_counter() - start
            namespace.stats['tin_builds'] += 1
        def __getattr__(self, name):
            return getattr(self.tri, name)
        def find_simplex(self, xy):
            start = perf_counter()
            result = self.tri.find_simplex(xy)
            if not getattr(namespace, 'in_v2_lookup', False):
                namespace.stats['lookup_seconds'] += perf_counter() - start
                namespace.stats['queries'] += len(xy)
            return result
    def locate(tri, xy, **kw):
        namespace.in_v2_lookup = True
        start = perf_counter()
        try:
            return raw_locate(tri, xy, **kw)
        finally:
            namespace.in_v2_lookup = False
            namespace.stats['lookup_seconds'] += perf_counter() - start
            namespace.stats['queries'] += len(xy)
    def classify(**kw):
        started = perf_counter()
        value = hashlib.sha256()
        value.update((source_hash + code_hash + args.engine
            + json.dumps(runtime, sort_keys=True)
            + report.get('baseline_source_sha256', '')).encode())
        for key in ('xyz', 'classification', 'fence_mask'):
            array = kw.get(key)
            if array is not None:
                value.update(str((key, array.shape, array.dtype)).encode())
                value.update(np.ascontiguousarray(array).tobytes())
        value.update(json.dumps(params, sort_keys=True).encode())
        key = value.hexdigest()
        saved = checkpoint / (key + '.npz')
        namespace.stats = dict(tin_seconds=0., lookup_seconds=0., tin_builds=0, queries=0)
        if saved in available:
            with np.load(saved, allow_pickle=False) as cached:
                kw['classification'][:] = cached['classification']
                result = json.loads(str(cached['result']))
                result['indices'] = cached['indices'].copy()
            with lock:
                report['checkpoint_hits'] += 1
                report['resumed'] = True
        else:
            result = function(**kw)
            if str(result.get('stop_reason', '')).lower() == 'cancelled':
                return result
            encoded = json.dumps(json_result(result), default=lambda x: x.item())
            # Separate streamed cells may contain identical ordered inputs.
            # Both still execute for a fresh timing. Serialize checkpoint IO
            # and validate duplicate outputs instead of racing Windows replace.
            with lock:
                if saved.exists():
                    with np.load(saved, allow_pickle=False) as prior:
                        np.testing.assert_array_equal(prior['classification'], kw['classification'])
                        np.testing.assert_array_equal(prior['indices'], result['indices'])
                        assert json.loads(str(prior['result'])) == json.loads(encoded)
                else:
                    temporary = saved.with_suffix('.' + uuid.uuid4().hex + '.temporary')
                    with temporary.open('wb') as handle:
                        np.savez(handle, classification=kw['classification'], indices=result['indices'],
                                 result=encoded)
                    os.replace(temporary, saved)
        stat = dict(namespace.stats, bounds=getattr(namespace, 'bounds', None),
                    points=len(kw['xyz']), elapsed_seconds=perf_counter()-started,
                    iterations=result['iterations'], seed_count=result['seed_count'],
                    stop_reason=result['stop_reason'], changed=result['changed'],
                    input_key=key)
        with lock:
            report['cells'].append(stat)
            report['cumulative_tin_seconds'] += stat['tin_seconds']
            report['cumulative_lookup_seconds'] += stat['lookup_seconds']
            dump(out / 'progress.json', report)
        print(f'CELL complete points={stat["points"]:,} iterations={stat["iterations"]} '
              f'elapsed={stat["elapsed_seconds"]:.1f}s changes={stat["changed"]:,}', flush=True)
        return result
    if baseline_namespace is not None:
        baseline_namespace['Delaunay'] = TimedTri
    from PySide6.QtCore import Qt, QSettings
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
    from gui.lnt_native import NativeView
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(out / 'settings'))
    qt = QApplication.instance() or QApplication([])
    app = QMainWindow()
    widget = QWidget(app); widget.setAttribute(Qt.WA_NativeWindow)
    app.setCentralWidget(widget)
    operation_start = perf_counter()
    native = NativeView(int(widget.winId()), 640, 480, str(destination), 64, 8.)
    last = [0.]
    def progress(pct, message=''):
        elapsed = perf_counter() - operation_start
        if monotonic() - last[0] >= 15:
            last[0] = monotonic()
            with lock:
                report.update(elapsed_seconds=elapsed, last_progress=str(message))
                dump(out / 'progress.json', report)
            print(f'{elapsed:.1f}s {pct}% {message}', flush=True)
    try:
        with environment(NAKSHA_GROUND_ENGINE_V2=args.engine, NAKSHA_GROUND_WORKERS=args.workers,
                         NAKSHA_GROUND_GPU=0, NAKSHA_GROUND_LOCATOR=0, NAKSHA_GROUND_V2_AUDIT=0), patch.object(
                stream, '_read_window', read), patch.object(tools, 'Delaunay', TimedTri), patch.object(
                tools, 'classify_ground_ptd', classify), patch.object(v2, 'find_simplex', locate), Meter() as meter:
            planning_start = perf_counter()
            plan, info = stream.plan_streamed_ground(destination, [], params,
                expected_total=len(original), progress=progress)
            report['planning_seconds'] = perf_counter() - planning_start
        report.update(meter.metrics, plan_stats=plan['stats'])
        report['unique_planned_ids'] = len(np.unique(stream._identity(plan['node'], plan['local'])))
        assert report['unique_planned_ids'] == len(plan['node'])
        np.savez(out / 'plan.npz', **{k: plan[k] for k in ('node', 'local', 'old_classes', 'new_classes')})
        node_starts = np.full(len(entries), -1, dtype=np.int64)
        node_starts[leaves] = starts[:-1]
        rows = node_starts[plan['node']] + plan['local']
        np.testing.assert_array_equal(original[rows], plan['old_classes'])
        report['status'] = 'COMMITTING_COPY'
        dump(out / 'progress.json', report)
        commit_start = perf_counter()
        assert native.set_classes(plan['node'], plan['local'], plan['new_classes'], plan['old_classes']), native.last_error
        report['native_commit_seconds'] = perf_counter() - commit_start
        (out / 'committed.flag').write_text('Only this disposable copy was committed.\n')
        native.frame(0.)
        expected = original.copy()
        expected[rows] = plan['new_classes']
        readback_start = perf_counter()
        after, _, _, _ = read_classes(str(destination))
        report['readback_class_differences'] = int(np.count_nonzero(after != expected))
        assert report['readback_class_differences'] == 0
        report['readback_seconds'] = perf_counter() - readback_start
        report['full_operation_seconds'] = perf_counter() - operation_start
        report['user_visible_seconds_including_v2_import'] = report['full_operation_seconds'] + report['v2_import_seconds']
        report['performance_360_seconds'] = report['user_visible_seconds_including_v2_import'] <= 360 and not report['resumed']
        report['source_sha256_after'] = digest(args.source)
        assert report['source_sha256_after'] == source_hash
        report['status'] = 'COMPLETE'
        dump(out / 'result.json', report)
        dump(out / 'progress.json', report)
        print(json.dumps({k: report[k] for k in ('status', 'full_operation_seconds',
            'native_commit_seconds', 'unique_planned_ids', 'readback_class_differences',
            'performance_360_seconds')}, indent=2), flush=True)
    finally:
        native.close(); app.close(); qt.processEvents()


if __name__ == '__main__':
    main()
