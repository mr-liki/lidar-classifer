"""Read-only reproducible Ground V0/V2 differential and stage benchmarks.

Never opens a dataset for writing. Full-operation/commit acceptance is NOT
inferred from these component/cell/planning measurements.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from threading import Event, Thread
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
import psutil
from scipy.spatial import Delaunay
from gui import lidar_classification_tools as tools
from gui import ground_ptd_engine_v2 as v2
from gui import native_ground_below_stream as stream


@contextmanager
def environment(**changes):
    before = {key: os.environ.get(key) for key in changes}
    os.environ.update({key: str(value) for key, value in changes.items()})
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class Meter:
    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.cpu = sum(self.process.cpu_times()[:2])
        self.started = perf_counter()
        self.stop = Event()
        def sample():
            while not self.stop.wait(.02):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.seconds = perf_counter() - self.started
        self.stop.set(); self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.metrics = dict(wall_seconds=self.seconds, peak_rss_gb=self.peak / 1e9,
            cpu_percent_of_machine=100 * (sum(self.process.cpu_times()[:2]) - self.cpu)
                                   / self.seconds / psutil.cpu_count())


def classify(xyz, original, params, enabled):
    stats = dict(tin_seconds=0., lookup_seconds=0., tin_builds=0,
                 lookup_calls=0, queries=0)
    anchors, messages = [], []
    sampler = tools._subsample_ground_tin_points
    locator = v2.find_simplex
    def timed_lookup(tri, xy, **kwargs):
        before = stats['lookup_seconds']
        count_before = stats['lookup_calls']
        started = perf_counter()
        result = locator(tri, xy, **kwargs)
        stats['lookup_seconds'] = before + perf_counter() - started
        if stats['lookup_calls'] == count_before:
            stats['lookup_calls'] += 1
            stats['queries'] += len(xy)
        return result
    def sample(points, ids, *args, **kwargs):
        anchors.append(hashlib.sha256(ids.tobytes()).hexdigest())
        return sampler(points, ids, *args, **kwargs)
    class TimedTri:
        def __init__(self, xy):
            started = perf_counter()
            self.tri = Delaunay(xy)
            stats['tin_seconds'] += perf_counter() - started
            stats['tin_builds'] += 1
        def __getattr__(self, name):
            return getattr(self.tri, name)
        def find_simplex(self, xy):
            started = perf_counter()
            result = self.tri.find_simplex(xy)
            stats['lookup_seconds'] += perf_counter() - started
            stats['lookup_calls'] += 1
            stats['queries'] += len(xy)
            return result
    values = original.copy()
    with environment(NAKSHA_GROUND_ENGINE_V2=enabled,
                     NAKSHA_GROUND_LOCATOR=0, NAKSHA_GROUND_GPU=0):
        with patch.object(tools, 'Delaunay', TimedTri), patch.object(
                tools, '_subsample_ground_tin_points', sample), patch.object(
                v2, 'find_simplex', timed_lookup), Meter() as meter:
            result = tools.classify_ground_ptd(xyz, values, **params,
                         progress_cb=lambda pct, msg: messages.append((pct, msg)))
    stats.update(meter.metrics)
    stats['other_ptd_seconds'] = meter.seconds - stats['tin_seconds'] - stats['lookup_seconds']
    stats.update(points=len(xyz), iterations=result['iterations'],
                 seed_count=result['seed_count'], changed=result['changed'],
                 stop_reason=result['stop_reason'])
    return values, result, anchors, messages, stats


def compare(left, right):
    differences = int(np.count_nonzero(left[0] != right[0]))
    assert differences == 0, f'{differences} different point classes'
    for key in left[1]:
        np.testing.assert_equal(left[1][key], right[1][key], err_msg=key)
    assert left[2] == right[2], 'anchor identities differ'
    assert left[3] == right[3], 'iteration/stop traces differ'
    return dict(class_differences=0, exact_indices=True,
                exact_result_fields=True, exact_anchor_traces=True)


def synthetic(n):
    rng = np.random.default_rng(7641)
    xy = rng.uniform(0., 180., (n, 2))
    z = 90 + .03 * xy[:, 0] + .8 * np.sin(xy[:, 1] / 17.)
    z += rng.choice([0., .03, .4, 3., 15.], n, p=[.3, .3, .2, .15, .05])
    cls = np.zeros(n, dtype=np.uint8)
    cls[::101] = 1
    return np.column_stack((xy + [256000., 4837000.], z)), cls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--points', type=int, default=60000)
    parser.add_argument('--lnt', type=Path)
    parser.add_argument('--cells', type=int, nargs='+', default=[7, 81])
    parser.add_argument('--context-half-size', type=float, default=60.)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--source-class', type=int, default=0)
    parser.add_argument('--ground-class', type=int, default=1)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(runtime=sys.version, logical_cpus=psutil.cpu_count(),
        available_ram_gb=psutil.virtual_memory().available / 1e9,
        mode='read-only component/cell benchmark; no native commit',
        full_marcena_operation='NOT RUN', native_commit_seconds=None, cases=[])
    def save():
        args.output.write_text(json.dumps(report, indent=2,
            default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item()),
            encoding='utf-8')
    params = dict(from_classes=[args.source_class], to_class=args.ground_class,
                  current_ground=[args.ground_class], max_building_size=60.,
                  terrascan_compatible_densification=True)
    # First V2 invocation includes any visible cold import/JIT/cache delay.
    xyz, classes = synthetic(args.points)
    baseline = classify(xyz, classes, params, '0')
    candidate = classify(xyz, classes, params, '1')
    report['cold_synthetic'] = dict(baseline=baseline[4], v2=candidate[4],
                                     parity=compare(baseline, candidate))
    save()
    workloads = [('synthetic_multi_iteration', xyz, classes, None)]
    source_stat = None
    if args.lnt:
        info = stream.inspect_directory(args.lnt, max_points=2**63-1)
        source_stat = info['size'], info['mtime_ns']
        report['source'] = {key: info[key] for key in ('path', 'total', 'size', 'mtime_ns')}
        regions = stream._regions(info, [], 110.)
        for index in args.cells:
            core = regions[index]
            cx, cy = .5 * (core[0] + core[2]), .5 * (core[1] + core[3])
            half = args.context_half_size
            bounds = cx-half, cy-half, cx+half, cy+half
            started = perf_counter()
            tile = stream._read_window(args.lnt, info, bounds,
                allowed_classes={args.source_class, args.ground_class}, max_context=2800000)
            read = perf_counter() - started
            report.setdefault('real_windows', []).append(dict(cell=index, bounds=bounds,
                read_seconds=read, points=len(tile['xyz']),
                source_count=int(np.count_nonzero(tile['classification'] == args.source_class)),
                ground_count=int(np.count_nonzero(tile['classification'] == args.ground_class))))
            workloads.append((f'MARCENA_cell_{index}', tile['xyz'], tile['classification'], read))
            print(f'Read real cell {index}: {len(tile["xyz"]):,} points in {read:.3f}s', flush=True)
    for name, xyz, classes, read in workloads:
        case = dict(name=name, read_seconds=read, runs=[])
        report['cases'].append(case)
        for repeat in range(args.repeats):
            results = {}
            # Alternate order to reduce systematic warm-cache bias.
            for enabled in (('0', '1') if repeat % 2 == 0 else ('1', '0')):
                print(f'{name}: repeat {repeat+1}, engine {enabled}', flush=True)
                results[enabled] = classify(xyz, classes, params, enabled)
            case['runs'].append(dict(baseline=results['0'][4], v2=results['1'][4],
                                      parity=compare(results['0'], results['1'])))
            save()
        b = np.median([run['baseline']['wall_seconds'] for run in case['runs']])
        c = np.median([run['v2']['wall_seconds'] for run in case['runs']])
        case.update(baseline_median_seconds=float(b), v2_median_seconds=float(c),
                    speedup=float(b/c))
        print(f'{name}: baseline {b:.3f}s, V2 {c:.3f}s, speedup {b/c:.3f}x', flush=True)
        save()
    if args.lnt:
        after = args.lnt.stat()
        assert source_stat == (after.st_size, after.st_mtime_ns), 'Source revision changed'
        report['source_revision_unchanged'] = True
    save()


if __name__ == '__main__':
    main()
