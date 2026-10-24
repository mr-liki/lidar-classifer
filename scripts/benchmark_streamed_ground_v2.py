"""Read-only differential on ORIGINAL streamed Ground cells, with original halos.

One original top-level cell is selected for this benchmark. Tile
size, context/splits, seed inputs, candidate order, limits and worker settings
are unchanged. No command here commits classifications.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_ground_ptd_v2 import Meter, environment
from gui import native_ground_below_stream as stream
from gui import lidar_classification_tools as tools
from gui import ground_ptd_engine_v2 as v2
from scipy.spatial import Delaunay


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest(handle)


def _digest(handle):
    value = hashlib.sha256()
    for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
        value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lnt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cells', nargs=1, type=int, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--source-class', type=int, required=True)
    parser.add_argument('--ground-class', type=int, required=True)
    args = parser.parse_args()
    info = stream.inspect_directory(args.lnt, max_points=2**63-1)
    regions = stream._regions(info, [], 110.)
    chosen = regions if args.cells is None else [regions[index] for index in args.cells]
    # Stage instrumentation shares counters, so benchmark one original cell
    # at a time. Internal recursive splits stay on its original worker.
    if len(chosen) > 1:
        raise ValueError('Select one original cell per benchmark for uncontended stage timings.')
    params = dict(from_classes=[args.source_class], to_class=args.ground_class,
                  current_ground=[args.ground_class], terrascan_compatible_densification=True)
    report = dict(source=str(args.lnt), total_points=info['total'], cells=args.cells,
                  workers=args.workers, params=params, source_sha256_before=digest(args.lnt),
                  native_commit_seconds=None, full_operation='NOT RUN', runs=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plans = []
    for enabled in ('0', '1'):
        stats = dict(tin_seconds=0., lookup_seconds=0., tin_builds=0, queries=0,
                     cells=[], engine=enabled)
        classify_original = tools.classify_ground_ptd
        locate_original = v2.find_simplex
        class TimedTri:
            def __init__(self, xy):
                start = perf_counter()
                self.tri = Delaunay(xy)
                stats['tin_seconds'] += perf_counter() - start
                stats['tin_builds'] += 1
            def __getattr__(self, key):
                return getattr(self.tri, key)
            def find_simplex(self, xy):
                start = perf_counter()
                result = self.tri.find_simplex(xy)
                stats['lookup_seconds'] += perf_counter() - start
                stats['queries'] += len(xy)
                return result
        def locate(tri, xy, **kwargs):
            before, q = stats['lookup_seconds'], stats['queries']
            start = perf_counter()
            result = locate_original(tri, xy, **kwargs)
            stats['lookup_seconds'] = before + perf_counter() - start
            stats['queries'] = q + len(xy)
            return result
        def classify(**kwargs):
            start = perf_counter()
            result = classify_original(**kwargs)
            stats['cells'].append(dict(points=len(kwargs['xyz']), seconds=perf_counter()-start,
                iterations=result['iterations'], seed_count=result['seed_count'],
                stop_reason=result['stop_reason']))
            return result
        print(f'Planning original cells {args.cells}, engine={enabled}', flush=True)
        with environment(NAKSHA_GROUND_ENGINE_V2=enabled, NAKSHA_GROUND_WORKERS=args.workers,
                         NAKSHA_GROUND_GPU=0, NAKSHA_GROUND_LOCATOR=0), patch.object(
                stream, '_regions', return_value=chosen), patch.object(tools, 'Delaunay', TimedTri), patch.object(
                tools, 'classify_ground_ptd', classify), patch.object(v2, 'find_simplex', locate), Meter() as meter:
            plan, _ = stream.plan_streamed_ground(args.lnt, [], params, expected_total=info['total'])
        stats.update(meter.metrics, plan_stats=plan['stats'])
        stats['other_ptd_seconds'] = sum(cell['seconds'] for cell in stats['cells']) - stats['tin_seconds'] - stats['lookup_seconds']
        plans.append(plan)
        report['runs'].append(stats)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Engine {enabled}: {meter.seconds:.3f}s, {len(plan["node"]):,} planned changes', flush=True)
    for key in ('node', 'local', 'old_classes', 'new_classes', 'file_stat'):
        np.testing.assert_equal(plans[0][key], plans[1][key], err_msg=key)
    report['source_sha256_after'] = digest(args.lnt)
    assert report['source_sha256_before'] == report['source_sha256_after']
    report['exact_plan_parity'] = True
    report['speedup'] = report['runs'][0]['wall_seconds'] / report['runs'][1]['wall_seconds']
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(exact_plan_parity=True, speedup=report['speedup'])), flush=True)


if __name__ == '__main__':
    main()
