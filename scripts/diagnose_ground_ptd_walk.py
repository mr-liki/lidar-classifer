"""Bounded native replay of a captured complete-query SciPy walk.

Uses a PREFIX, preserving the initial start state and original query order.
The parent enforces a process timeout. No interpreted geometry walks run here.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--max-queries', type=int, default=40000)
    parser.add_argument('--timeout', type=float, default=30.)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.max_queries <= 1_000_000 or not 1 <= args.timeout <= 120:
        parser.error('queries must be 1..1,000,000; timeout 1..120 seconds')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.worker:
        try:
            result = subprocess.run([sys.executable, __file__, str(args.capture),
                '--max-queries', str(args.max_queries), '--timeout', str(args.timeout),
                '--output', str(args.output), '--worker'], timeout=args.timeout)
            raise SystemExit(result.returncode)
        except subprocess.TimeoutExpired:
            args.output.write_text(json.dumps(dict(status='TIMEOUT', timeout=args.timeout)))
            raise SystemExit(2)
    import numpy as np
    from gui.ground_ptd_incremental import trace_ordered_walk
    with np.load(args.capture, allow_pickle=False) as data:
        n = min(args.max_queries, len(data['queries']))
        points = data['pts'][data['queries'][:n], :2]
        tri = SimpleNamespace(neighbors=data['neighbors'], equations=data['equations'],
            transform=data['transform'], min_bound=data['low'], max_bound=data['high'],
            paraboloid_scale=float(data['scale']), paraboloid_shift=float(data['shift']))
        started = perf_counter()
        observed, trace = trace_ordered_walk(tri, points)
        elapsed = perf_counter() - started
        differences = np.flatnonzero(observed != data['ref'][:n])
        old = np.flatnonzero(data['eng'][:n] != data['ref'][:n])
        report = dict(status='PASS' if not len(differences) else 'FAIL', queries=n,
            native_seconds=elapsed, raw_simplex_differences=len(differences),
            captured_shortcut_differences=len(old), max_paraboloid_cycles=int(trace[:, 2].max()),
            max_directed_cycles=int(trace[:, 3].max()), brute_fallbacks=int(trace[:, 4].sum()),
            first_captured_mismatch_query=int(data['queries'][old[0]]) if len(old) else None)
        if len(old):
            i = int(old[0])
            report['mismatch_replay'] = dict(query_index=i, before=int(data['eng'][i]),
                reference=int(data['ref'][i]), repaired=int(observed[i]), trace=trace[i].tolist())
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
