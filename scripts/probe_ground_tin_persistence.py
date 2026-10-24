"""Bounded parity probe of persistent Qhull versus authoritative rebuilds.

This does not enable an alternative topology in the application. Failed row
ordering, geometry or thinning comparisons are explicit blockers, not passes.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.worker:
        try:
            result = subprocess.run([sys.executable, __file__, '--tile', str(args.tile),
                '--output', str(args.output), '--worker'], timeout=60)
            raise SystemExit(result.returncode)
        except subprocess.TimeoutExpired:
            args.output.write_text(json.dumps(dict(status='TIMEOUT', timeout=60)))
            raise SystemExit(2)
    import numpy as np
    from scipy.spatial import Delaunay, QhullError
    def connectivity(tri):
        rows = np.sort(tri.simplices, axis=1)
        return rows[np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))]
    report = dict(status='COMPLETE', cases=[], production_topology_changed=False)
    rng = np.random.default_rng(20261024)
    for name, points in [('regular', rng.random((400, 2))),
                         ('projected', rng.uniform(0, 160, (400, 2))+[256000.,4836500.])]:
        ref = Delaunay(points)
        candidate = None
        row = dict(name=name, input_order='same append-only')
        try:
            candidate = Delaunay(points[:200], incremental=True)
            candidate.add_points(points[200:])
            row.update(exact_simplex_rows=bool(np.array_equal(ref.simplices, candidate.simplices)),
                geometric_connectivity_equal=bool(np.array_equal(connectivity(ref), connectivity(candidate))),
                persistent_options=candidate._qhull.options.decode())
        except QhullError as exc:
            row.update(refused=True, error=str(exc)[:800])
        finally:
            if candidate is not None:
                candidate.close()
        report['cases'].append(row)
    with np.load(args.tile, allow_pickle=False) as tile:
        names = sorted((k for k in tile.files if k.startswith('xy') and k[2:].isdigit()),
                       key=lambda k:int(k[2:]))
        transitions=[]
        previous=tile[names[0]]
        for name in names[1:]:
            current=tile[name]
            unchanged=np.array_equal(previous.view(np.uint64), current.view(np.uint64))
            append_order=(len(current)>=len(previous) and
                          np.array_equal(previous.view(np.uint64),current[:len(previous)].view(np.uint64)))
            transitions.append(dict(to=name, unchanged=unchanged, append_order=append_order))
            previous=current
        report['real_input_transitions']=transitions
        initial=tile[names[0]]
        ref=Delaunay(initial)
        real=dict(name='real_initial', vertices=len(initial), reference_simplices=len(ref.simplices),
                  reference_used_vertices=len(np.unique(ref.simplices)))
        started=perf_counter()
        try:
            candidate=Delaunay(initial, incremental=True)
            real.update(exact_simplex_rows=bool(np.array_equal(ref.simplices,candidate.simplices)),
                geometric_connectivity_equal=bool(np.array_equal(connectivity(ref),connectivity(candidate))),
                persistent_simplices=len(candidate.simplices),
                persistent_used_vertices=len(np.unique(candidate.simplices)),
                persistent_options=candidate._qhull.options.decode())
            candidate.close()
        except QhullError as exc:
            real.update(refused=True,error=str(exc)[:800])
        real['seconds']=perf_counter()-started
        report['cases'].append(real)
    report['persistent_tin_production_ready']=False
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
