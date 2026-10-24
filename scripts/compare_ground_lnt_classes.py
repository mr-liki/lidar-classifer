"""Read-only exact LNT differential: CPU vs Vulkan Ground classifications.

Compares all packed point fields other than classification, plus the exact
class byte at every (node_id, local_point_id). Never touches either input.
Use copies of the identical initial LNT before CPU and GPU runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


class LNTComparisonError(RuntimeError):
    pass


def compare_lnt(left, right, *, inspector=None, point_dtype=None,
                exactly=None, chunk=220_000, example_limit=10):
    if inspector is None or point_dtype is None or exactly is None:
        from gui.native_lnt_directory import inspect_directory, POINT, _exactly
        inspector = inspect_directory if inspector is None else inspector
        point_dtype = POINT if point_dtype is None else point_dtype
        exactly = _exactly if exactly is None else exactly
    left, right = Path(left), Path(right)
    if left.resolve() == right.resolve():
        raise LNTComparisonError('Two different LNT files are required.')
    a = inspector(left, max_points=2**63-1)
    b = inspector(right, max_points=2**63-1)
    if a['total'] != b['total'] or len(a['leaves']) != len(b['leaves']):
        raise LNTComparisonError('Point counts/leaf counts differ; not the same original LNT.')
    if not np.array_equal(a['origin'], b['origin']):
        raise LNTComparisonError('Different LNT coordinate origins.')
    if 'classification' not in point_dtype.names:
        raise LNTComparisonError('Packed point dtype missing classification.')
    verify_names = tuple(name for name in point_dtype.names if name != 'classification')
    differences = 0
    checked = 0
    first = []
    with open(left,'rb') as f0, open(right,'rb') as f1:
        for aa,bb in zip(a['leaves'],b['leaves']):
            na,offa,n,ca,ha = aa
            nb,offb,m,cb,hb = bb
            if na != nb or n != m or not np.array_equal(ca,cb) or ha != hb:
                raise LNTComparisonError(f'Packed node layout differs: {na!r} vs {nb!r}.')
            for start in range(0,int(n),chunk):
                count=min(chunk,int(n)-start)
                f0.seek(int(offa)+start*point_dtype.itemsize)
                f1.seek(int(offb)+start*point_dtype.itemsize)
                x=np.frombuffer(exactly(f0,count*point_dtype.itemsize,
                                        f'left leaf {na}'),dtype=point_dtype)
                y=np.frombuffer(exactly(f1,count*point_dtype.itemsize,
                                        f'right leaf {nb}'),dtype=point_dtype)
                for name in verify_names:
                    if not np.array_equal(x[name],y[name]):
                        raise LNTComparisonError(
                            f'Non-classification data changed: node={na} local={start} field={name}.')
                mismatch=np.flatnonzero(x['classification'] != y['classification'])
                differences += int(len(mismatch))
                checked += count
                for index in mismatch[:max(0,example_limit-len(first))]:
                    first.append((int(na),int(start+index),
                                  int(x['classification'][index]),
                                  int(y['classification'][index])))
    if checked != int(a['total']):
        raise LNTComparisonError('Unexpected LNT leaf sum vs declared total.')
    return {'points':checked, 'different_classifications':differences,
            'identical':differences==0,'examples':first}


def main():
    parser=argparse.ArgumentParser(description='Read-only exact class-byte and point-identity LNT comparison')
    parser.add_argument('cpu_lnt', help='CPU result LNT, from a copy of the original data')
    parser.add_argument('gpu_lnt', help='GPU result LNT, from a separate copy of the same original data')
    args=parser.parse_args()
    try:
        report=compare_lnt(args.cpu_lnt,args.gpu_lnt)
    except LNTComparisonError as err:
        print('[GROUND LNT COMPARE] INVALID COMPARISON:',err,flush=True)
        raise SystemExit(2) from err
    print(f'[GROUND LNT COMPARE] points={report["points"]:,} '
          f'differing_classes={report["different_classifications"]:,} '
          f'exact_parity={"YES" if report["identical"] else "NO"}',flush=True)
    for node,local,cls0,cls1 in report['examples']:
        print(f'  mismatch node={node} local={local} CPU={cls0} GPU={cls1}')
    raise SystemExit(0 if report['identical'] else 1)


if __name__ == '__main__':
    main()
