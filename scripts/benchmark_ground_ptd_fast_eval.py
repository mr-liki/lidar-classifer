"""Standalone synthetic angle-kernel benchmark, NOT full Ground PTD speed.

Parses no LNT and changes no classes. Reports exact masks/angles for rows still
eligible after mandatory slope, edge, distance and upward filters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gui.ground_ptd_fast_eval import sparse_angle_filter


def reference(points, simplices, inverse, pa, pn, edge, d, pd, base, **cfg):
    out = base.copy()
    pp = pa - d[:, np.newaxis] * pn
    psv = simplices[inverse]
    nearest = np.minimum(np.minimum(
        np.linalg.norm(points[psv[:, 0]] - pp, axis=1),
        np.linalg.norm(points[psv[:, 1]] - pp, axis=1)),
        np.linalg.norm(points[psv[:, 2]] - pp, axis=1))
    alpha = np.where(nearest < 1e-12, 90.0,
                     np.degrees(np.arctan2(pd, nearest)))
    allowed = np.full(len(pa), cfg['iteration_angle'], dtype=float)
    if cfg['reduce_angle_edge']:
        sm = edge < cfg['edge_length_threshold']
        allowed[sm] = cfg['iteration_angle'] * edge[sm] / cfg['edge_length_threshold']
    if cfg['use_distance_as_rating']:
        w = cfg['distance_weight'] / 100.0
        out &= ((1 - w) * np.where(allowed > 0, alpha / allowed, 1.0)
                + w * (pd / cfg['iteration_distance'])) <= 1.0
    else:
        out &= alpha <= allowed
    return out, alpha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=int, default=160000)
    parser.add_argument('--survival', type=float, default=.1,
                        help='Fraction surviving mandatory pre-angle gates (0..1).')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.candidates <= 2_000_000 or not 0 <= args.survival <= 1 or not 1 <= args.runs <= 30:
        parser.error('Use valid --candidates, --survival (0-1), and --runs')
    rng = np.random.default_rng(841)
    xyz = rng.uniform(0, 200, (2500, 3))
    triangles = rng.integers(0, len(xyz), (5000, 3))
    inv = rng.integers(0, len(triangles), args.candidates)
    pa = rng.uniform(0, 200, (args.candidates, 3))
    pn = rng.normal(size=pa.shape)
    pn /= np.linalg.norm(pn, axis=1, keepdims=True)
    edge = rng.uniform(.01, 30., args.candidates)
    pA = xyz[triangles[inv, 0]]
    d = np.einsum('ij,ij->i', pn, pa - pA)
    pd = np.abs(d)
    # Deterministic representative gate distribution. These are not real data.
    pre = rng.random(args.candidates) < args.survival
    cfg = dict(iteration_angle=6., reduce_angle_edge=True,
               edge_length_threshold=5., use_distance_as_rating=True,
               distance_weight=50., iteration_distance=1.4)
    samples = dict(reference=[], sparse=[])
    expected = reference(xyz, triangles, inv, pa, pn, edge, d, pd, pre, **cfg)
    np.testing.assert_array_equal(*(
        expected[0], sparse_angle_filter(xyz, triangles, inv, pa, pn,
            edge, d, pd, pre, **cfg)[0]))
    for _ in range(args.runs):
        for key, func in (('reference', reference), ('sparse', sparse_angle_filter)):
            begin = perf_counter()
            output = func(xyz, triangles, inv, pa, pn, edge, d, pd, pre, **cfg)
            samples[key].append(perf_counter() - begin)
            np.testing.assert_array_equal(output[0], expected[0])
            np.testing.assert_array_equal(output[1][pre], expected[1][pre])
    old = median(samples['reference'])
    new = median(samples['sparse'])
    report = dict(scope='isolated synthetic angle-kernel ONLY; no LNT/TIN/lookup',
                  candidates=args.candidates, mandatory_gate_survival=args.survival,
                  medians_seconds=dict(reference=old, sparse=new),
                  speedup=(old/new if new>0 else None),
                  exact_parity='all masks + active angles matched',
                  full_cloud_target_proven=False,
                  raw_times_seconds=samples)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
