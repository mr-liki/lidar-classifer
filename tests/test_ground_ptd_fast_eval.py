"""Exact-operator early rejection tests; no Qt, dataset or C++ DLL required."""
import json

import numpy as np
import pytest

from gui.ground_ptd_fast_eval import sparse_angle_filter
from gui.ground_ptd_profiler import report_ptd


def reference(tri_points, simplices, inverse, pa, pn, pme, d, pd, base, *,
              iteration_angle, reduce_angle_edge, edge_length_threshold,
              use_distance_as_rating, distance_weight, iteration_distance):
    mask = base.copy()
    pp = pa - d[:, np.newaxis] * pn
    psv = simplices[inverse]
    nr = np.minimum(
        np.minimum(np.linalg.norm(tri_points[psv[:, 0]] - pp, axis=1),
                   np.linalg.norm(tri_points[psv[:, 1]] - pp, axis=1)),
        np.linalg.norm(tri_points[psv[:, 2]] - pp, axis=1),
    )
    alpha = np.where(nr < 1e-12, 90.0, np.degrees(np.arctan2(pd, nr)))
    ea = np.full(len(pa), iteration_angle, dtype=float)
    if reduce_angle_edge:
        sm = pme < edge_length_threshold
        ea[sm] = iteration_angle * pme[sm] / edge_length_threshold
    if use_distance_as_rating:
        w = distance_weight / 100.0
        mask &= ((1 - w) * np.where(ea > 0, alpha / ea, 1.0)
                 + w * (pd / iteration_distance if iteration_distance > 0
                        else np.ones_like(pd))) <= 1.0
    else:
        mask &= alpha <= ea
    return mask, alpha


@pytest.mark.parametrize('seed', [1, 19, 71])
@pytest.mark.parametrize('num', [0, 1, 50, 13000])
@pytest.mark.parametrize('rating', [False, True])
@pytest.mark.parametrize('reduce', [False, True])
def test_sparse_exact_mask_and_active_angles(seed, num, rating, reduce):
    rng = np.random.default_rng(seed)
    ground = rng.normal(0, 30, (300, 3)).astype(np.float64)
    ground[:, 0] += 256000
    ground[:, 1] += 4837000
    simplices = rng.integers(0, 300, (100, 3))
    pa = rng.normal(0, 5, (num, 3)).astype(np.float64)
    pa[:, 0] += 256000
    pa[:, 1] += 4837000
    pn = rng.normal(0, 1, (num, 3))
    pn /= np.linalg.norm(pn, axis=1, keepdims=True)
    inv = rng.integers(0, len(simplices), num)
    pme = rng.uniform(0.01, 20, num)
    pA = ground[simplices[inv, 0]]
    d = np.einsum('ij,ij->i', pn, pa - pA)
    pd = np.abs(d)
    preliminary = (rng.random(num) > .9) & (pd <= 1.4)
    kwargs = dict(iteration_angle=6., reduce_angle_edge=reduce,
                  edge_length_threshold=5., use_distance_as_rating=rating,
                  distance_weight=65., iteration_distance=1.4)
    exp, angles_exp = reference(ground, simplices, inv, pa, pn,
                                pme, d, pd, preliminary, **kwargs)
    actual, angles_actual = sparse_angle_filter(
        ground, simplices, inv, pa, pn, pme, d, pd, preliminary, **kwargs)
    np.testing.assert_array_equal(actual, exp)
    np.testing.assert_array_equal(angles_actual[preliminary], angles_exp[preliminary])
    assert actual is not preliminary


def test_skips_expensive_triangle_geometry_if_mandatory_checks_fail(monkeypatch):
    tri = np.ones((3, 3), dtype=float)
    simplex = np.array([[0, 1, 2]], dtype=np.intp)
    zeros = np.zeros((20, 3), dtype=float)
    mask = np.zeros(20, dtype=bool)
    # All point indices out-of-range would raise if referenced. This proves
    # mandatory-rejected rows are not accessed by the sparse triangle kernel.
    inv = np.full(20, 999, dtype=np.intp)
    selected, angles = sparse_angle_filter(tri, simplex, inv, zeros, zeros,
        np.zeros(20), np.zeros(20), np.zeros(20), mask,
        iteration_angle=6., reduce_angle_edge=False, edge_length_threshold=5.,
        use_distance_as_rating=False, distance_weight=50., iteration_distance=1.)
    assert not selected.any()
    assert np.array_equal(angles, np.zeros(20))


def test_report_distinguishes_render_gpu_from_actual_cuda(tmp_path, monkeypatch, capsys):
    logfile = tmp_path / 'stage2.jsonl'
    monkeypatch.setenv('NAKSHA_GROUND_PROFILE', '1')
    monkeypatch.setenv('NAKSHA_GROUND_PROFILE_FILE', str(logfile))
    report_ptd(dict(working_points=500, source_candidates=220, iterations=7,
                    total_seconds=1.5, stages_seconds=dict(tin=.5, simplex=.3,
                    candidate_geometry=.5), backend=dict(cpp_queries=110,
                    numba_queries=0, scipy_queries=0, audited_queries=110),
                    gpu=dict(requested=True, calls=0, failed_calls=2,
                             rejected=0, seconds=0.0)))
    out = capsys.readouterr().out
    assert 'C++=YES' in out
    assert 'GroundCUDA=NO' in out
    assert 'VulkanRendering=SEPARATE' in out
    record = json.loads(logfile.read_text())
    assert record['gpu']['failed_calls'] == 2
    assert record['backend']['cpp_queries'] == 110
