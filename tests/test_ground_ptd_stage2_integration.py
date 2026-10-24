"""Full real PTD function parity across opt-in sparse-angle execution.

Requires the normal Naksha test environment (PySide6, SciPy 1.15.3, etc.).
Does not load or modify a user LAZ/LNT file and never touches a live project.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from gui.lidar_classification_tools import classify_ground_ptd


def _sample(n=4400):
    rng = np.random.default_rng(136)
    xy = rng.uniform([256000., 4837000.], [256180., 4837180.], (n, 2))
    z = .025 * (xy[:, 0] - 256000) + .85 * np.sin((xy[:, 1] - 4837000) / 18)
    z += rng.choice([0., .02, .25, 3., 14.], n,
                    p=[.3, .3, .2, .15, .05])
    xyz = np.column_stack((xy, z + 85.))
    classification = np.zeros(n, dtype=np.uint8)
    classification[::47] = 1
    classification[::91] = 5
    return xyz, classification


@pytest.mark.parametrize('fenced', [False, True])
@pytest.mark.parametrize('changes', [
    {}, {'seed_method': 'lowest_only'}, {'seed_method': 'ground_only'},
    {'add_only_upward': True}, {'use_distance_as_rating': True},
    {'class0_holdout_percent': 7.}, {'adaptive_thresholds': True},
    {'stop_triangulation': True}, {'iteration_distance': 0.},
    {'reduce_angle_edge': False}, {'complete_ground_from_final_tin': False},
    {'terrascan_compatible_densification': False},
])
def test_rejected_angle_rows_have_identical_class_indices_and_iteration_state(
        monkeypatch, fenced, changes):
    xyz, original = _sample()
    mask = ((xyz[:, 0] < 256135.) & (xyz[:, 1] > 4837025.)) if fenced else None
    params = dict(from_classes=[0], to_class=1, current_ground=[1],
                  max_building_size=25., terrascan_compatible_densification=True)
    params.update(changes)
    monkeypatch.setenv('NAKSHA_GROUND_GPU', '0')
    monkeypatch.setenv('NAKSHA_GROUND_ENGINE_V2', '1')
    monkeypatch.setenv('NAKSHA_GROUND_PROFILE', '0')
    values = []
    for optimized in ('0', '1'):
        monkeypatch.setenv('NAKSHA_GROUND_EARLY_REJECT', optimized)
        copied = original.copy()
        steps = []
        result = classify_ground_ptd(
            xyz, copied, fence_mask=mask, **params,
            progress_cb=lambda pct, msg: steps.append((pct, msg)))
        values.append((copied, result, steps))
    np.testing.assert_array_equal(values[0][0], values[1][0])
    np.testing.assert_array_equal(values[0][1]['indices'], values[1][1]['indices'])
    for key, value in values[0][1].items():
        np.testing.assert_equal(value, values[1][1][key], err_msg=key)
    assert values[0][2] == values[1][2]
    if fenced:
        np.testing.assert_array_equal(values[1][0][~mask], original[~mask])


def test_actual_ptd_profile_is_valid_and_reports_cpu_gpu_separately(
        tmp_path, monkeypatch):
    path = tmp_path / 'ground_trace.jsonl'
    monkeypatch.setenv('NAKSHA_GROUND_ENGINE_V2', '1')
    monkeypatch.setenv('NAKSHA_GROUND_EARLY_REJECT', '1')
    monkeypatch.setenv('NAKSHA_GROUND_PROFILE', '1')
    monkeypatch.setenv('NAKSHA_GROUND_PROFILE_FILE', str(path))
    monkeypatch.setenv('NAKSHA_GROUND_GPU', '0')
    xyz, values = _sample(2000)
    result = classify_ground_ptd(xyz, values, from_classes=[0], to_class=1,
                      current_ground=[1], max_building_size=25.,
                      terrascan_compatible_densification=True)
    reports = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(reports) == 1
    diag = reports[0]
    assert diag['gpu']['requested'] is False
    assert diag['gpu']['calls'] == 0
    assert diag['backend']['v2_active'] is True
    assert diag['early_reject'] is True
    assert diag['iterations'] == result['iterations']
    assert diag['total_seconds'] >= 0
    assert sum(diag['stages_seconds'].values()) <= diag['total_seconds'] + .001
