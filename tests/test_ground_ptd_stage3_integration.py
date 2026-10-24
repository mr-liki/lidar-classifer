"""End-to-end Ground PTD class and iteration parity for Stage 3 on Windows.

Run on the validated SciPy 1.15.3 Naksha environment with the freshly built
Stage 3 DLL. The GUI need not open, but PySide6 must be importable.
"""
import numpy as np
import pytest
from gui import lidar_classification_tools as tools


def _scene():
    rng = np.random.default_rng(20261024)
    xy = rng.uniform([256000., 4837000.], [256180., 4837180.], (5700, 2))
    z = 100 + .015 * (xy[:, 0] - 256000.) + .55 * np.sin((xy[:, 1] - 4837000.) / 12)
    z += rng.choice([0., .01, .15, 3., 10.], len(xy),
                    p=[.30, .25, .20, .20, .05])
    cl = np.zeros(len(xy), dtype=np.uint8)
    cl[::29] = 2
    cl[::37] = 5
    return np.c_[xy, z], cl


@pytest.mark.parametrize('fenced', [False, True])
@pytest.mark.parametrize('changes', [
    {}, {'seed_method': 'lowest_only'}, {'add_only_upward': True},
    {'use_distance_as_rating': True}, {'stop_triangulation': True},
    {'complete_ground_from_final_tin': False}
])
def test_stage3_identical_full_ptd_result(monkeypatch, fenced, changes):
    xyz, orig = _scene()
    fence = ((xyz[:, 0] < 256150) & (xyz[:, 1] > 4837020)) if fenced else None
    monkeypatch.setenv('NAKSHA_GROUND_CPP', '1')
    monkeypatch.setenv('NAKSHA_GROUND_ENGINE_V2', '1')
    monkeypatch.setenv('NAKSHA_GROUND_GPU', '0')
    monkeypatch.setenv('NAKSHA_GROUND_V2_AUDIT', '1')
    monkeypatch.setenv('NAKSHA_GROUND_EARLY_REJECT', '1')
    monkeypatch.setenv('NAKSHA_GROUND_PROFILE', '0')
    kwargs = dict(from_classes=[0, 5], to_class=2, current_ground=[2],
                  max_building_size=25., terrascan_compatible_densification=True)
    kwargs.update(changes)
    reference = None
    for grid in ('0', '1'):
        monkeypatch.setenv('NAKSHA_GROUND_CPP_GRID', grid)
        work = orig.copy()
        result = tools.classify_ground_ptd(xyz, work, fence_mask=fence, **kwargs)
        if reference is None:
            reference = work.copy(), result
        else:
            np.testing.assert_array_equal(work, reference[0])
            for key in result:
                np.testing.assert_equal(result[key], reference[1][key], err_msg=key)
        if fenced:
            np.testing.assert_array_equal(work[~fence], orig[~fence])
