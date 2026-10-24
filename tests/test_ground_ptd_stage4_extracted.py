"""Independent full PTD parity test without importing the Qt/Vulkan GUI.

Extract the UNCHANGED PTD functions from the current working source via AST.
This deliberately tests the actual BRUSH PTD implementation rather than
reimplementing its geometry or altering its TerraScan parameters.
"""
import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay

from gui import ground_ptd_engine_v2 as v2


@pytest.fixture(scope='module')
def classify():
    source = Path(__file__).resolve().parents[1] / 'gui' / 'lidar_classification_tools.py'
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    required = {
        '_select_ground_class0_holdout', '_subsample_ground_tin_points',
        '_bounded_ground_seed_indices', '_spatial_cell_low_indices',
        '_estimate_adaptive_ground_thresholds', 'classify_ground_ptd',
    }
    extracted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in required]
    if {n.name for n in extracted} != required:
        pytest.fail('PTD source layout changed; manual review required')
    scope = {'np': np, 'Delaunay': Delaunay}
    module = ast.Module(body=extracted, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), scope)
    return scope['classify_ground_ptd']


def _terrain():
    rng = np.random.default_rng(20261024)
    xy = rng.uniform([256000., 4837000.], [256180., 4837180.], (5700, 2))
    z = 100 + .015 * (xy[:, 0] - 256000.) + .55 * np.sin((xy[:, 1] - 4837000.) / 12)
    z += rng.choice([0., .01, .15, 3., 10.], len(xy), p=[.30, .25, .20, .20, .05])
    cl = np.zeros(len(xy), dtype=np.uint8)
    cl[::29] = 2
    cl[::37] = 5
    return np.c_[xy, z], cl


@pytest.mark.parametrize('fenced', [False, True])
@pytest.mark.parametrize('changed_settings', [
    {}, {'seed_method': 'lowest_only'}, {'add_only_upward': True},
    {'use_distance_as_rating': True}, {'stop_triangulation': True},
    {'complete_ground_from_final_tin': False},
])
def test_full_ptd_same_classes_and_iteration_structure(
        monkeypatch, classify, fenced, changed_settings):
    monkeypatch.setattr(v2.scipy, '__version__', '1.15.3')
    monkeypatch.setenv('NAKSHA_GROUND_GPU', '0')
    monkeypatch.setenv('NAKSHA_GROUND_PROFILE', '0')
    monkeypatch.setenv('NAKSHA_GROUND_EARLY_REJECT', '1')
    xyz, original = _terrain()
    fence = (xyz[:, 0] < 256150) & (xyz[:, 1] > 4837020) if fenced else None
    kwargs = dict(from_classes=[0, 5], to_class=2, current_ground=[2],
                  max_building_size=25., terrascan_compatible_densification=True)
    kwargs.update(changed_settings)
    comparison=[]
    for cpp_flag, v2_flag, stage4_flag in (
            ('0', '0', '0'), ('1', '1', '1')):
        monkeypatch.setenv('NAKSHA_GROUND_CPP', cpp_flag)
        monkeypatch.setenv('NAKSHA_GROUND_ENGINE_V2', v2_flag)
        monkeypatch.setenv('NAKSHA_GROUND_CPP_GRID', '1')
        monkeypatch.setenv('NAKSHA_GROUND_CPP_STAGE4', stage4_flag)
        monkeypatch.setenv('NAKSHA_GROUND_V2_AUDIT', '1')
        points_cls = original.copy()
        messages=[]
        result = classify(xyz, points_cls, fence_mask=fence,
                          progress_cb=lambda p, m: messages.append((p,m)), **kwargs)
        comparison.append((points_cls, result, messages))
    ref, observed=comparison
    np.testing.assert_array_equal(ref[0], observed[0])
    for k in ref[1]:
        np.testing.assert_equal(ref[1][k], observed[1][k], err_msg=k)
    assert ref[2] == observed[2]
    if fenced:
        np.testing.assert_array_equal(observed[0][~fence], original[~fence])
