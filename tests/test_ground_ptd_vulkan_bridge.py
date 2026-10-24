"""No GPU required. Check Vulkan C ABI protocol, fail-closed behavior, CPU
shadow audit and unchanged actual PTD output when accelerator makes errors.
Hardware Vulkan dispatch validation is separately in scripts/validate_*.py.
"""
import ast
import ctypes
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay

from gui import ground_ptd_vulkan as vk


class FakeVulkanDll:
    def __init__(self, reject_all=False, fail=False):
        self.reject_all = reject_all
        self.fail = fail
        self.calls = 0
        self.inputs = []

    def nk_ground_vk_max_batch(self):
        return 32768

    def nk_ground_vk_filter(self, context, packed_ptr, count, max_distance,
                            guard, upward, keep_ptr, msg_ptr, msg_cap):
        self.calls += 1
        if self.fail:
            return -1
        packed = np.ctypeslib.as_array(packed_ptr, shape=(count * 9,)).reshape(count, 9)
        self.inputs.append(packed.copy())
        keep = np.ctypeslib.as_array(keep_ptr, shape=(count,))
        if self.reject_all:
            keep[:] = 0
        else:
            d = np.einsum('ij,ij->i', packed[:, 0:3] - packed[:, 3:6], packed[:, 6:9])
            keep[:] = ((np.abs(d) <= max_distance + guard) &
                      (not upward or d >= -guard)).astype(np.uint32)
        return 0


def _inputs(n=7000):
    xyz = np.zeros((n, 3), dtype=np.float64)
    xyz[:, 2] = np.linspace(-2., 2., n)
    anchors = np.zeros_like(xyz)
    normals = np.tile(np.array([0., 0., 1.]), (n, 1))
    return xyz, anchors, normals


def test_real_bridge_packs_double_without_changing_input(monkeypatch):
    fake = FakeVulkanDll()
    monkeypatch.setenv('NAKSHA_GROUND_VULKAN_COMPUTE', '1')
    monkeypatch.setattr(vk, '_get_context', lambda: (fake, ctypes.c_void_p(1)))
    coords = _inputs()
    original = tuple(a.copy() for a in coords)
    out = vk.conservative_vulkan_prefilter(*coords, 0.5, add_only_upward=True)
    assert out is not None
    assert out['backend'] == 'vulkan_fp64_compute_distance_gate'
    assert out['keep'].shape == (len(coords[0]),)
    assert fake.calls == 1
    np.testing.assert_array_equal(fake.inputs[0][:, :3], coords[0])
    np.testing.assert_array_equal(fake.inputs[0][:, 3:6], coords[1])
    np.testing.assert_array_equal(fake.inputs[0][:, 6:9], coords[2])
    for arr, before in zip(coords, original):
        np.testing.assert_array_equal(arr, before)


def test_batched_dispatch_and_fail_closed(monkeypatch):
    fake = FakeVulkanDll()
    monkeypatch.setenv('NAKSHA_GROUND_VULKAN_COMPUTE', '1')
    monkeypatch.setenv('NAKSHA_GROUND_VK_BATCH', '8192')
    monkeypatch.setattr(vk, '_get_context', lambda: (fake, ctypes.c_void_p(1)))
    r = vk.conservative_vulkan_prefilter(*_inputs(20000), 0.9)
    assert r and r['candidates'] == 20000
    assert fake.calls == 3
    fake.fail = True
    assert vk.conservative_vulkan_prefilter(*_inputs(), 0.9) is None


def test_nonfinite_and_bad_type_never_go_to_gpu(monkeypatch):
    fake = FakeVulkanDll()
    monkeypatch.setenv('NAKSHA_GROUND_VULKAN_COMPUTE', '1')
    monkeypatch.setattr(vk, '_get_context', lambda: (fake, ctypes.c_void_p(1)))
    x, a, n = _inputs()
    x[5, 2] = np.nan
    assert vk.conservative_vulkan_prefilter(x, a, n, 0.1) is None
    x[5, 2] = 0
    assert vk.conservative_vulkan_prefilter(x.astype('f4'), a, n, 0.1) is None
    assert fake.calls == 0


@pytest.fixture(scope='module')
def classify():
    filename = Path(__file__).resolve().parents[1] / 'gui' / 'lidar_classification_tools.py'
    nodes = ast.parse(filename.read_text(encoding='utf-8'), filename=str(filename)).body
    names = {'_select_ground_class0_holdout', '_subsample_ground_tin_points',
             '_bounded_ground_seed_indices', '_spatial_cell_low_indices',
             '_estimate_adaptive_ground_thresholds', 'classify_ground_ptd'}
    selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == names
    scope = {'np': np, 'Delaunay': Delaunay}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 str(filename), 'exec'), scope)
    return scope['classify_ground_ptd']


@pytest.mark.parametrize('fenced', [True, False])
@pytest.mark.parametrize('upward', [True, False])
def test_even_false_gpu_rejections_cannot_change_ptd_classes(
        monkeypatch, classify, fenced, upward):
    """Every rejected point must be audited before discarding, even if the
    Vulkan driver's output was completely wrong. This tests real PTD code.
    """
    monkeypatch.setenv('NAKSHA_GROUND_ENGINE_V2', '0')
    monkeypatch.setenv('NAKSHA_GROUND_CPP', '0')
    monkeypatch.setenv('NAKSHA_GROUND_PROFILE', '0')
    monkeypatch.setenv('NAKSHA_GROUND_EARLY_REJECT', '1')
    monkeypatch.setenv('NAKSHA_GROUND_GPU', '0')
    monkeypatch.setenv('NAKSHA_GROUND_VULKAN_COMPUTE', '0')
    rng = np.random.default_rng(410)
    xy = rng.uniform(0, 48, (7800, 2)) + [256000, 4837000]
    z = 120 + .03 * (xy[:, 0] - 256000) + rng.uniform(0, 6, len(xy))
    xyz = np.c_[xy, z]
    c0 = np.zeros(len(xyz), dtype=np.uint8)
    c0[::23] = 2
    fence = (xy[:, 0] < 256036) if fenced else None
    args = dict(from_classes=[0], current_ground=[2], to_class=2,
                max_building_size=25., add_only_upward=upward,
                terrascan_compatible_densification=True)
    baseline = c0.copy()
    ref = classify(xyz, baseline, fence_mask=fence, **args)

    fake = FakeVulkanDll(reject_all=True)
    monkeypatch.setenv('NAKSHA_GROUND_VULKAN_COMPUTE', '1')
    monkeypatch.setattr(vk, '_get_context', lambda: (fake, ctypes.c_void_p(1)))
    tested = c0.copy()
    result = classify(xyz, tested, fence_mask=fence, **args)
    assert fake.calls > 0, 'actual PTD must pass enough candidates to use GPU'
    np.testing.assert_array_equal(tested, baseline)
    for key in ref:
        np.testing.assert_equal(ref[key], result[key], err_msg=key)
