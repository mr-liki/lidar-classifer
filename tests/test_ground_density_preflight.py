"""Independent tests for read-only LNT density preflight and stream parity."""
import importlib.util
import os
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

BASE = (Path(__file__).parent.parent / 'gui'
        if (Path(__file__).parent.parent / 'gui').exists()
        else Path(__file__).parent)
POINT = np.dtype([('x','<u2'), ('y','<u2'), ('z','<u2'),
                  ('classification','u1')])


def _modules(monkeypatch, info):
    gui = types.ModuleType('gui')
    gui.__path__ = [str(BASE)]
    directory = types.ModuleType('gui.native_lnt_directory')
    directory.LNTDirectoryError = type('LNTDirectoryError', (Exception,), {})
    directory.POINT = POINT
    directory._exactly = lambda handle, n, label: handle.read(n)
    directory.inspect_directory = lambda path, max_points: info
    profile = types.ModuleType('gui.ground_ptd_profiler')
    profile.enabled = lambda: False
    profile.log = lambda msg: None
    algorithm = types.ModuleType('gui.lidar_classification_tools')
    algorithm.called = []

    def classify_ground_ptd(*, xyz, classification, fence_mask, progress_cb,
                            abort_check, from_classes, current_ground,
                            to_class, **kwargs):
        algorithm.called.append(np.asarray(xyz).copy())
        mask = np.isin(classification, from_classes)
        if fence_mask is not None:
            mask &= fence_mask
        classification[mask] = to_class
        return {'stop_reason': 'stable'}

    algorithm.classify_ground_ptd = classify_ground_ptd
    for name, module in [('gui', gui), ('gui.native_lnt_directory', directory),
                         ('gui.ground_ptd_profiler', profile),
                         ('gui.lidar_classification_tools', algorithm)]:
        monkeypatch.setitem(sys.modules, name, module)
    return algorithm


def _cloud(tmp_path, points=320, seed=21):
    rng = np.random.default_rng(seed)
    rec = np.zeros(points, POINT)
    rec['x'] = rng.integers(12000, 53000, points, dtype=np.uint16)
    rec['y'] = rng.integers(12000, 53000, points, dtype=np.uint16)
    rec['z'] = rng.integers(12000, 53000, points, dtype=np.uint16)
    rec['classification'] = rng.choice([0, 1, 14], size=points, p=[.72,.20,.08])
    path = tmp_path / 'cloud.lnt'
    path.write_bytes(rec.tobytes())
    s = path.stat()
    info = {
        'origin': np.zeros(3, np.float64),
        'leaves': [(19, 0, points, np.zeros(3), 80.)],
        'header_bounds': np.array([-80.,-80.,-80.,80.,80.,80.]),
        'total': points, 'leaf_sum': points,
        'size': s.st_size, 'mtime_ns': s.st_mtime_ns,
    }
    return path, info, rec


def _load_density():
    spec = importlib.util.spec_from_file_location('density_preflight_test',
                            BASE/'ground_ptd_density_preflight.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load_stream(monkeypatch, info):
    algorithm = _modules(monkeypatch, info)
    spec = importlib.util.spec_from_file_location('gui.native_ground_below_stream',
                            BASE/'native_ground_below_stream.py')
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'gui.native_ground_below_stream', mod)
    spec.loader.exec_module(mod)
    return mod, algorithm


def test_density_index_bounds_and_exact_membership(tmp_path):
    path, info, rec = _cloud(tmp_path, points=1700)
    m = _load_density()
    index = m.DensityIndex(path, info, [0, 1], POINT,
                           lambda f,n,label: f.read(n), bin_size=4.)
    scale = 160 / 65535.
    x = -80 + scale * rec['x'].astype(np.float64)
    y = -80 + scale * rec['y'].astype(np.float64)
    valid = np.isin(rec['classification'], [0, 1])
    assert index.total_relevant == int(valid.sum())
    rng = np.random.default_rng(110)
    for i in range(175):
        a = rng.uniform(-90, 80, size=2)
        b = a + rng.uniform(.1, 70, size=2)
        bounds = (a[0],a[1],b[0],b[1])
        actual = int(np.count_nonzero(valid & (x>=a[0]) & (x<b[0]) &
                                               (y>=a[1]) & (y<b[1])))
        lo, hi = index.count_bounds(bounds)
        assert lo <= actual <= hi, (bounds, lo, actual, hi)
        for capacity in (0, 1, 20, 70, 200):
            assert index.fits(bounds, capacity) == (actual <= capacity)
    # Guaranteed zero coverage, without a false positive.
    assert index.count_bounds((200, 200, 220, 220)) == (0, 0)


def test_preflight_dense_parent_keeps_complete_original_halo(tmp_path):
    path, info, _ = _cloud(tmp_path, points=75)
    m = _load_density()
    ix = m.DensityIndex(path, info, [0, 1], POINT,
                        lambda f,n,label: f.read(n))
    core = (-90.,-90.,90.,90.)
    planned, stats = m.plan_windows(
        [core], ix, halo=90, base_limit=10, dense_limit=75,
        adaptive=True, prefer_dense=True)
    assert planned == [[(core,75)]]
    assert stats['dense_cells'] == 1
    assert stats['splits'] == 0


def test_preflight_rejects_impossible_halo_before_classifying(tmp_path, monkeypatch):
    path, info, rec = _cloud(tmp_path, points=140)
    mod, algorithm = _load_stream(monkeypatch, info)
    monkeypatch.setenv('NAKSHA_GROUND_PREFLIGHT', '1')
    monkeypatch.setenv('NAKSHA_GROUND_DENSE_ADAPTIVE', '0')
    monkeypatch.setenv('NAKSHA_GROUND_WORKERS', '2')
    before = path.read_bytes()
    # Tiny center region with halo wider than whole occupied cloud.
    with pytest.raises(mod.ScopedPlanningError, match='density preflight|Ground context'):
        mod.plan_streamed_ground(
            path, None, {'from_classes': [0], 'current_ground':[1],
                         'to_class':1}, expected_total=140, max_context=3,
            tile_size=250.)
    assert not algorithm.called
    assert path.read_bytes() == before


@pytest.mark.parametrize('workers', [1, 2])
def test_full_ground_plan_write_ids_and_classes_match_baseline(tmp_path, monkeypatch, workers):
    path, info, rec = _cloud(tmp_path, points=220)
    mod, algo = _load_stream(monkeypatch, info)
    before = path.read_bytes()
    params={'from_classes': [0], 'current_ground':[1], 'to_class':1}
    # First produce a reference classification with a sufficiently large
    # context and no preflight; compare to the adaptive full-context plan.
    monkeypatch.setenv('NAKSHA_GROUND_PREFLIGHT', '0')
    monkeypatch.setenv('NAKSHA_GROUND_WORKERS', '1')
    reference, _ = mod.plan_streamed_ground(
        path, None, params, expected_total=220, tile_size=220.,
        max_context=500)
    monkeypatch.setenv('NAKSHA_GROUND_PREFLIGHT', '1')
    monkeypatch.setenv('NAKSHA_GROUND_DENSE_ADAPTIVE', '1')
    monkeypatch.setenv('NAKSHA_GROUND_DENSE_MAX_CONTEXT', '500')
    monkeypatch.setenv('NAKSHA_GROUND_DENSE_PREFER_LARGE', '1')
    monkeypatch.setenv('NAKSHA_GROUND_WORKERS', str(workers))
    planned, _ = mod.plan_streamed_ground(
        path, None, params, expected_total=220, tile_size=220.,
        max_context=100)
    for k in ('node','local','old_classes','new_classes'):
        np.testing.assert_array_equal(planned[k], reference[k])
    assert planned['stats']['density_preflight']['dense_cells'] > 0
    assert path.read_bytes() == before
    assert all(np.all(np.isfinite(x)) for x in algo.called)


def test_memory_lease_prevents_parallel_large_context(monkeypatch):
    mod, _ = _load_stream(monkeypatch, {'leaves':[]})
    slots = mod._ContextSlots(2)
    normal_done = threading.Event()
    oversized_entered = threading.Event()

    def small():
        count = slots.acquire(False)
        try:
            assert not oversized_entered.is_set()
            time.sleep(.09)
        finally:
            slots.release(count)
            normal_done.set()

    def big():
        count = slots.acquire(True)
        oversized_entered.set()
        try:
            assert normal_done.is_set()
        finally:
            slots.release(count)

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(small)
        time.sleep(.01)
        b = pool.submit(big)
        a.result()
        b.result()


def test_mem_budget_fails_closed(monkeypatch):
    m = _load_density()
    import psutil
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: types.SimpleNamespace(
        available=4*1024**3))
    cap, available = m.ram_budgeted_extended_limit(2_800_000, 2)
    assert cap == 2_800_000
    assert available == 4*1024**3


def test_preflight_skips_context_with_no_source_candidates(tmp_path):
    path, info, rec = _cloud(tmp_path, points=170)
    m = _load_density()
    idx = m.DensityIndex(path, info, [0,1,14], POINT,
                         lambda f,n,label: f.read(n),
                         eligible_classes=[99])
    regions = [(-60,-60,60,60), (-25,-25,25,25)]
    plan, stats = m.plan_windows(regions, idx, halo=90, base_limit=4,
                                 dense_limit=400, adaptive=True)
    assert plan == [[],[]]
    assert stats['skipped_ineligible_cells'] == 2
    assert stats['dense_cells'] == 0


def test_density_preflight_cancellation_before_ptd(tmp_path, monkeypatch):
    path, info, rec = _cloud(tmp_path, points=450)
    mod, algo = _load_stream(monkeypatch, info)
    monkeypatch.setenv('NAKSHA_GROUND_WORKERS', '2')
    monkeypatch.setenv('NAKSHA_GROUND_PREFLIGHT', '1')
    with pytest.raises(mod.ScopedPlanningError, match='cancelled|Canceled'):
        mod.plan_streamed_ground(path, None,
                  {'from_classes': [0], 'current_ground':[1], 'to_class':1},
                  max_context=500, cancelled=lambda: True)
    assert not algo.called


@pytest.mark.parametrize('workers', [1, 2])
def test_preflight_keeps_original_multi_tile_changes_when_contexts_fit(
        tmp_path, monkeypatch, workers):
    path, info, rec = _cloud(tmp_path, points=330, seed=742)
    mod, _ = _load_stream(monkeypatch, info)
    params={'from_classes': [0], 'current_ground': [1], 'to_class':1}
    monkeypatch.setenv('NAKSHA_GROUND_WORKERS', str(workers))
    monkeypatch.setenv('NAKSHA_GROUND_PREFLIGHT', '0')
    before, _ = mod.plan_streamed_ground(
        path, [], params, tile_size=80., max_context=500)
    monkeypatch.setenv('NAKSHA_GROUND_PREFLIGHT', '1')
    monkeypatch.setenv('NAKSHA_GROUND_DENSE_ADAPTIVE', '1')
    monkeypatch.setenv('NAKSHA_GROUND_DENSE_PREFER_LARGE', '0')
    monkeypatch.setenv('NAKSHA_GROUND_DENSE_MAX_CONTEXT', '800')
    after, _ = mod.plan_streamed_ground(
        path, [], params, tile_size=80., max_context=500)
    for key in ('node', 'local', 'old_classes', 'new_classes'):
        np.testing.assert_array_equal(before[key], after[key])
    assert before['stats']['planned_changes'] == after['stats']['planned_changes']
    assert after['stats']['density_preflight']['dense_cells'] == 0


def test_large_world_xy_offsets_conservative_exact_density_count(tmp_path):
    path, info, rec = _cloud(tmp_path, points=600, seed=125)
    info['origin']=np.array([256000.,4837000.,0.])
    m=_load_density()
    idx=m.DensityIndex(path,info,[0,1],POINT,
                       lambda f,n,label:f.read(n), bin_size=4.)
    scale=160./65535.
    x=info['origin'][0] -80.+scale*rec['x'].astype(np.float64)
    y=info['origin'][1] -80.+scale*rec['y'].astype(np.float64)
    selected=np.isin(rec['classification'],[0,1])
    rng=np.random.default_rng(712)
    for _ in range(100):
        ax=256000.+rng.uniform(-80,70)
        ay=4837000.+rng.uniform(-80,70)
        bx=ax+rng.uniform(.001,80)
        by=ay+rng.uniform(.001,80)
        lo,hi=idx.count_bounds((ax,ay,bx,by))
        actual=np.count_nonzero(selected & (x>=ax) & (x<bx) &
                                           (y>=ay) & (y<by))
        assert lo <= actual <= hi
