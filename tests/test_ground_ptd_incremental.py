"""Exact-parity tests for the incremental native Ground PTD engine.

The reference is the untouched original ``classify_ground_ptd`` loaded from the
pre-change backup (when present) and the current code with every new
acceleration switched off. Parity means identical classification arrays,
identical result dictionaries, identical progress messages and identical
per-iteration TIN anchor sets -- not matching counts.
"""
import ast
import hashlib
import os
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
import pytest

from gui import ground_ptd_incremental as inc
from gui import lidar_classification_tools as tools
from gui import native_ground_below_stream as stream
from gui.native_lnt_directory import HEADER, DIR, ENTRY, POINT

BACKUP = Path(r'E:\NAKSHA_LIDAR_10_23_26\Naksha_Ground_Incremental_Backup_20261024'
              r'\gui\lidar_classification_tools.py')
OFF = dict(NAKSHA_GROUND_INCREMENTAL='0', NAKSHA_GROUND_TIN_REUSE='0',
           NAKSHA_GROUND_BLAS_SINGLE='0')
ON = dict(NAKSHA_GROUND_INCREMENTAL='1', NAKSHA_GROUND_TIN_REUSE='1',
          NAKSHA_GROUND_BLAS_SINGLE='1')

def _validation_ready():
    # Validation explicitly opts in without changing the application's default.
    from unittest.mock import patch
    with patch.dict(os.environ, NAKSHA_GROUND_INCREMENTAL='1'):
        return inc.available()


pytestmark = pytest.mark.skipif(not _validation_ready(), reason='native validation DLL unavailable')


@pytest.fixture(autouse=True)
def _native_test_environment(monkeypatch):
    monkeypatch.setenv('NAKSHA_GROUND_INCREMENTAL', '1')
    for name in ('NAKSHA_GROUND_ENGINE_V2', 'NAKSHA_GROUND_GPU',
                 'NAKSHA_GROUND_VULKAN_COMPUTE', 'NAKSHA_GROUND_VK_TIN',
                 'NAKSHA_GROUND_VK_LOCATOR'):
        monkeypatch.setenv(name, '0')


def _baseline():
    if not BACKUP.is_file():
        return None
    source = BACKUP.read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'classify_ground_ptd')
    namespace = dict(tools.__dict__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(BACKUP), 'exec'), namespace)
    return namespace['classify_ground_ptd']


BASELINE = _baseline()


def test_incremental_backend_is_explicit_opt_in(monkeypatch):
    monkeypatch.delenv('NAKSHA_GROUND_INCREMENTAL', raising=False)
    assert not inc.enabled()
    assert inc.create_engine(np.zeros((3, 3))) is None


def test_stale_certificate_dll_is_rejected(monkeypatch):
    from types import SimpleNamespace
    class Version:
        def __call__(self):
            return 1
    monkeypatch.setattr(inc, '_LIB', None)
    monkeypatch.setattr(inc, '_LIB_ERROR', None)
    monkeypatch.setattr(inc.ctypes, 'CDLL', lambda *a: SimpleNamespace(nk_ptd_api_version=Version()))
    assert inc._load() is None
    assert 'ABI mismatch' in inc._LIB_ERROR


def _captured():
    path = Path(__file__).resolve().parents[1]/'audit/ground_incremental_resume/mismatch1.npz'
    if not path.exists():
        pytest.skip('real mismatch capture unavailable on this machine')
    return np.load(path, allow_pickle=False)


def test_recorded_real_mismatch_raw_ordered_ids_and_cycle_limit():
    from types import SimpleNamespace
    with _captured() as data:
        tri = SimpleNamespace(neighbors=data['neighbors'], equations=data['equations'],
            transform=data['transform'], min_bound=data['low'], max_bound=data['high'],
            paraboloid_scale=float(data['scale']), paraboloid_shift=float(data['shift']))
        actual, trace = inc.trace_ordered_walk(tri, data['pts'][data['queries'], :2])
        np.testing.assert_array_equal(actual, data['ref'])
        bad = np.flatnonzero(data['eng'] != data['ref'])
        assert len(bad) == 1
        assert int(data['queries'][bad[0]]) == 48958
        assert (int(data['eng'][bad[0]]), int(actual[bad[0]])) == (26777, 48444)
        assert trace[:, 3].max() <= 1 + len(data['neighbors']) // 4
        assert trace[:, 4].sum() > 0  # exercises the missing Python fallback


def test_real_capture_engine_raw_and_audited_results_match(monkeypatch):
    from scipy.spatial import Delaunay
    monkeypatch.setenv('NAKSHA_GROUND_INCREMENTAL', '1')
    with _captured() as data:
        tri = Delaunay(data['points'])
        np.testing.assert_array_equal(tri.simplices, data['simplices'])
        kwargs = dict(terrain_rad=float(np.radians(88.)), iteration_angle=6.,
            iteration_distance=1.4, reduce_angle_edge=True, edge_length_threshold=5.,
            stop_triangulation=False, stop_edge_length=2., use_distance_as_rating=False,
            distance_weight=50., add_only_upward=False, one_per_simplex=True)
        engines = [inc.create_engine(data['pts'], **kwargs) for _ in range(2)]
        assert all(engine is not None for engine in engines)
        engines[0].audit = False
        engines[1].audit = True
        inc.AUDIT_FAILURES.clear()
        try:
            for mode in (0, 1):
                results = [engine.run(tri, data['tin_ids'], data['queries'], mode)
                           for engine in engines]
                assert all(result is not None for result in results)
                assert results[0][0] == results[1][0]
                np.testing.assert_array_equal(results[0][1], results[1][1])
                assert results[0][2] == results[1][2]
            assert not inc.AUDIT_FAILURES
            assert engines[1].stats['audit_checks'] == 2
            for engine in engines:
                assert engine.stats['cache_cert'] == engine.stats['grid_cert'] == 0
                assert engine.stats['eval_reused'] > 0
                assert engine.stats['persistent_tin_updates'] == 0
        finally:
            for engine in engines:
                engine.close()


def terrain(n=5000, seed=582, grid=False, duplicates=False):
    rng = np.random.default_rng(seed)
    if grid:
        side = int(np.sqrt(n))
        gx, gy = np.meshgrid(np.arange(side) * 0.5, np.arange(side) * 0.5)
        xy = np.c_[gx.ravel(), gy.ravel()][:n]
    else:
        xy = rng.uniform(0, 180, (n, 2))
    z = .025 * xy[:, 0] + .9 * np.sin(xy[:, 1] / 18)
    z += rng.choice([0., .02, .25, 3., 14.], len(xy), p=[.3, .3, .2, .15, .05])
    xyz = np.column_stack((xy + [256000., 4837000.], z + 85.))
    if duplicates:
        k = len(xyz) // 20
        xyz[-k:, :2] = xyz[:k, :2]           # identical XY, different Z
        xyz[-k:, 2] += rng.uniform(-.3, 4., k)
    cls = np.zeros(len(xyz), dtype=np.uint8)
    cls[::47] = 1
    cls[::91] = 5
    return xyz, cls


def params(**extra):
    base = dict(from_classes=[0], to_class=1, current_ground=[1],
                max_building_size=25., terrascan_compatible_densification=True)
    base.update(extra)
    return base


def run(monkeypatch, env, xyz, cls, kwargs, fence=None, function=None, audit=False):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('NAKSHA_GROUND_INC_AUDIT', '1' if audit else '0')
    monkeypatch.setenv('NAKSHA_GROUND_ENGINE_V2', '0')
    monkeypatch.setenv('NAKSHA_GROUND_GPU', '0')
    values, messages, anchors = cls.copy(), [], []
    sampler = tools._subsample_ground_tin_points

    def sample(points, ids, *a, **kw):
        anchors.append(np.array(ids, copy=True))
        return sampler(points, ids, *a, **kw)
    func = function or tools.classify_ground_ptd
    if function is not None:
        func.__globals__['_subsample_ground_tin_points'] = sample
    try:
        # Restore after EACH run: stacking wrappers records the reference run
        # into the earlier accelerated trace and creates false anchor failures.
        with monkeypatch.context() as recording:
            recording.setattr(tools, '_subsample_ground_tin_points', sample)
            result = func(xyz, values, fence_mask=fence,
                          progress_cb=lambda pct, msg: messages.append((pct, msg)), **kwargs)
    finally:
        if function is not None:
            func.__globals__['_subsample_ground_tin_points'] = sampler
    return values, result, messages, anchors


def assert_same(a, b):
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1].keys() == b[1].keys()
    for key in a[1]:
        np.testing.assert_equal(a[1][key], b[1][key], err_msg=key)
    assert a[2] == b[2]
    assert len(a[3]) == len(b[3])
    for left, right in zip(a[3], b[3]):
        np.testing.assert_array_equal(left, right)


OPTIONS = [
    {}, {'seed_method': 'lowest_only'}, {'seed_method': 'ground_only'},
    {'add_only_upward': True}, {'use_distance_as_rating': True},
    {'use_distance_as_rating': True, 'distance_weight': 100.0},
    {'class0_holdout_percent': 7.}, {'adaptive_thresholds': True},
    {'stop_triangulation': True}, {'iteration_distance': 0.},
    {'reduce_angle_edge': False}, {'complete_ground_from_final_tin': False},
    {'terrascan_compatible_densification': False},
    {'terrain_angle': 20.0, 'iteration_angle': 10.0, 'iteration_distance': 2.5},
]


@pytest.mark.parametrize('options', OPTIONS)
@pytest.mark.parametrize('fenced', [False, True])
def test_full_ptd_parity_with_original(monkeypatch, options, fenced):
    xyz, original = terrain()
    kwargs = params(**options)
    fence = ((xyz[:, 0] < 256135.) & (xyz[:, 1] > 4837025.)) if fenced else None
    inc.AUDIT_FAILURES.clear()
    fast = run(monkeypatch, ON, xyz, original, kwargs, fence, audit=True)
    assert not inc.AUDIT_FAILURES
    reference = run(monkeypatch, OFF, xyz, original, kwargs, fence)
    assert_same(fast, reference)
    if BASELINE is not None:
        assert_same(fast, run(monkeypatch, OFF, xyz, original, kwargs, fence, function=BASELINE))
    if fenced:
        np.testing.assert_array_equal(fast[0][~fence], original[~fence])


@pytest.mark.parametrize('geometry', ['grid', 'duplicates', 'dense'])
def test_degenerate_geometry_parity(monkeypatch, geometry):
    xyz, original = terrain(n=6400 if geometry != 'dense' else 20000, seed=7,
                            grid=geometry == 'grid', duplicates=geometry == 'duplicates')
    kwargs = params()
    inc.AUDIT_FAILURES.clear()
    fast = run(monkeypatch, ON, xyz, original, kwargs, audit=True)
    assert not inc.AUDIT_FAILURES
    assert_same(fast, run(monkeypatch, OFF, xyz, original, kwargs))


def test_engine_failure_mid_run_continues_on_original_path(monkeypatch):
    xyz, original = terrain(seed=3)
    kwargs = params()
    reference = run(monkeypatch, OFF, xyz, original, kwargs)
    calls = [0]
    real = inc.IncrementalPTD.run

    def flaky(self, *args, **kw):
        calls[0] += 1
        if calls[0] == 3:
            return None
        return real(self, *args, **kw)
    monkeypatch.setattr(inc.IncrementalPTD, 'run', flaky)
    fast = run(monkeypatch, ON, xyz, original, kwargs)
    assert calls[0] == 3  # disabled for the rest of the call after the refusal
    assert_same(fast, reference)


def test_float32_input_uses_original_path(monkeypatch):
    xyz, original = terrain(seed=4)
    xyz = (xyz - xyz.min(0)).astype(np.float32)
    created = []
    real = inc.create_engine
    monkeypatch.setattr(inc, 'create_engine', lambda *a, **k: created.append(1) or real(*a, **k))
    fast = run(monkeypatch, ON, xyz, original, params())
    assert not created
    assert_same(fast, run(monkeypatch, OFF, xyz, original, params()))


def test_blas_scope_restores_thread_count():
    lib = inc._scipy_openblas()
    if lib is None:
        pytest.skip('SciPy OpenBLAS not located')
    before = lib.scipy_openblas_get_num_threads()
    with inc.blas_single_thread():
        assert lib.scipy_openblas_get_num_threads() == 1
        with inc.blas_single_thread():
            assert lib.scipy_openblas_get_num_threads() == 1
        assert lib.scipy_openblas_get_num_threads() == 1
    assert lib.scipy_openblas_get_num_threads() == before


def test_transform_bits_identical_with_single_blas_thread():
    from scipy.spatial import Delaunay
    import scipy.spatial._qhull as qh
    rng = np.random.default_rng(9)
    xy = rng.uniform(0, 300, (60000, 2)) + [256000.37, 4836500.11]
    tri = Delaunay(xy)
    full = qh._get_barycentric_transforms(tri.points, tri.simplices, np.finfo(float).eps)
    with inc.blas_single_thread():
        single = qh._get_barycentric_transforms(tri.points, tri.simplices, np.finfo(float).eps)
    assert np.array_equal(full.view(np.uint64), single.view(np.uint64))


def write_lnt(path, xyz, classes):
    origin = xyz.min(0) - 1.
    bounds = (*origin, *(xyz.max(0) + 1.))
    half = 128.
    center = np.full(3, 128., dtype=float)
    scale = 2. * half / 65535.
    packed = np.zeros(len(xyz), dtype=POINT)
    for col, key in enumerate(('x', 'y', 'z')):
        packed[key] = np.rint((xyz[:, col] - origin[col]) / scale).astype(np.uint16)
    packed['classification'] = classes
    entries = []
    with path.open('wb') as handle:
        handle.write(bytes(64))
        for node, chunk in enumerate(np.array_split(packed, 4)):
            offset = handle.tell()
            chunk.tofile(handle)
            entries.append(ENTRY.pack(node, offset, chunk.nbytes, len(chunk),
                                      *center, half, 0, 1, bytes(10)))
        index = handle.tell()
        handle.write(DIR.pack(len(packed), len(entries)))
        for entry in entries:
            handle.write(entry)
        handle.seek(0)
        handle.write(HEADER.pack(b'LNT\x00', 2, 0, index, *bounds))


@pytest.mark.parametrize('workers', [1, 2, 4])
@pytest.mark.parametrize('fenced', [False, True])
def test_streamed_plan_parity(tmp_path, monkeypatch, workers, fenced):
    xyz, classes = terrain(3000)
    path = tmp_path / 'terrain.lnt'
    write_lnt(path, xyz, classes)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    fence = [np.array([[256025., 4837025.], [256150., 4837025.],
                       [256150., 4837150.], [256025., 4837150.]])] if fenced else []
    monkeypatch.setenv('NAKSHA_GROUND_WORKERS', str(workers))
    monkeypatch.setenv('NAKSHA_GROUND_GPU', '0')
    monkeypatch.setenv('NAKSHA_GROUND_ENGINE_V2', '0')
    plans = []
    for env in (OFF, ON):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        plan, _ = stream.plan_streamed_ground(path, fence, params(), tile_size=90.)
        plans.append(plan)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    for key in ('node', 'local', 'old_classes', 'new_classes', 'file_stat'):
        np.testing.assert_equal(plans[0][key], plans[1][key], err_msg=key)
    assert len(plans[1]['node']) > 0


def test_plan_result_chunks_match_record_list():
    rng = np.random.default_rng(2)
    ids = rng.choice(2**40, 5000, replace=False).astype(np.uint64)
    old = rng.integers(0, 5, 5000).astype(np.uint8)
    new = rng.integers(0, 5, 5000).astype(np.uint8)
    info = dict(size=1, mtime_ns=2)
    chunks = stream._ChangeChunks()
    for part in np.array_split(np.arange(5000), 7):
        chunks.add(ids[part], old[part], new[part])
    listed = [(int(k), int(a), int(b)) for k, a, b in zip(ids, old, new)]
    left = stream._plan_result(chunks, info, {})
    right = stream._plan_result(listed, info, {})
    for key in ('node', 'local', 'old_classes', 'new_classes'):
        np.testing.assert_array_equal(left[key], right[key])
        assert left[key].dtype == right[key].dtype
    duplicate = stream._ChangeChunks()
    duplicate.add(ids[:3], old[:3], new[:3])
    duplicate.add(ids[:1], old[:1], new[:1])
    with pytest.raises(stream.ScopedPlanningError, match='Duplicate'):
        stream._plan_result(duplicate, info, {})


@pytest.mark.parametrize('workers', [1, 2])
def test_cancel_after_native_work_never_writes_lnt(tmp_path, monkeypatch, workers):
    from threading import Event
    xyz, classes = terrain(5000)
    path = tmp_path / 'cancel.lnt'
    write_lnt(path, xyz, classes)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    cancelled = Event()
    completed = []
    engines = []
    real_run, real_create = inc.IncrementalPTD.run, inc.create_engine
    def create(*a, **kw):
        engine = real_create(*a, **kw)
        if engine is not None:
            engines.append(engine)
        return engine
    def run_then_cancel(self, *a, **kw):
        result = real_run(self, *a, **kw)
        completed.append(result)
        cancelled.set()
        return result
    monkeypatch.setattr(inc, 'create_engine', create)
    monkeypatch.setattr(inc.IncrementalPTD, 'run', run_then_cancel)
    for key, value in ON.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('NAKSHA_GROUND_WORKERS', str(workers))
    with pytest.raises(stream.ScopedPlanningError, match='cancelled'):
        stream.plan_streamed_ground(path, [], params(), tile_size=90.,
                                    cancelled=cancelled.is_set)
    assert completed and all(result is not None for result in completed)
    assert engines and all(not engine._handle for engine in engines)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
