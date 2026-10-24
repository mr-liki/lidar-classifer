"""Exact simplex, full PTD and real packed-leaf streamed differentials."""
import hashlib
import os
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
import pytest
from scipy.spatial import Delaunay

from gui import ground_ptd_engine_v2 as v2
from gui import lidar_classification_tools as tools
from gui import native_ground_below_stream as stream
from gui.native_lnt_directory import HEADER, DIR, ENTRY, POINT, read_exact_leaf_snapshot


def terrain(n=4000, seed=582):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 180, (n, 2))
    z = .025 * xy[:, 0] + .9 * np.sin(xy[:, 1] / 18)
    z += rng.choice([0., .02, .25, 3., 14.], n, p=[.3, .3, .2, .15, .05])
    xyz = np.column_stack((xy + [256000., 4837000.], z + 85.))
    cls = np.zeros(n, dtype=np.uint8)
    cls[::47] = 1
    cls[::91] = 5
    return xyz, cls


def params(**extra):
    return dict(from_classes=[0], to_class=1, current_ground=[1],
                max_building_size=25., terrascan_compatible_densification=True,
                **extra)


@pytest.mark.parametrize('geometry', ['random', 'grid', 'thin', 'collinear', 'duplicate'])
@pytest.mark.parametrize('offset', [(0., 0.), (256000., 4837000.)])
def test_all_simplex_ids_and_query_orders(geometry, offset):
    rng = np.random.default_rng(841)
    vertices = rng.random((120, 2))
    if geometry == 'grid':
        x, y = np.meshgrid(np.arange(10.) / 9, np.arange(10.) / 9)
        vertices = np.c_[x.ravel(), y.ravel()]
    elif geometry == 'thin':
        vertices[:, 1] *= 1e-6
    elif geometry == 'collinear':
        vertices[:, 1] = vertices[:, 0] * .25 + vertices[:, 1] * 1e-7
    elif geometry == 'duplicate':
        vertices = np.r_[vertices, vertices[:15], vertices[:5]]
    vertices += offset
    tri = Delaunay(vertices)
    edges = vertices[tri.simplices[:, :2]].mean(axis=1)
    centers = vertices[tri.simplices].mean(axis=1)
    random = vertices.min(0) + rng.uniform(-.1, 1.1, (1500, 2)) * np.ptp(vertices, axis=0)
    queries = np.r_[vertices, edges, np.nextafter(edges, np.inf),
                    np.nextafter(edges, -np.inf), centers, random]
    for order in (np.arange(len(queries)), rng.permutation(len(queries)),
                  np.arange(len(queries))[::-1]):
        xy = np.ascontiguousarray(queries[order])
        reference = tri.find_simplex(xy)
        observed, complete = v2._walk_2d(xy, tri.neighbors, tri.equations,
                                       tri.transform, tri.min_bound, tri.max_bound,
                                       tri.paraboloid_scale, tri.paraboloid_shift)
        if complete:
            np.testing.assert_array_equal(observed, reference)
        # Degenerate geometry must fall back to the FULL original operation.
        np.testing.assert_array_equal(v2.find_simplex(tri, xy), reference)
        engine = v2.GroundEngineV2()
        cached = engine.triangulate(vertices)
        np.testing.assert_array_equal(engine.locate(cached, xy), reference)
        np.testing.assert_array_equal(engine.locate(cached, xy), reference)
        assert engine.stats['lookup_reuses'] == 1


def test_failed_equivalence_discards_entire_proposal(monkeypatch):
    tri = Delaunay([[0., 0.], [1., 0.], [0., 1.]])
    xy = np.array([[.1, .1], [.5, .5], [3., 2.]])
    reference = tri.find_simplex(xy)
    monkeypatch.setattr(v2, '_walk_2d', lambda *args: (np.zeros(3, np.int32), True))
    stats = {}
    np.testing.assert_array_equal(v2.find_simplex(tri, xy, stats=stats), reference)
    assert stats['fallback'] == 'complete-query equivalence failure'


def test_bitwise_cache_keys_and_mutation_isolation():
    engine = v2.GroundEngineV2()
    xy = np.array([[0., 0.], [1., 0.], [0., 1.], [.9, .9]])
    tri = engine.triangulate(xy)
    assert engine.triangulate(xy.copy()) is tri
    query = np.array([[.2, .2], [.8, .8], [.5, .5]])
    original = engine.locate(tri, query)
    original[:] = -123
    np.testing.assert_array_equal(engine.locate(tri, query), tri.find_simplex(query))
    engine.locate(tri, query[::-1])
    assert engine.stats['lookup_calls'] == 2
    xy[0, 0] = -0.0
    assert engine.triangulate(xy) is not tri  # even signed zero changes the key
    xy[0, 0] = .001
    assert engine.triangulate(xy) is not tri


@pytest.mark.parametrize('n', [1, 37, 10000, 200000])
def test_linear_ranking_matches_all_lexsort_keys(n):
    rng = np.random.default_rng(123)
    point = rng.permutation(n).astype(np.intp)
    sid = rng.integers(0, 50, n, dtype=np.int32)
    distance = rng.integers(0, 5, n).astype(float)
    angle = rng.integers(0, 5, n).astype(float)
    distance[::13] = -0.
    passing = np.flatnonzero(rng.random(n) < .9)
    order = np.lexsort((point[passing], distance[passing], angle[passing], sid[passing]))
    s = sid[passing][order]
    expected = passing[order[np.r_[True, s[1:] != s[:-1]]]] if len(s) else passing
    actual = v2.choose_per_simplex(point, sid, distance, angle, passing, 50)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('options', [
    {}, {'seed_method': 'lowest_only'}, {'seed_method': 'ground_only'},
    {'add_only_upward': True}, {'use_distance_as_rating': True},
    {'class0_holdout_percent': 7.}, {'adaptive_thresholds': True},
    {'stop_triangulation': True}, {'iteration_distance': 0.},
    {'reduce_angle_edge': False}, {'complete_ground_from_final_tin': False},
    {'terrascan_compatible_densification': False},
])
def test_full_ptd_classes_ids_anchors_iterations_and_stop(monkeypatch, options):
    xyz, original = terrain()
    kwargs = params()
    kwargs.update(options)
    fence = (xyz[:, 0] < 256135.) & (xyz[:, 1] > 4837025.)
    outputs, traces, anchors = [], [], []
    sampler = tools._subsample_ground_tin_points
    for enabled in ('0', '1'):
        monkeypatch.setenv('NAKSHA_GROUND_ENGINE_V2', enabled)
        monkeypatch.setenv('NAKSHA_GROUND_LOCATOR', '0')
        monkeypatch.setenv('NAKSHA_GROUND_GPU', '0')
        values, messages, selected = original.copy(), [], []
        def sample(points, ids, *args, **kw):
            selected.append(ids.copy())
            return sampler(points, ids, *args, **kw)
        monkeypatch.setattr(tools, '_subsample_ground_tin_points', sample)
        result = tools.classify_ground_ptd(xyz, values, fence_mask=fence,
                    progress_cb=lambda pct, msg: messages.append((pct, msg)), **kwargs)
        outputs.append((values, result)); traces.append(messages); anchors.append(selected)
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1]['indices'], outputs[1][1]['indices'])
    for key in outputs[0][1]:
        np.testing.assert_equal(outputs[0][1][key], outputs[1][1][key], err_msg=key)
    assert traces[0] == traces[1]
    assert len(anchors[0]) == len(anchors[1])
    for left, right in zip(*anchors):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(outputs[1][0][~fence], original[~fence])
    assert outputs[1][1]['iterations'] > 1 or kwargs.get('iteration_distance') == 0.


@pytest.mark.parametrize('failure', ['triangulate', 'locate'])
def test_runtime_failure_uses_original_before_write(monkeypatch, failure):
    xyz, original = terrain(1500)
    monkeypatch.setenv('NAKSHA_GROUND_GPU', '0')
    monkeypatch.setenv('NAKSHA_GROUND_ENGINE_V2', '0')
    expected = original.copy()
    tools.classify_ground_ptd(xyz, expected, **params())
    monkeypatch.setenv('NAKSHA_GROUND_ENGINE_V2', '1')
    def fail(*args):
        raise RuntimeError('injected optional-engine failure')
    monkeypatch.setattr(v2.GroundEngineV2, failure, fail)
    actual = original.copy()
    tools.classify_ground_ptd(xyz, actual, **params())
    np.testing.assert_array_equal(actual, expected)


def write_lnt(path, xyz, classes):
    """Disposable valid packed-leaf fixture, preserving node/local identity."""
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


@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('fenced', [False, True])
def test_streamed_plans_exact_id_class_and_source_integrity(tmp_path, monkeypatch, workers, fenced):
    xyz, classes = terrain(2200)
    path = tmp_path / 'terrain.lnt'
    write_lnt(path, xyz, classes)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    fence = [np.array([[256025., 4837025.], [256150., 4837025.],
                       [256150., 4837150.], [256025., 4837150.]])] if fenced else []
    monkeypatch.setenv('NAKSHA_GROUND_WORKERS', str(workers))
    monkeypatch.setenv('NAKSHA_GROUND_GPU', '0')
    plans = []
    for enabled in ('0', '1'):
        monkeypatch.setenv('NAKSHA_GROUND_ENGINE_V2', enabled)
        plan, _ = stream.plan_streamed_ground(path, fence, params(), tile_size=90.)
        plans.append(plan)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    for key in ('node', 'local', 'old_classes', 'new_classes', 'file_stat'):
        np.testing.assert_equal(plans[0][key], plans[1][key], err_msg=key)
    ids = stream._identity(plans[1]['node'], plans[1]['local'])
    assert len(ids) > 0 and len(np.unique(ids)) == len(ids)
    snapshot, _ = read_exact_leaf_snapshot(path)
    all_ids = stream._identity(snapshot['node'], snapshot['local'])
    changed = np.isin(all_ids, ids)
    if fenced:
        assert np.all(stream._inside(snapshot['xyz'][changed, :2], stream._prepare_area(fence)))


@pytest.mark.parametrize('workers', [1, 2])
def test_streamed_cancellation_and_revision_rejection(tmp_path, monkeypatch, workers):
    path = tmp_path / 'terrain.lnt'
    write_lnt(path, *terrain(800))
    monkeypatch.setenv('NAKSHA_GROUND_WORKERS', str(workers))
    monkeypatch.setenv('NAKSHA_GROUND_ENGINE_V2', '1')
    before = path.read_bytes()
    with pytest.raises(stream.ScopedPlanningError, match='cancelled'):
        stream.plan_streamed_ground(path, [], params(), cancelled=lambda: True)
    assert path.read_bytes() == before
    info = stream.inspect_directory(path)
    os.utime(path, ns=(info['mtime_ns'], info['mtime_ns'] + 10000000))
    with pytest.raises(stream.ScopedPlanningError, match='changed during analysis'):
        stream._check_file(path, info)


def test_legacy_flag_cannot_reactivate_subset_locator(monkeypatch):
    from gui.ground_ptd_locator import find_simplex_exact
    calls = []
    class Tri:
        def find_simplex(self, xy):
            calls.append(xy)
            return np.array([0, -1, 1], dtype=np.int32)
    query = np.array([[0., 0.], [2., 2.], [1., 1.]])
    monkeypatch.setenv('NAKSHA_GROUND_LOCATOR', '1')
    np.testing.assert_array_equal(find_simplex_exact(Tri(), query), [0, -1, 1])
    assert len(calls) == 1 and calls[0] is query


def test_project_class_mapping_is_not_las_class_two():
    app = SimpleNamespace(class_palette={0: {'name': 'Default'},
                                        1: {'name': 'Ground'},
                                        2: {'name': 'Low vegetation'}})
    assert tools._resolve_ground_class_defaults(app) == (0, 1)


@pytest.mark.parametrize('delta', [-1, 0, 1])
@pytest.mark.parametrize('rating', [False, True])
def test_floating_distance_thresholds_duplicate_xy_and_edges(monkeypatch, delta, rating):
    distance = .2
    z = np.nextafter(distance, -np.inf if delta < 0 else np.inf) if delta else distance
    xyz = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [2., 2., 0.],
                    [.25, .25, z], [.25, .25, distance], [1., 1., z],
                    [0., 1., z], [3., 1., z]], dtype=np.float64)
    xyz[:, :2] += [256000., 4837000.]
    original = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
    outcomes = []
    for enabled in ('0', '1'):
        monkeypatch.setenv('NAKSHA_GROUND_ENGINE_V2', enabled)
        monkeypatch.setenv('NAKSHA_GROUND_GPU', '0')
        values = original.copy()
        result = tools.classify_ground_ptd(xyz, values, **params(
            seed_method='ground_only', iteration_distance=distance,
            iteration_angle=90., reduce_angle_edge=False,
            use_distance_as_rating=rating, distance_weight=100.))
        outcomes.append((values, result))
    np.testing.assert_array_equal(outcomes[0][0], outcomes[1][0])
    for key in outcomes[0][1]:
        np.testing.assert_equal(outcomes[0][1][key], outcomes[1][1][key])


@pytest.mark.parametrize('workers', [1, 2])
def test_mid_planning_cancel_keeps_source_bytes(tmp_path, monkeypatch, workers):
    path = tmp_path / 'cancel.lnt'
    write_lnt(path, *terrain(1800))
    before = path.read_bytes()
    monkeypatch.setenv('NAKSHA_GROUND_ENGINE_V2', '1')
    monkeypatch.setenv('NAKSHA_GROUND_WORKERS', str(workers))
    stopped = [False]
    def progress(pct, msg):
        stopped[0] = True
    with pytest.raises(stream.ScopedPlanningError, match='cancelled'):
        stream.plan_streamed_ground(path, [], params(), tile_size=90.,
                                    cancelled=lambda: stopped[0], progress=progress)
    assert path.read_bytes() == before


def test_unsupported_runtime_full_query_and_failed_audit_latches_off(monkeypatch):
    tri = Delaunay([[0., 0.], [1., 0.], [0., 1.]])
    xy = np.array([[.1, .1], [.5, .5], [3., 2.]])
    reference = tri.find_simplex(xy)
    with monkeypatch.context() as context:
        context.setattr(v2.scipy, '__version__', 'unsupported')
        np.testing.assert_array_equal(v2.find_simplex(tri, xy, verify=False), reference)
    engine = v2.GroundEngineV2()
    engine.audit = True
    cached = engine.triangulate(tri.points)
    monkeypatch.setattr(v2, '_walk_2d', lambda *args: (np.zeros(3, np.int32), True))
    np.testing.assert_array_equal(engine.locate(cached, xy), reference)
    assert engine._compiled_disabled
    # A different order must execute the original complete SciPy resolver.
    np.testing.assert_array_equal(engine.locate(cached, xy[::-1]), cached.find_simplex(xy[::-1]))
