"""Opt-in stateful C++ evaluation with reference-rebuilt Ground TINs.

SciPy/Qhull still builds every TIN exactly as before. The native engine only
replaces the per-iteration work after each Delaunay build -- the SciPy
``find_simplex`` lookup, the slope/distance/iteration-angle tests and the
one-best-candidate-per-triangle ranking -- with stateful native code:

* every query goes through the ordered SciPy walker with its carried state;
  cached/grid containment certificates are disabled after a real mismatch;
* candidate tests replicate NumPy's operation order bit-for-bit, and the
  replica is verified against NumPy on this machine before first use;
* cached test results are reused only while the candidate's triangle has the
  identical ordered vertex triple.

Any unsupported runtime, failed self-test, incomplete walk or native error
returns ``None`` so the caller continues with the original NumPy/SciPy code
from the same iteration. ``NAKSHA_GROUND_INCREMENTAL=0`` disables the engine;
``NAKSHA_GROUND_INC_AUDIT=1`` compares every lookup, test and selection with
SciPy/NumPy at run time (validation only; slow).
"""
from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np

_TRUE = ('1', 'true', 'yes', 'on')
_C_DBL = ctypes.POINTER(ctypes.c_double)
_C_I32 = ctypes.POINTER(ctypes.c_int32)
_C_I64 = ctypes.POINTER(ctypes.c_int64)
_C_U8 = ctypes.POINTER(ctypes.c_uint8)


class _Params(ctypes.Structure):
    _fields_ = [
        ('terrain_rad', ctypes.c_double),
        ('iteration_angle', ctypes.c_double),
        ('iteration_distance', ctypes.c_double),
        ('edge_length_threshold', ctypes.c_double),
        ('stop_edge_length', ctypes.c_double),
        ('rating_w', ctypes.c_double),
        ('rating_one_minus_w', ctypes.c_double),
        ('reduce_angle_edge', ctypes.c_int32),
        ('stop_triangulation', ctypes.c_int32),
        ('use_distance_as_rating', ctypes.c_int32),
        ('add_only_upward', ctypes.c_int32),
        ('one_per_simplex', ctypes.c_int32),
        ('reserved', ctypes.c_int32),
    ]


def enabled():
    # This experimental backend is never enabled by merely importing Ground.
    return os.environ.get('NAKSHA_GROUND_INCREMENTAL', '0').strip().lower() in _TRUE


def audit_enabled():
    return os.environ.get('NAKSHA_GROUND_INC_AUDIT', '0').strip().lower() in _TRUE


# ---------------------------------------------------------------------------
# SciPy's OpenBLAS: 2x2 LAPACK calls behind Delaunay.transform cost ~17 us each
# with the default thread pool and ~1 us single-threaded, with bit-identical
# results. Scope a single thread to Ground work only, refcounted across workers.
_BLAS_LOCK = Lock()
_BLAS_STATE = {'depth': 0, 'saved': None, 'lib': None, 'probed': False}


def _scipy_openblas():
    if _BLAS_STATE['probed']:
        return _BLAS_STATE['lib']
    _BLAS_STATE['probed'] = True
    try:
        import scipy
        base = Path(scipy.__file__).resolve().parent
        for folder in (base.parent / 'scipy.libs', base / '.libs', base.parent):
            for path in glob.glob(str(folder / '*scipy_openblas*.dll')):
                lib = ctypes.CDLL(path)
                if hasattr(lib, 'scipy_openblas_set_num_threads') and hasattr(
                        lib, 'scipy_openblas_get_num_threads'):
                    lib.scipy_openblas_set_num_threads.argtypes = [ctypes.c_int]
                    lib.scipy_openblas_set_num_threads.restype = None
                    lib.scipy_openblas_get_num_threads.argtypes = []
                    lib.scipy_openblas_get_num_threads.restype = ctypes.c_int
                    _BLAS_STATE['lib'] = lib
                    return lib
    except Exception:
        pass
    return None


class blas_single_thread:
    """Context manager; no-op when SciPy's OpenBLAS cannot be located."""

    def __enter__(self):
        if os.environ.get('NAKSHA_GROUND_BLAS_SINGLE', '1').strip().lower() not in _TRUE:
            return self
        with _BLAS_LOCK:
            lib = _scipy_openblas()
            if lib is None:
                return self
            if _BLAS_STATE['depth'] == 0:
                _BLAS_STATE['saved'] = int(lib.scipy_openblas_get_num_threads())
                lib.scipy_openblas_set_num_threads(1)
            _BLAS_STATE['depth'] += 1
            self._entered = True
        return self

    def __exit__(self, *exc):
        if getattr(self, '_entered', False):
            with _BLAS_LOCK:
                _BLAS_STATE['depth'] -= 1
                if _BLAS_STATE['depth'] == 0 and _BLAS_STATE['saved'] is not None:
                    _BLAS_STATE['lib'].scipy_openblas_set_num_threads(_BLAS_STATE['saved'])
                    _BLAS_STATE['saved'] = None
        return False


# ---------------------------------------------------------------------------
def _library_path():
    env = os.environ.get('NAKSHA_GROUND_INC_LIBRARY', '').strip()
    if env:
        return Path(env).expanduser().resolve()
    base = Path(__file__).resolve().parents[1] / 'native' / 'ground_ptd_incremental'
    for folder in ('build_verified', 'build'):
        for name in ('Release/naksha_ground_ptd_incremental.dll',
                     'naksha_ground_ptd_incremental.dll',
                     'libnaksha_ground_ptd_incremental.so'):
            if (base / folder / name).is_file():
                return base / folder / name
    return None


_LIB = None
_LIB_ERROR = None
_SELFTEST = None  # None = not run, True = passed, str = failure reason
_STATE_LOCK = Lock()
_WARNED = set()
AUDIT_FAILURES = []  # validation runs inspect this; production never audits


def _warn_once(message):
    if message not in _WARNED:
        _WARNED.add(message)
        print('[Ground incremental] ' + message, flush=True)


def _load():
    global _LIB, _LIB_ERROR
    if _LIB is not None or _LIB_ERROR is not None:
        return _LIB
    try:
        path = _library_path()
        if path is None or not path.is_file():
            raise FileNotFoundError('native incremental Ground engine is not built')
        lib = ctypes.CDLL(str(path))
        lib.nk_ptd_api_version.restype = ctypes.c_int
        if lib.nk_ptd_api_version() != 2:
            raise RuntimeError('incremental Ground engine ABI mismatch')
        lib.nk_ptd_create.argtypes = [_C_DBL, ctypes.c_int64, ctypes.POINTER(_Params)]
        lib.nk_ptd_create.restype = ctypes.c_void_p
        lib.nk_ptd_destroy.argtypes = [ctypes.c_void_p]
        lib.nk_ptd_destroy.restype = None
        lib.nk_ptd_iterate.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            _C_I64, ctypes.c_int64,
            _C_I32, _C_I32, _C_DBL, _C_DBL, ctypes.c_int64,
            _C_DBL,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double,
            _C_I64, ctypes.c_int64,
            _C_I64, _C_I64,
            _C_I32, _C_U8, _C_DBL, _C_DBL,
            _C_I64, _C_DBL]
        lib.nk_ptd_iterate.restype = ctypes.c_int
        lib.nk_ptd_evaluate_rows.argtypes = [
            _C_DBL, _C_DBL, _C_DBL, _C_DBL, ctypes.c_int64, ctypes.POINTER(_Params),
            _C_U8, _C_DBL, _C_DBL, _C_DBL, _C_DBL, _C_U8]
        lib.nk_ptd_evaluate_rows.restype = ctypes.c_int
        _LIB = lib
    except Exception as exc:
        _LIB_ERROR = f'{type(exc).__name__}: {exc}'
    return _LIB


def make_params(*, terrain_rad, iteration_angle, iteration_distance,
                reduce_angle_edge, edge_length_threshold, stop_triangulation,
                stop_edge_length, use_distance_as_rating, distance_weight,
                add_only_upward, one_per_simplex):
    w = distance_weight / 100.0  # identical Python expression to the reference
    return _Params(float(terrain_rad), float(iteration_angle), float(iteration_distance),
                   float(edge_length_threshold), float(stop_edge_length),
                   float(w), float(1 - w),
                   int(bool(reduce_angle_edge)), int(bool(stop_triangulation)),
                   int(bool(use_distance_as_rating)), int(bool(add_only_upward)),
                   int(bool(one_per_simplex)), 0)


def reference_rows(pa, A, B, C, *, terrain_rad, iteration_angle, iteration_distance,
                   reduce_angle_edge, edge_length_threshold, stop_triangulation,
                   stop_edge_length, use_distance_as_rating, distance_weight,
                   add_only_upward):
    """The reference classify_ground_ptd candidate tests, expression for expression."""
    from gui.ground_ptd_vulkan_tin import cpu_facets
    me, nor, good, _ = cpu_facets(A, B, C)
    sl = np.arccos(np.clip(nor[:, 2], -1, 1))
    tok = good & (sl <= terrain_rad)
    pn = nor; pme = me; pto = tok; pA = A
    act = pto.copy()
    if stop_triangulation:
        act &= pme >= stop_edge_length
    d = np.einsum('ij,ij->i', pn, pa - pA)
    pd = np.abs(d)
    act &= pd <= iteration_distance
    if add_only_upward:
        act &= d >= 0
    pp = pa - d[:, np.newaxis] * pn
    nr = np.minimum(np.minimum(np.linalg.norm(A - pp, axis=1),
                               np.linalg.norm(B - pp, axis=1)),
                    np.linalg.norm(C - pp, axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = np.where(nr < 1e-12, 90.0, np.degrees(np.arctan2(pd, nr)))
        ea = np.full(len(pa), iteration_angle, dtype=float)
        if reduce_angle_edge:
            sm = pme < edge_length_threshold
            ea[sm] = iteration_angle * pme[sm] / edge_length_threshold
        if use_distance_as_rating:
            w = distance_weight / 100.0
            act &= (1 - w) * np.where(ea > 0, alpha / ea, 1.0) + w * (
                pd / iteration_distance if iteration_distance > 0 else np.ones_like(pd)) <= 1.0
        else:
            act &= alpha <= ea
    return act, alpha, pd, me, nor, tok


def _synthetic_rows(rng, count):
    """World-coordinate triangles and candidates, including boundary cases."""
    origin = np.array([256000.0, 4836500.0, 85.0]) + rng.uniform(-500, 500, 3)
    size = rng.choice([0.002, 0.05, 0.5, 2.0, 5.0, 20.0, 80.0], count)[:, None]
    A = origin + rng.normal(size=(count, 3)) * size
    B = A + rng.normal(size=(count, 3)) * size
    C = A + rng.normal(size=(count, 3)) * size
    w = rng.dirichlet((1, 1, 1), count)
    pa = w[:, :1] * A + w[:, 1:2] * B + w[:, 2:] * C
    pa[:, 2] += rng.choice([0.0, 1e-9, 0.01, 0.2, 1.4, 3.0, -0.4], count) * rng.uniform(.5, 1.5, count)
    k = count // 10
    pa[:k] = A[:k] + rng.normal(size=(k, 3)) * 1e-13          # near a vertex (nr ~ 0)
    C[k:2 * k] = A[k:2 * k] + (B[k:2 * k] - A[k:2 * k]) * 0.5  # degenerate facets
    B[2 * k:3 * k, 2] = A[2 * k:3 * k, 2]
    C[2 * k:3 * k, 2] = A[2 * k:3 * k, 2]                      # flat facets
    return pa, A, B, C


def _selftest(lib):
    """Bit-compare the native candidate tests with NumPy on this machine."""
    rng = np.random.default_rng(20261024)
    combos = [
        dict(reduce_angle_edge=True, stop_triangulation=False, use_distance_as_rating=False,
             add_only_upward=False, iteration_distance=1.4, iteration_angle=6.0),
        dict(reduce_angle_edge=False, stop_triangulation=True, use_distance_as_rating=False,
             add_only_upward=True, iteration_distance=0.5, iteration_angle=4.0),
        dict(reduce_angle_edge=True, stop_triangulation=True, use_distance_as_rating=True,
             add_only_upward=False, iteration_distance=1.1, iteration_angle=8.0),
        dict(reduce_angle_edge=True, stop_triangulation=False, use_distance_as_rating=True,
             add_only_upward=True, iteration_distance=0.0, iteration_angle=10.0),
    ]
    for combo in combos:
        kwargs = dict(terrain_rad=float(np.radians(rng.uniform(15, 88))),
                      edge_length_threshold=5.0, stop_edge_length=2.0,
                      distance_weight=float(rng.choice([0.0, 37.5, 50.0, 100.0])), **combo)
        pa, A, B, C = (np.ascontiguousarray(x) for x in _synthetic_rows(rng, 120_000))
        ref = reference_rows(pa, A, B, C, **kwargs)
        n = len(pa)
        act = np.empty(n, np.uint8); alpha = np.empty(n); pd = np.empty(n)
        me = np.empty(n); nor = np.empty((n, 3)); tok = np.empty(n, np.uint8)
        params = make_params(one_per_simplex=True, **kwargs)
        rc = lib.nk_ptd_evaluate_rows(
            pa.ctypes.data_as(_C_DBL), A.ctypes.data_as(_C_DBL), B.ctypes.data_as(_C_DBL),
            C.ctypes.data_as(_C_DBL), n, ctypes.byref(params),
            act.ctypes.data_as(_C_U8), alpha.ctypes.data_as(_C_DBL), pd.ctypes.data_as(_C_DBL),
            me.ctypes.data_as(_C_DBL), nor.ctypes.data_as(_C_DBL), tok.ctypes.data_as(_C_U8))
        if rc != 0:
            return f'native evaluate_rows status {rc}'
        checks = (('act', act.astype(bool), ref[0]), ('tok', tok.astype(bool), ref[5]))
        for name, got, want in checks:
            if not np.array_equal(got, want):
                return f'{name} differs in {int(np.count_nonzero(got != want))} rows'
        for name, got, want in (('alpha', alpha, ref[1]), ('pd', pd, ref[2]),
                                ('max_edge', me, ref[3]), ('normal', nor, ref[4])):
            if not np.array_equal(np.asarray(got).view(np.uint64), np.asarray(want).view(np.uint64)):
                return f'{name} bits differ from NumPy'
    return True


def available():
    """True when the engine may be used in this process (cached self-test)."""
    global _SELFTEST
    if not enabled():
        return False
    try:
        import scipy
        if scipy.__version__ != '1.15.3':
            _warn_once(f'disabled: SciPy {scipy.__version__} walker parity is not validated')
            return False
    except Exception:
        return False
    lib = _load()
    if lib is None:
        _warn_once(f'unavailable ({_LIB_ERROR}); original Ground path used')
        return False
    with _STATE_LOCK:
        if _SELFTEST is None:
            try:
                _SELFTEST = _selftest(lib)
            except Exception as exc:
                _SELFTEST = f'self-test error {type(exc).__name__}: {exc}'
            if _SELFTEST is True:
                print('[Ground incremental] stateful C++ evaluation ready '
                      '(candidate bit-parity self-test passed; TINs use SciPy rebuilds)', flush=True)
    if _SELFTEST is not True:
        _warn_once(f'disabled: {_SELFTEST}')
        return False
    return True


class IncrementalPTD:
    """One engine per classify_ground_ptd call (per tile); not thread-shared."""

    def __init__(self, pts, params, kwargs=None):
        self._kwargs = dict(kwargs or {})
        self._lib = _load()
        self._pts = np.ascontiguousarray(pts, dtype=np.float64)
        if self._pts.ndim != 2 or self._pts.shape[1] < 3:
            raise ValueError('points must be N x 3')
        if self._pts.shape[1] != 3:
            self._pts = np.ascontiguousarray(self._pts[:, :3])
        self._params = params
        self._handle = self._lib.nk_ptd_create(
            self._pts.ctypes.data_as(_C_DBL), len(self._pts), ctypes.byref(params))
        if not self._handle:
            raise RuntimeError('native incremental engine creation failed')
        self.audit = audit_enabled()
        self.stats = dict(calls=0, queries=0, cache_cert=0, grid_cert=0, walked=0,
                          brute=0, eval_reused=0, eval_new=0, map_s=0.0, locate_s=0.0,
                          evaluate_s=0.0, select_s=0.0, transform_s=0.0, audit_checks=0)
        self.stats.update(topology_mode='reference_qhull_rebuild', persistent_tin_updates=0,
                          locator_mode='complete_ordered_walk', location_shortcuts=False)

    def close(self):
        if getattr(self, '_handle', None):
            self._lib.nk_ptd_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def run(self, tri, tin_ids, queries, mode):
        """Return ('ok', ids) / ('none_inside', None), or None to use the original path.

        mode 0: PTD iteration; mode 1: final surface completion (all passing).
        """
        if not self._handle:
            return None
        simplices = np.ascontiguousarray(tri.simplices, dtype=np.int32)
        neighbors = np.ascontiguousarray(tri.neighbors, dtype=np.int32)
        equations = np.ascontiguousarray(tri.equations, dtype=np.float64)
        points = np.ascontiguousarray(tri.points, dtype=np.float64)
        started = perf_counter()
        with blas_single_thread():
            transform = np.ascontiguousarray(tri.transform, dtype=np.float64)
        self.stats['transform_s'] += perf_counter() - started
        ntri = len(simplices)
        if (getattr(tri, 'ndim', 2) != 2 or simplices.shape != (ntri, 3) or
                neighbors.shape != (ntri, 3) or equations.shape != (ntri, 4) or
                transform.shape != (ntri, 3, 2) or points.ndim != 2 or points.shape[1] != 2):
            return None
        tin_ids = np.ascontiguousarray(tin_ids, dtype=np.int64)
        queries = np.ascontiguousarray(queries, dtype=np.int64)
        low = np.asarray(tri.min_bound, dtype=np.float64)
        high = np.asarray(tri.max_bound, dtype=np.float64)
        nq = len(queries)
        selected = np.empty(nq, np.int64)
        nsel = ctypes.c_int64(0)
        stats = (ctypes.c_int64 * 16)()
        times = (ctypes.c_double * 4)()
        if self.audit:
            sids = np.empty(nq, np.int32); act = np.empty(nq, np.uint8)
            alpha = np.empty(nq); pd = np.empty(nq)
            extra = (sids.ctypes.data_as(_C_I32), act.ctypes.data_as(_C_U8),
                     alpha.ctypes.data_as(_C_DBL), pd.ctypes.data_as(_C_DBL))
        else:
            extra = (None, None, None, None)
        rc = self._lib.nk_ptd_iterate(
            self._handle, int(mode),
            tin_ids.ctypes.data_as(_C_I64), len(tin_ids),
            simplices.ctypes.data_as(_C_I32), neighbors.ctypes.data_as(_C_I32),
            equations.ctypes.data_as(_C_DBL), transform.ctypes.data_as(_C_DBL), ntri,
            points.ctypes.data_as(_C_DBL),
            float(low[0]), float(low[1]), float(high[0]), float(high[1]),
            float(tri.paraboloid_scale), float(tri.paraboloid_shift),
            queries.ctypes.data_as(_C_I64), nq,
            selected.ctypes.data_as(_C_I64), ctypes.byref(nsel),
            *extra, stats, times)
        if rc not in (0, 2):
            _warn_once(f'native status {rc}; original path used for this tile')
            self.close()
            return None
        st = self.stats
        st['calls'] += 1; st['queries'] += int(stats[0]); st['cache_cert'] += int(stats[2])
        st['grid_cert'] += int(stats[3]); st['walked'] += int(stats[4]); st['brute'] += int(stats[5])
        st['eval_reused'] += int(stats[7]); st['eval_new'] += int(stats[8])
        st['map_s'] += times[0]; st['locate_s'] += times[1]
        st['evaluate_s'] += times[2]; st['select_s'] += times[3]
        if rc == 2:
            result = ('none_inside', None, int(stats[6]))
        else:
            result = ('ok', selected[:nsel.value].copy(), int(stats[6]))
        if self.audit:
            try:
                self._audit(tri, tin_ids, queries, mode, rc, sids,
                            act.astype(bool), alpha, pd, result)
            except AssertionError as exc:
                AUDIT_FAILURES.append(str(exc))
                print('[Ground incremental] AUDIT FAILURE: ' + str(exc), flush=True)
                self.close()
                return None  # original path recomputes this iteration
        return result

    def _audit(self, tri, tin_ids, queries, mode, rc, sids, act, alpha, pd, result):
        """Validation only: compare with SciPy find_simplex and NumPy tests."""
        with blas_single_thread():
            ref_sids = tri.find_simplex(self._pts[queries, :2])
        if not np.array_equal(ref_sids, sids):
            bad = np.flatnonzero(ref_sids != sids)
            raise AssertionError(f'incremental audit: {len(bad)} simplex IDs differ, '
                                 f'first query {int(queries[bad[0]])}')
        valid = ref_sids != -1
        if rc == 2:
            if valid.any():
                raise AssertionError('incremental audit: native reported no candidate inside TIN')
            return
        vq = queries[valid]; vs = ref_sids[valid]
        tri_points = self._pts[tin_ids]
        sv = tri.simplices[vs]
        A = tri_points[sv[:, 0]]; B = tri_points[sv[:, 1]]; C = tri_points[sv[:, 2]]
        p = self._params
        kw = {k: v for k, v in self._kwargs.items() if k != 'one_per_simplex'}
        ref = reference_rows(self._pts[vq], A, B, C, **kw)
        if not np.array_equal(ref[0], act[valid]):
            raise AssertionError('incremental audit: PTD acceptance flags differ')
        if not (np.array_equal(ref[1].view(np.uint64), alpha[valid].view(np.uint64)) and
                np.array_equal(ref[2].view(np.uint64), pd[valid].view(np.uint64))):
            raise AssertionError('incremental audit: angle/distance bits differ')
        pass_pos = np.flatnonzero(ref[0])
        if mode == 0 and p.one_per_simplex and len(pass_pos):
            order = np.lexsort((vq[pass_pos], ref[2][pass_pos], ref[1][pass_pos], vs[pass_pos]))
            ordered = vs[pass_pos][order]
            first = np.r_[True, ordered[1:] != ordered[:-1]]
            expected = vq[pass_pos[order[first]]]
        else:
            expected = vq[pass_pos]
        if not np.array_equal(np.sort(expected), np.sort(result[1])):
            raise AssertionError('incremental audit: selected Ground anchors differ')
        self.stats['audit_checks'] += 1


def create_engine(pts, **kwargs):
    """Return an IncrementalPTD or None (original path)."""
    if not available():
        return None
    try:
        return IncrementalPTD(pts, make_params(**kwargs), kwargs)
    except Exception as exc:
        _warn_once(f'creation failed ({type(exc).__name__}); original Ground path used')
        return None


def trace_ordered_walk(tri, query_xy):
    """Bounded hook into the actual engine walker, independent of opt-in.

    Run unknown captures in a subprocess with a wall timeout. Native cycles
    have the same finite directed-walk limit/fallback as the engine.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError(_LIB_ERROR or 'native diagnostic unavailable')
    xy = np.ascontiguousarray(query_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) > 1_000_000:
        raise ValueError('diagnostic requires at most 1,000,000 ordered XY queries')
    neighbors = np.ascontiguousarray(tri.neighbors, dtype=np.int32)
    equations = np.ascontiguousarray(tri.equations, dtype=np.float64)
    transform = np.ascontiguousarray(tri.transform, dtype=np.float64)
    ntri = len(neighbors)
    if neighbors.shape != (ntri, 3) or equations.shape != (ntri, 4) or transform.shape != (ntri, 3, 2):
        raise ValueError('invalid diagnostic triangle arrays')
    output = np.empty(len(xy), dtype=np.int32)
    trace = np.empty((len(xy), 5), dtype=np.int64)
    fn = lib.nk_ptd_trace_walk
    fn.restype = ctypes.c_int
    fn.argtypes = [_C_DBL, ctypes.c_int64, _C_I32, _C_DBL, _C_DBL, ctypes.c_int64,
                   *([ctypes.c_double] * 6), _C_I32, _C_I64]
    rc = fn(xy.ctypes.data_as(_C_DBL), len(xy), neighbors.ctypes.data_as(_C_I32),
            equations.ctypes.data_as(_C_DBL), transform.ctypes.data_as(_C_DBL), ntri,
            float(tri.min_bound[0]), float(tri.min_bound[1]),
            float(tri.max_bound[0]), float(tri.max_bound[1]),
            float(tri.paraboloid_scale), float(tri.paraboloid_shift),
            output.ctypes.data_as(_C_I32), trace.ctypes.data_as(_C_I64))
    if rc:
        raise RuntimeError(f'ordered diagnostic refused incomplete/invalid walk ({rc})')
    return output, trace
