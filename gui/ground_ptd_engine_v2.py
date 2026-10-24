"""Opt-in, per-PTD-call acceleration; no writes and no global geometry cache.

The 2-D walker preserves SciPy 1.15.3's full-array start state, neighbor
order and float64 operation order. Degeneracy/cycle failures discard the
ENTIRE proposal, never a subset. Optional audit uses the full original array.
TIN and query reuse requires bit-identical ordered inputs.

Walker adapted from scipy/spatial/_qhull.pyx (Pauli Virtanen, 2010).
See ground_ptd_engine_v2_LICENSE.txt for the SciPy BSD license.
"""
from __future__ import annotations

from time import perf_counter
import os
import numpy as np
import scipy
from scipy.spatial import Delaunay

try:
    from numba import njit
except (ImportError, OSError):
    njit = None


if njit is not None:
    @njit(cache=True, nogil=True, fastmath=False)
    def _brute_2d(x, y, neighbors, transform, eps):
        # Preserve simplex scan order and degenerate-neighbor broad tolerance.
        # This fallback is part of SciPy's algorithm, not nearest-vertex search.
        broad = np.sqrt(eps)
        c = np.empty(3, dtype=np.float64)
        for sid in range(len(neighbors)):
            if not np.isnan(transform[sid, 0, 0]):
                c[2] = 1.0
                inside = True
                for k in range(2):
                    c[k] = 0.0
                    c[k] += transform[sid, k, 0] * (x - transform[sid, 2, 0])
                    c[k] += transform[sid, k, 1] * (y - transform[sid, 2, 1])
                    c[2] -= c[k]
                    if not (-eps <= c[k] <= 1.0 + eps):
                        inside = False
                        break
                if inside and -eps <= c[2] <= 1.0 + eps:
                    return sid
            else:
                for k in range(3):
                    other = neighbors[sid, k]
                    if other == -1 or np.isnan(transform[other, 0, 0]):
                        continue
                    c[2] = 1.0
                    for m in range(2):
                        c[m] = 0.0
                        c[m] += transform[other, m, 0] * (x - transform[other, 2, 0])
                        c[m] += transform[other, m, 1] * (y - transform[other, 2, 1])
                        c[2] -= c[m]
                    inside = True
                    for m in range(3):
                        low = -broad if neighbors[other, m] == sid else -eps
                        if not (low <= c[m] <= 1.0 + eps):
                            inside = False
                            break
                    if inside:
                        return other
        return -1

    @njit(cache=True, nogil=True, fastmath=False)
    def _rank_per_simplex(vng, sids, distance, angle, passing, ntri):
        # Same lexicographic keys as lexsort((point, distance, angle, sid)),
        # including stable first occurrence on an exact tie. Passing PTD
        # distances/angles are finite; unsupported keys use the NumPy route.
        best = np.full(ntri, -1, dtype=np.intp)
        for pos in passing:
            sid = sids[pos]
            old = best[sid]
            if (old == -1 or angle[pos] < angle[old] or
                    (angle[pos] == angle[old] and
                     (distance[pos] < distance[old] or
                      (distance[pos] == distance[old] and vng[pos] < vng[old])))):
                best[sid] = pos
        return best[best >= 0]

    @njit(cache=True, nogil=True, fastmath=False)
    def _walk_2d(xy, neighbors, equations, transform, lo, hi, scale, shift):
        """Return (IDs, success); preserve one start state for ALL queries."""
        eps = 100.0 * np.finfo(np.float64).eps
        ntri = len(neighbors)
        out = np.full(len(xy), -1, dtype=np.int32)
        start = 0
        c = np.empty(3, dtype=np.float64)
        for row in range(len(xy)):
            x, y = xy[row, 0], xy[row, 1]
            if (x < lo[0] - eps or x > hi[0] + eps or
                    y < lo[1] - eps or y > hi[1] + eps or ntri <= 0):
                continue
            sid = start
            if sid < 0 or sid >= ntri:
                sid = 0
            z = 0.0
            z += x * x
            z += y * y
            z *= scale
            z += shift
            best = equations[sid, 3]
            best += equations[sid, 0] * x
            best += equations[sid, 1] * y
            best += equations[sid, 2] * z
            changed = True
            # Guard only aborts to the complete SciPy operation.
            steps = 0
            while changed:
                if best > 0:
                    break
                changed = False
                for k in range(3):
                    other = neighbors[sid, k]
                    if other == -1:
                        continue
                    dist = equations[other, 3]
                    dist += equations[other, 0] * x
                    dist += equations[other, 1] * y
                    dist += equations[other, 2] * z
                    if dist > best + eps * (1.0 + abs(best)):
                        sid = other
                        best = dist
                        changed = True
                steps += 1
                if steps > ntri + 1:
                    return out, False
            start = sid
            for cycle in range(1 + ntri // 4):
                inside = 1
                for k in range(3):
                    if k == 2:
                        c[2] = 1.0
                        c[2] -= c[0]
                        c[2] -= c[1]
                    else:
                        c[k] = 0.0
                        c[k] += transform[sid, k, 0] * (x - transform[sid, 2, 0])
                        c[k] += transform[sid, k, 1] * (y - transform[sid, 2, 1])
                    if c[k] < -eps:
                        other = neighbors[sid, k]
                        if other == -1:
                            start = sid
                            sid = -1
                            inside = 2
                            break
                        sid = other
                        inside = -1
                        break
                    elif c[k] <= 1.0 + eps:
                        pass
                    else:
                        inside = 0
                if inside == -1:
                    continue
                if inside == 2:
                    break
                if inside == 1:
                    start = sid
                    break
                sid = _brute_2d(x, y, neighbors, transform, eps)
                start = sid
                break
            else:
                sid = _brute_2d(x, y, neighbors, transform, eps)
                start = sid
            out[row] = sid
        return out, True
else:
    _walk_2d = None
    _rank_per_simplex = None


def choose_per_simplex(vng, sids, distance, angle, passing, ntri):
    """Linear-time exact finite-key equivalent of PTD's per-facet lexsort."""
    if (_rank_per_simplex is not None and np.isfinite(distance[passing]).all()
            and np.isfinite(angle[passing]).all()):
        try:
            return _rank_per_simplex(vng, sids, distance, angle, passing, ntri)
        except Exception:
            pass
    order = np.lexsort((vng[passing], distance[passing], angle[passing], sids[passing]))
    ordered = sids[passing][order]
    first = np.r_[True, ordered[1:] != ordered[:-1]]
    return passing[order[first]] if len(passing) else passing.copy()


def _same_bits(left, right):
    return (left is not None and left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def find_simplex(tri, query_xy, *, stats=None, verify=True):
    """Ordered 2-D SciPy algorithm, with full-query fallback and optional audit.

    No changed query order, search seed, transform arithmetic, tolerance,
    degenerate-neighbor handling, or simplex numbering. The public validation
    helper audits by default. V2 can explicitly request the tested compiled
    algorithm; NAKSHA_GROUND_V2_AUDIT=1 enables the full-query runtime audit.
    """
    stats = {} if stats is None else stats
    def fallback(reason):
        stats['fallback'] = reason
        reference = tri.find_simplex(query_xy)
        stats['scipy_queries'] = stats.get('scipy_queries', 0) + reference.size
        return reference
    if scipy.__version__ != '1.15.3':
        return fallback('unsupported runtime')
    xy = np.asarray(query_xy)
    if (xy.ndim != 2 or xy.shape[1] != 2 or xy.dtype != np.float64
            or tri.ndim != 2 or not np.isfinite(xy).all()):
        return fallback('unsupported input')
    # Vulkan GPU *parallel triangle locator* (separate from rendering).
    # It returns conservative, strictly interior simplex proposals. The
    # original CPU/SciPy reference is authoritative in the default full-audit
    # mode; all ambiguous queries use the exact CPU path. Source point order,
    # TIN triangle indices, per-facet ranks and LNT commits are unchanged.
    if os.environ.get('NAKSHA_GROUND_VK_LOCATOR', '0').lower() in (
            '1', 'true', 'yes', 'on'):
        try:
            from gui.ground_ptd_vulkan_locator import locate as _vulkan_locate
            result = _vulkan_locate(tri, np.ascontiguousarray(xy),
                                    lambda q: tri.find_simplex(q), stats)
            if result is not None:
                return result
        except Exception as exc:
            stats['vk_locator_fallback'] = type(exc).__name__
    # EXPERIMENTAL. This is an optional C++ implementation of the same
    # ordered walk, independent of the Vulkan renderer. Failure or incomplete
    # traversal always falls through to the original full SciPy/Numba path.
    cpp_requested = os.environ.get('NAKSHA_GROUND_CPP', '0').lower() in (
        '1', 'true', 'yes', 'on')
    if cpp_requested:
        try:
            from gui.ground_ptd_cpp import native_find_simplex
            result = native_find_simplex(tri, np.ascontiguousarray(xy), stats)
            stats['native_cpp_queries'] = stats.get('native_cpp_queries', 0) + len(xy)
            if verify:
                reference = tri.find_simplex(query_xy)
                stats['audited_queries'] = stats.get('audited_queries', 0) + len(xy)
                if not np.array_equal(result, reference):
                    stats['fallback'] = 'complete-query equivalence failure'
                    return reference
            return result
        except Exception as exc:
            stats['native_cpp_fallback'] = type(exc).__name__
    if _walk_2d is None:
        return fallback('numba unavailable')
    try:
        result, ok = _walk_2d(np.ascontiguousarray(xy), tri.neighbors,
                             tri.equations, tri.transform, tri.min_bound,
                             tri.max_bound, tri.paraboloid_scale,
                             tri.paraboloid_shift)
        if not ok:
            return fallback('incomplete ordered walk')
        stats['compiled_queries'] = stats.get('compiled_queries', 0) + len(xy)
        if verify:
            reference = tri.find_simplex(query_xy)
            stats['audited_queries'] = stats.get('audited_queries', 0) + len(xy)
            if not np.array_equal(result, reference):
                stats['fallback'] = 'complete-query equivalence failure'
                return reference
        return result
    except Exception as exc:
        return fallback(type(exc).__name__)


class GroundEngineV2:
    """One immutable TIN and one exact lookup retained within one PTD call.

    Immutable reuse and an ordered compiled 2-D locator, with full-query
    fallback. Cache memory is bounded to the most recent TIN and query.
    """
    def __init__(self, factory=Delaunay):
        self.factory = factory
        self.audit = os.environ.get('NAKSHA_GROUND_V2_AUDIT', '0').lower() in (
            '1', 'true', 'yes', 'on')
        self._compiled_disabled = False
        self._xy = self._tri = self._query = self._sids = None
        self.stats = dict(tin_builds=0, tin_reuses=0, lookup_calls=0,
                          lookup_reuses=0, tin_seconds=0.0, lookup_seconds=0.0)

    def triangulate(self, xy):
        xy = np.ascontiguousarray(xy, dtype=np.float64)
        if _same_bits(self._xy, xy):
            self.stats['tin_reuses'] += 1
            return self._tri
        # Release the previous terrain/search arrays before allocating the
        # next TIN. Keep peak memory bounded during changing-anchor iterations.
        self._xy = self._tri = self._query = self._sids = None
        started = perf_counter()
        tri = self.factory(xy.copy())
        self.stats['tin_seconds'] += perf_counter() - started
        self.stats['tin_builds'] += 1
        # Only publish a successfully built TIN. Own the comparison key;
        # never let a caller mutate it via its original input view.
        self._xy, self._tri = xy.copy(), tri
        self._query = self._sids = None
        return tri

    def locate(self, tri, xy):
        xy = np.ascontiguousarray(xy, dtype=np.float64)
        if tri is self._tri and _same_bits(self._query, xy):
            self.stats['lookup_reuses'] += 1
            return self._sids.copy()
        started = perf_counter()
        result = (tri.find_simplex(xy) if self._compiled_disabled else
                  find_simplex(tri, xy, stats=self.stats, verify=self.audit))
        self.stats['lookup_seconds'] += perf_counter() - started
        self.stats['lookup_calls'] += 1
        if tri is self._tri:
            if self.stats.get('fallback') == 'complete-query equivalence failure':
                # Never keep using compiled results after a detected mismatch.
                self._compiled_disabled = True
            self._query, self._sids = xy.copy(), result.copy()
        return result
