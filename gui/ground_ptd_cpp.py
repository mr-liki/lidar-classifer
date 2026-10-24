"""Optional native C++ PTD simplex walker; opt-in, read-only, parity first.

Only SciPy 1.15.3 is an approved production execution target. The C++
walker and its ABI live outside the existing Vulkan renderer.  The caller
must fall back to the COMPLETE SciPy query on every unsupported input,
load failure or nonzero native return code.  A platform must pass the full
SciPy audit (`NAKSHA_GROUND_V2_AUDIT=1`) before disabling auditing.

Usage (Windows, from project root, after building the native DLL):
    $env:NAKSHA_GROUND_ENGINE_V2 = '1'
    $env:NAKSHA_GROUND_CPP = '1'
    $env:NAKSHA_GROUND_V2_AUDIT = '1'

Do not confuse this with the conservative CUDA prefilter or Vulkan renderer.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from pathlib import Path

import numpy as np

_C_DBL = ctypes.POINTER(ctypes.c_double)
_C_INT = ctypes.POINTER(ctypes.c_int32)


def _library_candidates():
    env = os.environ.get('NAKSHA_GROUND_CPP_LIBRARY', '').strip()
    if env:
        # An explicit user path overrides discovery. Never silently substitute
        # another binary if a specified path is absent/incompatible.
        yield Path(env).expanduser().resolve()
        return
    base = Path(__file__).resolve().parents[1] / 'native' / 'ground_ptd' / 'build'
    for name in (
        'Release/naksha_ground_ptd_exact.dll',
        'naksha_ground_ptd_exact.dll',
        'libnaksha_ground_ptd_exact.so',
        'libnaksha_ground_ptd_exact.dylib',
    ):
        yield base / name


@lru_cache(maxsize=4)
def _load_library(path):
    lib = ctypes.CDLL(str(path))
    version = lib.naksha_ground_kernel_api_version
    version.restype = ctypes.c_int
    version.argtypes = []
    if version() != 1:
        raise RuntimeError('Native Ground PTD ABI mismatch')
    fn = lib.naksha_ground_find_simplex_2d
    fn.restype = ctypes.c_int
    fn.argtypes = [
        _C_DBL, ctypes.c_int64,
        _C_INT, _C_DBL, _C_DBL,
        ctypes.c_int64,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, _C_INT,
    ]
    return fn


@lru_cache(maxsize=4)
def _load_grid_function(path):
    # Call the existing ABI loader first: never accept a mismatched binary.
    _load_library(path)
    lib = ctypes.CDLL(str(path))
    fn = lib.naksha_ground_find_simplex_2d_grid
    fn.restype = ctypes.c_int
    fn.argtypes = [
        _C_DBL, ctypes.c_int64,
        _C_INT, _C_DBL, _C_DBL,
        ctypes.c_int64,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
        _C_INT, _C_DBL, ctypes.c_int64,
        _C_INT, ctypes.POINTER(ctypes.c_int64),
    ]
    return fn


@lru_cache(maxsize=4)
def _load_stage4_function(path):
    # Never bypass the Stage 1 ABI compatibility check.
    _load_library(path)
    lib = ctypes.CDLL(str(path))
    fn = lib.naksha_ground_find_simplex_2d_grid_stage4
    fn.restype = ctypes.c_int
    fn.argtypes = [
        _C_DBL, ctypes.c_int64,
        _C_INT, _C_DBL, _C_DBL,
        ctypes.c_int64,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
        _C_INT, _C_DBL, ctypes.c_int64,
        _C_INT, ctypes.POINTER(ctypes.c_int64), _C_DBL,
    ]
    return fn


def native_find_simplex(tri, query_xy, stats=None):
    """Run native lookup, preserving order. Raises for any unsafe condition.

    The caller owns all array buffers until the synchronous native call ends.
    This function does not change SciPy triangulation or any source arrays.
    """
    xy = np.asarray(query_xy)
    if (getattr(tri, 'ndim', None) != 2 or xy.ndim != 2 or
            xy.shape[1] != 2 or xy.dtype != np.float64 or
            not xy.flags.c_contiguous or not np.isfinite(xy).all()):
        raise ValueError('unsupported query layout')
    neighbors = np.asarray(tri.neighbors)
    equations = np.asarray(tri.equations)
    transform = np.asarray(tri.transform)
    n = len(neighbors)
    if (not 0 < n <= np.iinfo(np.int32).max or
            neighbors.shape != (n, 3) or neighbors.dtype != np.int32 or
            equations.shape != (n, 4) or equations.dtype != np.float64 or
            transform.shape != (n, 3, 2) or transform.dtype != np.float64 or
            not neighbors.flags.c_contiguous or
            not equations.flags.c_contiguous or
            not transform.flags.c_contiguous):
        raise ValueError('unsupported Delaunay array layout')
    low = np.asarray(tri.min_bound)
    high = np.asarray(tri.max_bound)
    if (low.shape != (2,) or high.shape != (2,) or
            not np.isfinite(low).all() or not np.isfinite(high).all()):
        raise ValueError('unsupported bounds')
    lib_path = next((path for path in _library_candidates() if path.is_file()), None)
    if lib_path is None:
        raise FileNotFoundError('native Ground PTD library was not built')
    fn = _load_library(str(lib_path))
    out = np.empty(len(xy), dtype=np.int32)
    # Stage 3 is a strictly interior spatial proposal. Every boundary,
    # outside-hull and uncertain query uses the unchanged ordered walker.
    # It is OFF unless explicitly opted in after full SciPy audit.
    if os.environ.get('NAKSHA_GROUND_CPP_GRID', '0').lower() in (
            '1', 'true', 'yes', 'on'):
        try:
            faces = np.asarray(tri.simplices)
            points = np.asarray(tri.points)
            if (faces.shape != (n, 3) or faces.dtype != np.int32 or
                    not faces.flags.c_contiguous or points.ndim != 2 or
                    points.shape[1] != 2 or points.dtype != np.float64 or
                    not points.flags.c_contiguous or not np.isfinite(points).all()):
                raise ValueError('unsupported indexed simplex layout')
            # Stage 4 is independently opt-in. It adds adaptive fine bins
            # and AABB pruning, but keeps exactly the original TIN and
            # unchanged Stage 1 ordered-walker fallback.
            stage4_on = os.environ.get('NAKSHA_GROUND_CPP_STAGE4', '0').lower() in (
                '1', 'true', 'yes', 'on')
            if stage4_on:
                try:
                    stage4_fn = _load_stage4_function(str(lib_path))
                    metrics = (ctypes.c_int64 * 7)()
                    timings = (ctypes.c_double * 2)()
                    status = stage4_fn(
                        xy.ctypes.data_as(_C_DBL), len(xy),
                        neighbors.ctypes.data_as(_C_INT),
                        equations.ctypes.data_as(_C_DBL),
                        transform.ctypes.data_as(_C_DBL), n,
                        float(low[0]), float(low[1]), float(high[0]), float(high[1]),
                        float(tri.paraboloid_scale), float(tri.paraboloid_shift),
                        faces.ctypes.data_as(_C_INT),
                        points.ctypes.data_as(_C_DBL), len(points),
                        out.ctypes.data_as(_C_INT), metrics, timings,
                    )
                    if status != 0:
                        raise RuntimeError(f'stage4 native kernel status {status}')
                    if stats is not None:
                        stats['grid_queries'] = stats.get('grid_queries', 0) + len(xy)
                        stats['grid_fast_hits'] = stats.get('grid_fast_hits', 0) + metrics[0]
                        stats['grid_ordered_fallbacks'] = (
                            stats.get('grid_ordered_fallbacks', 0) + len(xy) - metrics[0])
                        stats['stage4_queries'] = stats.get('stage4_queries', 0) + len(xy)
                        for key, value in (
                            ('stage4_probes', metrics[1]),
                            ('stage4_bbox_pruned', metrics[2]),
                            ('stage4_refined_queries', metrics[3]),
                            ('stage4_refined_bins', metrics[4]),
                            ('stage4_refined_entries', metrics[5]),
                            ('stage4_ambiguous', metrics[6]),
                        ):
                            stats[key] = stats.get(key, 0) + int(value)
                        for key, value in (
                            ('stage4_build_seconds', timings[0]),
                            ('stage4_lookup_seconds', timings[1]),
                        ):
                            stats[key] = stats.get(key, 0.0) + float(value)
                    return out
                except (AttributeError, ValueError, RuntimeError, OSError) as exc:
                    if stats is not None:
                        stats['stage4_fallback'] = type(exc).__name__
            grid_fn = _load_grid_function(str(lib_path))
            fast_hits = ctypes.c_int64(0)
            status = grid_fn(
                xy.ctypes.data_as(_C_DBL), len(xy),
                neighbors.ctypes.data_as(_C_INT),
                equations.ctypes.data_as(_C_DBL),
                transform.ctypes.data_as(_C_DBL), n,
                float(low[0]), float(low[1]), float(high[0]), float(high[1]),
                float(tri.paraboloid_scale), float(tri.paraboloid_shift),
                faces.ctypes.data_as(_C_INT),
                points.ctypes.data_as(_C_DBL), len(points),
                out.ctypes.data_as(_C_INT), ctypes.byref(fast_hits),
            )
            if status != 0:
                raise RuntimeError(f'grid kernel status {status}')
            if stats is not None:
                stats['grid_queries'] = stats.get('grid_queries', 0) + len(xy)
                stats['grid_fast_hits'] = stats.get('grid_fast_hits', 0) + fast_hits.value
                stats['grid_ordered_fallbacks'] = (
                    stats.get('grid_ordered_fallbacks', 0) + len(xy) - fast_hits.value)
            return out
        except (AttributeError, ValueError, RuntimeError, OSError) as exc:
            # Missing Stage 3 DLL export, unsafe input or unsuccessful proposal
            # is nonfatal. Run all queries via the known Stage 1 walker.
            if stats is not None:
                stats['grid_fallback'] = type(exc).__name__
    status = fn(
        xy.ctypes.data_as(_C_DBL), len(xy),
        neighbors.ctypes.data_as(_C_INT),
        equations.ctypes.data_as(_C_DBL),
        transform.ctypes.data_as(_C_DBL), n,
        float(low[0]), float(low[1]), float(high[0]), float(high[1]),
        float(tri.paraboloid_scale), float(tri.paraboloid_shift),
        out.ctypes.data_as(_C_INT),
    )
    if status != 0:
        raise RuntimeError(f'native Ground PTD incomplete/invalid lookup: {status}')
    return out
