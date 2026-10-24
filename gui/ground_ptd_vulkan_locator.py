"""Experimental Vulkan GPU 2-D TIN simplex location, WITHOUT changing PTD rules.

CPU SciPy still creates Delaunay connectivity. GPU proposes *strictly interior*
simplex IDs for parallel queries via a conservative CSR AABB spatial index.
Ambiguous/boundary/outside-hull queries always use the existing exact locator.

The default mode FULLY AUDITS all GPU replies against the unchanged CPU
reference, discarding the entire GPU result if even one ID differs. Audit
mode intentionally has no speed guarantee and must be used for validation.

NAKSHA_GROUND_VK_LOCATOR=1                  opt in
NAKSHA_GROUND_VK_LOCATOR_AUDIT=1            full CPU equivalence check (default)
NAKSHA_GROUND_VK_LOCATOR_EXPERIMENTAL_FAST=1 permit certified fast path
                                           only when audit explicitly 0.
"""
from __future__ import annotations

import atexit
import ctypes
import os
from pathlib import Path
from threading import RLock
from time import perf_counter

import numpy as np

_LOCK = RLock()
_CONTEXT = None
_DLL = None
_KEY = None
_WARNED = set()
_GRID_SIZE = 64
_MAX_ENTRIES = 2_500_000
_MAX_TRI = 500_000
_MIN_QUERIES = 8192
_STATS = dict(calls=0, queries=0, fast_proposals=0, cpu_fallback=0,
              audited=0, mismatches=0, gpu_seconds=0., index_seconds=0.)


def _on(name, default='0'):
    return os.environ.get(name, default).strip().lower() in ('1', 'yes', 'true', 'on')


def _enabled():
    return _on('NAKSHA_GROUND_VK_LOCATOR')


def _warn(s):
    if s not in _WARNED:
        _WARNED.add(s)
        print('[Ground Vulkan Locator] ' + s, flush=True)


def _paths():
    root = Path(__file__).resolve().parents[1]
    directory = root / 'native' / 'ground_vulkan_locator' / 'build_vulkan_locator'
    if os.name == 'nt':
        directory /= 'Release'
    name = 'naksha_ground_vulkan_locator.dll' if os.name == 'nt' else 'naksha_ground_vulkan_locator.so'
    dll = Path(os.environ.get('NAKSHA_GROUND_VK_LOCATOR_LIBRARY') or directory / name).expanduser().resolve()
    shader = Path(os.environ.get('NAKSHA_GROUND_VK_LOCATOR_SHADER') or dll.parent / 'ground_locate.comp.spv').expanduser().resolve()
    return dll, shader


def _bind(lib):
    lib.nk_ground_vk_locator_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int]
    lib.nk_ground_vk_locator_create.restype = ctypes.c_int
    lib.nk_ground_vk_locator_locate.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_double), ctypes.c_uint32,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_void_p, ctypes.c_int,
    ]
    lib.nk_ground_vk_locator_locate.restype = ctypes.c_int
    lib.nk_ground_vk_locator_release.argtypes = [ctypes.c_void_p]
    lib.nk_ground_vk_locator_release.restype = None


def _context():
    global _CONTEXT, _DLL, _KEY
    dll, shader = _paths()
    key = str(dll), str(shader)
    if _CONTEXT is not None:
        if key != _KEY:
            raise RuntimeError('GPU locator DLL/shader changed during session; restart Naksha')
        return _DLL, _CONTEXT
    if not dll.is_file() or not shader.is_file():
        raise FileNotFoundError('GPU locator DLL or shader missing (build first)')
    lib = ctypes.CDLL(str(dll))
    _bind(lib)
    ctx = ctypes.c_void_p()
    message = ctypes.create_string_buffer(512)
    status = lib.nk_ground_vk_locator_create(os.fsencode(shader), ctypes.byref(ctx), message, len(message))
    if status or not ctx.value:
        raise RuntimeError(message.value.decode('utf-8','replace') or f'Vulkan locator init status {status}')
    _CONTEXT, _DLL, _KEY = ctx, lib, key
    _warn(message.value.decode('utf-8', 'replace'))
    return lib, ctx


def _cleanup():
    global _CONTEXT, _DLL, _KEY
    with _LOCK:
        if _CONTEXT is not None:
            try: _DLL.nk_ground_vk_locator_release(_CONTEXT)
            except Exception: pass
            _CONTEXT = _DLL = _KEY = None


atexit.register(_cleanup)


def build_index(tri, grid_size=_GRID_SIZE):
    """Deterministic conservative CSR index ordered by original simplex ID.

    Data layout never reorders simplices or query points. AABB of every
    triangle is padded before integer bin rounding, and entries are
    stable-sorted by grid cell, then original SciPy simplex ID.
    """
    if grid_size != 64:
        raise ValueError('Vulkan shader requires exactly 64x64 bins')
    pts = np.asarray(tri.points)
    faces = np.asarray(tri.simplices)
    tr = np.asarray(tri.transform)
    if (pts.ndim != 2 or pts.shape[1] != 2 or pts.dtype != np.float64
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype != np.int32
            or tr.shape != (len(faces), 3, 2) or tr.dtype != np.float64
            or len(faces) <= 0 or len(faces) > _MAX_TRI
            or not np.isfinite(pts).all() or not np.isfinite(tr).all()
            or np.any(faces < 0) or np.any(faces >= len(pts))):
        raise ValueError('Unsupported or degenerate Delaunay input; use CPU reference')
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    ext = hi - lo
    if np.any(ext <= 1e-12) or not np.isfinite(ext).all():
        raise ValueError('Degenerate Delaunay extent')
    box = pts[faces]
    # Wide enough to include rounding uncertainty; interior certification is
    # performed separately in the shader, CPU handles all other queries.
    pad = np.maximum(ext * 1e-11, 1e-9)
    left = np.clip(np.floor((box.min(axis=1) - pad - lo) * (grid_size / ext)), 0, grid_size - 1).astype(np.int64)
    right = np.clip(np.floor((box.max(axis=1) + pad - lo) * (grid_size / ext)), 0, grid_size - 1).astype(np.int64)
    w = right[:, 0] - left[:, 0] + 1
    h = right[:, 1] - left[:, 1] + 1
    counts = w * h
    total = int(np.sum(counts, dtype=np.int64))
    if total <= 0 or total > _MAX_ENTRIES:
        raise ValueError(f'Grid too large ({total} links); use CPU reference')
    rep = np.repeat(np.arange(len(faces), dtype=np.uint32), counts)
    bases = np.repeat(np.cumsum(counts, dtype=np.int64) - counts, counts)
    delta = np.arange(total, dtype=np.int64) - bases
    cells = ((left[rep, 1] + delta // w[rep]) * grid_size
             + left[rep, 0] + delta % w[rep]).astype(np.int32)
    order = np.argsort(cells, kind='stable')
    sorted_cells = cells[order]
    entries = np.ascontiguousarray(rep[order], dtype=np.uint32)
    offsets = np.searchsorted(sorted_cells, np.arange(grid_size**2 + 1), side='left').astype(np.uint32)
    return (np.ascontiguousarray(tr, dtype=np.float64),
            np.ascontiguousarray(offsets), entries,
            float(lo[0]), float(lo[1]), float(grid_size/ext[0]), float(grid_size/ext[1]))


def _gpu_propose(tri, xy):
    started = perf_counter()
    tr, offsets, entries, lo_x, lo_y, inv_x, inv_y = build_index(tri)
    index_seconds = perf_counter() - started
    out = np.full(len(xy), -2, dtype=np.int32)
    with _LOCK:
        lib, ctx = _context()
        started = perf_counter()
        message = ctypes.create_string_buffer(512)
        d = ctypes.POINTER(ctypes.c_double)
        u = ctypes.POINTER(ctypes.c_uint32)
        i = ctypes.POINTER(ctypes.c_int32)
        # Native DLL performs bounded dispatches of <= 32768 queries; no
        # point-cloud buffer can exceed T400 VRAM capacity.
        status = lib.nk_ground_vk_locator_locate(
            ctx, tr.ctypes.data_as(d), len(tr), offsets.ctypes.data_as(u), len(offsets),
            entries.ctypes.data_as(u), len(entries), xy.ctypes.data_as(d), len(xy),
            lo_x, lo_y, inv_x, inv_y, out.ctypes.data_as(i), message, len(message))
        if status:
            raise RuntimeError('GPU locator dispatch failed: ' + message.value.decode('utf-8','replace'))
        seconds = perf_counter() - started
    if np.any((out < -2) | (out >= len(tr))):
        raise ValueError('Vulkan returned invalid simplex IDs')
    return out, index_seconds, seconds


def locate(tri, xy, cpu_reference, stats=None):
    """Exact-result GPU proposal; CPU callback remains final authority.

    Returns None when acceleration is disabled/inapplicable. In full-audit
    mode every ID is checked. Experimental fast mode (explicit double opt-in)
    only trusts conservatively certified interior proposals and obtains CPU
    IDs for all uncertain points; do not enable on production LAZ/LNT.
    """
    if not _enabled():
        return None
    query = np.asarray(xy)
    if (query.ndim != 2 or query.shape[1] != 2 or query.dtype != np.float64
            or len(query) < _MIN_QUERIES or not np.isfinite(query).all()):
        return None
    query = np.ascontiguousarray(query)
    try:
        proposals, build_sec, gpu_sec = _gpu_propose(tri, query)
        certified = proposals >= 0
        audit = _on('NAKSHA_GROUND_VK_LOCATOR_AUDIT', '1')
        unsafe_fast = _on('NAKSHA_GROUND_VK_LOCATOR_EXPERIMENTAL_FAST')
        if audit or not unsafe_fast:
            authoritative = np.asarray(cpu_reference(query), dtype=np.int32)
            if not np.array_equal(proposals[certified], authoritative[certified]):
                _STATS['mismatches'] += int(np.count_nonzero(proposals[certified] != authoritative[certified]))
                _warn('PROPOSAL MISMATCH: complete CPU result restored; stop GPU validation')
                return authoritative
            _STATS['audited'] += len(query)
            result = authoritative  # complete CPU reference, always exact
        else:
            result = proposals.copy()
            if np.any(~certified):
                result[~certified] = np.asarray(cpu_reference(query[~certified]), dtype=np.int32)
        _STATS['calls'] += 1
        _STATS['queries'] += len(query)
        _STATS['fast_proposals'] += int(np.count_nonzero(certified))
        _STATS['cpu_fallback'] += int(np.count_nonzero(~certified))
        _STATS['gpu_seconds'] += gpu_sec
        _STATS['index_seconds'] += build_sec
        if _STATS['calls'] == 1 or _STATS['calls'] % 10 == 0:
            print('[Ground Vulkan Locator] calls={} queries={} certified={} '
                  'CPU_uncertain={} full_CPU_audit={} grid={:.3f}s Vulkan={:.3f}s '
                  'Delaunay_topology=CPU_SciPy'.format(
                      _STATS['calls'], _STATS['queries'], _STATS['fast_proposals'],
                      _STATS['cpu_fallback'], (audit or not unsafe_fast),
                      _STATS['index_seconds'], _STATS['gpu_seconds']), flush=True)
        if stats is not None:
            stats['vk_locator_queries'] = stats.get('vk_locator_queries', 0) + len(query)
            stats['vk_locator_fast_proposals'] = stats.get('vk_locator_fast_proposals', 0) + int(np.count_nonzero(certified))
            stats['vk_locator_cpu_fallback'] = stats.get('vk_locator_cpu_fallback', 0) + int(np.count_nonzero(~certified))
            stats['vk_locator_index_seconds'] = stats.get('vk_locator_index_seconds', 0.) + build_sec
            stats['vk_locator_gpu_seconds'] = stats.get('vk_locator_gpu_seconds', 0.) + gpu_sec
            stats['vk_locator_audited'] = stats.get('vk_locator_audited', 0) + (len(query) if audit or not unsafe_fast else 0)
        return result
    except (RuntimeError, ValueError, FileNotFoundError, OSError, OverflowError) as exc:
        _warn(f'{type(exc).__name__}: {str(exc)[:140]}; CPU fallback')
        return None


def diagnostics():
    return _STATS.copy()
