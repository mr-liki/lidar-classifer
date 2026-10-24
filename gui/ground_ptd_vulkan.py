"""Opt-in REAL Vulkan FP64 compute distance gate for Ground PTD.

The DLL creates a headless compute-only Vulkan context independent of the
existing Naksha Vulkan renderer. No VTK interaction and no LNT writes.

Inputs/outputs are float64 candidate planes and uint32 keep flags. Every
proposed GPU rejection is checked by the authoritative CPU predicate in
classify_ground_ptd() before points can be skipped. CPU retains all final
classification, point ordering, angle tests and per-simplex decisions.

NAKSHA_GROUND_GPU=1                enable GPU in Ground settings (or explicitly
                                   NAKSHA_GROUND_VULKAN_COMPUTE=1)
NAKSHA_GROUND_VULKAN_COMPUTE=1     prefer Vulkan rather than CUDA (default 1)
NAKSHA_GROUND_VK_LIBRARY=...       optional exact DLL path
NAKSHA_GROUND_VK_SHADER=...        optional exact .spv path
NAKSHA_GROUND_VK_BATCH=32768      upper bounded by native 32768
"""
from __future__ import annotations

import atexit
import ctypes
import math
import os
from pathlib import Path
from threading import RLock
from time import perf_counter

import numpy as np

from gui.ground_ptd_gpu import _rounding_guard

_TRUE = {'1', 'true', 'yes', 'on'}
_FALSE = {'0', 'false', 'no', 'off'}
_LOCK = RLock()
_CONTEXT = None
_DLL = None
_CONTEXT_KEY = None
_ERRORS = set()


def _warn_once(message):
    if message not in _ERRORS:
        _ERRORS.add(message)
        print('[Ground Vulkan Compute] ' + message, flush=True)


def _requested():
    return os.environ.get('NAKSHA_GROUND_VULKAN_COMPUTE', '1').strip().lower() not in _FALSE


def _default_dll():
    root = Path(__file__).resolve().parent.parent
    filename = 'naksha_ground_vulkan_compute.dll' if os.name == 'nt' else 'naksha_ground_vulkan_compute.so'
    return root / 'native' / 'ground_vulkan_compute' / 'build_vulkan' / ('Release' if os.name == 'nt' else '') / filename


def _paths():
    dll_path = Path(os.environ.get('NAKSHA_GROUND_VK_LIBRARY') or _default_dll()).expanduser().resolve()
    shader_path = Path(os.environ.get('NAKSHA_GROUND_VK_SHADER') or (dll_path.parent / 'ground_distance.comp.spv')).expanduser().resolve()
    return dll_path, shader_path


def _bind(dll):
    dll.nk_ground_vk_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int]
    dll.nk_ground_vk_create.restype = ctypes.c_int
    dll.nk_ground_vk_filter.argtypes = [ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_uint32, ctypes.c_double,
        ctypes.c_double, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p, ctypes.c_int]
    dll.nk_ground_vk_filter.restype = ctypes.c_int
    dll.nk_ground_vk_release.argtypes = [ctypes.c_void_p]
    dll.nk_ground_vk_release.restype = None
    dll.nk_ground_vk_max_batch.argtypes = []
    dll.nk_ground_vk_max_batch.restype = ctypes.c_uint32


def _get_context():
    global _CONTEXT, _DLL, _CONTEXT_KEY
    dll_path, shader_path = _paths()
    key = (str(dll_path), str(shader_path))
    if _CONTEXT is not None:
        if key != _CONTEXT_KEY:
            raise RuntimeError('Vulkan compute DLL/shader changed during active session; restart Naksha')
        return _DLL, _CONTEXT
    if not dll_path.is_file() or not shader_path.is_file():
        raise FileNotFoundError('Ground Vulkan compute DLL or shader missing; use CPU fallback')
    dll = ctypes.CDLL(str(dll_path))
    _bind(dll)
    ctx = ctypes.c_void_p()
    message = ctypes.create_string_buffer(512)
    err = dll.nk_ground_vk_create(os.fsencode(str(shader_path)), ctypes.byref(ctx), message, len(message))
    if err != 0 or not ctx.value:
        raise RuntimeError(message.value.decode('utf-8', errors='replace') or 'Vulkan compute init failed')
    _DLL = dll
    _CONTEXT = ctx
    _CONTEXT_KEY = key
    print('[Ground Vulkan Compute] ' + message.value.decode('utf-8', errors='replace'), flush=True)
    return dll, ctx


def readiness():
    """Actual Vulkan compute dispatch readiness, not Vulkan-renderer detection."""
    if not _requested():
        return False, 'Ground Vulkan compute disabled explicitly.'
    try:
        with _LOCK:
            dll, context = _get_context()
            if not 0 < dll.nk_ground_vk_max_batch() <= 65536:
                return False, 'Ground Vulkan compute DLL has invalid batch limit.'
            # Use an actual dispatch; a device name alone is not readiness.
            simple = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1.]], np.float64)
            results = np.empty(1, np.uint32)
            msg = ctypes.create_string_buffer(512)
            rc = dll.nk_ground_vk_filter(context, simple.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                 1, .5, 0., 0, results.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), msg, len(msg))
            if rc or results[0] != 1:
                return False, 'Ground Vulkan smoke dispatch failed: ' + msg.value.decode('utf-8', errors='replace')
            return True, 'Vulkan FP64 compute ready and verified by GPU dispatch.'
    except Exception as exc:
        return False, 'Vulkan Ground compute unavailable: ' + str(exc)[:180]


def conservative_vulkan_prefilter(pts, anchors, normals, max_distance, *,
                                  add_only_upward=False, coordinate_scale=None):
    """Return necessary-condition keep mask or None, preserving CPU fallback.

    Preconditions intentionally match the earlier CUDA gate. This function
    never changes classes, candidates, source geometry or original point IDs.
    Any compute failure returns None so the CPU performs its original work.
    """
    if not _requested():
        return None
    pts = np.asarray(pts)
    anchors = np.asarray(anchors)
    normals = np.asarray(normals)
    if (pts.ndim != 2 or pts.shape[1] != 3 or pts.shape != anchors.shape or
            pts.shape != normals.shape or pts.dtype != np.float64 or
            anchors.dtype != np.float64 or normals.dtype != np.float64 or
            len(pts) < 4096 or not math.isfinite(float(max_distance)) or max_distance < 0):
        return None
    try:
        if coordinate_scale is None:
            if not (np.isfinite(pts).all() and np.isfinite(anchors).all() and np.isfinite(normals).all()):
                # Device must never decide about nonfinite data.
                return None
            scale = max(float(np.max(np.abs(pts))), float(np.max(np.abs(anchors))),
                        float(np.max(np.abs(normals))))
        else:
            scale = float(coordinate_scale)
        guard = _rounding_guard(scale, max_distance)
        if not math.isfinite(guard) or guard > max(1.0, float(max_distance)):
            return None
        requested_batch = int(os.environ.get('NAKSHA_GROUND_VK_BATCH', '32768'))
        batch_size = max(4096, min(32768, requested_batch))
    except (TypeError, ValueError, OverflowError):
        return None
    started = perf_counter()
    try:
        with _LOCK:
            dll, ctx = _get_context()
            batch_size = min(batch_size, int(dll.nk_ground_vk_max_batch()))
            if batch_size < 1:
                return None
            keep = np.empty(len(pts), dtype=bool)
            for start in range(0, len(pts), batch_size):
                stop = min(len(pts), start + batch_size)
                # Explicit fp64 layout: xyz, triangle anchor xyz, normal xyz.
                packed = np.empty((stop - start, 9), dtype=np.float64)
                packed[:, 0:3] = pts[start:stop]
                packed[:, 3:6] = anchors[start:stop]
                packed[:, 6:9] = normals[start:stop]
                flags = np.empty(stop - start, dtype=np.uint32)
                message = ctypes.create_string_buffer(512)
                rc = dll.nk_ground_vk_filter(ctx,
                    packed.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    len(packed), float(max_distance), float(guard),
                    int(bool(add_only_upward)),
                    flags.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                    message, len(message))
                if rc or not np.isin(flags, [0, 1]).all():
                    raise RuntimeError('Vulkan dispatch failed: ' + message.value.decode('utf-8', errors='replace'))
                keep[start:stop] = flags.astype(bool)
    except Exception as exc:
        _warn_once(type(exc).__name__ + ': ' + str(exc)[:160] + '; CPU fallback')
        return None
    return {'keep': keep, 'candidates': int(len(keep)),
            'rejected': int(np.count_nonzero(~keep)),
            'seconds': perf_counter() - started, 'margin': guard,
            'backend': 'vulkan_fp64_compute_distance_gate'}


@atexit.register
def _release():
    global _CONTEXT, _DLL, _CONTEXT_KEY
    with _LOCK:
        if _CONTEXT is not None and _DLL is not None:
            try:
                _DLL.nk_ground_vk_release(_CONTEXT)
            except Exception:
                pass
        _CONTEXT = None
        _DLL = None
        _CONTEXT_KEY = None
