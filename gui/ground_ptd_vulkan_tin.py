"""Opt-in real Vulkan GPU TIN *facet geometry* with byte-identical CPU audit.

This does NOT compute Delaunay connectivity. Qhull/SciPy still builds the
point-to-triangle topology. After locate, this shader computes triangle normals
and max edge length for the exact simplices chosen by SciPy. The original NumPy
formula is run in full, and any GPU facet with different FP64 bits is restored
from CPU before it can influence ground classification. A failure never changes
LNT, point IDs, topology, or classification decisions.

Only NAKSHA_GROUND_VK_TIN=1 opts in; safe default is CPU.
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
_CONTEXT_KEY = None
_WARNED = set()
_STATS = {"calls": 0, "facets": 0, "gpu_exact_facets": 0,
          "fallback_facets": 0, "gpu_seconds": 0.0, "cpu_seconds": 0.0}


def _enabled():
    return os.environ.get('NAKSHA_GROUND_VK_TIN', '0').strip().lower() in ('1','true','yes','on')


def _warn_once(message):
    if message not in _WARNED:
        _WARNED.add(message)
        print('[Ground Vulkan TIN] ' + message, flush=True)


def _paths():
    root = Path(__file__).resolve().parents[1]
    dllname = 'naksha_ground_vulkan_tin.dll' if os.name == 'nt' else 'naksha_ground_vulkan_tin.so'
    default_folder = root / 'native' / 'ground_vulkan_tin' / 'build_vulkan_tin'
    if os.name == 'nt':
        default_folder /= 'Release'
    dll = Path(os.environ.get('NAKSHA_GROUND_VK_TIN_LIBRARY') or default_folder / dllname).expanduser().resolve()
    shader = Path(os.environ.get('NAKSHA_GROUND_VK_TIN_SHADER') or dll.parent / 'tin_facets.comp.spv').expanduser().resolve()
    return dll, shader


def _bind(dll):
    dll.nk_ground_vk_tin_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int]
    dll.nk_ground_vk_tin_create.restype = ctypes.c_int
    dll.nk_ground_vk_tin_build.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_double), ctypes.c_void_p, ctypes.c_int]
    dll.nk_ground_vk_tin_build.restype = ctypes.c_int
    dll.nk_ground_vk_tin_release.argtypes = [ctypes.c_void_p]
    dll.nk_ground_vk_tin_release.restype = None
    dll.nk_ground_vk_tin_max_batch.argtypes = []
    dll.nk_ground_vk_tin_max_batch.restype = ctypes.c_uint32


def _get_context():
    global _CONTEXT, _DLL, _CONTEXT_KEY
    dll_path, shader_path = _paths()
    key = str(dll_path), str(shader_path)
    if _CONTEXT is not None:
        if key != _CONTEXT_KEY:
            raise RuntimeError('TIN shader/DLL paths changed in active process; restart Naksha')
        return _DLL, _CONTEXT
    if not dll_path.is_file() or not shader_path.is_file():
        raise FileNotFoundError('Vulkan TIN DLL/shader missing; use CPU')
    dll = ctypes.CDLL(str(dll_path))
    _bind(dll)
    message = ctypes.create_string_buffer(512)
    context = ctypes.c_void_p()
    rc = dll.nk_ground_vk_tin_create(os.fsencode(str(shader_path)), ctypes.byref(context),
                                      message, len(message))
    if rc != 0 or not context.value:
        raise RuntimeError(message.value.decode('utf-8','replace') or 'Vulkan TIN init failed')
    _DLL, _CONTEXT, _CONTEXT_KEY = dll, context, key
    print('[Ground Vulkan TIN] ' + message.value.decode('utf-8','replace'), flush=True)
    return dll, context


def cpu_facets(a, b, c):
    """Keep the original exact NumPy evaluation order and NaN behavior."""
    me = np.maximum(
        np.maximum(np.linalg.norm(b-a, axis=1), np.linalg.norm(c-b, axis=1)),
        np.linalg.norm(a-c, axis=1))
    nor = np.cross(b-a, c-a)
    nl = np.linalg.norm(nor, axis=1, keepdims=True)
    good = nl[:, 0] > 1e-12
    nor = nor / np.where(nl > 0, nl, 1.0)
    nor[nor[:, 2] < 0] *= -1
    return me, nor, good, nl[:, 0]


def _gpu_facets(a, b, c):
    if (a.dtype != np.float64 or b.dtype != np.float64 or c.dtype != np.float64 or
            a.shape != b.shape or a.shape != c.shape or a.ndim != 2 or a.shape[1] != 3 or
            not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all())):
        return None
    with _LOCK:
        dll, ctx = _get_context()
        capacity = int(dll.nk_ground_vk_tin_max_batch())
        if capacity < 1 or capacity > 32768:
            raise RuntimeError('Invalid Vulkan TIN max batch')
        results = np.empty((len(a),5), dtype=np.float64)
        for start in range(0,len(a),capacity):
            stop = min(start+capacity,len(a))
            packed = np.empty((stop-start,9), dtype=np.float64)
            packed[:,0:3], packed[:,3:6], packed[:,6:9] = a[start:stop], b[start:stop], c[start:stop]
            msg = ctypes.create_string_buffer(512)
            segment = np.empty((stop-start,5), dtype=np.float64)
            rc = dll.nk_ground_vk_tin_build(ctx,
                packed.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(packed),
                segment.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), msg, len(msg))
            if rc != 0 or not np.isfinite(segment).all():
                raise RuntimeError('Vulkan TIN dispatch invalid: ' + msg.value.decode('utf-8','replace'))
            results[start:stop] = segment
        return results


def exact_facet_geometry(a, b, c):
    """Return (max_edge, unit_normal, nondegenerate) with exact CPU parity.

    Vulkan can build candidate triangle planes. Any row with even one differing
    IEEE-754 bit is replaced by the authoritative CPU row. This includes tiny
    FP64 reductions and signed zero changes; no epsilon-based acceptance.
    """
    if not _enabled() or len(a) == 0:
        me, nor, good, _ = cpu_facets(a,b,c)
        return me, nor, good
    started_cpu = perf_counter()
    me, nor, good, norm = cpu_facets(a,b,c)
    elapsed_cpu = perf_counter()-started_cpu
    try:
        started_gpu = perf_counter()
        dev = _gpu_facets(a,b,c)
        elapsed_gpu = perf_counter()-started_gpu
        if dev is None or dev.shape != (len(me),5):
            raise RuntimeError('Invalid Vulkan TIN output layout')
        reference = np.empty((len(me),5), dtype=np.float64)
        reference[:,:3] = nor
        reference[:,3] = me
        reference[:,4] = norm
        equal = np.all(dev.view(np.uint64) == reference.view(np.uint64), axis=1)
        # The check above is mandatory even in production: changing a plane
        # normal can change PTD angle, slope, distance, or per-facet ranking.
        ngood = int(np.count_nonzero(equal))
        if ngood:
            nor[equal] = dev[equal,:3]
            me[equal] = dev[equal,3]
        with _LOCK:
            _STATS['calls'] += 1
            _STATS['facets'] += len(a)
            _STATS['gpu_exact_facets'] += ngood
            _STATS['fallback_facets'] += len(a)-ngood
            _STATS['gpu_seconds'] += elapsed_gpu
            _STATS['cpu_seconds'] += elapsed_cpu
            calls = _STATS['calls']
        if calls == 1 or calls % 25 == 0:
            print('[Ground Vulkan TIN] facet_GPU=YES calls={} facets={} exact={} CPU_fallback={} '
                  'gpu_dispatch_transfer={:.3f}s cpu_reference={:.3f}s '
                  'Delaunay_topology=CPU_SciPy'.format(
                       calls,_STATS['facets'],_STATS['gpu_exact_facets'],
                       _STATS['fallback_facets'],_STATS['gpu_seconds'],_STATS['cpu_seconds']),flush=True)
    except Exception as exc:
        _warn_once(type(exc).__name__ + ': '+ str(exc)[:180] + '; CPU TIN facet fallback')
    return me, nor, good


def readiness():
    """Perform actual TIN shader dispatch; not a rendering backend check."""
    try:
        a = np.array([[0.,0.,0.],[10.,20.,4.]],np.float64)
        b = np.array([[1.,0.,0.],[12.,20.,4.]],np.float64)
        c = np.array([[0.,1.,0.],[10.,21.,4.]],np.float64)
        got = _gpu_facets(a,b,c)
        maxedges, nor, _, n = cpu_facets(a,b,c)
        expected = np.column_stack((nor,maxedges,n))
        if got is None or got.shape != expected.shape or not np.allclose(got,expected,atol=1e-12,rtol=1e-12):
            return False,'Vulkan TIN shader hardware result mismatch'
        return True, 'Actual Vulkan TIN facet dispatch passed'
    except Exception as exc:
        return False, type(exc).__name__+': '+str(exc)[:180]


def stats():
    with _LOCK:
        return dict(_STATS)


@atexit.register
def _release():
    global _CONTEXT,_DLL,_CONTEXT_KEY
    with _LOCK:
        if _CONTEXT is not None and _DLL is not None:
            try:
                _DLL.nk_ground_vk_tin_release(_CONTEXT)
            except Exception:
                pass
        _CONTEXT=_DLL=_CONTEXT_KEY=None
