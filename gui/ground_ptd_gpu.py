"""Opt-in, conservative CUDA distance prefilter for Naksha Ground PTD.

This module NEVER decides which points become Ground. It only attempts to
exclude candidates that *cannot* pass the existing CPU point-to-TIN-plane
absolute-distance/upward rules. The original CPU rules (including exact
NumPy distances, angle and per-simplex ranking) run on EVERY survivor.

No data are written here. On any CUDA/precision failure the caller receives
None and must execute its unchanged CPU path.

Environment:
  NAKSHA_GROUND_GPU=1           opt in (default: disabled)
  NAKSHA_GROUND_GPU_BATCH=32768  CUDA candidate batch (4096..65536)

A CUDA-enabled PyTorch install is optional. VTK/Vulkan render acceleration is
unrelated to CUDA support, and the NVIDIA T400's FP64 throughput may limit the
benefit of this exactness-first pilot.
"""
from __future__ import annotations

import math
import os
import threading
from time import perf_counter

import numpy as np

_LOCK = threading.RLock()  # one CUDA PTD evaluation at a time on the 4GB T400
_WARNED = set()


def _enabled():
    return os.environ.get('NAKSHA_GROUND_GPU', '0').strip().lower() in (
        '1', 'true', 'yes', 'on'
    )


def _warn_once(msg):
    if msg not in _WARNED:
        _WARNED.add(msg)
        print('[Ground GPU] ' + msg, flush=True)


def _cuda_torch():
    if not _enabled():
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            _warn_once('CUDA-enabled PyTorch unavailable; original CPU Ground path preserved.')
            return None
        return torch
    except Exception as exc:
        _warn_once(f'PyTorch CUDA probe failed: {type(exc).__name__}; using CPU.')
        return None


def _rounding_guard(coordinate_scale, distance_threshold):
    """Overestimate double-rounding effects in a 3-term plane dot product.

    A generous factor (8192 ulps) covers CPU/GPU accumulation order, coordinate
    subtraction and GPU FMA differences for ordinary finite LiDAR coordinates.
    If the implied guard is too large or invalid, reject no points (caller CPU).
    This is deliberately not a fast FP32 approximation.
    """
    scale = max(1.0, float(coordinate_scale), abs(float(distance_threshold)))
    if not math.isfinite(scale):
        return math.inf
    return 8192.0 * np.finfo(np.float64).eps * scale


def _mask_from_double_distances(distance, max_distance, upward, guard):
    """Only reject unequivocal failures; nonfinite values stay on CPU."""
    dist = np.asarray(distance, dtype=np.float64)
    keep = ~np.isfinite(dist)
    keep |= np.abs(dist) <= float(max_distance) + float(guard)
    if upward:
        keep &= ~(np.isfinite(dist) & (dist < -float(guard)))
    return keep


def _torch_batched_mask(pts, anchors, normals, max_distance, upward,
                        guard, torch, device, batch_size):
    """Device-agnostic tensor loop; CPU torch device is used in unit tests only."""
    keep = np.empty(len(pts), dtype=bool)
    with torch.no_grad():
        for start in range(0, len(pts), batch_size):
            stop = min(start + batch_size, len(pts))
            # Preserve float64. Never float32 or quantize global coordinates.
            p = torch.as_tensor(np.ascontiguousarray(pts[start:stop], dtype=np.float64),
                                dtype=torch.float64, device=device)
            a = torch.as_tensor(np.ascontiguousarray(anchors[start:stop], dtype=np.float64),
                                dtype=torch.float64, device=device)
            n = torch.as_tensor(np.ascontiguousarray(normals[start:stop], dtype=np.float64),
                                dtype=torch.float64, device=device)
            d = ((p - a) * n).sum(dim=1)
            keep_tensor = (~torch.isfinite(d)) | (torch.abs(d) <= max_distance + guard)
            if upward:
                keep_tensor &= ~(torch.isfinite(d) & (d < -guard))
            keep[start:stop] = keep_tensor.to(device='cpu').numpy()
            del p, a, n, d, keep_tensor
    return keep


def conservative_cuda_prefilter(pts, anchors, normals, max_distance, *,
                                add_only_upward=False, coordinate_scale=None):
    """Return {'keep': bool[N], ...} or None to run the unchanged CPU path.

    Exact CPU classification always runs on all kept rows. For this predicate,
    GPU floating-point errors would have to exceed 8192 ulps of the tile's
    coordinate scale to create a false negative. The optional CPU shadow audit
    should be used to verify a new CUDA/driver/device configuration.
    """
    torch = _cuda_torch()
    if torch is None:
        return None
    coords = np.asarray(pts)
    bases = np.asarray(anchors)
    norm = np.asarray(normals)
    if (coords.ndim != 2 or coords.shape[1] != 3 or
        coords.shape != bases.shape or norm.shape != coords.shape or
        coords.dtype != np.float64 or bases.dtype != np.float64 or
        norm.dtype != np.float64 or len(coords) < 4096 or
        not math.isfinite(float(max_distance)) or max_distance < 0):
        return None

    # Evaluate a fixed worst-case coordinate bound for this working tile.
    # Always retain candidates with nonfinite inputs, letting the CPU decide.
    try:
        scale = (max(float(np.max(np.abs(coords))),
                     float(np.max(np.abs(bases))),
                     float(np.max(np.abs(norm))))
                 if coordinate_scale is None else float(coordinate_scale))
        guard = _rounding_guard(scale, max_distance)
        if not math.isfinite(guard) or guard > max(1.0, float(max_distance)):
            return None
        requested = int(os.environ.get('NAKSHA_GROUND_GPU_BATCH', '32768'))
        batch_size = min(max(requested, 4096), 65536)
    except (ValueError, OverflowError, TypeError):
        return None

    try:
        with _LOCK:
            device = torch.device('cuda:0')
            start = perf_counter()
            keep = _torch_batched_mask(coords, bases, norm, float(max_distance),
                                       bool(add_only_upward), float(guard),
                                       torch, device, batch_size)
            torch.cuda.synchronize(device)
            elapsed = perf_counter() - start
    except Exception as exc:
        _warn_once(f'CUDA evaluation failed ({type(exc).__name__}); using CPU for this batch.')
        return None

    return {
        'keep': keep,
        'candidates': int(len(coords)),
        'rejected': int(np.count_nonzero(~keep)),
        'seconds': float(elapsed),
        'margin': float(guard),
        'backend': 'cuda_torch_fp64_distance_gate',
    }
