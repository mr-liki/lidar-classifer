"""Bounded-memory spatial prefilter for Parallel's main-view LiDAR band.

Vulkan-only optimization. The legacy VTK mask generator is intentionally
unchanged and remains the fallback for small clouds. Uses float64 throughout
and includes all candidates that could satisfy the original segment test.
"""
from __future__ import annotations

import math
import numpy as np


def native_parallel_band_mask_xy(xyz, polyline_xy, left_tol, right_tol):
    """Same finite-segment signed XY band as ParallelToolDialog's VTK path.

    Avoids allocating N x 2 float64 temporary arrays for each line segment.
    Bounding-box prefilter is conservative (never clips a valid band match),
    and only the relevant candidates enter the exact dot/cross calculation.
    The returned mask still indexes the *original* point-cloud order.
    """
    if xyz is None or polyline_xy is None:
        return np.zeros(0, dtype=bool)
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    line = np.asarray(polyline_xy, dtype=np.float64)
    mask = np.zeros(len(points), dtype=bool)
    if line.ndim != 2 or line.shape[0] < 2:
        return mask
    left_tol, right_tol = max(0., float(left_tol)), max(0., float(right_tol))
    halo = max(left_tol, right_tol)
    px, py = points[:, 0], points[:, 1]
    for first, last in zip(line[:-1], line[1:]):
        a, b = first[:2], last[:2]
        v = b-a
        seg_len_sq = float(v @ v)
        if seg_len_sq <= 1e-12:
            continue
        seg_len = math.sqrt(seg_len_sq)
        # All valid projections lie between the endpoints and at most `halo`
        # from the segment. Include a small floating-point margin at large
        # UTM coordinates to avoid an accidental false negative at the edge.
        margin = max(1e-9, 8.*np.finfo(np.float64).eps *
                     max(1., float(np.max(np.abs(a))), float(np.max(np.abs(b)))))
        indexes = np.flatnonzero(
            (~mask) &
            (px >= min(a[0], b[0])-halo-margin) &
            (px <= max(a[0], b[0])+halo+margin) &
            (py >= min(a[1], b[1])-halo-margin) &
            (py <= max(a[1], b[1])+halo+margin)
        )
        if indexes.size == 0:
            continue
        rel = points[indexes, :2]-a
        projection = (rel @ v)/seg_len_sq
        signed = (v[0]*rel[:, 1]-v[1]*rel[:, 0])/seg_len
        matching = ((projection >= 0.) & (projection <= 1.) &
                    (signed >= -right_tol) & (signed <= left_tol))
        mask[indexes[matching]] = True
    return mask
