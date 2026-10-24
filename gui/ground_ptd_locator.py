"""Quarantined legacy locator; always executes the COMPLETE SciPy query.

The old nearest-vertex/subset fallback failed adversarial simplex/class parity.
The compatibility function remains available, but even explicit legacy opt-in
cannot reactivate that algorithm. Private helpers below are inactive.
"""
from __future__ import annotations

import os
import numpy as np
from scipy.spatial import cKDTree

try:
    from numba import njit
except ImportError:
    njit = None

if njit is not None:
    @njit(cache=True, nogil=True)
    def _vertex_to_simplex_csr(simplices, n_vertices):
        offsets = np.zeros(n_vertices + 1, dtype=np.int64)
        for i in range(len(simplices)):
            offsets[simplices[i, 0] + 1] += 1
            offsets[simplices[i, 1] + 1] += 1
            offsets[simplices[i, 2] + 1] += 1
        for i in range(n_vertices):
            offsets[i + 1] += offsets[i]
        write = offsets[:-1].copy()
        neighbors = np.empty(3 * len(simplices), dtype=np.int32)
        for i in range(len(simplices)):
            for j in range(3):
                v = simplices[i, j]
                neighbors[write[v]] = i
                write[v] += 1
        return offsets, neighbors

    @njit(cache=True, nogil=True)
    def _confident_interior(xy, nearest, transform, offsets, neighbors, guard):
        located = np.full(len(xy), -2, dtype=np.int32)
        for i in range(len(xy)):
            v = nearest[i]
            if v < 0 or v >= len(offsets) - 1:
                continue
            x0 = xy[i, 0]
            x1 = xy[i, 1]
            for j in range(offsets[v], offsets[v + 1]):
                sid = neighbors[j]
                # SciPy stores barycentric affine transform: first two rows
                # map point-minus-origin, third barycentric is 1 - sum.
                dx = x0 - transform[sid, 2, 0]
                dy = x1 - transform[sid, 2, 1]
                b0 = transform[sid, 0, 0] * dx + transform[sid, 0, 1] * dy
                b1 = transform[sid, 1, 0] * dx + transform[sid, 1, 1] * dy
                b2 = 1.0 - b0 - b1
                if b0 > guard and b1 > guard and b2 > guard:
                    located[i] = sid
                    break
        return located
else:
    _vertex_to_simplex_csr = None
    _confident_interior = None


def find_simplex_exact(triangulation, query_xy, *, stats=None):
    """Quarantined compatibility entry point: ONLY the complete SciPy query.

    The previous nearest-vertex/subset algorithm failed adversarial parity.
    Even explicit legacy opt-in must not reactivate it. Retain its private
    helpers for source compatibility, but never use them for classification.
    """
    result = triangulation.find_simplex(query_xy)
    if stats is not None:
        stats.update(resolved=0, fallback=int(result.size), total=int(result.size),
                     disabled_reason='legacy subset locator failed exact parity')
    return result
