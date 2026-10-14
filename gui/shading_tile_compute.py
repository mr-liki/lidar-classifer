"""Process-safe CPU kernels for large-cloud shading tiles.

There are deliberately no Qt or VTK imports here. Windows worker processes
can import this module cheaply while all renderer changes remain on the GUI
process.
"""
from __future__ import annotations

import numpy as np


def compute_owned_tin(local_points, bounds, offset):
    """Triangulate one halo tile and retain faces owned by its core bounds."""
    import triangle as tr

    local = np.asarray(local_points, dtype=np.float64)
    if len(local) < 3:
        empty_i = np.empty((0, 3), dtype=np.int32)
        return empty_i, np.empty((0, 3), dtype=np.float32)
    result = tr.triangulate({"vertices": local[:, :2]}, "Qz")
    faces = np.asarray(
        result.get("triangles", np.empty((0, 3))), dtype=np.int32,
    ).reshape(-1, 3)
    if len(faces) == 0:
        return faces, np.empty((0, 3), dtype=np.float32)

    b = tuple(float(v) for v in bounds)
    origin = np.asarray(offset, dtype=np.float64)
    centroids = (
        local[faces[:, 0], :2] + local[faces[:, 1], :2]
        + local[faces[:, 2], :2]
    ) / 3.0
    centroids += origin[:2]
    own = (
        (centroids[:, 0] >= b[0]) & (centroids[:, 0] <= b[1])
        & (centroids[:, 1] >= b[2]) & (centroids[:, 1] <= b[3])
    )
    faces = np.ascontiguousarray(faces[own], dtype=np.int32)
    if len(faces) == 0:
        return faces, np.empty((0, 3), dtype=np.float32)

    p0 = local[faces[:, 0]]
    p1 = local[faces[:, 1]]
    p2 = local[faces[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-10)
    flip = (normals[:, 2] < 0) & (np.abs(normals[:, 2]) > 0.3)
    normals[flip] *= -1.0
    return faces, np.ascontiguousarray(normals, dtype=np.float32)
