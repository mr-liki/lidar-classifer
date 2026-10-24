"""Opt-in exact-operator PTD early rejection (no class writes).

Distance/upward/slope/edge gates precede the expensive triangle-vertex angle
calculation in the existing algorithm. The gates are mandatory AND criteria:
rows already rejected cannot become accepted through the angle calculation.

This module uses the same NumPy operations, order, and float64 inputs as the
legacy reference on the surviving rows; it never estimates/quantizes a value.
Keep this experimental path disabled until the whole-project differential
suite and actual MARCENA point-ID/class comparisons succeed.
"""
from __future__ import annotations

import numpy as np


def sparse_angle_filter(tri_points, simplices, inverse, points, normals,
                        longest_edge, signed_distance, absolute_distance,
                        preliminary, *, iteration_angle, reduce_angle_edge,
                        edge_length_threshold, use_distance_as_rating,
                        distance_weight, iteration_distance):
    """Refine a copy of the mandatory-gate mask, returning (mask, angles).

    `angles` need only be defined for rows passing the mandatory gates; all
    passing/ranking rows receive the bit-for-bit same NumPy angle expression.
    The output mask retains the exact original order and length.
    """
    mask = np.asarray(preliminary, dtype=bool).copy()
    chosen = np.flatnonzero(mask)
    angles = np.empty(len(mask), dtype=np.float64)
    # Skipped angles are never used: these rows always remain rejected.
    angles.fill(0.0)
    if len(chosen) == 0:
        return mask, angles

    # EXACT original expressions, restricted to the surviving subsequence.
    pp = points[chosen] - signed_distance[chosen, np.newaxis] * normals[chosen]
    psv = simplices[inverse[chosen]]
    near = np.minimum(
        np.minimum(
            np.linalg.norm(tri_points[psv[:, 0]] - pp, axis=1),
            np.linalg.norm(tri_points[psv[:, 1]] - pp, axis=1),
        ),
        np.linalg.norm(tri_points[psv[:, 2]] - pp, axis=1),
    )
    measured = np.where(
        near < 1e-12, 90.0,
        np.degrees(np.arctan2(absolute_distance[chosen], near)),
    )
    angles[chosen] = measured
    allowed = np.full(len(chosen), iteration_angle, dtype=float)
    if reduce_angle_edge:
        sm = longest_edge[chosen] < edge_length_threshold
        allowed[sm] = (iteration_angle * longest_edge[chosen][sm]
                       / edge_length_threshold)
    if use_distance_as_rating:
        weight = distance_weight / 100.0
        accept = ((1 - weight) * np.where(allowed > 0, measured / allowed, 1.0)
                  + weight * (absolute_distance[chosen] / iteration_distance
                              if iteration_distance > 0 else np.ones_like(
                                  absolute_distance[chosen]))) <= 1.0
    else:
        accept = measured <= allowed
    mask[chosen] = accept
    return mask, angles
