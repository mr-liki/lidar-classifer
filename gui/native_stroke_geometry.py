"""Curve and Freehand geometry rules for the native Vulkan drawing adapter.

Keep this module free of Qt, VTK and native DLL dependencies. The authoritative
Curve algorithm is gui.curve_tools.CurveTool._interpolate_catmull_rom:
20 uniform Catmull-Rom samples per control-point segment, then the exact end.
"""
from __future__ import annotations

import math

VTK_CURVE_SAMPLES_PER_SEGMENT = 20


def catmull_rom_vtk(control_points, steps=VTK_CURVE_SAMPLES_PER_SEGMENT):
    """Sample the original VTK cubic, without changing its editable controls.

    XY and XYZ are both supported. Endpoint samples are bit-exact copies of
    the control coordinates; intermediate samples are only render geometry.
    """
    points = [tuple(float(v) for v in p) for p in control_points]
    if not points:
        return []
    if len(points) == 1:
        return points
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError('steps must be a positive integer')
    dimensions = len(points[0])
    if dimensions not in (2, 3) or any(len(p) != dimensions for p in points):
        raise ValueError('curve control points must all be 2D or all be 3D')
    if not all(math.isfinite(v) for p in points for v in p):
        raise ValueError('curve points must be finite')

    result = []
    for i in range(len(points) - 1):
        p0, p1 = points[max(0, i - 1)], points[i]
        p2, p3 = points[i + 1], points[min(len(points) - 1, i + 2)]
        for j in range(steps):
            t = j / steps
            t2, t3 = t * t, t * t * t
            result.append(tuple(0.5 * ((2 * b) + (-a + c) * t +
                                      (2 * a - 5 * b + 4 * c - d) * t2 +
                                      (-a + 3 * b - 3 * c + d) * t3)
                                for a, b, c, d in zip(p0, p1, p2, p3)))
    result.append(points[-1])
    return result


def drop_consecutive_duplicates(points):
    """Remove repeated adjacent controls/samples; preserve deliberate closure.

    Do not use a viewport-size tolerance here: distinct close controls are
    needed for intentional tight bends and precise cadastral coordinates.
    """
    result = []
    for point in points:
        p = tuple(point)
        if not result or p != result[-1]:
            result.append(p)
    return result


def snap_endpoint_near_start(points, candidate, tolerance):
    """Freehand starts magnetically closing only with a real polygonal stroke."""
    if len(points) < 3 or not math.isfinite(tolerance) or tolerance <= 0:
        return candidate
    start = points[0]
    # Require at least three distinct XY locations including the candidate
    # (otherwise a short line would be turned into a zero-area ring).
    # Stop after three unique points: this runs in near-constant time per
    # Freehand move, even for very long strokes with thousands of samples.
    distinct = set()
    for p in points[:-1]:
        distinct.add((float(p[0]), float(p[1])))
        if len(distinct) == 3:
            break
    if len(distinct) < 3:
        return candidate
    if math.hypot(candidate[0] - start[0], candidate[1] - start[1]) <= tolerance:
        return tuple(start)
    return candidate
