"""Pure geometry for Vulkan Measure Line/Path (also testable without Qt/GPU).

Measure Line: one cumulative total at the final endpoint.
Measure Path: cumulative distance at each vertex after the first.
Lengths use the 2-D plan-view coordinates supplied by NativeMeasure.
"""

import math


def measurement_labels(points, path=False):
    """Return ``[(vertex_index, cumulative_world_distance), ...]``.

    Ignore repeated/zero-length consecutive points so labels cannot stack at
    the same vertex. This never alters the underlying measurement geometry.
    """
    total = 0.0
    result = []
    last_index = None
    for index in range(1, len(points)):
        a, b = points[index - 1], points[index]
        distance = math.hypot(float(b[0]) - float(a[0]),
                              float(b[1]) - float(a[1]))
        if distance <= 1e-12:
            continue
        total += distance
        if path:
            result.append((index, total))
        else:
            last_index = index
    return result if path else ([(last_index, total)] if last_index is not None else [])


def measurement_distance(points):
    """Total 2-D route length, including every connected segment."""
    return sum(math.hypot(float(b[0]) - float(a[0]),
                          float(b[1]) - float(a[1]))
               for a, b in zip(points, points[1:]))


def measurement_text(distance, unit='m'):
    """Use the existing three-decimal Measure Settings unit convention."""
    if unit == 'km':
        return f'{distance / 1000.0:.3f} km'
    return f'{distance:.3f} m'
