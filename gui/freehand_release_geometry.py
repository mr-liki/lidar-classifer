"""Small, renderer-independent rules for completing a Freehand drag.

Only the *release* endpoint is magnetic; samples collected while dragging are
kept intact. A stroke is never forcibly connected to its beginning when it
ends outside the snap tolerance.
"""
from __future__ import annotations

import math


def _xy(point):
    try:
        x, y = float(point[0]), float(point[1])
        if math.isfinite(x) and math.isfinite(y):
            return x, y
    except (IndexError, TypeError, ValueError):
        pass
    return None


def snap_freehand_release(points, endpoint_candidates=(), tolerance=0.0):
    """Return (sampled_points, snap_kind) for a completed left-button drag.

    ``snap_kind`` is 'start', 'endpoint', or None. Existing external endpoint
    candidates are XY snapped on the working plane of the release point, so
    plan-view drawings cannot silently jump to an unrelated layer elevation.
    This function never creates another drawing or alters source objects.
    """
    result = [tuple(p) for p in points]
    if len(result) < 2 or _xy(result[-1]) is None:
        return result, None
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError):
        return result, None
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        return result, None

    release = result[-1]
    release_xy = _xy(release)
    previous_xy = _xy(result[-2])
    best = None

    def consider(candidate, kind, priority):
        nonlocal best
        target = _xy(candidate)
        if target is None:
            return
        distance = math.hypot(release_xy[0] - target[0], release_xy[1] - target[1])
        if distance > tolerance:
            return
        # Prevent an accidental two-point zero-length Freehand stroke.
        if len(result) == 2 and _xy(result[0]) == target:
            return
        # Do not select a target that would collapse the final segment.
        if previous_xy == target:
            return
        rank = (distance, priority)
        if best is None or rank < best[0]:
            best = (rank, kind, candidate)

    # A legitimate self-closure needs three distinct vertices BEFORE the
    # release sample. Snapping a single tiny scribble to itself is invalid.
    if len({_xy(p) for p in result[:-1] if _xy(p) is not None}) >= 3:
        consider(result[0], 'start', 0)
    for candidate in endpoint_candidates:
        consider(candidate, 'endpoint', 1)

    if best is None:
        return result, None
    _, kind, target = best
    x, y = _xy(target)
    if kind == 'start':
        # Exact vertex match, including Z for 3D drawings.
        result[-1] = result[0]
    else:
        # Snap in plan XY only; preserve the Freehand working plane.
        result[-1] = (x, y, *release[2:])
    return result, kind


def close_freehand_to_start(points):
    """Return (closed_points, valid) for ONE completed Freehand drag.

    Retain every sampled point, including the actual mouse-release point,
    and connect that point back to the exact first control point. No feature
    picking or zoom-dependent tolerance: release anywhere closes the stroke.
    Reject too-short strokes so no degenerate 2-point loop is committed.
    Never mutate the original input or other drawing objects.
    """
    try:
        result = [tuple(p) for p in points]
    except (TypeError, ValueError):
        return [], False
    if len(result) < 3 or any(_xy(p) is None for p in result):
        return result, False
    if len({_xy(p) for p in result}) < 3:
        return result, False
    if _xy(result[-1]) == _xy(result[0]):
        result[-1] = result[0]  # also preserve exact start Z
    else:
        result.append(result[0])  # never delete or move release sample
    return result, True
