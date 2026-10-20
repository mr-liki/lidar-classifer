"""Pure world-XY splitting for Vulkan's VTK-compatible Intersect session.

No Qt/VTK or GPU dependency. Unlike the drawing snap routine, this splitter
includes endpoint intersections (but ignores collinear overlapping segments).
It never mutates source drawing dictionaries or imported SNT entities.
"""
from __future__ import annotations

import math

MAX_INTERSECT_SEGMENTS = 1200
MAX_INTERSECT_PAIRS = 250000
_TOL = 1e-9


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def _segment_cross(a, b, c, d):
    """Return intersection XY and each segment's parameter, if unambiguous."""
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    denominator = _cross(rx, ry, sx, sy)
    scale = max(math.hypot(rx, ry) * math.hypot(sx, sy), 1e-12)
    if abs(denominator) <= scale * 1e-12:
        return None  # coincident/parallel lines have no single reliable cut
    qx, qy = c[0] - a[0], c[1] - a[1]
    t = _cross(qx, qy, sx, sy) / denominator
    u = _cross(qx, qy, rx, ry) / denominator
    if -_TOL <= t <= 1 + _TOL and -_TOL <= u <= 1 + _TOL:
        t = max(0.0, min(1.0, t))
        u = max(0.0, min(1.0, u))
        return (a[0] + t * rx, a[1] + t * ry), t, u
    return None


def _same(a, b):
    scale = max(1.0, abs(a[0]), abs(a[1]), abs(b[0]), abs(b[1]))
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= scale * 1e-12


def _add_unique(target, point):
    if not target or not _same(target[-1], point):
        target.append(point)


def _cut_segment(cuts, index, t, point):
    """Move t=0 on a segment to the prior segment's t=1, if possible."""
    if t <= _TOL:
        if index == 0:
            return
        index -= 1
        t = 1.0
    elif t >= 1.0 - _TOL:
        t = 1.0
    if not any(abs(existing_t - t) < 1e-9 for existing_t, _ in cuts[index]):
        cuts[index].append((t, tuple(point)))


def _parts_for(coords, cuts):
    current = [tuple(coords[0])]
    parts = []
    for index, endpoint in enumerate(coords[1:]):
        for t, point in sorted(cuts[index], key=lambda value: value[0]):
            _add_unique(current, point)
            if len(current) >= 2:
                parts.append(current)
            current = [point]
        _add_unique(current, tuple(endpoint))
    if len(current) >= 2:
        parts.append(current)
    return parts


def split_selected_features(features):
    """Split editable native features at crossings of any selected feature.

    Each entry is a descriptor with ``source``, ``coords`` and optionally
    ``drawing``. Selected read-only SNT figures can *cut* native figures but
    are never returned as editable parts. No mutations happen here.

    Returns ``(parts, count_of_distinct_crossings)``; an empty result means
    no crossings. A safety cap rejects large O(N**2) cases without work.
    """
    source_features = []
    total_segments = 0
    for feat in features:
        if feat.get('type') in ('text', 'point', 'circle', 'hatch'):
            continue  # analytic circle/hatch geometry cannot be split as straight edges
        try:
            coords = [(float(q[0]), float(q[1])) for q in feat.get('coords', ())]
        except (TypeError, ValueError, IndexError):
            continue
        if len(coords) < 2 or not all(math.isfinite(n) for q in coords for n in q):
            continue
        nsegments = sum(not _same(a, b) for a, b in zip(coords, coords[1:]))
        if not nsegments:
            continue
        total_segments += nsegments
        source_features.append((feat, coords))
    if total_segments > MAX_INTERSECT_SEGMENTS or total_segments * (total_segments-1)//2 > MAX_INTERSECT_PAIRS:
        raise ValueError('Intersect selection is too complex; select a smaller group of drawings')
    # AccuDraw records retain XYZ coordinates; this XY-only edit operation
    # must never leave their 3-D geometry metadata stale. They may cut other
    # drawings, but remain unchanged until true XYZ splitting is supported.
    editable = [(feat, coords) for feat, coords in source_features
                if feat['source'] == 'native' and feat.get('type') != 'accudraw']
    if not editable:
        return [], 0

    cuts = {id(feat): [[] for _ in range(len(coords) - 1)] for feat, coords in source_features}
    crossings = set()
    for left in range(len(source_features)):
        fa, ca = source_features[left]
        for right in range(left + 1, len(source_features)):
            fb, cb = source_features[right]
            for si, (a, b) in enumerate(zip(ca, ca[1:])):
                if _same(a, b):
                    continue
                for sj, (c, d) in enumerate(zip(cb, cb[1:])):
                    if _same(c, d):
                        continue
                    # Skip non-overlapping segment bounding boxes up front.
                    if (max(a[0], b[0]) < min(c[0], d[0]) or
                            max(c[0], d[0]) < min(a[0], b[0]) or
                            max(a[1], b[1]) < min(c[1], d[1]) or
                            max(c[1], d[1]) < min(a[1], b[1])):
                        continue
                    cross = _segment_cross(a, b, c, d)
                    if cross is None:
                        continue
                    point, ta, tb = cross
                    # Intersections shared only by endpoints do not create new
                    # segment pieces but are harmless as cut candidates.
                    crossings.add((round(point[0], 8), round(point[1], 8)))
                    _cut_segment(cuts[id(fa)], si, ta, point)
                    _cut_segment(cuts[id(fb)], sj, tb, point)
    if not crossings:
        # Match VTK: figures with no crossings stay as one selectable/deletable
        # whole part. The document still remains untouched unless removed.
        return [{'source': feat, 'coords': list(coords), 'deleted': False}
                for feat, coords in editable], 0
    result = []
    for feat, coords in editable:
        for points in _parts_for(coords, cuts[id(feat)]):
            if len(points) >= 2 and any(not _same(a, b) for a, b in zip(points, points[1:])):
                result.append({'source': feat, 'coords': points, 'deleted': False})
    return result, len(crossings)
