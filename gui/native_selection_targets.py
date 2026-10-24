"""Shared, dependency-free geometric acceptance rules for native drawing/selection.

Never mutate imported SNT entities here. This module can be regression-tested
without loading Qt, VTK or a GPU, so a scene with 7M LiDAR points is unnecessary.
"""
from __future__ import annotations

import math


def editable_draw(drawing):
    """Block/cell creation has a dedicated picker; not an editing selection fence."""
    return (isinstance(drawing, dict) and
            drawing.get('source') not in ('block_creation', 'cell_creation'))


def snap_draft_closure(points, candidate, tolerance):
    """Prefer the exact FIRST Curve control point over any unrelated scene snap."""
    if len(points) < 3 or not math.isfinite(tolerance) or tolerance <= 0:
        return candidate
    a = points[0]
    if math.hypot(float(candidate[0])-float(a[0]),
                  float(candidate[1])-float(a[1])) <= tolerance:
        return list(a) if isinstance(candidate, list) else tuple(a)
    return candidate


def close_freehand(points, tolerance):
    """Always close a real freehand region at release/right-click, exactly.

    A two-point accidental stroke is NOT a valid polygon; leave it an open
    line instead of silently manufacturing a zero-area self-intersecting ring.
    """
    result = list(points)
    if len(result) < 3 or len({(float(p[0]),float(p[1])) for p in result}) < 3:
        return result
    first, last = result[0], result[-1]
    if math.hypot(last[0]-first[0], last[1]-first[1]) <= max(0.,tolerance):
        result[-1] = first
    elif last != first:
        result.append(first)
    return result


def text_anchor(entity):
    """Read a render-space label anchor without assuming SNT text storage form."""
    raw = next((entity.get(name) for name in
                ('position','insertion_point','insert','anchor','points')
                if entity.get(name) is not None), None)
    if raw is None and entity.get('x') is not None and entity.get('y') is not None:
        raw = (entity['x'],entity['y'])
    try:
        if raw is None:return None
        if isinstance(raw,dict):raw=(raw.get('x'),raw.get('y'))
        if len(raw)>0 and isinstance(raw[0],(list,tuple)):raw=raw[0]
        pair=(float(raw[0]),float(raw[1]))
        return pair if all(map(math.isfinite,pair)) else None
    except (ValueError,TypeError,KeyError,IndexError):
        return None


def text_bounds(entity, anchor):
    """Conservative box for label hit-testing in world units.

    Use explicit CAD bounds when provided, else a font-height/rotation
    approximation. Always include the anchor, important for short labels.
    """
    bounds=entity.get('bounds')
    if bounds is None:bounds=entity.get('bbox')
    if bounds is not None:
        try:
            b=[float(v) for v in bounds]
            if len(b)>=4 and all(math.isfinite(v) for v in b[:4]):
                # CAD world bounds: (xmin,xmax,ymin,ymax), not a screen QRect.
                if b[0]<=b[1] and b[2]<=b[3]:
                    return min(b[0],anchor[0]),max(b[1],anchor[0]),min(b[2],anchor[1]),max(b[3],anchor[1])
        except (TypeError,ValueError):pass
    try:height=abs(float(entity.get('height') or entity.get('font_size') or 0.))
    except (TypeError,ValueError):height=0.
    if not math.isfinite(height):height=0.
    try:factor=abs(float(entity.get('width_factor') or 1.))
    except (TypeError,ValueError):factor=1.
    factor=min(max(factor,0.1),10.) if math.isfinite(factor) else 1.
    # The fallback catches intersections with the visible glyph, not just the anchor.
    width=height*0.65*len(str(entity.get('text') or entity.get('value') or ''))*factor
    try:angle=math.radians(float(entity.get('rotation') or 0.))
    except (TypeError,ValueError):angle=0.
    if not math.isfinite(angle):angle=0.
    c,s=math.cos(angle),math.sin(angle)
    x,y=anchor
    # Translation-only text insertion; rotation swaps axes as needed.
    corners=[(x+dx*c-dy*s,y+dx*s+dy*c)
             for dx,dy in ((0.,0.),(width,0.),(width,height),(0.,height))]
    return (min(v[0] for v in corners),max(v[0] for v in corners),
            min(v[1] for v in corners),max(v[1] for v in corners))


def box_overlap(a,b,inside=False):
    """(minx,maxx,miny,maxy) in both boxes; Inside requires full containment."""
    if inside:return b[0]<=a[0] and a[1]<=b[1] and b[2]<=a[2] and a[3]<=b[3]
    return not (a[1]<b[0] or a[0]>b[1] or a[3]<b[2] or a[2]>b[3])


# Parallel/Centerline accept only figures created by Draw, never attached SNT records.
# Keep source eligibility separate from the global snap index:
# Hatch and other tools legitimately need those SNT geometries.
_PARALLEL_NATIVE_LINE_TYPES = frozenset({
    'line', 'line_segment', 'smartline', 'smart_line', 'polyline',
    'lwpolyline', 'curve', 'parallel', 'parallel_line', 'centerline',
    'freehand', 'accudraw',
})
_PARALLEL_FORBIDDEN_ORIGINS = frozenset({
    'block_creation', 'cell_creation', 'snt_readonly',
    'measure_selection', 'snt_block', 'trajectory', 'flightline',
})


def parallel_line_source(feature):
    """True only for an open, nondegenerate native Draw line for Parallel."""
    return drawn_open_line_source(feature)


def drawn_open_line_source(feature):
    """Accept live Draw line descriptors, excluding imports and creation bounds."""
    if not isinstance(feature, dict):
        return False
    origin = str(feature.get('source', '') or '').lower()
    if origin != 'native':
        return False
    # Older callers may pass a flat native Draw descriptor rather than the
    # canonical {'drawing': original} wrapper. The dialog's _valid_source()
    # still independently validates live object identity before committing.
    record = feature.get('drawing') or feature
    if not isinstance(record, dict):
        return False
    kind = str(feature.get('type') or record.get('type') or '').lower()
    if kind not in _PARALLEL_NATIVE_LINE_TYPES:
        return False
    source = str(record.get('source', '') or '').lower()
    if source in _PARALLEL_FORBIDDEN_ORIGINS:
        return False
    if bool(record.get('imported')) or source in ('snt', 'snt_import'):
        return False
    if bool(record.get('closed')) or bool(record.get('is_closed')):
        return False
    coords = feature.get('coords')
    if coords is None:
        coords = record.get('coords') or record.get('points') or record.get('vertices')
    try:
        if coords is None or len(coords) < 2:
            return False
        first = (float(coords[0][0]), float(coords[0][1]))
        last = (float(coords[-1][0]), float(coords[-1][1]))
        if not all(math.isfinite(x) for p in coords for x in p[:2]):
            return False
        if first == last or not any(
                math.hypot(float(p[0]) - first[0], float(p[1]) - first[1]) > 1e-10
                for p in coords[1:]):
            return False
    except (IndexError, TypeError, ValueError, OverflowError):
        return False
    return True
