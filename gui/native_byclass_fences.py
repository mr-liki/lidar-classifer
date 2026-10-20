"""Vulkan Draw -> By Class fence adapter.

Native Draw keeps compact editor records, notably 2-corner rectangles and
center/edge circles. These are not polygons until expanded by the same Draw
geometry method used for painting. Do not mutate editor records or VTK data.
"""
from __future__ import annotations

import math

NATIVE_FENCE_TYPES = frozenset({
    'rectangle', 'circle', 'polygon', 'freehand', 'line', 'smart_line',
    'smartline', 'polyline', 'ortho', 'curve', 'hatch', 'parallel', 'centerline', 'accudraw',
})


def fence_identity(shape):
    """Stable across picker refreshes (which recreate geometry wrappers)."""
    if isinstance(shape, dict):
        if shape.get('source') == 'curve_tool':
            return id(shape.get('curve_data', shape))
        if shape.get('source') == 'native_draw':
            return id(shape.get('native_ref', shape))
        if shape.get('source_ref') is not None:
            return id(shape['source_ref'])
    return id(shape)


def _valid_ring(coords):
    """Return a closed, finite world-XY polygon or None for incomplete drafts."""
    try:
        points = [(float(p[0]), float(p[1])) for p in coords]
    except (ValueError, TypeError, IndexError):
        return None
    if len(points) < 3 or not all(math.isfinite(v) for p in points for v in p):
        return None
    if points[0] != points[-1]:
        points.append(points[0])
    # Reject collinear/zero-area lines; subtract the origin to avoid loss of
    # precision for UTM/world coordinates in the millions.
    ox, oy = points[0]
    area_twice = sum((x0 - ox) * (y1 - oy) - (x1 - ox) * (y0 - oy)
                     for (x0, y0), (x1, y1) in zip(points, points[1:]))
    if abs(area_twice) <= 1e-18:
        return None
    return points


def native_drawings(app):
    """Read finalized native drawings without allocating an input overlay."""
    from gui.native_draw import get_native_draw
    # Drawing ownership follows the visible viewport, independently of where
    # classification arrays live. Surface Vulkan keeps app.data populated and
    # has no _lnt_main_view, so want_native() is deliberately NOT a Draw gate.
    draw = get_native_draw(app, create=False)
    if draw is not None:
        return list(getattr(draw, 'drawings', ()) or ())
    view = getattr(app, '_lnt_main_view', None)
    if view is None:
        backend = getattr(app, 'render_backend', None)
        if (backend is not None and getattr(backend, 'active', False)
                and backend.vulkan_viewport_is_visible()
                and backend.owns_main_camera()):
            view = getattr(backend, '_plan_draw_view', None)
    draw = getattr(view, 'drawing', None)
    return list(getattr(draw, 'drawings', ()) or
                getattr(view, '_draw_saved_shapes', ()) or ())


def native_fence_records(app, allowed_types=None):
    """Expand finalized Vulkan Draw geometry on copies with stable source IDs.

    Supports both array-backed Vulkan and file-backed LNT, including drawings
    saved while the visible widget is replaced. No overlay is allocated.
    """
    from gui.native_draw import NativeDraw
    drawings = native_drawings(app)
    allowed = (NATIVE_FENCE_TYPES if allowed_types is None else
               NATIVE_FENCE_TYPES.intersection(str(t).lower() for t in allowed_types))
    result = []
    for drawing in drawings:
        if not isinstance(drawing, dict):
            continue
        shape_type = str(drawing.get('type', '')).strip().lower()
        if shape_type not in allowed or drawing.get('source') in ('block_creation', 'cell_creation'):
            continue
        # The Draw renderer expands rectangle (2 corners), circle (2 points)
        # and interpolated curve; raw ``drawing['coords']`` is NOT a fence.
        try:
            points = _valid_ring(NativeDraw._native_coords(drawing))
        except (TypeError, ValueError, IndexError):
            continue
        if points is None:
            continue
        result.append({
            'type': shape_type, 'coords': points,
            'source': 'native_draw', 'native_ref': drawing,
        })
    return result
