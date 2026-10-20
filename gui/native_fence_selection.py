"""Shared fence-selection geometry and Vulkan cyan-highlight integration.

This module does not change source drawings, source SNT entities or LiDAR classes.
Import Qt/Vulkan modules lazily so geometry conversion can be tested headlessly.
"""
from __future__ import annotations

from gui.native_byclass_fences import _valid_ring, fence_identity, NATIVE_FENCE_TYPES


FENCE_TYPES = NATIVE_FENCE_TYPES


def selected_fence_shapes(selected, *, native_draw=None, digitizer=None):
    """Expand selected Draw objects to *closed* nonzero-area world-XY fences.

    Native rectangles/circles store only two input points. NativeDraw's geometry
    expansion is authoritative. An open two-point line cannot become a fence.
    VTK keeps its original coordinate dimensions and algorithms.
    """
    result = []
    seen = set()
    for drawing in selected:
        if not isinstance(drawing, dict):
            continue
        kind = str(drawing.get('type', '')).strip().lower()
        if (kind not in FENCE_TYPES or drawing.get('source') in (
                'block_creation', 'cell_creation', 'snt_readonly',
                'measure_selection') or drawing.get('imported')):
            continue
        if native_draw is not None:
            # NativeSelectionManager.get() returns actual native Draw dictionaries.
            # Never call the VTK digitizer's coordinate converter on them.
            try:
                raw = native_draw._native_coords(drawing)
            except (IndexError, TypeError, ValueError):
                continue
            ring = _valid_ring(raw)
            if ring is None:
                continue
            shape = dict(drawing)
            shape.update(type=kind, coords=ring, source='native_draw',
                         native_ref=drawing)
        else:
            getter = getattr(digitizer, '_get_drawing_coords', None)
            raw = None
            if callable(getter):
                try:
                    raw = getter(drawing)
                except (IndexError, TypeError, ValueError):
                    pass
            if raw is None:
                raw = drawing.get('coords')
            if raw is None or _valid_ring(raw) is None:
                continue
            # Preserve source dimensionality on VTK: height and actor code can
            # rely on Z, but the polygon mask uses XY.
            coords = list(raw)
            if tuple(coords[0][:2]) != tuple(coords[-1][:2]):
                coords.append(coords[0])
            shape = dict(drawing)
            shape['coords'] = coords
            shape['type'] = kind
            shape.setdefault('source_ref', drawing)
        identity = fence_identity(shape) if native_draw is not None else id(drawing)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(shape)
    return result


def sync_native_fence_highlights(owner, fences):
    """Show the exact selected boundaries in native Vulkan, never VTK actors.

    Return True when the active main viewport is native. False means the VTK
    UI should retain its current VTK highlight handling.
    """
    app = getattr(owner, 'app', None)
    if app is None:
        return False
    from gui.native_draw import get_native_draw
    fence_list = list(fences)
    draw = get_native_draw(app, create=bool(fence_list))
    if draw is None:
        return False
    draw.set_classification_fence_highlights(owner, fence_list)
    return True


def clear_native_fence_highlights(owner):
    app = getattr(owner, 'app', None)
    if app is None:
        return False
    from gui.native_draw import get_native_draw
    draw = get_native_draw(app, create=False)
    if draw is None:
        return False
    draw.set_classification_fence_highlights(owner, ())
    return True
