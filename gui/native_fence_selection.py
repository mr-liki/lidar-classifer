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


def collect_fence_candidates(app, allowed_types=None, *, include_curves=True, native_records=None):
    """Enumerate each *valid completed* fence once across VTK, native Draw and Curve.

    Never use the point-cloud storage/backend to decide where drawings live.
    Source references give the pickers stable IDs across refreshes. We only
    duplicate geometry, never mutate the actual editable drawing or SNT data.
    ``native_records`` can be supplied by headless tests without loading Qt.
    """
    from gui.native_byclass_fences import (NATIVE_FENCE_TYPES, _valid_ring,
                                           fence_identity)

    allowed = (set(NATIVE_FENCE_TYPES) if allowed_types is None else
               set(str(t).strip().lower() for t in allowed_types) &
               set(NATIVE_FENCE_TYPES))
    if native_records is None:
        from gui.native_byclass_fences import native_fence_records
        native_records = native_fence_records(app, allowed)
    native_records = list(native_records)
    result, seen = [], set()
    # A native original also appearing in the legacy digitizer is one drawing,
    # not two separate fences in the list.
    native_original_ids = {id(f.get('native_ref')) for f in native_records
                           if isinstance(f, dict) and f.get('native_ref') is not None}

    def append_if_valid(shape):
        if not isinstance(shape, dict):
            return
        kind = str(shape.get('type', '')).strip().lower()
        if kind not in allowed or shape.get('imported') or shape.get('source') in (
                'snt_readonly', 'block_creation', 'cell_creation',
                'measure_selection'):
            return
        coords = shape.get('coords')
        ring = _valid_ring(coords) if coords is not None else None
        if ring is None:
            return
        identity = fence_identity(shape)
        if identity in seen:
            return
        seen.add(identity)
        result.append(shape)

    curve_tool = getattr(app, 'curve_tool', None)
    digitizer = getattr(app, 'digitizer', None)
    if digitizer is not None:
        getter = getattr(digitizer, '_get_drawing_coords', None)
        for drawing in getattr(digitizer, 'drawings', ()) or ():
            if not isinstance(drawing, dict) or id(drawing) in native_original_ids:
                continue
            if (drawing.get('source') == 'curve_tool' and curve_tool is not None
                    and any(drawing is item for item in
                            (getattr(curve_tool, 'finalized_actors', ()) or ()))):
                continue  # Curve owner supplies the authoritative interpolated geometry.
            kind = str(drawing.get('type', '')).strip().lower()
            if kind not in allowed or drawing.get('imported') or drawing.get('source') in (
                    'snt_readonly', 'block_creation', 'cell_creation', 'measure_selection'):
                continue
            is_cell = getattr(digitizer, 'is_cell_creation_drawing', None)
            if callable(is_cell) and is_cell(drawing):
                continue
            coords = None
            if callable(getter):
                try:
                    coords = getter(drawing)
                except (ValueError, TypeError, IndexError):
                    pass
            if coords is None:
                coords = drawing.get('coords')
            ring = _valid_ring(coords) if coords is not None else None
            if ring is None:
                continue
            # Keep VTK's original XYZ vertices, including Z, for its actors;
            # the Polygon mask will project to XY only when needed.
            vertices = list(coords)
            if tuple(vertices[0][:2]) != tuple(vertices[-1][:2]):
                vertices.append(vertices[0])
            wrapped = dict(drawing, type=kind, coords=vertices,
                           source_ref=drawing, _fence_normalized=True)
            append_if_valid(wrapped)

    for fence in native_records:
        append_if_valid(fence)

    if include_curves and curve_tool is not None:
        # The legacy get_curves_as_fences() mutates curve_data on read. Build
        # compatible descriptors from the authoritative finished strokes instead.
        # Preserve the original object for stable picker identities and deletion.
        for idx, original in enumerate(getattr(curve_tool, 'finalized_actors', ()) or ()):
            if not isinstance(original, dict):
                continue
            coords = original.get('interpolated')
            if coords is None:
                coords = original.get('coords')
            ring = _valid_ring(coords) if coords is not None else None
            if ring is None:
                continue
            vertices = list(coords)
            if tuple(vertices[0][:2]) != tuple(vertices[-1][:2]):
                vertices.append(vertices[0])
            append_if_valid(dict(original, type='curve', source='curve_tool',
                                 curve_data=original, curve_index=idx,
                                 coords=vertices, _fence_normalized=True))
    return result


def offer_fence_picker(dialog, app, *, only_if_candidates=False):
    """Open an existing tool's fence picker when Element Selection is empty.

    A zero Element Selection counter means *no preselection*, not *no drawings*.
    Keep the tool's normal picker, which can select one or many mixed fences.
    Returns False when no picker exists or optional tools have no candidates.
    """
    if only_if_candidates and not collect_fence_candidates(app):
        return False
    opener = getattr(dialog, 'select_fence', None)
    if not callable(opener):
        selector = getattr(dialog, '_fence_sel', None)
        opener = getattr(selector, 'open_fence_picker', None)
    if callable(opener):
        opener()
        return True
    return False
