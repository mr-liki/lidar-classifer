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


def _native_draw(app, *, create=False):
    if app is None:
        return None
    from gui.native_draw import get_native_draw
    return get_native_draw(app, create=create)


def _sync_canvas_selection(draw, fences, owner):
    """Keep the cyan Draw/Element Selection overlay in sync with fence pickers.

    The manager paints selected Draw objects on its own, independently of the
    classification dialog's cyan strokes. Otherwise it can keep rendering an
    obsolete circle even after the dialogue selection has changed to triangle.
    """
    bridge = getattr(draw, '_selection_bridge', None)
    manager = getattr(bridge, 'selection_manager', None)
    if manager is not None and manager is not owner:
        wanted = {id(shape['native_ref']) for shape in fences
                  if isinstance(shape, dict) and shape.get('native_ref') is not None}
        chosen = {}
        for index, drawing in enumerate(getattr(draw, 'drawings', ())):
            if id(drawing) in wanted:
                feature = bridge._native_feature(index, drawing)
                chosen[feature['key']] = feature
        current = getattr(manager, '_selected', {})
        if set(current) != set(chosen):
            manager._selected = chosen
            signal = getattr(manager, 'selection_changed', None)
            if signal is not None:
                signal.emit(len(chosen))
    # Standalone Draw > Select is separate from the Element Selection manager.
    # Its previous selected flag must not re-paint an obsolete cyan outline.
    current_ref = getattr(draw, '_selected_native_ref', None)
    if current_ref is not None and not any(
            isinstance(shape, dict) and shape.get('native_ref') is current_ref
            for shape in fences):
        draw._selected_native_ref = None
        draw.selected = None


def propagate_native_fence_selection(app, fences):
    """Mirror CURRENT fence membership into already-created classification UIs.

    Visual selection and persisted `classified_fence` are independent. Direct
    canvas selection, a picker Apply, and Element Selection all update the same
    group, so reopening an old dialog cannot silently resurrect old fences.
    No VTK fallback and no point classes/drawing metadata are modified.
    """
    from gui.native_draw import get_native_draw
    draw = get_native_draw(app, create=False) if app is not None else None
    if draw is None:
        return
    revision = getattr(draw, '_fence_selection_revision', 0)
    live = list(fences)
    seen = set()
    dialogs = []
    ribbons = getattr(getattr(app, 'ribbon_manager', None), 'ribbons', {}) or {}
    ribbon = ribbons.get('by_class') if hasattr(ribbons, 'get') else None
    if ribbon is not None:
        for attr in ('inside_fence_dialog', 'height_convert_dialog',
                     'closed_by_class_dialog'):
            dialogs.append(getattr(ribbon, attr, None))
    # The LiDAR Classification tools may use a shared FenceSelectorWidget.
    for attr in ('_classify_low_points_dlg', '_classify_isolated_points_dlg',
                 '_classify_ground_dlg', '_classify_surface_points_dlg',
                 '_classify_below_surface_dlg'):
        dialogs.append(getattr(app, attr, None))
    for dialog in dialogs:
        if dialog is None or id(dialog) in seen:
            continue
        seen.add(id(dialog))
        if hasattr(dialog, 'selected_fences'):
            dialog.selected_fences = list(live)
        widget = getattr(dialog, '_fence_sel', None)
        if widget is not None and hasattr(widget, 'selected_fences'):
            widget.selected_fences = list(live)
            widget._native_fence_selection_revision = revision
        dialog._native_fence_selection_revision = revision
        badge = getattr(dialog, 'fence_count_badge', None)
        if badge is not None and hasattr(badge, 'setText'):
            badge.setText(str(len(live)))


def sync_native_fence_highlights(owner, fences):
    """Commit a new active fence group in the actual Vulkan viewport.

    New picks REPLACE prior picks from any other dialog. Multiple fences in
    `fences` remain selected together. Classification point data is untouched.
    VTK keeps its original selection/highlight handling.
    """
    app = getattr(owner, 'app', None)
    fence_list = list(fences)
    draw = _native_draw(app, create=bool(fence_list))
    if draw is None:
        return False
    _sync_canvas_selection(draw, fence_list, owner)
    draw.set_classification_fence_highlights(owner, fence_list, replace=True)
    propagate_native_fence_selection(app, fence_list)
    try:
        owner._native_fence_selection_revision = draw._fence_selection_revision
    except (AttributeError, TypeError):
        pass
    return True


def preview_native_fence_highlights(owner, fences):
    """Noncommittal picker checkbox preview; cancel retains prior selection."""
    draw = _native_draw(getattr(owner, 'app', None), create=bool(fences))
    if draw is None:
        return False
    draw.set_classification_fence_highlights(owner, list(fences), preview=True)
    return True


def restore_native_fence_highlights(owner):
    """Reconcile an old dialog to the latest native selection, not vice versa.

    Reopening a minimized Fence/Close/Height dialog must never restore a stale
    cyan circle after the user has selected a new triangle group.
    """
    draw = _native_draw(getattr(owner, 'app', None), create=False)
    if draw is None:
        return False
    revision = getattr(draw, '_fence_selection_revision', 0)
    if getattr(owner, '_native_fence_selection_revision', None) != revision:
        current = list(getattr(draw, '_active_fence_shapes', ()))
        if hasattr(owner, 'selected_fences'):
            owner.selected_fences = list(current)
        widget = getattr(owner, '_fence_sel', None)
        if widget is not None and hasattr(widget, 'selected_fences'):
            widget.selected_fences = list(current)
        try:
            owner._native_fence_selection_revision = revision
        except (AttributeError, TypeError):
            pass
        badge = getattr(owner, 'fence_count_badge', None)
        if badge is not None and hasattr(badge, 'setText'):
            badge.setText(str(len(current)))
    return True


def clear_native_fence_highlights(owner):
    draw = _native_draw(getattr(owner, 'app', None), create=False)
    if draw is None:
        return False
    # Picker preview cleanup must never clear the committed selection.
    if getattr(draw, '_fence_picker_preview_owner', None) == id(owner):
        draw._fence_picker_preview_owner = None
        draw._fence_picker_preview_rings = []
        draw._paint()
        return True
    was_active = getattr(draw, '_active_fence_highlight_owner', None) == id(owner)
    if was_active:
        _sync_canvas_selection(draw, (), owner)
    draw.set_classification_fence_highlights(owner, ())
    if was_active:
        propagate_native_fence_selection(getattr(owner, 'app', None), ())
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
