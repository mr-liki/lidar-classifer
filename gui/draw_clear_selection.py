"""Read-only discovery of deletable Draw objects (not SNT or LiDAR data).

Clear uses object identity, not coordinate equality: two identical rectangles
remain independent choices. No mutation and no GUI dependencies in this module.
"""
from __future__ import annotations

from collections import Counter

PROTECTED_SOURCES = frozenset({
    'block_creation', 'cell_creation', 'snt_readonly', 'snt',
    'measure_selection', 'measure', 'imported_snt',
})

TOOL_NAMES = {
    'rectangle': 'Rectangle', 'circle': 'Circle',
    'polygon': 'Polygon', 'polyline': 'Polyline',
    'line': 'Line', 'smartline': 'Smart Line', 'smart_line': 'Smart Line',
    'ortho': 'Ortho', 'orthopolygon': 'Ortho Polygon',
    'freehand': 'Freehand', 'curve': 'Curve', 'hatch': 'Hatch',
    'hatcharea': 'Hatch', 'parallel': 'Parallel', 'centerline': 'Centerline',
    'accudraw': 'AccuDraw', 'text': 'Text', 'point': 'Point',
    'vertex': 'Vertex',
}


def is_clearable_draw(drawing, *, vtk=False, in_scope=None):
    """Only unsaved, user-owned Draw shapes may appear in Draw > Clear.

    Classified and unclassified drawings are both editable objects, but the
    Draw > Clear workflow protects classified fences until user confirmation.
    """
    if not isinstance(drawing, dict):
        return False
    if drawing.get('imported') or drawing.get('read_only'):
        return False
    if str(drawing.get('source', '')).lower() in PROTECTED_SOURCES:
        return False
    if drawing.get('snt_committed') or drawing.get('snt_stored'):
        return False
    if vtk and drawing.get('snt_file') and drawing.get('layer') and (
        str(drawing['layer']).upper() != 'DIGITIZER'
    ):
        return False
    if in_scope is not None and not in_scope(drawing):
        return False
    return bool(str(drawing.get('type', '')).strip())


def clearable_drawings(drawings, *, vtk=False, in_scope=None):
    """Snapshot current editable objects by identity, without copying them."""
    return [d for d in drawings if is_clearable_draw(d, vtk=vtk, in_scope=in_scope)]


def named_drawings(drawings, *, vtk=False, in_scope=None):
    """Return (drawing, tool label, item label) for a grouped checkable UI."""
    counts = Counter()
    result = []
    for drawing in clearable_drawings(drawings, vtk=vtk, in_scope=in_scope):
        kind = str(drawing.get('type', '')).strip().lower()
        tool = TOOL_NAMES.get(kind, kind.replace('_', ' ').title())
        counts[tool] += 1
        result.append((drawing, tool, f'{tool} {counts[tool]}'))
    return result


def select_drawings_by_identity(drawings, selected_ids, *, vtk=False, in_scope=None):
    """Recheck the live collection at deletion time; no stale popup deletions."""
    ids = set(selected_ids)
    return [d for d in clearable_drawings(drawings, vtk=vtk, in_scope=in_scope)
            if id(d) in ids]



def partition_drawings_for_clear(drawings, *, vtk=False, in_scope=None):
    """Return (unused, classified) DRAW objects without modifying the scene.

    A classified fence is one used by a successful By Class/LiDAR classification
    operation. Neither selection, preview, nor cyan highlight marks a fence used.
    Imported, committed, and otherwise protected drawings are excluded.
    """
    editable = clearable_drawings(drawings, vtk=vtk, in_scope=in_scope)
    unused = [d for d in editable if not d.get('classified_fence', False)]
    classified = [d for d in editable if d.get('classified_fence', False)]
    return unused, classified


def _xy_key(shape):
    """Exact coordinate key only for *unambiguous legacy* VTK handoffs."""
    if not isinstance(shape, dict):
        return None
    coords = shape.get('coords')
    if coords is None:
        return None
    try:
        pts = tuple((float(p[0]), float(p[1])) for p in coords)
    except (TypeError, ValueError, IndexError):
        return None
    if len(pts) < 3:
        return None
    return pts[:-1] if pts[0] == pts[-1] else pts


def mark_used_classification_fences(app, fences, *, operation=None):
    """Mark only live SOURCE Draw dictionaries after a successful point write.

    Native fence wrappers carry ``native_ref``; VTK wrappers ``source_ref`` or
    the original actor. A legacy coordinate match is used only when exactly
    one live, editable drawing has that geometry. Two identical rectangles
    are NEVER both marked because one was classified.

    This function does not touch LiDAR point classes, imported SNT data,
    selection/highlighting, history stacks, or source geometry.
    """
    fences = list(fences or ())
    if not fences:
        return 0
    candidates = []
    from gui.native_byclass_fences import native_drawings
    candidates.extend(clearable_drawings(native_drawings(app)))
    digitizer = getattr(app, 'digitizer', None)
    if digitizer is not None:
        scope = getattr(digitizer, 'drawing_in_edit_scope', None)
        candidates.extend(clearable_drawings(
            getattr(digitizer, 'drawings', ()), vtk=True,
            in_scope=scope if callable(scope) else None))
    # Keep candidate identity unique when a drawing is shared across stores.
    unique = {id(d): d for d in candidates}
    marked = set()
    for fence in fences:
        if not isinstance(fence, dict):
            continue
        source = None
        refs = [fence, fence.get('native_ref'), fence.get('source_ref')]
        for ref in refs:
            if id(ref) in unique:
                source = unique[id(ref)]
                break
        if source is None and fence.get('actor') is not None:
            matches = [d for d in unique.values()
                       if d.get('actor') is fence.get('actor')]
            if len(matches) == 1:
                source = matches[0]
        if source is None:
            key = _xy_key(fence)
            if key is not None:
                matches = [d for d in unique.values() if _xy_key(d) == key]
                if len(matches) == 1:
                    source = matches[0]
                elif len(matches) > 1:
                    # A saved macro may only retain coordinates. Refuse an
                    # automatic Draw Clear when two live shapes are identical
                    # rather than risk deleting the shape that was used.
                    app._draw_clear_ambiguous_classified_fence = True
                else:
                    # Saved macros store geometry, not Python object refs.
                    # Native rectangles/circles have compact input coords;
                    # compare their expanded fence boundaries ONLY when one
                    # live source drawing matches unambiguously.
                    try:
                        from gui.native_byclass_fences import native_fence_records
                        expanded = native_fence_records(app)
                    except (ImportError, RuntimeError, AttributeError):
                        expanded = ()
                    mapped = {id(r['native_ref']): r['native_ref'] for r in expanded
                              if _xy_key(r) == key and id(r.get('native_ref')) in unique}
                    if len(mapped) == 1:
                        source = next(iter(mapped.values()))
                    elif len(mapped) > 1:
                        app._draw_clear_ambiguous_classified_fence = True
        if source is None:
            continue
        source['classified_fence'] = True
        # Existing VTK serializers keep the classified_fence boolean on Undo.
        # Provenance is for UI diagnostics only; avoid creating history events.
        if operation:
            ops = source.setdefault('classified_operations', [])
            if operation not in ops:
                ops.append(operation)
        marked.add(id(source))
    return len(marked)

def prune_removed_native_fences(app, draw, removed_ids):
    """Forget deleted Draw objects in live By Class/Element Selection dialogs.

    Do NOT revert previously committed classification writes. A dialog-owned
    fence wrapper keeps ``native_ref`` pointing to its source Draw dictionary.
    Other non-deleted selections and cyan outlines must survive unchanged.
    """
    roots = [app]
    ribbon = getattr(app, 'ribbon_manager', None)
    for item in getattr(ribbon, 'ribbons', {}).values() if ribbon is not None else ():
        roots.append(item)
    seen = set()
    selected_names = (
        'inside_fence_dialog', 'closed_by_class_dialog', 'height_convert_dialog',
        'by_class_dialog', '_element_selection_dialog', 'element_selection_dialog',
        'low_points_dialog', 'isolated_dialog', 'ground_dialog',
        'surface_points_dialog', 'below_surface_dialog',
        '_fence_sel', 'fence_selector',
    )
    owners = []
    for root in roots:
        owners.append(root)
        for attr in selected_names:
            dlg = getattr(root, attr, None)
            if dlg is not None:
                owners.append(dlg)
                fence_widget = getattr(dlg, '_fence_sel', None)
                if fence_widget is not None:
                    owners.append(fence_widget)
    for owner in owners:
        if id(owner) in seen:
            continue
        seen.add(id(owner))
        fences = getattr(owner, 'selected_fences', None)
        if not isinstance(fences, list) or not fences:
            continue
        kept = [fence for fence in fences if not (
            isinstance(fence, dict) and (
                id(fence) in removed_ids or
                id(fence.get('native_ref')) in removed_ids or
                id(fence.get('source_ref')) in removed_ids
            )
        )]
        if len(kept) == len(fences):
            continue
        owner.selected_fences = kept
        # Keep exactly this owner's surviving cyan boundaries. An unrelated
        # dialog's selected fences are not cleared.
        draw.set_classification_fence_highlights(owner, kept)
        update = getattr(owner, '_update_status', None)
        if callable(update):
            try:
                update()
            except (AttributeError, RuntimeError):
                pass
