"""Parallel Classify Band against native LNT source identities.

The original VTK path is unchanged.  This adapter asks the existing native
section store for leaf records, evaluates the exact VTK XY band predicate in
float64, and commits through the same node/local writeback and Undo format as
native section classification.  It never tries to classify only rendered LOD
points; those are a visual subset, not source records.
"""
from __future__ import annotations

import math
import time
import numpy as np

from gui.native_parallel_band import native_parallel_band_mask_xy

# Protect an interactive 4-GB GPU workstation from enormous single-step edits.
MAX_BAND_EDITS = 8_000_000


class NativeBandError(RuntimeError):
    """A native query or commit cannot safely proceed."""


def select_band_rows(store, polyline_xy, left_tol, right_tol):
    """Return unique STORE rows inside the original finite-segment band.

    Each segment uses its own conservative XY bounding box, rather than one
    enormous polyline bounding box.  The SectionStore deduplicates exact native
    node/local identities across overlapping queries. Query failures are fatal,
    preventing partial classification of a multi-segment source.
    """
    line = np.asarray(polyline_xy, dtype=np.float64)
    if line.ndim != 2 or line.shape[1] < 2 or len(line) < 2:
        raise NativeBandError('Select an open line or polyline with at least two vertices.')
    line = line[:, :2]
    left, right = float(left_tol), float(right_tol)
    if (not np.isfinite(line).all() or not all(math.isfinite(v) for v in (left, right))
            or min(left, right) < 0):
        raise NativeBandError('Invalid geometry or negative/nonfinite band width.')
    halo = max(left, right)
    matched = []
    for a, b in zip(line[:-1], line[1:]):
        if float(np.dot(b-a, b-a)) <= 1.e-12:
            continue
        # At UTM-scale world coordinates keep query bounds conservatively open.
        low = np.nextafter(np.minimum(a, b) - halo, -np.inf)
        high = np.nextafter(np.maximum(a, b) + halo, np.inf)
        if not store.ensure_region((float(low[0]), float(low[1]), -1.e12),
                                   (float(high[0]), float(high[1]), 1.e12)):
            raise NativeBandError(getattr(store, 'last_error', '') or 'Native LNT query failed.')
        rows = np.asarray(store.last_region_indices, dtype=np.int64)
        if rows.size == 0:
            continue
        points = store.data['xyz'][rows]
        inside = native_parallel_band_mask_xy(points, (a, b), left, right)
        if np.any(inside):
            matched.append(rows[inside])
    if not matched:
        return np.empty(0, dtype=np.int64)
    out = np.unique(np.concatenate(matched))
    if out.size > MAX_BAND_EDITS:
        raise NativeBandError(f'Band contains {out.size:,} points, over the '
                              f'{MAX_BAND_EDITS:,} point safety limit; use a shorter source line.')
    return out


def classify_lnt_parallel_band(app, polyline_xy, left_tol, right_tol,
                               from_class, to_class, *, store=None):
    """Classify exact native LNT points; return {band_points, changed_points}.

    A missing native interface or a failed query is a hard error, not a fall
    back to partial rendered LOD points.  The file-backed .lnt working dataset
    is changed, not the original .las/.laz source; exporting that source is a
    separate workflow.  A successful flush is required before reporting success.
    """
    view = getattr(app, '_lnt_main_view', None)
    if view is None or getattr(view, 'native', None) is None:
        raise NativeBandError('No native LNT/LAZ point cloud is open.')
    if store is None:
        from gui.lnt_section_bridge import get_store
        store = get_store(app)
    if store is None:
        raise NativeBandError('Native point query/classification API unavailable.')
    if app.data is not store.data:
        raise NativeBandError('Classification store no longer owns app.data; retry after reloading.')
    # LNT section queries expose node/local/class but not LAS point_source_id.
    # A hidden-flight-line selection cannot be reproduced safely from those
    # records: refuse rather than silently classifying invisible flight lines.
    try:
        from gui.flight_line_filter import flight_line_visibility_for_slot
        line_visibility = flight_line_visibility_for_slot(app, slot=0)
    except ImportError:
        line_visibility = {}
    if any(not shown for shown in line_visibility.values()):
        source_ids = store.data.get('point_source_id')
        if source_ids is None or len(source_ids) != len(store.data['classification']):
            raise NativeBandError('Some flight lines are hidden, but native LNT queries '
                                  'do not expose point_source_id. Enable all flight lines '
                                  'or classify through a source-aware workflow.')

    to_class = int(to_class)
    if not 0 <= to_class <= 255:
        raise NativeBandError('Target class must be between 0 and 255.')
    if from_class is not None:
        from_class = int(from_class)
        if not 0 <= from_class <= 255:
            raise NativeBandError('Source class must be between 0 and 255.')

    rows = select_band_rows(store, polyline_xy, left_tol, right_tol)
    band_points = int(rows.size)
    if not band_points:
        return {'band_points': 0, 'changed_points': 0}
    classes = np.asarray(store.data['classification'])
    if classes.dtype != np.uint8:
        raise NativeBandError('Native classification data must be uint8.')
    if from_class is not None:
        rows = rows[classes[rows] == from_class]
    if any(not shown for shown in line_visibility.values()):
        from gui.flight_line_filter import filter_visible_flight_line_indices
        rows = filter_visible_flight_line_indices(app, rows, length=len(classes), slot=0)
    palette = getattr(app, 'class_palette', None) or {}
    if palette:
        shown = np.zeros(256, dtype=bool)
        for code, rule in palette.items():
            try:
                if 0 <= int(code) <= 255 and (not isinstance(rule, dict) or rule.get('show', True)):
                    shown[int(code)] = True
            except (ValueError, TypeError):
                continue
        rows = rows[shown[classes[rows]]]
    rows = rows[classes[rows] != to_class]
    if not rows.size:
        return {'band_points': band_points, 'changed_points': 0}
    old = classes[rows].copy()
    new = np.full(len(rows), to_class, dtype=np.uint8)
    classes[rows] = new
    try:
        # Same preview + native write-back path as the existing SectionClassifier.
        store.show(rows, old)
        store.flush()
        synced = getattr(store, '_synced', None)
        if synced is None or not np.array_equal(synced[rows], new):
            raise NativeBandError('Native LNT classification write was not confirmed. '
                                  'Do not export until the pending edits are resolved: '
                                  + str(getattr(store, 'last_error', 'unknown error')))
    except Exception as exc:
        # On failure, leave data pending for the SectionStore to retry/flush,
        # rather than falsely claiming that file data was rolled back.
        if isinstance(exc, NativeBandError):
            raise
        raise NativeBandError('Native classification write failed: ' + str(exc)) from exc
    stack = getattr(app, 'undo_stack', None)
    if stack is None:
        stack = app.undo_stack = []
    stack.append({'indices': rows.copy(), 'old_classes': old, 'new_classes': new})
    redo = getattr(app, 'redo_stack', None)
    if redo is not None:
        redo.clear()
    try:
        from gui.memory_manager import trim_undo_stack
        trim_undo_stack(app)
    except (ImportError, AttributeError):
        pass
    app._last_changed_indices = rows.copy()
    app._last_from_classes = [] if from_class is None else [from_class]
    app._last_classify_ts = time.time()
    app.classification_revision = int(getattr(app, 'classification_revision', 0) or 0) + 1
    return {'band_points': band_points, 'changed_points': int(len(rows))}
