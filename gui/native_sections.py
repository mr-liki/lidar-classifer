"""Native Vulkan section docks for an open LNT dataset.

Only the queried corridor is decoded for editing. Every selected leaf record
keeps its original node/local identity in LntSectionStore; rendering never
constructs VTK datasets, actors, cameras or interactors.
"""
from __future__ import annotations

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import QPoint, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import QWidget, QDockWidget, QVBoxLayout, QLabel, QMessageBox
from PySide6.QtWidgets import QComboBox, QDialog

from gui.render_backend import VulkanRenderBackend
from gui.lnt_section_bridge import get_store, LntCrossPicker
from gui.native_section_classify import SectionClassifier
from gui.native_pan import NativePanGesture, disarm_pan
from gui.native_section_display import (MESH_MODES, POINT_MODES, build_mesh_data, recolor_mesh,
                                       rainbow, visibility, draw_ranges, line_colors,
                                       depth_colors, color_range)

_mesh_workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix='section_mesh')

CROSS_SECTION_COUNT = 8
CUT_SLOT = 9


class _CrossViewSelectorDialog(QDialog):
    """Small selector whose position survives for this application session."""

    def __init__(self, app):
        super().__init__(app)
        self._app = app
        self._track_position = False

    def moveEvent(self, event):  # noqa: N802 - Qt API
        super().moveEvent(event)
        if self._track_position:
            self._app._native_cross_view_selector_position = QPoint(self.pos())


def _place_cross_view_selector(dialog, app):
    saved = getattr(app, '_native_cross_view_selector_position', None)
    if saved is not None:
        dialog.move(QPoint(saved))
        return
    # First appearance in this process: centre over the application window.
    centre = app.frameGeometry().center()
    dialog.move(centre - dialog.rect().center())


def _set_cross_ribbon_checked(app, checked):
    """Mirror the armed state on the Tools ▸ Sections ▸ Cross toggle."""
    try:
        ribbon = getattr(app, "ribbon_manager", None)
        tools = getattr(getattr(ribbon, "ribbons", None), "get", lambda *_: None)("tools")
        btn = getattr(tools, "cross_section_btn", None)
        if btn is not None and hasattr(btn, "setChecked"):
            btn.blockSignals(True)
            try:
                if bool(btn.isChecked()) != bool(checked):
                    btn.setChecked(bool(checked))
            finally:
                btn.blockSignals(False)
    except Exception:  # noqa: BLE001 - ribbon purely cosmetic
        pass


def manager(app, create=True):
    if getattr(app, '_lnt_main_view', None) is None:
        return None
    result = getattr(app, '_native_sections', None)
    if result is None and create:
        result = NativeSections(app)
        app._native_sections = result
    return result


try:   # one compiled pass for multi-million point corridors; numpy below otherwise
    from numba import njit as _njit
except Exception:  # noqa: BLE001
    _njit = None

if _njit is not None:
    @_njit(cache=True, nogil=True)
    def _corridor_kernel(xyz, p1x, p1y, dx, dy, length, half_width, buffer):
        n = xyz.shape[0]
        keep = np.empty(n, dtype=np.int64)
        count = 0
        for i in range(n):
            rx = xyz[i, 0] - p1x
            ry = xyz[i, 1] - p1y
            along = rx * dx + ry * dy
            across = ry * dx - rx * dy
            if along >= -buffer and along <= length + buffer and abs(across) <= half_width:
                keep[count] = i
                count += 1
        indices = keep[:count].copy()
        local = np.empty((count, 3), dtype=np.float64)
        core = np.empty(count, dtype=np.bool_)
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        if count == 0:
            return indices, local, core, np.zeros(3), np.zeros(3)
        for j in range(count):
            i = indices[j]
            rx = xyz[i, 0] - p1x
            ry = xyz[i, 1] - p1y
            v0 = rx * dx + ry * dy
            v1 = ry * dx - rx * dy
            v2 = xyz[i, 2]
            local[j, 0] = v0
            local[j, 1] = v1
            local[j, 2] = v2
            core[j] = v0 >= 0. and v0 <= length
            if v0 < lo[0]: lo[0] = v0
            if v0 > hi[0]: hi[0] = v0
            if v1 < lo[1]: lo[1] = v1
            if v1 > hi[1]: hi[1] = v1
            if v2 < lo[2]: lo[2] = v2
            if v2 > hi[2]: hi[2] = v2
        return indices, local, core, lo, hi


def section_geometry(xyz, p1, p2, half_width, buffer, bounds=False):
    """Exact double precision corridor test; endpoints/boundaries inclusive.
    Returns (indices, local, core) or, with bounds=True, also (lo, hi) of local."""
    p1, p2 = np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64)
    delta = p2[:2] - p1[:2]
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        raise ValueError('Section line has zero length')
    dx, dy = (float(v) for v in delta / length)
    if _njit is not None and len(xyz):
        indices, local, core, lo, hi = _corridor_kernel(
            np.ascontiguousarray(xyz, dtype=np.float64), float(p1[0]), float(p1[1]),
            dx, dy, length, float(half_width), float(buffer))
    else:
        # along = r.d, across = r.n with n = (-dy, dx); column-wise temporaries
        rx = xyz[:, 0] - p1[0]
        ry = xyz[:, 1] - p1[1]
        along = rx * dx
        along += ry * dy
        across = ry * dx
        across -= rx * dy
        selected = along >= -buffer
        selected &= along <= length + buffer
        selected &= np.abs(across) <= half_width
        indices = np.flatnonzero(selected)
        local = np.empty((len(indices), 3), dtype=np.float64)
        local[:, 0], local[:, 1], local[:, 2] = along[indices], across[indices], xyz[indices, 2]
        core = (local[:, 0] >= 0) & (local[:, 0] <= length)
        lo = local.min(axis=0) if len(local) else np.zeros(3)
        hi = local.max(axis=0) if len(local) else np.zeros(3)
    return (indices, local, core, lo, hi) if bounds else (indices, local, core)


class SectionCanvas(QWidget):
    mesh_ready = Signal(int, object, str)
    lines_ready = Signal(int, object, str)

    def __init__(self, owner, slot, parent):
        super().__init__(parent)
        self.owner, self.slot = owner, slot
        self.backend = VulkanRenderBackend()
        self.setAttribute(Qt.WA_NativeWindow)
        self.setAttribute(Qt.WA_PaintOnScreen)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setMinimumSize(180, 160)
        self.setFocusPolicy(Qt.StrongFocus)
        self.centre = np.zeros(3, dtype=np.float64)
        self.scale = 1.
        self.indices = np.empty(0, dtype=np.int64)
        self.local = np.empty((0, 3), dtype=np.float64)
        self._lo = self._hi = np.zeros(3, dtype=np.float64)
        self.axis = 0
        self.drag = None
        self.pan_gesture = NativePanGesture(self._pan_by)
        self.cut_armed = False
        self.cut_centre = None
        self.cut_source = None
        self.mode = 'class'
        self.palette = {}
        self._render_palette = {}      # the same palette keyed by stored slot
        self.shading_parameters = (45., 45., .25)
        self._display_revision = 0
        self._mesh_future = None
        self.mesh_ready.connect(self._finish_mesh, Qt.QueuedConnection)
        self.lines_ready.connect(self._finish_lines, Qt.QueuedConnection)
        self._source_ids = None
        # Geometry/shade cache for classification-only edits: recolouring an
        # existing mesh must never clear the viewport or re-triangulate.
        self._mesh_data = None
        self._mesh_mask = None
        self._mesh_key = None
        self._pending_mesh_revision = -1
        self._pending_mesh_mask = None
        self._pending_mesh_key = None
        self._surface_loaded = False      # the resident mesh matches the request
        self._surface_resident = False    # the engine holds a mesh on the GPU
        self._mesh_build_count = 0
        self._mesh_recolor_count = 0
        self._indices_sorted = True
        # Sparse classification row -> canvas-position lookup.  The section
        # store accumulates regions, so later section queries are not guaranteed
        # to return monotonically increasing store rows.  Falling back to a full
        # million-row compare whenever that happened was a major long-session
        # regression.  Build a sorted inverse lazily only for unsorted views.
        self._index_lookup_values = None
        self._index_lookup_pos = None
        self._dirty = True
        self._closed = False
        self._cut_preview = None
        self._bound_hwnd = 0               # native window the engine renders into
        self.setMouseTracking(True)
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.render)
        self.timer.start()
        self.classifier = SectionClassifier(self)
        self.shortcuts = []
        for key, callback in [('Shift+1', lambda: owner.arm_cut(False, self.slot)),
                              ('Shift+2', lambda: owner.arm_cut(True)),
                              ('Escape', self.escape), ('F', self.fit)]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

    def escape(self):
        # This QShortcut consumes Esc before any key filter sees it, so the
        # armed Pan tool must be disarmed here too.
        self.pan_gesture.finish()
        disarm_pan(self.owner.app)
        # Same for an armed measurement (XS): its Escape stands the whole XS
        # tool down, which keyPressEvent would never get to do here.
        measurement = getattr(self, 'measurement', None)
        if measurement is not None and measurement.key(Qt.Key_Escape):
            return
        # QShortcut activation bypasses keyPressEvent and the global KeyPress
        # filter. Route classification through its full Escape handler here as
        # well, including when the armed tool has no unfinished gesture.
        if self.classifier.active():
            self.classifier.key(Qt.Key_Escape)
            return
        if self.classifier.busy():
            self.classifier.cancel()
        else:
            self.owner.cancel_cut()

    def paintEngine(self):
        return None

    def paintEvent(self, event):
        self._dirty = True

    def start(self):
        if self._closed:
            return False
        hwnd = int(self.winId())
        rebind = bool(self.backend._handle) and hwnd != self._bound_hwnd
        if rebind:
            # Closing a section dock and taking the section again reparents
            # this canvas, which gives it a NEW native window. The engine's
            # swapchain is still bound to the destroyed one, so it renders but
            # never presents (blank window). Rebuild it on the current window.
            self.backend.shutdown()
        if not self.backend._handle:
            ratio = self.devicePixelRatioF()
            if not self.backend.initialize(hwnd,
                    max(1, round(self.width() * ratio)), max(1, round(self.height() * ratio))):
                raise RuntimeError('Could not create native section viewport: ' + self.backend.last_error())
            self._bound_hwnd = hwnd
            if rebind and len(self.local):
                # New engine instance: upload the same records, palette and
                # display mode again (set_points re-enters start(), now bound).
                self.set_points(self.local, self.indices, self.axis, fit=False,
                                bounds=(self._lo, self._hi))
            if self._cut_preview is not None:   # reopened window: redraw the preview
                self._draw_cut_preview()
            self._dirty = True
        return True

    def resizeEvent(self, event):
        if self.backend._handle:
            ratio = self.devicePixelRatioF()
            self.backend.resize(max(1, round(self.width() * ratio)), max(1, round(self.height() * ratio)))
        self._dirty = True
        super().resizeEvent(event)

    def set_points(self, local, indices, axis=0, fit=True, bounds=None):
        # Keep source rows for undo, including an entirely deleted section.
        if local is self.local and indices is self.indices and hasattr(self, '_deletion_source'):
            local, indices = self._deletion_source
        self._deletion_source = (local, indices)
        store = get_store(self.owner.app)
        if store is not None and hasattr(store, 'deleted'):
            keep = ~store.deleted[np.asarray(indices, dtype=np.int64)]
            local, indices = np.asarray(local)[keep], np.asarray(indices)[keep]
        if not np.array_equal(self.indices, indices):
            self._source_ids = None
        # New records/axis: any cached triangulation is no longer valid, and a
        # mesh from the previous corridor must not float over the new points.
        self._drop_mesh_cache()
        self._retire_mesh()
        self.local = np.ascontiguousarray(local, dtype=np.float64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self._indices_sorted = bool(len(self.indices) < 2 or np.all(np.diff(self.indices) > 0))
        self._index_lookup_values = None
        self._index_lookup_pos = None
        self.axis = axis
        if not self.start():
            return
        if not len(self.local):
            # Nothing in the corridor/cut: show an empty view (the engine
            # refuses a zero-point upload) and keep the old buffer hidden.
            self.backend.set_point_cloud_visible(False)
            self.apply_palette(self.palette, self.mode, sync=False)
            self.classifier.cancel()
            self._dirty = True
            return
        self.backend.set_point_cloud_visible(True)
        # Front view displays `local` as is (no copy); side view swaps columns.
        display = self.local if axis == 0 else np.ascontiguousarray(self.local[:, [1, 0, 2]])
        self._lo, self._hi = ((np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64))
                              if bounds is not None else (self.local.min(axis=0), self.local.max(axis=0)))
        store = get_store(self.owner.app)
        idx = self.indices
        first = int(idx[0])
        span = slice(first, first + len(idx)) if (
            int(idx[-1]) - first + 1 == len(idx) and bool(np.all(np.diff(idx) == 1))) else None
        gather = (lambda a: a[span]) if span is not None else (lambda a: a[idx])
        rgb8 = getattr(store, 'rgb8', None)
        if rgb8 is not None and len(rgb8) == len(store.data['rgb']):
            rgb = gather(rgb8)
        else:   # 0..1 floats -> the uint8 the engine expects (a plain cast would floor to 0/1)
            rgb = np.clip(gather(store.data['rgb']) * 255. + .5, 0, 255).astype(np.uint8)
        if not self.backend.set_point_cloud(display, rgb=rgb,
                classification=gather(store.data['classification']),
                intensity=gather(store.data['intensity'])):
            raise RuntimeError('Native section point upload failed')
        self.backend.set_point_size(3.)
        self.apply_palette(self.palette, self.mode, sync=False)
        if fit:
            self.fit(sync=False)
        self._dirty = True

    # ---- mesh refresh (classification edits must never flash the viewport) --
    def _drop_mesh_cache(self):
        self._mesh_data = self._mesh_mask = self._mesh_key = None
        self._pending_mesh_revision = -1
        self._pending_mesh_mask = self._pending_mesh_key = None

    def _retire_mesh(self):
        """Remove the resident mesh (leaving a mesh mode / new dataset)."""
        self._drop_mesh_cache()
        if getattr(self, '_closed', False):
            self._surface_loaded = self._surface_resident = False
            return
        if getattr(self, '_surface_loaded', False) or getattr(self, '_surface_resident', False):
            if not self.backend.clear_surface():
                raise RuntimeError('Could not clear the previous section mesh')
        self._surface_loaded = self._surface_resident = False
        self._dirty = True

    def positions_for_store_rows(self, rows):
        """Canvas positions for store-row ids, ``-1`` for rows not in this view.

        Fast path is a direct search on already sorted section indices.  For a
        later/overlapping section whose store rows arrive out of order, create one
        lazy sorted inverse and reuse it for every classification, brush frame,
        undo and redo.
        """
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        out = np.full(rows.size, -1, dtype=np.int64)
        if not rows.size or not len(self.indices):
            return out
        if self._indices_sorted:
            at = np.searchsorted(self.indices, rows)
            hit = at < len(self.indices)
            if np.any(hit):
                valid = np.flatnonzero(hit)
                exact = self.indices[at[valid]] == rows[valid]
                valid = valid[exact]
                out[valid] = at[valid]
            return out

        values = self._index_lookup_values
        order = self._index_lookup_pos
        if values is None or order is None or len(values) != len(self.indices):
            order64 = np.argsort(self.indices, kind='stable')
            values = np.ascontiguousarray(self.indices[order64], dtype=np.int64)
            order = np.asarray(order64, dtype=(np.int32 if len(self.indices) < 2**31 else np.int64))
            self._index_lookup_values = values
            self._index_lookup_pos = order
        at = np.searchsorted(values, rows)
        hit = at < len(values)
        if np.any(hit):
            valid = np.flatnonzero(hit)
            exact = values[at[valid]] == rows[valid]
            valid = valid[exact]
            out[valid] = order[at[valid]]
        return out

    def classes_all_visible(self, app=None):
        """True when a class edit cannot change this view's visibility mask."""
        app = self.owner.app if app is None else app
        if getattr(app, '_import_class_blocked', ()):
            return False
        return not any(not rule.get('show', True) for rule in self.palette.values())

    def _refresh_point_classification(self, classes):
        """Update classification-dependent point state without re-applying a display preset.

        Classification edits never change RGB/intensity/elevation/depth LUTs, point
        weights, source-line colours or the selected display mode.  Re-running
        ``apply_palette`` after every click needlessly re-uploaded those resources.
        This path uploads the one-byte class stream and, only when class visibility
        can actually change, recomputes draw ranges.
        """
        if self.mode in MESH_MODES:
            return False
        if not self.backend._handle:
            self._dirty = True
            return True
        classes = np.ascontiguousarray(classes, dtype=np.uint8).reshape(-1)
        if classes.size != len(self.indices):
            return False
        if not self.backend.set_point_attributes(classification=classes):
            return False

        app = self.owner.app
        visibility_can_change = (
            not self.classes_all_visible(app)
            or bool(getattr(app, '_import_class_blocked', ()))
        )
        if visibility_can_change:
            from gui.flight_line_filter import flight_line_visibility_for_slot
            store = get_store(app)
            source = store.data.get('point_source_id') if store is not None else None
            source_ids = (source[self.indices]
                          if source is not None and store is not None
                          and len(source) == len(store.data['xyz'])
                          else self._source_ids)
            lines = flight_line_visibility_for_slot(app, self.slot)
            if source_ids is None and (self.mode == 'line' or any(not shown for shown in lines.values())):
                return False
            mask = visibility(
                classes, self._render_palette, getattr(app, '_import_class_blocked', ()),
                source_ids, lines
            )
            firsts, counts = draw_ranges(mask)
            self.backend.set_point_draw_ranges(firsts, counts)
            self.backend.set_point_cloud_visible(bool(np.any(mask)))

        self._dirty = True
        return True

    def _mesh_options(self, app, mode):
        return dict(
            light=(float(getattr(app, 'last_shade_azimuth', 45.)),
                   float(getattr(app, 'last_shade_angle', 45.)),
                   float(getattr(app, 'shade_ambient', .25))),
            quality=getattr(app, 'shading_quality' if mode == 'shaded_class' else 'surface_quality', 'normal'),
            precision=float(getattr(app, 'surface_dedup_precision', 0.) or 0.),
            max_edge=float(getattr(app, 'surface_max_edge', 0.) or 0.),
            ramp=getattr(app, 'surface_color_ramp', None) or getattr(app, 'elevation_color_ramp', None))

    @staticmethod
    def _mesh_signature(mode, options):
        """Everything a triangulation depends on, except the record mask."""
        ramp = options.get('ramp') or ()
        try:
            ramp = tuple((float(stop[0]), tuple(int(c) for c in stop[1]))
                         for stop in sorted(ramp, key=lambda stop: stop[0]))
        except (TypeError, ValueError):
            ramp = repr(ramp)
        return (mode, options['quality'], options['light'], options['precision'],
                options['max_edge'], ramp)

    def _update_mesh(self, store, mask, classes):
        """Keep a shaded/surface mesh on screen across classification edits.

        A class edit normally changes only class ids, never the triangulation,
        so the resident mesh is recoloured from cached geometry (Surface bakes
        elevation, so it needs no upload at all). Only a real geometry change
        starts a worker, and the previous mesh stays visible until the new one
        is uploaded - the viewport is never cleared between the two.
        """
        app = self.owner.app
        options = self._mesh_options(app, self.mode)
        signature = self._mesh_signature(self.mode, options)
        cached = (self._mesh_data is not None and self._mesh_key == signature
                  and self._mesh_mask is not None and self._mesh_mask.shape == mask.shape
                  and bool(np.all(self._mesh_mask == mask)))
        if cached:
            if self._mesh_future is not None:
                self._mesh_future.cancel()
                self._mesh_future = None
            if self._recolor_resident(classes[mask]):
                return
        pending = (self._mesh_future is not None
                   and self._pending_mesh_revision == self._display_revision
                   and self._pending_mesh_key == signature
                   and self._pending_mesh_mask is not None
                   and bool(np.all(self._pending_mesh_mask == mask)))
        if pending:
            return
        if not np.any(mask):
            self._retire_mesh()
            return
        idx = self.indices
        xyz = np.ascontiguousarray(store.data['xyz'][idx][mask], dtype=np.float64)
        options['z_bounds'] = tuple(np.percentile(xyz[:, 2], [1., 99.]))
        display = self.local if self.axis == 0 else self.local[:, [1, 0, 2]]
        # Workers receive snapshots only. All renderer mutations happen in
        # _finish_mesh on the GUI thread, guarded by a display revision.
        revision = self._display_revision
        future = _mesh_workers.submit(build_mesh_data, xyz, np.ascontiguousarray(display[mask]),
                                      classes[mask].copy(), self._render_palette, self.mode, options)
        self._mesh_future = future
        self._pending_mesh_revision = revision
        self._pending_mesh_key = signature
        self._pending_mesh_mask = np.ascontiguousarray(mask)
        # The old mesh (or the point cloud, on first entry) stays on screen.
        self._surface_loaded = False
        self._dirty = True
        def complete(done):
            if done.cancelled():
                return
            try:
                result, error = done.result(), ''
            except Exception as exc:
                result, error = None, str(exc)
            try:
                self.mesh_ready.emit(revision, result, error)
            except RuntimeError:
                pass  # the dock was destroyed while its worker completed
        future.add_done_callback(complete)
        app.statusBar().showMessage(f'Building section {self.slot} {self.mode.replace("_", " ")}...', 3000)

    def _recolor_resident(self, classes_masked):
        """Re-bake the resident mesh's face colours; geometry is untouched."""
        data = self._mesh_data
        if data is None:
            return False
        if self.mode == 'surface':
            # Surface colours bake elevation and light only: a class edit
            # cannot change a single pixel.
            self._surface_loaded = self._surface_resident = True
            self._dirty = True
            return True
        colors = recolor_mesh(data, classes_masked, self._render_palette)
        if colors is None:
            return False
        update = getattr(self.backend, 'update_surface_colors', None)
        if update is None or not update(colors):
            # Compatibility with older engine DLLs.
            if not self.backend.set_surface(data['positions'], data['faces'], colors):
                return False
        data['colors'] = colors
        self._mesh_recolor_count += 1
        self._present_mesh()
        return True

    def _present_mesh(self):
        """Show a resident mesh: flat pass-through colours, points hidden."""
        self.backend.set_display_mode(1)
        self.backend.set_shading_parameters(0, 45., 45., 1.)
        self.backend.set_point_cloud_visible(False)
        self._surface_loaded = self._surface_resident = True
        self._dirty = True

    def apply_palette(self, palette, mode='class', sync=False):
        if mode not in POINT_MODES and mode not in MESH_MODES:
            raise ValueError(f'Unknown section display mode: {mode}')
        self.palette = {int(k): dict(v) for k, v in (palette or {}).items()}
        self.mode = mode
        # Native tables are keyed by the stored one-byte class slot, so user
        # numbers above 255 keep their colour, weight and visibility.
        from gui.class_slots import stored_palette
        self._render_palette = stored = stored_palette(self.owner.app, self.palette)
        self._display_revision += 1
        if self._mesh_future is not None:
            self._mesh_future.cancel()
            self._mesh_future = None
        if not self.backend._handle:
            return
        if not len(self.indices):
            self._retire_mesh()
            self._dirty = True
            return
        store = get_store(self.owner.app)
        app = self.owner.app
        idx = self.indices
        classes = np.asarray(store.data['classification'][idx], dtype=np.uint8)
        from gui.flight_line_filter import flight_line_visibility_for_slot
        source = store.data.get('point_source_id')
        source_ids = source[idx] if source is not None and len(source) == len(store.data['xyz']) else None
        if source_ids is None:
            source_ids = self._source_ids
        lines = flight_line_visibility_for_slot(app, self.slot)
        if source_ids is None and (mode == 'line' or any(not shown for shown in lines.values())):
            self._load_source_lines(store)
            return
        mask = visibility(classes, stored, getattr(app, '_import_class_blocked', ()), source_ids, lines)
        if mode == 'surface':
            for code, entry in stored.items():
                support = entry.get('surface', entry.get('surface_support', True))
                if support is not None and not support:
                    mask &= classes != code
        if mode in MESH_MODES:
            # The mesh owns the viewport; _update_mesh keeps whatever is on
            # screen until the replacement is uploaded, so an edit or a mode
            # switch never blanks the view.
            self._update_mesh(store, mask, classes)
            return
        self._retire_mesh()
        self.backend.set_point_attributes(classification=classes)
        firsts, counts = draw_ranges(mask)
        self.backend.set_point_draw_ranges(firsts, counts)
        if not np.any(mask):
            # An empty range list tells the engine to draw the whole buffer,
            # so hiding every class must hide the cloud (and its border
            # shells) explicitly - otherwise the view shows everything.
            self.backend.set_point_cloud_visible(False)
            self._dirty = True
            return
        display = self.local if self.axis == 0 else self.local[:, [1, 0, 2]]
        colors = np.full((256, 3), 128, dtype=np.uint8)
        weights = np.ones(256, dtype=np.float32)
        for code, rule in stored.items():   # stored slot (user > 255 mapped)
            if 0 <= code < 256:
                colors[code] = rule.get('color', (128, 128, 128))
                try:
                    weights[code] = max(.1, min(float(rule.get('weight', 1.)), 12.))
                except (ValueError, TypeError):
                    pass
        self.backend.set_point_luts(class_rgb=colors,
            elevation_rgb=rainbow(np.linspace(0, 1, 256), getattr(app, 'elevation_color_ramp', None)),
            intensity_rgb=np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1))
        # Visibility already narrowed the draw ranges above, so every class
        # left in them is shown; slots keep the LUT/resolution consistent.
        self.backend.set_class_visibility(np.ones(256, dtype=np.uint8))
        self.backend.set_class_weights(weights)
        # Line mode uses source IDs, never classification IDs. Restore original
        # RGB when leaving Line so Line colours cannot leak into RGB mode.
        if mode in ('line', 'depth') or getattr(self, '_point_colors_mode', '') in ('line', 'depth'):
            if mode == 'line' and source_ids is None:
                app.statusBar().showMessage('Line mode needs Point Source ID attributes; this LNT section has none.', 8000)
                return
            if mode == 'line':
                rgb = line_colors(source_ids, getattr(app, 'flight_line_colors', {}))
            elif mode == 'depth':
                rgb = depth_colors(display[:, 1], app)
            else:
                rgb8 = getattr(store, 'rgb8', None)
                rgb = rgb8[idx] if rgb8 is not None else np.clip(store.data['rgb'][idx] * 255. + .5, 0, 255).astype(np.uint8)
            if not self.backend.set_point_cloud(np.ascontiguousarray(display), rgb=rgb,
                    classification=classes, intensity=store.data['intensity'][idx]):
                raise RuntimeError('Could not upload section flight-line colours')
            self.backend.set_point_draw_ranges(firsts, counts)
        self._point_colors_mode = mode
        z = self.local[mask, 2]
        lo, hi = color_range(z, float(getattr(app, 'elevation_clip_low', 1.)),
                            float(getattr(app, 'elevation_clip_high', 99.)))
        self.backend.set_point_elevation_range(float(lo), max(float(hi), float(lo) + 1e-6))
        intensity = store.data['intensity'][idx][mask]
        lo, hi = color_range(intensity, float(getattr(app, 'intensity_clip_low', .5)),
                            float(getattr(app, 'intensity_clip_high', 99.8)), ignore_zero=True)
        self.backend.set_point_intensity_range(float(lo), max(float(hi), float(lo) + 1e-6),
            float(getattr(app, 'intensity_contrast', 1.)), float(getattr(app, 'intensity_gamma', 1.35)))
        self.backend.set_display_mode(0 if mode == 'depth' else POINT_MODES[mode])
        self.backend.set_point_cloud_visible(True)
        from gui.native_border import apply_border
        apply_border(self.owner.app, self.slot)
        self._dirty = True

    @Slot(int, object, str)
    def _finish_lines(self, revision, result, error):
        if self._closed or revision != self._display_revision:
            return
        if error:
            self.owner.app.statusBar().showMessage(f'Section {self.slot} Line mode: {error}', 10000)
            return
        self._source_ids = result
        self.apply_palette(self.palette, self.mode, sync=False)
        dialog = getattr(self.owner.app, 'display_mode_dialog', None)
        if getattr(self, '_open_lines_when_ready', False):
            self._open_lines_when_ready = False
            if dialog is not None and dialog.current_slot == self.slot:
                QTimer.singleShot(0, dialog._open_lines_dialog)

    def _load_source_lines(self, store):
        from gui.native_flight_lines import prepare
        app=self.owner.app
        view=getattr(app,'_lnt_main_view',None)
        if view is not None and getattr(view,'flight_lines',None) is not None:
            from gui.lnt_class_counts import read_directory
            _, entries=read_directory(view.lnt_path)
            self._source_ids=view.flight_lines.ids(store.node[self.indices],store.local[self.indices],entries)
            self.apply_palette(self.palette,self.mode,sync=False)
            if getattr(self,'_open_lines_when_ready',False):
                self._open_lines_when_ready=False
                dialog=getattr(app,'display_mode_dialog',None)
                if dialog is not None: QTimer.singleShot(0,dialog._open_lines_dialog)
            return
        if view is not None and hasattr(view,'app') and hasattr(store,'node'):
            prepare(view,lambda:self._load_source_lines(store) if not self._closed else None)
            return
        import os
        from gui.native_section_display import source_line_ids
        from gui.lnt_class_counts import read_directory
        app = self.owner.app
        path = str(getattr(app, '_lnt_source_path', '') or '')
        if not path or not os.path.isfile(path) or not path.lower().endswith(('.las', '.laz', '.lntset')):
            app.statusBar().showMessage('Line mode requires the original LAS/LAZ; LNT v2 does not store Point Source IDs.', 10000)
            return
        # Bound reconstruction error using the selected nodes' 16-bit cell step.
        lnt_path = app._lnt_main_view.lnt_path
        _, directory = read_directory(lnt_path)
        half = float(directory['half_size'][store.node[self.indices]].max())
        tolerance = 2. * half / 65535. * np.sqrt(3.) * .501
        if path.lower().endswith('.lntset'):
            # Collection directory centres share a common float32 origin.
            centres = directory['center'][store.node[self.indices]]
            tolerance += float(np.abs(np.spacing(centres)).max()) * np.sqrt(3.) * .501
        xyz = np.ascontiguousarray(store.data['xyz'][self.indices], dtype=np.float64)
        revision = self._display_revision
        future = _mesh_workers.submit(source_line_ids, path, xyz, tolerance,
                                      getattr(app, '_lnt_render_crs_wkt', None))
        self._mesh_future = future
        def complete(done):
            if done.cancelled():
                return
            try:
                result, error = done.result(), ''
            except Exception as exc:
                result, error = None, str(exc)
            try:
                self.lines_ready.emit(revision, result, error)
            except RuntimeError:
                pass
        future.add_done_callback(complete)
        app.statusBar().showMessage(f'Reading source flight lines for section {self.slot}...', 3000)

    @Slot(int, object, str)
    def _finish_mesh(self, revision, result, error):
        if self._closed or revision != self._display_revision:
            return
        if error:
            self._retire_mesh()
            self.backend.set_point_cloud_visible(True)
            self.owner.app.statusBar().showMessage(f'Section {self.slot} mesh failed: {error}', 8000)
        elif result is None:
            self._retire_mesh()
            self.backend.set_point_cloud_visible(True)
            self.owner.app.statusBar().showMessage(f'Section {self.slot}: no triangles for the selected points.', 5000)
        elif not self.backend.set_surface(result['positions'], result['faces'], result['colors']):
            self._retire_mesh()
            self.backend.set_point_cloud_visible(True)
            self.owner.app.statusBar().showMessage(f'Section {self.slot}: Vulkan mesh upload failed.', 8000)
        else:
            self._mesh_data = result
            self._mesh_mask = self._pending_mesh_mask
            self._mesh_key = self._pending_mesh_key
            self._mesh_build_count += 1
            # Colours already include the exact legacy lighting formula, and
            # the swap above is atomic - the old frame was never blanked.
            self._present_mesh()
        self._dirty = True

    def fit(self, sync=True):
        if not len(self.local):
            self.scale = 1.
            self._dirty = True
            return
        order = [self.axis, 1-self.axis, 2]
        lo, hi = self._lo[order], self._hi[order]
        self.centre = (lo + hi) * .5
        aspect = max(self.width(), 1) / max(self.height(), 1)
        self.scale = max((hi[2]-lo[2])*.55, (hi[0]-lo[0])*.55/aspect, .01)
        self._dirty = True
        if sync:
            self.owner.sync_from(self.slot)

    def render(self):
        if self._closed or not self.isVisible() or not self._dirty:
            return
        if not self.start():
            return
        self.backend.set_camera_view_up(0., 0., 1.)
        self.backend.set_camera_ortho(self.centre, (0., 1., 0.), self.scale,
                                      .001, max(10000., self.scale*100.))
        presented = getattr(self.backend, 'present_count', 0)
        self.backend.request_render()
        self._dirty = getattr(self.backend, 'present_count', 0) == presented

    @property
    def frame_backend(self):
        return self.backend

    def projection_snapshot(self):
        """Freeze the rendered camera, rather than the next interaction target."""
        frame = getattr(self.backend, 'present_count', 0)
        ratio = self.devicePixelRatioF()
        token = (frame, self.width(), self.height(), ratio)
        if token == getattr(self, '_projection_token', None):
            return self._projection_value
        mvp = self.backend.get_last_mvp()
        if mvp is None:
            return None
        mvp = np.array(mvp, dtype=np.float64, copy=True)
        origin = np.array(self.backend.get_render_origin_xyz(), dtype=np.float64, copy=True)
        w, h = self.width(), self.height()
        def project(horizontal, elevation):
            # Section records are (horizontal, depth, elevation); depth does
            # not change XY in this orthographic view.
            x, z = float(horizontal) - origin[0], float(elevation) - origin[2]
            clip = mvp @ np.array((x, 0., z, 1.))
            if abs(float(clip[3])) < 1e-12:
                return -1.e9, -1.e9
            return ((clip[0] / clip[3] * .5 + .5) * w,
                    (clip[1] / clip[3] * .5 + .5) * h)
        self._projection_token = token
        self._projection_value = (mvp.tobytes(), origin.tobytes(), w, h), project
        return self._projection_value

    # Orthographic mapping (camera looks along +depth, up = elevation):
    # screen x -> horizontal, screen y -> elevation; logical Qt pixels.
    def world_per_pixel(self):
        return 2*self.scale/max(1, self.height())

    def horizontal_at(self, x):
        return self.centre[0] + (x-self.width()*.5) * self.world_per_pixel()

    def section_at(self, x, y):
        f = self.world_per_pixel()
        return (self.centre[0] + (x-self.width()*.5)*f,
                self.centre[2] - (y-self.height()*.5)*f)

    def screen_from_section(self, h, z):
        f = max(self.world_per_pixel(), 1e-300)
        return ((h-self.centre[0])/f + self.width()*.5,
                (self.centre[2]-z)/f + self.height()*.5)

    def selection_points(self):
        """Horizontal/elevation view of the records, without a per-gesture copy."""
        return self.local[:, ::2] if self.axis == 0 else self.local[:, 1:]

    def visible_extent(self):
        half_w = self.width()*.5*self.world_per_pixel()
        return (self.centre[0]-half_w, self.centre[0]+half_w,
                self.centre[2]-self.scale, self.centre[2]+self.scale)

    # ---- cut preview (engine line pass, naksha-lidar cut tool look) --------
    # Before the centre click a white centre line follows the cursor; after
    # it, the white line stays at the centre and two orange depth lines follow
    # the cursor at centre +/- depth. All lines span far past the view height.
    def show_cut_preview(self, centre, half=None):
        self._cut_preview = (float(centre), None if half is None else float(half))
        overlay = self.classifier.overlay
        shapes = [shape for shape in overlay.shapes if shape[0] != 'cut_preview']
        if not self._draw_cut_preview():
            # DLL without the vector pass: same lines through the Qt overlay.
            shapes.append(('cut_preview', [(self._cut_preview[0], self.centre[2])],
                           (self._cut_preview[1], self.slot == CUT_SLOT)))
        overlay.set_shapes(shapes)
        self._dirty = True

    def _draw_cut_preview(self):
        preview = getattr(self, '_cut_preview', None)
        backend = self.backend
        if preview is None or not backend._handle or not backend.supports_vector_overlay():
            return False
        centre, half = preview
        edge = 0. if half is None else half
        reach = max(1e4, self.scale*1e3)
        y, z0, z1 = float(self.centre[1]), self.centre[2]-reach, self.centre[2]+reach
        xs = (centre, centre, centre-edge, centre-edge, centre+edge, centre+edge)
        xyz = np.array([(x, y, z) for x, z in zip(xs, (z0, z1)*3)], dtype=np.float64)
        rgb = np.array([(255, 255, 255)]*2 + [(255, 165, 0)]*4, dtype=np.uint8)
        restart = 0xFFFFFFFF
        indices = np.array([0, 1, restart, 2, 3, restart, 4, 5], dtype=np.uint32)
        if not backend.vector_reserve(6, 8):
            return False
        if not backend.vector_upload_strip(0, 6, xyz, rgb, 0, 8, indices):
            return False
        # Keep the centre line equally visible in Cut-from-Cross and
        # Cut-in-Cut.  The old 1 px cross-view line disappeared against dense
        # towers/conductors while the cut-view's 2 px line remained obvious.
        centre_px = 2.
        if half is None:
            return backend.vector_set_draw_ranges([0], [2], [centre_px])
        # The depth edges coincide with the centre at zero/small depth. Draw
        # the white centre last so orange cannot paint over it.
        return backend.vector_set_draw_ranges([3, 0], [5, 2], [2., centre_px])

    def clear_cut_preview(self):
        if getattr(self, '_cut_preview', None) is None:
            return
        self._cut_preview = None
        if self.backend._handle:
            self.backend.vector_clear_draw_ranges()
        overlay = self.classifier.overlay
        overlay.set_shapes([shape for shape in overlay.shapes if shape[0] != 'cut_preview'])
        self._dirty = True

    def _pan_by(self, dx, dy):
        factor = 2*self.scale/max(1, self.height())
        self.centre[0] -= dx*factor
        self.centre[2] += dy*factor
        self._dirty = True
        self.owner.sync_from(self.slot)

    def _pan_tool_input(self, event):
        """Armed Pan shortcut (Left - Click / Tap - Tap); tools keep their clicks."""
        measurement = getattr(self, 'measurement', None)
        if (self.cut_armed or self.classifier.active()
                or (measurement is not None and getattr(measurement, 'active', False))):
            self.pan_gesture.finish()
            return False
        return self.pan_gesture.handle(self.owner.app, event, self)

    def mousePressEvent(self, event):
        self.setFocus()
        cross_cut_active = self.owner.route_cross_cut_to_canvas(
            self, event.position())
        # Once the centre has been placed, the operation is locked to its
        # source view. Do not let a click in another cross view start locate,
        # classification or another tool underneath the pending cut.
        if cross_cut_active and not self.cut_armed:
            event.accept()
            return
        self.owner.active_slot = self.slot
        if self._pan_tool_input(event):
            return
        measurement = getattr(self, 'measurement', None)
        if measurement is not None and measurement.press(event.position(), event.button()):
            return
        if not self.cut_armed and self.classifier.press(event.position(), event.button()):
            return
        if (not self.cut_armed and event.button() == Qt.LeftButton
                and self.owner.locate(self, event.position())):
            return
        if event.button() == Qt.RightButton:
            # An armed cut owns right-click as Cancel. Otherwise an idle native
            # cross/cut canvas resumes the last classifier, matching VTK views.
            if self.cut_armed:
                self.owner.cancel_cut()
                event.accept()
                return
            helper = getattr(
                self.owner.app, "reactivate_last_classification_from_right_click", None
            )
            target = "cut" if self.slot == CUT_SLOT else "cross"
            if callable(helper) and helper(
                target, focus_widget=self, native=True
            ):
                event.accept()
                return
            self.owner.cancel_cut()
        elif event.button() == Qt.LeftButton and self.cut_armed:
            x = self.horizontal_at(event.position().x())
            if self.cut_centre is None:
                self.cut_centre = x
                self.show_cut_preview(x)
                self.owner.app.statusBar().showMessage('Click to set cut depth; right-click cancels', 5000)
            else:
                self.owner.finish_cut(self, x)
        elif event.button() == Qt.MiddleButton:
            # Middle-drag always pans. Left-drag pans only while the Pan
            # shortcut is armed (handled by _pan_tool_input above).
            self.drag = event.position()

    def mouseMoveEvent(self, event):
        cross_cut_active = self.owner.route_cross_cut_to_canvas(
            self, event.position())
        if cross_cut_active and not self.cut_armed:
            event.accept()
            return
        if self._pan_tool_input(event):
            return
        measurement = getattr(self, 'measurement', None)
        # A measurement gesture does not capture the middle button: section
        # panning remains available while Line/Path is active.
        middle_drag = bool(event.buttons() & Qt.MiddleButton)
        if measurement is not None and not middle_drag and measurement.move(event.position()):
            return
        if not self.cut_armed and self.drag is None and self.classifier.move(event.position()):
            return
        if self.cut_armed:
            x = self.horizontal_at(event.position().x())
            if self.cut_centre is None:
                self.show_cut_preview(x)
            else:
                self.show_cut_preview(self.cut_centre, abs(x-self.cut_centre))
        if self.drag is not None:
            delta = event.position()-self.drag
            self.drag = event.position()
            self._pan_by(delta.x(), delta.y())

    def mouseReleaseEvent(self, event):
        if self._pan_tool_input(event):
            return
        if self.drag is None and not self.cut_armed:
            self.classifier.release(event.position(), event.button())
        self.drag = None

    def mouseDoubleClickEvent(self, event):
        if self._pan_tool_input(event):
            return
        measurement = getattr(self, 'measurement', None)
        if measurement is not None and measurement.double_click(event.position(), event.button()):
            return
        if not self.classifier.double_click(event.position()):
            super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        self._pan_tool_input(event)
        measurement = getattr(self, 'measurement', None)
        if measurement is not None and measurement.key(event.key()):
            return
        if not self.classifier.key(event.key()):
            super().keyPressEvent(event)

    def leaveEvent(self, event):
        if not self.classifier.busy():
            self.classifier.overlay.set_shapes([])
        super().leaveEvent(event)

    def wheelEvent(self, event):
        notch = event.angleDelta().y()/120.
        if notch:
            pos = event.position()
            # Cursor-anchored zoom: remember the section point under the wheel,
            # scale about the canvas centre, then re-centre so that point stays
            # under the cursor - world_per_pixel() halves/doubles with scale.
            anchor = self.section_at(pos.x(), pos.y())
            s0 = max(self.scale, 1e-5)
            self.scale = max(1e-5, s0 * 1.15 ** -notch)
            if anchor is not None:
                f1 = self.world_per_pixel()
                self.centre[0] = anchor[0] - (pos.x()-self.width()*.5)*f1
                self.centre[2] = anchor[1] + (pos.y()-self.height()*.5)*f1
            self._dirty = True
            self.classifier._paint()
            self.owner.sync_from(self.slot)
        event.accept()

    def shutdown(self):
        self._closed = True
        self._display_revision += 1
        if self._mesh_future is not None:
            self._mesh_future.cancel()
            self._mesh_future = None
        # The engine (and its GPU mesh) is gone with the handle.
        self._drop_mesh_cache()
        self._surface_loaded = self._surface_resident = False
        self.timer.stop()
        measurement = getattr(self, 'measurement', None)
        if measurement is not None:
            measurement.shutdown()
        self.classifier.shutdown()
        self.backend.shutdown()


class NativeSections:
    def __init__(self, app):
        self.app = app
        self.views = {}
        self.docks = {}
        self.geometry = {}
        self.active_slot = 1
        self._cut_recipe = None
        self._cut_recipe_nested = False
        self._nested_cut_active = False
        self._cut_depth = 1.
        self._refresh = QTimer(app)
        self._refresh.setInterval(500)
        self._refresh.timeout.connect(self.refresh_edits)
        self._refresh.start()
        self._classes = {}
        self._syncing = False
        self._pending_display = {}
        dialog = getattr(app, 'display_mode_dialog', None) or getattr(app, 'display_dialog', None)
        if dialog is not None:
            self.configure_display_dialog(dialog)

    @staticmethod
    def configure_display_dialog(dialog):
        box = getattr(dialog, 'slot_box', None)
        if box is None:
            return
        selected = box.currentIndex()
        box.blockSignals(True)
        box.clear()
        box.addItems(['Main View'] + [f'View {i}' for i in range(1, CROSS_SECTION_COUNT+1)] + ['Cut Section'])
        box.setCurrentIndex(max(0, min(selected, CUT_SLOT)))
        box.blockSignals(False)

    def open(self, index, draw=True):
        slot = int(index)+1
        if not 1 <= slot <= CROSS_SECTION_COUNT:
            raise ValueError('Cross-section index must be 0..7')
        # Selecting/arming a target must not create or show an empty dock.
        # finalize() creates the viewport once the user completes the section.
        self.active_slot = slot
        self.app.section_controller.active_view = index
        if draw:
            self.cancel_point_actions()
            self.cancel_cut()
            self.cancel_cross()
            picker = LntCrossPicker(self.app, self.app._lnt_main_view)
            # Publish the armed tool where every Esc/cancel path looks for it
            # (VTK-era code checks these attributes). Otherwise Esc finds no
            # active cross-section tool and the native picker stays armed.
            try:
                self.app.cross_interactor = picker
                self.app.cross_section_active = True
                act = getattr(self.app, "cross_action", None)
                if act is not None and hasattr(act, "setChecked"):
                    act.setChecked(True)
                _set_cross_ribbon_checked(self.app, True)
                if hasattr(self.app, "set_cross_cursor_active"):
                    self.app.set_cross_cursor_active(True, "cross_section")
            except Exception:  # noqa: BLE001
                pass
        return self.views.get(slot)

    def open_next(self):
        for slot in range(1, CROSS_SECTION_COUNT+1):
            if slot not in self.docks or not self.docks[slot].isVisible():
                return self.open(slot-1)
        QMessageBox.information(self.app, 'Cross Sections', 'All eight cross-section views are open.')

    def select_view(self):
        app = self.app
        selected = max(0, min(int(getattr(app, '_cross_section_default_view', 0)),
                              CROSS_SECTION_COUNT-1))
        dialog = getattr(app, '_native_cross_view_selector', None)
        if dialog is None:
            dialog = _CrossViewSelectorDialog(app)
            dialog.setWindowTitle('Cross Section')
            dialog.setWindowModality(Qt.NonModal)
            layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel('Target viewport', dialog))
            combo = QComboBox(dialog)
            combo.addItems([f'View {i}' for i in range(1, CROSS_SECTION_COUNT+1)])
            combo.setMinimumWidth(170)
            layout.addWidget(combo)
            dialog.setFixedSize(dialog.sizeHint())
            _place_cross_view_selector(dialog, app)
            dialog.view_combo = combo
            app._native_cross_view_selector = dialog

            def changed(index):
                app._cross_section_default_view = index
                # Resolve the live manager: a file change replaces the old one.
                current = manager(app)
                if current is not None:
                    current.open(index, draw=False)

            combo.currentIndexChanged.connect(changed)
        combo = dialog.view_combo
        combo.blockSignals(True)
        combo.setCurrentIndex(selected)
        combo.blockSignals(False)
        dialog.show()
        dialog._track_position = True
        dialog.raise_()
        # Drawing starts immediately; the dropdown never asks for confirmation.
        return self.open(selected)

    def ensure_dock(self, slot):
        if slot in self.docks:
            self.docks[slot].show()
            self.docks[slot].raise_()
            return self.views[slot]
        dock = QDockWidget('Cut Section' if slot == CUT_SLOT else f'Cross Section View {slot}', self.app)
        frame = QWidget(dock)
        layout = QVBoxLayout(frame)
        canvas = SectionCanvas(self, slot, frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(canvas)
        dock.setWidget(frame)
        self.app.addDockWidget(Qt.RightDockWidgetArea, dock)
        self._float_dock(dock, slot)
        self.views[slot], self.docks[slot] = canvas, dock
        canvas.palette = dict(getattr(self.app, 'view_palettes', {}).get(slot)
                              or getattr(self.app, 'class_palette', {}) or {})
        if slot in self._pending_display:
            canvas.palette, canvas.mode = self._pending_display[slot]
        if slot != CUT_SLOT:
            if not isinstance(getattr(self.app, 'section_docks', None), dict):
                self.app.section_docks = {}
            self.app.section_docks[slot-1] = dock
        dock.show()
        canvas.start()
        self._apply_theme_background(canvas)
        if slot != CUT_SLOT and getattr(self.app, '_native_xs_active', False):
            from gui.native_shortcuts import arm_section_measure
            arm_section_measure(canvas, True)     # XS is on: new views measure too
        return canvas

    @staticmethod
    def _geometry_key(slot):
        return 'NativeCutSectionDock_geometry' if slot == CUT_SLOT else f'CrossSectionDock_{slot-1}_geometry'

    def _float_dock(self, dock, slot):
        """Same window behaviour as the VTK section docks: a floating, never
        re-docked window that remembers its geometry. Done before the canvas
        creates its renderer, so the Vulkan surface is never re-parented."""
        dock.setObjectName('NativeCutSectionDock' if slot == CUT_SLOT else f'CrossSectionDock_{slot-1}')
        dock.setContextMenuPolicy(Qt.NoContextMenu)
        dock.setFloating(True)
        dock.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable)
        dock.setAllowedAreas(Qt.NoDockWidgetArea)
        saved = None
        settings = getattr(self.app, 'settings', None)
        try:
            saved = settings.value(self._geometry_key(slot)) if settings is not None else None
        except Exception:  # noqa: BLE001 - a bad saved value just means default placement
            saved = None
        if not (saved is not None and dock.restoreGeometry(saved)):
            offset = 40 * (slot - 1)
            app = self.app
            dock.resize(520, 420)
            dock.move(app.x() + max(0, app.width() - 560) - offset, app.y() + 120 + offset)
        dock.visibilityChanged.connect(lambda visible, s=slot: None if visible else self._save_geometry(s))
        try:
            from gui.theme_manager import ThemeManager
            ThemeManager.apply_native_window_theme(dock)
        except Exception:  # noqa: BLE001 - cosmetic
            pass

    def _save_geometry(self, slot):
        dock, settings = self.docks.get(slot), getattr(self.app, 'settings', None)
        if dock is None or settings is None:
            return
        try:
            settings.setValue(self._geometry_key(slot), dock.saveGeometry())
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _apply_theme_background(canvas):
        try:
            from gui.theme_manager import ThemeManager
            rgb = ThemeManager.canvas_background_rgb()
        except Exception:  # noqa: BLE001
            rgb = (0., 0., 0.)
        canvas.backend.set_clear_color(*[float(c) for c in rgb])
        canvas._dirty = True

    def finalize(self, p1, p2, half_width, index=None, buffer=None):
        slot = int((getattr(self.app.section_controller, 'active_view', 0) or 0) if index is None else index)+1
        buffer = float(getattr(self.app, 'section_buffer', 2.) if buffer is None else buffer)
        store = get_store(self.app)
        if store is None or not store.ensure_for_section(p1, p2, half_width, buffer):
            raise RuntimeError(store.last_error if store else 'Native section query API unavailable')
        # Only the rows of the queried box can be in the corridor; testing them
        # (not the whole accumulated store) keeps a section's cost its own size.
        rows = getattr(store, 'last_region_all_indices', getattr(store, 'last_region_indices', None))
        if rows is None:
            indices, local, core, lo, hi = section_geometry(store.data['xyz'], p1, p2, half_width, buffer, True)
        else:
            xyz = store.data['xyz']
            if len(rows) and rows[-1] - rows[0] + 1 == len(rows) and bool(np.all(np.diff(rows) == 1)):
                start = int(rows[0])          # freshly appended box: a view, no gather
                hits, local, core, lo, hi = section_geometry(xyz[start:start+len(rows)], p1, p2, half_width, buffer, True)
                indices = hits + start
            else:
                rows = np.sort(rows)
                hits, local, core, lo, hi = section_geometry(xyz[rows], p1, p2, half_width, buffer, True)
                indices = rows[hits]
        canvas = self.ensure_dock(slot)
        self.geometry[slot] = (np.array(p1), np.array(p2), float(half_width), buffer)
        canvas.set_points(local, indices, canvas.axis, bounds=(lo, hi))
        self._classes[slot] = store.data['classification'][indices].copy()
        # Existing classification operations retain stable store indices.
        if core.all():          # no buffer records: views, not copies of the whole section
            core_parts = (indices, indices[:0], local, local[:0])
        else:
            core_parts = (indices[core], indices[~core], local[core], local[~core])
        for name, value in {'indices': indices, 'canonical_global_indices': indices,
                            'points_transformed': local,
                            'core_indices': core_parts[0], 'buffer_indices': core_parts[1],
                            'core_points': core_parts[2], 'buffer_points': core_parts[3],
                            'P1': p1, 'P2': p2, 'half_width': half_width}.items():
            setattr(self.app, f'section_{slot-1}_{name}', value)
        self.app.statusBar().showMessage(f'View {slot}: {len(indices):,} section records, native Vulkan', 5000)
        if self._cut_recipe and self._cut_recipe[0] == slot:
            _, _, _, axis, centre = self._cut_recipe
            self._cut_recipe = (slot, local.copy(), indices.copy(), axis, centre)
            self.update_cut_depth(self._cut_depth)
        self.sync_from(slot, data=True, display=True)

    def change_axis(self, slot, axis):
        canvas = self.views[slot]
        canvas.set_points(canvas.local, canvas.indices, axis)
        if self._cut_recipe and self._cut_recipe[0] == slot and axis != self._cut_recipe[3]:
            canvas.classifier.overlay.set_marks([])   # the slab is not visible from this side
        self.sync_from(slot)

    def linked_slots(self, slot):
        """Connected component of the eight cross views; links are bidirectional."""
        if not 1 <= slot <= CROSS_SECTION_COUNT:
            return set()
        linked = {slot}
        edges = [(int(target)+1, int(source)+1)
                 for target, source in getattr(self.app, 'view_sync_map', {}).items()
                 if 0 <= int(target) < CROSS_SECTION_COUNT and 0 <= int(source) < CROSS_SECTION_COUNT]
        while True:
            previous = len(linked)
            for target, source in edges:
                if target in linked or source in linked:
                    linked.update((target, source))
            if len(linked) == previous:
                return linked - {slot}

    def set_view_sync(self, target_view_num, source_view_num):
        target = int(target_view_num)
        if not 1 <= target <= CROSS_SECTION_COUNT:
            raise ValueError('Sync target must be View 1..8')
        mapping = getattr(self.app, 'view_sync_map', None)
        if mapping is None:
            mapping = self.app.view_sync_map = {}
        if source_view_num is None or int(source_view_num) == target:
            mapping.pop(target-1, None)
            return
        source = int(source_view_num)
        if not 1 <= source <= CROSS_SECTION_COUNT:
            raise ValueError('Sync source must be View 1..8')
        mapping[target-1] = source-1
        candidates = [source] + sorted(self.linked_slots(source))
        driver = next((s for s in candidates if s in self.views and len(self.views[s].indices)), None)
        if driver is not None:
            self.sync_from(driver, data=True, display=True)

    def apply_sync_map(self, mapping):
        """Apply the whole dialog/preset once, uploading each group member once."""
        updated = {int(target): int(source) for target, source in mapping.items()
                   if int(target) != int(source)}
        if any(not 0 <= value < CROSS_SECTION_COUNT
               for pair in updated.items() for value in pair):
            raise ValueError('Synchronized views must be View 1..8')
        previous = dict(getattr(self.app, 'view_sync_map', {}) or {})
        self.app.view_sync_map = updated
        visited = set()
        for target, source in updated.items():
            if previous.get(target) == source:
                continue
            slot = source+1
            group = {slot} | self.linked_slots(slot)
            if group & visited:
                continue
            visited.update(group)
            candidates = [slot] + sorted(group - {slot})
            driver = next((s for s in candidates if s in self.views and s in self.geometry), None)
            if driver is not None:
                self.sync_from(driver, data=True, display=True)

    def sync_from(self, slot, data=False, display=False):
        """Mirror native state once per peer, with no VTK observers or re-query."""
        if self._syncing or slot not in self.views:
            return
        source = self.views[slot]
        if not len(source.indices) and not data:
            return
        self._syncing = True
        try:
            for peer in sorted(self.linked_slots(slot)):
                target = self.views.get(peer)
                if target is None:
                    if not data:
                        continue
                    target = self.ensure_dock(peer)
                # Palette and display mode are per-view: a Display Mode Apply
                # targets ONLY the view selected in the dialog. Sync mirrors
                # section data and camera, never display state, so a peer keeps
                # its own class colors/visibility whether linked or not.
                if data or target.axis != source.axis:
                    target.classifier.cancel()
                    source_local, source_rows = getattr(source, '_deletion_source', (source.local, source.indices))
                    target.set_points(source_local, source_rows, source.axis, fit=False,
                                      bounds=(source._lo, source._hi))
                    self._classes[peer] = self._classes.get(slot, np.empty(0, dtype=np.uint8)).copy()
                elif display:
                    target.apply_palette(target.palette, target.mode, sync=False)
                if data:
                    if slot in self.geometry:
                        p1, p2, width, buffer = self.geometry[slot]
                        self.geometry[peer] = (p1.copy(), p2.copy(), width, buffer)
                    for name in ('indices', 'canonical_global_indices', 'points_transformed',
                                 'core_indices', 'buffer_indices', 'core_points', 'buffer_points',
                                 'P1', 'P2', 'half_width'):
                        value = getattr(self.app, f'section_{slot-1}_{name}', None)
                        setattr(self.app, f'section_{peer-1}_{name}', value)
                    if self._cut_recipe and self._cut_recipe[0] == peer:
                        _, _, _, axis, centre = self._cut_recipe
                        self._cut_recipe = (peer, source.local.copy(), source.indices.copy(), axis, centre)
                        self.update_cut_depth(self._cut_depth)
                target.centre = source.centre.copy()
                target.scale = source.scale
                target._dirty = True
        finally:
            self._syncing = False

    def world_from_section(self, slot, h, z):
        """Section-view position -> world XYZ. The coordinate the view looks
        along is unknown from a click: the section centre line (front view) or
        the middle of the section (side view) is used for it."""
        if slot not in self.geometry:
            return None
        p1, p2, _, _ = self.geometry[slot]
        delta = np.asarray(p2[:2], dtype=np.float64) - np.asarray(p1[:2], dtype=np.float64)
        length = float(np.linalg.norm(delta))
        if length <= 1e-9:
            return None
        direction = delta / length
        normal = np.array([-direction[1], direction[0]])
        along, across = (h, 0.) if self.views[slot].axis == 0 else (length * .5, h)
        xy = np.asarray(p1[:2], dtype=np.float64) + direction * along + normal * across
        return np.array([xy[0], xy[1], z], dtype=np.float64)

    def locate(self, canvas, pos):
        """Cross Section tool armed + click in a cross-section view: start the
        next section there (its P1) and centre the main view on it, like the
        VTK section locate."""
        view = getattr(self.app, '_lnt_main_view', None)
        picker = getattr(view, 'picker', None)
        if (picker is None or getattr(picker, 'slice_state', None) != 0
                or canvas.slot == CUT_SLOT or not len(canvas.indices)):
            return False
        point = self.world_from_section(canvas.slot, *canvas.section_at(pos.x(), pos.y()))
        if point is None:
            return False
        self.center_main_on(point[0], point[1])
        picker.P1, picker.slice_state = point, 1
        self.app.statusBar().showMessage(
            'Section locate: P1 set from View %d - click P2 in the main view' % canvas.slot, 6000)
        return True

    def center_main_on(self, x, y):
        view = getattr(self.app, '_lnt_main_view', None)
        native = getattr(view, 'native', None)
        widget = getattr(getattr(view, 'owner', None), 'vulkan_widget', None)
        if native is None or widget is None:
            return False
        ratio = max(widget.devicePixelRatioF(), 1.)
        for _ in range(3):                       # pan() is in mouse pixels; converge exactly
            screen = native.world_to_screen(x, y)
            if screen is None:
                return False
            dx = widget.width() * .5 - screen[0] / ratio
            dy = widget.height() * .5 - screen[1] / ratio
            if abs(dx) < .5 and abs(dy) < .5:
                return True
            native.pan(dx, dy)
        return True

    def update_buffer(self, slot, value):
        if slot in self.geometry:
            p1, p2, width, _ = self.geometry[slot]
            self.finalize(p1, p2, width, slot-1, value)

    def apply(self, slot, palette, mode):
        self._pending_display[slot] = (dict(palette), mode)
        # Display Mode Apply is strictly per-view: never mirror the palette or
        # the color mode into synced peers (sync carries data and camera only).
        if slot in self.views:
            self.views[slot].apply_palette(palette, mode, sync=False)
        return True

    def cancel_cross(self):
        view = getattr(self.app, '_lnt_main_view', None)
        picker = getattr(view, 'picker', None)
        if picker is not None:
            picker.cleanup()
        self.app.cross_interactor = None
        self.app.cross_section_active = False
        action = getattr(self.app, 'cross_action', None)
        if action is not None:
            blocked = action.blockSignals(True) if hasattr(action, 'blockSignals') else None
            action.setChecked(False)
            if blocked is not None:
                action.blockSignals(blocked)
        _set_cross_ribbon_checked(self.app, False)
        if hasattr(self.app, 'set_cross_cursor_active'):
            self.app.set_cross_cursor_active(False, 'cross_section')

    def cancel_point_actions(self):
        self.app.active_classify_tool = None
        view = getattr(self.app, '_lnt_main_view', None)
        for owner in [view] + list(self.views.values()):
            classifier = getattr(owner, 'classifier', None)
            if classifier is not None:
                classifier.cancel()
            measurement = getattr(owner, 'measurement', None)
            if measurement is not None:
                measurement.cancel()

    def arm_cut(self, nested=False, source_slot=None):
        if (nested and self._nested_cut_active and self.views.get(CUT_SLOT) is not None
                and self.views[CUT_SLOT].cut_armed):
            return True                      # already armed: keep a placed centre
        self.cancel_point_actions()
        self.cancel_cross()
        self.cancel_cut()
        selector = getattr(self.app, '_native_cross_view_selector', None)
        if selector is not None:
            selector.hide()
        slot = CUT_SLOT if nested else (source_slot or self.active_slot)
        if slot == CUT_SLOT and not nested:
            slot = next((s for s in self.views if s != CUT_SLOT), 0)
        canvas = self.views.get(slot)
        if canvas is None or not len(canvas.indices):
            self.app.statusBar().showMessage(
                'Create a cut section before using Cut-in-Cut' if nested else
                'Draw a cross section before creating a cut', 5000)
            return False
        self._nested_cut_active = bool(nested)
        canvas.cut_armed = True
        self.app.cut_section_mode_on = True
        if hasattr(self.app, 'set_cross_cursor_active'):
            self.app.set_cross_cursor_active(True, 'cut_section')
        canvas.setFocus()
        # Do not wait for a future MouseMoveEvent before showing feedback.
        # Shortcut activation commonly happens while the pointer is outside
        # the chosen cross-section canvas, which previously left Cut-from-
        # Cross looking inactive.  Use the current pointer when it is inside
        # the source canvas, otherwise start at the visible viewport centre.
        cursor = canvas.mapFromGlobal(QCursor.pos())
        preview_x = (cursor.x() if canvas.rect().contains(cursor)
                     else canvas.width() * .5)
        canvas.show_cut_preview(canvas.horizontal_at(preview_x))
        self.app.statusBar().showMessage('Cut: click the centre, then click to set depth; right-click cancels', 6000)
        return True

    def route_cross_cut_to_canvas(self, canvas, position):
        """Move an uncommitted Cut-in-Cross gesture to the hovered view.

        With several cross-section docks open, activating the tool must not
        permanently bind it to whichever dock happened to own focus. Before
        the first click the mouse owns routing; after the centre click the
        chosen canvas remains locked through depth selection/finalization.

        Returns True while a non-nested Cut-in-Cross operation owns section
        input, including on a non-owner canvas after the centre is locked.
        """
        if (not getattr(self.app, 'cut_section_mode_on', False)
                or self._nested_cut_active or canvas.slot == CUT_SLOT):
            return False
        if canvas.slot not in self.views or not len(canvas.indices):
            return False

        armed = next((view for view in self.views.values()
                      if view.slot != CUT_SLOT and view.cut_armed), None)
        # The first click commits the source view. Cursor travel during the
        # depth step must not steal the operation into another section.
        if armed is not None and armed.cut_centre is not None:
            return True

        if armed is not canvas:
            if armed is not None:
                armed.cut_armed = False
                armed.cut_centre = None
                armed.clear_cut_preview()
            canvas.cut_armed = True
            canvas.cut_centre = None
            self.active_slot = canvas.slot
            canvas.setFocus()

        x = max(0.0, min(float(position.x()), float(max(0, canvas.width() - 1))))
        canvas.show_cut_preview(canvas.horizontal_at(x))
        return True

    def cancel_cut(self):
        self._nested_cut_active = False
        self.app.cut_section_mode_on = False
        if hasattr(self.app, 'set_cross_cursor_active'):
            self.app.set_cross_cursor_active(False, 'cut_section')
        for canvas in self.views.values():
            canvas.cut_armed = False
            canvas.cut_centre = None
            canvas.clear_cut_preview()
            canvas.classifier.overlay.set_marks([])
        return True

    def finish_cut(self, source, edge):
        centre, depth = source.cut_centre, abs(edge-source.cut_centre)*2
        if depth < 1e-6:
            return
        nested = self._nested_cut_active and source.slot == CUT_SLOT
        if nested and not np.any(np.abs(source.local[:, source.axis]-centre) <= depth*.5):
            source.cut_centre = None
            source.clear_cut_preview()
            self.app.statusBar().showMessage('No points in cut range; click a new centre', 5000)
            return
        # Snapshot the source when taking a nested cut; replacing the cut slot
        # must not change the input of subsequent depth adjustments.
        source_local, source_rows = getattr(source, '_deletion_source', (source.local, source.indices))
        self._cut_recipe = (source.slot, source_local.copy(), source_rows.copy(),
                            source.axis, centre)
        self._cut_recipe_nested = nested
        self.cancel_cut()
        self.update_cut_depth(depth)
        if nested:
            # Cut-in-Cut stays armed on the result (naksha-lidar behaviour):
            # each next placement starts from the latest result, not its parent.
            self._nested_cut_active = True
            source.cut_armed = True
            self.app.cut_section_mode_on = True
            if hasattr(self.app, 'set_cross_cursor_active'):
                self.app.set_cross_cursor_active(True, 'cut_section')
            source.setFocus()
            self.app.statusBar().showMessage(
                'Cut-in-Cut ready: click next centre, then depth; Escape exits', 0)

    def update_cut_depth(self, depth):
        if self._cut_recipe is None:
            return
        self._cut_depth = float(depth)
        _, local, indices, axis, centre = self._cut_recipe
        selected = np.abs(local[:, axis]-centre) <= depth*.5
        canvas = self.ensure_dock(CUT_SLOT)
        output, output_axis = local[selected], 1-axis
        if self._cut_recipe_nested:
            # Rotate the displayed section coordinates clockwise by 90 degrees,
            # as the reference does, without touching store/world records.
            rotated = np.empty_like(output)
            rotated[:, 0] = output[:, 1-axis]
            rotated[:, 1] = -output[:, axis]
            rotated[:, 2] = output[:, 2]
            output, output_axis = rotated, 0
        canvas.set_points(output, indices[selected], output_axis)
        store = get_store(self.app)
        self._classes[CUT_SLOT] = store.data['classification'][canvas.indices].copy()
        controller = getattr(self.app, 'cut_section_controller', None)
        if controller is not None:
            controller.cut_indices = canvas.indices
            controller.cut_points = canvas.local
            controller.is_cut_view_active = True
        self.app.statusBar().showMessage(f'Cut: {selected.sum():,} records, depth {depth:.3f} m', 5000)

    def refresh_deletions(self):
        for slot, canvas in self.views.items():
            if hasattr(canvas, '_deletion_source'):
                local, rows = canvas._deletion_source
                canvas.set_points(local, rows, canvas.axis, fit=False)
                self._classes.pop(slot, None)

    def refresh_edits(self, force=False, changed=None, show_main=True):
        store = get_store(self.app)
        if store is None:
            return
        if changed is not None and show_main:
            store.show(*changed)             # main view recolours now; sparse write follows
        # Each tick would gather every view's classes (millions of records on
        # a large section); only do it when the store saw an edit since.
        generation = getattr(store, 'generation', None)
        if not force:
            # A committed classification whose exact rows were already pushed
            # to every section is waiting only for its debounced LNT write.
            # Do not let the 500 ms safety timer rescan millions of section rows
            # during that window.
            if int(getattr(store, '_dirty_rows_queued', 0) or 0) > 0:
                return
            if generation is not None and generation == getattr(self, '_seen_generation', None):
                return
        # ``store.show(changed)`` previews the exact rows immediately and its
        # debounced disk flush increments generation once later.  When we already
        # consume those exact rows here, anticipate that one persistence-only
        # generation so the 500 ms section timer does not perform a redundant
        # full-array gather/compare immediately after the write.
        self._seen_generation = (generation + 1
                                 if changed is not None and show_main and generation is not None
                                 else generation)
        # A gesture knows exactly which rows it changed. Mapping those onto
        # each view (sorted store rows) avoids gathering and comparing every
        # record of every view on each brush stroke.
        rows = new_classes = None
        if changed is not None:
            try:
                rows = np.asarray(changed[0], dtype=np.int64).reshape(-1)
                new_classes = np.asarray(store.data['classification'][rows], dtype=np.uint8)
            except Exception:  # noqa: BLE001 - any odd payload falls back to the full scan
                rows = new_classes = None
        classes = np.asarray(store.data['classification'])
        for slot, canvas in self.views.items():
            if not len(canvas.indices):
                continue
            cache = self._classes.get(slot)
            if rows is not None and rows.size and cache is not None:
                if len(cache) != len(canvas.indices):
                    cache = None
                else:
                    position = canvas.positions_for_store_rows(rows)
                    hit = position >= 0
                    if not hit.any():
                        continue            # this view never saw the edit
                    cache[position[hit]] = new_classes[hit]
                    if not canvas._refresh_point_classification(cache):
                        # Mesh modes, or an unusual source-line state, keep the
                        # existing correctness path.  Point modes avoid the full
                        # palette/LUT/display re-application above.
                        canvas.apply_palette(canvas.palette, canvas.mode, sync=False)
                    continue
            current = classes[canvas.indices]
            if not np.array_equal(current, cache):
                self._classes[slot] = current.copy()
                if not canvas._refresh_point_classification(current):
                    canvas.apply_palette(canvas.palette, canvas.mode, sync=False)

    def close(self):
        self._refresh.stop()
        self.cancel_cut()
        for slot, canvas in self.views.items():
            self._save_geometry(slot)
            canvas.shutdown()
            self.docks[slot].close()
            self.docks[slot].deleteLater()
            if slot != CUT_SLOT:
                getattr(self.app, 'section_docks', {}).pop(slot-1, None)
        self.views.clear()
        self.docks.clear()
