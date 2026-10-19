"""Native Vulkan section docks for an open LNT dataset.

Only the queried corridor is decoded for editing. Every selected leaf record
keeps its original node/local identity in LntSectionStore; rendering never
constructs VTK datasets, actors, cameras or interactors.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QWidget, QDockWidget, QVBoxLayout, QLabel, QMessageBox
from PySide6.QtWidgets import QComboBox, QDialog

from gui.render_backend import VulkanRenderBackend
from gui.lnt_section_bridge import get_store, LntCrossPicker
from gui.native_section_classify import SectionClassifier
from gui.native_pan import NativePanGesture, disarm_pan

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
        self.shading_parameters = (45., 45., .25)
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
        self.local = np.ascontiguousarray(local, dtype=np.float64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.axis = axis
        if not self.start():
            return
        if not len(self.local):
            # Nothing in the corridor/cut: show an empty view (the engine
            # refuses a zero-point upload) and keep the old buffer hidden.
            self.backend.set_point_cloud_visible(False)
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

    def apply_palette(self, palette, mode='class', sync=False):
        self.palette, self.mode = dict(palette or {}), mode
        if not self.backend._handle:
            return
        colors = np.full((256, 3), 128, dtype=np.uint8)
        visible = np.ones(256, dtype=np.uint8)
        # Per-class point weights (Display Mode "Weight" column). Mirrors the
        # VTK weight_lut: gl_PointSize = the view's fixed pixel size * weight.
        # Strictly per-view: this pushes to THIS view's renderer only, so
        # growing a class's points here never touches a synced sibling view.
        weights = np.ones(256, dtype=np.float32)
        for code, rule in (palette or {}).items():
            code = int(code)
            if 0 <= code < 256:
                colors[code] = rule.get('color', (128, 128, 128))
                visible[code] = bool(rule.get('show', True))
                try:
                    raw = float(rule.get('weight', 1.0))
                except (TypeError, ValueError):
                    raw = 1.0
                weights[code] = np.float32(max(0.1, min(raw, 12.0)))
        self.backend.set_point_luts(class_rgb=colors)
        self.backend.set_class_visibility(visible)
        if hasattr(self.backend, 'set_class_weights'):
            self.backend.set_class_weights(weights)
        # Surface mode is rendered as a Vulkan mesh (slab clip of the section),
        # not as points. Clear any previous mesh when leaving it.
        if mode != 'surface' and getattr(self, '_surface_loaded', False):
            try:
                self.backend.clear_surface()
            except Exception:
                pass
            self._surface_loaded = False

        if mode == 'surface':
            self.backend.set_point_cloud_visible(False)
            try:
                ok = self._upload_surface_slab()
            except Exception as exc:
                print(f"SECTION_MESH view={self.slot} status=surface_exception reason={exc}")
                ok = False
            if not ok:
                self.backend.set_point_cloud_visible(True)
            self._dirty = True
            if sync:
                self.owner.sync_from(self.slot, display=True)
            return

        # Points are visible for all point-based modes.
        self.backend.set_point_cloud_visible(True)
        modes = {'rgb': 0, 'class': 1, 'intensity': 2, 'elevation': 3,
                 'shaded_class': 4, 'depth': 6, 'line': 1}
        # Sections are projected point views. Native instant lighting shades
        # these records without constructing a terrain TIN across vertical faces.
        self.backend.set_display_mode(modes.get(mode, 1))
        if mode == 'shaded_class':
            self.backend.set_instant_shading_parameters(*self.shading_parameters)
        if len(self.local):
            self.backend.set_point_elevation_range(float(self._lo[2]), float(self._hi[2]) + 1e-6)
        self._dirty = True
        if sync:
            self.owner.sync_from(self.slot, display=True)

    def _upload_surface_slab(self):
        try:
            from gui.cross_section.section_shaded_surface import (
                _build_independent_surface_mesh, _slab_clip_and_transform,
            )
        except Exception as exc:
            print(f"   section surface import failed ({exc})")
            return False
        app = self.owner.app
        geom = getattr(self.owner, 'geometry', {}).get(self.slot)
        if geom is None:
            print(f"   section {self.slot}: no cutting line stored yet")
            return False
        p1, p2, half_width, _buffer = geom
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)
        hw = float(half_width) if half_width else 5.0
        app.section_view_idx_tmp = self.slot - 1
        view_idx = self.slot - 1
        setattr(app, f"section_{view_idx}_P1", p1)
        setattr(app, f"section_{view_idx}_P2", p2)
        setattr(app, f"section_{view_idx}_half_width", hw)

        # Ensure the section's own palette is available to the visibility mask.
        if not hasattr(app, 'display_mode_dialog') or app.display_mode_dialog is None:
            pass

        mesh = _build_independent_surface_mesh(app, view_idx)
        if mesh is None:
            print(f"   section {self.slot}: could not build surface mesh")
            return False
        result = _slab_clip_and_transform(mesh, p1, p2, hw)
        if result is None:
            print(f"   section {self.slot}: slab does not intersect mesh")
            return False
        local_pts, faces, point_rgb, cell_rgb = result
        positions = np.ascontiguousarray(local_pts, dtype=np.float64)
        faces_i = np.ascontiguousarray(faces, dtype=np.int32)
        if cell_rgb is not None:
            face_colors = np.ascontiguousarray(cell_rgb, dtype=np.uint8)
        elif point_rgb is not None:
            prgb = np.asarray(point_rgb, dtype=np.uint8)
            face_colors = np.ascontiguousarray(prgb[faces[:, 0]], dtype=np.uint8)
        else:
            face_colors = np.full((len(faces), 3), 200, dtype=np.uint8)
        ok = self.backend.set_surface(positions, faces_i, face_colors)
        if ok:
            self._surface_loaded = True
        else:
            print(f"   section {self.slot}: backend surface upload failed")
        return bool(ok)

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
        centre_px = 2. if self.slot == CUT_SLOT else 1.
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
                or (measurement is not None and measurement.active)):
            self.pan_gesture.finish()
            return False
        return self.pan_gesture.handle(self.owner.app, event)

    def mousePressEvent(self, event):
        self.setFocus()
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
        rows = getattr(store, 'last_region_indices', None)
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
                    target.set_points(source.local, source.indices, source.axis, fit=False,
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
        self.app.statusBar().showMessage('Cut: click the centre, then click to set depth; right-click cancels', 6000)
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
        self._cut_recipe = (source.slot, source.local.copy(), source.indices.copy(),
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

    def refresh_edits(self, force=False, changed=None):
        store = get_store(self.app)
        if store is None:
            return
        if changed is not None:
            store.show(*changed)             # main view recolours now; the write follows
        # Each tick would gather every view's classes (millions of records on
        # a large section); only do it when the store saw an edit since.
        generation = getattr(store, 'generation', None)
        if not force and generation is not None and generation == getattr(self, '_seen_generation', None):
            return
        self._seen_generation = generation
        for slot, canvas in self.views.items():
            if not len(canvas.indices):
                continue
            current = store.data['classification'][canvas.indices]
            if not np.array_equal(current, self._classes.get(slot)):
                canvas.backend.set_point_attributes(classification=current)
                self._classes[slot] = current.copy()
                canvas._dirty = True

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
