"""Rectangle / polygon selection + count/delete for the native plan view.

Block and Shape gestures are drawn by the Vulkan vector pass; candidate records come from the
section store (exact leaf records with class + XYZ).  Selection is a set of
store rows (file identities), counted into the Identify ribbon.

Pending deletions are separate native record exclusions. Only explicit Save
compacts the LAS/LAZ and its LNT companion.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox


class NativeSelectRect:
    def __init__(self, view):
        self.view = view
        self.app = view.app
        self.active = False
        self.pick_method = "Block"
        self.start = None
        self.current = None
        self.dragging = False
        self.vertices = []       # shape-mode world vertices
        self.selected_rows = np.empty(0, dtype=np.int64)
        self._canvas_instance = None
        self._overlay = None

    # ---- lifecycle ----------------------------------------------------
    def _canvas(self):
        from gui.native_main_classify import MainCanvas  # noqa: PLC0415
        # Reuse the native screen/world adapter for the whole tool session.
        if self._canvas_instance is None:
            self._canvas_instance = MainCanvas(self.view)
        return self._canvas_instance

    def _overlay_for(self, canvas):
        from gui.native_zoom_preview import VulkanSelectionPreview
        if self._overlay is None:
            self._overlay = VulkanSelectionPreview(self.view)
        return self._overlay

    def activate(self):
        self._overlay_for(self._canvas())
        drawing = getattr(self.view, 'drawing', None)
        if drawing is not None and drawing.active():
            drawing.cancel()
        measurement = getattr(self.view, 'measurement', None)
        if measurement is not None and measurement.active:
            measurement.cancel()
        self.deactivate_selection()
        self.active = True
        self.start = self.current = None
        self.dragging = False
        self.vertices = []
        try:
            self._canvas().setFocus()
        except Exception:
            pass
        self._status()
        return True

    def set_pick_method(self, method):
        if method in ("Block", "Shape"):
            self.deactivate_selection()
            self.pick_method = method
            self.start = self.current = None
            self.dragging = False
            self.vertices = []
            self._paint()
            self._status()

    def cancel(self):
        self.active = False
        self.deactivate_selection()
        try:
            if self._overlay is not None:
                self._overlay.clear()
        except Exception:
            pass

    def shutdown(self):
        """Hide the GPU preview before its Vulkan view disappears."""
        self.deactivate_selection()
        self.active = False
        self.start = self.current = None
        self.dragging = False
        self.vertices = []
        overlay, self._overlay = self._overlay, None
        if overlay is not None:
            try:
                overlay.shutdown()
            except (RuntimeError, ReferenceError, AttributeError):
                pass
        canvas, self._canvas_instance = self._canvas_instance, None
        if canvas is not None:
            canvas._closed = True

    def deactivate(self):
        self.deactivate_selection()
        self.active = False
        self.start = self.current = None
        self.dragging = False
        self.vertices = []
        try:
            if self._overlay is not None:
                self._overlay.clear()
        except Exception:
            pass

    def deactivate_selection(self):
        if self._overlay is not None:
            try:
                self._overlay.clear()
            except (RuntimeError, ReferenceError):
                pass
        self.selected_rows = np.empty(0, dtype=np.int64)
        self._update_count()

    # ---- events ---------------------------------------------------------
    def press(self, pos, button):
        if not self.active:
            return False
        if button == Qt.RightButton:
            return self._right_click()
        if button != Qt.LeftButton:
            return False
        try:
            w = self._canvas().section_at(pos.x(), pos.y())
        except Exception:
            return False
        if not self.dragging and not self.vertices:
            self.deactivate_selection()
        if self.pick_method == "Shape":
            self.vertices.append(w)
            self.current = w
            self._paint()
            return True
        if self.dragging and self.start is not None:
            self.current = w
            self._finish_block()
            return True
        self.start, self.current, self.dragging = w, w, True
        self._paint()
        return True

    def move(self, pos):
        if not self.active:
            return False
        if self.pick_method == 'Shape' and self.vertices:
            try:
                self.current = self._canvas().section_at(pos.x(), pos.y())
            except Exception:
                return False
            self._paint()
            return True
        if self.pick_method != 'Block' or not self.dragging or self.start is None:
            return False
        try:
            self.current = self._canvas().section_at(pos.x(), pos.y())
        except Exception:
            return False
        self._paint()
        return True

    def release(self, pos, button):
        if not self.active or self.pick_method != "Block":
            return False
        if not self.dragging or button != Qt.LeftButton:
            return False
        try:
            self.current = self._canvas().section_at(pos.x(), pos.y())
        except Exception:
            pass
        self._finish_block()
        return True

    def key(self, key):
        if not self.active:
            return False
        if key == Qt.Key_Escape:
            self.cancel()
            try:
                ribbon = getattr(getattr(self.app, "ribbon_manager", None),
                                 "ribbons", {}).get("identify")
                if ribbon is not None and hasattr(ribbon, "_deactivate_select_rectangle"):
                    ribbon._deactivate_select_rectangle()
            except Exception:
                pass
            return True
        return False

    def _right_click(self):
        if self.pick_method == "Shape":
            if len(self.vertices) >= 3:
                self._finish_shape()
            else:
                self.vertices = []
                self._paint()
            return True
        if self.dragging and self.start is not None and self.current is not None:
            self._finish_block()
            return True
        return False

    # ---- selection --------------------------------------------------------
    def _paint(self):
        try:
            ov = self._overlay_for(self._canvas())
            if self.pick_method == "Shape" and self.vertices:
                ov.set_polygon(self.vertices, self.current)
            elif self.start is not None and self.current is not None:
                ov.set_rectangle(self.start, self.current)
            else:
                ov.clear()
        except Exception as exc:
            self.deactivate()
            self.app.statusBar().showMessage(f'Selection preview failed: {exc}', 5000)

    def _status(self):
        try:
            if self.pick_method == "Shape":
                self.app.statusBar().showMessage(
                    "Native Select (Shape): left-click vertices, right-click to close", 5000)
            else:
                self.app.statusBar().showMessage(
                    "Native Select (Block): drag a rectangle", 5000)
        except Exception:
            pass

    def _finish_block(self):
        self.dragging = False
        a, b = self.start, self.current
        self.start = self.current = None
        self.deactivate_selection()
        if a is None or b is None:
            return
        x0, x1 = sorted((float(a[0]), float(b[0])))
        y0, y1 = sorted((float(a[1]), float(b[1])))
        if x1 - x0 < 1e-9 or y1 - y0 < 1e-9:
            self._paint()
            return
        self._overlay_for(self._canvas()).set_rectangle(a, b)
        self._query_box(x0, x1, y0, y1)
        self._confirm_selection()

    def _finish_shape(self):
        poly = list(self.vertices)
        self.vertices = []
        self.deactivate_selection()
        if len(poly) < 3:
            return
        self._overlay_for(self._canvas()).set_polygon(poly)
        p = np.asarray(poly, dtype=np.float64)
        self._query_box(float(p[:, 0].min()), float(p[:, 0].max()),
                        float(p[:, 1].min()), float(p[:, 1].max()),
                        polygon=p)
        self._confirm_selection()

    def _query_box(self, x0, x1, y0, y1, polygon=None):
        # A failed/new query must never leave an older deletion target armed.
        self.selected_rows = np.empty(0, dtype=np.int64)
        self._update_count()
        from gui.lnt_section_bridge import get_store  # noqa: PLC0415
        store = get_store(self.app)
        if store is None:
            return
        mn = (float(np.nextafter(x0, -np.inf)),
              float(np.nextafter(y0, -np.inf)), -1.0e12)
        mx = (float(np.nextafter(x1, np.inf)),
              float(np.nextafter(y1, np.inf)), 1.0e12)
        try:
            if not store.ensure_region(mn, mx):
                return
        except Exception:
            return
        rows = np.asarray(store.last_region_indices, dtype=np.int64).ravel()
        if rows.size == 0:
            self.selected_rows = rows
            self._update_count()
            return
        try:
            pts = np.asarray(store.data["xyz"])[rows]
        except Exception:
            return
        keep = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1)
                & (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
        if polygon is not None:
            try:
                from gui.native_tiled_runner import _points_in_polygons  # noqa: PLC0415
                keep &= _points_in_polygons(
                    pts[:, :2], np.asarray(polygon, dtype=np.float64))
            except Exception as exc:
                self.selected_rows = np.empty(0, dtype=np.int64)
                self._update_count()
                self.app.statusBar().showMessage(f'Shape selection failed: {exc}', 5000)
                return
        try:
            palette = getattr(self.app, "class_palette", None)
            if isinstance(palette, dict) and palette:
                from gui.class_slots import stored_palette  # noqa: PLC0415
                vis = [int(c) for c, i in stored_palette(self.app, palette).items()
                       if isinstance(i, dict) and i.get("show", True)]
                cls = np.asarray(store.data["classification"])[rows]
                keep &= np.isin(cls, np.asarray(vis))
        except Exception:
            pass
        if hasattr(store, 'deleted'):
            keep &= ~store.deleted[rows]
        self.selected_rows = rows[keep]
        self._update_count()

    def _update_count(self):
        n = int(self.selected_rows.size)
        try:
            ribbon = getattr(getattr(self.app, "ribbon_manager", None),
                             "ribbons", {}).get("identify")
            if ribbon is not None and hasattr(ribbon, "update_selected_count"):
                ribbon.update_selected_count(n)
        except Exception:
            pass
        if n:
            try:
                self.app.statusBar().showMessage(f"{n:,} points selected", 3000)
            except Exception:
                pass

    @property
    def selected_count(self):
        return int(self.selected_rows.size)

    def _confirm_selection(self):
        if not self.selected_count:
            return
        from gui.lnt_section_bridge import get_store
        store = get_store(self.app)
        try:
            self._overlay_for(self._canvas()).set_selected_points(store.data['xyz'][self.selected_rows])
            # Present outline and highlights before entering the modal confirmation.
            self.view.native.frame(0.0)
        except Exception as exc:
            self.deactivate_selection()
            self.app.statusBar().showMessage(f'Selection highlight failed: {exc}', 5000)
            return
        self.delete_selected_points()

    def delete_selected_points(self):
        if not self.selected_count:
            return
        from gui.lnt_section_bridge import get_store
        store = get_store(self.app)
        if store is None:
            return
        rows = store.live_rows(self.selected_rows)
        if not len(rows):
            self.deactivate_selection()
            return
        reply = QMessageBox.question(
            self.app, 'Delete Selection',
            f'Delete {len(rows):,} selected points?\n'
            'You can undo this change. Points are permanently removed only when you save.',
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        try:
            store.set_deleted(rows)
        except Exception as exc:
            self.app.statusBar().showMessage(f'Delete failed: {exc}', 5000)
            return
        entry = {'deleted_rows': rows.copy()}
        if not hasattr(self.app, 'undo_stack'):
            self.app.undo_stack = []
        self.app.undo_stack.append(entry)
        from gui.memory_manager import trim_undo_stack
        trim_undo_stack(self.app)
        if hasattr(self.app, 'redo_stack'):
            self.app.redo_stack.clear()
        self.deactivate_selection()
        self.app.statusBar().showMessage(
            f'Deleted {len(rows):,} points (Ctrl+Z to undo; Save to make permanent)', 5000)
