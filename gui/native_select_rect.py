"""Rectangle / polygon selection + count/delete for the native plan view.

No VTK: Block and Shape gestures are drawn on the click-through
``CanvasOverlay`` in world coordinates; the candidate records come from the
section store (exact leaf records with class + XYZ).  Selection is a set of
store rows (file identities), counted into the Identify ribbon.

Delete on native is intentionally read-only-safe: points are reclassified
to Noise -- the reversible, undoable equivalent -- because the append-only
.lnt octree has no record removal.
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
        self._overlay = None

    # ---- lifecycle ----------------------------------------------------
    def _canvas(self):
        from gui.native_main_classify import MainCanvas  # noqa: PLC0415
        return MainCanvas(self.view)

    def _overlay_for(self, canvas):
        from gui.native_section_classify import CanvasOverlay  # noqa: PLC0415
        if self._overlay is None or getattr(self._overlay, "canvas", None) is not canvas:
            self._overlay = CanvasOverlay(canvas)
        return self._overlay

    def activate(self):
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
                self._overlay.set_shapes([])
        except Exception:
            pass

    def deactivate(self):
        self.active = False
        self.start = self.current = None
        self.dragging = False
        self.vertices = []
        try:
            if self._overlay is not None:
                self._overlay.set_shapes([])
        except Exception:
            pass

    def deactivate_selection(self):
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
        if self.pick_method == "Shape":
            self.vertices.append(w)
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
        if not self.active or self.pick_method != "Block":
            return False
        if not self.dragging or self.start is None:
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
                ov.set_shapes([("polygon", list(self.vertices), None)])
            elif self.start is not None and self.current is not None:
                ov.set_shapes([("rect", [self.start, self.current], None)])
            else:
                ov.set_shapes([])
        except Exception:
            pass

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
        if a is None or b is None:
            return
        x0, x1 = sorted((float(a[0]), float(b[0])))
        y0, y1 = sorted((float(a[1]), float(b[1])))
        if x1 - x0 < 1e-9 or y1 - y0 < 1e-9:
            self._paint()
            return
        self._query_box(x0, x1, y0, y1)

    def _finish_shape(self):
        poly = list(self.vertices)
        self.vertices = []
        self._paint()
        if len(poly) < 3:
            return
        p = np.asarray(poly, dtype=np.float64)
        self._query_box(float(p[:, 0].min()), float(p[:, 0].max()),
                        float(p[:, 1].min()), float(p[:, 1].max()),
                        polygon=p)

    def _query_box(self, x0, x1, y0, y1, polygon=None):
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
            except Exception:
                pass
        try:
            palette = getattr(self.app, "class_palette", None)
            if isinstance(palette, dict) and palette:
                from gui.class_slots import stored_palette  # noqa: PLC0415
                vis = [int(c) for c, i in stored_palette(self.app, palette).items()
                       if isinstance(i, dict) and i.get("show", True)]
                if vis:
                    cls = np.asarray(store.data["classification"])[rows]
                    keep &= np.isin(cls, np.asarray(vis))
        except Exception:
            pass
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

    # ---- delete (reclassify-to-noise, undoable) -----------------------------
    def delete_selected_points(self):
        if self.selected_rows.size == 0:
            try:
                self.app.statusBar().showMessage("Nothing selected", 2500)
            except Exception:
                pass
            return
        from gui.lnt_section_bridge import get_store  # noqa: PLC0415
        store = get_store(self.app)
        if store is None:
            return
        n = int(self.selected_rows.size)
        reply = QMessageBox.question(
            self.app, "Delete Selection",
            f"Move {n:,} selected points to Noise (class 7)?\n"
            "Native clouds are append-only: this is the reversible, undoable "
            "equivalent of deletion.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        rows = np.asarray(self.selected_rows, dtype=np.int64)
        try:
            cur = np.asarray(store.data["classification"])
            old = cur[rows].copy()
            new = np.full(rows.size, 7, dtype=np.uint8)
            cur[rows] = new
        except Exception as exc:
            try:
                self.app.statusBar().showMessage(f"Delete failed: {exc}", 3000)
            except Exception:
                pass
            return
        try:
            from gui.native_tiled_runner import push_native_undo  # noqa: PLC0415
            push_native_undo(self.app, rows, old, new)
        except Exception:
            pass
        try:
            store.show(rows, old)
        except Exception:
            pass
        try:
            store.flush()
        except Exception:
            pass
        try:
            sections = getattr(self.app, "_native_sections", None)
            if sections is not None:
                sections.refresh_edits(force=True)
        except Exception:
            pass
        self.selected_rows = np.empty(0, dtype=np.int64)
        self._update_count()
        try:
            self.app.statusBar().showMessage(
                f"Moved {n:,} points to Noise (Ctrl+Z to undo)", 4000)
        except Exception:
            pass
