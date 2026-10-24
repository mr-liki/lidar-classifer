"""Zoom-rectangle for the native LNT/Vulkan plan viewport; no VTK.

Drag (or two clicks) a rectangle in world coordinates, then frame it with
the native ``frame_region(cx, cy, width)`` camera call.  Rubber band is
drawn by the Vulkan vector pass in the point cloud's render frame.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt


class NativeZoomRect:
    def __init__(self, view):
        self.view = view
        self.app = view.app
        self.active = False
        self.start = None      # world XY of drag start
        self.current = None    # world XY of drag current end
        self.dragging = False
        self._overlay = None
        self._canvas_instance = None
        self._shutdown = False

    # ---- lifecycle ----------------------------------------------------
    def _canvas(self):
        from gui.native_main_classify import MainCanvas  # noqa: PLC0415
        if self._canvas_instance is None:
            self._canvas_instance = MainCanvas(self.view)
        return self._canvas_instance

    def _overlay_for(self, canvas):
        from gui.native_zoom_preview import VulkanZoomPreview
        if self._overlay is None:
            self._overlay = VulkanZoomPreview(self.view)
        return self._overlay

    def activate(self):
        # Bind to the current LNT renderer; no Qt preview window is created.
        self._shutdown = False
        canvas = self._canvas()
        self._overlay_for(canvas)
        drawing = getattr(self.view, 'drawing', None)
        if drawing is not None and drawing.active():
            drawing.cancel()
        measurement = getattr(self.view, 'measurement', None)
        if measurement is not None and measurement.active:
            measurement.cancel()
        self.active = True
        self.start = self.current = None
        self.dragging = False
        try:
            self._canvas().setFocus()
        except Exception:
            pass
        self.app.statusBar().showMessage(
            "Native Zoom: drag a rectangle (left button)", 4000)
        return True

    def cancel(self):
        self.active = False
        self.start = self.current = None
        self.dragging = False
        try:
            if self._overlay is not None:
                self._overlay.clear()
        except Exception:
            pass

    def deactivate(self):
        self.cancel()

    def shutdown(self):
        """Release overlay callbacks before the native viewport is destroyed."""
        self._shutdown = True
        self.active = False
        self.start = self.current = None
        self.dragging = False
        overlay, self._overlay = self._overlay, None
        if overlay is not None:
            overlay.shutdown()
        canvas, self._canvas_instance = self._canvas_instance, None
        if canvas is not None:
            canvas._closed = True

    # ---- events (called from the native input filter) ------------------
    def press(self, pos, button):
        if not self.active or button != Qt.LeftButton:
            return False
        try:
            w = self._canvas().section_at(pos.x(), pos.y())
        except Exception:
            return False
        if self.dragging and self.start is not None:
            self.current = w
            self._finish()
            return True
        self.start, self.current, self.dragging = w, w, True
        self._paint()
        return True

    def move(self, pos):
        if not self.active or not self.dragging or self.start is None:
            return False
        try:
            self.current = self._canvas().section_at(pos.x(), pos.y())
        except Exception:
            return False
        self._paint()
        return True

    def release(self, pos, button):
        if not self.active or not self.dragging or button != Qt.LeftButton:
            return False
        try:
            self.current = self._canvas().section_at(pos.x(), pos.y())
        except Exception:
            self.current = self.current
        self._finish()
        return True

    def key(self, key):
        if not self.active:
            return False
        if key == Qt.Key_Escape:
            self.cancel()
            try:
                ribbon = getattr(getattr(self.app, "ribbon_manager", None),
                                 "ribbons", {}).get("identify")
                if ribbon is not None and hasattr(ribbon, "_deactivate_zoom_rectangle"):
                    ribbon._deactivate_zoom_rectangle()
            except Exception:
                pass
            return True
        return False

    # ---- internals ------------------------------------------------------
    def _paint(self):
        if self._shutdown:
            return
        try:
            canvas = self._canvas()
            ov = self._overlay_for(canvas)
            ov.set_rectangle(self.start, self.current)
        except Exception as exc:
            self.cancel()
            self.app.statusBar().showMessage(f'Zoom preview failed: {exc}', 5000)

    def _finish(self):
        self.dragging = False
        if self.start is None or self.current is None:
            self.start = self.current = None
            self._paint()
            return
        x0, y0 = float(self.start[0]), float(self.start[1])
        x1, y1 = float(self.current[0]), float(self.current[1])
        self.start = self.current = None
        self._paint()
        w, h = abs(x1 - x0), abs(y1 - y0)
        if w < 1e-9 or h < 1e-9:
            return
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        try:
            widget = self.view.owner.vulkan_widget
            aspect = float(widget.width()) / max(1.0, float(widget.height()))
        except Exception:
            aspect = 1.0
        width_m = max(w, h * max(aspect, 1e-9)) * 1.05
        try:
            self.view.native.frame_region(cx, cy, float(width_m))
        except Exception as exc:
            self.app.statusBar().showMessage(f"Native zoom failed: {exc}", 3000)
            return
        self.app.statusBar().showMessage(
            f"Zoomed to {w:.1f} x {h:.1f} m", 2500)
