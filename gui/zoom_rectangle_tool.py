"""Zoom Rectangle Tool (native Vulkan; VTK-free).

Thin compatibility shim: the real implementation is
:mod:`gui.native_zoom_rect` (``NativeZoomRect``), driven through the native
input filter.  This class keeps the historical constructor/attribute API so
existing callers keep working, and delegates activation to the native tool
when an .lnt view is open.
"""
from __future__ import annotations

from PySide6.QtCore import QObject


def _native_view(app):
    try:
        from gui.lnt_section_bridge import active_view
        return active_view(app)
    except Exception:
        return None


class ZoomRectangleTool(QObject):
    """Compatibility shim over the native zoom-rectangle tool."""

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.active = False
        self.start_pos = None
        self.end_pos = None
        self.is_dragging_zoom = False
        self.is_panning = False
        self.last_pan_pos = None
        # Compat attributes (native path never fills them).
        self.rubber_band_actor = None
        self.observer_ids = []
        self.interactor = None
        print("Zoom Tool initialized (native)")

    def _native(self, create=False):
        view = _native_view(self.app)
        if view is None:
            return None
        tool = getattr(view, "native_zoom", None)
        if tool is None and create:
            try:
                from gui.native_zoom_rect import NativeZoomRect
                tool = view.native_zoom = NativeZoomRect(view)
            except Exception:
                return None
        return tool

    def activate(self):
        if self.active:
            return
        native = self._native(create=True)
        if native is None:
            print("Zoom Rectangle needs an open native (.lnt) view")
            return
        native.activate()
        self.active = True
        print("Zoom Rectangle Tool activated (native)")

    def deactivate(self):
        if not self.active:
            return
        self.active = False
        self.start_pos = None
        self.end_pos = None
        self.is_dragging_zoom = False
        self.is_panning = False
        self.last_pan_pos = None
        try:
            native = self._native()
            if native is not None:
                native.cancel()
        except Exception:
            pass
        print("Zoom Rectangle Tool deactivated (native)")

    # ---- retired VTK entry points (kept for API compatibility) ----
    def on_left_button_down(self, obj, event):
        return None

    def on_mouse_move(self, obj, event):
        return None

    def on_left_button_up(self, obj, event):
        return None

    def on_right_button_down(self, obj, event):
        self.deactivate()
        return None

    def on_middle_button_guard(self, obj, event):
        return None

    def draw_rubber_band(self, start, end):
        return None

    def zoom_to_bounds(self, min_x, max_x, min_y, max_y):
        view = _native_view(self.app)
        if view is None:
            return None
        try:
            cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
            widget = view.owner.vulkan_widget
            aspect = float(widget.width()) / max(1.0, float(widget.height()))
        except Exception:
            return None
        width_m = max(max_x - min_x, (max_y - min_y) / max(aspect, 1e-9)) * 1.05
        try:
            view.native.frame_region(float(cx), float(cy), float(width_m))
        except Exception:
            return None
        return True
