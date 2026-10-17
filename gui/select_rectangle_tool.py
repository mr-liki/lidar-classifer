"""Select Rectangle Tool (native Vulkan; VTK-free).

COMPATIBILITY SHIM. The real implementation is gui/native_select_rect.py
(NativeSelectRect). Selection state is kept here in renderer-agnostic form
(store-row ids) so ribbon/UI code that reads selected_count /
get_selection_summary keeps working.
"""
from __future__ import annotations

import numpy as np


def _native_view(app):
    try:
        from gui.lnt_section_bridge import active_view
        return active_view(app)
    except Exception:
        return None
# SHIM-MARKER-SELECT-RECT-NATIVE


class SelectRectangleTool:
    """Compatibility shim over the native select tool."""

    def __init__(self, app):
        self.app = app
        self.active = False
        self.start_pos = None
        self.selected_mask = None
        self.selected_count = 0
        self.selected_dxf_actors = []
        self.is_drawing = False
        self.pick_method = None
        self.polygon_points = []
        self.world_polygon_points = []
        self.world_start_pos = None
        self.observer_ids = []
        print("Select Rectangle tool initialized (native)")

    def _native(self, create=False):
        view = _native_view(self.app)
        if view is None:
            return None
        tool = getattr(view, "native_select", None)
        if tool is None and create:
            try:
                from gui.native_select_rect import NativeSelectRect
                tool = view.native_select = NativeSelectRect(view)
            except Exception:
                return None
        return tool

    def _sync_from_native(self):
        try:
            native = self._native()
            if native is not None:
                self.selected_count = int(native.selected_count)
                return
        except Exception:
            pass
        self.selected_count = 0

    def activate(self):
        if self.active:
            return
        native = self._native(create=True)
        if native is None:
            print("Select Rectangle needs an open native (.lnt) view")
            return
        native.activate()
        self.active = True
        self.start_pos = None
        self.selected_mask = None
        self.selected_count = 0
        self.selected_dxf_actors = []
        self.is_drawing = False
        self.polygon_points = []
        print("Select Rectangle tool activated (native)")

    def deactivate(self):
        if not self.active:
            return
        self.active = False
        try:
            native = self._native()
            if native is not None:
                native.cancel()
        except Exception:
            pass
        self.selected_mask = None
        self.selected_count = 0
        self.selected_dxf_actors = []
        self.start_pos = None
        self.is_drawing = False
        print("Select Rectangle tool deactivated (native)")

    def set_pick_method(self, method):
        if method and method not in ["Block", "Shape"]:
            return
        self.pick_method = method
        self.is_drawing = False
        self.polygon_points = []
        try:
            native = self._native()
            if native is not None:
                native.set_pick_method(method)
        except Exception:
            pass

    def delete_selected_points(self):
        native = self._native()
        if native is None:
            return
        native.delete_selected_points()
        self._sync_from_native()

    def get_selection_summary(self):
        self._sync_from_native()
        return {"point_count": int(self.selected_count), "dxf_count": 0}

    def clear_selection(self):
        try:
            native = self._native()
            if native is not None:
                native.deactivate_selection()
        except Exception:
            pass
        self.selected_mask = None
        self.selected_count = 0
        self.selected_dxf_actors = []
        self.start_pos = None
        self.is_drawing = False

    def on_left_press(self, obj, event):
        return None

    def on_mouse_move(self, obj, event):
        return None

    def on_right_click(self, obj, event):
        return None
