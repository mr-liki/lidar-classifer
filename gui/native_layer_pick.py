"""Layer pick (Identify Layer) for the native LNT/Vulkan plan viewport; no VTK.

Click path: screen -> ``MainCanvas.section_at`` (native
``screen_to_world``) -> nearest visible GPU-vector linework segment in world
XY (``SNTLayerPickTool._pick_native_geometry``) -> footer label + click popup
+ status bar.  Mirrors ``NativeIdentify``/``NativeZoomRect`` so the shared
``_InputFilter`` in ``gui.lnt_open`` can drive it with no changes.
"""
from __future__ import annotations

from PySide6.QtCore import Qt


class NativeLayerPick:
    def __init__(self, view):
        self.view = view
        self.app = view.app
        self.active = False

    def _canvas(self):
        from gui.native_main_classify import MainCanvas  # noqa: PLC0415
        return MainCanvas(self.view)

    def _tool(self):
        return getattr(self.app, "snt_layer_pick_tool", None)

    def activate(self):
        self.active = True
        try:
            self._canvas().setFocus()
        except Exception:
            pass
        try:
            self.app.statusBar().showMessage(
                "Identify Layer: click SNT/DXF/GIS linework", 4000)
        except Exception:
            pass
        return True

    def cancel(self):
        self.active = False

    def deactivate(self):
        self.cancel()

    def press(self, pos, button):
        if not self.active or button != Qt.LeftButton:
            return False
        tool = self._tool()
        if tool is None or not getattr(tool, "active", False):
            return False
        try:
            canvas = self._canvas()
            wx, wy = canvas.section_at(pos.x(), pos.y())
        except Exception:
            return False
        try:
            upp = float(canvas.world_per_pixel())
        except Exception:
            upp = 1.0
        try:
            picked = tool._pick_native_geometry(float(wx), float(wy), upp)
        except Exception:
            picked = None
        if picked is None:
            try:
                tool._hide_click_popup()
                tool._set_footer_text(None, None, None)
                self.app.statusBar().showMessage(
                    "Identify Layer: no layer linework near click", 2500)
            except Exception:
                pass
            return True
        try:
            layer_name, source_name = picked
            entity_count = tool._resolve_layer_entity_count(
                source_name, layer_name)
            tool._set_footer_text(layer_name, source_name, entity_count)
            tool._show_click_popup(layer_name, source_name, entity_count)
            self.app.statusBar().showMessage(
                f"Layer: {layer_name}", 4000)
        except Exception:
            pass
        return True

    def move(self, pos):  # noqa: ARG002
        return False

    def release(self, pos, button):  # noqa: ARG002
        # Single-click tool: press already consumed the pick; staying active
        # (don't deactivate) so repeated clicks keep identifying.
        return False

    def key(self, key):
        if not self.active:
            return False
        if key == Qt.Key_Escape:
            self.cancel()
            try:
                tool = self._tool()
                if tool is not None and getattr(tool, "active", False):
                    tool.deactivate()
            except Exception:
                pass
            return True
        return False
