"""Identify (point pick) for the native LNT/Vulkan plan viewport; no VTK.

Click path: screen -> ``MainCanvas.section_at`` (native
``screen_to_world``) -> query a small world box through the section store
(exact leaf records with class + XYZ) -> nearest record in screen space ->
the shared ``IdentificationTool`` ribbon/statistics path.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt


class NativeIdentify:
    def __init__(self, view):
        self.view = view
        self.app = view.app
        self.active = False

    def _canvas(self):
        from gui.native_main_classify import MainCanvas  # noqa: PLC0415
        return MainCanvas(self.view)

    def _tool(self):
        return getattr(self.app, "identification_tool", None)

    def activate(self):
        tool = self._tool()
        # Stand down sibling interactions the same way the VTK tool owns clicks.
        try:
            if tool is not None and getattr(tool, "_cut_section_is_taking", lambda: False)():
                return False
        except Exception:
            pass
        self.active = True
        try:
            self._canvas().setFocus()
        except Exception:
            pass
        self.app.statusBar().showMessage("Native Identify: click a point", 4000)
        return True

    def cancel(self):
        self.active = False

    def deactivate(self):
        self.cancel()

    def press(self, pos, button):
        if not self.active or button != Qt.LeftButton:
            return False
        tool = self._tool()
        if tool is None:
            return False
        try:
            if getattr(tool, "_cut_section_is_taking", lambda: False)():
                return False
        except Exception:
            pass
        try:
            canvas = self._canvas()
            wx, wy = canvas.section_at(pos.x(), pos.y())
        except Exception:
            return False
        hit = self._pick(wx, wy)
        if hit is None:
            self.app.statusBar().showMessage("Native Identify: no point near click", 2500)
            return True
        store_row, xyz, class_code = hit
        try:
            name = tool.get_class_name(int(class_code))
        except Exception:
            name = str(int(class_code))
        try:
            tool.highlight_class(int(class_code))
        except Exception:
            pass
        try:
            tool._update_ribbon_info(int(class_code), name, tuple(float(v) for v in xyz),
                                     point_index=int(store_row))
        except Exception:
            pass
        try:
            tool.point_identified.emit(int(class_code), name, tuple(float(v) for v in xyz))
        except Exception:
            pass
        return True

    def move(self, pos):
        return False

    def key(self, key):
        if not self.active:
            return False
        if key == Qt.Key_Escape:
            self.cancel()
            try:
                ribbon = getattr(getattr(self.app, "ribbon_manager", None), "ribbons", {}).get("identify")
                if ribbon is not None and hasattr(ribbon, "_deactivate_identify"):
                    ribbon._deactivate_identify()
            except Exception:
                pass
            return True
        return False

    # ---- picking ------------------------------------------------------
    def _pick(self, wx, wy):
        """(store_row, xyz, class) of the nearest record to the click."""
        from gui.lnt_section_bridge import get_store  # noqa: PLC0415
        store = get_store(self.app)
        if store is None:
            return None
        try:
            wpp = self._canvas().world_per_pixel()
        except Exception:
            wpp = 1.0
        half = max(float(wpp) * 8.0, 0.5)
        mn = (float(np.nextafter(wx - half, -np.inf)),
              float(np.nextafter(wy - half, -np.inf)), -1.0e12)
        mx = (float(np.nextafter(wx + half, np.inf)),
              float(np.nextafter(wy + half, np.inf)), 1.0e12)
        try:
            if not store.ensure_region(mn, mx):
                return None
        except Exception:
            return None
        rows = np.asarray(store.last_region_indices, dtype=np.int64).ravel()
        if rows.size == 0:
            return None
        try:
            pts = np.asarray(store.data["xyz"])[rows]
            cls = np.asarray(store.data["classification"])[rows]
        except Exception:
            return None
        d2 = (pts[:, 0] - wx) ** 2 + (pts[:, 1] - wy) ** 2
        best = int(np.argmin(d2))
        if float(d2[best]) > (half * 3.0) ** 2:
            return None
        return int(rows[best]), pts[best], int(cls[best])
