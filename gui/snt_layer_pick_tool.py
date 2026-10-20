"""
SNT layer pick tool.

Provides a dedicated click mode in main view that identifies which SNT layer
the clicked feature belongs to, without reusing point-class identification.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QLabel


class SNTLayerPickTool(QObject):
    """Dedicated SNT layer identifier for main-view left-click."""

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.active = False
        self._main_observer = None
        self._prop_picker = None
        self._cell_picker = None
        self._native_pick = None
        self._surface_dispatch = False
        self._click_popup_label = None
        self._click_popup_timer = QTimer(self)
        self._click_popup_timer.setSingleShot(True)
        self._click_popup_timer.timeout.connect(self._hide_click_popup)
        print("✅ SNTLayerPickTool initialized")

    def activate(self):
        if self.active:
            print("⚠️ SNT layer pick tool already active")
            return

        # Keep click behavior deterministic: only one footer click tool at a time.
        self._deactivate_conflicting_tools()

        self.active = True
        # Native first: when the main canvas is the Vulkan viewport the VTK
        # interactor never sees clicks, so they must be picked natively
        # (segment hit-test against the GPU vector arena - no VTK).
        native_ok = self._attach_native_input()
        if native_ok:
            print("   ✅ SNT layer pick native click input armed (no VTK observers)")
        else:
            print("   ⚠️ SNT layer pick native arming failed - VTK fallback armed")
        # Always keep the legacy VTK observer as a fallback: it is harmless
        # when the Vulkan viewport consumes clicks (it simply never fires),
        # and it keeps Identify working if the canvas is VTK or the native
        # view goes away mid-session (e.g. .lnt closed while tool is armed).
        self._attach_main_observer()
        self._sync_cursor(True)
        self._sync_footer_button_state(True)
        self._set_footer_text(None, None, None)
        print("🎯 SNT layer pick tool ACTIVATED")

    def deactivate(self):
        if not self.active:
            return

        self.active = False
        if self._native_pick is not None:
            try:
                self._native_pick.deactivate()
            except Exception:
                pass
            self._native_pick = None
            self._surface_dispatch = False
        self._detach_main_observer()
        self._hide_click_popup()
        self._sync_cursor(False)
        self._sync_footer_button_state(False)
        self._set_footer_text(None, None, None)
        print("🎯 SNT layer pick tool DEACTIVATED")

    def _deactivate_conflicting_tools(self):
        for tool_name in ("point_sync_tool", "coordinate_pick_tool"):
            tool = getattr(self.app, tool_name, None)
            if tool is not None and getattr(tool, "active", False):
                try:
                    tool.deactivate()
                except Exception:
                    pass

    def _attach_main_observer(self):
        if self._main_observer is not None:
            return

        vtk_widget = getattr(self.app, "vtk_widget", None)
        if vtk_widget is None or not hasattr(vtk_widget, "interactor"):
            return

        try:
            self._main_observer = vtk_widget.interactor.AddObserver(
                "LeftButtonPressEvent",
                self._on_main_click,
                -0.9,
            )
            print("   ✅ SNT layer pick observer attached to Main View")
        except Exception as e:
            print(f"   ⚠️ Failed to attach SNT layer pick observer: {e}")

    def _detach_main_observer(self):
        if self._main_observer is None:
            return

        vtk_widget = getattr(self.app, "vtk_widget", None)
        if vtk_widget is None or not hasattr(vtk_widget, "interactor"):
            self._main_observer = None
            return

        try:
            vtk_widget.interactor.RemoveObserver(self._main_observer)
        except Exception as e:
            print(f"   ⚠️ Failed to remove SNT layer pick observer: {e}")
        finally:
            self._main_observer = None

    def _attach_native_input(self) -> bool:
        """Arm click input on the native Vulkan main viewport (no VTK).

        Two native surfaces:
          - an .lnt session: the session's ``_InputFilter`` forwards press/key
            to a ``NativeLayerPick`` tool hung on the view;
          - an SNT-only session: the promoted viewport surface dispatches
            plain left-clicks via ``dispatch_surface_left_click``.

        Returns False (and arms nothing) when the GPU vector arena is not
        in use, so the legacy VTK observer path keeps working there.
        """
        try:
            from gui.native_vector_overlay import _view_for
            view = _view_for(self.app)
        except Exception:
            view = None

        if view is not None and getattr(view, "native", None) is not None:
            from gui.native_attachments import is_native_main
            if is_native_main(self.app):
                try:
                    from gui.native_layer_pick import NativeLayerPick
                    lp = getattr(view, "native_layer_pick", None)
                    if lp is None or getattr(lp, "view", None) is not view:
                        lp = view.native_layer_pick = NativeLayerPick(view)
                    lp.activate()
                    self._native_pick = lp
                    return True
                except Exception as e:
                    print(f"   ⚠️ Native layer pick input arming failed: {e}")
                    return False

        try:
            from gui.native_vector_overlay import vector_layer_for
            native_layer = vector_layer_for(self.app)
        except Exception:
            native_layer = None
        if native_layer is None:
            return False

        # SNT-only promoted viewport: no _InputFilter to feed, the Vulkan
        # surface dispatches clicks instead.
        self._surface_dispatch = True
        return True

    def ensure_native_input(self) -> bool:
        """(Re-)arm native click input if an .lnt session is now live.

        Called lazily on every surface click so Identify keeps working when
        the point cloud loads after the tool was activated, or when the view
        is recreated mid-session. Never deactivates the tool.
        """
        if not self.active:
            return False
        try:
            from gui.native_attachments import is_native_main
            if not is_native_main(self.app):
                return self._native_pick is not None or self._surface_dispatch
            lp = getattr(self._native_pick, "view", None)
            from gui.native_vector_overlay import _view_for
            view = _view_for(self.app)
            if (self._native_pick is not None and view is not None
                    and lp is view and getattr(self._native_pick, "active", False)):
                return True
            return bool(self._attach_native_input())
        except Exception:
            return False

    def on_surface_click(self, world_xy, upp) -> bool:
        """A left-click to identify on the native Vulkan main canvas.

        ``world_xy`` is already resolved in world coordinates (from
        MainCamera2D - native screen_to_world, no VTK).
        """
        if not self.active:
            return False
        # The .lnt may have loaded AFTER this tool was activated (or been
        # closed/reopened mid-session): lazily arm the native pick so the
        # click never dies silently.
        self.ensure_native_input()
        try:
            wx, wy = float(world_xy[0]), float(world_xy[1])
        except Exception:
            return False
        try:
            picked = self._pick_native_geometry(wx, wy, upp)
        except Exception as e:
            print(f"   ⚠️ Native layer pick failed: {e}")
            self._hide_click_popup()
            return True

        if picked is None:
            self._hide_click_popup()
            self._set_footer_text(None, None, None)
            self.app.statusBar().showMessage("Identify Layer: no layer linework near click", 2500)
            return True

        layer_name, source_name = picked
        entity_count = self._resolve_layer_entity_count(source_name, layer_name)
        self._set_footer_text(layer_name, source_name, entity_count)
        self._show_click_popup(layer_name, source_name, entity_count)
        self.app.statusBar().showMessage(f"Layer: {layer_name}", 4000)
        return True

    def _pick_native_geometry(self, wx: float, wy: float, upp: float):
        """Nearest visible native linework segment to a world point.

        Searches every document of the GPU vector arena
        (``gui.native_vector_overlay``) in world XY - the chunk vertices keep
        authored world coordinates, so no projection pass is needed. Returns
        ``(layer_name, source_name)`` or None when nothing is within the
        pick tolerance.
        """
        import numpy as np

        try:
            from gui.native_vector_overlay import vector_layer_for
            layer = vector_layer_for(self.app)
        except Exception:
            layer = None
        if layer is None:
            return None

        tolerance = max(float(upp or 1.0) * 8.0, 1.0)
        tolerance_sq = float(tolerance) * float(tolerance)

        # Dynamic overlays (identify outlines, point targets ...) are camera
        # UI aids, not attachment linework - they must never report a level.
        internal_kinds = {"identify", "target", "classify", "fence", "measure"}

        best = None            # (dist_sq, layer_name, source_name)
        for key, document in list(getattr(layer, "documents", {}).items()):
            try:
                kind = str(key[0]).lower()
            except Exception:
                kind = ""
            if kind in internal_kinds:
                continue
            handles = document.get("handles") or {}
            for group_name, chunks in (document.get("gpu") or {}).items():
                handle = handles.get(group_name)
                if handle is not None and not getattr(handle, "_visible", True):
                    continue
                for chunk in chunks or ():
                    dist_sq, index = _chunk_nearest_segment(chunk, wx, wy)
                    if dist_sq is None or index is None:
                        continue
                    if dist_sq <= tolerance_sq and (best is None or dist_sq < best[0]):
                        best = (float(dist_sq), str(group_name), str(key[1]))

        if best is None:
            return None
        return best[1], best[2]

    def _on_main_click(self, obj, event):
        if not self.active:
            return

        actor = self._pick_main_actor()
        if actor is None:
            self._hide_click_popup()
            return

        layer_name, snt_name = self._resolve_actor_layer(actor)
        if not layer_name:
            self._hide_click_popup()
            return

        entity_count = self._resolve_layer_entity_count(snt_name, layer_name)
        self._set_footer_text(layer_name, snt_name, entity_count)
        self._show_click_popup(layer_name, snt_name, entity_count)

    def _pick_main_actor(self):
        vtk_widget = getattr(self.app, "vtk_widget", None)
        if vtk_widget is None or not hasattr(vtk_widget, "interactor"):
            return None

        # Get overlay renderer (where GIS vector layers live) and main renderer (where SNT/DXF layers live)
        overlay_renderer = None
        try:
            from gui.gis.gis_layers import _overlay_renderer
            overlay_renderer = _overlay_renderer(self.app)
        except Exception:
            pass
        main_renderer = getattr(vtk_widget, "renderer", None)

        renderers = [r for r in [overlay_renderer, main_renderer] if r is not None]
        if not renderers:
            return None

        try:
            x, y = vtk_widget.interactor.GetEventPosition()
        except Exception:
            return None

        try:
            import vtk

            if self._prop_picker is None:
                self._prop_picker = vtk.vtkPropPicker()

            if self._cell_picker is None:
                self._cell_picker = vtk.vtkCellPicker()
                self._cell_picker.SetTolerance(0.01)
        except Exception as e:
            print(f"   ⚠️ Failed to initialize SNT pickers: {e}")
            return None

        for ren in renderers:
            try:
                if self._prop_picker.Pick(x, y, 0, ren):
                    picked = self._prop_picker.GetViewProp()
                    if picked is not None:
                        return picked
            except Exception:
                pass

            try:
                if self._cell_picker.Pick(x, y, 0, ren):
                    picked = self._cell_picker.GetActor()
                    if picked is not None:
                        return picked
            except Exception:
                pass

        return None

    def _resolve_actor_layer(self, actor) -> Tuple[Optional[str], Optional[str]]:
        # Fast path: actor-level tags attached during SNT/DXF render.
        layer_name = str(getattr(actor, "_naksha_snt_layer", "") or "").strip()
        if layer_name:
            source_name = str(getattr(actor, "_naksha_snt_filename", "") or "").strip()
            return layer_name, source_name or None

        # Fallback path: resolve by object identity in attachment caches.
        for list_name in ["snt_attachments", "dxf_attachments"]:
            for attachment in getattr(self.app, list_name, []) or []:
                cache = attachment.get("actor_cache_map", {}) or {}
                source_name = attachment.get("filename") or attachment.get("full_path")
                for cached_layer, actors in cache.items():
                    if not actors:
                        continue
                    for cached_actor in actors:
                        if cached_actor is actor:
                            layer_value = str(cached_layer or "").strip()
                            if layer_value:
                                return layer_value, source_name

        # Fallback path 2: resolve by GIS layers registry (Shapefiles, GeoJSON, GDB, etc.)
        try:
            from gui.gis.gis_layers import _registry
            reg = _registry(self.app)
            for entry in reg:
                layer_actors = entry.get("actors", [])
                if actor in layer_actors:
                    layer_name = entry.get("name")
                    source_name = entry.get("path")
                    # Try to locate specific subtype/subclass if applicable
                    sub_layers = entry.get("sub_layers", {})
                    if sub_layers:
                        for sub_code, sub_info in sub_layers.items():
                            sub_actors = sub_info.get("actors", [])
                            if actor in sub_actors:
                                sub_label = sub_info.get("label")
                                if sub_label and sub_label != layer_name:
                                    return f"{layer_name} · {sub_label}", source_name
                    return layer_name, source_name
        except Exception:
            pass

        return None, None

    def _resolve_layer_entity_count(self, snt_name: Optional[str], layer_name: str) -> Optional[int]:
        lname = str(layer_name or "").strip()
        if not lname:
            return None

        snt_base = os.path.basename(str(snt_name)) if snt_name else None
        for list_name in ["snt_attachments", "dxf_attachments"]:
            for attachment in getattr(self.app, list_name, []) or []:
                att_name = attachment.get("filename") or attachment.get("full_path")
                if snt_base and att_name and os.path.basename(str(att_name)) != snt_base:
                    continue

                rows = None
                item = attachment.get("_snt_item") if isinstance(attachment, dict) else None
                if item is not None:
                    try:
                        refresh = getattr(item, "_refresh_live_layer_stats", None)
                        if callable(refresh):
                            rows = list(refresh())
                        else:
                            rows = getattr(item, "layer_stats_cache", None)
                    except Exception:
                        rows = getattr(item, "layer_stats_cache", None)
                if rows is None:
                    rows = attachment.get("layer_stats", []) or []

                for row in rows:
                    try:
                        row_layer = str(row[0] or "").strip()
                        row_count = int(row[1])
                    except Exception:
                        continue
                    if row_layer == lname:
                        return row_count
        return None

    def _set_footer_text(self, layer_name: Optional[str], snt_name: Optional[str], entity_count: Optional[int]):
        try:
            if hasattr(self.app, "_set_snt_layer_pick_footer_text"):
                self.app._set_snt_layer_pick_footer_text(layer_name, snt_name, entity_count)
        except Exception:
            pass

    def _sync_footer_button_state(self, enabled: bool):
        try:
            if hasattr(self.app, "_sync_snt_layer_pick_footer_button"):
                self.app._sync_snt_layer_pick_footer_button(bool(enabled))
        except Exception:
            pass

    def _sync_cursor(self, enabled: bool):
        try:
            if hasattr(self.app, "set_cross_cursor_active"):
                self.app.set_cross_cursor_active(bool(enabled), "snt_layer_pick")
        except Exception:
            pass

    def _show_click_popup(
        self,
        layer_name: str,
        snt_name: Optional[str],
        entity_count: Optional[int],
    ):
        layer_text = str(layer_name or "").strip()
        if not layer_text:
            return

        try:
            self._ensure_click_popup_label()
            label = self._click_popup_label
            if label is None:
                return

            label.setText(layer_text)
            label.adjustSize()

            global_pos = QCursor.pos()
            host = self.app
            local = host.mapFromGlobal(global_pos)
            x = int(local.x()) + 14
            y = int(local.y()) + 18

            margin = 8
            host_w = int(host.width())
            host_h = int(host.height())
            lbl_w = int(label.width())
            lbl_h = int(label.height())

            x = max(margin, min(x, host_w - lbl_w - margin))
            y = max(margin, min(y, host_h - lbl_h - margin))

            label.move(x, y)
            label.raise_()
            label.show()

            self._click_popup_timer.stop()
            self._click_popup_timer.start(2000)
        except Exception:
            pass

    def _hide_click_popup(self):
        try:
            if self._click_popup_timer.isActive():
                self._click_popup_timer.stop()
            if self._click_popup_label is not None:
                self._click_popup_label.hide()
        except Exception:
            pass

    def _ensure_click_popup_label(self):
        if self._click_popup_label is not None:
            return

        try:
            label = QLabel(self.app)
            label.setObjectName("sntLayerPickPopup")
            label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            label.setStyleSheet(
                """
                QLabel#sntLayerPickPopup {
                    background: rgba(20, 20, 24, 230);
                    color: #F6F7FB;
                    border: 1px solid rgba(120, 200, 255, 180);
                    border-radius: 6px;
                    padding: 4px 8px;
                    font-size: 11px;
                    font-weight: 600;
                }
                """
            )
            label.hide()
            self._click_popup_label = label
        except Exception:
            self._click_popup_label = None


def _chunk_nearest_segment(chunk, wx: float, wy: float):
    """(dist_sq, vertex_index) of the chunk segment closest to a world point.

    Chunk layout is the ``group_entities`` contract: ``xyz`` holds authored
    WORLD coordinates (rebinding/picking keeps them) and ``indices`` is the
    restart-separated strip list indexing into ``xyz``. Returns (None, None)
    for an unusable chunk.
    """
    import numpy as np

    try:
        xyz = np.asarray(chunk.get("xyz"))
        idx = np.asarray(chunk.get("indices"))
    except Exception:
        return None, None
    if xyz.ndim != 2 or xyz.shape[0] == 0 or xyz.shape[1] < 2 or idx.size < 2:
        return None, None

    a_idx = idx[:-1].astype(np.int64)
    b_idx = idx[1:].astype(np.int64)
    live = (a_idx != 0xFFFFFFFF) & (b_idx != 0xFFFFFFFF)
    a_idx = a_idx[live]
    b_idx = b_idx[live]
    if a_idx.size == 0:
        return None, None

    ax = xyz[a_idx, 0]
    ay = xyz[a_idx, 1]
    bx = xyz[b_idx, 0]
    by = xyz[b_idx, 1]

    lo_x = np.minimum(ax, bx)
    hi_x = np.maximum(ax, bx)
    lo_y = np.minimum(ay, by)
    hi_y = np.maximum(ay, by)

    # Cheap bbox prefilter before the exact segment distance.
    dx_bbox = np.maximum(np.maximum(lo_x - wx, wx - hi_x), 0.0)
    dy_bbox = np.maximum(np.maximum(lo_y - wy, wy - hi_y), 0.0)
    cand = (dx_bbox * dx_bbox + dy_bbox * dy_bbox) <= (64.0 * 64.0)
    if not np.any(cand):
        return None, None

    ax = ax[cand]
    ay = ay[cand]
    bx = bx[cand]
    by = by[cand]
    a_idx = a_idx[cand]

    vx = bx - ax
    vy = by - ay
    px = wx - ax
    py = wy - ay
    denom = vx * vx + vy * vy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(denom > 0.0, (px * vx + py * vy) / denom, 0.0)
    np.clip(t, 0.0, 1.0, out=t)
    cx = ax + t * vx
    cy = ay + t * vy
    dist_sq = (cx - wx) * (cx - wx) + (cy - wy) * (cy - wy)
    best = int(np.argmin(dist_sq))
    return float(dist_sq[best]), int(a_idx[best])
