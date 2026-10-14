"""Cross Section / Cut Section on a native .lnt cloud (no VTK point data involved).

While an .lnt is shown in the main viewport the application holds no point arrays
(`app.data` is empty by design), yet the Cross Section and Cut Section tools are written
against `app.data["xyz"]` / `["classification"]` and global point indices. This module
supplies exactly that, from the .lnt FILE, for the REGION a section needs:

  * LntSectionStore   queries the leaf points inside a section corridor through the native
                      DLL and keeps them as a compact `app.data`. Regions accumulate (a
                      later section appends points; earlier indices never move), so every
                      open section view keeps valid indices. Class edits made in the
                      section docks land in `app.data["classification"]`; a flush writes
                      the differences back INTO THE .lnt FILE (leaf points and their LOD
                      copies) and the open viewer refreshes - the edit survives closing the
                      file and shows in the main view.
  * LntCrossPicker    the three-click cross-section gesture (P1, P2, width) on the native
                      main view, replacing the VTK interactor style, which receives no
                      events while the Vulkan viewport is up.
  * LntSectionOverlay draws the rubber band and the finished section rectangles over the
                      native view (a translucent tool window, because a Qt widget cannot be
                      composited over a native Vulkan child window).

The section DOCKS keep their existing widgets: the engine allows one renderer per process,
and it is the main viewport's. What changed is where the points come from.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

# Largest number of points the section store may hold. A section is a corridor, not the
# cloud: this bounds memory (about 45 bytes per point) and refuses a runaway region.
MAX_STORE_POINTS = 30_000_000
FLUSH_INTERVAL_MS = 400


def active_view(app):
    """The LntMainView currently shown in the main viewport, or None."""
    v = getattr(app, "_lnt_main_view", None)
    if v is not None and getattr(v, "native", None) is not None:
        return v
    return None


def get_store(app):
    """The section store of the .lnt on screen, created on first use; None without one."""
    v = active_view(app)
    if v is None or not v.native.has_section_api():
        return None
    store = getattr(app, "_lnt_section_store", None)
    if store is None or store.view is not v:
        if store is not None:
            store.release()
        store = LntSectionStore(app, v)
        app._lnt_section_store = store
    return store


def release(app):
    """Flush pending edits and give app.data back. Safe to call at any time."""
    store = getattr(app, "_lnt_section_store", None)
    if store is not None:
        store.release()
        app._lnt_section_store = None
    v = getattr(app, "_lnt_main_view", None)
    overlay = getattr(v, "section_overlay", None) if v is not None else None
    if overlay is not None:
        overlay.close()
        v.section_overlay = None


class LntSectionStore:
    def __init__(self, app, view):
        self.app = app
        self.view = view
        self._saved_data = app.data
        self.keys = np.empty(0, dtype=np.uint64)
        self.node = np.empty(0, dtype=np.uint32)
        self.local = np.empty(0, dtype=np.uint32)
        self._synced = np.empty(0, dtype=np.uint8)       # classes as last written to the file
        self.data = {
            "xyz": np.empty((0, 3), dtype=np.float64),
            "classification": np.empty(0, dtype=np.uint8),
            "intensity": np.empty(0, dtype=np.float32),
            "rgb": np.empty((0, 3), dtype=np.float32),   # 0..1, as the loaders store it
            "_source_crs_wkt": None, "_runtime_crs_wkt": None,
            "_runtime_reprojected": False, "_source_crs_consistent": True,
            "_projection_report": None, "_layer_crs_info": [],
            "_lnt_section_store": True,
        }
        self.last_error = ""
        self.writes = 0
        self._timer = QTimer(app)
        self._timer.setInterval(FLUSH_INTERVAL_MS)
        self._timer.timeout.connect(self.flush)
        app.data = self.data
        self._timer.start()

    def __len__(self):
        return int(self.keys.shape[0])

    # ---- loading a region ----------------------------------------------------------
    def ensure_region(self, mn, mx):
        """Make sure every point inside the world box [mn, mx] is in app.data.
        Returns True on success; on failure last_error says why."""
        self.last_error = ""
        remaining = MAX_STORE_POINTS - len(self)
        res = self.view.native.query_points(mn, mx, max_points=max(remaining, 1) + len(self))
        if res is None:
            self.last_error = self.view.native.last_error or "the section query failed"
            return False
        n = int(res["xyz"].shape[0])
        if n == 0:
            return True
        k = (res["node"].astype(np.uint64) << np.uint64(32)) | res["local"].astype(np.uint64)
        fresh = ~np.isin(k, self.keys) if len(self) else np.ones(n, dtype=bool)
        m = int(fresh.sum())
        if m == 0:
            return True
        if len(self) + m > MAX_STORE_POINTS:
            self.last_error = (f"the section holds {len(self) + m:,} points (limit "
                               f"{MAX_STORE_POINTS:,}) - use a narrower section or clear the sections")
            return False
        # Points already held keep their (possibly edited, not yet flushed) classes.
        cls = np.asarray(self.data["classification"], dtype=np.uint8)
        d = self.data
        d["xyz"] = np.concatenate([d["xyz"], res["xyz"][fresh]])
        d["classification"] = np.concatenate([cls, res["classification"][fresh]])
        d["intensity"] = np.concatenate([d["intensity"], res["intensity"][fresh].astype(np.float32)])
        d["rgb"] = np.concatenate([d["rgb"], res["rgb"][fresh].astype(np.float32) / 255.0])
        self.keys = np.concatenate([self.keys, k[fresh]])
        self.node = np.concatenate([self.node, res["node"][fresh]])
        self.local = np.concatenate([self.local, res["local"][fresh]])
        self._synced = np.concatenate([self._synced, res["classification"][fresh]])
        return True

    def ensure_for_section(self, P1, P2, half_width, buffer):
        """Load the corridor of a cross section (same box finalize_section pre-filters on)."""
        p1 = np.asarray(P1[:2], dtype=np.float64)
        p2 = np.asarray(P2[:2], dtype=np.float64)
        v = p2 - p1
        length = float(np.linalg.norm(v))
        if length < 1e-9:
            return True
        d = v / length
        perp = np.array([-d[1], d[0]])
        mx_ = half_width * abs(perp[0]) + (length + buffer) * abs(d[0])
        my_ = half_width * abs(perp[1]) + (length + buffer) * abs(d[1])
        lo = (min(p1[0], p2[0]) - mx_, min(p1[1], p2[1]) - my_, -1.0e12)
        hi = (max(p1[0], p2[0]) + mx_, max(p1[1], p2[1]) + my_, 1.0e12)
        return self.ensure_region(lo, hi)

    # ---- writing edits back --------------------------------------------------------
    def flush(self):
        """Write every classification that changed since the last flush into the .lnt."""
        if self.app.data is not self.data:
            return 0                                     # something replaced app.data: not ours
        try:
            cur = np.asarray(self.data["classification"])
            if cur.shape[0] != self._synced.shape[0]:
                return 0
            cur = cur.astype(np.uint8, copy=False)
            changed = np.flatnonzero(cur != self._synced)
            if changed.size == 0:
                return 0
            ok = self.view.native.set_classes(self.node[changed], self.local[changed], cur[changed])
            if not ok:
                self.last_error = self.view.native.last_error
                print(f"[LNT-SECTION] could not write {changed.size:,} class edits: {self.last_error}")
                return 0
            self._synced[changed] = cur[changed]
            self.writes += int(changed.size)
            print(f"[LNT-SECTION] wrote {changed.size:,} class edits into the .lnt file")
            return int(changed.size)
        except Exception as exc:                          # noqa: BLE001 - never break the UI timer
            print(f"[LNT-SECTION] flush failed: {exc}")
            return 0

    def release(self):
        try:
            self._timer.stop()
        except Exception:                                 # noqa: BLE001
            pass
        self.flush()
        if self.app.data is self.data:
            self.app.data = self._saved_data


# ===================================================================================
# overlay
# ===================================================================================
class LntSectionOverlay(QWidget):
    """Translucent, click-through tool window over the native viewport."""

    def __init__(self, view):
        super().__init__(view.app, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowTransparentForInput
                         | Qt.WindowDoesNotAcceptFocus)
        self.view = view
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.preview = None            # ("line", [(x, y)...]) | ("rect", [(x, y)...]) world XY
        self.sections = {}             # view index -> 4 world corners
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._last = None

    def _widget(self):
        return getattr(self.view.owner, "vulkan_widget", None)

    def start(self):
        self._tick()
        self.show()
        self._timer.start()

    def set_preview(self, kind, pts):
        self.preview = (kind, [(float(p[0]), float(p[1])) for p in pts])
        self.update()

    def clear_preview(self):
        self.preview = None
        self.update()

    def set_section(self, view_idx, corners):
        self.sections[int(view_idx)] = [(float(c[0]), float(c[1])) for c in corners]
        self.update()

    def clear_sections(self):
        self.sections.clear()
        self.update()

    def _tick(self):
        w = self._widget()
        if w is None or not w.isVisible() or self.view.native is None:
            self.hide()
            return
        if not self.isVisible():
            self.show()
        rect = QRect(w.mapToGlobal(QPoint(0, 0)), w.size())
        if self.geometry() != rect:
            self.setGeometry(rect)
        nv = self.view.native
        state = (rect.x(), rect.y(), rect.width(), rect.height(), self.view.mode_2d,
                 tuple(nv.world_to_screen(0.0, 0.0) or ()), tuple(nv.world_to_screen(1.0, 1.0) or ()))
        if state != self._last:
            self._last = state
            self.update()

    def _to_qt(self, xy):
        w = self._widget()
        s = self.view.native.world_to_screen(xy[0], xy[1])
        if s is None or w is None:
            return None
        r = max(w.devicePixelRatioF(), 1.0)
        return QPointF(s[0] / r, s[1] / r)

    def paintEvent(self, _ev):
        if self.view.native is None or not self.view.mode_2d:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        for idx, corners in self.sections.items():
            pts = [self._to_qt(c) for c in corners]
            if any(q is None for q in pts):
                continue
            p.setPen(QPen(QColor(255, 0, 255), 2))
            p.drawPolygon(QPolygonF(pts))
            p.setPen(QPen(QColor(255, 255, 0), 2))
            mid_a = QPointF((pts[0].x() + pts[3].x()) / 2, (pts[0].y() + pts[3].y()) / 2)
            mid_b = QPointF((pts[1].x() + pts[2].x()) / 2, (pts[1].y() + pts[2].y()) / 2)
            p.drawLine(mid_a, mid_b)
            p.drawText(mid_a + QPointF(4, -4), str(idx + 1))
        if self.preview is not None:
            kind, pts_w = self.preview
            pts = [self._to_qt(c) for c in pts_w]
            if all(q is not None for q in pts):
                pen = QPen(QColor(255, 0, 255), 2, Qt.DashLine)
                p.setPen(pen)
                if kind == "rect":
                    p.drawPolygon(QPolygonF(pts))
                else:
                    p.drawPolyline(QPolygonF(pts))
        p.end()


def get_overlay(app):
    v = active_view(app)
    if v is None:
        return None
    ov = getattr(v, "section_overlay", None)
    if ov is None:
        ov = LntSectionOverlay(v)
        v.section_overlay = ov
        ov.start()
    return ov


# ===================================================================================
# the three-click gesture
# ===================================================================================
class LntCrossPicker:
    """Stands in for CrossSectionInteractor while the native viewport is up.

    click 1 = P1, click 2 = P2, click 3 = width (distance of the click from the centre
    line) and finalize - the same gesture, the same finalize_section() call."""

    def __init__(self, app, lnt_view):
        self.app = app
        self.view = lnt_view
        self.slice_state = 0
        self.P1 = None
        self.P2 = None
        self.section_controller = getattr(app, "section_controller", None)
        self.on_section_complete = None
        self._cleaned_up = False
        lnt_view.picker = self
        app.statusBar().showMessage(
            "Cross section (.lnt): click P1, click P2, then click to set the width", 6000)

    # -- helpers -------------------------------------------------------------------
    def _world(self, qpos):
        w = getattr(self.view.owner, "vulkan_widget", None)
        r = max(w.devicePixelRatioF(), 1.0) if w is not None else 1.0
        wp = self.view.native.screen_to_world(qpos.x() * r, qpos.y() * r)
        if wp is None:
            self.app.statusBar().showMessage(
                "Cross section needs the 2D plan view: press D to leave 3D orbit", 5000)
            return None
        return np.array(wp, dtype=np.float64)

    def _overlay(self):
        return get_overlay(self.app)

    @staticmethod
    def _corners(P1, P2, half_width):
        v = np.asarray(P2[:2]) - np.asarray(P1[:2])
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return None
        d = v / n
        perp = np.array([-d[1], d[0]])
        a, b = np.asarray(P1[:2]), np.asarray(P2[:2])
        return [a + perp * half_width, b + perp * half_width, b - perp * half_width,
                a - perp * half_width]

    # -- gesture ---------------------------------------------------------------------
    def on_press(self, qpos):
        """Left click. Returns True when the click was consumed."""
        if self._cleaned_up:
            return False
        pt = self._world(qpos)
        if pt is None:
            return True
        sc = self.section_controller
        if sc is None:
            return True
        if self.slice_state == 0:
            try:
                sc.clear()
            except Exception as exc:                      # noqa: BLE001
                print(f"[LNT-SECTION] clear() failed: {exc}")
            self.P1, self.slice_state = pt, 1
        elif self.slice_state == 1:
            if np.linalg.norm(pt[:2] - self.P1[:2]) < 1e-6:
                return True
            self.P2, self.slice_state = pt, 2
        else:
            self._finish(pt)
        return True

    def on_move(self, qpos):
        if self._cleaned_up or self.slice_state == 0:
            return
        pt = self._world(qpos)
        if pt is None:
            return
        ov = self._overlay()
        if ov is None:
            return
        if self.slice_state == 1:
            ov.set_preview("line", [self.P1, pt])
        else:
            hw = self._half_width(pt)
            c = self._corners(self.P1, self.P2, hw)
            if c is not None:
                ov.set_preview("rect", c)

    def _half_width(self, pt):
        v = self.P2[:2] - self.P1[:2]
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return 0.0
        d = v / n
        perp = np.array([-d[1], d[0]])
        return abs(float(np.dot(pt[:2] - self.P2[:2], perp)))

    def _finish(self, pt):
        sc = self.section_controller
        hw = self._half_width(pt)
        P1, P2 = self.P1, self.P2
        self.P1 = self.P2 = None
        self.slice_state = 0
        ov = self._overlay()
        if ov is not None:
            ov.clear_preview()
        if hw < 1e-3:
            self.app.statusBar().showMessage("Section width is zero: click farther from the line", 4000)
            return
        sc.half_width = hw
        try:
            sc.finalize_section(P1, P2)
            view_idx = getattr(sc, "active_view", None)
            if view_idx is None:
                view_idx = 0
            if ov is not None:
                c = self._corners(P1, P2, hw)
                if c is not None:
                    ov.set_section(view_idx, c)
            if hasattr(self.app, "_on_section_updated"):
                self.app._on_section_updated(view_idx)
        except Exception as exc:                          # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[LNT-SECTION] finalize failed: {exc}")

    # -- lifecycle (same surface as CrossSectionInteractor) ---------------------------
    def cancel(self):
        self.P1 = self.P2 = None
        self.slice_state = 0
        ov = getattr(self.view, "section_overlay", None)
        if ov is not None:
            ov.clear_preview()

    def cleanup(self):
        self._cleaned_up = True
        self.cancel()
        if getattr(self.view, "picker", None) is self:
            self.view.picker = None
