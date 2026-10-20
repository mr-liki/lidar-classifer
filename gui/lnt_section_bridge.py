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

Native section docks are implemented in gui.native_sections. They share the
main viewport's Vulkan device and retain these source identities for edits.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

# Largest number of points the section store may hold. A section is a corridor, not the
# cloud: this bounds memory (about 45 bytes per point) and refuses a runaway region.
MAX_STORE_POINTS = 30_000_000

try:
    from numba import njit as _njit
except Exception:  # noqa: BLE001
    _njit = None

if _njit is not None:
    @_njit(cache=True, nogil=True)
    def _merge_searchsorted(haystack, needles):
        # np.searchsorted(haystack, needles) (side='left') for SORTED needles:
        # one linear merge instead of a binary search per needle.
        out = np.empty(needles.shape[0], dtype=np.int64)
        j, n = 0, haystack.shape[0]
        for i in range(needles.shape[0]):
            v = needles[i]
            while j < n and haystack[j] < v:
                j += 1
            out[i] = j
        return out


    @_njit(cache=True, nogil=True)
    def _merge_rows(haystack, positions, needles):
        # Store row of each SORTED needle key, -1 when the store lacks it.
        out = np.empty(needles.shape[0], dtype=np.int64)
        j, n = 0, haystack.shape[0]
        for i in range(needles.shape[0]):
            v = needles[i]
            while j < n and haystack[j] < v:
                j += 1
            out[i] = positions[j] if j < n and haystack[j] == v else -1
        return out


def _store_rows(haystack, positions, needles, needles_sorted):
    if _njit is not None and needles_sorted and len(needles) > 4096:
        return _merge_rows(haystack, positions, needles)
    rows = np.full(len(needles), -1, dtype=np.int64)
    at = np.searchsorted(haystack, needles)
    hit = at < len(haystack)
    hit[hit] = haystack[at[hit]] == needles[hit]
    rows[hit] = positions[at[hit]]
    return rows


def _sorted_positions(haystack, needles, needles_sorted):
    if _njit is not None and needles_sorted and len(needles) > 4096:
        return _merge_searchsorted(haystack, needles)
    return np.searchsorted(haystack, needles)
FLUSH_INTERVAL_MS = 400
SHOW_WRITE_DELAY_MS = 30     # after show(): write once the recoloured frame is presented


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


def release(app, flush_pending=True):
    """Flush pending edits (unless discarded) and give app.data back. Safe anytime."""
    native_sections = getattr(app, '_native_sections', None)
    if native_sections is not None:
        native_sections.close()
        app._native_sections = None
    store = getattr(app, "_lnt_section_store", None)
    if store is not None:
        try:
            store.release(flush_pending=flush_pending)
        except TypeError:
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
        # A timer flush is not a user Save. Keep a separate checkpoint so
        # Don't Save cannot undo classifications already explicitly saved.
        self._saved_classes = np.empty(0, dtype=np.uint8)
        self.rgb8 = np.empty((0, 3), dtype=np.uint8)     # file colours as uploaded to section views
        self.generation = 0                              # bumps whenever flush finds class edits
        # keys kept sorted (with their store positions) so a query is matched
        # against the store in O(m log n), not an np.isin over the whole store
        self._node_rows = {}                             # leaf node -> (local -> store row)
        self._buffers = {}                               # column name -> backing array with spare rows
        self.last_region_indices = np.empty(0, dtype=np.int64)  # store rows of the last query box
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

    def _append(self, name, cur, new):
        """cur + new as one array. When cur is the head of this column's backing buffer
        the new rows are written into its spare capacity (no copy of the existing rows);
        otherwise (first use, full, or cur replaced elsewhere) a buffer with 50 % headroom
        is allocated."""
        n0, n1 = cur.shape[0], cur.shape[0] + new.shape[0]
        buf = self._buffers.get(name)
        if (buf is None or cur.base is not buf or buf.shape[0] < n1 or buf.dtype != cur.dtype
                or (n0 and cur.ctypes.data != buf.ctypes.data)):
            dtype = cur.dtype if n0 else new.dtype
            buf = np.empty((max(n1, n0 + n0 // 2, 4096),) + new.shape[1:], dtype=dtype)
            buf[:n0] = cur
            self._buffers[name] = buf
        buf[n0:n1] = new
        return buf[:n1]

    # ---- loading a region ----------------------------------------------------------
    def ensure_region(self, mn, mx):
        """Make sure every point inside the world box [mn, mx] is in app.data.
        Returns True on success; on failure last_error says why."""
        self.last_error = ""
        self.last_region_indices = np.empty(0, dtype=np.int64)
        remaining = MAX_STORE_POINTS - len(self)
        res = self.view.native.query_points(mn, mx, max_points=max(remaining, 1) + len(self))
        return self._ingest(res)

    def ensure_stroke(self, path, radius, square):
        """Make sure every point a plan-view brush stroke covers is in app.data; on success
        last_region_indices holds exactly those store rows (no bounding-box surplus)."""
        self.last_error = ""
        self.last_region_indices = np.empty(0, dtype=np.int64)
        remaining = MAX_STORE_POINTS - len(self)
        res = self.view.native.query_stroke(path, radius, square,
                                            max_points=max(remaining, 1) + len(self))
        return self._ingest(res)

    def _ingest(self, res):
        if res is None:
            self.last_error = self.view.native.last_error or "the section query failed"
            return False
        n = int(res["xyz"].shape[0])
        if n == 0:
            return True
        node, local = res["node"], res["local"]
        base = len(self)
        # Store rows by identity: one row table per leaf node (local -> row, -1 = absent),
        # so matching costs O(query) however large the store has grown. The queries
        # return records grouped node by node.
        cuts = np.flatnonzero(node[1:] != node[:-1]) + 1
        bounds = np.concatenate(([0], cuts, [n]))
        rows = np.empty(n, dtype=np.int64)
        tables = self._node_rows
        for s0, s1 in zip(bounds[:-1], bounds[1:]):
            loc = local[s0:s1]
            tbl = tables.get(int(node[s0]))
            if tbl is None:
                rows[s0:s1] = -1
            else:
                inside = loc < tbl.shape[0]
                part = np.full(s1 - s0, -1, dtype=np.int64)
                part[inside] = tbl[loc[inside]]
                rows[s0:s1] = part
        fresh = rows < 0
        m = int(np.count_nonzero(fresh))
        if m == 0:
            self.last_region_indices = rows
            return True
        if base + m > MAX_STORE_POINTS:
            self.last_error = (f"the section holds {base + m:,} points (limit "
                               f"{MAX_STORE_POINTS:,}) - use a narrower section or clear the sections")
            return False
        take = (lambda a: a) if m == n else (lambda a: a[fresh])
        d = self.data
        cls_new = np.asarray(take(res["classification"]), dtype=np.uint8)
        rgb8 = take(res["rgb"])
        # Points already held keep their (possibly edited, not yet flushed) classes.
        # Columns grow into spare capacity: only the new rows are written.
        d["xyz"] = self._append("xyz", d["xyz"], take(res["xyz"]))
        d["classification"] = self._append("classification", np.asarray(d["classification"], dtype=np.uint8), cls_new)
        d["intensity"] = self._append("intensity", d["intensity"], take(res["intensity"]).astype(np.float32))
        self.rgb8 = self._append("rgb8", self.rgb8, rgb8)
        d["rgb"] = self._append("rgb", d["rgb"], np.multiply(rgb8, np.float32(1.0 / 255.0), dtype=np.float32))
        new_node, new_local = take(node), take(local)
        self.keys = self._append("keys", self.keys,
                                 (new_node.astype(np.uint64) << np.uint64(32)) | new_local.astype(np.uint64))
        self.node = self._append("node", self.node, new_node)
        self.local = self._append("local", self.local, new_local)
        self._synced = self._append("synced", self._synced, cls_new)
        self._saved_classes = self._append("saved_classes", self._saved_classes, cls_new)
        rows[fresh] = np.arange(base, base + m, dtype=np.int64)
        for s0, s1 in zip(bounds[:-1], bounds[1:]):
            nd, loc = int(node[s0]), local[s0:s1]
            tbl = tables.get(nd)
            need = int(loc.max()) + 1
            if tbl is None or tbl.shape[0] < need:
                grown = np.full(max(need, 2 * (0 if tbl is None else tbl.shape[0])), -1, dtype=np.int64)
                if tbl is not None:
                    grown[:tbl.shape[0]] = tbl
                tables[nd] = tbl = grown
            tbl[loc] = rows[s0:s1]
        self.last_region_indices = rows
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
        corners = np.array([p1-buffer*d-half_width*perp,
                            p1-buffer*d+half_width*perp,
                            p2+buffer*d-half_width*perp,
                            p2+buffer*d+half_width*perp])
        # Expand by one ULP so query rounding cannot lose boundary records.
        low = np.nextafter(corners.min(axis=0), -np.inf)
        high = np.nextafter(corners.max(axis=0), np.inf)
        lo = (low[0], low[1], -1.0e12)
        hi = (high[0], high[1], 1.0e12)
        return self.ensure_region(lo, hi)

    # ---- showing edits at once ------------------------------------------------------
    def show(self, indices, old_classes):
        """Display class edits of these store rows in the main view now (surface tiles are
        recoloured in place, nothing re-meshed) and present a frame; the file write
        follows right after, off the gesture's critical path. Returns the native patch
        statistics (tiles, vertices, coverage, rebuilt) or None."""
        idx = np.asarray(indices, dtype=np.int64)
        nv = getattr(self.view, "native", None)
        if idx.size == 0 or nv is None:
            return None
        stats = nv.preview_classes(self.node[idx], self.local[idx], self.data["xyz"][idx],
                                   old_classes, np.asarray(self.data["classification"])[idx])
        if stats is not None:
            self._preview_pending = True
            nv.frame(0.0)                                # present the recoloured view now
        QTimer.singleShot(SHOW_WRITE_DELAY_MS, self.flush)
        return stats

    def preview_live(self, indices, old_classes):
        """Preview speculative class edits without scheduling an LNT write.

        Used by the native Vulkan brush while the left mouse button is held.
        The canonical in-memory class array already contains the speculative
        values, so this only patches resident GPU data and presents a frame.
        ``flush()`` is transaction-gated below, which guarantees an interrupted
        stroke can be rolled back without ever reaching the .lnt file.
        """
        idx = np.asarray(indices, dtype=np.int64)
        nv = getattr(self.view, "native", None)
        if idx.size == 0 or nv is None:
            return None
        old = np.asarray(old_classes, dtype=np.uint8)
        if old.size != idx.size:
            raise ValueError("preview_live: old-class length mismatch")
        stats = nv.preview_classes(
            self.node[idx], self.local[idx], self.data["xyz"][idx],
            old, np.asarray(self.data["classification"])[idx],
        )
        if stats is not None:
            self._preview_pending = True
            nv.frame(0.0)
        return stats

    # ---- writing edits back --------------------------------------------------------
    def flush(self):
        """Write every classification that changed since the last flush into the .lnt."""
        # A native brush stroke is speculative until physical left-button
        # release.  The CPU class array and Vulkan previews change live, but
        # neither this periodic timer nor show()'s single-shot writer may make
        # those bytes durable while the gesture is incomplete.  Cancellation
        # restores app.data first, then releases this gate.
        if bool(getattr(self.app, "_native_brush_transaction_active", False)):
            return 0
        if self.app.data is not self.data:
            return 0                                     # something replaced app.data: not ours
        try:
            cur = np.asarray(self.data["classification"])
            if cur.shape[0] != self._synced.shape[0]:
                return 0
            cur = cur.astype(np.uint8, copy=False)
            changed = np.flatnonzero(cur != self._synced)
            if changed.size == 0:
                if getattr(self, '_preview_pending', False):
                    # Undo can restore the on-disk classes before the delayed write.
                    # Complete that preview transaction too, without rewriting points.
                    if self.view.native.set_classes(self.node[:0], self.local[:0], cur[:0]):
                        self._preview_pending = False
                return 0
            self.generation += 1
            ok = self.view.native.set_classes(self.node[changed], self.local[changed], cur[changed])
            if not ok:
                self.last_error = self.view.native.last_error
                print(f"[LNT-SECTION] could not write {changed.size:,} class edits: {self.last_error}")
                return 0
            self._synced[changed] = cur[changed]
            self._preview_pending = False
            self.writes += int(changed.size)
            print(f"[LNT-SECTION] wrote {changed.size:,} class edits into the .lnt file")
            return int(changed.size)
        except Exception as exc:                          # noqa: BLE001 - never break the UI timer
            print(f"[LNT-SECTION] flush failed: {exc}")
            return 0

    def mark_saved(self):
        """Advance the discard checkpoint after a successful Save of this document.

        Undo/redo history is intentionally retained: undoing a saved action is
        a new unsaved change, which Don't Save must restore to this checkpoint.
        """
        current = np.asarray(self.data['classification'], dtype=np.uint8)
        if current.shape != self._saved_classes.shape:
            raise RuntimeError('Native save checkpoint does not match the section store.')
        self._saved_classes[:] = current
        self._discarded = False

    def discard_unflushed_edits(self, stop_timer=True):
        """Discard only changes since the last successful explicit Save.

        Timer flushes may already have written unsaved edits to the LNT file;
        restore those bytes from the save checkpoint, not from undo history.
        """
        reverted = 0
        try:
            if stop_timer:
                try:
                    self._timer.stop()
                except Exception:
                    pass
            self._preview_pending = False
            data = self.data
            if not isinstance(data, dict):
                return 0
            cur = np.asarray(data.get("classification", None))
            if cur.size == 0:
                return 0
            saved = self._saved_classes
            if saved.shape != cur.shape:
                raise RuntimeError('Cannot discard: native save checkpoint is incomplete.')
            reverted = int(np.count_nonzero(cur != saved))
            if self._synced.shape != saved.shape:
                raise RuntimeError('Cannot discard: native disk state is incomplete.')
            rows = np.flatnonzero(saved != self._synced)
            if rows.size:
                nv = getattr(self.view, 'native', None)
                if nv is None or not nv.set_classes(self.node[rows], self.local[rows], saved[rows]):
                    raise OSError(getattr(nv, 'last_error', '') or 'Could not restore saved native classes.')
                print(f"[LNT-SECTION] reverted {rows.size:,} class edits on Don't Save")
            # Only acknowledge the rollback after its native write succeeded.
            cur[:] = saved
            self._synced[:] = saved
            self._preview_pending = False
            self._discarded = True
        except Exception as exc:                          # noqa: BLE001 - never break clear
            self.last_error = str(exc)
            print(f"[LNT-SECTION] discard_unflushed_edits failed: {exc}")
            return 0
        return int(reverted)

    def release(self, flush_pending=True):
        try:
            self._timer.stop()
        except Exception:                                 # noqa: BLE001
            pass
        if flush_pending and not getattr(self, "_discarded", False):
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
        # Freshness for finished boxes: a box belongs to the dataset revision
        # that produced it. _tick compares against the live LNT revision and
        # drops boxes from a previous file, so opening a new cloud can never
        # leave old magenta rectangles hanging over the fresh data (the extra
        # boxes floating outside the cloud in the report).
        self._section_rev = {}
        self._lnt_rev = None

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
        self._section_rev[int(view_idx)] = self._lnt_rev
        self.update()

    def clear_sections(self):
        self.sections.clear()
        self._section_rev.clear()
        self.update()

    def _tick(self):
        w = self._widget()
        if w is None or not w.isVisible() or self.view.native is None:
            self.hide()
            return
        if not self.isVisible():
            self.show()
        # Drop boxes drawn for a previous .lnt before re-framing: the native
        # view object is rebuilt per file, so its identity is the revision.
        rev = id(self.view.native)
        if self._lnt_rev != rev:
            self._lnt_rev = rev
            if self.sections:
                self.sections.clear()
                self._section_rev.clear()
                self.preview = None
                self.update()
        # The overlay is a top-level tool window painted in widget-LOCAL
        # logical pixels (paintEvent draws the QPolygonF in local coords), so
        # frame it from the Vulkan widget's own size — never from its global
        # top-left. Offsetting by the global origin double-counts the position
        # (global pos + local coords) and shifts every preview box down-right
        # away from the cursor, exactly the offset boxes in the report.
        if self.size() != w.size():
            self.resize(w.size())
        top_left = w.mapToGlobal(QPoint(0, 0))
        if self.pos() != top_left:
            self.move(top_left)
        nv = self.view.native
        pos = self.pos()
        state = (pos.x(), pos.y(), self.width(), self.height(), self.view.mode_2d,
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
        app = self.view.app
        color = QColor.fromRgbF(*getattr(app, 'cross_line_color', (1., 0., 1.)))
        width = max(1, int(getattr(app, 'cross_line_width', 3)))
        style = {'solid': Qt.SolidLine, 'dashed': Qt.DashLine,
                 'dotted': Qt.DotLine, 'dash-dot': Qt.DashDotLine,
                 'dash-dot-dot': Qt.DashDotDotLine}.get(
                     getattr(app, 'cross_line_style', 'solid'), Qt.SolidLine)
        for idx, corners in self.sections.items():
            pts = [self._to_qt(c) for c in corners]
            if any(q is None for q in pts):
                continue
            p.setPen(QPen(color, width, style))
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
                pen = QPen(color, width, style)
                p.setPen(pen)
                if kind == "rect":
                    p.drawPolygon(QPolygonF(pts))
                    # Same yellow centreline as a finished section, so the
                    # live corridor's orientation reads while dragging.
                    p.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
                    mid_a = QPointF((pts[0].x() + pts[3].x()) / 2, (pts[0].y() + pts[3].y()) / 2)
                    mid_b = QPointF((pts[1].x() + pts[2].x()) / 2, (pts[1].y() + pts[2].y()) / 2)
                    p.drawLine(mid_a, mid_b)
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
            # A fresh gesture for this target view replaces its old finished
            # box: otherwise the previous corridor stays on screen next to the
            # live rubber band and the user cannot tell which box is being
            # drawn (the skewed/offset boxes in the report).
            try:
                ov = self._overlay()
                idx = getattr(sc, "active_view", None)
                if ov is not None and idx is not None and int(idx) in ov.sections:
                    del ov.sections[int(idx)]
                    ov.update()
            except Exception:  # noqa: BLE001
                pass
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
            # The rubber band only shows the area while it is being picked:
            # once the section is taken nothing stays on the main view.
            # Repaint now so it is gone before the section is computed.
            ov.clear_preview()
            ov.clear_sections()
            ov.repaint()
        if hw < 1e-3:
            self.app.statusBar().showMessage("Section width is zero: click farther from the line", 4000)
            return
        sc.half_width = hw
        try:
            sc.finalize_section(P1, P2)
            view_idx = getattr(sc, "active_view", None)
            if view_idx is None:
                view_idx = 0
            if getattr(self.app, '_native_sections', None) is None and hasattr(self.app, "_on_section_updated"):
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
