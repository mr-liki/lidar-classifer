"""Run By Class tools on a native .lnt main view (no VTK point data).

All By Class cores (Convert/Close/Height/Fence in gui/menu_sidebar_system.py,
classify_* in gui/lidar_classification_tools.py, run_macro in
gui/by_class_macro.py) are pure numpy/scipy on ``xyz`` + ``classification``
arrays. On a native view the points live in the .lnt FILE (``app.data`` is
empty), so this module runs the SAME cores tile by tile against direct native
queries and writes back with ``set_classes`` (which also recolours the view):

* scope: whole cloud, or a fence-polygon union (tiles only cover the scope)
* tiling: XY grid sized for ~5M points/tile; a tile that still overflows the
  per-query cap is split recursively (halo preserved)
* halo: per-tool context margin (search radius etc.); only core-box rows are
  eligible, halo rows are read-only context. kNN-plane tools (Surface, Below
  Surface) and Height TIN support have no bounded range, so their halo is a
  generous heuristic and tile seams are approximate for them (documented).
* flight-line + class-palette masks are applied per tile, exactly like VTK
* one undo step ``{node, local, old_classes, new_classes}`` (file identity,
  understood by native_classification_history) per run, capped at 10 by
  trim_undo_stack like every other classification edit
* everything runs on the GUI thread (native lnt_view_* calls have GUI-thread
  affinity and must never run in the QThread workers the VTK path uses), with
  a modal progress dialog and cancel support. Cancelled runs keep already
  written tiles; the single undo step reverts them.

Limitations vs VTK: no per-point flight-line filtering (native queries do not
return point_source_id, so the filter degrades to all-True), no fence picker
(fences must already exist as data), interactive line pickers need VTK.
"""

from __future__ import annotations

import struct

import numpy as np
from PySide6.QtWidgets import QApplication, QProgressDialog

from gui.flight_line_filter import flight_line_visibility_mask
from gui.memory_manager import trim_undo_stack

TILE_TARGET_POINTS = 5_000_000
TILE_QUERY_CAP = 8_000_000
KNN_HALO = 50.0          # heuristic context for unbounded-kNN plane tools
HEIGHT_HALO_MIN = 25.0   # floor under 2x max TIN edge for Height
_Z_FULL = (-1.0e12, 1.0e12)


# --------------------------------------------------------------------------
# native view + cloud scope
# --------------------------------------------------------------------------

def native_main_view(app):
    """The live native .lnt main view, or None (VTK path then applies)."""
    view = getattr(app, "_lnt_main_view", None)
    if view is None or getattr(view, "native", None) is None:
        return None
    if not view.native.has_section_api():
        return None
    return view


def using_vtk_data(app):
    """True when point arrays are loaded (VTK or section store): native-file
    path applies only when those are absent."""
    data = getattr(app, "data", None)
    if not isinstance(data, dict):
        return False
    xyz = data.get("xyz")
    return xyz is not None and len(xyz) > 0


def want_native(app):
    """True when By Class must run against the .lnt file instead of app.data."""
    return native_main_view(app) is not None and not using_vtk_data(app)


def cloud_bbox(view):
    """(x0, y0, x1, y1, n_points) from the .lnt file header."""
    with open(view.lnt_path, "rb") as handle:
        head = struct.unpack("<4sHHQ6d", handle.read(64))
    xmin, ymin, _zmin, xmax, ymax, _zmax = head[4:10]
    trailer_at = head[3]
    with open(view.lnt_path, "rb") as handle:
        handle.seek(trailer_at)
        points, _nodes = struct.unpack("<QQ", handle.read(16))
    return float(xmin), float(ymin), float(xmax), float(ymax), int(points)


# --------------------------------------------------------------------------
# eligibility masks (same rules as the VTK path, per tile)
# --------------------------------------------------------------------------

def visible_class_mask(app, cls):
    """Hidden palette classes cannot be reclassified (copy of the VTK rule)."""
    palette = getattr(app, "class_palette", None)
    if isinstance(palette, dict) and palette:
        visible = [int(c) for c, info in palette.items()
                   if isinstance(info, dict) and info.get("show", True)]
        return np.isin(cls, np.asarray(visible))
    return np.ones(len(cls), dtype=bool)


def _single_polygon_mask(xy, coords):
    try:
        from matplotlib.path import Path
    except ImportError:
        Path = None
    poly = np.asarray(coords, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] < 2:
        return np.zeros(len(xy), dtype=bool)
    poly_xy = poly[:, :2]
    points_xy = np.asarray(xy)[:, :2]
    lo = poly_xy.min(axis=0)
    hi = poly_xy.max(axis=0)
    bbox_mask = ((points_xy[:, 0] >= lo[0]) & (points_xy[:, 0] <= hi[0])
                 & (points_xy[:, 1] >= lo[1]) & (points_xy[:, 1] <= hi[1]))
    inside = np.zeros(len(xy), dtype=bool)
    if not np.any(bbox_mask):
        return inside
    if Path is not None:
        inside[bbox_mask] = Path(poly_xy).contains_points(points_xy[bbox_mask])
        return inside
    # Compact ray-cast fallback (even-odd rule).
    px, py = points_xy[bbox_mask, 0], points_xy[bbox_mask, 1]
    vx, vy = poly_xy[:, 0], poly_xy[:, 1]
    x1, y1 = vx[:-1], vy[:-1]
    x2, y2 = vx[1:], vy[1:]
    hit = np.zeros(len(px), dtype=bool)
    for ax, ay, bx, by in zip(x1, y1, x2, y2):
        cond = ((ay > py) != (by > py)) & (px < (bx - ax) * (py - ay) / ((by - ay) + 1e-300) + ax)
        hit ^= cond
    inside[bbox_mask] = hit
    return inside


def inside_any_polygon(xy, polys):
    """Union point-in-polygon over world-coordinate fence polygons."""
    inside = np.zeros(len(np.asarray(xy)), dtype=bool)
    for coords in polys or []:
        try:
            inside |= _single_polygon_mask(xy, coords)
        except Exception:
            continue
    return inside


def fence_coords_list(fences):
    """World-coordinate polygon list from dialog fence dicts."""
    polys = []
    for fence in fences or []:
        if not isinstance(fence, dict):
            continue
        coords = fence.get("coords", [])
        try:
            arr = np.asarray(coords, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 2:
            polys.append(arr[:, :2])
    return polys


def height_filter_mask(xy_or_z, dialog):
    """Shared height-widget filter (Close/Fence/Height dialogs)."""
    try:
        enabled = dialog.height_filter_enabled.isChecked()
    except Exception:
        return None
    if not enabled:
        return None
    z = np.asarray(xy_or_z)[:, 2] if np.asarray(xy_or_z).ndim == 2 else np.asarray(xy_or_z)
    try:
        mode = dialog.height_mode_combo.currentData()
    except Exception:
        mode = "within"
    try:
        if mode == "within":
            return ((z >= dialog.min_height_spin.value())
                    & (z <= dialog.max_height_spin.value()))
        if mode == "above":
            return z >= dialog.min_height_spin.value()
        if mode == "below":
            return z <= dialog.max_height_spin.value()
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# tiling
# --------------------------------------------------------------------------

def _tile_grid(x0, y0, x1, y1, total_points, target_points):
    """Partition the bbox into core boxes of ~target_points (area-weighted)."""
    area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    if area <= 0 or total_points <= 0:
        return [((x0, y0), (x1, y1))]
    n_tiles = max(1, int(round(total_points / float(target_points))))
    aspect = (x1 - x0) / max(y1 - y0, 1e-9)
    nx = max(1, int(round((n_tiles * aspect) ** 0.5)))
    ny = max(1, int(round(n_tiles / nx)))
    while nx * ny < n_tiles:
        if (x1 - x0) / nx > (y1 - y0) / ny:
            nx += 1
        else:
            ny += 1
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    return [((xs[i], ys[j]), (xs[i + 1], ys[j + 1])) for i in range(nx) for j in range(ny)]


def _expand_box(core, halo, bounds):
    (cx0, cy0), (cx1, cy1) = core
    (bx0, by0), (bx1, by1) = bounds
    return ((max(cx0 - halo, bx0), max(cy0 - halo, by0)),
            (min(cx1 + halo, bx1), min(cy1 + halo, by1)))


def _query_split(nv, core, halo, bounds, out):
    """Query halo(core); split core recursively when it exceeds the cap."""
    (cx0, cy0), (cx1, cy1) = core
    (hx0, hy0), (hx1, hy1) = _expand_box(core, halo, bounds)
    res = nv.query_points((hx0, hy0, _Z_FULL[0]), (hx1, hy1, _Z_FULL[1]),
                          max_points=TILE_QUERY_CAP)
    if res is not None:
        out.append((res, core))
        return True
    if "too many" not in str(getattr(nv, "last_error", "") or "").lower():
        return False
    if (cx1 - cx0) <= 0 or (cy1 - cy0) <= 0:
        return False
    mx, my = 0.5 * (cx0 + cx1), 0.5 * (cy0 + cy1)
    for sub in (((cx0, cy0), (mx, my)), ((mx, cy0), (cx1, my)),
                ((cx0, my), (mx, cy1)), ((mx, my), (cx1, cy1))):
        if not _query_split(nv, sub, halo, bounds, out):
            return False
    return True


# --------------------------------------------------------------------------
# tiled execution
# --------------------------------------------------------------------------

class _Progress:
    """Maps per-tile + in-core progress onto one modal dialog; GUI thread."""

    def __init__(self, title, total_tiles, parent):
        self.total = max(1, int(total_tiles))
        self.done_spans = 0.0
        self.span = 1.0 / self.total
        self.dlg = QProgressDialog(title, "Cancel", 0, 100, parent)
        try:
            self.dlg.setWindowModality(2)  # Qt.WindowModal
        except Exception:
            pass
        try:
            self.dlg.setMinimumDuration(0)
            self.dlg.setValue(0)
            self.dlg.show()
        except Exception:
            pass

    def start_tile(self, index):
        self.done_spans = index * self.span
        self._set(0.0, "")

    def core_progress(self, pct, msg=""):
        try:
            self._set(float(pct), msg)
        except Exception:
            pass

    def _set(self, pct, msg):
        value = self.done_spans + self.span * min(max(pct, 0.0), 100.0) / 100.0
        self.dlg.setValue(int(value * 100))
        if msg:
            try:
                self.dlg.setLabelText(str(msg)[:160])
            except Exception:
                pass
        QApplication.processEvents()

    def cancelled(self):
        try:
            return bool(self.dlg.wasCanceled())
        except Exception:
            return False

    def close(self):
        try:
            self.dlg.close()
        except Exception:
            pass


def run_tiled(app, view, name, halo, op, scope=None, parent=None):
    """Run ``op`` over the cloud (or fence scope) tile by tile.

    ``op(tile, eligible, report)`` receives the native query dict of one
    tile+halo box, a boolean eligible mask over its rows (core box only),
    and ``report`` with ``.core_progress(pct, msg)`` / ``.cancelled()``.
    It mutates ``tile["classification"]`` in place and returns a result
    dict (at least ``{"changed": int}``). Non-eligible rows are restored
    afterwards (same postcondition as the VTK path).

    Returns ``{"changed": total, "cancelled": bool, "tiles": n}`` and pushes
    ONE file-identity undo step when anything was written.
    """
    nv = view.native
    x0, y0, x1, y1, _total = cloud_bbox(view)
    bounds = ((x0, y0), (x1, y1))
    if scope is not None:
        (sx0, sy0), (sx1, sy1) = scope
        x0, y0 = max(x0, sx0), max(y0, sy0)
        x1, y1 = min(x1, sx1), min(y1, sy1)
        if x1 <= x0 or y1 <= y0:
            return {"changed": 0, "cancelled": False, "tiles": 0}
    cores = _tile_grid(x0, y0, x1, y1, _total, TILE_TARGET_POINTS)

    prog = _Progress(f"{name} (native)…", len(cores), parent if parent is not None else app)
    undo_node, undo_local, undo_old, undo_new = [], [], [], []
    changed_total = 0
    cancelled = False
    tiles_done = 0
    try:
        for index, core in enumerate(cores):
            if prog.cancelled():
                cancelled = True
                break
            if getattr(view, "native", None) is None:
                cancelled = True
                break
            prog.start_tile(index)
            got = []
            if not _query_split(nv, core, halo, bounds, got):
                try:
                    app.statusBar().showMessage(f"{name}: query failed: {nv.last_error}", 5000)
                except Exception:
                    pass
                print(f"[NATIVE-BY-CLASS] {name}: query failed: {nv.last_error}")
                cancelled = True
                break
            for res, res_core in got:
                if prog.cancelled():
                    cancelled = True
                    break
                n = int(res["xyz"].shape[0])
                if n == 0:
                    continue
                (rx0, ry0), (rx1, ry1) = res_core
                eligible = ((res["xyz"][:, 0] >= rx0) & (res["xyz"][:, 0] <= rx1)
                            & (res["xyz"][:, 1] >= ry0) & (res["xyz"][:, 1] <= ry1))
                if not np.any(eligible):
                    continue
                before = res["classification"].astype(np.uint8, copy=True)
                result = op(res, eligible, prog) or {}
                # Postcondition (mirrors the VTK path): nothing outside the
                # eligible scope may change, even if a core forgets its fence.
                fixed = np.asarray(res["classification"])
                if fixed.shape != before.shape:
                    continue
                fixed[~eligible] = before[~eligible]
                changed = np.flatnonzero(fixed != before)
                if changed.size:
                    ok = nv.set_classes(np.ascontiguousarray(res["node"][changed]),
                                        np.ascontiguousarray(res["local"][changed]),
                                        np.ascontiguousarray(fixed[changed]))
                    if not ok:
                        print(f"[NATIVE-BY-CLASS] {name}: write failed: {nv.last_error}")
                        cancelled = True
                        break
                    undo_node.append(np.asarray(res["node"][changed], dtype=np.uint32))
                    undo_local.append(np.asarray(res["local"][changed], dtype=np.uint32))
                    undo_old.append(np.asarray(before[changed], dtype=np.uint8))
                    undo_new.append(np.asarray(fixed[changed], dtype=np.uint8))
                    changed_total += int(changed.size)
                tiles_done += 1
                try:
                    nv.frame(0.0)
                except Exception:
                    pass
                QApplication.processEvents()
            if cancelled:
                break
    finally:
        prog.close()

    if undo_node:
        node = np.concatenate(undo_node)
        local = np.concatenate(undo_local)
        app.undo_stack.append({
            "node": node,
            "local": local,
            "old_classes": np.concatenate(undo_old),
            "new_classes": np.concatenate(undo_new),
        })
        for stack_name in ("redo_stack", "redostack"):
            stack = getattr(app, stack_name, None)
            if isinstance(stack, list):
                stack.clear()
        trim_undo_stack(app)
    try:
        verb = "cancelled after" if cancelled else "done:"
        app.statusBar().showMessage(
            f"{name} (native): {changed_total:,} points changed, {verb} {tiles_done} tiles", 6000)
    except Exception:
        pass
    print(f"[NATIVE-BY-CLASS] {name}: changed={changed_total:,} "
          f"tiles={tiles_done} cancelled={int(bool(cancelled))}")
    return {"changed": changed_total, "cancelled": bool(cancelled), "tiles": tiles_done}


# --------------------------------------------------------------------------
# per-tool halo rules (mirror each core's neighborhood range)
# --------------------------------------------------------------------------

def halo_for_core(func_name, kwargs):
    """Context margin so a tile+halo contains every neighbor a core can read."""
    get = dict(kwargs or {}).get
    if func_name == "classify_low_points":
        return max(0.0, float(get("within", 5.0)))
    if func_name == "classify_isolated_points":
        return max(0.0, float(get("within", 5.0)))
    if func_name == "classify_ground_ptd":
        return max(float(get("max_building_size", 60.0)),
                   float(get("iteration_distance", 1.40)) * 4.0,
                   float(get("edge_length_threshold", 5.0)) * 2.0
                   if get("reduce_angle_edge", True) else 0.0,
                   5.0)
    if func_name in ("classify_surface_points", "classify_below_surface"):
        return KNN_HALO  # unbounded kNN planes: generous heuristic, seams approximate
    if func_name == "classify_find_wires":
        return max(0.0, float(get("corridor_width", 3.0)))
    if func_name == "classify_poles_by_centerline":
        return max(0.0, float(get("radius", 1.0)))
    if func_name == "run_macro":
        return macro_halo(kwargs.get("steps", []))
    return 0.0


def macro_halo(steps):
    """Max tile halo over macro steps (same per-op rules as above)."""
    halo = 0.0
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        op = step.get("operation")
        params = step.get("params", {}) or {}
        if op == "Close":
            halo = max(halo, float(params.get("radius", 1.0)))
        elif op == "Height":
            halo = max(halo, 2.0 * float(params.get("max_triangle", 100.0)),
                       HEIGHT_HALO_MIN)
        elif op in ("Low Points", "Isolated"):
            halo = max(halo, float(params.get("within", 5.0)))
        elif op == "Ground":
            halo = max(halo, float(params.get("max_building_size", 60.0)),
                       float(params.get("iteration_distance", 1.4)) * 4.0, 5.0)
        elif op in ("Surface Points", "Below Surface"):
            halo = max(halo, KNN_HALO)
        elif op == "Find Wires":
            halo = max(halo, float(params.get("corridor_width", 3.0)))
        elif op == "Classify by Centerline":
            halo = max(halo, float(params.get("radius", 1.0)))
    return max(0.0, halo)


# --------------------------------------------------------------------------
# eligibility assembly shared by the menu_sidebar native ops
# --------------------------------------------------------------------------

def tile_eligible(app, tile, eligible, fence_polys=None):
    """Intersect core eligibility with flight-line + palette (+fence) rules."""
    eligible = np.asarray(eligible, dtype=bool)
    n = int(tile["xyz"].shape[0])
    if eligible.shape[0] != n:
        eligible = np.zeros(n, dtype=bool)
    try:
        eligible &= np.asarray(
            flight_line_visibility_mask(app, length=n, slot=0), dtype=bool)
    except Exception:
        pass
    try:
        eligible &= visible_class_mask(app, np.asarray(tile["classification"]))
    except Exception:
        pass
    if fence_polys:
        try:
            eligible &= inside_any_polygon(tile["xyz"][:, :2], fence_polys)
        except Exception:
            pass
    return eligible
