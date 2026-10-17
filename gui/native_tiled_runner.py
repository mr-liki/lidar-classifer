"""Shared tiled runner for native (.lnt) classification / conversion tools.

While an .lnt is shown in the main viewport the application holds no
monolithic point arrays (``app.data`` is the small, accumulating
``LntSectionStore``).  Every algorithm written against
``app.data["xyz"]`` / ``["classification"]`` therefore runs through this
module on native:

* read the cloud envelope from the .lnt 64-byte header (bounds_min/max),
* walk it in XY tiles,
* per tile: ``store.ensure_region`` (native ``query_points``) -> run the
  caller's per-tile callback on the tile rows -> write the new classes,
* one file-identity undo entry (``indices`` / ``old_classes`` /
  ``new_classes``, the exact contract
  ``gui.native_shortcuts.native_classification_history`` requires), then
  ``store.show`` (instant recolour) + ``store.flush`` (file write-back).

All inputs degrade gracefully on native: a missing/foreign view, a missing
section API, a failed tile query, or a missing helper (flight-line filter,
palette, refresh signal) skips that piece instead of raising.
"""
from __future__ import annotations

import os
import struct

import numpy as np

# .lnt header: magic[4] version[2] reserved[2] index_offset[8]
#             bounds_min[3 x f8] bounds_max[3 x f8]  (offsets 16 / 40)
_LNT_HEADER_FMT = "<4sHHQ3d3d"
_LNT_HEADER_SIZE = struct.calcsize(_LNT_HEADER_FMT)

DEFAULT_TILE_SIZE = 500.0
_Z_FULL = 1.0e12


def is_native(app) -> bool:
    """True when an .lnt view with a section API is on screen."""
    try:
        from gui.lnt_section_bridge import active_view, get_store  # noqa: PLC0415
        view = active_view(app)
        if view is None or not view.native.has_section_api():
            return False
        return get_store(app) is not None
    except Exception:
        return False


def native_view(app):
    try:
        from gui.lnt_section_bridge import active_view  # noqa: PLC0415
        return active_view(app)
    except Exception:
        return None


def native_store(app):
    try:
        from gui.lnt_section_bridge import get_store  # noqa: PLC0415
        return get_store(app)
    except Exception:
        return None


def read_lnt_bounds(lnt_path):
    """(bounds_min, bounds_max) as float64 (3,) arrays, or (None, None)."""
    try:
        with open(lnt_path, "rb") as f:
            head = f.read(_LNT_HEADER_SIZE)
        if len(head) < _LNT_HEADER_SIZE or head[:3] != b"LNT":
            return None, None
        parts = struct.unpack(_LNT_HEADER_FMT, head)
        return (np.asarray(parts[4:7], dtype=np.float64),
                np.asarray(parts[7:10], dtype=np.float64))
    except Exception:
        return None, None


def cloud_bounds(app):
    """Envelope of the open .lnt, or (None, None) when unavailable."""
    view = native_view(app)
    path = getattr(view, "lnt_path", None) if view is not None else None
    if path and os.path.isfile(path):
        bmin, bmax = read_lnt_bounds(path)
        if bmin is not None:
            return bmin, bmax
    return None, None

def iter_tiles(bmin, bmax, tile_size=DEFAULT_TILE_SIZE):
    """Yield (mn, mx) world boxes covering the XY envelope."""
    ts = float(tile_size) if tile_size and float(tile_size) > 0 else DEFAULT_TILE_SIZE
    nx = max(1, int(np.ceil((bmax[0] - bmin[0]) / ts)))
    ny = max(1, int(np.ceil((bmax[1] - bmin[1]) / ts)))
    for ix in range(nx):
        x0 = bmin[0] + ix * ts
        x1 = bmax[0] if ix == nx - 1 else x0 + ts
        for iy in range(ny):
            y0 = bmin[1] + iy * ts
            y1 = bmax[1] if iy == ny - 1 else y0 + ts
            mn = (float(np.nextafter(x0, -np.inf)),
                  float(np.nextafter(y0, -np.inf)), -_Z_FULL)
            mx = (float(np.nextafter(x1, np.inf)),
                  float(np.nextafter(y1, np.inf)), _Z_FULL)
            yield mn, mx


def visible_classes(app):
    """Sorted visible class codes from the palette, or None (no restriction)."""
    try:
        palette = getattr(app, "class_palette", None)
        if isinstance(palette, dict) and palette:
            vis = [int(c) for c, i in palette.items()
                   if isinstance(i, dict) and i.get("show", True)]
            return vis or None
    except Exception:
        pass
    return None


def push_native_undo(app, indices, old_classes, new_classes):
    """Append one file-identity undo entry; returns True on success."""
    try:
        idx = np.asarray(indices, dtype=np.int64).ravel()
        old = np.asarray(old_classes, dtype=np.uint8).ravel()
        new = np.asarray(new_classes, dtype=np.uint8).ravel()
        if idx.size == 0:
            return False
        entry = {
            "indices": idx.copy(),
            "oldclasses": old.copy(), "old_classes": old.copy(),
            "newclasses": new.copy(), "new_classes": new.copy(),
        }
        stack = getattr(app, "undo_stack", getattr(app, "undostack", None))
        if stack is None:
            app.undo_stack = stack = []
        stack.append(entry)
        redo = getattr(app, "redo_stack", getattr(app, "redostack", None))
        if redo is not None:
            try:
                redo.clear()
            except Exception:
                pass
        try:
            from gui.memory_manager import trim_undo_stack  # noqa: PLC0415
            trim_undo_stack(app)
        except Exception:
            max_steps = getattr(app, "_max_undo_steps", 10)
            try:
                while len(stack) > max_steps:
                    stack.pop(0)
            except Exception:
                pass
        return True
    except Exception:
        return False

def _points_in_polygons(pts, poly):
    """Ray-casting point-in-polygon for one ring; pure numpy fallback."""
    try:
        x, y = pts[:, 0], pts[:, 1]
        inside = np.zeros(len(pts), dtype=bool)
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = float(poly[i, 0]), float(poly[i, 1])
            xj, yj = float(poly[j, 0]), float(poly[j, 1])
            cond = (((yi > y) != (yj > y))
                    & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi))
            inside ^= cond
            j = i
        return inside
    except Exception:
        return np.ones(len(pts), dtype=bool)


def _refresh_native(app, store, indices):
    try:
        rows = np.asarray(indices, dtype=np.int64)
        shown = np.asarray(store.data["classification"])[rows]
        store.show(rows, shown)
    except Exception:
        try:
            store.flush()
        except Exception:
            pass
    try:
        sections = getattr(app, "_native_sections", None)
        if sections is not None:
            sections.refresh_edits(force=True)
    except Exception:
        pass
    try:
        sig = getattr(app, "classification_finished", None)
        if sig is not None and hasattr(sig, "emit"):
            mask = np.zeros(int(store.data["classification"].shape[0]), dtype=bool)
            mask[np.asarray(indices, dtype=np.int64)] = True
            sig.emit(mask)
    except Exception:
        pass


def run_tiled(app, label, apply_tile, tile_size=DEFAULT_TILE_SIZE,
              progress=None, fence_xy=None):
    """Run ``apply_tile(tile)`` over every cloud tile; commit one undo step.

    ``apply_tile`` receives ``{"rows", "xyz", "cls", "store"}`` and returns
    ``(local_changed, new_values)`` -- positions into the tile arrays and
    their new classes -- or ``None`` when the tile is unaffected.
    """
    from gui.lnt_section_bridge import get_store  # noqa: PLC0415
    empty_result = {"changed": 0, "indices": np.empty(0, dtype=np.int64),
                    "tiles": 0, "empty_tiles": 0, "label": label}
    store = get_store(app)
    if store is None:
        empty_result["error"] = "no native store"
        return empty_result
    bmin, bmax = cloud_bounds(app)
    if bmin is None:
        empty_result["error"] = "no cloud bounds"
        return empty_result
    fence = None
    if fence_xy is not None:
        try:
            fence = np.asarray(fence_xy, dtype=np.float64).reshape(-1, 2)
            if fence.shape[0] < 3:
                fence = None
        except Exception:
            fence = None

    all_idx, all_old, all_new = [], [], []
    tiles = empty = 0
    for mn, mx in iter_tiles(bmin, bmax, tile_size):
        tiles += 1
        try:
            ok = store.ensure_region(mn, mx)
        except Exception:
            ok = False
        if not ok:
            continue
        rows = np.asarray(store.last_region_indices, dtype=np.int64).ravel()
        if rows.size == 0:
            empty += 1
            continue
        if fence is not None:
            try:
                pts = np.asarray(store.data["xyz"])[rows][:, :2]
                rows = rows[_points_in_polygons(pts, fence)]
            except Exception:
                pass
            if rows.size == 0:
                empty += 1
                continue
        tile = {"rows": rows,
                "xyz": np.asarray(store.data["xyz"])[rows],
                "cls": np.asarray(store.data["classification"])[rows],
                "store": store}
        try:
            out = apply_tile(tile)
        except Exception as exc:  # noqa: BLE001 - one bad tile never kills the run
            print(f"[NATIVE-TILED] {label}: tile {tiles} skipped ({exc})")
            continue
        if not out:
            continue
        local, values = out
        local = np.asarray(local).ravel()
        values = np.asarray(values, dtype=np.uint8).ravel()
        if local.size == 0 or values.size != local.size:
            continue
        cur = np.asarray(store.data["classification"])[rows[local]]
        keep = cur != values
        if not np.any(keep):
            continue
        local, values = local[keep], values[keep]
        hit_rows = rows[local]
        old = np.asarray(store.data["classification"])[hit_rows].copy()
        np.asarray(store.data["classification"])[hit_rows] = values
        all_idx.append(hit_rows)
        all_old.append(old)
        all_new.append(values.copy())
        if progress is not None:
            try:
                progress(tiles, int(sum(a.size for a in all_idx)))
            except Exception:
                pass
    if not all_idx:
        empty_result.update({"tiles": tiles, "empty_tiles": empty})
        return empty_result
    indices = np.concatenate(all_idx).astype(np.int64, copy=False)
    old_all = np.concatenate(all_old)
    new_all = np.concatenate(all_new)
    _, uniq = np.unique(indices, return_index=True)
    if uniq.size != indices.size:
        order = np.sort(uniq)
        indices, old_all, new_all = indices[order], old_all[order], new_all[order]
    push_native_undo(app, indices, old_all, new_all)
    try:
        app.classification_revision = int(getattr(app, "classification_revision", 0)) + 1
    except Exception:
        pass
    _refresh_native(app, store, indices)
    try:
        store.flush()
    except Exception:
        pass
    print(f"[NATIVE-TILED] {label}: {indices.size:,} pts over {tiles} tiles committed")
    return {"changed": int(indices.size), "indices": indices,
            "tiles": tiles, "empty_tiles": empty, "label": label}
