"""AI classification on an open .lnt - native write-back, no VTK.

The AI pipeline (gui/ai_dialog.py) works on a full in-memory dataset in SOURCE
point order (app.data: xyz, intensity, classification, return fields). A
streamed .lnt holds no such array, but every .lnt record maps to exactly one
source point: the builder (native/lnt/include/LNTBuilderImpl.h) quantizes each
source point to 21 bits per axis inside the header's cube bounds, Morton-encodes
it, stable-radix-sorts by that key and writes the leaf records as
pts[ids[begin..end)] in directory order. Recomputing the same keys and a stable
sort here gives that permutation exactly (verified on a 14.9M-point tile:
positions within half a leaf step, classes byte-identical).

So an AI run on an .lnt:
  1. builds the dataset from the source LAS/LAZ (real intensity and return
     fields) with the CURRENT classes read from the .lnt (user edits included),
  2. lets the unchanged AI dialog run on it as app.data,
  3. writes every class that differs from what the .lnt holds through
     NativeView.set_classes (the same path manual classification uses: file,
     LOD copies, the view, class counts), keeps the section store in step, and
  4. hands app.data back once no AI dialog is open any more.
"""
import os
import struct

import numpy as np
from PySide6.QtCore import QObject, QTimer

from gui.lnt_class_counts import read_directory

_AI_DIALOGS = ("_ai_progress_dialog", "_ai_fence_dialog", "_ai_mapping_dialog")
_POINT_BYTES, _CLASS_BYTE = 11, 10


def active_bridge(app):
    bridge = getattr(app, "_lnt_ai_bridge", None)
    return bridge if bridge is not None and bridge.active else None


def _expand_bits_21(v):
    v = v.astype(np.uint64) & np.uint64(0x1FFFFF)
    v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    v = (v | (v << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    v = (v | (v << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v << np.uint64(2))) & np.uint64(0x1249249249249249)
    return v


def _quantize_21(val, lo, hi):
    """LNTMath.h quantize_to_21bit, vectorized (same IEEE operations)."""
    if not hi > lo:
        return np.zeros(len(val), dtype=np.uint32)
    u = (val - lo) / (hi - lo)
    q = np.minimum(u * 2097152.0, 2097151.0)
    q = np.where(u >= 1.0, 2097151.0, q)
    return np.where(u > 0.0, q, 0.0).astype(np.uint32)


_SPREAD_7 = None


def morton_keys(xyz, bounds_min, bounds_max):
    """The builder's 63-bit Morton key of every point (same bits as
    _expand_bits_21 per axis, built from 7-bit lookup tables: fewer passes)."""
    global _SPREAD_7
    if _SPREAD_7 is None:
        _SPREAD_7 = _expand_bits_21(np.arange(128, dtype=np.uint32))
    keys = np.zeros(len(xyz), dtype=np.uint64)
    for axis in range(3):
        q = _quantize_21(xyz[:, axis], bounds_min[axis], bounds_max[axis])
        for part in range(3):
            keys |= _SPREAD_7[(q >> np.uint32(7 * part)) & np.uint32(127)] << np.uint64(axis + 21 * part)
    return keys


def source_order(xyz, bounds_min, bounds_max):
    """Source index of every .lnt leaf record, in record order (the builder's ids)."""
    return np.argsort(morton_keys(xyz, bounds_min, bounds_max), kind="stable")


def _lnt_bounds(lnt):
    with open(lnt, "rb") as f:
        header = f.read(64)
    return (np.array(struct.unpack_from("<3d", header, 16)),
            np.array(struct.unpack_from("<3d", header, 40)))


def lnt_classes_at(lnt, xyz, indices, entries, samples=256):
    """Classes the .lnt holds now (user edits included) for source points ``indices``.

    Same mapping as read_lnt_records - record k is the k-th point in stable
    Morton-key order - without ordering the whole cloud: a point's record is
    the number of smaller keys plus its rank among equal keys (ties keep source
    order).  So only the keys are sorted (np.sort, no stable argsort) and only
    the requested points are ranked.  Raises when the .lnt was not built from
    ``xyz``.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return np.empty(0, dtype=np.uint8)
    bmin, bmax = _lnt_bounds(lnt)
    keys = morton_keys(xyz, bmin, bmax)
    sorted_keys = np.sort(keys)
    query = keys[indices]
    by_key = np.argsort(query)                    # sorted queries: cache-friendly search
    q_sorted = query[by_key]
    first = np.searchsorted(sorted_keys, q_sorted, side="left")
    last = np.searchsorted(sorted_keys, q_sorted, side="right")
    del sorted_keys
    rank = np.empty(indices.size, dtype=np.int64)
    rank[by_key] = first
    tied = np.zeros(indices.size, dtype=bool)
    tied[by_key] = (last - first) > 1
    if tied.any():
        # Equal keys keep source order: add how many equal-key points come first.
        tie_keys = np.unique(query[tied])
        pos = np.minimum(np.searchsorted(tie_keys, keys), tie_keys.size - 1)
        members = np.flatnonzero(tie_keys[pos] == keys)        # ascending source index
        member_keys = keys[members]
        grouped = np.argsort(member_keys, kind="stable")
        group_start = np.searchsorted(member_keys[grouped], member_keys[grouped], side="left")
        within = np.empty(members.size, dtype=np.int64)
        within[grouped] = np.arange(members.size) - group_start
        rank[tied] += within[np.searchsorted(members, indices[tied])]
    del keys

    leaves = np.flatnonzero(entries["children_mask"] == 0)
    counts = entries["point_count"][leaves].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.searchsorted(starts, rank, side="right") - 1    # last leaf starting at/below
    leaf = leaves[slot]
    offset = entries["byte_offset"][leaf].astype(np.int64) + (rank - starts[slot]) * _POINT_BYTES
    mm = np.memmap(lnt, dtype=np.uint8, mode="r")
    try:
        classes = np.asarray(mm[offset + _CLASS_BYTE])
        # Decode a spread of the looked-up records: they must be these source
        # points, or the .lnt is stale/foreign and its classes must not be used.
        picks = np.unique(np.linspace(0, indices.size - 1, min(samples, indices.size)).astype(np.int64))
        rows = np.asarray(mm[offset[picks, None] + np.arange(6)]).copy().view("<u2").astype(np.float64)
        e = entries[leaf[picks]]
        half = e["half_size"].astype(np.float64)[:, None]
        world = (bmin + e["center"].astype(np.float64) - half) + rows * (2 * half) / 65535.0
        err = np.abs(world - xyz[indices[picks]])
        if (err > 0.51 * (2 * half / 65535.0) + 1e-9).any():
            raise RuntimeError("The open .lnt no longer matches its source point cloud "
                               "(it was built from a different file). Reopen the source "
                               "file to rebuild the .lnt, then try again.")
    finally:
        del mm
    return classes


def read_lnt_records(lnt, xyz, total, entries, progress=lambda pct, text: None):
    """Map every .lnt record to its source point.

    Returns (order, rec_node, rec_local, rec_cls): record k is source point
    order[k], stored at (rec_node[k], rec_local[k]) with class rec_cls[k] - the
    .lnt's current class, user edits included.  Raises when the .lnt was not
    built from ``xyz``.
    """
    progress(35, "Matching the .lnt records to the source points…")
    bmin, bmax = _lnt_bounds(lnt)
    order = source_order(xyz, bmin, bmax)          # record k <- source order[k]

    leaves = np.flatnonzero(entries["children_mask"] == 0)
    counts = entries["point_count"][leaves].astype(np.int64)
    rec_node = np.repeat(leaves.astype(np.uint32), counts)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rec_local = (np.arange(total, dtype=np.int64) - np.repeat(starts, counts)).astype(np.uint32)

    progress(60, "Reading the current classes from the .lnt…")
    rec_cls = np.empty(total, dtype=np.uint8)
    mm = np.memmap(lnt, dtype=np.uint8, mode="r")
    try:
        for leaf, start, count in zip(leaves.tolist(), starts.tolist(), counts.tolist()):
            if count:
                off = int(entries["byte_offset"][leaf]) + _CLASS_BYTE
                rec_cls[start:start + count] = mm[off:off + _POINT_BYTES * count:_POINT_BYTES]
        _verify_records(mm, entries, leaves, starts, counts, xyz, order, bmin)
    finally:
        del mm
    return order, rec_node, rec_local, rec_cls


def _verify_records(mm, entries, leaves, starts, counts, xyz, order, bmin, samples=64):
    """Decode a spread of leaves and check they are the source points the
    permutation claims - a stale or foreign .lnt must never be written to."""
    picks = np.unique(np.linspace(0, len(leaves) - 1, min(samples, len(leaves))).astype(int))
    for i in picks.tolist():
        leaf, start, count = int(leaves[i]), int(starts[i]), int(counts[i])
        if not count:
            continue
        e = entries[leaf]
        off = int(e["byte_offset"])
        raw = np.asarray(mm[off:off + _POINT_BYTES * count]).reshape(count, _POINT_BYTES)
        rel = raw[:, :6].copy().view("<u2").astype(np.float64)
        half = float(e["half_size"])
        world = (bmin + e["center"].astype(np.float64) - half) + rel * (2 * half) / 65535.0
        step = 2 * half / 65535.0
        err = np.abs(world - xyz[order[start:start + count]]).max()
        if err > 0.51 * step + 1e-9:
            raise RuntimeError("The open .lnt no longer matches its source point cloud "
                               "(it was built from a different file). Reopen the source "
                               "file to rebuild the .lnt, then try again.")


class LntAiBridge(QObject):
    def __init__(self, app, view, parent=None):
        super().__init__(parent or app)
        self.app, self.view = app, view
        self.active = False
        self._saved_data = None
        self._watch = QTimer(self)
        self._watch.setInterval(500)
        self._watch.timeout.connect(self.release_if_idle)

    # ---- dataset ------------------------------------------------------------
    def prepare(self, source, progress=lambda pct, text: None):
        """Build the AI dataset for the open .lnt; raises with a user-facing reason."""
        import laspy
        lnt = self.view.lnt_path
        store = getattr(self.app, "_lnt_section_store", None)
        if store is not None and getattr(store, "view", None) is self.view:
            store.flush()                          # the .lnt must hold every shown edit
        progress(5, "Reading the source point cloud…")
        las = laspy.read(source)
        n = int(len(las.points))
        total, entries = read_directory(lnt)
        if n != total:
            raise RuntimeError(f"{os.path.basename(source)} has {n:,} points but the open "
                               f".lnt has {total:,}; reopen the file to rebuild its .lnt.")
        xyz = np.column_stack([np.asarray(las.x, dtype=np.float64),
                               np.asarray(las.y, dtype=np.float64),
                               np.asarray(las.z, dtype=np.float64)])
        dims = {str(d) for d in las.point_format.dimension_names}
        intensity = (np.asarray(las.intensity, dtype=np.float32) if "intensity" in dims else None)
        rn = (np.asarray(las.return_number, dtype=np.uint8) if "return_number" in dims else None)
        nr = (np.asarray(las.number_of_returns, dtype=np.uint8) if "number_of_returns" in dims else None)
        del las

        order, rec_node, rec_local, rec_cls = read_lnt_records(lnt, xyz, total, entries, progress)

        self.node = np.empty(total, dtype=np.uint32)
        self.local = np.empty(total, dtype=np.uint32)
        self.node[order] = rec_node                     # source index -> .lnt identity
        self.local[order] = rec_local
        classification = np.empty(total, dtype=np.uint8)
        classification[order] = rec_cls
        self.on_disk = classification.copy()            # what the .lnt holds right now
        progress(95, "Starting AI classification…")
        return {
            "xyz": xyz, "intensity": intensity, "classification": classification,
            "return_number": rn, "number_of_returns": nr,
            "rgb": None, "_source_file_path": str(source), "loaded_file": str(source),
            "_lnt_ai_dataset": True,
        }

    # ---- session --------------------------------------------------------------
    def begin(self, dataset):
        self._saved_data = self.app.data
        self.app.data = dataset
        self.dataset = dataset
        self.active = True
        self._watch.start()
        print(f"[LNT-AI] AI dataset ready: {len(dataset['xyz']):,} source-order points "
              f"mapped onto {os.path.basename(self.view.lnt_path)}")

    def release_if_idle(self):
        """Hand app.data back once every AI dialog has closed (any exit path)."""
        if not self.active:
            return
        from PySide6.QtWidgets import QApplication
        busy = QApplication.activeModalWidget() is not None   # a prompt between AI steps
        for name in _AI_DIALOGS:
            dialog = getattr(self.app, name, None)
            try:
                if dialog is not None and (dialog.isVisible() or dialog.isMinimized()):
                    busy = True
            except RuntimeError:
                pass
        if busy:
            self._idle_ticks = 0
            return
        # Idle twice in a row: the hand-over from one AI window to the next is
        # synchronous, so a real gap never spans two ticks.
        self._idle_ticks = getattr(self, "_idle_ticks", 0) + 1
        if self._idle_ticks >= 2 or not self._watch.isActive():
            self.end()

    def end(self):
        if not self.active:
            return
        try:
            self.commit()                       # nothing may be left unwritten
        finally:
            self.active = False
            self._watch.stop()
            if self.app.data is self.dataset:
                still_open = getattr(self.app, "_lnt_main_view", None) is self.view and \
                    getattr(self.view, "native", None) is not None
                self.app.data = self._saved_data if still_open else None
            self.dataset = self._saved_data = None
            self.node = self.local = self.on_disk = None
            print("[LNT-AI] AI session closed; the .lnt view owns the data again")

    # ---- write-back -------------------------------------------------------------
    def commit(self):
        """Write every class the AI dataset holds that the .lnt does not. Returns the count."""
        if not self.active or self.view.native is None:
            return 0
        cur = np.asarray(self.dataset.get("classification"), dtype=np.uint8)
        if cur.shape != self.on_disk.shape:
            print("[LNT-AI] classification length changed; nothing written")
            return 0
        rows = np.flatnonzero(cur != self.on_disk)
        if rows.size == 0:
            return 0
        node, local, new = self.node[rows], self.local[rows], cur[rows]
        if not self.view.native.set_classes(node, local, new):
            raise RuntimeError(f"Could not write AI classes into the .lnt: "
                               f"{self.view.native.last_error}")
        self.on_disk[rows] = new
        print(f"[LNT-AI] wrote {rows.size:,} AI class changes into the .lnt")
        self._sync_views(node, local, new)
        return int(rows.size)

    def identity(self, mask):
        """(node, local) of the source points in `mask` - for undo entries."""
        return self.node[mask].copy(), self.local[mask].copy()

    def _sync_views(self, node, local, new):
        sync_store_rows(self.app, self.view, node, local, new)
        try:
            self.view.native.frame(0.0)
        except Exception:                        # noqa: BLE001
            pass


def sync_store_rows(app, view, node, local, new):
    """Keep section-store rows (loaded section points) equal to what was just written,
    so a later flush cannot write their stale classes back, then recolour sections."""
    store = getattr(app, "_lnt_section_store", None)
    if store is None or getattr(store, "view", None) is not view or not len(store.node):
        return
    key_store = store.node.astype(np.uint64) << np.uint64(32) | store.local.astype(np.uint64)
    key_new = np.asarray(node, dtype=np.uint64) << np.uint64(32) | np.asarray(local, dtype=np.uint64)
    order = np.argsort(key_store, kind="stable")
    pos = np.searchsorted(key_store[order], key_new)
    pos = np.minimum(pos, len(order) - 1)
    hit = key_store[order][pos] == key_new
    if not hit.any():
        return
    rows = order[pos[hit]]
    values = np.asarray(new, dtype=np.uint8)[hit]
    store.data["classification"][rows] = values
    store._synced[rows] = values
    store.generation += 1
    sections = getattr(app, "_native_sections", None)
    if sections is not None:
        sections.refresh_edits(force=True)


# ---- fences and highlights (AI > Fence tab on an .lnt) -----------------------------
def native_fence_drawings(app):
    """The Draw tab's native drawings, in the shape AIFenceRunDialog validates:
    two-click rectangles become their 4 corners, circles centre + radius."""
    view = getattr(app, "_lnt_main_view", None)
    drawing = getattr(view, "drawing", None)
    out = []
    for d in list(getattr(drawing, "drawings", None) or []):
        if not isinstance(d, dict):
            continue
        kind = str(d.get("type", "")).lower()
        pts = [(float(p[0]), float(p[1])) for p in d.get("coords", []) if len(p) >= 2]
        fence = dict(d, source="digitizer")
        if kind == "rectangle" and len(pts) >= 2:
            (x0, y0), (x1, y1) = pts[0], pts[1]
            fence["coords"] = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        elif kind == "circle" and len(pts) >= 2:
            (cx, cy), (ex, ey) = pts[0], pts[1]
            fence["center"] = (cx, cy, 0.0)
            fence["radius"] = float(np.hypot(ex - cx, ey - cy))
        else:
            fence["coords"] = pts
        out.append(fence)
    return out


def show_native_highlights(app, shapes):
    """Paint fence highlights (hover / selected) on the native main view overlay."""
    view = getattr(app, "_lnt_main_view", None)
    if view is None or getattr(view, "native", None) is None:
        return
    overlay = getattr(view, "_ai_fence_overlay", None)
    if overlay is None:
        if not shapes:
            return
        from gui.native_main_classify import MainCanvas
        from gui.native_section_classify import CanvasOverlay
        overlay = view._ai_fence_overlay = CanvasOverlay(MainCanvas(view))
    overlay.set_shapes([("drawing", s["coords"], {"type": "polyline", "color": s["color"],
                                                  "width": s["width"]})
                        for s in shapes if s and len(s.get("coords", ())) >= 2])


# ---- entry point --------------------------------------------------------------------
def start_lnt_ai(app):
    """AI > Start with an .lnt open. Returns False when no .lnt is open."""
    view = getattr(app, "_lnt_main_view", None)
    if view is None or getattr(view, "native", None) is None:
        return False
    from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog
    from gui.ai_dialog import show_ai_classification_dialog
    bridge = active_bridge(app)
    if bridge is not None:                     # an AI window is already open: focus it
        show_ai_classification_dialog(app)
        return True
    source = getattr(app, "_lnt_source_path", None)
    if not source or not os.path.isfile(source) or \
            os.path.splitext(source)[1].lower() not in (".las", ".laz"):
        QMessageBox.warning(app, "AI Classification",
                            "AI needs the source LAS/LAZ of the open .lnt, which was not found:\n"
                            f"{source or '(unknown)'}")
        return True
    dlg = QProgressDialog("Preparing AI classification…", None, 0, 100, app)
    dlg.setWindowTitle("AI Classification")
    dlg.setMinimumDuration(0)
    dlg.setValue(0)

    def progress(pct, text):
        dlg.setLabelText(text)
        dlg.setValue(int(pct))
        QApplication.processEvents()

    bridge = LntAiBridge(app, view)
    try:
        dataset = bridge.prepare(source, progress)
    except Exception as exc:                   # noqa: BLE001 - shown to the user
        dlg.close()
        print(f"[LNT-AI] cannot prepare AI dataset: {exc}")
        QMessageBox.warning(app, "AI Classification", str(exc))
        return True
    dlg.close()
    app._lnt_ai_bridge = bridge
    bridge.begin(dataset)
    try:
        show_ai_classification_dialog(app)
    finally:
        bridge.release_if_idle()               # declined/early exit: hand app.data back now
    return True
