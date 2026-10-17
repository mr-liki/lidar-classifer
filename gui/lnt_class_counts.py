"""Exact per-class point counts of an open .lnt - read from the file, no VTK.

The .lnt keeps every source record once, in a leaf node (children_mask == 0);
inner nodes only hold LOD copies. Each record is an 11-byte LNTPoint whose
class code is byte 10 (native/lnt/include/LNTFormat.h). So the exact class
histogram is a bincount over the class byte of every leaf record: one pass at
open time (background thread, ~0.4 s per 15M points), after which every class
write is applied as a delta. NativeView.set_classes - the one path all writers
(section flush, undo revert, By Class) go through - reports each write to
before_write()/after_write(); the old class bytes are read from the file just
before the native write overwrites them.
"""
import struct
import threading

import numpy as np
from PySide6.QtCore import QObject, Signal

_HEADER = struct.Struct("<4sHHQ")            # magic, version, reserved, index_offset
_DIR_HEADER = struct.Struct("<QQ")           # total_points, total_nodes
_ENTRY = np.dtype([
    ("node_id", "<u8"), ("byte_offset", "<u8"), ("byte_length", "<u4"),
    ("point_count", "<u4"), ("center", "<f4", 3), ("half_size", "<f4"),
    ("children_mask", "u1"), ("tree_level", "u1"), ("reserved", "u1", 10)])
_POINT_BYTES = 11
_CLASS_BYTE = 10


def read_directory(path):
    """(total_points, node entries) of an .lnt file."""
    with open(path, "rb") as f:
        magic, _version, _reserved, index_offset = _HEADER.unpack(f.read(_HEADER.size))
        if magic != b"LNT\0":
            raise ValueError(f"{path} is not an .lnt file")
        f.seek(index_offset)
        total, nodes = _DIR_HEADER.unpack(f.read(_DIR_HEADER.size))
        entries = np.frombuffer(f.read(_ENTRY.itemsize * nodes), dtype=_ENTRY)
    if len(entries) != nodes:
        raise ValueError(f"{path}: truncated node directory")
    return int(total), entries


def scan_class_histogram(path, entries=None, cancelled=lambda: False):
    """int64[256] point count per class code over every leaf record, or None if cancelled."""
    if entries is None:
        _total, entries = read_directory(path)
    hist = np.zeros(256, dtype=np.int64)
    leaves = entries[entries["children_mask"] == 0]
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    try:
        for offset, count in zip(leaves["byte_offset"].tolist(), leaves["point_count"].tolist()):
            if cancelled():
                return None
            if count:
                start = offset + _CLASS_BYTE
                hist += np.bincount(mm[start:start + _POINT_BYTES * count:_POINT_BYTES],
                                    minlength=256)
    finally:
        del mm
    return hist


class LntClassCounts(QObject):
    """Live class histogram of one open .lnt. `ready` fires when counts change."""

    ready = Signal()

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path
        self.total_points, self._entries = read_directory(path)
        self._hist = None            # file histogram, exact once the scan finished
        self._lock = threading.Lock()
        self._scanning = False
        self._rescan = False         # a write landed during a scan: its effect is unknown
        self._closed = False
        self.start()

    def start(self):
        with self._lock:
            if self._scanning:
                self._rescan = True
                return
            self._scanning, self._rescan = True, False
        threading.Thread(target=self._scan, name="lnt-class-counts", daemon=True).start()

    def _scan(self):
        try:
            hist = scan_class_histogram(self.path, self._entries, lambda: self._closed)
        except Exception as exc:     # noqa: BLE001 - counts are informational
            print(f"[LNT] class count scan failed: {exc}")
            hist = None
        with self._lock:
            self._scanning = False
            again = self._rescan and not self._closed
            if not again and hist is not None:
                self._hist = hist
                print(f"[LNT] class counts: {int(hist.sum()):,} records in "
                      f"{int(np.count_nonzero(hist))} classes")
        if again:
            self.start()
        elif hist is not None and not self._closed:
            self.ready.emit()        # queued to the GUI thread (receiver lives there)

    def close(self):
        self._closed = True

    # ---- write tracking (called by NativeView.set_classes) ----------------------
    def before_write(self, node, local, classification):
        """Old class bytes of the records about to be written (last write per record
        wins, as in the native writer), or None when the delta cannot be computed."""
        with self._lock:
            if self._scanning or self._hist is None:
                self._rescan = self._scanning
                return None
        try:
            node = np.asarray(node, dtype=np.int64)
            local = np.asarray(local, dtype=np.int64)
            new = np.asarray(classification, dtype=np.uint8)
            key = node * (1 << 32) + local
            _, last = np.unique(key[::-1], return_index=True)
            pick = len(key) - 1 - last
            node, local, new = node[pick], local[pick], new[pick]
            pos = (self._entries["byte_offset"][node].astype(np.int64)
                   + local * _POINT_BYTES + _CLASS_BYTE)
            mm = np.memmap(self.path, dtype=np.uint8, mode="r")
            try:
                old = np.array(mm[pos])
            finally:
                del mm
            return old, new
        except Exception as exc:     # noqa: BLE001
            print(f"[LNT] class count delta unavailable ({exc}); recounting")
            return None

    def after_write(self, token, ok):
        """Apply a finished write; anything inexact falls back to a recount."""
        if ok and token is not None:
            old, new = token
            delta = (np.bincount(new, minlength=256).astype(np.int64)
                     - np.bincount(old, minlength=256))
            with self._lock:
                if self._hist is not None and not self._scanning:
                    self._hist += delta
                    token = "applied"
            if token == "applied":
                if delta.any():
                    self.ready.emit()
                return
        self.start()

    # ---- reading ----------------------------------------------------------------
    def counts(self, store=None):
        """{class: count} including edits shown on screen but not yet written
        (section store rows whose class differs from the file), or None while counting."""
        with self._lock:
            if self._hist is None or self._scanning:
                return None
            hist = self._hist.copy()
        if store is not None:
            try:
                cur = np.asarray(store.data["classification"], dtype=np.uint8)
                synced = np.asarray(store._synced, dtype=np.uint8)
                if cur.shape == synced.shape:
                    dirty = cur != synced
                    if dirty.any():
                        hist += np.bincount(cur[dirty], minlength=256)
                        hist -= np.bincount(synced[dirty], minlength=256)
            except Exception:        # noqa: BLE001
                pass
        return {int(c): int(n) for c, n in enumerate(hist) if n > 0}
