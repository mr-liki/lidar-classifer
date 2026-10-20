"""Fields - the point data table of the open native .lnt cloud (no VTK).

Native port of the reference implementation
(``E:\\naksha-lidar\\gui\\dialogs\\view_fields_table.py``): the same "Fields"
checklist, the same table behaviour (column selection, sorting, select by
class, CSV export, point highlight, zoom to point, delete menu), but every
access goes through the native Vulkan pipeline:

* the rows come from the .lnt FILE - the native session keeps no point arrays
  in memory (``app.data`` is empty by design), so the file is the only source
  of truth.  The .lnt keeps every source record exactly once, in a leaf node
  (inner nodes hold LOD copies), so reading the leaf records IS the cloud;
* the point highlight is drawn by the native overlay (gui.native_highlight)
  and the camera move is the engine's own ``frame_region``;
* a class edit is written straight into the .lnt through the engine
  (``NativeView.set_classes``) and is undoable by .lnt identity, the same
  history format the AI classification bridge uses.

The native format stores 11 bytes per record (relative XYZ, RGB, intensity,
class), so the table offers exactly those attributes - the .lnt does not keep
the LAS extras (time, line, echo, amplitude, ...).
"""
from __future__ import annotations

import csv
import os
import struct
from collections import OrderedDict

import numpy as np

from PySide6.QtCore import (QAbstractTableModel, QItemSelection, QItemSelectionModel,
                            QModelIndex, QObject, Qt, QThread, QTimer, Signal)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDialog,
                               QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel,
                               QLayout, QListWidget, QListWidgetItem, QMenu,
                               QMenuBar, QMessageBox, QPushButton, QTableView,
                               QVBoxLayout)

from gui.lnt_class_counts import read_directory

POINT = np.dtype([('rel', '<u2', 3), ('rgb', 'u1', 3), ('intensity', 'u1'),
                  ('classification', 'u1')])
CLASS_BYTE = 10          # offset of the class byte inside one LNTPoint

# Column kinds understood by the model.
K_CODE = "code"          # stored class slot -> user class number
K_DESC = "desc"          # stored class slot -> class name
K_FLOAT3 = "float3"      # XYZ, 3 decimals
K_INT0 = "int0"          # integer attribute, 0 decimals
K_RGB = "rgb"            # "R.. G.. B.."
K_HSV = "hsv"            # "H S V"

# Ordered column definitions (header, kind, source key).
COLUMN_DEFS = [
    ("Class", K_CODE, "classification"),
    ("Description", K_DESC, "classification"),
    ("Easting", K_FLOAT3, "x"),
    ("Northing", K_FLOAT3, "y"),
    ("Elevation", K_FLOAT3, "z"),
    ("Intensity", K_INT0, "intensity"),
    ("Color RGB", K_RGB, "rgb"),
    ("Color HSV", K_HSV, "rgb"),
]

# The attributes the native .lnt keeps, in the order of the reference list.
# The names are the reference's, so the shared "view_fields/checked" setting
# keeps working for both implementations.
NATIVE_FIELD_SPECS = [
    ("Class", ("classification",), True),
    ("Description", ("classification",), True),
    ("Easting", ("x",), True),
    ("Northing", ("y",), True),
    ("Elevation", ("z",), True),
    ("Intensity", ("intensity",), True),
    ("Color RGB", ("red", "green", "blue"), True),
]

# Native clouds are append-only: the reversible equivalent of a deletion is
# moving the points to the noise class (what the native Select tool does).
NOISE_CLASS = 7


def _rgb_to_hsv(rgb):
    r, g, b = (int(rgb[0]) / 255.0, int(rgb[1]) / 255.0, int(rgb[2]) / 255.0)
    mx, mn = max(r, g, b), min(r, g, b)
    diff = mx - mn
    if diff == 0:
        h = 0.0
    elif mx == r:
        h = (60.0 * ((g - b) / diff)) % 360.0
    elif mx == g:
        h = (60.0 * ((b - r) / diff)) + 120.0
    else:
        h = (60.0 * ((r - g) / diff)) + 240.0
    s = 0.0 if mx == 0 else diff / mx
    return "%d %d %d" % (round(h), round(s * 100), round(mx * 100))


def point_text(point, kind, key, codes, names):
    """Display text of one record for one column (table cells and CSV agree)."""
    if kind == K_CODE:
        return str(codes[int(point["classification"])])
    if kind == K_DESC:
        code = codes[int(point["classification"])]
        return names.get(code, "Class %s" % code)
    if kind == K_FLOAT3:
        axis = {"x": 0, "y": 1, "z": 2}[key]
        return "%.3f" % float(point["xyz"][axis])
    if kind == K_INT0:
        return "%d" % int(point["intensity"])
    if kind == K_RGB:
        return "R%d G%d B%d" % (int(point["rgb"][0]), int(point["rgb"][1]),
                                int(point["rgb"][2]))
    if kind == K_HSV:
        return _rgb_to_hsv(point["rgb"])
    return ""


class NativeFieldsReader:
    """Bounded page cache over the original leaf records of an .lnt.

    The .lnt keeps every source record once, in a leaf node; inner nodes only
    hold LOD copies.  Reading the leaves therefore yields the whole cloud, in
    the file's own order (the row index used by this table).
    """

    PAGE = 4096

    def __init__(self, path):
        self.path = str(path)
        with open(path, "rb") as stream:
            magic, version, _, _, *bounds = struct.unpack("<4sHHQ6d", stream.read(64))
        if magic != b"LNT\0" or version != 2:
            raise ValueError("Unsupported native point-cloud format")
        self.origin = np.asarray(bounds[:3], dtype=np.float64)
        self.bounds_min = self.origin
        self.bounds_max = np.asarray(bounds[3:6], dtype=np.float64)
        _, entries = read_directory(path)
        self.nodes = np.flatnonzero((entries["children_mask"] == 0)
                                    & (entries["point_count"] > 0))
        self.leaves = entries[self.nodes]
        self.ends = np.cumsum(self.leaves["point_count"], dtype=np.int64)
        self.count = int(self.ends[-1]) if len(self.ends) else 0
        self.starts = (np.concatenate(([0], self.ends[:-1])) if self.count
                       else np.empty(0, dtype=np.int64))
        self._bytes = np.memmap(path, dtype=np.uint8, mode="r")
        self._cache = OrderedDict()
        self._leaf_start = {int(node): int(self.starts[index])
                            for index, node in enumerate(self.nodes.tolist())}
        for leaf in self.leaves:
            if int(leaf["byte_offset"]) + int(leaf["point_count"]) * POINT.itemsize > len(self._bytes):
                self.close()
                raise ValueError("Truncated native point records")

    def close(self):
        self._cache.clear()
        data = getattr(self, "_bytes", None)
        if data is not None:
            mmap = getattr(data, "_mmap", None)
            if mmap is not None:
                try:
                    mmap.close()
                except Exception:
                    pass
            self._bytes = None

    def invalidate(self):
        self._cache.clear()

    def read(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        if np.any(rows < 0) or np.any(rows >= self.count):
            raise IndexError("Native point row is out of range")
        leaf_idx = np.searchsorted(self.ends, rows, side="right")
        local = rows - self.starts[leaf_idx]
        leaf = self.leaves[leaf_idx]
        offsets = leaf["byte_offset"].astype(np.int64) + local * POINT.itemsize
        raw = np.asarray(self._bytes[offsets[:, None] + np.arange(POINT.itemsize)]).copy()
        points = raw.reshape(-1).view(POINT)
        half = leaf["half_size"].astype(np.float64)[:, None]
        xyz = (self.origin + leaf["center"].astype(np.float64) - half
               + points["rel"] * (2 * half / 65535.))
        return dict(xyz=xyz, classification=points["classification"].copy(),
                    intensity=points["intensity"].copy(), rgb=points["rgb"].copy(),
                    node=self.nodes[leaf_idx].astype(np.uint32),
                    local=local.astype(np.uint32))

    def point(self, row):
        page = int(row) // self.PAGE
        if page not in self._cache:
            start = page * self.PAGE
            self._cache[page] = self.read(np.arange(start, min(start + self.PAGE, self.count)))
            if len(self._cache) > 4:
                self._cache.popitem(last=False)
        self._cache.move_to_end(page)
        data = self._cache[page]
        return {key: values[int(row) % self.PAGE] for key, values in data.items()}

    def row_for(self, node, local):
        """Table row of the record identified by (node, local), or None.

        The section store keeps the same (node, local) identity for the points
        it holds, so a point picked in the 3D view can be located in this
        table without guessing.
        """
        start = self._leaf_start.get(int(node))
        if start is None:
            return None
        row = start + int(local)
        return row if 0 <= row < self.count else None

    def rows_for_slots(self, slots):
        """Table rows whose stored class slot is one of `slots`.

        Only the class byte of every record is touched, so this is a fast scan
        of the file even for a cloud with millions of points.
        """
        wanted = np.zeros(256, dtype=bool)
        for slot in slots or ():
            slot = int(slot)
            if 0 <= slot < 256:
                wanted[slot] = True
        if not wanted.any():
            return np.empty(0, dtype=np.int64)
        found = []
        row = 0
        for leaf in self.leaves:
            count = int(leaf["point_count"])
            if count:
                start = int(leaf["byte_offset"]) + CLASS_BYTE
                classes = np.asarray(self._bytes[start:start + POINT.itemsize * count:POINT.itemsize])
                hits = np.flatnonzero(wanted[classes])
                if hits.size:
                    found.append(hits.astype(np.int64) + row)
            row += count
        return np.concatenate(found) if found else np.empty(0, dtype=np.int64)

    def slots_for_rows(self, rows):
        """(node, local, old class) of the given table rows."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            empty_u32 = np.empty(0, dtype=np.uint32)
            return empty_u32, empty_u32, np.empty(0, dtype=np.uint8)
        node, local, old = [], [], []
        for start in range(0, rows.size, self.PAGE):
            data = self.read(rows[start:start + self.PAGE])
            node.append(data["node"])
            local.append(data["local"])
            old.append(data["classification"])
        return (np.concatenate(node).astype(np.uint32),
                np.concatenate(local).astype(np.uint32),
                np.concatenate(old).astype(np.uint8))


class NativeFieldsModel(QAbstractTableModel):
    """Virtual table model over the .lnt records.

    Only the cells the view asks for are read; the records come from the file
    through a small page cache, so a multi-million point cloud stays
    responsive.
    """

    def __init__(self, reader, app, parent=None):
        super().__init__(parent)
        self.reader = reader
        self.app = app
        self.columns = list(COLUMN_DEFS)
        self.order = None            # int64 array: visible row -> source row
        self.codes = ()
        self.names = {}
        self.refresh_classes()

    # -- class codes / names ------------------------------------------------
    def refresh_classes(self):
        """Re-read the user class numbers and names (palette may have changed)."""
        from gui.class_slots import code_for
        from gui.dialogs.view_fields_table import CLASS_NAMES

        codes = tuple(code_for(self.app, slot) for slot in range(256))
        names = dict(CLASS_NAMES)
        for code, rule in (getattr(self.app, "class_palette", {}) or {}).items():
            try:
                code = int(code)
            except (TypeError, ValueError):
                continue
            if isinstance(rule, dict):
                label = rule.get("name") or rule.get("description")
                if label:
                    names[code] = label
        changed = codes != self.codes or names != self.names
        self.codes, self.names = codes, names
        if changed:
            # The file holds the stored slots; a remapped slot changes what the
            # table shows, so the cached pages are stale.
            self.reader.invalidate()

    # -- Qt model API -------------------------------------------------------
    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self.order) if self.order is not None else self.reader.count

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.columns[section][0]
        return section + 1

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role != Qt.DisplayRole:
            return None
        header, kind, key = self.columns[index.column()]
        point = self.reader.point(self.source_row(index.row()))
        return point_text(point, kind, key, self.codes, self.names)

    # -- rows / ordering ----------------------------------------------------
    def source_row(self, row):
        """Table row of the visible row (the record's index in the file)."""
        return int(self.order[row]) if self.order is not None else int(row)

    def visible_row(self, source_row):
        """Visible row of a table row, or None when it is filtered out."""
        if self.order is None:
            return int(source_row) if 0 <= source_row < self.reader.count else None
        matches = np.flatnonzero(self.order == int(source_row))
        return int(matches[0]) if matches.size else None

    def set_order(self, order):
        self.beginResetModel()
        self.order = order
        self.endResetModel()

    def set_visible_columns(self, headers):
        """Restrict the visible columns to the given header names.

        Returns True when at least one column matched, False otherwise.
        """
        wanted = set(headers or ())
        if not wanted:
            return False
        columns = [column for column in COLUMN_DEFS if column[0] in wanted]
        if not columns:
            return False
        self.beginResetModel()
        self.columns = columns
        self.endResetModel()
        return True

    def class_code_for_row(self, row):
        """User class number of a visible row, or None."""
        try:
            point = self.reader.point(self.source_row(row))
        except (IndexError, ValueError):
            return None
        return self.codes[int(point["classification"])]

    def rows_for_class(self, class_code):
        """Visible rows belonging to a user class number, as an int64 array."""
        codes = np.asarray(self.codes, dtype=np.int64)
        if self.order is None:
            slots = np.flatnonzero(codes == int(class_code))
            if slots.size == 0:
                return np.empty(0, dtype=np.int64)
            return self.reader.rows_for_slots(slots.tolist())
        rows = []
        for start in range(0, self.reader.count, self.reader.PAGE):
            data = self.reader.read(np.arange(start, min(start + self.reader.PAGE,
                                                         self.reader.count)))
            match = codes[data["classification"]] == int(class_code)
            if match.any():
                rows.append(np.flatnonzero(match).astype(np.int64) + start)
        if not rows:
            return np.empty(0, dtype=np.int64)
        source = np.concatenate(rows)
        visible = np.flatnonzero(np.isin(self.order, source))
        return visible.astype(np.int64, copy=False)

    def sort_by_class_code(self):
        """Stable sort by class, used when a class selection is fragmented."""
        try:
            self.set_order(np.argsort(np.asarray(self.codes, dtype=np.int64)[
                self._slot_column()], kind="stable"))
            return True
        except Exception:
            return False

    def _slot_column(self):
        slots = np.empty(self.reader.count, dtype=np.uint8)
        for start in range(0, self.reader.count, self.reader.PAGE):
            data = self.reader.read(np.arange(start, min(start + self.reader.PAGE,
                                                         self.reader.count)))
            slots[start:start + len(data["classification"])] = data["classification"]
        return slots


class FieldsTask(QThread):
    """Background work for the native Fields table.

    Three jobs, all streaming the file page by page so the GUI stays alive:
    sorting the cloud by one column, writing a CSV export, and finding the
    points inside a fence.
    """

    done = Signal(object, str)          # (result, error)

    def __init__(self, path, kind, key, descending=False, codes=(), names=None,
                 columns=None, rows=None, csv_path=None, polygons=None,
                 inside=True, parent=None):
        super().__init__(parent)
        self.path = path
        self.kind, self.key, self.descending = kind, key, bool(descending)
        self.codes = codes
        self.names = names or {}
        self.columns = columns or []
        self.rows = rows
        self.csv_path = csv_path
        self.polygons = polygons
        self.inside = bool(inside)

    def run(self):
        reader = None
        try:
            reader = NativeFieldsReader(self.path)
            if self.csv_path:
                self._export(reader)
                self.done.emit({"csv": self.csv_path}, "")
                return
            if self.polygons:
                self.done.emit({"rows": self._fence(reader)}, "")
                return
            order = self._sort(reader)
            self.done.emit({"order": order}, "")
        except Exception as exc:       # noqa: BLE001 - reported to the user
            self.done.emit(None, str(exc))
        finally:
            if reader is not None:
                reader.close()

    # -- fence --------------------------------------------------------------
    def _fence(self, reader):
        """Rows inside (or outside) the given polygons."""
        from gui.native_tiled_runner import _points_in_polygons
        found = []
        for start in range(0, reader.count, reader.PAGE):
            if self.isInterruptionRequested():
                return None
            rows = np.arange(start, min(start + reader.PAGE, reader.count))
            data = reader.read(rows)
            mask = np.zeros(len(rows), dtype=bool)
            for polygon in self.polygons:
                mask |= _points_in_polygons(data["xyz"], polygon)
            if not self.inside:
                mask = ~mask
            if mask.any():
                found.append(rows[mask])
        return np.concatenate(found) if found else np.empty(0, dtype=np.int64)

    # -- sorting ------------------------------------------------------------
    def _sort(self, reader):
        keys = np.empty(reader.count, dtype=np.float64)
        code_map = np.asarray(self.codes, dtype=np.float64)
        for start in range(0, reader.count, reader.PAGE):
            if self.isInterruptionRequested():
                return None
            rows = np.arange(start, min(start + reader.PAGE, reader.count))
            data = reader.read(rows)
            if self.kind in (K_CODE, K_DESC):
                keys[rows] = code_map[data["classification"]]
            elif self.kind == K_FLOAT3:
                axis = {"x": 0, "y": 1, "z": 2}[self.key]
                keys[rows] = data["xyz"][:, axis]
            elif self.kind == K_INT0:
                keys[rows] = data["intensity"]
            else:                       # rgb / hsv: sort by luminance
                rgb = data["rgb"].astype(np.float64)
                keys[rows] = (rgb[:, 0] * 0.299 + rgb[:, 1] * 0.587
                              + rgb[:, 2] * 0.114)
        order = np.argsort(keys, kind="stable")
        if self.descending:
            order = order[::-1]
        return order

    # -- csv ----------------------------------------------------------------
    def _export(self, reader):
        rows = self.rows
        if rows is None:
            rows = np.arange(reader.count, dtype=np.int64)
        rows = np.asarray(rows, dtype=np.int64)
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow([column[0] for column in self.columns])
            for start in range(0, rows.size, reader.PAGE):
                if self.isInterruptionRequested():
                    return
                chunk = rows[start:start + reader.PAGE]
                data = reader.read(chunk)
                for index in range(len(chunk)):
                    point = {key: values[index] for key, values in data.items()}
                    writer.writerow([point_text(point, kind, key, self.codes, self.names)
                                     for _header, kind, key in self.columns])


class NativeFieldsDialog(QDialog):
    """The point data table of the open native cloud (Display > Fields)."""

    def __init__(self, app, view):
        super().__init__(app)
        self.app, self.view = app, view
        self._task = None
        self._closed = False
        self._context_row = None
        self._class_selected_rows = None
        self._programmatic_class_selection = False
        self._pending_description = None
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowSystemMenuHint
                            | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint
                            | Qt.WindowCloseButtonHint)
        self.resize(1400, 720)
        self.setMinimumSize(0, 0)
        self.setSizeGripEnabled(True)
        from gui.theme_manager import get_dialog_stylesheet
        self.setStyleSheet(get_dialog_stylesheet() + self._extra_stylesheet())

        self.reader = NativeFieldsReader(view.lnt_path)
        self.model = NativeFieldsModel(self.reader, app, self)
        self.setWindowTitle("%s - %s points" % (getattr(view, "name",
                                                         os.path.basename(view.lnt_path)),
                                                format(self.reader.count, ",")))

        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SetNoConstraint)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._build_menubar(layout)
        self._build_table(layout)
        self.status = QLabel("All native points. Intensity/RGB are the values the "
                              "native cloud stores.")
        layout.addWidget(self.status)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    # -- look ---------------------------------------------------------------
    def _extra_stylesheet(self):
        from gui.theme_manager import ThemeColors
        c = ThemeColors
        return """
        QTableView#fieldsTable {
            background-color: %s;
            gridline-color: %s;
            color: %s;
            border: none;
            font-size: 11px;
        }
        QTableView#fieldsTable::item:selected {
            background-color: %s;
            color: %s;
        }
        QHeaderView::section {
            background-color: %s;
            color: %s;
            border: 1px solid %s;
            padding: 4px 6px;
        }
        """ % (c.get("bg_input"), c.get("border_light"), c.get("text_primary"),
               "#FFFFFF", "#000000",
               c.get("bg_panel", c.get("bg_input")), c.get("text_primary"),
               c.get("border_light"))

    def _build_menubar(self, layout):
        menubar = QMenuBar()
        menubar.setObjectName("fieldsMenuBar")
        # Direct action - clicking "Fields" opens the View Fields checklist.
        menubar.addAction("Fields", self._choose_fields)
        delete_menu = menubar.addMenu("Delete")
        delete_menu.addAction("By Class", self._delete_by_class)
        delete_menu.addAction("By Point", self._delete_by_point)
        delete_menu.addAction("By line", self._delete_by_line)
        delete_menu.addSeparator()
        delete_menu.addAction("Inside Fence", lambda: self._delete_by_fence(True))
        delete_menu.addAction("Outside Fence", lambda: self._delete_by_fence(False))
        layout.addWidget(menubar)

    def _build_table(self, layout):
        self.table = QTableView()
        self.table.setObjectName("fieldsTable")
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSortIndicatorShown(True)
        self.table.verticalHeader().setVisible(False)
        self.table.clicked.connect(self._on_row_clicked)
        self.table.doubleClicked.connect(self._go_to_selected)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._build_context_menu)
        self.table.selectionModel().selectionChanged.connect(
            self._on_table_selection_changed)
        self.table.horizontalHeader().sectionClicked.connect(self._sort_section)
        layout.addWidget(self.table, 1)

    # -- lifecycle ----------------------------------------------------------
    def current_cloud(self):
        return (getattr(self.app, "_lnt_main_view", None) is self.view
                and self.view.native is not None)

    def refresh(self):
        """Keep the table in step with the cloud (palette, class edits)."""
        if not self.current_cloud():
            self.close()
            return
        self.model.refresh_classes()
        rows = self.model.rowCount()
        if rows:
            self.model.dataChanged.emit(self.model.index(0, 0),
                                        self.model.index(rows - 1,
                                                        max(self.model.columnCount() - 1, 0)))

    def refresh_data(self, filename=None, app_data=None):
        """Compatibility hook (the point-count widget calls it after a reload).

        The native table always shows the .lnt on screen, so there is nothing
        to re-point - it only has to re-read the class names.
        """
        self.refresh()

    def closeEvent(self, event):
        self._closed = True
        self.timer.stop()
        if self._task is not None:
            self._task.requestInterruption()
        from gui.native_highlight import clear_highlight_group
        clear_highlight_group(self.app, "fields-point")
        try:
            self.reader.close()
        except Exception:
            pass
        super().closeEvent(event)

    # -- field selection ----------------------------------------------------
    def _choose_fields(self):
        """Open the View Fields checklist and apply it to the table."""
        from gui.dialogs.view_fields_dialog import ViewFieldsDialog
        chooser = ViewFieldsDialog(None, self, field_specs=NATIVE_FIELD_SPECS)
        if chooser.exec() != QDialog.Accepted:
            return
        selected = chooser.selected_fields()
        if not selected:
            # An empty choice would leave the table without a single column;
            # fall back to the default attributes of the native cloud.
            selected = [label for label, _dims, _always in NATIVE_FIELD_SPECS
                        if label in ("Class", "Description", "Easting",
                                     "Northing", "Elevation")]
        # Keep the Identify ribbon in sync with the choice.
        self.app.selected_view_fields = selected
        if not self.model.set_visible_columns(selected):
            QMessageBox.information(self, "View Fields",
                                    "No matching columns to display for the selected "
                                    "fields.")
            return
        self.status.setText("%s column(s) shown" % self.model.columnCount())

    # -- sorting ------------------------------------------------------------
    def _sort_section(self, column):
        if self._task is not None:
            return
        header = self.table.horizontalHeader()
        descending = (header.sortIndicatorSection() == column
                      and header.sortIndicatorOrder() == Qt.AscendingOrder)
        header.setSortIndicator(column, Qt.DescendingOrder if descending
                                else Qt.AscendingOrder)
        _header, kind, key = self.model.columns[column]
        self._start_task(kind, key, descending)

    def _start_task(self, kind, key, descending=False, rows=None, csv_path=None,
                    polygons=None, inside=True):
        if self._task is not None or not self.current_cloud():
            return
        task = FieldsTask(self.reader.path, kind, key, descending=descending,
                          codes=self.model.codes, names=dict(self.model.names),
                          columns=list(self.model.columns), rows=rows,
                          csv_path=csv_path, polygons=polygons, inside=inside,
                          parent=self.app)
        self._task = task
        # Keep the worker alive if this window is closed while it finishes.
        tasks = getattr(self.app, "_native_fields_tasks", None)
        if tasks is None:
            tasks = self.app._native_fields_tasks = []
        tasks.append(task)
        task.done.connect(self._task_done)
        task.finished.connect(lambda: tasks.remove(task) if task in tasks else None)
        task.finished.connect(task.deleteLater)
        self.status.setText("Working...")
        task.start()

    def _task_done(self, result, error):
        self._task = None
        if self._closed or not self.current_cloud():
            return
        if error:
            QMessageBox.warning(self, "Fields", error)
            self.status.setText("Failed")
            return
        if not result:
            return
        if "order" in result and result["order"] is not None:
            self.model.set_order(result["order"])
            self.status.setText("%s native points shown" % format(self.model.rowCount(), ","))
        if "csv" in result:
            self.status.setText("Exported %s" % result["csv"])
        if "rows" in result and result["rows"] is not None:
            description = self._pending_description or "the selected points"
            self._pending_description = None
            self._apply_delete(result["rows"], description)

    # -- selection helpers --------------------------------------------------
    def _selected_rows(self):
        if self._class_selected_rows is not None:
            return self._class_selected_rows
        selection = self.table.selectionModel()
        if selection is None or not selection.hasSelection():
            return np.empty(0, dtype=np.int64)
        return np.asarray(sorted({index.row() for index in selection.selectedRows()}),
                          dtype=np.int64)

    def _on_table_selection_changed(self, _selected, _deselected):
        if not self._programmatic_class_selection:
            self._class_selected_rows = None

    def _primary_action_row(self):
        if self._context_row is not None and 0 <= int(self._context_row) < self.model.rowCount():
            return int(self._context_row)
        rows = self._selected_rows()
        return int(rows[0]) if rows.size else None

    # -- row interactions ---------------------------------------------------
    def _on_row_clicked(self, index):
        if not index.isValid():
            return
        self._highlight_row(index.row())

    def _highlight_row(self, row, focus=True):
        """Mark the record in the 3D view and report it in the status line."""
        try:
            point = self.reader.point(self.model.source_row(row))
        except (IndexError, ValueError):
            return
        from gui.native_highlight import highlight_overlay_for
        overlay = highlight_overlay_for(self.app)
        if overlay is not None:
            x, y = float(point["xyz"][0]), float(point["xyz"][1])
            overlay.set_group("fields-point", [("drawing", [(x, y)],
                dict(type="point", color="#ff2d2d", width=5, source="attachment"))])
        code = self.model.codes[int(point["classification"])]
        self.status.setText("Point %s | Class %s | X %.3f  Y %.3f  Z %.3f" %
                            (self.model.source_row(row) + 1, code, *point["xyz"]))
        if focus:
            self._select_row(row)

    def _select_row(self, row):
        selection = self.table.selectionModel()
        if selection is None:
            return
        last = max(self.model.columnCount() - 1, 0)
        selection.clearSelection()
        selection.select(QItemSelection(self.model.index(row, 0),
                                        self.model.index(row, last)),
                         QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
        self.table.setCurrentIndex(self.model.index(row, 0))
        self.table.scrollTo(self.model.index(row, 0), QTableView.PositionAtCenter)

    def highlight_point(self, point_index):
        """Select and scroll to the row of a point picked in the 3D view.

        The native Identify tool hands over the SECTION STORE row of the
        picked record, so the record is located through the store's
        (node, local) identity - the same identity the .lnt file uses.
        """
        if not self.isVisible():
            return
        row = self._row_for_picked_point(point_index)
        if row is None:
            return
        visible = self.model.visible_row(row)
        if visible is None:
            return
        self._select_row(visible)

    def _row_for_picked_point(self, point_index):
        try:
            point_index = int(point_index)
        except (TypeError, ValueError):
            return None
        store = getattr(self.app, "_lnt_section_store", None)
        if store is not None and len(getattr(store, "node", ())):
            try:
                node = int(np.asarray(store.node)[point_index])
                local = int(np.asarray(store.local)[point_index])
                return self.reader.row_for(node, local)
            except (IndexError, ValueError, TypeError):
                return None
        # No store: the index already is a table row.
        if 0 <= point_index < self.reader.count:
            return point_index
        return None

    def _go_to_selected(self):
        """Zoom the native camera to the selected record."""
        row = self._primary_action_row()
        if row is None:
            return
        try:
            point = self.reader.point(self.model.source_row(row))
        except (IndexError, ValueError):
            return
        x, y, z = (float(v) for v in point["xyz"])
        try:
            self.view.set_2d(True)
            self.view.native.frame_region(x, y, max(self._point_radius() * 40.0, 1.0))
        except Exception as exc:
            print(f"[Fields] camera zoom failed: {exc}")
            return
        self._highlight_row(row, focus=False)
        self.status.setText("Zoomed to point %s (%.3f, %.3f, %.3f)"
                            % (self.model.source_row(row) + 1, x, y, z))

    def _point_radius(self):
        """A marker size for the cloud (the reference used the same rule)."""
        try:
            span = np.linalg.norm(self.reader.bounds_max - self.reader.bounds_min)
            return max(float(span) / 5000.0, 1e-4)
        except Exception:
            return 1.0

    def _identify_selected(self):
        """Identify the selected record (the native Identify path)."""
        row = self._primary_action_row()
        if row is None:
            return
        try:
            point = self.reader.point(self.model.source_row(row))
        except (IndexError, ValueError):
            return
        code = self.model.codes[int(point["classification"])]
        tool = getattr(self.app, "identification_tool", None)
        if tool is None:
            QMessageBox.information(self, "Identify",
                                    "The Identification tool is not available.")
            return
        try:
            name = tool.get_class_name(int(code))
        except Exception:
            name = str(code)
        xyz = tuple(float(v) for v in point["xyz"])
        store_row = self._store_row_for_source(self.model.source_row(row))
        try:
            tool.highlight_class(int(code))
        except Exception:
            pass
        update = getattr(tool, "_update_ribbon_info", None)
        if callable(update):
            try:
                update(int(code), name, xyz, point_index=store_row)
                tool.point_identified.emit(int(code), name, xyz)
            except Exception as exc:
                print(f"[Fields] identify failed: {exc}")
        self._highlight_row(row, focus=False)
        self.status.setText("Identified point %s - class %s" %
                            (self.model.source_row(row) + 1, code))

    def _store_row_for_source(self, source_row):
        """The section store row of a table row, when the store holds it."""
        store = getattr(self.app, "_lnt_section_store", None)
        if store is None or not len(getattr(store, "node", ())):
            return None
        try:
            data = self.reader.read(np.asarray([int(source_row)], dtype=np.int64))
            node = np.uint64(int(data["node"][0]))
            local = np.uint64(int(data["local"][0]))
            key = (node << np.uint64(32)) | local
            store_key = (np.asarray(store.node, dtype=np.uint64) << np.uint64(32)
                         | np.asarray(store.local, dtype=np.uint64))
            hits = np.flatnonzero(store_key == key)
            return int(hits[0]) if hits.size else None
        except Exception:
            return None

    def _select_by_class(self):
        row = self._context_row
        if row is None:
            current = self.table.currentIndex()
            row = current.row() if current.isValid() else None
        if row is None or row < 0:
            return
        class_code = self.model.class_code_for_row(row)
        if class_code is None:
            QMessageBox.information(self, "Select by Class",
                                    "Classification is not available.")
            return
        rows = self.model.rows_for_class(class_code)
        if rows.size == 0:
            return

        def runs(values):
            if values.size == 0:
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
            breaks = np.flatnonzero(np.diff(values) != 1)
            starts = np.r_[0, breaks + 1]
            ends = np.r_[breaks, values.size - 1]
            return values[starts], values[ends]

        run_starts, run_ends = runs(rows)
        if run_starts.size > 5000 and self.model.sort_by_class_code():
            rows = self.model.rows_for_class(class_code)
            run_starts, run_ends = runs(rows)
            self.status.setText("Table sorted by Class for efficient class selection")
        last = max(self.model.columnCount() - 1, 0)
        selection = QItemSelection()
        for start_row, end_row in zip(run_starts, run_ends):
            selection.select(self.model.index(int(start_row), 0),
                             self.model.index(int(end_row), last))
        selection_model = self.table.selectionModel()
        self._programmatic_class_selection = True
        try:
            selection_model.select(selection, QItemSelectionModel.ClearAndSelect
                                   | QItemSelectionModel.Rows)
            first = self.model.index(int(rows[0]), 0)
            self.table.setCurrentIndex(first)
            self.table.scrollTo(first, QTableView.PositionAtCenter)
            self._class_selected_rows = rows
        finally:
            self._programmatic_class_selection = False
        self.status.setText("Selected %s row(s) in class %s" %
                            (format(int(rows.size), ","), class_code))

    # -- clipboard / export -------------------------------------------------
    def _copy_selected_rows(self):
        rows = self._selected_rows()
        if rows.size == 0:
            return
        if rows.size > 100_000:
            reply = QMessageBox.question(
                self, "Copy Selected Rows",
                "Copy %s rows to the clipboard?\n\nThis can use a large amount of "
                "memory. For very large selections, Export Selection to CSV is "
                "recommended." % format(int(rows.size), ","),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        lines = ['\t'.join(self.model.columns[column][0]
                           for column in range(self.model.columnCount()))]
        for row in rows:
            cells = []
            for column in range(self.model.columnCount()):
                value = self.model.data(self.model.index(int(row), column),
                                        Qt.DisplayRole)
                cells.append(str(value or ""))
            lines.append("\t".join(cells))
        QApplication.clipboard().setText("\n".join(lines))
        self.status.setText("Copied %s row(s) to clipboard" % format(int(rows.size), ","))

    def _export_csv(self):
        selected = self._selected_rows()
        if selected.size:
            rows = np.asarray([self.model.source_row(int(row)) for row in selected],
                              dtype=np.int64)
            default_name = "selected_points.csv"
        else:
            total = self.model.rowCount()
            if total > 1_000_000:
                QMessageBox.warning(
                    self, "Export CSV",
                    "This would export %s rows. Exporting the full table to CSV "
                    "can be very large; select a subset of rows to export instead."
                    % format(total, ","))
                return
            rows = None
            default_name = "point_fields.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", default_name,
                                              "CSV Files (*.csv)")
        if not path:
            return
        _header, kind, key = self.model.columns[0]
        self._start_task(kind, key, rows=rows, csv_path=path)

    # -- delete menu --------------------------------------------------------
    def _pick_codes_dialog(self, title, label_text, items):
        """items: [(code, text)]. Returns the chosen codes, or None."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        from gui.theme_manager import get_dialog_stylesheet
        dialog.setStyleSheet(get_dialog_stylesheet())
        dialog.resize(320, 380)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(label_text))
        list_widget = QListWidget()
        for code, text in items:
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, code)
            list_widget.addItem(item)
        layout.addWidget(list_widget, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        if dialog.exec() != QDialog.Accepted:
            return None
        chosen = [list_widget.item(index).data(Qt.UserRole)
                  for index in range(list_widget.count())
                  if list_widget.item(index).checkState() == Qt.Checked]
        return chosen or None

    def _class_items(self):
        """(user class number, label, count) for every class in the cloud."""
        counts = None
        counter = getattr(self.view, "class_counts", None)
        if counter is not None:
            try:
                counts = counter.counts(getattr(self.app, "_lnt_section_store", None))
            except Exception:
                counts = None
        if not counts:
            try:
                from gui.lnt_class_counts import scan_class_histogram
                from gui.class_slots import code_for
                histogram = scan_class_histogram(self.reader.path)
                counts = {}
                for slot, count in enumerate(histogram.tolist()):
                    if count:
                        counts[code_for(self.app, slot)] = int(count)
            except Exception:
                counts = {}
        items = []
        for code, count in sorted((counts or {}).items(), key=lambda item: item[0]):
            items.append((int(code), "%d - %s (%s pts)"
                          % (int(code), self.model.names.get(int(code), "Class %d" % int(code)),
                             format(int(count), ","))))
        return items

    def _delete_by_class(self):
        items = self._class_items()
        if not items:
            QMessageBox.information(self, "Delete - By Class",
                                    "No classification data loaded.")
            return
        chosen = self._pick_codes_dialog(
            "Delete By Class", "Select classification code(s) to delete:", items)
        if not chosen:
            return
        rows = self.reader.rows_for_slots(self._slots_for_codes(chosen))
        self._apply_delete(rows, "class(es) %s" % ", ".join(str(code) for code in chosen))

    def _slots_for_codes(self, codes):
        """Stored class slots that carry the given user class numbers."""
        from gui.class_slots import code_for
        wanted = set(int(code) for code in codes)
        return [slot for slot in range(256) if int(code_for(self.app, slot)) in wanted]

    def _delete_by_point(self):
        selection = self.table.selectionModel()
        if selection is None or not selection.hasSelection():
            QMessageBox.information(self, "Delete - By Point",
                                    "Select a row in the table first.")
            return
        row = int(selection.selectedRows()[0].row())
        self._apply_delete(np.asarray([self.model.source_row(row)], dtype=np.int64),
                           "the selected row")

    def _delete_by_line(self):
        QMessageBox.information(
            self, "Delete - By Line",
            "Line numbers (point_source_id) are not kept by the native cloud.")

    def _fence_shapes(self):
        """Fence shapes drawn in the app (native draw tool or VTK digitizer)."""
        shapes = []
        digitizer = getattr(self.app, "digitizer", None)
        for drawing in (getattr(digitizer, "drawings", None) or []):
            if isinstance(drawing, dict):
                shapes.append(drawing)
        draw = getattr(self.view, "drawing", None)
        for drawing in (getattr(draw, "drawings", None) or []):
            if isinstance(drawing, dict):
                shapes.append(drawing)
        return [shape for shape in shapes
                if str(shape.get("type", "")).lower() in
                ("rectangle", "circle", "polygon", "freehand", "line",
                 "smart_line", "polyline", "smartline")]

    def _delete_by_fence(self, inside=True):
        shapes = self._fence_shapes()
        if not shapes:
            QMessageBox.information(
                self, "Delete - Fence",
                "No fence shape found. Draw a shape with the Draw tools first.")
            return
        items = [(index, "%s #%d" % (str(shape.get("type", "shape")).title(), index + 1))
                 for index, shape in enumerate(shapes)]
        chosen = self._pick_codes_dialog("Select Fence",
                                         "Select the fence shape(s) to use:", items)
        if not chosen:
            return
        from gui.menu_sidebar_system import _prepare_fence_shape
        polygons = []
        for index in chosen:
            shape = _prepare_fence_shape(shapes[int(index)])
            coords = shape.get("coords")
            if coords is None or len(coords) < 3:
                continue
            polygons.append(np.asarray(coords, dtype=np.float64))
        if not polygons:
            QMessageBox.information(self, "Delete - Fence",
                                    "The selected shape is not a closed area.")
            return
        label = "inside" if inside else "outside"
        self._pending_description = "point(s) %s the selected fence" % label
        self._start_task(self.model.columns[0][1], self.model.columns[0][2],
                         polygons=polygons, inside=inside)

    def _apply_delete(self, rows, description):
        """Move the given records to the noise class (the native, undoable
        equivalent of deleting points - a native cloud is append-only)."""
        rows = np.asarray(rows, dtype=np.int64)
        count = int(rows.size)
        if count == 0:
            QMessageBox.information(self, "Delete",
                                    "No points matched - nothing to delete.")
            return
        reply = QMessageBox.question(
            self, "Confirm Delete",
            "Delete %s point(s) - %s?\n\nNative clouds are append-only: the points "
            "are moved to the noise class (class %d) and can be undone with Ctrl+Z."
            % (format(count, ","), description, NOISE_CLASS),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        node, local, old = self.reader.slots_for_rows(rows)
        new = np.full(count, NOISE_CLASS, dtype=np.uint8)
        if not self.view.native.set_classes(node, local, new):
            QMessageBox.warning(self, "Delete",
                                self.view.native.last_error or "The class write failed.")
            return
        self._push_undo(node, local, old, new)
        # The section store keeps its own copy of the classes: it must not
        # write the stale ones back over the file.
        try:
            from gui.lnt_ai_bridge import sync_store_rows
            sync_store_rows(self.app, self.view, node, local, new)
        except Exception:
            pass
        self.reader.invalidate()
        self.model.refresh_classes()
        self.refresh()
        self.status.setText("Deleted %s point(s) - moved to class %d"
                            % (format(count, ","), NOISE_CLASS))

    def _push_undo(self, node, local, old, new):
        """One undo step, by .lnt identity (the native history format)."""
        entry = {
            "lnt_path": self.view.lnt_path,
            "lnt_node": np.asarray(node, dtype=np.uint32),
            "lnt_local": np.asarray(local, dtype=np.uint32),
            "old_classes": np.asarray(old, dtype=np.uint8),
            "new_classes": np.asarray(new, dtype=np.uint8),
        }
        stack = getattr(self.app, "undo_stack", None)
        if stack is None:
            stack = []
            self.app.undo_stack = stack
        stack.append(entry)
        redo = getattr(self.app, "redo_stack", None)
        if redo is not None:
            try:
                redo.clear()
            except Exception:
                pass
        try:
            from gui.memory_manager import trim_undo_stack
            trim_undo_stack(self.app)
        except Exception:
            pass

    # -- context menu -------------------------------------------------------
    def _build_context_menu(self, pos):
        index = self.table.indexAt(pos)
        if not index.isValid():
            return
        self._context_row = index.row()
        selection = self.table.selectionModel()
        if selection is not None and not selection.isRowSelected(index.row(), QModelIndex()):
            self._class_selected_rows = None
            selection.select(self.model.index(index.row(), 0),
                             QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
            self.table.setCurrentIndex(self.model.index(index.row(), 0))
        class_code = self.model.class_code_for_row(index.row())
        menu = QMenu(self)
        copy_label = ("Copy Row" if self._selected_rows().size <= 1
                      else "Copy Selected Rows")
        menu.addAction(copy_label, self._copy_selected_rows)
        menu.addAction("Export Selection to CSV", self._export_csv)
        if class_code is not None:
            menu.addSeparator()
            menu.addAction("Select by Class - %d (%s)"
                           % (class_code, self.model.names.get(class_code,
                                                              "Class %d" % class_code)),
                           self._select_by_class)
        menu.addSeparator()
        menu.addAction("Zoom to Point in 3D", self._go_to_selected)
        menu.addAction("Identify Point", self._identify_selected)
        menu.exec(self.table.viewport().mapToGlobal(pos))


def open_native_fields(app):
    """Open the Fields table for the cloud on screen (never a partial section)."""
    view = getattr(app, "_lnt_main_view", None)
    if view is None or getattr(view, "native", None) is None:
        QMessageBox.information(app, "View Fields",
                                "Load a native point cloud first to view its fields.")
        return
    existing = getattr(app, "_view_fields_table_dialog", None)
    if isinstance(existing, NativeFieldsDialog) and existing.view is view:
        existing.refresh()
        existing.showNormal()
        existing.raise_()
        existing.activateWindow()
        return
    if existing is not None:
        existing.close()
    try:
        dialog = NativeFieldsDialog(app, view)
    except (OSError, ValueError) as exc:
        QMessageBox.warning(app, "View Fields", str(exc))
        return
    app._view_fields_table_dialog = dialog

    def finished(_result):
        if getattr(app, "_view_fields_table_dialog", None) is dialog:
            app._view_fields_table_dialog = None

    dialog.finished.connect(finished)
    dialog.show()
