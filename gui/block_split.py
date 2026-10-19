"""Block Split: write one point-cloud file per created block (no VTK).

Shared by the legacy Block Creation Settings dialog and the native Vulkan one
(gui/native_create.py).  Points come from the in-memory cloud, the open .lnt's
source LAS/LAZ (with the .lnt's current classes), or the LAZ/LAS files listed in
the PRJ next to the loaded SNT.
"""
import json
import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def _get_block_dialog_style():
    from gui.theme_manager import ThemeColors as _TC

    return f"""
QDialog {{
    background-color: {_TC.get('bg_secondary')};
}}
QWidget#dialogContent {{
    background-color: {_TC.get('bg_secondary')};
    color: {_TC.get('text_primary')};
}}
QGroupBox {{
    background-color: {_TC.get('bg_primary')};
    border: 1px solid {_TC.get('border_light')};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    padding-bottom: 6px;
    font-weight: bold;
    color: {_TC.get('text_primary')};
    font-size: 9.5pt;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    font-size: 9.5pt;
    letter-spacing: 0.3px;
    color: {_TC.get('accent')};
}}
QLabel {{
    color: {_TC.get('text_primary')};
    font-size: 9pt;
    font-weight: 500;
}}
QComboBox, QDoubleSpinBox, QLineEdit, QListWidget {{
    background-color: {_TC.get('bg_input')};
    color: {_TC.get('text_primary')};
    border: 1px solid {_TC.get('border_light')};
    border-radius: 4px;
    padding: 2px 8px;
    min-height: 22px;
    max-height: 22px;
    font-size: 9pt;
}}
QListWidget {{
    max-height: none;
}}
QComboBox:hover, QDoubleSpinBox:hover, QLineEdit:hover, QListWidget:hover {{
    border: 1px solid {_TC.get('accent')};
    background-color: {_TC.get('bg_button_hover')};
}}
QLineEdit:read-only {{
    color: {_TC.get('text_secondary')};
}}
QListWidget::item {{
    padding: 3px 4px;
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QPushButton {{
    background-color: {_TC.get('bg_button')};
    color: {_TC.get('text_primary')};
    border: 1px solid {_TC.get('border_light')};
    border-radius: 4px;
    padding: 2px 12px;
    font-weight: bold;
    font-size: 9pt;
    min-height: 22px;
    max-height: 22px;
}}
QPushButton:hover {{
    background-color: {_TC.get('bg_button_hover')};
    border-color: {_TC.get('accent')};
}}
QPushButton:pressed {{
    background-color: {_TC.get('accent')};
    color: {_TC.get('text_on_active')};
}}
QPushButton:checked {{
    background-color: {_TC.get('accent')};
    border-color: {_TC.get('accent_hover')};
    color: {_TC.get('text_on_active')};
}}
QPushButton#okButton {{
    background-color: {_TC.get('accent')};
    border-color: {_TC.get('accent')};
    color: {_TC.get('text_on_active')};
}}
QPushButton#okButton:hover {{
    background-color: {_TC.get('accent_hover')};
}}
QPushButton#splitButton {{
    background-color: #1a1a00;
    border: 1px solid #ccff00;
    color: #ccff00;
    font-weight: bold;
}}
QPushButton#splitButton:hover {{
    background-color: #2e2e00;
    border-color: #eeff44;
    color: #eeff44;
}}
QPushButton#splitButton:pressed {{
    background-color: #ccff00;
    color: #000000;
}}
"""


def _points_in_polygon(x, y, polygon):
    """Vectorised ray cast: which of the (x, y) points lie inside ``polygon``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    inside = np.zeros(x.shape[0], dtype=bool)
    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
        )
        inside ^= crosses
        j = i
    return inside


def _effective_block_rings(block):
    """World rings of a block after the sliver merge ([] when unchanged)."""
    rings = []
    for ring in (block or {}).get("effective_polygon") or []:
        try:
            pts = [(float(point[0]), float(point[1])) for point in ring]
        except Exception:
            continue
        if len(pts) >= 4:
            rings.append(pts)
    return rings


def _block_mask_rings(block):
    """Rings deciding which points belong to a block (merged geometry wins).

    Module level on purpose: the Block Creation Settings save path and the Split
    Blocks dialog both need it, so the post-merge ``effective_polygon`` is
    honoured by every writer instead of only whichever class defined it.
    """
    rings = _effective_block_rings(block)
    if rings:
        return [np.asarray(ring, dtype=np.float64) for ring in rings]
    polygon = np.asarray((block or {}).get("polygon"), dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[0] < 3:
        return []
    return [polygon]


def _ring_shape(ring):
    """Prepared shapely area equal to the ray cast's even-odd region, or None.

    A corridor ring self-intersects on the inside of a sharp CL bend.  GEOS
    predicates are unreliable on such a ring, so it is rebuilt with the
    "linework" make_valid, whose faces alternate inside/outside exactly like
    the even-odd ray cast.  None (ray cast it) if GEOS cannot build it.
    """
    try:
        import shapely
        shape = shapely.Polygon(np.asarray(ring, dtype=np.float64)[:, :2])
        if not shape.is_valid:
            shape = shapely.make_valid(shape, method="linework")
        if shape.is_empty:
            return None
        shapely.prepare(shape)
        return shape
    except Exception:
        return None


def _points_in_ring_list(x, y, rings, shapes=None):
    """Union of the point-in-polygon masks of several rings.

    Valid rings are tested in GEOS (prepared polygons, linear in the points
    instead of points x vertices in Python).  A point exactly on an outline
    counts as inside; the caller's ``assigned`` mask gives it to one block.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.zeros(np.shape(x), dtype=bool)
    if shapes is None:
        shapes = [_ring_shape(ring) for ring in rings]
    for ring, shape in zip(rings, shapes):
        if shape is not None:
            import shapely
            mask |= shapely.intersects_xy(shape, x, y)
        else:
            mask |= _points_in_polygon(x, y, ring)
    return mask


def _block_entries(blocks):
    """Per block: its mask rings, prepared shapes and bounding box."""
    entries = []
    for block in blocks:
        mask_rings = _block_mask_rings(block)
        if not mask_rings:
            continue
        corners = np.vstack([ring[:, :2] for ring in mask_rings])
        entries.append({
            "block": block,
            "polygon": np.asarray(block.get("polygon"), dtype=np.float64),
            "mask_rings": mask_rings,
            "shapes": [_ring_shape(ring) for ring in mask_rings],
            "min": corners.min(axis=0),
            "max": corners.max(axis=0),
        })
    return entries


def _entries_touching(entries, lo, hi):
    """Entries whose bounding box overlaps the box lo..hi."""
    return [
        entry for entry in entries
        if entry["max"][0] >= lo[0] and entry["min"][0] <= hi[0]
        and entry["max"][1] >= lo[1] and entry["min"][1] <= hi[1]
    ]


_POOL = None


def _pool():
    """Shared worker threads for the per-block polygon tests (GEOS drops the GIL)."""
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 2))),
                                   thread_name_prefix="block-split")
    return _POOL


def _inside_box_and_rings(x, y, entry):
    """Ascending indices of the chunk points inside one block."""
    lo, hi = entry["min"], entry["max"]
    idx = np.flatnonzero((x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1]))
    if idx.size == 0:
        return idx
    return idx[_points_in_ring_list(x[idx], y[idx], entry["mask_rings"], entry["shapes"])]


def _select_chunk(x, y, entries):
    """[(entry, ascending point indices)] for one chunk, in block priority order.

    A point belongs to the first block (in ``entries`` order) that contains it.
    Only blocks whose box overlaps the chunk's are tested; their polygon tests
    run in parallel and the priority is applied afterwards, in block order.
    """
    if x.size == 0:
        return []
    active = _entries_touching(entries, (x.min(), y.min()), (x.max(), y.max()))
    if not active:
        return []
    if len(active) == 1:
        inside = [_inside_box_and_rings(x, y, active[0])]
    else:
        inside = list(_pool().map(lambda entry: _inside_box_and_rings(x, y, entry), active))
    assigned = np.zeros(x.shape[0], dtype=bool) if len(active) > 1 else None
    picked = []
    for entry, sel in zip(active, inside):
        if assigned is not None and sel.size:
            sel = sel[~assigned[sel]]
            assigned[sel] = True
        if sel.size:
            picked.append((entry, sel))
    return picked


def _read_ahead(chunks):
    """Yield from ``chunks`` while the next item is already being read/decompressed."""
    import queue
    import threading

    box = queue.Queue(maxsize=1)
    done = object()

    def feed():
        try:
            for item in chunks:
                box.put(item)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
            box.put(exc)
        box.put(done)

    threading.Thread(target=feed, name="block-split-read", daemon=True).start()
    while True:
        item = box.get()
        if item is done:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


class BlockSplitSaveDialog(QDialog):
    def __init__(self, app, blocks, parent=None):
        super().__init__(parent or app)
        self.app = app
        self.blocks = sorted(blocks or [], key=lambda item: item.get("priority", 0))
        self.source_path = self._get_source_path()
        self.source_stem = self.source_path.stem if self.source_path else "pointcloud"
        self.output_ext = self._get_output_ext()
        self.parent_dir = self._default_output_parent()
        self.setWindowTitle("Save Split Blocks As")
        self.setModal(True)
        self.resize(520, 430)
        self.setStyleSheet(_get_block_dialog_style())
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 5, 18, 10)
        layout.setSpacing(12)

        blocks_group = QGroupBox("Blocks")
        blocks_layout = QVBoxLayout(blocks_group)
        blocks_layout.setContentsMargins(15, 10, 15, 10)
        blocks_layout.setSpacing(8)
        self.block_list = QListWidget()
        for block in self.blocks:
            self.block_list.addItem(self._block_file_stem(block) + self.output_ext)
        blocks_layout.addWidget(self.block_list, 1)
        layout.addWidget(blocks_group, 1)

        location_group = QGroupBox("Save Location")
        location_form = QFormLayout(location_group)
        location_form.setLabelAlignment(Qt.AlignRight)
        location_form.setHorizontalSpacing(12)
        location_form.setVerticalSpacing(8)
        location_form.setContentsMargins(15, 10, 15, 10)

        self.folder_name_edit = QLineEdit(f"{self.source_stem}_Split_Files")
        # Runtime shortcuts remain available everywhere else in this dialog.
        # Only literal filename entry owns alphabet keys while it has focus.
        self.folder_name_edit.setProperty("nakshaStrictTextInput", True)
        self.folder_name_edit.textChanged.connect(self._refresh_destination_path)
        location_form.addRow("Folder name:", self.folder_name_edit)

        location_row = QWidget()
        location_row_layout = QHBoxLayout(location_row)
        location_row_layout.setContentsMargins(0, 0, 0, 0)
        location_row_layout.setSpacing(8)
        self.parent_dir_edit = QLineEdit(str(self.parent_dir))
        self.parent_dir_edit.setReadOnly(True)
        location_row_layout.addWidget(self.parent_dir_edit, 1)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_parent_dir)
        location_row_layout.addWidget(self.browse_btn)
        location_form.addRow("Save in:", location_row)

        self.destination_label = QLabel("")
        self.destination_label.setWordWrap(True)
        location_form.addRow("Destination:", self.destination_label)
        layout.addWidget(location_group)
        self._refresh_destination_path()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.save_btn = QPushButton("Save")
        self.save_btn.setObjectName("okButton")
        self.save_btn.setDefault(True)
        self.cancel_btn = QPushButton("Cancel")
        self.save_btn.clicked.connect(self._save_split_blocks)
        self.folder_name_edit.returnPressed.connect(self._save_split_blocks)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

    def _get_source_path(self):
        for attr in ("loaded_file", "current_file_path", "last_save_path"):
            value = getattr(self.app, attr, None)
            if value:
                try:
                    return Path(value)
                except Exception:
                    pass
        data = getattr(self.app, "data", None)
        if isinstance(data, dict):
            value = data.get("_source_file_path") or data.get("filename")
            if value:
                try:
                    return Path(value)
                except Exception:
                    pass
        for snt_file in self._loaded_snt_paths():
            return snt_file
        return None

    def _loaded_snt_paths(self):
        paths = []
        for attachment in getattr(self.app, "snt_attachments", []) or []:
            full_path = attachment.get("full_path") if isinstance(attachment, dict) else None
            if full_path:
                try:
                    paths.append(Path(full_path))
                except Exception:
                    pass
        return paths

    def _get_output_ext(self):
        if self.source_path and self.source_path.suffix.lower() in (".las", ".laz"):
            return self.source_path.suffix.lower()
        return ".laz"

    def _default_output_parent(self):
        if self.source_path and self.source_path.parent:
            return self.source_path.parent
        return Path(os.getcwd())

    def _output_parent(self):
        return self.parent_dir

    def _output_dir(self):
        folder_name = self.folder_name_edit.text().strip()
        return self._output_parent() / folder_name if folder_name else self._output_parent()

    def _refresh_destination_path(self):
        if hasattr(self, "destination_label"):
            self.destination_label.setText(f"Destination: {self._output_dir()}")

    def _browse_parent_dir(self):
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Save Location",
            str(self.parent_dir),
        )
        if not selected:
            return
        self.parent_dir = Path(selected)
        self.parent_dir_edit.setText(str(self.parent_dir))
        self._refresh_destination_path()

    @staticmethod
    def _distance_token(distance):
        distance = float(distance or 0.0)
        if abs(distance - round(distance)) < 1e-6:
            return str(int(round(distance)))
        return f"{distance:.4f}".rstrip("0").rstrip(".")

    def _block_file_stem(self, block):
        serial = int(block.get("priority", 0) or 0)
        distance_token = self._distance_token(block.get("distance", 0.0))
        return f"{self.source_stem}-{distance_token}-{serial:02d}"

    def _save_split_blocks(self):
        folder_name = self.folder_name_edit.text().strip()
        if not folder_name:
            QMessageBox.warning(self, "Split Blocks", "Enter a folder name.")
            return

        output_dir = self._output_dir()
        progress = QProgressDialog("Saving split blocks...", None, 0, 1000, self)
        progress.setWindowTitle("Split Blocks")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        self.save_btn.setEnabled(False)

        def work():
            output_dir.mkdir(parents=True, exist_ok=True)
            saved, skipped, saved_blocks = self._write_block_files(output_dir)
            if saved_blocks:
                self._set_progress(1.0, "Creating SNT and PRJ files...")
                saved_blocks = self._prepare_manifest_blocks(saved_blocks)
                manifest_base = output_dir.name or self.source_stem
                self._write_prj_manifest(output_dir / f"{manifest_base}.prj", saved_blocks)
                self._write_snt_manifest(output_dir / f"{manifest_base}.snt", saved_blocks)
            return saved, skipped, saved_blocks

        try:
            saved, skipped, saved_blocks = self._run_in_worker(work, progress)
        except Exception as exc:
            progress.close()
            self.save_btn.setEnabled(True)
            QMessageBox.critical(self, "Split Blocks", f"Failed to save split files:\n{exc}")
            return
        finally:
            progress.close()

        if skipped:
            manifest_message = (
                f"Created {output_dir.name}.snt and {output_dir.name}.prj.\n\n"
                if saved_blocks else ""
            )
            QMessageBox.information(
                self,
                "Split Blocks",
                f"Saved {saved} block file(s) to:\n{output_dir}\n\n"
                f"{manifest_message}"
                f"{skipped} block(s) had no points and were skipped.",
            )
        else:
            manifest_message = (
                f"\n\nCreated {output_dir.name}.snt and {output_dir.name}.prj."
                if saved_blocks else ""
            )
            QMessageBox.information(
                self,
                "Split Blocks",
                f"Saved {saved} block file(s) to:\n{output_dir}{manifest_message}",
            )
        self.accept()

    def _run_in_worker(self, work, progress):
        """Run ``work`` off the UI thread, keeping the window and progress live.

        Reading, point assignment and LAZ compression (numpy, GEOS, lazrs) release
        the GIL, so the UI thread only repaints the progress meanwhile.  ``work``
        must not touch Qt; it reports through _set_progress.
        """
        import threading
        from PySide6.QtCore import QEventLoop

        result = {}

        def target():
            try:
                result["value"] = work()
            except BaseException as exc:  # noqa: BLE001 - re-raised on the UI thread
                result["error"] = exc

        self._progress_state = (0.0, "Saving split blocks...")
        thread = threading.Thread(target=target, name="block-split-save", daemon=True)
        thread.start()
        shown = None
        while thread.is_alive():
            QApplication.processEvents(QEventLoop.AllEvents, 50)
            thread.join(0.05)
            state = self._progress_state
            if state != shown:
                shown = state
                progress.setValue(int(state[0] * 1000))
                progress.setLabelText(state[1])
        if "error" in result:
            raise result["error"]
        return result["value"]

    def _set_progress(self, fraction, text):
        """Called from the save worker thread; the UI thread polls it (no Qt here)."""
        self._progress_state = (max(0.0, min(1.0, float(fraction))), str(text))

    def _select_in_arrays(self, x, y, entries, text, chunk_size=2_000_000):
        """[(entry, ascending point indices)] for whole x/y arrays, chunk by chunk.

        Same assignment as one pass over all points: earlier chunks never hold a
        point of a later one, and inside a chunk the first block wins.
        """
        parts = {id(entry): [] for entry in entries}
        total = int(x.shape[0])
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            self._set_progress(stop / max(total, 1), text)
            for entry, sel in _select_chunk(x[start:stop], y[start:stop], entries):
                parts[id(entry)].append(sel + start)
        return [
            (entry, np.concatenate(parts[id(entry)]) if parts[id(entry)] else np.empty(0, np.int64))
            for entry in entries
        ]

    def _lnt_view(self):
        """The native .lnt main view when one is open (its points live in the file)."""
        view = getattr(self.app, "_lnt_main_view", None)
        if view is None or getattr(view, "native", None) is None or not getattr(view, "lnt_path", None):
            return None
        return view

    def _write_block_files_from_lnt(self, output_dir, view):
        """Split the open .lnt's source LAS/LAZ with the classes the .lnt holds now.

        The .lnt keeps only position and class per point, so every other field
        (intensity, returns, GPS time, colour, ...) and the header come from the
        source file; the classes come from the .lnt so the user's edits are in
        the split files.
        """
        import laspy
        from gui.lnt_ai_bridge import lnt_classes_at
        from gui.lnt_class_counts import read_directory

        source = getattr(self.app, "_lnt_source_path", None)
        if not source or not os.path.isfile(source) or \
                os.path.splitext(source)[1].lower() not in (".las", ".laz"):
            raise RuntimeError(
                "Split needs the source LAS/LAZ of the open .lnt, which was not found:\n"
                f"{source or '(unknown)'}"
            )
        store = getattr(self.app, "_lnt_section_store", None)
        if store is not None and getattr(store, "view", None) is view:
            store.flush()                          # the .lnt must hold every shown edit

        self._set_progress(0.0, f"Reading {Path(source).name}...")
        las = laspy.read(source)
        total, directory = read_directory(view.lnt_path)
        if int(len(las.points)) != int(total):
            raise RuntimeError(
                f"{os.path.basename(source)} has {len(las.points):,} points but the open "
                f".lnt has {total:,}; reopen the file to rebuild its .lnt."
            )
        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        print(
            f"[BlockSplit] points source = open .lnt {Path(view.lnt_path).name} "
            f"(fields from {Path(source).name}) points={int(total):,}"
        )

        src_header = las.header
        assigned_count = 0
        saved = 0
        entries = _block_entries(self.blocks)
        skipped = len(self.blocks) - len(entries)
        saved_blocks = []
        selections = self._select_in_arrays(x, y, entries, "Assigning points to blocks...")

        # Only the written points need their current .lnt class.
        self._set_progress(1.0, "Reading the current classes from the .lnt...")
        written = np.concatenate([sel for _entry, sel in selections] or [np.empty(0, np.int64)])
        xyz = np.column_stack([x, y, np.asarray(las.z, dtype=np.float64)])
        written_classes = lnt_classes_at(view.lnt_path, xyz, written, directory)
        del xyz
        legacy_bits = (las.point_format.id < 6 and written_classes.size
                       and int(written_classes.max()) > 31)
        class_of = {}
        start = 0
        for entry, sel in selections:
            class_of[id(entry)] = written_classes[start:start + sel.size]
            start += sel.size

        for number, (entry, sel) in enumerate(selections, 1):
            block, polygon = entry["block"], entry["polygon"]
            if sel.size == 0:
                skipped += 1
                continue
            self._set_progress(number / len(selections), f"Writing block {number} of {len(selections)}...")

            output_path = output_dir / (self._block_file_stem(block) + self.output_ext)
            header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
            header.scales = src_header.scales
            header.offsets = src_header.offsets
            for vlr in src_header.vlrs:
                if getattr(vlr, "user_id", "") == "NakshaAI" and getattr(vlr, "record_id", 0) == 1002:
                    continue                       # the source's own drawings; this block's go below
                header.vlrs.append(vlr)
            drawing_data = self._serialize_block_drawings(block)
            if drawing_data:
                header.vlrs.append(laspy.VLR(
                    user_id="NakshaAI",
                    record_id=1002,
                    description="Digitized drawings (lines, polygons, annotations)",
                    record_data=drawing_data,
                ))
            points = las.points[sel]
            classes = class_of[id(entry)]
            if legacy_bits:
                # Legacy point formats store 5 class bits; keep the high bits in
                # the flag bits exactly as the regular LAS save does.
                points.classification = classes & 0x1F
                points.synthetic = ((classes >> 5) & 1).astype(bool)
                points.key_point = ((classes >> 6) & 1).astype(bool)
                points.withheld = ((classes >> 7) & 1).astype(bool)
            else:
                points.classification = classes
            with laspy.open(str(output_path), mode="w", header=header) as writer:
                writer.write_points(points)
            assigned_count += int(sel.size)
            saved += 1
            saved_blocks.append({
                "block": block,
                "file_name": output_path.name,
                "label": output_path.stem,
                "polygon": polygon,
                "point_count": int(sel.size),
            })
            self._report_block_written(output_path, block, int(sel.size))

        print(
            f"[BlockSplit] summary files={saved} skipped={skipped} "
            f"blocks={len(self.blocks)} points={assigned_count}"
        )
        self._report_unassigned_points(x.shape[0] - assigned_count, x.shape[0])
        return saved, skipped, saved_blocks

    def _write_block_files(self, output_dir):
        # An open .lnt owns the full cloud; app.data is then empty or only a
        # section's partial subset, so it must not be what gets split.
        view = self._lnt_view()
        if view is not None:
            return self._write_block_files_from_lnt(output_dir, view)
        data = getattr(self.app, "data", None)
        if not isinstance(data, dict) or data.get("xyz") is None:
            prj_sources = self._prj_source_files()
            if prj_sources:
                self._report_missing_prj_sources(len(prj_sources))
                return self._write_block_files_from_prj(output_dir, prj_sources)
            raise RuntimeError(
                "No point cloud data is loaded, and no LAZ/LAS files could be "
                "found from the PRJ next to the loaded SNT."
            )

        xyz = np.asarray(data["xyz"])
        if xyz.ndim != 2 or xyz.shape[1] < 3 or xyz.shape[0] == 0:
            raise RuntimeError("Loaded point cloud data is empty or invalid.")
        # Which cloud is being split decides whether the output can be complete:
        # a loaded cloud may be one file, or a decimated display copy, so say so.
        print(
            f"[BlockSplit] points source = in-memory loaded cloud "
            f"points={int(xyz.shape[0]):,}"
        )

        assigned = np.zeros(xyz.shape[0], dtype=bool)
        saved = 0
        entries = _block_entries(self.blocks)
        skipped = len(self.blocks) - len(entries)
        saved_blocks = []
        selections = self._select_in_arrays(
            np.ascontiguousarray(xyz[:, 0], dtype=np.float64),
            np.ascontiguousarray(xyz[:, 1], dtype=np.float64),
            entries, "Assigning points to blocks...",
        )

        for number, (entry, sel) in enumerate(selections, 1):
            block, polygon = entry["block"], entry["polygon"]
            if sel.size == 0:
                skipped += 1
                continue
            self._set_progress(number / len(selections), f"Writing block {number} of {len(selections)}...")
            mask = np.zeros(xyz.shape[0], dtype=bool)
            mask[sel] = True

            output_path = output_dir / (self._block_file_stem(block) + self.output_ext)
            self._write_las_subset(output_path, data, mask, block)
            assigned[mask] = True
            saved += 1
            saved_blocks.append({
                "block": block,
                "file_name": output_path.name,
                "label": output_path.stem,
                "polygon": polygon,
                "point_count": int(np.count_nonzero(mask)),
            })
            self._report_block_written(output_path, block, int(np.count_nonzero(mask)))

        print(
            f"[BlockSplit] summary files={saved} skipped={skipped} "
            f"blocks={len(self.blocks)} points={int(np.count_nonzero(assigned))}"
        )
        self._report_unassigned_points(int(xyz.shape[0]) - int(np.count_nonzero(assigned)), xyz.shape[0])
        return saved, skipped, saved_blocks

    @staticmethod
    def _report_block_written(output_path, block, point_count):
        """Per block: how many points it took, and with which geometry/CL.

        ``mask=merged`` means the post-merge outline decided the points (the
        right answer for a block that absorbed a sliver); ``mask=outline`` means
        its own stored polygon did, which is how a dropped merge shows up.
        """
        merged_rings = _effective_block_rings(block)
        carried = [
            chain for chain in (block.get("absorbed_cl_chains") or [])
            if len(chain) >= 2
        ]
        own_cl = 1 if len(block.get("cl_pts") or []) >= 2 else 0
        print(
            f"[BlockSplit] file={Path(output_path).name} points={int(point_count)} "
            f"mask={'merged' if merged_rings else 'outline'} "
            f"rings={len(merged_rings) or 1} clChains={own_cl + len(carried)}"
            + (f" absorbedCl={len(carried)}" if carried else "")
        )

    @staticmethod
    def _report_unassigned_points(unassigned, total):
        """Say so when points land in no block, instead of dropping them silently.

        After a sliver is absorbed the copied points must still be covered by the
        parent's merged rings; if they are not, they would vanish from every
        output file without a word.
        """
        total = int(total)
        if total <= 0:
            return
        unassigned = int(unassigned)
        if unassigned <= 0:
            print(f"[BlockSplit] all {total} loaded points assigned to a block")
            return
        print(
            f"[BlockSplit] WARNING {unassigned}/{total} points "
            f"({100.0 * unassigned / total:.3f}%) lie outside every block outline "
            f"and were not written (the blocks only cover the corridor around the "
            f"CL lines, so points beyond its width belong to no block)"
        )

    def _prj_source_files(self):
        """LAZ/LAS files listed in the PRJ that sits next to each loaded SNT."""
        try:
            from gui.snt_attachment import _parse_snt_adjacent_prj_blocks
        except Exception:
            return []
        files = []
        seen = set()
        missing = []
        for snt_file in self._loaded_snt_paths():
            for entry in _parse_snt_adjacent_prj_blocks(str(snt_file)) or []:
                file_path = entry.get("file_path")
                if not file_path:
                    # Listed in the project but not resolvable on disk.  Blocks over
                    # its footprint can never receive points, so remember it.
                    name = entry.get("block_file") or entry.get("label")
                    if name:
                        missing.append(str(name))
                    continue
                key = os.path.normcase(os.path.abspath(file_path))
                if key in seen:
                    continue
                seen.add(key)
                files.append(Path(file_path))
        self._missing_prj_sources = missing
        return files

    def _report_missing_prj_sources(self, split_count):
        """Say which project point clouds the split cannot read, and why it matters.

        A PRJ commonly lists a whole tiling while only part of it is present.  The
        split reads what exists, so every created block outside those files gets
        no points and is skipped - which looks like the split gave up.
        """
        missing = list(getattr(self, "_missing_prj_sources", []) or [])
        if not missing:
            return
        listed = split_count + len(missing)
        preview = ", ".join(missing[:6])
        if len(missing) > 6:
            preview += f", ... (+{len(missing) - 6} more)"
        print(
            f"[BlockSplit] WARNING only {split_count} of {listed} point cloud(s) "
            f"listed in the PRJ are on disk. The missing {len(missing)} CANNOT be "
            f"split, so any block over their footprint will get no points: {preview}"
        )

    def _write_block_files_from_prj(self, output_dir, source_files, chunk_size=2_000_000):
        """Split PRJ-listed LAZ/LAS files one by one, streaming points into per-block outputs."""
        import laspy

        blocks = _block_entries(self.blocks)
        for entry in blocks:
            entry.update({
                "path": output_dir / (self._block_file_stem(entry["block"]) + self.output_ext),
                "writer": None,
                "count": 0,
            })
        skipped = len(self.blocks) - len(blocks)

        def open_writer(entry, src_header):
            header = laspy.LasHeader(
                point_format=src_header.point_format, version=src_header.version
            )
            header.scales = src_header.scales
            header.offsets = src_header.offsets
            for vlr in src_header.vlrs:
                try:
                    header.vlrs.append(vlr)
                except Exception:
                    pass
            drawing_data = self._serialize_block_drawings(entry["block"])
            if drawing_data:
                header.vlrs.append(laspy.VLR(
                    user_id="NakshaAI",
                    record_id=1002,
                    description="Digitized drawings (lines, polygons, annotations)",
                    record_data=drawing_data,
                ))
            entry["header"] = header
            return laspy.open(str(entry["path"]), mode="w", header=header)

        total_seen = 0
        total_assigned = 0
        source_bounds = {}
        empty_blocks = []
        print(
            f"[BlockSplit] points source = {len(source_files)} LAZ/LAS file(s) "
            f"listed in the PRJ next to the loaded SNT"
        )
        total_points = 0
        for source in source_files:
            try:
                with laspy.open(str(source)) as reader:
                    total_points += int(reader.header.point_count)
            except Exception:
                pass
        try:
            for file_index, source in enumerate(source_files):
                file_seen = 0
                file_assigned = 0
                with laspy.open(str(source)) as reader:
                    src_header = reader.header
                    source_bounds[source.name] = (
                        np.asarray(src_header.mins, dtype=float)[:2],
                        np.asarray(src_header.maxs, dtype=float)[:2],
                    )
                    if not _entries_touching(blocks, *source_bounds[source.name]):
                        # No block over this tile: none of its points can be
                        # written, so it is not decompressed at all.
                        file_seen = int(src_header.point_count)
                        total_seen += file_seen
                        self._set_progress(total_seen / max(total_points, 1),
                                           f"Skipped {source.name} (no block over it)")
                        print(f"[BlockSplit] source={source.name} points={file_seen:,} "
                              f"skipped (outside every block)")
                        continue
                    for chunk in _read_ahead(reader.chunk_iterator(chunk_size)):
                        x = np.asarray(chunk.x)
                        y = np.asarray(chunk.y)
                        total_seen += int(x.shape[0])
                        file_seen += int(x.shape[0])
                        self._set_progress(
                            total_seen / max(total_points, 1),
                            f"Splitting {source.name} ({file_index + 1} of {len(source_files)})",
                        )
                        for entry, sel in _select_chunk(x, y, blocks):
                            file_assigned += int(sel.size)
                            total_assigned += int(sel.size)
                            if entry["writer"] is None:
                                entry["writer"] = open_writer(entry, src_header)
                            subset = chunk[sel]
                            out_header = entry["header"]
                            if (
                                subset.point_format.id != out_header.point_format.id
                                or not np.allclose(src_header.scales, out_header.scales)
                                or not np.allclose(src_header.offsets, out_header.offsets)
                            ):
                                converted = laspy.ScaleAwarePointRecord.zeros(
                                    sel.size, header=out_header
                                )
                                converted.copy_fields_from(subset)
                                converted.x = x[sel]
                                converted.y = y[sel]
                                converted.z = np.asarray(chunk.z)[sel]
                                subset = converted
                            entry["writer"].write_points(subset)
                            entry["count"] += int(sel.size)
                # Per source file: proves every input was read whole and how much
                # of it the blocks claimed, instead of one opaque grand total.
                print(
                    f"[BlockSplit] source={source.name} points={file_seen:,} "
                    f"assigned={file_assigned:,} "
                    f"outsideBlocks={file_seen - file_assigned:,} "
                    f"({100.0 * (file_seen - file_assigned) / max(file_seen, 1):.3f}%)"
                )
        finally:
            for entry in blocks:
                if entry["writer"] is not None:
                    try:
                        entry["writer"].close()
                    except Exception as exc:
                        print(f"Failed to close split file {entry['path']}: {exc}")

        self._report_unassigned_points(total_seen - total_assigned, total_seen)
        self._report_unsplit_siblings(source_files)

        saved = 0
        saved_blocks = []
        for entry in blocks:
            if entry["count"] == 0:
                skipped += 1
                empty_blocks.append(entry)
                continue
            saved += 1
            self._report_block_written(entry["path"], entry["block"], entry["count"])
            saved_blocks.append({
                "block": entry["block"],
                "file_name": entry["path"].name,
                "label": entry["path"].stem,
                "polygon": entry["polygon"],
                "point_count": entry["count"],
            })
        self._report_empty_blocks(empty_blocks, source_bounds, len(blocks))
        print(
            f"[BlockSplit] summary files={saved} skipped={skipped} "
            f"blocks={len(self.blocks)} points={total_assigned}"
        )
        return saved, skipped, saved_blocks

    @staticmethod
    def _report_empty_blocks(entries, source_bounds, total_blocks):
        """Say which blocks received no points, and whether data even exists there.

        Distinguishes the two very different causes: a block outside every source
        file's footprint (no points exist to take) versus one that overlaps a
        source file and still came back empty (a block-outline problem).
        """
        if not entries:
            return
        outside = 0
        overlapping = []
        for entry in entries:
            lo, hi = entry["min"], entry["max"]
            hits = [
                name for name, (slo, shi) in source_bounds.items()
                if hi[0] >= slo[0] and lo[0] <= shi[0]
                and hi[1] >= slo[1] and lo[1] <= shi[1]
            ]
            if hits:
                overlapping.append((entry["path"].name, hits))
            else:
                outside += 1
        print(
            f"[BlockSplit] {len(entries)} of {total_blocks} block(s) got no points: "
            f"{outside} lie outside every split point cloud, "
            f"{len(overlapping)} overlap a cloud"
        )
        for name, hits in overlapping[:8]:
            print(f"[BlockSplit]   block={name} points=0 but overlaps "
                  f"{', '.join(hits[:2])} - check the block outline")
        if len(overlapping) > 8:
            print(f"[BlockSplit]   ... {len(overlapping) - 8} more overlapping but empty")

    @staticmethod
    def _report_unsplit_siblings(source_files):
        """Name point clouds that sit beside the sources but are not split.

        The split reads what the project PRJ lists.  Any other LAZ/LAS in the same
        folder (a ``_deleted_001``/``_backup`` copy, for instance) is therefore
        left untouched, which otherwise looks like the split stopped early.
        """
        try:
            import laspy
        except Exception:
            return
        folders = {Path(str(path)).parent for path in source_files}
        referenced = {
            os.path.normcase(os.path.abspath(str(path))) for path in source_files
        }
        extra = []
        for folder in folders:
            for pattern in ("*.laz", "*.las"):
                for candidate in sorted(folder.glob(pattern)):
                    key = os.path.normcase(os.path.abspath(str(candidate)))
                    if key in referenced:
                        continue
                    try:
                        with laspy.open(str(candidate)) as reader:
                            count = int(reader.header.point_count)
                    except Exception:
                        count = 0
                    extra.append((candidate.name, count))
        if not extra:
            return
        listed = ", ".join(f"{name} ({count:,} points)" for name, count in extra)
        print(
            f"[BlockSplit] NOTE {len(extra)} point cloud(s) in the source folder are "
            f"NOT listed in the PRJ and were not split: {listed}"
        )

    def _write_prj_manifest(self, output_path, saved_blocks):
        distance = 0.0
        for item in saved_blocks:
            try:
                distance = max(distance, float(item["block"].get("distance", 0.0) or 0.0))
            except Exception:
                pass

        block_size = int(round(distance)) if distance > 0 else 0
        lines = [
            "[TerraScan project]",
            "Scanner=AirborneLidar",
            f"Storage={self.output_ext.lstrip('.').upper()}1.2",
            "StoreTime=1",
            "StoreColor=0",
            "StoreEchoLen=0",
            "StoreParam=0",
            "StoreReflectance=0",
            "StoreDeviation=0",
            "StoreReliability=0",
            "StoreDistance=0",
            "StoreGroup=0",
            "StoreNormal=0",
            "StoreImage=0",
            "RequireLock=0",
            "Description=",
            "FirstPointId=1",
            "BlockRounded=1",
            f"BlockSize={block_size}",
            "BlockGroupCount=1000000",
            "BlockNaming=0",
            "BlockPrefix=pt",
            "",
        ]

        for index, item in enumerate(saved_blocks, start=1):
            lines.append(f"Block {item['file_name']}")
            lines.append(f"GroupFirst={index * 1000000}")
            lines.append("GroupCount=1000000")
            rings = item.get("export_rings") or [self._closed_polygon_xy(item["polygon"])]
            for x, y in rings[0]:
                lines.append(f" {x:.4f} {y:.4f}")
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")

    def _write_snt_manifest(self, output_path, saved_blocks):
        import shutil
        import tempfile

        try:
            import ezdxf
            import snt_core
        except Exception as exc:
            raise RuntimeError(f"SNT export backend is unavailable: {exc}") from exc

        temp_root = Path(tempfile.mkdtemp(prefix="naksha_split_snt_"))
        temp_dxf = temp_root / f"{output_path.stem}.dxf"
        temp_snt = temp_root / f"{output_path.stem}.snt"

        try:
            doc = ezdxf.new("R2010")
            for layer_name, color in (("BL", 4), ("CL", 1), ("FileNames", 2)):
                if not doc.layers.has_entry(layer_name):
                    layer = doc.layers.new(layer_name)
                else:
                    layer = doc.layers.get(layer_name)
                layer.color = color

            msp = doc.modelspace()
            for item in saved_blocks:
                cl_points = self._polyline_xyz(item["block"].get("cl_pts"))
                if len(cl_points) >= 2:
                    msp.add_polyline3d(
                        cl_points,
                        dxfattribs={"layer": "CL", "color": 1},
                    )

                # CL chains of the side blocks this block absorbed: their points
                # went into this block, so their CL belongs in its export too.
                for carried_chain in item["block"].get("absorbed_cl_chains") or []:
                    carried_points = self._polyline_xyz(carried_chain)
                    if len(carried_points) >= 2:
                        msp.add_polyline3d(
                            carried_points,
                            dxfattribs={"layer": "CL", "color": 1},
                        )

                bl_line_chains = self._open_block_line_chains(
                    self._block_bl_line_chains(item)
                )
                rings = item.get("export_rings") or [self._closed_polygon_xy(item["polygon"])]
                rings = [ring for ring in rings if len(ring) >= 4]
                if not bl_line_chains and not rings:
                    continue
                if bl_line_chains:
                    for chain in bl_line_chains:
                        msp.add_polyline3d(
                            chain,
                            dxfattribs={"layer": "BL", "color": 4},
                        )
                # Always write the closed ring so the SNT block-click PIP test works.
                # bl_line_chains are open polylines (visual only); the closed lwpolyline
                # is what build_snt_block_polygons needs to form a hit-testable polygon.
                for polygon in rings:
                    msp.add_lwpolyline(
                        polygon[:-1],
                        format="xy",
                        close=True,
                        dxfattribs={"layer": "BL", "color": 4},
                    )

                label_pos = item.get("label_position") or self._block_label_position(item["block"], item["polygon"])
                label_polygon = rings[0] if rings else self._closed_polygon_xy(item["polygon"])
                label_height = self._block_label_height(label_polygon, item["label"])
                text = msp.add_text(
                    item["label"],
                    dxfattribs={
                        "layer": "FileNames",
                        "color": 2,
                        "height": label_height,
                    },
                )
                try:
                    from ezdxf.enums import TextEntityAlignment
                    text.set_placement(label_pos, align=TextEntityAlignment.MIDDLE_CENTER)
                except Exception:
                    text.dxf.insert = label_pos

            doc.saveas(temp_dxf)

            result = None
            try:
                result = snt_core.convert(
                    dxf_path=temp_dxf,
                    snt_path=temp_snt,
                    description=output_path.stem,
                )
            except TypeError as exc:
                if "snt_path" not in str(exc):
                    raise
                result = snt_core.convert(
                    dxf_path=temp_dxf,
                    description=output_path.stem,
                )

            produced = Path(getattr(result, "snt_path", temp_snt))
            if not produced.exists():
                fallback = temp_dxf.with_suffix(".snt")
                if fallback.exists():
                    produced = fallback
            if not produced.exists():
                raise RuntimeError(f"SNT conversion did not produce output file: {produced}")

            shutil.copy2(produced, output_path)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def _prepare_manifest_blocks(self, saved_blocks):
        try:
            from shapely.geometry import Polygon
            from shapely.ops import unary_union
        except Exception:
            return [
                {
                    **item,
                    "export_rings": [self._closed_polygon_xy(item["polygon"])],
                    "label_position": self._block_label_position(item["block"], item["polygon"]),
                }
                for item in saved_blocks
            ]

        prepared = []
        higher_priority_geoms = []

        for item in saved_blocks:
            original_ring = self._closed_polygon_xy(item["polygon"])
            effective_rings = _effective_block_rings(item.get("block"))
            try:
                if effective_rings:
                    # A small intersection sliver was absorbed into this block, so its
                    # stored rings are already the final topology: export them as-is and
                    # clip later blocks against them (no sliver is written out twice).
                    rings = [list(ring) for ring in effective_rings]
                    label_position = self._label_position_from_geometry(Polygon(rings[0]), item)
                    prepared.append({
                        **item,
                        "export_rings": rings,
                        "label_position": label_position,
                    })
                    merged_geom = unary_union([Polygon(ring) for ring in rings])
                    if not merged_geom.is_empty:
                        higher_priority_geoms.append(merged_geom)
                    continue

                geom = Polygon(original_ring)
                if not geom.is_valid:
                    geom = geom.buffer(0)

                if higher_priority_geoms and not geom.is_empty:
                    # Only reshape the exported ring when it actually needs to be
                    # clipped against a higher-priority block. Reshaping unconditionally
                    # (e.g. shapely's buffer(0) repair of a self-intersecting ring from a
                    # sharp CL bend) can produce a polygon that no longer matches the
                    # bl_line_chains boundary already drawn on screen, so the reloaded
                    # SNT shows both the real boundary and a stray shapely-repaired sliver.
                    export_geom = geom.difference(unary_union(higher_priority_geoms))
                    if export_geom.is_empty:
                        export_geom = geom
                    rings = self._rings_from_geometry(export_geom)
                    label_position = self._label_position_from_geometry(export_geom, item)
                else:
                    rings = [original_ring]
                    label_position = self._block_label_position(item["block"], item["polygon"])

                if not rings:
                    rings = [original_ring]

                prepared.append({
                    **item,
                    "export_rings": rings,
                    "label_position": label_position,
                })

                if not geom.is_empty:
                    higher_priority_geoms.append(geom)
            except Exception:
                prepared.append({
                    **item,
                    "export_rings": [original_ring],
                    "label_position": self._block_label_position(item["block"], item["polygon"]),
                })

        return prepared

    @classmethod
    def _rings_from_geometry(cls, geom):
        if geom is None or geom.is_empty:
            return []

        geoms = []
        geom_type = getattr(geom, "geom_type", "")
        if geom_type == "Polygon":
            geoms = [geom]
        elif geom_type == "MultiPolygon":
            geoms = list(geom.geoms)
        else:
            return []

        rings = []
        for poly in geoms:
            if poly.is_empty or float(poly.area) <= 1e-9:
                continue
            ring = [(float(x), float(y)) for x, y in list(poly.exterior.coords)]
            if len(ring) >= 4:
                rings.append(ring)
        rings.sort(key=cls._ring_area_abs, reverse=True)
        return rings

    @staticmethod
    def _ring_area_abs(ring):
        area = 0.0
        pts = list(ring or [])
        for idx in range(len(pts) - 1):
            area += pts[idx][0] * pts[idx + 1][1] - pts[idx + 1][0] * pts[idx][1]
        return abs(area) * 0.5

    def _label_position_from_geometry(self, geom, item):
        z = self._block_label_position(item["block"], item["polygon"])[2]
        try:
            if geom is None or geom.is_empty:
                raise ValueError("empty geometry")
            point = geom.centroid
            if not geom.contains(point):
                point = geom.representative_point()
            return (float(point.x), float(point.y), float(z))
        except Exception:
            return self._block_label_position(item["block"], item["polygon"])

    @staticmethod
    def _closed_polygon_xy(polygon):
        pts = []
        for point in np.asarray(polygon, dtype=np.float64):
            if len(point) < 2:
                continue
            pts.append((float(point[0]), float(point[1])))
        if pts and (abs(pts[0][0] - pts[-1][0]) > 1e-9 or abs(pts[0][1] - pts[-1][1]) > 1e-9):
            pts.append(pts[0])
        return pts

    @staticmethod
    def _polyline_xyz(points):
        out = []
        for point in list(points or []):
            try:
                if point is None or len(point) < 2:
                    continue
                x = float(point[0])
                y = float(point[1])
                z = float(point[2]) if len(point) > 2 else 0.0
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    out.append((x, y, z))
            except Exception:
                continue
        return out

    @classmethod
    def _block_bl_line_chains(cls, item):
        chains = []
        block = item.get("block") if isinstance(item, dict) else None
        for drawing in (block or {}).get("drawings") or []:
            chain = cls._polyline_xyz(drawing.get("coords") or drawing.get("points"))
            if len(chain) >= 2:
                chains.append(chain)
        return chains

    @staticmethod
    def _open_block_line_chains(chains):
        """Cut self-closing drawn lines into open chains - geometry unchanged.

        A block whose whole boundary survives the merge ends up with ONE closed
        loop of drawn lines (its neighbours did not take any of its edges).  The
        SNT loader reads a closed polyline sitting on a block-boundary layer as a
        block outline and hides it from the line pass, so such a block would
        export - and reload - with no visible block lines at all.  Splitting the
        loop in two leaves every coordinate in place but makes both halves open
        chains, which are drawn as ordinary lines.
        """
        out = []
        for chain in chains or []:
            points = list(chain)
            if len(points) < 3:
                if len(points) >= 2:
                    out.append(points)
                continue
            first, last = points[0], points[-1]
            closes = (
                abs(float(first[0]) - float(last[0])) <= 1e-3
                and abs(float(first[1]) - float(last[1])) <= 1e-3
            )
            if not closes:
                out.append(points)
                continue
            loop = points[:-1]
            if len(loop) < 3:
                out.append(points)
                continue
            middle = len(loop) // 2
            out.append(loop[: middle + 1])
            out.append(loop[middle:] + [loop[0]])
        return out

    @staticmethod
    def _block_label_position(block, polygon):
        above = list(block.get("above_pts") or [])
        below = list(block.get("below_pts") or [])
        if above and below:
            idx = min(len(above), len(below)) // 2
            idx = max(0, min(idx, len(above) - 1, len(below) - 1))
            a = above[idx]
            b = below[idx]
            z = ((float(a[2]) if len(a) > 2 else 0.0) + (float(b[2]) if len(b) > 2 else 0.0)) * 0.5
            return ((float(a[0]) + float(b[0])) * 0.5, (float(a[1]) + float(b[1])) * 0.5, z)

        pts = np.asarray(polygon, dtype=np.float64)
        return (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])), 0.0)

    @staticmethod
    def _block_label_height(polygon, label):
        pts = np.asarray(polygon, dtype=np.float64)
        span_x = float(np.max(pts[:, 0]) - np.min(pts[:, 0])) if pts.size else 0.0
        span_y = float(np.max(pts[:, 1]) - np.min(pts[:, 1])) if pts.size else 0.0
        block_span = max(span_x, span_y, 1.0)
        by_name = block_span / max(len(str(label or "")) * 0.75, 1.0)
        return max(1.0, min(25.0, by_name))

    def _write_las_subset(self, output_path, data, mask, block):
        import laspy
        from gui.save_pointcloud import (
            _choose_point_format,
            _intensity_to_uint16,
            _rgb_to_las16,
        )

        xyz = np.asarray(data["xyz"])[mask]
        n_points = xyz.shape[0]

        classes = data.get("classification")
        if classes is None or np.asarray(classes).shape[0] != mask.shape[0]:
            classes = np.zeros(n_points, dtype=np.uint8)
        else:
            classes = np.asarray(classes)[mask].astype(np.uint8, copy=False)

        rgb = data.get("rgb")
        rgb_subset = None
        if rgb is not None and np.asarray(rgb).shape[0] == mask.shape[0]:
            rgb_subset = np.asarray(rgb)[mask]

        intensity = data.get("intensity")
        intensity_subset = None
        if intensity is not None and np.asarray(intensity).shape[0] == mask.shape[0]:
            intensity_subset = np.asarray(intensity)[mask]

        las_version = self._infer_las_version(data, classes)
        major, minor = map(int, las_version.split("."))
        rgb16 = _rgb_to_las16(rgb_subset)
        point_format_id = _choose_point_format(las_version, has_rgb=(rgb16 is not None))
        header = laspy.LasHeader(point_format=point_format_id, version=f"{major}.{minor}")

        try:
            import pyproj
            if getattr(self.app, "project_crs_wkt", None):
                header.add_crs(pyproj.CRS.from_wkt(self.app.project_crs_wkt))
            elif getattr(self.app, "project_crs_epsg", None):
                header.add_crs(pyproj.CRS.from_epsg(self.app.project_crs_epsg))
        except Exception as exc:
            print(f"Failed to embed split-file CRS: {exc}")

        las = laspy.LasData(header)
        las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        las.classification = classes
        if rgb16 is not None:
            las.red, las.green, las.blue = rgb16[:, 0], rgb16[:, 1], rgb16[:, 2]
        intensity16 = _intensity_to_uint16(intensity_subset, n_points=n_points)
        if intensity16 is not None:
            las.intensity = intensity16

        drawing_data = self._serialize_block_drawings(block)
        if drawing_data:
            las.vlrs.append(laspy.VLR(
                user_id="NakshaAI",
                record_id=1002,
                description="Digitized drawings (lines, polygons, annotations)",
                record_data=drawing_data,
            ))

        las.write(str(output_path))

    @staticmethod
    def _infer_las_version(data, classes):
        version = data.get("input_format_version") or data.get("las_version") or data.get("version")
        if isinstance(version, tuple) and len(version) >= 2:
            las_version = f"{int(version[0])}.{int(version[1])}"
        else:
            las_version = str(version or "1.4")
        if las_version not in ("1.2", "1.4"):
            las_version = "1.4"
        max_cls = int(np.asarray(classes).max()) if np.asarray(classes).size else 0
        if max_cls > 31 and las_version == "1.2":
            las_version = "1.4"
        return las_version

    @staticmethod
    def _serialize_block_drawings(block):
        serializable = []
        for drawing in block.get("drawings", []):
            coords = drawing.get("coords")
            if coords is None:
                coords = drawing.get("points")
            if coords is None:
                continue
            try:
                points = np.asarray(coords, dtype=float).tolist()
            except Exception:
                continue
            if not points:
                continue
            serializable.append({
                "type": drawing.get("type", "smartline"),
                "points": points,
                "color": [float(c) for c in drawing.get("original_color", (0.0, 0.7, 1.0))],
                "width": float(drawing.get("original_width", 3)),
            })
        if not serializable:
            return b""
        return json.dumps(serializable, separators=(",", ":")).encode("utf-8")


class BlockEditRefreshMixin:
    """Turn the (possibly edited) block outline drawings back into block
    polygons for Split.  Shared by the VTK and the native Vulkan dialogs."""

    @staticmethod
    def _clean_block_edit_chain(points):
        cleaned = []
        for point in list(points or []):
            try:
                xyz = (
                    float(point[0]),
                    float(point[1]),
                    float(point[2]) if len(point) > 2 else 0.0,
                )
            except (TypeError, ValueError, IndexError):
                continue
            if not cleaned or np.hypot(xyz[0] - cleaned[-1][0], xyz[1] - cleaned[-1][1]) > 1e-9:
                cleaned.append(xyz)
        return cleaned

    @staticmethod
    def _same_block_vertex(first, second, tolerance=1e-6):
        return np.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1])) <= tolerance

    def _normalised_block_drawings(self, drawings):
        """Return copied drawings with duplicated block corners made coherent."""
        normalised = []
        junctions = {}
        for drawing in list(drawings or []):
            copied = dict(drawing)
            copied["coords"] = self._clean_block_edit_chain(drawing.get("coords"))
            normalised.append(copied)

            original = drawing.get("block_creation_original_coords") or []
            vertex_groups = drawing.get("block_creation_vertex_groups") or []
            for vertex_index, junction_id in enumerate(vertex_groups):
                if (
                    junction_id is None
                    or vertex_index >= len(copied["coords"])
                    or vertex_index >= len(original)
                ):
                    continue
                junctions.setdefault(junction_id, []).append(
                    (copied, vertex_index, original[vertex_index])
                )

        reconciled = []
        for junction_id, members in junctions.items():
            moved = [
                copied["coords"][vertex_index]
                for copied, vertex_index, original in members
                if not self._same_block_vertex(copied["coords"][vertex_index], original)
            ]
            target = moved[0] if moved else members[0][0]["coords"][members[0][1]]
            if any(not self._same_block_vertex(point, target, tolerance=0.02) for point in moved[1:]):
                # Two edits dragged the same shared corner to different spots.
                # Stitch at the average of the moved members instead of
                # rejecting the whole split: the block still closes and every
                # edit stays approximately honoured.  (Previously this flagged
                # the block invalid and blocked Split entirely.)
                target = tuple(
                    float(np.mean([point[axis] for point in moved]))
                    for axis in range(3)
                )
                reconciled.append(junction_id)
            for copied, vertex_index, _original in members:
                copied["coords"][vertex_index] = tuple(target)

        # Also remove tiny gaps created by independent edits at two endpoints.
        endpoints = []
        for copied in normalised:
            coords = copied["coords"]
            if len(coords) < 2:
                continue
            endpoints.append((copied, 0))
            endpoints.append((copied, len(coords) - 1))

        visited = set()
        for start_index, (copied, vertex_index) in enumerate(endpoints):
            if start_index in visited:
                continue
            cluster = [start_index]
            visited.add(start_index)
            queue = [start_index]
            while queue:
                current_index = queue.pop()
                current_drawing, current_vertex = endpoints[current_index]
                current_point = current_drawing["coords"][current_vertex]
                for candidate_index, (candidate_drawing, candidate_vertex) in enumerate(endpoints):
                    if candidate_index in visited:
                        continue
                    candidate_point = candidate_drawing["coords"][candidate_vertex]
                    if self._same_block_vertex(current_point, candidate_point, tolerance=0.02):
                        visited.add(candidate_index)
                        queue.append(candidate_index)
                        cluster.append(candidate_index)
            if len(cluster) < 2:
                continue
            points = [endpoints[index][0]["coords"][endpoints[index][1]] for index in cluster]
            target = tuple(float(np.mean([point[axis] for point in points])) for axis in range(3))
            for endpoint_index in cluster:
                endpoint_drawing, endpoint_vertex = endpoints[endpoint_index]
                endpoint_drawing["coords"][endpoint_vertex] = target

        return normalised, reconciled

    @classmethod
    def _map_original_to_current(cls, original, current, tolerance=1e-6):
        """Pair pre-edit vertices with their edited counterparts.

        Extend / vertex-delete edits insert or drop points, which used to make
        the whole drawing unmappable (strict length equality) so its edits were
        invisible to Split and the block was rejected as "not closed".  An extra
        current point is treated as an insertion and a missing one as a
        deletion; everything else pairs in order so a moved vertex still maps
        back onto its original ring position.
        """
        pairs = []
        i = j = 0
        while i < len(original) and j < len(current):
            original_point = original[i]
            current_point = current[j]
            if cls._same_block_vertex(original_point, current_point, tolerance):
                pairs.append((original_point, current_point))
                i += 1
                j += 1
            elif j + 1 < len(current) and cls._same_block_vertex(
                original_point, current[j + 1], tolerance
            ):
                j += 1  # a vertex was inserted at j
            elif i + 1 < len(original) and cls._same_block_vertex(
                original[i + 1], current_point, tolerance
            ):
                i += 1  # a vertex was deleted from the original
            else:
                pairs.append((original_point, current_point))
                i += 1
                j += 1
        while i < len(original) and j < len(current):
            pairs.append((original[i], current[j]))
            i += 1
            j += 1
        return pairs

    def _cached_block_polygon(self, block):
        try:
            raw_polygon = np.asarray(block.get("polygon"), dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if raw_polygon.ndim != 2 or raw_polygon.shape[0] < 3:
            return None

        ring = []
        for point in raw_polygon:
            xy = (float(point[0]), float(point[1]))
            if not ring or not self._same_block_vertex(xy, ring[-1]):
                ring.append(xy)
        if len(ring) < 3:
            return None
        if not self._same_block_vertex(ring[0], ring[-1]):
            ring.append(ring[0])
        return np.asarray(ring, dtype=np.float64)

    def _patched_cached_block_polygon(self, block, drawings):
        """Apply edited preview vertices to the original full block ring."""
        cached = self._cached_block_polygon(block)
        if cached is None:
            return None
        ring = np.array(cached[:-1], dtype=np.float64, copy=True)
        changed = False

        for drawing in list(drawings or []):
            original = drawing.get("block_creation_original_coords") or []
            current = self._clean_block_edit_chain(drawing.get("coords"))
            for original_point, current_point in self._map_original_to_current(
                original, current
            ):
                if self._same_block_vertex(original_point, current_point):
                    continue
                distances = np.hypot(ring[:, 0] - original_point[0], ring[:, 1] - original_point[1])
                for ring_index in np.flatnonzero(distances <= 1e-5):
                    ring[ring_index] = (float(current_point[0]), float(current_point[1]))
                    changed = True

        if not changed:
            return None
        patched = np.vstack((ring, ring[0]))
        try:
            from shapely.geometry import Polygon

            polygon = Polygon(patched)
            if polygon.is_empty or not polygon.is_valid or float(polygon.area) <= 1e-9:
                return None
        except Exception:
            return None
        return patched

    def _block_drawings_changed(self, block, live_drawings):
        original_drawings = list(block.get("drawings", []) or [])
        live_ids = {id(drawing) for drawing in live_drawings}
        if any(id(drawing) not in live_ids for drawing in original_drawings):
            return True

        for drawing in live_drawings:
            original = drawing.get("block_creation_original_coords")
            if original is None:
                continue
            current = self._clean_block_edit_chain(drawing.get("coords"))
            if len(current) != len(original):
                return True
            if any(
                not self._same_block_vertex(current_point, original_point)
                for current_point, original_point in zip(current, original)
            ):
                return True
        return False

    def _polygon_from_block_drawings(self, drawings):
        chains = []
        for drawing in list(drawings or []):
            coords = self._clean_block_edit_chain(
                drawing.get("coords") if isinstance(drawing, dict) else None
            )
            if len(coords) >= 2:
                chains.append(coords)

        if not chains:
            return None

        try:
            from shapely.geometry import LineString, Polygon
            from shapely.ops import polygonize, unary_union
        except Exception:
            return None

        try:
            line_geoms = []
            for chain in chains:
                xy = [(float(p[0]), float(p[1])) for p in chain]
                if len(xy) >= 2:
                    line_geoms.append(LineString(xy))
            if not line_geoms:
                return None

            merged = unary_union(line_geoms)
            polygons = [poly for poly in polygonize(merged) if not poly.is_empty and float(poly.area) > 1e-9]

            if not polygons:
                for chain in chains:
                    xy = [(float(p[0]), float(p[1])) for p in chain]
                    if len(xy) >= 4 and abs(xy[0][0] - xy[-1][0]) <= 1e-9 and abs(xy[0][1] - xy[-1][1]) <= 1e-9:
                        poly = Polygon(xy)
                        if not poly.is_empty and float(poly.area) > 1e-9:
                            polygons.append(poly)

            if not polygons:
                return None

            poly = max(polygons, key=lambda item: float(item.area))
            ring = [(float(x), float(y)) for x, y in list(poly.exterior.coords)]
            return np.array(ring, dtype=np.float64) if len(ring) >= 4 else None
        except Exception as exc:
            print(f"Block edit polygon rebuild failed: {exc}")
            return None

    def _refresh_blocks_from_live_drawings(self, current_blocks, live_ids):
        """(refreshed blocks, [(priority, reason)] rejected) for the blocks whose
        outline drawings are still in the view (``live_ids``: ids of drawings)."""
        refreshed = []
        invalid_blocks = []

        for block in current_blocks:
            priority = int(block.get("priority", len(refreshed) + 1))
            original_drawings = list(block.get("drawings", []) or [])
            live_drawings = [
                drawing for drawing in original_drawings
                if id(drawing) in live_ids
            ]
            if not live_drawings:
                invalid_blocks.append(
                    (priority, "none of its cyan edges remain in the view - undo the deletion or rebuild the preview")
                )
                continue

            has_deleted_drawing = len(live_drawings) != len(original_drawings)
            changed = self._block_drawings_changed(block, live_drawings)
            normalised_drawings, reconciled = self._normalised_block_drawings(live_drawings)
            if reconciled:
                print(
                    f"[Split] block priority {priority}: stitched shared corner(s) "
                    f"{', '.join(reconciled)} that separate edits had pulled apart"
                )

            if not changed:
                # Overlap clipping deliberately leaves some visible edge groups open.
                # Their cached block polygon is still the authoritative valid shape.
                polygon = self._cached_block_polygon(block)
            else:
                # A vertex move normally maps directly back to the full original ring.
                # This keeps clipped helper edges from turning a valid block into an open one.
                polygon = None if has_deleted_drawing else self._patched_cached_block_polygon(
                    block, normalised_drawings
                )
                if polygon is None:
                    polygon = self._polygon_from_block_drawings(normalised_drawings)
                if polygon is None:
                    # The edits could not be mapped onto a closed outline (a
                    # deleted edge, clipped helper edges, an ambiguous shape).
                    # The cached ring still describes this block, so proceed
                    # with it rather than blocking the entire Split.
                    polygon = self._cached_block_polygon(block)
                    if polygon is not None:
                        print(
                            f"[Split] block priority {priority}: edited edges do not close "
                            "on their own - proceeding with the last valid outline"
                        )
            if polygon is None or len(polygon) < 4:
                invalid_blocks.append(
                    (priority, "its edited edges do not close into a polygon - fix or undo that edit")
                )
                continue

            updated = dict(block)
            updated["drawings"] = live_drawings
            updated["polygon"] = polygon
            # Vertex edits invalidate the small-fragment merge FOR THAT BLOCK: its
            # merged rings no longer describe these drawings, so export is left to
            # re-polygonize.  An untouched block keeps its merged rings - dropping
            # them here would undo the merge on every plain Create -> Split, and the
            # points of an absorbed sliver would then belong to no block at all.
            updated["effective_polygon"] = (
                None if changed else block.get("effective_polygon")
            )
            updated["above_pts"] = []
            updated["below_pts"] = []
            refreshed.append(updated)

        return refreshed, invalid_blocks
