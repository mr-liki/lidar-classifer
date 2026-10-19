import os
import json
import weakref
import shutil
import tempfile
import numpy as np
import laspy
from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtCore import Qt

# LAS versions this app can write. LAS 1.0 is written through the identical
# 1.1 header layout with the version bytes stamped back afterwards, because
# laspy can read a 1.0 header but refuses to create one.
LAS_VERSIONS = ("1.0", "1.1", "1.2", "1.3", "1.4")

_LAS_VERSION_RANK = {version: rank for rank, version in enumerate(LAS_VERSIONS)}
_LAS_VERSION_BYTE_OFFSET = 24  # header version major byte; minor follows


def _version_rank(las_version: str) -> int:
    return _LAS_VERSION_RANK.get(str(las_version or "").strip(), len(LAS_VERSIONS) - 1)


def _version_supports_rgb(las_version: str) -> bool:
    """RGB channels (point format 2) only exist from LAS 1.2 onwards."""
    return _version_rank(las_version) >= _LAS_VERSION_RANK["1.2"]


def _las_version_of(las) -> str:
    version = las.header.version
    return f"{int(version.major)}.{int(version.minor)}"


def _writable_header_version(las_version: str) -> str:
    """Header version laspy accepts; LAS 1.0 borrows the 1.1 layout."""
    text = str(las_version or "").strip()
    return "1.1" if text == "1.0" else text


def _parse_filter(selected_filter: str):
    """Map a 'LAZ 1.2 (*.laz)' style filter to (extension, LAS version)."""
    sf = (selected_filter or "").strip().lower()
    if "laz" in sf:
        ext = ".laz"
    elif "las" in sf:
        ext = ".las"
    else:
        ext = ".laz"
    for version in LAS_VERSIONS:
        if version in sf:
            return ext, version
    return ext, "1.4"

# Display Mode class table columns: Show, Code, Description, Draw, Lvl, Color, Weight.
_CLASS_TABLE_COLOR_COLUMN = 5


def _table_row_color(table, row):
    """RGB tuple from a Display Mode class row's Color column, else None."""
    try:
        item = table.item(row, _CLASS_TABLE_COLOR_COLUMN)
        if item is None:
            return None
        brush = item.background()
        if brush.style() == Qt.NoBrush:
            # Never colored: leave the row unswatched instead of drawing black.
            return None
        qcolor = brush.color()
        if not qcolor.isValid():
            return None
        return tuple(int(channel) for channel in qcolor.getRgb()[:3])
    except Exception:
        return None


def _unique_point_source_ids(source_ids):
    """Sorted unique Point Source IDs in one linear pass.

    ``np.unique`` sorts every point (about a second on tens of millions of
    them). Point Source IDs are unsigned 16-bit, so a bincount returns the same
    sorted list far cheaper; anything unexpected falls back to ``np.unique``.
    """
    arr = np.asarray(source_ids)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        return []
    try:
        if np.issubdtype(arr.dtype, np.integer):
            counts = np.bincount(arr)
            return [int(value) for value in np.nonzero(counts)[0]]
    except Exception:
        pass
    return [int(value) for value in np.unique(arr)]


def _remember_line_ids(app, source_ids, line_ids):
    """Cache a scan for this exact array (weak ref: holds no extra memory)."""
    try:
        app._save_points_line_ids_cache = (weakref.ref(source_ids), list(line_ids))
    except TypeError:
        # Arrays are weak-referenceable; a plain sequence is not, so hold it.
        app._save_points_line_ids_cache = (source_ids, list(line_ids))


def _cached_line_ids(app, source_ids):
    """Previously scanned ids for this array, or None when not scanned yet."""
    cache = getattr(app, "_save_points_line_ids_cache", None)
    if not cache:
        return None
    reference, line_ids = cache
    try:
        stored = reference()
    except TypeError:
        stored = reference
    return list(line_ids) if stored is source_ids else None


def _flight_line_color(line_id):
    """Same stable high-contrast RGB Display Mode's Line rendering uses."""
    lid = int(line_id)
    return (
        (lid * 67 + 53) % 256,
        (lid * 131 + 97) % 256,
        (lid * 193 + 181) % 256,
    )


def _line_items_with_colors(app, line_ids):
    """(line_id, "Line N", (r, g, b)) entries with the Line swatch colors."""
    colors = dict(getattr(app, "flight_line_colors", {}) or {})
    items = []
    for line_id in sorted(int(value) for value in line_ids):
        color = colors.get(line_id)
        if color is None:
            color = _flight_line_color(line_id)
        items.append((line_id, f"Line {line_id}", tuple(color)))
    return items


def _loaded_las_path(app):
    """Existing LAS/LAZ file behind the loaded dataset, or None."""
    for candidate in (
        getattr(app, "loaded_file", None),
        getattr(app, "last_save_path", None),
    ):
        if (
            candidate
            and str(candidate).lower().endswith((".las", ".laz"))
            and os.path.exists(str(candidate))
        ):
            return str(candidate)
    return None


def _remember_no_line_file(app, path, mtime, n_points):
    """Remember that this file has no usable ids for this dataset."""
    try:
        app._save_points_no_line_file = (path, mtime, n_points)
    except Exception:
        pass


def _recover_point_source_array(app, data):
    """Point Source ID column read straight from the loaded LAS/LAZ file.

    Legacy load paths leave it out of ``data``; the array is installed only
    when its length matches the in-memory xyz, so a filtered import can never
    be paired with unfiltered ids. A negative result is remembered per file
    (path + mtime + point count) so the file is opened at most once until it
    or the dataset changes. Display Mode's list is never consulted.
    """
    try:
        if not isinstance(data, dict):
            return None
        existing = data.get("point_source_id")
        if existing is not None:
            return existing
        xyz = data.get("xyz")
        if xyz is None:
            return None
        path = _loaded_las_path(app)
        if path is None:
            return None
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None
        blocked = getattr(app, "_save_points_no_line_file", None)
        if blocked == (path, mtime, len(xyz)):
            return None
        chunks = []
        with laspy.open(path) as reader:
            dims = {
                str(d).lower()
                for d in reader.header.point_format.dimension_names
            }
            if "point_source_id" not in dims:
                _remember_no_line_file(app, path, mtime, len(xyz))
                return None
            for points in reader.chunk_iterator(2_000_000):
                chunks.append(
                    np.asarray(points.point_source_id, dtype=np.uint16).copy()
                )
        recovered = (
            np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint16)
        )
        if len(recovered) != len(xyz):
            print(
                "Save points: flight-line recovery skipped, file/data lengths "
                f"differ ({len(recovered):,} != {len(xyz):,})"
            )
            _remember_no_line_file(app, path, mtime, len(xyz))
            return None
        data["point_source_id"] = recovered
        return recovered
    except Exception as exc:
        print(f"Save points: could not read flight lines from the file: {exc}")
        return None


def _available_line_items(app, data=None):
    """Flight lines for Save points, read straight from the loaded dataset.

    The source is the Point Source ID array in the loaded data (the same
    values Display Mode's Lines button reads), recovered from the loaded
    LAS/LAZ file when a legacy load path omitted it. Display Mode's computed
    list is never consulted, so Save points does not depend on that window.
    Scans are cached per array; the popup stays independent of the dataset
    size because the work is deferred (see _line_items_plan).
    """
    try:
        source_ids = None
        if isinstance(data, dict):
            source_ids = data.get("point_source_id")
        if source_ids is None:
            source_ids = _recover_point_source_array(app, data)
            if source_ids is None:
                return []
        try:
            if not len(source_ids):
                return []
        except TypeError:
            return []
        line_ids = _cached_line_ids(app, source_ids)
        if line_ids is None:
            line_ids = _unique_point_source_ids(source_ids)
            _remember_line_ids(app, source_ids, line_ids)
        return _line_items_with_colors(app, line_ids)
    except Exception as exc:
        print(f"Save points: could not read flight lines: {exc}")
        return []


def _line_items_plan(app, data):
    """(items, provider) for the Line field, without walking every point.

    Items come back directly when the Point Source IDs are already in the
    loaded data (the loader stores them, or an earlier scan cached them).
    When a legacy load path left them out, a provider is returned that reads
    the column straight from the loaded LAS/LAZ file the first time the Line
    list is opened: the same source Display Mode's Lines button uses, but
    never that window's computed list, so opening Save points stays
    independent of the dataset size.
    """
    source_ids = None
    if isinstance(data, dict):
        source_ids = data.get("point_source_id")
    if source_ids is None:
        if _loaded_las_path(app) is None:
            # No readable LAS/LAZ file behind this dataset: keep it disabled.
            return [], None
        return None, (lambda: _available_line_items(app, data))
    try:
        if not len(source_ids):
            return [], None
    except TypeError:
        return [], None
    cached = _cached_line_ids(app, source_ids)
    if cached is not None:
        return _line_items_with_colors(app, cached), None
    return None, (lambda: _available_line_items(app, data))


def _format_filters() -> str:
    """QFileDialog filter list: every LAS/LAZ version, grouped by container."""
    return ";;".join(
        f"{container} {version} (*.{container.lower()})"
        for container in ("LAS", "LAZ")
        for version in LAS_VERSIONS
    )


def _ensure_ext(path: str, ext: str) -> str:
    root, cur_ext = os.path.splitext(path)
    if cur_ext.lower() != ext.lower():
        return root + ext
    return path

def _rgb_to_las16(rgb_arr):
    """Convert RGB to LAS 16-bit (0..65535). Accepts float(0..1) / uint8 / uint16."""
    if rgb_arr is None:
        return None
    rgb = np.asarray(rgb_arr)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        return None

    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb, 0.0, 1.0)
        return (rgb * 65535.0).round().astype(np.uint16)

    mx = int(rgb.max()) if rgb.size else 0
    if mx <= 255:
        return (rgb.astype(np.uint16) * 256)
    return np.clip(rgb, 0, 65535).astype(np.uint16)

def _intensity_to_uint16(intensity_arr, n_points: int):
    """LAS intensity is uint16. If intensity is missing, return None."""
    if intensity_arr is None:
        return None
    inten = np.asarray(intensity_arr)
    if inten.shape[0] != n_points:
        return None

    if np.issubdtype(inten.dtype, np.floating):
        mx = float(np.nanmax(inten)) if inten.size else 0.0
        if mx <= 1.0:
            inten = np.clip(inten, 0.0, 1.0) * 65535.0
        inten = np.clip(inten, 0.0, 65535.0)
        return inten.round().astype(np.uint16)

    return np.clip(inten, 0, 65535).astype(np.uint16)

def _choose_point_format(las_version: str, has_rgb: bool) -> int:
    """
    Smallest point format valid for the chosen LAS version:
      - LAS 1.4: 6 (no RGB) or 7 (RGB)
      - LAS 1.2/1.3: 0 (no RGB) or 2 (RGB)
      - LAS 1.0/1.1: 0 - point format 2 (RGB) arrived with LAS 1.2, so those
        versions fall back to format 0 and the caller drops the colour data.
    """
    if _version_rank(las_version) == _LAS_VERSION_RANK["1.4"]:
        return 7 if has_rgb else 6
    if has_rgb and _version_supports_rgb(las_version):
        return 2
    return 0


def _apply_rgb(las, rgb16, las_version: str) -> bool:
    """Write RGB into ``las`` when the target version carries colour.

    Point format 2 (the first format with colour) arrived with LAS 1.2, so
    1.0/1.1 targets keep the geometry and drop the colour with a notice.
    """
    if rgb16 is None:
        return False
    if not _version_supports_rgb(las_version):
        print(
            f"⚠️ RGB dropped: LAS {las_version} has no colour channels "
            "(colour needs LAS 1.2 or newer)"
        )
        return False
    las.red, las.green, las.blue = rgb16[:, 0], rgb16[:, 1], rgb16[:, 2]
    return True


def _import_options_reduce_points(import_options) -> bool:
    if not isinstance(import_options, dict):
        return False
    if bool(import_options.get("only_class")) and bool(import_options.get("class_codes")):
        return True
    if bool(import_options.get("only_every")) and int(import_options.get("nth_point", 1) or 1) > 1:
        return True
    return False


def _same_path(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    try:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
    except Exception:
        return os.path.normcase(a) == os.path.normcase(b)


def _assign_classification_to_las(las, classes_u8: np.ndarray) -> None:
    version = las.header.version
    las_version = f"{int(version.major)}.{int(version.minor)}"

    if las_version == "1.4":
        las.classification = np.clip(classes_u8, 0, 255).astype(np.uint8, copy=False)
        return

    max_cls = int(classes_u8.max()) if classes_u8.size else 0
    if max_cls > 31:
        las.classification = (classes_u8 & 0x1F).astype(np.uint8, copy=False)
        las.synthetic = ((classes_u8 >> 5) & 1).astype(bool)
        las.key_point = ((classes_u8 >> 6) & 1).astype(bool)
        las.withheld = ((classes_u8 >> 7) & 1).astype(bool)
        return

    las.classification = np.clip(classes_u8, 0, 31).astype(np.uint8, copy=False)


# "Attributes to save" label -> LAS dimensions cleared in the written file
# when the label is unchecked.  Easting/Northing/Elevation are deliberately
# absent: a LAS point cannot exist without coordinates, and the dialog marks
# those rows always available.  "Description" follows "Class" - it is derived
# from the classification code rather than stored per point.  "Time" and
# "Date" share gps_time and are cleared only when neither is chosen.  "Echo"
# also clears Number of Returns because both describe the return pattern.
# A dimension with no row here - scan angle, shown as "Angle" in View
# Fields - keeps its source values.
_ATTRIBUTE_WRITE_DIMS = {
    "Class": ("classification",),
    "Line": ("point_source_id",),
    "Echo": ("return_number", "number_of_returns"),
    "Amplitude": ("amplitude",),
    "Intensity": ("intensity",),
    "Reflectance": ("reflectance",),
    "Deviation": ("deviation",),
    "Reliability": ("reliability",),
    "Color RGB": ("red", "green", "blue"),
}


def _attr_selected(selection, label):
    """True when the label is checked, or when no Attributes choice was made."""
    return selection is None or label in selection


def _apply_attribute_selection(las, selection):
    """Clear every LAS dimension the user unchecked in "Attributes to save".

    Runs on both write paths - the source-preserving rewrite (which copies
    every source field) and the synthesized save - just before the file is
    written, so unchecked attributes carry no values in the saved file.
    Coordinates are never touched, and neither is a dimension with no row
    in the list (scan angle).  Returns the cleared labels for the log.
    """
    if selection is None or las is None:
        return []
    checked = set(selection)
    dim_names = [str(d) for d in las.point_format.dimension_names]
    dims = {name.casefold() for name in dim_names}
    cleared = set()

    def clear(names, label):
        for name in names:
            if name.casefold() not in dims:
                continue
            try:
                current = np.asarray(getattr(las, name))
                setattr(las, name, np.zeros_like(current))
                cleared.add(label)
            except Exception as exc:
                print(f"⚠️ Could not clear {label} ({name}): {exc}")

    for label, names in _ATTRIBUTE_WRITE_DIMS.items():
        if label not in checked:
            clear(names, label)

    if "Image" not in checked:
        # Vendor extra-byte dimension ("imageid", "photo", ...) discovered
        # exactly the way View Fields discovers it.
        try:
            from gui.dialogs.view_fields_dialog import find_image_dimension_name

            image_name = find_image_dimension_name(dim_names)
        except Exception:
            image_name = None
        if image_name is not None:
            clear((image_name,), "Image")
    if "Time" not in checked and "Date" not in checked:
        # One gps_time field backs both the Time and the Date column.
        clear(("gps_time",), "Time/Date")
    if cleared:
        print(
            "Save points: cleared unchecked attribute(s): "
            + ", ".join(sorted(cleared))
        )
    return sorted(cleared)


def _try_build_preserved_las(
    *,
    source_path: str | None,
    target_path: str,
    xyz,
    classes_u8,
    rgb16,
    intensity16,
    import_options=None,
    requested_version: str | None = None,
    crs_wkt: str | None = None,
    crs_epsg=None,
):
    if not source_path:
        return None, "no source file available"
    if not os.path.exists(source_path):
        return None, f"source file missing: {source_path}"

    source_ext = os.path.splitext(source_path)[1].lower()
    target_ext = os.path.splitext(target_path)[1].lower()
    if source_ext not in (".las", ".laz"):
        return None, f"source is not LAS/LAZ: {source_ext}"
    if target_ext != source_ext:
        return None, f"target format {target_ext} differs from loaded source {source_ext}"
    if _import_options_reduce_points(import_options):
        return None, "loaded dataset is filtered/sampled and no longer matches the full source file"

    try:
        las = laspy.read(source_path)
    except Exception as exc:
        return None, f"could not read source for preserved save: {exc}"

    source_points = int(len(las.points))
    current_points = int(len(np.asarray(xyz)))
    if source_points != current_points:
        return None, (
            f"point-count mismatch between source ({source_points:,}) and loaded data "
            f"({current_points:,})"
        )

    source_version = f"{int(las.header.version.major)}.{int(las.header.version.minor)}"
    if requested_version and requested_version != source_version:
        return None, f"requested LAS version {requested_version} differs from source {source_version}"

    dims = {str(d).lower() for d in las.point_format.dimension_names}
    if rgb16 is not None and not {"red", "green", "blue"}.issubset(dims):
        return None, "loaded RGB data cannot be preserved because source point format has no RGB channels"
    if intensity16 is not None and "intensity" not in dims:
        return None, "loaded intensity data cannot be preserved because source point format has no intensity channel"

    xyz = np.asarray(xyz)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    _assign_classification_to_las(las, np.asarray(classes_u8, dtype=np.uint8, copy=False))

    if rgb16 is not None and {"red", "green", "blue"}.issubset(dims):
        las.red = rgb16[:, 0]
        las.green = rgb16[:, 1]
        las.blue = rgb16[:, 2]
    if intensity16 is not None and "intensity" in dims:
        las.intensity = intensity16

    try:
        import pyproj
        if crs_wkt:
            las.header.add_crs(pyproj.CRS.from_wkt(crs_wkt))
        elif crs_epsg:
            las.header.add_crs(pyproj.CRS.from_epsg(crs_epsg))
    except Exception as exc:
        print(f"⚠️ Preserved-save CRS update skipped: {exc}")

    return las, None


def _empty_i64():
    return np.empty(0, dtype=np.int64)


def _empty_i32():
    return np.empty(0, dtype=np.int32)


def _as_i64(values):
    if values is None:
        return _empty_i64()
    return np.asarray(values, dtype=np.int64)


def _as_i32(values):
    if values is None:
        return _empty_i32()
    return np.asarray(values, dtype=np.int32)


def _inspect_fenced_parent_session(app):
    data = getattr(app, "data", None)
    if not isinstance(data, dict):
        return None, None, False

    session_id = data.get("_fence_session_id")
    if not session_id:
        return None, None, False

    session = getattr(app, "_fence_parent_session", None)
    if not isinstance(session, dict) or session.get("session_id") != session_id:
        return None, (
            "The fenced source-file session is no longer available.\n"
            "Reload the fence before using Save."
        ), True

    xyz = data.get("xyz")
    if xyz is None:
        return None, "No fenced data is currently loaded.", True
    point_count = int(len(np.asarray(xyz)))

    source_files = [str(p) for p in list(session.get("source_files") or []) if str(p)]
    if not source_files:
        return None, "This fenced load has no parent-file list attached.", True

    source_file_ids = data.get("_fence_source_file_ids")
    source_point_indices = data.get("_fence_source_point_indices")
    if source_file_ids is None or source_point_indices is None:
        return None, (
            "The fenced point-source mapping is missing.\n"
            "Reload the fence before using Save."
        ), True

    source_file_ids = _as_i32(source_file_ids)
    source_point_indices = _as_i64(source_point_indices)
    if source_file_ids.shape[0] != point_count or source_point_indices.shape[0] != point_count:
        return None, (
            "The fenced dataset no longer matches its source-point map.\n"
            "Use Save As for the fenced subset, or reload the fence before using Save."
        ), True

    if np.any(source_file_ids < 0) or np.any(source_file_ids >= len(source_files)):
        return None, "The fenced source-file IDs are out of range.", True

    duplicate_owner_indices = _as_i64(session.get("duplicate_owner_indices"))
    duplicate_source_file_ids = _as_i32(session.get("duplicate_source_file_ids"))
    duplicate_source_point_indices = _as_i64(session.get("duplicate_source_point_indices"))

    if not (
        duplicate_owner_indices.shape[0] ==
        duplicate_source_file_ids.shape[0] ==
        duplicate_source_point_indices.shape[0]
    ):
        return None, "The fenced duplicate-point mapping is inconsistent.", True

    if duplicate_owner_indices.size:
        if np.any(duplicate_owner_indices < 0) or np.any(duplicate_owner_indices >= point_count):
            return None, "The fenced duplicate-point owners are out of range.", True
        if np.any(duplicate_source_file_ids < 0) or np.any(duplicate_source_file_ids >= len(source_files)):
            return None, "The fenced duplicate-point file IDs are out of range.", True

    return {
        "session_id": session_id,
        "point_count": point_count,
        "source_files": source_files,
        "operation": str(session.get("operation") or "fence"),
        "primary_grid": session.get("primary_grid"),
        "buffer_width": session.get("buffer_width"),
        "source_file_ids": source_file_ids,
        "source_point_indices": source_point_indices,
        "duplicate_owner_indices": duplicate_owner_indices,
        "duplicate_source_file_ids": duplicate_source_file_ids,
        "duplicate_source_point_indices": duplicate_source_point_indices,
    }, None, True


def has_fenced_parent_writeback(app) -> bool:
    session, _error, is_fence_mode = _inspect_fenced_parent_session(app)
    # Ordinary Load-by-Fence sessions are subset-only and must never trigger
    # an implicit write into their complete source tiles. Buffered-grid loads
    # retain their explicit multi-parent ownership workflow.
    return bool(
        is_fence_mode
        and session is not None
        and session.get("operation") == "buffered_grid"
    )


def save_current_pointcloud_in_place(app) -> bool:
    """Silently save the currently editable dataset to its real source."""
    from .native_save import native_active, save_native
    if native_active(app):
        return save_native(app)
    data = getattr(app, "data", None)
    if not isinstance(data, dict) or data.get("xyz") is None:
        return False

    session, error, is_fence_mode = _inspect_fenced_parent_session(app)
    if is_fence_mode:
        if session is None:
            if error:
                print(f"Fence in-place save skipped: {error}")
            return False
        if session.get("operation") != "buffered_grid":
            # A fenced subset has no safe single in-place destination. Require
            # an explicit subset Save As instead of modifying full parents.
            return False
        return _save_fenced_parent_files(
            app, session, output_dir=None, in_place=True, show_messages=False,
        )

    path = getattr(app, "last_save_path", None) or getattr(app, "loaded_file", None)
    return save_pointcloud_quick(app, path) if path else False


def _prompt_fenced_save_mode(app, session):
    if session.get("operation") != "buffered_grid":
        return "subset"

    source_count = len(session.get("source_files") or [])
    first_source = os.path.basename(session["source_files"][0]) if source_count else "parent file"

    box = QMessageBox(app)
    box.setIcon(QMessageBox.Question)
    box.setWindowTitle("Save Buffered Grid")
    box.setText("Choose how to save the buffered grid edits.")
    box.setInformativeText(
        f"Source grid files: {source_count}\n"
        f"Primary grid: {session.get('primary_grid') or first_source}\n"
        f"Buffer width: {float(session.get('buffer_width') or 0.0):.2f}\n\n"
        "Original source files: writes each edited point back to its own LAS/LAZ file.\n"
        "Full parent copies: creates separately saved complete tile copies.\n"
        "Buffered subset only: saves all displayed points into one combined file."
    )
    subset_btn = box.addButton(
        "Buffered Subset Only",
        QMessageBox.AcceptRole,
    )
    parents_btn = box.addButton("Full Parent Files", QMessageBox.ActionRole)
    originals_btn = (
        box.addButton("Save to Original Source Files", QMessageBox.DestructiveRole)
        if session.get("operation") == "buffered_grid" else None
    )
    cancel_btn = box.addButton(QMessageBox.Cancel)
    box.exec()

    clicked = box.clickedButton()
    if clicked == parents_btn:
        return "parents"
    if originals_btn is not None and clicked == originals_btn:
        return "in_place"
    if clicked == subset_btn:
        return "subset"
    if clicked == cancel_btn:
        return None
    return None


def _copy_prj_sidecar(app, source_path: str, target_path: str) -> None:
    src_prj = os.path.splitext(source_path)[0] + ".prj"
    dst_prj = os.path.splitext(target_path)[0] + ".prj"
    try:
        if os.path.exists(src_prj):
            shutil.copyfile(src_prj, dst_prj)
            return
        if getattr(app, "project_crs_wkt", None):
            with open(dst_prj, "w") as fh:
                fh.write(app.project_crs_wkt)
        elif getattr(app, "project_crs_epsg", None):
            with open(dst_prj, "w") as fh:
                fh.write(f"EPSG:{app.project_crs_epsg}")
    except Exception as exc:
        print(f"⚠️ Failed to copy/write .prj for {os.path.basename(target_path)}: {exc}")


def _parent_upgrade_point_format(las_obj) -> int:
    dims = {str(d).lower() for d in las_obj.point_format.dimension_names}
    has_rgb = {"red", "green", "blue"}.issubset(dims)
    has_waveform = any(
        name in dims for name in (
            "wavepacket_index",
            "byte_offset_to_waveform_data",
            "waveform_packet_size_in_bytes",
            "return_point_waveform_location",
        )
    )
    if has_waveform:
        return 10 if has_rgb else 9
    return 7 if has_rgb else 6


def _build_fenced_parent_assignments(app, session):
    data = getattr(app, "data", {}) or {}
    point_count = int(session["point_count"])
    if point_count <= 0:
        return {}, None, None

    positions = np.arange(point_count, dtype=np.int64)
    source_files = list(session["source_files"])
    source_file_ids = session["source_file_ids"]
    source_point_indices = session["source_point_indices"]
    duplicate_owner_indices = session["duplicate_owner_indices"]
    duplicate_source_file_ids = session["duplicate_source_file_ids"]
    duplicate_source_point_indices = session["duplicate_source_point_indices"]

    assignments = {}
    for file_id, source_path in enumerate(source_files):
        point_idx_parts = []
        data_idx_parts = []

        primary_mask = (source_file_ids == file_id)
        if np.any(primary_mask):
            point_idx_parts.append(source_point_indices[primary_mask])
            data_idx_parts.append(positions[primary_mask])

        dup_mask = (duplicate_source_file_ids == file_id)
        if np.any(dup_mask):
            point_idx_parts.append(duplicate_source_point_indices[dup_mask])
            data_idx_parts.append(duplicate_owner_indices[dup_mask])

        if not point_idx_parts:
            continue

        point_indices = np.concatenate(point_idx_parts).astype(np.int64, copy=False)
        data_indices = np.concatenate(data_idx_parts).astype(np.int64, copy=False)

        order = np.argsort(point_indices, kind="stable")
        point_indices = point_indices[order]
        data_indices = data_indices[order]

        if point_indices.size > 1:
            keep_mask = np.ones(point_indices.size, dtype=bool)
            keep_mask[:-1] = point_indices[:-1] != point_indices[1:]
            point_indices = point_indices[keep_mask]
            data_indices = data_indices[keep_mask]

        assignments[source_path] = {
            "point_indices": point_indices,
            "data_indices": data_indices,
        }

    rgb16 = _rgb_to_las16(data.get("rgb"))
    intensity16 = _intensity_to_uint16(data.get("intensity"), n_points=point_count)
    return assignments, rgb16, intensity16


def _save_fenced_parent_files(app, session, output_dir=None, in_place=True, show_messages=True):
    data = getattr(app, "data", None)
    if not isinstance(data, dict) or data.get("xyz") is None:
        if show_messages:
            QMessageBox.warning(app, "Save", "No fenced data is loaded.")
        return False

    classes = data.get("classification")
    if classes is None:
        classes = np.zeros(session["point_count"], dtype=np.uint8)
    classes_u8 = np.asarray(classes).astype(np.uint8, copy=False)

    assignments, rgb16, intensity16 = _build_fenced_parent_assignments(app, session)
    if not assignments:
        if show_messages:
            QMessageBox.information(app, "Save", "No parent-file updates were found for this fence selection.")
        return False

    if not in_place:
        if not output_dir:
            if show_messages:
                QMessageBox.information(app, "Save As", "No output folder was selected.")
            return False

        basenames = {}
        for source_path in assignments:
            base_key = os.path.basename(source_path).lower()
            prev = basenames.get(base_key)
            if prev is not None and os.path.normcase(prev) != os.path.normcase(source_path):
                if show_messages:
                    QMessageBox.critical(
                        app,
                        "Save As",
                        "Two different parent files share the same output name.\n"
                        "Choose another workflow or rename the source files first."
                    )
                return False
            basenames[base_key] = source_path

    written_paths = []
    updated_points_total = 0

    try:
        for source_path, assignment in assignments.items():
            point_indices = np.asarray(assignment["point_indices"], dtype=np.int64)
            data_indices = np.asarray(assignment["data_indices"], dtype=np.int64)
            if point_indices.size == 0:
                continue

            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Parent file not found: {source_path}")

            target_path = source_path if in_place else os.path.join(output_dir, os.path.basename(source_path))
            target_dir = os.path.dirname(target_path) or os.getcwd()
            os.makedirs(target_dir, exist_ok=True)

            las = laspy.read(source_path)
            total_points = int(len(las.points))
            if np.any(point_indices < 0) or np.any(point_indices >= total_points):
                raise IndexError(
                    f"Source-point index out of range for {os.path.basename(source_path)} "
                    f"(file points={total_points:,})."
                )

            source_version = f"{las.header.version.major}.{las.header.version.minor}"
            source_max_cls = int(classes_u8[data_indices].max()) if data_indices.size else 0
            if source_max_cls > 31 and source_version == "1.2":
                try:
                    if not hasattr(laspy, "convert"):
                        raise RuntimeError("laspy.convert is not available in this runtime")
                    las = laspy.convert(
                        las,
                        point_format_id=_parent_upgrade_point_format(las),
                        file_version="1.4",
                    )
                    print(
                        f"⚠️ Fence parent write-back auto-upgraded {os.path.basename(source_path)} "
                        f"to LAS 1.4 (classification max={source_max_cls})"
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not upgrade {os.path.basename(source_path)} to LAS 1.4 "
                        f"for classification values above 31: {exc}"
                    ) from exc

            las.classification[point_indices] = classes_u8[data_indices]

            dims = {str(d).lower() for d in las.point_format.dimension_names}
            if rgb16 is not None and {"red", "green", "blue"}.issubset(dims):
                las.red[point_indices] = rgb16[data_indices, 0]
                las.green[point_indices] = rgb16[data_indices, 1]
                las.blue[point_indices] = rgb16[data_indices, 2]
            if intensity16 is not None and "intensity" in dims:
                las.intensity[point_indices] = intensity16[data_indices]

            fd, tmp_path = tempfile.mkstemp(
                prefix=os.path.splitext(os.path.basename(target_path))[0] + "_tmp_",
                suffix=os.path.splitext(target_path)[1],
                dir=target_dir,
            )
            os.close(fd)
            try:
                _write_las_for_version(las, tmp_path)
                os.replace(tmp_path, target_path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

            if not in_place:
                _copy_prj_sidecar(app, source_path, target_path)

            written_paths.append(target_path)
            updated_points_total += int(point_indices.size)

        if not written_paths:
            if show_messages:
                QMessageBox.information(app, "Save", "No parent files were written.")
            return False

        setattr(app, "_last_fence_parent_write_paths", list(written_paths))

        if hasattr(app, "statusBar"):
            scope_text = "source files" if in_place else "full parent copies"
            app.statusBar().showMessage(
                f"Saved fence edits into {len(written_paths)} {scope_text}.",
                6000,
            )

        if show_messages:
            scope_text = "original parent files" if in_place else "full parent-file copies"
            QMessageBox.information(
                app,
                "Save",
                f"Saved fence edits into {len(written_paths)} {scope_text}.\n"
                f"Updated point records: {updated_points_total:,}"
            )

        print(
            f"💾 Fence parent write-back complete: files={len(written_paths)} "
            f"updated_points={updated_points_total:,} in_place={in_place}"
        )
        for path in written_paths:
            print(f"   ↳ {path}")
        return True

    except Exception as exc:
        print(f"❌ Fence parent write-back failed: {exc}")
        if show_messages:
            QMessageBox.critical(app, "Save", f"Fence parent write-back failed:\n\n{exc}")
        return False

def _serialize_drawings(app) -> bytes:
    """
    Serialize all drawing objects from app into JSON bytes.
    ✅ FIXED: Proper numpy array handling
    """
    try:
        # Get drawings from digitizer if available
        if hasattr(app, 'digitizer') and hasattr(app.digitizer, 'drawings'):
            drawings = app.digitizer.drawings
            print(f"📐 Found {len(drawings)} drawings in digitizer")
        else:
            drawings = getattr(app, "drawings", [])
        
        if not drawings:
            print("ℹ️ No drawings to serialize")
            return b""
        
        serializable = []
        for obj in drawings:
            # ✅ FIXED: Handle numpy arrays properly
            coords = None
            if 'coords' in obj:
                coords = obj['coords']
            elif 'points' in obj:
                coords = obj['points']
            
            # Check if coords is valid
            if coords is None:
                print(f"  ⚠️ Skipping drawing with no coordinates")
                continue
            
            # Check if it's an empty array/list
            try:
                if isinstance(coords, np.ndarray):
                    if coords.size == 0:
                        print(f"  ⚠️ Skipping drawing with empty array")
                        continue
                elif len(coords) == 0:
                    print(f"  ⚠️ Skipping drawing with empty list")
                    continue
            except Exception:
                print(f"  ⚠️ Skipping drawing with invalid coordinates")
                continue
            
            obj_copy = {
                'type': obj.get('type', 'line'),
                'points': []
            }
            
            # Convert coordinates safely
            try:
                if isinstance(coords, np.ndarray):
                    obj_copy['points'] = coords.tolist()
                elif isinstance(coords, list):
                    obj_copy['points'] = [[float(c) for c in pt] for pt in coords]
                else:
                    print(f"  ⚠️ Unknown coordinate format: {type(coords)}")
                    continue
            except Exception as e:
                print(f"  ⚠️ Failed to convert coordinates: {e}")
                continue
            
            # Copy metadata
            if 'text' in obj:
                obj_copy['text'] = str(obj['text'])
            
            # Handle color
            if 'original_color' in obj:
                obj_copy['color'] = [float(c) for c in obj['original_color']]
            elif 'color' in obj:
                color = obj['color']
                if isinstance(color, (list, tuple, np.ndarray)):
                    obj_copy['color'] = [float(c) for c in color]
            
            # Handle width
            if 'original_width' in obj:
                obj_copy['width'] = float(obj['original_width'])
            elif 'width' in obj:
                obj_copy['width'] = float(obj['width'])
            
            # Naksha SNT/SBM workflow metadata
            if 'layer' in obj:
                obj_copy['layer'] = str(obj['layer'])
            if 'snt_file' in obj:
                obj_copy['snt_file'] = str(obj['snt_file'])
            if 'snt_committed' in obj:
                obj_copy['snt_committed'] = bool(obj['snt_committed'])
            if 'snt_stored' in obj:
                obj_copy['snt_stored'] = bool(obj['snt_stored'])
            if 'sbm_class' in obj:
                obj_copy['sbm_class'] = obj['sbm_class']
            if 'source' in obj:
                obj_copy['source'] = str(obj['source'])
            
            serializable.append(obj_copy)
        
        if not serializable:
            print("ℹ️ No valid drawings after processing")
            return b""
        
        json_str = json.dumps(serializable, separators=(',', ':'))
        print(f"✅ Serialized {len(serializable)} drawing(s) to JSON ({len(json_str)} bytes)")
        return json_str.encode('utf-8')
    
    except Exception as e:
        print(f"⚠️ Failed to serialize drawings: {e}")
        import traceback
        traceback.print_exc()
        return b""


def _deserialize_drawings(data: bytes) -> list:
    """
    Deserialize drawing objects from JSON bytes.
    Returns list of drawing dictionaries.
    """
    try:
        if not data:
            return []
        json_str = data.decode('utf-8')
        drawings = json.loads(json_str)
        
        # Convert lists back to numpy arrays where needed
        for obj in drawings:
            if 'points' in obj and isinstance(obj['points'], list):
                obj['points'] = np.array(obj['points'])
            if 'properties' in obj and isinstance(obj['properties'], dict):
                for key, val in obj['properties'].items():
                    if isinstance(val, list) and key in ['color', 'vertices']:
                        obj['properties'][key] = np.array(val)      
        return drawings  
    except Exception as e:
        print(f"⚠️ Failed to deserialize drawings: {e}")
        return []

def _sidecar_drawings_path(las_path: str) -> str:
    root, _ = os.path.splitext(las_path)
    return root + ".naksha_drawings.json"


def _save_drawings_sidecar(las_path: str, drawing_data: bytes) -> str:
    """
    Persist oversized drawing payloads next to the LAS/LAZ file.
    """
    sidecar_path = _sidecar_drawings_path(las_path)
    payload = {
        "encoding": "utf-8",
        "drawings_json": drawing_data.decode("utf-8", errors="replace"),
    }
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return sidecar_path


def _remove_drawings_sidecar(las_path: str) -> None:
    sidecar_path = _sidecar_drawings_path(las_path)
    try:
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)
    except Exception:
        pass


def _remove_drawings_vlr(las) -> None:
    """Drop Naksha drawings VLR entries before writing."""
    las.vlrs = [
        v for v in getattr(las, "vlrs", [])
        if not (v.user_id.strip() == "NakshaAI" and int(v.record_id) == 1002)
    ]


def _attach_drawings_storage(las, las_path: str, drawing_data: bytes, description: str, *,
                             verbose: bool = True) -> str | None:
    """
    Store drawings in a normal VLR when they fit; otherwise save them to a
    JSON sidecar and ensure no oversize VLR remains attached to the LAS object.

    Returns the sidecar path when sidecar fallback is used, else None.
    """
    _remove_drawings_vlr(las)

    if not drawing_data:
        _remove_drawings_sidecar(las_path)
        return None

    max_vlr_len = np.iinfo("uint16").max
    if len(drawing_data) > max_vlr_len:
        sidecar_path = _save_drawings_sidecar(las_path, drawing_data)
        if verbose:
            print(f"📎 Drawing payload too large for VLR; saved sidecar instead → {sidecar_path}")
        return sidecar_path

    drawing_vlr = laspy.VLR(
        user_id="NakshaAI",
        record_id=1002,
        description=description,
        record_data=drawing_data,
    )
    las.vlrs.append(drawing_vlr)
    _remove_drawings_sidecar(las_path)
    return None


def _write_las_for_version(las, path: str, las_version: str | None = None) -> None:
    """Write ``las`` to ``path``, including LAS 1.0 targets.

    laspy reads a LAS 1.0 header but refuses to create one, and the 1.0/1.1
    headers share a layout (1.1 only pinned down the file-source-id and
    global-encoding fields, both written as 0 here). So a 1.0 target is
    written as 1.1 and its two version bytes are stamped to (1, 0).
    """
    target = str(las_version or _las_version_of(las)).strip()
    if target != "1.0":
        las.write(path)
        return

    if _las_version_of(las) == "1.0":
        # Object read from a 1.0 source: convert to the equivalent 1.1 first
        # (same point format, same dimensions) so laspy will write it.
        las = laspy.convert(las, file_version="1.1")
    las.write(path)
    with open(path, "r+b") as handle:
        handle.seek(_LAS_VERSION_BYTE_OFFSET)
        handle.write(bytes((1, 0)))
    print(f"📌 LAS 1.0 written over the identical 1.1 header layout: {os.path.basename(path)}")


def _atomic_write_las(las, path: str, las_version: str | None = None) -> None:
    """
    Write to a temp file in the same directory and replace the destination only
    after a successful full write so failures cannot truncate the live file.
    """
    out_dir = os.path.dirname(path) or "."
    suffix = os.path.splitext(path)[1] or ".laz"
    fd, tmp_path = tempfile.mkstemp(prefix=".naksha_save_", suffix=suffix, dir=out_dir)
    os.close(fd)
    try:
        _write_las_for_version(las, tmp_path, las_version)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise

def load_drawings_from_las(las_path: str, app):
    """
    Load digitized drawings from VLR in existing LAS/LAZ file.
    ✅ FIXED: Stores drawings temporarily, to be restored AFTER point cloud renders
    """
    try:
        with laspy.open(las_path) as reader:
            for vlr in reader.header.vlrs:
                if vlr.user_id.strip() == "NakshaAI" and int(vlr.record_id) == 1002:
                    print("📐 Found drawing data in LAS file")
                    drawings = _deserialize_drawings(vlr.record_data)
                    
                    if not hasattr(app, 'digitizer') or not hasattr(app.digitizer, 'drawings'):
                        print("⚠️ Digitizer not available, skipping drawing load")
                        return False
                    
                    # ✅ CRITICAL FIX: Store for later, don't add to renderer yet
                    app._pending_drawings_restore = drawings
                    print(f"💾 Stored {len(drawings)} drawings for post-render restoration")
                    
                    return True

        sidecar_path = _sidecar_drawings_path(las_path)
        if os.path.exists(sidecar_path):
            print(f"📐 Found drawing sidecar: {sidecar_path}")
            with open(sidecar_path, "r", encoding="utf-8") as f:
                sidecar = json.load(f)
            drawings_json = sidecar.get("drawings_json", "")
            drawings = _deserialize_drawings(drawings_json.encode("utf-8"))
            if drawings:
                if not hasattr(app, 'digitizer') or not hasattr(app.digitizer, 'drawings'):
                    print("⚠️ Digitizer not available, skipping drawing load")
                    return False
                app._pending_drawings_restore = drawings
                print(f"💾 Stored {len(drawings)} drawings from sidecar for post-render restoration")
                return True

        if hasattr(app, 'digitizer'):
            if not hasattr(app.digitizer, 'drawings'):
                app.digitizer.drawings = []
        return False
    except Exception as e:
        print(f"⚠️ Failed to load drawings from {las_path}: {e}")
        import traceback
        traceback.print_exc()
        
        if hasattr(app, 'digitizer'):
            if not hasattr(app.digitizer, 'drawings'):
                app.digitizer.drawings = []
        
        return False
    
def finalize_drawing_render(app):
    """
    ✅ Restore drawings AFTER point cloud is rendered - forces them ON TOP
    """
    if not hasattr(app, '_pending_drawings_restore'):
        return
    
    drawings = app._pending_drawings_restore
    delattr(app, '_pending_drawings_restore')
    
    if not drawings:
        return
    
    print(f"\n{'='*60}")
    print(f"🎨 RESTORING {len(drawings)} DRAWINGS ON TOP")
    print(f"{'='*60}")
    
    try:
        app.digitizer.drawings = []  # Clear existing
        renderer = app.vtk_widget.renderer
        
        for drawing_data in drawings:
            coords = drawing_data.get('points', [])
            
            if len(coords) == 0:
                continue
            
            dtype = drawing_data.get('type', 'line')
            color = drawing_data.get('color', (1, 0, 0))
            width = drawing_data.get('width', 2)
            
            # Create actor
            actor = None
            curve_tool = getattr(app, "curve_tool", None)
            if dtype == 'curve' and curve_tool is not None and hasattr(curve_tool, '_create_curve_actor'):
                try:
                    actor = curve_tool._create_curve_actor(coords, color=color, width=width)
                except Exception as e:
                    print(f"  ⚠️ Curve actor creation failed, falling back to polyline: {e}")
            
            if actor is None:
                actor = app.digitizer._make_polyline_actor(
                    coords,
                    color=color,
                    width=width
                )
            
            if actor:
                # ✅ Force rendering ON TOP with multiple strategies
                try:
                    # Strategy 1: Polygon offset (pulls toward camera)
                    mapper = actor.GetMapper()
                    if mapper:
                        mapper.SetResolveCoincidentTopologyToPolygonOffset()
                        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2.0, -2.0)
                    
                    # Strategy 2: Thick, bright lines
                    prop = actor.GetProperty()
                    prop.SetLineWidth(max(5, width * 2))
                    prop.SetOpacity(1.0)
                    
                    # Strategy 3: Disable lighting
                    prop.LightingOff()
                    
                    # Strategy 4: Render as tubes for visibility
                    prop.SetRenderLinesAsTubes(True)
                    
                except Exception as e:
                    print(f"  ⚠️ Render setup warning: {e}")
                
                # Add to renderer (will be added AFTER point cloud)
                renderer.AddActor(actor)
                
                entry = {
                    'type': dtype,
                    'coords': coords,
                    'actor': actor,
                    'bounds': actor.GetBounds(),
                    'original_color': color,
                    'original_width': width
                }
                
                if 'text' in drawing_data:
                    entry['text'] = drawing_data['text']
                
                # Restore SNT/SBM metadata if present
                for key in ('layer', 'snt_file', 'snt_committed', 'snt_stored', 'sbm_class', 'source'):
                    if key in drawing_data:
                        entry[key] = drawing_data[key]
                
                # If it's a curve, also register it in curve_tool
                if dtype == 'curve' and curve_tool is not None:
                    curve_data = {
                        'actor': actor,
                        'control_points': coords,
                        'interpolated': coords,
                        'coords': [tuple(pt) for pt in coords],
                        'type': 'curve',
                        'source': 'curve_tool',
                        'color': color,
                        'original_color': color,
                        'original_width': width,
                    }
                    # Copy other restored metadata to curve_data too
                    for key in ('layer', 'snt_file', 'snt_committed', 'snt_stored', 'sbm_class'):
                        if key in entry:
                            curve_data[key] = entry[key]
                    entry = curve_data
                    curve_tool.finalized_actors.append(curve_data)

                app.digitizer.drawings.append(entry)
        
        print(f"✅ {len(app.digitizer.drawings)} drawings restored ON TOP")
        
        # Reset clipping range
        renderer.ResetCameraClippingRange()
        
        # Force render
        renderer.GetRenderWindow().Render()
        
        print(f"{'='*60}\n")       
    except Exception as e:
        print(f"❌ Drawing restoration failed: {e}")
        import traceback
        traceback.print_exc()
    
def debug_renderer_setup(app):
    """Debug helper to understand renderer configuration"""
    print("\n🔍 RENDERER DEBUG:")
    
    if hasattr(app, 'digitizer'):
        print(f"  Digitizer renderer: {id(app.digitizer.renderer)}")
    
    if hasattr(app, 'vtk_widget'):
        print(f"  VTK widget renderer: {id(app.vtk_widget.renderer)}")
        
    if hasattr(app, 'renderer'):
        print(f"  App renderer: {id(app.renderer)}")
    
    # Check if they're the same object
    if hasattr(app, 'digitizer') and hasattr(app, 'vtk_widget'):
        same = app.digitizer.renderer == app.vtk_widget.renderer
        print(f"  Same renderer? {same}")    
    print()    

def _coordinates_for_las_output(app, data, xyz):
    """Return (xyz_to_write, crs_wkt, crs_epsg) without CRS relabelling.

    LiDAR can be transformed into the Naksha project CRS for display.  A normal
    Save/backup should round-trip the original source CRS when all loaded tiles
    share one source CRS; otherwise (mixed-source project) it writes project
    coordinates with the project CRS.
    """
    arr = np.asarray(xyz)
    runtime_value = None
    source_value = None
    if isinstance(data, dict):
        runtime_value = data.get("_runtime_crs_wkt")
        source_value = data.get("_source_crs_wkt")
        source_consistent = bool(data.get("_source_crs_consistent", True))
    else:
        source_consistent = True
    if runtime_value is None:
        runtime_value = getattr(app, "canvas_crs", None) or getattr(app, "project_crs_wkt", None) or getattr(app, "project_crs_epsg", None)
    if not source_consistent:
        source_value = None

    try:
        from gui.projection_engine import parse_crs, transform_points
        runtime_crs = parse_crs(runtime_value)
        source_crs = parse_crs(source_value) if source_value else None
    except Exception:
        runtime_crs = None
        source_crs = None

    output_crs = source_crs or runtime_crs
    xyz_out = arr
    if runtime_crs is not None and source_crs is not None:
        try:
            if not runtime_crs.equals(source_crs):
                xyz_out, _ = transform_points(
                    arr, runtime_crs, source_crs,
                    copy=True, chunk_size=1_000_000,
                )
                print(f"Saving LiDAR back in native source CRS: {source_crs.name}")
        except Exception as exc:
            # Never write transformed coordinates with the wrong CRS. If the
            # reverse operation fails, keep runtime/project coordinates and CRS.
            print(f"CRS round-trip warning: {exc}; saving in project CRS instead")
            xyz_out = arr
            output_crs = runtime_crs

    wkt = None
    epsg = None
    if output_crs is not None:
        try:
            wkt = output_crs.to_wkt()
        except Exception:
            wkt = None
        try:
            epsg = output_crs.to_epsg()
        except Exception:
            epsg = None
    return xyz_out, wkt, epsg


def save_pointcloud(app, path=None, file_format=None, las_version=None, show_dialog=True):
    """
    ✅ Enhanced Save behavior:
    - Saves point cloud data (XYZ, RGB, classification, intensity)
    - Saves digitized drawings (lines, polygons, annotations) as VLR
    - Drawings are fully editable after reload
    """

    from .native_save import native_active, save_native
    if native_active(app):
        return save_native(app, path, show_dialog=show_dialog, version=las_version)

    data = getattr(app, "data", None)
    if data is None or "xyz" not in data or data["xyz"] is None:
        print("⚠️ No dataset loaded")
        return False

    fence_session, fence_error, is_fence_mode = _inspect_fenced_parent_session(app)
    if is_fence_mode and fence_session is not None:
        is_buffered_grid = fence_session.get("operation") == "buffered_grid"
        if is_buffered_grid and show_dialog:
            save_mode = _prompt_fenced_save_mode(app, fence_session)
            if save_mode is None:
                return False
            if save_mode == "in_place":
                return _save_fenced_parent_files(
                    app,
                    fence_session,
                    output_dir=None,
                    in_place=True,
                    show_messages=True,
                )
            if save_mode == "parents":
                first_source = fence_session["source_files"][0]
                start_dir = os.path.dirname(first_source) if first_source else ""
                output_dir = QFileDialog.getExistingDirectory(
                    app,
                    "Save Full Parent Files",
                    start_dir,
                )
                if not output_dir:
                    return False
                return _save_fenced_parent_files(
                    app,
                    fence_session,
                    output_dir=output_dir,
                    in_place=False,
                    show_messages=True,
                )
        elif is_buffered_grid:
            return _save_fenced_parent_files(
                app,
                fence_session,
                output_dir=None,
                in_place=True,
                show_messages=False,
            )
    elif is_fence_mode and fence_error:
        if show_dialog:
            QMessageBox.warning(
                app,
                "Save Fenced Data",
                fence_error + "\n\nContinuing with fenced-subset Save As only.",
            )
        else:
            QMessageBox.warning(app, "Save", fence_error)
            return False

    xyz = np.asarray(data["xyz"])
    n = xyz.shape[0]

    classes = data.get("classification")
    if classes is None or np.asarray(classes).shape[0] != n:
        classes = np.zeros(n, dtype=np.uint8)
    else:
        classes = np.asarray(classes)

    # RGB/intensity conversion walks every point, so it waits until after the
    # options popup and the file browser: opening Save points must not depend
    # on how many points are loaded.
    save_classes = None
    save_lines = None
    save_fences = None
    save_attributes = None

    # ---------------------------------------------------------
    # ✅ Save As dialog
    # ---------------------------------------------------------
    if show_dialog:
        if path:
            start_dir = os.path.dirname(path)
            base_name = os.path.splitext(os.path.basename(path))[0] or "untitled"
        elif getattr(app, "last_save_path", None):
            start_dir = os.path.dirname(app.last_save_path)
            base_name = os.path.splitext(os.path.basename(app.last_save_path))[0] or "untitled"
        elif getattr(app, "loaded_file", None):
            start_dir = os.path.dirname(app.loaded_file)
            base_name = os.path.splitext(os.path.basename(app.loaded_file))[0] or "untitled"
        elif is_fence_mode and fence_session is not None:
            first_source = fence_session["source_files"][0]
            start_dir = os.path.dirname(first_source)
            base_name = os.path.splitext(os.path.basename(first_source))[0] + "_fence_subset"
        else:
            start_dir = ""
            base_name = "untitled"

        default_path = os.path.join(start_dir, base_name) if start_dir else base_name

        filters = _format_filters()
        default_filter = "LAZ 1.4 (*.laz)"

        try:
            loaded = getattr(app, "loaded_file", None) or getattr(app, "current_file_path", None)
            if loaded:
                loaded_ext = os.path.splitext(loaded)[1].lower()
                src_ver = None
                if hasattr(app, "data") and isinstance(app.data, dict):
                    src_ver = app.data.get("las_version") or app.data.get("version")

                if src_ver is None and loaded_ext in (".las", ".laz") and os.path.exists(loaded):
                    try:
                        # Header only: laspy.read() would decode every point
                        # before the popup could appear.
                        with laspy.open(loaded) as reader:
                            hdr = reader.header
                        src_ver = f"{hdr.version.major}.{hdr.version.minor}"
                    except Exception:
                        src_ver = None

                if src_ver not in LAS_VERSIONS:
                    src_ver = "1.4"

                if loaded_ext == ".las":
                    default_filter = f"LAS {src_ver} (*.las)"
                else:
                    default_filter = f"LAZ {src_ver} (*.laz)"

        except Exception:
            pass

        # Options popup first; the file browser opens only after OK.
        from .save_points_dialog import (
            SavePointsDialog,
            collect_save_fences,
        )

        # Class choices are exactly the classes listed in the Display Mode
        # class table (loaded PTC plus any rows the user added there). The app
        # keeps default palettes even with no PTC, so those are NOT used: with
        # an empty table the Save class box stays blank.
        class_items = []
        try:
            table = getattr(getattr(app, "display_mode_dialog", None), "table", None)
            if table is not None:
                seen = set()
                for row in range(table.rowCount()):
                    code_item = table.item(row, 1)
                    if code_item is None:
                        continue
                    try:
                        code = int(code_item.text())
                    except (TypeError, ValueError):
                        continue
                    if code in seen:
                        continue
                    seen.add(code)
                    desc_item = table.item(row, 2)
                    lvl_item = table.item(row, 4)
                    desc = str(
                        (lvl_item.text() if lvl_item else "")
                        or (desc_item.text() if desc_item else "")
                    ).strip()
                    entry = (code, f"{code} - {desc}" if desc else str(code))
                    color = _table_row_color(table, row)
                    if color is not None:
                        # Same color the Display Mode table shows for the class.
                        entry = entry + (color,)
                    class_items.append(entry)
                class_items.sort(key=lambda item: item[0])
        except Exception as exc:
            print(f"Save points: could not read Display Mode class table: {exc}")
            class_items = []

        app_data = getattr(app, "data", None)

        # Line choices: read straight from the loaded dataset's Point Source
        # IDs (recovered from the LAS/LAZ file when a legacy load path left
        # them out). Display Mode's own list is never consulted.
        line_items, line_provider = _line_items_plan(app, app_data)

        options_dlg = SavePointsDialog(
            app,
            class_items,
            default_format=default_filter.split(" (")[0],
            filename=getattr(app, "loaded_file", None),
            available_dimensions=app_data.keys() if isinstance(app_data, dict) else (),
            line_items=line_items,
            line_provider=line_provider,
            fence_items=collect_save_fences(app),
        )
        if options_dlg.exec() != SavePointsDialog.Accepted:
            return False
        save_classes = options_dlg.selected_classes()
        save_lines = options_dlg.selected_lines()
        save_attributes = options_dlg.selected_attributes()
        # Older callers/tests replace the popup, so read the fence choice
        # defensively: no fence attribute means no fence filter.
        save_fences = (
            options_dlg.selected_fences()
            if hasattr(options_dlg, "selected_fences")
            else None
        )
        if save_lines is not None:
            print(
                "Save points: flight lines selected for save: "
                f"{', '.join(str(line) for line in save_lines)}"
            )
        if save_attributes is not None:
            print(
                "Save points: attributes selected for save: "
                f"{', '.join(save_attributes) or '(none)'}"
            )
        if save_fences:
            print(
                "Save points: fence filter on: "
                f"{len(save_fences)} fence(s) selected"
            )
        default_filter = f"{options_dlg.selected_format()} (*.{'laz' if options_dlg.selected_format().startswith('LAZ') else 'las'})"

        picked_path, selected_filter = QFileDialog.getSaveFileName(
            app,
            "Save As",
            default_path,
            filters,
            default_filter
        )
        if not picked_path:
            return False

        ext, picked_version = _parse_filter(selected_filter)
        path = _ensure_ext(picked_path, ext)

        las_version = picked_version
        file_format = "laz" if ext == ".laz" else "las"

    if not path:
        print("⚠️ No output path provided")
        return False

    ext = os.path.splitext(path)[1].lower()
    if ext not in (".las", ".laz"):
        print(f"⚠️ Unsupported extension: {ext}")
        return False

    rgb16 = _rgb_to_las16(data.get("rgb"))
    intensity16 = _intensity_to_uint16(data.get("intensity"), n_points=n)

    source_path = getattr(app, "loaded_file", None)
    import_options = data.get("import_options") if isinstance(data, dict) else None

    # Flight-line ids (Point Source ID) travel with the subset so a save that
    # only wrote certain lines still identifies them after reload.
    point_source = None
    if isinstance(data, dict) and data.get("point_source_id") is not None:
        candidate = np.asarray(data.get("point_source_id"))
        if candidate.ndim == 1 and candidate.shape[0] == n:
            point_source = candidate
    if point_source is None and save_lines is not None:
        # The Line options come straight from the loaded file; recover the
        # array the same way when the popup's provider never ran (custom or
        # older dialogs), so the selection can still be applied.
        recovered = _recover_point_source_array(app, data)
        if recovered is not None:
            candidate = np.asarray(recovered)
            if candidate.ndim == 1 and candidate.shape[0] == n:
                point_source = candidate

    # "Save class" from the options popup: keep only the chosen classes. A
    # subset no longer matches the source file's point count, so the
    # source-preserving rewrite is skipped and the synthesized path is used.
    keep = None
    if save_classes is not None:
        keep = np.isin(np.asarray(classes), np.asarray(save_classes))
        if not keep.any():
            QMessageBox.warning(app, "Save As", f"No points in class(es) {save_classes}.")
            return False
        print(f"Save class filter: classes {save_classes} -> {int(keep.sum()):,} of {keep.size:,} points")

    # "Line" from the options popup: keep only the chosen flight lines, i.e.
    # the same Point Source IDs Display Mode's Lines button lists.
    if save_lines is not None:
        if point_source is None:
            QMessageBox.warning(
                app,
                "Save As",
                "Flight-line data is not available for the loaded dataset, "
                "so the Line selection cannot be applied.",
            )
            return False
        line_keep = np.isin(point_source, np.asarray(save_lines))
        if not line_keep.any():
            QMessageBox.warning(app, "Save As", f"No points in line(s) {save_lines}.")
            return False
        keep = line_keep if keep is None else (keep & line_keep)
        print(
            f"Save line filter: lines {save_lines} -> "
            f"{int(keep.sum()):,} of {keep.size:,} points"
        )

    # "Inside fence only" from the options popup: keep the points inside
    # the fence(s) picked from the fence tool's By Class list, joined with
    # the Save class and Line choices above.
    if save_fences:
        from .save_points_dialog import fence_inside_mask

        fence_keep = fence_inside_mask(
            xyz, save_fences, digitizer=getattr(app, "digitizer", None)
        )
        if not fence_keep.any():
            QMessageBox.warning(
                app, "Save As", "No points inside the selected fence(s)."
            )
            return False
        keep = fence_keep if keep is None else (keep & fence_keep)
        print(
            f"Save fence filter: {len(save_fences)} selected fence(s) -> "
            f"{int(keep.sum()):,} of {keep.size:,} points"
        )

    if keep is not None:
        xyz = xyz[keep]
        classes = np.asarray(classes)[keep]
        if rgb16 is not None:
            rgb16 = rgb16[keep]
        if intensity16 is not None:
            intensity16 = intensity16[keep]
        if point_source is not None:
            point_source = point_source[keep]

    # Infer version if needed
    if las_version is None:
        probe = path
        if not os.path.exists(probe):
            probe = getattr(app, "loaded_file", None)

        if probe and os.path.exists(probe):
            try:
                with laspy.open(probe) as reader:
                    hv = reader.header.version
                    las_version = f"{hv.major}.{hv.minor}"
            except Exception as e:
                print(f"⚠️ Could not infer LAS version from '{probe}': {e}")

    if las_version is None:
        las_version = "1.4"

    if (
        not show_dialog
        and _same_path(path, source_path)
        and _import_options_reduce_points(import_options)
    ):
        print(
            "❌ Refusing in-place save: the loaded dataset is a filtered/sampled subset, "
            "so overwriting the original file would reduce it."
        )
        return False

    # ✅ LAS 1.2: 5-bit classification (0-31). For codes > 31, use user_data
    # field (MicroStation/TerraScan approach) — never remap or truncate.
    classes_u8 = np.asarray(classes).astype(np.uint8, copy=False)
    max_cls = int(classes_u8.max()) if classes_u8.size else 0

    pack_class_into_flags = False
    if max_cls > 31 and las_version != "1.4":
        # Point formats 0-5 (LAS 1.0-1.3) store a 5-bit classification plus the
        # synthetic/key_point/withheld flag bits. MicroStation/TerraScan
        # compatible packing writes the full code straight into that byte, so
        # the on-disk byte EQUALS the real code (0-255).
        pack_class_into_flags = True
        print(f"\n{'='*60}")
        print(f"📌 LAS {las_version} with extended classes (max={max_cls})")
        print(f"   Packing full code into classification byte via flag bits")
        print(f"   (MicroStation/TerraScan compatible)")
        print(f"{'='*60}\n")

    xyz_to_write, save_crs_wkt, save_crs_epsg = _coordinates_for_las_output(app, data, xyz)

    preserved_las, preserve_reason = _try_build_preserved_las(
        source_path=source_path,
        target_path=path,
        xyz=xyz_to_write,
        classes_u8=classes_u8,
        rgb16=rgb16,
        intensity16=intensity16,
        import_options=import_options,
        requested_version=las_version,
        crs_wkt=save_crs_wkt,
        crs_epsg=save_crs_epsg,
    )
    if preserved_las is not None:
        # "Attributes to save": the rewrite copies every source field, so the
        # unchecked ones are cleared here, just before the file is written.
        _apply_attribute_selection(preserved_las, save_attributes)
        try:
            drawing_data = _serialize_drawings(app)
            if drawing_data:
                decoded = json.loads(drawing_data.decode("utf-8"))
                print(f"✅ Prepared {len(decoded)} drawings for VLR")
                sidecar_path = _attach_drawings_storage(
                    preserved_las,
                    path,
                    drawing_data,
                    "Digitized drawings (lines, polygons, annotations)",
                    verbose=True,
                )
                if sidecar_path is None:
                    print(f"📐 Saved {len(decoded)} drawing object(s) to VLR")
        except Exception as e:
            print(f"⚠️ Failed to save drawings: {e}")

        _atomic_write_las(preserved_las, path)
        print(f"💾 Saved {ext.upper()} {las_version} using source-preserving rewrite: {path}")

        prj_path = os.path.splitext(path)[0] + ".prj"
        try:
            if save_crs_wkt:
                try:
                    from gui.projection_engine import write_prj_sidecar
                    write_prj_sidecar(path, save_crs_wkt, esri_compatible=True)
                except Exception:
                    with open(prj_path, "w", encoding="utf-8") as f:
                        f.write(save_crs_wkt)
                print(f"📌 CRS saved to {prj_path}")
            elif save_crs_epsg:
                with open(prj_path, "w", encoding="utf-8") as f:
                    f.write(f"EPSG:{save_crs_epsg}")
                print(f"📌 EPSG code saved to {prj_path}")
        except Exception as e:
            print(f"⚠️ Failed to write .prj: {e}")

        app.last_save_path = path
        app.last_save_format = file_format
        app.last_save_version = las_version
        if hasattr(app, "statusBar"):
            app.statusBar().showMessage(f"Saved: {path}", 5000)
        return True

    print(f"ℹ️ Falling back to synthesized save path: {preserve_reason}")

    point_format_id = _choose_point_format(
        las_version,
        has_rgb=(rgb16 is not None)
        and _attr_selected(save_attributes, "Color RGB"),
    )
    header = laspy.LasHeader(
        point_format=point_format_id,
        version=_writable_header_version(las_version),
    )

    # Embed CRS
    try:
        import pyproj
        if save_crs_wkt:
            crs = pyproj.CRS.from_wkt(save_crs_wkt)
            header.add_crs(crs)
            print("📌 Embedded output CRS from WKT into LAS header")
        elif save_crs_epsg:
            crs = pyproj.CRS.from_epsg(save_crs_epsg)
            header.add_crs(crs)
            print(f"📌 Embedded output CRS EPSG:{save_crs_epsg} into LAS header")
    except Exception as e:
        print(f"⚠️ Failed to embed CRS into {ext.upper()} header: {e}")

    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz_to_write[:, 0], xyz_to_write[:, 1], xyz_to_write[:, 2]

    # Classification handling
    if las_version == "1.4":
        las.classification = np.clip(classes_u8, 0, 255).astype(np.uint8, copy=False)
    elif pack_class_into_flags:
        # Codes > 31 on LAS 1.0-1.3: write the raw classification byte exactly
        # the way TerraScan/MicroStation does. laspy exposes the byte as four
        # bit-packed sub-fields on point formats 0-5, so the on-disk byte ends
        # up bit-for-bit equal to the real code (0-255). MicroStation reads the
        # whole byte and shows the correct value — no VLR, no user_data needed.
        las.classification = (classes_u8 & 0x1F).astype(np.uint8, copy=False)
        las.synthetic = ((classes_u8 >> 5) & 1).astype(bool)
        las.key_point = ((classes_u8 >> 6) & 1).astype(bool)
        las.withheld  = ((classes_u8 >> 7) & 1).astype(bool)
    else:
        # Codes <= 31 fit the plain 5-bit classification field
        las.classification = np.clip(classes_u8, 0, 31).astype(np.uint8, copy=False)

    _apply_rgb(
        las,
        rgb16 if _attr_selected(save_attributes, "Color RGB") else None,
        las_version,
    )
    if intensity16 is not None:
        las.intensity = intensity16
    if point_source is not None:
        # Keep flight-line identity: Point Source ID is what Display Mode's
        # Lines button reads back after the file is loaded again.
        las.point_source_id = np.asarray(point_source, dtype=np.uint16)

    # "Attributes to save": clear every unchecked attribute before writing.
    _apply_attribute_selection(las, save_attributes)


    # ---------------------------------------------------------
    # ✅ NEW: Save digitized drawings as VLR (record_id=1002)
    # ---------------------------------------------------------
    try:
        drawing_data = _serialize_drawings(app)
        if drawing_data:
            decoded = json.loads(drawing_data.decode('utf-8'))
            print(f"✅ Prepared {len(decoded)} drawings for VLR")

            try:
                sidecar_path = _attach_drawings_storage(
                    las,
                    path,
                    drawing_data,
                    "Digitized drawings (lines, polygons, annotations)",
                    verbose=True,
                )
                if sidecar_path is None:
                    print(f"📐 Saved {len(decoded)} drawing object(s) to VLR")
            except Exception:
                raise
            

    except Exception as e:
        print(f"⚠️ Failed to save drawings: {e}")

    _atomic_write_las(las, path, las_version)
    print(f"💾 Saved {ext.upper()} {las_version}, Point Format {point_format_id}: {path}")

    # .prj sidecar
    prj_path = os.path.splitext(path)[0] + ".prj"
    try:
        if save_crs_wkt:
            try:
                from gui.projection_engine import write_prj_sidecar
                write_prj_sidecar(path, save_crs_wkt, esri_compatible=True)
            except Exception:
                with open(prj_path, "w", encoding="utf-8") as f:
                    f.write(save_crs_wkt)
            print(f"📌 CRS saved to {prj_path}")
        elif save_crs_epsg:
            with open(prj_path, "w", encoding="utf-8") as f:
                f.write(f"EPSG:{save_crs_epsg}")
            print(f"📌 EPSG code saved to {prj_path}")
    except Exception as e:
        print(f"⚠️ Failed to write .prj: {e}")

    app.last_save_path = path
    app.last_save_format = file_format
    app.last_save_version = las_version
    if hasattr(app, "statusBar"):
        app.statusBar().showMessage(f"Saved: {path}", 5000)
    return True


# ---------------- QUICK AUTO-BACKUP SAVE ----------------
def save_pointcloud_quick(app, path):
    """
    Silent quick-save version used for auto-backup.
    ✅ Also saves drawings.
    """
    from .native_save import native_active, save_native
    if native_active(app):
        return save_native(app, path, reopen=False)
    try:
        data = getattr(app, "data", None)
        if data is None or "xyz" not in data or data["xyz"] is None:
            print("⚠️ No data to save for backup")
            return False

        fence_session, fence_error, is_fence_mode = _inspect_fenced_parent_session(app)
        if is_fence_mode and fence_session is not None:
            return _save_fenced_parent_files(
                app,
                fence_session,
                output_dir=None,
                in_place=True,
                show_messages=False,
            )
        if is_fence_mode and fence_error:
            print(f"⚠️ Fence quick-save skipped: {fence_error}")
            return False

        ext = os.path.splitext(path)[1].lower()
        if ext not in (".las", ".laz"):
            print(f"⚠️ Unsupported backup extension: {ext}")
            return False

        xyz = np.asarray(data["xyz"])
        n = xyz.shape[0]

        # Determine backup LAS version
        # Prefer the last explicit save version so backups match Save / Save As.
        las_version = getattr(app, "last_save_version", None)
        try:
            v = data.get("input_format_version", None)
            if las_version not in LAS_VERSIONS and isinstance(v, tuple) and len(v) >= 2:
                las_version = f"{int(v[0])}.{int(v[1])}"
        except Exception:
            pass

        if las_version not in LAS_VERSIONS:
            probe = getattr(app, "loaded_file", None)
            if probe and os.path.exists(probe):
                try:
                    with laspy.open(probe) as reader:
                        hv = reader.header.version
                        las_version = f"{hv.major}.{hv.minor}"
                except Exception as e:
                    print(f"⚠️ Could not infer LAS version for backup: {e}")

        if las_version not in LAS_VERSIONS:
            las_version = "1.4"

        source_path = getattr(app, "loaded_file", None)
        import_options = data.get("import_options") if isinstance(data, dict) else None
        if _same_path(path, source_path) and _import_options_reduce_points(import_options):
            print(
                "❌ Quick-save refused: the current dataset is a filtered/sampled subset, "
                "so saving over the source would reduce it."
            )
            return False

        classes = data.get("classification", np.zeros(n, dtype=np.uint8))
        classes_u8 = np.asarray(classes).astype(np.uint8, copy=False)
        if classes_u8.shape[0] != n:
            classes_u8 = np.zeros(n, dtype=np.uint8)

        rgb16 = _rgb_to_las16(data.get("rgb"))
        intensity16 = _intensity_to_uint16(data.get("intensity"), n_points=n)

        point_source = None
        if data.get("point_source_id") is not None:
            candidate = np.asarray(data.get("point_source_id"))
            if candidate.ndim == 1 and candidate.shape[0] == n:
                point_source = candidate

        xyz_to_write, save_crs_wkt, save_crs_epsg = _coordinates_for_las_output(app, data, xyz)

        preserved_las, preserve_reason = _try_build_preserved_las(
            source_path=source_path,
            target_path=path,
            xyz=xyz_to_write,
            classes_u8=classes_u8,
            rgb16=rgb16,
            intensity16=intensity16,
            import_options=import_options,
            requested_version=las_version,
            crs_wkt=save_crs_wkt,
            crs_epsg=save_crs_epsg,
        )
        if preserved_las is not None:
            try:
                drawing_data = _serialize_drawings(app)
                if drawing_data:
                    _attach_drawings_storage(
                        preserved_las,
                        path,
                        drawing_data,
                        "Digitized drawings",
                        verbose=False,
                    )
            except Exception as e:
                print(f"⚠️ Backup: failed to save drawings: {e}")

            _atomic_write_las(preserved_las, path)
            print(f"💾 Auto-backup saved → {path} (preserved source layout)")
            return True

        print(f"ℹ️ Auto-backup fallback to synthesized save path: {preserve_reason}")

        max_cls = int(classes_u8.max()) if classes_u8.size else 0
        use_user_data_for_class = (max_cls > 31 and las_version != "1.4")

        point_format = _choose_point_format(las_version, has_rgb=(rgb16 is not None))
        header = laspy.LasHeader(
            point_format=point_format,
            version=_writable_header_version(las_version),
        )

        try:
            import pyproj
            if save_crs_wkt:
                header.add_crs(pyproj.CRS.from_wkt(save_crs_wkt))
            elif save_crs_epsg:
                header.add_crs(pyproj.CRS.from_epsg(save_crs_epsg))
        except Exception as e:
            print(f"⚠️ CRS embedding skipped: {e}")

        las = laspy.LasData(header)
        las.x, las.y, las.z = xyz_to_write[:, 0], xyz_to_write[:, 1], xyz_to_write[:, 2]

        if las_version == "1.4":
            las.classification = np.clip(classes_u8, 0, 255).astype(np.uint8, copy=False)
        elif use_user_data_for_class:
            # Codes > 31 on LAS 1.0-1.3: pack full code into the raw classification
            # byte via flag bits (TerraScan/MicroStation compatible). On-disk byte
            # == real code (0-255). No VLR / user_data needed.
            las.classification = (classes_u8 & 0x1F).astype(np.uint8, copy=False)
            las.synthetic = ((classes_u8 >> 5) & 1).astype(bool)
            las.key_point = ((classes_u8 >> 6) & 1).astype(bool)
            las.withheld  = ((classes_u8 >> 7) & 1).astype(bool)
        else:
            # Codes <= 31 fit the plain 5-bit classification field
            las.classification = np.clip(classes_u8, 0, 31).astype(np.uint8, copy=False)

        _apply_rgb(las, rgb16, las_version)
        if intensity16 is not None:
            las.intensity = intensity16
        if point_source is not None:
            las.point_source_id = np.asarray(point_source, dtype=np.uint16)

        # ✅ Save drawings in backup
        try:
            drawing_data = _serialize_drawings(app)
            if drawing_data:
                _attach_drawings_storage(
                    las,
                    path,
                    drawing_data,
                    "Digitized drawings",
                    verbose=False,
                )
        except Exception as e:
            print(f"⚠️ Backup: failed to save drawings: {e}")

        _atomic_write_las(las, path, las_version)
        print(f"💾 Auto-backup saved → {path} (version={las_version}, point_format={point_format})")
        return True

    except Exception as e:
        print(f"⚠️ Quick-save failed: {e}")
        return False
