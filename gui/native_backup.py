"""Backup of the classified NATIVE point cloud - written by the native engine.

WHY THIS EXISTS
    Closing the application (or losing power) must never overwrite the point
    cloud the user loaded. The native engine used to flush the class edits
    straight into the loaded document, so the file the user opened ended up
    holding work that was never asked to be saved there, and there was no
    separate copy of it anywhere.

    The classified result now goes to a SEPARATE file next to the loaded
    document - 123_backup.laz - produced by the native exporter: no VTK, no
    render readback, and the source file is never written. The file that was
    loaded is left exactly as it was found (the class edits are rolled back
    out of it AFTER the backup succeeded, so nothing is ever lost).

THE SETTINGS
    The Auto-Backup Settings dialog (gui/backup_settings_dialog.py) writes
    them into QSettings("NakshaAI", "LidarApp"):

        backup_enabled           the periodic auto-backup
        backup_use_custom_path   backup folder: custom, or next to the original
        backup_custom_path       the custom folder
        backup_interval_minutes  how often the periodic backup runs
        backup_naming_format     _backup / _YYYYMMDD_HHMMSS / _autosave
        backup_on_exit           write the backup when the application closes
        backup_keep_original     leave the loaded document unchanged afterwards

    The same keys drive the periodic backup and the backup written when the
    application closes, so the dialog configures both.
"""
from __future__ import annotations

import os
import time

SETTINGS_ORG = "NakshaAI"
SETTINGS_APP = "LidarApp"

NAMING_BACKUP = 0
NAMING_TIMESTAMP = 1
NAMING_AUTOSAVE = 2


# =============================================================================
# settings
# =============================================================================
def settings() -> dict:
    """The Auto-Backup Settings, exactly as the dialog stored them."""
    from PySide6.QtCore import QSettings

    store = QSettings(SETTINGS_ORG, SETTINGS_APP)
    try:
        interval = int(store.value("backup_interval_minutes", 5))
    except (TypeError, ValueError):
        interval = 5
    try:
        naming = int(store.value("backup_naming_format", NAMING_BACKUP))
    except (TypeError, ValueError):
        naming = NAMING_BACKUP
    return {
        "enabled": bool(store.value("backup_enabled", True, type=bool)),
        "use_custom_path": bool(store.value("backup_use_custom_path", False, type=bool)),
        "custom_path": str(store.value("backup_custom_path", "", type=str) or ""),
        "interval_minutes": max(1, min(60, interval)),
        "naming_format": naming if naming in (NAMING_BACKUP, NAMING_TIMESTAMP,
                                              NAMING_AUTOSAVE) else NAMING_BACKUP,
        "on_exit": bool(store.value("backup_on_exit", True, type=bool)),
        "keep_original": bool(store.value("backup_keep_original", True, type=bool)),
    }


# =============================================================================
# what is open
# =============================================================================
def active(app) -> bool:
    """True when a native (.lnt / Vulkan) point cloud is open."""
    view = getattr(app, "_lnt_main_view", None)
    return view is not None and getattr(view, "native", None) is not None


def document(app):
    """The native document the class edits are written into."""
    view = getattr(app, "_lnt_main_view", None)
    if view is None:
        return None
    path = getattr(view, "lnt_path", None)
    return str(path) if path else None


def source_for(app):
    """The LAS/LAZ the document was built from, or the document itself."""
    try:
        from gui.native_save import _source_for
        source = _source_for(app)
    except Exception:
        source = None
    if source:
        return str(source)
    return document(app) or getattr(app, "loaded_file", None)


def class_writes(app) -> int:
    """How many class records this session wrote into the native document."""
    view = getattr(app, "_lnt_main_view", None)
    native = getattr(view, "native", None) if view is not None else None
    return int(getattr(native, "class_writes", 0) or 0)


def has_work(app) -> bool:
    """True when this session classified something that is not backed up yet."""
    return class_writes(app) > 0


def _backup_state(app) -> dict:
    state = getattr(app, "_native_backup_state", None)
    if state is None:
        state = {}
        try:
            app._native_backup_state = state
        except Exception:
            pass
    return state


def backup_is_current(app, destination) -> bool:
    """True when the backup on disk already holds this session's work."""
    state = _backup_state(app)
    key = os.path.normcase(os.path.abspath(destination))
    entry = state.get(key)
    if not entry:
        return False
    writes, stamp = entry
    if writes != class_writes(app):
        return False
    try:
        return int(os.path.getmtime(destination)) == int(stamp)
    except OSError:
        return False


def _mark_current(app, destination) -> None:
    state = _backup_state(app)
    key = os.path.normcase(os.path.abspath(destination))
    try:
        stamp = int(os.path.getmtime(destination))
    except OSError:
        stamp = 0
    state[key] = (class_writes(app), stamp)


# =============================================================================
# where the backup goes
# =============================================================================
def backup_path(app, source=None, *, timestamp=None):
    """The backup file path: the Auto-Backup Settings decide folder and name.

    The default is the folder the point cloud was loaded from, with the name
    the naming format asks for (123_backup.laz by default).
    """
    try:
        cfg = settings()
        source = source or source_for(app)
        if not source:
            return None
        source = os.path.abspath(os.fspath(source))
        base = os.path.splitext(os.path.basename(source))[0]
        ext = os.path.splitext(source)[1].lower()
        if ext not in (".las", ".laz"):
            ext = ".laz"
        if cfg["naming_format"] == NAMING_TIMESTAMP:
            stamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
            name = f"{base}_{stamp}{ext}"
        elif cfg["naming_format"] == NAMING_AUTOSAVE:
            name = f"{base}_autosave{ext}"
        else:
            name = f"{base}_backup{ext}"
        folder = ""
        if cfg["use_custom_path"] and cfg["custom_path"]:
            folder = os.path.abspath(cfg["custom_path"])
        if not folder:
            folder = os.path.dirname(source)
        if not folder:
            folder = os.getcwd()
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, name)
    except Exception as exc:                       # noqa: BLE001 - never break the caller
        print(f"[NATIVE-BACKUP] could not compute the backup path: {exc}")
        return None


def _protected_paths(app):
    """Files the backup must never be written over."""
    paths = set()
    for candidate in (source_for(app), document(app), getattr(app, "loaded_file", None)):
        if candidate:
            paths.add(os.path.normcase(os.path.abspath(os.fspath(candidate))))
    return paths


# =============================================================================
# the export
# =============================================================================
def export_backup(app, destination=None, *, progress=None, flush=True,
                  mapping=None, drawings=None):
    """Write the classified point cloud to `destination` with the native exporter.

    The loaded document and its LAS/LAZ source are never written: the export
    reads them and produces a NEW file. `flush` writes the pending class edits
    into the native document first (it must run on the GUI thread); pass
    flush=False when the caller has already flushed.
    """
    from gui import native_save

    if not active(app):
        raise RuntimeError("No native point cloud is open.")
    lnt = document(app)
    if not lnt:
        raise RuntimeError("The native document has no path.")
    if flush:
        native_save.flush_edits(app)
    source = source_for(app)
    if destination is None:
        destination = backup_path(app)
    if not destination:
        raise RuntimeError("No backup destination could be determined.")
    destination = os.path.abspath(os.fspath(destination))
    if os.path.normcase(destination) in _protected_paths(app):
        raise ValueError("The backup destination is the loaded file itself; "
                         "refusing to overwrite it.")
    if mapping is None:
        from gui.class_slots import _state
        mapping = dict(_state(app)["code_to_slot"])
    if drawings is None:
        from gui.save_pointcloud import _serialize_drawings
        drawings = _serialize_drawings(app)
    view = getattr(app, "_lnt_main_view", None)
    cache = getattr(view, "_save_mapping_cache", None)
    if cache is None:
        cache = {}
        try:
            view._save_mapping_cache = cache
        except Exception:
            pass
    result = native_save.write_pair(
        source, lnt, destination, mapping=mapping, cache=cache,
        companion=False, drawings=drawings,
        progress=progress or (lambda n, text: None),
    )
    _mark_current(app, destination)
    return result


def restore_loaded_document(app) -> int:
    """Roll the class edits back out of the loaded native document.

    Called ONLY after a successful backup, so the work is safe in the backup
    file. The document keeps the classes it had when it was opened (or at the
    last explicit Save), which is what "the loaded file is never overwritten"
    means. Returns the number of records restored.
    """
    store = getattr(app, "_lnt_section_store", None)
    view = getattr(app, "_lnt_main_view", None)
    if store is None or view is None:
        return 0
    if getattr(store, "view", None) is not view:
        return 0
    try:
        reverted = store.discard_unflushed_edits()
    except Exception as exc:                       # noqa: BLE001 - best effort
        print(f"[NATIVE-BACKUP] could not restore the loaded document: {exc}")
        return 0
    if reverted:
        print(f"[NATIVE-BACKUP] restored {reverted:,} class records in the loaded "
              f"document; the classified copy is in the backup file")
    return reverted
