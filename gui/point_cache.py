"""Persistent disk-backed point cache for very large LAS/LAZ datasets.

Phase 2C goals
--------------
* Keep the existing ``app.data``/NumPy API intact so classification, saving,
  sections, shading and undo/redo do not need a simultaneous rewrite.
* On a warm load, bypass LAZ decompression and return ``numpy.memmap`` arrays
  backed by a local SSD cache.
* Mutable arrays (XYZ/classification) are opened copy-on-write.  Editing never
  modifies the cache files themselves, so an unsaved session cannot poison a
  future load.
* Cache validity is tied to source path + size + nanosecond mtime + import
  options.  Saving/replacing the source therefore invalidates the old cache.
* The manifest is written last; interrupted builds are ignored safely.

The cache is intentionally presentation-neutral.  It stores canonical point
order and attributes only.  Render/spatial metadata may live beside it, but is
owned by the respective subsystems.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, Optional

import numpy as np


# Bumped because classification loading now preserves LAS 1.2 extended class
# values reconstructed from the legacy flag bits.  A cache made by the older
# loader may contain the collapsed base value (for example 19 instead of 51)
# and must never be served as if it were canonical.
CACHE_SCHEMA = 2
DEFAULT_THRESHOLD = 5_000_000
DEFAULT_MAX_GB = 12.0
DEFAULT_MAX_ENTRIES = 6
DEFAULT_MIN_FREE_GB = 20.0
_PRUNE_DONE = False


def _env_bool(name: str, default: bool = True) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw not in ("0", "false", "off", "no")


def cache_enabled() -> bool:
    return _env_bool("NAKSHA_POINT_CACHE", True)


def cache_threshold() -> int:
    raw = str(os.getenv("NAKSHA_POINT_CACHE_THRESHOLD", DEFAULT_THRESHOLD)).replace(",", "")
    try:
        return max(250_000, int(raw))
    except Exception:
        return DEFAULT_THRESHOLD


def cache_root() -> Path:
    raw = str(os.getenv("NAKSHA_POINT_CACHE_DIR", "")).strip()
    if raw:
        return Path(raw).expanduser()
    local = os.getenv("LOCALAPPDATA")
    if local:
        return Path(local) / "NakshaAI" / "point_cache"
    return Path.home() / ".cache" / "nakshaai" / "point_cache"


def _max_cache_bytes() -> int:
    raw = str(os.getenv("NAKSHA_POINT_CACHE_MAX_GB", DEFAULT_MAX_GB)).strip()
    try:
        return int(max(2.0, float(raw)) * (1024 ** 3))
    except Exception:
        return int(DEFAULT_MAX_GB * (1024 ** 3))


def _max_cache_entries() -> int:
    raw = str(os.getenv("NAKSHA_POINT_CACHE_MAX_ENTRIES", DEFAULT_MAX_ENTRIES)).strip()
    try:
        return max(1, min(64, int(raw)))
    except Exception:
        return int(DEFAULT_MAX_ENTRIES)


def _min_free_bytes() -> int:
    raw = str(os.getenv("NAKSHA_POINT_CACHE_MIN_FREE_GB", DEFAULT_MIN_FREE_GB)).strip()
    try:
        return int(max(2.0, float(raw)) * (1024 ** 3))
    except Exception:
        return int(DEFAULT_MIN_FREE_GB * (1024 ** 3))


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def normalize_import_options(options: Optional[dict]) -> dict:
    """Return only load options that can change the canonical arrays."""
    options = dict(options or {})
    attrs = dict(options.get("attributes") or {})
    # Keep all attribute booleans: loading Color/Intensity changes cached arrays.
    return {
        "only_class": bool(options.get("only_class", False)),
        "class_codes": sorted(int(v) for v in (options.get("class_codes") or [])),
        "only_every": bool(options.get("only_every", False)),
        "nth_point": int(options.get("nth_point", 1) or 1),
        "attributes": {str(k): bool(v) for k, v in sorted(attrs.items())},
    }


def _source_identity(filename: str) -> dict:
    p = Path(filename)
    st = p.stat()
    try:
        resolved = str(p.resolve())
    except Exception:
        resolved = str(p.absolute())
    return {
        "path": os.path.normcase(resolved),
        "size": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
    }


def cache_key(filename: str, options: Optional[dict]) -> str:
    payload = {
        "schema": CACHE_SCHEMA,
        "source": _source_identity(filename),
        "options": normalize_import_options(options),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cache_dir_for_source(filename: str, options: Optional[dict]) -> Path:
    key = cache_key(filename, options)
    return cache_root() / key[:2] / key


def cache_dir_from_array(array) -> Optional[Path]:
    """Return the Phase-2C cache directory for a memmapped cached array."""
    try:
        fn = getattr(array, "filename", None)
        if not fn:
            return None
        p = Path(str(fn)).resolve().parent
        if (p / "manifest.json").is_file():
            return p
    except Exception:
        pass
    return None


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def _load_manifest(cache_dir: Path) -> Optional[dict]:
    try:
        return json.loads(_manifest_path(cache_dir).read_text(encoding="utf-8"))
    except Exception:
        return None


def _manifest_valid(manifest: dict, filename: str, options: Optional[dict]) -> bool:
    try:
        if int(manifest.get("schema", -1)) != CACHE_SCHEMA:
            return False
        if manifest.get("source") != _source_identity(filename):
            return False
        if manifest.get("options") != normalize_import_options(options):
            return False
        arrays = manifest.get("arrays") or {}
        if "xyz" not in arrays or "classification" not in arrays:
            return False
        return True
    except Exception:
        return False


def _array_file(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.npy"


def try_load_lidar_cache(filename: str, options: Optional[dict]) -> Optional[Dict[str, Any]]:
    if not cache_enabled():
        return None
    try:
        cdir = cache_dir_for_source(filename, options)
    except Exception:
        return None
    manifest = _load_manifest(cdir)
    if not manifest or not _manifest_valid(manifest, filename, options):
        return None

    t0 = time.perf_counter()
    arrays = manifest.get("arrays") or {}
    loaded: Dict[str, Any] = {}
    try:
        # Copy-on-write mmap is the compatibility sweet spot: callers can mutate
        # classification/XYZ normally, while the persistent cache stays pristine.
        for name in ("xyz", "rgb", "intensity", "classification", "point_source_id"):
            entry = arrays.get(name)
            if not entry:
                loaded[name] = None
                continue
            path = cdir / str(entry["file"])
            loaded[name] = np.load(path, mmap_mode="c", allow_pickle=False)

        meta = dict(manifest.get("metadata") or {})
        version = meta.get("input_format_version")
        if isinstance(version, list):
            version = tuple(int(v) for v in version)
        loaded.update(
            crs_wkt=meta.get("crs_wkt"),
            crs_epsg=meta.get("crs_epsg"),
            input_format_version=version,
            input_point_format=meta.get("input_point_format"),
            import_options=options,
            type="las",
            _point_cache_dir=str(cdir),
            _point_cache_key=cdir.name,
            _point_cache_hit=True,
        )
        # Touch directory mtime for best-effort LRU cleanup.
        try:
            os.utime(cdir, None)
        except Exception:
            pass
        elapsed = (time.perf_counter() - t0) * 1000.0
        n = int(len(loaded["xyz"]))
        print(
            "PHASE2C_POINT_CACHE status=hit "
            f"points={n:,} mmap=1 load_ms={elapsed:.1f} dir={cdir}"
        )
        global _PRUNE_DONE
        if not _PRUNE_DONE:
            _PRUNE_DONE = True
            _prune_cache(cache_root(), keep=cdir)
        return loaded
    except Exception as exc:
        print(f"PHASE2C_POINT_CACHE status=load_failed error={exc}")
        return None


def _write_npy_atomic(path: Path, array: np.ndarray) -> dict:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, np.asarray(array), allow_pickle=False)
    os.replace(tmp, path)
    return {
        "file": path.name,
        "dtype": str(np.asarray(array).dtype),
        "shape": list(np.asarray(array).shape),
        "bytes": int(np.asarray(array).nbytes),
    }


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for fn in files:
                try:
                    total += int((Path(root) / fn).stat().st_size)
                except Exception:
                    pass
    except Exception:
        pass
    return total


def _prune_cache(root: Path, keep: Optional[Path] = None) -> None:
    """Best-effort bounded LRU cleanup.

    The cache is disposable acceleration data, never project data.  Phase 2C.2
    therefore bounds it by three independent constraints: total bytes, recent
    entry count and a minimum free-disk reserve.  Cleanup failure never breaks a
    dataset load.
    """
    try:
        if not root.exists():
            return
        max_bytes = _max_cache_bytes()
        max_entries = _max_cache_entries()
        min_free = _min_free_bytes()
        entries = []
        total = 0
        now = time.time()

        for prefix in list(root.iterdir()):
            if not prefix.is_dir():
                continue
            for cdir in list(prefix.iterdir()):
                if not cdir.is_dir():
                    continue
                name = cdir.name
                # Interrupted atomic builds are safe to delete once clearly stale.
                if ".tmp-" in name:
                    try:
                        if now - cdir.stat().st_mtime > 6 * 3600:
                            shutil.rmtree(cdir, ignore_errors=True)
                    except Exception:
                        pass
                    continue
                manifest_path = cdir / "manifest.json"
                if not manifest_path.is_file():
                    # A cache directory without a manifest can never be opened.
                    try:
                        if now - cdir.stat().st_mtime > 24 * 3600:
                            shutil.rmtree(cdir, ignore_errors=True)
                    except Exception:
                        pass
                    continue
                size = _dir_size(cdir)
                total += size
                try:
                    mtime = cdir.stat().st_mtime
                except Exception:
                    mtime = 0.0
                entries.append((mtime, size, cdir))

        try:
            free = int(shutil.disk_usage(root).free)
        except Exception:
            free = 1 << 62

        def over_limits(entry_count: int, total_bytes: int, free_bytes: int) -> bool:
            return (
                total_bytes > max_bytes
                or entry_count > max_entries
                or free_bytes < min_free
            )

        removed = 0; removed_bytes = 0
        live_count = len(entries)
        if over_limits(live_count, total, free):
            for _mtime, size, cdir in sorted(entries, key=lambda x: x[0]):
                if keep is not None and cdir == keep:
                    continue
                if not over_limits(live_count, total, free):
                    break
                shutil.rmtree(cdir, ignore_errors=True)
                total -= size
                free += size
                live_count -= 1
                removed += 1
                removed_bytes += size

        if removed:
            print(
                "PHASE2C2_CACHE_PRUNE status=ok "
                f"removed={removed} freed_gb={removed_bytes/(1024**3):.2f} "
                f"entries={live_count} cache_gb={total/(1024**3):.2f} "
                f"max_gb={max_bytes/(1024**3):.1f} max_entries={max_entries} "
                f"free_gb={free/(1024**3):.1f} min_free_gb={min_free/(1024**3):.1f}"
            )
    except Exception as exc:
        print(f"PHASE2C2_CACHE_PRUNE status=warning error={exc}")


def store_lidar_cache(
    filename: str,
    options: Optional[dict],
    data: Dict[str, Any],
    *,
    reopen_memmap: bool = True,
) -> Optional[Dict[str, Any]]:
    """Persist canonical LAS arrays and optionally reopen them as COW memmaps."""
    if not cache_enabled():
        return None
    xyz = data.get("xyz") if isinstance(data, dict) else None
    classification = data.get("classification") if isinstance(data, dict) else None
    if xyz is None or classification is None or len(xyz) < cache_threshold():
        return None

    t0 = time.perf_counter()
    try:
        cdir = cache_dir_for_source(filename, options)
        existing = _load_manifest(cdir)
        if existing and _manifest_valid(existing, filename, options):
            return try_load_lidar_cache(filename, options) if reopen_memmap else None

        cdir.parent.mkdir(parents=True, exist_ok=True)
        # Build in a sibling temp directory and publish with one directory rename.
        tmp = Path(tempfile.mkdtemp(prefix=f"{cdir.name}.tmp-", dir=str(cdir.parent)))
        arrays_meta = {}
        try:
            for name in ("xyz", "rgb", "intensity", "classification", "point_source_id"):
                arr = data.get(name)
                if arr is None:
                    continue
                arrays_meta[name] = _write_npy_atomic(tmp / f"{name}.npy", arr)

            manifest = {
                "schema": CACHE_SCHEMA,
                "created": time.time(),
                "source": _source_identity(filename),
                "options": normalize_import_options(options),
                "arrays": arrays_meta,
                "metadata": {
                    "crs_wkt": data.get("crs_wkt"),
                    "crs_epsg": data.get("crs_epsg"),
                    "input_format_version": _jsonable(data.get("input_format_version")),
                    "input_point_format": data.get("input_point_format"),
                },
            }
            (tmp / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            if cdir.exists():
                shutil.rmtree(cdir, ignore_errors=True)
            os.replace(tmp, cdir)
            tmp = None
        finally:
            if tmp is not None and tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

        elapsed = (time.perf_counter() - t0) * 1000.0
        size_mb = _dir_size(cdir) / (1024.0 * 1024.0)
        print(
            "PHASE2C_POINT_CACHE status=built "
            f"points={len(xyz):,} size_mb={size_mb:.1f} build_ms={elapsed:.1f} dir={cdir}"
        )
        _prune_cache(cache_root(), keep=cdir)
        return try_load_lidar_cache(filename, options) if reopen_memmap else None
    except Exception as exc:
        print(f"PHASE2C_POINT_CACHE status=build_failed error={exc}")
        return None
