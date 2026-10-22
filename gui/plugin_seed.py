"""Seed bundled plugins into the per-user plugins folder (frozen builds only).

`gui/plugin_manager.py` reads plugins from `%LOCALAPPDATA%/NakshaAI-LiDAR/plugins`
once frozen, because Program Files is not writable. The installer payload (the
`plugins/` tree bundled into `_internal`) therefore has to be copied in once, and
refreshed when a newer build ships a newer plugin version.

The marker file `.bundled-version` records which bundled version was copied, so
a user-modified plugin is left alone until the package itself changes.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

MARKER = ".bundled-version"
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", MARKER)


def bundled_plugins_dir() -> Optional[Path]:
    """The read-only `plugins/` payload inside the frozen app, if any."""
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return None
    candidate = Path(base) / "plugins"
    return candidate if candidate.is_dir() else None


def user_plugins_dir() -> Path:
    """Mirror of `gui.plugin_manager._get_plugins_dir()` for frozen builds."""
    local_appdata = os.getenv("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "NakshaAI-LiDAR" / "plugins"
    return Path.home() / ".naksha" / "plugins"


def _version_of(plugin_dir: Path) -> str:
    try:
        manifest = json.loads((plugin_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    for key in ("version", "release", "build"):
        value = manifest.get(key)
        if value:
            return str(value)
    return ""


def _marker_value(plugin_dir: Path) -> str:
    try:
        return (plugin_dir / MARKER).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def seed_bundled_plugins(force: bool = False) -> List[str]:
    """Copy/refresh bundled plugins and return the names that were written."""
    if not getattr(sys, "frozen", False) and not force:
        return []

    source_root = bundled_plugins_dir()
    if source_root is None:
        return []

    target_root = user_plugins_dir()
    seeded: List[str] = []
    try:
        target_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[plugin-seed] cannot create {target_root}: {exc}")
        return seeded

    for source in sorted(source_root.iterdir()):
        if not source.is_dir() or not (source / "manifest.json").is_file():
            continue

        version = _version_of(source)
        target = target_root / source.name
        if not force and target.is_dir() and _marker_value(target) == version:
            continue

        try:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target, ignore=_IGNORE)
            (target / MARKER).write_text(version, encoding="utf-8")
            seeded.append(source.name)
        except OSError as exc:
            print(f"[plugin-seed] failed to install {source.name}: {exc}")

    if seeded:
        print(f"[plugin-seed] installed {len(seeded)} bundled plugin(s) "
              f"into {target_root}: {', '.join(seeded)}")
    return seeded
