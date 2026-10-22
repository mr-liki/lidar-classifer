"""native_build_freshness.py - does the prebuilt Vulkan DLL match the sources?

`native/naksha_vulkan/build_msvc/naksha_vulkan.dll` is a prebuilt binary. No
Python code in this tree can rebuild it (it needs MSVC + CMake; see
`build_native.ps1`), so the startup self-check has to decide whether the DLL on
disk is the build the current native sources describe.

Timestamps cannot answer that question. A git checkout, a project restore, a
deploy step or a backup copy re-stamps every file it writes, so the sources
routinely look "newer" than a DLL that was linked from exactly those sources -
the .h/.hpp/.cpp files land a few hundred milliseconds *after* the binary in
copy order. Comparing raw mtimes therefore aborts startup on good builds (and
still misses a header edit that leaves the mtime ordering intact).

Content is checked instead, strongest first:

1. Export surface. Every ``nkv_*`` the narrow C API header declares must be
   exported by the DLL. A missing export is a hard break, not a suspicion:
   ``gui/render_backend.py`` binds those symbols through ctypes, so the failure
   would otherwise surface later as an ``AttributeError`` at a call site, long
   after startup, in the middle of a render.
2. Build stamp (opt-in). :func:`write_build_stamp` records a sha256 over the
   native sources plus the DLL at the moment a build succeeded. Once a stamp
   exists, any later source change without a rebuild is a *proven* mismatch -
   including body-only edits that leave the export surface identical.

mtimes are still reported as evidence, never as the verdict. Everything here is
stdlib-only, imports nothing heavy and never raises: the self-check must not be
able to break the app.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import sys
from typing import Dict, List, Optional, Sequence, Set

NATIVE_DIR_REL = os.path.join("native", "naksha_vulkan")
DLL_REL = os.path.join(NATIVE_DIR_REL, "build_msvc", "naksha_vulkan.dll")
HEADER_REL = os.path.join(
    NATIVE_DIR_REL, "include", "naksha", "naksha_vulkan_c_api.h"
)
STAMP_REL = os.path.join(
    NATIVE_DIR_REL, "build_msvc", "native_build_stamp.json"
)

# Source files that are inputs to the native build. Shader sources are included
# because their compiled .spv siblings are loaded from the DLL's own directory
# at runtime - editing a .vert/.frag without recompiling leaves a stale binary
# sitting next to a current source.
SOURCE_SUFFIXES = (".cpp", ".hpp", ".h", ".glsl", ".vert", ".frag")

# Directories never walked for sources (CMake build trees).
_BUILD_PREFIX = "build"

_EXPORT_DECL_RE = re.compile(
    r"NKV_API\b[^;{]*?\b(nkv_[A-Za-z0-9_]+)\s*\(", re.MULTILINE
)


def dll_candidates(project_root: str) -> List[str]:
    """DLL paths the runtime will try, in ``render_backend.py``'s own order.

    The candidate list is taken from the backend when it is importable so the
    check can never drift from what actually gets loaded; the mirrored list is
    only a fallback for a broken/short-circuited import.
    """
    cands: List[str] = []
    try:
        from gui import render_backend as _rb
        for cand in getattr(_rb, "_DLL_CANDIDATES", ()) or ():
            if cand:
                cands.append(os.path.abspath(cand))
    except Exception:
        pass
    cands.append(os.path.join(project_root, DLL_REL))
    cands.append(os.path.join(project_root, NATIVE_DIR_REL, "build",
                              "naksha_vulkan.dll"))
    cands.append(os.path.join(project_root, "naksha_vulkan.dll"))
    seen: Set[str] = set()
    out: List[str] = []
    for cand in cands:
        key = os.path.normcase(os.path.abspath(cand))
        if key not in seen:
            seen.add(key)
            out.append(os.path.abspath(cand))
    return out


def resolve_dll(project_root: str, dll_path: str = "") -> str:
    """The DLL to check: an explicit path, else the first candidate on disk.

    Falls back to the preferred build path when nothing exists, so the report
    can still name what was looked for.
    """
    if dll_path:
        return os.path.abspath(dll_path)
    cands = dll_candidates(project_root)
    for cand in cands:
        if os.path.isfile(cand):
            return cand
    return cands[0] if cands else os.path.join(project_root, DLL_REL)


def project_paths(project_root: str) -> Dict[str, str]:
    """Absolute paths of the four artifacts this module reasons about."""
    return {
        "native_dir": os.path.join(project_root, NATIVE_DIR_REL),
        "dll": resolve_dll(project_root),
        "header": os.path.join(project_root, HEADER_REL),
        "stamp": os.path.join(project_root, STAMP_REL),
    }


# ---------------------------------------------------------------------------
# PE export table (stdlib only - no pefile dependency)
# ---------------------------------------------------------------------------
def pe_exported_names(dll_path: str) -> Optional[Set[str]]:
    """Names in a PE file's export table.

    Returns ``None`` when the file is missing or is not a readable PE image, so
    callers can tell "cannot verify" apart from "verified, empty".
    """
    try:
        with open(dll_path, "rb") as fh:
            blob = fh.read()
    except OSError:
        return None
    if len(blob) < 0x40 or blob[:2] != b"MZ":
        return None

    try:
        e_lfanew = struct.unpack_from("<I", blob, 0x3C)[0]
        if blob[e_lfanew:e_lfanew + 4] != b"PE\0\0":
            return None
        coff = e_lfanew + 4
        (n_sections,) = struct.unpack_from("<H", blob, coff + 2)
        (opt_size,) = struct.unpack_from("<H", blob, coff + 16)
        opt = coff + 20
        (magic,) = struct.unpack_from("<H", blob, opt)
        if magic == 0x10B:            # PE32
            data_dirs = opt + 96
        elif magic == 0x20B:          # PE32+
            data_dirs = opt + 112
        else:
            return None
        export_rva, export_size = struct.unpack_from("<II", blob, data_dirs)
        if not export_rva or not export_size:
            return set()

        sections = []
        sec = opt + opt_size
        for _ in range(n_sections):
            (vsize, vaddr, raw_size, raw_ptr) = struct.unpack_from(
                "<IIII", blob, sec + 8
            )
            sections.append((vaddr, vsize or raw_size, raw_ptr, raw_size))
            sec += 40

        def _rva_to_offset(rva: int) -> Optional[int]:
            for vaddr, vsize, raw_ptr, raw_size in sections:
                if vaddr <= rva < vaddr + vsize:
                    off = raw_ptr + (rva - vaddr)
                    # Data must live inside the raw section bytes.
                    if 0 <= rva - vaddr < raw_size:
                        return off
                    return None
            return None

        exp = _rva_to_offset(export_rva)
        if exp is None:
            return None
        (n_names,) = struct.unpack_from("<I", blob, exp + 24)
        (names_rva,) = struct.unpack_from("<I", blob, exp + 32)
        names_off = _rva_to_offset(names_rva)
        if names_off is None:
            return None

        out: Set[str] = set()
        for i in range(n_names):
            (name_rva,) = struct.unpack_from("<I", blob, names_off + 4 * i)
            off = _rva_to_offset(name_rva)
            if off is None:
                continue
            end = blob.find(b"\0", off)
            if end < 0:
                continue
            out.add(blob[off:end].decode("latin-1"))
        return out
    except (struct.error, IndexError, ValueError):
        return None


def c_api_declared_exports(header_path: str) -> Optional[Set[str]]:
    """``nkv_*`` functions the narrow C API header declares (``NKV_API``)."""
    try:
        with open(header_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    return set(_EXPORT_DECL_RE.findall(text))


# ---------------------------------------------------------------------------
# Native source digest
# ---------------------------------------------------------------------------
def native_source_files(native_dir: str) -> List[str]:
    """Native source files, relative to ``native_dir``, sorted, build trees out."""
    found: List[str] = []
    for dirpath, dirnames, filenames in os.walk(native_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(_BUILD_PREFIX)]
        for name in filenames:
            if not name.lower().endswith(SOURCE_SUFFIXES):
                continue
            full = os.path.join(dirpath, name)
            found.append(os.path.relpath(full, native_dir).replace("\\", "/"))
    found.sort()
    return found


def native_source_digest(native_dir: str) -> Dict[str, object]:
    """sha256 over every native source file's path and content.

    Returns ``{"digest", "files", "bytes", "latest_mtime", "newest"}``. The
    digest is the stamp's payload, so a later edit changes it even when the
    export surface does not.
    """
    digest = hashlib.sha256()
    files: List[str] = []
    total = 0
    latest = 0.0
    newest = ""
    for rel in native_source_files(native_dir):
        full = os.path.join(native_dir, rel)
        try:
            with open(full, "rb") as fh:
                data = fh.read()
            mtime = os.path.getmtime(full)
        except OSError:
            continue
        files.append(rel)
        total += len(data)
        if mtime > latest:
            latest = mtime
            newest = rel
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        # Git may check text out as LF or CRLF on Windows. Those are the same
        # build inputs; checkout line endings must never demand a native rebuild.
        normalized = data.replace(b"\r\n", b"\n")
        digest.update(hashlib.sha256(normalized).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return {
        "digest": digest.hexdigest(),
        "files": len(files),
        "bytes": total,
        "latest_mtime": latest,
        "newest": newest,
        "paths": files,
    }


def sha256_file(path: str) -> str:
    """Hex sha256 of a file, or "" when it cannot be read."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def read_build_stamp(stamp_path: str) -> Optional[Dict[str, object]]:
    """Parsed build stamp, or ``None`` when absent/unreadable."""
    try:
        with open(stamp_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_build_stamp(project_root: str, dll_path: str = "") -> Dict[str, object]:
    """Record the current sources + DLL as a matched pair.

    Call this immediately after a successful native rebuild (``build_native.ps1``
    does). It is the only place a stamp is created: nothing infers provenance
    that was not observed, so before a stamp exists the export check is the only
    evidence and the report says so.
    """
    paths = project_paths(project_root)
    dll = resolve_dll(project_root, dll_path)
    sources = native_source_digest(paths["native_dir"])
    dll_exports = pe_exported_names(dll)
    declared = c_api_declared_exports(paths["header"]) or set()
    stamp = {
        "created": _now_iso(),
        "dll": os.path.relpath(dll, project_root).replace("\\", "/"),
        "dll_sha256": sha256_file(dll),
        "dll_exports": len(dll_exports) if dll_exports is not None else None,
        "missing_exports": sorted(declared - (dll_exports or set())),
        "source_digest": sources["digest"],
        "source_files": sources["files"],
        "source_newest": sources["newest"],
    }
    tmp = paths["stamp"] + ".tmp"
    os.makedirs(os.path.dirname(paths["stamp"]), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, paths["stamp"])
    return stamp


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def assess(project_root: str, dll_path: str = "") -> Dict[str, object]:
    """Decide whether the deployed Vulkan DLL matches the native sources.

    Never raises. ``rebuild_required`` is True only on positive evidence of a
    mismatch - a declared export the binary does not provide, or a build stamp
    that no longer matches the sources.
    """
    paths = project_paths(project_root)
    dll = resolve_dll(project_root, dll_path)
    report: Dict[str, object] = {
        "dll_path": dll if os.path.isfile(dll) else "<not found>",
        "dll_mtime": 0.0,
        "dll_sha256": "",
        "dll_exports": None,
        "declared_exports": 0,
        "missing_exports": [],
        "source_digest": "",
        "source_files": 0,
        "source_latest_mtime": 0.0,
        "source_newest": "",
        "stamp_present": False,
        "stamp_created": "",
        "stamp_digest": "",
        "verdict": "UNVERIFIED",
        "rebuild_required": False,
        "reasons": [],
        "notes": [],
    }
    reasons: List[str] = report["reasons"]        # type: ignore[assignment]
    notes: List[str] = report["notes"]            # type: ignore[assignment]

    sources = native_source_digest(paths["native_dir"])
    report["source_digest"] = str(sources["digest"])[:16]
    report["source_files"] = sources["files"]
    report["source_latest_mtime"] = sources["latest_mtime"]
    report["source_newest"] = sources["newest"]

    if not os.path.isfile(dll):
        report["verdict"] = "DLL MISSING"
        notes.append(
            "no prebuilt Vulkan DLL in this tree; the app runs on the VTK "
            "backend (opt-in Vulkan)"
        )
        return report

    try:
        report["dll_mtime"] = os.path.getmtime(dll)
    except OSError:
        pass
    report["dll_sha256"] = sha256_file(dll)

    exported = pe_exported_names(dll)
    declared = c_api_declared_exports(paths["header"])
    report["dll_exports"] = len(exported) if exported is not None else None
    if declared:
        report["declared_exports"] = len(declared)
    if exported is None:
        notes.append("DLL export table unreadable - export surface not verified")
    elif declared is None:
        notes.append(
            f"C API header missing ({HEADER_REL}) - export surface not verified"
        )
    else:
        missing = sorted(declared - exported)
        report["missing_exports"] = missing
        if missing:
            reasons.append(
                "NAKSHA VULKAN DLL IS STALE - REBUILD REQUIRED. The DLL does "
                f"not export {len(missing)} nkv_* function(s) declared by "
                f"native/{HEADER_REL.replace(os.sep, '/')}: "
                + ", ".join(missing[:8])
                + ("..." if len(missing) > 8 else "")
                + " (rebuild with native\\naksha_vulkan\\build_msvc)"
            )

    stamp = read_build_stamp(paths["stamp"])
    frozen = bool(getattr(sys, "frozen", False))
    if stamp is None:
        notes.append(
            "no build stamp - source provenance unverified (run "
            "write_native_build_stamp.py after a native rebuild to pin it)"
        )
    elif frozen:
        # A packaged build ships the DLL, the C API header and the stamp, but
        # not the .cpp/.h sources the stamp was taken over. Recomputing the
        # digest there would always mismatch, so provenance is reported as
        # recorded rather than re-derived. The export check above is still a
        # real verification of the binary, and dev trees are unaffected.
        report["stamp_present"] = True
        report["stamp_created"] = str(stamp.get("created", ""))
        report["stamp_digest"] = str(stamp.get("source_digest", ""))
        notes.append(
            "frozen build - build stamp reported as recorded; native sources "
            "are not shipped, so the source digest is not re-derived"
        )
    else:
        report["stamp_present"] = True
        report["stamp_created"] = str(stamp.get("created", ""))
        report["stamp_digest"] = str(stamp.get("source_digest", ""))
        stamp_digest = str(stamp.get("source_digest", ""))
        if not stamp_digest:
            notes.append("build stamp has no source_digest - ignored")
        elif stamp_digest != sources["digest"]:
            reasons.append(
                "NAKSHA VULKAN DLL IS STALE - REBUILD REQUIRED. Native sources "
                "changed since the DLL was built and verified (stamp "
                f"{stamp_digest[:16]} of {report['stamp_created']} != current "
                f"{str(sources['digest'])[:16]}). Rebuild with "
                "native\\naksha_vulkan\\build_msvc, then run "
                "write_native_build_stamp.py."
            )

    report["rebuild_required"] = bool(reasons)
    report["verdict"] = "REBUILD REQUIRED" if reasons else "OK"
    return report
