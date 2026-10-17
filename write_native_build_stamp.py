"""write_native_build_stamp.py - pin the native DLL to the sources it was built from.

`main.py`'s startup self-check can always verify that the deployed
`native/naksha_vulkan/build_msvc/naksha_vulkan.dll` exports every `nkv_*` the
C API header declares. It cannot know whether a change to a .cpp/.hpp body (same
export surface) was compiled in - unless a build stamp says which sources that
DLL was linked from.

Run this straight after a successful native rebuild (`build_native.ps1` calls it
automatically):

    py write_native_build_stamp.py

From then on, editing any native source without rebuilding makes the startup
self-check fail with the proven mismatch instead of a guess, and a fresh
checkout / restore / backup copy - which re-stamps every file it writes - no
longer looks stale.

Exit codes: 0 stamped (or --check found a match), 1 mismatch/missing, 2 usage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gui.native_build_freshness import (  # noqa: E402
    assess,
    project_paths,
    read_build_stamp,
    write_build_stamp,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Record (or verify) the native source digest the Vulkan DLL "
                    "was built from.",
    )
    parser.add_argument(
        "--dll", default="",
        help="DLL to stamp (default: native/naksha_vulkan/build_msvc/naksha_vulkan.dll)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="verify the current stamp instead of writing one (no writes)",
    )
    args = parser.parse_args(argv)

    paths = project_paths(PROJECT_ROOT)
    dll = args.dll or paths["dll"]

    if not os.path.isfile(dll):
        print(f"[stamp] FAIL: no DLL at {dll}")
        return 1

    if args.check:
        report = assess(PROJECT_ROOT, dll)
        print(f"[stamp] dll            : {dll}")
        print(f"[stamp] verdict        : {report['verdict']}")
        print(f"[stamp] stamp_present  : {report['stamp_present']}")
        print(f"[stamp] stamp_digest   : {report['stamp_digest']}")
        print(f"[stamp] source_digest  : {report['source_digest']}")
        for reason in report["reasons"]:
            print(f"[stamp] reason         : {reason}")
        return 0 if not report["rebuild_required"] else 1

    before = read_build_stamp(paths["stamp"])
    stamp = write_build_stamp(PROJECT_ROOT, dll)
    print(f"[stamp] dll            : {dll}")
    print(f"[stamp] dll_exports    : {stamp['dll_exports']}")
    print(f"[stamp] missing_exports: {stamp['missing_exports'] or 'none'}")
    print(f"[stamp] source_files   : {stamp['source_files']}")
    print(f"[stamp] source_digest  : {stamp['source_digest'][:16]}")
    print(f"[stamp] written        : {paths['stamp']}")
    if before is not None and before.get("source_digest") != stamp["source_digest"]:
        print("[stamp] note           : superseded an older stamp")
    print(json.dumps({k: stamp[k] for k in ("created", "source_files")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
