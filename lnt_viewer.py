"""Launcher for the native .lnt viewer (native/lnt/lnt_viewer.exe).

    py lnt_viewer.py                      # file dialog
    py lnt_viewer.py path\\to\\cloud.las    # -> cloud.lnt beside it
    py lnt_viewer.py path\\to\\cloud.laz    # decompressed on the fly -> cloud.lnt beside it
    py lnt_viewer.py cloud.lnt --sse 4    # extra args go to the viewer

The .lnt is created ONCE beside the source and reused while it is newer than the
source. Controls: left-drag orbit | right-drag pan | wheel zoom | [ ] SSE |
1/2/3 colour | R reset | Esc quit.
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(ROOT, "native", "lnt", "lnt_viewer.exe")
DLL = os.path.join(ROOT, "native", "naksha_vulkan", "build_msvc", "naksha_vulkan.dll")
VENV_PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")


def _reexec_in_venv():
    """The global `py` has no PySide6/laspy; the project's venv does."""
    if os.path.isfile(VENV_PY) and \
            os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(VENV_PY):
        sys.exit(subprocess.call([VENV_PY, os.path.abspath(__file__)] + sys.argv[1:]))


def _pick_file():
    from PySide6.QtWidgets import QApplication, QFileDialog
    app = QApplication.instance() or QApplication(sys.argv[:1])
    path, _ = QFileDialog.getOpenFileName(
        None, "Open point cloud", ROOT,
        "Point clouds (*.las *.laz *.lnt);;All files (*)")
    del app
    return path


def _laz_to_las(laz, las, chunk=2_000_000):
    """Stream-decompress LAZ -> uncompressed LAS (laspy + lazrs)."""
    import laspy
    with laspy.open(laz) as src:
        header = laspy.LasHeader(point_format=src.header.point_format,
                                 version=src.header.version)
        header.scales, header.offsets = src.header.scales, src.header.offsets
        total, done = src.header.point_count, 0
        with laspy.open(las, mode="w", header=header) as dst:
            for pts in src.chunk_iterator(chunk):
                dst.write_points(pts)
                done += len(pts)
                print(f"[laz] decompressed {done:,}/{total:,}", end="\r", flush=True)
    print()


def main(argv):
    if not os.path.isfile(EXE):
        print(f"missing {EXE}\nbuild it (see the header of native/lnt/lnt_viewer.cpp)")
        return 1
    args = list(argv)
    if not args or args[0].startswith("--"):
        path = _pick_file()
        if not path:
            return 0
        args.insert(0, path)
    env = dict(os.environ)
    env.setdefault("LNT_VULKAN_DLL", DLL)

    src = args[0]
    tmp = None
    if src.lower().endswith(".laz"):
        lnt = src[:-4] + ".lnt"
        if os.path.isfile(lnt) and os.path.getmtime(lnt) >= os.path.getmtime(src):
            print(f"[laz] {lnt} is up to date, reusing it")
            args[0] = lnt
        else:
            fd, tmp = tempfile.mkstemp(suffix=".las", prefix="lnt_")
            os.close(fd)
            print(f"[laz] decompressing {src}")
            try:
                _laz_to_las(src, tmp)
            except Exception as exc:
                print(f"[laz] FAILED: {type(exc).__name__}: {exc}")
                os.path.exists(tmp) and os.remove(tmp)
                return 1
            args[0] = tmp
            args += ["--lnt-out", lnt]
    try:
        return subprocess.call([EXE] + args, cwd=ROOT, env=env)
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    _reexec_in_venv()
    sys.exit(main(sys.argv[1:]))
