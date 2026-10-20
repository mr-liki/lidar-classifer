"""File > Open with several LAZ files: individual LNTs, Vulkan stays the viewport.

Runs the real application window and the real multi-file routing, so it also
proves the individual LNT collection never falls back to VTK.

Run: venv/Scripts/python.exe audit/lnt_native_batch_gui_test.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("NAKSHA_RENDER_BACKEND", "auto")

import laspy
import numpy as np
from gui.laspy_compat import install_opaque_extra_bytes_fix
from gui.lnt_collection import read_collection
from pyproj import CRS
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication, QFileDialog

install_opaque_extra_bytes_fix()

results = []


def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {detail}", flush=True)


def pump(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(.01)


def write_laz(path, count, line_id, rgb, epsg=32643):
    header = laspy.LasHeader(point_format=3 if rgb else 1, version="1.2")
    # Reproduce customer files whose type-0 Extra Bytes options are 9 bytes,
    # which older laspy getters misread as scale/no-data flags.
    header.add_extra_dim(laspy.ExtraBytesParams(name='opaque', type='9u1'))
    header.scales = [.001] * 3
    header.offsets = [500000., 4000000., 0.]
    if epsg is not None:
        header.add_crs(CRS.from_epsg(epsg))
    rng = np.random.default_rng(line_id)
    with laspy.open(path, mode="w", header=header) as writer:
        for start in range(0, count, 5000):
            n = min(5000, count - start)
            record = laspy.ScaleAwarePointRecord.zeros(n, header=header)
            record.x = rng.uniform(0., 100., n) + 500000. + (line_id - 11) * 120.
            record.y = rng.uniform(0., 100., n) + 4000000.
            record.z = rng.uniform(10., 18., n)
            record.classification = np.full(n, 2 + line_id - 11, np.uint8)
            record.intensity = rng.integers(1, 60000, n, dtype=np.uint16)
            record.point_source_id = np.full(n, line_id, np.uint16)
            record['opaque'] = np.full((n, 9), line_id, np.uint8)
            if rgb:
                record.red = np.full(n, 30000, np.uint16)
                record.green = np.full(n, 20000, np.uint16)
                record.blue = np.full(n, 10000, np.uint16)
            writer.write_points(record)


def _source_ids(path):
    with laspy.open(path) as reader:
        return {int(v) for v in np.unique(reader.read().point_source_id)}


def _point_count(path):
    with laspy.open(path) as reader:
        return int(reader.header.point_count)


with tempfile.TemporaryDirectory(prefix="laz_batch_gui_") as folder:
    folder = Path(folder)
    first, second = folder / "flight_1.laz", folder / "flight_2.laz"
    write_laz(first, 30_000, 11, rgb=True)
    write_laz(second, 25_000, 12, rgb=False)
    sources = [str(first), str(second)]
    check("both inputs are real, decompressible LAZ files",
          all(Path(p).suffix.lower() == ".laz" and _point_count(p) > 0 for p in sources))

    app = QApplication(sys.argv)
    from gui.app_window import NakshaApp

    win = NakshaApp()
    win.show()
    pump(4.0)
    backend = win.render_backend
    check("the Vulkan runtime initialized at startup",
          backend.vulkan_widget is not None and backend.vulkan_backend is not None)

    QFileDialog.getOpenFileNames = staticmethod(lambda *args, **kwargs: (sources, ""))
    started = time.monotonic()
    win.open_file()
    deadline = time.monotonic() + 120.
    while time.monotonic() < deadline:
        pump(.05)
        if getattr(win, "_lnt_main_view", None) is not None and getattr(win, "_lnt_build", None) is None:
            break
    first_open = time.monotonic() - started
    view = getattr(win, "_lnt_main_view", None)
    check("two LAZ files open through File > Open", view is not None and view.native is not None)
    if view is None or view.native is None:
        print("  cannot continue without a native view", flush=True)
        os._exit(1)

    check("the viewport references the individual LNTs through a small index",
          str(win._lnt_source_path).lower().endswith(".lntset"),
          str(win._lnt_source_path))
    check("the selection index stays beside the LAZ files",
          Path(win._lnt_source_path).resolve().parents[1]
          == folder.resolve() / "naksha_native_batches",
          str(win._lnt_source_path))
    check("the original LAZ files are remembered for reopen/recovery",
          [os.path.normcase(os.path.realpath(p)) for p in win._lnt_source_paths]
          == [os.path.normcase(os.path.realpath(p)) for p in sources])
    check("no VTK point array was populated", win.data is None)
    check("every input point reached the native file", view.total_points == 55_000,
          f"{view.total_points:,}")
    check("each LAZ has an independently reusable LNT",
          all(Path(p).with_suffix('.lnt').is_file() for p in sources))
    check("no combined LAS or LNT was created",not list(folder.rglob('combined.*')))
    check("Point Source IDs remain available in the original files",
          set().union(*(_source_ids(p) for p in sources))=={11,12})
    check("the Vulkan viewport owns LiDAR (VTK LiDAR off)",
          backend.vulkan_owns_lidar_viewport() and not backend.vtk_lidar_rendering_enabled())
    check("the native viewport erases and presents a frame", bool(view.native.frame(1 / 60.)))
    check("native presentation modes still apply", view.apply_shading("intensity")
          and view.apply_shading("class"))

    # Reopening the same pair must hit the batch cache and stay native.
    previous = view
    from gui.lnt_open import open_with_lnt
    started = time.monotonic()
    open_with_lnt(win, sources)
    deadline = time.monotonic() + 60.
    while time.monotonic() < deadline and getattr(win, "_lnt_build", None) is not None:
        pump(.05)
    cached_worker = time.monotonic() - started   # merge/build step only
    deadline = time.monotonic() + 60.
    while time.monotonic() < deadline:
        pump(.05)
        view = getattr(win, "_lnt_main_view", None)
        if view is not None and view is not previous and getattr(win, "_lnt_build", None) is None:
            break
    check("a cached reopen skips conversion and returns immediately",
          cached_worker < 5.0,
          f"build step {cached_worker:.2f}s vs {first_open:.2f}s for the first open")
    check("reopening the LAZ pair reuses the cached native dataset",
          view is not None and view is not previous and view.total_points == 55_000)
    check("reopen still leaves Vulkan in charge",
          backend.vulkan_owns_lidar_viewport() and not backend.vtk_lidar_rendering_enabled()
          and win.data is None)

    # Reopen just one original LAZ: it must use its existing individual LNT.
    from unittest.mock import patch
    with patch('gui.lnt_native.convert_las',side_effect=AssertionError('individual source was reconverted')):
        open_with_lnt(win,str(first))
        pump(.3)
        view=getattr(win,'_lnt_main_view',None)
        check("one original LAZ reopens from its individual LNT without conversion",
              view is not None and view.total_points==30_000 and Path(view.lnt_path)==first.with_suffix('.lnt'))
        check("single-file reopen needs no build worker",getattr(win,'_lnt_build',None) is None)

    # ---- the reported failure: LAZ files with NO CRS under a project CRS ----
    # Before the fix this raised "Some selected files have no CRS" and File >
    # Open stopped on a "Could not build the .lnt file" dialog.
    plain_1, plain_2 = folder / "plain_1.laz", folder / "plain_2.laz"
    write_laz(plain_1, 20_000, 21, rgb=True, epsg=None)
    write_laz(plain_2, 15_000, 22, rgb=False, epsg=None)
    check("the CRS-less inputs really declare no CRS",
          all(laspy.open(p).header.parse_crs() is None for p in (plain_1, plain_2)))
    from PySide6.QtWidgets import QMessageBox
    real_critical = QMessageBox.critical
    popups = []
    QMessageBox.critical = staticmethod(lambda *args, **kwargs: popups.append(args[1:]))
    previous = view
    win.project_crs_wkt = CRS.from_epsg(32643).to_wkt()   # a project CRS, as a real project has
    QFileDialog.getOpenFileNames = staticmethod(
        lambda *args, **kwargs: ([str(plain_1), str(plain_2)], ""))
    win.open_file()
    deadline = time.monotonic() + 180.
    while time.monotonic() < deadline and getattr(win, "_lnt_build", None) is not None:
        pump(.05)
    plain_view = getattr(win, "_lnt_main_view", None)
    check("LAZ files without a CRS open instead of popping 'Could not build'",
          not popups and plain_view is not None and plain_view is not previous,
          str(popups[0][1]) if popups else "no dialog")
    check("every point of the CRS-less pair reached the native file",
          plain_view is not None and plain_view.total_points == 35_000,
          f"{plain_view.total_points:,}" if plain_view is not None else "no view")
    check("the selection index retains the project CRS",
          plain_view is not None and getattr(win, "_lnt_source_path", None) is not None
          and CRS.from_user_input(read_collection(win._lnt_source_path)['target_crs']) == CRS.from_epsg(32643),
          str(getattr(win, "_lnt_source_path", None)))
    check("the CRS-less batch also stays on the LAZ drive",
          plain_view is not None
          and Path(win._lnt_source_path).resolve().parents[1]
          == folder.resolve() / "naksha_native_batches",
          str(getattr(win, "_lnt_source_path", None)))
    check("Vulkan still owns the viewport after the CRS-less open",
          plain_view is not None and backend.vulkan_owns_lidar_viewport()
          and not backend.vtk_lidar_rendering_enabled() and win.data is None)
    QMessageBox.critical = real_critical
    win.project_crs_wkt = None
    view = plain_view

    if view is not None:
        # The same teardown the app performs on quit: release the native
        # renderer without rebuilding the owner's VTK renderer.
        view.stop(restore=False)
        pump(.3)
    print(f"{sum(results)}/{len(results)} LAZ batch GUI checks passed", flush=True)
    # Skip interpreter/GPU teardown: the vendored DLL has a pre-existing
    # access violation at process exit that is unrelated to these checks.
    os._exit(0 if all(results) else 1)
