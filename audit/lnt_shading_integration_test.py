"""Integration test: shading modes on a native .lnt cloud inside the real application.

Run from the repo root with:  set PYTHONUTF8=1 && venv\Scripts\python.exe audit\lnt_shading_integration_test.py
Opens the application window (needs a GPU). It checks that the Display UI path
(app.set_display_mode) reaches the NATIVE shading API for every mode - no VTK - that the
value ranges come from the file, and that mode switches are instant. The pixels themselves
are verified against the reference ramps by Section U of native/lnt/main.cpp.
"""
import os, sys, time, tempfile, traceback
if os.name == "nt":
    # Failed GPU tests must return a failure status instead of leaving a
    # Windows fault dialog on the user's desktop.
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
SCRATCH = tempfile.mkdtemp(prefix="lnt_shading_itest_")

import numpy as np
import laspy

results = []
def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {detail}", flush=True)

las_path = os.path.join(SCRATCH, "shade.las")
lnt_path = os.path.join(SCRATCH, "shade.lnt")
rng = np.random.default_rng(11)
n = 300_000
x0, y0 = 600000.0, 4100000.0
xs = x0 + rng.random(n) * 300.0
ys = y0 + rng.random(n) * 300.0
zs = 50.0 + rng.random(n) * 40.0                      # 50..90 m
hdr = laspy.LasHeader(point_format=3, version="1.2")
hdr.scales = np.array([0.001] * 3); hdr.offsets = np.array([x0, y0, 50.0])
las = laspy.LasData(hdr)
las.x, las.y, las.z = xs, ys, zs
las.classification = (rng.random(n) * 8).astype(np.uint8)
las.intensity = (rng.random(n) * 65535).astype(np.uint16)
las.red = las.green = las.blue = (rng.random(n) * 65535).astype(np.uint16)
las.write(las_path)

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QMessageBox

def pump(sec):
    end = time.time() + sec
    while time.time() < end:
        QCoreApplication.processEvents()
        time.sleep(0.01)

try:
    from gui import lnt_open, lnt_native as nat
    lnt_open.build_lnt(las_path, lambda p, t: None, lambda: False)
    app = QApplication(sys.argv)
    from gui.app_window import NakshaApp
    win = NakshaApp()
    win.show()
    pump(3.0)
    win.data = None
    lnt_open._show(win, lnt_path, "shade.lnt")
    pump(3.0)
    view = getattr(win, "_lnt_main_view", None)
    check("the .lnt is shown in the native viewport", view is not None and view.native is not None)
    nv = view.native
    check("the engine DLL has the shading API", nv.has_shading())

    st = nv.data_stats()
    check("value ranges come from the file (Z 1st..99th percentile inside 50..90 m)",
          st is not None and 50.0 <= st["z_p1"] < st["z_p99"] <= 90.0 and st["i_max"] > st["i_min"],
          f"{st}")

    # every UI mode, through the same entry the Display dialog uses
    modes = ["class", "rgb", "intensity", "elevation", "shaded", "shaded_class", "classification"]
    t0 = time.perf_counter()
    ok_all = True
    for m in modes:
        ok = bool(win.set_display_mode(m))
        ok_all = ok_all and ok and view.shading_name == m
        pump(0.15)
    check("Display UI -> native shading works for class, rgb, intensity, elevation, shaded",
          ok_all, f"{(time.perf_counter() - t0):.2f} s for {len(modes)} modes incl. 0.15 s frame pumps")

    # the UI path never touches VTK data: app.data stays empty through all of it
    check("no point data was created for shading (app.data is still empty)", not win.data)

    # settings: a ramp / range change applies on the next switch, instantly
    view.ramp_elevation = nat.RAMP_BLUE_RED
    view.z_range = (60.0, 80.0)
    t1 = time.perf_counter()
    ok = view.apply_shading("elevation")
    ms = (time.perf_counter() - t1) * 1000.0
    check("a ramp + range change is one uniform write (well under a frame)", ok and ms < 20.0,
          f"{ms:.2f} ms")
    view.intensity_range = (0.25, 0.75)
    check("intensity bounds apply", view.apply_shading("intensity"))
    view.edl = 0.8
    check("normal shading with Eye-Dome Lighting applies", view.apply_shading("shaded"))
    # Same native palette entry used by Display; wait for worker + atomic swap.
    palette = np.zeros((256, 4), dtype=np.uint8)
    palette[:, :3] = (200, 100, 50)
    palette[:, 3] = 1
    def wait_method(method, seconds=10):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            pump(0.05)
            if nv.surface_method() == method:
                # A second frame activates the pending GPU set.
                pump(0.2)
                return True
        return False
    check("multiple visible sampled classes choose barycentric", wait_method(1))
    palette[:, 3] = 0
    palette[2, 3] = 1
    check("single class palette applies", nv.set_classification_lut(palette))
    check("hiding classes rebuilds and switches to facet", wait_method(0))
    palette[5, 3] = 1
    check("second class palette applies", nv.set_classification_lut(palette))
    check("showing another class switches back to barycentric", wait_method(1))
    check("Surface with several classes applies", view.apply_shading("surface"))
    # apply_shading pushes the application palette, which contains multiple classes.
    check("Surface follows the same multiple class rule", wait_method(1))
    palette[:, 3] = 0
    palette[2, 3] = 1
    check("Surface single class palette applies", nv.set_classification_lut(palette))
    check("Surface with one class switches to facet", wait_method(0))
    check("viewer keeps rendering throughout visibility changes", nv.frame(0.016))
    check("an unknown mode name is refused, not silently coerced", not view.apply_shading("bogus"))
    check("the keyboard shortcuts 1-5 are bound to the native modes",
          all(f'apply_shading("{m}")' in open(os.path.join(ROOT, "gui", "lnt_open.py"), encoding="utf-8").read()
              for m in ("class", "intensity", "elevation", "rgb", "shaded")))

    lnt_open.close_lnt_view(win)
    pump(1.0)
except Exception:
    traceback.print_exc()
    results.append(False)

try:
    if "win" in globals():
        # Use the application's full cleanup (Vulkan shutdown and VTK
        # Finalize) before unloading NVIDIA's OpenGL DLL at process exit.
        question = QMessageBox.question
        QMessageBox.question = lambda *args, **kwargs: QMessageBox.Yes
        try:
            win.close()
        finally:
            QMessageBox.question = question
        win.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        pump(0.2)
    if "app" in globals():
        app.quit()
except Exception:
    traceback.print_exc()
    results.append(False)

print(f"\n{sum(results)}/{len(results)} integration checks passed", flush=True)
raise SystemExit(0 if all(results) else 1)
