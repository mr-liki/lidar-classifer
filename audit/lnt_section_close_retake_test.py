"""Close -> retake: Cross / Cut Section windows closed by hand must draw again.

Takes a cross section and a cut on a native .lnt in the real app, closes the
section windows the way the user does (the title-bar X), takes them again and
checks the reopened windows present real frames with the points in them. Repeats
the cycle a few times. No VTK in the .lnt path.

Run from the repo root:  set PYTHONUTF8=1 && venv\\Scripts\\python.exe audit\\lnt_section_close_retake_test.py
A crash in nvoglv64.dll right at exit is harmless teardown noise."""
import os, sys, time, traceback
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import tempfile
SCRATCH = tempfile.mkdtemp(prefix="lnt_section_retake_")
os.environ.setdefault("NAKSHA_RENDER_BACKEND", "auto")

import numpy as np
import laspy

results = []
def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {detail}", flush=True)

las_path = os.path.join(SCRATCH, "retake.las")
lnt_path = os.path.join(SCRATCH, "retake.lnt")
rng = np.random.default_rng(3)
n = 300_000
x0, y0 = 500000.0, 4000000.0
xs = x0 + rng.random(n) * 400.0
ys = y0 + rng.random(n) * 400.0
zs = 100.0 + rng.random(n) * 2.0
cls = np.full(n, 2, dtype=np.uint8)
wall = np.abs(xs - (x0 + 200.0)) < 1.5
zs[wall] = 100.0 + rng.random(int(wall.sum())) * 15.0
cls[wall] = 6
hdr = laspy.LasHeader(point_format=3, version="1.2")
hdr.scales = np.array([0.001, 0.001, 0.001]); hdr.offsets = np.array([x0, y0, 100.0])
las = laspy.LasData(hdr)
las.x, las.y, las.z = xs, ys, zs
las.classification = cls
las.intensity = (rng.random(n) * 60000).astype(np.uint16)
las.red = las.green = las.blue = (rng.random(n) * 65535).astype(np.uint16)
las.write(las_path)

from PySide6.QtCore import QCoreApplication, QPointF
from PySide6.QtWidgets import QApplication

def pump(sec):
    end = time.time() + sec
    while time.time() < end:
        QCoreApplication.processEvents()
        time.sleep(0.01)

try:
    from gui import lnt_open
    lnt_open.build_lnt(las_path, lambda p, t: None, lambda: False)
    app = QApplication(sys.argv)
    from gui.app_window import NakshaApp
    win = NakshaApp()
    win.show()
    pump(2.0)
    win.data = None
    lnt_open._show(win, lnt_path, "retake.lnt")
    pump(3.0)
    view = win._lnt_main_view
    nv = view.native
    nv.frame_region(x0 + 200.0, y0 + 200.0, 500.0)
    pump(1.0)
    win.current_view = "top"
    from PySide6.QtWidgets import QInputDialog
    QInputDialog.getItem = staticmethod(lambda *a, **k: ("View 1", True))
    from gui.native_sections import CUT_SLOT

    # The section path must stay native Vulkan: record any VTK module that gets
    # imported (and who imported it) from here on, and any VTK widget created.
    import builtins, traceback as _tb
    vtk_before = {m for m in sys.modules if m == "vtk" or m.startswith("vtkmodules")}
    vtk_imports = []
    _real_import = builtins.__import__
    def _watch_import(name, *a, **k):
        if (name == "vtk" or name.startswith("vtkmodules")) and name not in sys.modules:
            vtk_imports.append((name, "".join(_tb.format_stack(limit=6)[:-1])))
        return _real_import(name, *a, **k)
    builtins.__import__ = _watch_import

    w = view.owner.vulkan_widget
    dpr = max(w.devicePixelRatioF(), 1.0)
    def click_world(x, y):
        s = nv.world_to_screen(x, y)
        win.cross_interactor.on_press(QPointF(s[0] / dpr, s[1] / dpr))

    def take_section(view_index, y):
        win._cross_section_default_view = view_index
        win.enable_cross_section_mode()
        pump(0.5)
        click_world(x0 + 150.0, y0 + y)
        click_world(x0 + 250.0, y0 + y)
        click_world(x0 + 250.0, y0 + y + 10.0)
        pump(1.0)

    def drawn(canvas):
        """Frames presented to the window since `mark`, and points in the picture."""
        if not hasattr(canvas, "_dbg_hwnd"):
            canvas._dbg_hwnd = int(canvas.winId())
        before = canvas.backend.present_count
        rc = canvas.backend.render_count
        canvas._dirty = True
        pump(0.8)
        if os.environ.get("RETAKE_DEBUG"):
            print("   dbg closed", canvas._closed, "visible", canvas.isVisible(), "dirty", canvas._dirty,
                  "renders", canvas.backend.render_count - rc, "handle", canvas.backend._handle,
                  "size", canvas.width(), canvas.height(), "hwnd", int(canvas.winId()), "first", getattr(canvas, "_dbg_hwnd", None), "err", canvas.backend.last_error(), flush=True)
        frame = canvas.backend.capture_frame()
        bg = frame[0, 0, :3].astype(int) if frame is not None else None
        lit = 0 if frame is None else int(np.any(np.abs(frame[..., :3].astype(int) - bg) > 30, axis=2).sum())
        return canvas.backend.present_count - before, lit

    for cycle in range(3):
        print(f"--- cycle {cycle + 1} ---", flush=True)
        take_section(0, 200.0)
        take_section(1, 260.0)
        ns = win._native_sections
        for slot in (1, 2):
            cv = ns.views.get(slot)
            ok = cv is not None and ns.docks[slot].isVisible()
            presents, lit = drawn(cv) if ok else (0, 0)
            check(f"cycle {cycle + 1}: Cross Section View {slot} is shown and draws its points",
                  ok and presents > 0 and lit > 200, f"presents {presents}, lit px {lit}")
        cv1 = ns.views[1]
        # capture_frame reads the Vulkan swapchain, so a Qt fallback cannot
        # accidentally satisfy these preview checks.
        for mode in ('class', 'shaded_class'):
            cv1.apply_palette(cv1.palette, mode)
            centre = float(cv1.centre[0])
            half = cv1.scale * .35
            cv1.show_cut_preview(centre)
            pump(.3)
            frame = cv1.backend.capture_frame()
            def color_columns(frame, rgb):
                if frame is None:
                    return 0
                mask = np.all(np.abs(frame[..., :3].astype(int) - rgb) < 12, axis=2)
                return int((mask.sum(axis=0) > frame.shape[0] * .9).sum())
            check(f"cycle {cycle + 1}: {mode} native white centre spans full height",
                  color_columns(frame, (255, 255, 255)) >= 1)
            cv1.show_cut_preview(centre, half)
            pump(.3)
            frame = cv1.backend.capture_frame()
            orange = color_columns(frame, (255, 165, 0))
            check(f"cycle {cycle + 1}: {mode} native orange depth lines span full height",
                  orange >= 2 and color_columns(frame, (255, 255, 255)) >= 1,
                  f"orange columns {orange}")
            cv1.clear_cut_preview()
            pump(.3)
            check(f"cycle {cycle + 1}: {mode} native preview clears",
                  color_columns(cv1.backend.capture_frame(), (255, 165, 0)) == 0)
        cv1.apply_palette(cv1.palette, 'class')
        if getattr(win, "cross_interactor", None) is not None:
            win.cross_interactor.cancel()
        cv1.cut_centre = 50.0
        ns.finish_cut(cv1, 52.0)
        pump(1.0)
        cut = ns.views.get(CUT_SLOT)
        ok = cut is not None and ns.docks[CUT_SLOT].isVisible() and len(cut.indices) > 100
        presents, lit = drawn(cut) if ok else (0, 0)
        check(f"cycle {cycle + 1}: Cut Section is shown and draws its points",
              ok and presents > 0 and lit > 50, f"presents {presents}, lit px {lit}")

        # the user closes every section window with its title-bar X
        for slot in (1, 2, CUT_SLOT):
            ns.docks[slot].close()
        pump(1.0)
        check(f"cycle {cycle + 1}: closing hides the section windows",
              not any(ns.docks[s].isVisible() for s in (1, 2, CUT_SLOT)))

    builtins.__import__ = _real_import
    vtk_new = {m for m in sys.modules if m == "vtk" or m.startswith("vtkmodules")} - vtk_before
    for name, stack in vtk_imports[:3]:
        print(f"   VTK import of {name} from:", stack, flush=True)
    check("taking/closing/retaking sections imports no VTK module (native Vulkan only)",
          not vtk_new and not vtk_imports, f"{sorted(vtk_new)[:5]}")
    vtk_widgets = [w for w in QApplication.allWidgets() if "vtk" in type(w).__name__.lower()]
    check("no VTK render widget exists on the .lnt session", not vtk_widgets,
          f"{[type(w).__name__ for w in vtk_widgets][:5]}")
    lnt_open.close_lnt_view(win)
    pump(0.5)
except Exception:
    traceback.print_exc()
    results.append(False)

print(f"\n{sum(results)}/{len(results)} close/retake checks passed", flush=True)
os._exit(0 if all(results) else 1)
