"""Integration test: Cross Section + Cut Section on a native .lnt cloud inside the real app.

Run from the repo root with:  set PYTHONUTF8=1 && venv\Scripts\python.exe audit\lnt_section_integration_test.py
It opens the application window (needs a GPU) and exits 0 when every check passes. A crash
in nvoglv64.dll right at exit is harmless Qt/OpenGL teardown noise."""
import os, sys, time, traceback
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import tempfile
SCRATCH = tempfile.mkdtemp(prefix="lnt_section_itest_")
os.environ.setdefault("NAKSHA_RENDER_BACKEND", "auto")

import numpy as np
import laspy

results = []
def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {detail}", flush=True)

# ---- 1. a synthetic LAS with structure: ground (class 2) + a wall (class 6) -------------
las_path = os.path.join(SCRATCH, "itest_cs.las")
lnt_path = os.path.join(SCRATCH, "itest_cs.lnt")
for p in (lnt_path, lnt_path + ".tmp"):
    if os.path.exists(p):
        os.remove(p)
rng = np.random.default_rng(7)
n = 400_000
x0, y0 = 500000.0, 4000000.0
xs = x0 + rng.random(n) * 400.0
ys = y0 + rng.random(n) * 400.0
zs = 100.0 + rng.random(n) * 2.0
cls = np.full(n, 2, dtype=np.uint8)
wall = (np.abs(xs - (x0 + 200.0)) < 1.5) & (rng.random(n) < 1.0)
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
print(f"LAS written: {n:,} points, {int(wall.sum()):,} in the wall", flush=True)

from PySide6.QtCore import QCoreApplication, QPointF, Qt
from PySide6.QtWidgets import QApplication

def pump(sec):
    end = time.time() + sec
    while time.time() < end:
        QCoreApplication.processEvents()
        time.sleep(0.01)

try:
    from gui import lnt_open
    lnt_open.build_lnt(las_path, lambda p, t: None, lambda: False)
    check("the .lnt builds from the LAS", os.path.isfile(lnt_path))

    app = QApplication(sys.argv)
    from gui.app_window import NakshaApp
    win = NakshaApp()
    win.show()
    pump(3.0)
    win.data = None

    lnt_open._show(win, lnt_path, "itest_cs.lnt")
    pump(3.0)
    view = getattr(win, "_lnt_main_view", None)
    check("the .lnt is shown in the native main viewport", view is not None and view.native is not None)
    nv = view.native
    check("the section API is available in the loaded DLL", nv.has_section_api())
    check("app.data is empty before any section (the native view holds no point array)",
          not win.data)

    # ---- screen <-> world round trip ----
    wsc = nv.world_to_screen(x0 + 200.0, y0 + 200.0)
    back = nv.screen_to_world(*wsc)
    check("world -> screen -> world round-trips on the 2D plan view",
          back is not None and abs(back[0] - (x0 + 200.0)) < 1e-6 and abs(back[1] - (y0 + 200.0)) < 1e-6,
          f"{back}")

    # frame the data so the section line is on screen
    nv.frame_region(x0 + 200.0, y0 + 200.0, 500.0)
    pump(1.0)

    # ---- 2. arm the Cross Section tool ----
    win.current_view = "top"
    win.enable_cross_section_mode()
    pump(1.0)
    picker = getattr(win, "cross_interactor", None)
    check("arming Cross Section installs the native picker", type(picker).__name__ == "LntCrossPicker",
          type(picker).__name__)

    w = view.owner.vulkan_widget
    dpr = max(w.devicePixelRatioF(), 1.0)
    def click_world(x, y):
        s = nv.world_to_screen(x, y)
        cur = getattr(win, "cross_interactor", None)       # the tool re-arms itself after a section
        cur.on_press(QPointF(s[0] / dpr, s[1] / dpr))

    # a section line across the wall (x = 200 is the wall), 100 m long, width 20 m each side
    P1 = (x0 + 150.0, y0 + 200.0)
    P2 = (x0 + 250.0, y0 + 200.0)
    click_world(*P1)
    click_world(*P2)
    check("two clicks arm the width step", picker.slice_state == 2, f"state {picker.slice_state}")
    click_world(P2[0], P2[1] + 20.0)          # width 20 m
    pump(2.0)

    data = win.data
    check("the section corridor was loaded into app.data from the file",
          isinstance(data, dict) and data.get("xyz") is not None and len(data["xyz"]) > 1000,
          f"{0 if not data else len(data['xyz']):,} points")
    store = getattr(win, "_lnt_section_store", None)
    check("a section store is active", store is not None)
    if store is not None and data:
        xyz = data["xyz"]
        inside = ((xyz[:, 0] >= P1[0] - 3) & (xyz[:, 0] <= P2[0] + 3) &
                  (np.abs(xyz[:, 1] - P1[1]) <= 25))
        check("the loaded points cover the corridor and stay near it",
              inside.sum() > 0 and xyz[:, 1].min() >= P1[1] - 30 and xyz[:, 1].max() <= P1[1] + 30,
              f"y range {xyz[:, 1].min() - y0:.1f}..{xyz[:, 1].max() - y0:.1f}")
        # exactness against the source LAS
        mask = ((xs >= xyz[:, 0].min() - 0.01) & (xs <= xyz[:, 0].max() + 0.01) &
                (ys >= xyz[:, 1].min() - 0.01) & (ys <= xyz[:, 1].max() + 0.01))
        mism = abs(len(xyz) - int(mask.sum()))
        check("point count equals the LAS points in the same box (nothing lost, nothing added)",
              mism <= max(2, int(0.002 * len(xyz))), f"{len(xyz):,} vs {int(mask.sum()):,}")

    sc = win.section_controller
    docks = getattr(win, "section_docks", {})
    check("a cross-section dock exists for the section", len(docks) > 0, f"docks {list(docks)}")
    vi = sc.active_view
    core = getattr(win, f"section_{vi}_core_points", None)
    check("the section's core points were computed for the active view",
          core is not None and len(core) > 100, f"{0 if core is None else len(core):,} core points")

    # ---- 3. edit classes in the section and write them back ----
    if store is not None and data is not None:
        idx = getattr(win, f"section_{vi}_core_indices", None)
        if idx is None:
            idx = getattr(win, f"section_{vi}_indices", None)
        check("the section knows which app.data points it shows", idx is not None and len(idx) > 100)
        if idx is not None:
            z = data["xyz"][idx, 2]
            pick = np.asarray(idx)[z > 108.0]                    # the top of the wall
            before = data["classification"][pick].copy()
            data["classification"][pick] = 9
            wrote = store.flush()
            check("class edits flush into the .lnt", wrote == len(pick) and len(pick) > 0,
                  f"{wrote} of {len(pick)} written")
            # read the FILE back through a fresh query
            res = nv.query_points((x0 - 5, y0 + 180.0, 0), (x0 + 405, y0 + 220.0, 1e6))
            top = res["xyz"][:, 2] > 108.0
            wallpts = (np.abs(res["xyz"][:, 0] - (x0 + 200.0)) < 1.5) & top
            sel_in_box = (res["xyz"][:, 0] >= P1[0] - 1e-6) & (res["xyz"][:, 0] <= P2[0] + 1e-6)
            changed = res["classification"][wallpts & sel_in_box]
            check("the file now holds the new class for the edited points",
                  len(changed) > 0 and np.all(changed == 9),
                  f"{int((changed == 9).sum())} of {len(changed)}")
            untouched = res["classification"][~(wallpts & sel_in_box)]
            check("points that were not edited kept their class", not np.any(untouched == 9),
                  f"{int((untouched == 9).sum())} unexpected")
            # ---- Cut Section on the cross-section view (runs on the same app.data indices) ----
            ctrl = win.cut_section_controller
            ctrl._cut_source = 'cross'
            ctrl.center_point = np.array([50.0, 0.0, 107.0])     # along = 50 m: the wall
            ctrl.dynamic_depth = 4.0
            try:
                ctrl._finalize_dynamic_cut_section()
            except Exception:
                traceback.print_exc()
            pump(2.0)
            cp = getattr(ctrl, "cut_points", None)
            cmap = getattr(ctrl, "_cut_index_map", None)
            check("Cut Section produced a cut from the cross-section view",
                  cp is not None and cmap is not None and len(cp) > 100 and len(cp) == len(cmap),
                  f"{0 if cp is None else len(cp):,} cut points")
            check("the cut view is active", bool(getattr(ctrl, "is_cut_view_active", False)))
            if cmap is not None and len(cmap):
                cx = win.data["xyz"][np.asarray(cmap, dtype=np.int64), 0]
                check("every cut point maps back to the right app.data point (within the cut depth)",
                      np.all(np.abs(cx - (x0 + 200.0)) <= 4.0 + 0.5) and int(np.asarray(cmap).max()) < len(win.data["xyz"]),
                      f"x {cx.min() - x0:.1f}..{cx.max() - x0:.1f}")
                sel = np.asarray(cmap, dtype=np.int64)[:50]
                win.data["classification"][sel] = 7
                try:
                    ctrl.onclassificationchanged(sel)
                except Exception:
                    traceback.print_exc()
                wrote = store.flush()
                check("a class edit made in the cut view is written into the .lnt",
                      wrote == len(sel), f"{wrote} of {len(sel)} written")
                res2 = nv.query_points((x0 - 5, y0 + 150, 0), (x0 + 405, y0 + 250, 1e6))
                keys_new = set(map(tuple, np.round(win.data["xyz"][sel], 3)))
                keys_file = {tuple(r) for r in np.round(res2["xyz"][res2["classification"] == 7], 3)}
                check("the file holds class 7 for exactly those points",
                      len(keys_new) > 0 and keys_new <= keys_file, f"{len(keys_new & keys_file)} of {len(keys_new)}")

            # a second section reuses the same store; indices of the first stay valid
            n_before = len(data["xyz"])
            check("the tool re-arms itself after a section (a fresh picker is installed on the view)",
                  type(getattr(win, "cross_interactor", None)).__name__ == "LntCrossPicker" and
                  view.picker is win.cross_interactor and win.cross_interactor is not picker)
            click_world(x0 + 100.0, y0 + 300.0)
            click_world(x0 + 120.0, y0 + 300.0)
            click_world(x0 + 120.0, y0 + 305.0)
            pump(2.0)
            check("a second section appends points without moving the first one's indices",
                  len(win.data["xyz"]) > n_before and
                  np.array_equal(win.data["classification"][pick], np.full(len(pick), 9, dtype=np.uint8)),
                  f"{n_before:,} -> {len(win.data['xyz']):,} points")

    # ---- 4. closing the view flushes and returns app.data ----
    win.data["classification"][:5] = 11
    lnt_open.close_lnt_view(win)
    pump(1.0)
    check("closing the .lnt view hands app.data back", not getattr(win, "_lnt_section_store", None) and not win.data)
except Exception:
    traceback.print_exc()
    results.append(False)

print(f"\n{sum(results)}/{len(results)} integration checks passed", flush=True)
os._exit(0 if all(results) else 1)
