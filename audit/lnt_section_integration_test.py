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

    # ---- shared gesture driver for the native classification tools ----
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    from gui.native_section_classify import select as tool_select
    win.brush_radius, win.brush_shape, win.brush_preview_px = 1.0, "circle", 12.0
    win.parallel_line_mode = "symmetric"            # skip the first-use settings dialogs
    ctx = {}

    def gesture(target, tool, pts):
        """Drive `tool` on `target` (a native canvas widget) with real Qt mouse input.
        Returns the shape the classifier saw, in its own coordinates."""
        cv = ctx["canvas"]
        qs = [QPoint(*map(round, cv.screen_from_section(*pt))) for pt in pts]
        seen = [cv.section_at(q.x(), q.y()) for q in qs]
        if tool in ("rectangle", "circle", "parallel_line"):            # press-drag-release
            QTest.mousePress(target, Qt.LeftButton, Qt.NoModifier, qs[0])
            QTest.mouseMove(target, qs[1])
            QTest.mouseRelease(target, Qt.LeftButton, Qt.NoModifier, qs[1])
        elif tool == "polygon":                                          # clicks + Enter
            for q in qs:
                QTest.mouseClick(target, Qt.LeftButton, Qt.NoModifier, q)
                pump(0.05)
            QTest.keyClick(target, Qt.Key_Return)
        elif tool in ("freehand", "brush"):
            QTest.mousePress(target, Qt.LeftButton, Qt.NoModifier, qs[0])
            for q in qs[1:]:
                QTest.mouseMove(target, q)
                pump(0.02)
            QTest.mouseRelease(target, Qt.LeftButton, Qt.NoModifier, qs[-1])
            if tool == "freehand":
                seen = seen + [seen[-1]]
        elif tool == "point":
            QTest.mouseClick(target, Qt.LeftButton, Qt.NoModifier, qs[0])
            seen = [seen[0], seen[0]]
        pump(0.3)
        return seen

    def radius_for(tool):
        cv = ctx["canvas"]
        if tool == "brush":
            return ctx["clf"].brush_radius_px() * cv.world_per_pixel()
        if tool == "point":
            return 4. * cv.world_per_pixel()
        return None

    # frame the data so the section line is on screen
    nv.frame_region(x0 + 200.0, y0 + 200.0, 500.0)
    pump(1.0)


    # ---- 1b. classification on the native MAIN plan view, before any section ----
    # (app.data is None here: the records live in the .lnt, not a VTK array)
    main_tests = {"rectangle": [(x0 + 20, y0 + 120), (x0 + 40, y0 + 140)],
                  "circle": [(x0 + 70, y0 + 130), (x0 + 78, y0 + 130)],
                  "polygon": [(x0 + 100, y0 + 120), (x0 + 130, y0 + 125), (x0 + 110, y0 + 145)],
                  "freehand": [(x0 + 150, y0 + 120), (x0 + 175, y0 + 122), (x0 + 170, y0 + 145), (x0 + 152, y0 + 140)],
                  "brush": [(x0 + 30, y0 + 160), (x0 + 36, y0 + 160), (x0 + 42, y0 + 162)],
                  "point": [(x0 + 80, y0 + 160)]}
    for k, (tool, pts) in enumerate(main_tests.items()):
        win.set_classify_tool(tool)
        pump(0.3)
        win.from_classes, win.to_class = None, 20 + k
        mc = getattr(view, "classifier", None)
        if mc is None or win.active_classify_tool != tool:
            check(f"main view: {tool} arms natively", False, f"active {win.active_classify_tool}")
            continue
        ctx["canvas"], ctx["clf"] = mc.canvas, mc
        vtk_attached = getattr(win, "classify_interactor", None) is not None
        res0 = nv.query_points((x0 - 1, y0 + 100, 0), (x0 + 200, y0 + 175, 1e6))
        seen = gesture(view.owner.vulkan_widget, tool, pts)
        pump(.06)  # the preview is immediate; file persistence is deferred
        res1 = nv.query_points((x0 - 1, y0 + 100, 0), (x0 + 200, y0 + 175, 1e6))
        same = len(res0["xyz"]) == len(res1["xyz"]) and np.array_equal(res0["xyz"], res1["xyz"])
        xy = res0["xyz"][:, :2]
        want = tool_select(tool, xy, seen, win, radius_for(tool))
        lo_h, hi_h, lo_z, hi_z = mc.canvas.visible_extent()
        want &= (xy[:, 0] >= lo_h) & (xy[:, 0] <= hi_h) & (xy[:, 1] >= lo_z) & (xy[:, 1] <= hi_z)
        shown = mc._visible_classes()
        if shown is not None:
            want &= np.isin(res0["classification"], shown)
        changed = res1["classification"] != res0["classification"]
        check(f"main view: {tool} classifies exactly its shape in the file (ribbon, no VTK)",
              same and not vtk_attached and want.sum() > 0 and np.array_equal(changed, want)
              and np.all(res1["classification"][want] == 20 + k),
              f"{int(want.sum())} expected, {int(changed.sum())} changed, vtk {vtk_attached}")
    win.set_classify_tool(None)
    from gui import lnt_section_bridge
    lnt_section_bridge.release(win)        # flush; the section checks below start from an empty store
    pump(0.2)
    check("main-view edits leave app.data empty once the store is released", not win.data)

    # ---- 2. arm the Cross Section tool ----
    win.current_view = "top"
    # The native tool asks for the target view in a modal dialog; answer it.
    from PySide6.QtWidgets import QInputDialog
    QInputDialog.getItem = staticmethod(lambda *a, **k: ("View 1", True))
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
            # ---- Native section dock: real Vulkan canvas, cut, classify ----
            check("the cross-section tool stays armed after a section",
                  type(getattr(win, "cross_interactor", None)).__name__ == "LntCrossPicker" and
                  view.picker is win.cross_interactor)
            ns = getattr(win, "_native_sections", None)
            check("the section dock is a native Vulkan canvas (no VTK widget)",
                  ns is not None and (vi + 1) in ns.views and ns.views[vi + 1].backend._handle,
                  type(ns).__name__)
            canvas = ns.views[vi + 1]
            pump(0.5)
            canvas.fit()
            check("the section dock is a floating window like the VTK docks",
                  ns.docks[vi + 1].isFloating() and ns.docks[vi + 1].isVisible())
            lp = getattr(win, "cross_interactor", None)
            ok = lp is not None and lp.slice_state == 0 and ns.locate(
                canvas, QPointF(*canvas.screen_from_section(50.0, 107.0)))
            pump(0.5)
            sc_xy = nv.world_to_screen(x0 + 200.0, y0 + 200.0)
            wdg = view.owner.vulkan_widget
            r = max(wdg.devicePixelRatioF(), 1.0)
            check("section locate seeds P1 at the clicked spot and centres the main view on it",
                  ok and np.allclose(lp.P1[:2], (x0 + 200.0, y0 + 200.0), atol=1e-6) and lp.slice_state == 1
                  and abs(sc_xy[0] / r - wdg.width() / 2) < 2 and abs(sc_xy[1] / r - wdg.height() / 2) < 2,
                  f"P1 {None if lp is None or lp.P1 is None else (lp.P1[:2] - (x0, y0)).round(3)} screen {sc_xy}")
            if lp is not None:
                lp.cancel()
            canvas.cut_centre = 50.0                              # along = 50 m: the wall
            ns.finish_cut(canvas, 52.0)                           # depth 4 m
            pump(1.0)
            from gui.native_sections import CUT_SLOT
            cut = ns.views.get(CUT_SLOT)
            cmap = None if cut is None else cut.indices
            check("Cut Section produced a native cut from the cross-section view",
                  cmap is not None and len(cmap) > 100, f"{0 if cmap is None else len(cmap):,} cut points")
            check("the cut view is active", bool(getattr(win.cut_section_controller, "is_cut_view_active", False)))
            if cmap is not None and len(cmap):
                cx = win.data["xyz"][np.asarray(cmap, dtype=np.int64), 0]
                check("every cut point maps back to the right app.data point (within the cut depth)",
                      np.all(np.abs(cx - (x0 + 200.0)) <= 2.0 + 1e-6) and int(np.asarray(cmap).max()) < len(win.data["xyz"]),
                      f"x {cx.min() - x0:.1f}..{cx.max() - x0:.1f}")
            # Exercise the real shortcut/controller routing without allowing
            # the nested operation to enter the legacy VTK implementation.
            import builtins
            real_import = builtins.__import__
            def native_cut_import(name, *args, **kwargs):
                assert name != 'vtk' and not name.startswith('vtkmodules'), name
                return real_import(name, *args, **kwargs)
            builtins.__import__ = native_cut_import
            try:
                win.cut_section_controller.activate_from_cut_shortcut()
                parent, parent_ids, axis = cut.local.copy(), cut.indices.copy(), cut.axis
                centre = float(np.median(parent[:, axis]))
                half = max(.1, float(np.ptp(parent[:, axis])) * .3)
                cut.cut_centre = centre
                ns.finish_cut(cut, centre + half)
                keep = np.abs(parent[:, axis] - centre) <= half + 1e-9
                check('Cut-in-Cut shortcut filters native records and rearms without VTK',
                      np.array_equal(cut.indices, parent_ids[keep]) and cut.cut_armed
                      and ns._nested_cut_active and cut.axis == 0)
                check('Cut-in-Cut rotates the native result 90 degrees',
                      np.allclose(cut.local, np.column_stack((parent[keep, 1-axis],
                                  -parent[keep, axis], parent[keep, 2]))))
                ids_before_cancel = cut.indices.copy()
                handled = win.cut_section_controller.cancel_persistent_cut_in_cut()
                check('Escape controller route exits native Cut-in-Cut and preserves result',
                      handled and not cut.cut_armed and np.array_equal(cut.indices, ids_before_cancel))
            finally:
                builtins.__import__ = real_import

            # ---- Above / Below Line from the Classify ribbon, on the native dock ----
            from PySide6.QtCore import QPoint
            from PySide6.QtTest import QTest
            canvas.fit()
            pump(0.3)
            win.from_classes = None
            win.classify_interactors = {}
            win.set_classify_tool("above_line")
            pump(0.3)
            win.to_class = 13                      # picked in the Class Picker after arming
            check("Above Line arms on a native section (no VTK interactor attached)",
                  win.active_classify_tool == "above_line" and not win.classify_interactors
                  and canvas.classifier.active())

            loc = canvas.local[:, [canvas.axis, 2]]
            ids = canvas.indices
            shown = canvas.classifier._visible_classes()

            def expected(tool, a, b):
                """The tool's own rule on the pixel-rounded line ends, limited to shown classes."""
                (x1, z1), (x2, z2) = a, b
                line_z = (z2 - z1) / (x2 - x1 + 1e-9) * (loc[:, 0] - x1) + z1
                sel = (loc[:, 0] >= min(x1, x2)) & (loc[:, 0] <= max(x1, x2))
                sel &= (loc[:, 1] >= line_z) if tool == "above_line" else (loc[:, 1] <= line_z)
                if shown is not None:
                    sel &= np.isin(win.data["classification"][ids], shown)
                return sel

            def to_px(pt):
                q = QPoint(*map(round, canvas.screen_from_section(*pt)))
                return q, canvas.section_at(q.x(), q.y())

            # press-drag-release form
            (pa, a), (pb, b) = to_px((20.0, 101.5)), to_px((40.0, 101.5))
            cls_before = win.data["classification"][ids].copy()
            above_sel = expected("above_line", a, b)
            QTest.mousePress(canvas, Qt.LeftButton, Qt.NoModifier, pa)
            QTest.mouseMove(canvas, pb)
            shapes = list(canvas.classifier.overlay.shapes)
            check("the Above Line preview draws the line with upward guides",
                  len(shapes) == 1 and shapes[0][0] == "guided_line" and shapes[0][2][0] is True,
                  f"{[(k, e[0] if e else e) for k, _, e in shapes]}")
            QTest.mouseRelease(canvas, Qt.LeftButton, Qt.NoModifier, pb)
            pump(0.3)
            after = win.data["classification"][ids]
            check("Above Line classifies exactly the shown points above the drawn line",
                  above_sel.sum() > 0 and np.all(after[above_sel] == 13)
                  and np.array_equal(after[~above_sel], cls_before[~above_sel]),
                  f"{int(above_sel.sum())} selected, {int((after != cls_before).sum())} changed, "
                  f"status '{win.statusBar().currentMessage()}'")
            from gui.native_main_classify import ensure_classifier
            main_cls = ensure_classifier(view)
            check("the plan view refuses line tools (section-only, as in VTK)",
                  main_cls is not None and not main_cls.active())

            win.set_classify_tool("below_line")
            pump(0.3)
            win.to_class = 14
            # click-click form: press/release at P1, then at P2
            (pa, a), (pb, b) = to_px((0.0, 101.0)), to_px((30.0, 101.0))
            cls_before = win.data["classification"][ids].copy()
            below_sel = expected("below_line", a, b)
            for p in (pa, pb):
                QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, p)
                pump(0.2)
            after = win.data["classification"][ids]
            check("Below Line classifies exactly the shown points below the drawn line",
                  below_sel.sum() > 0 and np.all(after[below_sel] == 14)
                  and np.array_equal(after[~below_sel], cls_before[~below_sel]),
                  f"{int(below_sel.sum())} selected, {int((after != cls_before).sum())} changed")
            store.flush()
            res = nv.query_points((x0 - 5, y0 + 180.0, 0), (x0 + 405, y0 + 220.0, 1e6))
            check("line-tool edits are written into the .lnt",
                  int((res["classification"] == 13).sum()) >= int(above_sel.sum())
                  and int((res["classification"] == 14).sum()) >= int(below_sel.sum()),
                  f"13: {int((res['classification'] == 13).sum())}, 14: {int((res['classification'] == 14).sum())}")

            # ---- every other tool from the ribbon on the native section dock ----
            ctx["canvas"], ctx["clf"] = canvas, canvas.classifier
            shown = canvas.classifier._visible_classes()
            sec_tests = {"parallel_line": [(5.0, 100.5), (25.0, 101.5)],
                         "rectangle": [(60.0, 100.0), (66.0, 102.0)],
                         "circle": [(70.0, 101.0), (71.5, 101.0)],
                         "polygon": [(74.0, 100.0), (80.0, 100.2), (77.0, 102.0)],
                         "freehand": [(82.0, 100.0), (86.0, 100.2), (85.0, 102.0), (82.5, 101.6)],
                         "brush": [(88.0, 101.0), (89.0, 101.0), (90.0, 101.2)],
                         "point": [(93.0, 101.0)]}
            for k, (tool, pts) in enumerate(sec_tests.items()):
                win.set_classify_tool(tool)
                pump(0.3)
                win.from_classes, win.to_class = None, 30 + k
                ok_armed = win.active_classify_tool == tool and canvas.classifier.active()
                loc = canvas.local[:, [canvas.axis, 2]]
                ids = canvas.indices
                before = win.data["classification"][ids].copy()
                seen = gesture(canvas, tool, pts)
                want = tool_select(tool, loc, seen, win, radius_for(tool))
                lo_h, hi_h, lo_z, hi_z = canvas.visible_extent()
                want &= (loc[:, 0] >= lo_h) & (loc[:, 0] <= hi_h) & (loc[:, 1] >= lo_z) & (loc[:, 1] <= hi_z)
                if shown is not None:
                    want &= np.isin(before, shown)
                after = win.data["classification"][ids]
                check(f"section dock: {tool} classifies exactly its shape (ribbon, no VTK)",
                      ok_armed and not win.classify_interactors and want.sum() > 0
                      and np.array_equal(after != before, want) and np.all(after[want] == 30 + k),
                      f"{int(want.sum())} expected, {int((after != before).sum())} changed")
            store.flush()
            res = nv.query_points((x0 - 5, y0 + 180.0, 0), (x0 + 405, y0 + 220.0, 1e6))
            got = {t: int((res["classification"] == 30 + k).sum()) for k, t in enumerate(sec_tests)}
            check("section tool edits are written into the .lnt", all(got.values()), f"{got}")
            win.set_classify_tool(None)

            # a second section reuses the same store; indices of the first stay valid
            # (arming a classify tool disarms Cross Section, as with VTK: re-arm it)
            win.enable_cross_section_mode()
            pump(1.0)
            n_before = len(data["xyz"])
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
