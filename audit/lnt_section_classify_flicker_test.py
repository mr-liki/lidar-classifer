"""Cross-section classify: no viewport flicker, incremental recolour, speed.

Real Vulkan frames. A classification edit in Shaded/Surface must keep the
resident mesh on screen (never clear it, never retriangulate), must land on
exactly the colours a fresh build would produce, and must stay fast. Switching
display modes keeps the previous image until the new mesh is uploaded.

Run: venv/Scripts/python.exe audit/lnt_section_classify_flicker_test.py
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from gui import native_sections
from gui.native_sections import NativeSections, CUT_SLOT
from gui.native_section_display import build_mesh_data

qt = QApplication([])
app = QMainWindow()
app.resize(1000, 760)
app.show()

# Shaded/Surface build a TIN from the XY footprint, so the audit uses a
# real (tilted) ground plane, exactly like the display audit.
gx, gy = np.meshgrid(np.linspace(0., 10., 81), np.linspace(-2., 2., 25))
xyz = np.column_stack((gx.ravel(), gy.ravel(), (.3 * gx + .7 * gy).ravel()))
classes = np.where(xyz[:, 0] < 5., 2, 6).astype(np.uint8)
palette = {2: dict(color=(240, 40, 20), show=True, weight=1.),
           6: dict(color=(20, 220, 80), show=True, weight=1.)}
store = SimpleNamespace(
    data=dict(xyz=xyz, classification=classes,
              intensity=np.linspace(1, 255, len(xyz), dtype=np.float32),
              rgb=np.full((len(xyz), 3), .5, dtype=np.float32)),
    rgb8=np.full((len(xyz), 3), 128, dtype=np.uint8), generation=0)
store.show = lambda *args, **kwargs: None
app._lnt_main_view = SimpleNamespace()
app.view_palettes, app.class_palette, app.data = {}, palette, store.data
app.undo_stack, app.redo_stack = [], []
sections = NativeSections(app)
app._native_sections = sections


def wait_for(test, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qt.processEvents()
        if test():
            return True
        time.sleep(.01)
    raise AssertionError('Timed out: ' + app.statusBar().currentMessage())


def present(canvas, frames=3):
    for _ in range(frames):
        canvas.render()
        qt.processEvents()
        time.sleep(.005)
    return canvas.backend.capture_frame()


def occupied(frame):
    return frame is not None and int(np.ptp(frame[:, :, :3])) > 10


def classify(rows, to_class=6):
    """Mirror native_section_classify._commit: classes, then refresh."""
    rows = np.asarray(rows, dtype=np.int64)
    old = classes[rows].copy()
    classes[rows] = to_class
    store.generation += 1
    start = time.perf_counter()
    sections.refresh_edits(force=True, changed=(rows, old))
    return time.perf_counter() - start


def band(lo, hi):
    return np.flatnonzero((xyz[:, 0] >= lo) & (xyz[:, 0] < hi))


X_EDGES = (0., 2., 4., 6., 8., 10.)


def fresh_build(canvas, mask):
    options = canvas._mesh_options(app, canvas.mode)
    idx = canvas.indices
    xyz_masked = np.ascontiguousarray(store.data['xyz'][idx][mask], dtype=np.float64)
    options['z_bounds'] = tuple(np.percentile(xyz_masked[:, 2], [1., 99.]))
    display = canvas.local if canvas.axis == 0 else canvas.local[:, [1, 0, 2]]
    return build_mesh_data(xyz_masked, np.ascontiguousarray(display[mask]),
                           classes[mask].copy(), canvas.palette, canvas.mode, options)


status = 0
try:
    with patch('gui.native_sections.get_store', return_value=store):
        canvas = sections.ensure_dock(1)
        canvas.resize(560, 460)
        canvas.set_points(xyz, np.arange(len(xyz)), axis=0)
        # finalize() seeds this cache; the audit drives set_points directly.
        sections._classes[1] = classes[canvas.indices].copy()
        canvas.fit(sync=False)
        canvas.backend.set_point_size(3.)
        wait_for(lambda: canvas.backend.present_count > 0)

        cleared = [0]
        real_clear = canvas.backend.clear_surface
        canvas.backend.clear_surface = lambda: (cleared.__setitem__(0, cleared[0] + 1),
                                                real_clear())[1]

        # ---- Shaded Class: recolour in place, never clear / retriangulate ----
        canvas.apply_palette(palette, 'shaded_class')
        assert wait_for(lambda: canvas._surface_loaded), 'shaded mesh never appeared'
        assert occupied(present(canvas)), 'shaded mesh rendered nothing'
        builds = canvas._mesh_build_count
        recolours = canvas._mesh_recolor_count
        cleared[0] = 0
        before = present(canvas)
        timings = []
        for step, (lo, hi) in enumerate(zip(X_EDGES[:-1], X_EDGES[1:])):
            timings.append(classify(band(lo, hi)))
            frame = present(canvas)
            assert occupied(frame), f'shaded_class went blank at step {step}'
            assert canvas._surface_loaded, 'recolour must keep the mesh live'
        assert cleared[0] == 0, 'classify cleared the mesh'
        assert canvas._mesh_build_count == builds, 'classify retriangulated the section'
        assert canvas._mesh_recolor_count == recolours + len(timings), 'recolour did not run'
        assert not np.array_equal(before, present(canvas)), 'recoloured frame is unchanged'
        # Exactly what a full rebuild would have produced (same geometry, same colours).
        mask = np.ones(len(canvas.indices), dtype=bool)
        fresh = fresh_build(canvas, mask)
        resident = canvas._mesh_data
        np.testing.assert_array_equal(fresh['faces'], resident['faces'])
        np.testing.assert_array_equal(fresh['positions'], resident['positions'])
        np.testing.assert_array_equal(fresh['colors'], resident['colors'])
        shaded_frame = present(canvas)
        print(f'PASS shaded_class recolour: {len(timings)} edits, '
              f'max {max(timings)*1000:.1f} ms, identical to a fresh build', flush=True)

        # ---- The real Classify tool (rectangle click-drag) on the mesh ------
        from PySide6.QtCore import QPointF, Qt
        app.to_class, app.from_classes = 2, None
        app.active_classify_tool = 'rectangle'
        before_tool = present(canvas)
        recolours = canvas._mesh_recolor_count
        cleared[0] = 0
        p0 = QPointF(*canvas.screen_from_section(1.5, -1.5))
        p1 = QPointF(*canvas.screen_from_section(4.5, 1.5))
        assert canvas.classifier.press(p0, Qt.LeftButton)
        canvas.classifier.release(p1, Qt.LeftButton)
        frame = present(canvas)
        assert occupied(frame), 'classify tool blanked the shaded viewport'
        assert canvas._surface_loaded and cleared[0] == 0
        assert canvas._mesh_build_count == builds
        assert canvas._mesh_recolor_count == recolours + 1
        assert not np.array_equal(before_tool, frame), 'tool edit did not recolour'
        app.active_classify_tool = None
        print('PASS rectangle classify tool recolours the resident mesh in place', flush=True)

        # ---- Surface: a class edit cannot change a pixel, and no rebuild -----
        canvas.apply_palette(palette, 'surface')
        assert wait_for(lambda: canvas._surface_loaded), 'surface mesh never appeared'
        assert occupied(present(canvas)), 'surface mesh rendered nothing'
        surface_builds = canvas._mesh_build_count
        surface_recolours = canvas._mesh_recolor_count
        cleared[0] = 0
        before_surface = present(canvas)
        surface_timings = [classify(band(3., 7.)), classify(band(7., 9.5))]
        assert cleared[0] == 0, 'surface classify cleared the mesh'
        assert canvas._mesh_build_count == surface_builds, 'surface classify retriangulated'
        after_surface = present(canvas)
        assert occupied(after_surface)
        assert np.array_equal(before_surface, after_surface), \
            'surface pixels changed on a class-only edit'
        assert not np.array_equal(shaded_frame, after_surface), \
            'shaded and surface frames should differ'
        mask = np.ones(len(canvas.indices), dtype=bool)
        fresh = fresh_build(canvas, mask)
        np.testing.assert_array_equal(fresh['colors'], canvas._mesh_data['colors'])
        print(f'PASS surface classify: pixels untouched, '
              f'max {max(surface_timings)*1000:.1f} ms, no rebuild', flush=True)

        # ---- Point modes: no XYZ re-upload, no blank frame -------------------
        for mode in ('class', 'rgb', 'intensity', 'elevation', 'depth'):
            canvas.apply_palette(palette, mode)
            assert wait_for(lambda: canvas.backend.present_count > 0)
            assert occupied(present(canvas)), mode + ' rendered nothing'
            uploads = canvas.backend.get_point_position_upload_count()
            cleared[0] = 0
            classify(band(0., 1.5))
            frame = present(canvas)
            assert occupied(frame), mode + ' went blank after classify'
            assert canvas.backend.get_point_position_upload_count() == uploads, \
                mode + ' re-uploaded XYZ on a class edit'
            assert cleared[0] == 0, mode + ' cleared a mesh it never owned'
            print(f'PASS {mode} classify: no XYZ re-upload, frame populated', flush=True)

        # ---- Mode switches never blank the viewport mid-build ----------------
        real_build = native_sections.build_mesh_data

        def slow_build(*args, **kwargs):
            time.sleep(.35)
            return real_build(*args, **kwargs)

        canvas.apply_palette(palette, 'intensity')
        present(canvas)
        landed = {}
        with patch('gui.native_sections.build_mesh_data', slow_build):
            for mode in ('shaded_class', 'surface', 'shaded_class'):
                # Display Mode Apply goes through NativeSections.apply().
                assert sections.apply(canvas.slot, palette, mode)
                assert not canvas._surface_loaded, 'mode switch must wait for the new mesh'
                frames = []
                for _ in range(6):
                    frames.append(present(canvas, frames=1))
                    time.sleep(.04)
                assert all(occupied(f) for f in frames), mode + ' blanked while building'
                assert wait_for(lambda: canvas._surface_loaded), mode + ' mesh never landed'
                landed[mode] = present(canvas)
                assert occupied(landed[mode]), mode + ' rendered nothing after the switch'
        assert not np.array_equal(landed['shaded_class'], landed['surface']), \
            'shaded and surface must render differently after landing'
        print('PASS mode switches keep the previous frame until the new mesh is up', flush=True)

        # ---- Speed: a large section's edits stay interactive ----------------
        bx, by = np.meshgrid(np.linspace(0., 20., 501), np.linspace(-2., 2., 201))
        big_xyz = np.column_stack((bx.ravel(), by.ravel(), (.3 * bx + .7 * by).ravel()))
        big = dict(xyz=big_xyz,
                   classification=np.where(big_xyz[:, 0] < 10., 2, 6).astype(np.uint8),
                   intensity=np.linspace(1, 255, len(big_xyz), dtype=np.float32),
                   rgb=np.full((len(big_xyz), 3), .5, dtype=np.float32))
        big_store = SimpleNamespace(data=big, rgb8=np.full((len(big_xyz), 3), 128, np.uint8),
                                    generation=0)
        big_store.show = lambda *args, **kwargs: None
        # Its own manager so the small section above is not refreshed with the
        # big store's rows.
        big_sections = NativeSections(app)
        big_canvas = big_sections.ensure_dock(1)
        big_canvas.resize(560, 460)
        with patch('gui.native_sections.get_store', return_value=big_store):
            big_canvas.set_points(big_xyz, np.arange(len(big_xyz)), axis=0)
            big_sections._classes[1] = big['classification'][big_canvas.indices].copy()
            big_canvas.fit(sync=False)
            big_canvas.apply_palette(palette, 'shaded_class')
            assert wait_for(lambda: big_canvas._surface_loaded, 180)
            times = []
            geometry_uploads = big_canvas.backend.get_surface_upload_count()
            for step, lo in enumerate((1., 4., 7., 10.)):
                rows = np.flatnonzero((big_xyz[:, 0] >= lo) & (big_xyz[:, 0] < lo + 1.5))
                old = big['classification'][rows].copy()
                big['classification'][rows] = 2 if step % 2 else 6
                start = time.perf_counter()
                big_sections.refresh_edits(force=True, changed=(rows, old))
                times.append(time.perf_counter() - start)
                assert big_canvas._surface_loaded
            assert big_canvas._mesh_build_count == 1, 'large classify rebridged the mesh'
            # Contrast: dropping the geometry cache forces the old path's full
            # retriangulation for the same section.
            if hasattr(big_canvas.backend._dll, 'nkv_update_surface_colors'):
                assert big_canvas.backend.get_surface_upload_count() == geometry_uploads, \
                    'class edits re-uploaded surface geometry'
            start = time.perf_counter()
            big_canvas._drop_mesh_cache()
            big_canvas.apply_palette(palette, 'shaded_class')
            assert wait_for(lambda: big_canvas._surface_loaded, 180)
            rebuild = time.perf_counter() - start
            print(f'PASS large section ({len(big_xyz):,} pts, '
                  f'{len(big_canvas._mesh_data["faces"]):,} faces): classify '
                  f'{min(times)*1000:.0f}-{max(times)*1000:.0f} ms per edit '
                  f'(full retriangulation: {rebuild*1000:.0f} ms)', flush=True)
            assert max(times) < 1.5, f'classify too slow: {max(times):.2f}s'
            big_sections.close()

    print('ALL SECTION CLASSIFY FLICKER CHECKS PASSED', flush=True)
    for dock in list(sections.docks.values()):
        dock.close()
    sections.close()
except BaseException:
    import traceback
    traceback.print_exc()
    status = 1
finally:
    sys.stdout.flush()
    sys.stderr.flush()
sys.exit(status)
