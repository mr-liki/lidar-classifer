"""Real Vulkan multi-viewport and exact corridor/cut regression checks."""
from pathlib import Path
import sys
import os
from types import SimpleNamespace
from unittest.mock import Mock
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from gui.native_sections import NativeSections, section_geometry, CUT_SLOT
from gui import lnt_native

qapp = QApplication([])
app = QMainWindow()
app.resize(1400, 800)
main_canvas = QWidget(app)
main_canvas.setAttribute(__import__('PySide6.QtCore', fromlist=['Qt']).Qt.WA_NativeWindow)
app.setCentralWidget(main_canvas)
app.show()
qapp.processEvents()
if os.environ.get('NAKSHA_VULKAN_VALIDATION') == '1':
    from gui.render_backend import VulkanRenderBackend
    main_backend = VulkanRenderBackend()
    assert main_backend.initialize(int(main_canvas.winId()), 800, 600)
    native = SimpleNamespace(has_section_api=lambda: True, frame=lambda dt: main_backend.request_render(), close=main_backend.shutdown)
else:
    native = lnt_native.NativeView(int(main_canvas.winId()), 800, 600,
                                   str(ROOT/'native/lnt/test_shading.lnt'), 64, 8.)
app._lnt_main_view = SimpleNamespace(native=native)
app.section_controller = SimpleNamespace(active_view=0)
app.section_buffer = 0.
app.view_palettes = {}
app.class_palette = {2: {'color': (255, 120, 0), 'show': True},
                     5: {'color': (0, 255, 0), 'show': True}}
xyz = np.array([[0,0,10], [1,0,11], [2,0,12], [3,0,13], [4,0,14],
                [2,1,12], [2,-1,12], [2,1.0001,12], [-1,0,10]], dtype=np.float64)
store = SimpleNamespace(view=app._lnt_main_view, ensure_for_section=Mock(return_value=True),
                        data={'xyz': xyz, 'classification': np.array([2,5,2,5,2,5,2,5,2], dtype=np.uint8),
                              'rgb': np.ones((9,3), dtype=np.float32),
                              'intensity': np.arange(9, dtype=np.float32)})
app._lnt_section_store = store
store.show = Mock()
app.data = store.data
sections = NativeSections(app)
app._native_sections = sections
try:
    indices, local, core = section_geometry(xyz, (0,0,0), (4,0,0), 1., 0.)
    assert indices.tolist() == [0,1,2,3,4,5,6]
    assert core.all()
    assert np.array_equal(local[:,2], xyz[indices,2])
    print('PASS corridor includes every boundary record, preserves exact source indices', flush=True)
    for index in range(8):
        sections.open(index, draw=False)
        sections.finalize((0,0,0), (4,0,0), 1., index, 0.)
        sections.views[index+1].resize(240,180)
        qapp.processEvents()
        sections.views[index+1].render()
        native.frame(.016)
    assert len({v.backend._handle for v in sections.views.values()}) == 8
    if hasattr(native, 'set_class_weights'):
        main_weights = np.ones(256, dtype=np.float32)
        main_weights[2] = 5.0
        assert native.set_class_weights(main_weights), \
            'main .lnt view must accept the per-class weight table (nkv_set_class_weights)'
        print('PASS main native view accepts per-class weights through lnt_view_set_class_weights', flush=True)
    canvas = sections.views[8]
    print('DEBUG section:', canvas.isVisible(), canvas.centre, canvas.scale, canvas.backend.get_frame_stats(), canvas.backend.get_point_position_upload_count(), canvas.backend.get_point_count(), canvas.backend.get_render_origin(), flush=True)
    canvas._dirty = True
    canvas.render()
    qapp.processEvents()
    import time as _time
    _time.sleep(0.4)
    qapp.processEvents()
    canvas.backend.set_camera_view_up(0.,0.,1.)
    canvas.backend.set_camera_ortho(canvas.centre,(0.,1.,0.),canvas.scale)
    before = canvas.backend.get_frame_stats()
    canvas.backend.request_render()
    _time.sleep(0.4)
    after = canvas.backend.get_frame_stats()
    presented = (after[0] if isinstance(after, (list, tuple)) else after) - (before[0] if isinstance(before, (list, tuple)) else before)
    print('DEBUG presented', presented, before, after, flush=True)
    assert presented >= 1, 'Section viewport did not present a frame'
    assert canvas.backend.get_point_count() == 7
    assert canvas.backend.get_point_position_upload_count() >= 1
    source = sections.views[8]
    sections.open(7)
    picker = app._lnt_main_view.picker
    assert app.cross_section_active
    assert sections.arm_cut(source_slot=8)
    assert picker._cleaned_up and app._lnt_main_view.picker is None
    assert not app.cross_section_active and app.cross_interactor is None
    assert source.cut_armed and app.cut_section_mode_on
    sections.open(7)
    assert not source.cut_armed and not app.cut_section_mode_on
    assert app.cross_section_active and app._lnt_main_view.picker is not None
    assert sections.arm_cut(source_slot=8)
    print('PASS cross/cut activation is mutually exclusive in both directions', flush=True)
    source.cut_centre = 2.
    source.cut_armed = True
    from PySide6.QtCore import QPointF
    source.mouseMoveEvent(SimpleNamespace(position=lambda: QPointF(*source.screen_from_section(2.6, 12.)),
                                          buttons=lambda: Qt.NoButton))
    assert source.classifier.overlay.shapes[0][0] == 'cut_preview'
    sections.finish_cut(source, 2.6)
    cut = sections.views[CUT_SLOT]
    assert cut.indices.tolist() == [2,5,6]
    assert cut.axis == 1
    assert not source.classifier.overlay.marks
    assert not source.classifier.overlay.shapes, 'completed cut clears its preview'
    assert not cut.classifier.overlay.marks
    from PySide6.QtWidgets import QPushButton, QDoubleSpinBox
    cut_frame = sections.docks[CUT_SLOT].widget()
    assert not cut_frame.findChildren(QPushButton)
    assert not cut_frame.findChildren(QDoubleSpinBox)
    assert cut_frame.layout().count() == 1
    assert np.allclose(sections.world_from_section(8, 3., 12.5), (3., 0., 12.5))
    from PySide6.QtCore import QPointF
    picker = SimpleNamespace(slice_state=0, P1=None)
    app._lnt_main_view.picker = picker
    app.statusBar()
    assert sections.locate(source, QPointF(*source.screen_from_section(3., 12.)))
    assert picker.slice_state == 1 and np.allclose(picker.P1, (3., 0., 12.))
    assert not sections.locate(cut, QPointF(5, 5)), 'cut view never locates'
    app._lnt_main_view.picker = None
    print('PASS cut preview clears on completion; cut window is viewport-only; section locate seeds the next P1', flush=True)
    cut.cut_centre = 0.
    sections.finish_cut(cut, .1)
    assert cut.indices.tolist() == [2]
    assert not any(v.classifier.overlay.marks for v in sections.views.values()), 'nested cut has no cross source'
    assert len(sections.views) == 9
    sections.update_cut_depth(2.)
    assert cut.indices.tolist() == [2,5,6]
    assert np.array_equal(cut.local[:,2], xyz[cut.indices,2])
    print('PASS eight cross sections plus one cut; nested cuts preserve identities', flush=True)
    for slot, canvas in sections.views.items():
        uploads = canvas.backend.get_point_position_upload_count()
        weight_pushes = canvas.backend.get_class_weight_update_count()
        canvas.apply_palette({2: {'color': (255,0,0), 'show': True, 'weight': 4.0},
                              5: {'color': (0,255,0), 'show': False, 'weight': 1.0}}, 'class')
        canvas.centre[0] += .1
        canvas.scale *= .9
        canvas._dirty = True
        canvas.render()
        native.frame(.016)
        assert canvas.backend.get_point_position_upload_count() == uploads
        assert canvas.backend.get_class_weight_update_count() == weight_pushes + 1, \
            'apply_palette must push the per-class weight table'
    for mode in ('rgb', 'intensity', 'elevation', 'depth', 'shaded_class', 'class'):
        cut.apply_palette(app.class_palette, mode)
        cut.render()
        native.frame(.016)
    print('PASS native section color modes and shading; palette/weight changes do not upload positions', flush=True)
    # Pixel proof of the Weight column: raising ONE class's weight must grow
    # only that class's points. Captured offscreen through nkv_capture_frame
    # (same frame-slot UBO and camera the live view uses), so this is a real
    # GPU measurement, not a state assertion.
    probe = sections.views[1]
    def _red_pixels(frame):
        return int(((frame[..., 0] > 200) & (frame[..., 1] < 80) & (frame[..., 2] < 80)).sum())
    probe.backend.set_point_size(3.)
    probe.apply_palette({2: {'color': (255, 0, 0), 'show': True, 'weight': 1.0},
                         5: {'color': (0, 255, 0), 'show': True, 'weight': 1.0}}, 'class')
    probe._dirty = True
    probe.render()
    before_w = probe.backend.capture_frame()
    assert before_w is not None, 'offscreen capture unavailable'
    uploads_w = probe.backend.get_point_position_upload_count()
    probe.apply_palette({2: {'color': (255, 0, 0), 'show': True, 'weight': 8.0},
                         5: {'color': (0, 255, 0), 'show': True, 'weight': 1.0}}, 'class')
    probe._dirty = True
    probe.render()
    after_w = probe.backend.capture_frame()
    assert after_w is not None
    assert probe.backend.get_point_position_upload_count() == uploads_w, \
        'a weight change must never re-upload positions'
    red_before, red_after = _red_pixels(before_w), _red_pixels(after_w)
    assert red_before > 0 and red_after > red_before, (
        f'weight 8x must enlarge class-2 points (red pixels {red_before} -> {red_after})')
    print(f'PASS weight->size pixel proof: class-2 red coverage {red_before} -> {red_after} px at 8x (no position upload)', flush=True)
    store.data['classification'][2] = 5
    sections.refresh_edits()
    assert sections._classes[CUT_SLOT][0] == 5
    print('PASS independent palette/visibility, camera and class attribute updates', flush=True)
    # A chain/cycle joins all eight native views into one bidirectional group.
    app.view_sync_map = {}
    before_sync = {s: v.backend.get_point_position_upload_count() for s, v in sections.views.items()}
    sections.apply_sync_map({slot-1: slot-2 for slot in range(2, 9)})
    for slot in range(2, 9):
        assert sections.views[slot].backend.get_point_position_upload_count() == before_sync[slot]+1
    assert sections.views[1].backend.get_point_position_upload_count() == before_sync[1]
    sections.set_view_sync(1, 8)
    assert sections.linked_slots(8) == set(range(1, 8))
    uploads = {s: v.backend.get_point_position_upload_count() for s, v in sections.views.items()}
    driver = sections.views[8]
    driver.drag = QPointF(100, 100)
    driver.mouseMoveEvent(SimpleNamespace(position=lambda: QPointF(110, 105),
                                          buttons=lambda: Qt.NoButton))
    driver.drag = None
    driver.wheelEvent(SimpleNamespace(angleDelta=lambda: QPoint(0, 120), accept=lambda: None))
    for slot in range(1, 9):
        assert np.array_equal(sections.views[slot].centre, driver.centre)
        assert sections.views[slot].scale == driver.scale
        assert sections.views[slot].backend.get_point_position_upload_count() == uploads[slot]
    sections.views[3].fit()
    assert all(v.scale == sections.views[3].scale for s, v in sections.views.items() if s != CUT_SLOT)
    # Display state is per-view: a palette/mode change on one grouped view must
    # not repaint its synced peers (sync mirrors data and camera only).
    peers_before = {s: (dict(v.palette), v.mode) for s, v in sections.views.items()
                    if s != CUT_SLOT and v is not driver}
    driver.apply_palette({2: {'color': (80, 90, 100), 'show': False},
                          5: {'color': (10, 20, 30), 'show': True}}, 'intensity')
    assert driver.mode == 'intensity' and driver.palette[2]['show'] is False
    for s, (palette, mode) in peers_before.items():
        peer = sections.views[s]
        assert peer.mode == mode and peer.palette == palette, (
            f'View {s} display state changed by a View {driver.slot} palette apply')
    # Display Mode Apply entry point: only the selected slot changes even while
    # every view belongs to one synced group.
    before_apply = {s: (dict(v.palette), v.mode) for s, v in sections.views.items() if s != CUT_SLOT}
    weight_counts = {s: v.backend.get_class_weight_update_count()
                     for s, v in sections.views.items()}
    sections.apply(3, {2: {'color': (7, 8, 9), 'show': False, 'weight': 9.0},
                       5: {'color': (1, 2, 3), 'show': False, 'weight': 0.2}}, 'class')
    target = sections.views[3]
    assert target.mode == 'class' and target.palette[2]['show'] is False
    assert target.palette[2]['weight'] == 9.0 and target.palette[5]['weight'] == 0.2
    assert target.backend.get_class_weight_update_count() == weight_counts[3] + 1, \
        'weights must reach the target slot renderer'
    for s, (palette, mode) in before_apply.items():
        if s == 3:
            continue
        peer = sections.views[s]
        assert peer.mode == mode and peer.palette == palette, (
            f'View {s} display state leaked from a Display Mode apply to View 3')
        assert peer.backend.get_class_weight_update_count() == weight_counts[s], (
            f'View {s} renderer received a weight table from a View 3 apply')
    print('PASS per-view display apply: palette/mode/weights never leak to synced peers', flush=True)
    sections.change_axis(4, 1)
    assert all(v.axis == 1 for s, v in sections.views.items() if s != CUT_SLOT)
    sections.finalize((0,0,0), (3,0,0), 1., 5, .5)
    driver = sections.views[6]
    for slot in range(1, 9):
        peer = sections.views[slot]
        assert np.array_equal(peer.indices, driver.indices)
        assert np.array_equal(peer.local, driver.local)
        assert sections.geometry[slot][3] == .5
        assert np.array_equal(getattr(app, f'section_{slot-1}_core_indices'), app.section_5_core_indices)
    edited = int(driver.indices[0])
    old_class = int(store.data['classification'][edited])
    for value in (6, old_class):
        store.data['classification'][edited] = value
        sections.refresh_edits(force=True)
        for slot in range(1, 9):
            assert sections._classes[slot][0] == value
    sections.finalize((100,100,0), (104,100,0), .1, 5, 0.)
    assert all(not len(v.indices) for s, v in sections.views.items() if s != CUT_SLOT)
    sections.finalize((0,0,0), (3,0,0), 1., 5, .5)
    sections.set_view_sync(1, None)
    sections.set_view_sync(2, None)
    detached = sections.views[1].scale
    driver.wheelEvent(SimpleNamespace(angleDelta=lambda: QPoint(0, 120), accept=lambda: None))
    assert sections.views[1].scale == detached
    app.view_sync_map.clear()
    print('PASS eight-view native sync: chains/cycles, bidirectional pan/zoom/fit, axis, section/buffer, class edits/restoration, empty sections, unlink; camera changes never upload points', flush=True)
    cut._dirty = True
    cut.render()
    _time.sleep(0.3)
    assert cut.backend.get_point_count() == len(cut.indices) > 0
    # Destroy sections before their HWNDs and keep the main native renderer alive.
    sections.close()
    native.frame(.016)
    print('PASS section shutdown leaves main LNT viewport operational', flush=True)
finally:
    sections.close()
    native.close()
    app.close()
    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qapp.quit()

