"""Native section classification: screen mapping vs real Vulkan pixels, and every
drawable tool's selection/undo/visibility/From-class rules on store indices."""
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import builtins
_real_import = builtins.__import__
def _without_vtk(name, *args, **kwargs):
    if name == 'vtk' or name.startswith(('vtk.', 'vtkmodules', 'pyvista')):
        raise AssertionError('Border attempted a legacy renderer import: ' + name)
    return _real_import(name, *args, **kwargs)
builtins.__import__ = _without_vtk

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from gui.native_sections import NativeSections, CUT_SLOT
from gui import lnt_native

qapp = QApplication([])
app = QMainWindow()
app.resize(1200, 800)
main_canvas = QWidget(app)
main_canvas.setAttribute(Qt.WA_NativeWindow)
app.setCentralWidget(main_canvas)
app.show()
qapp.processEvents()
native = lnt_native.NativeView(int(main_canvas.winId()), 600, 400,
                               str(ROOT / 'native/lnt/test_shading.lnt'), 64, 8.)
app._lnt_main_view = SimpleNamespace(native=native)
app.section_controller = SimpleNamespace(active_view=0)
app.section_buffer = 0.
app.view_palettes = {}
app.undo_stack, app.redo_stack = [], []
app.class_palette = {c: {'color': (255, 255, 255), 'show': True} for c in (1, 2, 5, 6)}

# A grid of records on the section plane y=0: h = 0..10, z = 0..10 (1 m pitch).
hh, zz = np.meshgrid(np.arange(11.), np.arange(11.))
xyz = np.column_stack([hh.ravel(), np.zeros(hh.size), zz.ravel()])
classes = np.full(len(xyz), 2, dtype=np.uint8)
classes[::7] = 5
store = SimpleNamespace(view=app._lnt_main_view, ensure_for_section=Mock(return_value=True),
                        data={'xyz': xyz, 'classification': classes,
                              'rgb': np.ones((len(xyz), 3), dtype=np.float32),
                              'intensity': np.zeros(len(xyz), dtype=np.float32)})
app._lnt_section_store = store
store.show = Mock()
app.data = store.data
sections = NativeSections(app)
app._native_sections = sections


from gui.native_border import apply_border
from PySide6.QtCore import QSettings
from unittest.mock import patch
settings_dir = ROOT / 'scratch' / 'native_border_settings'
settings_dir.mkdir(exist_ok=True)
test_settings = QSettings(str(settings_dir / 'border.ini'), QSettings.IniFormat)
test_settings.clear()
settings_patch = patch('PySide6.QtCore.QSettings', side_effect=lambda *a, **k: test_settings)
settings_patch.start()
from gui.display_mode import DisplayModeDialog
app.view_borders = {0: 0., 1: 0., 2: 0.}
app._naksha_border_logic_modes = {0: 1, 1: 1, 2: 1}
app.display_mode = 'class'
app._lnt_main_view.lnt_path = str(ROOT / 'native/lnt/test_shading.lnt')
dialog = None
try:
    sections.open(0, draw=False)
    sections.finalize((0, 0, 0), (10, 0, 0), .5, 0, 0.)
    canvas = sections.views[1]
    canvas.resize(400, 400)
    qapp.processEvents()
    canvas.fit()
    canvas.backend.set_clear_color(.3, .3, .3)
    canvas.apply_palette(app.class_palette, 'class')
    def capture():
        canvas._dirty = True
        canvas.render()
        qapp.processEvents()
        frame = canvas.backend.capture_frame()
        assert frame is not None
        return frame
    test_palette = dict(canvas.palette)
    baseline = capture()
    uploads = canvas.backend.get_point_position_upload_count()
    dialog = DisplayModeDialog(app)
    app.display_mode_dialog = dialog
    dialog.current_slot = 1
    dialog.view_borders[1] = 0
    dialog._select_border_mode(0, push_gpu=False)
    # Real +/- button signals (5% steps), with the legacy path forbidden.
    with patch('gui.display_mode._uam_sync', side_effect=AssertionError('legacy border path')):
        for _ in range(20):
            dialog.border_plus_btn.click()
        assert dialog.view_borders[1] == 100
        assert dialog.border_value_display.text() == '100%'
        point = capture()
        black = lambda f: int(np.count_nonzero(np.all(f[..., :3] < 8, axis=2)))
        assert black(point) > black(baseline) + 100
        assert app.view_borders[0] == 0 and app.view_borders[2] == 0
        assert canvas.backend.get_point_position_upload_count() == uploads
        print('PASS Border +/- draws real black Vulkan rings in 5% steps with zero point uploads', flush=True)
        for mode in (1, 2):
            dialog._select_border_mode(mode)
            frame = capture()
            assert black(frame) > black(baseline) + 100, mode
            assert app._naksha_border_logic_modes[1] == mode
        assert canvas.backend.get_point_position_upload_count() == uploads
        print('PASS Structured and Hybrid render through native GPU passes', flush=True)
        canvas.backend.set_point_size(36.)
        frames = []
        for mode in (0, 1, 2):
            dialog._select_border_mode(mode)
            frames.append(capture())
        assert not np.array_equal(frames[0], frames[1])
        assert not np.array_equal(frames[1], frames[2])
        from PIL import Image
        Image.fromarray(np.concatenate(frames, axis=1)).save(ROOT / 'scratch' / 'native_border_modes.png')
        canvas.backend.set_point_size(3.)
        print('PASS Per-Point, Structured and Hybrid produce distinct pixels on overlapping returns', flush=True)

        for _ in range(20):
            dialog.border_minus_btn.click()
        assert dialog.view_borders[1] == 0
        assert np.array_equal(capture(), baseline)
        print('PASS 0% restores identical pixels; changes remain isolated to the chosen view', flush=True)
        dialog.current_slot = 0
        dialog.view_borders[0] = 0
        dialog.border_plus_btn.click()
        assert app.point_border_percent == 5
        assert app.view_borders[1] == 0
        print('PASS main LNT renderer handle accepts the same native border controls', flush=True)
    app.view_borders[1] = 100
    apply_border(app, 1, 100, 0)
    canvas.apply_palette(test_palette, 'shaded_class')
    shaded_border = capture()
    apply_border(app, 1, 0, 0)
    assert np.array_equal(capture(), shaded_border)
    canvas.apply_palette(test_palette, 'class')
    app.view_borders[1] = 100
    apply_border(app, 1)
    assert black(capture()) > black(baseline) + 100
    print('PASS Shaded Class disables borders and class mode can restore them without point uploads', flush=True)
    canvas.apply_palette({c: dict(rule, show=False) for c, rule in test_palette.items()}, 'class')
    hidden = capture()
    assert np.all(hidden == hidden[0, 0])
    print('PASS hidden classes leave no border shells or depth fragments', flush=True)
    dialog._save_timer.stop()
    dialog.save_global_settings()
    saved = test_settings.value('global_view_borders')
    assert float(saved['0']) == 5
    saved_modes = test_settings.value('global_view_border_modes')
    assert int(saved_modes['1']) == int(dialog.view_border_modes[1])
    assert not any(name == 'vtk' or name.startswith('vtkmodules') for name in sys.modules)
    print('PASS border settings persist; dialog and native border workflow import no VTK', flush=True)
finally:
    if dialog is not None:
        dialog._save_timer.stop()
        dialog.hide()
        dialog._allow_native_close = True
        dialog.close()
    sections.close()
    native.close()
    app.close()
    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qapp.quit()
    settings_patch.stop()
    builtins.__import__ = _real_import
