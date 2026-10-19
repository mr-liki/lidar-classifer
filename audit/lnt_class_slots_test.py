"""Native section classification: screen mapping vs real Vulkan pixels, and every
drawable tool's selection/undo/visibility/From-class rules on store indices."""
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
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


import json
import uuid
from contextlib import contextmanager

@contextmanager
def test_directory():
    path = ROOT / "scratch" / ("class_slots_" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
from PySide6.QtGui import QMouseEvent
from gui.class_slots import slot_for, code_for, stored_palette

def click(h, z):
    pos = QPointF(*canvas.screen_from_section(h, z))
    for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
        event = QMouseEvent(kind, pos, canvas.mapToGlobal(pos.toPoint()).toPointF(),
                            Qt.LeftButton, Qt.LeftButton if kind == QEvent.MouseButtonPress else Qt.NoButton,
                            Qt.NoModifier)
        QApplication.sendEvent(canvas, event)

def rectangle():
    click(1, 1)
    click(6, 6)

try:
    with test_directory() as tmp:
        app._lnt_main_view.lnt_path = str(Path(tmp) / 'sample.lnt')
        sections.open(0, draw=False)
        sections.finalize((0, 0, 0), (10, 0, 0), .5, 0, 0.)
        canvas = sections.views[1]
        canvas.resize(400, 400)
        qapp.processEvents()
        canvas.fit()
        app.active_classify_tool = 'rectangle'
        app.to_class, app.from_classes = 999, None
        before = classes.copy()
        used = set(map(int, classes)) | set(app.class_palette)
        rectangle()
        slot = slot_for(app, 999)
        assert slot is not None and slot not in used
        selected = classes == slot
        assert selected.any() and np.all(classes[~selected] == before[~selected])
        assert '-> class 999' in app.statusBar().currentMessage()
        sidecar = Path(app._lnt_main_view.lnt_path + '.classmap.json')
        assert json.loads(sidecar.read_text())['code_to_slot'] == {'999': slot}
        assert code_for(app, slot) == 999
        del app._class_slot_state
        assert slot_for(app, 999) == slot and code_for(app, slot) == 999
        print('PASS real Qt clicks classify to 999 using a free persisted byte slot', flush=True)
        rule = {'color': (13, 71, 229), 'show': True}
        app.class_palette[999] = rule
        canvas.apply_palette(app.class_palette, 'class')
        assert stored_palette(app, app.class_palette)[slot] == rule
        app.to_class, app.from_classes = 6, [999]
        rectangle()
        assert np.array_equal(classes == 6, selected)
        assert np.all(classes[~selected] == before[~selected])
        print('PASS shown class 999 palette and From [999] -> 6 preserve exact selection', flush=True)
        from gui.class_slots import NoFreeSlot
        with patch('gui.class_slots._save', side_effect=OSError('sidecar unavailable')):
            app.to_class, app.from_classes = 1001, None
            unchanged = classes.copy()
            rectangle()
            assert np.array_equal(classes, unchanged)
            assert 'sidecar unavailable' in app.statusBar().currentMessage()
            assert slot_for(app, 1001) is None
        print('PASS sidecar write failure reports status and rolls back the map', flush=True)
        with patch('gui.lnt_class_counts.scan_class_histogram', return_value=np.ones(256, dtype=np.int64)), \
                patch('gui.class_slots.os.path.isfile', return_value=True):
            try:
                slot_for(app, 1002, allocate=True)
            except NoFreeSlot:
                pass
            else:
                raise AssertionError('allocation ignored full-file classes outside the section')
        print('PASS allocation checks full-file classes when background counts are unavailable', flush=True)
        # AI datasets carry user numbers; only native writes carry byte slots.
        from gui.lnt_ai_bridge import LntAiBridge
        writer = Mock(return_value=True)
        bridge = SimpleNamespace(
            active=True, app=app, view=SimpleNamespace(native=SimpleNamespace(set_classes=writer)),
            dataset={'classification': np.array([999, 6], dtype=np.int64)},
            on_disk=np.array([2, 6], dtype=np.uint8),
            node=np.array([1, 1], dtype=np.uint32),
            local=np.array([0, 1], dtype=np.uint32), _sync_views=Mock())
        assert LntAiBridge.commit(bridge) == 1
        assert writer.call_args.args[2].dtype == np.uint8
        assert writer.call_args.args[2].tolist() == [slot]
        assert LntAiBridge.commit(bridge) == 0
        print('PASS AI write-back maps user class 999 to its byte slot', flush=True)
        app.from_classes = None
        app.to_class = 1000
        app.class_palette = {c: {'color': (255, 255, 255), 'show': True} for c in range(256)}
        unchanged = classes.copy()
        rectangle()
        assert np.array_equal(classes, unchanged)
        assert 'all 256 class slots' in app.statusBar().currentMessage()
        print('PASS slot exhaustion reports status through real Qt events without mutation', flush=True)
finally:
    sections.close()
    native.close()
    app.close()
    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qapp.quit()
