"""Real Qt classify -> Save -> Don't Save -> reopen must retain saved classes."""
from pathlib import Path
import sys
import shutil
import uuid
from types import SimpleNamespace
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import builtins
real_import = builtins.__import__
def no_vtk(name, *a, **kw):
    if name == 'vtk' or name.startswith(('vtkmodules', 'pyvista')):
        raise AssertionError('Native persistence imported VTK: ' + name)
    return real_import(name, *a, **kw)
builtins.__import__ = no_vtk
import numpy as np
import laspy
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from gui import lnt_native
from gui.native_sections import NativeSections
from gui.native_save import save_native, read_classes
from gui.class_slots import slot_for, code_for
qapp = QApplication([])
root = ROOT / 'scratch' / ('saved_classes_' + uuid.uuid4().hex)
root.mkdir()
try:
    for code, extension in ((6, '.las'), (999, '.las'), (6, '.laz'), (999, '.laz')):
        source = root / f'classes_{code}_{extension[1:]}{extension}'
        lnt = source.with_suffix('.lnt')
        las = laspy.LasData(laspy.LasHeader(point_format=7, version='1.4'))
        xx, zz = np.meshgrid(np.arange(11.), np.arange(11.))
        las.x, las.y, las.z = xx.ravel(), np.zeros(xx.size), zz.ravel()
        las.classification = np.full(xx.size, 2, np.uint8)
        las.red = las.green = las.blue = np.full(xx.size, 65535, np.uint16)
        las.write(source)
        converter_source = source
        if extension == '.laz':
            converter_source = source.with_suffix('.input.las')
            las.write(converter_source)
        lnt_native.convert_las(str(converter_source), str(lnt))
        app = QMainWindow()
        app.resize(1000, 700)
        widget = QWidget(app)
        widget.setAttribute(Qt.WA_NativeWindow)
        app.setCentralWidget(widget)
        app.show()
        qapp.processEvents()
        native = lnt_native.NativeView(int(widget.winId()), 1000, 700, str(lnt), 64, 8.)
        app._lnt_main_view = SimpleNamespace(native=native, lnt_path=str(lnt))
        app._lnt_source_path = app.loaded_file = str(source)
        app.data = None
        app.section_controller = SimpleNamespace(active_view=0)
        app.section_buffer = 0.
        app.view_palettes = {}
        app.class_palette = {c: {'color': (255, 255, 255), 'show': True} for c in (2, 6, 7, 999)}
        app.undo_stack, app.redo_stack = [], []
        sections = NativeSections(app)
        app._native_sections = sections
        try:
            sections.open(0, draw=False)
            sections.finalize((0, 0, 0), (10, 0, 0), .5, 0, 0.)
            canvas = sections.views[1]
            canvas.resize(400, 400)
            qapp.processEvents()
            canvas.fit()
            store = app._lnt_section_store
            app.active_classify_tool = 'rectangle'
            app.to_class, app.from_classes = code, None
            def rectangle():
                for h, z in ((1.5, 1.5), (5.5, 5.5)):
                    pos = QPointF(*canvas.screen_from_section(h, z))
                    for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
                        event = QMouseEvent(kind, pos, canvas.mapToGlobal(pos.toPoint()).toPointF(),
                                            Qt.LeftButton, Qt.LeftButton if kind == QEvent.MouseButtonPress else Qt.NoButton,
                                            Qt.NoModifier)
                        QApplication.sendEvent(canvas, event)
            rectangle()
            slot = slot_for(app, code)
            assert np.count_nonzero(store.data['classification'] == slot) > 0
            assert save_native(app)
            saved = read_classes(str(lnt))[0].copy()
            assert np.count_nonzero(saved == slot) > 0
            assert np.count_nonzero(laspy.read(source).classification == slot) == np.count_nonzero(saved == slot)
            store.discard_unflushed_edits(stop_timer=False)
            assert np.array_equal(read_classes(str(lnt))[0], saved)
            assert save_native(app)  # unchanged Save also retains the checkpoint
            print(f"PASS class {code}: Save -> Don't Save with no further edits keeps saved classes", flush=True)
            # Undo remains available across Save. A later edit may even undo
            # a saved action; Don't Save must still restore the checkpoint.
            assert app.undo_stack
            from gui.native_shortcuts import native_classification_history
            assert native_classification_history(app)
            store.flush()
            assert not np.array_equal(read_classes(str(lnt))[0], saved)
            store.discard_unflushed_edits(stop_timer=False)
            assert np.array_equal(read_classes(str(lnt))[0], saved)
            assert save_native(app)
            print(f"PASS class {code}: undo of a saved action is discardable without losing the save checkpoint", flush=True)
            app.to_class = 7
            rectangle()
            assert np.count_nonzero(store.data['classification'] == 7) > 0
            store.flush()
            # A failed save must not turn the later edit into the checkpoint.
            checkpoint = store._saved_classes.copy()
            with patch('gui.native_save.write_pair', side_effect=OSError('injected save failure')), \
                 patch('PySide6.QtWidgets.QMessageBox.critical'):
                assert not save_native(app)
            assert np.array_equal(store._saved_classes, checkpoint)
            app.undo_stack.clear()  # rollback must also survive history trimming
            store.discard_unflushed_edits()
            after_discard = read_classes(str(lnt))[0]
            assert np.array_equal(after_discard, saved), f"Don't Save erased saved class {code}"
            print(f"PASS class {code}: real Qt classify -> Save -> further edit -> Don't Save preserves saved LAS and LNT", flush=True)
        finally:
            sections.close()
            native.close()
            app.close()
            qapp.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        # Independent reopen of the native file, not the original store or renderer.
        app2 = QMainWindow()
        widget2 = QWidget(app2)
        widget2.setAttribute(Qt.WA_NativeWindow)
        app2.setCentralWidget(widget2)
        app2.show()
        qapp.processEvents()
        reopened = lnt_native.NativeView(int(widget2.winId()), 600, 400, str(lnt), 64, 8.)
        try:
            app2._lnt_main_view = SimpleNamespace(native=reopened, lnt_path=str(lnt))
            result = reopened.query_points((0., -1., 0.), (10., 1., 10.))
            assert result is not None
            assert np.count_nonzero(result['classification'] == slot) == np.count_nonzero(saved == slot)
            assert code_for(app2, slot) == code
            print(f'PASS class {code}: fresh Vulkan reopen retains stored class and user number', flush=True)
        finally:
            reopened.close()
            app2.close()
            qapp.processEvents()
finally:
    shutil.rmtree(root)
    qapp.quit()
    builtins.__import__ = real_import
