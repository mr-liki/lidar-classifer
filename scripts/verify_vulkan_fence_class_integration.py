"""Live Vulkan acceptance on a generated cloud; never opens user point files.

Run from the project root with venv/Scripts/python.exe. Artifacts are written
under diagnostics/fence_class_integration/live_vulkan.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop('QT_QPA_PLATFORM', None)
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding='utf-8', errors='replace')

import laspy
import numpy as np
from PySide6.QtCore import QPointF, Qt, QSettings
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QListWidget, QCheckBox, QPushButton

from gui.lnt_native import NativeView, convert_las
from gui.native_draw import get_native_draw
from gui.native_byclass_fences import native_fence_records
from gui import native_byclass_parity as parity
from gui.menu_sidebar_system import ByClassDialog, ClosedByClassDialog, ByClassHeightDialog, InsideFenceDialog
from gui import lidar_classification_tools as classification_tools


def main():
    out = ROOT / 'diagnostics/fence_class_integration/live_vulkan'
    out.mkdir(parents=True, exist_ok=True)
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(out / 'settings'))
    xx, yy = np.meshgrid(np.arange(12, dtype=float), np.arange(12, dtype=float))
    ground = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    vegetation = ground + (0., 0., 1.)
    xyz = np.vstack([ground, vegetation])
    classes = np.r_[np.zeros(len(ground), dtype=np.uint8),
                    np.full(len(vegetation), 2, dtype=np.uint8)]
    classes[len(ground)::12] = 7
    las = laspy.LasData(laspy.LasHeader(point_format=3, version='1.2'))
    las.header.scales = [.001, .001, .001]
    las.x, las.y, las.z = xyz.T
    las.classification = classes
    las_path, lnt_path = out / 'fixture.las', out / 'fixture.lnt'
    las.write(str(las_path))
    convert_las(str(las_path), str(lnt_path))

    qt = QApplication([])
    app = QMainWindow()
    app.setWindowTitle('Naksha Vulkan fence integration acceptance')
    widget = QWidget(app)
    widget.setAttribute(Qt.WA_NativeWindow)
    app.setCentralWidget(widget)
    app.resize(1000, 700)
    app.show()
    qt.processEvents()
    ratio = widget.devicePixelRatioF()
    native = NativeView(int(widget.winId()), int(widget.width()*ratio),
                        int(widget.height()*ratio), str(lnt_path), 64, 8.)
    owner = SimpleNamespace(vulkan_widget=widget, vulkan_backend=None)
    view = SimpleNamespace(app=app, owner=owner, native=native,
                           mode_2d=True, lnt_path=str(lnt_path))
    app._lnt_main_view = view
    app.data = None
    app.undo_stack, app.redo_stack = [], []
    app.class_palette = {0: {'name': 'Ground reference', 'color': (220, 160, 20), 'show': True},
                         2: {'name': 'Vegetation', 'color': (30, 240, 50), 'show': True},
                         7: {'name': 'Target', 'color': (255, 60, 60), 'show': True}}
    draw = None
    results = {'points': len(xyz), 'operations': [], 'errors': []}
    dialogs = []
    ask, report = parity._ask, parity._report
    parity._ask = lambda *args: True
    parity._report = lambda dialog, message, error=False: results['errors'].append(message) if error else None
    try:
        native.set_2d(True)
        native.plan_view()
        for _ in range(4):
            native.frame(.016)
            qt.processEvents()
        draw = get_native_draw(app, create=True)

        def click(x, y, button=Qt.LeftButton):
            px, py = draw.canvas.screen_from_section(x, y)
            draw.press(QPointF(px, py), button)
            draw.release(QPointF(px, py), button)

        assert draw.activate('rectangle')
        click(1.25, 1.25)
        click(4.75, 4.75)
        assert draw.activate('circle')
        click(8., 8.)
        click(9.25, 8.)
        assert draw.activate('polyline')
        click(6., 1.)
        click(9., 1.)
        click(8., 4.)
        click(8., 4., Qt.RightButton)
        draw.cancel()
        fences = native_fence_records(app)
        assert len(fences) == 3, [(f['type'], f['coords']) for f in fences]
        assert {f['type'] for f in fences} == {'rectangle', 'circle', 'polyline'}
        _, snapshot = parity.read_snapshot(app)
        original = snapshot['classification'].copy()
        polygon_mask = parity._fence_mask(snapshot['xyz'], [f['coords'] for f in fences[:2]])
        assert 0 < polygon_mask.sum() < len(original)

        for mode, cls in [('convert', ByClassDialog), ('fence', InsideFenceDialog),
                          ('close', ClosedByClassDialog), ('height', ByClassHeightDialog)]:
            dialog = cls(app, None)
            dialogs.append(dialog)
            dialog.show()
            qt.processEvents()
            for index in range(dialog.from_list.count()):
                item = dialog.from_list.item(index)
                item.setSelected(item.data(Qt.UserRole) == 2)
            dialog.to_combo.setCurrentIndex(dialog.to_combo.findData(7))
            if mode != 'convert':
                dialog.select_fence()
                picker = dialog._fence_selection_dialog
                rows = picker.findChild(QListWidget)
                assert rows.count() == 3
                for index in (0, 1):
                    rows.itemWidget(rows.item(index)).findChild(QCheckBox).setChecked(True)
                assert id(picker) in draw._classification_fence_highlights
                next(b for b in picker.findChildren(QPushButton) if 'Apply' in b.text()).click()
                assert len(dialog.selected_fences) == 2
            if mode == 'close':
                dialog.radius_spin.setValue(20.)
            if mode == 'height':
                dialog.ref_class_combo.setCurrentIndex(dialog.ref_class_combo.findData(0))
                dialog.min_height_spin.setValue(.5)
                dialog.max_height_spin.setValue(1.5)
                dialog.max_triangle_spin.setValue(20.)
            assert parity.route_convert_dialog(dialog, mode, preview=False)
            assert not results['errors'], results['errors']
            _, after = parity.read_snapshot(app)
            changed = after['classification'] != original
            expected = (original == 2) & (np.ones(len(original), bool) if mode == 'convert' else polygon_mask)
            assert changed.sum() > 0, f'{mode} changed no source points'
            assert np.array_equal(changed, expected), f'{mode}: unexpected class/fence scope'
            assert np.all(after['classification'][changed] == 7)
            step = app.undo_stack.pop()
            assert native.set_classes(step['lnt_node'], step['lnt_local'], step['old_classes'])
            _, restored = parity.read_snapshot(app)
            assert np.array_equal(restored['classification'], original), f'{mode}: undo restoration failed'
            results['operations'].append({'mode': mode, 'changed': int(changed.sum()),
                                          'fences': 0 if mode == 'convert' else 2, 'undo_restored': True})
            if mode == 'fence':
                dialog.grab().save(str(out / 'fence_dialog.png'))
            dialog.close()
            qt.processEvents()

        alignment = np.array([[0., 2., 0.], [5., 2., 0.], [11., 2., 0.]])
        algorithms = [
            ('classify_low_points', {'ground_classes': [0], 'search_mode': 'single', 'within': 2.}),
            ('classify_isolated_points', {'in_classes': [2], 'if_fewer_than': 2, 'within': .1}),
            ('classify_ground_ptd', {'current_ground': 0, 'max_building_size': 4., 'iteration_distance': 1.4}),
            ('classify_surface_points', {'tolerance': .1}),
            ('classify_below_surface', {'limit': 1., 'num_neighbors': 8}),
            ('classify_find_wires', {'alignment_coords': [alignment], 'corridor_width': 20., 'min_height_above_ground': 0.}),
            ('classify_poles_by_centerline', {'centerline_coords': [alignment], 'radius': 20., 'down_height': 2.}),
        ]
        for name, extra in algorithms:
            func = getattr(classification_tools, name)
            params = dict(from_classes=[2], to_class=7, **extra)
            expected = original.copy()
            result = func(xyz=snapshot['xyz'], classification=expected,
                          fence_mask=polygon_mask.copy(), **params)
            parity._check_algorithm_result(result)
            expected[~polygon_mask] = original[~polygon_mask]
            dialog = classification_tools._BaseClassifyDialog(app, name)
            dialogs.append(dialog)
            selector = classification_tools.FenceSelectorWidget(app, parent=dialog)
            selector.selected_fences = fences[:2]
            dialog._fence_sel = selector
            assert parity.route_algorithm_dialog(dialog, func, params, name)
            assert not results['errors'], results['errors']
            _, after = parity.read_snapshot(app)
            assert np.array_equal(after['classification'], expected), f'{name}: VTK core parity mismatch'
            changed = after['classification'] != original
            assert not changed[~polygon_mask].any(), f'{name}: wrote outside selected fences'
            if changed.any():
                step = app.undo_stack.pop()
                assert native.set_classes(step['lnt_node'], step['lnt_local'], step['old_classes'])
            _, restored = parity.read_snapshot(app)
            assert np.array_equal(restored['classification'], original)
            results['operations'].append({'mode': name, 'changed': int(changed.sum()),
                                          'fences': 2, 'vtk_core_match': True, 'undo_restored': True})
            dialog.close()
            qt.processEvents()
        results['passed'] = True
        print(json.dumps(results, indent=2))
    finally:
        parity._ask, parity._report = ask, report
        for dialog in dialogs:
            dialog.close()
        if draw is not None:
            draw.shutdown()
        qt.processEvents()
        native.close()
        app.close()
        qt.processEvents()
        (out / 'result.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
