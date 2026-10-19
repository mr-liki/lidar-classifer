"""Native Vulkan Block/Cell Creation end-to-end on the real engine (no VTK).

Reproduces the reported flow: an SNT is attached, an .lnt is open in the native
main viewport, then Create Block is used - open dialog, arm, pick an SNT
centreline in the viewport, Create - and every Qt overlay (selection, created
rings, SNT text labels) paints over the live engine frames.

Regression guards:
- selection/candidate/drawing coords must stay 2-wide (x, y): 3-wide tuples
  raise inside QWidget::paintEvent's Python override, which escalates into an
  access violation in shiboken6 (the reported crash).
- _split_polyline must split long CLs at 'Distance' without IndexError.

Run from the repo root:  venv\\Scripts\\python.exe audit\\native_block_creation_test.py
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtCore import Qt, QPointF, QCoreApplication, QEvent

qapp = QApplication([])

checks = []
def check(label, ok, detail=""):
    checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {detail}", flush=True)


win = QMainWindow()
win.resize(1200, 800)
canvas = QWidget(win)
canvas.setAttribute(Qt.WA_NativeWindow)
win.setCentralWidget(canvas)
win.show()
qapp.processEvents()

from gui import lnt_native
native = lnt_native.NativeView(int(canvas.winId()), 800, 600,
                               str(ROOT / 'native/lnt/test_shading.lnt'), 64, 8.)

# The application's own backend object exactly as it is during an .lnt session:
# the LNT view owns the live renderer; this one is shut down (handle 0) but its
# DLL is loaded, which is what the vector proxy copies.
from gui.render_backend import VulkanRenderBackend
app_backend = VulkanRenderBackend()

win.snt_attachments = [{
    'filename': 'blocks.snt',
    'full_path': str(ROOT / 'blocks.snt'),
    'parsed': {'entities': [
        {'type': 'LINE', 'vertices': [(0., 0., 10.), (3., 0., 10.)], '_snt_entity_id': 1},
    ]},
}]
win.class_palette = {2: {'color': (255, 0, 0), 'show': True}}
win.data = None

view = SimpleNamespace(app=win, native=native, mode_2d=True, name='test',
                       lnt_path=str(ROOT / 'native/lnt/test_shading.lnt'))
view.owner = SimpleNamespace(vulkan_widget=canvas, vulkan_backend=app_backend)
win._lnt_main_view = view
win.render_backend = view.owner

native.set_2d(1)
native.frame_region(1.5, 0., 30.)
for _ in range(3):
    native.frame(.016)
    qapp.processEvents()
time.sleep(.2)
qapp.processEvents()

ratio = canvas.devicePixelRatioF()
mid = native.world_to_screen(1.5, 0.)
check('engine maps the SNT CL midpoint to a pixel',
      mid is not None and 0 <= mid[0] <= 800 and 0 <= mid[1] <= 600, str(mid))

try:
    from gui.native_create import (_split_polyline, get_native_creation,
                                   show_native_block_dialog, show_native_cell_dialog)

    # 0. Pure geometry: long CLs split at 'Distance' with 2-wide coords.
    chains = _split_polyline([(0., 0.), (1000., 0.)], 250.)
    check('1 km CL splits into 4 x 250 m chains', len(chains) == 4,
          f"{len(chains)} chain(s)")
    check('split chains keep 2-wide coords',
          all(len(p) == 2 for chain in chains for p in chain))
    check('split chains cover the full line',
          chains[0][0] == (0., 0.) and chains[-1][-1] == (1000., 0.))

    # 1. Ribbon entry point: the NATIVE dialog opens (never the VTK one).
    ok = show_native_block_dialog(win)
    dialog = getattr(win, 'native_block_creation_dialog', None)
    check('Create Block opens the native dialog', ok and dialog is not None
          and dialog.isVisible(), f"visible={getattr(dialog, 'isVisible', lambda: False)()}")
    controller = get_native_creation(win, create=True)
    check('dialog and ribbon share one controller',
          controller is not None and dialog.controller is controller)

    # 2. Select CL: arm, then click the SNT centreline through the real mapping.
    dialog.select_btn.click()
    candidates = controller.candidates()
    check('arm collected the SNT centreline',
          controller.active and any(c.get('source') == 'snt_cl' for c in candidates))
    check('candidate coords are 2-wide (overlay contract)',
          all(len(p) == 2 for c in candidates for p in c.get('coords', [])))
    pos = QPointF(mid[0] / ratio, mid[1] / ratio)
    # A click selects on release (a press may still become a drag rectangle).
    handled = controller.press(pos, Qt.LeftButton, Qt.NoModifier)
    handled = controller.release(pos, Qt.LeftButton) and handled
    check('viewport click selects the CL', handled and len(controller.selected) == 1,
          f"selected={len(controller.selected)}")
    selection = [s for s in controller.draw.overlay.shapes
                 if s[0] == 'drawing']
    check('selection is drawn on the transparent overlay', bool(selection))
    check('selection coords are 2-wide (overlay contract)',
          all(len(p) == 2 for s in selection for p in s[1]))

    # 3. Create: native corridor rings, no VTK.
    dialog.total.setValue(60.)
    before = len(controller.draw.drawings)
    dialog.create()
    rings = [d for d in controller.draw.drawings if d.get('source') == 'block_creation']
    check('Create built native block rings', len(rings) >= 1 and
          len(controller.draw.drawings) > before, f"{len(rings)} ring(s)")
    # A block is drawn as its edges (above, below, two caps), as in the
    # reference; together they must close around the CL.
    records = getattr(win, 'block_creation_generated_blocks', None) or []
    edges = (records[0].get('drawings') or []) if records else []
    ends = [tuple(round(v, 6) for v in d['coords'][i][:2]) for d in edges for i in (0, -1)]
    check('block edges close into a corridor around the CL',
          len(edges) >= 3 and all(ends.count(p) % 2 == 0 for p in ends),
          f"{len(edges)} edge(s)")

    # 4. The created ring paints; frames keep flowing with the dialog open.
    #    (Any exception inside QWidget::paintEvent's Python override used to
    #    escalate into an access violation in shiboken6 here.)
    for _ in range(3):
        native.frame(.016)
        qapp.processEvents()
    time.sleep(.25)
    qapp.processEvents()
    painted = any(s[0] == 'drawing' and (s[2] or {}).get('source') == 'block_creation'
                  for s in controller.draw.overlay.shapes)
    check('created ring is shown on the overlay', painted)

    # 5. SNT text labels (world_z world projection) paint on their own overlay,
    #    exactly like the vector layer's label overlay does.
    from gui.native_main_classify import MainCanvas
    from gui.native_section_classify import CanvasOverlay
    label_overlay = CanvasOverlay(MainCanvas(view))
    label_overlay.set_shapes([('drawing', [(1., 0.)],
                               {'type': 'text', 'source': 'attachment', 'text': 'BLOCK_A',
                                'world_z': 10.0, 'height': 2.0, 'color': '#ffffff',
                                'width': 1})])
    label_overlay._follow(immediate=True)
    qapp.processEvents()
    check('SNT label overlay survived a paint through the engine MVP',
          bool(label_overlay.shapes))

    # 6. Cell Creation rides the same native controller.
    cell_ok = show_native_cell_dialog(win)
    cell_dialog = getattr(win, 'native_cell_creation_dialog', None)
    cell_dialog.select_btn.click()
    controller.press(pos, Qt.LeftButton, Qt.NoModifier)
    cell_dialog.diameter.setValue(3000.)
    cell_dialog.create()
    cells = [d for d in controller.draw.drawings if d.get('source') == 'cell_creation']
    check('Create Cell built native cells', cell_ok and len(cells) >= 1, f"{len(cells)} cell(s)")
    check('cell center stays a 3-wide world point',
          bool(cells) and len(cells[0].get('center', ())) == 3)
    for _ in range(2):
        native.frame(.016)
        qapp.processEvents()

    check('VTK was never imported', 'vtk' not in sys.modules)
    check('gui.block_creation_settings_dialog never imported',
          'gui.block_creation_settings_dialog' not in sys.modules)

    dialog.close()
    cell_dialog.close()
    label_overlay._timer.stop()
    label_overlay.close()
finally:
    native.close()
    win.close()
    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qapp.quit()

print(f"\n{sum(checks)}/{len(checks)} checks passed")
sys.exit(0 if all(checks) else 1)
