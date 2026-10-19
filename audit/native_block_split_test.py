"""Native Block Creation > Split on an open .lnt, end to end (no VTK).

Builds an .lnt from native/lnt/input_test.las, marks every record class 9 in the
.lnt (standing in for user edits), creates blocks in the native dialog and
splits: each output LAS must hold the source points of its block with the
.lnt's classes and every other field of the source, no point twice, plus the
.prj/.snt manifests.  Neither VTK nor the legacy dialog may be imported.

Run from the repo root:  venv\Scripts\python.exe audit\native_block_split_test.py
"""
import shutil, tempfile
import sys, os, glob, io, contextlib
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, laspy
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QDialog
qapp = QApplication([])
ROOT = Path(__file__).resolve().parents[1]
SP = Path(tempfile.mkdtemp(prefix='block_split_test_'))
src = str(SP / 'src.las')
shutil.copy(ROOT / 'native/lnt/input_test.las', src)
from gui.lnt_open import build_lnt
lnt = build_lnt(src, lambda p, t: None, lambda: False)
print('built', lnt)
# Simulate user edits: every record in the .lnt becomes class 9.
from gui.lnt_class_counts import read_directory
total, entries = read_directory(lnt)
mm = np.memmap(lnt, dtype=np.uint8, mode='r+')
for e in entries:
    if e['children_mask'] == 0 and e['point_count']:
        off = int(e['byte_offset']) + 10
        mm[off:off + 11 * int(e['point_count']):11] = 9
mm.flush(); del mm

las = laspy.read(src)
x, y = np.asarray(las.x), np.asarray(las.y)
print('points', len(x), 'fmt', las.point_format.id, 'bounds', x.min(), x.max(), y.min(), y.max())
cx0, cx1, cy = x.min(), x.max(), (y.min() + y.max()) / 2
cl1 = [(cx0 - 1, cy, 0.), (cx1 + 1, cy, 0.)]
cl2 = [((cx0 + cx1) / 2, y.min() - 1, 0.), ((cx0 + cx1) / 2, y.max() + 1, 0.)]
half = (y.max() - y.min()) / 6

win = QMainWindow()
win._lnt_source_path = src; win.loaded_file = src
win._lnt_main_view = SimpleNamespace(native=object(), lnt_path=lnt)
from gui import native_create as nc
draw = SimpleNamespace(drawings=[], redo=[], selected=None, vertex_target=None, _paint=lambda: None, aux_shapes=[])
ctl = nc.NativeCreation.__new__(nc.NativeCreation)
ctl.view = None; ctl.app = win; ctl.draw = draw; ctl.canvas = None; ctl.mode = None
ctl.selected = [{'type': 'polyline', 'coords': cl1}, {'type': 'polyline', 'coords': cl2}]
ctl.hovered = None; ctl._press_px = ctl._drag_end_px = None; ctl.paint = lambda: None
nc.get_native_creation = lambda app, create=False: ctl
dlg = nc.NativeBlockDialog(win); dlg.show(); qapp.processEvents()
hidden_before = not dlg.split_btn.isVisible()
dlg.mode_box.setCurrentIndex(1); dlg.above.setValue(half); dlg.below.setValue(half)
dlg.distance.setValue((cx1 - cx0) / 2.5)
with contextlib.redirect_stdout(io.StringIO()): dlg.create()
shown_after = dlg.split_btn.isVisible() and len(win.block_creation_generated_blocks) > 0

# Drive Split -> save dialog -> Save, with message boxes auto-answered.
from gui import block_split
msgs = []
QMessageBox.information = staticmethod(lambda *a, **k: msgs.append(a[2]))
QMessageBox.warning = staticmethod(lambda *a, **k: msgs.append('WARN ' + a[2]))
QMessageBox.critical = staticmethod(lambda *a, **k: msgs.append('CRIT ' + a[2]))
orig_exec = block_split.BlockSplitSaveDialog.exec
def fake_exec(self):
    self.parent_dir = SP; self.folder_name_edit.setText('out')
    self._save_split_blocks(); return self.result()
block_split.BlockSplitSaveDialog.exec = fake_exec
dlg.split()
print('messages:', [m.splitlines()[0] for m in msgs])
out = sorted(glob.glob(str(SP / 'out' / '*')))
print('outputs', [Path(o).name for o in out])
files = [o for o in out if o.endswith('.las')]
n = 0; ok_fields = True; classes = set()
key_src = {}
src_t = np.asarray(las.gps_time) if 'gps_time' in las.point_format.dimension_names else None
ix = {(a, b, c): i for i, (a, b, c) in enumerate(zip(las.X, las.Y, las.Z))}
seen = set()
for f in files:
    o = laspy.read(f); n += len(o.points); classes |= set(np.unique(o.classification).tolist())
    for j, k in enumerate(zip(o.X, o.Y, o.Z)):
        i = ix[k]; seen.add(i)
        if o.intensity[j] != las.intensity[i] or (src_t is not None and o.gps_time[j] != src_t[i]):
            ok_fields = False
    assert o.header.point_format.id == las.point_format.id
checks = {
    'Split hidden until blocks exist': hidden_before,
    'Split shown after Create': shown_after,
    'every written point is a distinct source point': n == len(seen) and n > 0,
    'split files carry the .lnt classes': classes == {9},
    'other source fields preserved': ok_fields,
    'prj and snt manifests written': any(o.endswith('.prj') for o in out) and any(o.endswith('.snt') for o in out),
}
checks['blocks cleared after a successful split'] = not any(d.get('source') == 'block_creation' for d in draw.drawings)
checks['VTK never imported'] = 'vtk' not in sys.modules
checks['legacy VTK dialog never imported'] = 'gui.block_creation_settings_dialog' not in sys.modules
for label, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
shutil.rmtree(SP, ignore_errors=True)
sys.exit(0 if all(checks.values()) else 1)
