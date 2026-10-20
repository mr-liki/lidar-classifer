"""Real Vulkan cross/cut display modes, asynchronous transitions, no VTK.

Run: venv/Scripts/python.exe audit/native_section_display_test.py
"""
import importlib.abc
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class NoVtk(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('vtk', 'vtkmodules', 'pyvista'):
            raise AssertionError('Native display imported ' + fullname)

sys.meta_path.insert(0, NoVtk())
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from gui.native_sections import NativeSections, CUT_SLOT
from gui.native_section_display import build_mesh, depth_colors, source_line_ids, visibility, color_range

qt = QApplication([])
app = QMainWindow()
app.resize(900, 650)
app.show()
x, y = np.meshgrid(np.linspace(0., 10., 41), np.linspace(-2., 2., 17))
xyz = np.column_stack((x.ravel(), y.ravel(), (.3*x + .7*y).ravel()))
classes = np.where(xyz[:, 0] < 5., 2, 6).astype(np.uint8)
palette = {2: dict(color=(255, 60, 20), show=True, weight=1.),
           6: dict(color=(20, 220, 80), show=True, weight=1.)}
ids = np.where(classes == 2, 11, 22).astype(np.uint16)
store = SimpleNamespace(data=dict(xyz=xyz, classification=classes,
    intensity=np.linspace(1, 255, len(xyz), dtype=np.float32),
    rgb=np.full((len(xyz), 3), .8, dtype=np.float32), point_source_id=ids),
    rgb8=np.full((len(xyz), 3), 204, dtype=np.uint8), generation=0)
app._lnt_main_view = SimpleNamespace()
app.view_palettes = {}
app.class_palette = palette
app.data = store.data
sections = NativeSections(app)
app._native_sections = sections

# Numeric parity with the old two DISTINCT lighting formulas. A constant
# colour ramp isolates Surface's additive ambient from Class's hard floor.
normal = np.array([-.3, -.7, 1.])
normal /= np.linalg.norm(normal)
az, el, ambient = 30., 40., .25
base = np.array([200., 100., 50.])
options = dict(light=(az, el, ambient), precision=0., max_edge=0., quality='normal',
               z_bounds=(-3., 6.), ramp=[(0., tuple(base)), (1., tuple(base))])
display = xyz[:, [1, 0, 2]].copy()  # side-view transform
for mode in ('shaded_class', 'surface'):
    mesh = build_mesh(xyz, display, np.full(len(xyz), 2, np.uint8),
                      {2: dict(color=base)}, mode, options)
    assert mesh is not None
    # Geographic azimuth in Shaded Class, mathematical azimuth in Surface.
    angle = np.deg2rad(90. - az if mode == 'shaded_class' else az)
    elev = np.deg2rad(el)
    light = np.array([np.cos(elev)*np.cos(angle), np.cos(elev)*np.sin(angle), np.sin(elev)])
    dot = max(float(normal @ light), 0.)
    shade = max(dot, ambient) if mode == 'shaded_class' else ambient + (1.-ambient)*dot
    expected = (base * shade).astype(np.uint8)
    assert np.max(np.abs(mesh[2].astype(int) - expected.astype(int))) <= 1
    assert all(tuple(p) in set(map(tuple, display)) for p in mesh[0])
np.testing.assert_array_equal(visibility(np.array([2, 6, 7]), palette), [True, True, False])
assert color_range(np.full(4, 8.), 1, 99) == (7.5, 8.5)
print('PASS legacy lighting formulas, local transforms, and visibility', flush=True)

def wait_for(test, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qt.processEvents()
        if test():
            return
        time.sleep(.01)
    raise AssertionError('Timed out: ' + app.statusBar().currentMessage())

try:
    with patch('gui.native_sections.get_store', return_value=store):
        canvas = sections.ensure_dock(1)
        canvas.set_points(xyz, np.arange(len(xyz)))
        images = {}
        for mode in ('class', 'rgb', 'intensity', 'elevation', 'depth', 'line', 'shaded_class', 'surface'):
            canvas.apply_palette(palette, mode)
            if mode in ('surface', 'shaded_class'):
                wait_for(lambda: getattr(canvas, '_surface_loaded', False))
            canvas.render()
            qt.processEvents()
            frame = canvas.backend.capture_frame()
            assert frame is not None, mode
            assert np.ptp(frame[:, :, :3]) > 10, mode + ' rendered a flat frame'
            images[mode] = frame
            print('PASS Vulkan pixels:', mode, flush=True)
        assert not np.array_equal(images['class'], images['line'])
        assert not np.array_equal(images['shaded_class'], images['surface'])

        # A late mesh completion must never replace a newly selected point mode.
        canvas.apply_palette(palette, 'surface')
        stale = canvas._display_revision
        canvas.apply_palette(palette, 'rgb')
        canvas._finish_mesh(stale, (xyz, np.array([[0, 1, 42]], np.int32), np.array([[255, 0, 0]], np.uint8)), '')
        assert not canvas._surface_loaded and canvas.mode == 'rgb'

        # Edits invalidate meshes and visibility in the section, without
        # changing another view's independent display choice.
        cut = sections.ensure_dock(CUT_SLOT)
        cut.set_points(xyz, np.arange(len(xyz)), axis=1)
        cut.apply_palette(palette, 'elevation')
        canvas.apply_palette(palette, 'surface')
        wait_for(lambda: canvas._surface_loaded)
        classes[:20] = 6
        sections.refresh_edits(force=True)
        wait_for(lambda: canvas._surface_loaded)
        assert cut.mode == 'elevation'

        hidden = {c: dict(rule, show=False) for c, rule in palette.items()}
        canvas.apply_palette(hidden, 'surface')
        assert not canvas._surface_loaded
        canvas.set_points(np.empty((0, 3)), np.empty(0, np.int64))
        assert not canvas._surface_loaded
        assert np.all(depth_colors(np.ones(5), SimpleNamespace()) == 127)
        assert not any(n.split('.')[0] in ('vtk', 'vtkmodules', 'pyvista') for n in sys.modules)
        print('PASS transitions, edits, empty selection, cut frame, and no VTK', flush=True)

        # Recover the attribute LNT v2 omits from a real, chunk-readable LAS.
        import laspy
        with tempfile.TemporaryDirectory(prefix='section_lines_') as folder:
            source = str(Path(folder) / 'lines.las')
            las = laspy.LasData(laspy.LasHeader(point_format=3, version='1.2'))
            las.header.scales = [.001] * 3
            las.x, las.y, las.z = xyz.T
            las.point_source_id = ids
            las.write(source)
            np.testing.assert_array_equal(source_line_ids(source, xyz, .001), ids)
            # Actual .lnt cell quantisation -> asynchronous Line mode recovery.
            from gui import lnt_native
            from gui.lnt_class_counts import read_directory
            lnt = str(Path(folder) / 'lines.lnt')
            lnt_native.convert_las(source, lnt)
            _, directory = read_directory(lnt)
            import struct
            origin = np.array(struct.unpack('<3d', Path(lnt).read_bytes()[16:40]))
            parts, node_parts = [], []
            with open(lnt, 'rb') as stream:
                for node, entry in enumerate(directory):
                    if entry['children_mask'] or not entry['point_count']:
                        continue
                    stream.seek(int(entry['byte_offset']))
                    raw = np.frombuffer(stream.read(int(entry['point_count']) * 11), dtype=np.dtype([
                        ('relative', '<u2', 3), ('rest', 'u1', 5)]))
                    parts.append(origin + entry['center'].astype(float) - float(entry['half_size']) +
                                 raw['relative'].astype(float) / 65535. * 2. * float(entry['half_size']))
                    node_parts.append(np.full(len(raw), node, dtype=np.uint32))
            reconstructed = np.concatenate(parts)
            node_ids = np.concatenate(node_parts)
            app._lnt_source_path = source
            app._lnt_main_view.lnt_path = lnt
            store.node = node_ids
            store.data = dict(xyz=reconstructed, classification=np.full(len(reconstructed), 2, np.uint8),
                              intensity=np.ones(len(reconstructed), np.float32),
                              rgb=np.full((len(reconstructed), 3), .8, np.float32))
            store.rgb8 = np.full((len(reconstructed), 3), 204, np.uint8)
            canvas.set_points(reconstructed, np.arange(len(reconstructed)))
            canvas.apply_palette(palette, 'line')
            wait_for(lambda: canvas._source_ids is not None)
            assert set(canvas._source_ids.tolist()) == {11, 22}
            print('PASS original LAS -> quantised LNT -> native Line mode recovery', flush=True)
finally:
    sections.close()
    app.close()
    qt.processEvents()
