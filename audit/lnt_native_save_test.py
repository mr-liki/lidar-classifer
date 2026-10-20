"""Native paired Save/Save As: full source preservation, slots, failure and fast repeat."""
from pathlib import Path
import builtins
import hashlib
import os
import shutil
import sys
import time
from types import SimpleNamespace
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import laspy
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from gui import lnt_native
from gui.native_save import write_pair, read_classes, save_native
from gui.lnt_ai_bridge import source_order, _lnt_bounds

qapp = QApplication.instance() or QApplication([])
scratch = ROOT / 'scratch' / ('native_save_' + uuid.uuid4().hex)
scratch.mkdir()
real_import = builtins.__import__
def no_vtk(name, *args, **kwargs):
    if name == 'vtk' or name.startswith(('vtk.', 'vtkmodules')):
        raise AssertionError('Native saving attempted a VTK import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = no_vtk

try:
    source = scratch / 'original.las'
    lnt = source.with_suffix('.lnt')
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.scales = [.001, .001, .001]
    header.offsets = [680000., 5000000., 0.]
    header.add_extra_dim(laspy.ExtraBytesParams(name='custom_id', type=np.uint32))
    header.vlrs.append(laspy.VLR(user_id='test', record_id=42, record_data=b'preserve me'))
    las = laspy.LasData(header)
    n = int(os.environ.get('LNT_SAVE_TEST_POINTS', '10003'))
    rng = np.random.default_rng(42)
    las.x = 680000 + rng.uniform(0, 100, n)
    las.y = 5000000 + rng.uniform(0, 100, n)
    las.z = rng.uniform(0, 20, n)
    las.classification = np.full(n, 2, np.uint8)
    las.intensity = rng.integers(0, 65536, n, dtype=np.uint16)
    las.gps_time = np.arange(n, dtype=float) + 1234.5
    las.point_source_id = (np.arange(n) % 3).astype(np.uint16)
    las.return_number = np.full(n, 2, np.uint8)
    las.number_of_returns = np.full(n, 3, np.uint8)
    las.synthetic = (np.arange(n) % 2).astype(np.uint8)
    las.withheld = (np.arange(n) % 3 == 0).astype(np.uint8)
    las.scan_angle_rank = (np.arange(n) % 61 - 30).astype(np.int8)
    las.red, las.green, las.blue = [rng.integers(0, 65536, n, dtype=np.uint16) for _ in range(3)]
    las.custom_id = np.arange(n, dtype=np.uint32)
    las.write(source)
    lnt_native.convert_las(str(source), str(lnt))
    order = source_order(np.column_stack((las.x, las.y, las.z)), *_lnt_bounds(str(lnt)))
    initial, entries, leaves, starts = read_classes(str(lnt))
    record_slots = initial.copy()
    record_slots[::7] = 255
    record_slots[1::7] = 6
    with open(lnt, 'r+b') as handle:
        for i, node in enumerate(leaves):
            entry = entries[node]
            handle.seek(int(entry['byte_offset']))
            raw = bytearray(handle.read(int(entry['point_count']) * 11))
            raw[10::11] = record_slots[starts[i]:starts[i + 1]].tobytes()
            handle.seek(int(entry['byte_offset']))
            handle.write(raw)
    expected = np.empty(n, np.uint8)
    expected[order] = record_slots
    cache = {}
    t = time.perf_counter()
    result = write_pair(str(source), str(lnt), str(source), mapping={999: 255}, cache=cache)
    first = time.perf_counter() - t
    saved = laspy.read(source)
    assert saved.header.version == laspy.header.Version(1, 4)
    assert np.array_equal(saved.classification, expected)
    assert np.array_equal(saved.naksha_class, np.where(expected == 255, 999, expected.astype(np.uint64)))
    for dim in las.point_format.dimension_names:
        if dim not in ('classification', 'scan_angle_rank'):
            assert np.array_equal(np.asarray(saved[dim]), np.asarray(las[dim])), dim
    assert np.allclose(saved.scan_angle * .006, las.scan_angle_rank, atol=.0031)
    assert any(v.user_id == 'test' and v.record_data == b'preserve me' for v in saved.header.vlrs)
    assert source.with_suffix('.lnt.classmap.json').is_file()
    assert lnt.stat().st_mtime_ns >= source.stat().st_mtime_ns
    print(f'PASS Save updates LAS + LNT, preserves source coordinates/attributes/flags/VLRs and class 999 ({first:.3f}s)', flush=True)
    t = time.perf_counter()
    with patch('gui.lnt_ai_bridge.morton_keys', side_effect=AssertionError('mapping recomputed')):
        repeat = write_pair(str(source), str(lnt), str(source), mapping={999: 255}, cache=cache)
    assert repeat['unchanged']
    print(f'PASS unchanged Save skips coordinate sorting and file rewrite ({time.perf_counter() - t:.3f}s)', flush=True)
    target = scratch / 'copy.laz'
    before_lnt = hashlib.sha256(lnt.read_bytes()).digest()
    write_pair(str(source), str(lnt), str(target), mapping={999: 255}, cache=cache)
    laz = laspy.read(target)
    assert np.array_equal(laz.points.array, saved.points.array)
    assert hashlib.sha256(target.with_suffix('.lnt').read_bytes()).digest() == before_lnt
    print('PASS Save As writes complete LAZ and byte-identical LNT companion with class map', flush=True)
    # Saving an already compressed source must also preserve its attributes.
    write_pair(str(target), str(target.with_suffix('.lnt')), str(target), mapping={999: 255}, cache={})
    assert np.array_equal(laspy.read(target).points.array, saved.points.array)
    print('PASS Save overwrites an original LAZ while retaining all source dimensions', flush=True)
    filtered = scratch / 'filtered.las'
    write_pair(str(source), str(lnt), str(filtered), mapping={999: 255}, cache=cache, classes=[999])
    subset = laspy.read(filtered)
    assert len(subset.points) == np.count_nonzero(expected == 255)
    assert np.all(subset.naksha_class == 999)
    subset_slots, *_ = read_classes(str(filtered.with_suffix('.lnt')))
    assert len(subset_slots) == len(subset.points) and np.all(subset_slots == 255)
    print('PASS class-filtered Save As rebuilds a matching full-detail native companion', flush=True)
    original_digest = hashlib.sha256(source.read_bytes()).digest()
    with patch('gui.native_save.shutil.copyfile', side_effect=OSError('injected copy failure')):
        try:
            write_pair(str(source), str(lnt), str(scratch / 'failed.las'), cache=cache)
        except OSError:
            pass
        else:
            raise AssertionError('copy failure ignored')
    assert not (scratch / 'failed.las').exists()
    assert hashlib.sha256(source.read_bytes()).digest() == original_digest
    print('PASS staging failure leaves source and destination LAS untouched', flush=True)
    # A later publication error must restore an existing destination companion.
    old_pair = (target.read_bytes(), target.with_suffix('.lnt').read_bytes(),
                Path(str(target.with_suffix('.lnt')) + '.classmap.json').read_bytes())
    real_replace = os.replace
    def fail_final(staged, destination):
        if str(staged).endswith('points.laz') and str(destination) == str(target.resolve()):
            raise PermissionError('injected final replacement failure')
        return real_replace(staged, destination)
    with patch('gui.native_save.os.replace', side_effect=fail_final):
        try:
            write_pair(str(source), str(lnt), str(target), mapping={999: 255}, cache=cache)
        except PermissionError:
            pass
        else:
            raise AssertionError('publish failure ignored')
    assert old_pair == (target.read_bytes(), target.with_suffix('.lnt').read_bytes(),
                        Path(str(target.with_suffix('.lnt')) + '.classmap.json').read_bytes())
    print('PASS final publication failure restores the previous destination pair and sidecars', flush=True)
    # Ordinary subsequent LAS edits use the memory-mapped copy path.
    copy_target = scratch / 'fast.las'
    t = time.perf_counter()
    with patch('laspy.mmap', wraps=laspy.mmap) as mmap:
        write_pair(str(source), str(lnt), str(copy_target), mapping={999: 255}, cache=cache)
        assert mmap.call_count == 1
    assert np.array_equal(laspy.read(copy_target).points.array, saved.points.array)
    print(f'PASS compatible LAS save copies bytes and patches only class fields ({time.perf_counter() - t:.3f}s)', flush=True)
    standalone = scratch / 'standalone.las'
    write_pair(None, str(lnt), str(standalone), mapping={999: 255})
    standalone_data = laspy.read(standalone)
    assert len(standalone_data.points) == n
    assert np.array_equal(standalone_data.classification, record_slots)
    assert standalone.with_suffix('.lnt').is_file()
    print('PASS standalone LNT exports all leaf records and a native companion', flush=True)
    attributes = scratch / 'attributes.las'
    write_pair(str(source), str(lnt), str(attributes), mapping={999: 255}, cache=cache,
               attributes=['Class', 'Color RGB', 'Line', 'Time', 'Echo'])
    assert not np.any(laspy.read(attributes).intensity)
    assert len(read_classes(str(attributes.with_suffix('.lnt')))[0]) == n
    print('PASS Save As attribute choices apply to LAS and rebuilt LNT', flush=True)
    foreign = scratch / 'foreign.las'
    altered = laspy.read(source)
    altered.x = altered.x + 1
    altered.write(foreign)
    unchanged = foreign.read_bytes()
    try:
        write_pair(str(foreign), str(lnt), str(foreign), mapping={999: 255}, cache={})
    except ValueError:
        pass
    else:
        raise AssertionError('foreign source was accepted')
    assert foreign.read_bytes() == unchanged
    print('PASS mismatched source coordinates are rejected before writing', flush=True)
    # Exercise the actual shared GUI entry point with app.data absent, as in
    # a freshly opened native session; a section subset must never be exported.
    app = QMainWindow()
    app._lnt_main_view = SimpleNamespace(native=object(), lnt_path=str(lnt))
    app._lnt_source_path = str(source)
    app.loaded_file = str(source)
    app.data = None
    app._class_slot_state = {'path': str(lnt), 'code_to_slot': {999: 255}, 'slot_to_code': {255: 999}}
    with patch('gui.native_save.write_pair', wraps=write_pair) as writer:
        assert save_native(app)
        assert writer.call_count == 1
    assert laspy.read(source).header.point_count == n
    from gui.save_pointcloud import save_pointcloud, save_current_pointcloud_in_place
    with patch('gui.native_save.save_native', return_value=True) as saver:
        assert save_pointcloud(app, show_dialog=False)
        assert save_current_pointcloud_in_place(app)
        assert saver.call_count == 2
    # Drive the Save As entry, including real Qt worker and switch to new pair.
    from unittest.mock import Mock
    options = Mock()
    options.Accepted = options.exec.return_value = 1
    options.selected_classes.return_value = None
    options.selected_lines.return_value = None
    options.selected_attributes.return_value = None
    options.selected_fences.return_value = []
    options.selected_format.return_value = 'LAZ 1.4'
    gui_target = str(scratch / 'gui-copy.laz')
    with patch('gui.save_points_dialog.SavePointsDialog', return_value=options), \
         patch('PySide6.QtWidgets.QFileDialog.getSaveFileName', return_value=(gui_target, 'LAZ 1.4 (*.laz)')), \
         patch('gui.lnt_open.open_with_lnt') as opened:
        assert save_pointcloud(app, show_dialog=True)
        opened.assert_called_once_with(app, gui_target)
    assert laspy.read(gui_target).header.point_count == n
    assert Path(gui_target).with_suffix('.lnt').is_file()
    print('PASS Save As options route through native worker and switch to the new LAS/LAZ + LNT pair', flush=True)
    app.close()
    print('PASS real Qt worker saves the full native document with app.data=None, without VTK', flush=True)
finally:
    builtins.__import__ = real_import
    shutil.rmtree(scratch)
