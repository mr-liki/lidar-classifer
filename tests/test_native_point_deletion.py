"""Native Vulkan selection deletion: scratch records only; no survey data."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox
from gui import lnt_native, lnt_section_bridge, native_save
from gui.native_select_rect import NativeSelectRect
from gui.native_shortcuts import native_classification_history


@pytest.fixture(scope='module')
def qt():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def cloud(tmp_path):
    import laspy
    source = tmp_path / 'source.las'
    header = laspy.LasHeader(point_format=7, version='1.4')
    header.add_extra_dim(laspy.ExtraBytesParams(name='record_id', type=np.uint32))
    las = laspy.LasData(header)
    las.x = [0, 0, 1, 2, 3, 4]
    las.y = [0, 0, 1, 2, 3, 4]
    las.z = [1, 1, 2, 3, 4, 5]
    las.classification = [255, 255, 2, 7, 0, 31]
    las.intensity = [10, 20, 30, 40, 50, 60]
    las.record_id = np.arange(6, dtype=np.uint32)
    las.gps_time = np.arange(6) + 100.5
    las.red = np.arange(6, dtype=np.uint16) + 100
    las.write(source)
    document = tmp_path / 'source.lnt'
    assert lnt_native.convert_las(str(source), str(document))
    return source, document


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.data = {}
        self.undo_stack, self.redo_stack, self.messages = [], [], []
        self.class_palette = {}
    def statusBar(self):
        return self
    def showMessage(self, message, timeout=0):
        self.messages.append(message)


class Native:
    def __init__(self):
        self.masks = []
        self.success = True
    def has_section_api(self):
        return True
    def set_deleted(self, node, local, classes):
        self.masks.append((node.copy(), local.copy(), classes.copy()))
        return self.success
    def frame(self, dt):
        pass


class Overlay:
    def __init__(self):
        self.outline, self.points = None, None
    def clear(self):
        self.outline, self.points = None, None
    def set_rectangle(self, a, b):
        self.outline = ('rectangle', a, b)
    def set_polygon(self, p):
        self.outline = ('polygon', p)
    def set_selected_points(self, p):
        self.points = p.copy()


@pytest.fixture
def selection(qt):
    app = App()
    view = SimpleNamespace(app=app, native=Native())
    app._lnt_main_view = view
    store = lnt_section_bridge.LntSectionStore(app, view)
    app._lnt_section_store = store
    records = {'xyz': np.array([[0., 0, 1], [0, 0, 1], [1, 1, 2], [2, 2, 3]]),
               'classification': np.array([255, 255, 7, 2], dtype=np.uint8),
               'intensity': np.ones(4, np.uint8), 'rgb': np.ones((4, 3), np.uint8),
               'node': np.zeros(4, np.uint32), 'local': np.arange(4, dtype=np.uint32)}
    assert store._ingest(records)
    view.native.query_points = lambda *args, **kwargs: records
    selector = NativeSelectRect(view)
    view.native_select = selector
    selector._overlay, selector._canvas_instance = Overlay(), SimpleNamespace()
    yield app, view, store, selector
    store._timer.stop()
    store._write_timer.stop()
    app.close()


@pytest.mark.parametrize('method', ['Block', 'Shape'])
@pytest.mark.parametrize('reply', [QMessageBox.Yes, QMessageBox.No, QMessageBox.Cancel])
def test_completed_selection_confirmation(selection, monkeypatch, method, reply):
    app, view, store, selector = selection
    before = store.data['classification'].copy()
    calls = []
    def confirm(*args):
        assert selector._overlay.outline[0] == ('rectangle' if method == 'Block' else 'polygon')
        assert len(selector._overlay.points) == 3
        assert selector.selected_rows.tolist() == [0, 1, 2]
        calls.append(args)
        return reply
    monkeypatch.setattr(QMessageBox, 'question', confirm)
    if method == 'Block':
        selector.start, selector.current = (-1, -1, 0), (1.5, 1.5, 0)
        selector._finish_block()
    else:
        selector.vertices = [(-1, -1, 0), (2, -1, 0), (2, 1.5, 0), (-1, 1.5, 0)]
        selector._finish_shape()
    assert len(calls) == 1
    assert store.deleted.tolist() == ([True, True, True, False] if reply == QMessageBox.Yes else [False]*4)
    np.testing.assert_array_equal(store.data['classification'], before)
    assert bool(app.undo_stack) == (reply == QMessageBox.Yes)


def test_multiple_deletions_undo_redo_discard_and_duplicate_identity(selection, monkeypatch):
    app, view, store, selector = selection
    monkeypatch.setattr(QMessageBox, 'question', lambda *a: QMessageBox.Yes)
    selector.selected_rows = np.array([0])  # only one of two coincident records
    selector.delete_selected_points()
    assert store.deleted.tolist() == [True, False, False, False]
    selector._query_box(-1, 3, -1, 3)
    assert selector.selected_rows.tolist() == [1, 2, 3]
    selector.selected_rows = np.array([2])
    selector.delete_selected_points()
    assert store.deleted.tolist() == [True, False, True, False]
    assert native_classification_history(app)
    assert store.deleted.tolist() == [True, False, False, False]
    assert native_classification_history(app, redo=True)
    assert store.deleted.tolist() == [True, False, True, False]
    assert store.discard_unflushed_edits() == 2
    assert not store.deleted.any()


def test_failed_delete_and_failed_query_keep_no_old_target(selection):
    app, view, store, selector = selection
    view.native.success = False
    with pytest.raises(RuntimeError):
        store.set_deleted([1])
    assert not store.deleted.any()
    selector.selected_rows = np.array([1])
    store.ensure_region = lambda *args: False
    selector._query_box(0, 1, 0, 1)
    assert selector.selected_count == 0


def test_empty_selection_no_confirmation(selection, monkeypatch):
    app, view, store, selector = selection
    monkeypatch.setattr(QMessageBox, 'question', lambda *a: pytest.fail('empty confirmation'))
    selector.start, selector.current = (50, 50, 0), (51, 51, 0)
    selector._finish_block()
    assert selector.selected_count == 0
    assert not store.deleted.any()


def identities(document, remove):
    slots, entries, leaves, starts = native_save.read_classes(document)
    leaf_positions = np.searchsorted(starts[1:], remove, side='right')
    return leaves[leaf_positions].astype(np.uint32), (remove - starts[leaf_positions]).astype(np.uint32)


@pytest.mark.parametrize('extension', ['.las', '.laz'])
@pytest.mark.parametrize('all_points', [False, True])
def test_save_compacts_both_files_and_preserves_survivor_attributes(cloud, extension, all_points):
    import laspy
    source, document = cloud
    before = laspy.read(source)
    slots, entries, leaves, starts = native_save.read_classes(document)
    order = native_save.source_mapping(source, document, entries, leaves, starts, {}, lambda *a: None)
    remove = np.arange(6) if all_points else np.array([0, 3])
    output = source.with_name('saved' + extension)
    result = native_save.write_pair(source, document, output, deleted=identities(document, remove))
    keep = np.ones(6, bool)
    keep[order[remove]] = False
    saved = laspy.read(output)
    np.testing.assert_array_equal(saved.points.array, before.points.array[keep])
    assert result['count'] == keep.sum()
    native_classes, *_ = native_save.read_classes(output.with_suffix('.lnt'))
    assert len(native_classes) == keep.sum()
    np.testing.assert_array_equal(laspy.read(source).points.array, before.points.array)


def test_in_place_save_bypasses_unchanged_cache_and_publish_failure_rolls_back(cloud, monkeypatch):
    source, document = cloud
    cache = {}
    native_save.write_pair(source, document, source, cache=cache)
    original_source, original_native = source.read_bytes(), document.read_bytes()
    deletions = identities(document, np.array([0]))
    publish = native_save._publish
    def fail(*args):
        raise OSError('injected publication failure')
    monkeypatch.setattr(native_save, '_publish', fail)
    with pytest.raises(OSError, match='injected'):
        native_save.write_pair(source, document, source, cache=cache, deleted=deletions)
    assert source.read_bytes() == original_source
    assert document.read_bytes() == original_native
    monkeypatch.setattr(native_save, '_publish', publish)
    result = native_save.write_pair(source, document, source, cache=cache, deleted=deletions)
    assert result['count'] == 5 and not result['unchanged']
    assert len(native_save.read_classes(document)[0]) == 5


def test_vulkan_highlight_chunking_and_lifecycle(monkeypatch):
    from gui import native_zoom_preview as preview, native_main_classify
    class Layer:
        def __init__(self):
            self.documents, self.uploads, self.removed = {}, [], []
        def put_entities(self, kind, name, entities, **kwargs):
            self.uploads.append((kind, name, kwargs['prepared_groups']))
            return {'ok': True}
        def remove_file(self, kind, name):
            self.removed.append((kind, name))
    layer = Layer()
    monkeypatch.setattr(preview, 'vector_layer_for', lambda *a, **k: layer)
    monkeypatch.setattr(preview, '_cloud_z_bounds', lambda app: (0, 5))
    monkeypatch.setattr(native_main_classify, 'MainCanvas', lambda view: SimpleNamespace(visible_extent=lambda: (0, 100, 0, 100)))
    overlay = preview.VulkanSelectionPreview(SimpleNamespace(app=object()))
    points = np.zeros((65537, 3))
    points[:, 0] = np.arange(len(points))
    overlay.set_selected_points(points)
    chunks = layer.uploads[-1][2][overlay.LAYER]
    assert [c['v_count'] for c in chunks] == [65536*4, 4]
    for c in chunks:
        assert (c['indices'].reshape(-1, 3)[:, 2] == preview.RESTART_INDEX).all()
        assert np.isfinite(c['xyz']).all()
    overlay.shutdown()
    assert (overlay.KIND, overlay.NAME + '_points') in layer.removed
    assert (overlay.KIND, overlay.NAME) in layer.removed

@pytest.mark.parametrize('tiled', [True, False])
def test_real_vulkan_points_and_surfaces_respect_deletions(tmp_path, monkeypatch, tiled):
    """Use a private hidden Win32 window, a real Vulkan device, and scratch data."""
    if os.name != 'nt':
        pytest.skip('Win32 native viewer integration')
    import ctypes
    import time
    from ctypes import wintypes
    import laspy
    monkeypatch.setenv('NAKSHA_LNT_SURFACE_LOD', '1' if tiled else '0')
    monkeypatch.setenv('NAKSHA_LNT_SURFACE_PERSIST', '0')
    source, document = tmp_path / 'grid.las', tmp_path / 'grid.lnt'
    las = laspy.LasData(laspy.LasHeader(point_format=7, version='1.4'))
    x, y = np.meshgrid(np.arange(5), np.arange(5))
    las.x, las.y, las.z = x.ravel(), y.ravel(), np.zeros(25)
    las.classification = np.full(25, 255, dtype=np.uint8)
    las.write(source)
    lnt_native.convert_las(str(source), str(document))
    user32 = ctypes.windll.user32
    user32.CreateWindowExW.restype = wintypes.HWND
    hwnd = user32.CreateWindowExW(0, 'STATIC', 'Scratch Vulkan deletion test',
                                0x80000000, 0, 0, 400, 400, None, None, None, None)
    assert hwnd
    view = None
    try:
        view = lnt_native.NativeView(hwnd, 400, 400, str(document))
        view.set_2d(True)
        def pump(predicate):
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                assert view.frame(.02)
                if predicate():
                    return
                time.sleep(.01)
            pytest.fail(f'Native draw did not converge: points={view.stats.points_rendered}, surface={view.surface_point_counts()}, error={view.surface_error()}')
        pump(lambda: view.stats.points_rendered == 25)
        records = view.query_points((-1, -1, -1), (6, 6, 1))
        assert view.set_deleted(records['node'][:1], records['local'][:1], records['classification'][:1])
        pump(lambda: view.stats.points_rendered == 24)
        assert view.set_shading(lnt_native.SHADE_NORMAL)
        pump(lambda: view.surface_point_counts()[1] == 24)
        assert not view.surface_error()
        assert view.set_deleted(records['node'], records['local'], records['classification'])
        pump(lambda: view.surface_point_counts()[1] == 0)
        empty = np.empty(0, dtype=np.uint32)
        assert view.set_deleted(empty, empty, np.empty(0, np.uint8))
        pump(lambda: view.surface_point_counts()[1] == 25)
        view.close()  # join any private derived-mesh writer before reading
        np.testing.assert_array_equal(native_save.read_classes(document)[0], np.full(25, 255, np.uint8))
    finally:
        if view is not None:
            view.close()
        user32.DestroyWindow(wintypes.HWND(hwnd))


def test_partial_publication_failure_restores_native_and_las(cloud, monkeypatch):
    source, document = cloud
    original_source, original_native = source.read_bytes(), document.read_bytes()
    original_lines = Path(str(document) + '.lines').read_bytes()
    replace = os.replace
    failed = False
    def injected(src, dst):
        nonlocal failed
        if str(dst) == str(source) and not failed:
            failed = True
            raise OSError('fail after companion publication')
        return replace(src, dst)
    monkeypatch.setattr(os, 'replace', injected)
    with pytest.raises(OSError, match='after companion'):
        native_save.write_pair(source, document, source, deleted=identities(document, np.array([0])))
    assert failed
    assert source.read_bytes() == original_source
    assert document.read_bytes() == original_native
    assert Path(str(document) + '.lines').read_bytes() == original_lines


@pytest.mark.parametrize('fails', [False, True])
def test_save_entrypoint_rebinds_compacted_document_only_after_success(qt, cloud, monkeypatch, fails):
    from gui import lnt_open, save_pointcloud
    source, document = cloud
    app = App()
    app._lnt_main_view = SimpleNamespace(lnt_path=str(document), native=Native())
    app._lnt_source_path = str(source)
    app._lnt_source_paths = [str(source)]
    deleted = identities(document, np.array([0]))
    store = SimpleNamespace(view=app._lnt_main_view, deletion_identities=lambda: deleted)
    app._lnt_section_store = store
    app.undo_stack.append({'deleted_rows': np.array([0])})
    reopened, errors = [], []
    monkeypatch.setattr(native_save, 'flush_edits', lambda app: None)
    monkeypatch.setattr(save_pointcloud, '_serialize_drawings', lambda app: None)
    monkeypatch.setattr(lnt_open, 'open_with_lnt', lambda app, destination: reopened.append(destination))
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: errors.append(args))
    before = source.read_bytes()
    if fails:
        monkeypatch.setattr(native_save, 'write_pair', lambda *a, **k: (_ for _ in ()).throw(OSError('save failure')))
    assert native_save.save_native(app) == (not fails)
    if fails:
        assert app.undo_stack and not reopened and errors
        assert source.read_bytes() == before
        assert native_save.pending_deletions(app) is not None
    else:
        assert reopened == [str(source)]
        assert not app.undo_stack
        assert len(native_save.read_classes(document)[0]) == 5
    app.close()


def test_sections_and_classifier_exclude_deleted_rows_and_undo_restores(selection):
    from gui.native_sections import SectionCanvas, NativeSections
    from gui.native_section_classify import SectionClassifier
    app, view, store, selector = selection
    rows = np.arange(4)
    local = store.data['xyz'].copy()
    canvas = SimpleNamespace(owner=SimpleNamespace(app=app), axis=0,
        local=np.empty((0, 3)), indices=np.empty(0, dtype=np.int64),
        _drop_mesh_cache=lambda: None, _retire_mesh=lambda: None, start=lambda: False)
    canvas.set_points = lambda *a, **k: SectionCanvas.set_points(canvas, *a, **k)
    canvas.set_points(local, rows)
    app._native_sections = SimpleNamespace(views={1: canvas}, _classes={})
    app._native_sections.refresh_deletions = lambda: NativeSections.refresh_deletions(app._native_sections)
    store.set_deleted([0, 2])
    assert canvas.indices.tolist() == [1, 3]
    store.set_deleted([0, 2], False)
    assert canvas.indices.tolist() == [0, 1, 2, 3]
    # Even a stale cached canvas / brush candidate list cannot reclassify deleted rows.
    classifier = SectionClassifier.__new__(SectionClassifier)
    classifier.app = app
    classifier.canvas = SimpleNamespace(indices=rows, exact_selection='rectangle')
    classifier._visible_classes = lambda: None
    store.set_deleted([1])
    assert classifier.candidates('rectangle', []).tolist() == [0, 2, 3]


def test_workflow_does_not_import_vtk(selection, cloud, monkeypatch):
    import builtins
    app, view, store, selector = selection
    original_import = builtins.__import__
    def native_only(name, *args, **kwargs):
        if name == 'vtk' or name.startswith(('vtkmodules', 'pyvista')):
            raise AssertionError('VTK requested by native deletion workflow')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', native_only)
    monkeypatch.setattr(QMessageBox, 'question', lambda *a: QMessageBox.Yes)
    selector.start, selector.current = (-1, -1, 0), (1.5, 1.5, 0)
    selector._finish_block()
    assert native_classification_history(app)
    assert native_classification_history(app, redo=True)
    source, document = cloud
    native_save.write_pair(source, document, source.with_name('native_only.las'),
                           deleted=identities(document, np.array([0])))
    selector.cancel()
    selector.shutdown()



def test_backup_excludes_pending_deletions_without_publishing_companion(qt, cloud):
    from gui import native_backup
    import laspy
    source, document = cloud
    before_source, before_native = source.read_bytes(), document.read_bytes()
    app = App()
    app._lnt_main_view = SimpleNamespace(lnt_path=str(document), native=Native())
    app._lnt_source_path = str(source)
    app._lnt_source_paths = [str(source)]
    app._lnt_section_store = SimpleNamespace(
        deletion_identities=lambda: identities(document, np.array([0])), deletion_revision=1)
    destination = source.with_name('backup.laz')
    assert native_backup.has_work(app)
    result = native_backup.export_backup(app, destination, flush=False, mapping={}, drawings=b'')
    assert result['count'] == 5
    assert len(laspy.read(destination).points) == 5
    assert not destination.with_suffix('.lnt').exists()
    assert source.read_bytes() == before_source and document.read_bytes() == before_native
    app.close()


def test_empty_saved_companion_reopens_in_native_vulkan(tmp_path):
    if os.name != 'nt':
        pytest.skip('Win32 native viewer integration')
    import ctypes
    from ctypes import wintypes
    import struct
    import laspy
    source, document = tmp_path / 'empty.las', tmp_path / 'empty.lnt'
    laspy.LasData(laspy.LasHeader(point_format=7, version='1.4')).write(source)
    document.write_bytes(struct.pack('<4sHHQ6dQQ', b'LNT\0', 2, 0, 64, 0, 0, 0, 1, 1, 1, 0, 0))
    user32 = ctypes.windll.user32
    user32.CreateWindowExW.restype = wintypes.HWND
    hwnd = user32.CreateWindowExW(0, 'STATIC', 'Empty native scratch cloud', 0x80000000,
                                0, 0, 400, 400, None, None, None, None)
    view = None
    try:
        view = lnt_native.NativeView(hwnd, 400, 400, str(document))
        assert view.info()[0] == 0
        assert view.frame(.02)
        assert view.stats.points_rendered == 0
    finally:
        if view:
            view.close()
        user32.DestroyWindow(wintypes.HWND(hwnd))
    # A later Save of an empty cloud must keep a valid empty pair.
    result = native_save.write_pair(source, document, source)
    assert result['count'] == 0
    assert len(native_save.read_classes(document)[0]) == 0


def test_native_coarse_lod_does_not_resurrect_deleted_records(tmp_path):
    if os.name != 'nt':
        pytest.skip('Win32 native viewer integration')
    import ctypes
    import time
    import laspy
    from ctypes import wintypes
    source, document = tmp_path / 'lod.las', tmp_path / 'lod.lnt'
    las = laspy.LasData(laspy.LasHeader(point_format=7, version='1.4'))
    x, y = np.meshgrid(np.arange(245), np.arange(245))  # exceeds native 60k leaf capacity
    las.x, las.y, las.z = x.ravel(), y.ravel(), np.zeros(x.size)
    las.write(source)
    lnt_native.convert_las(str(source), str(document))
    assert len(native_save.read_classes(document)[2]) > 1
    user32 = ctypes.windll.user32
    user32.CreateWindowExW.restype = wintypes.HWND
    hwnd = user32.CreateWindowExW(0, 'STATIC', 'Native LOD scratch test', 0x80000000,
                                0, 0, 400, 400, None, None, None, None)
    view = None
    try:
        view = lnt_native.NativeView(hwnd, 400, 400, str(document))
        view.set_2d(True)
        view.set_sse(1.e9)
        view.set_point_budget((x.size + 1) // 2)  # keep the strided root cut
        def pump(expected):
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                assert view.frame(.02)
                if view.stats.points_rendered == expected:
                    return
                time.sleep(.01)
            pytest.fail(f'Expected {expected} native LOD points; got {view.stats.points_rendered}')
        root_count = (x.size + 1) // 2
        pump(root_count)
        records = view.query_points((-1, -1, -1), (246, 246, 1))
        # First exclude a record absent from the root sample, then a sampled
        # record. An earlier unresolved stride must never poison later deletions.
        assert view.set_deleted(records['node'][1:2], records['local'][1:2], records['classification'][1:2])
        pump(root_count)
        assert view.set_deleted(records['node'][:2], records['local'][:2], records['classification'][:2])
        pump(root_count - 1)
        assert view.set_deleted(records['node'], records['local'], records['classification'])
        pump(0)
        empty = np.empty(0, np.uint32)
        assert view.set_deleted(empty, empty, np.empty(0, np.uint8))
        pump(root_count)
    finally:
        if view:
            view.close()
        user32.DestroyWindow(wintypes.HWND(hwnd))
