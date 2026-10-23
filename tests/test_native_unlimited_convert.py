"""Exact native Convert beyond the former cap, including Undo and rollback."""
import gc
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gui.native_lnt_directory import (
    HEADER, DIR, ENTRY, POINT, LNTDirectoryError, plan_exact_class_conversion,
)
from gui.native_conversion_history import ConversionJournal, ConversionCancelled
import gui.native_conversion_history as history


def make_lnt(path, total=13, leaf_size=200_000):
    leaves = []
    with open(path, 'wb') as handle:
        handle.write(bytes(HEADER.size))
        for start in range(0, total, leaf_size):
            count = min(leaf_size, total - start)
            points = np.zeros(count, dtype=POINT)
            points['classification'] = (np.arange(count) + start) % 2 + 2
            offset = handle.tell()
            points.tofile(handle)
            leaves.append((len(leaves), offset, count, (0.5, 0.5, 0.5), 0.5))
        index_at = handle.tell()
        handle.write(DIR.pack(total, len(leaves)))
        for node, offset, count, center, half in leaves:
            handle.write(ENTRY.pack(node + 1, offset, count * POINT.itemsize, count,
                                    *center, half, 0, 1, bytes(10)))
        handle.seek(0)
        handle.write(HEADER.pack(b'LNT\x00', 2, 0, index_at, 0., 0., 0., 1., 1., 1.))
    return leaves


class NativeWriter:
    """Enforce bounded calls and emulate partially failing native writes."""
    last_error = 'injected write failure'

    def __init__(self, path, leaves):
        self.values = {}
        with open(path, 'rb') as handle:
            for node, offset, count, *_ in leaves:
                handle.seek(offset)
                self.values[node] = np.fromfile(handle, dtype=POINT, count=count)['classification'].copy()
        self.calls = 0
        self.largest_batch = 0
        self.fail_at = None
        self.fail_always = False

    def has_section_api(self):
        return True

    def frame(self, _dt):
        pass

    def set_classes(self, node, local, values):
        self.calls += 1
        self.largest_batch = max(self.largest_batch, len(values))
        assert len(values) <= history.BATCH_SIZE
        fail = self.fail_always or self.calls == self.fail_at
        if fail:  # Simulate a writer that edits some records before returning false.
            node, local, values = node[:2], local[:2], values[:2]
        for key in np.unique(node):
            rows = node == key
            self.values[int(key)][local[rows]] = values[rows]
        return not fail

    def assert_target(self, target):
        assert all(np.all(values == target) for values in self.values.values())

    def assert_original(self):
        start = 0
        for values in self.values.values():
            np.testing.assert_array_equal(values, (np.arange(len(values)) + start) % 2 + 2)
            start += len(values)


@pytest.fixture
def planned(tmp_path, monkeypatch):
    monkeypatch.setattr(history, 'BATCH_SIZE', 4)
    path = tmp_path / 'test.lnt'
    leaves = make_lnt(path)
    journal = ConversionJournal(7)
    plan_exact_class_conversion(path, [2, 3], 7, journal=journal, chunk_size=3)
    journal.seal()
    native = NativeWriter(path, leaves)
    yield journal, native, path
    journal.close()


def test_more_than_eight_million_edits_roundtrip_in_bounded_batches(tmp_path):
    path = tmp_path / 'large.lnt'
    total = 8_000_123
    leaves = make_lnt(path, total)
    journal = ConversionJournal(7)
    try:
        plan, info = plan_exact_class_conversion(
            path, [2, 3], 7, expected_total=total, journal=journal)
        journal.seal()
        assert journal.count == total
        assert info['total'] == total
        assert set(plan) == {'journal', 'file_stat', 'stats'}  # No whole-edit-set arrays.
        assert os.path.getsize(journal.path) == total * history.RECORD.itemsize
        native = NativeWriter(path, leaves)
        journal.replay(native)
        native.assert_target(7)
        journal.replay(native, forward=False)
        native.assert_original()
        journal.replay(native)
        native.assert_target(7)
        assert native.largest_batch <= 250_000
    finally:
        journal.close()
    assert not Path(journal.path).exists()


@pytest.mark.parametrize('forward', [True, False])
def test_partial_native_failure_rolls_back_current_and_prior_batches(planned, forward):
    journal, native, _ = planned
    if not forward:
        journal.replay(native)
    native.fail_at = native.calls + 2
    with pytest.raises(RuntimeError, match='rolled back'):
        journal.replay(native, forward=forward)
    if forward:
        native.assert_original()
    else:
        native.assert_target(7)


@pytest.mark.parametrize('forward', [True, False])
def test_cancel_after_writes_rolls_back_without_partial_conversion(planned, forward):
    journal, native, _ = planned
    if not forward:
        journal.replay(native)
    cancelled = False
    def report(done, total):
        nonlocal cancelled
        cancelled = True
    with pytest.raises(ConversionCancelled, match='rolled back'):
        journal.replay(native, forward=forward, progress=report, cancelled=lambda: cancelled)
    if forward:
        native.assert_original()
    else:
        native.assert_target(7)


def test_corrupt_history_rejected_before_any_write(planned):
    journal, native, _ = planned
    with open(journal.path, 'r+b') as handle:
        handle.truncate(1)
    with pytest.raises(RuntimeError, match='incomplete'):
        journal.replay(native)
    assert native.calls == 0


def test_failed_rollback_retains_recovery_file(planned):
    journal, native, _ = planned
    native.fail_always = True
    with pytest.raises(RuntimeError, match='Recovery history retained'):
        journal.replay(native)
    journal.close()
    assert Path(journal.path).exists()
    Path(journal.path).unlink()


def test_filters_and_target_noop_match_exact_original_classes(tmp_path):
    path = tmp_path / 'test.lnt'
    make_lnt(path)
    plan, _ = plan_exact_class_conversion(path, [2, 3], 3, visible_classes=[2])
    assert len(plan['node']) == 7
    assert np.all(plan['old_classes'] == 2)
    plan, _ = plan_exact_class_conversion(path, [2], 7, visible_classes=[3])
    assert len(plan['node']) == 0


def test_cancelled_preflight_does_not_write_source(tmp_path):
    path = tmp_path / 'test.lnt'
    make_lnt(path)
    before = path.read_bytes()
    journal = ConversionJournal(7)
    try:
        with pytest.raises(LNTDirectoryError, match='cancelled'):
            plan_exact_class_conversion(path, [2, 3], 7, journal=journal,
                                        cancelled=lambda: journal.count > 0, chunk_size=3)
        assert path.read_bytes() == before
    finally:
        journal.close()


def test_changed_file_rejected_during_preflight(tmp_path):
    path = tmp_path / 'test.lnt'
    make_lnt(path)
    def progress(*_):
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    with pytest.raises(LNTDirectoryError, match='changed while'):
        plan_exact_class_conversion(path, [2], 7, progress=progress)


def test_history_trim_and_discard_release_disk_journal(planned):
    from gui.memory_manager import trim_undo_stack
    journal, _, _ = planned
    app = SimpleNamespace(undo_stack=[{'lnt_conversion': journal}], redo_stack=[])
    trim_undo_stack(app, max_steps=0)
    assert not app.undo_stack
    assert not Path(journal.path).exists()
    discarded = ConversionJournal(7)
    path = discarded.path
    del discarded
    gc.collect()
    assert not Path(path).exists()


def test_native_dialog_conversion_undo_redo_and_preview(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMainWindow, QDialog
    import gui.native_byclass_parity as parity
    import gui.lnt_ai_bridge as bridge
    import gui.point_count_widget as counts
    from gui.native_shortcuts import native_classification_history

    qt = QApplication.instance() or QApplication([])
    path = tmp_path / 'test.lnt'
    leaves = make_lnt(path)
    native = NativeWriter(path, leaves)
    app = QMainWindow()
    dialog = QDialog(app)
    dialog.app = app
    class ForbiddenVTK:
        def __getattribute__(self, name):
            raise AssertionError('Legacy VTK accessed: ' + name)
    app.vtk_widget = ForbiddenVTK()
    view = SimpleNamespace(lnt_path=str(path), native=native)
    app._lnt_main_view = view
    app.undo_stack, app.redo_stack = [], []
    app.class_palette = {}
    monkeypatch.setattr(parity, 'MAX_EXACT_POINTS', 2)
    monkeypatch.setattr(counts, 'refresh_point_statistics', lambda *_: None)
    synced = []
    monkeypatch.setattr(bridge, 'sync_store_rows', lambda a, v, n, l, c: synced.append(c.copy()))
    try:
        def forbidden_small_cloud(*_):
            raise AssertionError('Whole-cloud snapshot fallback used.')
        preview = parity._with_progress(
            dialog, 'Convert', forbidden_small_cloud, preview=True, convert_spec=([2, 3], 7))
        assert preview == 13 and native.calls == 0 and not app.undo_stack
        amount = parity._with_progress(
            dialog, 'Convert', forbidden_small_cloud, convert_spec=([2, 3], 7))
        assert amount == 13 and len(app.undo_stack) == 1
        journal = app.undo_stack[-1]['lnt_conversion']
        native.assert_target(7)
        assert native_classification_history(app)
        native.assert_original()
        assert len(app.redo_stack) == 1 and not app.undo_stack
        assert native_classification_history(app, redo=True)
        native.assert_target(7)
        assert len(app.undo_stack) == 1 and not app.redo_stack
        assert synced and not app._native_byclass_busy
        journal.close()
    finally:
        app.undo_stack.clear()
        app.redo_stack.clear()
        dialog.close()
        app.close()
        qt.processEvents()

def test_section_sync_failure_does_not_skip_remaining_native_rollback(planned):
    journal, native, _ = planned
    calls = 0
    def sync(*_):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise RuntimeError('section synchronization failed')
    with pytest.raises(RuntimeError, match='Recovery history retained'):
        journal.replay(native, sync=sync)
    native.assert_original()
    Path(journal.path).unlink()


def test_history_write_failure_keeps_entry_in_undo_stack(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMainWindow
    from gui.native_shortcuts import native_classification_history
    import gui.lnt_ai_bridge as bridge

    qt = QApplication.instance() or QApplication([])
    monkeypatch.setattr(history, 'BATCH_SIZE', 4)
    path = tmp_path / 'test.lnt'
    leaves = make_lnt(path)
    native = NativeWriter(path, leaves)
    journal = ConversionJournal(7)
    plan_exact_class_conversion(path, [2, 3], 7, journal=journal)
    journal.seal()
    journal.replay(native)
    native.fail_at = native.calls + 2
    app = QMainWindow()
    app._lnt_main_view = SimpleNamespace(lnt_path=str(path), native=native)
    app.undo_stack = [{'lnt_path': str(path), 'lnt_conversion': journal}]
    app.redo_stack = []
    monkeypatch.setattr(bridge, 'sync_store_rows', lambda *_: None)
    try:
        assert native_classification_history(app)
        native.assert_target(7)
        assert len(app.undo_stack) == 1 and not app.redo_stack
        assert not app._native_byclass_busy
        assert 'rolled back' in app.statusBar().currentMessage()
    finally:
        journal.close()
        app.undo_stack.clear()
        app.close()
        qt.processEvents()

@pytest.mark.parametrize("query_threshold", [2, 100])
@pytest.mark.parametrize("sources,matched,changed,already", [
    (None, 13, 7, 6),
    ([2], 7, 7, 0),
    ([2, 3], 13, 7, 6),
    ([3], 6, 0, 6),
])
def test_convert_dialog_ignores_viewport_and_hidden_classes_for_every_file_size(
        tmp_path, monkeypatch, query_threshold, sources, matched, changed, already):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QDialog, QListWidget, QListWidgetItem, QLabel,
    )
    from unittest.mock import Mock
    import gui.native_byclass_parity as parity
    import gui.lnt_ai_bridge as bridge
    import gui.point_count_widget as counts

    qt = QApplication.instance() or QApplication([])
    path = tmp_path / 'all-leaves.lnt'
    leaves = make_lnt(path, leaf_size=4)
    native = NativeWriter(path, leaves)
    native.query_box = Mock(side_effect=AssertionError('Viewport query used by Convert'))
    app = QMainWindow()
    dialog = QDialog(app)
    dialog.app = app
    dialog.from_list = QListWidget(dialog)
    for code in ([None] if sources is None else sources):
        item = QListWidgetItem(str(code), dialog.from_list)
        item.setData(Qt.UserRole, code)
    # Qt's default single-selection policy is bypassed by returning this explicit selection.
    selected = [dialog.from_list.item(i) for i in range(dialog.from_list.count())]
    dialog.from_list.selectedItems = lambda: selected
    dialog.to_combo = SimpleNamespace(currentData=lambda: 3)
    dialog.info_label = QLabel(dialog)
    dialog.ribbon_parent = SimpleNamespace(update_status=Mock())
    dialog.selected_fences = [object()]  # Convert must ignore any selected fence.
    class ForbiddenVTK:
        def __getattribute__(self, name):
            raise AssertionError('Legacy VTK accessed: ' + name)
    app.vtk_widget = ForbiddenVTK()
    app._lnt_main_view = SimpleNamespace(lnt_path=str(path), native=native)
    app.undo_stack, app.redo_stack = [], []
    app.class_palette = {2: {'show': False}, 3: {'show': False}}
    app.flight_line_visibility = {17: False}
    monkeypatch.setattr(parity, 'MAX_EXACT_POINTS', query_threshold)
    monkeypatch.setattr(parity, '_ask', lambda *_: True)
    reports = []
    monkeypatch.setattr(parity, '_report', lambda d, message, error=False: reports.append((message, error)))
    monkeypatch.setattr(counts, 'refresh_point_statistics', lambda *_: None)
    monkeypatch.setattr(bridge, 'sync_store_rows', lambda *_: None)
    try:
        assert parity.route_convert_dialog(dialog, 'convert')
        assert len(reports) == 1 and not reports[0][1]
        assert dialog._native_convert_stats == dict(
            scanned=13, matched=matched, changed=changed, already_target=already)
        if sources == [3]:
            native.assert_original()
        else:
            native.assert_target(3)  # All leaves, including those outside any viewport.
        native.query_box.assert_not_called()
        assert len(app.undo_stack) == int(changed > 0)
        text = dialog.info_label.text()
        assert 'Scanned 13' in text and f'matched {matched}' in text
        assert f'Changed {changed}' in text and f'target class: {already}' in text
        if changed:
            assert sum(len(old) for _, _, old in app.undo_stack[0]['lnt_conversion'].batches()) == changed
    finally:
        for entry in app.undo_stack:
            entry['lnt_conversion'].close()
        app.undo_stack.clear()
        app.redo_stack.clear()
        dialog.close()
        app.close()
        qt.processEvents()


def test_full_file_any_class_accounting_includes_existing_target_points(tmp_path):
    path = tmp_path / 'test.lnt'
    make_lnt(path, total=10_001, leaf_size=200)
    plan, _ = plan_exact_class_conversion(path, None, 3, chunk_size=75)
    assert plan['stats'] == dict(scanned=10_001, matched=10_001,
                                changed=5_001, already_target=5_000)
    assert len(plan['node']) == 5_001
    assert len(np.unique(plan['node'])) == 51
