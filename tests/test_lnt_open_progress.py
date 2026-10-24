"""First-open builds and native loading share one continuous progress window."""
import os
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from gui import lnt_open


class Dialog:
    def __init__(self, *args):
        self.values, self.labels = [], []
        self.closed = 0
        self.cancelled = False
        self.canceled = SimpleNamespace(disconnect=Mock())
        self.title = ''

    def setValue(self, value):
        self.values.append(value)

    def value(self):
        return self.values[-1] if self.values else 0

    def setLabelText(self, text):
        self.labels.append(text)

    def wasCanceled(self):
        return self.cancelled

    def close(self):
        self.closed += 1

    def setWindowTitle(self, title):
        self.title = title

    def setWindowModality(self, *args):
        pass

    def setMinimumDuration(self, *args):
        pass

    def setAutoClose(self, *args):
        pass

    def setAutoReset(self, *args):
        pass


@pytest.fixture
def opening(tmp_path, monkeypatch):
    qt = QApplication.instance() or QApplication([])
    app = QWidget()
    app._lnt_main_view = None
    app._lnt_quit_hook = True
    app._lnt_import_class_filter = {2}
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    source = tmp_path / 'survey.laz'
    document = tmp_path / 'survey.lnt'
    document.write_bytes(b'scratch native document')
    dialogs, views = [], []

    def make_dialog(*args):
        dialog = Dialog(*args)
        dialogs.append(dialog)
        return dialog

    class View:
        def __init__(self, app, working, name):
            self.app, self.lnt_path, self.name = app, working, name
            self.started = False
            views.append(self)

        def start(self):
            dialog = dialogs[-1]
            assert not dialog.closed
            assert dialog.value() == 99
            assert 100 not in dialog.values
            assert self.lnt_path != str(document)
            self.started = True

        def stop(self):
            lnt_open._discard_session_lnt(self)

    monkeypatch.setattr(lnt_open, 'QProgressDialog', make_dialog)
    monkeypatch.setattr(lnt_open, 'LntMainView', View)
    monkeypatch.setattr(lnt_open, 'close_lnt_view', Mock())
    monkeypatch.setattr(lnt_open, '_keep_vector_overlay', Mock())
    monkeypatch.setattr(lnt_open, '_apply_import_class_filter', Mock())
    monkeypatch.setattr(lnt_open, '_refresh_total_points', Mock())
    from gui import native_border
    monkeypatch.setattr(native_border, 'apply_border', Mock())
    errors = Mock()
    monkeypatch.setattr(lnt_open.QMessageBox, 'critical', errors)
    worker = SimpleNamespace(sources=[str(source)], render_source=str(source),
                             target_crs_wkt=None, cancel=Mock())
    thread = SimpleNamespace(quit=Mock(), wait=Mock())
    state = SimpleNamespace(app=app, source=source, document=document, dialogs=dialogs,
                            views=views, errors=errors, worker=worker, thread=thread,
                            make_dialog=make_dialog, qt=qt)
    yield state
    for view in views:
        view.stop()
    app.close()


def build_ui(state):
    dialog = state.make_dialog()
    dialog.setWindowTitle('Optimizing point cloud')
    dialog.setValue(0)
    ui = lnt_open._BuildUi(state.app, str(state.source), dialog, state.thread, state.worker)
    state.app._lnt_build = (state.thread, state.worker, dialog, ui)
    return ui, dialog


def test_first_build_continues_into_loading_in_same_dialog(opening):
    ui, dialog = build_ui(opening)
    ui.on_progress(50, 'Decompressing LAZ')
    ui.on_progress(100, 'Built native document')
    assert dialog.value() == 85
    ui.on_finished(str(opening.document), '')
    assert opening.dialogs == [dialog]
    assert dialog.title == 'Optimizing point cloud'
    assert dialog.values == sorted(dialog.values)
    assert dialog.values[-1] == 100
    assert any(label.startswith('Loading ') for label in dialog.labels)
    assert any(label.startswith('Opening ') for label in dialog.labels)
    assert dialog.closed == 1
    assert opening.views[0].started
    assert opening.app._lnt_build is None
    opening.thread.quit.assert_called_once()
    opening.worker.cancel.assert_not_called()


def test_cached_open_keeps_standalone_loading_dialog(opening, monkeypatch):
    from gui import lnt_batch
    monkeypatch.setattr(lnt_batch, 'cached_render_cache', lambda *a: str(opening.document))
    monkeypatch.setattr(lnt_open, '_BuildWorker', Mock(side_effect=AssertionError('Cached file rebuilt')))
    assert lnt_open.open_with_lnt(opening.app, str(opening.source))
    assert len(opening.dialogs) == 1
    dialog = opening.dialogs[0]
    assert dialog.title == 'Loading point cloud'
    assert dialog.values[0] == 0 and dialog.values[-1] == 100
    assert opening.views[0].started
    assert dialog.closed == 1


@pytest.mark.parametrize('error', ['cancelled', 'conversion failed'])
def test_build_failure_or_cancellation_closes_without_loading(opening, error):
    ui, dialog = build_ui(opening)
    ui.on_finished('', error)
    assert not opening.views
    assert opening.dialogs == [dialog]
    assert dialog.closed == 1
    assert opening.app._lnt_import_class_filter is None
    assert opening.errors.call_count == (error != 'cancelled')


def test_cancel_at_build_completion_does_not_start_loading(opening):
    ui, dialog = build_ui(opening)
    dialog.cancelled = True
    ui.on_finished(str(opening.document), '')
    assert not opening.views
    assert dialog.closed == 1
    opening.errors.assert_not_called()


@pytest.mark.parametrize('stage', ['Loading ', 'Opening '])
def test_cancel_during_loading_cleans_private_copy_and_keeps_source(opening, monkeypatch, stage):
    ui, dialog = build_ui(opening)
    original = dialog.setLabelText

    def cancel_loading(text):
        original(text)
        if text.startswith(stage):
            dialog.cancelled = True

    monkeypatch.setattr(dialog, 'setLabelText', cancel_loading)
    ui.on_finished(str(opening.document), '')
    assert not opening.views
    assert opening.dialogs == [dialog] and dialog.closed == 1
    assert not list((opening.document.parent / 'NakshaAI' / 'lnt_sessions').iterdir())
    assert opening.document.read_bytes() == b'scratch native document'
    opening.errors.assert_not_called()


def test_copy_failure_closes_existing_dialog(opening, monkeypatch):
    ui, dialog = build_ui(opening)
    monkeypatch.setattr(lnt_open, '_session_lnt_copy', Mock(side_effect=OSError('disk full')))
    ui.on_finished(str(opening.document), '')
    assert not opening.views
    assert opening.dialogs == [dialog] and dialog.closed == 1
    opening.errors.assert_called_once()


def test_native_start_failure_closes_existing_dialog_and_cleans_session(opening, monkeypatch):
    ui, dialog = build_ui(opening)
    monkeypatch.setattr(lnt_open.LntMainView, 'start', Mock(side_effect=RuntimeError('native start failed')))
    ui.on_finished(str(opening.document), '')
    assert opening.dialogs == [dialog] and dialog.closed == 1
    assert 100 not in dialog.values
    assert not list((opening.document.parent / 'NakshaAI' / 'lnt_sessions').iterdir())
    opening.errors.assert_called_once()


def test_build_progress_never_moves_back_between_build_stages(opening):
    ui, dialog = build_ui(opening)
    for percent in [10, 70, 100, 0, 50, 100]:
        ui.on_progress(percent, 'Build stage')
    assert dialog.values == sorted(dialog.values)
    assert dialog.value() == 85
