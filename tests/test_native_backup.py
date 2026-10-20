"""The classified point cloud is backed up to a SEPARATE file, never over the original.

Closing the application (or losing power) used to leave the loaded point cloud
holding classifications that were never asked to be saved there. The native
exporter now writes the classified cloud to its own file - 123_backup.laz next
to the file that was loaded - and the loaded document is left alone.

These tests cover the native (Vulkan/.lnt) path only: no VTK is involved.
"""
import os
import shutil
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings

from gui import native_backup, native_save

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_LNT = os.path.join(ROOT, "native", "lnt", "test.lnt")
SETTINGS_KEYS = ("backup_enabled", "backup_use_custom_path", "backup_custom_path",
                 "backup_interval_minutes", "backup_naming_format",
                 "backup_on_exit", "backup_keep_original")


class _FakeNative:
    """The native view's engine handle."""

    def __init__(self, writes=0):
        self.class_writes = writes


class _FakeView:
    def __init__(self, lnt_path, writes=0):
        self.lnt_path = lnt_path
        self.native = _FakeNative(writes)
        self._save_mapping_cache = None


class _FakeApp:
    """The bits of the main window the backup path uses."""

    def __init__(self, document, source=None, writes=0):
        self._lnt_main_view = _FakeView(document, writes)
        self._lnt_source_path = source
        self._lnt_source_paths = [source] if source else []
        self.loaded_file = source or document
        self.data = None
        self._lnt_section_store = None
        self._lnt_ai_bridge = None
        self._active_backup_worker = None
        self.messages = []

    def statusBar(self):
        return self

    def showMessage(self, text, timeout=0):
        self.messages.append(text)

    def _backup_progress_dialog(self, destination):
        return None                      # no window in a test

    def _on_backup_progress(self, percent, text):
        pass


@pytest.fixture(autouse=True)
def _settings_round_trip():
    """The Auto-Backup Settings live in the user's QSettings: put them back."""
    store = QSettings("NakshaAI", "LidarApp")
    previous = {key: store.value(key) for key in SETTINGS_KEYS}
    try:
        yield
    finally:
        for key, value in previous.items():
            store.setValue(key, value)
        store.sync()


def _write_settings(**values):
    store = QSettings("NakshaAI", "LidarApp")
    for key, value in values.items():
        store.setValue(key, value)
    store.sync()


@pytest.fixture
def source_and_document():
    """A LAS source and the native document built from it by the app."""
    import laspy
    import numpy as np
    from gui import lnt_native as nat
    work = tempfile.mkdtemp(prefix="naksha_backup_src_")
    try:
        source = os.path.join(work, "123.las")
        header = laspy.LasHeader(point_format=7, version="1.4")
        header.offsets = np.array([680000.0, 5000000.0, 0.0])
        header.scales = np.array([0.001, 0.001, 0.001])
        las = laspy.LasData(header)
        count = 2000
        rng = np.random.default_rng(3)
        las.x = 680000.0 + rng.random(count) * 200.0
        las.y = 5000000.0 + rng.random(count) * 200.0
        las.z = 100.0 + rng.random(count) * 20.0
        las.classification = np.zeros(count, dtype=np.uint8)
        las.write(source)
        document = os.path.join(work, "123.lnt")
        assert nat.convert_las(source, document)
        yield source, document
    finally:
        shutil.rmtree(work, ignore_errors=True)


@pytest.fixture
def document():
    if not os.path.isfile(SAMPLE_LNT):
        pytest.skip("no sample .lnt")
    work = tempfile.mkdtemp(prefix="naksha_backup_test_")
    path = os.path.join(work, "123.lnt")
    shutil.copyfile(SAMPLE_LNT, path)
    try:
        yield path
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# where the backup goes
# --------------------------------------------------------------------------- #
def test_backup_lands_next_to_the_loaded_file(document):
    _write_settings(backup_use_custom_path=False, backup_naming_format=0)
    app = _FakeApp(document)
    path = native_backup.backup_path(app)
    assert os.path.basename(path) == "123_backup.laz"
    assert os.path.dirname(path) == os.path.dirname(document)


def test_backup_uses_the_custom_folder(document):
    custom = os.path.join(os.path.dirname(document), "backups")
    _write_settings(backup_use_custom_path=True, backup_custom_path=custom,
                    backup_naming_format=0)
    app = _FakeApp(document)
    path = native_backup.backup_path(app)
    assert os.path.dirname(path) == custom
    assert os.path.basename(path) == "123_backup.laz"


def test_backup_naming_formats(document):
    app = _FakeApp(document)
    _write_settings(backup_use_custom_path=False, backup_naming_format=0)
    assert os.path.basename(native_backup.backup_path(app)) == "123_backup.laz"
    _write_settings(backup_naming_format=2)
    assert os.path.basename(native_backup.backup_path(app)) == "123_autosave.laz"
    _write_settings(backup_naming_format=1)
    stamped = os.path.basename(native_backup.backup_path(app))
    assert stamped.startswith("123_") and stamped.endswith(".laz")
    assert stamped != "123_backup.laz"


def test_backup_never_targets_the_loaded_file(document):
    """The naming can never resolve to the file that was loaded."""
    _write_settings(backup_use_custom_path=False, backup_naming_format=0)
    app = _FakeApp(document)
    path = native_backup.backup_path(app)
    assert os.path.normcase(path) != os.path.normcase(document)


# --------------------------------------------------------------------------- #
# the export
# --------------------------------------------------------------------------- #
def test_export_writes_a_new_file_and_leaves_the_document_alone(document):
    before = os.stat(document)
    app = _FakeApp(document, writes=1)
    destination = os.path.join(os.path.dirname(document), "123_backup.laz")
    result = native_backup.export_backup(app, destination, flush=False)
    assert os.path.isfile(destination)
    assert result["count"] > 0
    after = os.stat(document)
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    # A backup is a LAS/LAZ file, not a native document.
    assert not os.path.isfile(os.path.join(os.path.dirname(document), "123_backup.lnt"))


def test_export_refuses_to_overwrite_the_loaded_file(document):
    app = _FakeApp(document, writes=1)
    with pytest.raises(ValueError):
        native_backup.export_backup(app, document, flush=False)


def test_export_keeps_the_class_map_beside_the_backup(document):
    """The user class numbers must survive with the backup."""
    app = _FakeApp(document, writes=1)
    destination = os.path.join(os.path.dirname(document), "123_backup.laz")
    native_backup.export_backup(app, destination, flush=False, mapping={300: 5})
    assert os.path.isfile(destination + ".classmap.json")


def test_write_pair_without_a_companion_publishes_only_the_backup(document):
    destination = os.path.join(os.path.dirname(document), "123_backup.laz")
    native_save.write_pair(None, document, destination, mapping={}, cache={},
                           companion=False, progress=lambda n, text: None)
    assert os.path.isfile(destination)
    assert not os.path.isfile(os.path.join(os.path.dirname(document), "123_backup.lnt"))


# --------------------------------------------------------------------------- #
# the point mapping is kept between sessions
# --------------------------------------------------------------------------- #
def test_the_point_mapping_is_kept_beside_the_document(document):
    """Rebuilding it costs a full pass over the source (seconds for big files).

    The export needs it every time - including the backup written while the
    application closes - so it is kept next to the document and reused by the
    next session.
    """
    destination = os.path.join(os.path.dirname(document), "123_backup.laz")
    native_save.write_pair(None, document, destination, mapping={}, cache={},
                           companion=False, progress=lambda n, text: None)
    # A document without a LAS/LAZ source needs no mapping; the cache files are
    # only written when there is a source to map.
    assert not os.path.isfile(document + ".sourceorder.npy")


def test_the_point_mapping_cache_is_reused_by_the_next_session(source_and_document):
    source, document = source_and_document
    destination = os.path.join(os.path.dirname(document), "123_backup.laz")
    first = native_save.write_pair(source, document, destination, mapping={}, cache={},
                                   companion=False, progress=lambda n, text: None)
    assert os.path.isfile(document + ".sourceorder.npy")
    assert os.path.isfile(document + ".sourceorder.json")
    # A new session starts with an empty cache: the mapping comes from disk.
    second = native_save.write_pair(source, document, destination, mapping={}, cache={},
                                    companion=False, progress=lambda n, text: None)
    assert second["count"] == first["count"]


# --------------------------------------------------------------------------- #
# the exit path
# --------------------------------------------------------------------------- #
def test_exit_backup_writes_the_classified_copy(document):
    _write_settings(backup_on_exit=True, backup_keep_original=True,
                    backup_use_custom_path=False, backup_naming_format=0)
    app = _FakeApp(document, writes=1)
    from gui.app_window import NakshaApp
    NakshaApp._backup_native_on_exit(app)
    backup = os.path.join(os.path.dirname(document), "123_backup.laz")
    assert os.path.isfile(backup), "the exit backup was not written"
    assert app.messages, "the user was not told where the backup went"


def test_exit_backup_is_skipped_without_classifications(document):
    _write_settings(backup_on_exit=True)
    app = _FakeApp(document, writes=0)
    from gui.app_window import NakshaApp
    NakshaApp._backup_native_on_exit(app)
    assert not os.path.isfile(os.path.join(os.path.dirname(document), "123_backup.laz"))


def test_exit_backup_can_be_switched_off(document):
    _write_settings(backup_on_exit=False)
    app = _FakeApp(document, writes=1)
    from gui.app_window import NakshaApp
    NakshaApp._backup_native_on_exit(app)
    assert not os.path.isfile(os.path.join(os.path.dirname(document), "123_backup.laz"))
