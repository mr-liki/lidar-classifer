"""Disk-backed, bounded-memory history for native LNT class conversions."""
from __future__ import annotations

import os
import tempfile
import weakref

import numpy as np

RECORD = np.dtype([('node', '<u4'), ('local', '<u4'), ('old', 'u1')])
BATCH_SIZE = 250_000


class ConversionCancelled(RuntimeError):
    pass


def _cleanup(handle, path):
    handle.close()
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


class ConversionJournal:
    """Keep exact original classes on disk, owned by one Undo/Redo entry."""

    def __init__(self, target):
        self.target = int(target)
        if not 0 <= self.target <= 255:
            raise ValueError('To-class outside uint8 range.')
        self.count = 0
        self._file = tempfile.NamedTemporaryFile(
            prefix='naksha-convert-', suffix='.undo', delete=False)
        self.path = self._file.name
        self._cleanup = weakref.finalize(self, _cleanup, self._file, self.path)

    def append(self, node, local, old):
        records = np.empty(len(old), dtype=RECORD)
        records['node'], records['local'], records['old'] = node, local, old
        records.tofile(self._file)
        self.count += len(records)

    def seal(self):
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self.validate()

    def validate(self):
        if not self._file.closed:
            raise RuntimeError('Conversion history is not complete.')
        if os.path.getsize(self.path) != self.count * RECORD.itemsize:
            raise RuntimeError('Conversion history is incomplete; no writes attempted.')

    def batches(self, limit=None):
        remaining = self.count if limit is None else min(self.count, limit)
        with open(self.path, 'rb') as handle:
            while remaining:
                size = min(BATCH_SIZE, remaining)
                records = np.fromfile(handle, dtype=RECORD, count=size)
                if len(records) != size:
                    raise RuntimeError('Conversion history became unreadable.')
                yield (np.ascontiguousarray(records['node']),
                       np.ascontiguousarray(records['local']),
                       np.ascontiguousarray(records['old']))
                remaining -= size

    def close(self):
        self._cleanup()

    def replay(self, native, forward=True, progress=None, cancelled=None, sync=None):
        """Apply or undo in batches; restore every attempted batch on failure."""
        self.validate()
        attempted = 0
        try:
            for node, local, old in self.batches():
                if cancelled is not None and cancelled():
                    raise ConversionCancelled('Convert cancelled.')
                target = np.full(len(old), self.target, dtype=np.uint8)
                values = target if forward else old
                # Include the current batch: a native failure may partially write it.
                attempted += len(old)
                if not native.set_classes(node, local, values):
                    raise RuntimeError('Native classification write failed: ' +
                                       str(getattr(native, 'last_error', '')))
                if sync is not None:
                    sync(node, local, values)
                if progress is not None:
                    progress(attempted, self.count)
            if cancelled is not None and cancelled():
                raise ConversionCancelled('Convert cancelled.')
        except Exception as error:
            rollback_error = None
            try:
                for node, local, old in self.batches(limit=attempted):
                    values = old if forward else np.full(len(old), self.target, dtype=np.uint8)
                    # A failure restoring one batch or its section cache must
                    # not prevent attempts to restore every remaining batch.
                    try:
                        if not native.set_classes(node, local, values):
                            raise RuntimeError(str(getattr(native, 'last_error', '')))
                        if sync is not None:
                            sync(node, local, values)
                    except Exception as restore_error:
                        rollback_error = restore_error
            except Exception as read_error:
                rollback_error = read_error
            if rollback_error is not None:
                # Preserve the only recovery record if storage/native writes fail.
                self._cleanup.detach()
                raise RuntimeError(
                    f'{error} Rollback failed: {rollback_error}. '
                    f'Recovery history retained at {self.path}') from error
            if isinstance(error, ConversionCancelled):
                raise ConversionCancelled('Convert cancelled; all changes rolled back.') from error
            raise RuntimeError(f'{error} All attempted changes were rolled back.') from error


def conversion_history(app, view, source, destination, redo=False):
    """Route one disk-backed history entry through the native writer only."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QProgressDialog
    from gui.lnt_ai_bridge import active_bridge, sync_store_rows

    entry = source[-1]
    if entry['lnt_path'] != view.lnt_path:
        app.statusBar().showMessage('This conversion belongs to another point cloud.', 4000)
        return True
    if active_bridge(app) is not None:
        app.statusBar().showMessage('Close AI classification before Undo/Redo.', 4000)
        return True
    journal = entry['lnt_conversion']
    app._native_byclass_busy = True
    progress = QProgressDialog('Replaying conversion...', 'Cancel', 0, 100, app)
    progress.setWindowTitle('Redo Convert' if redo else 'Undo Convert')
    progress.setWindowModality(Qt.ApplicationModal)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.show()
    try:
        def report(done, total):
            progress.setValue(int(100 * done / max(total, 1)))
            progress.setLabelText(f'{"Redo" if redo else "Undo"} Convert: {done:,}/{total:,}')
            QApplication.processEvents()
        journal.replay(view.native, forward=redo, progress=report,
                       cancelled=progress.wasCanceled,
                       sync=lambda n, l, c: sync_store_rows(app, view, n, l, c))
        destination.append(source.pop())
        app.classification_revision = int(getattr(app, 'classification_revision', 0)) + 1
        view.native.frame(0.0)
        from gui.point_count_widget import refresh_point_statistics
        refresh_point_statistics(app)
        app.statusBar().showMessage(
            f'Convert {"redone" if redo else "undone"} ({journal.count:,} points)', 4000)
    except Exception as error:
        app.statusBar().showMessage(str(error), 8000)
    finally:
        progress.close()
        app._native_byclass_busy = False
    return True
