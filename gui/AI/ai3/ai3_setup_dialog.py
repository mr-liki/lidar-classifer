"""Qt front end for creating / verifying the AI model 3.0 runtime on this computer.

``ensure_runtime(parent)`` is what the AI dialog calls before a run: it returns True when the
runtime is ready, offering to create or repair it (with the size and internet requirement stated
up front, and a cancellable progress window) when it is not.
"""
from __future__ import annotations

import threading
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QApplication, QDialog, QLabel, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QVBoxLayout)

from . import ai3_runtime as rt
from . import setup_ai3_runtime as setup


class _WorkThread(QThread):
    progress = Signal(int, str)
    done = Signal(bool, str)

    def __init__(self, mode: str, repair: bool = False, variant: str = "auto"):
        super().__init__()
        self.mode = mode                    # "setup" | "verify"
        self.repair = repair
        self.variant = variant
        self.cancel_event = threading.Event()

    def run(self) -> None:
        try:
            if self.mode == "verify":
                self.progress.emit(10, "Checking the AI model 3.0 runtime (PDAL, GPU, model files)...")
                status = rt.verify_runtime()
                self.done.emit(bool(status.ok), status.message)
            else:
                status = setup.run_setup(variant=self.variant, repair=self.repair,
                                         progress=self.progress.emit, cancel=self.cancel_event)
                self.done.emit(bool(status.ok), status.message)
        except setup.SetupCancelled as exc:
            self.done.emit(False, str(exc))
        except setup.SetupError as exc:
            self.done.emit(False, str(exc))
        except Exception as exc:                                  # never let the thread die silently
            self.done.emit(False, f"{type(exc).__name__}: {exc}")


class Ai3SetupDialog(QDialog):
    """Progress window for a setup or verification run. exec() returns Accepted on success."""

    def __init__(self, mode: str = "setup", repair: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI model 3.0 - runtime setup" if mode == "setup"
                            else "AI model 3.0 - checking runtime")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.message = ""

        layout = QVBoxLayout(self)
        self._headline = QLabel("Setting up AI model 3.0 on this computer. This is done once; "
                                "it downloads about 3 GB and installs about 8 GB." if mode == "setup"
                                else "Checking the AI model 3.0 runtime...")
        self._headline.setWordWrap(True)
        layout.addWidget(self._headline)
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        layout.addWidget(self._bar)
        self._status = QLabel("Starting...")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(300)
        self._log.setMinimumHeight(140)
        layout.addWidget(self._log)
        self._cancel = QPushButton("Cancel")
        self._cancel.clicked.connect(self._on_cancel)
        layout.addWidget(self._cancel)

        self._thread = _WorkThread(mode, repair=repair)
        self._thread.progress.connect(self._on_progress)
        self._thread.done.connect(self._on_done)

    def showEvent(self, event):                     # start when first shown
        super().showEvent(event)
        if not self._thread.isRunning() and not self._thread.isFinished():
            self._thread.start()

    def _on_progress(self, percent: int, text: str) -> None:
        self._bar.setValue(max(self._bar.value(), int(percent)))
        self._status.setText(text)
        self._log.appendPlainText(f"{int(percent):3d}%  {text}")

    def _on_done(self, ok: bool, message: str) -> None:
        self.message = message
        self._thread.wait(3000)
        if ok:
            self._bar.setValue(100)
            self.accept()
        else:
            self.reject()

    def _on_cancel(self) -> None:
        self._cancel.setEnabled(False)
        self._status.setText("Cancelling...")
        self._thread.cancel_event.set()

    def reject(self) -> None:
        if self._thread.isRunning():                # window close = cancel, then wait for it
            self._on_cancel()
            self._thread.wait(15000)
        super().reject()


def ensure_runtime(parent=None) -> bool:
    """True when the AI model 3.0 runtime is ready to use, setting it up (with consent) if needed."""
    status = rt.runtime_status()
    if status.ok:
        return True
    if status.engine_ok is False:
        QMessageBox.critical(parent, "AI model 3.0", status.message)
        return False

    if status.state == "unverified":
        dialog = Ai3SetupDialog(mode="verify", parent=parent)
        if dialog.exec() == QDialog.Accepted:
            return True
        status = rt.runtime_status()
        if status.state not in ("broken", "missing", "incomplete"):
            status.state = "broken"
        verify_message = dialog.message
    else:
        verify_message = ""

    repair = status.state == "broken"
    try:
        info = setup.plan()
    except Exception:
        info = {}
    gpu_line = ("An NVIDIA GPU was found, so the GPU build will be installed."
                if info.get("variant") == "cuda"
                else "No usable NVIDIA GPU was found, so the CPU build will be installed "
                     "(classification will be slow).")
    text = (
        ("The AI model 3.0 runtime is damaged and must be rebuilt.\n\n" + verify_message + "\n\n"
         if repair else "AI model 3.0 needs a one-time setup on this computer.\n\n")
        + f"{gpu_line}\n\n"
        "It downloads about 3 GB, installs about 8 GB under\n"
        f"{rt.runtime_root()}\n"
        "and needs an internet connection while it runs (typically 10-30 minutes).\n\n"
        "Set it up now?"
    )
    if QMessageBox.question(parent, "AI model 3.0 - setup", text,
                            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) != QMessageBox.Yes:
        return False

    dialog = Ai3SetupDialog(mode="setup", repair=repair, parent=parent)
    result = dialog.exec()
    if result == QDialog.Accepted:
        QMessageBox.information(parent, "AI model 3.0", "AI model 3.0 is ready to use.")
        return True
    QMessageBox.warning(parent, "AI model 3.0 - setup did not finish",
                        (dialog.message or "Setup was cancelled.")
                        + f"\n\nDetails: {rt.runtime_root() / 'logs' / 'setup.log'}")
    return False
