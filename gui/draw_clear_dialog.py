"""VTK-parity Draw > Clear, for native Vulkan and the original VTK digitizer.

Clear removes *unused* user drawings as one undoable action, then shows the
original-style 'Clear Classified Fences' checkbox list for successful-classify
fences. A dismiss/Keep Selected never deletes classified fences. No point
classification is changed by deleting a drawing.
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from gui.draw_clear_selection import named_drawings, partition_drawings_for_clear


def _drawing_summary(drawing, native=False):
    """Render metadata like the old VTK popup without altering source geometry."""
    kind = str(drawing.get('type', '')).lower()
    if native:
        try:
            from gui.native_draw import NativeDraw
            coords = NativeDraw._native_coords(drawing)
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            coords = drawing.get('coords')
            if coords is None:
                coords = ()
    else:
        coords = drawing.get('coords')
        if coords is None:
            coords = ()
    try:
        xy = [(float(p[0]), float(p[1])) for p in coords]
        xy = [(x, y) for x, y in xy if math.isfinite(x) and math.isfinite(y)]
    except (TypeError, ValueError, IndexError):
        xy = []
    if not xy:
        return '0 pts | dimensions unavailable'
    w = max(p[0] for p in xy) - min(p[0] for p in xy)
    h = max(p[1] for p in xy) - min(p[1] for p in xy)
    # A closed polygon repeats its start vertex; VTK's log counts the original
    # stored vertices rather than automatically adding the closing one.
    raw = drawing.get('coords')
    try:
        count = len(raw)
    except (TypeError, ValueError):
        count = len(xy)
    if native and len(xy) >= 3:
        count = len(xy) - int(xy[0] == xy[-1])
    if kind == 'circle' and isinstance(drawing.get('radius'), (int, float)):
        return f'{count} pts | radius {float(drawing["radius"]):.2f}m'
    return f'{count} pts | {w:.1f}x{h:.1f}m'


class DrawClearDialog(QDialog):
    """Matches the original Clear Classified Fences UI and its button semantics.

    Select All = check all, Clear All = uncheck all (per VTK source),
    Keep Selected = close without deleting, Delete Checked = delete checked.
    """
    def __init__(self, entries, *, native=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Clear Classified Fences')
        self.resize(500, 420)
        self.setMinimumSize(460, 340)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self._selected_ids = set()
        self._checks = []
        self.setStyleSheet('''
            QDialog { background-color: #181818; color: #eeeeee; }
            QLabel { color: #e9e9e9; background: transparent; }
            QScrollArea { border: 0; background: #181818; }
            QScrollArea > QWidget > QWidget { background: #181818; }
            QFrame#classifiedFenceRow { background: #2d2d2d; border-radius: 7px; }
            QFrame#classifiedFenceRow:hover { background: #363636; }
            QCheckBox { color: #eeeeee; background: transparent; }
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 3px;
                                   border: 1px solid #454545; background-color: #242424; }
            QCheckBox::indicator:checked { background-color: #e94141;
                                           border-color: #e94141; }
            QPushButton { border: 1px solid #343434; border-radius: 7px;
                          color: #ffffff; background: #2a2a2a; padding: 10px 12px; }
            QPushButton:hover { background: #414141; }
            QPushButton#dangerBtn { background: #d92c32; border-color: #d92c32; }
            QPushButton#dangerBtn:hover { background: #ea4044; }
            QPushButton#primaryBtn { background: #303942; border-color: #52616c; }
        ''')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(10)
        title = QLabel('Select classified fences to DELETE', self)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('color: #f04444; font-weight: bold; font-size: 13px; padding: 6px;')
        layout.addWidget(title)
        help_text = QLabel('Unchecked fences will be kept. Checked fences will be removed.', self)
        help_text.setAlignment(Qt.AlignCenter)
        help_text.setStyleSheet('color: #cccccc; font-size: 10px; padding-bottom: 8px;')
        layout.addWidget(help_text)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll_contents = QWidget(scroll)
        rows = QVBoxLayout(scroll_contents)
        rows.setContentsMargins(0, 2, 0, 2)
        rows.setSpacing(5)
        for number, (drawing, tool, _label) in enumerate(entries, 1):
            item = QFrame(scroll_contents)
            item.setObjectName('classifiedFenceRow')
            hbox = QHBoxLayout(item)
            hbox.setContentsMargins(14, 8, 12, 8)
            hbox.setSpacing(12)
            checkbox = QCheckBox(item)
            checkbox.setAccessibleName(f'Delete classified fence {number}, {tool}')
            checkbox.setChecked(False)
            hbox.addWidget(checkbox)
            accent = QFrame(item)
            accent.setFixedSize(6, 31)
            accent.setStyleSheet('background: #219de6; border-radius: 3px;')
            hbox.addWidget(accent)
            label = QLabel(item)
            layer = drawing.get('layer') or drawing.get('level') or '0'
            label.setText(f'{layer}  #{number}: {tool}\n{_drawing_summary(drawing, native=native)}')
            label.setStyleSheet('color: #ededed; font-size: 10px;')
            hbox.addWidget(label, 1)
            self._checks.append((checkbox, id(drawing)))
            rows.addWidget(item)
        rows.addStretch(1)
        scroll.setWidget(scroll_contents)
        layout.addWidget(scroll, 1)

        actions = QHBoxLayout()
        actions.setSpacing(7)
        select_all = QPushButton('Select All', self)
        select_all.setObjectName('dangerBtn')
        select_all.clicked.connect(lambda: self._set_all(True))
        actions.addWidget(select_all)
        clear_all = QPushButton('Clear All', self)
        clear_all.setToolTip('Uncheck all classified fences (VTK behavior)')
        clear_all.clicked.connect(lambda: self._set_all(False))
        actions.addWidget(clear_all)
        actions.addStretch(1)
        keep = QPushButton('Keep Selected', self)
        keep.setObjectName('primaryBtn')
        keep.setToolTip('Close without deleting any classified fences')
        keep.clicked.connect(self.reject)
        actions.addWidget(keep)
        delete = QPushButton('Delete Checked', self)
        delete.setObjectName('dangerBtn')
        delete.setToolTip('Remove only checked classified fence drawings')
        delete.clicked.connect(self._accept_selected)
        actions.addWidget(delete)
        layout.addLayout(actions)

    def _set_all(self, checked):
        for checkbox, _ in self._checks:
            checkbox.setChecked(bool(checked))

    def _accept_selected(self):
        self._selected_ids = {identifier for check, identifier in self._checks
                              if check.isChecked()}
        if not self._selected_ids:
            # The old VTK dialog simply keeps all fences when none are checked.
            self.reject()
            return
        self.accept()

    @property
    def selected_ids(self):
        return set(self._selected_ids)


def show_draw_clear_dialog(app):
    """Only Draw > Clear uses this two-phase delete; classification Clear does not."""
    from gui.native_draw import get_native_draw
    # Existing Vulkan Draw may have been saved while the viewport was replaced.
    draw = get_native_draw(app, create=True, allow_snt=True)
    digitizer = None if draw is not None else getattr(app, 'digitizer', None)
    if draw is None and digitizer is None:
        return 0
    if getattr(app, '_draw_clear_ambiguous_classified_fence', False):
        QMessageBox.warning(
            app, 'Clear Drawings Blocked',
            'A previously classified fence cannot be identified uniquely '
            '(for example, a saved macro used one of two identical shapes). '
            'Automatic Clear is blocked to protect classified drawings. '
            'Resolve the fence identity in the source workflow before clearing.')
        return 0
    if draw is not None:
        drawings, vtk, in_scope = draw.drawings, False, None
        delete = draw.delete_drawings
    else:
        drawings, vtk = digitizer.drawings, True
        scope = getattr(digitizer, 'drawing_in_edit_scope', None)
        in_scope = scope if callable(scope) else None
        delete = digitizer.delete_drawings_by_identity

    unused, classified = partition_drawings_for_clear(
        drawings, vtk=vtk, in_scope=in_scope)
    if not unused and not classified:
        QMessageBox.information(app, 'Clear Drawings',
                                'No user-created drawings are available to clear.')
        return 0

    # Original VTK sequence: remove unused drawings first; protect every used
    # classification fence until the user explicitly checks it in the popup.
    removed = delete({id(item) for item in unused}) if unused else 0
    # Recheck live objects after deleting unused drawings (undo may be active).
    entries = [(drawing, tool, name) for drawing, tool, name in named_drawings(
        drawings, vtk=vtk, in_scope=in_scope)
        if drawing.get('classified_fence', False)]
    if not entries:
        try:
            app.statusBar().showMessage(
                f'Draw Clear: {removed} unused drawing(s) cleared; Ctrl+Z to undo', 5000)
        except (AttributeError, RuntimeError):
            pass
        return removed

    dialog = DrawClearDialog(entries, native=draw is not None, parent=app)
    if dialog.exec() == QDialog.Accepted:
        removed += delete(dialog.selected_ids)
    try:
        app.statusBar().showMessage(
            f'Draw Clear: {removed} drawing(s) removed; classified fences not checked are kept',
            6000)
    except (AttributeError, RuntimeError):
        pass
    return removed
