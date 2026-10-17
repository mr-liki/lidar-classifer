"""Native Vulkan Block and Cell creation for the LNT plan viewport.

This module intentionally contains no VTK imports or compatibility fallback.
Selection, hit testing, previews and generated geometry use the native canvas
and the shared Qt overlay above the Vulkan swapchain.
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractSpinBox, QColorDialog, QComboBox, QDialog, QDoubleSpinBox,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QVBoxLayout,
)

from gui.native_draw import get_native_draw
from gui.theme_manager import ThemeColors, ThemeManager, get_dialog_stylesheet


def _xyz(point):
    return float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0


def _distance_to_segment(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    d2 = dx * dx + dy * dy
    t = max(0.0, min(1.0, ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / d2)) if d2 else 0.0
    q = a[0] + t*dx, a[1] + t*dy
    return math.hypot(p[0]-q[0], p[1]-q[1])


def _split_polyline(coords, distance):
    """Split a polyline into chains no longer than distance."""
    pts = [_xyz(p) for p in coords]
    if len(pts) < 2 or distance <= 0:
        return [pts]
    result, chain, remaining = [], [pts[0]], float(distance)
    for start, end in zip(pts, pts[1:]):
        a = start
        seg = math.hypot(end[0]-a[0], end[1]-a[1])
        while seg > remaining and seg > 1e-12:
            t = remaining / seg
            cut = tuple(a[i] + (end[i]-a[i])*t for i in range(3))
            chain.append(cut); result.append(chain)
            chain, a = [cut], cut
            seg = math.hypot(end[0]-a[0], end[1]-a[1]); remaining = float(distance)
        if seg > 1e-12:
            chain.append(end); remaining -= seg
        elif chain[-1] != end:
            chain.append(end)
        if remaining <= 1e-9:
            result.append(chain); chain, remaining = [end], float(distance)
    if len(chain) > 1:
        result.append(chain)
    return result


def _corridor(chain, left_width, right_width):
    """Return a stable mitered corridor ring around an open 2D chain."""
    if len(chain) < 2:
        return []
    normals = []
    for a, b in zip(chain, chain[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        normals.append((-dy/length, dx/length) if length else (0.0, 0.0))
    vertex_normals = [normals[0]]
    for first, second in zip(normals, normals[1:]):
        nx, ny = first[0]+second[0], first[1]+second[1]
        length = math.hypot(nx, ny)
        vertex_normals.append((nx/length, ny/length) if length else second)
    vertex_normals.append(normals[-1])
    left = [(p[0]+n[0]*left_width, p[1]+n[1]*left_width) for p, n in zip(chain, vertex_normals)]
    right = [(p[0]-n[0]*right_width, p[1]-n[1]*right_width) for p, n in zip(chain, vertex_normals)]
    return left + list(reversed(right)) + [left[0]]


class NativeCreation:
    def __init__(self, view):
        self.view = view
        self.app = view.app
        self.draw = get_native_draw(self.app, create=True)
        self.canvas = self.draw.canvas
        self.mode = None
        self.selected = []
        self.hovered = None
        self._candidate_lines = []

    @property
    def active(self):
        return self.mode in ('block', 'cell')

    def arm(self, mode):
        self.mode = mode
        self._candidate_lines = self._collect_candidates()
        self.draw.cancel()
        measurement = getattr(self.view, 'measurement', None)
        if measurement is not None:
            measurement.cancel()
        classifier = getattr(self.view, 'classifier', None)
        if classifier is not None:
            classifier.cancel()
        self.canvas.setCursor(Qt.PointingHandCursor)
        self.app.statusBar().showMessage(
            'Select CL: click a native line; Shift+click adds more. Middle-drag pans.', 0)
        self.paint()

    def disarm(self):
        self.mode = None; self.hovered = None
        self.canvas.setCursor(Qt.ArrowCursor)
        self.paint()

    def _collect_candidates(self):
        candidates = [d for d in self.draw.drawings
                      if d.get('type') in ('line', 'smartline', 'polyline', 'centerline')
                      and d.get('source') not in ('block_creation', 'cell_creation')]
        for attachment in getattr(self.app, 'snt_attachments', []) or []:
            for entity in (attachment.get('parsed') or {}).get('entities', []) or []:
                kind = str(entity.get('type', '')).upper()
                coords = entity.get('vertices') or entity.get('coords') or entity.get('points') or []
                if kind in ('LINE', 'POLYLINE', 'LWPOLYLINE') and len(coords) >= 2 and not entity.get('closed'):
                    candidates.append({'type': 'polyline', 'coords': [_xyz(p) for p in coords],
                                       'source': 'snt_cl', 'snt_entity_id': entity.get('_snt_entity_id'),
                                       'snt_filename': attachment.get('filename')})
        return candidates

    def candidates(self):
        return self._candidate_lines

    def hit(self, pos):
        p = self.canvas.section_at(pos.x(), pos.y())
        best = None
        for drawing in self.candidates():
            coords = drawing.get('coords', [])
            for a, b in zip(coords, coords[1:]):
                distance = _distance_to_segment(p, a, b)
                if best is None or distance < best[0]:
                    best = distance, drawing
        tolerance = max(self.canvas.world_per_pixel()*14.0, 1e-9)
        return best[1] if best and best[0] <= tolerance else None

    def press(self, pos, button, modifiers=Qt.NoModifier):
        if not self.active:
            return False
        if button == Qt.RightButton:
            self.disarm(); return True
        if button != Qt.LeftButton:
            return False
        hit = self.hit(pos)
        if not (modifiers & Qt.ShiftModifier):
            self.selected.clear()
        if hit is not None and not any(item is hit for item in self.selected):
            self.selected.append(hit)
        self.paint(); self._notify_dialog(); return True

    def move(self, pos):
        if not self.active:
            return False
        hit = self.hit(pos)
        if hit is not self.hovered:
            self.hovered = hit; self.paint()
        return True

    def key(self, key):
        if key == Qt.Key_Escape:
            self.disarm(); return True
        return False

    def paint(self):
        shapes = []
        for drawing in self.selected:
            shapes.append(('drawing', drawing.get('coords', []),
                           {'type': 'polyline', 'selected': True, 'source': 'creation_selection'}))
        if self.hovered is not None and not any(x is self.hovered for x in self.selected):
            shapes.append(('drawing_preview', self.hovered.get('coords', []), {'type': 'polyline'}))
        self.draw.aux_shapes = shapes
        self.draw._paint()

    def clear_selection(self):
        self.selected.clear(); self.hovered = None; self.paint(); self._notify_dialog()

    def _notify_dialog(self):
        dialog = getattr(self.app, f'native_{self.mode}_creation_dialog', None)
        if dialog is not None:
            dialog.update_selection()

    def create_cells(self, diameter_mm, color, width):
        radius = float(diameter_mm) / 2000.0
        unique = []
        for line in self.selected:
            for p in line.get('coords', []):
                p = _xyz(p)
                if not any(math.hypot(p[0]-q[0], p[1]-q[1]) <= .01 for q in unique):
                    unique.append(p)
        for center in unique:
            ring = [(center[0]+radius*math.cos(i*2*math.pi/72),
                     center[1]+radius*math.sin(i*2*math.pi/72)) for i in range(72)]
            ring.append(ring[0])
            self.draw.drawings.append({'type': 'polyline', 'coords': ring, 'source': 'cell_creation',
                                       'center': center, 'radius': radius, 'color': color, 'width': width})
        self.draw.redo.clear(); self.draw._paint()
        return len(unique)

    def create_blocks(self, left_width, right_width, distance):
        created = 0
        for line in self.selected:
            for chain in _split_polyline(line.get('coords', []), distance):
                ring = _corridor(chain, left_width, right_width)
                if ring:
                    self.draw.drawings.append({'type': 'polyline', 'coords': ring,
                                               'source': 'block_creation', 'centerline': chain,
                                               'tolerance_above': left_width,
                                               'tolerance_below': right_width})
                    created += 1
        self.draw.redo.clear(); self.draw._paint()
        return created


def get_native_creation(app, create=False):
    view = getattr(app, '_lnt_main_view', None)
    if view is None or getattr(view, 'native', None) is None or not getattr(view, 'mode_2d', False):
        return None
    controller = getattr(view, 'creation', None)
    if controller is None and create:
        controller = view.creation = NativeCreation(view)
    return controller


class _BaseDialog(QDialog):
    mode = ''
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.controller = get_native_creation(app, create=True)
        self.setWindowTitle(f'{self.mode.title()} Creation Settings')
        self.setProperty('renderBackend', 'native-vulkan')
        self.setModal(False); self.setFixedWidth(410)
        self.setStyleSheet(get_dialog_stylesheet())
        self.root = QVBoxLayout(self)
        group = QGroupBox('CL'); row = QHBoxLayout(group)
        self.select_btn = QPushButton('Select CL'); self.select_btn.setCheckable(True)
        self.select_btn.clicked.connect(self._select)
        self.selection_label = QLabel('No CL selected')
        row.addWidget(self.select_btn); row.addWidget(self.selection_label, 1)
        self.root.addWidget(group)

    def _select(self, checked):
        if checked: self.controller.arm(self.mode)
        else: self.controller.disarm()

    def update_selection(self):
        count = len(self.controller.selected)
        self.selection_label.setText(f'{count} CL selected' if count else 'No CL selected')

    def _buttons(self, create_callback):
        row = QHBoxLayout(); clear = QPushButton('Clear'); create = QPushButton('Create')
        clear.clicked.connect(self._clear); create.clicked.connect(create_callback)
        row.addWidget(clear); row.addStretch(); row.addWidget(create); self.root.addLayout(row)

    def _clear(self):
        self.controller.clear_selection(); self.reset()

    def reset(self): pass

    def closeEvent(self, event):
        self.controller.disarm(); self.select_btn.setChecked(False); super().closeEvent(event)


class NativeBlockDialog(_BaseDialog):
    mode = 'block'
    def __init__(self, app):
        super().__init__(app)
        group = QGroupBox('Block Settings'); form = QFormLayout(group)
        self.mode_box = QComboBox(); self.mode_box.addItems(['Symmetric (total width)', 'Asymmetric (above / below)'])
        self.total = self._spin(60.0, ' m'); self.above = self._spin(1.0, ' m'); self.below = self._spin(1.0, ' m')
        self.distance = self._spin(250.0, ' m')
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        form.addRow('Mode:', self.mode_box); form.addRow('Total width:', self.total)
        form.addRow('Tolerance above:', self.above); form.addRow('Tolerance below:', self.below)
        form.addRow('Distance:', self.distance); self.root.addWidget(group); self._buttons(self.create)
        self._mode_changed(0); self.adjustSize(); ThemeManager.apply_native_window_theme(self)

    @staticmethod
    def _spin(value, suffix):
        spin = QDoubleSpinBox(); spin.setRange(0., 999999.); spin.setDecimals(4)
        spin.setValue(value); spin.setSuffix(suffix); spin.setButtonSymbols(QAbstractSpinBox.NoButtons); return spin

    def _mode_changed(self, index):
        symmetric = index == 0; self.total.setEnabled(symmetric)
        self.above.setEnabled(not symmetric); self.below.setEnabled(not symmetric)

    def create(self):
        if not self.controller.selected:
            self.app.statusBar().showMessage('Select at least one CL first', 3000); return
        if self.mode_box.currentIndex() == 0:
            above = below = self.total.value()/2.0
        else: above, below = self.above.value(), self.below.value()
        count = self.controller.create_blocks(above, below, self.distance.value())
        self.app.statusBar().showMessage(f'Created {count} native Vulkan block(s)', 4000)

    def reset(self):
        self.mode_box.setCurrentIndex(0); self.total.setValue(60.); self.above.setValue(1.)
        self.below.setValue(1.); self.distance.setValue(250.)


class NativeCellDialog(_BaseDialog):
    mode = 'cell'
    def __init__(self, app):
        super().__init__(app)
        group = QGroupBox('Cell Settings'); form = QFormLayout(group)
        self.diameter = QDoubleSpinBox(); self.diameter.setRange(0., 999999.); self.diameter.setDecimals(4)
        self.diameter.setValue(3000.); self.diameter.setSuffix(' mm'); self.diameter.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.color = QColor('#2979ff'); self.color_btn = QPushButton(self.color.name().upper())
        self.color_btn.clicked.connect(self.pick_color)
        self.width = QSpinBox(); self.width.setRange(1, 100); self.width.setValue(2); self.width.setSuffix(' px')
        self.width.setButtonSymbols(QAbstractSpinBox.NoButtons)
        form.addRow('Total Diameter:', self.diameter); form.addRow('Color:', self.color_btn)
        form.addRow('Thickness:', self.width); self.root.addWidget(group); self._buttons(self.create)
        self._color_style(); self.adjustSize(); ThemeManager.apply_native_window_theme(self)

    def _color_style(self):
        self.color_btn.setText(self.color.name().upper())
        self.color_btn.setStyleSheet(f'background:{self.color.name()}; color:white;')

    def pick_color(self):
        chosen = QColorDialog.getColor(self.color, self, 'Select Cell Color')
        if chosen.isValid(): self.color = chosen; self._color_style()

    def create(self):
        if not self.controller.selected:
            self.app.statusBar().showMessage('Select at least one CL first', 3000); return
        count = self.controller.create_cells(self.diameter.value(), self.color.name(), self.width.value())
        self.app.statusBar().showMessage(f'Created {count} native Vulkan cell(s)', 4000)

    def reset(self):
        self.diameter.setValue(3000.); self.width.setValue(2); self.color = QColor('#2979ff'); self._color_style()


def _show(app, attr, cls):
    if get_native_creation(app, create=True) is None:
        app.statusBar().showMessage('Block and Cell creation require the native Vulkan plan view', 4000)
        return True
    dialog = getattr(app, attr, None)
    try:
        if dialog is None or not dialog.windowTitle(): raise RuntimeError
    except (RuntimeError, ReferenceError):
        dialog = cls(app); setattr(app, attr, dialog)
    dialog.show(); dialog.raise_(); dialog.activateWindow()
    return True


def show_native_block_dialog(app):
    return _show(app, 'native_block_creation_dialog', NativeBlockDialog)


def show_native_cell_dialog(app):
    return _show(app, 'native_cell_creation_dialog', NativeCellDialog)
