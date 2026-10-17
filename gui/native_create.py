"""Native Vulkan Block and Cell creation for the LNT plan viewport.

This module intentionally contains no VTK imports or compatibility fallback.
Selection, hit testing, previews and generated geometry use the native canvas
and the shared Qt overlay above the Vulkan swapchain.
"""
from __future__ import annotations

import math
import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractSpinBox, QColorDialog, QComboBox, QDialog, QDoubleSpinBox,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QVBoxLayout,
)

from gui.native_draw import get_native_draw
from gui.theme_manager import ThemeColors, ThemeManager, get_dialog_stylesheet


def _xy(point):
    return float(point[0]), float(point[1])


def _distance_to_segment(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    d2 = dx * dx + dy * dy
    t = max(0.0, min(1.0, ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / d2)) if d2 else 0.0
    q = a[0] + t*dx, a[1] + t*dy
    return math.hypot(p[0]-q[0], p[1]-q[1])


def _split_polyline(coords, distance):
    """Split a polyline into chains no longer than distance."""
    pts = [_xy(p) for p in coords]
    if len(pts) < 2 or distance <= 0:
        return [pts]
    result, chain, remaining = [], [pts[0]], float(distance)
    for start, end in zip(pts, pts[1:]):
        a = start
        seg = math.hypot(end[0]-a[0], end[1]-a[1])
        while seg > remaining and seg > 1e-12:
            t = remaining / seg
            cut = tuple(a[i] + (end[i]-a[i])*t for i in range(len(a)))
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
        self.draw = get_native_draw(self.app, create=True, allow_snt=True)
        self.canvas = self.draw.canvas
        self.mode = None
        self.selected = []
        self.hovered = None
        self._candidate_lines = []
        self._mouse_tracking_before_selection = None
        self._press_px = None
        self._drag_end_px = None

    @property
    def active(self):
        return self.mode in ('block', 'cell')

    def arm(self, mode):
        self._press_px = self._drag_end_px = None
        self.mode = mode
        self._sync_buttons()
        self._candidate_lines = self._collect_candidates()
        self.draw.cancel()
        measurement = getattr(self.view, 'measurement', None)
        if measurement is not None:
            measurement.cancel()
        classifier = getattr(self.view, 'classifier', None)
        if classifier is not None:
            classifier.cancel()
        if self._mouse_tracking_before_selection is None:
            self._mouse_tracking_before_selection = self.canvas.hasMouseTracking()
        self.canvas.setMouseTracking(True)
        self.canvas.setCursor(Qt.PointingHandCursor)
        self.app.statusBar().showMessage(
            'Select CL: click lines to add them, or drag a rectangle to select several. Middle-drag pans.', 0)
        self.paint()

    def disarm(self):
        self._press_px = self._drag_end_px = None
        self.mode = None; self.hovered = None
        if self._mouse_tracking_before_selection is not None:
            self.canvas.setMouseTracking(self._mouse_tracking_before_selection)
            self._mouse_tracking_before_selection = None
        self.canvas.setCursor(Qt.ArrowCursor)
        self.paint()
        self._sync_buttons()

    def _sync_buttons(self):
        """Select CL stays checked (highlighted) exactly while its mode is armed -
        whether it was switched by the button, Clear, closing, right-click or Esc."""
        for mode in ('block', 'cell'):
            dialog = getattr(self.app, f'native_{mode}_creation_dialog', None)
            button = getattr(dialog, 'select_btn', None)
            if button is None:
                continue
            try:
                armed = self.mode == mode
                if button.isChecked() != armed:
                    button.blockSignals(True); button.setChecked(armed); button.blockSignals(False)
            except RuntimeError:
                pass                            # dialog already deleted

    def _collect_candidates(self):
        candidates = [d for d in self.draw.drawings
                      if d.get('type') in ('line', 'smartline', 'polyline', 'centerline')
                      and d.get('source') not in ('block_creation', 'cell_creation')]
        for attachment in getattr(self.app, 'snt_attachments', []) or []:
            # Match the project-space geometry actually uploaded to Vulkan.
            # parsed is source-space metadata and may not contain any entities.
            entities = attachment.get('_render_entities')
            if entities is None:
                entities = attachment.get('entities')
            if entities is None:
                entities = (attachment.get('parsed') or {}).get('entities', [])
            for entity in entities or []:
                kind = str(entity.get('type', '')).strip().upper()
                if kind not in ('LINE', 'POLYLINE', 'LWPOLYLINE'):
                    continue
                layer = str(entity.get('layer', '')).strip()
                if set(re.findall(r'[a-z0-9]+', layer.casefold())) & {'grid', 'bbox', 'trajectory', 'traj', 'trj'}:
                    continue
                if kind != 'LINE' and entity.get('closed'):
                    continue
                coords = entity.get('points') or entity.get('vertices') or entity.get('coords')
                if not coords and kind == 'LINE' and entity.get('start') is not None and entity.get('end') is not None:
                    coords = [entity['start'], entity['end']]
                if not coords or len(coords) < 2:
                    continue
                try:
                    coords = [_xy(p) for p in coords]
                except (TypeError, ValueError, IndexError):
                    continue
                if not all(math.isfinite(v) for p in coords for v in p):
                    continue
                if math.hypot(coords[-1][0]-coords[0][0], coords[-1][1]-coords[0][1]) <= 1e-9:
                    continue
                candidates.append({'type': 'polyline', 'coords': coords, 'layer': layer,
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
        self._press_px = (float(pos.x()), float(pos.y()))
        self._drag_end_px = None
        return True

    @staticmethod
    def _segment_in_box(a, b, box):
        xmin, xmax, ymin, ymax = box
        dx, dy = b[0] - a[0], b[1] - a[1]
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, a[0]-xmin), (dx, xmax-a[0]),
                     (-dy, a[1]-ymin), (dy, ymax-a[1])):
            if abs(p) < 1e-12:
                if q < 0.0:
                    return False
                continue
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return False
        return True

    def _lines_in_rectangle(self, start, end):
        box = (min(start[0], end[0]), max(start[0], end[0]),
               min(start[1], end[1]), max(start[1], end[1]))
        hits = []
        for drawing in self.candidates():
            points = [self.canvas.screen_from_section(*p[:2])
                      for p in drawing.get('coords', [])]
            if any(self._segment_in_box(a, b, box) for a, b in zip(points, points[1:])):
                hits.append(drawing)
        return hits

    def release(self, pos, button):
        if not self.active or button != Qt.LeftButton or self._press_px is None:
            return False
        start = self._press_px
        end = (float(pos.x()), float(pos.y()))
        self._press_px = self._drag_end_px = None
        if math.hypot(end[0]-start[0], end[1]-start[1]) >= 4.0:
            hits = self._lines_in_rectangle(start, end)
        else:
            hit = self.hit(pos)
            hits = [hit] if hit is not None else []
        existing = {id(drawing) for drawing in self.selected}
        for hit in hits:
            if id(hit) not in existing:
                self.selected.append(hit)
                existing.add(id(hit))
        self.hovered = None
        self.paint()
        self._notify_dialog()
        return True

    def move(self, pos):
        if not self.active:
            return False
        if self._press_px is not None:
            end = (float(pos.x()), float(pos.y()))
            if self._drag_end_px is not None or math.hypot(
                    end[0]-self._press_px[0], end[1]-self._press_px[1]) >= 4.0:
                self._drag_end_px = end
                self.hovered = None
                self.paint()
                return True
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
            shapes.append(('drawing_preview', self.hovered.get('coords', []),
                           {'type': 'polyline', 'color': '#00ffff', 'width': 5, 'line_style': 'solid'}))
        if self._press_px is not None and self._drag_end_px is not None:
            corners = [self.canvas.section_at(*p) for p in (self._press_px, self._drag_end_px)]
            shapes.append(('drawing_preview', corners,
                           {'type': 'rectangle', 'color': '#00ffff', 'width': 2,
                            'line_style': 'dashed'}))
        self.draw.aux_shapes = shapes
        self.draw._paint()

    def clear_selection(self):
        self._press_px = self._drag_end_px = None
        self.selected.clear(); self.hovered = None; self.paint(); self._notify_dialog()

    def _notify_dialog(self):
        # Block and Cell share the selection even after Select CL is disarmed.
        for mode in ('cell', 'block'):
            dialog = getattr(self.app, f'native_{mode}_creation_dialog', None)
            if dialog is not None:
                try:
                    dialog.update_selection()
                except RuntimeError:
                    pass  # Qt may have already deleted a closed dialog.

    def create_cells(self, diameter_mm, color, width):
        radius = float(diameter_mm) / 2000.0
        # Keep source geometry with the cells, independently of the transient
        # Select CL state. Shared by this batch so thousands of cells do not
        # each copy the same centreline coordinates.
        source_lines = tuple({
            'coords': tuple(tuple(float(v) for v in p[:3]) for p in line.get('coords', [])),
            'layer': line.get('layer', ''),
            'snt_filename': line.get('snt_filename'),
            'snt_entity_id': line.get('snt_entity_id'),
        } for line in self.selected)
        unique = []
        for line in self.selected:
            for p in line.get('coords', []):
                p = _xy(p)
                if not any(math.hypot(p[0]-q[0], p[1]-q[1]) <= .01 for q in unique):
                    unique.append(p)
        for center in unique:
            ring = [(center[0]+radius*math.cos(i*2*math.pi/72),
                     center[1]+radius*math.sin(i*2*math.pi/72)) for i in range(72)]
            ring.append(ring[0])
            self.draw.drawings.append({'type': 'polyline', 'coords': ring, 'source': 'cell_creation',
                                       'center': (center[0], center[1], 0.0), 'radius': radius,
                                       'source_cl_lines': source_lines,
                                       'color': color, 'width': width})
        self.draw.redo.clear(); self.draw._paint()
        return len(unique)

    def clear_cells(self):
        """Remove generated cells from the drawing overlay and pending redo."""
        drawings = self.draw.drawings
        selected_index = getattr(self.draw, 'selected', None)
        selected = (drawings[selected_index] if isinstance(selected_index, int)
                    and 0 <= selected_index < len(drawings) else None)
        kept = [drawing for drawing in drawings if drawing.get('source') != 'cell_creation']
        removed = len(drawings) - len(kept)
        drawings[:] = kept
        self.draw.redo[:] = [drawing for drawing in self.draw.redo
                             if drawing.get('source') != 'cell_creation']
        if removed:
            self.draw.selected = next((i for i, drawing in enumerate(kept)
                                       if drawing is selected), None)
            self.draw.vertex_target = None
        self.draw._paint()
        return removed

    def create_blocks(self, left_width, right_width, distance):
        from gui.native_block_geometry import NativeBlockGeometry
        geometry = NativeBlockGeometry(self.app, self.selected, left_width, right_width, distance)
        records = geometry._generate_block_lines()
        # Reference behaviour: Create regenerates the blocks, preserving cells,
        # source CLs and all unrelated drawings.
        self.draw.drawings[:] = [d for d in self.draw.drawings if d.get('source') != 'block_creation']
        for record in records:
            self.draw.drawings.extend(record.get('drawings') or [])
        self.draw.redo.clear()
        self.draw.selected = None
        self.draw.vertex_target = None
        self.draw._paint()
        return len(records)



def get_native_creation(app, create=False):
    # .lnt plan view, or the Vulkan viewport of an SNT-only session: Block/Cell
    # work from SNT centrelines whether or not a point cloud is loaded.
    from gui.native_draw import native_plan_view
    view = native_plan_view(app, allow_snt=True)
    if view is None:
        return None
    controller = getattr(view, 'creation', None)
    if controller is None and create:
        controller = view.creation = NativeCreation(view)
        if getattr(app, '_lnt_main_view', None) is not view:
            from gui.native_measure import ensure_snt_input
            ensure_snt_input(view)       # SNT-only: route the Vulkan widget's clicks here
    return controller


class _BaseDialog(QDialog):
    mode = ''
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.controller = get_native_creation(app, create=True)
        self.setWindowTitle(f'{self.mode.title()} Creation Settings')
        self.setProperty('renderBackend', 'native-vulkan')
        self.setSizeGripEnabled(False)
        self.setModal(False); self.setFixedWidth(410)
        accent = ThemeColors.get('accent') or '#2979ff'
        hover = ThemeColors.get('accent_hover') or accent
        # Select CL is a toggle: checked = selection mode is armed, so it must look on.
        self.setStyleSheet(get_dialog_stylesheet() + f"""
            QPushButton#selectClButton:checked {{
                background-color: {accent}; border: 1px solid {hover}; color: white; font-weight: bold;
            }}
            QPushButton#selectClButton:checked:hover {{ background-color: {hover}; }}
        """)
        self.root = QVBoxLayout(self)
        group = QGroupBox('CL'); row = QHBoxLayout(group)
        self.select_btn = QPushButton('Select CL'); self.select_btn.setCheckable(True)
        self.select_btn.setObjectName('selectClButton')
        self.select_btn.setToolTip('Hover to highlight CL lines. Click each line or drag a rectangle to add multiple lines. Click this button again to stop')
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
        self.controller.clear_selection()
        self.update_selection()
        self.reset()

    def reset(self): pass

    def closeEvent(self, event):
        if self.controller.mode == self.mode:
            self.controller.disarm()
        self.select_btn.setChecked(False); super().closeEvent(event)


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
        self.setFixedSize(self.size())

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
        self.setFixedSize(self.size())

    def _clear(self):
        self.controller.clear_cells()
        super()._clear()

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
        app.statusBar().showMessage('Block and Cell creation need the 2D plan view, with an SNT '
                                    'attached or a point cloud loaded', 4000)
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
