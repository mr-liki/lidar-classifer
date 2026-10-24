"""Line, path, grid-area and block-area measurements for native Vulkan viewports (no VTK objects)."""
import math

from PySide6.QtCore import QEvent, QObject, Qt
from gui.native_section_classify import CanvasOverlay
from gui.measure_settings_dialog import load_measure_settings
from gui.native_measure_geometry import measurement_distance, measurement_text


class NativeMeasure:
    def __init__(self, canvas, path=False, grid_area=False, area_mode=None):
        self.canvas, self.path = canvas, path
        # None: line/path; 'grid': SNT grid cell; 'block': PRJ/SNT/drawn block.
        self.area_mode = area_mode or ('grid' if grid_area else None)
        self.overlay = CanvasOverlay(canvas)
        self.points = []
        self.completed = []
        self.completed_modes = []
        self.grids = []             # picked grid cells / blocks: points/label/area/kind
        self.hover = None
        self.active = True
        self.undone = []
        self._clear_undo_snapshot = None
        self._clear_redo_snapshot = None
        self._picker = None

    def can_history(self, redo=False):
        """Whether Undo/Redo would actually change this measurement overlay."""
        if redo:
            return bool(self.undone or self._clear_redo_snapshot is not None)
        return bool(
            self.points or self.grids or self.completed
            or self._clear_undo_snapshot is not None
        )

    def _new_measurement_action(self):
        """A fresh edit invalidates redo of an undone Clear transaction."""
        self._clear_redo_snapshot = None
        self._clear_undo_snapshot = None

    @property
    def app(self):
        return self.canvas.owner.app

    @property
    def grid_area(self):
        return self.area_mode == 'grid'

    @grid_area.setter
    def grid_area(self, value):
        self.area_mode = 'grid' if value else None

    def _style(self, path=None):
        styles = load_measure_settings()
        key = getattr(self, 'style_key', None)    # 'cross_section' for XS
        if key not in styles:
            key = 'path' if (self.path if path is None else path) else 'line'
        return dict(styles[key])

    @staticmethod
    def distance(points):
        return measurement_distance(points)

    @staticmethod
    def _text(distance, unit):
        return measurement_text(distance, unit)

    def _status(self, message, timeout=4000):
        bar = getattr(self.app, 'statusBar', lambda: None)()
        if bar is not None:
            bar.showMessage(message, timeout)

    def _grid_style(self, kind='grid'):
        """Area outline style: Measure Settings 'grid'/'block' unit/label size."""
        style = dict((load_measure_settings().get(kind) or {}))
        # Grid cells: cyan; blocks: orange, like the VTK block outline.
        style.setdefault('color', (1.0, 0.55, 0.0) if kind == 'block' else (0.3, 1.0, 0.85))
        style.setdefault('width', 2)
        return style

    @staticmethod
    def _area_text(area, unit):
        if unit == 'km':
            return f'{area / 1_000_000.0:.3f} km²'
        return f'{area:,.2f} m²'

    def _pick_area(self, world, kind):
        """Grid cell ('grid') or block ('block') containing the world XY, or None.

        Uses the pick-only index (measure_area_index.AreaIndex, the same one
        the VTK tool inherits): pure data over PRJ files,
        app.snt_block_polygons and digitizer drawings — VTK is never
        imported. Status messages for "nothing attached"
        / "nothing here" are emitted here so the caller only handles a hit.
        """
        name = 'Block' if kind == 'block' else 'Grid'
        try:
            if world is None or len(world) < 2:
                return None
            if self._picker is None or self._picker.app is not self.app:
                from gui.measure_area_index import AreaIndex
                self._picker = AreaIndex(self.app)
            picker = self._picker
            point = (float(world[0]), float(world[1]), 0.0)
            if kind == 'block':
                if not picker._load_block_boundaries():
                    self._status('No PRJ/SNT/Draw block boundaries found for the '
                                 'attached overlays', 4000)
                    return None
                found = picker._find_block_at_world_point(point)
            else:
                if not picker._load_grid_boundaries():
                    self._status('No measurable SNT grid boundaries are attached', 4000)
                    return None
                found = picker._find_grid_at_world_point(point)
            if found is None:
                self._status(f'No {name.lower()} found at this location', 2500)
            return found
        except Exception as exc:
            self._status(f'{name} pick failed: {exc}', 3000)
            return None

    def _pick_grid(self, world):
        return self._pick_area(world, 'grid')

    def _finish(self):
        if len(self.points) < 2 or self.distance(self.points) <= 1e-12:
            self.points, self.hover = [], None
            self._paint()
            return False
        points = list(self.points)
        self._new_measurement_action()
        self.completed.append(points)
        self.completed_modes.append(bool(self.path))
        # Finishing a draft invalidates its per-vertex redo history. Otherwise
        # redo could append an old point to an unrelated new measurement.
        self.undone.clear()
        style = self._style()
        self._status(f'{"Path" if self.path else "Line"}: '
                     f'{self._text(self.distance(points), style.get("unit", "m"))}')
        self.points, self.hover = [], None
        self._paint()
        return True

    def _paint(self):
        # Highlight *the existing measurement stroke* in its own overlay,
        # rather than drawing a second polygon into the Draw overlay (which
        # could appear underneath the measurement layer on some Qt windows).
        # No calls to selection_manager.get_features() here: that would prune
        # and repaint recursively while Qt is rendering this overlay.
        owner = getattr(self.canvas, 'owner', None)
        draw = getattr(owner, 'drawing', None)
        bridge = getattr(draw, '_selection_bridge', None)
        selected = (bridge.selection_manager._selected if bridge is not None else {})
        hovered = getattr(bridge, '_hover_feature', None) if bridge is not None else None
        hovered_key = hovered.get('key') if hovered and hovered.get('source') == 'measure' else None
        shapes = [('measurement', points,
                   {'style': self._style(self.completed_modes[i] if i < len(self.completed_modes) else False),
                    'path': self.completed_modes[i] if i < len(self.completed_modes) else False,
                    'complete': True, 'highlight_scope': 'measure',
                    'selected': ('measure', id(self), id(points)) in selected,
                    'hovered': ('measure', id(self), id(points)) == hovered_key})
                  for i, points in enumerate(self.completed)]
        for cell in self.grids:
            # Closed cell outline + centroid area label (see CanvasOverlay:
            # a 'measurement' shape carrying 'area' paints as a filled cell).
            shapes.append(('measurement', list(cell['points']),
                           {'style': self._grid_style(cell.get('kind', 'grid')),
                            'area': float(cell['area']),
                            'grid_label': cell['label'],
                            'complete': True}))
        if self.points:
            preview = list(self.points)
            if (self.hover is not None and len(self.hover) >= 2
                    and (self.hover[0] != self.points[-1][0]
                         or self.hover[1] != self.points[-1][1])):
                preview.append(self.hover)
            shapes.append(('measurement', preview,
                           {'style': self._style(), 'path': self.path, 'complete': False}))
        self.overlay.set_shapes(shapes)

    def press(self, pos, button):
        if not self.active:
            return False
        if self.area_mode:
            # Area mode: one left click resolves (and toggles) the grid cell
            # or block under the cursor. No vertex collecting, no finish.
            if button != Qt.LeftButton:
                return True
            kind = self.area_mode
            name = 'Block' if kind == 'block' else 'Grid'
            world = self.canvas.section_at(pos.x(), pos.y())
            grid = self._pick_area(world, kind)
            if grid is None:
                return True
            label = str(grid.get('label', ''))
            for index, cell in enumerate(self.grids):
                if cell['label'] == label and cell.get('kind', 'grid') == kind:
                    self._new_measurement_action()
                    self.grids.pop(index)
                    self._status(f'{name} {label} deselected', 2000)
                    self._paint()
                    return True
            area = float(grid.get('area', 0.0))
            self._new_measurement_action()
            self.grids.append({
                'points': [(float(x), float(y)) for x, y in grid['points']],
                'label': label,
                'area': area,
                'kind': kind,
            })
            style = self._grid_style(kind)
            self._status(
                f'▦ {name} {label} area: '
                f'{self._area_text(area, style.get("unit", "m"))}',
                6000,
            )
            self._paint()
            return True
        if button == Qt.RightButton:
            if self.path or getattr(self, 'style_key', None) != 'cross_section':
                # Line and Path both have SmartLine-style multi-click input.
                # Cross-section measurement remains the existing two-click tool.
                self._finish()
            else:
                self.points, self.hover = [], None
                self._paint()
            return True
        if button != Qt.LeftButton:
            return False
        point = self.canvas.section_at(pos.x(), pos.y())
        if point is None or len(point) < 2:
            return True
        if self.points and point[0] == self.points[-1][0] and point[1] == self.points[-1][1]:
            return True  # A double-click must not create a zero-length leg.
        self.points.append(point)
        self._clear_redo_snapshot = None
        self._clear_undo_snapshot = None
        self.undone.clear()
        if (getattr(self, 'style_key', None) == 'cross_section'
                and not self.path and len(self.points) == 2):
            self._finish()  # Preserve cross-section's original two-click workflow.
        else:
            self._paint()
            if len(self.points) > 1:
                style = self._style()
                name = 'Path' if self.path else 'Line'
                self._status(f'{name} total: '
                             f'{self._text(self.distance(self.points), style.get("unit", "m"))}; '
                             'click next point or right-click to finish')
        return True

    def move(self, pos):
        if not self.active:
            return False
        self.hover = self.canvas.section_at(pos.x(), pos.y()) if self.points else None
        self._paint()
        return True

    def double_click(self, pos, button):
        if (not self.active or self.area_mode or button != Qt.LeftButton
                or getattr(self, 'style_key', None) == 'cross_section'):
            return False
        point = self.canvas.section_at(pos.x(), pos.y())
        if (point is not None and len(point) >= 2
                and (not self.points or point[0] != self.points[-1][0]
                     or point[1] != self.points[-1][1])):
            self.points.append(point)
        self._finish()
        return True

    def key(self, key):
        if not self.active:
            return False
        if key == Qt.Key_Escape:
            if getattr(self, 'style_key', None) == 'cross_section':
                # XS is one tool across every section window: stand it down
                # through its button, not just this window's line.
                from gui.native_shortcuts import set_xs
                set_xs(self.app, False)
                self.cancel()
                self._status('Cross-section measurement disabled', 2500)
                return True
            self.cancel()
            self._status('Measurement tool deactivated', 2500)
            return True
        if key in (Qt.Key_Backspace, Qt.Key_Delete) and self.points:
            self.points.pop()
            self.hover = None
            self._paint()
            return True
        return False

    def cancel(self):
        was_active = self.active
        self.active = False
        self.points, self.hover = [], None
        self._paint()
        if was_active:
            self._refresh_ribbon()

    def _refresh_ribbon(self):
        """Keep the Measure ribbon highlight in step with the live tool state."""
        try:
            from gui.native_shortcuts import refresh_measure_ribbon
            refresh_measure_ribbon(self.app)
        except Exception as exc:      # noqa: BLE001 - sync must never break a tool
            print(f'⚠️ Measure ribbon sync failed: {exc}')

    def clear(self):
        if self.points or self.completed or self.grids:
            self._clear_undo_snapshot = (
                list(self.points),
                [list(points) for points in self.completed],
                list(self.completed_modes),
                [dict(cell, points=list(cell.get('points', []))) for cell in self.grids],
            )
        else:
            self._clear_undo_snapshot = None
        self._clear_redo_snapshot = None
        self.points, self.completed, self.completed_modes, self.hover = [], [], [], None
        self.grids = []
        self.undone.clear()
        self._paint()

    def history(self, redo=False):
        handled = False
        if redo and self.undone:
            kind, value = self.undone.pop()
            if kind == 'grids':
                self.grids.append(value)
            elif kind == 'completed':
                points, path = value
                self.completed.append(points)
                self.completed_modes.append(path)
            else:
                getattr(self, kind).append(value)
            handled = True
        elif redo and self._clear_redo_snapshot is not None:
            snapshot = self._clear_redo_snapshot
            self._clear_redo_snapshot = None
            self._clear_undo_snapshot = snapshot
            self.points, self.completed, self.completed_modes, self.grids = [], [], [], []
            self.hover = None
            handled = True
        elif not redo:
            if self.points:
                self.undone.append(('points', self.points.pop()))
                handled = True
            elif self.grids:
                self.undone.append(('grids', self.grids.pop()))
                handled = True
            elif self.completed:
                path = self.completed_modes.pop() if self.completed_modes else False
                self.undone.append(('completed', (self.completed.pop(), path)))
                handled = True
            elif self._clear_undo_snapshot is not None:
                snapshot = self._clear_undo_snapshot
                self._clear_undo_snapshot = None
                self._clear_redo_snapshot = snapshot
                points, completed, modes, grids = snapshot
                self.points = list(points)
                self.completed = [list(item) for item in completed]
                self.completed_modes = list(modes)
                self.grids = [dict(cell, points=list(cell.get('points', []))) for cell in grids]
                self.hover = None
                handled = True
        self._paint()
        return handled

    def shutdown(self):
        self.cancel()
        self.overlay.shutdown()


class NativeMeasureInput(QObject):
    """Feeds the Vulkan widget's input to the native tools of an SNT-only session.

    An .lnt session routes through lnt_open's filter; without one, the promoted
    SNT viewport's widget only navigates. Same order and rules as lnt_open:
    Block/Cell creation (view.creation) first, then measurement. While a tool
    is active it owns left/right clicks, moves and keys; middle-drag always
    stays with the camera so the user can pan while selecting or measuring.
    """

    def __init__(self, view, widget):
        super().__init__(widget)
        self.view = view
        widget.installEventFilter(self)

    def eventFilter(self, obj, ev):
        t = ev.type()
        creation = getattr(self.view, 'creation', None)
        if creation is not None and creation.active:
            if t == QEvent.MouseButtonPress and ev.button() != Qt.MiddleButton:
                if creation.press(ev.position(), ev.button(), ev.modifiers()):
                    obj.setFocus()
                    return True
            elif t == QEvent.MouseMove and not ev.buttons() & Qt.MiddleButton:
                return creation.move(ev.position())
            elif t == QEvent.MouseButtonRelease and ev.button() != Qt.MiddleButton:
                creation.release(ev.position(), ev.button())
                return True      # no right-click context menu while selecting
            elif t == QEvent.KeyPress:
                return creation.key(ev.key())
            return False
        measurement = getattr(self.view, 'measurement', None)
        if measurement is None or not measurement.active:
            return False
        if t == QEvent.MouseButtonPress and ev.button() != Qt.MiddleButton:
            if measurement.press(ev.position(), ev.button()):
                obj.setFocus()
                return True
        elif t == QEvent.MouseMove and not ev.buttons() & Qt.MiddleButton:
            return measurement.move(ev.position())
        elif t == QEvent.MouseButtonDblClick:
            return measurement.double_click(ev.position(), ev.button())
        elif t == QEvent.MouseButtonRelease and ev.button() != Qt.MiddleButton:
            return True      # no right-click context menu while measuring
        elif t == QEvent.KeyPress:
            return measurement.key(ev.key())
        return False


def ensure_snt_input(view):
    """Install the SNT-viewport input filter once per view (measure and creation share it)."""
    if getattr(view, '_measure_input', None) is None:
        widget = getattr(getattr(view, 'owner', None), 'vulkan_widget', None)
        if widget is not None:
            view._measure_input = NativeMeasureInput(view, widget)
    return getattr(view, '_measure_input', None)
