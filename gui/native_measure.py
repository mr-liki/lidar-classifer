"""Line, path, grid-area and block-area measurements for native Vulkan viewports (no VTK objects)."""
import math

from PySide6.QtCore import QEvent, QObject, Qt
from gui.native_section_classify import CanvasOverlay
from gui.measure_settings_dialog import load_measure_settings


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
        self._picker = None

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
        return dict(styles['path' if (self.path if path is None else path) else 'line'])

    @staticmethod
    def distance(points):
        return sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(points, points[1:]))

    @staticmethod
    def _text(distance, unit):
        if unit == 'km':
            return f'{distance / 1000.0:.3f} km'
        return f'{distance:.3f} m'

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
        if len(self.points) < 2:
            self.points, self.hover = [], None
            self._paint()
            return False
        points = list(self.points)
        self.completed.append(points)
        self.completed_modes.append(bool(self.path))
        style = self._style()
        self._status(f'{"Path" if self.path else "Line"}: '
                     f'{self._text(self.distance(points), style.get("unit", "m"))}')
        self.points, self.hover = [], None
        self._paint()
        return True

    def _paint(self):
        shapes = [('measurement', points,
                   {'style': self._style(self.completed_modes[i] if i < len(self.completed_modes) else False),
                    'path': self.completed_modes[i] if i < len(self.completed_modes) else False,
                    'complete': True})
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
            shapes.append(('measurement', self.points + ([self.hover] if self.hover else []),
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
                    self.grids.pop(index)
                    self._status(f'{name} {label} deselected', 2000)
                    self._paint()
                    return True
            area = float(grid.get('area', 0.0))
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
            if self.path:
                self._finish()
            else:
                self.points, self.hover = [], None
                self._paint()
            return True
        if button != Qt.LeftButton:
            return False
        self.points.append(self.canvas.section_at(pos.x(), pos.y()))
        self.undone.clear()
        if not self.path and len(self.points) == 2:
            self._finish()
        else:
            self._paint()
            if self.path and len(self.points) > 1:
                style = self._style()
                self._status(f'Path: {self._text(self.distance(self.points), style.get("unit", "m"))}; '
                             'click next point or right-click to finish')
        return True

    def move(self, pos):
        if not self.active:
            return False
        self.hover = self.canvas.section_at(pos.x(), pos.y()) if self.points else None
        self._paint()
        return True

    def double_click(self, pos, button):
        if not self.active or not self.path or button != Qt.LeftButton:
            return False
        point = self.canvas.section_at(pos.x(), pos.y())
        if not self.points or point != self.points[-1]:
            self.points.append(point)
        self._finish()
        return True

    def key(self, key):
        if not self.active:
            return False
        if key == Qt.Key_Escape:
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
        self.active = False
        self.points, self.hover = [], None
        self._paint()

    def clear(self):
        self.points, self.completed, self.completed_modes, self.hover = [], [], [], None
        self.grids = []
        self.undone.clear()
        self._paint()

    def history(self, redo=False):
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
        elif not redo:
            if self.points:
                self.undone.append(('points', self.points.pop()))
            elif self.grids:
                self.undone.append(('grids', self.grids.pop()))
            elif self.completed:
                path = self.completed_modes.pop() if self.completed_modes else False
                self.undone.append(('completed', (self.completed.pop(), path)))
        self._paint()
        return True

    def shutdown(self):
        self.cancel()
        self.overlay._timer.stop()
        self.overlay.close()


class NativeMeasureInput(QObject):
    """Feeds the Vulkan widget's input to view.measurement (SNT-only sessions).

    An .lnt session routes through lnt_open's filter; without one, the
    promoted SNT viewport's widget only navigates. Same rules as lnt_open:
    while a measurement is active it owns left/right clicks, moves and keys,
    and middle-drag stays with the camera so the user can pan while measuring.
    """

    def __init__(self, view, widget):
        super().__init__(widget)
        self.view = view
        widget.installEventFilter(self)

    def eventFilter(self, obj, ev):
        measurement = getattr(self.view, 'measurement', None)
        if measurement is None or not measurement.active:
            return False
        t = ev.type()
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
