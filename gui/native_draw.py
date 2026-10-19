"""Drawing tools for the native LNT/Vulkan plan viewport; no VTK objects."""
from __future__ import annotations

import math
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QInputDialog

from gui.native_main_classify import MainCanvas
from gui.native_section_classify import CanvasOverlay


TOOL_MAP = {
    'smartline': 'smartline', 'line': 'line', 'orthopolygon': 'ortho',
    'polyline': 'polyline', 'polyline_permanent': 'polyline',
    'rectangle': 'rectangle', 'rectangle_permanent': 'rectangle',
    'circle': 'circle', 'circle_permanent': 'circle',
    'freehand': 'freehand', 'freehand_permanent': 'freehand',
    'text': 'text', 'gispoint': 'point', 'vertex': 'vertex',
    'move_vertex': 'movevertex', 'movevertex': 'movevertex',
    'delete_vertex': 'deletevertex', 'deletevertex': 'deletevertex',
    'select_drawing': 'select', 'selectdrawing': 'select', 'select': 'select',
    'curve_point': 'curve', 'curve': 'curve', 'hatcharea': 'hatch',
    'accudraw': 'accudraw',
    'parallel': 'parallel', 'centerline': 'centerline',
}


class NativeDraw:
    def __init__(self, view):
        self.view = view
        self.app = view.app
        self.canvas = MainCanvas(view)
        self.overlay = CanvasOverlay(self.canvas)
        self.drawings = []
        self.redo = []
        self.tool = None
        self.points = []
        self.hover = None
        self.dragging = False
        self.selected = None
        self.vertex_target = None
        self.aux_shapes = []
        self.reload_styles()

    def reload_styles(self):
        from gui.draw_settings_dialog import load_draw_settings
        self.styles = load_draw_settings()
        self._paint()

    def _style(self, tool):
        from PySide6.QtGui import QColor
        key = {'ortho': 'orthopolygon', 'hatch': 'hatcharea'}.get(tool, tool)
        style = dict(self.styles.get(key, {}))
        if 'color' in style:
            style['color'] = QColor.fromRgbF(*style['color']).name()
        if 'fill_color' in style:
            style['fill_color'] = QColor.fromRgbF(*style['fill_color']).name()
        return style

    def activate(self, name):
        tool = TOOL_MAP.get(str(name or '').lower().replace(' ', '_'))
        if tool is None:
            return False
        measurement = getattr(self.view, 'measurement', None)
        if measurement is not None:
            measurement.cancel()
        classifier = getattr(self.view, 'classifier', None)
        if classifier is not None:
            classifier.cancel()
        sections = getattr(self.app, '_native_sections', None)
        if sections is not None:
            sections.cancel_cross(); sections.cancel_cut()
        self.cancel(stand_down=False)
        self.tool = tool
        self.canvas.setFocus()
        self.app.statusBar().showMessage(self._instruction(), 5000)
        return True

    def _instruction(self):
        if self.tool in ('line', 'rectangle', 'circle', 'accudraw'):
            return f'Native {self.tool}: click two points'
        if self.tool == 'polyline':
            return 'Native polygon: click vertices; right-click or double-click to close'
        if self.tool in ('smartline', 'ortho', 'curve', 'hatch'):
            return f'Native {self.tool}: click points; right-click or double-click to finish'
        if self.tool == 'freehand':
            return 'Native Freehand: hold and drag the left button'
        return f'Native {self.tool}: click a position'

    def active(self):
        return self.tool is not None

    def _point(self, pos):
        p = self.canvas.section_at(pos.x(), pos.y())
        digitizer = getattr(self.app, 'digitizer', None)
        if digitizer is not None and getattr(digitizer, 'snap_enabled', False) and self.drawings:
            mode = getattr(digitizer, 'snap_mode', 'nearby')
            candidates = []
            if mode in ('nearby', 'keypoint'):
                candidates = [q for d in self.drawings for q in d.get('coords', [])]
            elif mode == 'midpoint':
                candidates = [((a[0]+b[0])*.5, (a[1]+b[1])*.5)
                              for d in self.drawings for a, b in zip(d.get('coords', []), d.get('coords', [])[1:])]
            elif mode == 'center':
                for d in self.drawings:
                    coords = d.get('coords', [])
                    if coords:
                        candidates.append((sum(q[0] for q in coords)/len(coords),
                                           sum(q[1] for q in coords)/len(coords)))
            elif mode == 'intercept':
                hit = self._nearest_segment(p)
                candidates = [hit[3]] if hit else []
            if candidates:
                snap = min(candidates, key=lambda q: math.hypot(p[0]-q[0], p[1]-q[1]))
                if math.hypot(p[0]-snap[0], p[1]-snap[1]) <= self.canvas.world_per_pixel()*14.:
                    p = snap
        if self.tool == 'ortho' and self.points:
            a = self.points[-1]
            return (p[0], a[1]) if abs(p[0]-a[0]) >= abs(p[1]-a[1]) else (a[0], p[1])
        if self.tool == 'accudraw' and self.points:
            a = self.points[-1]
            dx, dy = p[0]-a[0], p[1]-a[1]
            angle = round(math.atan2(dy, dx)/(math.pi/4))*(math.pi/4)
            length = math.hypot(dx, dy)
            return a[0]+length*math.cos(angle), a[1]+length*math.sin(angle)
        return p

    def _commit(self, kind=None, points=None, **extra):
        pts = list(points if points is not None else self.points)
        if not pts:
            return False
        self.drawings.append({'type': kind or self.tool, 'coords': pts, **extra})
        self.redo.clear()
        self.points, self.hover, self.dragging = [], None, False
        self._paint()
        return True

    def _nearest_vertex(self, point):
        best = None
        for di, drawing in enumerate(self.drawings):
            for vi, vertex in enumerate(drawing.get('coords', [])):
                distance = math.hypot(point[0]-vertex[0], point[1]-vertex[1])
                if best is None or distance < best[0]:
                    best = distance, di, vi
        tolerance = max(self.canvas.world_per_pixel() * 12., 1e-9)
        return best if best is not None and best[0] <= tolerance else None

    def _nearest_segment(self, point):
        best = None
        for di, drawing in enumerate(self.drawings):
            coords = drawing.get('coords', [])
            for si, (a, b) in enumerate(zip(coords, coords[1:])):
                dx, dy = b[0]-a[0], b[1]-a[1]
                d2 = dx*dx + dy*dy
                t = max(0., min(1., ((point[0]-a[0])*dx + (point[1]-a[1])*dy)/d2)) if d2 else 0.
                q = a[0]+t*dx, a[1]+t*dy
                distance = math.hypot(point[0]-q[0], point[1]-q[1])
                if best is None or distance < best[0]:
                    best = distance, di, si, q
        tolerance = max(self.canvas.world_per_pixel() * 12., 1e-9)
        return best if best is not None and best[0] <= tolerance else None

    def _finish(self):
        minimum = 3 if self.tool in ('polyline', 'hatch') else 2
        if len(self.points) < minimum:
            self.points, self.hover = [], None
            self._paint()
            return False
        pts = list(self.points)
        if self.tool in ('polyline', 'hatch', 'ortho') and pts[0] != pts[-1]:
            pts.append(pts[0])
        return self._commit(points=pts)

    def press(self, pos, button):
        if not self.active() or button == Qt.MiddleButton:
            return False
        if button == Qt.RightButton:
            if self.tool in ('smartline', 'polyline', 'ortho', 'curve', 'hatch'):
                self._finish()
            else:
                self.points, self.hover, self.dragging = [], None, False
                self._paint()
            return True
        if button != Qt.LeftButton:
            return False
        p = self._point(pos)
        if self.tool == 'select':
            hit = self._nearest_segment(p) or self._nearest_vertex(p)
            self.selected = hit[1] if hit else None
            self._paint()
            return True
        if self.tool == 'parallel':
            if self.selected is None or self.selected >= len(self.drawings):
                self.app.statusBar().showMessage('Select a drawing before Parallel', 2500)
                return True
            source = self.drawings[self.selected]
            hit = self._nearest_segment(p)
            if hit:
                q = hit[3]; dx, dy = p[0]-q[0], p[1]-q[1]
                self._commit('parallel', [(x+dx, y+dy) for x, y in source.get('coords', [])])
            return True
        if self.tool == 'centerline':
            if self.selected is None or self.selected >= len(self.drawings):
                self.app.statusBar().showMessage('Select the first drawing before Centerline', 2500)
                return True
            hit = self._nearest_segment(p)
            if hit and hit[1] != self.selected:
                a = self.drawings[self.selected].get('coords', [])
                b = self.drawings[hit[1]].get('coords', [])
                n = min(len(a), len(b))
                self._commit('centerline', [((a[i][0]+b[i][0])*.5, (a[i][1]+b[i][1])*.5)
                                            for i in range(n)])
            return True
        if self.tool == 'deletevertex':
            hit = self._nearest_vertex(p)
            if hit:
                _, di, vi = hit
                before = dict(self.drawings[di]); before['coords'] = list(before['coords'])
                coords = self.drawings[di]['coords']
                if len(coords) <= 2:
                    self.redo.append(self.drawings.pop(di))
                else:
                    coords.pop(vi); self.redo.append(before)
                self._paint()
            return True
        if self.tool == 'movevertex':
            if self.vertex_target is None:
                hit = self._nearest_vertex(p)
                self.vertex_target = (hit[1], hit[2]) if hit else None
            else:
                di, vi = self.vertex_target
                before = dict(self.drawings[di]); before['coords'] = list(before['coords'])
                self.drawings[di]['coords'][vi] = p
                self.redo.append(before)
                self.vertex_target = None
            self._paint()
            return True
        if self.tool == 'vertex':
            hit = self._nearest_segment(p)
            if hit:
                _, di, si, _ = hit
                self.drawings[di]['coords'].insert(si + 1, p)
            else:
                self._commit('point', [p])
            self._paint()
            return True
        if self.tool == 'text':
            text, ok = QInputDialog.getText(self.app, 'Draw Text', 'Text:')
            if ok and text:
                self._commit('text', [p], text=text)
            return True
        if self.tool == 'point':
            self._commit('point', [p])
            return True
        self.points.append(p)
        if self.tool == 'freehand':
            self.dragging = True
        elif self.tool in ('line', 'rectangle', 'circle', 'accudraw') and len(self.points) == 2:
            self._commit()
        self._paint()
        return True

    def move(self, pos, buttons=Qt.NoButton):
        if not self.active() or buttons & Qt.MiddleButton:
            return False
        p = self._point(pos)
        if self.tool == 'freehand' and self.dragging and buttons & Qt.LeftButton:
            self.points.append(p)
        else:
            self.hover = p if self.points else None
        self._paint()
        return True

    def release(self, pos, button):
        if not self.active() or button != Qt.LeftButton:
            return False
        if self.tool == 'freehand' and self.dragging:
            self.points.append(self._point(pos))
            self._commit()
            return True
        return self.tool in ('line', 'rectangle', 'circle', 'accudraw')

    def double_click(self, pos, button):
        if self.active() and button == Qt.LeftButton and self.tool in ('smartline', 'polyline', 'ortho', 'curve', 'hatch'):
            p = self._point(pos)
            if not self.points or p != self.points[-1]:
                self.points.append(p)
            self._finish()
            return True
        return False

    def key(self, key):
        if not self.active():
            return False
        if key == Qt.Key_Escape:
            self.cancel()
            return True
        if key in (Qt.Key_Backspace, Qt.Key_Delete) and self.points:
            self.points.pop(); self._paint(); return True
        return False

    def history(self, redo=False):
        source, target = (self.redo, self.drawings) if redo else (self.drawings, self.redo)
        if source:
            target.append(source.pop())
            self._paint()
        return True

    def clear(self):
        if self.drawings:
            self.redo.extend(reversed(self.drawings))
            self.drawings.clear()
        self.points, self.hover = [], None
        self._paint()

    def cancel(self, stand_down=True):
        self.points, self.hover, self.dragging, self.vertex_target = [], None, False, None
        if stand_down:
            self.tool = None
        self._paint()

    def _sync_block_vectors(self):
        """Keep authored blocks in the same camera frame as attached linework."""
        blocks = [d for d in self.drawings if d.get('source') == 'block_creation']
        resident = getattr(self, '_block_vector_layer', None)
        name = '__native_created_blocks__'
        if not blocks:
            if resident is not None:
                resident.remove('drawing', name)
                self._present_vector_frame(resident)
            self._block_vector_layer = None
            self._block_vector_signature = None
            return False
        from gui.native_vector_overlay import vector_layer_for
        layer = vector_layer_for(self.app)
        if layer is None:
            return False
        entities = []
        for drawing in blocks:
            color = drawing.get('color', (0, 179, 255))
            if isinstance(color, str):
                from PySide6.QtGui import QColor
                color = QColor(color).getRgb()[:3]
            entities.append({'type': 'polyline', 'layer': 'Created blocks',
                             'points': drawing.get('coords', []), 'color': tuple(color),
                             '_lineweight': max(0, (int(drawing.get('width', 3)) - 1) * 2)})
        signature = (id(layer), tuple((tuple(tuple(p) for p in e['points']),
                                      e['color'], e['_lineweight']) for e in entities))
        if signature != getattr(self, '_block_vector_signature', None):
            if layer.put_entities('drawing', name, entities) is None:
                return False
            self._block_vector_layer = layer
            self._block_vector_signature = signature
            self._present_vector_frame(layer)
        return True

    def _present_vector_frame(self, layer):
        """Show changed block linework now, not at the next zoom/pan.

        Uploading only updates the GPU draw ranges. An .lnt view presents every
        tick (its vector backend's request_render is a no-op), but the promoted
        Vulkan viewport of an SNT-only session renders on demand, so ask it for
        one frame - coalesced, since Create/Clear may sync several times.
        """
        if getattr(self, '_vector_frame_pending', False):
            return
        self._vector_frame_pending = True

        def present():
            self._vector_frame_pending = False
            backend = getattr(layer, '_backend', None)
            try:
                if backend is not None:
                    backend.request_render()
            except Exception as exc:  # noqa: BLE001 - a missed frame must not break editing
                print(f'[native draw] block linework frame request failed: {exc}')

        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, present)

    def _paint(self):
        gpu_blocks = self._sync_block_vectors()
        shapes = []
        for i, drawing in enumerate(self.drawings):
            if gpu_blocks and drawing.get('source') == 'block_creation':
                continue
            data = self._style(drawing.get('type'))
            data.update(drawing)
            data['selected'] = i == self.selected
            shapes.append(('drawing', drawing.get('coords', []), data))
        if self.points:
            preview = self.points + ([self.hover] if self.hover else [])
            # Draw Polyline is the polygon tool in the ribbon. Show the closing
            # edge during placement, rather than waiting until completion, so
            # the preview always communicates that the result is a closed ring.
            if self.tool == 'polyline' and len(preview) >= 3 and preview[-1] != preview[0]:
                preview = preview + [preview[0]]
            shapes.append(('drawing_preview', preview, {'type': self.tool, **self._style(self.tool)}))
        shapes.extend(self.aux_shapes)
        self.overlay.set_shapes(shapes)

    def shutdown(self):
        layer = getattr(self, '_block_vector_layer', None)
        if layer is not None:
            layer.remove('drawing', '__native_created_blocks__')
        self.cancel()
        self.overlay._timer.stop()
        self.overlay.close()


def native_plan_view(app, allow_snt=False):
    """The native 2D plan view: the open .lnt's, or - with allow_snt - the promoted
    Vulkan viewport of an SNT-only session (engine MVP mapping, no point stream)."""
    view = getattr(app, '_lnt_main_view', None)
    if view is not None and getattr(view, 'native', None) is not None:
        return view if getattr(view, 'mode_2d', False) else None
    if allow_snt:
        from gui.native_vector_overlay import _view_for, snt_viewport_active
        if snt_viewport_active(app):
            view = _view_for(app)
            if view is not None and getattr(getattr(view, 'owner', None), 'vulkan_widget', None) is not None:
                return view
    return None


def get_native_draw(app, create=False, allow_snt=False):
    view = native_plan_view(app, allow_snt)
    if view is None:
        return None
    draw = getattr(view, 'drawing', None)
    if draw is None and create:
        draw = view.drawing = NativeDraw(view)
    return draw


def activate_native_draw(app, tool):
    draw = get_native_draw(app, create=True)
    return bool(draw and draw.activate(tool))
