"""Footer point targeting on native viewports, without VTK picking or actors."""
import math
import numpy as np
from PySide6.QtCore import QObject, QEvent, Qt
from PySide6.QtWidgets import QApplication


class NativePointTarget(QObject):
    def __init__(self, tool):
        super().__init__(tool)
        self.tool, self.app = tool, tool.app
        self._section_backends = []
        self._layer = None

    def start(self):
        QApplication.instance().installEventFilter(self)

    def stop(self):
        QApplication.instance().removeEventFilter(self)
        if self._layer is not None:
            self._layer.remove('target', '__point_target__')
            self._layer = None
        for backend in self._section_backends:
            backend.vector_clear_draw_ranges()
            backend.request_render()
        self._section_backends.clear()

    def eventFilter(self, obj, event):
        if not self.tool.active:
            return False
        view = getattr(self.app, '_lnt_main_view', None) or getattr(self.app, '_snt_viewport_view', None)
        main = getattr(getattr(view, 'owner', None), 'vulkan_widget', None)
        sections = getattr(getattr(self.app, '_native_sections', None), 'views', {})
        section = next((v for v in sections.values() if v is obj), None)
        if obj is not main and section is None:
            return False
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self.tool.deactivate()
            return True
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            if section is None:
                from gui.native_identify import NativeIdentify
                picker = NativeIdentify(view)
                canvas = picker._canvas()
                wx, wy = canvas.section_at(event.position().x(), event.position().y())
                hit = picker._pick(wx, wy)
                row = hit[0] if hit is not None else None
                if hit is not None:
                    sx, sy = canvas.screen_from_section(float(hit[1][0]), float(hit[1][1]))
                    if (sx-event.position().x())**2 + (sy-event.position().y())**2 > 8.**2:
                        row = None
            else:
                row = self.pick_section(section, event.position())
            if row is None:
                self.app.statusBar().showMessage('Point target: no point near click', 2500)
            else:
                self.highlight(row)
            return True
        if event.type() in (QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick) and event.button() == Qt.LeftButton:
            return True
        return False

    @staticmethod
    def pick_section(canvas, pos):
        if not len(canvas.indices):
            return None
        h, z = canvas.section_at(pos.x(), pos.y())
        points = canvas.local
        d2 = (points[:, canvas.axis] - h)**2 + (points[:, 2] - z)**2
        best = int(np.argmin(d2))
        if d2[best] > (8. * canvas.world_per_pixel())**2:
            return None
        return int(canvas.indices[best])

    @staticmethod
    def ring(point, radius, horizontal=0, vertical=1):
        coords = []
        for angle in np.linspace(0., 2.*math.pi, 49):
            p = list(map(float, point))
            p[horizontal] += radius * math.cos(angle)
            p[vertical] += radius * math.sin(angle)
            coords.append(tuple(p))
        coords[-1] = coords[0]
        return {'type': 'polyline', 'points': coords, 'color': (255, 255, 255), '_lineweight': 2}

    def highlight(self, row):
        from gui.lnt_section_bridge import get_store
        from gui.native_main_classify import MainCanvas
        from gui.native_vector_overlay import vector_layer_for, group_entities
        store = get_store(self.app)
        if store is None or row < 0 or row >= len(store.data['xyz']):
            return False
        xyz = np.asarray(store.data['xyz'])[row]
        code = int(store.data['classification'][row])
        view = getattr(self.app, '_lnt_main_view', None)
        if view is None:
            return False
        self.tool.selected_global_index = int(row)
        layer = vector_layer_for(self.app)
        if layer is not None:
            radius = MainCanvas(view).world_per_pixel() * 11.
            layer.put_entities('target', '__point_target__', [self.ring(xyz, radius)])
            self._layer = layer
        for backend in self._section_backends:
            backend.vector_clear_draw_ranges()
            backend.request_render()
        self._section_backends.clear()
        for canvas in getattr(getattr(self.app, '_native_sections', None), 'views', {}).values():
            matches = np.flatnonzero(canvas.indices == row)
            if not len(matches):
                continue
            point = canvas.local[int(matches[0])].copy()
            # Section upload swaps horizontal/across columns for side views.
            point = point[[canvas.axis, 1-canvas.axis, 2]]
            entity = self.ring(point, canvas.world_per_pixel()*11., 0, 2)
            chunks = group_entities([entity])['0']
            chunk = chunks[0]
            backend = canvas.backend
            if backend.vector_reserve(chunk['v_count'], chunk['i_count']):
                if backend.vector_upload_strip(0, chunk['v_count'], chunk['xyz'], chunk['rgb'], 0, chunk['i_count'], chunk['indices']):
                    backend.vector_set_draw_ranges([0], [chunk['i_count']], [2.])
                    backend.request_render()
                    self._section_backends.append(backend)
        # Display information without the legacy class-highlight actor path.
        info = getattr(self.app, 'identification_tool', None)
        name = info.get_class_name(code) if info is not None else str(code)
        label = getattr(self.app, 'snt_layer_pick_footer_label', None)
        if label is not None:
            label.setText(f'Class: {code} : {name}')
            label.setToolTip(f'Point #{row:,}  X: {xyz[0]:.4f}  Y: {xyz[1]:.4f}  Z: {xyz[2]:.4f}')
        self.app.statusBar().showMessage(f'Point #{row:,} | Class {code}: {name} | X {xyz[0]:.4f}  Y {xyz[1]:.4f}  Z {xyz[2]:.4f}', 8000)
        return True
