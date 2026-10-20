"""Footer point targeting on native viewports, without VTK picking or actors."""
import math
import numpy as np
from PySide6.QtCore import QObject, QEvent, Qt, QTimer
from PySide6.QtWidgets import QApplication


class NativePointTarget(QObject):
    def __init__(self, tool):
        super().__init__(tool)
        self.tool, self.app = tool, tool.app
        self._section_backends = []
        self._layer = None
        self._camera_signature = None
        self._follow_timer = QTimer(self)
        self._follow_timer.setInterval(50)
        self._follow_timer.timeout.connect(self._follow)

    def start(self):
        from gui.native_vector_overlay import _view_for
        view = _view_for(self.app)  # SNT/native-data sessions need their mapping too.
        self.app.active_classify_tool = None
        for name in ('classifier', 'measurement', 'native_identify', 'drawing'):
            interaction = getattr(view, name, None)
            if interaction is not None:
                interaction.cancel()
        for name in ('identification_tool', 'snt_layer_pick_tool'):
            interaction = getattr(self.app, name, None)
            if interaction is not None and getattr(interaction, 'active', False):
                interaction.deactivate()
        sections = getattr(self.app, '_native_sections', None)
        if sections is not None:
            sections.cancel_cut()
            sections.cancel_cross()
            sections.cancel_point_actions()
        QApplication.instance().installEventFilter(self)
        self._follow_timer.start()

    def stop(self):
        self._follow_timer.stop()
        QApplication.instance().removeEventFilter(self)
        if self._layer is not None:
            self._layer.remove('target', '__point_target__')
            self._layer._backend.request_render()
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
                row = self.pick_main(view, event.position())
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
        data = getattr(canvas.owner.app, 'data', None) or {}
        classes = data.get('classification')
        if classes is not None:
            visible = np.ones(256, dtype=bool)
            for code, rule in canvas.palette.items():
                visible[int(code)] = bool(rule.get('show', True))
            d2[~visible[np.asarray(classes)[canvas.indices]]] = np.inf
        best = int(np.argmin(d2))
        if d2[best] > (8. * canvas.world_per_pixel())**2:
            return None
        return int(canvas.indices[best])

    def pick_main(self, view, pos):
        from gui.lnt_section_bridge import get_store
        from gui.native_main_classify import MainCanvas
        from gui.native_vector_overlay import _backend
        canvas = MainCanvas(view)
        store = get_store(self.app)
        if store is not None:
            # Query exact source records in a small plan-view neighbourhood.
            wx, wy = canvas.section_at(pos.x(), pos.y())
            radius = max(canvas.world_per_pixel() * 8., .01)
            if not store.ensure_region((wx-radius, wy-radius, -1e12),
                                       (wx+radius, wy+radius, 1e12)):
                return None
            rows = np.asarray(store.last_region_indices, dtype=np.int64)
            data = store.data
        else:
            data = getattr(self.app, 'data', None) or {}
            rows = None
        xyz = data.get('xyz')
        if xyz is None or not len(xyz):
            return None
        backend = _backend(self.app)
        mvp = backend.get_last_mvp()
        if mvp is None:
            return None
        origin = np.asarray(backend.get_render_origin_xyz(), dtype=np.float64)
        ratio = canvas.devicePixelRatioF()
        width, height = canvas.width()*ratio, canvas.height()*ratio
        from gui.class_slots import stored_palette
        visible = np.ones(256, dtype=bool)
        for code, rule in stored_palette(self.app, getattr(self.app, 'class_palette', {}) or {}).items():
            visible[int(code)] = bool(rule.get('show', True))
        classes = data.get('classification')
        count = len(xyz) if rows is None else len(rows)
        plan_box = None
        matrix = np.asarray(mvp)
        if np.all(np.abs(matrix[:2, 2]) < 1e-12) and np.all(np.abs(matrix[3, :3]) < 1e-12):
            wx, wy = canvas.section_at(pos.x(), pos.y())
            radius = canvas.world_per_pixel() * 8.
            plan_box = (wx-radius, wx+radius, wy-radius, wy+radius)
        best_row, best_distance, best_depth = None, 64., np.inf
        # Bound temporary memory even for full in-memory survey clouds.
        for start in range(0, count, 262144):
            ids = np.arange(start, min(start+262144, count)) if rows is None else rows[start:start+262144]
            points = np.asarray(xyz)[ids]
            if plan_box is not None:
                x0, x1, y0, y1 = plan_box
                near = ((points[:, 0] >= x0) & (points[:, 0] <= x1) &
                        (points[:, 1] >= y0) & (points[:, 1] <= y1))
                ids, points = ids[near], points[near]
                if not len(ids):
                    continue
            points = points - origin
            clip = points @ np.asarray(mvp)[:, :3].T + np.asarray(mvp)[:, 3]
            with np.errstate(divide='ignore', invalid='ignore'):
                ndc = clip[:, :3] / clip[:, 3, None]
            distance = (((ndc[:, 0]*.5+.5)*width/ratio-pos.x())**2 +
                        ((ndc[:, 1]*.5+.5)*height/ratio-pos.y())**2)
            valid = (clip[:, 3] > 0) & np.all(np.isfinite(ndc), axis=1) & (ndc[:, 2] >= 0) & (ndc[:, 2] <= 1)
            if classes is not None:
                valid &= visible[np.asarray(classes)[ids]]
            candidates = np.flatnonzero(valid & (distance <= best_distance))
            if not len(candidates):
                continue
            order = np.lexsort((ndc[candidates, 2], distance[candidates]))
            i = candidates[order[0]]
            if distance[i] < best_distance or ndc[i, 2] < best_depth:
                best_row, best_distance, best_depth = int(ids[i]), float(distance[i]), float(ndc[i, 2])
        return best_row

    def _follow(self):
        if self.tool.selected_global_index is None:
            return
        from gui.native_vector_overlay import _backend
        backend = _backend(self.app)
        mvp = backend.get_last_mvp() if backend is not None else None
        sections = getattr(getattr(self.app, '_native_sections', None), 'views', {})
        view = getattr(self.app, '_lnt_main_view', None) or getattr(self.app, '_snt_viewport_view', None)
        widget = getattr(getattr(view, 'owner', None), 'vulkan_widget', None)
        signature = (None if mvp is None else np.asarray(mvp).tobytes(),
                     None if widget is None else (widget.width(), widget.height(), widget.devicePixelRatioF()),
                     getattr(self.app, 'classification_revision', 0),
                     tuple((slot, tuple(c.centre), c.scale, c.width(), c.height(), c.isVisible())
                           for slot, c in sections.items()))
        if signature != self._camera_signature:
            self._camera_signature = signature
            self.highlight(self.tool.selected_global_index, update_info=False)

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

    def highlight(self, row, update_info=True):
        from gui.lnt_section_bridge import get_store
        from gui.native_main_classify import MainCanvas
        from gui.native_vector_overlay import vector_layer_for, group_entities
        store = get_store(self.app)
        data = store.data if store is not None else (getattr(self.app, 'data', None) or {})
        if data.get('xyz') is None or row < 0 or row >= len(data['xyz']):
            return False
        xyz = np.asarray(data['xyz'])[row]
        from gui.class_slots import code_for
        code = code_for(self.app, int(data['classification'][row]))   # user class number
        from gui.native_vector_overlay import _view_for
        view = _view_for(self.app)
        if view is None:
            return False
        self.tool.selected_global_index = int(row)
        layer = vector_layer_for(self.app)
        if layer is not None and (getattr(self.app, 'class_palette', {}) or {}).get(code, {}).get('show', True):
            radius = MainCanvas(view).world_per_pixel() * 11.
            layer.put_entities('target', '__point_target__', [self.ring(xyz, radius)])
            layer._backend.request_render()
            self._layer = layer
        elif self._layer is not None:
            self._layer.remove('target', '__point_target__')
            self._layer._backend.request_render()
        for backend in self._section_backends:
            backend.vector_clear_draw_ranges()
            backend.request_render()
        self._section_backends.clear()
        for canvas in getattr(getattr(self.app, '_native_sections', None), 'views', {}).values():
            if not canvas.palette.get(code, {}).get('show', True):
                continue
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
        if not update_info:
            return True
        # Display information without the legacy class-highlight actor path.
        info = getattr(self.app, 'identification_tool', None)
        name = info.get_class_name(code) if info is not None else str(code)
        if info is not None:
            info._update_ribbon_info(code, name, tuple(float(v) for v in xyz), point_index=int(row))
            info.point_identified.emit(code, name, tuple(float(v) for v in xyz))
        label = getattr(self.app, 'snt_layer_pick_footer_label', None)
        if label is not None:
            label.setText(f'Class: {code} : {name}')
            label.setToolTip(f'Point #{row:,}  X: {xyz[0]:.4f}  Y: {xyz[1]:.4f}  Z: {xyz[2]:.4f}')
            label.setVisible(True)
        self.app.statusBar().showMessage(f'Point #{row:,} | Class {code}: {name} | X {xyz[0]:.4f}  Y {xyz[1]:.4f}  Z {xyz[2]:.4f}', 8000)
        return True
