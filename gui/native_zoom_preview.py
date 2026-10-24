"""Zoom's five-vertex rectangle in the shared Vulkan vector arena.

No QWidget, Qt painting, event filter or follow timer is created here. The
existing LNT render loop presents the rectangle using the cloud's camera.
"""
import numpy as np

from gui.native_vector_overlay import vector_layer_for, _cloud_z_bounds, RESTART_INDEX


class VulkanZoomPreview:
    KIND = '__tool_preview__'
    NAME = '__zoom_rectangle__'
    LAYER = '__zoom_rectangle__'

    def __init__(self, view):
        self.view = view
        self.layer = vector_layer_for(view.app, gpu_only=True)
        if self.layer is None:
            raise RuntimeError('The Vulkan vector pass is unavailable for Zoom')
        self.key = (self.KIND, self.NAME)

    def set_rectangle(self, start, end):
        if start is None or end is None:
            self.clear()
            return
        a, b = start, end
        bounds = _cloud_z_bounds(self.view.app)
        z = bounds[1] + max(50., bounds[1] - bounds[0]) if bounds else 0.
        xyz = np.array([(a[0], a[1], z), (b[0], a[1], z),
                        (b[0], b[1], z), (a[0], b[1], z),
                        (a[0], a[1], z)], dtype=np.float64)
        self._set_points(xyz)

    def _set_points(self, xyz):
        if not np.isfinite(xyz).all():
            raise ValueError('Tool preview contains nonfinite coordinates')
        document = self.layer.documents.get(self.key)
        if (document is None or
                document['gpu'][self.LAYER][0]['v_count'] != len(xyz)):
            result = self.layer.put_entities(self.KIND, self.NAME, [{
                'type': 'polyline', 'points': xyz.tolist(),
                'layer': self.LAYER, 'color': (255, 165, 0), '_lineweight': 2,
            }])
            document = self.layer.documents.get(self.key)
            if not result or document is None or not document['gpu'].get(self.LAYER):
                raise RuntimeError('Vulkan Zoom rectangle upload failed')
        else:
            # Reuse the allocated five vertices, including after arena compaction.
            # The document retains CPU data for the normal renderer rebind path.
            chunks = document['gpu'][self.LAYER]
            chunk = chunks[0]
            indices = chunk['indices'].copy()
            indices[indices != RESTART_INDEX] += np.uint32(chunk['v_first'])
            if not self.layer._backend.vector_upload_strip(
                    chunk['v_first'], chunk['v_count'], xyz, chunk['rgb'],
                    chunk['i_first'], chunk['i_count'], indices):
                raise RuntimeError('Vulkan Zoom rectangle update failed')
            chunk['xyz'] = xyz
        document['handles'][self.LAYER]._visible = True
        self.layer._push_draw_ranges()

    def clear(self):
        document = self.layer.documents.get(self.key)
        if document is not None:
            document['handles'][self.LAYER]._visible = False
            self.layer._push_draw_ranges()

    def shutdown(self):
        self.clear()


class VulkanSelectionPreview(VulkanZoomPreview):
    NAME = '__point_selection__'
    LAYER = '__point_selection__'

    def set_polygon(self, vertices, hover=None):
        points = list(vertices)
        if hover is not None and (not points or tuple(hover) != tuple(points[-1])):
            points.append(hover)
        if len(points) < 2:
            self.clear()
            return
        if len(points) >= 3:
            points.append(points[0])
        bounds = _cloud_z_bounds(self.view.app)
        z = bounds[1] + max(50., bounds[1] - bounds[0]) if bounds else 0.
        xyz = np.asarray([(p[0], p[1], z) for p in points], dtype=np.float64)
        # Grow geometrically. Repeated trailing vertices are degenerate segments
        # and draw nothing, so mouse motion reuses the same GPU strip.
        capacity = max(8, 1 << (len(xyz) - 1).bit_length())
        if len(xyz) < capacity:
            xyz = np.vstack([xyz, np.repeat(xyz[-1:], capacity - len(xyz), axis=0)])
        self._set_points(xyz)

    def set_selected_points(self, points):
        """Exact XY highlights, bounded uploads through the shared Vulkan arena."""
        name = self.NAME + '_points'
        self.layer.remove_file(self.KIND, name)
        points = np.asarray(points, dtype=np.float64)
        if not len(points):
            return
        bounds = _cloud_z_bounds(self.view.app)
        z = bounds[1] + max(50., bounds[1] - bounds[0]) if bounds else 0.
        from gui.native_main_classify import MainCanvas
        canvas = MainCanvas(self.view)
        x0, x1, y0, y1 = canvas.visible_extent()
        radius = max(x1 - x0, y1 - y0) / 600.
        chunks = []
        for at in range(0, len(points), 65536):
            p = points[at:at + 65536]
            xyz = np.repeat(p, 4, axis=0)
            xyz[:, 2] = z
            xyz[0::4, 0] -= radius
            xyz[1::4, 0] += radius
            xyz[2::4, 1] -= radius
            xyz[3::4, 1] += radius
            indices = np.empty((len(p) * 2, 3), dtype=np.uint32)
            indices[:, :2] = np.arange(len(p) * 4, dtype=np.uint32).reshape(-1, 2)
            indices[:, 2] = RESTART_INDEX
            chunks.append({'xyz': xyz, 'rgb': np.tile(np.array([255, 255, 0], dtype=np.uint8), (len(xyz), 1)),
                           'indices': indices.ravel(), 'v_count': len(xyz),
                           'i_count': indices.size, 'lineweight': 2})
        if not self.layer.put_entities(self.KIND, name, [], prepared_groups={self.LAYER: chunks}):
            raise RuntimeError('Vulkan selection highlight upload failed')

    def clear(self):
        super().clear()
        self.layer.remove_file(self.KIND, self.NAME + '_points')

    def shutdown(self):
        self.clear()
        self.layer.remove_file(self.KIND, self.NAME)
