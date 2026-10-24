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
    COLOR = (255, 165, 0)
    LINEWEIGHT = 2

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
                'layer': self.LAYER, 'color': self.COLOR,
                '_lineweight': self.LINEWEIGHT,
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


class VulkanTempFencePreview(VulkanSelectionPreview):
    """Yellow temporary-classification fence on the native Vulkan canvas."""
    COLOR = (255, 255, 0)
    LINEWEIGHT = 3

    def __init__(self, view, name):
        self.NAME = str(name)
        self.LAYER = str(name)
        super().__init__(view)
