"""Native 2D main-view classification using exact LNT source records."""
from types import SimpleNamespace
import numpy as np

from gui.lnt_section_bridge import get_store
from gui.native_section_classify import SectionClassifier


class MainCanvas:
    line_tools = False      # Above/Below/Parallel Line are cross/cut-section tools

    def __init__(self, view):
        self.view = view
        self.widget = view.owner.vulkan_widget
        self.owner = SimpleNamespace(app=view.app, refresh_edits=self.refresh_edits)
        self.axis = 0
        self.indices = np.empty(0, dtype=np.int64)
        self.local = np.empty((0, 3), dtype=np.float64)
        self._selection_points = np.empty((0, 2), dtype=np.float64)
        self._closed = False
        self._region = None
        self._store = None
        self.exact_selection = None   # tool whose shape the last native query already applied

    def __getattr__(self, name):
        return getattr(self.widget, name)

    @property
    def palette(self):
        return getattr(self.view.app, 'class_palette', {})

    @property
    def frame_backend(self):
        from gui.native_vector_overlay import _backend
        return _backend(self.view.app) or getattr(self.view.owner, 'vulkan_backend', None)

    def section_at(self, x, y):
        ratio = self.widget.devicePixelRatioF()
        world = self.view.native.screen_to_world(x*ratio, y*ratio)
        if world is None:
            raise RuntimeError('Native screen-to-world mapping failed')
        return world[0], world[1]

    def screen_from_section(self, x, y):
        pixel = self.view.native.world_to_screen(x, y)
        if pixel is None:
            return -1.e9, -1.e9
        ratio = self.widget.devicePixelRatioF()
        return pixel[0]/ratio, pixel[1]/ratio

    def projection_snapshot(self):
        # The controller can already contain the next pan/zoom target. Use
        # the same rendered MVP as the point cloud, once for the whole paint.
        from gui.native_vector_overlay import _MvpScreenMapping
        mapping = getattr(self, '_frame_mapping', None)
        if mapping is None:
            mapping = self._frame_mapping = _MvpScreenMapping(SimpleNamespace(
                vulkan_backend=self.frame_backend, vulkan_widget=self.widget))
        snapshot = mapping.projection_snapshot()
        if snapshot is None:
            return None
        key, project = snapshot
        ratio = self.widget.devicePixelRatioF()
        def logical(x, y):
            px, py = project(x, y)
            return px / ratio, py / ratio
        return (key, ratio), logical

    def world_projection_snapshot(self, z):
        """SNT labels use their displayed height; point tools keep their own plane."""
        from gui.native_vector_overlay import _MvpScreenMapping, _backend
        owner = SimpleNamespace(vulkan_backend=_backend(self.view.app),
                                vulkan_widget=self.widget)
        mapping = _MvpScreenMapping(owner)
        snapshot = mapping.projection_snapshot(z)
        if snapshot is None:
            return None
        key, project = snapshot
        ratio = self.widget.devicePixelRatioF()
        return (key, ratio), lambda x, y: tuple(v / ratio for v in project(x, y))

    def world_per_pixel(self):
        a, b = self.section_at(0., 0.), self.section_at(1., 0.)
        return float(np.hypot(b[0]-a[0], b[1]-a[1]))

    def visible_extent(self):
        corners = np.array([self.section_at(x, y) for x, y in (
            (0, 0), (self.width(), 0), (0, self.height()), (self.width(), self.height()))])
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        return lo[0], hi[0], lo[1], hi[1]

    def prepare_selection(self, tool, shape, radius):
        self.exact_selection = None
        store = get_store(self.view.app)
        if tool == 'brush' and store is not None and self.view.native.has_stroke_query():
            # Native query_stroke already performs the exact swept-brush geometry.
            # Keep only its store-row identities here: building/copying an Nx2 XY
            # array for those same rows on every live brush batch is redundant and
            # becomes visible on very large clouds.
            from gui.native_section_classify import brush_is_square
            if not store.ensure_stroke(shape, radius, brush_is_square(self.view.app)):
                raise RuntimeError(store.last_error)
            self.indices = np.asarray(store.last_region_indices, dtype=np.int64)
            self._selection_points = np.empty((0, 2), dtype=np.float64)
            self._store = store
            self._region = None            # not a box: a later box tool queries again
            self.exact_selection = 'brush'
            return
        x0, x1, y0, y1 = self.visible_extent()
        lo, hi = np.array([x0, y0]), np.array([x1, y1])
        points = np.asarray(shape, dtype=np.float64)
        if tool in ('brush', 'point'):
            lo = np.maximum(lo, points.min(axis=0)-radius)
            hi = np.minimum(hi, points.max(axis=0)+radius)
        elif tool == 'circle':
            distance = float(np.linalg.norm(points[-1]-points[0]))
            lo = np.maximum(lo, points[0]-distance)
            hi = np.minimum(hi, points[0]+distance)
        elif tool in ('rectangle', 'polygon', 'freehand'):
            lo = np.maximum(lo, points.min(axis=0))
            hi = np.minimum(hi, points.max(axis=0))
        if np.any(lo > hi):
            self.indices = np.empty(0, dtype=np.int64)
            self.local = np.empty((0, 3), dtype=np.float64)
            self._selection_points = np.empty((0, 2), dtype=np.float64)
            self._region = None
            return
        if store is self._store and self._region is not None and np.all(lo >= self._region[0]) and np.all(hi <= self._region[1]):
            return
        if store is None or not store.ensure_region(
                (*np.nextafter(lo, -np.inf), -1.e12), (*np.nextafter(hi, np.inf), 1.e12)):
            raise RuntimeError(store.last_error if store else 'Native classification store unavailable')
        self._take_rows(store)
        self._region = lo.copy(), hi.copy()

    def _take_rows(self, store):
        self.indices = store.last_region_indices.copy()
        # Main-view classification only needs plan X/Y.  Building a temporary
        # Nx3 XYZ array, another reordered Nx3 array, then an Nx2 selection
        # array made three full copies for every large gesture.
        xyz = store.data['xyz']
        self._selection_points = np.empty((len(self.indices), 2), dtype=np.float64)
        np.take(xyz[:, 0], self.indices, out=self._selection_points[:, 0])
        np.take(xyz[:, 1], self.indices, out=self._selection_points[:, 1])
        self._store = store

    def selection_points(self):
        return self._selection_points

    def refresh_edits(self, force=False, changed=None):
        store = get_store(self.view.app)
        if store is not None:
            if changed is not None:
                store.show(*changed)         # recolour now; sparse write is debounced
            else:
                # A render refresh is not an explicit Save.  Never turn it into a
                # full multi-million-row class diff on the GUI thread.
                store.flush(force_scan=False)
        sections = getattr(self.view.app, '_native_sections', None)
        if sections is not None:
            # Preserve the exact changed-row payload so every open cross/cut view
            # can use its sparse update path.  The old call dropped it and forced
            # a full classification-array gather/compare in every section.
            sections.refresh_edits(force=True, changed=changed, show_main=False)


def ensure_classifier(view):
    if not view.mode_2d or view.owner is None:
        return None
    if getattr(view, 'classifier', None) is None:
        view.classifier = SectionClassifier(MainCanvas(view))
    return view.classifier
