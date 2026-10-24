"""ctypes binding for native/lnt/lnt_view.dll (the .lnt converter + streaming viewer).

All view calls must be made on the GUI thread that created the view.
"""
import ctypes
import os
from ctypes import (POINTER, Structure, c_char_p, c_double, c_float, c_int,
                    c_uint32, c_uint64, c_void_p, create_string_buffer)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LNT_DIR = os.path.join(ROOT, "native", "lnt")
# Viewer API in main_shading.dll; GPU Shading uses the compact Vulkan engine below.
# Use the camera-preset build. Keeping a distinct deployed filename also lets a
# corrected renderer be installed while an older main_shading.dll is still
# mapped by a running Windows process.
LNT_DLL = os.path.join(_LNT_DIR, "main_display_modes_elevation.dll")
_VULKAN_DIR = os.path.join(ROOT, "native", "naksha_vulkan", "build_msvc")
# Main and section renderers share the same engine module/device pool.
VULKAN_DLL = os.path.join(_VULKAN_DIR, "naksha_vulkan.dll")

MODE_CLASS, MODE_INTENSITY, MODE_ELEVATION, MODE_NEUTRAL, MODE_DEPTH = 1, 2, 3, 5, 6

# Wheel zoom geometry: one notch scales the 2D-plan distance by this fraction
# (engine parity - native/lnt/lnt_view_api.cpp lnt_view_zoom asks for
# -goal_distance * 0.12 per step, i.e. a 1 - 0.12*steps distance factor).
ZOOM_STEP_FRACTION = 0.12

# Presentation modes of lnt_view_set_shading (LNTShadingModes.h). NOT the engine's
# MODE_* numbers above: these are the shading API's own enumeration.
SHADE_RGB, SHADE_INTENSITY, SHADE_ELEVATION, SHADE_CLASSIFICATION, SHADE_NORMAL = 0, 1, 2, 3, 4
RAMP_GRAYSCALE, RAMP_HUE, RAMP_BLUE_RED, RAMP_LAND_TOPO = 0, 1, 2, 3


class ShadingUniforms(Structure):
    """Mirror of lnt::ShadingUniforms - exactly 40 bytes."""
    _pack_ = 1
    _fields_ = [("display_mode", c_uint32), ("color_ramp", c_uint32),
                ("elevation_min", c_float), ("elevation_max", c_float),
                ("intensity_min", c_float), ("intensity_max", c_float),
                ("light_direction", c_float * 3), ("edl_strength", c_float)]


assert ctypes.sizeof(ShadingUniforms) == 40


class LntViewStats(Structure):
    _fields_ = [("active_nodes", c_uint32), ("points_rendered", c_uint32),
                ("points_wanted", c_uint32), ("pending_loads", c_uint32),
                ("evictions", c_uint32), ("_pad", c_uint32),
                ("cache_used_bytes", c_uint64), ("cache_capacity_bytes", c_uint64),
                ("frame_ms", c_double), ("update_ms", c_double),
                ("sse", c_double), ("distance", c_double)]


_lib = None


def lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.isfile(LNT_DLL):
        raise RuntimeError(f"missing {LNT_DLL} (build it with native/lnt/build_lnt_view.bat)")
    d = ctypes.CDLL(LNT_DLL)
    d.lnt_convert_las.argtypes = [c_char_p, c_char_p, c_char_p, c_int,
                                  POINTER(c_uint64), POINTER(c_uint64)]
    d.lnt_convert_las.restype = c_int
    d.lnt_view_open.argtypes = [c_void_p, c_int, c_int, c_char_p, c_char_p,
                                c_uint32, c_double, c_char_p, c_int]
    d.lnt_view_open.restype = c_void_p
    d.lnt_view_open_on_window.argtypes = [c_void_p, c_int, c_int, c_char_p, c_char_p,
                                          c_uint32, c_double, c_char_p, c_int]
    d.lnt_view_open_on_window.restype = c_void_p
    d.lnt_view_attach.argtypes = [c_uint64, c_int, c_int, c_char_p, c_char_p,
                                  c_uint32, c_double, c_char_p, c_int]
    d.lnt_view_attach.restype = c_void_p
    if hasattr(d, "lnt_view_request_redraw"):
        d.lnt_view_request_redraw.argtypes = [c_void_p]
        d.lnt_view_request_redraw.restype = None
    d.lnt_view_close.argtypes = [c_void_p]
    d.lnt_view_resize.argtypes = [c_void_p, c_int, c_int]
    d.lnt_view_resize.restype = c_int
    for n in ("lnt_view_orbit", "lnt_view_pan"):
        getattr(d, n).argtypes = [c_void_p, c_double, c_double]
    for n in ("lnt_view_zoom", "lnt_view_set_sse"):
        getattr(d, n).argtypes = [c_void_p, c_double]
    d.lnt_view_set_mode.argtypes = [c_void_p, c_int]
    d.lnt_view_reset.argtypes = [c_void_p]
    if hasattr(d, "lnt_view_renderer_handle"):    # the view's own nkv renderer (vector overlay)
        d.lnt_view_renderer_handle.argtypes = [c_void_p]
        d.lnt_view_renderer_handle.restype = c_uint64
    if hasattr(d, "lnt_view_plan_view"):          # absent in older builds of the DLL
        d.lnt_view_plan_view.argtypes = [c_void_p]
    if hasattr(d, "lnt_view_set_luts"):
        d.lnt_view_set_luts.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p]
        d.lnt_view_set_luts.restype = c_int
    if hasattr(d, "lnt_view_set_classification_lut"):
        d.lnt_view_set_classification_lut.argtypes = [c_void_p, c_void_p]
        d.lnt_view_set_classification_lut.restype = c_int
    if hasattr(d, "lnt_view_set_class_weights"):   # per-class point sizes (Weight column)
        d.lnt_view_set_class_weights.argtypes = [c_void_p, c_void_p]
        d.lnt_view_set_class_weights.restype = c_int
    if hasattr(d, "lnt_view_set_2d"):
        d.lnt_view_set_2d.argtypes = [c_void_p, c_int]
    if hasattr(d, "lnt_view_set_view"):
        d.lnt_view_set_view.argtypes = [c_void_p, c_int]
    if hasattr(d, "lnt_query_aabb"):             # section tools: point query + class write-back
        d.lnt_query_aabb.argtypes = [c_void_p, c_void_p, c_void_p, c_uint64, c_char_p, c_int]
        d.lnt_query_aabb.restype = ctypes.c_int64
        d.lnt_query_fetch.argtypes = [c_void_p] * 7
        d.lnt_query_fetch.restype = c_int
    if hasattr(d, "lnt_query_stroke"):           # brush stroke: only the swept records
        d.lnt_query_stroke.argtypes = [c_void_p, c_void_p, c_uint64, ctypes.c_double, c_int, c_uint64,
                                       c_char_p, c_int]
        d.lnt_query_stroke.restype = ctypes.c_int64
        d.lnt_query_release.argtypes = [c_void_p]
        d.lnt_set_classes.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_uint64,
                                      c_char_p, c_int]
        d.lnt_set_classes.restype = c_int
    if hasattr(d, "lnt_preview_classes"):        # instant class display before the write
        d.lnt_preview_classes.argtypes = [c_void_p] * 6 + [c_uint64, c_void_p]
        d.lnt_preview_classes.restype = c_int
        d.lnt_view_screen_to_world.argtypes = [c_void_p, c_double, c_double, c_void_p]
        d.lnt_view_screen_to_world.restype = c_int
        d.lnt_view_world_to_screen.argtypes = [c_void_p, c_double, c_double, c_void_p]
        d.lnt_view_world_to_screen.restype = c_int
        d.lnt_view_frame_region.argtypes = [c_void_p, c_double, c_double, c_double]
    if hasattr(d, "lnt_view_surface_method"):
        d.lnt_view_surface_method.argtypes = [c_void_p]
        d.lnt_view_surface_method.restype = c_int
    if hasattr(d, "lnt_view_surface_error"):
        d.lnt_view_surface_error.argtypes = [c_void_p]
        d.lnt_view_surface_error.restype = c_char_p
    if hasattr(d, "lnt_view_surface_building"):
        d.lnt_view_surface_building.argtypes = [c_void_p]
        d.lnt_view_surface_building.restype = c_int
    if hasattr(d, "lnt_view_surface_point_counts"):
        d.lnt_view_surface_point_counts.argtypes = [c_void_p, POINTER(c_uint64)]
        d.lnt_view_surface_point_counts.restype = c_int
    if hasattr(d, "lnt_view_surface_lod_stats"):     # persistent viewport surface LOD
        d.lnt_view_surface_lod_stats.argtypes = [c_void_p, POINTER(c_uint64), c_int]
        d.lnt_view_surface_lod_stats.restype = c_int
        d.lnt_view_surface_lod_active.argtypes = [c_void_p]
        d.lnt_view_surface_lod_active.restype = c_int
        d.lnt_view_surface_persist_message.argtypes = [c_void_p]
        d.lnt_view_surface_persist_message.restype = c_char_p
        d.lnt_view_set_surface_lod_px.argtypes = [c_void_p, c_double]
        d.lnt_view_set_surface_lod_px.restype = c_int
        d.lnt_view_surface_flush.argtypes = [c_void_p]
        d.lnt_view_surface_flush.restype = c_int
    if hasattr(d, "lnt_view_set_surface_elevation"):
        d.lnt_view_set_surface_elevation.argtypes = [c_void_p, c_int, ctypes.c_uint32,
                                                     c_double, c_double]
        d.lnt_view_set_surface_elevation.restype = c_int
    if hasattr(d, "lnt_view_set_surface_shading"):   # crisp shaded-class lighting (VTK parity)
        d.lnt_view_set_surface_shading.argtypes = [c_void_p, c_double, c_double, c_double,
                                                   c_double]
        d.lnt_view_set_surface_shading.restype = c_int
    if hasattr(d, "lnt_view_set_surface_single_elevation"):
        d.lnt_view_set_surface_single_elevation.argtypes = [c_void_p, c_double]
        d.lnt_view_set_surface_single_elevation.restype = c_int
    if hasattr(d, "lnt_view_show_external_surface"):
        d.lnt_view_show_external_surface.argtypes = [c_void_p]
        d.lnt_view_show_external_surface.restype = c_int
    if hasattr(d, "lnt_view_capture"):
        d.lnt_view_capture.argtypes = [c_void_p, c_void_p, c_uint64,
                                       POINTER(c_uint32), POINTER(c_uint32)]
        d.lnt_view_capture.restype = c_int
    if hasattr(d, "lnt_view_set_shading"):        # RGB / Intensity / Elevation / Class / Normal
        d.lnt_view_set_shading.argtypes = [c_void_p, c_void_p]
        d.lnt_view_set_shading.restype = c_int
        d.lnt_view_has_shading.argtypes = [c_void_p]
        d.lnt_view_has_shading.restype = c_int
        d.lnt_view_data_stats.argtypes = [c_void_p, c_void_p]
        d.lnt_view_data_stats.restype = c_int
    if hasattr(d, "lnt_view_set_point_budget"):   # upper bound on points drawn per frame
        d.lnt_view_set_point_budget.argtypes = [c_void_p, c_uint64]
    if hasattr(d, "lnt_view_set_clip_planes"):    # cross-section clip volume + corridor
        d.lnt_view_set_clip_planes.argtypes = [c_void_p, c_void_p, c_uint32]
        d.lnt_view_set_clip_planes.restype = c_int
        d.lnt_view_has_clip_volume.argtypes = [c_void_p]
        d.lnt_view_has_clip_volume.restype = c_int
        d.lnt_view_set_corridor.argtypes = [c_void_p, c_void_p, c_void_p]
        d.lnt_view_clear_corridor.argtypes = [c_void_p]
        d.lnt_view_corridor_node_count.argtypes = [c_void_p, c_void_p, c_void_p]
        d.lnt_view_corridor_node_count.restype = c_uint64
    d.lnt_view_frame.argtypes = [c_void_p, c_float, POINTER(LntViewStats)]
    d.lnt_view_frame.restype = c_int
    d.lnt_view_info.argtypes = [c_void_p, POINTER(c_uint64), POINTER(c_uint64), POINTER(c_double)]
    if hasattr(d, 'lnt_view_zoom_at'):
        d.lnt_view_zoom_at.argtypes = [c_void_p, c_double, c_double, c_double]
        d.lnt_view_zoom_at.restype = c_int
    _lib = d
    return d


def convert_las(las_path, lnt_path):
    """Native LAS -> .lnt. Returns (points, nodes); raises RuntimeError."""
    err = create_string_buffer(512)
    pts, nodes = c_uint64(0), c_uint64(0)
    ok = lib().lnt_convert_las(os.fsencode(las_path), os.fsencode(lnt_path), err, 512,
                               ctypes.byref(pts), ctypes.byref(nodes))
    if not ok:
        raise RuntimeError(err.value.decode("utf-8", "replace") or "native conversion failed")
    return pts.value, nodes.value


class NativeView:
    """One open .lnt viewer rendering into a child window of `parent_hwnd`."""

    def __init__(self, hwnd, width, height, lnt_path, budget_mb=512, sse=8.0):
        """Create the dedicated LNT renderer on `hwnd` (a Qt widget's native window).
        Main and section views share the engine's Vulkan device; each HWND
        has its own renderer handle."""
        err = create_string_buffer(512)
        self.class_writes = 0            # class records written into the document
        self.h = lib().lnt_view_open_on_window(c_void_p(int(hwnd)), int(width), int(height),
                                               os.fsencode(lnt_path), os.fsencode(VULKAN_DLL),
                                               int(budget_mb), float(sse), err, 512)
        if not self.h:
            raise RuntimeError(err.value.decode("utf-8", "replace") or "viewer open failed")
        self.stats = LntViewStats()
        self.last_error = ""
        # Device-pixel extent this renderer draws into (lnt_view_screen_to_world
        # and lnt_view_world_to_screen both work in this space). Needed by
        # zoom_at() to convert the anchor shift into a frame_region width.
        self._w, self._h = int(width), int(height)

    def close(self):
        if self.h:
            lib().lnt_view_close(c_void_p(self.h))
            self.h = None

    def renderer_handle(self):
        """The nkv renderer handle this view draws with (0 if unavailable). Same
        naksha_vulkan.dll module as the app backend, so nkv_* calls accept it."""
        fn = getattr(lib(), "lnt_view_renderer_handle", None)
        if not self.h or fn is None:
            return 0
        return int(fn(c_void_p(self.h)) or 0)

    def _ok(self):
        return bool(self.h)

    def resize(self, w, h):
        self._w, self._h = int(w), int(h)
        return self._ok() and lib().lnt_view_resize(c_void_p(self.h), int(w), int(h))

    def orbit(self, dx, dy):
        self._ok() and lib().lnt_view_orbit(c_void_p(self.h), dx, dy)

    def pan(self, dx, dy):
        self._ok() and lib().lnt_view_pan(c_void_p(self.h), dx, dy)

    def zoom(self, steps):
        self._ok() and lib().lnt_view_zoom(c_void_p(self.h), steps)

    def zoom_at(self, steps, px, py):
        """Zoom one wheel notch about the device-pixel point (px, py).

        Keeps the world point under the cursor fixed while the scale changes
        by the engine's own per-notch factor (ZOOM_STEP_FRACTION).

        Current engines anchor using the actual clamped zoom scale, including
        at maximum zoom. The code below also supports older engines:
        lnt_view_zoom only dollies about the view centre and
        eases distance toward a goal, so the anchor is emulated EXACTLY with
        the engine's own affine 2D-plan mapping: screen = centre +
        (world - target) * metres_per_px. Centre-dollying scales that factor
        by 1/k; re-centring through lnt_view_frame_region (an exact target +
        distance setter, no glide) then holds the cursor's world point in
        place: target = anchor - (cursor - centre) * new_metres_per_px.

        2D plan only - in 3D orbit the mapping APIs report no plan projection
        and the caller must fall back to the engine's centre zoom(). Returns
        False when the anchor could not be applied (no plan mapping, zero-size
        view, predated DLL, clamped extreme distances).
        """
        if not steps or not self._ok():
            return False
        anchored = getattr(lib(), 'lnt_view_zoom_at', None)
        if anchored is not None:
            return bool(anchored(c_void_p(self.h), float(steps), float(px), float(py)))
        anchor = self.screen_to_world(px, py)
        probe = self.screen_to_world(px + 1.0, py)
        if not anchor or not probe:
            return False                       # 3D orbit mode: no plan mapping
        m0 = abs(probe[0] - anchor[0])         # metres per device pixel now
        if m0 <= 0.0:
            return False
        k = max(0.05, 1.0 - ZOOM_STEP_FRACTION * float(steps))
        m1 = m0 * k
        w, h = float(getattr(self, "_w", 0) or 0), float(getattr(self, "_h", 0) or 0)
        if w < 1.0 or h < 1.0:
            return False
        self.frame_region(anchor[0] - (px - .5 * w) * m1,
                          anchor[1] + (py - .5 * h) * m1,
                          w * m1)
        # Prove the engine actually applied it: frame_region silently no-ops
        # on a DLL that predates it, and the engine clamps extreme distances.
        after = self.screen_to_world(px + 1.0, py)
        if not after or abs(abs(after[0] - anchor[0]) - m1) > m1 * 1e-6 + 1e-12:
            return False
        return True

    def set_sse(self, px):
        self._ok() and lib().lnt_view_set_sse(c_void_p(self.h), px)

    def set_mode(self, mode):
        self._ok() and lib().lnt_view_set_mode(c_void_p(self.h), int(mode))

    def set_lines(self, colors, visible):
        import numpy as np
        fn=getattr(lib(),'lnt_view_set_lines',None)
        if fn is None: return False
        fn.argtypes=[c_void_p,c_void_p,c_void_p]; fn.restype=c_int
        colors=np.ascontiguousarray(colors,dtype=np.uint8)
        visible=np.ascontiguousarray(visible,dtype=np.uint8)
        if colors.shape!=(65536,3) or visible.shape!=(65536,): raise ValueError('Invalid flight-line palette')
        return bool(fn(c_void_p(self.h),c_void_p(colors.ctypes.data),c_void_p(visible.ctypes.data)))

    def set_luts(self, class_rgb, elev_rgb, int_rgb):
        """Three (256,3) uint8 numpy tables: class palette, elevation ramp, intensity ramp."""
        if not (self._ok() and hasattr(lib(), "lnt_view_set_luts")):
            return False
        import numpy as np
        keep = [np.ascontiguousarray(a, dtype=np.uint8) if a is not None else None
                for a in (class_rgb, elev_rgb, int_rgb)]
        ptr = [c_void_p(a.ctypes.data) if a is not None else c_void_p(0) for a in keep]
        return bool(lib().lnt_view_set_luts(c_void_p(self.h), *ptr))

    def set_classification_lut(self, rgba):
        """Upload a (256,4) uint8 class color/visibility table."""
        if not (self._ok() and hasattr(lib(), "lnt_view_set_classification_lut")):
            return False
        import numpy as np
        table = np.ascontiguousarray(rgba, dtype=np.uint8)
        if table.shape != (256, 4):
            raise ValueError("classification LUT must have shape (256, 4)")
        return bool(lib().lnt_view_set_classification_lut(
            c_void_p(self.h), c_void_p(table.ctypes.data)))

    def set_class_weights(self, weights):
        """Upload a 256-float per-class point-size weight table (1.0 = neutral).

        Native form of the VTK weight_lut: gl_PointSize scales with the weight,
        per view. Uniform-only in the engine, so this never re-uploads a point.
        False when the DLL predates lnt_view_set_class_weights."""
        if not (self._ok() and hasattr(lib(), "lnt_view_set_class_weights")):
            return False
        import numpy as np
        table = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1)
        if table.size != 256:
            raise ValueError("class weights must have 256 entries")
        return bool(lib().lnt_view_set_class_weights(
            c_void_p(self.h), c_void_p(table.ctypes.data)))

    def set_2d(self, on):
        """2D plan (orthographic, orbit locked) or 3D perspective orbit."""
        if self._ok() and hasattr(lib(), "lnt_view_set_2d"):
            lib().lnt_view_set_2d(c_void_p(self.h), 1 if on else 0)

    def plan_view(self):
        """Top-down plan view, keeping target and distance (no-op on an old DLL)."""
        if self._ok() and hasattr(lib(), "lnt_view_plan_view"):
            lib().lnt_view_plan_view(c_void_p(self.h))

    def set_view(self, preset):
        """Native camera preset: 3d/top/front/side; never calls VTK."""
        values = {'3d': 0, 'top': 1, 'front': 2, 'side': 3}
        value = values.get(str(preset).lower())
        if value is None or not (self._ok() and hasattr(lib(), "lnt_view_set_view")):
            return False
        lib().lnt_view_set_view(c_void_p(self.h), value)
        return True

    # ---- section tools -----------------------------------------------------------
    def has_section_api(self):
        return self._ok() and hasattr(lib(), "lnt_query_aabb")

    def query_points(self, mn, mx, max_points=0):
        """Every leaf point inside the world box [mn, mx] (inclusive).

        Returns a dict of numpy arrays - xyz (n,3 float64), classification (uint8),
        intensity (uint8), rgb (n,3 uint8), node (uint32) and local (uint32), the last two
        being the point's identity for set_classes() - or None when the query failed or
        exceeded max_points (self.last_error says why)."""
        import numpy as np
        self.last_error = ""
        if not self.has_section_api():
            self.last_error = "this lnt_view DLL has no section API (rebuild native/lnt)"
            return None
        err = create_string_buffer(256)
        a, b = (c_double * 3)(*[float(v) for v in mn]), (c_double * 3)(*[float(v) for v in mx])
        n = lib().lnt_query_aabb(c_void_p(self.h), a, b, int(max_points), err, 256)
        return self._fetch_query(n, err)

    def has_stroke_query(self):
        return self._ok() and hasattr(lib(), "lnt_query_stroke")

    def query_stroke(self, path, radius, square, max_points=0):
        """Every leaf point a plan-view brush stroke covers: the round (or square) brush of
        `radius` swept along the XY `path`. Same result layout as query_points()."""
        import numpy as np
        self.last_error = ""
        if not self.has_stroke_query():
            self.last_error = "this lnt_view DLL has no stroke query (rebuild native/lnt)"
            return None
        path = np.ascontiguousarray(np.asarray(path, dtype=np.float64).reshape(-1, 2))
        err = create_string_buffer(256)
        n = lib().lnt_query_stroke(c_void_p(self.h), c_void_p(path.ctypes.data), len(path),
                                   float(radius), int(bool(square)), int(max_points), err, 256)
        return self._fetch_query(n, err)

    def _fetch_query(self, n, err):
        import numpy as np
        if n < 0:
            self.last_error = err.value.decode("utf-8", "replace") or (
                "too many points in the section" if n == -2 else "query failed")
            lib().lnt_query_release(c_void_p(self.h))
            return None
        out = {
            "xyz": np.empty((n, 3), dtype=np.float64),
            "classification": np.empty(n, dtype=np.uint8),
            "intensity": np.empty(n, dtype=np.uint8),
            "rgb": np.empty((n, 3), dtype=np.uint8),
            "node": np.empty(n, dtype=np.uint32),
            "local": np.empty(n, dtype=np.uint32),
        }
        if n:
            ok = lib().lnt_query_fetch(
                c_void_p(self.h), *[c_void_p(out[k].ctypes.data) for k in
                                    ("xyz", "classification", "intensity", "rgb", "node", "local")])
            if not ok:
                self.last_error = "fetching the section points failed"
                lib().lnt_query_release(c_void_p(self.h))
                return None
        lib().lnt_query_release(c_void_p(self.h))
        return out

    def set_classes(self, node, local, classification, old_classification=None):
        """Write classification values identified by ``node/local`` into the .lnt.

        ``old_classification`` is optional and is used only by the live class-count
        tracker.  Supplying the section store's known pre-write bytes avoids a
        second file lookup on the interactive classification path; the native write
        itself is unchanged.
        """
        import numpy as np
        self.last_error = ""
        if not self.has_section_api():
            self.last_error = "no section API"
            return False
        node = np.ascontiguousarray(node, dtype=np.uint32)
        local = np.ascontiguousarray(local, dtype=np.uint32)
        cls = np.ascontiguousarray(classification, dtype=np.uint8)
        if not (len(node) == len(local) == len(cls)):
            self.last_error = "array length mismatch"
            return False
        if len(node) == 0:
            return True
        # Live class histogram (gui/lnt_class_counts.py): old bytes before, delta after.
        counts = getattr(self, "class_counts", None)
        token = (counts.before_write(node, local, cls, old_classification)
                 if counts is not None else None)
        err = create_string_buffer(256)
        ok = lib().lnt_set_classes(c_void_p(self.h), c_void_p(node.ctypes.data),
                                   c_void_p(local.ctypes.data), c_void_p(cls.ctypes.data),
                                   len(node), err, 256)
        if not ok:
            self.last_error = err.value.decode("utf-8", "replace") or "write failed"
        else:
            # How many class records this session wrote into the document. The
            # backup uses it to tell "there is work to save" from "nothing was
            # classified", so a close never writes a pointless full copy.
            self.class_writes = int(getattr(self, "class_writes", 0) or 0) + len(node)
        if counts is not None:
            counts.after_write(token, bool(ok))
        return bool(ok)

    def preview_classes(self, node, local, xyz, old_class, new_class):
        """Show class edits at once, before set_classes() writes them: the resident surface
        tiles are recoloured in place (no re-meshing) and point tiles re-uploaded. Returns
        (tiles, vertices, coverage, rebuilt) or None when the DLL predates it."""
        import numpy as np
        if not (self._ok() and hasattr(lib(), "lnt_preview_classes")):
            return None
        node = np.ascontiguousarray(node, dtype=np.uint32)
        local = np.ascontiguousarray(local, dtype=np.uint32)
        xyz = np.ascontiguousarray(xyz, dtype=np.float64)
        old_class = np.ascontiguousarray(old_class, dtype=np.uint8)
        new_class = np.ascontiguousarray(new_class, dtype=np.uint8)
        n = len(node)
        if not (len(local) == len(old_class) == len(new_class) == n and xyz.shape == (n, 3)):
            raise ValueError("preview_classes: array length mismatch")
        stats = (ctypes.c_uint32 * 4)()
        ok = lib().lnt_preview_classes(c_void_p(self.h), c_void_p(node.ctypes.data),
                                       c_void_p(local.ctypes.data), c_void_p(xyz.ctypes.data),
                                       c_void_p(old_class.ctypes.data), c_void_p(new_class.ctypes.data),
                                       n, stats)
        return tuple(stats) if ok else None

    def screen_to_world(self, px, py):
        """Device-pixel window position -> world (x, y, z); None in 3D orbit mode."""
        if not (self._ok() and hasattr(lib(), "lnt_view_screen_to_world")):
            return None
        out = (c_double * 3)()
        return tuple(out) if lib().lnt_view_screen_to_world(c_void_p(self.h), px, py, out) else None

    def world_to_screen(self, x, y):
        if not (self._ok() and hasattr(lib(), "lnt_view_world_to_screen")):
            return None
        out = (c_double * 2)()
        return tuple(out) if lib().lnt_view_world_to_screen(c_void_p(self.h), x, y, out) else None

    def frame_region(self, cx, cy, width_m):
        if self._ok() and hasattr(lib(), "lnt_view_frame_region"):
            lib().lnt_view_frame_region(c_void_p(self.h), cx, cy, width_m)

    # ---- shading modes (all on the GPU; no VTK) ---------------------------------------
    def has_shading(self):
        return (self._ok() and hasattr(lib(), "lnt_view_set_shading")
                and bool(lib().lnt_view_has_shading(c_void_p(self.h))))

    def set_shading(self, mode, ramp=RAMP_GRAYSCALE, z_range=(0.0, 1.0), intensity_range=(0.0, 1.0),
                    light=(0.5, -0.5, -0.7071), edl=0.0):
        """Switch the presentation: mode is a SHADE_* value, ramp a RAMP_* value, z_range the
        world Z the Elevation ramp spans, intensity_range fractions 0..1 of the stored
        intensity, light the direction the light travels, edl the Eye-Dome Lighting strength.
        Shaded mode also requests a native surface mesh. False on an engine DLL without the API."""
        if not self.has_shading():
            return False
        u = ShadingUniforms(int(mode), int(ramp), float(z_range[0]), float(z_range[1]),
                            float(intensity_range[0]), float(intensity_range[1]),
                            (c_float * 3)(*[float(v) for v in light]), float(edl))
        return bool(lib().lnt_view_set_shading(c_void_p(self.h), ctypes.byref(u)))

    def surface_method(self):
        return lib().lnt_view_surface_method(c_void_p(self.h)) if self._ok() and hasattr(lib(), "lnt_view_surface_method") else 0

    def surface_error(self):
        if self._ok() and hasattr(lib(), "lnt_view_surface_error"):
            return (lib().lnt_view_surface_error(c_void_p(self.h)) or b"").decode("utf-8", "replace")
        return ""

    def surface_building(self):
        return (self._ok() and hasattr(lib(), "lnt_view_surface_building")
                and bool(lib().lnt_view_surface_building(c_void_p(self.h))))

    def surface_point_counts(self):
        counts = (c_uint64 * 3)()
        if self._ok() and hasattr(lib(), "lnt_view_surface_point_counts"):
            lib().lnt_view_surface_point_counts(c_void_p(self.h), counts)
        return tuple(counts)

    # Field order of lnt_view_surface_lod_stats (LNTSurfaceLod.h LodStat).
    SURFACE_LOD_STATS = (
        "drawn_tiles", "drawn_exact_tiles", "source_covered", "source_exact", "represented_drawn",
        "triangles_drawn", "coverage_drawn", "gpu_tiles", "gpu_vertices", "cpu_tiles", "cpu_bytes",
        "queued", "inflight", "built", "loaded", "uploaded", "gpu_evicted", "cpu_evicted",
        "load_failures", "archive_tiles", "commits", "archive_bytes", "total_points", "wanted_missing")

    def surface_lod_active(self):
        """True while Shading is drawn from persistent viewport-LOD surface tiles."""
        return (self._ok() and hasattr(lib(), "lnt_view_surface_lod_active")
                and bool(lib().lnt_view_surface_lod_active(c_void_p(self.h))))

    def surface_lod_stats(self):
        """dict of SURFACE_LOD_STATS, or None before the LOD path was used. source_covered
        counts original records inside the drawn tiles (coarse tiles represent them through
        the hierarchy); source_exact those drawn at full resolution; represented_drawn the
        records/representatives that are actually mesh vertices or coverage points."""
        if not (self._ok() and hasattr(lib(), "lnt_view_surface_lod_stats")):
            return None
        out = (c_uint64 * len(self.SURFACE_LOD_STATS))()
        n = lib().lnt_view_surface_lod_stats(c_void_p(self.h), out, len(out))
        return dict(zip(self.SURFACE_LOD_STATS[:n], list(out)[:n])) if n else None

    def surface_persist_message(self):
        """A new surface-archive save problem (e.g. access denied), or ""."""
        if self._ok() and hasattr(lib(), "lnt_view_surface_persist_message"):
            return (lib().lnt_view_surface_persist_message(c_void_p(self.h)) or b"").decode("utf-8", "replace")
        return ""

    def set_surface_elevation(self, on, ramp=RAMP_HUE, z_range=(0.0, 1.0)):
        """Surface feature: the Shading mesh coloured by elevation. Call right after
        set_shading(SHADE_NORMAL). False on a DLL without the call."""
        if not (self._ok() and hasattr(lib(), "lnt_view_set_surface_elevation")):
            return False
        return bool(lib().lnt_view_set_surface_elevation(
            c_void_p(self.h), 1 if on else 0, int(ramp), float(z_range[0]), float(z_range[1])))

    def set_surface_shading(self, azimuth_deg, sharpness, ambient, light_elevation_deg=0.0):
        """The Shading mesh's hillshade controls, exactly as the VTK Shaded-Classification
        view derives them (_configure_microstation_color_blend_lighting): azimuth is the
        compass direction the light comes FROM, sharpness the raw 0..999 Sharpness slider,
        ambient the brightness / shadow floor, light_elevation_deg <= 0 keeps the default sun
        elevation. Geometry is untouched: with a crisp mesh resident only the lighting
        changes. False on a DLL without the call (the mesh keeps its CPU-baked shade)."""
        if not (self._ok() and hasattr(lib(), "lnt_view_set_surface_shading")):
            return False
        return bool(lib().lnt_view_set_surface_shading(
            c_void_p(self.h), float(azimuth_deg), float(sharpness), float(ambient),
            float(light_elevation_deg)))

    def set_surface_single_elevation(self, elevation_deg):
        """Light elevation for a cut with ONE visible class (app.last_shade_angle): that case
        is the plain hillshade baked into flat cell colours, as in the VTK view."""
        if not (self._ok() and hasattr(lib(), "lnt_view_set_surface_single_elevation")):
            return False
        return bool(lib().lnt_view_set_surface_single_elevation(
            c_void_p(self.h), float(elevation_deg)))

    def show_external_surface(self):
        """Present a mesh uploaded through the shared Vulkan renderer handle."""
        if not (self._ok() and hasattr(lib(), "lnt_view_show_external_surface")):
            return False
        return bool(lib().lnt_view_show_external_surface(c_void_p(self.h)))

    def capture(self):
        """The engine's current frame as an (h, w, 4) uint8 array, or None."""
        if not (self._ok() and hasattr(lib(), "lnt_view_capture")):
            return None
        import numpy as np
        w, h = c_uint32(0), c_uint32(0)
        if not lib().lnt_view_capture(c_void_p(self.h), None, 0, ctypes.byref(w), ctypes.byref(h)):
            return None
        buf = (ctypes.c_uint8 * (w.value * h.value * 4))()
        if not lib().lnt_view_capture(c_void_p(self.h), buf, len(buf), ctypes.byref(w), ctypes.byref(h)):
            return None
        return np.frombuffer(buf, dtype=np.uint8).reshape(h.value, w.value, 4).copy()

    def data_stats(self):
        """Value ranges sampled from the root LOD node: dict with z_min, z_max, z_p1, z_p99,
        i_min, i_max - or None."""
        if not (self._ok() and hasattr(lib(), "lnt_view_data_stats")):
            return None
        out = (c_double * 6)()
        if not lib().lnt_view_data_stats(c_void_p(self.h), out):
            return None
        return dict(zip(("z_min", "z_max", "z_p1", "z_p99", "i_min", "i_max"), tuple(out)))

    def set_point_budget(self, points):
        """Cap on the points the cut may hold (0 = unlimited). Lower = faster, coarser."""
        if self._ok() and hasattr(lib(), "lnt_view_set_point_budget"):
            lib().lnt_view_set_point_budget(c_void_p(self.h), int(points))

    def set_upload_budget_ms(self, ms):
        """Per-frame GPU upload time slice in ms (0 = unlimited; the view loop defaults to 1 ms)."""
        if self._ok() and hasattr(lib(), "lnt_view_set_upload_budget_ms"):
            lib().lnt_view_set_upload_budget_ms(c_void_p(self.h), c_double(ms))

    def set_clip_planes(self, planes):
        """GPU cross-section clip volume: up to 4 world-space (A, B, C, D) planes; a point
        is kept where A*x + B*y + C*z + D >= 0. None / empty switches clipping off.
        Returns False when the engine DLL has no nkv_set_clip_volume yet."""
        if not (self._ok() and hasattr(lib(), "lnt_view_set_clip_planes")):
            return False
        import numpy as np
        arr = np.ascontiguousarray(np.asarray(planes if planes is not None else [], dtype=np.float64)
                                   .reshape(-1, 4)[:4])
        ptr = c_void_p(arr.ctypes.data) if len(arr) else c_void_p(0)
        return bool(lib().lnt_view_set_clip_planes(c_void_p(self.h), ptr, len(arr)))

    def set_corridor(self, mn, mx):
        """Restrict streaming to a world AABB: nodes outside it are never loaded."""
        if self._ok() and hasattr(lib(), "lnt_view_set_corridor"):
            a, b = (c_double * 3)(*mn), (c_double * 3)(*mx)
            lib().lnt_view_set_corridor(c_void_p(self.h), a, b)

    def clear_corridor(self):
        if self._ok() and hasattr(lib(), "lnt_view_clear_corridor"):
            lib().lnt_view_clear_corridor(c_void_p(self.h))

    def reset(self):
        self._ok() and lib().lnt_view_reset(c_void_p(self.h))

    def request_redraw(self):
        fn = getattr(lib(), "lnt_view_request_redraw", None)
        if self._ok() and fn is not None:
            fn(c_void_p(self.h))

    def frame(self, dt):
        return self._ok() and bool(lib().lnt_view_frame(c_void_p(self.h), float(dt),
                                                         ctypes.byref(self.stats)))

    def info(self):
        pts, nodes, ext = c_uint64(), c_uint64(), (c_double * 3)()
        lib().lnt_view_info(c_void_p(self.h), ctypes.byref(pts), ctypes.byref(nodes), ext)
        return pts.value, nodes.value, tuple(ext)
