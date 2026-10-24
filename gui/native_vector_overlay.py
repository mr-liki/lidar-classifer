"""SNT/DXF CAD linework on the native Vulkan main viewport.

WHY THIS EXISTS
    In native-main mode the visible overlay linework went through the
    transparent Qt overlay (gui/native_section_classify.py's CanvasOverlay):
    every entity was re-projected world-then-screen and re-stroked with
    QPainter on a 33 ms repaint. That is fine for a drawing tool's own shapes
    and completely untenable for an SNT attach - the reference N4_BLOCK_932.snt
    has 1,125,237 line entities, and a per-frame QPainter pass over them cannot
    keep up.

    This module uploads that linework to the engine's vector arena ONCE
    (nkv_vector_reserve + nkv_vector_upload_strip) and lets the engine draw it
    (~one draw call per layer/colour group, no per-frame CPU cost). Layer and
    file visibility become DRAW-RANGE toggles, never re-uploads.

LABELS
    World-sized CAD text is flattened into glyph contours at attachment time
    and uploaded alongside the linework. Both use the same Vulkan camera and
    presentation frame. Text without an authored height uses the Qt overlay.

WHAT IS *NOT* CHANGED
    The entity dicts stay the source of truth for picking, digitizing, saving
    (snt_core.direct_write), PRJ matching and block polygons. Only the GPU
    buffers move. The handle surface (SetVisibility / GetVisibility / GetBounds /
    Modified) is inherited from gui/native_attachments.NativeAttachmentHandle,
    so the Layers widgets, identify and block-click code need no change.

    Without the vector entry points in the loaded DLL - or when the pass could
    not be built - `vector_overlay_available()` is False and the caller keeps the
    existing Qt-overlay linework path untouched.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from gui.native_attachments import (
    NativeAttachmentHandle,
    NativeAttachmentLayer,
    _entity_record,
    is_native_main,
)

# Primitive-restart index: separates consecutive line strips inside one group so
# the whole group is a single vkCmdDrawIndexed. Must match
# VectorOverlayRenderer::kRestartIndex.
RESTART_INDEX = 0xFFFFFFFF

# Polylines per upload call. Mirrors the VTK branch's
# NAKSHA_SNT_LINE_CHUNK_POINTS (2,000,000) so a chunk boundary lands in the same
# place on both paths and an upload call stays a bounded staging write.
_LINE_CHUNK_POINTS_DEFAULT = 2_000_000

# Fallback marker arm length (world metres) for 'point' entities when the caller
# supplies no size hint. A point has no extent of its own, so it is drawn as a
# small cross; callers that know the view scale pass marker_m explicitly.
_MARKER_M_DEFAULT = 0.1

# SNT lineweight -> pixel width, the same formula the VTK line actor uses
# (gui/snt_attachment._build_line_actor: pw = 1.0 + lw * 0.5, else 1.0).
_LINE_WIDTH_BASE = 1.0
_LINE_WIDTH_PER_WEIGHT = 0.5
# Fallback ceiling when the loaded DLL predates the Phase-0
# VkPhysicalDeviceLimits.lineWidthRange probe (the T400 reports [1.0, 64.0]).
_LINE_WIDTH_MAX_PX = 64.0


def _env_int(name: str, default: int, minimum: int = 1000) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _backend(app):
    """The live VulkanRenderBackend for the main viewport, or None.

    In an .lnt session Naksha's own renderer is shut down and the LNT view draws
    with a dedicated nkv renderer, so the linework must go to THAT renderer."""
    view = getattr(app, "_lnt_main_view", None)
    if view is not None and getattr(view, "native", None) is not None:
        return lnt_vector_backend(view)
    owner = getattr(app, "render_backend", None)
    return getattr(owner, "vulkan_backend", None)


def lnt_vector_backend(view):
    """A VulkanRenderBackend bound to an .lnt view's own nkv renderer.

    The LNT engine loads the same naksha_vulkan.dll module as the app backend,
    so the app backend's ctypes wrappers work unchanged on the LNT handle, and
    nkv_render (which the LNT view calls every frame) records the vector pass.
    Presentation, resize and destruction stay with the LNT view."""
    native = getattr(view, "native", None)
    getter = getattr(native, "renderer_handle", None)
    owner = getattr(view, "owner", None) or getattr(getattr(view, "app", None), "render_backend", None)
    base = getattr(owner, "vulkan_backend", None)
    if getter is None or base is None or getattr(base, "_dll", None) is None:
        return None
    try:
        handle = int(getter() or 0)
    except Exception:
        handle = 0
    if not handle:
        return None
    proxy = getattr(view, "_vector_backend", None)
    if proxy is None or getattr(proxy, "_handle", 0) != handle:
        proxy = copy.copy(base)
        proxy._handle = handle
        import weakref
        proxy._frame_overlays = weakref.WeakSet()
        for attr in ("_nkv_line_width_range", "_nkv_line_width_logged"):
            proxy.__dict__.pop(attr, None)
        proxy.request_render = lambda *a, **k: None   # the LNT view's timer presents
        proxy.resize = lambda *a, **k: None
        proxy.shutdown = lambda *a, **k: None          # the LNT view destroys its renderer
        try:
            view._vector_backend = proxy
        except Exception:
            pass
    return proxy


# ---------------------------------------------------------------------------
# Phase-0 probe: VkPhysicalDeviceLimits.lineWidthRange
# ---------------------------------------------------------------------------

def line_width_range(backend):
    """(min, max) pixel line width the DEVICE accepts, probed once per backend.

    This is the Phase-0 question the whole width design hinges on. `wideLines`
    being supported only means >1.0 is *allowed*; the authoritative answer is
    limits.lineWidthRange, and many GPUs report [1,1], which forces 1 px lines in
    core Vulkan. The engine has already forced max to 1.0 when the wideLines
    feature is not enabled, so `max == 1.0` reads as "thick strokes are
    impossible on this device" and a thicker line would need CPU ribbon
    tessellation (quads) rather than vkCmdSetLineWidth.

    The probe result is cached on the backend object, so it costs one DLL call
    per renderer and never a per-frame query. A DLL without the probe reports
    the historical ceiling instead of failing.
    """
    if backend is None:
        return (1.0, _LINE_WIDTH_MAX_PX)
    cached = getattr(backend, "_nkv_line_width_range", None)
    if cached is not None:
        return cached
    raw = None
    getter = getattr(backend, "get_line_width_range", None)
    if getter is not None:
        try:
            raw = getter()
        except Exception:
            raw = None
    if raw and len(raw) == 2 and float(raw[1]) > 0.0:
        lo = max(1.0, float(raw[0]))
        hi = float(raw[1])
        rng = (lo, hi if hi >= lo else lo)
    else:
        rng = (1.0, _LINE_WIDTH_MAX_PX)
    try:
        backend._nkv_line_width_range = rng
    except Exception:
        pass
    return rng


def wide_lines_available(backend) -> bool:
    """True when the device can draw a stroke thicker than 1 px.

    False means every group draws 1 px regardless of its authored lineweight -
    the honest outcome on a [1,1] device, and the trigger for the CPU ribbon
    fallback the spec reserves for that case.
    """
    return line_width_range(backend)[1] > 1.0


def _log_line_width_probe(backend, rng) -> None:
    """Phase-0 probe line, once per renderer, matching this project's [TAG] logs.

    `max == 1.0` is the [1,1] case: the GPU cannot draw a thick line in core
    Vulkan, so lineweights degrade to 1 px and thicker strokes would need CPU
    ribbon tessellation. `max > 1.0` is the case the per-range dynamic width
    path covers, and it is the T400's answer ([1.0, 64.0]).
    """
    if backend is None:
        return
    if getattr(backend, "_nkv_line_width_logged", False):
        return
    try:
        backend._nkv_line_width_logged = True
    except Exception:
        pass
    low, high = rng
    if high > 1.0:
        print(f"[VECTOR OVERLAY] lineWidthRange=[{low:.1f}, {high:.1f}] wideLines=YES "
              f"- per-layer-group widths via dynamic vkCmdSetLineWidth", flush=True)
    else:
        print(f"[VECTOR OVERLAY] lineWidthRange=[{low:.1f}, {high:.1f}] wideLines=NO "
              f"- thick strokes fall back to 1 px (ribbon tessellation would be "
              f"required for heavier lineweights)", flush=True)


def vector_overlay_available(app) -> bool:
    """True when the engine can draw the vector overlay for this app.

    Requires a native main viewport the engine can own (an .lnt session, or an
    SNT-only session where §5 promotion is allowed), plus a backend whose loaded
    DLL built the vector pass; anything else returns False and the caller keeps
    linework on the Qt/VTK overlay rather than silently drawing nothing.
    """
    if not snt_viewport_capable(app):
        return False
    backend = _backend(app)
    if backend is None:
        return False
    try:
        return bool(backend.supports_vector_overlay())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# §5 hard dependency: who owns the visible canvas
# ---------------------------------------------------------------------------

def _owner(app):
    """The app's AppRenderBackendOwner - it owns the widget and the viewport."""
    return getattr(app, "render_backend", None)


def engine_can_draw_vectors(app) -> bool:
    """True when the loaded engine was built with the vector pass at all.

    Deliberately independent of who owns the canvas: this answers only "can the
    DLL draw lines", so callers can ask before paying for a promotion attempt.
    """
    owner = _owner(app)
    if owner is None or not getattr(owner, "active", False):
        return False
    backend = getattr(owner, "vulkan_backend", None)
    if backend is None:
        return False
    try:
        return bool(backend.supports_vector_overlay())
    except Exception:
        return False


def snt_viewport_active(app) -> bool:
    """The engine's surface IS the visible canvas right now.

    Either an .lnt session installed it (is_native_main), or this process
    promoted it for an SNT attach and has not handed it back yet.
    """
    if is_native_main(app):
        return True
    owner = _owner(app)
    return bool(owner is not None and getattr(owner, "_viewport_installed", False)
                and engine_can_draw_vectors(app))


def snt_viewport_capable(app) -> bool:
    """Whether an SNT attach SHOULD move the visible canvas to the engine.

    §5 states that SNT-on-Vulkan is only visible when the Vulkan main viewport
    is installed, and asks Phase 1 to install it in SNT-attach mode. Two reasons
    to decline and let the untouched VTK path draw instead:

      * the engine cannot draw vectors (no pass / no renderer / backend off), so
        promoting would swap in a canvas that shows nothing;
      * a VTK session with data is on screen: install_as_main_viewport() switches
        VTK's LiDAR actors off, so the loaded cloud would vanish. That attach
        keeps its current canvas and its current linework, unchanged.

    An .lnt session is always capable - it owns the canvas already.
    """
    if is_native_main(app):
        return True
    if not engine_can_draw_vectors(app):
        return False
    if getattr(app, "data", None):
        return False       # a VTK-rendered cloud would be switched off
    return getattr(_owner(app), "vulkan_widget", None) is not None


def promote_snt_viewport(app) -> bool:
    """Make the engine's surface the visible canvas for this SNT attach (§5).

    Idempotent, and a no-op inside an .lnt session (which owns the viewport and
    whose camera we must not touch). install_as_main_viewport() does the real
    work: raise the native child over the interactor cell, resize the swapchain
    to it, install the VTK->rig camera mirror, and switch VTK's LiDAR actors off
    (its render loop, overlays and Qt UI keep running).
    """
    if is_native_main(app):
        return True
    if not snt_viewport_capable(app):
        return False
    if snt_viewport_active(app):
        return True            # already promoted by an earlier attach
    owner = _owner(app)
    if owner is None:
        return False
    if not owner.vulkan_viewer_host_is_ready():
        print("[SNT vectors] viewport promotion refused: viewer host not ready; "
              "linework stays on the VTK/Qt overlay")
        return False
    if not owner.install_as_main_viewport():
        print("[SNT vectors] viewport promotion refused: install failed; "
              "linework stays on the VTK/Qt overlay")
        return False
    app._snt_viewport_promoted = True
    backend = _backend(app)
    try:
        if backend is not None:
            backend.request_render()   # first frame, so the MVP is real for labels
    except Exception:
        pass
    print("[SNT vectors] Vulkan main viewport installed for the SNT attach "
          "(central viewport cell only; VTK render loop, overlays and Qt UI "
          "keep working)")
    return True


def demote_snt_viewport(app) -> None:
    """Hand the canvas back to VTK when we were the ones who took it.

    The mirror image of promote_snt_viewport(): an .lnt session is never touched
    (it owns its viewport), and nothing happens unless we promoted, so a failed
    or emptied SNT attach ends in exactly the state it started in - VTK visible,
    LiDAR actors on, camera sync left as LntMainView.stop() leaves it.
    """
    if is_native_main(app) or not getattr(app, "_snt_viewport_promoted", False):
        return
    app._snt_viewport_promoted = False
    owner = _owner(app)
    if owner is None:
        return
    try:
        widget = getattr(owner, "vulkan_widget", None)
        if widget is not None:
            widget.hide()
        owner._viewport_installed = False
        owner.set_vtk_lidar_rendering(True)
        print("[SNT vectors] main viewport returned to VTK (no SNT linework left)")
    except Exception as exc:
        print(f"[SNT vectors] could not restore the VTK viewport: {exc}")


class _MvpScreenMapping:
    """world <-> framebuffer pixels, read from the engine's own MVP.

    In an .lnt session MainCanvas gets this from the C++ NativeView. An
    SNT-only promoted session has no point-cloud stream, so the mapping comes
    from nkv_get_last_mvp - the matrix the linework was rasterised with, re-read
    on every call. Labels therefore land on their entities and follow
    pan/zoom/resize exactly like the lines they belong to.

    Screen coordinates are the ENGINE's framebuffer pixels (what
    nkv_capture_frame returns), which is what the NativeView returns too -
    MainCanvas divides by devicePixelRatioF() to reach Qt's logical space.
    """

    def __init__(self, owner):
        self._owner = owner

    def _backend(self):
        return getattr(self._owner, "vulkan_backend", None)

    def _size(self):
        widget = getattr(self._owner, "vulkan_widget", None)
        if widget is None:
            backend = self._backend()
            ext = getattr(backend, "_last_extent", None) if backend is not None else None
            if ext and ext[0] > 0 and ext[1] > 0:
                return int(ext[0]), int(ext[1])
            return 0, 0
        try:
            ratio = float(widget.devicePixelRatioF() or 1.0)
        except Exception:
            ratio = 1.0
        # _last_extent is the requested resize, which may be logical pixels or
        # temporarily clamped by Win32. Qt labels need the live widget's physical
        # extent; MainCanvas converts it back to logical coordinates exactly once.
        return max(1, int(widget.width() * ratio)), max(1, int(widget.height() * ratio))

    @staticmethod
    def _plane_z(origin) -> float:
        """World Z the 2-D section mapping samples at.

        MainCanvas gives only (x, y) - the main view is a plan view, so what is
        needed is where that (x, y) sits in the frame. Under the top-down
        orthographic camera the SNT auto-fit installs, Z does not move the
        screen XY at all; using the pinned render origin's Z (which _pin_origin
        set from the linework's own bounds) keeps the depth sane if the user
        orbits instead.
        """
        try:
            return float(origin[2])
        except Exception:
            return 0.0

    def _mvp_and_origin(self):
        backend = self._backend()
        if backend is None:
            return None, None
        getter = getattr(backend, "get_last_mvp", None)
        if getter is None:
            return None, None
        mvp = getter()
        if mvp is None:
            return None, None
        try:
            origin = np.asarray(backend.get_render_origin_xyz(), dtype=np.float64)
        except Exception:
            origin = np.zeros(3, dtype=np.float64)
        return np.array(mvp, dtype=np.float64, copy=True), origin.copy()

    def world_to_screen(self, x, y):
        """(px, py) in framebuffer pixels, or None before the first frame."""
        mvp, origin = self._mvp_and_origin()
        w, h = self._size()
        if mvp is None or not w or not h:
            return None
        z = self._plane_z(origin)
        clip = mvp @ np.array([float(x) - origin[0], float(y) - origin[1],
                               z - origin[2], 1.0], dtype=np.float64)
        if not np.all(np.isfinite(clip)) or abs(float(clip[3])) < 1e-12:
            return None
        sx = (float(clip[0]) / float(clip[3]) * 0.5 + 0.5) * w
        sy = (float(clip[1]) / float(clip[3]) * 0.5 + 0.5) * h
        if not (np.isfinite(sx) and np.isfinite(sy)):
            return None
        return sx, sy

    def projection_snapshot(self, world_z=None):
        """One immutable frame projection shared by every label in a paint."""
        backend = self._backend()
        frames = getattr(backend, 'present_count', getattr(backend, 'render_count', None))
        size = self._size()
        token = (frames, size, world_z) if isinstance(frames, int) else None
        if token is not None and token == getattr(self, '_snapshot_token', None):
            return self._snapshot_value
        mvp, origin = self._mvp_and_origin()
        w, h = size
        if mvp is None or not w or not h:
            return None
        key = (mvp.tobytes(), origin.tobytes(), w, h, world_z)
        rz = 0.0 if world_z is None else float(world_z) - origin[2]
        def project(x, y):
            # Plan-view labels lie on the render-origin plane.
            rx, ry = float(x) - origin[0], float(y) - origin[1]
            cw = mvp[3, 0] * rx + mvp[3, 1] * ry + mvp[3, 2] * rz + mvp[3, 3]
            if not np.isfinite(cw) or abs(cw) < 1e-12:
                return -1.e9, -1.e9
            return ((mvp[0, 0] * rx + mvp[0, 1] * ry + mvp[0, 2] * rz + mvp[0, 3]) / cw * .5 + .5) * w, \
                   ((mvp[1, 0] * rx + mvp[1, 1] * ry + mvp[1, 2] * rz + mvp[1, 3]) / cw * .5 + .5) * h
        value = key, project
        if token is not None:
            self._snapshot_token, self._snapshot_value = token, value
        return value

    def screen_to_world(self, px, py):
        """[x, y, z] on the plan-view plane, or None when it cannot be solved."""
        mvp, origin = self._mvp_and_origin()
        w, h = self._size()
        if mvp is None or not w or not h:
            return None
        try:
            inv = np.linalg.inv(mvp)
        except np.linalg.LinAlgError:
            return None
        nx = (float(px) / w) * 2.0 - 1.0
        ny = (float(py) / h) * 2.0 - 1.0

        def unproject(nz):
            # Vulkan NDC depth is 0..1 (near..far); any two distinct samples lie
            # on the same view ray, so this is the ray through this pixel.
            q = inv @ np.array([nx, ny, float(nz), 1.0], dtype=np.float64)
            if not np.all(np.isfinite(q)) or abs(float(q[3])) < 1e-12:
                return None
            return q[:3] / float(q[3]) + origin     # render space -> world

        p0, p1 = unproject(0.0), unproject(1.0)
        if p0 is None or p1 is None:
            return None
        direction = p1 - p0
        if abs(float(direction[2])) < 1e-12:
            return None          # ray parallel to the plane
        target = self._plane_z(origin)
        hit = p0 + (target - float(p0[2])) / float(direction[2]) * direction
        if not np.all(np.isfinite(hit)):
            return None
        return [float(hit[0]), float(hit[1]), float(hit[2])]

    def has_stroke_query(self):
        """No point-cloud stream here, so no exact brush query (MainCanvas falls
        back to a region query, which reports the store as unavailable - the same
        degradation an .lnt-less session already had for classification."""
        return False


class _SntViewportView:
    """The promoted viewport as a 'view' for an SNT-only session (no .lnt).

    NativeAttachmentLayer needs a view to hang its classifier and canvas on -
    that canvas is what hosts the label overlay - and MainCanvas needs a
    world<->screen mapping from `view.native`. This supplies exactly those two
    things from the engine and nothing point-cloud specific, so anything that
    would need the stream (classification store, stroke query) reports
    unavailable instead of crashing on an absent NativeView.
    """

    mode_2d = True
    picker = None

    def __init__(self, app, owner):
        self.app = app
        self.owner = owner
        self.name = "SNT viewport"
        self.classifier = None
        self.native = _MvpScreenMapping(owner)
        self._closed = False


def _view_for(app):
    """The view the overlay layer attaches to.

    An .lnt session: its own view, unchanged. Otherwise the promoted-viewport
    view, created once per app so a second SNT attach reuses the same classifier
    - and therefore the same label-overlay window - instead of stacking a new one.
    """
    view = getattr(app, "_lnt_main_view", None)
    if view is not None and getattr(view, "native", None) is not None:
        return view
    owner = _owner(app)
    if owner is None:
        return None
    view = getattr(app, "_snt_viewport_view", None)
    if view is None or getattr(view, "owner", None) is not owner:
        view = _SntViewportView(app, owner)
        try:
            app._snt_viewport_view = view
        except Exception:
            pass
    return view


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _points3(entity: Dict, close_ring: bool = True) -> Optional[np.ndarray]:
    """Entity vertices as float64 (N,3), closing a ring when the entity says so.

    Closed polylines repeat their first vertex, which is the same convention the
    VTK _build_line_actor path used - a closed strip without the repeat renders
    with one missing edge.
    """
    pts = entity.get("points")
    if not pts:
        pts = entity.get("vertices")
    if not pts:
        return None
    try:
        a = np.asarray(pts, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 2:
        return None
    if a.shape[1] < 3:
        a = np.hstack([a, np.zeros((a.shape[0], 1), dtype=np.float64)])
    elif a.shape[1] > 3:
        a = a[:, :3]
    if close_ring and bool(entity.get("closed")) and a.shape[0] >= 3:
        try:
            # Coordinates are survey eastings/northings: relative tolerance
            # would incorrectly close endpoints metres apart. Avoid a NumPy
            # reduction allocation for each of a million short polylines.
            already = all(abs(float(x) - float(y)) <= 1e-6
                          for x, y in zip(a[0], a[-1]))
        except Exception:
            already = False
        if not already:
            a = np.vstack([a, a[0]])
    return np.ascontiguousarray(a)


def _marker_strips(xyz: np.ndarray, arm: float) -> List[np.ndarray]:
    """A small '+' for each marker point: two 2-vertex line strips per point.

    A line strip cannot rasterise a single vertex, and a zero-length segment
    rasterises nothing, so a marker is drawn as two short segments. The arm
    length is in WORLD units because the pass draws world-space geometry.
    """
    if xyz.size == 0 or arm <= 0.0:
        return []
    out: List[np.ndarray] = []
    for p in xyz:
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        out.append(np.array([[x - arm, y, z], [x + arm, y, z]], dtype=np.float64))
        out.append(np.array([[x, y - arm, z], [x, y + arm, z]], dtype=np.float64))
    return out


def _pack_strips(strips: List[np.ndarray], rgb: np.ndarray):
    """Flatten strips into one vertex array + one restart-separated index array.

    Returns (xyz, rgb, indices) where indices are LOCAL to this chunk (the
    uploader rebases them onto the arena offset) and a RESTART separates each
    strip from the next.
    """
    strips = [s for s in strips if s is not None and len(s) >= 2]
    if not strips:
        return None
    total = sum(len(s) for s in strips)
    xyz = np.ascontiguousarray(np.concatenate(strips, axis=0), dtype=np.float64)
    color = np.empty((total, 3), dtype=np.uint8)
    lengths = np.fromiter((len(s) for s in strips), dtype=np.int64,
                          count=len(strips))
    # Allocate once rather than two tiny arrays per strip plus concatenate.
    ends = np.cumsum(lengths + 1) - 1
    indices = np.arange(total + len(strips), dtype=np.uint32)
    indices -= np.repeat(np.arange(len(strips), dtype=np.uint32), lengths + 1)
    indices[ends] = RESTART_INDEX
    indices = indices[:-1]
    color[:] = rgb
    return xyz, color, indices


def _entity_rgb(entity: Dict) -> np.ndarray:
    """Authored RGB byte for one entity, defaulting to white like the VTK path."""
    values = entity.get("color")
    if values is None:
        values = (255, 255, 255)
    try:
        rgb = [int(round(float(v))) for v in list(values)[:3]]
    except (TypeError, ValueError):
        rgb = [255, 255, 255]
    while len(rgb) < 3:
        rgb.append(255)
    return np.array([max(0, min(255, c)) for c in rgb], dtype=np.uint8)


def group_entities(entities, marker_m: float = _MARKER_M_DEFAULT, is_cancelled=None):
    """Groups linework into per-(colour, layer, lineweight) chunk lists.

    Returns {layer_name: [chunk, ...]} where each chunk is
        {'xyz', 'rgb', 'indices', 'v_count', 'i_count', 'lineweight'}
    and several chunks per layer appear when a layer carries more than one
    (colour, lineweight) group or when a group exceeds the upload chunk cap
    (the arena stays contiguous across them). Because line width is dynamic
    per-draw-range state, each chunk is drawn at its own authored lineweight.
    Text is deliberately not collected here - it stays on the Qt overlay.
    """
    chunk_points = _env_int("NAKSHA_SNT_LINE_CHUNK_POINTS", _LINE_CHUNK_POINTS_DEFAULT)
    groups: Dict[str, List[Dict]] = {}
    pending: Dict[tuple, Dict] = {}
    color_cache = {}

    def flush(key):
        state = pending.pop(key, None)
        if state is not None:
            groups.setdefault(key[0], []).append(state["chunk"])
        return state
    for entity_index, entity in enumerate(entities or ()):
        if (entity_index & 0x3FFF) == 0 and is_cancelled is not None and is_cancelled():
            from gui.snt_attachment import _SntCancelled
            raise _SntCancelled()
        try:
            etype = str(entity.get("type", "")).strip().lower()
        except Exception:
            continue
        if etype in ("line", "polyline", "lwpolyline", "polygon"):
            strips = [_points3(entity)]
        elif etype == "3dface":
            # A 3D face is drawn as its closed outline, matching the VTK
            # branch's _build_face_outline_actor (render_face_outlines).
            strips = [_points3(entity, close_ring=True)]
        elif etype == "point":
            pos = entity.get("position") or entity.get("point")
            pts = np.asarray([pos], dtype=np.float64) if pos else np.empty((0, 3))
            strips = _marker_strips(pts.reshape(-1, 3), float(marker_m))
        else:
            continue   # text / anything else is not engine geometry

        strips = [s for s in strips if s is not None and len(s) >= 2]
        if not strips:
            continue

        layer = str(entity.get("layer", "0") or "0")
        raw_color = tuple(entity.get("color") or (255, 255, 255))
        rgb = color_cache.get(raw_color)
        if rgb is None:
            rgb = color_cache[raw_color] = _entity_rgb(entity)
        weight = int(entity.get("_lineweight", 0) or 0)
        # (colour, layer, lineweight) - the spec's grouping key. Line weight is
        # part of the key because line width is per-draw-range dynamic state, so
        # two lineweights inside one layer/colour become two groups = two draw
        # calls, each at its own authored width, instead of one width for the
        # whole file (which is what the single-global-width version did).
        key = (layer, raw_color, weight)

        state = pending.get(key)
        added = sum(len(s) for s in strips)
        if state is not None and state["chunk"]["v_count"] + added > chunk_points:
            flush(key)
            state = None
        if state is None:
            state = pending[key] = {
                "chunk": {"strips": [], "rgb": rgb, "v_count": 0,
                          "lineweight": weight, "layer": layer},
            }
        state["chunk"]["strips"].extend(strips)
        state["chunk"]["v_count"] += added

    for key in list(pending):
        flush(key)

    # Flatten each chunk's strips into the arena-ready arrays.
    out: Dict[str, List[Dict]] = {}
    for layer, chunks in groups.items():
        built: List[Dict] = []
        for chunk in chunks:
            if is_cancelled is not None and is_cancelled():
                from gui.snt_attachment import _SntCancelled
                raise _SntCancelled()
            packed = _pack_strips(chunk["strips"], chunk["rgb"])
            if packed is None:
                continue
            xyz, rgb, indices = packed
            built.append({
                "xyz": xyz, "rgb": rgb, "indices": indices,
                "v_count": int(xyz.shape[0]), "i_count": int(indices.shape[0]),
                "lineweight": int(chunk["lineweight"]),
            })
        if built:
            out[layer] = built
    return out


def _line_width_px(lineweight: int, rng=None) -> float:
    """Authored lineweight -> a pixel width the DEVICE will accept.

    Clamped into the probed lineWidthRange, because vkCmdSetLineWidth outside
    that range is an invalid command. On a [1,1] device every stroke therefore
    comes out 1 px: the correct degradation, not a silent invalid draw.
    """
    width = (_LINE_WIDTH_BASE + int(lineweight or 0) * _LINE_WIDTH_PER_WEIGHT)
    low, high = rng if rng else (1.0, _LINE_WIDTH_MAX_PX)
    return float(max(low, min(width, high)))


# ---------------------------------------------------------------------------
# Handles / layer
# ---------------------------------------------------------------------------

class VectorLayerHandle(NativeAttachmentHandle):
    """NativeAttachmentHandle whose visibility covers GPU linework too.

    The whole behaviour is inherited on purpose: SetVisibility() flips
    `_visible` and calls layer.refresh(), and NativeVectorLayer.refresh() pushes
    the draw ranges AND updates the Qt text overlay from that same flag. So the
    Layers widget, identify and block-click code keep working unmodified, and
    the two representations of "this layer is on" can never disagree.
    """

    def __init__(self, layer, document, layer_name, bounds, chunks):
        super().__init__(layer, document, layer_name, bounds)
        # Arena-resident chunks drawn by this layer, filled by put_entities().
        self.chunks: List[Dict] = list(chunks or [])

    @property
    def line_weight(self) -> int:
        """Widest authored lineweight in this layer (Qt-side report/debug only).

        A property, not a cached attribute: _reserve() swaps `chunks` when the
        arena is rebuilt, and a stale maximum would misreport the layer.
        """
        return max((int(c.get("lineweight", 0) or 0) for c in self.chunks),
                   default=0)


def _cloud_z_bounds(app):
    """Full-file bounds, independent of the currently streamed point LOD."""
    view = getattr(app, "_lnt_main_view", None)
    if view is not None and getattr(view, "native", None) is not None:
        cached = getattr(view, "_snt_cloud_z_bounds", None)
        if cached is not None:
            return cached
        try:
            import struct
            from gui.lnt_class_counts import read_directory
            with open(view.lnt_path, "rb") as stream:
                header = struct.unpack("<4sHHQ6d", stream.read(64))
            _, entries = read_directory(view.lnt_path)
            leaves = entries[(entries["children_mask"] == 0) & (entries["point_count"] > 0)]
            if not len(leaves):
                return None
            # The file header is a cubic octree envelope, often hundreds of
            # metres taller than the terrain. Populated leaf boxes give a tight,
            # conservative ceiling without decoding every source point or
            # trusting the root LOD sample (which can miss the tallest points).
            centres = leaves["center"][:, 2].astype(np.float64) + header[6]
            half = leaves["half_size"].astype(np.float64)
            bounds = float(np.min(centres - half)), float(np.max(centres + half))
            if np.all(np.isfinite(bounds)) and bounds[1] >= bounds[0]:
                view._snt_cloud_z_bounds = bounds
                return bounds
        except (OSError, AttributeError, ValueError, KeyError, struct.error):
            return None
        return None
    data = getattr(app, "data", None)
    if data is not None and "xyz" in data and len(data["xyz"]):
        bounds = getattr(app, "data_bounds", None)
        if bounds is not None and len(bounds) >= 6:
            z0, z1 = float(bounds[4]), float(bounds[5])
        else:
            z = np.asarray(data["xyz"])[:, 2]
            z0, z1 = float(np.nanmin(z)), float(np.nanmax(z))
        if np.isfinite(z0) and np.isfinite(z1) and z1 >= z0:
            return z0, z1
    return None


class NativeVectorLayer(NativeAttachmentLayer):
    """NativeAttachmentLayer whose polyline geometry lives in the GPU arena.

    `documents[key]['groups']` carries fallback Qt text records, and
    `documents[key]['gpu']` carries the linework and world-sized glyphs in
    arena-resident chunks per layer. Everything else (refresh, remove,
    remove_file, shutdown) is inherited.
    """

    def __init__(self, app):
        # The view may be an .lnt session's, or the promoted-viewport view an
        # SNT-only attach gets (§5); NativeAttachmentLayer takes it either way.
        super().__init__(app, view=_view_for(app))
        self._backend = _backend(app)
        self._v_used = 0
        self._i_used = 0
        self._line_width_px = _LINE_WIDTH_BASE
        self._origin_pinned = False
        self._width_range = line_width_range(self._backend)
        _log_line_width_probe(self._backend, self._width_range)

    # ---- upload ----------------------------------------------------------
    def put_entities(self, kind: str, name: str, entities, bounds=None, prepared_groups=None):
        """Uploads the linework and registers the Qt-overlay text records.

        Returns {layer_name: [handle]} - the same shape
        gui.native_attachments.render_entities returns, so the SNT caller's
        actor_cache handling is unchanged.
        """
        entities = list(entities or ())
        gpu_groups = prepared_groups if prepared_groups is not None else group_entities(entities)
        # World-sized CAD labels must share the linework's Vulkan camera frame.
        # A separate transparent window can present an older frame during pan.
        from gui.native_label_layout import native_text_entities
        glyph_entities = []
        text_grouped: Dict[str, List] = {}
        for entity in entities or ():
            try:
                if str(entity.get("type", "")).strip().lower() != "text":
                    continue
            except Exception:
                continue
            if entity.get("_naksha_label_font_size") == 0:
                continue
            glyphs = native_text_entities(entity)
            if glyphs is not None:
                glyph_entities.extend(glyphs)
                continue
            record = _entity_record(entity)
            if record is None:
                continue
            position = entity.get("position") or ()
            record[1]["source_z"] = float(position[2]) if len(position) > 2 else 0.0
            text_grouped.setdefault(
                str(entity.get("layer", "0") or "0"), []).append(record)

        if glyph_entities:
            # Copy the prepared map: it may also be retained by the attachment.
            gpu_groups = {layer: list(chunks) for layer, chunks in gpu_groups.items()}
            for layer, chunks in group_entities(glyph_entities).items():
                gpu_groups.setdefault(layer, []).extend(chunks)

        key = (str(kind).lower(), str(Path(name)))

        # Re-attaching a file REPLACES its document. The arena is append-only,
        # so the replaced file's chunks would otherwise stay reserved forever;
        # dropping it and forcing a compaction re-uploads every remaining
        # document at fresh offsets and reclaims the space.
        previous = self.documents.get(key, {})
        replacing = key in self.documents
        if replacing:
            self.documents.pop(key, None)

        self._pin_origin(gpu_groups)
        new_v = sum(c["v_count"] for chunks in gpu_groups.values() for c in chunks)
        new_i = sum(c["i_count"] for chunks in gpu_groups.values() for c in chunks)
        resident_v = sum(c["v_count"] for document in self.documents.values()
                         for chunks in (document.get("gpu") or {}).values()
                         for c in chunks)
        resident_i = sum(c["i_count"] for document in self.documents.values()
                         for chunks in (document.get("gpu") or {}).values()
                         for c in chunks)

        # `_v_used`/`_i_used` count every offset the arena has handed out, while
        # resident_v/resident_i count what the LIVE documents still occupy. A
        # difference means space was freed (remove_file / remove / a replaced
        # document) and the append cursor now sits past the live data - appending
        # there would straddle the gap and run off the end of the capacity, and
        # the upload would be silently refused. So a freed document forces the
        # same compaction a replace does, re-uploading what is still resident.
        freed = (resident_v < self._v_used) or (resident_i < self._i_used)
        if not self._reserve(resident_v + new_v, resident_i + new_i,
                             force=replacing or freed):
            # None (not {}) tells the caller the ENGINE could not take the file,
            # so an SNT-only attach hands the viewport back and lets the VTK path
            # draw instead of showing a canvas with nothing on it.
            return None

        document = {"kind": key[0], "groups": text_grouped, "handles": {}, "gpu": gpu_groups,
                    "entities": entities, "bounds": bounds}
        self._upload_document(document)

        layer_names = set(text_grouped) | set(gpu_groups)
        document["handles"] = {
            layer_name: VectorLayerHandle(
                self, key, layer_name, bounds,
                document["gpu"].get(layer_name, []))
            for layer_name in layer_names
        }
        # Keep layer-widget handles and visibility valid when label edits rebuild
        # a document's glyph geometry.
        for layer_name, handle in previous.get("handles", {}).items():
            if layer_name in document["handles"]:
                handle.chunks = list(document["gpu"].get(layer_name, []))
                document["handles"][layer_name] = handle
        if previous:
            previous.clear()
            previous.update(document)
            document = previous
        self.documents[key] = document
        self._push_line_width()
        self.refresh()
        return {layer_name: [handle]
                for layer_name, handle in document["handles"].items()}

    def _pin_origin(self, gpu_groups) -> None:
        """Pins the render origin from the LINEWORK bounds while nothing is resident.

        The engine otherwise derives the origin from the FIRST uploaded chunk, so
        a survey-wide SNT would be biased to whichever layer happened to upload
        first - and a far layer would then lose float32 precision. That is the
        SNT-only case the native viewport supports (empty cloud, GPU lines).

        Deliberately conservative: only while no vertex/index is resident AND the
        origin is still unpinned, so a loaded point cloud's origin (which the
        whole frame shares) can never be moved underneath it.
        """
        if self._origin_pinned or self._backend is None:
            return
        if self._v_used or self._i_used:
            return
        lo = hi = None
        pending = list(gpu_groups.items())
        for document in self.documents.values():
            pending.extend((document.get("gpu") or {}).items())
        for _layer, chunks in pending:
            for chunk in chunks:
                arr = chunk.get("xyz")
                if arr is None or arr.size == 0:
                    continue
                chunk_lo, chunk_hi = arr.min(axis=0), arr.max(axis=0)
                lo = chunk_lo if lo is None else np.minimum(lo, chunk_lo)
                hi = chunk_hi if hi is None else np.maximum(hi, chunk_hi)
        if lo is None:
            return
        try:
            current = self._backend.get_render_origin_xyz()
        except Exception:
            current = (0.0, 0.0, 0.0)
        if current != (0.0, 0.0, 0.0):
            self._origin_pinned = True   # already authoritative: leave it alone
            return
        centre = (lo + hi) * 0.5
        try:
            self._backend.set_render_origin(
                float(centre[0]), float(centre[1]), float(centre[2]))
        except Exception:
            return
        self._origin_pinned = True

    def _reserve(self, need_v: int, need_i: int, force: bool = False) -> bool:
        """Grows the arena to fit `need_v`/`need_i`, re-uploading when it must.

        A reserve that reports 2 means every earlier offset moved, so the only
        correct response is to re-upload every resident chunk at its new offset -
        which is exactly why the chunk arrays are kept in memory. `force`
        compacts even when the capacity already fits (a replaced document's
        chunks cannot be freed in place, so the arena is rebuilt instead).
        """
        if self._backend is None:
            return False
        try:
            rc = int(self._backend.vector_reserve(max(1, need_v), max(1, need_i)))
        except Exception:
            return False
        if rc == 0:
            return False
        if force or rc == 2:
            # (Re)allocated: nothing resident survives, so rebuild from scratch.
            self._v_used = 0
            self._i_used = 0
            for key, document in self.documents.items():
                document.setdefault("kind", key[0])
                self._upload_document(document)
                for layer_name, handle in document["handles"].items():
                    handle.chunks = document["gpu"].get(layer_name, [])
        return True

    def _upload_document(self, document):
        offset = 0.0
        cloud = _cloud_z_bounds(self.app) if document.get("kind") == "snt" else None
        if cloud is not None:
            lows = []
            for chunks in document.get("gpu", {}).values():
                for chunk in chunks:
                    z = chunk["xyz"][:, 2]
                    finite = z[np.isfinite(z)]
                    if finite.size:
                        lows.append(float(finite.min()))
            for records in document.get("groups", {}).values():
                lows.extend(float(data.get("source_z", 0.0)) for _, data in records)
            if lows:
                z0, z1 = cloud
                target = z1 + max((z1 - z0) * 0.5, 50.0)
                offset = max(0.0, target - min(lows))
        document["z_offset"] = offset
        document["gpu"] = {name: self._upload_chunks(chunks, offset)
                           for name, chunks in document.get("gpu", {}).items()}
        for records in document.get("groups", {}).values():
            for _, data in records:
                if document.get("kind") == "snt":
                    data["world_z"] = float(data.get("source_z", 0.0)) + offset

    def _upload_chunks(self, chunks, z_offset=0.0) -> List[Dict]:
        resident: List[Dict] = []
        for chunk in chunks:
            v_first, i_first = self._v_used, self._i_used
            indices = chunk["indices"].copy()
            live = indices != RESTART_INDEX
            indices[live] += np.uint32(v_first)
            # Keep authored world coordinates for rebinding, picking and saving.
            xyz = chunk["xyz"]
            if z_offset:
                xyz = xyz.copy()
                xyz[:, 2] += z_offset
            ok = False
            try:
                ok = bool(self._backend.vector_upload_strip(
                    v_first, chunk["v_count"], xyz, chunk["rgb"],
                    i_first, chunk["i_count"], indices))
            except Exception:
                ok = False
            if not ok:
                continue
            resident.append({
                "v_first": int(v_first), "i_first": int(i_first),
                "v_count": int(chunk["v_count"]), "i_count": int(chunk["i_count"]),
                "lineweight": int(chunk.get("lineweight", 0) or 0),
                "xyz": chunk["xyz"], "rgb": chunk["rgb"],
                "indices": chunk["indices"],
            })
            self._v_used += chunk["v_count"]
            self._i_used += chunk["i_count"]
        return resident

    def _push_line_width(self) -> None:
        """The fallback width used by any range that carries no explicit one.

        Every chunk this module uploads DOES carry its own width (see
        _push_draw_ranges), so this is the fallback a widthless caller would get:
        the widest authored lineweight in the document set, clamped to the
        device's probed lineWidthRange.
        """
        if self._backend is None:
            return
        weight = 0
        for document in self.documents.values():
            for chunks in (document.get("gpu") or {}).values():
                for chunk in chunks:
                    weight = max(weight, int(chunk.get("lineweight", 0) or 0))
        width = _line_width_px(weight, self._width_range)
        if width != self._line_width_px:
            self._line_width_px = width
            try:
                self._backend.vector_set_line_width(width)
            except Exception:
                pass

    # ---- visibility ------------------------------------------------------
    def remove(self, kind, name):
        super().remove(kind, name)
        self._demote_if_empty()

    def remove_file(self, kind, name):
        super().remove_file(kind, name)
        self._demote_if_empty()

    def _demote_if_empty(self) -> None:
        """Hand the canvas back to VTK once the last SNT file is gone (§5).

        Promoting was only ever justified by this layer's geometry; with nothing
        resident the viewport must return to the state the session started in.
        """
        if self.documents:
            return
        demote_snt_viewport(self.app)

    def refresh(self) -> None:
        # Base class: feeds the Qt overlay from documents[*]['groups'] (now TEXT
        # only) honouring handle._visible, and drives the transparent window.
        super().refresh()
        self._push_draw_ranges()

    def _push_draw_ranges(self) -> None:
        """The union of every VISIBLE layer's resident chunk windows, with widths.

        One window per chunk = one draw call per chunk (at that chunk's own
        authored pixel width), so "draw calls ~= layer groups" is a measurement,
        not an assumption. A toggle changes only this list; no buffer is touched
        and no vertex or index is re-uploaded.
        """
        if self._backend is None:
            return
        firsts: List[int] = []
        counts: List[int] = []
        widths: List[float] = []
        for document in self.documents.values():
            gpu = document.get("gpu") or {}
            for layer_name, chunks in gpu.items():
                handle = document["handles"].get(layer_name)
                if handle is None or not handle._visible:
                    continue
                for chunk in chunks:
                    if chunk["i_count"] <= 0:
                        continue
                    firsts.append(int(chunk["i_first"]))
                    counts.append(int(chunk["i_count"]))
                    widths.append(_line_width_px(chunk.get("lineweight", 0),
                                                 self._width_range))
        try:
            if firsts:
                self._backend.vector_set_draw_ranges(firsts, counts, widths)
            else:
                self._backend.vector_clear_draw_ranges()
        except Exception:
            pass

    def rebind(self) -> bool:
        """Move every resident document onto the app's CURRENT viewport renderer.

        Used when the renderer under the linework is replaced (an .lnt opened over
        an attached SNT, or closed again). Geometry comes from the CPU copies kept
        in each chunk; handles are updated in place, so visibility toggles keep
        working. Returns False when the new renderer cannot draw vectors."""
        from gui.native_attachments import CanvasOverlay, ensure_classifier

        view = _view_for(self.app)
        backend = _backend(self.app)
        if view is None or backend is None:
            return False
        try:
            if not backend.supports_vector_overlay():
                return False
        except Exception:
            return False
        if view is not self.view:
            try:
                self.overlay._timer.stop()
                self.overlay.close()
            except Exception:
                pass
            classifier = ensure_classifier(view)
            if classifier is None:
                return False
            self.view = view
            self.overlay = CanvasOverlay(classifier.canvas)
        self._backend = backend
        self._v_used = 0
        self._i_used = 0
        self._origin_pinned = False
        self._line_width_px = _LINE_WIDTH_BASE
        self._width_range = line_width_range(backend)
        _log_line_width_probe(backend, self._width_range)
        self._pin_origin({})
        need_v = sum(c["v_count"] for d in self.documents.values()
                     for chunks in (d.get("gpu") or {}).values() for c in chunks)
        need_i = sum(c["i_count"] for d in self.documents.values()
                     for chunks in (d.get("gpu") or {}).values() for c in chunks)
        if not self._reserve(need_v, need_i, force=True):
            return False
        self._push_line_width()
        self.refresh()
        print(f"[SNT vectors] re-uploaded {len(self.documents)} file(s) "
              f"({need_v:,} vertices) to the current viewport renderer")
        return True

    def shutdown(self) -> None:
        if self._backend is not None:
            try:
                self._backend.vector_clear()
            except Exception:
                pass
        super().shutdown()
        # Nothing resident any more: an SNT-only session gives the canvas back.
        # (An .lnt session is untouched inside demote_snt_viewport.)
        self._demote_if_empty()


def vector_layer_for(app) -> Optional[NativeVectorLayer]:
    """The app's NativeVectorLayer, created on first use, or None if unusable.

    Rebuilt when the view changes (a new canvas means a new overlay layer) and
    when the engine cannot draw vectors at all - in which case the caller keeps
    the Qt-overlay linework path.
    """
    if not snt_viewport_capable(app):
        return None
    if not vector_overlay_available(app):
        return None
    view = _view_for(app)
    if view is None:
        return None
    layer = getattr(app, "_native_vector_layer", None)
    if layer is not None and (layer.view is not view
                              or layer._backend is not _backend(app)):
        # The old arena died with its renderer: re-upload the kept chunks into
        # the new one IN PLACE, so the Layers widgets' handles stay valid.
        if not layer.rebind():
            return None
    elif layer is None:
        layer = app._native_vector_layer = NativeVectorLayer(app)
    return layer


def render_linework(app, kind: str, attachment: Dict, entities=None,
                    bounds=None, prepared_groups=None):
    """Uploads `entities`' linework and registers their labels on the Qt overlay.

    Returns {layer_name: [handle]} - the same contract as
    gui.native_attachments.render_entities - or None when the engine cannot draw
    vectors, which tells the caller to fall back to the VTK/Qt-overlay path.

    This is also where §5 is decided: with no .lnt open the engine's surface is
    hidden behind VTK, so GPU linework would be drawn where nobody can see it.
    The viewport is therefore promoted BEFORE anything is uploaded, and handed
    straight back if the upload is refused - a refusal then leaves exactly the
    pre-existing VTK canvas and VTK path in place, rather than a black surface.
    """
    layer = vector_layer_for(app)
    if layer is None:
        return None
    if not promote_snt_viewport(app):
        return None
    source = entities if entities is not None else attachment.get("entities", [])
    name = attachment.get("full_path") or attachment.get("filename") or kind
    if bounds is None:
        bounds = attachment.get("fit_bounds") or attachment.get("bounds")
    cache = layer.put_entities(kind, name, source, bounds, prepared_groups=prepared_groups)
    if cache is None:
        demote_snt_viewport(app)
        return None
    return cache


def rebind_after_viewport_change(app) -> bool:
    """Called when the renderer behind the main viewport is replaced (an .lnt
    view started or stopped). Keeps attached SNT/DXF linework on screen."""
    layer = getattr(app, "_native_vector_layer", None)
    if layer is None or not layer.documents:
        return False
    if not is_native_main(app) and not promote_snt_viewport(app):
        return False
    return vector_layer_for(app) is layer
