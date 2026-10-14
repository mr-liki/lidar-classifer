"""Open .las / .laz / .lnt through the native .lnt pipeline (no .naksha cache).

First open of a source builds `<source>.lnt` beside it (LAZ is decompressed with
laspy+lazrs, then converted by the native C++ builder). Every later open finds the
.lnt newer than the source and loads it directly.
"""
import ctypes
import os
import shutil
import tempfile
import time

from PySide6.QtCore import QEvent, QObject, QThread, Qt, QTimer, Slot, Signal
from PySide6.QtWidgets import QMessageBox, QProgressDialog

from gui import lnt_native as nat

LNT_SUFFIXES = (".las", ".laz", ".lnt")


def lnt_path_for(source):
    return source if source.lower().endswith(".lnt") else os.path.splitext(source)[0] + ".lnt"


LNT_FORMAT_VERSION = 2      # 2: points carry a classification byte (11 B/pt)


def lnt_file_version(path):
    """The format version stored in an .lnt header (uint16 at offset 4), or 0."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        if head[:3] != b"LNT" or len(head) < 6:
            return 0
        return int.from_bytes(head[4:6], "little")
    except OSError:
        return 0


def lnt_is_current(source):
    """True when a usable .lnt of the CURRENT format exists for `source`. An .lnt
    written by an older build (no classification) is treated as stale and rebuilt."""
    lnt = lnt_path_for(source)
    if source.lower().endswith(".lnt"):
        return os.path.isfile(source) and lnt_file_version(source) == LNT_FORMAT_VERSION
    return (os.path.isfile(lnt) and lnt_file_version(lnt) == LNT_FORMAT_VERSION
            and os.path.getmtime(lnt) >= os.path.getmtime(source))


class Cancelled(Exception):
    pass


def _laz_to_las(laz, las, progress, cancelled, chunk=2_000_000):
    import laspy
    backends = laspy.LazBackend.detect_available()
    backend = (laspy.LazBackend.LazrsParallel
               if laspy.LazBackend.LazrsParallel in backends else None)
    with laspy.open(laz, laz_backend=backend) as src:
        required = (src.header.offset_to_point_data
                    + src.header.point_count * src.header.point_format.size)
        free = shutil.disk_usage(os.path.dirname(las)).free
        if required > free:
            raise RuntimeError(
                f"LAZ needs about {required / 1024**3:.1f} GiB of temporary space "
                f"to decompress; only {free / 1024**3:.1f} GiB is free")
        header = laspy.LasHeader(point_format=src.header.point_format,
                                 version=src.header.version)
        header.scales, header.offsets = src.header.scales, src.header.offsets
        total, done = max(1, src.header.point_count), 0
        started = time.perf_counter()
        with laspy.open(las, mode="w", header=header) as dst:
            for pts in src.chunk_iterator(chunk):
                if cancelled():
                    raise Cancelled()
                dst.write_points(pts)
                done += len(pts)
                rate = done / max(time.perf_counter() - started, 0.001) / 1_000_000
                progress(5 + 85.0 * done / total,
                         f"Decompressing LAZ  {done:,}/{total:,} points ({rate:.1f} M/s)")


def build_lnt(source, progress, cancelled):
    """Builds <source>.lnt. progress(percent, text); raises on failure."""
    lnt = lnt_path_for(source)
    tmp_las = None
    tmp_lnt = lnt + ".tmp"
    try:
        las = source
        if source.lower().endswith(".laz"):
            fd, tmp_las = tempfile.mkstemp(suffix=".las", prefix="lnt_")
            os.close(fd)
            _laz_to_las(source, tmp_las, progress, cancelled)
            las = tmp_las
        if cancelled():
            raise Cancelled()
        progress(92, "Building .lnt (octree + LOD)")
        pts, nodes = nat.convert_las(las, tmp_lnt)
        os.replace(tmp_lnt, lnt)
        progress(100, f"Built {os.path.basename(lnt)}: {pts:,} points, {nodes:,} nodes")
        return lnt
    finally:
        for p in (tmp_las, tmp_lnt):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


class _BuildWorker(QObject):
    progress = Signal(float, str)
    finished = Signal(str, str)        # (lnt_path or "", error or "")

    def __init__(self, source):
        super().__init__()
        self.source = source
        self._cancel = False

    def cancel(self):
        self._cancel = True

    @Slot()
    def run(self):
        try:
            lnt = build_lnt(self.source, lambda p, t: self.progress.emit(p, t),
                            lambda: self._cancel)
            self.finished.emit(lnt, "")
        except Cancelled:
            self.finished.emit("", "cancelled")
        except Exception as exc:   # noqa: BLE001 - surfaced to the user
            self.finished.emit("", f"{type(exc).__name__}: {exc}")


class _InputFilter(QObject):
    """Installed on Naksha's own Vulkan viewport widget. While an .lnt is shown,
    mouse / wheel / keys drive the .lnt camera and are CONSUMED, so the widget's
    existing camera-rig handlers (made for the VTK-synced path) do not also act."""

    def __init__(self, view):
        super().__init__(view)
        self.view = view
        self._last = None

    def eventFilter(self, obj, ev):   # noqa: N802 (Qt API)
        t = ev.type()
        nv = self.view.native
        if nv is None:
            return False
        E = QEvent
        if t == E.MouseButtonPress:
            picker = getattr(self.view, "picker", None)
            if picker is not None and ev.button() == Qt.LeftButton:
                # An armed Cross Section tool owns the left button (P1 / P2 / width).
                obj.setFocus()
                self._last = None
                picker.on_press(ev.position())
                return True
            self._last = ev.position()
            obj.setFocus()
            return True
        if t == E.MouseButtonRelease:
            self._last = None
            return True
        if t == E.MouseMove:
            picker = getattr(self.view, "picker", None)
            if picker is not None:
                picker.on_move(ev.position())
            if self._last is None:
                return True
            p = ev.position()
            dx, dy = p.x() - self._last.x(), p.y() - self._last.y()
            self._last = p
            b = ev.buttons()
            if self.view.mode_2d:
                # 2D plan mode: any drag pans the map; there is no orbit.
                if b & (Qt.LeftButton | Qt.RightButton | Qt.MiddleButton):
                    nv.pan(dx, dy)
            elif b & Qt.LeftButton and not (ev.modifiers() & Qt.ShiftModifier):
                nv.orbit(dx, dy)
            elif b & (Qt.RightButton | Qt.MiddleButton) or (
                    b & Qt.LeftButton and ev.modifiers() & Qt.ShiftModifier):
                nv.pan(dx, dy)
            return True
        if t == E.Wheel:
            nv.zoom(ev.angleDelta().y() / 120.0)
            return True
        if t == E.KeyPress:
            k = ev.key()
            if k == Qt.Key_Escape and getattr(self.view, "picker", None) is not None:
                self.view.picker.cancel()
            elif k == Qt.Key_R:
                nv.reset()
                self.view.set_2d(True)
            elif k == Qt.Key_D:
                self.view.set_2d(not self.view.mode_2d)
            elif k == Qt.Key_1:
                self.view.apply_shading("class")
            elif k == Qt.Key_2:
                self.view.apply_shading("intensity")
            elif k == Qt.Key_3:
                self.view.apply_shading("elevation")
            elif k == Qt.Key_4:
                self.view.apply_shading("rgb")
            elif k == Qt.Key_5:
                self.view.apply_shading("shaded")
            elif k == Qt.Key_6:
                self.view.apply_shading("surface")
            elif k in (Qt.Key_BracketRight, Qt.Key_Up):
                self.view.adjust_sse(+1)
            elif k in (Qt.Key_BracketLeft, Qt.Key_Down):
                self.view.adjust_sse(-1)
            else:
                return False
            return True
        return False


class LntMainView(QObject):
    """Shows an .lnt in the EXISTING Naksha main viewport.

    Naksha already owns the engine's one renderer and its Vulkan widget; this takes
    that widget over as the main viewport (the app's own install_as_main_viewport),
    attaches the native streamer to the same renderer, and pumps frames."""

    def __init__(self, app, lnt_path, name):
        super().__init__(app)
        self.app, self.lnt_path, self.name = app, lnt_path, name
        self.native = None
        self.owner = None
        self.sse = 8.0
        self._palette_baseline = None
        self.mode_2d = True          # plan view by default; D toggles 3D orbit
        # Presentation settings (changed by the Display UI; applied by apply_shading()).
        self.ramp_elevation = nat.RAMP_HUE
        self.ramp_intensity = nat.RAMP_GRAYSCALE
        self.z_range = None              # None -> the data's 1st..99th percentile Z
        self.intensity_range = (0.0, 1.0)
        self.light = (0.5, -0.5, -0.7071)
        self.edl = 0.4
        # Crisp shaded-classification lighting (VTK parity). These are the VTK view's own
        # defaults; _push_surface_shading() refreshes them from the application's Shading
        # controls before every Shading mesh, so the .lnt view and the VTK view light the
        # terrain identically.
        self.shading_azimuth = 45.0
        self.shading_sharpness = 45.0
        self.shading_ambient = 0.25
        self.shading_name = "class"
        self._stats = None
        self.picker = None           # LntCrossPicker while the Cross Section tool is armed
        self.section_overlay = None  # rubber band / section rectangles over the native view
        self._filter = None
        self._owner_renderer_down = False
        self._t = time.perf_counter()
        self._size = (0, 0)
        self._fails = 0
        self._n = 0
        self._timer = QTimer(self)
        self._timer.setInterval(8)
        self._timer.timeout.connect(self._tick)

    # -- lifecycle -------------------------------------------------------
    # The engine allows ONE renderer per process and Naksha's own renderer carries
    # state (arena, modes, passes) that a streamed .lnt must not inherit. So while
    # an .lnt is shown, Naksha's renderer is shut down and a DEDICATED LNT renderer
    # runs on the same viewport window; when the .lnt is closed the LNT renderer is
    # destroyed and Naksha's is re-created on that window (its own rebind routine).
    def start(self):
        owner = getattr(self.app, "render_backend", None)
        vb = getattr(owner, "vulkan_backend", None)
        widget = getattr(owner, "vulkan_widget", None)
        if owner is None or vb is None or widget is None or not getattr(vb, "_handle", 0):
            raise RuntimeError("the Vulkan renderer is not available in this session "
                               "(see the NAKSHA RENDER BACKEND banner at startup)")
        if not owner.install_as_main_viewport():
            raise RuntimeError("could not switch the main viewport to the Vulkan renderer")
        self.owner = owner
        w, h = self._px(widget)
        hwnd = int(widget.winId())
        vb.shutdown()                       # nkv_destroy_renderer: frees the one-per-process slot
        self._owner_renderer_down = True
        self.native = nat.NativeView(hwnd, w, h, self.lnt_path, 2048, self.sse)   # 2 GiB tile cache
        self._size = (w, h)
        # COLOURS: one flat default colour for every class until the user loads a .ptc in
        # the Display dialog and clicks Apply. Whatever palette the application already
        # holds (restored from an earlier session, a remembered / bound .ptc) is NOT used:
        # it is only remembered, so that a palette Applied later can be told apart from it.
        self._palette_baseline = getattr(self.app, "class_palette", None)
        self._push_neutral()
        self._stats = self.native.data_stats()
        if not self.apply_shading("class"):
            self.native.set_mode(nat.MODE_CLASS)
        try:
            self.app.display_mode = "class"
        except Exception:   # noqa: BLE001
            pass
        self._filter = _InputFilter(self)
        widget.installEventFilter(self._filter)
        widget.setFocusPolicy(Qt.StrongFocus)
        pts, nodes, ext = self.native.info()
        self._status(f"LNT: {self.name} - {pts:,} points, {nodes:,} nodes, "
                     f"{ext[0]:,.0f} x {ext[1]:,.0f} m  [2D plan: drag=pan  wheel=zoom  D=3D orbit  "
                     f"[ ]=detail  1/2/3=colour  R=reset]")
        self._t = time.perf_counter()
        self._timer.start()
        print(f"[LNT] showing {self.lnt_path} in the main viewport ({pts:,} points)")

    def stop(self, restore=True):
        self._timer.stop()
        try:
            if self.picker is not None:
                self.picker.cleanup()
            from gui import lnt_section_bridge
            lnt_section_bridge.release(self.app)     # flush class edits, hand app.data back
        except Exception as exc:   # noqa: BLE001
            print(f"[LNT] section state release failed: {exc}")
        owner, widget = self.owner, getattr(self.owner, "vulkan_widget", None)
        if widget is not None and self._filter is not None:
            try:
                widget.removeEventFilter(self._filter)
            except Exception:   # noqa: BLE001
                pass
        if self.native is not None:
            self.native.close()          # destroys the LNT renderer (frees the slot)
            self.native = None
        if owner is not None and not restore:
            pass        # application is quitting: just free the renderer, rebuild nothing
        elif owner is not None:
            try:
                if getattr(self, "_owner_renderer_down", False):
                    w, h = self._px(widget)
                    # Naksha's own routine: re-create ITS renderer on the live HWND.
                    owner._rebind_renderer_to_current_hwnd(w, h)
                    self._owner_renderer_down = False
                if widget is not None:
                    widget.hide()
                owner._viewport_installed = False
                owner.set_vtk_lidar_rendering(True)      # VTK shows its LiDAR again
            except Exception as exc:   # noqa: BLE001
                print(f"[LNT] could not restore Naksha's renderer cleanly: {exc}")
        self.owner = None

    # -- colour ----------------------------------------------------------
    def _luts_for(self, source):
        from gui.render_backend import build_point_luts
        return build_point_luts(source)

    def _push_neutral(self):
        """One flat colour for every class (Naksha's built-in default class colour)."""
        try:
            from types import SimpleNamespace
            from gui.ptc_table import default_palette
            shim = SimpleNamespace(
                class_palette=default_palette(), class_weight=1.0,
                elevation_color_ramp=getattr(self.app, "elevation_color_ramp", None))
            cls_lut, elev_lut, int_lut = self._luts_for(shim)
            if self.native is not None:
                self.native.set_luts(cls_lut, elev_lut, int_lut)
        except Exception as exc:   # noqa: BLE001
            print(f"[LNT] neutral colours failed: {exc}")

    def _palette_was_applied(self):
        """True when the Display dialog has stored a NEW palette since this view opened.
        Apply always assigns a fresh dict to app.class_palette, so identity (not content)
        is what tells an Apply from the palette that was already there."""
        pal = getattr(self.app, "class_palette", None)
        return bool(pal) and pal is not self._palette_baseline

    def _push_applied_palette(self):
        try:
            import numpy as np
            packed = np.full((256, 4), 128, dtype=np.uint8)
            packed[:, 3] = 1
            weight = float(getattr(self.app, "class_weight", 1.0) or 1.0)
            for code, entry in self.app.class_palette.items():
                code = int(code)
                if 0 <= code < 256:
                    packed[code, :3] = np.clip(
                        np.asarray(entry.get("color", (128, 128, 128)),
                                   dtype=np.float64) * weight, 0, 255).astype(np.uint8)
                    packed[code, 3] = int(bool(entry.get("show", True)))
            # Only the CLASS palette comes from the Display dialog. The elevation and
            # intensity ramps are generated on the GPU by apply_shading(); the VTK-side
            # tables built by build_point_luts() must not overwrite them.
            ok = (self.native is not None and
                  self.native.set_classification_lut(packed))
            if ok:
                self._palette_baseline = self.app.class_palette
            print("[LNT] applied the palette from the Display dialog (Apply)")
            return ok
        except Exception as exc:   # noqa: BLE001
            print(f"[LNT] palette push failed: {exc}")
            return False

    # UI mode names (app.set_display_mode) -> shading presentation modes (native, no VTK)
    _UI_TO_SHADING = {"class": nat.SHADE_CLASSIFICATION, "classification": nat.SHADE_CLASSIFICATION,
                      "rgb": nat.SHADE_RGB, "intensity": nat.SHADE_INTENSITY,
                      "elevation": nat.SHADE_ELEVATION,
                      "shaded_class": nat.SHADE_NORMAL, "shaded": nat.SHADE_NORMAL}

    def _push_surface_shading(self):
        """Mirror the application's Shaded-Classification light controls onto the native view.

        The native Shading mesh is lit by the SAME engine model as the VTK view
        (gui/shading_display.py::_configure_microstation_color_blend_lighting), so it needs
        the same azimuth / Sharpness / Ambient. Falls back to the VTK defaults when the
        application has never exposed those controls."""
        app = self.app
        azimuth = getattr(app, "last_shade_azimuth", None)
        sharpness = getattr(app, "shading_sharpness_angle", None)
        ambient = getattr(app, "shade_ambient", None)
        light_elevation = getattr(app, "shading_light_elevation", None)
        try:
            self.shading_azimuth = float(azimuth if azimuth is not None else self.shading_azimuth)
            self.shading_sharpness = float(
                sharpness if sharpness is not None else self.shading_sharpness)
            self.shading_ambient = float(ambient if ambient is not None else self.shading_ambient)
            elevation = float(light_elevation) if light_elevation is not None else 0.0
        except Exception:   # noqa: BLE001
            return False
        if self.native is None:
            return False
        # Single-class light elevation, as gui/shading_display.py reads it. The mesh always
        # uses every visible point (no Fast/Normal thinning), so quality is not pushed.
        try:
            single_elevation = float(getattr(app, "last_shade_angle", 45.0))
        except (TypeError, ValueError):
            single_elevation = 45.0
        self.native.set_surface_single_elevation(single_elevation)
        return self.native.set_surface_shading(
            self.shading_azimuth, self.shading_sharpness, self.shading_ambient, elevation)

    def apply_shading(self, ui_mode):
        """Select a native presentation mode. Shaded builds a triangulated, lit
        surface from the visible .lnt cut; other modes update GPU uniforms."""
        name = str(ui_mode or "").strip().lower()
        if name == "neutral":
            if self.native is None:
                return False
            self.native.set_mode(nat.MODE_NEUTRAL)
            self.shading_name = name
            return True
        if name == "surface":
            # Surface = the Shading mesh coloured by elevation (native, no VTK).
            if self.native is None:
                return False
            self._push_surface_shading()
            st = self._stats or {}
            z0, z1 = self.z_range or (st.get("z_p1", 0.0), st.get("z_p99", 1.0))
            if not z1 > z0:
                z1 = z0 + 1.0
            ok = (self.native.set_shading(nat.SHADE_NORMAL, self.ramp_elevation, (z0, z1),
                                          self.intensity_range, self.light, 0.0)
                  and self.native.set_surface_elevation(True, self.ramp_elevation, (z0, z1)))
            if ok:
                self.shading_name = name
            return ok
        mode = self._UI_TO_SHADING.get(name)
        if mode is None or self.native is None:
            return False
        if mode == nat.SHADE_NORMAL:      # Shaded: the lit triangulated mesh, not a point pass
            self._push_surface_shading()
        st = self._stats or {}
        z0, z1 = self.z_range or (st.get("z_p1", 0.0), st.get("z_p99", 1.0))
        if not z1 > z0:
            z1 = z0 + 1.0
        ramp = self.ramp_intensity if mode == nat.SHADE_INTENSITY else self.ramp_elevation
        ok = self.native.set_shading(mode, ramp, (z0, z1), self.intensity_range, self.light,
                                     self.edl if mode == nat.SHADE_NORMAL else 0.0)
        if ok:
            self.shading_name = name
        return ok

    def set_app_display_mode(self, ui_mode):
        """Entry from MainWindow.set_display_mode (Display dialog Apply / ribbon). A palette
        is pushed ONLY if the dialog has just applied one; otherwise the colours stay as
        they are."""
        if self._palette_was_applied():
            self._push_applied_palette()
        if not self.apply_shading(ui_mode) and self.native is not None:
            self.native.set_mode(nat.MODE_CLASS)         # an unknown name falls back to class
        try:
            self.app.display_mode = ui_mode
        except Exception:   # noqa: BLE001
            pass
        print(f"[LNT] display mode -> {ui_mode} (native shading)")
        return True

    # -- controls --------------------------------------------------------
    def set_2d(self, on):
        self.mode_2d = bool(on)
        if self.native is not None:
            self.native.set_2d(self.mode_2d)
        self._status("LNT view: 2D plan (drag = pan, wheel = zoom, D = 3D orbit)" if self.mode_2d
                     else "LNT view: 3D orbit (drag = orbit, right-drag = pan, D = back to 2D)")

    def adjust_sse(self, delta):
        self.sse = max(0.0, min(50.0, self.sse + delta))
        if self.native is not None:
            self.native.set_sse(self.sse)
        self._status(f"LNT detail (SSE) = {self.sse:.0f} px  (0 = full resolution)")

    def _status(self, text):
        try:
            self.app.statusBar().showMessage(text)
        except Exception:   # noqa: BLE001
            pass

    @staticmethod
    def _px(widget):
        r = widget.devicePixelRatioF()
        return max(1, int(widget.width() * r)), max(1, int(widget.height() * r))

    def _tick(self):
        nv, widget = self.native, getattr(self.owner, "vulkan_widget", None)
        if nv is None or widget is None:
            return
        size = self._px(widget)
        if size != self._size:
            self._size = size
            nv.resize(*size)
        now = time.perf_counter()
        dt, self._t = now - self._t, now
        if nv.frame(dt):
            self._fails = 0
            self._n += 1
            surface_mode = self.shading_name in ("shaded", "shaded_class", "surface")
            error = nv.surface_error() if surface_mode else ""
            if error and error != getattr(self, "_surface_reported_error", ""):
                self._status(f"LNT Shading: {error}")
            self._surface_reported_error = error
            persist = nv.surface_persist_message() if surface_mode and self._n % 30 == 0 else ""
            if persist:                       # the DLL already logged it to stderr
                self._status(f"LNT Shading: {persist}")
            lod = nv.surface_lod_stats() if surface_mode and self._n % 60 == 0 and nv.surface_lod_active() else None
            if self._n % 60 == 0 and error:
                self._status(f"LNT Shading: {error}")
            elif lod is not None:
                pending = lod["queued"] + lod["inflight"] + lod["wanted_missing"]
                self._status(
                    f"LNT {self.name} | surface LOD: {lod['drawn_tiles']:,} tiles "
                    f"({lod['drawn_exact_tiles']:,} full resolution) | {lod['source_covered']:,} source records "
                    f"covered, {lod['source_exact']:,} at full resolution | {lod['represented_drawn']:,} "
                    f"records/representatives as vertices, {lod['triangles_drawn']:,} triangles"
                    + (f" | refining ({pending:,} tiles pending)" if pending else ""))
            elif self._n % 60 == 0 and surface_mode and nv.surface_building():
                self._status(f"LNT {self.name}: building full-detail shaded surface…")
            elif self._n % 60 == 0 and surface_mode:
                source, visible, coverage = nv.surface_point_counts()
                self._status(
                    f"LNT {self.name} | {visible:,} visible records accounted for | "
                    f"{visible-coverage:,} triangle-referenced | {coverage:,} coverage | "
                    f"{source:,} source records")
            elif self._n % 60 == 0:
                s = nv.stats
                self._status(
                    f"LNT {self.name} | nodes {s.active_nodes:,} | points "
                    f"{s.points_rendered:,}/{s.points_wanted:,} | cache "
                    f"{s.cache_used_bytes / 1048576:.0f}/{s.cache_capacity_bytes / 1048576:.0f} MiB | "
                    f"loading {s.pending_loads} | {s.frame_ms:.1f} ms | detail {s.sse:.0f} px | "
                    f"mesh {'barycentric' if nv.surface_method() else 'facet'}")
        else:
            self._fails += 1
            if self._fails == 60:
                self._timer.stop()
                print("[LNT] rendering stopped after repeated frame failures")


class _BuildUi(QObject):
    """Lives in the GUI thread: the worker's signals reach these slots QUEUED, so
    every widget / engine call below runs on the GUI thread (the engine's window
    must be created on a thread with a message loop)."""

    def __init__(self, main_window, source, dlg, thread, worker):
        super().__init__(main_window)
        self.mw, self.source, self.dlg = main_window, source, dlg
        self.thread, self.worker = thread, worker
        self.name = os.path.basename(source)

    @Slot(float, str)
    def on_progress(self, p, text):
        self.dlg.setLabelText(text)
        self.dlg.setValue(int(p))

    @Slot(str, str)
    def on_finished(self, lnt, error):
        self.thread.quit()
        self.thread.wait(3000)
        try:
            self.dlg.canceled.disconnect()
        except Exception:   # noqa: BLE001
            pass
        self.dlg.close()
        self.mw._lnt_build = None
        if error == "cancelled":
            print("[LNT] build cancelled")
        elif error:
            print(f"[LNT] build failed: {error}")
            QMessageBox.critical(self.mw, "Open point cloud",
                                 f"Could not build the .lnt file:\n\n{error}")
        else:
            print(f"[LNT] built {lnt}")
            _show(self.mw, lnt, self.name)

    @Slot()
    def on_cancel(self):
        self.worker.cancel()


def open_with_lnt(main_window, source):
    """Entry point from MainWindow.open_file. Always returns True (handled): on
    failure the user sees the reason; there is deliberately no .naksha fallback."""
    name = os.path.basename(source)
    main_window._lnt_source_path = source
    if lnt_is_current(source):
        _show(main_window, lnt_path_for(source), name)
        return True

    dlg = QProgressDialog(f"Building .lnt for {name} (first open only)...", "Cancel", 0, 100,
                          main_window)
    dlg.setWindowTitle("Optimizing point cloud")
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    dlg.setValue(0)

    thread = QThread(main_window)
    worker = _BuildWorker(source)
    worker.moveToThread(thread)
    ui = _BuildUi(main_window, source, dlg, thread, worker)
    main_window._lnt_build = (thread, worker, dlg, ui)      # keep alive

    worker.progress.connect(ui.on_progress)
    worker.finished.connect(ui.on_finished)
    dlg.canceled.connect(ui.on_cancel)
    thread.started.connect(worker.run)
    thread.start()
    return True


def close_lnt_view(app):
    """Return the main viewport to VTK if an .lnt is currently shown."""
    v = getattr(app, "_lnt_main_view", None)
    if v is not None:
        v.stop()
        v.deleteLater()
        app._lnt_main_view = None
        print("[LNT] closed the .lnt view; main viewport returned to VTK")


def _show(app, lnt_path, name):
    close_lnt_view(app)
    view = LntMainView(app, lnt_path, name)
    try:
        view.start()
    except Exception as exc:   # noqa: BLE001 - surfaced to the user
        view.stop()
        print(f"[LNT] cannot show {lnt_path}: {exc}")
        QMessageBox.critical(app, "Open point cloud", f"Could not show the point cloud:\n\n{exc}")
        return
    app._lnt_main_view = view
    app.data = None   # full-cloud records are owned by the native file, not a VTK array
    app.loaded_file = getattr(app, "_lnt_source_path", lnt_path)
    app.setWindowTitle(f"NakshaAI-LiDAR - {name}")
    if not getattr(app, "_lnt_quit_hook", False):
        app._lnt_quit_hook = True
        from PySide6.QtWidgets import QApplication
        # Free the LNT renderer before Qt tears the window down at exit.
        QApplication.instance().aboutToQuit.connect(
            lambda: getattr(app, "_lnt_main_view", None)
            and app._lnt_main_view.stop(restore=False))
