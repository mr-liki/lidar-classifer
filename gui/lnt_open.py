"""Open .las / .laz / .lnt through the native .lnt pipeline (no .naksha cache).

First open of a source builds `<source>.lnt` beside it (LAZ is decompressed with
laspy+lazrs, then converted by the native C++ builder). Every later open finds the
.lnt newer than the source and loads it directly.
"""
import ctypes
import json
import os
import shutil
import tempfile
import time

from PySide6.QtCore import QEvent, QObject, QThread, Qt, QTimer, Slot, Signal
from PySide6.QtWidgets import QMessageBox, QProgressDialog

from gui import lnt_native as nat
from gui.native_pan import NativePanGesture

LNT_SUFFIXES = (".las", ".laz", ".lnt", ".lntset")


def lnt_path_for(source):
    return source if source.lower().endswith(".lnt") else os.path.splitext(source)[0] + ".lnt"


LNT_FORMAT_VERSION = 2      # 2: points carry a classification byte (11 B/pt)


def lnt_file_version(path):
    """Format version of a structurally complete cache, or 0 if truncated."""
    try:
        with open(path, 'rb') as f:
            head=f.read(64)
            if len(head)!=64 or head[:4]!=b'LNT\0': return 0
            index=int.from_bytes(head[8:16],'little')
            size=os.fstat(f.fileno()).st_size
            if index<64 or index>size-16: return 0
            f.seek(index)
            directory=f.read(16)
            nodes=int.from_bytes(directory[8:16],'little')
            if len(directory)!=16 or not nodes or nodes>(size-index-16)//52: return 0
            return int.from_bytes(head[4:6],'little')
    except OSError:
        return 0


def _source_signature(source):
    stat = os.stat(source)
    return dict(source=os.path.normcase(os.path.realpath(source)), size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, format=LNT_FORMAT_VERSION)


def _source_cache_current(source, lnt):
    if not os.path.isfile(lnt) or lnt_file_version(lnt) != LNT_FORMAT_VERSION:
        return False
    try:
        with open(lnt + '.source.json', encoding='utf-8') as f:
            return json.load(f) == _source_signature(source)
    except FileNotFoundError:
        return os.path.getmtime(lnt) >= os.path.getmtime(source) # compatible older cache
    except (OSError, ValueError):
        return False


def _publish_source_cache(source, staged, destination, signature):
    if _source_signature(source) != signature:
        raise RuntimeError(f'{source} changed while its cache was building')
    if lnt_file_version(staged) != LNT_FORMAT_VERSION:
        raise RuntimeError(f'The converter did not produce a valid cache for {source}')
    # Staging and destination are on the source drive, so replacement is atomic.
    metadata = os.fspath(staged) + '.source.json'
    with open(metadata, 'w', encoding='utf-8') as f:
        json.dump(signature, f, sort_keys=True)
    os.replace(staged, destination)
    os.replace(metadata, os.fspath(destination) + '.source.json')


def lnt_is_current(source):
    """Source fingerprints survive classification edits to the cache's mtime."""
    source = os.fspath(source)
    if source.lower().endswith('.lnt'):
        return os.path.isfile(source) and lnt_file_version(source) == LNT_FORMAT_VERSION
    return _source_cache_current(source, lnt_path_for(source))


class Cancelled(Exception):
    pass


def _laz_to_las(laz, las, progress, cancelled, chunk=2_000_000):
    import laspy
    backends = laspy.LazBackend.detect_available()
    backend = (laspy.LazBackend.LazrsParallel
               if laspy.LazBackend.LazrsParallel in backends else None)
    with laspy.open(laz, laz_backend=backend) as src:
        required = (src.header.offset_to_point_data
                    + src.header.point_count * (src.header.point_format.size + 22)
                    + 8 * 1024**2)
        free = shutil.disk_usage(os.path.dirname(las)).free
        if required > free:
            raise RuntimeError(
                f"Not enough free space beside {laz}: need about {required / 1024**3:.2f} GiB "
                f"for decompression and its LNT; only {free / 1024**3:.2f} GiB is free")
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


def _scratch_dir(source):
    """Require scratch space beside the source; never fall back to system TEMP."""
    folder = os.path.dirname(os.path.realpath(os.fspath(source)))
    try:
        fd, probe = tempfile.mkstemp(prefix="naksha_probe_", dir=folder)
        os.close(fd)
        os.remove(probe)
    except OSError as exc:
        raise OSError(f"Cannot build the cache beside {source}: {exc}. "
                      "Choose a writable source folder with enough free space.") from exc
    return folder


def build_lnt(source, progress, cancelled):
    """Build/reuse an individual LNT; all staging lives beside its LAS/LAZ."""
    source = os.path.realpath(os.fspath(source))
    if cancelled():
        raise Cancelled()
    if lnt_is_current(source):
        progress(100, f"Using cached {os.path.basename(lnt_path_for(source))}")
        return lnt_path_for(source)
    lnt = lnt_path_for(source)
    signature = _source_signature(source)
    folder = _scratch_dir(source)
    if source.lower().endswith('.las'):
        import laspy
        with laspy.open(source) as reader:
            required = reader.header.point_count * 22 + 8 * 1024**2
        if shutil.disk_usage(folder).free < required:
            raise OSError(f"Not enough free space beside {source} to build its LNT cache")
    with tempfile.TemporaryDirectory(prefix="naksha_lnt_", dir=folder) as staging:
        staged_lnt = os.path.join(staging, "cache.lnt")
        las = source
        if source.lower().endswith(".laz"):
            las = os.path.join(staging, "decompressed.las")
            _laz_to_las(source, las, progress, cancelled)
        if cancelled():
            raise Cancelled()
        progress(92, f"Building {os.path.basename(lnt)} (octree + LOD)")
        pts, nodes = nat.convert_las(las, staged_lnt)
        if cancelled():
            raise Cancelled()
        _publish_source_cache(source, staged_lnt, lnt, signature)
    progress(100, f"Built {os.path.basename(lnt)}: {pts:,} points, {nodes:,} nodes")
    return lnt


class _BuildWorker(QObject):
    progress = Signal(float, str)
    finished = Signal(str, str)        # (lnt_path or "", error or "")

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.sources = [os.fspath(p) for p in (source if isinstance(source, (list, tuple)) else [source])]
        self.render_source = self.sources[0]
        self.target_crs_wkt = None
        self._cancel = False

    def cancel(self):
        self._cancel = True

    @Slot()
    def run(self):
        try:
            if len(self.sources) > 1:
                from gui.lnt_batch import build_batch, BatchCancelled
                try:
                    self.render_source, lnt = build_batch(self.sources,
                        lambda p, t: self.progress.emit(p, t), lambda: self._cancel,
                        target_crs_wkt=self.target_crs_wkt)
                except BatchCancelled:
                    raise Cancelled()
            else:
                lnt = build_lnt(self.sources[0], lambda p, t: self.progress.emit(p, t),
                                lambda: self._cancel)
                if self.target_crs_wkt and not self.sources[0].lower().endswith('.lnt'):
                    from pyproj import CRS
                    from gui.lnt_batch import _projected_cache
                    lnt = _projected_cache(self.sources[0],CRS.from_user_input(self.target_crs_wkt),
                        lambda p,t:self.progress.emit(p,t),lambda:self._cancel,500_000) or lnt
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
        self._pan = NativePanGesture(self._pan_main)

    def _pan_main(self, dx, dy):
        nv = self.view.native
        if nv is not None:
            nv.pan(dx, dy)

    def eventFilter(self, obj, ev):   # noqa: N802 (Qt API)
        t = ev.type()
        nv = self.view.native
        if nv is None:
            return False
        E = QEvent
        creation = getattr(self.view, 'creation', None)
        if creation is not None and creation.active:
            middle_drag = t == E.MouseMove and bool(ev.buttons() & Qt.MiddleButton)
            if (t == E.MouseButtonPress and ev.button() != Qt.MiddleButton
                    and creation.press(ev.position(), ev.button(), ev.modifiers())):
                self._last = None; obj.setFocus(); return True
            if t == E.MouseMove and not middle_drag and creation.move(ev.position()):
                return True
            if t == E.KeyPress and creation.key(ev.key()):
                return True
            if t == E.MouseButtonRelease and ev.button() != Qt.MiddleButton:
                creation.release(ev.position(), ev.button())
                self._last = None; return True
        drawing = getattr(self.view, 'drawing', None)
        if drawing is not None and drawing.active():
            middle_drag = t == E.MouseMove and bool(ev.buttons() & Qt.MiddleButton)
            if (t == E.MouseButtonPress and ev.button() != Qt.MiddleButton
                    and drawing.press(ev.position(), ev.button())):
                self._last = None; obj.setFocus(); return True
            if t == E.MouseMove and not middle_drag and drawing.move(ev.position(), ev.buttons()):
                return True
            if t == E.MouseButtonRelease and ev.button() != Qt.MiddleButton:
                if drawing.release(ev.position(), ev.button()):
                    self._last = None; return True
            if t == E.MouseButtonDblClick and drawing.double_click(ev.position(), ev.button()):
                return True
            if t == E.KeyPress and drawing.key(ev.key()):
                return True
        measurement = getattr(self.view, 'measurement', None)
        if measurement is not None and measurement.active:
            # Measurement owns left/right clicks, but middle-drag always stays
            # with the native camera so the user can pan while placing points.
            middle_drag = t == E.MouseMove and bool(ev.buttons() & Qt.MiddleButton)
            if (t == E.MouseButtonPress and ev.button() != Qt.MiddleButton
                    and measurement.press(ev.position(), ev.button())):
                self._last = None
                obj.setFocus()
                return True
            if t == E.MouseMove and not middle_drag and measurement.move(ev.position()):
                return True
            if t == E.MouseButtonDblClick and measurement.double_click(ev.position(), ev.button()):
                return True
            if t == E.KeyPress and measurement.key(ev.key()):
                return True
            if t == E.MouseButtonRelease and ev.button() != Qt.MiddleButton:
                self._last = None
                return True
        classifier = getattr(self.view, 'classifier', None)
        if classifier is not None and self.view.mode_2d and classifier.active():
            if t == E.MouseButtonPress and classifier.press(ev.position(), ev.button()):
                self._last = None
                obj.setFocus()
                return True
            if t == E.MouseMove and (classifier.move(ev.position()) or ev.buttons() & Qt.LeftButton):
                return True
            if t == E.MouseButtonRelease and classifier.release(ev.position(), ev.button()):
                self._last = None
                return True
            if t == E.MouseButtonDblClick and classifier.double_click(ev.position()):
                return True
            if t == E.KeyPress and classifier.key(ev.key()):
                return True
        # Identify-tab tools own their gestures while armed (native Vulkan,
        # no VTK observers). Order: select/zoom swallow drags; identify takes
        # single clicks. Middle-drag always stays with the camera. An armed
        # Cross Section picker owns left-press and is checked first.
        if t == E.MouseButtonPress and ev.button() == Qt.LeftButton:
            _armed_picker = getattr(self.view, "picker", None)
            if _armed_picker is not None:
                obj.setFocus()
                self._last = None
                _armed_picker.on_press(ev.position())
                return True
        for _attr in ("native_identify", "native_zoom", "native_select",
                      "native_layer_pick"):
            _tool = getattr(self.view, _attr, None)
            if _tool is None or not getattr(_tool, "active", False):
                continue
            _middle_drag = t == E.MouseMove and bool(ev.buttons() & Qt.MiddleButton)
            if t == E.MouseButtonPress and ev.button() != Qt.MiddleButton:
                if _tool.press(ev.position(), ev.button()):
                    self._last = None
                    obj.setFocus()
                    return True
            elif t == E.MouseMove and not _middle_drag:
                if _tool.move(ev.position()):
                    return True
            elif t == E.MouseButtonRelease and ev.button() != Qt.MiddleButton:
                _release = getattr(_tool, "release", None)
                if callable(_release) and _release(ev.position(), ev.button()):
                    self._last = None
                    return True
                self._last = None
                return True
            elif t == E.KeyPress:
                if _tool.key(ev.key()):
                    return True
        # Armed Pan shortcut (Left - Click / Tap - Tap): every tool above
        # keeps its own clicks, so this only sees otherwise free left input.
        if self._pan.handle(getattr(self.view, "app", None), ev):
            if t == E.MouseButtonPress:
                obj.setFocus()
            self._last = None
            return True
        if t == E.MouseButtonPress:
            picker = getattr(self.view, "picker", None)
            if picker is not None and ev.button() == Qt.LeftButton:
                # Handled above (armed Cross Section owns left-press); kept as
                # a safety net in case a future branch reorders the filter.
                obj.setFocus()
                self._last = None
                picker.on_press(ev.position())
                return True
            # In 2D plan mode only the physical middle button owns camera
            # panning. Left clicks remain available for tools/selection and a
            # bare left drag must never move the point cloud.
            self._last = (
                ev.position()
                if not self.view.mode_2d or ev.button() == Qt.MiddleButton
                else None
            )
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
                # 2D plan mode: middle-drag is the sole pan gesture.
                if b & Qt.MiddleButton:
                    nv.pan(dx, dy)
            elif b & Qt.LeftButton and not (ev.modifiers() & Qt.ShiftModifier):
                nv.orbit(dx, dy)
            elif b & (Qt.RightButton | Qt.MiddleButton) or (
                    b & Qt.LeftButton and ev.modifiers() & Qt.ShiftModifier):
                nv.pan(dx, dy)
            return True
        if t == E.Wheel:
            steps = ev.angleDelta().y() / 120.0
            # Cursor-anchored zoom: the world point under the wheel stays put.
            # Falls back to the engine's centre dolly in 3D orbit mode.
            pos = ev.position()
            try:
                ratio = obj.devicePixelRatioF()
            except (AttributeError, RuntimeError):   # noqa: BLE001 - Qt state
                ratio = 1.0
            if not nv.zoom_at(steps, pos.x() * ratio, pos.y() * ratio):
                nv.zoom(steps)
            return True
        if t == E.KeyPress:
            k = ev.key()
            if k == Qt.Key_Escape and getattr(self.view, "picker", None) is not None:
                try:
                    self.view.picker.cancel()
                finally:
                    # Esc must fully stand the tool down (flags, cursor, ribbon
                    # toggle), not just clear the in-progress clicks: route it
                    # through the same app-level helper every other Esc path
                    # uses so the tool does not stay armed.
                    app = getattr(self.view, "app", None)
                    if app is not None and hasattr(app, "deactivate_cross_section_tool"):
                        try:
                            app.deactivate_cross_section_tool()
                        except Exception:  # noqa: BLE001
                            pass
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
        self.native_identify = None  # NativeIdentify while the Identify tool is armed
        self.native_zoom = None      # NativeZoomRect while the Zoom tool is armed
        self.native_select = None    # NativeSelectRect while the Select tool is armed
        self.native_layer_pick = None  # NativeLayerPick while Identify Layer is armed
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
        # The cross-section gesture is click-P1, HOVER to P2, click, HOVER for
        # width, click: hover moves arrive only with mouse tracking on. Without
        # it the live rubber band never follows the cursor and the user only
        # sees the (surprising) finished box.
        try:
            widget.setMouseTracking(True)
        except Exception:  # noqa: BLE001
            pass
        pts, nodes, ext = self.native.info()
        self.total_points = int(pts)        # footer "Total Points" (app.update_total_points_label)
        # Point Statistics: exact per-class counts, scanned from the file in the
        # background and kept current by every class write (NativeView.set_classes).
        self.ptc_applied = False            # classes are listed once a .ptc is Applied
        self.class_counts = None
        try:
            from gui.lnt_class_counts import LntClassCounts
            self.class_counts = LntClassCounts(self.lnt_path, self)
            self.class_counts.ready.connect(self._refresh_point_statistics)
            self.native.class_counts = self.class_counts
        except Exception as exc:   # noqa: BLE001 - statistics are informational
            print(f"[LNT] class counts unavailable: {exc}")
        self._log(f"LNT: {self.name} - {pts:,} points, {nodes:,} nodes, "
                  f"{ext[0]:,.0f} x {ext[1]:,.0f} m", force=True)
        self._t = time.perf_counter()
        self._timer.start()
        print(f"[LNT] showing {self.lnt_path} in the main viewport ({pts:,} points)")

    def stop(self, restore=True, flush_pending=True, keep_native_viewport=False):
        """Tear the .lnt view down.

        `keep_native_viewport`: the point cloud is gone but the CANVAS stays
        native. The owner's renderer is re-bound to the live HWND and keeps
        presenting (empty), `_viewport_installed` stays True and VTK's LiDAR
        actors stay off, so native tools (measure, SNT/DXF linework, the next
        .lnt open) never bounce through the VTK viewport and back - clearing
        data must not change which viewport the session lives in.
        """
        try:
            from gui.lnt_ai_bridge import active_bridge
            bridge = active_bridge(self.app)
            if bridge is not None and bridge.view is self:
                bridge.end()             # write the AI result before the renderer goes
        except Exception as exc:   # noqa: BLE001
            print(f"[LNT] AI session close failed: {exc}")
        drawing = getattr(self, 'drawing', None)
        if drawing is not None:
            drawing.shutdown()
            self.drawing = None
        measurement = getattr(self, 'measurement', None)
        if measurement is not None:
            measurement.shutdown()
            self.measurement = None
        classifier = getattr(self, 'classifier', None)
        if classifier is not None:
            classifier.canvas._closed = True
            classifier.shutdown()
            self.classifier = None
        for _attr in ("native_identify", "native_zoom", "native_select",
                      "native_layer_pick"):
            try:
                _tool = getattr(self, _attr, None)
                if _tool is not None and hasattr(_tool, "cancel"):
                    _tool.cancel()
            except Exception:  # noqa: BLE001
                pass
            try:
                setattr(self, _attr, None)
            except Exception:  # noqa: BLE001
                pass
        self._timer.stop()
        try:
            if self.picker is not None:
                self.picker.cleanup()
            from gui import lnt_section_bridge
            try:
                lnt_section_bridge.release(self.app, flush_pending=flush_pending)
            except TypeError:
                lnt_section_bridge.release(self.app)     # flush class edits, hand app.data back
        except Exception as exc:   # noqa: BLE001
            print(f"[LNT] section state release failed: {exc}")
        owner, widget = self.owner, getattr(self.owner, "vulkan_widget", None)
        if widget is not None and self._filter is not None:
            try:
                widget.removeEventFilter(self._filter)
            except Exception:   # noqa: BLE001
                pass
        counts = getattr(self, "class_counts", None)
        if counts is not None:
            counts.close()
            self.class_counts = None
        if self.native is not None:
            self.native.class_counts = None
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
                if keep_native_viewport:
                    # Stay on the native canvas: the surface keeps presenting
                    # (empty) so the promoted-viewport toolchain keeps working
                    # and the next .lnt open re-uses the installed viewport.
                    owner._viewport_installed = True
                    try:
                        owner.resync_camera(present=True)
                    except Exception:   # noqa: BLE001 - pose push is cosmetic
                        pass
                else:
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
            # Per-class point weights (Display Mode "Weight" column). Native
            # form of the VTK weight_lut: gl_PointSize = base pixel size *
            # clamp(weight, 0.1, 12.0). Uniform-only in the engine, so raising a
            # weight never re-uploads a point.
            weights = np.ones(256, dtype=np.float32)
            weight = float(getattr(self.app, "class_weight", 1.0) or 1.0)
            for code, entry in self.app.class_palette.items():
                code = int(code)
                if 0 <= code < 256:
                    packed[code, :3] = np.clip(
                        np.asarray(entry.get("color", (128, 128, 128)),
                                   dtype=np.float64) * weight, 0, 255).astype(np.uint8)
                    packed[code, 3] = int(bool(entry.get("show", True)))
                    try:
                        raw = float(entry.get("weight", 1.0))
                    except (TypeError, ValueError):
                        raw = 1.0
                    weights[code] = np.float32(max(0.1, min(raw, 12.0)))
            # Only the CLASS palette comes from the Display dialog. The elevation and
            # intensity ramps are generated on the GPU by apply_shading(); the VTK-side
            # tables built by build_point_luts() must not overwrite them.
            ok = (self.native is not None and
                  self.native.set_classification_lut(packed))
            if ok:
                pushed = (hasattr(self.native, "set_class_weights")
                          and self.native.set_class_weights(weights))
                if not pushed:
                    print("[LNT] per-class weights not pushed (engine DLL "
                          "predates lnt_view_set_class_weights)")
                self._palette_baseline = self.app.class_palette
                self.ptc_applied = True
                self._refresh_point_statistics()
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
        if name == "depth":
            # Genuine Vulkan view-space depth. It is derived in the point
            # shader from resident XYZ and the live native camera; no VTK
            # colours, actors, lookup tables, or point-buffer rebuild.
            if self.native is None:
                return False
            self.native.set_mode(nat.MODE_DEPTH)
            self.shading_name = name
            return True
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
        normalized = str(ui_mode or "").strip().lower()
        native_only = normalized in ("depth", "intensity", "elevation")
        applied = self.apply_shading(normalized)
        if not applied and native_only:
            # Never fall through to a VTK display implementation for these
            # modes while the native LNT viewport owns the point cloud.
            raise RuntimeError(f"native Vulkan display mode unavailable: {normalized}")
        if not applied and self.native is not None:
            self.native.set_mode(nat.MODE_CLASS)         # an unknown name falls back to class
        try:
            self.app.display_mode = normalized
        except Exception:   # noqa: BLE001
            pass
        print(f"[LNT] display mode -> {normalized} (native Vulkan)")
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

    def _refresh_point_statistics(self):
        try:
            from gui.point_count_widget import refresh_point_statistics
            refresh_point_statistics(self.app)
        except Exception as exc:   # noqa: BLE001
            print(f"[LNT] point statistics refresh failed: {exc}")

    _LOG_INTERVAL_S = 10.0

    def _log(self, text, force=False):
        """Engine diagnostics (nodes, cache, LOD tiles...) go to the log, not the
        status bar. The render tick produces one per second, so the log gets
        at most one line per _LOG_INTERVAL_S."""
        now = time.perf_counter()
        if force or now - getattr(self, "_log_t", -1e9) >= self._LOG_INTERVAL_S:
            self._log_t = now
            print(f"[LNT] {text}")

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
                self._log(
                    f"LNT {self.name} | surface LOD: {lod['drawn_tiles']:,} tiles "
                    f"({lod['drawn_exact_tiles']:,} full resolution) | {lod['source_covered']:,} source records "
                    f"covered, {lod['source_exact']:,} at full resolution | {lod['represented_drawn']:,} "
                    f"records/representatives as vertices, {lod['triangles_drawn']:,} triangles"
                    + (f" | refining ({pending:,} tiles pending)" if pending else ""))
            elif self._n % 60 == 0 and surface_mode and nv.surface_building():
                self._status(f"LNT {self.name}: building full-detail shaded surface…")
            elif self._n % 60 == 0 and surface_mode:
                source, visible, coverage = nv.surface_point_counts()
                self._log(
                    f"LNT {self.name} | {visible:,} visible records accounted for | "
                    f"{visible-coverage:,} triangle-referenced | {coverage:,} coverage | "
                    f"{source:,} source records")
            elif self._n % 60 == 0:
                s = nv.stats
                self._log(
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
        self.name = ', '.join(os.path.basename(p) for p in worker.sources)

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
            self.mw._lnt_source_path = self.worker.render_source
            self.mw._lnt_source_paths = list(self.worker.sources)
            self.mw._lnt_render_crs_wkt = self.worker.target_crs_wkt
            _show(self.mw, lnt, self.name)

    @Slot()
    def on_cancel(self):
        self.worker.cancel()


def open_with_lnt(main_window, source):
    """Entry point from MainWindow.open_file. Always returns True (handled): on
    failure the user sees the reason; there is deliberately no .naksha fallback."""
    sources = list(source) if isinstance(source, (list, tuple)) else [source]
    target_crs_wkt = getattr(main_window,'project_crs_wkt',None)
    if len(sources)==1 and os.fspath(sources[0]).lower().endswith('.lntset'):
        from gui.lnt_collection import read_collection
        try:
            group=read_collection(sources[0])
            sources=group['sources']
            target_crs_wkt=target_crs_wkt or group.get('target_crs')
            source=sources if len(sources)>1 else sources[0]
        except (OSError,ValueError) as exc:
            QMessageBox.critical(main_window,'Open point cloud',f'Cannot read the LNT selection index: {exc}')
            return True
    if not sources:
        return True
    if len(sources) == 1:
        source = sources[0]
    name = ', '.join(os.path.basename(p) for p in sources)
    if len(sources) == 1:
        from gui.lnt_batch import cached_render_cache
        try:
            cached = cached_render_cache(source,target_crs_wkt)
        except (OSError,ValueError):
            cached = None
        if cached:
            main_window._lnt_source_path = source
            main_window._lnt_source_paths = sources
            main_window._lnt_render_crs_wkt = target_crs_wkt
            _show(main_window, cached, name)
            return True

    # Rebuilding a source cache must release readers/writers of that same
    # individual LNT first (Windows denies replacing an open cache file).
    current = getattr(main_window, '_lnt_main_view', None)
    previous = {os.path.normcase(os.path.realpath(p)) for p in getattr(main_window, '_lnt_source_paths', ())}
    if current is not None and any(
            os.path.normcase(os.path.realpath(p)) in previous and not lnt_is_current(p)
            for p in sources):
        close_lnt_view(main_window, reopening=True)

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
    worker.target_crs_wkt = target_crs_wkt
    worker.moveToThread(thread)
    ui = _BuildUi(main_window, source, dlg, thread, worker)
    main_window._lnt_build = (thread, worker, dlg, ui)      # keep alive

    worker.progress.connect(ui.on_progress)
    worker.finished.connect(ui.on_finished)
    dlg.canceled.connect(ui.on_cancel)
    thread.started.connect(worker.run)
    thread.start()
    return True


def _keep_vector_overlay(app):
    """The renderer behind the viewport was replaced: re-upload attached SNT/DXF
    linework to the one now drawing, so it does not vanish."""
    try:
        from gui.native_vector_overlay import rebind_after_viewport_change
        rebind_after_viewport_change(app)
    except Exception as exc:   # noqa: BLE001 - never block opening/closing a cloud
        print(f"[LNT] attached linework could not follow the viewport: {exc}")


def close_lnt_view(app, flush_pending=True, reopening=False, keep_native_viewport=False):
    """Return the main viewport to VTK if an .lnt is currently shown.

    `reopening`: another .lnt is about to be shown, so attached linework moves
    straight to that view instead of bouncing through Naksha's renderer.

    `keep_native_viewport`: the point cloud was CLEARED (Clear Project / Clear
    Data), not replaced - so the (now empty) native canvas stays installed
    instead of handing the viewport back to VTK. Native tools therefore keep
    routing to Vulkan after a clear. Falls back to the VTK hand-back when the
    engine cannot serve the promoted viewport, which would otherwise leave the
    tools drawing where nobody can see them.
    """
    v = getattr(app, "_lnt_main_view", None)
    if v is not None:
        try:
            v.stop(flush_pending=flush_pending, keep_native_viewport=keep_native_viewport)
        except TypeError:
            try:
                v.stop(flush_pending=flush_pending)
            except TypeError:
                v.stop()
        v.deleteLater()
        app._lnt_main_view = None
        if keep_native_viewport and not _kept_viewport_usable(app):
            keep_native_viewport = False
            _return_viewport_to_vtk(app)
        print("[LNT] closed the .lnt view"
              + ("; native viewport kept (empty canvas)"
                 if keep_native_viewport else "; main viewport returned to VTK"))
        if not reopening:
            _keep_vector_overlay(app)
            _refresh_total_points(app)


def _kept_viewport_usable(app) -> bool:
    """True when a kept native canvas can still serve the promoted-viewport
    toolchain (measure, SNT/DXF linework).

    Same gate as gui.native_vector_overlay.snt_viewport_active: the owner
    renderer has just been re-bound by LntMainView.stop(), so the vector pass
    answers against the live handle. Without it the Vulkan surface would cover
    a viewport whose only working tools are VTK's - i.e. invisible.
    """
    try:
        from gui.native_vector_overlay import engine_can_draw_vectors
        owner = getattr(app, "render_backend", None)
        widget = getattr(owner, "vulkan_widget", None) if owner is not None else None
        return (widget is not None
                and bool(getattr(owner, "_viewport_installed", False))
                and engine_can_draw_vectors(app))
    except Exception:   # noqa: BLE001 - a broken probe must not strand the canvas
        return False


def _return_viewport_to_vtk(app) -> None:
    """Hand the main canvas back to VTK (the pre-keep behaviour)."""
    owner = getattr(app, "render_backend", None)
    if owner is None:
        return
    try:
        app._snt_viewport_promoted = False     # §5 flag follows the canvas
        widget = getattr(owner, "vulkan_widget", None)
        if widget is not None:
            widget.hide()
        owner._viewport_installed = False
        owner.set_vtk_lidar_rendering(True)
    except Exception as exc:   # noqa: BLE001
        print(f"[LNT] could not restore the VTK viewport: {exc}")


def take_back_viewport_for_vtk_load(app) -> bool:
    """A VTK-side load must own the visible canvas.

    The native surface can be promoted while nothing is loaded (measure tools
    on a freshly started app, an SNT-only attach). Once a VTK load populates
    app.data with no .lnt session drawing the cloud, that opaque surface would
    cover a cloud nobody can see - hand the viewport back to VTK. Left
    untouched: an .lnt session (it owns the canvas), resident native linework
    (its upload would be stranded), and sessions that were never promoted.
    """
    try:
        from gui.native_vector_overlay import is_native_main, _owner
        if is_native_main(app):
            return False
        owner = _owner(app)
        if owner is None or not getattr(owner, "_viewport_installed", False):
            return False
        layer = getattr(app, "_native_vector_layer", None)
        if layer is not None and getattr(layer, "documents", None):
            return False                    # SNT/DXF linework is resident
        data = getattr(app, "data", None)
        xyz = data.get("xyz") if isinstance(data, dict) else None
        if xyz is None:
            return False
        try:
            if len(xyz) == 0:
                return False
        except Exception:   # noqa: BLE001 - exotic array: assume real data
            pass
        _return_viewport_to_vtk(app)
        print("[LNT] VTK load owns the canvas again (native viewport handed back)")
        return True
    except Exception as exc:   # noqa: BLE001 - never block a load on this
        print(f"[LNT] VTK-load viewport hand-back skipped: {exc}")
        return False


def _refresh_total_points(app):
    """Footer "Total Points": the open .lnt's record count, or app.data's once closed."""
    update = getattr(app, "update_total_points_label", None)
    if callable(update):
        try:
            update()
        except Exception as exc:   # noqa: BLE001
            print(f"[LNT] total points label update failed: {exc}")


def _show(app, lnt_path, name):
    close_lnt_view(app, reopening=True)
    view = LntMainView(app, lnt_path, name)
    try:
        view.start()
    except Exception as exc:   # noqa: BLE001 - surfaced to the user
        view.stop()
        _keep_vector_overlay(app)
        print(f"[LNT] cannot show {lnt_path}: {exc}")
        QMessageBox.critical(app, "Open point cloud", f"Could not show the point cloud:\n\n{exc}")
        return
    app._lnt_main_view = view
    _keep_vector_overlay(app)
    app.data = None   # full-cloud records are owned by the native file, not a VTK array
    app.loaded_file = getattr(app, "_lnt_source_path", lnt_path)
    app.setWindowTitle(f"NakshaAI-LiDAR - {name}")
    _refresh_total_points(app)
    if not getattr(app, "_lnt_quit_hook", False):
        app._lnt_quit_hook = True
        from PySide6.QtWidgets import QApplication
        # Free the LNT renderer before Qt tears the window down at exit.
        QApplication.instance().aboutToQuit.connect(
            lambda: getattr(app, "_lnt_main_view", None)
            and app._lnt_main_view.stop(restore=False))
