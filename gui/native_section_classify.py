"""Classification tools for the native Vulkan section canvases.

The VTK ClassificationInteractor cannot run here (a native section canvas has no
VTK renderer or interactor), so the drawable tools are re-implemented on the
canvas' exact 2-D mapping: screen x -> horizontal section coordinate, screen y
-> elevation. Selections are evaluated in float64 against the canvas'
`local` records and committed through the shared `app.data` store indices, so
the LntSectionStore flush writes them back into the .lnt file and every other
open section refreshes from the same array.

Tool semantics follow gui.classification_tools for cross/cut sections:
only records inside the visible viewport, of a class shown in this view's
palette, and of a From class (when set) change.
"""
from __future__ import annotations

import time
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt
from PySide6.QtGui import QColor, QCursor, QCursor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QPolygonF, QTransform
from PySide6.QtWidgets import QWidget
from gui.native_measure_geometry import measurement_labels, measurement_text

DRAWABLE_TOOLS = ("rectangle", "circle", "polygon", "freehand", "brush", "point",
                  "above_line", "below_line", "parallel_line")
_TWO_CLICK = ("rectangle", "circle", "above_line", "below_line", "parallel_line")
LINE_TOOLS = ("above_line", "below_line", "parallel_line")
_CLICK_SLOP_PX = 4.0
# Two-point tools use tap-tap by default.  A press becomes a drag only after
# the cursor crosses the threshold *after* the short settle window.  These
# values deliberately mirror the proven VTK ClassificationInteractor path.
_TAP_GRACE_SEC = 0.12
_TAP_DRAG_THRESHOLD_PX = 6.0
# Two-point tools use tap-tap by default.  A press becomes a drag only after
# the cursor crosses the threshold *after* the short settle window.  These
# values deliberately mirror the proven VTK ClassificationInteractor path.
_TAP_GRACE_SEC = 0.12
_TAP_DRAG_THRESHOLD_PX = 6.0


def _points_in_polygon(px, py, poly):
    """Even-odd rule, vectorised over the points; boundary-inclusive enough for picking."""
    inside = np.zeros(px.shape[0], dtype=bool)
    xs, ys = poly[:, 0], poly[:, 1]
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi, xj, yj = xs[i], ys[i], xs[j], ys[j]
        crosses = (yi > py) != (yj > py)
        if np.any(crosses):
            x_at = (xj - xi) * (py[crosses] - yi) / ((yj - yi) or 1e-300) + xi
            hit = np.zeros_like(inside)
            hit[crosses] = px[crosses] <= x_at
            inside ^= hit
        j = i
    return inside


def parallel_tolerances(app):
    def width(name, default):
        try:
            value = float(getattr(app, name, default))
        except (TypeError, ValueError):
            value = default
        return max(0., min(value if np.isfinite(value) else default, 10000.))
    if str(getattr(app, "parallel_line_mode", "symmetric")).strip().lower() == "asymmetric":
        return width("parallel_line_tol_above", .2), width("parallel_line_tol_below", .2)
    half = width("parallel_line_total_width", .4) * .5
    return half, half


def line_style(tool):
    """(QColor, width px) the user chose for this tool in Classify Tools settings."""
    try:
        from gui.classify_settings_dialog import load_classify_style
        style = load_classify_style(tool)
    except Exception:
        style = None
    if not style:
        return QColor(255, 255, 0), 2 if tool not in ("polygon", "brush") else 3
    return QColor.fromRgbF(*style["color"]), max(1, int(style["width"]))


def parallel_band_edges(p1, p2, above, below):
    """The two band boundaries `select` uses: offset along the upward line normal."""
    p1, p2 = np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64)
    length = float(np.hypot(*(p2 - p1)))
    if length <= 1e-12:
        return []
    d = (p2 - p1) / length
    n = np.array([-d[1], d[0]])
    if n[1] < 0:
        n = -n
    return [[tuple(p1 + n * above), tuple(p2 + n * above)],
            [tuple(p1 - n * below), tuple(p2 - n * below)]]


def brush_sweep(h, z, path, radius, square):
    """Records the brush covered along its whole stroke: the union of the brush shape
    swept along every segment of `path` (a capsule for the round brush; for the square
    brush, every point the axis-aligned square passes over), not just at the sampled
    mouse positions - so a fast stroke leaves no gaps."""
    mask = np.zeros(len(h), dtype=bool)
    path = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if not len(path) or not len(h) or radius <= 0:
        return mask
    r = float(radius)
    lo, hi = path.min(axis=0) - r, path.max(axis=0) + r
    near = np.flatnonzero((h >= lo[0]) & (h <= hi[0]) & (z >= lo[1]) & (z <= hi[1]))
    if not near.size:
        return mask
    hs, zs = h[near], z[near]
    # Bucket the nearby records on a 2r grid so each segment tests only its own cells.
    cell = 2.0 * r
    cx = np.minimum(((hs - lo[0]) / cell).astype(np.int64), int((hi[0] - lo[0]) / cell))
    cz = np.minimum(((zs - lo[1]) / cell).astype(np.int64), int((hi[1] - lo[1]) / cell))
    ncx = int(cx.max()) + 1
    key = cz * ncx + cx
    order = np.argsort(key, kind="stable")
    skey = key[order]
    hit = np.zeros(near.size, dtype=bool)
    segments = [(path[0], path[0])] if len(path) == 1 else zip(path[:-1], path[1:])
    for a, b in segments:
        slo, shi = np.minimum(a, b) - r, np.maximum(a, b) + r
        x0, x1 = (max(0, int((slo[0] - lo[0]) // cell)), min(ncx - 1, int((shi[0] - lo[0]) // cell)))
        z0, z1 = (max(0, int((slo[1] - lo[1]) // cell)), int((shi[1] - lo[1]) // cell))
        parts = []
        for row in range(z0, z1 + 1):
            i0 = np.searchsorted(skey, row * ncx + x0, "left")
            i1 = np.searchsorted(skey, row * ncx + x1, "right")
            if i1 > i0:
                parts.append(order[i0:i1])
        if not parts:
            continue
        cand = np.concatenate(parts) if len(parts) > 1 else parts[0]
        cand = cand[~hit[cand]]
        if not cand.size:
            continue
        px, pz = hs[cand] - a[0], zs[cand] - a[1]
        dx, dz = b[0] - a[0], b[1] - a[1]
        if square:
            # Some t in [0, 1] with |px - t dx| <= r and |pz - t dz| <= r.
            t_lo, t_hi = np.zeros(cand.size), np.ones(cand.size)
            ok = np.ones(cand.size, dtype=bool)
            for p, d in ((px, dx), (pz, dz)):
                if d == 0.0:
                    ok &= np.abs(p) <= r
                else:
                    t1, t2 = (p - r) / d, (p + r) / d
                    t_lo = np.maximum(t_lo, np.minimum(t1, t2))
                    t_hi = np.minimum(t_hi, np.maximum(t1, t2))
            inside = ok & (t_lo <= t_hi)
        else:
            length2 = dx * dx + dz * dz
            t = np.clip((px * dx + pz * dz) / length2, 0.0, 1.0) if length2 > 0.0 else 0.0
            ex, ez = px - t * dx, pz - t * dz
            inside = ex * ex + ez * ez <= r * r
        hit[cand[inside]] = True
    mask[near[hit]] = True
    return mask


def brush_is_square(app):
    return str(getattr(app, "brush_shape", "circle")).lower() == "rectangle"


def select(tool, pts2d, shape, app, radius=None):
    """Boolean mask over pts2d (N x 2, horizontal/elevation) for a finished shape."""
    h, z = pts2d[:, 0], pts2d[:, 1]
    shape = np.asarray(shape, dtype=np.float64)
    if tool == "rectangle":
        (x1, z1), (x2, z2) = shape[:2]
        return ((h >= min(x1, x2)) & (h <= max(x1, x2)) &
                (z >= min(z1, z2)) & (z <= max(z1, z2)))
    if tool in ("circle", "point"):
        centre, r = shape[0], (float(np.hypot(*(shape[1] - shape[0]))) if radius is None else radius)
        return (h - centre[0]) ** 2 + (z - centre[1]) ** 2 <= r * r
    if tool == "brush":
        return brush_sweep(h, z, shape, radius, brush_is_square(app))
    if tool in ("polygon", "freehand"):
        if len(shape) < 3:
            return np.zeros(len(h), dtype=bool)
        lo, hi = shape.min(axis=0), shape.max(axis=0)
        box = (h >= lo[0]) & (h <= hi[0]) & (z >= lo[1]) & (z <= hi[1])
        mask = np.zeros(len(h), dtype=bool)
        candidates = np.flatnonzero(box)
        mask[candidates] = _points_in_polygon(h[candidates], z[candidates], shape)
        return mask
    if tool in ("above_line", "below_line"):
        (x1, z1), (x2, z2) = shape[:2]
        slope = (z2 - z1) / (x2 - x1 + 1e-9)
        line_z = slope * (h - x1) + z1
        strip = (h >= min(x1, x2)) & (h <= max(x1, x2))
        return strip & ((z >= line_z) if tool == "above_line" else (z <= line_z))
    if tool == "parallel_line":
        p1, p2 = shape[0], shape[1]
        length = float(np.hypot(*(p2 - p1)))
        if length <= 1e-12:
            return np.zeros(len(h), dtype=bool)
        d = (p2 - p1) / length
        n = np.array([-d[1], d[0]])
        if n[1] < 0:
            n = -n
        above, below = parallel_tolerances(app)
        rel = pts2d - p1
        t, signed = rel @ d, rel @ n
        return (t >= 0) & (t <= length) & (signed <= above) & (signed >= -below)
    return np.zeros(len(h), dtype=bool)


def _convex_hull(points):
    """Convex hull of a few QPointF (monotone chain) as a QPolygonF."""
    pts = sorted(set((p.x(), p.y()) for p in points))
    if len(pts) <= 2:
        return QPolygonF([QPointF(x, y) for x, y in pts])
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return QPolygonF([QPointF(x, y) for x, y in lower[:-1] + upper[:-1]])


def _drawing_vertex_markers_visible(kind, data):
    """Draw vertex handles only while placing or explicitly editing a drawing.

    Completed CAD elements are strokes, not permanent point-marker overlays.
    Keeping this policy in the Qt overlay also covers imported selections and
    other callers that do not go through NativeDraw._paint().
    """
    dtype = str(data.get('type', 'polyline')).lower()
    if dtype in ('text', 'freehand', 'point', 'circle') or data.get('source') in (
            'block_creation', 'cell_creation'):
        return False
    if 'show_vertices' in data:
        return bool(data['show_vertices'])
    # Drawing previews contain control vertices. Finished drawings are clean,
    # even when their geometry has many tessellation or interpolation points.
    return kind == 'drawing_preview' and dtype not in ('centerline', 'parallel')


class CanvasOverlay(QWidget):
    """Click-through tool window over one native canvas (a Qt widget cannot paint
    over the native Vulkan child window). Shapes are kept in section coordinates
    and re-projected on paint, so they follow pan and zoom."""

    def __init__(self, canvas):
        super().__init__(canvas.window(), Qt.Tool | Qt.FramelessWindowHint |
                         Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.canvas = canvas
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.shapes = []           # (kind, [(h, z)...], extra)
        self.marks = []            # persistent cut-slab lines [(h, z), (h, z)]
        self._content_revision = 0
        self._follow_state = None
        self._paint_projection = None
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._follow)
        backend = getattr(canvas, 'frame_backend', None)
        if backend is not None:
            import weakref
            overlays = getattr(backend, '_frame_overlays', None)
            if overlays is None:
                backend._frame_overlays = overlays = weakref.WeakSet()
            overlays.add(self)

    def set_shapes(self, shapes):
        self.shapes = list(shapes)
        self._changed()

    def set_marks(self, marks):
        self.marks = [list(m) for m in marks]
        self._changed()

    def _changed(self):
        self._content_revision += 1
        if self.shapes or self.marks:
            self._follow()
            self._timer.start()
        else:
            self._timer.stop()
            self.hide()
        self.update()

    def _follow(self, immediate=False):
        canvas = self.canvas
        if not (self.shapes or self.marks) or not canvas.isVisible() or canvas._closed:
            self.hide()
            return
        top = canvas.window()
        if self.parentWidget() is not top:
            # The dock floats as its own window: stay owned by (and above) it.
            self.setParent(top, self.windowFlags())
        top_left = canvas.mapToGlobal(canvas.rect().topLeft())
        if self.pos() != top_left:
            self.move(top_left)
        if self.size() != canvas.size():
            self.resize(canvas.size())
        getter = getattr(canvas, 'projection_snapshot', None)
        snapshot = getter() if getter is not None else None
        if snapshot is not None:
            key, project = snapshot
            state = (key, self._content_revision, top_left.x(), top_left.y(),
                     canvas.width(), canvas.height())
            self._paint_projection = project
            if state == self._follow_state and self.isVisible():
                return
            self._follow_state = state
        if not self.isVisible():
            self.show()
        if immediate:
            self.repaint()
        else:
            self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        to_qt = self._paint_projection or self.canvas.screen_from_section
        label_cells = {}
        painter.setPen(QPen(QColor(255, 165, 0), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        for a, b in self.marks:
            painter.drawLine(QPointF(*to_qt(*a)), QPointF(*to_qt(*b)))
        base_to_qt = to_qt
        world_planes = {}
        for kind, pts, extra in self.shapes:
            to_qt = base_to_qt
            world_z = extra.get("world_z") if kind == "drawing" and isinstance(extra, dict) else None
            world_projection = getattr(self.canvas, "world_projection_snapshot", None)
            if world_z is not None and world_projection is not None:
                if world_z not in world_planes:
                    world_planes[world_z] = world_projection(world_z)
                snapshot = world_planes[world_z]
                if snapshot is not None:
                    to_qt = snapshot[1]
            if kind == "brush_cursor_px":
                # Already in this overlay's Qt logical-pixel coordinates.  Keeping
                # the brush here avoids the section/world round trip that made the
                # Vulkan cursor lag or wobble behind the OS pointer.
                qpts = [QPointF(float(p[0]), float(p[1])) for p in pts]
            else:
                qpts = [QPointF(*to_qt(p[0], p[1])) for p in pts]
            painter.setPen(QPen(QColor(255, 255, 0), 1.5, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            if kind == 'measurement' and len(qpts) >= 2:
                options = extra or {}
                style = options.get('style', {})
                rgb = style.get('color', (0., 1., .47))
                color = QColor(*(max(0, min(255, round(float(v) * 255))) for v in rgb))
                # Selection is a transient overlay state: never mutate a
                # user's configured stroke/label color or the measured values.
                stroke_color = (QColor('#00ffff') if (options.get('highlight_scope') == 'measure'
                                and options.get('area') is None
                                and (options.get('selected') or options.get('hovered'))) else color)
                painter.setPen(QPen(stroke_color, max(1, int(style.get('width', 2)))))
                font = painter.font()
                font.setPointSize(max(7, int(style.get('label_font_size', 16))))
                painter.setFont(font)
                area = options.get('area')
                if area is not None:
                    # Picked SNT grid cell: closed outline, light fill, and the
                    # exact polygon area centred inside the cell.
                    fill = QColor(color)
                    fill.setAlpha(36)
                    painter.setBrush(fill)
                    painter.drawPolygon(QPolygonF(qpts))
                    painter.setBrush(color)
                    for point in qpts:
                        painter.drawEllipse(point, 3., 3.)
                    unit = style.get('unit', 'm')
                    text = (f'{float(area) / 1e6:.3f} km²' if unit == 'km'
                            else f'{float(area):,.2f} m²')
                    label = options.get('grid_label')
                    if label:
                        text = f'{label}: {text}'
                    bounds = QFontMetricsF(font).boundingRect(text)
                    centre = sum(qpts, QPointF()) / len(qpts)
                    painter.drawText(centre + QPointF(-bounds.width() / 2.,
                                                      bounds.height() / 2. - 2.),
                                     text)
                else:
                    # The measured segment(s) - and, while placing, the rubber band
                    # to the cursor - in the Measure Settings width and line style.
                    pen = painter.pen()
                    painter.setPen(QPen(stroke_color, max(1, int(style.get('width', 2))), {
                        'dashed': Qt.DashLine, 'dotted': Qt.DotLine,
                        'dash-dot': Qt.DashDotLine, 'dash-dot-dot': Qt.DashDotDotLine,
                    }.get(str(style.get('line_style', 'solid')), Qt.SolidLine)))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPolyline(QPolygonF(qpts))
                    painter.setPen(pen)
                    painter.setBrush(stroke_color)
                    for point in qpts:
                        painter.drawEllipse(point, 3., 3.)
                    # QPainter.drawText uses the PEN color, not the brush.
                    # Restore the configured measurement color for all labels.
                    painter.setPen(QPen(color, max(1, int(style.get('width', 2)))))
                    painter.setBrush(color)
                    # Line: a single sum at the final endpoint.
                    # Path: cumulative distances at P2, P3, ... final endpoint.
                    # Never label individual segment lengths as path totals.
                    is_path = bool(options.get('path'))
                    unit = style.get('unit', 'm')
                    for index, total in measurement_labels(pts, path=is_path):
                        label = measurement_text(total, unit)
                        if is_path and index == len(qpts) - 1:
                            label = f'Total {label}'
                        painter.drawText(qpts[index] + QPointF(8., -8.), label)
            elif kind in ('drawing', 'drawing_preview') and qpts:
                data = extra or {}
                dtype = str(data.get('type', 'polyline')).lower()
                preview = kind == 'drawing_preview'
                configured_color = QColor(str(data.get('color', '')))
                color = (QColor('#00ffff') if (data.get('highlight_scope') == 'draw'
                         and data.get('source') not in ('snt', 'attachment', 'snt_readonly',
                                                        'block_creation', 'cell_creation')
                         and (data.get('selected') or data.get('hovered'))) else
                         QColor(255, 80, 255) if data.get('selected') else
                         configured_color if configured_color.isValid() else
                         QColor(255, 255, 0) if preview else QColor(0, 220, 255))
                line_style = {'solid': Qt.SolidLine, 'dashed': Qt.DashLine,
                              'dotted': Qt.DotLine, 'dash-dot': Qt.DashDotLine,
                              'dash-dot-dot': Qt.DashDotDotLine}.get(
                                  data.get('style'), Qt.DashLine if preview else Qt.SolidLine)
                painter.setPen(QPen(color, max(1, int(data.get('width', 2))),
                                    line_style))
                painter.setBrush(Qt.NoBrush)
                if data.get('fill_enabled'):
                    fill = QColor(data.get('fill_color', color))
                    fill.setAlphaF(max(0., min(1., float(data.get('fill_opacity', .3)))))
                    painter.setBrush(fill)
                if dtype == 'circle' and len(qpts) >= 2:
                    r = math.hypot(qpts[1].x()-qpts[0].x(), qpts[1].y()-qpts[0].y())
                    painter.drawEllipse(qpts[0], r, r)
                elif dtype == 'rectangle' and len(qpts) >= 2:
                    painter.drawRect(QRectF(qpts[0], qpts[1]).normalized())
                elif dtype == 'text':
                    text = str(data.get('text', ''))
                    angle = -float(data.get('rotation', 0.) or 0.)
                    transform = QTransform().translate(qpts[0].x(), qpts[0].y()).rotate(angle)
                    font = QFont(painter.font())
                    # Native Text uses the VTK text palette's font settings;
                    # imported world-height SNT labels retain their own sizing.
                    if data.get('source') != 'attachment':
                        font.setFamily(str(data.get('font_family',font.family())))
                        if data.get('font_size') is not None:
                            font.setPixelSize(max(6, min(512, int(data['font_size']))))
                        font.setBold(bool(data.get('bold',False)))
                        font.setItalic(bool(data.get('italic',False)))
                    world_sized = data.get('source') == 'attachment' and float(data.get('height', 0.) or 0.) > 0.
                    if world_sized:
                        from gui.native_label_layout import world_text_transform
                        font = QFont(font)
                        font.setPixelSize(64)
                        metrics = QFontMetricsF(font)
                        transform = world_text_transform(to_qt, *pts[0], float(data['height']),
                                                         -angle, metrics.capHeight())
                        screen_height = math.hypot(transform.m21(), transform.m22()) * metrics.capHeight()
                        if screen_height < 2.:
                            continue
                    if data.get('source') == 'attachment':
                        rect = transform.mapRect(QFontMetricsF(font).boundingRect(text))
                        if not rect.intersects(QRectF(self.rect())):
                            continue
                        if not world_sized:
                            rect = rect.adjusted(-2., -2., 2., 2.)
                            cells = [(x, y) for x in range(math.floor(rect.left()/128), math.floor(rect.right()/128)+1)
                                     for y in range(math.floor(rect.top()/32), math.floor(rect.bottom()/32)+1)]
                            if any(rect.intersects(other) for cell in cells for other in label_cells.get(cell, ())):
                                continue
                            for cell in cells:
                                label_cells.setdefault(cell, []).append(rect)
                    painter.save()
                    painter.setFont(font)
                    painter.setWorldTransform(transform, True)
                    painter.drawText(QPointF(0., 0.), text)
                    painter.restore()
                elif dtype == 'point':
                    painter.setBrush(color)
                    painter.drawEllipse(qpts[0], 4., 4.)
                elif dtype == 'hatch' and len(qpts) >= 3:
                    # Unlike a translucent polygon fill, actual hatch lines
                    # are clipped to the closed boundary in world coordinates.
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPolygon(QPolygonF(qpts))
                    for a,b in data.get('hatch_segments', []):
                        painter.drawLine(QPointF(*to_qt(*a)), QPointF(*to_qt(*b)))
                elif len(qpts) >= 2:
                    if data.get('fill_enabled') and len(qpts) >= 3:
                        painter.drawPolygon(QPolygonF(qpts))
                    else:
                        painter.drawPolyline(QPolygonF(qpts))
                # Control handles belong to an ACTIVE placement or an
                # EXPLICIT Vertex Display/Edit mode, not to finished geometry.
                # Rectangle and circle use their two true control points, not
                # extra points on the tessellated perimeter. Curves use control
                # points rather than their interpolated samples. Avoid even
                # building the handle list when hidden (large CAD polylines).
                if _drawing_vertex_markers_visible(kind, data):
                    if dtype == 'rectangle' and len(qpts) >= 2:
                        a,b=qpts[0],qpts[1]
                        # The four visible rectangle corners are editable;
                        # the drawing record stores only two diagonal points.
                        vertex_points=[a,QPointF(b.x(),a.y()),b,QPointF(a.x(),b.y())]
                    elif dtype == 'curve' and data.get('control_points'):
                        vertex_points=[QPointF(*to_qt(*p)) for p in data['control_points']]
                    else:
                        vertex_points=qpts
                        if len(qpts)>2 and qpts[0]==qpts[-1]:
                            vertex_points=qpts[:-1]
                    for point in vertex_points:
                        painter.drawEllipse(point, 2.5, 2.5)
            elif kind == "cut_preview" and qpts:
                # Fallback for DLLs without the vector pass; same look as the
                # engine preview (white centre, orange depth lines, full height).
                half, cut_view = extra
                x0, h = qpts[0].x(), float(self.height())
                if half is not None:
                    dx = half / max(self.canvas.world_per_pixel(), 1e-300)
                    painter.setPen(QPen(QColor(255, 165, 0), 2))
                    for x in (x0 - dx, x0 + dx):
                        painter.drawLine(QPointF(x, 0.), QPointF(x, h))
                painter.setPen(QPen(QColor(255, 255, 255), 2 if cut_view else 1))
                painter.drawLine(QPointF(x0, 0.), QPointF(x0, h))
            elif kind == "stroke" and qpts:
                # The brush's painted trail: exactly the area brush_sweep selects.
                square, r, (color, _width) = extra
                fill = QColor(color)
                fill.setAlpha(110)
                painter.setPen(Qt.NoPen)
                painter.setBrush(fill)
                if square:
                    path = QPainterPath()
                    path.setFillRule(Qt.WindingFill)
                    pairs = [(qpts[0], qpts[0])] if len(qpts) == 1 else zip(qpts[:-1], qpts[1:])
                    for a, b in pairs:
                        corners = [QPointF(p.x() + sx * r, p.y() + sy * r)
                                   for p in (a, b) for sx in (-1, 1) for sy in (-1, 1)]
                        path.addPolygon(_convex_hull(corners))
                    painter.drawPath(path)
                elif len(qpts) == 1:
                    painter.drawEllipse(qpts[0], r, r)
                else:
                    pen = QPen(fill, 2 * r, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPolyline(QPolygonF(qpts))     # one path: uniform alpha
            elif kind == "brush_cursor_px" and qpts:
                square, r, (color, width) = extra
                width = max(1., float(width))
                pen = QPen(color, width, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                centre = qpts[0]
                if square:
                    painter.drawRect(QRectF(centre.x() - r, centre.y() - r, 2 * r, 2 * r))
                else:
                    painter.drawEllipse(centre, r, r)

                # Same visual language as the established brush cursor: a stable
                # crosshair and tiny centre ring make the actual stamp centre clear.
                guide = QPen(color, max(1., width * .75), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                guide.setCosmetic(True)
                painter.setPen(guide)
                ch = max(3., float(r) * .30)
                painter.drawLine(QPointF(centre.x() - ch, centre.y()),
                                 QPointF(centre.x() + ch, centre.y()))
                painter.drawLine(QPointF(centre.x(), centre.y() - ch),
                                 QPointF(centre.x(), centre.y() + ch))
                dot = max(1.5, float(r) * .06)
                painter.drawEllipse(centre, dot, dot)
            elif kind in ("rect", "polyline", "polygon", "circle_px", "guided_line"):
                # Classify previews use the tool's Classify Tools colour/width (VTK parity).
                color, width = extra[-1] if kind in ("circle_px", "guided_line") else extra
                pen = QPen(color, width)
                painter.setPen(pen)
                if kind == "rect" and len(qpts) == 2:
                    painter.drawRect(QRectF(qpts[0], qpts[1]).normalized())
                elif kind == "polyline" and len(qpts) >= 2:
                    painter.drawPolyline(QPolygonF(qpts))
                elif kind == "polygon" and len(qpts) >= 2:
                    painter.drawPolygon(QPolygonF(qpts))
                elif kind == "circle_px" and qpts:
                    square, r, _style = extra
                    if square:
                        painter.drawRect(QRectF(qpts[0].x() - r, qpts[0].y() - r, 2 * r, 2 * r))
                    else:
                        painter.drawEllipse(qpts[0], r, r)
                elif kind == "guided_line" and len(qpts) == 2:
                    # The line plus dashed verticals from both ends toward the side
                    # being classified (up for Above Line, down for Below Line).
                    painter.drawLine(qpts[0], qpts[1])
                    pen.setStyle(Qt.CustomDashLine)
                    pen.setDashPattern([8. / width, 6. / width])
                    painter.setPen(pen)
                    edge = 0. if extra[0] else float(self.height())
                    for q in qpts:
                        painter.drawLine(q, QPointF(q.x(), edge))
        painter.end()


def _class_lut(slots):
    """256-entry mask of stored class slots (see gui/class_slots.py)."""
    lut = np.zeros(256, dtype=np.bool_)
    for code in slots:
        try:
            code = int(code)
        except (TypeError, ValueError):
            continue
        if 0 <= code <= 255:
            lut[code] = True
    return lut


def _iter_native_classifiers(app):
    """Yield every live native Vulkan classifier exactly once.

    The native main/cross/cut canvases deliberately share classification state on
    ``app`` but own independent Qt overlays.  Brush preview lifecycle therefore
    has to be broadcast to every native canvas or a cursor left in a background
    view can remain visible after the stroke finishes elsewhere.
    """
    seen = set()
    view = getattr(app, "_lnt_main_view", None)
    classifier = getattr(view, "classifier", None) if view is not None else None
    if classifier is not None:
        seen.add(id(classifier))
        yield classifier

    sections = getattr(app, "_native_sections", None)
    for canvas in getattr(sections, "views", {}).values() if sections is not None else ():
        classifier = getattr(canvas, "classifier", None)
        if classifier is not None and id(classifier) not in seen:
            seen.add(id(classifier))
            yield classifier


def disarm_native_brush_preview(app, reason="inactive"):
    """Hide every native brush preview without deactivating the Brush tool.

    Brush selection and Brush *stroke preview* are intentionally separate states.
    Selecting Brush (shortcut/ribbon/right-click reactivation) leaves the tool
    persistent but visually idle.  The QPainter cursor exists only while the user
    is physically holding the left button for a live transactional stroke.
    Releasing or cancelling the stroke clears only this preview ownership; the
    selected Brush remains ready for the next left press.
    """
    was_armed = bool(getattr(app, "_native_brush_preview_armed", False))
    app._native_brush_preview_armed = False
    app._native_brush_preview_owner = None
    for classifier in _iter_native_classifiers(app):
        hide = getattr(classifier, "hide_brush_preview", None)
        if callable(hide):
            hide()
    if was_armed:
        print(f"NATIVE_BRUSH_PREVIEW state=hidden reason={reason}")


def cancel_native_brush_transaction(app, reason="tool_terminated"):
    """Roll back any in-progress native brush stroke, without history or disk writes.

    Native Vulkan brush painting is transactional: while the left button is held
    the canonical in-memory class array and the Vulkan/section previews are updated
    live, but LNT flushing is suspended.  Only a *real* left-button release turns
    that speculative paint into one undoable classification operation.  Any tool
    switch, Escape, view shutdown, or other interruption calls this helper first so
    partially painted points return to their exact pre-stroke classes.
    """
    cancelled = 0
    for classifier in _iter_native_classifiers(app):
        cancel = getattr(classifier, "cancel_pending_brush_transaction", None)
        if callable(cancel):
            try:
                cancelled += int(cancel(reason=reason) or 0)
            except Exception as exc:
                print(f"NATIVE_BRUSH_CANCEL status=error reason={reason} error={exc}")
    return cancelled


def notify_native_classification_activation(app, tool_name):
    """Publish an explicit native tool activation to all Vulkan canvases.

    Native shortcuts and the ribbon bypass the legacy VTK attach step.  This
    activation serial is the native equivalent: it lets a Brush->stroke->Brush
    sequence re-arm the cursor even when ``active_classify_tool`` has the same
    string value before and after the second activation.
    """
    native_view = getattr(app, "_lnt_main_view", None)
    sections = getattr(app, "_native_sections", None)
    if getattr(native_view, "native", None) is None and sections is None:
        return False

    # Switching/re-arming while a stroke is down is a cancellation boundary,
    # never an implicit commit.  Roll the speculative paint back *before* the
    # new tool state becomes visible to the canvases.
    cancel_native_brush_transaction(app, reason="tool_activation")

    serial = int(getattr(app, "_native_classify_activation_serial", 0) or 0) + 1
    app._native_classify_activation_serial = serial
    is_brush = tool_name == "brush"
    # Tool selection must never create a hover cursor. Brush is persistent, but
    # the preview belongs to an individual left-button stroke only.
    app._native_brush_preview_armed = False
    app._native_brush_preview_owner = None

    for classifier in _iter_native_classifiers(app):
        activated = getattr(classifier, "external_tool_activated", None)
        if callable(activated):
            activated(tool_name, serial)

    if is_brush:
        print(
            "NATIVE_BRUSH_PREVIEW state=idle trigger=tool_selected "
            f"backend=qpainter trail=0 activation={serial}"
        )
    return True


class SectionClassifier:
    """Mouse gesture state machine for one canvas; reads the app's active tool,
    From/To classes and brush settings on every event (no attach step)."""

    def __init__(self, canvas):
        self.canvas = canvas
        self.app = canvas.owner.app
        self.overlay = CanvasOverlay(canvas)
        self.points = []           # gesture vertices in section coordinates
        self.press_px = None
        self.dragging = False
        # Hybrid tap-tap / drag arbitration for rectangle, circle and line tools.
        # ``dragging`` only means the left button is currently down; the gesture
        # does not become a true drag until ``_two_point_drag_latched`` flips.
        self._press_time = 0.0
        self._press_max_move_px = 0.0
        self._two_point_drag_latched = False
        self.hover = None
        # Cursor preview is kept in Qt logical pixels so the circle is locked
        # exactly to the mouse.  Classification still uses float64 section/world
        # coordinates; only the visual cursor bypasses world->screen round trips.
        self.hover_px = None
        self._tool = None
        self._activation_serial = int(getattr(self.app, "_native_classify_activation_serial", 0) or 0)
        self._brush_cursor_logged = False
        # One active native brush stroke is a transaction.  Points are changed
        # in app.data live for immediate visual feedback, but the store's LNT
        # writer is blocked until release.  These arrays preserve the *first*
        # class value touched for exact rollback/undo.
        self._brush_tx_active = False
        self._brush_tx_seen = None
        self._brush_tx_indices = []
        self._brush_tx_old_classes = []
        self._brush_tx_frame_indices = []
        self._brush_tx_frame_old = []
        self._brush_tx_to_slot = None
        self._brush_tx_to_code = None
        self._brush_tx_from_slots = None
        self._brush_tx_from_user = None
        self._brush_tx_last_preview = 0.0
        self._brush_tx_changed = 0

    # ---- state ---------------------------------------------------------------
    def tool(self):
        tool = getattr(self.app, "active_classify_tool", None)
        if tool in LINE_TOOLS and not getattr(self.canvas, "line_tools", True):
            return None            # plan view: line tools are section-only (as in VTK)
        return tool if tool in DRAWABLE_TOOLS else None

    def active(self):
        return self.tool() is not None

    def busy(self):
        return bool(self.points)

    def cancel(self):
        if self._brush_tx_active:
            self.cancel_pending_brush_transaction(reason="gesture_cancel")
        self.points, self.press_px, self.dragging = [], None, False
        self._press_time = 0.0
        self._press_max_move_px = 0.0
        self._two_point_drag_latched = False
        self._paint()

    def shutdown(self):
        if self._brush_tx_active:
            self.cancel_pending_brush_transaction(reason="view_shutdown")
        self.cancel()
        self.overlay._timer.stop()
        self.overlay.close()

    def _brush_view_name(self):
        if not getattr(self.canvas, "line_tools", True):
            return "main"
        try:
            from gui.native_sections import CUT_SLOT
            return "cut" if getattr(self.canvas, "slot", None) == CUT_SLOT else "section"
        except Exception:
            return "section"

    def hide_brush_preview(self):
        """Remove brush-only shapes immediately; preserve unrelated overlay marks."""
        self.hover_px = None
        shapes = [shape for shape in self.overlay.shapes
                  if shape[0] not in ("stroke", "circle_px", "brush_cursor_px")]
        if len(shapes) != len(self.overlay.shapes):
            self.overlay.set_shapes(shapes)

    def external_tool_activated(self, tool_name, serial=None):
        """Reset transient gesture state on an explicit shortcut/ribbon activation."""
        if self._brush_tx_active:
            self.cancel_pending_brush_transaction(reason="tool_switch")
        self.points = []
        self.press_px = None
        self.dragging = False
        self._press_time = 0.0
        self._press_max_move_px = 0.0
        self._two_point_drag_latched = False
        self.hover = None
        self.hover_px = None
        self._tool = tool_name if tool_name in DRAWABLE_TOOLS else None
        self._activation_serial = int(serial or 0)
        self._brush_cursor_logged = False
        # Never carry a preview from the old tool into the newly armed tool.
        if self.overlay.shapes:
            self.overlay.set_shapes([])
        # Brush selection is intentionally preview-free. The first visible
        # cursor is created by the next physical left-button press, never by a
        # shortcut, ribbon click, ClassPicker update, or right-click reactivation.

    def _sync_tool(self):
        tool = self.tool()
        if tool != self._tool:
            self._tool = tool
            self.cancel()
        return tool

    # ---- events --------------------------------------------------------------
    def press(self, pos, button):
        tool = self._sync_tool()
        if tool is None and button == Qt.RightButton:
            remembered = getattr(self.app, 'last_classify_tool', None)
            if remembered in DRAWABLE_TOOLS and (
                    remembered not in LINE_TOOLS or getattr(self.canvas, 'line_tools', True)):
                # Recall an idle tool without resurrecting partial gestures in
                # another section. Measurement and cut tools consume their own clicks.
                manager = getattr(self.app, '_native_sections', self.canvas.owner)
                for canvas in getattr(manager, 'views', {}).values():
                    canvas.classifier.cancel()
                self.cancel()
                previous = getattr(self.app, '_right_click_reactivating', False)
                self.app._right_click_reactivating = True
                try:
                    saved = getattr(self.app, 'last_classify_from_classes', None)
                    self.app.from_classes = list(saved) if saved is not None else None
                    self.app.to_class = getattr(self.app, 'last_classify_to_class', None)
                    self.app.set_classify_tool(remembered)
                finally:
                    self.app._right_click_reactivating = previous
                return True
        if tool is None:
            if (button == Qt.LeftButton and not getattr(self.canvas, "line_tools", True)
                    and getattr(self.app, "active_classify_tool", None) in LINE_TOOLS):
                self.app.statusBar().showMessage(
                    "Line classification tools work only in cross-section views", 3000)
            return False
        if button == Qt.RightButton:
            if tool == "polygon" and len(self.points) >= 3:
                self._finish_polygon()
                return True
            if self.points:
                self.cancel()
                return True
            if tool == "brush":
                # Persistent Brush owns its idle right-click so it cannot fall
                # through into Locate/Cut/context handlers. Right-click is not
                # a preview trigger; only a left press starts a Brush stroke.
                return True
            return False
        if button != Qt.LeftButton:
            return False
        if not self.points:
            self.app.last_classify_tool = tool
            source = getattr(self.app, 'from_classes', None)
            self.app.last_classify_from_classes = list(source) if source is not None else None
            self.app.last_classify_to_class = getattr(self.app, 'to_class', None)
        pt = self.canvas.section_at(pos.x(), pos.y())
        self.press_px = (pos.x(), pos.y())
        self._press_time = time.monotonic()
        self._press_max_move_px = 0.0
        self._two_point_drag_latched = False
        self.hover = pt
        self.hover_px = (float(pos.x()), float(pos.y()))
        if tool in ("brush", "freehand"):
            self.points, self.dragging = [pt], True
            if tool == "brush":
                # The tool may remain selected indefinitely, but preview ownership
                # starts only on this physical left press. Hide every other native
                # canvas so only the stroke's viewport can draw a cursor.
                self.app._native_brush_preview_armed = True
                self.app._native_brush_preview_owner = id(self)
                self._brush_cursor_logged = False
                for classifier in _iter_native_classifiers(self.app):
                    if classifier is not self:
                        classifier.hide_brush_preview()
                self._paint()
                if not self._begin_brush_transaction(pt):
                    self.points, self.press_px, self.dragging = [], None, False
                    disarm_native_brush_preview(self.app, reason="stroke_begin_failed")
                    self._paint()
                    return True
                print(
                    "NATIVE_BRUSH_STROKE status=begin mode=live_transaction "
                    f"view={self._brush_view_name()} radius_px={self.brush_radius_px():.2f} "
                    "tool=persistent"
                )
        elif tool == "point":
            self._commit(tool, [pt, pt])
        elif tool == "polygon":
            if len(self.points) >= 3:
                first_px = self.canvas.screen_from_section(*self.points[0])
                if np.hypot(pos.x() - first_px[0], pos.y() - first_px[1]) <= _CLICK_SLOP_PX:
                    self._finish_polygon()
                    return True
            self.points.append(pt)
        elif tool in _TWO_CLICK:
            if not self.points:
                # First physical press starts a *candidate* gesture only.  Tap is
                # the priority: unless a deliberate drag is latched while this
                # button is held, release leaves P1 armed and the rubber-band
                # preview follows the cursor until the confirming second tap.
                self.points, self.dragging = [pt], True
            else:
                # Fast tap-tap finalizes on the second PRESS.  Doing this on press
                # (rather than waiting for release) prevents OS/Qt double-click
                # coalescing from dropping the confirming tap.
                changed = self._commit(tool, [self.points[0], pt])
                self.points = []
                self.dragging = False
                self._press_max_move_px = 0.0
                self._two_point_drag_latched = False
                print(
                    "NATIVE_CLASSIFY_GESTURE mode=tap_finalize "
                    f"tool={tool} changed={int(changed or 0)}"
                )
        self._paint()
        return True

    def move(self, pos):
        tool = self._sync_tool()
        self.hover_px = (float(pos.x()), float(pos.y()))
        self.hover = self.canvas.section_at(pos.x(), pos.y())
        if tool is None:
            if self.overlay.shapes:
                self.overlay.set_shapes([])
            return False
        if tool in _TWO_CLICK and self.dragging and self.press_px is not None:
            # Track peak motion at native mouse-event cadence.  Movement inside the
            # settle window is ignored, so hand jitter from a fast click cannot
            # silently convert tap-tap into drag mode.  Once the threshold is
            # crossed after the grace period the gesture is permanently latched as
            # a drag, even if the cursor later returns near P1.
            if (time.monotonic() - self._press_time) >= _TAP_GRACE_SEC:
                distance = float(np.hypot(pos.x() - self.press_px[0],
                                          pos.y() - self.press_px[1]))
                if distance > self._press_max_move_px:
                    self._press_max_move_px = distance
                if (not self._two_point_drag_latched
                        and self._press_max_move_px >= _TAP_DRAG_THRESHOLD_PX):
                    self._two_point_drag_latched = True
                    print(
                        "NATIVE_CLASSIFY_GESTURE mode=drag_latched "
                        f"tool={tool} moved_px={self._press_max_move_px:.1f}"
                    )
        if tool == "brush" and self.dragging and self.points:
            # Sample the stroke every quarter brush radius (at least 2 px): the sweep
            # between samples is exact, so this only bounds the work, not the coverage.
            last = self.canvas.screen_from_section(*self.points[-1])
            if np.hypot(pos.x() - last[0], pos.y() - last[1]) >= max(2., .25 * self.brush_radius_px()):
                previous = self.points[-1]
                self.points.append(self.hover)
                self._apply_live_brush_segment(previous, self.hover)
        elif tool == "freehand" and self.dragging:
            last = self.canvas.screen_from_section(*self.points[-1])
            if np.hypot(pos.x() - last[0], pos.y() - last[1]) >= 2:
                self.points.append(self.hover)
        self._paint()
        return self.busy()

    def release(self, pos, button):
        tool = self.tool()
        if tool is None or button != Qt.LeftButton:
            return False
        pt = self.canvas.section_at(pos.x(), pos.y())
        if tool == "brush" and self.dragging:
            self.dragging = False
            # Include the physical release position even when the last move was
            # below the sampling threshold, then atomically turn the speculative
            # live paint into one history entry + one delayed LNT write.
            if self.points:
                previous = self.points[-1]
                if previous != pt:
                    self.points.append(pt)
                    self._apply_live_brush_segment(previous, pt, force_preview=True)
            changed = self._finalize_brush_transaction()
            stroke_samples = len(self.points)
            self.points = []
            # A completed stroke hides its cursor/trail but keeps Brush selected.
            # The very next left press starts another independent transaction; no
            # shortcut/ribbon reactivation is required.
            disarm_native_brush_preview(self.app, reason="stroke_complete")
            print(
                "NATIVE_BRUSH_STROKE status=commit "
                f"view={self._brush_view_name()} changed={int(changed or 0)} "
                f"samples={stroke_samples} preview=hidden tool=persistent"
            )
        elif tool == "freehand" and self.dragging:
            self.dragging = False
            shape, self.points = self.points + [pt], []
            self._commit(tool, shape)
        elif tool in _TWO_CLICK and self.dragging:
            self.dragging = False
            held_sec = max(0.0, time.monotonic() - self._press_time)
            release_move = (float(np.hypot(pos.x() - self.press_px[0],
                                           pos.y() - self.press_px[1]))
                            if self.press_px is not None else 0.0)
            # Exact parity with the proven VTK hybrid input model: a fast click's
            # settle/jitter window is always treated as a tap.  Afterwards only a
            # real threshold crossing opts into drag-to-drag behavior.
            moved_px = 0.0 if held_sec < _TAP_GRACE_SEC else max(
                release_move, self._press_max_move_px
            )
            is_drag = bool(
                self._two_point_drag_latched
                or moved_px >= _TAP_DRAG_THRESHOLD_PX
            )
            if is_drag and self.points:
                changed = self._commit(tool, [self.points[0], pt])
                self.points = []
                print(
                    "NATIVE_CLASSIFY_GESTURE mode=drag_finalize "
                    f"tool={tool} moved_px={moved_px:.1f} "
                    f"changed={int(changed or 0)}"
                )
            elif self.points:
                # TAP #1: keep P1, clear button-down state, and leave the live
                # rubber-band preview active.  TAP #2 will finalize in press().
                print(
                    "NATIVE_CLASSIFY_GESTURE mode=tap_anchor "
                    f"tool={tool} held_ms={held_sec * 1000.0:.0f} "
                    f"moved_px={moved_px:.1f}"
                )
            self._press_max_move_px = 0.0
            self._two_point_drag_latched = False
        self.press_px = None
        self._press_time = 0.0
        self._paint()
        return True

    def double_click(self, pos):
        tool = self._sync_tool()
        if tool == "polygon" and len(self.points) >= 3:
            self._finish_polygon()
            return True
        if tool in _TWO_CLICK:
            # Qt can replace the second MouseButtonPress of a very fast pair with
            # MouseButtonDblClick.  If TAP #1 is armed, route that event through
            # the exact same second-tap finalizer.  If the normal press already
            # finalized it, simply swallow the double-click so it cannot start a
            # new accidental gesture or reach window/camera handlers.
            if self.points and not self.dragging:
                print(f"NATIVE_CLASSIFY_GESTURE mode=double_click_bridge tool={tool}")
                self.press(pos, Qt.LeftButton)
            return True
        # Point is already a single-tap tool.  Swallow a Qt double-click so one
        # physical double-click cannot leak into another tool/window handler.
        if tool == "point":
            return True
        return False

    def key(self, key):
        if key in (Qt.Key_Return, Qt.Key_Enter) and self.tool() == "polygon" and len(self.points) >= 3:
            self._finish_polygon()
            return True
        if key == Qt.Key_Escape and self.points:
            self.cancel()
            return True
        return False

    def _finish_polygon(self):
        """Commit the current closed polygon and remove its live preview."""
        shape, self.points = list(self.points), []
        self.press_px = None
        self.dragging = False
        self._commit("polygon", shape)
        self._paint()

    # ---- preview -------------------------------------------------------------
    def brush_radius_px(self):
        try:
            return max(1., float(getattr(self.app, "brush_preview_px", 20.)))
        except (TypeError, ValueError):
            return 20.

    # ---- native live-brush transaction --------------------------------------
    def _clear_brush_transaction_state(self):
        self._brush_tx_active = False
        self._brush_tx_seen = None
        self._brush_tx_indices = []
        self._brush_tx_old_classes = []
        self._brush_tx_frame_indices = []
        self._brush_tx_frame_old = []
        self._brush_tx_to_slot = None
        self._brush_tx_to_code = None
        self._brush_tx_from_slots = None
        self._brush_tx_from_user = None
        self._brush_tx_last_preview = 0.0
        self._brush_tx_changed = 0

    def _release_brush_transaction_lock(self):
        app = self.app
        if getattr(app, "_native_brush_transaction_owner", None) == id(self):
            app._native_brush_transaction_active = False
            app._native_brush_transaction_owner = None

    def _resolve_brush_transaction_classes(self):
        """Resolve public class numbers once at stroke start and keep them stable."""
        app = self.app
        to_code = getattr(app, "to_class", None)
        if to_code is None:
            app.statusBar().showMessage("Target class not set.", 2500)
            return None
        from gui.class_slots import NoFreeSlot, slot_for, slots_for
        try:
            to_slot = slot_for(app, to_code, allocate=True)
        except (NoFreeSlot, OSError) as exc:
            app.statusBar().showMessage(str(exc), 6000)
            return None
        if to_slot is None:
            app.statusBar().showMessage(
                f"Class {to_code} is not a valid class number.", 4000
            )
            return None
        from_user = getattr(app, "from_classes", None)
        from_slots = None
        if from_user:
            from_slots = np.asarray(slots_for(app, from_user), dtype=np.int64)
        return int(to_slot), to_code, from_slots, list(from_user or [])

    def _begin_brush_transaction(self, first_point):
        """Begin speculative live painting; no disk write/history exists yet."""
        resolved = self._resolve_brush_transaction_classes()
        if resolved is None:
            return False

        # Only one native Vulkan brush transaction may own the shared class
        # array at a time.  This also protects against a stale drag in a hidden
        # cross/cut window when the user starts in another viewport.
        owner = getattr(self.app, "_native_brush_transaction_owner", None)
        if bool(getattr(self.app, "_native_brush_transaction_active", False)) and owner != id(self):
            cancel_native_brush_transaction(self.app, reason="new_brush_owner")

        # Make every previously *committed* classification the transaction's
        # baseline before we suppress the store's periodic writer.
        try:
            from gui.lnt_section_bridge import get_store
            store = get_store(self.app)
            if store is not None:
                store.flush()
        except Exception as exc:
            print(f"NATIVE_BRUSH_BASELINE status=flush_warning error={exc}")

        self._clear_brush_transaction_state()
        (self._brush_tx_to_slot, self._brush_tx_to_code,
         self._brush_tx_from_slots, self._brush_tx_from_user) = resolved
        self._brush_tx_active = True
        self.app._native_brush_transaction_active = True
        self.app._native_brush_transaction_owner = id(self)

        data = getattr(self.app, "data", None)
        if not isinstance(data, dict) or "classification" not in data:
            self._release_brush_transaction_lock()
            self._clear_brush_transaction_state()
            try:
                self.app.statusBar().showMessage(
                    "Native classification store is not available.", 3500
                )
            except Exception:
                pass
            return False
        classes = np.asarray(data["classification"])
        self._brush_tx_seen = np.zeros(len(classes), dtype=np.bool_)

        # A mouse-down is a real brush stamp.  Paint it immediately so the user
        # never sees a cursor moving over points that wait for release.
        self._apply_live_brush_segment(first_point, first_point, force_preview=True)
        if self._brush_tx_active:
            print(
                "NATIVE_BRUSH_LIVE status=enabled transaction=speculative "
                "commit=left_release cancel=rollback "
                f"view={self._brush_view_name()} "
                f"display_mode={getattr(self.app, 'display_mode', 'class')}"
            )
        return bool(self._brush_tx_active)

    def _grow_brush_seen(self, size):
        seen = self._brush_tx_seen
        if seen is None:
            self._brush_tx_seen = np.zeros(int(size), dtype=np.bool_)
            return self._brush_tx_seen
        if len(seen) < size:
            grown = np.zeros(int(size), dtype=np.bool_)
            grown[:len(seen)] = seen
            self._brush_tx_seen = grown
        return self._brush_tx_seen

    def _flush_live_brush_preview(self, force=False):
        """Push accumulated speculative classes to Vulkan at ~30 FPS.

        The QPainter cursor is updated on every mouse move.  Class-stream/GPU
        updates are batched to keep motion smooth.  This method never schedules
        an LNT write; the store timer is also blocked by the transaction flag.
        """
        if not self._brush_tx_active or not self._brush_tx_frame_indices:
            return 0
        now = time.perf_counter()
        if not force and (now - self._brush_tx_last_preview) < (1.0 / 30.0):
            return 0

        if len(self._brush_tx_frame_indices) == 1:
            indices = self._brush_tx_frame_indices[0]
            old = self._brush_tx_frame_old[0]
        else:
            indices = np.concatenate(self._brush_tx_frame_indices)
            old = np.concatenate(self._brush_tx_frame_old)
        self._brush_tx_frame_indices = []
        self._brush_tx_frame_old = []
        self._brush_tx_last_preview = now

        try:
            from gui.lnt_section_bridge import get_store
            store = get_store(self.app)
            if store is not None:
                store.preview_live(indices, old)

            # Cross/cut renderers hold their own resident class stream.  Update
            # only attributes (never XYZ/camera/display mode), preserving RGB,
            # intensity, elevation, depth, shading and surface/line behaviour.
            sections = getattr(self.app, "_native_sections", None)
            if sections is not None:
                sections.refresh_edits(force=True)
        except Exception as exc:
            print(f"NATIVE_BRUSH_LIVE status=preview_warning error={exc}")
        return int(indices.size)

    def _apply_live_brush_segment(self, start, end, force_preview=False):
        """Classify only newly touched records for one swept cursor segment."""
        if not self._brush_tx_active:
            return 0
        try:
            indices = self.candidates("brush", [start, end])
            if not indices.size:
                self._flush_live_brush_preview(force=force_preview)
                return 0

            classes = np.asarray(self.app.data["classification"])
            seen = self._grow_brush_seen(len(classes))
            # Native source identities are unique, but the swept segments overlap.
            # A point's pre-stroke class is captured exactly once.
            indices = np.asarray(indices, dtype=np.int64)
            indices = indices[~seen[indices]]
            if not indices.size:
                self._flush_live_brush_preview(force=force_preview)
                return 0

            if self._brush_tx_from_slots is not None and self._brush_tx_from_slots.size:
                indices = indices[_class_lut(self._brush_tx_from_slots)[classes[indices]]]
            if indices.size:
                indices = indices[classes[indices] != np.asarray(
                    self._brush_tx_to_slot, dtype=classes.dtype
                )]
            if not indices.size:
                self._flush_live_brush_preview(force=force_preview)
                return 0

            old = classes[indices].copy()
            seen[indices] = True
            classes[indices] = np.asarray(self._brush_tx_to_slot, dtype=classes.dtype)
            self._brush_tx_indices.append(indices.copy())
            self._brush_tx_old_classes.append(old)
            self._brush_tx_frame_indices.append(indices.copy())
            self._brush_tx_frame_old.append(old.copy())
            self._brush_tx_changed += int(indices.size)
            self._flush_live_brush_preview(force=force_preview)
            return int(indices.size)
        except Exception as exc:
            print(f"NATIVE_BRUSH_LIVE status=error error={exc}")
            self.cancel_pending_brush_transaction(reason="live_error")
            return 0

    def _brush_transaction_arrays(self):
        if not self._brush_tx_indices:
            return (np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.uint8))
        if len(self._brush_tx_indices) == 1:
            return self._brush_tx_indices[0], self._brush_tx_old_classes[0]
        return (np.concatenate(self._brush_tx_indices),
                np.concatenate(self._brush_tx_old_classes))

    def _finalize_brush_transaction(self):
        """Commit live paint on physical release as one undo/write operation."""
        if not self._brush_tx_active:
            return 0
        self._flush_live_brush_preview(force=True)
        indices, old = self._brush_transaction_arrays()
        app = self.app
        to_slot = self._brush_tx_to_slot
        to_code = self._brush_tx_to_code
        from_user = list(self._brush_tx_from_user or [])

        try:
            if indices.size == 0:
                return 0
            classes = np.asarray(app.data["classification"])
            new = np.full(indices.size, to_slot, dtype=classes.dtype)

            stack = getattr(app, "undo_stack", None)
            if stack is None:
                stack = app.undo_stack = []
            stack.append({
                "indices": indices.copy(),
                "old_classes": old.copy(),
                "new_classes": new,
            })
            redo = getattr(app, "redo_stack", None)
            if redo is not None:
                redo.clear()
            from gui.memory_manager import trim_undo_stack
            trim_undo_stack(app)

            app._last_changed_indices = indices.copy()
            app._last_from_classes = from_user
            app._last_classify_ts = time.time()
            app.classification_revision = int(
                getattr(app, "classification_revision", 0) or 0
            ) + 1

            # The speculative interval ends *before* refresh_edits(), so the
            # established store.show()->delayed flush path can persist this one
            # successful stroke.  It never ran while the button was down.
            self._release_brush_transaction_lock()
            try:
                self.canvas.owner.refresh_edits(force=True, changed=(indices, old))
            except Exception as exc:
                # The class transaction is already committed in app.data and
                # history.  Never let a renderer-refresh exception escape a Qt
                # mouse callback; the store timer can still persist the edit.
                print(f"NATIVE_BRUSH_COMMIT status=refresh_warning error={exc}")
            app.statusBar().showMessage(
                f"{indices.size:,} points -> class {to_code}", 3000
            )
            return int(indices.size)
        finally:
            self._release_brush_transaction_lock()
            self._clear_brush_transaction_state()

    def cancel_pending_brush_transaction(self, reason="cancel"):
        """Restore an unfinished live stroke exactly; add no undo and write nothing."""
        if not self._brush_tx_active:
            return 0
        app = self.app
        indices, old = self._brush_transaction_arrays()
        restored = int(indices.size)
        try:
            if indices.size:
                classes = np.asarray(app.data["classification"])
                current = classes[indices].copy()
                classes[indices] = old
                try:
                    from gui.lnt_section_bridge import get_store
                    store = get_store(app)
                    if store is not None:
                        # Reverse the speculative Vulkan colours/classes without
                        # scheduling a write. app.data is already back to baseline.
                        store.preview_live(indices, current)
                    sections = getattr(app, "_native_sections", None)
                    if sections is not None:
                        sections.refresh_edits(force=True)
                except Exception as exc:
                    print(f"NATIVE_BRUSH_CANCEL status=preview_warning error={exc}")
        finally:
            self._release_brush_transaction_lock()
            self._clear_brush_transaction_state()
            self.points = []
            self.press_px = None
            self.dragging = False
            disarm_native_brush_preview(app, reason=reason)
            try:
                self.overlay.set_shapes([])
            except Exception:
                pass

        print(
            "NATIVE_BRUSH_CANCEL status=rolled_back "
            f"view={self._brush_view_name()} restored={restored} reason={reason}"
        )
        return restored

    def _paint(self):
        tool, hover, pts = self.tool(), self.hover, self.points
        shapes = []
        if tool is None or hover is None:
            self.overlay.set_shapes([])
            return
        style = line_style(tool)
        if tool == "brush":
            # QPainter cursor only (trail=0), no Vulkan/VTK render per mouse move.
            # Tool selection is not a preview state: the cursor exists only while
            # this classifier owns an active left-button stroke.
            preview_owned = (
                bool(getattr(self.app, "_native_brush_preview_armed", False))
                and getattr(self.app, "_native_brush_preview_owner", None) == id(self)
                and self.dragging
            )
            if not preview_owned:
                if self.overlay.shapes:
                    self.hide_brush_preview()
                return
            if self.hover_px is None:
                if self.overlay.shapes:
                    self.overlay.set_shapes([])
                return
            square = str(getattr(self.app, "brush_shape", "circle")).lower() == "rectangle"
            shapes.append(("brush_cursor_px", [self.hover_px],
                           (square, self.brush_radius_px(), style)))
            if not self._brush_cursor_logged:
                self._brush_cursor_logged = True
                print(
                    "PHASE2B4_BRUSH_CURSOR status=enabled backend=qpainter "
                    f"view={self._brush_view_name()} mode=cursor_locked "
                    "trail=0 vtk_render_per_move=0"
                )
        elif tool == "point":
            shapes.append(("circle_px", [hover], (False, 4., style)))
        elif pts:
            if tool == "rectangle":
                shapes.append(("rect", [pts[0], hover], style))
            elif tool == "circle":
                r = float(np.hypot(hover[0] - pts[0][0], hover[1] - pts[0][1]))
                ring = [(pts[0][0] + r * np.cos(a), pts[0][1] + r * np.sin(a))
                        for a in np.linspace(0, 2 * np.pi, 64)]
                shapes.append(("polyline", ring, style))
            elif tool in ("above_line", "below_line"):
                shapes.append(("guided_line", [pts[0], hover], (tool == "above_line", style)))
            elif tool == "parallel_line":
                shapes.append(("polyline", [pts[0], hover], style))
                shapes.extend(("polyline", edge, style) for edge in parallel_band_edges(
                    pts[0], hover, *parallel_tolerances(self.app)))
            elif tool == "polygon":
                shapes.append(("polygon", list(pts) + [hover], style))
            elif tool == "freehand":
                shapes.append(("polyline", list(pts) + [hover], style))
        self.overlay.set_shapes(shapes)

    # ---- commit --------------------------------------------------------------
    def _visible_classes(self):
        palette = self.canvas.palette or {}
        if not palette:
            return None
        from gui.class_slots import stored_palette
        return np.array([slot for slot, rule in stored_palette(self.app, palette).items()
                         if rule.get("show", True)], dtype=np.int64)

    def candidates(self, tool, shape):
        """Store indices this gesture would classify (before the From filter)."""
        canvas = self.canvas
        radius = None
        if tool == "brush":
            radius = self.brush_radius_px() * canvas.world_per_pixel()
        elif tool == "point":
            radius = 4. * canvas.world_per_pixel()
        prepare = getattr(canvas, 'prepare_selection', None)
        if prepare is not None:
            try:
                prepare(tool, shape, radius)
            except RuntimeError as exc:
                self.app.statusBar().showMessage(str(exc), 5000)
                return np.empty(0, dtype=np.int64)
        if not len(canvas.indices):
            return np.empty(0, dtype=np.int64)
        points = getattr(canvas, "selection_points", None)
        pts2d = points() if points is not None else canvas.local[:, [canvas.axis, 2]]
        if getattr(canvas, "exact_selection", None) == tool:
            mask = np.ones(len(pts2d), dtype=bool)   # the native query applied the shape
        else:
            mask = select(tool, pts2d, shape, self.app, radius)
        # Only what the view shows: inside the viewport, visible class.
        lo_h, hi_h, lo_z, hi_z = canvas.visible_extent()
        mask &= (pts2d[:, 0] >= lo_h) & (pts2d[:, 0] <= hi_h) & (pts2d[:, 1] >= lo_z) & (pts2d[:, 1] <= hi_z)
        indices = canvas.indices[mask]
        visible = self._visible_classes()
        if visible is not None and indices.size:
            classes = self.app.data["classification"]
            indices = indices[_class_lut(visible)[classes[indices]]]
        return indices

    def _commit(self, tool, shape):
        app = self.app
        to_class = getattr(app, "to_class", None)
        if to_class is None:
            app.statusBar().showMessage("Target class not set.", 2500)
            return 0
        # User class numbers can be anything (e.g. 999); points store a one-byte
        # slot (gui/class_slots.py). Never raise here: an exception escaping a
        # Qt mouse override crashes the app.
        from gui.class_slots import NoFreeSlot, slot_for, slots_for
        to_code = to_class
        try:
            to_class = slot_for(app, to_code, allocate=True)
        except (NoFreeSlot, OSError) as exc:
            app.statusBar().showMessage(str(exc), 6000)
            return 0
        if to_class is None:
            app.statusBar().showMessage(f"Class {to_code} is not a valid class number.", 4000)
            return 0
        indices = self.candidates(tool, shape)
        if not indices.size:
            return 0
        classes = app.data["classification"]
        from_classes = getattr(app, "from_classes", None)
        if from_classes and indices.size:
            indices = indices[_class_lut(slots_for(app, from_classes))[classes[indices]]]
        if indices.size:
            indices = indices[classes[indices] != np.asarray(to_class, dtype=classes.dtype)]
        if indices.size == 0:
            app.statusBar().showMessage("No visible/from-class points in selection.", 2500)
            return 0
        # Native queries contain one row per exact source identity and masks
        # preserve that uniqueness.  np.unique sorted millions of already
        # unique rows on every commit.
        indices = np.asarray(indices, dtype=np.int64)
        old = classes[indices].copy()
        new = np.full(indices.size, to_class, dtype=classes.dtype)
        stack = getattr(app, "undo_stack", None)
        if stack is None:
            stack = app.undo_stack = []
        stack.append({"indices": indices, "old_classes": old, "new_classes": new})
        redo = getattr(app, "redo_stack", None)
        if redo is not None:
            redo.clear()
        from gui.memory_manager import trim_undo_stack
        trim_undo_stack(app)
        classes[indices] = to_class
        app._last_changed_indices = indices.copy()
        app._last_from_classes = list(from_classes or [])
        app._last_classify_ts = time.time()
        app.classification_revision = int(getattr(app, "classification_revision", 0) or 0) + 1
        self.canvas.owner.refresh_edits(force=True, changed=(indices, old))
        app.statusBar().showMessage(f"{indices.size:,} points -> class {to_code}", 3000)
        return int(indices.size)
