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
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QPolygonF, QTransform
from PySide6.QtWidgets import QWidget
from gui.native_measure_geometry import measurement_labels, measurement_text

DRAWABLE_TOOLS = ("rectangle", "circle", "polygon", "freehand", "brush", "point",
                  "above_line", "below_line", "parallel_line")
_TWO_CLICK = ("rectangle", "circle", "above_line", "below_line", "parallel_line")
LINE_TOOLS = ("above_line", "below_line", "parallel_line")
_CLICK_SLOP_PX = 4.0


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
            elif kind == "cut_preview" and len(qpts) == 2:
                painter.setPen(QPen(QColor(255, 165, 0), 2, Qt.DashLine))
                painter.drawRect(QRectF(qpts[0], qpts[1]).normalized())
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
        self.hover = None
        self._tool = None

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
        self.points, self.press_px, self.dragging = [], None, False
        self._paint()

    def shutdown(self):
        self.cancel()
        self.overlay._timer.stop()
        self.overlay.close()

    def _sync_tool(self):
        tool = self.tool()
        if tool != self._tool:
            self._tool = tool
            self.cancel()
        return tool

    # ---- events --------------------------------------------------------------
    def press(self, pos, button):
        tool = self._sync_tool()
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
            return False
        if button != Qt.LeftButton:
            return False
        pt = self.canvas.section_at(pos.x(), pos.y())
        self.press_px = (pos.x(), pos.y())
        if tool in ("brush", "freehand"):
            # The brush only paints while the button is down; the whole stroke is
            # classified once, on release (one undo step, one refresh).
            self.points, self.dragging = [pt], True
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
                self.points, self.dragging = [pt], True
            else:
                self._commit(tool, [self.points[0], pt])
                self.points = []
        self._paint()
        return True

    def move(self, pos):
        tool = self._sync_tool()
        self.hover = self.canvas.section_at(pos.x(), pos.y())
        if tool is None:
            if self.overlay.shapes:
                self.overlay.set_shapes([])
            return False
        if tool == "brush" and self.dragging and self.points:
            # Sample the stroke every quarter brush radius (at least 2 px): the sweep
            # between samples is exact, so this only bounds the work, not the coverage.
            last = self.canvas.screen_from_section(*self.points[-1])
            if np.hypot(pos.x() - last[0], pos.y() - last[1]) >= max(2., .25 * self.brush_radius_px()):
                self.points.append(self.hover)
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
        moved = (self.press_px is not None and
                 np.hypot(pos.x() - self.press_px[0], pos.y() - self.press_px[1]) > _CLICK_SLOP_PX)
        if tool == "brush" and self.dragging:
            self.dragging = False
            stroke, self.points = self.points + [pt], []
            self._commit(tool, stroke)
        elif tool == "freehand" and self.dragging:
            self.dragging = False
            shape, self.points = self.points + [pt], []
            self._commit(tool, shape)
        elif tool in _TWO_CLICK and self.dragging:
            self.dragging = False
            if moved and self.points:          # press-drag-release form
                self._commit(tool, [self.points[0], pt])
                self.points = []
        self.press_px = None
        self._paint()
        return True

    def double_click(self, pos):
        if self.tool() == "polygon" and len(self.points) >= 3:
            self._finish_polygon()
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

    def _paint(self):
        tool, hover, pts = self.tool(), self.hover, self.points
        shapes = []
        if tool is None or hover is None:
            self.overlay.set_shapes([])
            return
        style = line_style(tool)
        if tool == "brush":
            square = str(getattr(self.app, "brush_shape", "circle")).lower() == "rectangle"
            if pts:
                shapes.append(("stroke", list(pts) + [hover], (square, self.brush_radius_px(), style)))
            shapes.append(("circle_px", [hover], (square, self.brush_radius_px(), style)))
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
        return np.array([int(c) for c, rule in palette.items() if rule.get("show", True)],
                        dtype=np.int64)

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
            lut = np.zeros(256, dtype=np.bool_)
            lut[np.asarray(visible, dtype=np.uint8)] = True
            indices = indices[lut[classes[indices]]]
        return indices

    def _commit(self, tool, shape):
        app = self.app
        to_class = getattr(app, "to_class", None)
        if to_class is None:
            app.statusBar().showMessage("Target class not set.", 2500)
            return 0
        indices = self.candidates(tool, shape)
        if not indices.size:
            return 0
        classes = app.data["classification"]
        from_classes = getattr(app, "from_classes", None)
        if from_classes and indices.size:
            lut = np.zeros(256, dtype=np.bool_)
            lut[np.asarray(from_classes, dtype=np.uint8)] = True
            indices = indices[lut[classes[indices]]]
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
        app.statusBar().showMessage(f"{indices.size:,} points -> class {to_class}", 3000)
        return int(indices.size)
