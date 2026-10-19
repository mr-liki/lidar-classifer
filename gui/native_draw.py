"""Drawing tools for the native LNT/Vulkan plan viewport; no VTK objects."""
from __future__ import annotations

import math
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QInputDialog

from gui.native_main_classify import MainCanvas
from gui.native_section_classify import CanvasOverlay
from gui.native_vector_pick import (
    SntPickIndex, nearest_segment, segment_projection, segment_intersection,
    polygon_centroid, offset_open_line, centerline_between,
)


TOOL_MAP = {
    'smartline': 'smartline', 'line': 'line', 'orthopolygon': 'ortho',
    'polyline': 'polyline', 'polyline_permanent': 'polyline',
    'rectangle': 'rectangle', 'rectangle_permanent': 'rectangle',
    'circle': 'circle', 'circle_permanent': 'circle',
    'freehand': 'freehand', 'freehand_permanent': 'freehand',
    'text': 'text', 'gispoint': 'point', 'gis_point': 'point', 'vertex': 'vertex',
    'move_vertex': 'movevertex', 'movevertex': 'movevertex',
    'delete_vertex': 'deletevertex', 'deletevertex': 'deletevertex',
    'select_drawing': 'select', 'selectdrawing': 'select', 'select': 'select',
    'curve_point': 'curve', 'curve': 'curve', 'hatcharea': 'hatch',
    'accudraw': 'accudraw',
    'parallel': 'parallel', 'centerline': 'centerline',
}


def _catmull_rom(points, steps=16):
    """Interpolate an open smooth curve without replacing its editable control points."""
    pts = list(points)
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    for i in range(len(pts) - 1):
        p0, p1 = pts[max(0, i-1)], pts[i]
        p2, p3 = pts[i+1], pts[min(len(pts)-1, i+2)]
        for k in range(1, steps+1):
            t = k / steps
            t2, t3 = t*t, t*t*t
            out.append(tuple(.5 * ((2*b) + (-a+c)*t + (2*a-5*b+4*c-d)*t2
                                    + (-a+3*b-3*c+d)*t3)
                             for a,b,c,d in zip(p0,p1,p2,p3)))
    return out


def _inside_polygon(p, poly):
    """Even-odd point-in-polygon for closed native Draw elements."""
    x, y = p; inside = False
    for a,b in zip(poly, list(poly[1:]) + [poly[0]]):
        if (a[1] > y) != (b[1] > y):
            crossing = a[0] + (y - a[1]) * (b[0]-a[0]) / (b[1]-a[1])
            if x < crossing:
                inside = not inside
    return inside


class _PrecisionPaletteAdapter:
    """Expose the existing AccuDrawDialog API to the native plan-view draft."""
    def __init__(self, draw):
        self.draw = draw
    @property
    def active(self):
        return self.draw.tool == 'accudraw'
    def add_point_from_dialog(self):
        return self.draw.add_xyz_from_dialog()
    def undo_last_point(self):
        return self.draw.history(redo=False)
    def finish(self):
        return self.draw._finish()
    def finish_for_tool_switch(self, next_tool=''):
        self.draw.cancel()
    def deactivate(self, cancel=True):
        self.draw.cancel()


class NativeDraw:
    def __init__(self, view):
        self.view = view
        self.app = view.app
        self.canvas = MainCanvas(view)
        self.overlay = CanvasOverlay(self.canvas)
        self.drawings = []
        self.redo = []
        # Pending vertices are separate from the document/object history.
        # Ctrl+Z/Y edits the current placement first, one clicked vertex at a time.
        self.point_redo = []
        # Line and Smart Line place vertices the same way. A Line stores each
        # completed edge as a separate selectable object; keep the active
        # Line-session objects so Ctrl+Z/Y edits the latest edge, not the
        # unfinished starting anchor or an unrelated drawing.
        self._line_session_refs = []
        self._line_session_undone = []
        # XYZ values follow live AccuDraw vertices (the canvas is a 2-D plan).
        self._z_points = []
        self._z_redo = []
        self._last_z = 0.0
        self._precision_dialog = None
        self._precision_adapter = None
        self._hatch_dialog = None
        self._hatch_mode = 2  # select by points; same default as legacy Hatch
        self._angle_sticky = False
        self._last_pointer_pos = None
        self.tool = None
        self.points = []
        self.hover = None
        self.dragging = False
        self.selected = None
        self._selected_native_ref = None
        self._selected_imported = None
        self._selection_scene_token = None
        self._parallel_source = None
        self._centerline_first = None
        self._snt_pick_index = None
        self._snt_pick_token = None
        self._snap_target = None
        self.vertex_target = None
        # Pending Move Vertex keeps its BEFORE object immutable until commit.
        # Live mouse-drag previews replace only an editable native drawing.
        self._vertex_move_before = None
        self._vertex_move_original = None
        self._vertex_move_dragged = False
        self._vertex_insert_pending = False
        self._text_options = None
        self.aux_shapes = []
        self._selection_bridge = None  # active Vulkan-only Select panel
        # The legacy Parallel/Centerline dialogs own interaction when Vulkan is
        # visible. Keep their original algorithms and controls; only picking and
        # result delivery need an adapter. Never register VTK observers here.
        self._legacy_tool_dialog = None
        self._legacy_dialog_highlights = []
        self._object_actions = []
        self._object_redo = []
        self.reload_styles()

    def reload_styles(self):
        from gui.draw_settings_dialog import load_draw_settings
        self.styles = load_draw_settings()
        self._paint()

    def _style(self, tool):
        from PySide6.QtGui import QColor
        key = {'ortho': 'orthopolygon', 'hatch': 'hatcharea'}.get(tool, tool)
        style = dict(self.styles.get(key, {}))
        if 'color' in style:
            style['color'] = QColor.fromRgbF(*style['color']).name()
        if 'fill_color' in style:
            style['fill_color'] = QColor.fromRgbF(*style['fill_color']).name()
        return style

    def activate(self, name):
        tool = TOOL_MAP.get(str(name or '').lower().replace(' ', '_'))
        if tool is None:
            return False
        # Switching ribbon tools must close the previous dialog's native
        # routing; its VTK branch is unaffected by this native-only state.
        old_dialog = self._legacy_tool_dialog
        if old_dialog is not None and tool != self.tool:
            self._legacy_tool_dialog = None
            self._legacy_dialog_highlights = []
            try:
                old_dialog.close()
            except RuntimeError:
                pass
        measurement = getattr(self.view, 'measurement', None)
        if measurement is not None:
            measurement.cancel()
        classifier = getattr(self.view, 'classifier', None)
        if classifier is not None:
            classifier.cancel()
        sections = getattr(self.app, '_native_sections', None)
        if sections is not None:
            sections.cancel_cross(); sections.cancel_cut()
        self.cancel(stand_down=False)
        if tool != 'select' and self._selection_bridge is not None:
            self._selection_bridge._active = False
            self._selection_bridge = None
        self.tool = tool
        # A Select action can choose a native drawing or an imported SNT line.
        # Parallel may reuse that selection; Centerline chooses two sources.
        self._parallel_source = self._selected_feature() if tool == 'parallel' else None
        self._centerline_first = None
        if tool == 'accudraw':
            # The original VTK AccuDraw popup is reusable: its fields and
            # buttons call this adapter, while native viewport owns placement.
            from gui.accudraw_tool import AccuDrawDialog
            self._precision_adapter = _PrecisionPaletteAdapter(self)
            self._precision_dialog = AccuDrawDialog(self._precision_adapter, parent=self.app)
            self._precision_dialog.set_status(
                'Click vertices or type XYZ and press Enter. Right-click / Finish completes the polyline.')
            self._precision_dialog.show()
            self._precision_dialog.raise_()
        elif tool == 'hatch':
            from gui.hatch_area_tool import HatchAreaDialog
            self._hatch_mode = 2
            self._hatch_dialog = HatchAreaDialog(parent=self.app)
            self._hatch_dialog.method_changed.connect(self._hatch_change_method)
            self._hatch_dialog.set_status(
                'Points: click 3+ vertices; Element/Block: pick a closed SNT/native polygon. '
                'Rectangle: two corners. Right-click to finish.')
            self._hatch_dialog.show()
            self._hatch_dialog.raise_()
        elif tool == 'text':
            # Use the SAME font/text palette as the original VTK digitizer,
            # but keep positioning/snap in the visible Vulkan widget.
            from gui.digitize_tools import TextEditDialog
            from PySide6.QtWidgets import QDialog
            dialog = TextEditDialog(current_text='New Text', current_size=75,
                                    current_font='Arial', current_bold=True,
                                    current_italic=False, current_color=(1, 0, 0),
                                    parent=self.app)
            if dialog.exec() != QDialog.Accepted:
                self.cancel()
                return False
            values = dialog.get_values()
            if not str(values.get('text', '')).strip():
                self.cancel()
                return False
            self._text_options = values
        self.canvas.setFocus()
        self.app.statusBar().showMessage(self._instruction(), 5000)
        return True

    def _instruction(self):
        if self.tool == 'line':
            return ('Native Line: click connected vertices continuously; each segment '
                    'selects independently. Right-click, double-click, or Enter to finish.')
        if self.tool in ('rectangle', 'circle'):
            return f'Native {self.tool}: click two points'
        if self.tool == 'accudraw':
            return 'Native XYZ/Angle: click vertices or enter XYZ; right-click to finish'
        if self.tool == 'hatch':
            return 'Native Hatch: choose method and spacing; right-click to complete boundary'
        if self.tool == 'polyline':
            return 'Native polygon: click vertices; right-click or double-click to close'
        if self.tool in ('smartline', 'ortho', 'curve', 'hatch'):
            return f'Native {self.tool}: click points; right-click or double-click to finish'
        if self.tool == 'select':
            return 'Native Select: click a visible SNT vector or native drawing'
        if self.tool == 'parallel':
            return ('Native Parallel: click source line then click on offset side; '
                    'or Select a line first')
        if self.tool == 'centerline':
            return 'Native Centerline: click two different open line features'
        if self.tool == 'freehand':
            return 'Native Freehand: hold and drag the left button'
        return f'Native {self.tool}: click a position'

    def active(self):
        return self.tool is not None

    def bind_legacy_dialog(self, dialog):
        """Hand Vulkan clicks to the REAL VTK dialog instead of a replacement UI."""
        self._legacy_tool_dialog = dialog
        self._legacy_dialog_highlights = []
        self._last_pointer_pos = None
        self._paint()

    def unbind_legacy_dialog(self, dialog):
        if self._legacy_tool_dialog is not dialog:
            return
        self._legacy_tool_dialog = None
        self._legacy_dialog_highlights = []
        self._last_pointer_pos = None
        if self.tool in ('parallel', 'centerline'):
            self.cancel()
        else:
            self._paint()

    def set_legacy_dialog_highlights(self, dialog, features):
        if self._legacy_tool_dialog is not dialog:
            return
        self._legacy_dialog_highlights = list(features)
        self._paint()

    def _scene_token(self):
        """Invalidate imported geometry and selections after SNT load/unload."""
        records = []
        for att in getattr(self.app, 'snt_attachments', ()) or ():
            entities = att.get('_render_entities')
            if entities is None:
                entities = att.get('entities') or ()
            records.append((id(att), id(entities), len(entities)))
        return tuple(records)

    def _snt_index(self):
        token = self._scene_token()
        if token != self._snt_pick_token:
            self._snt_pick_token = token
            self._snt_pick_index = None
            # A former SNT selection must never refer to a removed/replaced file.
            if (self._selected_imported is not None and
                    self._selection_scene_token != token):
                self._selected_imported = None
                self._parallel_source = None
                self._centerline_first = None
        if self._snt_pick_index is None:
            self._snt_pick_index = SntPickIndex(
                getattr(self.app, 'snt_attachments', ()) or ())
        return self._snt_pick_index

    def _snt_visible(self, doc_key, layer_name):
        layer = getattr(self.app, '_native_vector_layer', None)
        if layer is None:
            # The independent LNT view also accepts SNT attachments, which can
            # have a different draw-layer owner. A missing visibility manager
            # is not proof that the attached file was hidden.
            return True
        document = getattr(layer, 'documents', {}).get(doc_key)
        if document is None:
            return False   # detached or not uploaded into the visible engine
        handle = document.get('handles', {}).get(layer_name)
        return bool(handle is not None and getattr(handle, '_visible', True))

    @staticmethod
    def _native_coords(drawing):
        """Geometry actually displayed by the overlay (circle/rect included)."""
        coords = [tuple(p[:2]) for p in drawing.get('coords', [])]
        dtype = drawing.get('type')
        if dtype == 'curve':
            return _catmull_rom(coords)
        if dtype == 'rectangle' and len(coords) >= 2:
            a,b=coords[0],coords[1]
            return [a,(b[0],a[1]),b,(a[0],b[1]),a]
        if dtype == 'circle' and len(coords) >= 2:
            a,b=coords[0],coords[1]
            radius=math.hypot(b[0]-a[0],b[1]-a[1])
            if radius<=1.e-12:return [a]
            return [(a[0]+radius*math.cos(k*math.tau/48.),
                     a[1]+radius*math.sin(k*math.tau/48.))
                    for k in range(49)]
        return coords

    def _near_scene(self, p, tolerance):
        # Native objects are typically few: no scene index needed for them.
        for i,drawing in enumerate(self.drawings):
            coords=self._native_coords(drawing)
            if not coords:continue
            lo_x=min(q[0] for q in coords);hi_x=max(q[0] for q in coords)
            lo_y=min(q[1] for q in coords);hi_y=max(q[1] for q in coords)
            if (lo_x-tolerance <= p[0] <= hi_x+tolerance and
                    lo_y-tolerance <= p[1] <= hi_y+tolerance):
                yield {'source':'native', 'index':i, 'coords':coords,
                       'type':drawing.get('type','line'), 'key':('native',id(drawing)),
                       'drawing':drawing}
        index = self._snt_index()
        yield from index.nearby(p, tolerance, visible=self._snt_visible)

    @staticmethod
    def _feature_distance(p, feature):
        coords=feature['coords']
        if len(coords)==1:
            return math.dist(p,coords[0])
        info=nearest_segment(p,coords)
        return info[0] if info else math.inf

    def _pick_feature(self, p, tolerance=None, open_only=False, exclude=None):
        if tolerance is None:
            tolerance=max(self.canvas.world_per_pixel()*12., 1.e-9)
        best=None
        for feature in self._near_scene(p, tolerance):
            coords=feature['coords']
            if open_only and (len(coords)<2 or coords[0]==coords[-1]):
                continue
            if exclude is not None and feature['key']==exclude['key']:
                continue
            distance=self._feature_distance(p,feature)
            # Prefer clicking the actual stroke. Closed-polygon interior is a
            # fallback only, so giant SNT block bounds do not steal line picks.
            if distance<=tolerance:
                rank=(distance,0 if feature['source']=='native' else 1)
                if best is None or rank<best[0]:best=(rank,feature)
        if best:
            return best[1]
        if open_only:
            return None
        # Second pass: a click inside a closed polygon selects the polygon.
        for feature in self._near_scene(p,tolerance):
            coords=feature['coords']
            if len(coords)>=4 and coords[0]==coords[-1] and _inside_polygon(p,coords[:-1]):
                return feature
        return None

    def _selected_feature(self):
        if (self.selected is not None and 0<=self.selected<len(self.drawings)
                and self.drawings[self.selected] is self._selected_native_ref):
            drawing=self.drawings[self.selected]
            return {'source':'native','index':self.selected,
                    'coords':self._native_coords(drawing),
                    'key':('native',id(drawing)), 'type':drawing.get('type','line'),
                    'drawing':drawing}
        if self._selected_imported is not None:
            if self._selection_scene_token != self._scene_token():
                self._selected_imported=None
            elif self._snt_visible(self._selected_imported['doc_key'],
                                  self._selected_imported['layer']):
                return self._selected_imported
        return None

    def _valid_source(self, feature):
        if feature is None:return False
        if feature['source']=='native':
            return any(d is feature.get('drawing') for d in self.drawings)
        if feature['source']=='snt':
            # Re-check the active document and attachment identity; a detached
            # source is not a valid input to a newly created drawing.
            return (feature.get('scene_token')==self._scene_token()
                    and self._snt_visible(feature['doc_key'],feature['layer']))
        return False

    def _store_selection(self, feature):
        if feature is None:
            self.selected = None
            self._selected_native_ref = None
            self._selected_imported = None
            self._selection_scene_token = None
        elif feature['source']=='native':
            self.selected = feature['index']
            self._selected_native_ref = feature['drawing']
            self._selected_imported = None
            self._selection_scene_token = None
        else:
            self.selected = None
            self._selected_native_ref = None
            self._selected_imported = dict(feature,scene_token=self._scene_token())
            self._selection_scene_token = self._scene_token()
        self._paint()

    @staticmethod
    def _parallel_output(source, click):
        coords=source['coords']
        if len(coords)<2 or coords[0]==coords[-1]:
            return None
        segment=nearest_segment(click,coords)
        if segment is None:return None
        _,idx,foot,_=segment
        a,b=coords[idx],coords[idx+1]
        dx,dy=b[0]-a[0],b[1]-a[1]
        length=math.hypot(dx,dy)
        if length<=1.e-12:return None
        signed=((click[0]-foot[0])*(-dy)+(click[1]-foot[1])*dx)/length
        result=offset_open_line(coords,signed)
        return (result,signed) if result is not None else None

    def _snap_at(self,p,mode,tol):
        features=list(self._near_scene(p,tol))
        best=None
        endpoint=None
        def offer(q, vertex=False):
            nonlocal best, endpoint
            distance=math.dist(p,q)
            if distance<=tol and (best is None or distance<best[0]):
                best=(distance,tuple(q))
            # Nearby mode prioritizes a real endpoint in a tight radius over
            # a perpendicular projection onto the adjacent segment.
            if vertex and distance<=tol*.55 and (endpoint is None or distance<endpoint[0]):
                endpoint=(distance,tuple(q))
        if mode == 'intercept':
            # Compare only nearby segments. Reject intersections of adjacent
            # segments of the same polyline (those are ordinary vertices).
            nearby=[]
            for feature in features:
                for si,(a,b) in enumerate(zip(feature['coords'],feature['coords'][1:])):
                    if segment_projection(p,a,b)[0]<=tol:
                        nearby.append((feature['key'],si,a,b))
            for idx,one in enumerate(nearby):
                for two in nearby[idx+1:]:
                    if one[0]==two[0] and abs(one[1]-two[1])<=1:
                        continue
                    crossed=segment_intersection(one[2],one[3],two[2],two[3])
                    if crossed is not None:offer(crossed)
        else:
            for feature in features:
                coords=feature['coords']
                # Smart Line doesn't snap to its *current* unfinished vertex.
                # Line commits each segment as an independent drawing, so its
                # newest endpoint also appears in the spatial picker. Ignore
                # that endpoint/segment for automatic nearby snapping or a
                # short next segment could magnetically collapse to zero.
                recent_line = (mode == 'nearby' and self.tool == 'line'
                               and bool(self.points) and bool(self._line_session_refs)
                               and feature.get('source') == 'native'
                               and feature.get('drawing') is self._line_session_refs[-1])
                if mode in ('nearby','keypoint'):
                    for q in coords:
                        if recent_line and tuple(q[:2]) == self.points[-1]:
                            continue
                        offer(q,vertex=True)
                if mode == 'center':
                    if feature.get('type')=='circle' and feature['source']=='native':
                        offer(feature['drawing']['coords'][0][:2])
                    elif len(coords)>=4 and coords[0]==coords[-1]:
                        center=polygon_centroid(coords)
                        if center is not None:offer(center)
                for a,b in zip(coords,coords[1:]):
                    if recent_line:
                        continue
                    dist,foot,_=segment_projection(p,a,b)
                    if mode == 'nearby' and dist<=tol:offer(foot)
                    if mode == 'midpoint':offer(((a[0]+b[0])*.5,(a[1]+b[1])*.5))
            # A draft may snap back to its own earlier points, not the most
            # recent vertex (which would prevent drawing the next segment).
            if mode in ('nearby','keypoint'):
                for q in self.points[:-1]:offer(q,vertex=True)
        if mode=='nearby' and endpoint is not None:
            return endpoint[1]
        return best[1] if best else None

    def _point(self, pos):
        p=self.canvas.section_at(pos.x(),pos.y())
        did_snap=False
        self._snap_target=None
        digitizer=getattr(self.app,'digitizer',None)
        # Both Smart Line AND Line snap automatically in the VTK digitizer.
        # Text placement and Vertex insertion also need endpoint magnetism.
        # Explicit Snap mode takes priority over this nearby default.
        smart_auto = self.tool in ('smartline', 'line', 'text', 'vertex')
        explicit_snap = bool(digitizer is not None and getattr(digitizer,'snap_enabled',False))
        if self.tool not in ('select','parallel','centerline') and (smart_auto or explicit_snap):
            mode=(getattr(digitizer,'snap_mode',None) or 'nearby') if explicit_snap else 'nearby'
            if mode in ('nearby','keypoint','midpoint','center','intercept'):
                candidate=self._snap_at(p,mode,max(self.canvas.world_per_pixel()*14.,1.e-9))
                if candidate is not None:
                    p=candidate;did_snap=True;self._snap_target=p
        if self.tool=='ortho' and self.points:
            a=self.points[-1]
            # Ortho is an explicit axis constraint. A non-axis-aligned target
            # cannot override it, even when Snap is enabled.
            p=((p[0],a[1]) if abs(p[0]-a[0])>=abs(p[1]-a[1]) else (a[0],p[1]))
            if did_snap and self._snap_target!=p:
                self._snap_target=None
            return p
        if self.tool=='accudraw' and self.points:
            if did_snap:
                self._angle_sticky=False
                return p
            a=self.points[-1]
            dx,dy=p[0]-a[0],p[1]-a[1]
            length=math.hypot(dx,dy)
            if length<=1.e-12:return p
            dlg=self._precision_dialog
            angle=dlg.target_angle_degrees() if dlg is not None else 0.
            tolerance=dlg.angle_tolerance_degrees() if dlg is not None else 2.
            if len(self.points)>=2:
                prev=self.points[-2]
                angle+=math.degrees(math.atan2(a[1]-prev[1],a[0]-prev[0]))
            cursor_angle=math.degrees(math.atan2(dy,dx))
            diff=abs((cursor_angle-angle+180.)%360.-180.)
            snap_tol=tolerance*(3.5 if self._angle_sticky else 1.)
            self._angle_sticky=diff<=snap_tol
            if self._angle_sticky:
                theta=math.radians(angle)
                return (a[0]+length*math.cos(theta),a[1]+length*math.sin(theta))
        return p

    # Freehand is continuous drag sampling, not a sequence of clicked vertices.
    # The other placement tools share a single point-level history, including
    # two-click geometry (line/rectangle/circle) before it is finalized.
    _POINT_TOOLS = frozenset({
        'smartline', 'polyline', 'ortho', 'curve', 'hatch',
        'line', 'rectangle', 'circle', 'accudraw',
    })

    def _hatch_change_method(self, method_id):
        self._hatch_mode = int(method_id)
        self.points.clear()
        self.point_redo.clear()
        self._z_redo.clear()
        self.hover = None
        dlg = self._hatch_dialog
        if dlg is not None:
            instructions = {
                0: 'Select by Element: click inside a closed native or SNT polygon.',
                2: 'Select by Points: click at least 3 boundary vertices; right-click to hatch.',
                3: 'Select by Rectangle: click two opposing rectangle corners.',
                6: 'Select Block: click inside a visible closed SNT block boundary.',
            }
            dlg.set_status(instructions.get(self._hatch_mode,
                'Fence, Freehand, and Union require additional geometry handling; '
                'use Element, Points, Rectangle, or Block in Vulkan.'))
        self._paint()

    def _hatch_metadata(self, boundary):
        """Use legacy hatch geometry generator, with a bounded overlay density."""
        dlg = self._hatch_dialog
        spacing = max(0.1, float(dlg.spacing if dlg is not None else 1.0))
        angle = float(dlg.angle if dlg is not None else 45.0)
        coords = [(float(x),float(y)) for x,y in boundary]
        if coords[0] == coords[-1]:
            coords = coords[:-1]
        if len(coords) < 3:
            return None
        # Do not silently replace user-entered spacing with a coarser hatch.
        # Abort excessively dense hatch safely so dimensions stay exact.
        ext = (max(x for x,y in coords)-min(x for x,y in coords)
               + max(y for x,y in coords)-min(y for x,y in coords))
        if ext / spacing > 20000.:
            if dlg is not None:
                dlg.set_status(f'Hatch too dense at {spacing:g} units. '
                               f'Increase spacing to at least {ext/20000.:.3f} for this polygon.')
            return None
        from gui.hatch_area_tool import generate_hatch_lines
        segments = generate_hatch_lines([(x,y,0.0) for x,y in coords], spacing, angle)
        if len(segments) > 30000:
            if dlg is not None:
                dlg.set_status('Hatch exceeds the 30,000-segment safety limit. Increase spacing.')
            return None
        return {'hatch_segments': [((a[0],a[1]),(b[0],b[1])) for a,b in segments],
                'hatch_spacing': spacing, 'hatch_effective_spacing': spacing,
                'hatch_angle': angle}

    def _commit_hatch(self, boundary, **extra):
        metadata = self._hatch_metadata(boundary)
        if metadata is None:
            return False
        return self._commit('hatch', boundary, **metadata, **extra)

    def _hatch_existing(self, p, block_only=False):
        """Hatch a visible polygon using a NEW overlay; SNT geometry stays untouched."""
        if not block_only:
            for drawing in reversed(self.drawings):
                pts = drawing.get('coords', [])
                if (len(pts) >= 4 and pts[0] == pts[-1]
                        and drawing.get('type') != 'hatch' and _inside_polygon(p, pts[:-1])):
                    boundary = list(pts)
                    return self._commit_hatch(boundary)
        # Pick the smallest containing SNT ring (rather than the bounding box
        # of the first large polygon), respecting layer visibility. Block mode
        # permits only explicitly named BLOCK-level SNT boundaries.
        candidate = None
        for feature in self._snt_index().nearby(p, 0., visible=self._snt_visible):
            pts = feature['coords']
            if (len(pts) < 4 or pts[0] != pts[-1]
                    or not _inside_polygon(p, pts[:-1])):
                continue
            if block_only and 'block' not in feature.get('layer','').lower():
                continue
            box_area=((max(q[0] for q in pts)-min(q[0] for q in pts)) *
                      (max(q[1] for q in pts)-min(q[1] for q in pts)))
            if candidate is None or box_area<candidate[0]:
                candidate=(box_area, feature)
        if candidate is not None:
            boundary=list(candidate[1]['coords'])
            return self._commit_hatch(boundary, source_key=candidate[1]['key'])
        if self._hatch_dialog is not None:
            self._hatch_dialog.set_status('No visible closed SNT block/polygon at this click. '
                                          'Check the SNT layer or use Select by Points.')
        return False

    def _set_palette_xyz(self, xyz):
        dlg = self._precision_dialog
        if dlg is None:
            return
        try:
            if any(widget.hasFocus() for widget in (dlg.x_edit,dlg.y_edit,dlg.z_edit)):
                return  # Do not overwrite XYZ actively being typed.
            dlg.set_xyz(xyz)
        except (AttributeError, RuntimeError):
            pass

    def add_xyz_from_dialog(self):
        """Add user-entered precise XYZ; Z is preserved in the drawing record."""
        dlg = self._precision_dialog
        if self.tool != 'accudraw' or dlg is None:
            return False
        try:
            x,y,z = (float(v) for v in dlg.get_xyz())
            if not all(math.isfinite(v) for v in (x,y,z)):
                raise ValueError('finite XYZ required')
        except (ValueError, TypeError):
            dlg.set_status('Enter finite numeric X, Y, Z coordinates.')
            return False
        if self.points and (x,y) == self.points[-1]:
            dlg.set_status('Duplicate XY: move to a different position in plan view.')
            return False
        self.point_redo.clear(); self._z_redo.clear()
        self._last_z = z
        self.points.append((x,y)); self._z_points.append(z)
        self._refresh_placement_preview()
        self._paint()
        dlg.set_status(f'Added XYZ vertex {len(self.points)}. Right-click or Finish to complete.')
        return True

    def _close_palettes(self):
        for name in ('_hatch_dialog','_precision_dialog'):
            dlg = getattr(self,name,None)
            if dlg is not None:
                dlg.close()
                setattr(self,name,None)
        self._precision_adapter = None

    def _refresh_placement_preview(self):
        """Re-anchor Ortho/AccuDraw hover to the last *remaining* vertex."""
        self.hover = None
        if self.points and self._last_pointer_pos is not None:
            try:
                self.hover = self._point(self._last_pointer_pos)
            except RuntimeError:
                pass  # Transient unavailable viewport: don't retain stale hover.

    def _commit(self, kind=None, points=None, **extra):
        pts = list(points if points is not None else self.points)
        if not pts:
            return False
        before = list(self.drawings)
        self.drawings.append({'type': kind or self.tool, 'coords': pts, **extra})
        self.record_object_change(before)
        self.redo.clear()
        self.point_redo.clear()
        self._z_redo.clear()
        self._z_points.clear()
        self._angle_sticky = False
        self.points, self.hover, self.dragging = [], None, False
        # Hide any green snap indicator when a drawing finishes. The current
        # vertex still exists in the saved geometry and remains snappable.
        self._snap_target = None
        self._paint()
        return True

    def record_object_change(self, before):
        """Undo grouped geometry edits as one atomic operation.

        Older point-by-point placement Undo remains higher priority. Snapshots
        contain only native Draw objects, never imported SNT records.
        """
        self._object_actions.append((list(before), list(self.drawings)))
        if len(self._object_actions) > 100:
            del self._object_actions[:-100]
        self._object_redo.clear()
        self.redo.clear()
        self._paint()

    def _finish_line(self):
        """End only the active Line chain; each completed edge already exists."""
        self.points.clear()
        self.point_redo.clear()
        self._line_session_refs.clear()
        self._line_session_undone.clear()
        self.hover = None
        self._snap_target = None
        self._last_pointer_pos = None
        self._paint()
        return True

    def _line_history(self, redo=False):
        """Undo/redo one Line segment without touching older unrelated objects.

        Individual segments share document Undo snapshots with other Draw tools.
        Identity checks prevent stale session state from undoing an edit made by
        another tool or replacing the wrong drawing after switching renderers.
        """
        if redo:
            if self.point_redo:
                self.points[:] = [self.point_redo.pop()]
                self._refresh_placement_preview()
                self._paint()
                return True
            if self._line_session_undone:
                segment = self._line_session_undone[-1]
                if not self._object_redo:
                    return False
                before, after = self._object_redo[-1]
                if not after or after[-1] is not segment:
                    return False
                self._object_redo.pop()
                self.drawings[:] = after
                self._object_actions.append((before, after))
                self._line_session_refs.append(self._line_session_undone.pop())
                self.points[:] = [segment['coords'][-1]]
                self._refresh_placement_preview()
                self._paint()
                return True
            return False

        if self._line_session_refs:
            segment = self._line_session_refs[-1]
            if not self._object_actions:
                return False
            before, after = self._object_actions[-1]
            if (not after or after[-1] is not segment or
                    not self.drawings or self.drawings[-1] is not segment):
                return False
            self._object_actions.pop()
            self.drawings[:] = before
            self._object_redo.append((before, after))
            self._line_session_undone.append(self._line_session_refs.pop())
            self.points[:] = [segment['coords'][0]]
            self._refresh_placement_preview()
            self._paint()
            return True
        if self.points:
            # Only the first click is uncommitted and has no segment to undo.
            self.point_redo.append(self.points.pop())
            self.hover = None
            self._paint()
            return True
        return False

    def _nearest_vertex(self, point):
        best = None
        for di, drawing in enumerate(self.drawings):
            for vi, vertex in enumerate(drawing.get('coords', [])):
                distance = math.hypot(point[0]-vertex[0], point[1]-vertex[1])
                if best is None or distance < best[0]:
                    best = distance, di, vi
        tolerance = max(self.canvas.world_per_pixel() * 12., 1e-9)
        return best if best is not None and best[0] <= tolerance else None

    def _vertex_move_preview(self, destination):
        """Replace the editable shape for a live preview without mutating Undo's BEFORE."""
        if self.vertex_target is None or self._vertex_move_original is None:
            return False
        di, vi = self.vertex_target
        if di >= len(self.drawings):
            return False
        original = self._vertex_move_original
        coords = list(original['coords'])
        if not coords or vi >= len(coords):
            return False
        destination = tuple(destination[:2])
        closed = len(coords) > 2 and coords[0] == coords[-1]
        if closed and vi in (0, len(coords)-1):
            coords[0] = coords[-1] = destination
        else:
            coords[vi] = destination
        new_obj = dict(original, coords=coords)
        self.drawings[di] = new_obj
        self._paint()
        return True

    def _vertex_move_finish(self, destination):
        """Commit one Undo entry after drag-release or second-click placement."""
        if self.vertex_target is None or self._vertex_move_before is None:
            return False
        before = self._vertex_move_before
        original = self._vertex_move_original
        self._vertex_move_preview(destination)
        di, vi = self.vertex_target
        new_obj = self.drawings[di]
        moved = new_obj['coords'][vi] != original['coords'][vi]
        if moved:
            self.record_object_change(before)
        else:
            self.drawings[di] = original
        self.vertex_target = None
        self._vertex_move_before = None
        self._vertex_move_original = None
        self._vertex_move_dragged = False
        self.dragging = False
        self._paint()
        return moved

    def _nearest_segment(self, point):
        best = None
        for di, drawing in enumerate(self.drawings):
            coords = drawing.get('coords', [])
            for si, (a, b) in enumerate(zip(coords, coords[1:])):
                dx, dy = b[0]-a[0], b[1]-a[1]
                d2 = dx*dx + dy*dy
                t = max(0., min(1., ((point[0]-a[0])*dx + (point[1]-a[1])*dy)/d2)) if d2 else 0.
                q = a[0]+t*dx, a[1]+t*dy
                distance = math.hypot(point[0]-q[0], point[1]-q[1])
                if best is None or distance < best[0]:
                    best = distance, di, si, q
        tolerance = max(self.canvas.world_per_pixel() * 12., 1e-9)
        return best if best is not None and best[0] <= tolerance else None

    def _finish(self):
        minimum = 3 if self.tool in ('polyline', 'hatch') else 2
        if len(self.points) < minimum:
            # Keep one-point drafts editable; do not destroy history on a
            # premature Finish/right-click.
            self._paint()
            return False
        pts = list(self.points)
        if self.tool in ('polyline', 'hatch', 'ortho') and pts[0] != pts[-1]:
            pts.append(pts[0])
        if self.tool == 'hatch':
            return self._commit_hatch(pts)
        if self.tool == 'accudraw':
            xyz = [(p[0],p[1],z) for p,z in zip(self.points, self._z_points)]
            return self._commit('accudraw', points=pts, xyz_coords=xyz)
        return self._commit(points=pts)

    def press(self, pos, button):
        if not self.active() or button == Qt.MiddleButton:
            return False
        if self.tool == 'select' and self._selection_bridge is not None:
            return self._selection_bridge.press(pos,button)
        dialog = self._legacy_tool_dialog
        if dialog is not None and self.tool in ('parallel', 'centerline'):
            # The dialog controls whether a figure-pick is pending. Do not
            # accidentally create the simplified native parallel/centerline or
            # pan/orbit the camera while a dialog is open.
            if button == Qt.RightButton:
                if getattr(dialog, '_select_mode', False):
                    dialog._on_finish_selection()
                return True
            if button != Qt.LeftButton:
                return False
            if getattr(dialog, '_select_mode', False):
                try:
                    p = self.canvas.section_at(pos.x(), pos.y())
                    feature = self._pick_feature(p, open_only=self.tool == 'centerline')
                except (RuntimeError, TypeError, ValueError):
                    feature = None
                if feature is not None and feature['source'] == 'snt':
                    feature = dict(feature, scene_token=self._scene_token())
                dialog._on_native_feature_picked(feature)
            return True
        if button == Qt.RightButton:
            if self.tool == 'line':
                self._finish_line()
            elif self.tool in ('smartline', 'polyline', 'ortho', 'curve', 'hatch', 'accudraw'):
                self._finish()
            else:
                self.points, self.hover, self.dragging = [], None, False
                self.point_redo.clear()
                if self.tool == 'centerline': self._centerline_first=None
                if self.tool == 'parallel': self._parallel_source=None
                self._paint()
            return True
        if button != Qt.LeftButton:
            return False
        try:
            p = self._point(pos)
        except RuntimeError as exc:
            self.app.statusBar().showMessage(str(exc), 4000)
            return True
        self._last_pointer_pos = pos
        if self.tool == 'hatch':
            if self._hatch_mode in (0,6):
                self._hatch_existing(p,block_only=self._hatch_mode==6)
                return True
            if self._hatch_mode not in (2,3):
                if self._hatch_dialog is not None:
                    self._hatch_dialog.set_status('Unsupported Vulkan hatch mode. '
                                                  'Choose Points / Element / Rectangle.')
                return True
        if self.tool == 'select':
            feature=self._pick_feature(p)
            self._store_selection(feature)
            if feature is not None:
                label=('SNT' if feature['source']=='snt' else 'native Draw')
                self.app.statusBar().showMessage(f'Selected {label} {feature["type"]}',3000)
            else:
                self.app.statusBar().showMessage('Nothing selectable at cursor',2500)
            return True
        if self.tool == 'parallel':
            if self._parallel_source is not None and not self._valid_source(self._parallel_source):
                self._parallel_source=None
                self._store_selection(None)
            if self._parallel_source is None:
                source=self._pick_feature(p,open_only=True)
                if source is None:
                    self.app.statusBar().showMessage('Click an existing open SNT/native line first',3500)
                    return True
                self._store_selection(source)
                self._parallel_source=self._selected_feature()
                self.app.statusBar().showMessage('Click on the side and at the offset distance',4000)
                return True
            result=self._parallel_output(self._parallel_source,p)
            if result is None or abs(result[1])<=max(self.canvas.world_per_pixel()*.5,1.e-9):
                self.app.statusBar().showMessage('Choose an offset point away from the source line',3000)
                return True
            points,distance=result
            self._commit('parallel',points, offset_distance=distance,
                         source_key=self._parallel_source['key'])
            self._last_pointer_pos=None
            self._paint()
            self.app.statusBar().showMessage(f'Parallel offset: {distance:.3f} drawing units',4000)
            # Stay in Parallel so users can place another offset from source.
            return True
        if self.tool == 'centerline':
            if self._centerline_first is not None and not self._valid_source(self._centerline_first):
                self._centerline_first=None
                self._store_selection(None)
            if self._centerline_first is None:
                first=self._pick_feature(p,open_only=True)
                if first is None:
                    self.app.statusBar().showMessage('Click the first existing open SNT/native line',3500)
                    return True
                self._store_selection(first)
                self._centerline_first=self._selected_feature()
                self.app.statusBar().showMessage('Click a second DIFFERENT open line',4000)
                return True
            second=self._pick_feature(p,open_only=True,exclude=self._centerline_first)
            if second is None:
                self.app.statusBar().showMessage('Choose a different open line for Centerline',3500)
                return True
            center=centerline_between(self._centerline_first['coords'],second['coords'])
            if center is None:
                self.app.statusBar().showMessage('Centerline needs two non-degenerate open lines',3500)
                return True
            self._commit('centerline',center,source_keys=(self._centerline_first['key'],second['key']))
            self._centerline_first=None
            self._store_selection(None)
            self.app.statusBar().showMessage('Centerline created; choose the first line for another',4000)
            return True
        if self.tool == 'deletevertex':
            hit = self._nearest_vertex(p)
            if hit:
                _, di, vi = hit
                drawing = self.drawings[di]
                coords = list(drawing.get('coords', []))
                closed = len(coords) >= 4 and coords[0] == coords[-1]
                if closed:
                    unique = coords[:-1]
                    if len(unique) <= 3:
                        self.app.statusBar().showMessage(
                            'A closed polygon needs at least three vertices; no change made.', 4000)
                        return True
                    unique.pop(0 if vi == len(coords)-1 else vi)
                    coords = unique + [unique[0]]
                else:
                    if len(coords) <= 2:
                        self.app.statusBar().showMessage(
                            'A line needs at least two vertices; no change made.', 4000)
                        return True
                    coords.pop(vi)
                # Never mutate the source object in an Undo snapshot in-place;
                # every editable point deletion is one reversible native action.
                before = list(self.drawings)
                self.drawings[di] = dict(drawing, coords=coords)
                self.record_object_change(before)
                self._paint()
            return True
        if self.tool == 'movevertex':
            if self.vertex_target is None:
                hit = self._nearest_vertex(p)
                if hit:
                    _, di, vi = hit
                    self.vertex_target = (di, vi)
                    self._vertex_move_original = self.drawings[di]
                    self._vertex_move_before = list(self.drawings)
                    self._vertex_move_dragged = False
                    self.dragging = False
            else:
                # Backward-compatible tap/tap placement; holding the first
                # click and dragging is finalized in release(), below.
                self._vertex_move_finish(p)
            self._paint()
            return True
        if self.tool == 'vertex':
            if self.vertex_target is not None and self._vertex_insert_pending:
                self._vertex_move_finish(p)
                self._vertex_insert_pending = False
                return True
            hit = self._nearest_segment(p)
            if hit:
                _, di, si, _ = hit
                before = list(self.drawings)
                original = self.drawings[di]
                coords = list(original['coords'])
                coords.insert(si + 1, p)
                self.drawings[di] = dict(original, coords=coords)
                self.record_object_change(before)
                if getattr(getattr(self.app, 'digitizer', None), 'vertex_auto_drag', False):
                    self.vertex_target = (di, si+1)
                    self._vertex_move_original = self.drawings[di]
                    self._vertex_move_before = list(self.drawings)
                    self._vertex_insert_pending = True
            else:
                feature = self._pick_feature(p)
                if feature is not None and feature['source']=='snt':
                    segment = nearest_segment(p, feature['coords'])
                    if segment is not None:
                        coords = list(feature['coords'])
                        coords.insert(segment[1]+1, p)
                        self._commit('polyline', coords, source_key=feature['key'])
                    else:
                        self._commit('point', [p])
                else:
                    self._commit('point', [p])
            self._paint()
            return True
        if self.tool == 'text':
            options = self._text_options
            if options is None:
                self.app.statusBar().showMessage('Text palette cancelled; activate Text again.', 3000)
                return True
            red, green, blue = options.get('color', (1., 0., 0.))
            hex_color = '#%02x%02x%02x' % tuple(
                max(0, min(255, round(float(c)*255))) for c in (red,green,blue))
            self._commit('text', [p], text=options['text'], color=hex_color,
                         font_family=options.get('font_family','Arial'),
                         font_size=int(options.get('font_size',75)),
                         bold=bool(options.get('bold',False)),
                         italic=bool(options.get('italic',False)))
            return True
        if self.tool == 'point':
            self._commit('point', [p])
            return True
        if self.tool == 'line':
            if not self.points:
                # First click starts the chain; no document geometry yet.
                self.point_redo.clear()
                self._line_session_undone.clear()
                self.points[:] = [p]
                self.hover = None
                self._paint()
                return True
            start = self.points[-1]
            if p == start:
                # Ignore coincident clicks (including snapping onto the same
                # endpoint); never create a zero-length segment.
                return True
            self.point_redo.clear()
            self._line_session_undone.clear()
            self._commit('line', [start, p])
            self._line_session_refs.append(self.drawings[-1])
            if len(self._line_session_refs) > 100:
                del self._line_session_refs[:-100]
            # _commit() clears pending vertices. Preserve ONLY the new endpoint
            # as the start of the next independent segment, without duplicating
            # the already-committed geometry in the live preview.
            self.points[:] = [p]
            self.hover = None
            self._paint()
            return True
        if self.tool == 'accudraw' and self.points and p == self.points[-1]:
            if self._precision_dialog is not None:
                self._precision_dialog.set_status('Duplicate point ignored. Move cursor or enter new XYZ.')
            return True
        if self.tool in self._POINT_TOOLS:
            # Any fresh click after Undo creates a new branch of the shape.
            self.point_redo.clear(); self._z_redo.clear()
        self.points.append(p)
        if self.tool == 'accudraw':
            self._z_points.append(self._last_z)
            self._angle_sticky = False
            self._set_palette_xyz((p[0],p[1],self._last_z))
        if self.tool == 'freehand':
            self.dragging = True
        elif self.tool == 'hatch' and self._hatch_mode == 3 and len(self.points) == 2:
            a,b = self.points
            boundary = [a,(b[0],a[1]),b,(a[0],b[1]),a]
            self._commit_hatch(boundary)
        elif self.tool in ('rectangle', 'circle') and len(self.points) == 2:
            self._commit()
        self._paint()
        return True

    def move(self, pos, buttons=Qt.NoButton):
        if not self.active() or buttons & Qt.MiddleButton:
            return False
        if self.tool=='select' and self._selection_bridge is not None:
            return self._selection_bridge.move(pos,buttons)
        if self._legacy_tool_dialog is not None and self.tool in ('parallel', 'centerline'):
            # Avoid repainting millions of points on every cursor move.
            return True
        try:
            p = self._point(pos)
        except RuntimeError:
            return True
        self._last_pointer_pos = pos
        if self.tool == 'accudraw' and self.points:
            self._set_palette_xyz((p[0],p[1],self._last_z))
        if self.tool in ('movevertex','vertex') and self.vertex_target is not None:
            if buttons & Qt.LeftButton:
                _, vi = self.vertex_target
                origin = self._vertex_move_original['coords'][vi]
                if math.dist(p, origin) > max(2.*self.canvas.world_per_pixel(), 1.e-10):
                    self._vertex_move_dragged = True
                    self.dragging = True
                    self._vertex_move_preview(p)
            elif self.tool == 'movevertex' or self._vertex_insert_pending:
                # Tap/tap mode: keep a live preview even without the button held.
                self._vertex_move_preview(p)
            return True
        if self.tool == 'parallel' and self._parallel_source is not None:
            self._paint()
            return True
        if self.tool == 'freehand' and self.dragging and buttons & Qt.LeftButton:
            self.points.append(p)
        else:
            self.hover = p if self.points else None
        self._paint()
        return True

    def release(self, pos, button):
        if not self.active() or button != Qt.LeftButton:
            return False
        if self.tool=='select' and self._selection_bridge is not None:
            return self._selection_bridge.release(pos,button)
        if self.tool in ('movevertex','vertex') and self.vertex_target is not None:
            if self._vertex_move_dragged:
                try:
                    destination=self._point(pos)
                except RuntimeError:
                    destination=self._vertex_move_original['coords'][self.vertex_target[1]]
                self._vertex_move_finish(destination)
                self._vertex_insert_pending = False
            return True
        if self.tool == 'freehand' and self.dragging:
            try:
                self.points.append(self._point(pos))
            except RuntimeError:
                self.cancel(stand_down=False)
                return True
            self._commit()
            return True
        return self.tool in ('line', 'rectangle', 'circle')

    def double_click(self, pos, button):
        if self.tool=='select' and self._selection_bridge is not None:
            return self._selection_bridge.double_click(pos,button)
        if self.tool == 'line' and button == Qt.LeftButton:
            # Qt may deliver the double-click at a new vertex without a second
            # ordinary press. Match Smart Line: include that final position once.
            if self.points:
                try:
                    end = self._point(pos)
                except RuntimeError:
                    end = None
                if end is not None and end != self.points[-1]:
                    self.press(pos, Qt.LeftButton)
            self._finish_line()
            return True
        if self.active() and button == Qt.LeftButton and self.tool in ('smartline', 'polyline', 'ortho', 'curve', 'hatch', 'accudraw'):
            try:
                p = self._point(pos)
            except RuntimeError:
                return True
            if not self.points or p != self.points[-1]:
                self.point_redo.clear(); self._z_redo.clear()
                self.points.append(p)
                if self.tool == 'accudraw':
                    self._z_points.append(self._last_z)
            self._finish()
            return True
        return False

    def key(self, key):
        if not self.active():
            return False
        if self.tool=='select' and self._selection_bridge is not None:
            if self._selection_bridge.key(key):return True
        if key == Qt.Key_Escape:
            if self._legacy_tool_dialog is not None:
                dialog = self._legacy_tool_dialog
                if getattr(dialog, '_select_mode', False):
                    dialog._on_finish_selection()
                else:
                    dialog.reject()
                return True
            self.cancel()
            return True
        if key in (Qt.Key_Return, Qt.Key_Enter) and self.tool == 'line':
            self._finish_line()
            return True
        if key in (Qt.Key_Return, Qt.Key_Enter) and self.tool in (
                'smartline','polyline','ortho','curve','hatch','accudraw'):
            self._finish()
            return True
        if key in (Qt.Key_Backspace, Qt.Key_Delete) and self.points:
            if self.tool in self._POINT_TOOLS:
                return self.history(redo=False)
            self.points.pop()
            self._refresh_placement_preview()
            self._paint()
            return True
        return False

    def history(self, redo=False):
        if self.tool == 'line' and (self.points or self.point_redo or
                                    self._line_session_refs or self._line_session_undone):
            return self._line_history(redo=redo)
        # While a multi-click placement is in progress, document/object Undo
        # must never delete a *completed* drawing. The point redo list remains
        # populated even after Undo removes the first and final pending vertex.
        if self.active() and self.tool in self._POINT_TOOLS and (self.points or self.point_redo):
            if redo:
                if self.point_redo:
                    self.points.append(self.point_redo.pop())
                    if self.tool == 'accudraw':
                        self._z_points.append(self._z_redo.pop())
            elif self.points:
                self.point_redo.append(self.points.pop())
                if self.tool == 'accudraw':
                    self._z_redo.append(self._z_points.pop())
            self._angle_sticky = False
            if self.tool == 'accudraw' and self.points:
                self._last_z = self._z_points[-1]
                self._set_palette_xyz((*self.points[-1],self._last_z))
            self._refresh_placement_preview()
            self._paint()
            return True

        if self.vertex_target is not None and self._vertex_move_before is not None:
            # Ctrl+Z during a pending mouse drag cancels the preview; it must
            # not undo an unrelated already-completed geometry object.
            self.drawings[:] = self._vertex_move_before
            self.vertex_target = None
            self._vertex_move_before = None
            self._vertex_move_original = None
            self._vertex_move_dragged = False
            self._vertex_insert_pending = False
            self.dragging = False
            self._paint()
            return True
        # Grouped edit actions (Move/Copy/Rotate/Mirror/Delete) share the same
        # document history as completed native Draw shapes.
        changes = self._object_redo if redo else self._object_actions
        target_actions = self._object_actions if redo else self._object_redo
        if changes:
            before, after = changes.pop()
            self.drawings[:] = after if redo else before
            target_actions.append((before,after))
            if self._selection_bridge is not None:
                self._selection_bridge.selection_manager._prune()
            self._paint()
            return True
        # Pre-existing legacy Draw history, if any, remains available.
        if hasattr(self, '_object_actions'):
            return True
        source, target = (self.redo, self.drawings) if redo else (self.drawings, self.redo)
        if source:
            target.append(source.pop())
            self._paint()
        return True

    def clear(self):
        self._line_session_refs.clear()
        self._line_session_undone.clear()
        if self.drawings:
            before=list(self.drawings)
            self.drawings.clear()
            self.record_object_change(before)
        self.points, self.hover = [], None
        self.point_redo.clear(); self._z_redo.clear(); self._z_points.clear()
        self._last_pointer_pos = None
        self._store_selection(None)
        self._paint()

    def cancel(self, stand_down=True):
        self._line_session_refs.clear()
        self._line_session_undone.clear()
        if self.vertex_target is not None and self._vertex_move_before is not None:
            self.drawings[:] = self._vertex_move_before
        self._vertex_move_before = None
        self._vertex_move_original = None
        self._vertex_move_dragged = False
        self._vertex_insert_pending = False
        self._text_options = None
        self._close_palettes()
        self.points, self.hover, self.dragging, self.vertex_target = [], None, False, None
        self.point_redo.clear(); self._z_redo.clear(); self._z_points.clear()
        self._last_pointer_pos = None
        self._angle_sticky = False
        self._snap_target = None
        self._parallel_source = None
        self._centerline_first = None
        if stand_down:
            self.tool = None
        self._paint()

    def _paint(self):
        shapes = []
        bridge = self._selection_bridge
        chosen = (bridge.selection_manager._selected if bridge is not None else {})
        for i, drawing in enumerate(self.drawings):
            data = self._style(drawing.get('type'))
            data.update(drawing)
            data['selected'] = (('native', id(drawing)) in chosen if bridge is not None
                                else i == self.selected and drawing is self._selected_native_ref)
            data['show_vertices'] = bool(self.tool in ('movevertex','vertex','deletevertex') or (
                bridge is not None and getattr(bridge,'_vertex_display',False)
                and data['selected']))
            coords = drawing.get('coords', [])
            if drawing.get('type') == 'curve':
                data['control_points'] = list(coords)
                coords = _catmull_rom(coords)
            shapes.append(('drawing', coords, data))
        if self.points:
            preview = self.points + ([self.hover] if self.hover else [])
            # Draw Polyline is the polygon tool in the ribbon. Show the closing
            # edge during placement, rather than waiting until completion, so
            # the preview always communicates that the result is a closed ring.
            if self.tool == 'polyline' and len(preview) >= 3 and preview[-1] != preview[0]:
                preview = preview + [preview[0]]
            data = {'type': self.tool, **self._style(self.tool)}
            if self.tool == 'accudraw' and self._angle_sticky:
                data['color'] = '#22ffd7'  # angle-match preview
            if self.tool == 'curve':
                data['control_points'] = list(self.points)
                preview = _catmull_rom(preview)
            shapes.append(('drawing_preview', preview, data))
        # Show an imported SNT selection without modifying/rendering its GPU
        # buffers. The selected line is highlighted on the same MVP overlay.
        imported=self._selected_feature() if self._selected_imported is not None else None
        if imported is not None and imported['source']=='snt' and self._selection_bridge is None:
            shapes.append(('drawing',imported['coords'],
                           {'type':'polyline','selected':True,'width':3}))
        if self.tool=='parallel' and self._parallel_source is not None and self._last_pointer_pos is not None:
            try:
                click=self.canvas.section_at(self._last_pointer_pos.x(),
                                             self._last_pointer_pos.y())
                parallel=self._parallel_output(self._parallel_source,click)
            except (RuntimeError,TypeError,ValueError):
                parallel=None
            if parallel is not None and abs(parallel[1])>1.e-9:
                shapes.append(('drawing_preview', parallel[0],
                               {'type':'parallel','color':'#39ffc0','width':2}))
        if self._snap_target is not None and self.tool not in ('select','parallel','centerline'):
            shapes.append(('drawing_preview',[self._snap_target],
                           {'type':'point','color':'#00e676','width':3}))
        # Highlight current dialog picks, without copying/mutating imported SNT
        # GPU geometry. A removed attachment is invalidated automatically.
        if self._legacy_tool_dialog is not None:
            for feat in self._legacy_dialog_highlights:
                if not self._valid_source(feat):
                    continue
                coords = feat['coords']
                shapes.append(('drawing', coords,
                               {'type':'point' if len(coords)==1 else 'polyline',
                                'selected':True,'width':3}))
        if self._selection_bridge is not None:
            for feat in self._selection_bridge.highlight_features():
                if feat['source']=='snt':
                    shapes.append(('drawing',feat['coords'],
                                   {'type':'point' if len(feat['coords'])==1 else 'polyline',
                                    'selected':True,'width':3}))
            shapes.extend(self._selection_bridge.preview_shapes())
        shapes.extend(self.aux_shapes)
        self.overlay.set_shapes(shapes)

    def shutdown(self):
        dialog = self._legacy_tool_dialog
        self._legacy_tool_dialog = None
        self._legacy_dialog_highlights = []
        if dialog is not None:
            try:
                dialog.close()
            except RuntimeError:
                pass
        self.cancel()
        self.overlay._timer.stop()
        self.overlay.close()


def get_native_draw(app, create=False):
    """Return the draw layer owning the *visible* native viewport.

    LNT's independent Vulkan widget/input filter retains its original path.
    Otherwise a visible ordinary Vulkan plan viewport gets a separate Draw
    overlay for SNT and LAZ (including mixed loads in either order). A VTK
    viewport retains the existing legacy digitizer and is never hijacked.
    """
    view = getattr(app, '_lnt_main_view', None)
    if view is not None:
        if getattr(view, 'native', None) is None or not getattr(view, 'mode_2d', False):
            return None
    else:
        owner = getattr(app, 'render_backend', None)
        if (owner is None or not getattr(owner, 'active', False)
                or not owner.vulkan_viewport_is_visible()
                or not owner.owns_main_camera()):
            return None
        view = getattr(owner, '_plan_draw_view', None)
        if view is None and create:
            # The SNT Vulkan vector layer already has an exact framebuffer
            # mapping (_MvpScreenMapping) for its Qt text overlay. Reuse that
            # view and MainCanvas instead of inventing a second camera mapping:
            # new lines land on the same pixels as loaded SNT entities.
            from gui.native_vector_overlay import _view_for
            view = _view_for(app)
            if view is None:
                return None
            view._surface_draw = True
            owner._plan_draw_view = view
        if view is None:
            return None
    draw = getattr(view, 'drawing', None)
    if (draw is not None and getattr(view, '_surface_draw', False)
            and draw.canvas.widget is not view.owner.vulkan_widget):
        # LNT can replace the app's Vulkan QWidget while open. Retire a
        # stale overlay before drawing in the replacement viewport.
        view._draw_saved_shapes = list(draw.drawings)
        draw.shutdown()
        view.drawing = draw = None
    if draw is None and create:
        draw = view.drawing = NativeDraw(view)
        if getattr(view, '_surface_draw', False):
            draw.drawings.extend(getattr(view, '_draw_saved_shapes', ()))
            view._draw_saved_shapes = []
            draw._paint()
    return draw


def activate_native_draw(app, tool):
    draw = get_native_draw(app, create=True)
    return bool(draw and draw.activate(tool))
