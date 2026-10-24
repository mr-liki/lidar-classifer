"""Drawing tools for the native LNT/Vulkan plan viewport; no VTK objects."""
from __future__ import annotations

import math
import time
from functools import lru_cache
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QInputDialog

from gui.native_selection_targets import (
    close_freehand, snap_draft_closure, editable_draw, parallel_line_source,
    drawn_open_line_source,
)
from gui.native_stroke_geometry import (
    catmull_rom_vtk, drop_consecutive_duplicates, snap_endpoint_near_start,
)
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
    # Internal interactive operation, entered only from a native text menu.
    'movetext': 'movetext', 'move_text': 'movetext',
}


@lru_cache(maxsize=16)
def _catmull_rom_cached(controls):
    """Reuse recent static splines without caching cursor-dependent geometry."""
    return tuple(catmull_rom_vtk(controls))


def _catmull_rom(points, steps=20):
    """Sample the same Catmull-Rom interpolation as original VTK CurveTool."""
    if steps == 20:
        controls = tuple(tuple(p) for p in points)
        # Never retain huge sampled curves in the LRU. Large strokes are
        # rendered on demand, preventing preview caches from growing with
        # LiDAR-scene working sets or very long user-created Curves.
        if len(controls) <= 128:
            return list(_catmull_rom_cached(controls))
        return catmull_rom_vtk(controls)
    return catmull_rom_vtk(points, steps=steps)


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
        self._hover_selected_native_ref = None
        self._hover_pick_at = 0.0
        self._selection_scene_token = None
        self._parallel_source = None
        self._centerline_first = None
        self._snt_pick_index = None
        self._snt_pick_token = None
        self._snap_target = None
        # VTK Freehand: left release pauses, another drag resumes, right-click
        # commits. Keep per-drag checkpoints so Undo never deletes a finished
        # figure while a Freehand draft remains pending.
        self._freehand_before_drag = None
        self._freehand_undo_strokes = []
        self._freehand_redo_strokes = []
        self._freehand_last_snap_at = 0.0
        self.vertex_target = None
        # Pending Move Vertex keeps its BEFORE object immutable until commit.
        # Live mouse-drag previews replace only an editable native drawing.
        self._vertex_move_before = None
        self._vertex_move_original = None
        self._vertex_move_dragged = False
        self._vertex_insert_pending = False
        self._text_options = None
        # Move native text by replacing its immutable world-coordinate record.
        # No LiDAR/classification/SNT entity is edited.
        self._moving_text_ref = None
        self._moving_text_hover = None
        self._moving_text_grab_offset = (0., 0.)
        self.aux_shapes = []
        self._selection_bridge = None  # active Vulkan-only Select panel
        # The legacy Parallel/Centerline dialogs own interaction when Vulkan is
        # visible. Keep their original algorithms and controls; only picking and
        # result delivery need an adapter. Never register VTK observers here.
        self._legacy_tool_dialog = None
        self._legacy_dialog_highlights = []
        # Dialog-owned selected fence outlines, independent of the VTK actors.
        # Geometry is world XY; class colors and native point buffers are untouched.
        self._classification_fence_highlights = {}
        self._classification_fence_highlight_refs = {}
        # Selection is a VIEW state, not a classification/used-fence flag.
        # Exactly one group owns the permanent cyan selection at a time. The
        # popup's checkbox preview has a separate temporary overlay.
        self._active_fence_highlight_owner = None
        self._active_fence_shapes = []
        self._fence_selection_revision = 0
        self._fence_picker_preview_owner = None
        self._fence_picker_preview_rings = []
        self._fence_picker_hover_owner = None
        self._fence_picker_hover_ring = None
        self._object_actions = []
        self._object_redo = []
        self._shutting_down = False
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

    def _source_visual_style(self, feature):
        """Copy the visible appearance of the selected native or SNT line.

        Parallel is a derived CAD entity, not a transient rubber-band. Its
        authored appearance must survive style edits to unrelated tools.
        """
        if feature.get('source') == 'native':
            drawing = feature['drawing']
            configured = self._style(drawing.get('type'))
            configured.update({k: drawing[k] for k in ('color', 'width', 'style')
                               if drawing.get(k) is not None})
            # Older ParallelToolDialog records persisted 'line_style' only.
            # Their original stroke must remain inheritable too.
            if drawing.get('style') is None and drawing.get('line_style') is not None:
                configured['style'] = drawing['line_style']
            rgb = configured.get('color')
            if isinstance(rgb, (tuple, list)) and len(rgb) >= 3:
                # Native records may carry byte RGB or normalized RGB values.
                components = [float(c) for c in rgb[:3]]
                if all(0. <= c <= 1. for c in components):
                    components = [255. * c for c in components]
                configured['color'] = '#%02x%02x%02x' % tuple(
                    max(0, min(255, round(c))) for c in components)
            return {k: configured[k] for k in ('color', 'width', 'style')
                    if k in configured}
        if feature.get('source') == 'snt':
            entity = feature.get('entity') or {}
            # Use the *same* normalization and lineweight mapping used by the
            # Vulkan SNT vector renderer, not the Parallel tool's defaults.
            from gui.native_vector_overlay import _entity_rgb, _line_width_px
            r, g, b = (int(v) for v in _entity_rgb(entity))
            result = {'color': f'#{r:02x}{g:02x}{b:02x}',
                      'width': max(1, round(_line_width_px(entity.get('_lineweight', 0))))}
            raw = str(entity.get('style') or entity.get('line_style') or 'solid').lower()
            raw = {'dash': 'dashed', 'dot': 'dotted', 'dashdot': 'dash-dot',
                   'dashdotdot': 'dash-dot-dot'}.get(raw, raw)
            result['style'] = (raw if raw in ('solid', 'dashed', 'dotted',
                                             'dash-dot', 'dash-dot-dot') else 'solid')
            return result
        return {}

    def _native_text_at(self, pos):
        """Pick actual screen-space glyph bounds at *any* camera zoom.

        Unlike an ordinary drawing's one-point geometry, a label is picked
        throughout its text, not only within 12 px of its insertion anchor.
        This uses the same projection/transform as CanvasOverlay.paintEvent.
        """
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QFontMetricsF
        from gui.native_label_layout import native_draw_text_layout
        from gui.draw_clear_selection import is_clearable_draw
        snapshot = self.canvas.projection_snapshot()
        project = snapshot[1] if snapshot is not None else self.canvas.screen_from_section
        screen = QPointF(pos.x(), pos.y())
        for drawing in reversed(self.drawings):
            if (drawing.get('type') != 'text' or
                    not is_clearable_draw(drawing)):
                continue
            coords = drawing.get('coords')
            if coords is None or len(coords) == 0 or not str(drawing.get('text', '')).strip():
                continue
            font, transform = native_draw_text_layout(project, coords[0], drawing,
                                                      self.overlay.font())
            bounds = QFontMetricsF(font).boundingRect(str(drawing['text']))
            if not transform.mapRect(bounds).adjusted(-4, -4, 4, 4).contains(screen):
                continue
            inverse, ok = transform.inverted()
            if not ok:
                continue
            # Precise rotated hit-test, with a small screen-pixel allowance.
            scale = max(0.1, min(math.hypot(transform.m11(), transform.m12()),
                                 math.hypot(transform.m21(), transform.m22())))
            pad = 4. / scale
            if bounds.adjusted(-pad, -pad, pad, pad).contains(inverse.map(screen)):
                return drawing
        return None

    def try_text_context(self, pos):
        """Consume a right-click ONLY when it hits an editable native label."""
        if getattr(self, '_shutting_down', False):
            return False
        drawing = self._native_text_at(pos)
        if drawing is None:
            return False
        # Showing QMenu synchronously from MouseButtonPress creates a nested
        # event loop before button release on Windows; defer it one Qt tick.
        from PySide6.QtCore import QTimer
        global_pos = self.canvas.widget.mapToGlobal(pos.toPoint())
        # Keep the original local hit position to preserve the user's grab
        # point when Move Text is selected from the context menu.
        from PySide6.QtCore import QPointF
        hit_pos = QPointF(pos)
        QTimer.singleShot(0, lambda d=drawing, p=global_pos, h=hit_pos:
                          self._show_text_context_menu(d, p, h))
        return True

    def _show_text_context_menu(self, drawing, global_pos, hit_pos=None):
        if getattr(self, '_shutting_down', False) or not any(
                d is drawing for d in self.drawings):
            return
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self.app)
        edit_action = menu.addAction('Edit Text...')
        move_action = menu.addAction('Move Text...')
        delete_action = menu.addAction('Delete Text')
        chosen = menu.exec(global_pos)
        if chosen == edit_action:
            self._edit_native_text(drawing)
        elif chosen == move_action:
            self._begin_native_text_move(drawing, hit_pos)
        elif chosen == delete_action:
            self.delete_drawings([id(drawing)])

    def _begin_native_text_move(self, drawing, hit_pos=None):
        """Arm a single text relocation, preserving the grabbed glyph position.

        This internal tool receives input through both native LNT and promoted
        Vulkan viewport routes; the prior Draw/Select tool is deactivated via
        the ordinary activation path. Original geometry is unchanged until
        the destination click, so Cancel cannot damage the document.
        """
        if not any(item is drawing for item in self.drawings):
            return False
        coords = drawing.get('coords') or ()
        if not coords or drawing.get('type') != 'text':
            return False
        anchor = tuple(coords[0])
        if len(anchor) < 2:
            return False
        delta = (0., 0.)
        if hit_pos is not None:
            try:
                picked = self.canvas.section_at(hit_pos.x(), hit_pos.y())
                delta = (float(anchor[0])-float(picked[0]),
                         float(anchor[1])-float(picked[1]))
            except (RuntimeError, TypeError, ValueError):
                pass  # Fall back to selecting a new anchor directly.
        if not self.activate('movetext'):
            return False
        self._moving_text_ref = drawing
        self._moving_text_grab_offset = delta
        self._moving_text_hover = (float(anchor[0]), float(anchor[1]))
        self.app.statusBar().showMessage(
            'Move Text: move pointer and left-click to place; Esc/right-click cancels',
            7000)
        self._paint()
        return True

    def _text_move_world_at(self, pos):
        """Project cursor to XY without snapping to the label being moved."""
        world = self.canvas.section_at(pos.x(), pos.y())
        dx, dy = self._moving_text_grab_offset
        return (float(world[0]) + dx, float(world[1]) + dy)

    def _commit_native_text_move(self, pos):
        """Commit the new world position as exactly one Draw Undo/Redo step."""
        original = self._moving_text_ref
        if original is None or not any(d is original for d in self.drawings):
            self.cancel()
            return True
        try:
            xy = self._text_move_world_at(pos)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.app.statusBar().showMessage(f'Move Text: cannot project point: {exc}', 3000)
            return True  # Keep the active move; user can try again or cancel.
        old_coords = list(original['coords'])
        old_anchor = tuple(old_coords[0])
        tolerance = max(1.e-10, 0.1 * self.canvas.world_per_pixel())
        if math.dist(xy, old_anchor[:2]) <= tolerance:
            self.cancel()
            self.app.statusBar().showMessage('Move Text: position unchanged', 2500)
            return True
        old_coords[0] = (xy[0], xy[1], *old_anchor[2:])
        before = list(self.drawings)
        replacement = dict(original, coords=old_coords)
        index = next(i for i, d in enumerate(self.drawings) if d is original)
        self.drawings[index] = replacement
        self._moving_text_ref = None
        self._moving_text_hover = None
        self.record_object_change(before)
        self._prune_deleted_native_refs({id(original)})
        self.cancel()
        self.app.statusBar().showMessage('Text moved (Ctrl+Z to undo)', 3500)
        return True

    def _edit_native_text(self, drawing):
        """Reopen the existing editor; commit one immutable Undo/Redo step."""
        if not any(d is drawing for d in self.drawings):
            return False
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QDialog
        from gui.digitize_tools import TextEditDialog
        raw_color = drawing.get('color')
        if raw_color is None:
            raw_color = '#ff0000'
        if isinstance(raw_color, (tuple, list)) and len(raw_color) >= 3:
            channels = [float(v) for v in raw_color[:3]]
            if all(0. <= c <= 1. for c in channels):
                color = QColor.fromRgbF(*channels)
            else:
                color = QColor(*(max(0, min(255, round(c))) for c in channels))
        else:
            color = QColor(str(raw_color))
        if not color.isValid():
            color = QColor('#ff0000')
        dialog = TextEditDialog(
            current_text=str(drawing.get('text', '')),
            current_size=max(1, int(drawing.get('font_size') or 75)),
            current_font=drawing.get('font_family', 'Arial'),
            current_bold=bool(drawing.get('bold', False)),
            current_italic=bool(drawing.get('italic', False)),
            current_color=(color.redF(), color.greenF(), color.blueF()),
            parent=self.app,
        )
        if dialog.exec() != QDialog.Accepted:
            return False
        values = dialog.get_values()
        if not str(values.get('text', '')).strip():
            self.app.statusBar().showMessage('Text cannot be empty; edit cancelled.', 3000)
            return False
        old_size = max(1, int(drawing.get('font_size') or 75))
        new_size = max(1, int(values.get('font_size') or old_size))
        old_height = float(drawing.get('height') or 0.)
        from gui.native_label_layout import native_draw_text_height
        # Existing world-sized text must not jump in size when editing at a
        # different zoom. First-time edits to old records establish a height.
        new_height = (old_height * new_size / old_size if old_height > 0. else
                      native_draw_text_height(values, self.canvas.world_per_pixel()))
        rgb = values.get('color', (1., 0., 0.))
        hex_color = '#%02x%02x%02x' % tuple(
            max(0, min(255, round(float(c) * 255))) for c in rgb[:3])
        before = list(self.drawings)
        replacement = dict(drawing, text=values['text'], color=hex_color,
                           font_family=values.get('font_family', 'Arial'),
                           font_size=new_size, bold=bool(values.get('bold')),
                           italic=bool(values.get('italic')), height=new_height)
        idx = next(i for i, d in enumerate(self.drawings) if d is drawing)
        self.drawings[idx] = replacement
        self.record_object_change(before)
        # No in-place mutation: the previous object remains valid in Undo.
        # Remove only stale UI selection references, not unrelated fences.
        self._prune_deleted_native_refs({id(drawing)})
        self.app.statusBar().showMessage('Text updated (Ctrl+Z to undo)', 3000)
        return True

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
        if tool != 'select' and self._selection_bridge is not None:
            if getattr(self._selection_bridge,'_intersect_mode',False):
                self._selection_bridge.exit_intersect_mode(commit=True)
        self.cancel(stand_down=False)
        # Pick up settings saved while this native viewport was not visible.
        from gui.draw_settings_dialog import load_draw_settings
        self.styles = load_draw_settings()
        self._hover_selected_native_ref = None
        self._hover_pick_at = 0.0
        if tool != 'select' and self._selection_bridge is not None:
            self._selection_bridge._active = False
            self._selection_bridge = None
            # Tool switch must also clear stale cyan in the separate
            # Measure Line/Path overlay (Draw's overlay was just reset).
            repaint_measure = getattr(measurement, '_paint', None)
            if callable(repaint_measure):
                repaint_measure()
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
            values = dialog.get_values() if dialog.exec() == QDialog.Accepted else None
            if values is None or not str(values.get('text', '')).strip():
                # Cancelled (Esc / Cancel / empty text): the Text tool stands
                # down here. Still report the activation as handled, otherwise
                # the ribbon falls back to the VTK digitizer, whose own Text
                # dialog would reopen behind this one.
                self.cancel()
                clear_buttons = getattr(self.app, '_clear_ribbon_active_buttons', None)
                if callable(clear_buttons):
                    clear_buttons('draw')
                self.app.statusBar().showMessage('Text tool deactivated', 2500)
                return True
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
            return ('Native Freehand: left-drag to draw, release to pause, '
                    'drag to resume, right-click to finish')
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

    def set_classification_fence_hover(self, owner, shape=None):
        """Preview one picker row without changing selected fences or drawings."""
        from gui.native_byclass_fences import _valid_ring
        key = id(owner)
        if shape is None:
            if getattr(self, '_fence_picker_hover_owner', None) != key:
                return
            self._fence_picker_hover_owner = None
            self._fence_picker_hover_ring = None
        else:
            ring = _valid_ring(shape.get('coords', ()))
            self._fence_picker_hover_owner = key if ring else None
            self._fence_picker_hover_ring = ring
        self._paint()

    def set_classification_fence_highlights(self, owner, fences, *,
                                            preview=False, replace=False):
        """Show only the *current* selection in cyan, never historical fences.

        `preview` belongs to a checkbox picker and does not commit selection.
        A stale dialog clearing/closing cannot erase a newer owner's selection.
        `replace=True` is used by actual canvas selection (including a click on
        empty canvas), which must be authoritative even when the set is empty.
        """
        from gui.native_byclass_fences import _valid_ring, fence_identity
        rings, refs, shapes, seen = [], [], [], set()
        for shape in fences:
            if not isinstance(shape, dict):
                continue
            coords = shape.get('coords')
            ring = _valid_ring(coords) if coords is not None else None
            if ring is None:
                continue
            identity = fence_identity(shape)
            if identity in seen:
                continue
            seen.add(identity)
            rings.append(ring)
            shapes.append(shape)
            ref = shape.get('native_ref')
            if ref is None and any(shape is obj for obj in getattr(self, 'drawings', ())):
                ref = shape
            refs.append(id(ref) if ref is not None else None)

        key = id(owner)
        # Some older selection/Undo sessions construct an uninitialized
        # controller before its overlay is attached. Initialize lazily rather
        # than breaking Stage 1-5 selection tests or viewport restoration.
        if not hasattr(self, '_classification_fence_highlights'):
            self._classification_fence_highlights = {}
        if not hasattr(self, '_classification_fence_highlight_refs'):
            self._classification_fence_highlight_refs = {}
        if not hasattr(self, '_active_fence_highlight_owner'):
            self._active_fence_highlight_owner = None
        if not hasattr(self, '_fence_picker_preview_owner'):
            self._fence_picker_preview_owner = None
            self._fence_picker_preview_rings = []
        if preview:
            self._fence_picker_preview_owner = key
            self._fence_picker_preview_rings = rings
            self._paint()
            return

        active = getattr(self, '_active_fence_highlight_owner', None)
        if not rings and key != active and not replace:
            # Stale dialog cleanup; an older owner must not clear new picks.
            if self._fence_picker_preview_owner == key:
                self._fence_picker_preview_owner = None
                self._fence_picker_preview_rings = []
                self._paint()
            return
        previous = list(getattr(self, '_active_fence_shapes', ()))
        previous_ids = [fence_identity(item) for item in previous]
        new_ids = [fence_identity(item) for item in shapes]
        # Retire ALL previous dialog outlines rather than rendering their union.
        self._classification_fence_highlights.clear()
        self._classification_fence_highlight_refs.clear()
        if rings:
            self._classification_fence_highlights[key] = rings
            self._classification_fence_highlight_refs[key] = refs
        self._active_fence_highlight_owner = key if rings else None
        self._active_fence_shapes = shapes
        if previous_ids != new_ids:
            self._fence_selection_revision = getattr(self, '_fence_selection_revision', 0) + 1
        # A new committed selection supersedes any checkbox preview.
        self._fence_picker_preview_owner = None
        self._fence_picker_preview_rings = []
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
        for i,drawing in enumerate(getattr(self, 'drawings', ())):
            coords=self._native_coords(drawing)
            if not coords:continue
            lo_x=min(q[0] for q in coords);hi_x=max(q[0] for q in coords)
            lo_y=min(q[1] for q in coords);hi_y=max(q[1] for q in coords)
            if (lo_x-tolerance <= p[0] <= hi_x+tolerance and
                    lo_y-tolerance <= p[1] <= hi_y+tolerance):
                yield {'source':'native', 'index':i, 'coords':coords,
                       'type':drawing.get('type','line'), 'key':('native',id(drawing)),
                       'drawing':drawing}
        # Avoid creating/scanning an SNT index for an empty attachment list;
        # native drawings and raw Freehand moves never need a LiDAR pass.
        if not getattr(self.app, 'snt_attachments', None):
            return
        index = self._snt_index()
        yield from index.nearby(p, tolerance, visible=self._snt_visible)

    def _pick_editable_draw(self, p, tolerance=None):
        """Standalone Draw > Select cannot pick imported SNT reference blocks.

        The Hatch picker retains its separate SNT inputs.
        The Element Selection panel additionally supports measured Line/Path.
        """
        if tolerance is None:
            tolerance=max(self.canvas.world_per_pixel()*12.,1e-9)
        best=(tolerance,None)
        for i,d in enumerate(self.drawings):
            if not editable_draw(d):continue
            coords=self._native_coords(d)
            if not coords:continue
            feat={'source':'native','index':i,'coords':coords,
                  'type':d.get('type','line'),'key':('native',id(d)),
                  'drawing':d}
            if len(coords)>2 and coords[0]==coords[-1] and _inside_polygon(p,coords):
                distance=0.
            else:
                distance=self._feature_distance(p,feat)
            if distance<=best[0]:best=(distance,feat)
        return best[1]

    @staticmethod
    def _feature_distance(p, feature):
        coords=feature['coords']
        if len(coords)==1:
            return math.dist(p,coords[0])
        info=nearest_segment(p,coords)
        return info[0] if info else math.inf

    def _pick_feature(self, p, tolerance=None, open_only=False, exclude=None, predicate=None):
        if tolerance is None:
            tolerance=max(self.canvas.world_per_pixel()*12., 1.e-9)
        best=None
        for feature in self._near_scene(p, tolerance):
            if predicate is not None and not predicate(feature):
                continue
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
            if predicate is not None and not predicate(feature):
                continue
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
        # A standalone Draw > Select click is a new selection, not a request
        # to keep old cyan fences from a previous By Class dialog. Selection
        # never undoes a previously applied LiDAR classification.
        from gui.native_fence_selection import selected_fence_shapes
        selected_shapes = selected_fence_shapes(
            [feature['drawing']] if feature is not None
            and feature.get('source') == 'native' else [], native_draw=self,
        )
        self.set_classification_fence_highlights(self, selected_shapes, replace=True)
        from gui.native_fence_selection import propagate_native_fence_selection
        propagate_native_fence_selection(self.app, selected_shapes)

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
        if mode=='center':
            # Line stores independent edges. A completed closed chain is
            # still a closed Draw figure even when no edge is near the click.
            chains={}
            for drawing in getattr(self,'drawings',()):
                chain=drawing.get('line_chain')
                if (chain and drawing.get('type')=='line' and
                        not drawing.get('imported') and editable_draw(drawing)):
                    chains.setdefault(chain,[]).append(drawing)
            for chain,edges in chains.items():
                ring=[]
                for edge in edges:
                    coords=edge.get('coords',[])
                    if len(coords)<2 or (ring and tuple(ring[-1])!=tuple(coords[0])):
                        break
                    ring.extend(coords if not ring else coords[1:])
                else:
                    if len(ring)>=4 and tuple(ring[0])==tuple(ring[-1]):
                        features.append(dict(source='native',type='polyline',
                            coords=ring,key=('line_chain',chain)))
        best=None
        endpoint=None
        containing_center=None
        line_keypoint=None
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
                if mode in ('nearby','keypoint') and feature.get('type') != 'circle':
                    # Only a native Curve's original control endpoints can be
                    # CAD endpoint targets: its spline tessellation must not
                    # create artificial editable/snappable vertices.
                    vertex_coords = coords
                    if (feature.get('type') == 'curve' and
                            feature.get('source') == 'native'):
                        controls = feature.get('drawing', {}).get('coords', ())
                        vertex_coords = ([controls[0], controls[-1]] if controls
                                         else ())
                    if mode=='keypoint' and len(coords)>=2 and len(vertex_coords):
                        hit=nearest_segment(p,coords)
                        if hit is not None and hit[0]<=tol:
                            # Pick the line by its stroke, then its nearest
                            # real vertex. Vertex distance has no radius cap.
                            vertex=min((tuple(q[:2]) for q in vertex_coords),
                                       key=lambda q: math.dist(p,q))
                            rank=(hit[0],math.dist(p,vertex),vertex)
                            if line_keypoint is None or rank<line_keypoint:
                                line_keypoint=rank
                    for q in vertex_coords:
                        if recent_line and tuple(q[:2]) == self.points[-1]:
                            continue
                        offer(tuple(q[:2]), vertex=True)
                if mode == 'center':
                    center=None
                    inside=False
                    if feature.get('type')=='circle' and feature['source']=='native':
                        controls=feature['drawing']['coords']
                        center=tuple(controls[0][:2])
                        if len(controls)>=2:
                            radius=math.dist(center,controls[1][:2])
                            inside=radius>0. and math.dist(p,center)<=radius
                    elif len(coords)>=4 and coords[0]==coords[-1]:
                        center=polygon_centroid(coords)
                        inside=_inside_polygon(p,coords[:-1])
                    if center is not None:
                        if inside and feature.get('source')=='native':
                            # Interior selection is by figure containment,
                            # independent of the normal pixel snap radius.
                            rank=(math.dist(p,center),tuple(center))
                            if containing_center is None or rank<containing_center:
                                containing_center=rank
                        offer(center)
                for a,b in zip(coords,coords[1:]):
                    if recent_line:
                        continue
                    dist,foot,_=segment_projection(p,a,b)
                    if mode == 'nearby' and dist<=tol:offer(foot)
                    if mode == 'midpoint':offer(((a[0]+b[0])*.5,(a[1]+b[1])*.5))
            # A draft may snap back to its own earlier points, not the most
            # recent vertex (which would prevent drawing the next segment).
            if mode in ('nearby','keypoint') and self.tool not in ('freehand', 'curve'):
                for q in self.points[:-1]:offer(q,vertex=True)
        if containing_center is not None:
            return containing_center[1]
        if line_keypoint is not None:
            return line_keypoint[2]
        if mode=='nearby' and endpoint is not None:
            return endpoint[1]
        return best[1] if best else None

    def _point(self, pos, snap=True):
        p=self.canvas.section_at(pos.x(),pos.y())
        raw_p=p  # Curve closure is based on cursor position, before scene snapping.
        did_snap=False
        self._snap_target=None
        digitizer=getattr(self.app,'digitizer',None)
        explicit_snap = bool(digitizer is not None and getattr(digitizer,'snap_enabled',False))
        exact_snap_mode = getattr(digitizer,'snap_mode',None) if explicit_snap else None
        if (snap and exact_snap_mode in ('center','keypoint') and
                self.tool not in ('select','parallel','centerline')):
            candidate=self._snap_at(p,exact_snap_mode,max(self.canvas.world_per_pixel()*14.,1.e-9))
            if candidate is not None:
                # Explicit geometric targets win over draft closure and
                # angle/axis constraints, so placement stays on the target.
                self._snap_target=candidate
                return candidate
        # Both Smart Line AND Line snap automatically in the VTK digitizer.
        # Text placement and Vertex insertion also need endpoint magnetism.
        # Explicit Snap mode takes priority over this nearby default.
        # Freehand snaps at start/end only; raw drag samples remain untouched.
        # Exact closure of the CURVE's first control point takes priority over
        # unrelated SNT or linework geometry, using VTK's 14px snap radius.
        if snap and self.tool == 'curve' and len(self.points) >= 3:
            tolerance = max(self.canvas.world_per_pixel() * 14., 1e-9)
            closure = snap_draft_closure(self.points, raw_p, tolerance)
            if closure != raw_p:
                self._snap_target = tuple(closure[:2])
                return self._snap_target
        smart_auto = self.tool in ('smartline', 'line', 'text', 'vertex', 'freehand', 'curve')
        if (snap and self.tool not in ('select','parallel','centerline')
                and (smart_auto or explicit_snap)):
            mode=(getattr(digitizer,'snap_mode',None) or 'nearby') if explicit_snap else 'nearby'
            if mode in ('nearby','midpoint','intercept'):
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
        if (kind or self.tool) == 'freehand':
            self._freehand_before_drag = None
            self._freehand_undo_strokes.clear()
            self._freehand_redo_strokes.clear()
            self._freehand_last_snap_at = 0.0
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
        from gui.native_history_routing import record_native_action
        record_native_action(self.app, 'draw')
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
            if drawing.get('type') == 'circle' or drawing.get('source') in ('block_creation','cell_creation'):
                continue  # No perimeter tessellation or protected source vertices.
            if drawing.get('type') == 'rectangle':
                vertices = self._native_coords(drawing)[:4]
            else:
                vertices = drawing.get('coords', [])
            for vi, vertex in enumerate(vertices):
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
        destination = tuple(destination[:2])
        if original.get('type') == 'rectangle':
            corners = self._native_coords(original)[:4]
            if len(corners) != 4 or vi >= 4:
                return False
            # Drag any of four visible corners against its opposite corner.
            # The rectangle remains a true two-control-point rectangle.
            coords = [destination, corners[(vi + 2) % 4]]
        else:
            if not coords or vi >= len(coords):
                return False
            closed = len(coords) > 2 and coords[0] == coords[-1]
            if closed and vi in (0, len(coords)-1):
                coords[0] = coords[-1] = destination
            else:
                coords[vi] = destination
        new_obj = dict(original, coords=coords)
        self.drawings[di] = new_obj
        if original.get('type') == 'line' and self._vertex_move_before is not None:
            # Line commits each edge separately. Move a shared chain corner
            # on every incident edge, as Smart Line does within one record.
            origin = tuple(original['coords'][vi][:2])
            chain = original.get('line_chain')
            for other_di, other in enumerate(self._vertex_move_before):
                if other_di == di or other.get('type') != 'line':
                    continue
                if other.get('imported') or other.get('source') in (
                        'block_creation', 'cell_creation', 'snt', 'snt_readonly'):
                    continue
                if chain is not None:
                    if other.get('line_chain') != chain:
                        continue
                elif other.get('line_chain') is not None or abs(other_di - di) != 1:
                    # Older drawings have no chain ID. Consecutively stored
                    # edges sharing this corner are the recoverable topology.
                    continue
                other_coords = list(other.get('coords', []))
                matches = [i for i, point in enumerate(other_coords)
                           if tuple(point[:2]) == origin]
                if matches:
                    for i in matches:
                        other_coords[i] = destination
                    self.drawings[other_di] = dict(other, coords=other_coords)
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
        if original.get('type') == 'rectangle':
            moved = tuple(destination[:2]) != self._native_coords(original)[vi]
        else:
            moved = new_obj['coords'][vi] != original['coords'][vi]
        if moved:
            self.record_object_change(before)
            # The edited dict replaced the selected object. Retire any old
            # highlight/fence identity; stale highlights can draw a synthetic
            # closing edge across an otherwise OPEN edited polyline.
            self._prune_deleted_native_refs({id(old) for old, new in
                zip(before, self.drawings) if old is not new})
        else:
            self.drawings[:] = before
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
            if drawing.get('type') == 'circle' or drawing.get('source') in ('block_creation','cell_creation'):
                continue
            coords = (self._native_coords(drawing) if drawing.get('type') == 'rectangle'
                      else drawing.get('coords', []))
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

    def _freehand_end_snap(self, force=False):
        """Preview a magnetic LAST point without changing any raw samples.

        Completed native/SNT geometry is queried at most every 50 ms while
        dragging; the final right-click always recomputes exact world coords.
        """
        if len(self.points) < 2:
            self._snap_target = None
            return None
        endpoint = self.points[-1]
        tolerance = max(self.canvas.world_per_pixel() * 14., 1e-9)
        own_start = snap_endpoint_near_start(self.points, endpoint, tolerance)
        if own_start != endpoint:
            self._snap_target = tuple(own_start[:2])
            return self._snap_target
        now = time.monotonic()
        if not force and now - self._freehand_last_snap_at < .05:
            return self._snap_target
        self._freehand_last_snap_at = now
        digitizer = getattr(self.app, 'digitizer', None)
        explicit = bool(digitizer is not None and getattr(digitizer, 'snap_enabled', False))
        mode = (getattr(digitizer, 'snap_mode', None) or 'nearby') if explicit else 'nearby'
        candidate = self._snap_at(endpoint, mode, tolerance) if mode in (
            'nearby', 'keypoint', 'midpoint', 'center', 'intercept') else None
        self._snap_target = (tuple(candidate) if candidate is not None and
                             tuple(candidate) != tuple(endpoint) else None)
        return self._snap_target

    def _finish_freehand(self):
        """VTK parity: commit only on right-click, after any paused drag(s)."""
        self.dragging = False
        if len(self.points) < 2:
            self.points.clear()
            self.hover = None
            self._snap_target = None
            self._freehand_before_drag = None
            self._freehand_undo_strokes.clear()
            self._freehand_redo_strokes.clear()
            self._paint()
            return False
        pts = drop_consecutive_duplicates(self.points)
        # At commit: bypass the preview throttle, magnetize only final sample.
        self.points = pts
        endpoint = self._freehand_end_snap(force=True)
        if endpoint is not None and endpoint != pts[-1]:
            # Two-point strokes are permitted as open lines. Do not snap the
            # endpoint onto their own start and create a zero-length feature.
            if len(pts) > 2 or endpoint != pts[0]:
                pts[-1] = endpoint
        pts = drop_consecutive_duplicates(pts)
        if len(pts) < 2:
            return False
        pts = close_freehand(pts, max(self.canvas.world_per_pixel() * 14., 1e-9))
        return self._commit('freehand', pts)

    def _finish(self):
        minimum = 3 if self.tool in ('polyline', 'hatch') else 2
        if len(self.points) < minimum:
            # Keep one-point drafts editable; do not destroy history on a
            # premature Finish/right-click.
            self._paint()
            return False
        pts = list(self.points)
        if self.tool == 'curve':
            # Never serialize duplicate adjacent controls as zero-length spans.
            pts = drop_consecutive_duplicates(pts)
            if len(pts) < 2:
                return False
            pts[-1] = snap_draft_closure(pts[:-1], pts[-1],
                                        max(self.canvas.world_per_pixel()*14.,1e-9))
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
        if self.tool == 'movetext':
            if button == Qt.RightButton:
                self.cancel()  # Right-click cancels without opening another menu.
                return True
            return self._commit_native_text_move(pos) if button == Qt.LeftButton else False
        # Existing text is editable on right-click even when the Text tool is
        # still armed. Never hijack Right-click Finish for another Draw tool.
        if button == Qt.RightButton and self.tool in ('text', 'select'):
            if self.try_text_context(pos):
                return True
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
                    feature = self._pick_feature(
                        p, open_only=self.tool == 'centerline',
                        predicate=drawn_open_line_source,
                    )
                except (RuntimeError, TypeError, ValueError):
                    feature = None
                if feature is not None and feature['source'] == 'snt':
                    feature = dict(feature, scene_token=self._scene_token())
                dialog._on_native_feature_picked(feature)
            return True
        if button == Qt.RightButton:
            if self.tool == 'freehand':
                self._finish_freehand()
            elif self.tool == 'line':
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
        if self.tool == 'freehand':
            # A second drag RESUMES this pending stroke, as in original VTK.
            # Scene snapping is applied only to the very first sample.
            try:
                p = (self._point(pos) if not self.points else
                     self.canvas.section_at(pos.x(), pos.y()))
            except RuntimeError as exc:
                self.app.statusBar().showMessage(str(exc), 4000)
                return True
            self._freehand_before_drag = list(self.points)
            self._freehand_redo_strokes.clear()
            if not self.points or math.dist(p, self.points[-1]) >= max(
                    self.canvas.world_per_pixel() * 2., 1e-10):
                self.points.append(tuple(p))
            self.dragging = True
            self.hover = None
            self._freehand_last_snap_at = 0.0
            self._snap_target = None
            self._paint()
            return True
        try:
            # Add Vertex must pick the clicked CAD edge, not a nearby SNT
            # trajectory/block endpoint found by the global snap engine.
            # Pending tap/drag Move Vertex still uses the normal snap path.
            p = (self.canvas.section_at(pos.x(), pos.y())
                 if self.tool == 'vertex' and self.vertex_target is None
                 else self._point(pos))
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
            feature=self._pick_editable_draw(p)
            self._store_selection(feature)
            if feature is not None:
                label=('SNT' if feature['source']=='snt' else 'native Draw')
                self.app.statusBar().showMessage(f'Selected {label} {feature["type"]}',3000)
            else:
                self.app.statusBar().showMessage('Nothing selectable at cursor',2500)
            return True
        if self.tool == 'parallel':
            if self._parallel_source is not None and not (
                    self._valid_source(self._parallel_source) and
                    parallel_line_source(self._parallel_source)):
                self._parallel_source=None
                self._store_selection(None)
            if self._parallel_source is None:
                source=self._pick_feature(p,open_only=True, predicate=parallel_line_source)
                if source is None:
                    self.app.statusBar().showMessage('Click an open line drawn with the Draw tools first',3500)
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
            self._commit('parallel', points, offset_distance=distance,
                         source_key=self._parallel_source['key'],
                         **self._source_visual_style(self._parallel_source))
            self._last_pointer_pos=None
            self._paint()
            self.app.statusBar().showMessage(f'Parallel offset: {distance:.3f} drawing units',4000)
            # Stay in Parallel so users can place another offset from source.
            return True
        if self.tool == 'centerline':
            if self._centerline_first is not None and not (
                    drawn_open_line_source(self._centerline_first) and
                    self._valid_source(self._centerline_first)):
                self._centerline_first=None
                self._store_selection(None)
            if self._centerline_first is None:
                first=self._pick_feature(p,open_only=True, predicate=drawn_open_line_source)
                if first is None:
                    self.app.statusBar().showMessage('Click the first open line drawn with the Draw tools',3500)
                    return True
                self._store_selection(first)
                self._centerline_first=self._selected_feature()
                self.app.statusBar().showMessage('Click a second DIFFERENT open line',4000)
                return True
            second=self._pick_feature(p,open_only=True,exclude=self._centerline_first,
                                      predicate=drawn_open_line_source)
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
                is_rectangle = drawing.get('type') == 'rectangle'
                # Removing a rectangle corner creates a triangle. Convert to
                # a closed editable polyline instead of retaining a malformed
                # two-point rectangle primitive.
                coords = (self._native_coords(drawing) if is_rectangle
                          else list(drawing.get('coords', [])))
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
                new_drawing = dict(drawing, coords=coords)
                if is_rectangle:
                    new_drawing['type'] = 'polyline'
                self.drawings[di] = new_drawing
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
                _, di, si, foot = hit
                before = list(self.drawings)
                original = self.drawings[di]
                is_rectangle = original.get('type') == 'rectangle'
                # An inserted point breaks four-corner rectangle topology.
                # Preserve its full outline as a closed polyline with the new
                # editable vertex, never as invalid rectangle control points.
                coords = (self._native_coords(original) if is_rectangle
                          else list(original['coords']))
                # Insert ON the selected segment. A click tolerance or a
                # neighbouring SNT snap must not create a stray zigzag edge.
                # Users can reposition the new vertex with Move Vertex.
                vertex = foot
                coords.insert(si + 1, vertex)
                new_drawing = dict(original, coords=coords)
                if is_rectangle:
                    new_drawing['type'] = 'polyline'
                self.drawings[di] = new_drawing
                self.record_object_change(before)
                self._prune_deleted_native_refs({id(original)})
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
                        # The SNT source stays read-only; any new native
                        # clone subdivides the picked edge without a kink.
                        coords.insert(segment[1]+1, segment[2])
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
            from gui.native_label_layout import native_draw_text_height
            # Freeze label height in CAD/world units at the moment of placement.
            # The font size remains an editable typographic point value.
            font_size = max(1, int(options.get('font_size') or 75))
            height = native_draw_text_height(options, self.canvas.world_per_pixel())
            self._commit('text', [p], text=options['text'], color=hex_color,
                         font_family=options.get('font_family', 'Arial'),
                         font_size=font_size, height=height,
                         bold=bool(options.get('bold', False)),
                         italic=bool(options.get('italic', False)))
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
            # Persist chain identity so later vertex edits do not pull an
            # unrelated Line figure that happens to touch the same point.
            from uuid import uuid4
            chain = (self._line_session_refs[0].get('line_chain')
                     if self._line_session_refs else None) or uuid4().hex
            self._commit('line', [start, p], line_chain=chain)
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
        if self.tool == 'curve' and self.points and p == self.points[-1]:
            return True  # Ignore duplicate adjacent curve controls, including snaps.
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
        if self.tool == 'movetext':
            if self._moving_text_ref is None or not any(
                    d is self._moving_text_ref for d in self.drawings):
                self.cancel()
                return True
            try:
                xy = self._text_move_world_at(pos)
            except (RuntimeError, TypeError, ValueError):
                return True
            last = self._moving_text_hover
            if last is None or math.dist(last, xy) > max(
                    0.5 * self.canvas.world_per_pixel(), 1.e-10):
                self._moving_text_hover = xy
                self._paint()
            return True
        if self.tool == 'select':
            if self._selection_bridge is not None:
                return self._selection_bridge.move(pos, buttons)
            # Draw > Select also has hover feedback without a selection panel.
            # Limit expensive native-DRAW picks to 20 Hz; never scan the SNT or
            # LiDAR entity stores just to color a hovered object.
            now = time.monotonic()
            if now - getattr(self, '_hover_pick_at', 0.) < .05:
                return True
            self._hover_pick_at = now
            try:
                p = self.canvas.section_at(pos.x(), pos.y())
            except RuntimeError:
                return True
            hit = self._pick_editable_draw(p)
            ref = hit['drawing'] if hit is not None else None
            if ref is not getattr(self, '_hover_selected_native_ref', None):
                self._hover_selected_native_ref = ref
                self._paint()
            return True
        if self._legacy_tool_dialog is not None and self.tool in ('parallel', 'centerline'):
            # Avoid repainting millions of points on every cursor move.
            return True
        if self.tool == 'curve' and not self.points:
            # VTK only previews Curve after the first placed control point;
            # do not run SNT picking against every idle cursor movement.
            return True
        if self.tool == 'freehand':
            if self.dragging and buttons & Qt.LeftButton:
                try:
                    p = self.canvas.section_at(pos.x(), pos.y())
                except RuntimeError:
                    return True
                self._last_pointer_pos = pos
                # VTK filters jitter at 2 screen pixels, not 0.5 pixels.
                if (not self.points or math.dist(p, self.points[-1]) >= max(
                        self.canvas.world_per_pixel() * 2., 1e-10)):
                    self.points.append(tuple(p))
                    self._freehand_end_snap()
                    self._paint()
            # Paused Freehand never draws a moving cursor rubber-band line.
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
                original = self._vertex_move_original
                origin = (self._native_coords(original)[vi]
                          if original.get('type') == 'rectangle'
                          else original['coords'][vi])
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
                    original=self._vertex_move_original
                    vi=self.vertex_target[1]
                    destination=(self._native_coords(original)[vi]
                                 if original.get('type') == 'rectangle'
                                 else original['coords'][vi])
                self._vertex_move_finish(destination)
                self._vertex_insert_pending = False
            return True
        if self.tool == 'freehand' and self.dragging:
            try:
                p = self.canvas.section_at(pos.x(), pos.y())
                if not self.points or math.dist(p, self.points[-1]) >= max(
                        self.canvas.world_per_pixel() * 2., 1e-10):
                    self.points.append(tuple(p))
            except RuntimeError:
                pass  # Keep existing samples if the viewport is unavailable.
            self.dragging = False
            self.hover = None
            self._freehand_undo_strokes.append(self._freehand_before_drag or [])
            self._freehand_before_drag = None
            self._freehand_end_snap(force=True)
            self._paint()
            self.app.statusBar().showMessage(
                'Freehand paused: left-drag to resume; right-click to close/finish', 4000)
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
        if (self._selection_bridge is not None and
                getattr(self._selection_bridge,'_intersect_mode',False)):
            return self._selection_bridge.intersect_history(redo=redo)
        if self.tool == 'line' and (self.points or self.point_redo or
                                    self._line_session_refs or self._line_session_undone):
            return self._line_history(redo=redo)
        if self.tool == 'freehand' and (self.points or self.dragging or
                                        self._freehand_undo_strokes or
                                        self._freehand_redo_strokes):
            if self.dragging and not redo:
                # Undo an unfinished drag, not a previously committed object.
                self.points[:] = self._freehand_before_drag or []
                self.dragging = False
                self._freehand_before_drag = None
            elif redo and self._freehand_redo_strokes:
                self._freehand_undo_strokes.append(list(self.points))
                self.points[:] = self._freehand_redo_strokes.pop()
            elif not redo and self._freehand_undo_strokes:
                self._freehand_redo_strokes.append(list(self.points))
                self.points[:] = self._freehand_undo_strokes.pop()
            else:
                return True  # No document Undo while a stroke is pending.
            self._snap_target = None
            self.hover = None
            self._paint()
            return True
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

        if self.tool == 'movetext' and self._moving_text_ref is not None:
            # No history entry until placed; Undo/Redo only cancels the preview.
            self.cancel()
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
            prior = {id(item) for item in self.drawings}
            self.drawings[:] = after if redo else before
            target_actions.append((before,after))
            live = {id(item) for item in self.drawings}
            removed_ids = prior - live
            if removed_ids:
                self._prune_deleted_native_refs(removed_ids)
            elif self._selection_bridge is not None:
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

    def _prune_deleted_native_refs(self, removed_ids):
        """Sync live Draw selection/fence dialogs after shape removal."""
        selected_ref = getattr(self, '_selected_native_ref', None)
        if selected_ref is not None and id(selected_ref) in removed_ids:
            self._store_selection(None)
        elif selected_ref is not None:
            self.selected = next((i for i, obj in enumerate(self.drawings)
                                  if obj is selected_ref), None)
        hover_ref = getattr(self, '_hover_selected_native_ref', None)
        if hover_ref is not None and id(hover_ref) in removed_ids:
            self._hover_selected_native_ref = None
        all_highlights = getattr(self, '_classification_fence_highlights', {})
        highlight_refs = getattr(self, '_classification_fence_highlight_refs', {})
        for key, rings in list(all_highlights.items()):
            refs = highlight_refs.get(key, [])
            surviving = [(ring, refs[i] if i < len(refs) else None)
                         for i, ring in enumerate(rings)
                         if i >= len(refs) or refs[i] not in removed_ids]
            if surviving:
                self._classification_fence_highlights[key] = [item[0] for item in surviving]
                highlight_refs[key] = [item[1] for item in surviving]
            else:
                self._classification_fence_highlights.pop(key, None)
                highlight_refs.pop(key, None)
        if not all_highlights:
            self._active_fence_highlight_owner = None
        old_fence_count = len(getattr(self, '_active_fence_shapes', ()))
        self._active_fence_shapes = [shape for shape in
            getattr(self, '_active_fence_shapes', ())
            if (shape.get('native_ref') is None or
                id(shape['native_ref']) not in removed_ids)]
        if len(self._active_fence_shapes) != old_fence_count:
            self._fence_selection_revision = getattr(self, '_fence_selection_revision', 0) + 1
            from gui.native_fence_selection import propagate_native_fence_selection
            propagate_native_fence_selection(self.app, self._active_fence_shapes)
        bridge = getattr(self, '_selection_bridge', None)
        if bridge is not None:
            bridge.selection_manager._prune()
        from gui.draw_clear_selection import prune_removed_native_fences
        prune_removed_native_fences(self.app, self, removed_ids)
        self._paint()

    def delete_drawings(self, selected_ids):
        """Delete exactly chosen editable Draw objects; one atomic Undo step.

        Imported SNT, saved/committed objects, point classes, and all unchosen
        drawings remain untouched. Never compare NumPy-bearing dicts by value.
        """
        from gui.draw_clear_selection import select_drawings_by_identity
        removed = select_drawings_by_identity(self.drawings, selected_ids)
        if not removed:
            return 0
        removed_ids = {id(item) for item in removed}
        before = list(self.drawings)
        self.drawings[:] = [item for item in self.drawings if id(item) not in removed_ids]
        self._line_session_refs[:] = [item for item in self._line_session_refs
                                      if id(item) not in removed_ids]
        self._line_session_undone[:] = [item for item in self._line_session_undone
                                        if id(item) not in removed_ids]
        # Commit the Undo snapshot before potentially invalid Qt dialogs are
        # inspected; a stale popup must not make the deletion irreversible.
        self.record_object_change(before)
        try:
            self._prune_deleted_native_refs(removed_ids)
        except (AttributeError, RuntimeError) as exc:
            print(f'[Draw Clear] stale selection cleanup warning: {exc}')
        return len(removed)

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
        self._moving_text_ref = None
        self._moving_text_hover = None
        self._moving_text_grab_offset = (0., 0.)
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
        self._moving_text_ref = None
        self._moving_text_hover = None
        self._moving_text_grab_offset = (0., 0.)
        self._close_palettes()
        self.points, self.hover, self.dragging, self.vertex_target = [], None, False, None
        self.point_redo.clear(); self._z_redo.clear(); self._z_points.clear()
        self._last_pointer_pos = None
        self._angle_sticky = False
        self._snap_target = None
        self._hover_selected_native_ref = None
        self._freehand_before_drag = None
        self._freehand_undo_strokes.clear()
        self._freehand_redo_strokes.clear()
        self._freehand_last_snap_at = 0.0
        self._parallel_source = None
        self._centerline_first = None
        if stand_down:
            self.tool = None
        self._paint()

    def _sync_block_vectors(self):
        """Keep authored blocks in the same camera frame as attached linework."""
        blocks = [d for d in self.drawings if d.get('source') == 'block_creation']
        resident = getattr(self, '_block_vector_layer', None)
        name = '__native_created_blocks__'
        if not blocks:
            if resident is not None:
                resident.remove('drawing', name)
                self._present_vector_frame(resident)
            self._block_vector_layer = None
            self._block_vector_signature = None
            return False
        from gui.native_vector_overlay import vector_layer_for
        layer = vector_layer_for(self.app)
        if layer is None:
            return False
        entities = []
        for drawing in blocks:
            color = drawing.get('color', (0, 179, 255))
            if isinstance(color, str):
                from PySide6.QtGui import QColor
                color = QColor(color).getRgb()[:3]
            entities.append({'type': 'polyline', 'layer': 'Created blocks',
                             'points': drawing.get('coords', []), 'color': tuple(color),
                             '_lineweight': max(0, (int(drawing.get('width', 3)) - 1) * 2)})
        signature = (id(layer), tuple((tuple(tuple(p) for p in e['points']),
                                      e['color'], e['_lineweight']) for e in entities))
        if signature != getattr(self, '_block_vector_signature', None):
            if layer.put_entities('drawing', name, entities) is None:
                return False
            self._block_vector_layer = layer
            self._block_vector_signature = signature
            self._present_vector_frame(layer)
        return True

    def _present_vector_frame(self, layer):
        """Show changed block linework now, not at the next zoom/pan.

        Uploading only updates the GPU draw ranges. An .lnt view presents every
        tick (its vector backend's request_render is a no-op), but the promoted
        Vulkan viewport of an SNT-only session renders on demand, so ask it for
        one frame - coalesced, since Create/Clear may sync several times.
        """
        if getattr(self, '_vector_frame_pending', False):
            return
        self._vector_frame_pending = True

        def present():
            self._vector_frame_pending = False
            backend = getattr(layer, '_backend', None)
            try:
                if backend is not None:
                    backend.request_render()
            except Exception as exc:  # noqa: BLE001 - a missed frame must not break editing
                print(f'[native draw] block linework frame request failed: {exc}')

        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, present)

    def _paint(self):
        if getattr(self,'_shutting_down',False):
            return  # Closing must not re-upload native block GPU vectors.
        gpu_blocks = self._sync_block_vectors()
        shapes = []
        bridge = self._selection_bridge
        chosen = (bridge.selection_manager._selected if bridge is not None else {})
        # An open picker previews a REPLACEMENT group. Obsolete selection
        # strokes must not stay cyan behind that preview.
        picker_preview = getattr(self, '_fence_picker_preview_owner', None) is not None
        if picker_preview:
            chosen = {}
        hidden_in_intersect = (bridge._intersect_source_ids if bridge is not None
                               and bridge._intersect_mode else set())
        hovered = (getattr(bridge, '_hover_feature', None) if bridge is not None
                   else None)
        hovered_ref = (hovered.get('drawing') if hovered is not None
                       and hovered.get('source') == 'native' else
                       getattr(self, '_hover_selected_native_ref', None)
                       if bridge is None and self.tool == 'select' else None)
        transform_hidden = (
            bridge.transform_hidden_drawing_ids()
            if bridge is not None and hasattr(bridge, 'transform_hidden_drawing_ids')
            else set()
        )
        for i, drawing in enumerate(self.drawings):
            if (drawing is getattr(self, '_moving_text_ref', None)
                    and getattr(self, '_moving_text_hover', None) is not None):
                continue  # The live preview replaces the original, not duplicates it.
            if id(drawing) in transform_hidden:
                continue  # Transform preview replaces its committed sources this frame.
            if id(drawing) in hidden_in_intersect:
                continue  # VTK-style overlay replaces original with split pieces.
            if gpu_blocks and drawing.get('source') == 'block_creation':
                continue
            data = self._style(drawing.get('type'))
            data.update(drawing)
            # Backward compatibility with lines saved by the old Parallel
            # dialog, which used 'line_style' instead of the overlay's 'style'.
            if drawing.get('style') is None and drawing.get('line_style') is not None:
                data['style'] = drawing['line_style']
            data['selected'] = (('native', id(drawing)) in chosen if bridge is not None
                                else not picker_preview and i == self.selected
                                and drawing is self._selected_native_ref)
            # Only editable native Draw objects get cyan. Block/cell geometry
            # and imported SNT linework retain their existing special colors.
            data['highlight_scope'] = 'draw' if editable_draw(drawing) else None
            data['hovered'] = editable_draw(drawing) and drawing is hovered_ref
            data['show_vertices'] = bool(drawing.get('type') != 'circle' and (
                self.tool in ('movevertex','vertex','deletevertex') or (
                bridge is not None and getattr(bridge,'_vertex_display',False)
                and data['selected'])))
            coords = drawing.get('coords', [])
            if drawing.get('type') == 'curve':
                data['control_points'] = list(coords)
                coords = _catmull_rom(coords)
            shapes.append(('drawing', coords, data))
        if (getattr(self, '_moving_text_ref', None) is not None
                and getattr(self, '_moving_text_hover', None) is not None):
            # Only the placement preview moves; original world coordinates and
            # every font/style field remain immutable until the left-click.
            label = self._moving_text_ref
            data = dict(self._style('text'), **label)
            data['show_vertices'] = False
            data['selected'] = False
            data['hovered'] = False
            shapes.append(('drawing_preview', [self._moving_text_hover], data))
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
                # Original VTK: stationary smooth curve through placed controls,
                # plus a straight grey/yellow rubber band to the mouse. The
                # moving cursor must not bend already completed segments.
                data['control_points'] = list(self.points)
                data['show_vertices'] = False
                preview = _catmull_rom(self.points)
                if self.hover is not None and self.hover != self.points[-1]:
                    shapes.append(('drawing', [self.points[-1], self.hover],
                                   {'type': 'polyline',
                                    'color': '#ffdf4a' if self._snap_target is not None
                                             else '#a8adb7',
                                    'width': 1, 'show_vertices': False}))
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
                               {'type': 'parallel', 'show_vertices': False,
                                **self._source_visual_style(self._parallel_source)}))
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
                if feat['source'] == 'snt':
                    # SNT text has a separate read-only outline. Measure
                    # Line/Path highlight in their own overlay, not here.
                    shapes.append(('drawing',feat['coords'],
                                   {'type':'point' if len(feat['coords'])==1 else 'polyline',
                                    'selected':True,'width':3}))
            shapes.extend(self._selection_bridge.preview_shapes())
        # The VTK renderer is hidden behind the Vulkan main viewport. Only
        # the CURRENT fence group is cyan; formerly classified shapes are not
        # selected just because their class edits are still committed.
        # The active Draw/Element Selection path already paints a chosen source
        # object cyan, so skip a second overlay for exactly that same object.
        active_key = getattr(self, '_active_fence_highlight_owner', None)
        active_rings = (() if picker_preview else
                        getattr(self, '_classification_fence_highlights', {}).get(active_key, ()))
        active_refs = getattr(self, '_classification_fence_highlight_refs', {}).get(active_key, ())
        selection_refs = {key[1] for key in chosen if key[0] == 'native'}
        if bridge is None and getattr(self, '_selected_native_ref', None) is not None and getattr(self, 'selected', None) is not None:
            selection_refs.add(id(self._selected_native_ref))
        # Fence classification may close an OPEN line into an area, but that
        # virtual closure is not a real Draw edge. Repaint native reference
        # highlights using their exact stored/display geometry instead.
        live_native = ({id(d): d for d in self.drawings} if active_rings else {})
        for i, ring in enumerate(active_rings):
            ref_id = active_refs[i] if i < len(active_refs) else None
            if ref_id in selection_refs:
                continue
            if ref_id is not None:
                real_drawing = live_native.get(ref_id)
                if real_drawing is None:
                    continue  # An obsolete selected object was edited away.
                visible_coords = self._native_coords(real_drawing)
                if not visible_coords:
                    continue
            else:
                visible_coords = ring  # Explicit classification fence shape.
            shapes.append(('drawing', visible_coords, {
                'type': 'polyline', 'color': '#00ffff', 'selected': True,
                'highlight_scope': 'draw', 'width': 4,
                'show_vertices': False,
            }))
        # The picker is a non-destructive preview. When it closes, the last
        # committed selection becomes visible again, not an obsolete dialog.
        for ring in getattr(self, '_fence_picker_preview_rings', ()):
            shapes.append(('drawing', ring, {
                'type': 'polyline', 'color': '#00ffff', 'selected': True,
                'highlight_scope': 'draw', 'width': 4,
                'show_vertices': False,
            }))
        shapes.extend(self.aux_shapes)
        hover_ring = getattr(self, '_fence_picker_hover_ring', None)
        if hover_ring:
            shapes.append(('drawing', hover_ring, {
                'type': 'polyline', 'color': '#00ffff', 'width': 6,
                'style': 'solid', 'show_vertices': False,
                'source': 'fence_picker_hover',
            }))
        self.overlay.set_shapes(shapes)

    def shutdown(self):
        # Prevent cancel()/dialog close from repainting and repopulating the
        # block GPU vector layer after shutdown removed it.
        self._shutting_down = True
        dialog = self._legacy_tool_dialog
        self._legacy_tool_dialog = None
        self._legacy_dialog_highlights = []
        self._classification_fence_highlights.clear()
        self._classification_fence_highlight_refs.clear()
        self._active_fence_highlight_owner = None
        self._active_fence_shapes = []
        self._fence_picker_preview_owner = None
        self._fence_picker_preview_rings = []
        self._fence_picker_hover_owner = None
        self._fence_picker_hover_ring = None
        if dialog is not None:
            try:
                dialog.close()
            except RuntimeError:
                pass
        self.cancel()
        layer = getattr(self, '_block_vector_layer', None)
        if layer is not None:
            layer.remove('drawing', '__native_created_blocks__')
        self._block_vector_layer = None
        self._block_vector_signature = None
        self.canvas._closed = True
        self.overlay.shutdown()


def native_plan_view(app, allow_snt=False):
    """Return the native LNT or promoted SNT Vulkan plan view."""
    view = getattr(app, '_lnt_main_view', None)

    if view is not None and getattr(view, 'native', None) is not None:
        return view if getattr(view, 'mode_2d', False) else None

    if allow_snt:
        from gui.native_vector_overlay import _view_for, snt_viewport_active
        if snt_viewport_active(app):
            view = _view_for(app)
            if (view is not None and
                    getattr(getattr(view, 'owner', None),
                            'vulkan_widget', None) is not None):
                return view

    return None


def get_native_draw(app, create=False, allow_snt=False, existing_any_mode=False):
    """Find Draw on Vulkan; explicit creation promotes an empty viewport.

    ``existing_any_mode`` is intentionally read/delete-only: it exposes an
    already-created controller while the LNT camera is in 3D, but never creates
    one there. Editing and placement remain restricted to the 2D plan view.
    """
    view = getattr(app, '_lnt_main_view', None)

    if view is not None:
        if getattr(view, 'native', None) is None:
            return None
        if not getattr(view, 'mode_2d', False):
            if existing_any_mode and not create:
                return getattr(view, 'drawing', None)
            return None
    else:
        owner = getattr(app, 'render_backend', None)
        view = None

        if create and owner is not None:
            # Match Measure on a fresh launch: install the native canvas
            # before checking visibility/camera ownership. Promotion refuses
            # a loaded legacy cloud or an engine without the vector pass.
            from gui.native_vector_overlay import promote_snt_viewport, snt_viewport_active
            if not snt_viewport_active(app):
                promote_snt_viewport(app)

        if (owner is not None and
                getattr(owner, 'active', False) and
                owner.vulkan_viewport_is_visible() and
                owner.owns_main_camera()):

            view = getattr(owner, '_plan_draw_view', None)

            if view is None and create:
                from gui.native_vector_overlay import _view_for
                view = _view_for(app)

                if view is not None:
                    view._surface_draw = True
                    owner._plan_draw_view = view

        if view is None and allow_snt:
            view = native_plan_view(app, allow_snt=True)
            if view is not None:
                view._surface_draw = True

        if view is None:
            return None
    draw = getattr(view, 'drawing', None)
    if (draw is not None and getattr(view, '_surface_draw', False)
            and draw.canvas.widget is not view.owner.vulkan_widget):
        # LNT can replace the app's Vulkan QWidget while open. Retire a
        # stale overlay before drawing in the replacement viewport.
        view._draw_saved_shapes = list(draw.drawings)
        view._draw_saved_history = (list(getattr(draw, '_object_actions', ())),
                                    list(getattr(draw, '_object_redo', ())))
        draw.shutdown()
        view.drawing = draw = None
    if draw is None and create:
        draw = view.drawing = NativeDraw(view)
        if getattr(view, '_surface_draw', False):
            draw.drawings.extend(getattr(view, '_draw_saved_shapes', ()))
            view._draw_saved_shapes = []
            saved_history = getattr(view, '_draw_saved_history', None)
            if saved_history is not None:
                draw._object_actions[:] = saved_history[0]
                draw._object_redo[:] = saved_history[1]
                view._draw_saved_history = None
            draw._paint()
    return draw


def _return_to_plan_view_for_draw(app):
    """Draw places plan XY only. A 3D orbit (D / Shift+P / View ribbon) must
    not make activation fall through to the VTK digitizer hidden under the
    Vulkan surface, so a Draw tool brings the main view back to Top."""
    view = getattr(app, '_lnt_main_view', None)
    if view is not None:
        if getattr(view, 'native', None) is None or getattr(view, 'mode_2d', False):
            return
    elif getattr(app, 'data', None) or not (
            getattr(app, 'is_3d_mode', False) or
            getattr(app, '_main_view_3d_user_enabled', False)):
        return  # 2D already, or a VTK-rendered cloud keeps the legacy path
    change = getattr(app, 'handle_view_change', None)
    if callable(change) and change('top') is not False:
        if view is None or getattr(view, 'mode_2d', False):
            return
    if view is not None:
        # Renderer without view presets: same as the native TopView shortcut.
        view.set_2d(True)
        view.native.plan_view()
        app.current_view = 'top'
    app.is_3d_mode = False
    app._main_view_3d_user_enabled = False


def activate_native_draw(app, tool):
    if str(tool or '').lower().replace(' ', '_') not in TOOL_MAP:
        return False
    _return_to_plan_view_for_draw(app)
    draw = get_native_draw(app, create=True)
    return bool(draw and draw.activate(tool))
