"""Element Selection panel controller for the *visible* native Vulkan plan.

VTK's ElementSelectTool remains the owner of VTK interaction. In a Vulkan
main view VTK actors cannot receive mouse picks because the Vulkan widget is
above the QVTK interactor. This adapter uses NativeDraw's same coordinate and
SNT visibility mappings, never writes to imported source entity dictionaries,
and never touches the VTK camera or LiDAR point buffers.
"""
from __future__ import annotations

import math
import time
from bisect import bisect_left, bisect_right
from pathlib import Path
from gui.native_selection_targets import editable_draw, text_anchor, text_bounds, box_overlap
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QApplication, QInputDialog

from gui.native_selection_geometry import (
    rect_selects, polygon_selects, line_selects, translate, rotate, mirror,
)
from gui.native_vector_pick import nearest_segment
from gui.native_intersect_geometry import split_selected_features


class NativeSelectionManager(QObject):
    selection_changed = Signal(int)
    MAX_BULK = 5000  # Avoid hanging GUI on million-entity SNT attachments.

    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller
        self._selected = {}

    def _prune(self):
        draw = self.controller.draw
        # O(native drawings + selection), not O(their product). Select All can
        # hold thousands of native objects on a large CAD/LiDAR scene.
        native_ids={id(d) for d in draw.drawings}
        scene=draw._scene_token()
        valid={}
        for key,feat in self._selected.items():
            if feat['source']=='native':
                live=(id(feat.get('drawing')) in native_ids
                      and editable_draw(feat.get('drawing')))
            elif feat['source']=='measure':
                measure=feat.get('measure')
                live=(measure is not None and
                      measure is getattr(getattr(draw,'view',None),'measurement',None) and
                      any(points is feat.get('points')
                          for points in getattr(measure,'completed',())))
            else:
                live=(feat.get('scene_token')==scene
                      and feat.get('type') in ('text','mtext')
                      and draw._snt_visible(feat['doc_key'],feat['layer']))
            if live:valid[key]=feat
        if len(valid) != len(self._selected):
            self._selected = valid
            self.selection_changed.emit(len(valid))
        return valid

    def get_features(self):
        return list(self._prune().values())

    def get(self):
        # These are *descriptors*, not mutable SNT entities. The legacy
        # Classify fence dialogs only consume type and coords.
        result=[]
        for f in self.get_features():
            if f['source']=='native':
                result.append(f['drawing'])
            elif f['source']=='measure':
                result.append({'type':f['type'],'coords':list(f['coords']),
                               'source':'measure_selection'})
            else:
                result.append({'type':f['type'],'coords':list(f['coords']),
                               'source':'snt_readonly','layer':f['layer'],
                               'imported':True})
        return result

    def count(self):
        return len(self._prune())

    def _sync_fence_highlights(self):
        """Canvas selection replaces old cyan classification-dialog outlines.

        The manager already implements new/add/subtract/invert, so retain its
        complete current group instead of just the most recently clicked ring.
        """
        draw = self.controller.draw
        # VTK's/older Stage 1-5 test controllers deliberately do not have
        # native Draw fence overlays. Their selection logic stays unchanged.
        setter = getattr(draw, 'set_classification_fence_highlights', None)
        native_coords = getattr(draw, '_native_coords', None)
        if not callable(setter) or not callable(native_coords):
            return
        from gui.native_fence_selection import (
            selected_fence_shapes, propagate_native_fence_selection,
        )
        chosen_drawings = [feature['drawing']
                           for feature in self._selected.values()
                           if feature.get('source') == 'native']
        fences = selected_fence_shapes(chosen_drawings, native_draw=draw)
        setter(self, fences, replace=True)
        propagate_native_fence_selection(getattr(draw, 'app', None), fences)

    def _repaint(self):
        self.controller.draw._paint()
        # The Measure overlay is a different Qt window/layer. Trigger it as
        # well so selected/unselected Line and Path strokes change in place.
        self.controller._repaint_measure_overlay()

    def clear(self, emit=True):
        self._selected.clear()
        # Deactivation must not erase a DIFFERENT dialog's newer selection;
        # explicit canvas selection (apply) always supersedes the old group.
        if getattr(self.controller.draw, '_active_fence_highlight_owner', None) == id(self):
            self._sync_fence_highlights()
        self._repaint()
        if emit:self.selection_changed.emit(0)

    def apply(self, features, mode='new'):
        old = dict(self._prune())
        if mode=='new':old.clear()
        for feat in features:
            if feat is None:continue
            if feat['source']=='snt':
                # Selection dialog is NOT the grid/block picker. Imported SNT
                # linework is outside the editing scope; only SNT Text mode
                # exposes read-only label selections for clipboard copying.
                if (self.controller._method!='snt_text' or
                        feat.get('type') not in ('text','mtext')):continue
                feat=dict(feat,scene_token=self.controller.draw._scene_token())
            elif feat['source']=='native' and not editable_draw(feat.get('drawing')):
                continue
            key=feat['key']
            if mode=='subtract':old.pop(key,None)
            elif mode=='invert':
                if key in old:old.pop(key)
                else:old[key]=feat
            else:old[key]=feat
        if len(old)>self.MAX_BULK:
            self.controller.message(f'Selection capped at {self.MAX_BULK:,} elements for responsiveness')
            old=dict(list(old.items())[:self.MAX_BULK])
        self._selected=old
        self._sync_fence_highlights()
        self._repaint()
        self.selection_changed.emit(len(old))

    def select_all(self,predicate=None):
        # Element Selection = editable Draw + completed Measure Line/Path,
        # NEVER arbitrary imported SNT block boundaries or point-cloud data.
        features=[self.controller._native_feature(i,d)
                  for i,d in enumerate(self.controller.draw.drawings)
                  if editable_draw(d) and (predicate is None or predicate(d))]
        if predicate is None:
            features.extend(self.controller._measure_features())
        self.apply(features,'new')

    def invert_all(self):
        candidates=[self.controller._native_feature(i,d)
                    for i,d in enumerate(self.controller.draw.drawings)
                    if editable_draw(d)]
        candidates.extend(self.controller._measure_features())
        chosen=self._prune()
        # Toggle only eligible drawing/measurement objects. SNT text keeps its
        # own dedicated mode; the block boundary must never be inverted in.
        self.apply([f for f in candidates if f['key'] not in chosen], 'new')


class NativeSelectionController(QObject):
    """Vulkan interaction half of the existing SelectionModeDialog."""
    def __init__(self, draw):
        super().__init__()
        self.draw=draw
        self.app=draw.app
        self.selection_manager=NativeSelectionManager(self)
        self._method='individual'
        self._enclose_mode='overlap'
        self._active=False
        self._move_mode=False
        self._copy_mode=False
        self._intersect_mode=False
        self._intersect_parts=[]
        self._intersect_source_ids=set()
        self._intersect_selected_part=None
        self._intersect_hovered_part=None
        self._intersect_delete_history=[]
        self._intersect_redo_history=[]
        self._intersect_scene_token=None
        self._start=None
        self._hover=None
        self._hover_feature=None
        self._hover_pick_at=0.
        self._gesture=[]
        self._drag_origin=None
        self._offset_origin=None
        self._vertex_display=False
        self._text_index_token=None
        self._text_index=[]
        self._text_min_x=[]
        self._text_prefix_max_x=[]
        self._snt_copy_sources=[]

    def message(self, msg, timeout=3500):
        bar=getattr(self.app,'statusBar',lambda:None)()
        if bar is not None:bar.showMessage(msg,timeout)

    def is_active(self):
        return (self._active and self.draw.tool == 'select' and
                getattr(self.draw, '_selection_bridge', None) is self)

    def activate(self, method='individual'):
        if self.draw.tool!='select' and not self.draw.activate('select'):
            return False
        self._active=True
        self._method=method
        self.draw._selection_bridge=self
        self._reset_gesture()
        return True

    def deactivate(self):
        if self._intersect_mode:
            self.exit_intersect_mode(commit=True)
        self._active=False
        self._reset_gesture()
        self._move_mode=self._copy_mode=self._intersect_mode=False
        self._snt_copy_sources=[]
        if self.draw._selection_bridge is self:
            self.draw._selection_bridge=None
        self.selection_manager.clear()
        if self.draw.tool=='select':self.draw.cancel()

    def set_method(self, method):
        if self._intersect_mode:
            self.exit_intersect_mode(commit=True)
        self._method=method
        self._copy_mode=self._move_mode=False
        self._snt_copy_sources=[]
        self._reset_gesture()
        self.selection_manager.clear()
        self.message(f'Vulkan Select: {method.replace("_"," ")}')

    def set_enclose_mode(self, mode):
        self._enclose_mode=mode

    def _repaint_measure_overlay(self):
        measurement = getattr(getattr(self.draw, 'view', None), 'measurement', None)
        repaint = getattr(measurement, '_paint', None)
        if callable(repaint):
            repaint()

    def _set_hover_feature(self, feature):
        old = self._hover_feature
        old_key = old.get('key') if old is not None else None
        new_key = feature.get('key') if feature is not None else None
        if old_key == new_key:
            return
        self._hover_feature = feature
        if getattr(self.draw, '_selection_bridge', None) is self:
            self.draw._paint()
        if ((old is not None and old.get('source') == 'measure') or
                (feature is not None and feature.get('source') == 'measure')):
            self._repaint_measure_overlay()

    def _reset_gesture(self):
        self._start=self._hover=self._drag_origin=self._offset_origin=None
        old_hover = self._hover_feature
        self._hover_feature=None
        self._hover_pick_at=0.
        self._gesture=[]
        if getattr(self.draw,'_selection_bridge',None) is self:
            self.draw._paint()
        if old_hover is not None and old_hover.get('source') == 'measure':
            self._repaint_measure_overlay()

    @staticmethod
    def _selection_modifier():
        mods=QApplication.keyboardModifiers()
        if mods & Qt.ControlModifier and mods & Qt.ShiftModifier:return 'invert'
        if mods & Qt.ControlModifier:return 'add'
        if mods & Qt.ShiftModifier:return 'subtract'
        return 'new'

    def _native_feature(self, index, drawing):
        return {'source':'native', 'index':index, 'drawing':drawing,
                'coords':self.draw._native_coords(drawing),
                'type':drawing.get('type','line'),
                'key':('native',id(drawing))}

    def _measure_features(self):
        """Completed main-viewport Line/Path measurements, never grid areas."""
        measure=getattr(getattr(self.draw,'view',None),'measurement',None)
        if measure is None:return []
        result=[]
        modes=getattr(measure,'completed_modes',())
        for i,points in enumerate(getattr(measure,'completed',())):
            if len(points)<2:continue
            coords=[tuple(p[:2]) for p in points]
            if not all(math.isfinite(v) for p in coords for v in p):continue
            result.append({'source':'measure','measure':measure,'points':points,
                           'index':i,'coords':coords,
                           'type':'measure_path' if i<len(modes) and modes[i] else 'measure_line',
                           'key':('measure',id(measure),id(points))})
        return result

    def _pick_editable(self,p):
        """Pick only Draw and Measure Line/Path, never imported SNT blocks."""
        from gui.native_draw import _inside_polygon
        tol=max(self.draw.canvas.world_per_pixel()*12.,1.e-9)
        best=(tol,None)
        lo=(p[0]-tol,p[1]-tol);hi=(p[0]+tol,p[1]+tol)
        for feat in self._features_bbox(lo,hi):
            coords=feat['coords']
            if len(coords)>2 and coords[0]==coords[-1] and _inside_polygon(p,coords):
                dist=0.
            else:
                dist=self.draw._feature_distance(p,feat)
            if dist<=best[0]:best=(dist,feat)
        return best[1]

    def _features_bbox(self, lower, upper):
        a,b=lower,upper
        minx,maxx=sorted((a[0],b[0]));miny,maxy=sorted((a[1],b[1]))
        count=0
        for i,d in enumerate(self.draw.drawings):
            if not editable_draw(d):continue
            coords=self.draw._native_coords(d)
            if not coords:continue
            if not (max(p[0] for p in coords)<minx or min(p[0] for p in coords)>maxx or
                    max(p[1] for p in coords)<miny or min(p[1] for p in coords)>maxy):
                yield self._native_feature(i,d);count+=1
                if count>=self.selection_manager.MAX_BULK:return
        for feat in self._measure_features():
            coords=feat['coords']
            if (max(p[0] for p in coords)<minx or min(p[0] for p in coords)>maxx or
                    max(p[1] for p in coords)<miny or min(p[1] for p in coords)>maxy):
                continue
            yield feat;count+=1
            if count>=self.selection_manager.MAX_BULK:return

    def _select_area(self, method, vertices):
        if not vertices:return
        if method == 'shape' and len(vertices) < 3:return
        if method in ('block', 'cline') and len(vertices) < 2:return
        a,b=vertices[0],vertices[-1]
        if method=='shape':
            bounds=[(min(p[0] for p in vertices),min(p[1] for p in vertices)),
                    (max(p[0] for p in vertices),max(p[1] for p in vertices))]
        else:
            bounds=[(min(a[0],b[0]), min(a[1],b[1])),
                    (max(a[0],b[0]), max(a[1],b[1]))]
        tol=self.draw.canvas.world_per_pixel()*10. if method=='cline' else 0.
        bounds=[(bounds[0][0]-tol,bounds[0][1]-tol),
                (bounds[1][0]+tol,bounds[1][1]+tol)]
        inside=self._enclose_mode=='inside'
        hits=[]
        for feat in self._features_bbox(*bounds):
            coords=feat['coords']
            if method=='block' and rect_selects(coords,a,b,inside):hits.append(feat)
            elif method=='shape' and polygon_selects(coords,vertices,inside):hits.append(feat)
            elif method=='cline' and line_selects(coords,a,b):hits.append(feat)
        modifier=self._selection_modifier()
        self.selection_manager.apply(hits,modifier)
        print(f'[Native Element Selection] {method}: hits={len(hits)} '
              f'mode={modifier} selected={self.selection_manager.count()}', flush=True)
        self.message(f'{method.title()}: {len(hits)} shape(s) hit; '
                     f'{self.selection_manager.count()} element(s) selected')
        self._reset_gesture()

    def _ensure_text_index(self):
        """Cache world-space SNT text bounds, not legacy VTK actors.

        Text has a separate user-facing pick method. Native GPU layer linework
        and the transparent text overlay both originate from _render_entities.
        """
        token=self.draw._scene_token()
        if token==self._text_index_token:return
        records=[]
        for attachment in getattr(self.app,'snt_attachments',()) or ():
            entities=attachment.get('_render_entities')
            if entities is None:entities=attachment.get('entities') or ()
            name=attachment.get('full_path') or attachment.get('filename') or ''
            doc_key=('snt',str(Path(str(name))))
            for idx,entity in enumerate(entities):
                if not isinstance(entity,dict):continue
                if str(entity.get('type','')).strip().lower() not in ('text','mtext'):
                    continue
                anchor=text_anchor(entity)
                if anchor is None:continue
                bounds=text_bounds(entity,anchor)
                layer=str(entity.get('layer','0') or '0')
                records.append((*bounds,anchor,doc_key,layer,idx,entity))
        records.sort(key=lambda row:row[0])
        self._text_index=records
        self._text_min_x=[row[0] for row in records]
        self._text_prefix_max_x=[]
        xmax=-math.inf
        for row in records:
            xmax=max(xmax,row[1]);self._text_prefix_max_x.append(xmax)
        self._text_index_token=token

    def _query_text(self,bounds,inside=False):
        self._ensure_text_index()
        minx,maxx,miny,maxy=bounds
        lower=bisect_left(self._text_prefix_max_x,minx)
        upper=bisect_right(self._text_min_x,maxx)
        token=self.draw._scene_token()
        result=[]
        for x0,x1,y0,y1,anchor,doc_key,layer,idx,entity in self._text_index[lower:upper]:
            if not box_overlap((x0,x1,y0,y1),bounds,inside=inside):continue
            if not self.draw._snt_visible(doc_key,layer):continue
            result.append({'source':'snt','coords':[anchor], 'type':'text',
                           'doc_key':doc_key,'layer':layer,'entity':entity,
                           'bbox':(x0,x1,y0,y1),
                           'key':(doc_key,idx),'scene_token':token})
            if len(result)>=self.selection_manager.MAX_BULK:break
        return result

    def _text_pick(self,p):
        tol=max(self.draw.canvas.world_per_pixel()*15.,1.e-9)
        near=self._query_text((p[0]-tol,p[0]+tol,p[1]-tol,p[1]+tol))
        best=None
        for feat in near:
            x0,x1,y0,y1=feat['bbox']
            # Distance to the LABEL BOX, not only its insertion anchor.
            dist=math.hypot(max(x0-p[0],0.,p[0]-x1),
                            max(y0-p[1],0.,p[1]-y1))
            if dist<=tol and (best is None or dist<best[0]):
                best=(dist,feat)
        return best[1] if best else None

    def _select_text_rectangle(self,start,end):
        minx,maxx=sorted((start[0],end[0]))
        miny,maxy=sorted((start[1],end[1]))
        hits=self._query_text((minx,maxx,miny,maxy))
        self.selection_manager.apply(hits,self._selection_modifier())
        self.message(f'SNT Text: {len(hits)} label(s) matched; '
                     f'{self.selection_manager.count()} selected')
        self._reset_gesture()

    def _visible_text_features(self):
        self._ensure_text_index()
        hits=[]
        token=self.draw._scene_token()
        for x0,x1,y0,y1,anchor,doc_key,layer,idx,entity in self._text_index:
            if not self.draw._snt_visible(doc_key,layer):continue
            hits.append({'source':'snt','coords':[anchor], 'type':'text',
                         'doc_key':doc_key,'layer':layer,'entity':entity,
                         'bbox':(x0,x1,y0,y1),
                         'key':(doc_key,idx),'scene_token':token})
            if len(hits)>=self.selection_manager.MAX_BULK:break
        return hits

    def select_all_snt_text(self):
        self.selection_manager.apply(self._visible_text_features(),'new')

    def invert_snt_text_selection(self):
        candidates=self._visible_text_features()
        current=self.selection_manager._prune()
        self.selection_manager.apply([f for f in candidates if f['key'] not in current],'new')

    def press(self,pos,button):
        if not self.is_active():return False
        if self._intersect_mode:
            if button==Qt.RightButton:
                self._intersect_delete_part()
            elif button==Qt.LeftButton:
                try:p=self.draw.canvas.section_at(pos.x(),pos.y())
                except RuntimeError:return True
                self._intersect_selected_part=self._intersect_nearest_part(p,20.)
                self._intersect_hovered_part=None
                self.draw._paint()
                if self._intersect_selected_part is None:
                    self.message('Intersect: click a highlighted segment to select')
                else:
                    self.message('Intersect: segment selected (red). Right-click to remove it')
            return True
        if button==Qt.RightButton:
            if self._copy_mode and self._method=='snt_text':
                self.exit_copy_mode()
                self.message('SNT text placement cancelled; original labels unchanged')
                return True
            if self._method=='shape' and len(self._gesture)>=3:
                self._select_area('shape',list(self._gesture))
            else:self._reset_gesture()
            return True
        if button!=Qt.LeftButton:return False
        if self._method in ('block','shape','cline','snt_text'):
            print(f'[Native Element Selection] press method={self._method} '
                  f'copy={self._copy_mode}', flush=True)
        try:p=self.draw.canvas.section_at(pos.x(),pos.y())
        except RuntimeError:return True
        if self._copy_mode and self._method=='snt_text':
            # A stale placement flag from an older dialog must never paste an
            # imported SNT label. SNT Text is selection + clipboard only.
            self.exit_copy_mode()
        if self._move_mode or self._copy_mode:
            if not self.selection_manager.count():
                feat=(self._text_pick(p) if self._method=='snt_text'
                      else self._pick_editable(p))
                if feat:self.selection_manager.apply([feat]);self.message('Click start point, then target')
                else:self.message('Select an element first')
                return True
            if self._offset_origin is None:
                self._offset_origin=p
                self._hover=p
                self.draw._paint()
                self.message('Click target point (or drag) to place geometry')
            else:
                self._place_transform(p)
            return True
        if self._method=='individual':
            hit=self._pick_editable(p)
            self._set_hover_feature(hit)
            mode=self._selection_modifier()
            if self._intersect_mode and mode=='new':mode='add'
            self.selection_manager.apply([hit] if hit else [],mode)
        elif self._method=='snt_text':
            self._set_hover_feature(None)
            # VTK SNT Text uses a *dragged rectangle*. A click without a
            # meaningful drag still picks the nearest visible text label.
            self._start=p
        elif self._method=='shape':
            self._set_hover_feature(None)
            # Double-clicks generate duplicate click positions on some Qt
            # input paths. Keep only distinct polygon corners.
            if (not self._gesture or
                    math.dist(self._gesture[-1],p)>max(self.draw.canvas.world_per_pixel(),1.e-10)):
                self._gesture.append(p)
        elif self._method in ('block','cline'):
            self._set_hover_feature(None)
            if self._start is None:
                self._start=p
            elif self._method=='cline':
                self._select_area('cline',[self._start,p])
        self.draw._paint()
        return True

    def move(self,pos,buttons):
        if not self.is_active():return False
        try:self._hover=self.draw.canvas.section_at(pos.x(),pos.y())
        except RuntimeError:return True
        if self._intersect_mode:
            if self._intersect_selected_part is None:
                hovered=self._intersect_nearest_part(self._hover,15.)
                if hovered!=self._intersect_hovered_part:
                    self._intersect_hovered_part=hovered
                    self.draw._paint()
            return True
        if (self._method == 'individual'
                and not (self._gesture or self._start is not None or
                         self._offset_origin is not None or self._move_mode or self._copy_mode)):
            # Hover picks target ONLY editable Draw + completed Measure objects,
            # never the million-entity SNT/LiDAR GPU buffers. 20 Hz upper bound.
            now = time.monotonic()
            if now - self._hover_pick_at >= .05:
                self._hover_pick_at = now
                self._set_hover_feature(self._pick_editable(self._hover))
        elif self._hover_feature is not None:
            self._set_hover_feature(None)
        if (self._gesture or self._start is not None or
                self._offset_origin is not None):
            self.draw._paint()
        return True

    def release(self,pos,button):
        if not self.is_active():return False
        if self._intersect_mode:return True
        if button!=Qt.LeftButton:return False
        if (self._move_mode or self._copy_mode) and self._offset_origin is not None:
            try:p=self.draw.canvas.section_at(pos.x(),pos.y())
            except RuntimeError:return True
            if math.dist(p,self._offset_origin)>self.draw.canvas.world_per_pixel()*3:
                self._place_transform(p)
            return True
        if self._method == 'cline' and self._start is not None:
            # Support VTK-style drawing the crossing line by either click-tap
            # at both endpoints or press-drag-release.
            try:p=self.draw.canvas.section_at(pos.x(),pos.y())
            except RuntimeError:return True
            if math.dist(self._start,p)>self.draw.canvas.world_per_pixel()*3.:
                self._select_area('cline',[self._start,p])
            return True
        if self._method in ('block','snt_text') and self._start is not None:
            try:p=self.draw.canvas.section_at(pos.x(),pos.y())
            except RuntimeError:self._reset_gesture();return True
            if self._method=='snt_text':
                if math.dist(self._start,p)>self.draw.canvas.world_per_pixel()*3.:
                    self._select_text_rectangle(self._start,p)
                else:
                    hit=self._text_pick(p)
                    self.selection_manager.apply([hit] if hit else [],
                                                 self._selection_modifier())
                    self.message(f'SNT Text: {self.selection_manager.count()} selected')
                    self._reset_gesture()
                return True
            if math.dist(self._start,p)>self.draw.canvas.world_per_pixel()*3.:
                self._select_area('block',[self._start,p])
            # A click followed by no drag allows another click to set corner.
            return True
        return True

    def double_click(self,pos,button):
        if self._intersect_mode:return True
        if not self.is_active() or button!=Qt.LeftButton:return False
        if self._method=='shape' and len(self._gesture)>=3:
            self._select_area('shape',list(self._gesture))
            return True
        if self._method=='cline' and self._start is not None:
            try:p=self.draw.canvas.section_at(pos.x(),pos.y())
            except RuntimeError:return True
            self._select_area('cline',[self._start,p])
            return True
        return False

    def key(self,key):
        if not self.is_active():return False
        if self._intersect_mode and key==Qt.Key_Escape:
            self.exit_intersect_mode(commit=False)
            self.message('Intersect cancelled; no drawings changed')
            return True
        if self._intersect_mode and key in (Qt.Key_Enter,Qt.Key_Return):
            self.exit_intersect_mode(commit=True)
            return True
        if key==Qt.Key_Escape:
            self._reset_gesture()
            self._move_mode=self._copy_mode=False
            self._snt_copy_sources=[]
            return True
        if key in (Qt.Key_Enter,Qt.Key_Return) and self._method=='shape':
            if len(self._gesture)>=3:self._select_area('shape',list(self._gesture))
            return True
        return False

    def preview_shapes(self):
        if self._intersect_mode:
            shapes=[]
            for idx,part in enumerate(self._intersect_parts):
                if part['deleted']:continue
                source=part['source'].get('drawing',{})
                color=str(source.get('color','#00dcff'))
                width=max(1,int(source.get('width',2)))
                if idx==self._intersect_selected_part:
                    color='#ff1717';width=max(width,5)
                elif idx==self._intersect_hovered_part:
                    # Hover is cyan; a chosen segment stays red as a deliberate
                    # destructive Intersect operation warning.
                    color='#00ffff';width=max(width,4)
                shapes.append(('drawing',part['coords'],
                               {'type':'polyline','color':color,'width':width}))
            return shapes
        pts=[]
        if self._method=='shape':pts=list(self._gesture)
        elif self._start is not None:pts=[self._start]
        if pts and self._hover is not None:pts.append(self._hover)
        shapes=[]
        if self._offset_origin is not None and self._hover is not None:
            dx=self._hover[0]-self._offset_origin[0]
            dy=self._hover[1]-self._offset_origin[1]
            for feat in self.selection_manager.get_features()[:100]:
                # The preview is still smooth/render-sampled, but must not
                # advertise its samples as new editable control vertices.
                is_curve = feat.get('source') == 'native' and feat.get('type') == 'curve'
                shapes.append(('drawing_preview',translate(feat['coords'],dx,dy),
                               {'type':'polyline','color':'#79d6ff','width':2,
                                'show_vertices': not is_curve}))
        if pts:
            if self._method in ('block','snt_text') and len(pts)>=2:
                a,b=pts[0],pts[-1]
                pts=[a,(b[0],a[1]),b,(a[0],b[1]),a]
            shapes.append(('drawing_preview',pts,
                           {'type':'polyline','color':'#ffd54f','width':2}))
        # SNT Text is read-only selection. Copy goes to the system clipboard,
        # not to a native Draw paste ghost; never render duplicate label text.
        if self._method=='snt_text':
            # Outline true/approximated label bounds, so a selected label is
            # visibly highlighted even though imported SNT text stays read-only.
            for feat in self.selection_manager.get_features()[:200]:
                if feat['source']!='snt' or 'bbox' not in feat:continue
                x0,x1,y0,y1=feat['bbox']
                # Degenerate-height text anchors still get a visible marker.
                pad=self.draw.canvas.world_per_pixel()*2.
                if x0==x1:x0-=pad;x1+=pad
                if y0==y1:y0-=pad;y1+=pad
                # Highlight the ORIGINAL label's bounds only. Solid outline +
                # low-alpha fill are not a movable/pasteable text preview.
                shapes.append(('drawing',[(x0,y0),(x1,y0),(x1,y1),
                                           (x0,y1),(x0,y0)],
                               {'type':'polyline','color':'#ffbb32','width':2,
                                'style':'solid','show_vertices':False,
                                'fill_enabled':True,'fill_color':'#ffbb32',
                                'fill_opacity':0.12}))
        if self._vertex_display:
            for feat in self.selection_manager.get_features()[:200]:
                # Circle perimeter samples are rendering detail, not CAD vertices.
                if feat['type'] in ('circle', 'text'):
                    continue
                coords = feat['coords']
                if feat['type'] == 'curve' and feat['source'] == 'native':
                    coords = feat['drawing'].get('coords', [])
                if len(coords) > 2 and coords[0] == coords[-1]:
                    coords = coords[:-1]  # don't show the first/last corner twice
                for point in coords[:200]:
                    shapes.append(('drawing_preview',[point],
                                   {'type':'point','color':'#f6bd60','width':3}))
        return shapes

    def highlight_features(self):
        # The Intersect preview replaces its sources with selectable pieces.
        if self._intersect_mode:return []
        return self.selection_manager.get_features()[:500]

    def transform_hidden_drawing_ids(self):
        """Native sources temporarily replaced by the active transform ghost."""
        if self._offset_origin is None or not (self._move_mode or self._copy_mode):
            return set()
        return {
            id(feature['drawing'])
            for feature in self.selection_manager.get_features()
            if feature.get('source') == 'native' and feature.get('drawing') is not None
        }

    def _place_transform(self, target):
        """Retire the ghost before any document/selection callback repaints.

        Selection changes repaint synchronously inside _transform_selected.
        Leaving the anchor live there applies the displacement a second time
        to the newly selected result and leaves that wrong frame on screen.
        """
        origin = self._offset_origin
        if origin is None:
            return
        dx, dy = target[0] - origin[0], target[1] - origin[1]
        self._offset_origin = None
        self._hover = target
        self._transform_selected(lambda coords: translate(coords, dx, dy),
                                 copy=self._copy_mode)
        self.draw._paint()

    def _transform_selected(self,transform,copy=False):
        selected=self.selection_manager.get_features()
        if not selected:return self.message('Select at least one element')
        if self._method=='snt_text':
            return self.message('Imported SNT Text is read-only; Copy copies label text')
        # Measurement line/path is a selectable overlay, not an editable Draw
        # document. Never silently turn a Move/Delete of a measurement into a
        # new drawing, which would leave the original measurement in place.
        non_draw=[f for f in selected if f['source']=='measure']
        if non_draw and not copy:
            return self.message('Measure Line/Path selection is read-only for Move/Rotate/Extend; '
                                'the Measure tab still owns measurement edits')
        before=list(self.draw.drawings)
        replaced=[];created=[];imported=0
        for feat in selected:
            # `feat['coords']` is the *rendered*, 20-samples-per-span curve.
            # It is valid for hit-testing, never for storing an editable copy.
            # Use original control vertices for native curves instead.
            if feat['source'] == 'native' and feat['drawing'].get('type') == 'curve':
                coords = feat['drawing'].get('coords', ())
            else:
                coords = feat['coords']
            if not coords:continue
            result=transform(coords)
            if not result or any(not math.isfinite(v) for p in result for v in p[:2]):
                continue
            if feat['source']=='native':
                orig=feat['drawing']
                new=dict(orig,coords=list(result))
                # Circle/rectangle/hatch store special non-vertex geometry.
                # Preserve the old VTK-compatible polyline conversion there.
                # A Curve, however, stores real control vertices in 'coords'.
                # Keep type='curve' so the native renderer samples the same
                # Catmull-Rom spline on the translated/rotated/mirrored copy.
                if orig.get('type') in ('circle','rectangle','hatch'):
                    new['type']='polyline'
                    new.pop('hatch_segments',None)
                elif orig.get('type') == 'curve':
                    new.pop('interpolated', None)
                    new.pop('control_points', None)
                if not copy:replaced.append(orig)
            elif feat['source']=='measure':
                # Copying a measurement yields an independent Draw overlay;
                # the original measurement and its units remain unchanged.
                new={'type':'polyline' if len(coords)>1 else 'point',
                     'coords':list(result),'derived_from_measure':True}
            else:
                imported+=1
                new={'type':'polyline' if len(coords)>1 else 'point',
                     'coords':list(result),'derived_from_snt':True}
            created.append(new)
        if not created:return self.message('No editable geometry created')
        ids={id(d) for d in replaced}
        self.draw.drawings[:]=[d for d in self.draw.drawings if id(d) not in ids]
        first_new=len(self.draw.drawings)
        self.draw.drawings.extend(created)
        self.draw.record_object_change(before)
        self.selection_manager.clear()
        self.selection_manager.apply([self._native_feature(first_new+i,d)
                                      for i,d in enumerate(created)])
        if imported:self.message('Imported SNT preserved; edited result is a new native overlay')
        else:self.message(f'{len(created)} element(s) updated')

    def enter_move_mode(self):
        if not self.selection_manager.count():return self.message('Select geometry before Move')
        if self._method=='snt_text':
            return self.message('SNT Text labels are read-only. Use Copy to copy their text.')
        if any(f['source']=='measure' for f in self.selection_manager.get_features()):
            return self.message('Measure Line/Path is selection-only; use Measure tab for edits')
        self._move_mode=True;self._copy_mode=False;self._offset_origin=None
        self.message('Move: click anchor, then target; Escape cancels')

    def exit_move_mode(self,cancel=False):
        self._move_mode=False;self._offset_origin=None

    def enter_copy_mode(self):
        if self._method=='snt_text':
            selected=[f for f in self.selection_manager.get_features()
                      if f.get('source')=='snt']
            if not selected:
                return self.message('Select SNT text label(s) before Copy')
            # Clipboard-only by design: keep ORIGINAL SNT labels selected and
            # highlighted; never arm an editable Draw label/paste preview.
            # exit_copy_mode also clears any stale placement state.
            self.exit_copy_mode()
            self.copy_selected_snt_text_to_clipboard()
            self.message(f'Copied {len(selected)} SNT text label(s) to clipboard; '
                         'original labels remain highlighted')
            return True
        if not self.selection_manager.count():return self.message('Select geometry before Copy')
        self._copy_mode=True;self._move_mode=False;self._offset_origin=None
        self.message('Copy: click anchor, then target; Escape cancels')

    def exit_copy_mode(self):
        self._copy_mode=False;self._offset_origin=None
        self._snt_copy_sources=[]
        self.draw._paint()

    def _place_snt_copies(self, position):
        """Commit one reversible Draw operation; never mutate an SNT file."""
        from gui.native_snt_text_copy import copy_group
        if (not self._snt_copy_sources or
                self.draw._scene_token() != self._snt_copy_sources[0].get('scene_token')):
            self.exit_copy_mode()
            self.message('SNT source changed; copy cancelled safely')
            return False
        pasted = copy_group(self._snt_copy_sources, position)
        if not pasted:
            self.exit_copy_mode()
            self.message('No valid SNT text geometry to paste')
            return False
        before = list(self.draw.drawings)
        self.draw.drawings.extend(pasted)
        self.draw.record_object_change(before)
        self.exit_copy_mode()
        self.message(f'Pasted {len(pasted)} new text label(s); originals preserved; '
                     'Ctrl+Z undoes paste. Use Individual to select/edit copies')
        print(f'[Native Element Selection] snt_text pasted={len(pasted)} '
              f'undo_group=1', flush=True)
        return True

    def _notify_intersect_button(self, active):
        popup=getattr(self.app,'_element_selection_dialog',None)
        if popup is not None:
            button=getattr(popup,'btn_intersect',None)
            if button is not None:button.setChecked(active)
            label=getattr(popup,'lbl_count',None)
            if label is not None:
                label.setText('Intersect Active' if active else
                              f'Selection: {self.selection_manager.count()}')

    def _intersect_nearest_part(self, point, tolerance_px):
        tol=max(1e-10,self.draw.canvas.world_per_pixel()*tolerance_px)
        best=(tol,None)
        for index,part in enumerate(self._intersect_parts):
            if part['deleted']:continue
            nearest=nearest_segment(point,part['coords'])
            if nearest is not None and nearest[0] <= best[0]:
                best=(nearest[0],index)
        return best[1]

    def enter_intersect_mode(self):
        """VTK Intersect: preview all pieces; select left, remove right, commit on exit."""
        if not self.is_active():return False
        if self._intersect_mode:return True
        selection=self.selection_manager.get_features()
        if not any(f['source']=='native' and
                   f['drawing'].get('source') not in ('block_creation','cell_creation')
                   for f in selection):
            self.message('Intersect needs editable native Draw elements; SNT remains read-only')
            return False
        # Protected GPU-rendered creation blocks must not be modified here.
        selection=[f for f in selection if f['source']!='native' or
                   f['drawing'].get('source') not in ('block_creation','cell_creation')]
        try:
            parts,crossings=split_selected_features(selection)
        except ValueError as exc:
            self.message(str(exc));return False
        if not parts:
            self.message('Intersect: select editable lines or polygons')
            return False
        self._intersect_parts=parts
        self._intersect_source_ids={id(p['source']['drawing']) for p in parts}
        self._intersect_selected_part=None
        self._intersect_hovered_part=None
        self._intersect_delete_history=[]
        self._intersect_redo_history=[]
        self._intersect_scene_token=self.draw._scene_token()
        self._intersect_mode=True
        self._move_mode=self._copy_mode=False
        self._offset_origin=None
        self.selection_manager.clear()
        self._notify_intersect_button(True)
        self.message(f'Intersect: {crossings} crossing(s), {len(parts)} pieces. '
                     'Left-click piece, right-click remove; click Intersect to finish',0)
        return True

    def _intersect_delete_part(self):
        index=self._intersect_selected_part
        if index is None or self._intersect_parts[index]['deleted']:
            self.message('Intersect: first left-click the piece to remove')
            return
        self._intersect_parts[index]['deleted']=True
        self._intersect_delete_history.append(index)
        self._intersect_redo_history=[]
        self._intersect_selected_part=None
        self._intersect_hovered_part=None
        self.draw._paint()
        self.message('Intersect: piece marked for removal; click Intersect to commit')

    def intersect_history(self, redo=False):
        """Ctrl+Z/Y restores a piece during the preview, without editing the document."""
        if not self._intersect_mode:return False
        stack=self._intersect_redo_history if redo else self._intersect_delete_history
        if not stack:
            self.message('Intersect: no pending piece deletions to undo/redo')
            return True
        index=stack.pop()
        self._intersect_parts[index]['deleted']=bool(redo)
        target=self._intersect_delete_history if redo else self._intersect_redo_history
        target.append(index)
        self._intersect_selected_part=self._intersect_hovered_part=None
        self.draw._paint()
        return True

    def exit_intersect_mode(self,commit=True):
        """Commit only explicitly removed native parts, atomically and Undo-safe."""
        if not self._intersect_mode:return False
        parts=self._intersect_parts
        source_ids=set(self._intersect_source_ids)
        scene_token=self._intersect_scene_token
        self._intersect_mode=False
        self._intersect_parts=[]
        self._intersect_source_ids=set()
        self._intersect_selected_part=None
        self._intersect_hovered_part=None
        self._intersect_delete_history=[]
        self._intersect_redo_history=[]
        self._intersect_scene_token=None
        self._notify_intersect_button(False)
        if not commit:
            self.draw._paint()
            return True
        if scene_token!=self.draw._scene_token():
            self.draw._paint()
            self.message('Intersect discarded because attached SNT scene changed')
            return False
        deleted_ids={id(part['source']['drawing']) for part in parts if part['deleted']}
        if not deleted_ids:
            self.draw._paint()
            self.message('Intersect ended; no geometry changed')
            return True
        # A source is only replaced if the user actually removed one of its
        # pieces. All unaffected drawings preserve their identity and metadata.
        before=list(self.draw.drawings)
        replacements={}
        for original in before:
            if id(original) not in deleted_ids or id(original) not in source_ids:
                continue
            surviving=[p['coords'] for p in parts
                       if p['source'].get('drawing') is original and not p['deleted']]
            pieces=[]
            for coords in surviving:
                derived=dict(original)
                derived['coords']=list(coords)
                if (original.get('type') in ('rectangle','curve','circle','hatch') or
                        len(coords)<4 and coords[0]!=coords[-1] and
                        original.get('type') in ('polyline','ortho')):
                    derived['type']='polyline'
                for stale in ('hatch_segments','control_points','interpolated'):
                    derived.pop(stale,None)
                pieces.append(derived)
            replacements[id(original)]=pieces
        updated=[]
        for original in before:
            if id(original) in replacements:updated.extend(replacements[id(original)])
            else:updated.append(original)
        self.draw.drawings[:]=updated
        self.draw.record_object_change(before)
        self.selection_manager.clear()
        self.message(f'Intersect committed: {sum(p["deleted"] for p in parts)} piece(s) removed')
        return True

    def extend(self):
        if self._method=='snt_text':
            return self.message('Imported SNT Text is read-only')
        selected=self.selection_manager.get_features()
        if not selected:return self.message('Select an open line to extend')
        value,ok=QInputDialog.getDouble(self.app,'Extend','Extend end by (world units):',1.,0.,1e9,3)
        if not ok:return
        def extension(points):
            if len(points)<2 or points[0]==points[-1]:return None
            a,b=points[-2],points[-1];dx,dy=b[0]-a[0],b[1]-a[1]
            length=math.hypot(dx,dy)
            if length<1e-12:return None
            return list(points[:-1])+[(b[0]+dx*value/length,b[1]+dy*value/length)]
        self._transform_selected(extension,copy=False)

    def rotate_selected(self):
        if self._method=='snt_text':
            return self.message('Imported SNT Text is read-only')
        angle,ok=QInputDialog.getDouble(self.app,'Rotate','Degrees counter-clockwise:',90.,-360.,360.,3)
        if not ok:return
        pivot=self._pivot()
        if pivot is not None:self._transform_selected(lambda coords:rotate(coords,angle,pivot))

    def mirror_selected(self):
        if self._method=='snt_text':
            return self.message('Imported SNT Text is read-only')
        axis,ok=QInputDialog.getItem(self.app,'Mirror','Axis:',
                                     ['horizontal','vertical'],0,False)
        if not ok:return
        pivot=self._pivot()
        if pivot is not None:self._transform_selected(lambda coords:mirror(coords,axis,pivot))

    def _pivot(self):
        pts=[p for feat in self.selection_manager.get_features() for p in feat['coords']]
        if not pts:self.message('Select at least one element');return None
        return sum(p[0] for p in pts)/len(pts),sum(p[1] for p in pts)/len(pts)

    def delete_selected(self):
        selected=self.selection_manager.get_features()
        target={id(f['drawing']) for f in selected if f['source']=='native'}
        imported=sum(f['source']=='snt' for f in selected)
        measured=sum(f['source']=='measure' for f in selected)
        if target:
            old=list(self.draw.drawings)
            self.draw.drawings[:]=[d for d in self.draw.drawings if id(d) not in target]
            self.draw.record_object_change(old)
        self.selection_manager.clear()
        if imported or measured:
            self.message(f'{len(target)} drawings deleted; {imported} imported SNT text and '
                         f'{measured} measurements protected')
        else:self.message(f'{len(target)} drawing(s) deleted')

    def copy_selected_snt_text_to_clipboard(self):
        texts=[]
        for f in self.selection_manager.get_features():
            e=f.get('entity',{})
            if f.get('type')=='text':
                texts.append(str(e.get('text') or e.get('value') or ''))
        if texts:
            QApplication.clipboard().setText('\n'.join(texts))
            self.message(f'Copied {len(texts)} SNT text label(s)')
        else:self.message('Select SNT text label(s) first')
