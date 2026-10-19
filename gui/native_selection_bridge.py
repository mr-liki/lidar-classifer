"""Element Selection panel controller for the *visible* native Vulkan plan.

VTK's ElementSelectTool remains the owner of VTK interaction. In a Vulkan
main view VTK actors cannot receive mouse picks because the Vulkan widget is
above the QVTK interactor. This adapter uses NativeDraw's same coordinate and
SNT visibility mappings, never writes to imported source entity dictionaries,
and never touches the VTK camera or LiDAR point buffers.
"""
from __future__ import annotations

import math
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QApplication, QInputDialog

from gui.native_selection_geometry import (
    rect_selects, polygon_selects, line_selects, translate, rotate, mirror,
)
from gui.native_vector_pick import segment_intersection


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
                live=id(feat.get('drawing')) in native_ids
            else:
                live=(feat.get('scene_token')==scene
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
            else:
                result.append({'type':f['type'],'coords':list(f['coords']),
                               'source':'snt_readonly','layer':f['layer'],
                               'imported':True})
        return result

    def count(self):
        return len(self._prune())

    def clear(self, emit=True):
        self._selected.clear()
        self.controller.draw._paint()
        if emit:self.selection_changed.emit(0)

    def apply(self, features, mode='new'):
        old = dict(self._prune())
        if mode=='new':old.clear()
        for feat in features:
            if feat is None:continue
            if feat['source']=='snt':
                feat=dict(feat,scene_token=self.controller.draw._scene_token())
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
        self.controller.draw._paint()
        self.selection_changed.emit(len(old))

    def select_all(self,predicate=None):
        # For safety don't select every SNT feature: on a large map that could
        # imply millions of CPU QPainter highlights. All means editable Draw
        # objects here; users still pick imported SNT by click/fence.
        features=[]
        for i,d in enumerate(self.controller.draw.drawings):
            if predicate is not None and not predicate(d):continue
            features.append(self.controller._native_feature(i,d))
        index=self.controller.draw._snt_index()
        if predicate is None and len(index.records)+len(features)<=self.MAX_BULK:
            features.extend(index.query_bbox(-math.inf,math.inf,-math.inf,math.inf,
                                             visible=self.controller.draw._snt_visible))
        elif index.records:
            self.controller.message('Large SNT: Select All includes native drawings only. Use a selection fence for SNT.')
        self.apply(features,'new')

    def invert_all(self):
        features=[self.controller._native_feature(i,d)
                  for i,d in enumerate(self.controller.draw.drawings)]
        previous={f['key']:f for f in self.get_features() if f['source']=='snt'}
        for f in features:
            if f['key'] not in self._selected:previous[f['key']]=f
        self._selected=dict(list(previous.items())[:self.MAX_BULK])
        self.controller.draw._paint()
        self.selection_changed.emit(len(self._selected))


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
        self._start=None
        self._hover=None
        self._gesture=[]
        self._drag_origin=None
        self._offset_origin=None
        self._vertex_display=False
        self._text_index_token=None
        self._text_index=[]

    def message(self, msg, timeout=3500):
        bar=getattr(self.app,'statusBar',lambda:None)()
        if bar is not None:bar.showMessage(msg,timeout)

    def is_active(self):
        return self._active and self.draw.tool=='select'

    def activate(self, method='individual'):
        if self.draw.tool!='select' and not self.draw.activate('select'):
            return False
        self._active=True
        self._method=method
        self.draw._selection_bridge=self
        self._reset_gesture()
        return True

    def deactivate(self):
        self._active=False
        self._reset_gesture()
        self._move_mode=self._copy_mode=self._intersect_mode=False
        if self.draw._selection_bridge is self:
            self.draw._selection_bridge=None
        self.selection_manager.clear()
        if self.draw.tool=='select':self.draw.cancel()

    def set_method(self, method):
        self._method=method
        self._reset_gesture()
        self.selection_manager.clear()
        self.message(f'Vulkan Select: {method.replace("_"," ")}')

    def set_enclose_mode(self, mode):
        self._enclose_mode=mode

    def _reset_gesture(self):
        self._start=self._hover=self._drag_origin=self._offset_origin=None
        self._gesture=[]
        if getattr(self.draw,'_selection_bridge',None) is self:
            self.draw._paint()

    @staticmethod
    def _selection_modifier():
        mods=QApplication.keyboardModifiers()
        if mods & Qt.ControlModifier:return 'add'
        if mods & Qt.ShiftModifier:return 'subtract'
        return 'new'

    def _native_feature(self, index, drawing):
        return {'source':'native', 'index':index, 'drawing':drawing,
                'coords':self.draw._native_coords(drawing),
                'type':drawing.get('type','line'),
                'key':('native',id(drawing))}

    def _features_bbox(self, lower, upper):
        a,b=lower,upper
        minx,maxx=sorted((a[0],b[0]));miny,maxy=sorted((a[1],b[1]))
        count=0
        for i,d in enumerate(self.draw.drawings):
            coords=self.draw._native_coords(d)
            if not coords:continue
            if not (max(p[0] for p in coords)<minx or min(p[0] for p in coords)>maxx or
                    max(p[1] for p in coords)<miny or min(p[1] for p in coords)>maxy):
                yield self._native_feature(i,d);count+=1
                if count>=self.selection_manager.MAX_BULK:return
        for feat in self.draw._snt_index().query_bbox(minx,maxx,miny,maxy,
                                                      visible=self.draw._snt_visible):
            yield feat;count+=1
            if count>=self.selection_manager.MAX_BULK:return

    def _select_area(self, method, vertices):
        if not vertices:return
        a,b=vertices[0],vertices[-1]
        if method=='shape':
            bounds=[(min(p[0] for p in vertices),min(p[1] for p in vertices)),
                    (max(p[0] for p in vertices),max(p[1] for p in vertices))]
        else:
            bounds=[a,b]
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
        self.selection_manager.apply(hits,self._selection_modifier())
        self.message(f'Selected {self.selection_manager.count()} element(s)')
        self._reset_gesture()

    def _text_pick(self,p):
        """SNT text is read-only; index label anchors once per attachment set."""
        from bisect import bisect_left, bisect_right
        from pathlib import Path
        token=self.draw._scene_token()
        if token!=self._text_index_token:
            anchors=[]
            for attachment in getattr(self.app,'snt_attachments',()) or ():
                records=attachment.get('_render_entities')
                if records is None:records=attachment.get('entities') or ()
                name=attachment.get('full_path') or attachment.get('filename') or ''
                doc_key=('snt',str(Path(str(name))))
                for idx,entity in enumerate(records):
                    if not isinstance(entity,dict):continue
                    if str(entity.get('type','')).lower() not in ('text','mtext'):continue
                    layer=str(entity.get('layer','0') or '0')
                    anchor=next((entity[key] for key in
                                 ('position','insertion_point','insert','points')
                                 if entity.get(key) is not None),None)
                    try:
                        if anchor is None:continue
                        if len(anchor) and isinstance(anchor[0],(list,tuple)):
                            anchor=anchor[0]
                        q=(float(anchor[0]),float(anchor[1]))
                    except (ValueError,IndexError,TypeError):continue
                    if not all(math.isfinite(v) for v in q):continue
                    anchors.append((q[0],q[1],doc_key,layer,idx,entity))
            anchors.sort(key=lambda row:row[0])
            self._text_index=anchors
            self._text_index_token=token
        tol=max(self.draw.canvas.world_per_pixel()*15,1e-9)
        xs=[row[0] for row in self._text_index]
        lower=bisect_left(xs,p[0]-tol)
        upper=bisect_right(xs,p[0]+tol)
        best=None
        for x,y,doc_key,layer,idx,entity in self._text_index[lower:upper]:
            if not self.draw._snt_visible(doc_key,layer):continue
            dist=math.hypot(x-p[0],y-p[1])
            if dist<=tol and (best is None or dist<best[0]):
                feat={'source':'snt','coords':[(x,y)], 'type':'text',
                      'doc_key':doc_key,'layer':layer,'entity':entity,
                      'key':(doc_key,idx),'scene_token':token}
                best=(dist,feat)
        return best[1] if best else None

    def _visible_text_features(self):
        # Build/cache the label index once; it does not draw a pick marker.
        self._text_pick((0.,0.))
        features=[]
        for x,y,doc_key,layer,idx,entity in self._text_index:
            if not self.draw._snt_visible(doc_key,layer):continue
            features.append({'source':'snt','coords':[(x,y)],'type':'text',
                             'doc_key':doc_key,'layer':layer,'entity':entity,
                             'key':(doc_key,idx),'scene_token':self.draw._scene_token()})
            if len(features)>=self.selection_manager.MAX_BULK:
                self.message('Text selection capped for responsiveness')
                break
        return features

    def select_all_snt_text(self):
        self.selection_manager.apply(self._visible_text_features(),'new')

    def invert_snt_text_selection(self):
        candidates=self._visible_text_features()
        current=self.selection_manager._prune()
        self.selection_manager.apply([f for f in candidates if f['key'] not in current],'new')

    def press(self,pos,button):
        if not self.is_active():return False
        if button==Qt.RightButton:
            if self._method=='shape' and len(self._gesture)>=3:
                self._select_area('shape',list(self._gesture))
            else:self._reset_gesture()
            return True
        if button!=Qt.LeftButton:return False
        try:p=self.draw.canvas.section_at(pos.x(),pos.y())
        except RuntimeError:return True
        if self._move_mode or self._copy_mode:
            if not self.selection_manager.count():
                feat=self.draw._pick_feature(p)
                if feat:self.selection_manager.apply([feat]);self.message('Click start point, then target')
                else:self.message('Select an element first')
                return True
            if self._offset_origin is None:
                self._offset_origin=p
                self.message('Click target point (or drag) to place geometry')
            else:
                dx=p[0]-self._offset_origin[0];dy=p[1]-self._offset_origin[1]
                self._transform_selected(lambda c:translate(c,dx,dy),copy=self._copy_mode)
                self._offset_origin=None
            return True
        if self._method=='individual':
            hit=self.draw._pick_feature(p)
            mode=self._selection_modifier()
            if self._intersect_mode and mode=='new':mode='add'
            self.selection_manager.apply([hit] if hit else [],mode)
        elif self._method=='snt_text':
            hit=self._text_pick(p)
            self.selection_manager.apply([hit] if hit else [],self._selection_modifier())
        elif self._method=='shape':
            self._gesture.append(p)
        elif self._method in ('block','cline'):
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
        # Idle hover does not need a Qt overlay repaint, which is important
        # when a huge SNT is resident in GPU vector buffers.
        if self._gesture or self._start is not None or self._offset_origin is not None:
            self.draw._paint()
        return True

    def release(self,pos,button):
        if not self.is_active():return False
        if button!=Qt.LeftButton:return False
        if (self._move_mode or self._copy_mode) and self._offset_origin is not None:
            try:p=self.draw.canvas.section_at(pos.x(),pos.y())
            except RuntimeError:return True
            if math.dist(p,self._offset_origin)>self.draw.canvas.world_per_pixel()*3:
                dx=p[0]-self._offset_origin[0];dy=p[1]-self._offset_origin[1]
                self._transform_selected(lambda coords:translate(coords,dx,dy),
                                         copy=self._copy_mode)
                self._offset_origin=None
            return True
        if self._method=='block' and self._start is not None:
            try:p=self.draw.canvas.section_at(pos.x(),pos.y())
            except RuntimeError:self._reset_gesture();return True
            if math.dist(self._start,p)>self.draw.canvas.world_per_pixel()*3.:
                self._select_area('block',[self._start,p])
            # A click followed by no drag allows another click to set corner.
            return True
        return True

    def double_click(self,pos,button):
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
        if key==Qt.Key_Escape:
            self._reset_gesture()
            self._move_mode=self._copy_mode=False
            return True
        if key in (Qt.Key_Enter,Qt.Key_Return) and self._method=='shape':
            if len(self._gesture)>=3:self._select_area('shape',list(self._gesture))
            return True
        return False

    def preview_shapes(self):
        pts=[]
        if self._method=='shape':pts=list(self._gesture)
        elif self._start is not None:pts=[self._start]
        if pts and self._hover is not None:pts.append(self._hover)
        shapes=[]
        if self._offset_origin is not None and self._hover is not None:
            dx=self._hover[0]-self._offset_origin[0]
            dy=self._hover[1]-self._offset_origin[1]
            for feat in self.selection_manager.get_features()[:100]:
                shapes.append(('drawing_preview',translate(feat['coords'],dx,dy),
                               {'type':'polyline','color':'#79d6ff','width':2}))
        if pts:
            if self._method=='block' and len(pts)>=2:
                a,b=pts[0],pts[-1]
                pts=[a,(b[0],a[1]),b,(a[0],b[1]),a]
            shapes.append(('drawing_preview',pts,
                           {'type':'polyline','color':'#ffd54f','width':2}))
        if self._vertex_display:
            for feat in self.selection_manager.get_features()[:200]:
                for point in feat['coords'][:200]:
                    shapes.append(('drawing_preview',[point],
                                   {'type':'point','color':'#f6bd60','width':3}))
        return shapes

    def highlight_features(self):
        # Don't re-stroke million-element SNT: only the actual selection.
        return self.selection_manager.get_features()[:500]

    def _transform_selected(self,transform,copy=False):
        selected=self.selection_manager.get_features()
        if not selected:return self.message('Select at least one element')
        before=list(self.draw.drawings)
        replaced=[];created=[];imported=0
        for feat in selected:
            coords=feat['coords']
            if not coords:continue
            result=transform(coords)
            if not result or any(not math.isfinite(v) for p in result for v in p[:2]):
                continue
            if feat['source']=='native':
                orig=feat['drawing']
                new=dict(orig,coords=list(result))
                # Circle/rectangle/hatch/curve have non-vertex source data.
                # They must be stored as transformed polyline geometry, not
                # left with a stale centre/radius, hatch scanlines or spline.
                if orig.get('type') in ('circle','rectangle','hatch','curve'):
                    new['type']='polyline'
                    new.pop('hatch_segments',None)
                if not copy:replaced.append(orig)
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
        self._move_mode=True;self._copy_mode=False;self._offset_origin=None
        self.message('Move: click anchor, then target; Escape cancels')

    def exit_move_mode(self,cancel=False):
        self._move_mode=False;self._offset_origin=None

    def enter_copy_mode(self):
        if self._method=='snt_text':return self.copy_selected_snt_text_to_clipboard()
        if not self.selection_manager.count():return self.message('Select geometry before Copy')
        self._copy_mode=True;self._move_mode=False;self._offset_origin=None
        self.message('Copy: click anchor, then target; Escape cancels')

    def exit_copy_mode(self):
        self._copy_mode=False;self._offset_origin=None

    def enter_intersect_mode(self):
        self._intersect_mode=True
        self.message('Intersect: select two crossing open lines, then press Intersect again to split the first')

    def exit_intersect_mode(self,commit=False):
        self._intersect_mode=False
        if not commit:
            return
        selected=self.selection_manager.get_features()
        if len(selected)<2:return self.message('Intersect needs two selected lines')
        one,two=selected[:2]
        ca,cb=one['coords'],two['coords']
        for i,(a,b) in enumerate(zip(ca,ca[1:])):
            for c,d in zip(cb,cb[1:]):
                p=segment_intersection(a,b,c,d)
                if p is None:continue
                before=list(self.draw.drawings)
                left=ca[:i+1]+[p];right=[p]+ca[i+1:]
                if one['source']=='native':
                    self.draw.drawings.remove(one['drawing'])
                for part in (left,right):
                    self.draw.drawings.append({'type':'polyline','coords':part,
                                               'derived_from_snt':one['source']=='snt'})
                self.draw.record_object_change(before)
                self.selection_manager.clear()
                return self.message('Intersection: split first line into two editable parts')
        self.message('No interior segment intersection found')

    def extend(self):
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
        angle,ok=QInputDialog.getDouble(self.app,'Rotate','Degrees counter-clockwise:',90.,-360.,360.,3)
        if not ok:return
        pivot=self._pivot()
        if pivot is not None:self._transform_selected(lambda coords:rotate(coords,angle,pivot))

    def mirror_selected(self):
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
        if target:
            old=list(self.draw.drawings)
            self.draw.drawings[:]=[d for d in self.draw.drawings if id(d) not in target]
            self.draw.record_object_change(old)
        self.selection_manager.clear()
        if imported:self.message(f'{len(target)} drawings deleted; {imported} imported SNT entities protected')
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
