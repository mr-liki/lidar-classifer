"""Saved, ordered By Class classification macros."""
import copy
import json
import os
import numpy as np
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QListWidget, QMessageBox, QPushButton,
    QSpinBox, QVBoxLayout, QWidget, QAbstractItemView)
from gui.lidar_classification_tools import (_BaseClassifyDialog, _collect_alignment_sources,
    classify_low_points, classify_isolated_points, classify_ground_ptd,
    classify_surface_points, classify_below_surface, classify_find_wires,
    classify_poles_by_centerline, _make_alignment_highlight_actor,
    ClassSelectorDialog, _get_las_classes, _get_class_colors,
    _make_color_icon, _make_multi_class_icon, _multi_class_label,
    _resolve_ground_class_defaults)

# key, label, kind, default, low, high
FIELDS = {
    'Convert': [],
    'Fence': [],
    'Height': [('reference_class','Reference class','class',1,0,255),('min_height','Min height (m)','float',0,-10000,10000),('max_height','Max height (m)','float',3,-10000,10000),('max_triangle','Max TIN edge (m)','float',100,.01,999999)],
    'Close': [('radius','Radius (m)','float',1,.01,1000)],
    'Low Points': [('search_mode','Search','choice:groups|single','groups',None,None),('max_count','Max count','int',6,1,100000),('more_than','Vertical gap (m)','float',.5,0,1000),('within','Within (m)','float',5,.01,10000)],
    'Isolated': [('in_classes','Neighbor classes','classes',[1,2,3,4,5],None,None),('if_fewer_than','Fewer than','int',1,1,100000),('within','Within (m)','float',5,.01,10000)],
    'Ground': [('current_ground','Current ground','class',2,0,255),('max_building_size','Max building size (m)','float',60,.01,10000),('terrain_angle','Terrain angle','float',88,0,90),('iteration_angle','Iteration angle','float',6,0,90),('iteration_distance','Iteration distance','float',1.4,0,1000)],
    'Surface Points': [('tolerance','Tolerance (m)','float',.05,.001,10)],
    'Below Surface': [('surface_type','Surface','choice:planar|curved','planar',None,None),('limit','Limit (m)','float',4,0,10000),('z_tolerance','Z tolerance (m)','float',.1,0,1000),('num_neighbors','Neighbors','int',25,3,1000)],
    'Find Wires': [('corridor_width','Total width (m)','float',3,.01,1000),('min_height_above_ground','Height above ground (m)','float',15,0,1000)],
    'Classify by Centerline': [('radius','Vertex radius (m)','float',1,.01,1000),('down_height','Height down from top (m)','float',5,0,1000)],
}
DEFAULTS = {'Convert':([1],2),'Fence':([1],2),'Height':([3,4],5),'Close':([1],2),'Low Points':([1],7),'Isolated':([1],7),'Ground':([1],2),'Surface Points':([1],2),'Below Surface':([1],7),'Find Wires':([4],14),'Classify by Centerline':([3,4],16)}
LINE_OPS = {'Find Wires','Classify by Centerline'}
# Close and Height may also receive step-specific fence polygons. Their fence
# selection is optional for backward-compatible loading of existing macros.
GEOMETRY_OPS = LINE_OPS | {'Fence', 'Close', 'Height'}
REQUIRED_GEOMETRY_OPS = LINE_OPS | {'Fence'}

def _collect_fence_sources(app):
    """Fetch VTK and Vulkan Draw fences as real world-coordinate polygons."""
    from gui.native_byclass_fences import NATIVE_FENCE_TYPES, _valid_ring
    sources = []
    digitizer = getattr(app, 'digitizer', None)
    drawings = list(getattr(digitizer, 'drawings', []) or [])
    for index, drawing in enumerate(drawings, 1):
        if not isinstance(drawing, dict):
            continue
        kind = str(drawing.get('type', '')).lower()
        if (kind not in NATIVE_FENCE_TYPES | {'rect', 'fence', 'closed_polyline'}
                or drawing.get('source') in ('cell_creation', 'block_creation')):
            continue
        try:
            getter = getattr(digitizer, '_get_drawing_coords', None)
            raw = getter(drawing) if callable(getter) else drawing.get('coords')
            if _valid_ring(raw) is None:
                continue
            coords = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            continue
        if coords.ndim != 2 or len(coords) < 3 or coords.shape[1] < 2:
            continue
        if not np.all(np.isfinite(coords)):
            continue
        sources.append({'label': f'Drawing {index}: {kind}', 'coords': coords,
                        'key': tuple(np.round(coords[:, :2].ravel(), 3))})

    # Native Draw stores rectangle/circle as two compact points. Reuse the
    # same expansion as FenceSelectorWidget and the By Class popup rather
    # than copying raw two-point control geometry to saved macros.
    try:
        from gui.native_byclass_fences import native_fence_records
        for index, fence in enumerate(native_fence_records(app), 1):
            coords = np.asarray(fence['coords'], dtype=float)
            sources.append({'label': f"Vulkan Draw {index}: {fence['type']}",
                            'coords': coords,
                            'key': tuple(np.round(coords[:, :2].ravel(), 3))})
    except Exception as exc:
        print(f'[Native By Class] macro fence enumeration failed: {exc}')
    return sources

def _inside_polygon(xy,polygon):
    vertices = np.asarray(polygon,dtype=float)[:,:2]
    x,y = xy[:,0],xy[:,1]
    inside = np.zeros(len(xy),dtype=bool)
    previous = vertices[-1]
    for current in vertices:
        x1,y1 = previous; x2,y2 = current
        crossing = ((y1 > y) != (y2 > y)) & (x < (x2-x1)*(y-y1)/(y2-y1+1e-300)+x1)
        inside ^= crossing
        previous = current
    return inside
FORMAT = 'naksha-by-class-macro'

class MacroClassPicker(QWidget):
    """A step-local display backed by the normal By Class selector dialog."""
    changed = Signal()

    def __init__(self, app, value, *, multi=False, allow_any=False, parent=None):
        super().__init__(parent)
        self.app = app
        self.multi = multi
        self.allow_any = allow_any
        self.codes = [None] if value is None else (list(value) if multi else [int(value)])
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.button = QPushButton()
        self.button.setMinimumWidth(190)
        self.button.setIconSize(QSize(32, 16))
        self.button.clicked.connect(self._choose)
        layout.addWidget(self.button)
        self._refresh()

    def value(self):
        if self.multi:
            return None if self.codes == [None] else list(self.codes)
        return int(self.codes[0])

    def _refresh(self):
        labels = _get_las_classes(self.app)
        colors = _get_class_colors(self.app)
        icon = None
        if self.multi and self.codes != [None]:
            icon = _make_multi_class_icon(self.codes, colors)
        elif not self.multi and self.codes[0] in colors:
            icon = _make_color_icon(colors[self.codes[0]], 16)
        self.button.setIcon(icon or QIcon())
        if self.codes == [None]:
            self.button.setText('Any class')
            self.button.setToolTip('Any eligible source class')
        elif self.multi:
            self.button.setText(_multi_class_label(self.codes))
            self.button.setToolTip(', '.join(labels.get(code, str(code)) for code in self.codes))
        else:
            self.button.setText(labels.get(self.codes[0], str(self.codes[0])))
            self.button.setToolTip(self.button.text())

    def _choose(self):
        dialog = ClassSelectorDialog(self.codes, self.app, multi=self.multi,
                                     extended=self.multi, parent=self.window())
        if not dialog.exec():
            return
        codes = dialog.selected_codes()
        if not codes:
            return
        if codes == [None] and not self.allow_any:
            QMessageBox.warning(self, 'Select Classes', 'Choose one or more specific classes.')
            return
        self.codes = list(codes)
        self._refresh()
        self.changed.emit()


def classes(value):
    if value is None or value == 'any': return None
    if isinstance(value,str): value = [s.strip() for s in value.split(',') if s.strip()]
    if not isinstance(value,(list,tuple)) or not value: raise ValueError('Enter source classes or any.')
    values = [int(v) for v in value]
    if any(v < 0 or v > 255 for v in values): raise ValueError('Classes must be 0-255.')
    return values

def new_step(op, app=None):
    source, target = DEFAULTS[op]
    params = {key: copy.deepcopy(default) for key, _, _, default, _, _ in FIELDS[op]}
    if app is not None:
        labels = _get_las_classes(app)
        names = {int(code): str(label).split(' - ', 1)[-1].strip().lower()
                 for code, label in labels.items()}

        def find(*wanted, fallback):
            return next((code for code, name in names.items() if name in wanted), fallback)

        unclassified, ground = _resolve_ground_class_defaults(app)
        low_veg = find('low vegetation', fallback=unclassified)
        medium_veg = find('medium vegetation', fallback=3)
        high_veg = find('high vegetation', fallback=4)
        noise = find('low point (noise)', 'noise', 'low points', fallback=unclassified)
        source, target = {
            'Convert': ([unclassified], ground),
            'Fence': ([unclassified], ground),
            'Height': ([unclassified], low_veg),
            'Close': ([unclassified], ground),
            'Low Points': ([unclassified], noise),
            'Isolated': ([unclassified], noise),
            'Ground': ([unclassified], ground),
            'Surface Points': ([unclassified], find('surface', 'surface points', fallback=unclassified)),
            'Below Surface': ([ground], noise),
            'Find Wires': ([high_veg], find('bare conductors', 'wire - conductor', fallback=high_veg)),
            'Classify by Centerline': ([medium_veg, high_veg], find('pylons or poles', 'transmission tower', fallback=medium_veg)),
        }[op]
        if op == 'Height':
            params['reference_class'] = ground
        elif op == 'Ground':
            params['current_ground'] = ground
    return {'operation': op, 'from_classes': list(dict.fromkeys(source)),
            'to_class': target, 'params': params, 'lines': []}

def validate_step(step):
    if not isinstance(step,dict) or step.get('operation') not in FIELDS: raise ValueError('Unknown By Class operation.')
    op = step['operation']; classes(step.get('from_classes'))
    target = int(step.get('to_class'))
    if not 0 <= target <= 255: raise ValueError('Target class must be 0-255.')
    params = step.get('params')
    if not isinstance(params,dict): raise ValueError('Missing parameters.')
    for key,_,kind,_,low,high in FIELDS[op]:
        if key not in params: raise ValueError('Missing '+key)
        val = params[key]
        if kind == 'classes':
            if classes(val) is None: raise ValueError('Neighbor classes must be specific classes.')
        elif kind == 'class':
            if not 0 <= int(val) <= 255: raise ValueError('Invalid '+key)
        elif kind.startswith('choice:'):
            if val not in kind.split(':',1)[1].split('|'): raise ValueError('Invalid '+key)
        else:
            number = int(val) if kind == 'int' else float(val)
            if not np.isfinite(number) or not low <= number <= high: raise ValueError('Invalid '+key)
    if op == 'Height' and params['min_height'] > params['max_height']: raise ValueError('Minimum height exceeds maximum height.')
    if op in GEOMETRY_OPS:
        if op in REQUIRED_GEOMETRY_OPS and not step.get('lines'):
            raise ValueError('Select geometry for '+op)
        for line in step.get('lines', []):
            coords = np.asarray(line.get('coords'),dtype=float)
            minimum = 3 if op in {'Fence', 'Close', 'Height'} else 2
            if coords.ndim != 2 or len(coords) < minimum or coords.shape[1] < 2 or not np.all(np.isfinite(coords)):
                raise ValueError('Invalid geometry coordinates.')

def save_macro(path,steps):
    if not steps: raise ValueError('Add an operation first.')
    for step in steps: validate_step(step)
    temporary = path+'.tmp'
    try:
        with open(temporary,'w',encoding='utf-8') as handle:
            json.dump({'format':FORMAT,'version':1,'steps':steps},handle,indent=2,allow_nan=False)
            handle.write('\n')
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.remove(temporary)

def load_macro(path):
    with open(path,'r',encoding='utf-8') as handle: payload = json.load(handle)
    if not isinstance(payload,dict) or payload.get('format') != FORMAT or payload.get('version') != 1: raise ValueError('Not a Naksha By Class macro.')
    steps = payload.get('steps')
    if not isinstance(steps,list) or not steps: raise ValueError('Macro has no operations.')
    for step in steps: validate_step(step)
    return steps

def run_macro(xyz,classification,steps,fence_mask=None,progress_cb=None,abort_check=None):
    if not steps: raise ValueError('Macro has no operations.')
    for step in steps: validate_step(step)
    xyz = np.asarray(xyz); cls = np.asarray(classification)
    if len(xyz) != len(cls): raise ValueError('Point and class counts differ.')
    original = cls.copy()
    eligible = np.ones(len(cls),bool) if fence_mask is None else np.asarray(fence_mask,dtype=bool)
    if len(eligible) != len(cls): raise ValueError('Invalid classification mask.')
    try:
        for index,step in enumerate(steps):
            if abort_check and abort_check(): raise InterruptedError('cancelled')
            op = step['operation']; source = classes(step['from_classes']); target = int(step['to_class']); params = dict(step['params'])
            if progress_cb: progress_cb(int(index*100/len(steps)),f'{index+1}/{len(steps)}: {op}')
            if op in {'Convert','Close','Height','Fence'}:
                mask = eligible & (cls != target)
                if source is not None: mask &= np.isin(cls,source)
                # Step-specific fences use world XY; they constrain output for
                # Fence, Close, and Height. Convert remains class-wide.
                step_polys = step.get('lines', []) if op != 'Convert' else []
                if op == 'Fence' or (op in {'Close','Height'} and step_polys):
                    candidates = np.flatnonzero(mask & np.all(np.isfinite(xyz[:,:2]),axis=1))
                    mask[:] = False
                    if len(candidates):
                        inside = np.zeros(len(candidates),dtype=bool)
                        for shape in step_polys:
                            inside |= _inside_polygon(xyz[candidates,:2],shape['coords'])
                        mask[candidates[inside]] = True
                if op == 'Close':
                    from scipy.spatial import cKDTree
                    finite = np.all(np.isfinite(xyz[:,:3]),axis=1)
                    refs_mask = (cls == target) & finite & eligible
                    if step_polys:
                        inside_refs = np.zeros(len(xyz),dtype=bool)
                        valid_refs = np.flatnonzero(refs_mask)
                        for shape in step_polys:
                            inside_refs[valid_refs] |= _inside_polygon(xyz[valid_refs,:2],shape['coords'])
                        refs_mask &= inside_refs
                    refs = xyz[refs_mask]; candidates = np.flatnonzero(mask & finite)
                    if len(candidates) and not len(refs):
                        raise ValueError(f'Step {index+1}: no To-class reference points in Close scope; refusing class-wide fallback.')
                    mask[:] = False
                    if len(candidates):
                        dist,_ = cKDTree(refs[:,:3]).query(xyz[candidates,:3],distance_upper_bound=params['radius'])
                        mask[candidates[dist < params['radius']]] = True
                elif op == 'Height':
                    finite = np.all(np.isfinite(xyz[:,:3]),axis=1)
                    refs = xyz[(cls == params['reference_class']) & finite]
                    if step_polys:
                        local_refs = np.zeros(len(refs), dtype=bool)
                        for shape in step_polys:
                            local_refs |= _inside_polygon(refs[:,:2],shape['coords'])
                        from gui.height_reference_surface import has_tin_xy_support
                        if has_tin_xy_support(refs[local_refs]):
                            refs = refs[local_refs]  # fallback if fence reference points cannot form TIN
                    candidates = np.flatnonzero(mask & finite)
                    if not len(refs): raise ValueError(f'Step {index+1}: reference class has no points.')
                    mask[:] = False
                    if len(candidates):
                        from gui.height_reference_surface import heights_above_reference_tin
                        heights,supported,_ = heights_above_reference_tin(refs,xyz[candidates],params['max_triangle'])
                        mask[candidates[supported&(heights >= params['min_height'])&(heights <= params['max_height'])]] = True
                cls[mask] = target
                continue
            kwargs = dict(params)
            if op == 'Isolated': kwargs['in_classes'] = classes(kwargs['in_classes'])
            if op in LINE_OPS:
                coords = [np.asarray(line['coords'],dtype=float) for line in step['lines']]
                kwargs['alignment_coords' if op == 'Find Wires' else 'centerline_coords'] = coords
            classifier = {'Low Points':classify_low_points,'Isolated':classify_isolated_points,'Ground':classify_ground_ptd,'Surface Points':classify_surface_points,'Below Surface':classify_below_surface,'Find Wires':classify_find_wires,'Classify by Centerline':classify_poles_by_centerline}[op]
            result = classifier(xyz,cls,source,target,fence_mask=eligible,progress_cb=None,abort_check=abort_check,**kwargs)
            reason = str(result.get('stop_reason','')).lower()
            if reason in {'cancelled','canceled','aborted'}: raise InterruptedError('cancelled')
            if op in LINE_OPS and reason and reason not in {'no candidates','no points'}: raise ValueError(f'Step {index+1}: {reason}')
        if abort_check and abort_check(): raise InterruptedError('cancelled')
    except InterruptedError:
        cls[:] = original
        return {'routine':'by_class_macro','changed':0,'indices':np.empty(0,dtype=np.intp),'stop_reason':'cancelled'}
    except Exception:
        cls[:] = original
        raise
    indices = np.flatnonzero(cls != original)
    if progress_cb: progress_cb(100,'Macro complete')
    return {'routine':'by_class_macro','changed':len(indices),'indices':indices}

class ByClassMacroDialog(_BaseClassifyDialog):
    """Non-modal sequence editor; each row is a separate operation instance."""
    _keep_visible_during_run = True

    def __init__(self, app, parent=None):
        super().__init__(app, 'By Class Macro', parent)
        self.setWindowModality(Qt.NonModal)
        self.steps = []
        self.path = None
        self._loading = False
        self._geometry_actors = []
        self._build_ui()
        self.resize(780, 620)

    def _build_ui(self):
        root = self._content_layout
        toolbar = QHBoxLayout()
        self.operation = QComboBox()
        self.operation.addItems(FIELDS)
        toolbar.addWidget(self.operation, 1)
        for text,slot in [('Add',self._add),('Open',self._open),('Save',self._save),('Save As',self._save_as)]:
            button = QPushButton(text); button.clicked.connect(slot); toolbar.addWidget(button)
        root.addLayout(toolbar)
        body = QHBoxLayout()
        left = QVBoxLayout()
        self.steps_list = QListWidget()
        self.steps_list.currentRowChanged.connect(self._show_step)
        left.addWidget(self.steps_list, 1)
        controls = QHBoxLayout()
        for text,slot in [('Up',lambda:self._move(-1)),('Down',lambda:self._move(1)),('Duplicate',self._duplicate),('Remove',self._remove)]:
            button = QPushButton(text); button.clicked.connect(slot); controls.addWidget(button)
        left.addLayout(controls)
        body.addLayout(left, 2)
        self.editor = QWidget()
        self.form = QFormLayout(self.editor)
        self.form.setLabelAlignment(Qt.AlignRight)
        body.addWidget(self.editor, 3)
        root.addLayout(body, 1)
        self.path_label = QLabel('Unsaved macro')
        root.addWidget(self.path_label)
        footer = QHBoxLayout()
        footer.addStretch()
        close = QPushButton('Close'); close.clicked.connect(self.close); footer.addWidget(close)
        run = QPushButton('Run Macro'); run.clicked.connect(self._run); footer.addWidget(run)
        root.addLayout(footer)
        self.widgets = {}
        self._show_step(-1)

    def _selected(self):
        row = self.steps_list.currentRow()
        return self.steps[row] if 0 <= row < len(self.steps) else None

    def _refresh_list(self, selected=None):
        if selected is None: selected = self.steps_list.currentRow()
        self.steps_list.blockSignals(True)
        self.steps_list.clear()
        for index,step in enumerate(self.steps,1):
            source = step['from_classes']
            label = 'any' if source is None else ','.join(map(str,source))
            self.steps_list.addItem(f"{index}. {step['operation']}   {label} -> {step['to_class']}")
        self.steps_list.blockSignals(False)
        self.steps_list.setCurrentRow(min(selected,len(self.steps)-1))
        self._show_step(self.steps_list.currentRow())

    def _add(self):
        self.steps.append(new_step(self.operation.currentText(), self.app))
        self._refresh_list(len(self.steps)-1)

    def _remove(self):
        row = self.steps_list.currentRow()
        if row >= 0:
            del self.steps[row]
            self._refresh_list(row)

    def _duplicate(self):
        row = self.steps_list.currentRow()
        if row >= 0:
            self.steps.insert(row+1,copy.deepcopy(self.steps[row]))
            self._refresh_list(row+1)

    def _move(self,offset):
        row = self.steps_list.currentRow(); other = row+offset
        if 0 <= row < len(self.steps) and 0 <= other < len(self.steps):
            self.steps[row],self.steps[other] = self.steps[other],self.steps[row]
            self._refresh_list(other)

    def _clear_geometry_highlights(self):
        digitizer = getattr(self.app, 'digitizer', None)
        if digitizer is not None:
            for actor in self._geometry_actors:
                try: digitizer._remove_actor_from_overlay(actor)
                except Exception: pass
        self._geometry_actors = []
        try: self.app.vtk_widget.render()
        except Exception: pass

    def _update_geometry_highlights(self):
        self._clear_geometry_highlights()
        if self._selected() is None or self._selected()['operation'] not in GEOMETRY_OPS:
            return
        for line in self._selected()['lines']:
            actor = _make_alignment_highlight_actor(self, line['coords'])
            if actor is not None: self._geometry_actors.append(actor)
        try: self.app.vtk_widget.render()
        except Exception: pass

    def closeEvent(self, event):
        self._clear_geometry_highlights()
        super().closeEvent(event)

    def _clear_form(self):
        self._clear_geometry_highlights()
        while self.form.rowCount(): self.form.removeRow(0)
        self.widgets = {}

    def _show_step(self,row):
        self._loading = True
        self._clear_form()
        step = self._selected()
        if step is None:
            self.form.addRow(QLabel('Select an operation to edit.'))
            self._loading = False
            return
        op = step['operation']
        title = QLabel(op); title.setStyleSheet('font-weight: bold; font-size: 14px;')
        self.form.addRow(title)
        source = MacroClassPicker(self.app, step['from_classes'], multi=True, allow_any=True)
        source.changed.connect(self._sync_step)
        self.form.addRow('From classes:', source)
        self.widgets['from_classes'] = source
        target = MacroClassPicker(self.app, step['to_class'])
        target.changed.connect(self._sync_step)
        self.form.addRow('To class:', target)
        self.widgets['to_class'] = target
        for key,label,kind,default,low,high in FIELDS[op]:
            value = step['params'].get(key,default)
            if kind == 'class':
                widget = MacroClassPicker(self.app, value)
                widget.changed.connect(self._sync_step)
            elif kind == 'classes':
                widget = MacroClassPicker(self.app, value, multi=True)
                widget.changed.connect(self._sync_step)
            elif kind == 'int':
                widget = QSpinBox(); widget.setRange(low,high); widget.setValue(int(value)); widget.valueChanged.connect(self._sync_step)
            elif kind == 'float':
                widget = QDoubleSpinBox(); widget.setDecimals(3); widget.setRange(low,high); widget.setValue(float(value)); widget.valueChanged.connect(self._sync_step)
            elif kind.startswith('choice:'):
                widget = QComboBox(); widget.addItems(kind.split(':',1)[1].split('|')); widget.setCurrentText(value); widget.currentTextChanged.connect(self._sync_step)
            self.form.addRow(label+':',widget); self.widgets[key] = widget
        if op in GEOMETRY_OPS:
            self.lines_list = QListWidget()
            self.lines_list.setSelectionMode(QAbstractItemView.MultiSelection)
            self.lines_list.setMinimumHeight(110)
            selected = {tuple(np.round(np.asarray(line['coords'])[:,:2].ravel(),3)) for line in step['lines']}
            self.line_sources = (_collect_fence_sources(self.app) if op in {'Fence','Close','Height'}
                                 else _collect_alignment_sources(self.app))
            for source in self.line_sources:
                self.lines_list.addItem(source['label'])
                if source['key'] in selected: self.lines_list.item(self.lines_list.count()-1).setSelected(True)
            self.lines_list.itemSelectionChanged.connect(self._geometry_selection_changed)
            self.form.addRow('Drawn fence:' if op in {'Fence','Close','Height'} else 'SNT / drawn CL:',self.lines_list)
            actions = QWidget(); actions_row = QHBoxLayout(actions); actions_row.setContentsMargins(0,0,0,0)
            refresh = QPushButton('Refresh'); refresh.clicked.connect(lambda: self._show_step(self.steps_list.currentRow()))
            clear = QPushButton('Clear selection'); clear.clicked.connect(self._clear_geometry_selection)
            actions_row.addWidget(refresh); actions_row.addWidget(clear)
            self.form.addRow(actions)
            self.form.addRow(QLabel(f"{len(step['lines'])} saved shape(s)"))
        self._loading = False
        self._update_geometry_highlights()

    def _clear_geometry_selection(self):
        step = self._selected()
        if step is None: return
        step['lines'] = []
        self.lines_list.clearSelection()
        self._update_geometry_highlights()

    def _geometry_selection_changed(self):
        self._sync_step()
        self._update_geometry_highlights()

    def _sync_step(self):
        if self._loading: return
        step = self._selected()
        if step is None: return
        source = self.widgets['from_classes'].value()
        step['from_classes'] = source
        step['to_class'] = self.widgets['to_class'].value()
        for key,_,kind,_,_,_ in FIELDS[step['operation']]:
            widget = self.widgets[key]
            if kind.startswith('choice:'):
                step['params'][key] = widget.currentText()
            else:
                step['params'][key] = widget.value()
        if step['operation'] in GEOMETRY_OPS:
            available = {source['key'] for source in self.line_sources}
            unmatched = [line for line in step['lines'] if tuple(np.round(np.asarray(line['coords'])[:,:2].ravel(),3)) not in available]
            step['lines'] = unmatched + [{'label':source['label'],'coords':np.asarray(source['coords']).tolist()}
                             for i,source in enumerate(self.line_sources) if self.lines_list.item(i).isSelected()]
        row = self.steps_list.currentRow()
        self.steps_list.item(row).setText(f"{row+1}. {step['operation']}   {('any' if source is None else ','.join(map(str,source)))} -> {step['to_class']}")
        return True

    def _open(self):
        path,_ = QFileDialog.getOpenFileName(self,'Open By Class Macro','','Naksha macros (*.naksha-macro.json);;JSON (*.json)')
        if not path: return
        try:
            loaded = load_macro(path)
        except Exception as exc:
            QMessageBox.warning(self,'Cannot Open Macro',str(exc)); return
        self.steps = loaded; self.path = path; self.path_label.setText(path); self._refresh_list(0)

    def _save_as(self):
        previous = self.path
        self.path = None
        self._save()
        if self.path is None: self.path = previous

    def _save(self):
        self._sync_step()
        path = self.path
        if not path:
            path,_ = QFileDialog.getSaveFileName(self,'Save By Class Macro','','Naksha macros (*.naksha-macro.json)')
            if not path: return
            if not path.lower().endswith('.json'): path += '.naksha-macro.json'
        try: save_macro(path,self.steps)
        except Exception as exc:
            QMessageBox.warning(self,'Cannot Save Macro',str(exc)); return
        self.path = path; self.path_label.setText(path)

    def _run(self):
        self._sync_step()
        if not self.steps:
            QMessageBox.warning(self,'Empty Macro','Add an operation first.'); return
        try:
            for step in self.steps: validate_step(step)
        except Exception as exc:
            QMessageBox.warning(self,'Invalid Macro',str(exc)); return
        self._run_algorithm(run_macro,{'steps':copy.deepcopy(self.steps)},'By Class Macro')
