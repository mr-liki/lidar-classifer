"""Native selection/Intersect tests, run without a Vulkan GPU or PySide6.

The Qt stub only supplies QObject/signals/input constants; all source geometry
and production controller logic is imported from the actual patched GUI code.
"""
import importlib
import math
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / 'gui').is_dir():  # development bundle layout
    ROOT = ROOT / 'patched'
sys.path.insert(0, str(ROOT))

import pytest
from gui.native_intersect_geometry import split_selected_features, MAX_INTERSECT_SEGMENTS


def feature(name, coords, source='native', typ='line'):
    drawing={'type':typ, 'coords':list(coords), 'name':name}
    return {'source':source,'type':typ,'coords':list(coords),
            'drawing':drawing,'key':('native',id(drawing))}


def test_two_crossing_lines_make_four_segments_and_do_not_modify_inputs():
    one = feature('horizontal', [(0,0),(10,0)])
    two = feature('vertical', [(5,-5),(5,5)])
    original=[list(one['drawing']['coords']),list(two['drawing']['coords'])]
    parts,n=split_selected_features([one,two])
    assert n==1
    assert len(parts)==4
    assert [(p['coords'][0],p['coords'][-1]) for p in parts]==[
        ((0,0),(5,0)),((5,0),(10,0)),((5,-5),(5,0)),((5,0),(5,5))]
    assert [one['drawing']['coords'],two['drawing']['coords']]==original


def test_three_crossings_and_closed_polygon():
    across=feature('horizontal', [(0,0),(10,0)])
    cross1=feature('vertical1',[(3,-5),(3,5)])
    cross2=feature('vertical2',[(7,-5),(7,5)])
    parts,n=split_selected_features([across,cross1,cross2])
    assert n==2
    assert [p['coords'] for p in parts if p['source'] is across]==[
        [(0,0),(3,0)],[(3,0),(7,0)],[(7,0),(10,0)]]
    box=feature('box',[(0,0),(10,0),(10,10),(0,10),(0,0)],typ='polyline')
    cutter=feature('cutter',[(5,-1),(5,11)])
    parts,n=split_selected_features([box,cutter])
    assert n==2
    assert len([p for p in parts if p['source'] is box])==3


def test_endpoint_crossing_keeps_only_nonzero_parts():
    one=feature('ending',[(0,0),(5,0)])
    two=feature('vertical',[(5,-5),(5,5)])
    parts,n=split_selected_features([one,two])
    assert n==1
    assert len([p for p in parts if p['source'] is one])==1
    assert len([p for p in parts if p['source'] is two])==2
    assert all(len(p['coords'])>=2 and p['coords'][0]!=p['coords'][-1] for p in parts)


def test_unsafe_collinear_overlap_is_not_modified():
    one=feature('one',[(0,0),(10,0)])
    two=feature('two',[(5,0),(15,0)])
    parts,n=split_selected_features([one,two])
    assert n==0 and len(parts)==2
    assert parts[0]['coords']==[(0,0),(10,0)]


def test_readonly_snt_can_cut_but_never_be_modified():
    one=feature('native',[(0,0),(10,0)])
    snt=feature('imported',[(5,-1),(5,1)],source='snt')
    original=snt['drawing'].copy()
    parts,n=split_selected_features([one,snt])
    assert n==1 and len(parts)==2
    assert all(p['source'] is one for p in parts)
    assert snt['drawing']==original
    assert split_selected_features([snt,snt])==([],0)


def test_circle_tessellation_is_not_an_intersect_source():
    one=feature('circle',[(0,0),(1,1)],typ='circle')
    two=feature('line',[(0,-1),(0,1)])
    parts,n=split_selected_features([one,two])
    assert n==0 and len(parts)==1
    assert parts[0]['source'] is two


def test_large_complexity_rejected_without_computation():
    huge=feature('huge',[(float(x),0.) for x in range(MAX_INTERSECT_SEGMENTS+2)],typ='polyline')
    with pytest.raises(ValueError, match='too complex'):
        split_selected_features([huge,feature('other',[(0,-1),(0,1)])])


@pytest.fixture(scope='module')
def runtime():
    # Stub only GUI IO. The controller and native drawing geometry are real.
    qt=types.ModuleType('PySide6.QtCore')
    qtw=types.ModuleType('PySide6.QtWidgets')
    class Qt:
        LeftButton=1;RightButton=2;MiddleButton=4;NoButton=0
        ControlModifier=0x01;ShiftModifier=0x02
        Key_Escape=27;Key_Enter=13;Key_Return=10
    qt.Qt=Qt
    class QObject:
        def __init__(self,*a,**kw):pass
    qt.QObject=QObject
    class Signal:
        def __init__(self,*a): self.history=[]
        def emit(self,*a):self.history.append(a)
    qt.Signal=Signal
    class QApplication:
        modifiers=0
        @staticmethod
        def keyboardModifiers():return QApplication.modifiers
    qtw.QApplication=QApplication
    qtw.QInputDialog=type('QInputDialog',(),{})
    sys.modules['PySide6']=types.ModuleType('PySide6')
    sys.modules['PySide6.QtCore']=qt
    sys.modules['PySide6.QtWidgets']=qtw
    main=types.ModuleType('gui.native_main_classify')
    main.MainCanvas=type('MainCanvas',(),{})
    sys.modules['gui.native_main_classify']=main
    overlay=types.ModuleType('gui.native_section_classify')
    overlay.CanvasOverlay=type('CanvasOverlay',(),{})
    sys.modules['gui.native_section_classify']=overlay
    bridge=importlib.import_module('gui.native_selection_bridge')
    draw_module=importlib.import_module('gui.native_draw')
    return Qt,QApplication,bridge,draw_module


class Pos:
    def __init__(self,x,y):self._x=x;self._y=y
    def x(self):return self._x
    def y(self):return self._y


class Canvas:
    def section_at(self,x,y):return (float(x),float(y))
    def world_per_pixel(self):return .1


class Status:
    def __init__(self): self.messages=[]
    def showMessage(self,msg,*args):self.messages.append(msg)


class App:
    def __init__(self):self.bar=Status()
    def statusBar(self):return self.bar


class DrawStub:
    def __init__(self,drawings):
        self.drawings=drawings
        self.app=App()
        self.canvas=Canvas()
        self.tool='select'
        self._selection_bridge=None
        self.actions=[]
        self.paints=0
    def _scene_token(self):return ('same',)
    def _native_coords(self,d):return d['coords']
    def _paint(self):self.paints+=1
    def record_object_change(self,before):self.actions.append((list(before),list(self.drawings)))


def make_controller(bridge,draw):
    tool=bridge.NativeSelectionController(draw)
    tool.activate('individual')
    tool.selection_manager.apply([tool._native_feature(i,obj) for i,obj in enumerate(draw.drawings)])
    return tool


def test_intersect_select_delete_undo_redo_and_atomic_commit(runtime):
    Qt,AppQt,bridge,_=runtime
    left={'type':'line','coords':[(0.,0.),(10.,0.)],'color':'#123456'}
    cutter={'type':'line','coords':[(5.,-5.),(5.,5.)],'color':'#789abc'}
    draw=DrawStub([left,cutter]);tool=make_controller(bridge,draw)
    assert tool.enter_intersect_mode()
    assert tool._intersect_mode and len(tool._intersect_parts)==4
    assert draw.drawings==[left,cutter]
    assert tool.press(Pos(2.,0.),Qt.LeftButton)
    chosen=tool._intersect_selected_part
    assert chosen is not None
    tool.press(Pos(2.,0.),Qt.RightButton)
    assert tool._intersect_parts[chosen]['deleted']
    assert tool.intersect_history(redo=False)
    assert not tool._intersect_parts[chosen]['deleted']
    assert tool.intersect_history(redo=True)
    assert tool._intersect_parts[chosen]['deleted']
    assert tool.exit_intersect_mode(commit=True)
    assert not tool._intersect_mode
    assert len(draw.actions)==1
    assert len(draw.drawings)==2
    assert draw.drawings[0]['coords']==[(5.,0.),(10.,0.)]
    assert draw.drawings[1] is cutter
    assert draw.drawings[0]['color']=='#123456'
    before, after = draw.actions[0]
    assert before[0] is left and before[1] is cutter
    assert after[0] is draw.drawings[0]


def test_intersect_escape_preserves_everything(runtime):
    Qt,AppQt,bridge,_=runtime
    a={'type':'line','coords':[(0,0),(10,0)]}
    b={'type':'line','coords':[(5,-5),(5,5)]}
    draw=DrawStub([a,b]);tool=make_controller(bridge,draw)
    assert tool.enter_intersect_mode()
    tool.press(Pos(2,0),Qt.LeftButton)
    tool.press(Pos(2,0),Qt.RightButton)
    assert tool.key(Qt.Key_Escape)
    assert not tool._intersect_mode
    assert draw.drawings==[a,b] and not draw.actions


def test_ctrl_shift_selection_inverts(runtime):
    Qt,AppQt,bridge,_=runtime
    AppQt.modifiers=Qt.ControlModifier | Qt.ShiftModifier
    assert bridge.NativeSelectionController._selection_modifier()=='invert'
    AppQt.modifiers=Qt.ControlModifier
    assert bridge.NativeSelectionController._selection_modifier()=='add'
    AppQt.modifiers=Qt.ShiftModifier
    assert bridge.NativeSelectionController._selection_modifier()=='subtract'
    AppQt.modifiers=0


def test_native_rectangle_four_corner_pick_and_move(runtime):
    Qt,AppQt,bridge,module=runtime
    draw=module.NativeDraw.__new__(module.NativeDraw)
    rect={'type':'rectangle','coords':[(0.,0.),(10.,10.)]}
    circle={'type':'circle','coords':[(30.,30.),(35.,30.)]}
    draw.drawings=[rect,circle]
    draw.canvas=Canvas()
    draw._paint=lambda:None
    draw.record_object_change=lambda before:None
    hit=draw._nearest_vertex((10.,0.))
    assert hit and hit[1:]==(0,1)
    assert draw._nearest_segment((10.,5.))[1:3]==(0,1)
    draw.vertex_target=(0,1)
    draw._vertex_move_original=rect
    draw._vertex_move_before=list(draw.drawings)
    assert draw._vertex_move_preview((12.,-2.))
    new=draw.drawings[0]
    assert new['type']=='rectangle'
    assert new['coords']==[(12.,-2.),(0.,10.)]
    draw._vertex_move_finish((12.,-2.))
    assert draw.vertex_target is None
    assert draw._nearest_vertex((35.,30.)) is None
    assert draw._nearest_segment((35.,30.)) is None


def test_rectangle_add_delete_are_closed_polyline_and_circle_rejected(runtime):
    Qt,AppQt,bridge,module=runtime
    draw=module.NativeDraw.__new__(module.NativeDraw)
    draw.drawings=[{'type':'rectangle','coords':[(0.,0.),(10.,10.)]}]
    draw.canvas=Canvas();draw._paint=lambda:None
    draw._point=lambda pos:(pos.x(),pos.y())
    draw._last_pointer_pos=None
    draw.active=lambda:True
    draw.app=App()
    draw.record_object_change=lambda before:None
    draw._nearest_vertex=lambda p:None
    draw.tool='vertex'
    draw._legacy_tool_dialog=None
    draw._selection_bridge=None
    draw.vertex_target=None;draw._vertex_insert_pending=False
    draw.press(Pos(5.,0.),Qt.LeftButton)
    item=draw.drawings[0]
    assert item['type']=='polyline'
    assert item['coords'][0]==item['coords'][-1]
    assert (5.,0.) in item['coords']
    draw.tool='deletevertex'
    draw._nearest_vertex=lambda p:(0,0,1)
    draw.press(Pos(5.,0.),Qt.LeftButton)
    item=draw.drawings[0]
    assert item['coords'][0]==item['coords'][-1]
    assert len(item['coords'])==5  # closed quadrilateral after removing inserted vertex


def test_freehand_start_and_end_snap_but_not_middle(runtime):
    Qt,AppQt,bridge,module=runtime
    draw=module.NativeDraw.__new__(module.NativeDraw)
    draw.canvas=Canvas();draw.app=App()
    draw.drawings=[{'type':'line','coords':[(0,0),(10,0)]}]
    draw._near_scene=lambda p,tol:[{'source':'native','type':'line','coords':[(0.,0.),(10.,0.)],
                                  'drawing':draw.drawings[0],'key':('native',0)}]
    draw.tool='freehand';draw.points=[]
    draw._snap_target=None
    p=draw._point(Pos(.4,.4))
    assert p==(0.,0.) or p==(.4,0.)
    assert draw._snap_target is not None
    raw=draw._point(Pos(.4,.4),snap=False)
    assert raw==(.4,.4)
    assert draw._snap_target is None


def test_selection_fences_operate_on_existing_shapes_only():
    from gui.native_selection_geometry import rect_selects, polygon_selects, line_selects
    box=[(2.,2.),(8.,2.),(8.,8.),(2.,8.),(2.,2.)]
    inside_line=[(3.,3.),(7.,3.)]
    crossing_line=[(-1.,5.),(11.,5.)]
    assert rect_selects(inside_line,(0,0),(10,10),inside=True)
    assert not rect_selects(crossing_line,(0,0),(10,10),inside=True)
    assert rect_selects(crossing_line,(0,0),(10,10),inside=False)
    assert polygon_selects(inside_line,box,inside=True)
    assert not polygon_selects(crossing_line,box,inside=True)
    assert polygon_selects(crossing_line,box,inside=False)
    assert line_selects(crossing_line,(5,0),(5,10))
    assert not line_selects(inside_line,(5,5),(7,5))


def test_intersect_no_delete_keeps_original_identity(runtime):
    Qt,AppQt,bridge,_=runtime
    a={'type':'line','coords':[(0,0),(10,0)]}
    b={'type':'line','coords':[(5,-5),(5,5)]}
    draw=DrawStub([a,b]);tool=make_controller(bridge,draw)
    assert tool.enter_intersect_mode()
    assert tool.exit_intersect_mode(commit=True)
    assert draw.drawings[0] is a and draw.drawings[1] is b
    assert not draw.actions


def test_intersect_changed_scene_discards_pending_deletions(runtime):
    Qt,AppQt,bridge,_=runtime
    a={'type':'line','coords':[(0,0),(10,0)]}
    b={'type':'line','coords':[(5,-5),(5,5)]}
    draw=DrawStub([a,b]);tool=make_controller(bridge,draw)
    assert tool.enter_intersect_mode()
    tool.press(Pos(2,0),Qt.LeftButton)
    tool.press(Pos(2,0),Qt.RightButton)
    draw._scene_token=lambda:('changed_snt',)
    assert not tool.exit_intersect_mode(commit=True)
    assert draw.drawings==[a,b] and not draw.actions


def test_imported_snt_line_not_added_to_native_editing_selection(runtime):
    Qt,AppQt,bridge,_=runtime
    a={'type':'line','coords':[(0.,0.),(10.,0.)]}
    draw=DrawStub([a]);tool=make_controller(bridge,draw)
    cutter={'type':'line','coords':[(5.,-5.),(5.,5.)]}
    imported={'source':'snt','type':'line','coords':cutter['coords'],
              'doc_key':('snt','doc'), 'layer':'survey',
              'scene_token':draw._scene_token(), 'key':('snt',0)}
    draw._snt_visible=lambda *args:True
    tool.selection_manager.apply([imported],mode='add')
    # Stage 2 user scope: imported SNT geometry is NOT a Draw/Measure target.
    assert tool.selection_manager.count()==1
    assert all(f['source']=='native' for f in tool.selection_manager.get_features())
    assert tool.enter_intersect_mode()
    assert tool.exit_intersect_mode(commit=False)
    assert draw.drawings == [a]
    assert cutter['coords']==[(5.,-5.),(5.,5.)]


def test_native_undo_restores_intersect_source_objects(runtime):
    Qt,AppQt,bridge,module=runtime
    draw=module.NativeDraw.__new__(module.NativeDraw)
    draw.canvas=Canvas();draw.app=App()
    a={'type':'line','coords':[(0.,0.),(10.,0.)]}
    b={'type':'line','coords':[(5.,-5.),(5.,5.)]}
    draw.drawings=[a,b]
    draw.tool='select';draw.active=lambda:True
    draw._paint=lambda:None;draw._scene_token=lambda:('same',)
    draw._object_actions=[];draw._object_redo=[]
    draw._selection_bridge=None
    draw.points=[];draw.point_redo=[];draw.redo=[];draw.vertex_target=None
    draw.record_object_change=module.NativeDraw.record_object_change.__get__(draw)
    ctrl=bridge.NativeSelectionController(draw)
    ctrl.activate('individual')
    ctrl.selection_manager.apply([ctrl._native_feature(i,obj) for i,obj in enumerate(draw.drawings)])
    assert ctrl.enter_intersect_mode()
    ctrl.press(Pos(2,0),Qt.LeftButton)
    ctrl.press(Pos(2,0),Qt.RightButton)
    # While the session is armed, document-level Undo must only restore a part.
    assert draw.history(redo=False)
    assert not any(p['deleted'] for p in ctrl._intersect_parts)
    assert draw.history(redo=True)
    assert any(p['deleted'] for p in ctrl._intersect_parts)
    ctrl.exit_intersect_mode(commit=True)
    assert draw.drawings[0] is not a
    assert draw.history(redo=False)
    assert draw.drawings[0] is a and draw.drawings[1] is b
    assert draw.history(redo=True)
    assert draw.drawings[0]['coords']==[(5.,0.),(10.,0.)]


def test_rectangle_drag_on_fourth_corner_no_index_error(runtime):
    Qt,AppQt,bridge,module=runtime
    draw=module.NativeDraw.__new__(module.NativeDraw)
    original={'type':'rectangle','coords':[(0.,0.),(10.,10.)]}
    draw.drawings=[original];draw.canvas=Canvas();draw.app=App()
    draw._paint=lambda:None
    draw._point=lambda pos, snap=True:(pos.x(),pos.y())
    draw.active=lambda:True
    draw.tool='movevertex';draw._legacy_tool_dialog=None
    draw._selection_bridge=None
    draw.vertex_target=(0,3)
    draw._vertex_move_original=original
    draw._vertex_move_before=list(draw.drawings)
    draw._vertex_move_dragged=False
    draw._last_pointer_pos=None
    draw.move(Pos(-2.,8.),buttons=Qt.LeftButton)
    assert draw.drawings[0]['type']=='rectangle'
    assert draw.drawings[0]['coords']==[(-2.,8.),(10.,0.)]


def test_freehand_closes_region_even_when_release_is_far(runtime):
    Qt,AppQt,bridge,module=runtime
    draw=module.NativeDraw.__new__(module.NativeDraw)
    draw.canvas=Canvas();draw.app=App()
    draw._paint=lambda:None
    draw.tool='freehand'
    draw._commit=lambda kind,points: (kind,list(points))
    draw.points=[(0.,0.),(4.,4.),(.1,.1)]
    assert draw._finish_freehand()==('freehand',[(0.,0.),(4.,4.),(0.,0.)])
    draw.points=[(0.,0.),(4.,4.),(2.,1.)]
    assert draw._finish_freehand()==('freehand',[(0.,0.),(4.,4.),(2.,1.),(0.,0.)])


def test_vulkan_overlay_hides_original_intersect_sources(runtime):
    Qt,AppQt,bridge,module=runtime
    draw=module.NativeDraw.__new__(module.NativeDraw)
    draw.canvas=Canvas();draw.app=App()
    a={'type':'line','coords':[(0.,0.),(10.,0.)]}
    b={'type':'line','coords':[(5.,-5.),(5.,5.)]}
    draw.drawings=[a,b]
    draw.tool='select';draw.active=lambda:True
    draw._scene_token=lambda:('same',)
    draw._style=lambda name:{'color':'#123456','width':2}
    draw._sync_block_vectors=lambda:False
    draw._selected_imported=None;draw._legacy_tool_dialog=None
    draw._snap_target=None;draw._parallel_source=None
    draw._last_pointer_pos=None;draw.aux_shapes=[];draw.points=[]
    draw._shutting_down=False
    class Overlay:
        def set_shapes(self,shapes):self.shapes=shapes
    draw.overlay=Overlay()
    ctrl=bridge.NativeSelectionController(draw)
    ctrl.activate('individual')
    ctrl.selection_manager.apply([ctrl._native_feature(i,obj) for i,obj in enumerate(draw.drawings)])
    assert ctrl.enter_intersect_mode()
    draw._paint()
    assert len(draw.overlay.shapes)==4
    assert all(shape[0]=='drawing' for shape in draw.overlay.shapes)
    assert all(len(shape[1])==2 for shape in draw.overlay.shapes)
    ctrl.exit_intersect_mode(commit=False)
    draw._paint()
    assert len(draw.overlay.shapes)==2


def test_freehand_right_click_finishes_current_stroke(runtime):
    Qt,AppQt,bridge,module=runtime
    draw=module.NativeDraw.__new__(module.NativeDraw)
    draw.canvas=Canvas();draw.app=App()
    draw._paint=lambda:None
    draw.active=lambda:True
    draw._legacy_tool_dialog=None
    draw._selection_bridge=None
    draw.tool='freehand';draw.points=[(0.,0.),(4.,4.)]
    done=[]
    draw._commit=lambda kind,points:done.append((kind, list(points))) or True
    assert draw.press(Pos(2.,2.),Qt.RightButton)
    assert done==[('freehand',[(0.,0.),(4.,4.)])]


def test_accudraw_xyz_is_kept_readonly_until_3d_split_is_supported():
    xyz=feature('xyz',[(0,0),(10,0)],typ='accudraw')
    xyz['drawing']['xyz_coords']=[(0.,0.,10.),(10.,0.,20.)]
    crossing=feature('crossing',[(5,-5),(5,5)])
    parts,n=split_selected_features([xyz,crossing])
    assert n==1
    assert all(p['source'] is crossing for p in parts)
    assert xyz['drawing']['xyz_coords']==[(0.,0.,10.),(10.,0.,20.)]


def test_vtk_intersect_can_remove_single_non_crossing_line(runtime):
    Qt,AppQt,bridge,_=runtime
    a={'type':'line','coords':[(0.,0.),(10.,0.)]}
    draw=DrawStub([a]);tool=make_controller(bridge,draw)
    assert tool.enter_intersect_mode()
    assert len(tool._intersect_parts)==1
    tool.press(Pos(2,0),Qt.LeftButton)
    tool.press(Pos(2,0),Qt.RightButton)
    assert tool.exit_intersect_mode(commit=True)
    assert draw.drawings==[]
    assert len(draw.actions)==1


def test_rectangle_release_when_world_pick_unavailable_preserves_state(runtime):
    Qt,AppQt,bridge,module=runtime
    draw=module.NativeDraw.__new__(module.NativeDraw)
    original={'type':'rectangle','coords':[(0.,0.),(10.,10.)]}
    draw.drawings=[original];draw.canvas=Canvas();draw.app=App()
    draw._paint=lambda:None
    draw._point=lambda pos: (_ for _ in ()).throw(RuntimeError('lost viewport'))
    draw.active=lambda:True
    draw.tool='movevertex';draw._selection_bridge=None
    draw.vertex_target=(0,3)
    draw._vertex_move_original=original
    draw._vertex_move_before=list(draw.drawings)
    draw._vertex_move_dragged=True
    draw.record_object_change=lambda before:None
    assert draw.release(Pos(0,0),Qt.LeftButton)
    assert draw.drawings[0]['type']=='rectangle'
    assert draw.vertex_target is None
