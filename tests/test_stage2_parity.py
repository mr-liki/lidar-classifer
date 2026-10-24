"""Stage 2 acceptance tests: closure, scoped selection, and SNT Text rectangle.

These load production controller logic using the same lightweight Qt stub as
Stage 1. GUI/GPU gestures still require Windows interactive acceptance.
"""
import math
import sys
from types import SimpleNamespace

import pytest
from test_stage1_parity import runtime as stage1_runtime, App, Canvas, DrawStub, Pos
from gui.native_selection_targets import (
    editable_draw, snap_draft_closure, close_freehand, text_anchor, text_bounds, box_overlap,
)


@pytest.fixture(scope='module')
def runtime():
    # Stage 1 and Stage 2 must use the SAME Qt stubs and loaded production
    # modules when both test files run in one pytest process.
    if 'gui.native_selection_bridge' not in sys.modules:
        return stage1_runtime.__wrapped__()
    bridge=sys.modules['gui.native_selection_bridge']
    return bridge.Qt,bridge.QApplication,bridge,sys.modules['gui.native_draw']


def prepared_draw(drawings=(), measurement=None, attachments=()):
    draw=DrawStub(list(drawings))
    draw.view=SimpleNamespace(measurement=measurement)
    draw.app.snt_attachments=list(attachments)
    draw._snt_visible=lambda key,layer:True
    draw._scene_token=lambda:tuple((id(x),id(x.get('_render_entities')),len(x.get('_render_entities',())))
                                    for x in draw.app.snt_attachments)
    draw._feature_distance=lambda p,feat: min((math.dist(p,q) for q in feat['coords']), default=math.inf) \
        if len(feat['coords'])==1 else min(
            __import__('gui.native_vector_pick',fromlist=['segment_projection']).segment_projection(p,a,b)[0]
            for a,b in zip(feat['coords'],feat['coords'][1:]))
    return draw


def controller(bridge,draw,method='individual'):
    instance=bridge.NativeSelectionController(draw)
    assert instance.activate(method)
    return instance


def test_closure_priority_requires_three_controls_and_nearby():
    pts=[(0.,0.),(4.,0.),(5.,5.)]
    assert snap_draft_closure(pts,(0.1,-0.1),.4)==(0.,0.)
    assert snap_draft_closure(pts,(2.,2.),.4)==(2.,2.)
    assert snap_draft_closure(pts[:2],(0.01,0.),.4)==(0.01,0.)


def test_closure_keeps_3d_z_coordinate_and_candidate_type():
    pts=[[0.,0.,23.],[4.,0.,26.],[6.,3.,24.]]
    assert snap_draft_closure(pts,[.01,.01,5.],.5)==pts[0]
    assert isinstance(snap_draft_closure(pts,[.01,.01,5.],.5),list)
    assert snap_draft_closure(pts,[.01,.01,5.],float('inf'))!=pts[0]


def test_freehand_auto_closes_far_release_and_preserves_last_sample():
    source=[(1.,1.),(5.,3.),(5.,8.),(14.,12.)]
    result=close_freehand(source,.2)
    assert result==source+[source[0]]
    assert source[-1]==(14.,12.)
    assert result[0] is result[-1]


def test_freehand_near_start_uses_exact_start_and_avoids_duplicate():
    source=[(1.,1.),(5.,3.),(5.,8.),(1.01,.99)]
    result=close_freehand(source,.2)
    assert len(result)==len(source) and result[-1]==source[0]


def test_freehand_avoids_degenerate_two_point_ring():
    assert close_freehand([(0,0),(2,3)],1.)==[(0,0),(2,3)]
    assert close_freehand([(0,0),(2,3),(0,0)],1.)==[(0,0),(2,3),(0,0)]


def test_edit_scope_excludes_authoritative_block_and_cell_geometry():
    assert editable_draw({'type':'line'})
    assert not editable_draw({'type':'polyline','source':'block_creation'})
    assert not editable_draw({'type':'circle','source':'cell_creation'})


def test_rectangle_and_shape_fences_ignore_snt_and_created_block(runtime):
    Qt,AppQt,bridge,_=runtime
    block={'type':'polyline','coords':[(1,1),(5,1),(5,5),(1,5),(1,1)],
           'source':'block_creation'}
    line={'type':'line','coords':[(2,2),(3,3)]}
    draw=prepared_draw([block,line])
    # Any attempt to use native SNT spatial index for normal editing is a bug.
    draw._snt_index=lambda:pytest.fail('Imported SNT index accessed by Block Selection')
    tool=controller(bridge,draw,'block')
    tool._select_area('block',[(0,0),(8,8)])
    assert tool.selection_manager.count()==1
    assert tool.selection_manager.get_features()[0]['drawing'] is line
    tool.set_method('shape')
    tool._select_area('shape',[(0,0),(8,0),(8,8),(0,8)])
    assert tool.selection_manager.count()==1


def test_individual_picker_rejects_imported_snt(runtime):
    Qt,AppQt,bridge,_=runtime
    draw=prepared_draw([{'type':'line','coords':[(0,0),(2,0)]}])
    draw._snt_index=lambda:pytest.fail('Do not touch SNT while picking edit targets')
    tool=controller(bridge,draw,'individual')
    assert tool.press(Pos(1,0),Qt.LeftButton)
    assert tool.selection_manager.count()==1
    assert tool.selection_manager.get_features()[0]['source']=='native'


def test_measure_line_and_path_are_selectable_not_grid(runtime):
    Qt,AppQt,bridge,_=runtime
    measure=SimpleNamespace(completed=[[(1.,1.),(4.,1.)],[(2.,2.),(4.,4.),(5.,4.)]],
                            completed_modes=[False,True],grids=[{'points':[(0,0)]}])
    draw=prepared_draw([],measurement=measure)
    tool=controller(bridge,draw,'block')
    tool._select_area('block',[(0,0),(7,7)])
    features=tool.selection_manager.get_features()
    assert len(features)==2
    assert [f['type'] for f in features]==['measure_line','measure_path']
    assert all(f['measure'] is measure for f in features)


def test_selected_measurement_is_pruned_after_clear(runtime):
    Qt,AppQt,bridge,_=runtime
    measure=SimpleNamespace(completed=[[(1.,1.),(4.,1.)]],completed_modes=[False])
    draw=prepared_draw([],measurement=measure)
    tool=controller(bridge,draw,'block')
    tool._select_area('block',[(0,0),(7,7)])
    assert tool.selection_manager.count()==1
    measure.completed.clear()
    assert tool.selection_manager.count()==0


def test_select_all_excludes_snt_and_created_blocks(runtime):
    Qt,AppQt,bridge,_=runtime
    line={'type':'line','coords':[(1,1),(2,1)]}
    protected={'type':'polyline','source':'block_creation','coords':[(0,0),(2,2)]}
    measure=SimpleNamespace(completed=[[(1,3),(4,3)]],completed_modes=[False])
    draw=prepared_draw([line,protected],measurement=measure)
    draw._snt_index=lambda:pytest.fail('select all consulted SNT')
    tool=controller(bridge,draw,'block')
    tool.selection_manager.select_all()
    assert {f['source'] for f in tool.selection_manager.get_features()}=={'native','measure'}
    assert tool.selection_manager.count()==2


def test_measure_selection_is_protected_against_destructive_delete(runtime):
    Qt,AppQt,bridge,_=runtime
    measure=SimpleNamespace(completed=[[(1,1),(4,1)]],completed_modes=[False])
    draw=prepared_draw([],measurement=measure)
    tool=controller(bridge,draw,'block')
    tool._select_area('block',[(0,0),(7,7)])
    tool.delete_selected()
    assert len(measure.completed)==1
    assert not draw.actions


def test_snt_text_anchor_and_bounds_include_rotated_labels():
    entity={'type':'TEXT','position':(100.,200.,0.),'text':'HELLO','height':5.,'rotation':90}
    assert text_anchor(entity)==(100.,200.)
    box=text_bounds(entity,(100.,200.))
    assert box[0]<=95 and box[2]<=200 and box[3]>210
    assert box_overlap(box,(90,110,195,220))
    assert not box_overlap(box,(120,130,195,220))


def test_text_drag_selects_glyph_not_only_anchor(runtime):
    Qt,AppQt,bridge,_=runtime
    text={'type':'text','text':'SNT GRID LABEL','position':(100.,200.,0.),
          'height':8.,'layer':'FileNames'}
    block={'type':'polygon','points':[(0,0),(1000,0),(1000,1000),(0,1000)],'layer':'BLOCKS'}
    snt={'full_path':'C:/map/grid.snt','_render_entities':[text,block]}
    draw=prepared_draw([],attachments=[snt])
    tool=controller(bridge,draw,'snt_text')
    # Draw over the visible glyph to the right, without enclosing the anchor.
    assert tool.press(Pos(112,198),Qt.LeftButton)
    assert tool.release(Pos(145,215),Qt.LeftButton)
    assert tool.selection_manager.count()==1
    chosen=tool.selection_manager.get_features()[0]
    assert chosen['entity'] is text and chosen['source']=='snt'
    assert chosen['layer']=='FileNames'
    assert text['position']==(100.,200.,0.)


def test_snt_text_click_selects_bbox_region(runtime):
    Qt,AppQt,bridge,_=runtime
    text={'type':'mtext','text':'EXISTING LABEL','position':(100.,200.,0.),
          'height':8.,'layer':'FileNames'}
    draw=prepared_draw([],attachments=[{'filename':'file.snt','_render_entities':[text]}])
    tool=controller(bridge,draw,'snt_text')
    assert tool.press(Pos(110,204),Qt.LeftButton)
    assert tool.release(Pos(110,204),Qt.LeftButton)
    assert tool.selection_manager.count()==1


def test_snt_text_ctrl_shift_inverts_drag_result(runtime):
    Qt,AppQt,bridge,_=runtime
    text={'type':'text','text':'HELLO','position':(100.,200.),'height':5.,'layer':'T'}
    draw=prepared_draw([],attachments=[{'filename':'file.snt','_render_entities':[text]}])
    tool=controller(bridge,draw,'snt_text')
    tool.press(Pos(99,199),Qt.LeftButton)
    tool.release(Pos(122,209),Qt.LeftButton)
    assert tool.selection_manager.count()==1
    AppQt.modifiers=Qt.ControlModifier | Qt.ShiftModifier
    try:
        tool.press(Pos(99,199),Qt.LeftButton)
        tool.release(Pos(122,209),Qt.LeftButton)
        assert tool.selection_manager.count()==0
    finally:AppQt.modifiers=0


def test_hidden_snt_text_not_pickable(runtime):
    Qt,AppQt,bridge,_=runtime
    text={'type':'text','text':'INVISIBLE','position':(100.,200.),'height':8.,'layer':'Off'}
    draw=prepared_draw([],attachments=[{'filename':'file.snt','_render_entities':[text]}])
    draw._snt_visible=lambda doc,layer:False
    tool=controller(bridge,draw,'snt_text')
    tool._select_text_rectangle((0,0),(200,400))
    assert tool.selection_manager.count()==0


def test_text_index_invalidated_when_attachment_list_changes(runtime):
    Qt,AppQt,bridge,_=runtime
    first={'filename':'first.snt','_render_entities':[{'type':'text','text':'FIRST',
                 'position':(10.,10.),'height':3.}]}
    second={'filename':'second.snt','_render_entities':[{'type':'text','text':'SECOND',
                  'position':(100.,100.),'height':3.}]}
    draw=prepared_draw([],attachments=[first])
    tool=controller(bridge,draw,'snt_text')
    assert len(tool._visible_text_features())==1
    draw.app.snt_attachments=[second]
    assert len(tool._visible_text_features())==1
    assert tool._visible_text_features()[0]['entity']['text']=='SECOND'


def test_snt_text_only_accepted_in_explicit_text_mode(runtime):
    Qt,AppQt,bridge,_=runtime
    text={'type':'text','position':(2,2),'text':'T','height':4}
    draw=prepared_draw([],attachments=[{'filename':'file.snt','_render_entities':[text]}])
    tool=controller(bridge,draw,'block')
    feat={'source':'snt','type':'text','coords':[(2,2)],'entity':text,
          'key':(('snt','file.snt'),0),'doc_key':('snt','file.snt'),'layer':'0'}
    tool.selection_manager.apply([feat])
    assert tool.selection_manager.count()==0
    tool.set_method('snt_text')
    tool.selection_manager.apply([feat])
    assert tool.selection_manager.count()==1


def test_native_curve_point_snap_prioritizes_self_start(runtime):
    Qt,AppQt,bridge,native=runtime
    draw=native.NativeDraw.__new__(native.NativeDraw)
    draw.canvas=Canvas();draw.app=App();draw.tool='curve'
    draw.points=[(0.,0.),(5.,0.),(5.,5.)]
    draw._snap_target=None
    # An external SNT line is closer than the cursor to start; own start wins.
    draw._snap_at=lambda p,mode,tol:(.5,.5)
    draw._near_scene=lambda p,tol:[]
    candidate=draw._point(Pos(.2,.1))
    assert candidate==(0.,0.)
    assert draw._snap_target==(0.,0.)


def test_native_curve_finish_replaces_nearby_last_control_with_first(runtime):
    Qt,AppQt,bridge,native=runtime
    draw=native.NativeDraw.__new__(native.NativeDraw)
    draw.canvas=Canvas();draw.tool='curve';draw._commit=lambda **kw:kw
    draw.points=[(0.,0.),(5.,0.),(5.,5.),(.1,.1)]
    result=draw._finish()
    assert result['points'][-1]==result['points'][0]


def test_text_selection_preview_has_rectangle(runtime):
    Qt,AppQt,bridge,_=runtime
    draw=prepared_draw([])
    tool=controller(bridge,draw,'snt_text')
    tool._start=(2,3);tool._hover=(5,8)
    outlines=tool.preview_shapes()
    assert outlines and outlines[0][1]==[(2,3),(5,3),(5,8),(2,8),(2,3)]


def test_selected_measurement_emits_highlight(runtime):
    Qt,AppQt,bridge,native=runtime
    measure=SimpleNamespace(completed=[[(1,1),(4,1)]],completed_modes=[False])
    draw=prepared_draw([],measurement=measure)
    tool=controller(bridge,draw,'block')
    tool._select_area('block',[(0,0),(6,6)])
    assert tool.highlight_features()[0]['type']=='measure_line'


def test_new_source_methods_do_not_write_snt_entities(runtime):
    Qt,AppQt,bridge,_=runtime
    text={'type':'text','text':'KEEP','position':(100.,200.,0.),'height':4.,'layer':'T'}
    attachment={'full_path':'file.snt','_render_entities':[text]}
    original=dict(text)
    draw=prepared_draw([],attachments=[attachment])
    tool=controller(bridge,draw,'snt_text')
    tool._select_text_rectangle((0,0),(200,400))
    tool.delete_selected()
    assert text==original
    assert tool.selection_manager.count()==0


def test_standalone_draw_select_ignores_snt_block(runtime):
    Qt,AppQt,bridge,native=runtime
    draw=native.NativeDraw.__new__(native.NativeDraw)
    draw.drawings=[{'type':'line','coords':[(0,0),(6,0)]},
                   {'type':'polyline','coords':[(0,0),(6,0),(6,6),(0,0)],
                    'source':'block_creation'}]
    draw.canvas=Canvas()
    draw._near_scene=lambda p,tol:pytest.fail('Stand-alone Draw Select touched SNT scene')
    assert draw._pick_editable_draw((3,0))['drawing'] is draw.drawings[0]
    assert draw._pick_editable_draw((3,4)) is None


def test_measurement_fence_cline_selects_path(runtime):
    Qt,AppQt,bridge,_=runtime
    measure=SimpleNamespace(completed=[[(0,2),(9,2)]],completed_modes=[True])
    draw=prepared_draw([],measurement=measure)
    tool=controller(bridge,draw,'cline')
    tool._select_area('cline',[(4,0),(4,6)])
    assert tool.selection_manager.count()==1
    assert tool.selection_manager.get_features()[0]['type']=='measure_path'


def test_measure_move_refused_without_mutating_original(runtime):
    Qt,AppQt,bridge,_=runtime
    original=[(0,0),(4,0)]
    measure=SimpleNamespace(completed=[list(original)],completed_modes=[False])
    draw=prepared_draw([],measurement=measure)
    tool=controller(bridge,draw,'block')
    tool._select_area('block',[(-1,-1),(10,10)])
    tool.enter_move_mode()
    assert not tool._move_mode
    assert measure.completed[0]==original
    assert 'selection-only' in draw.app.bar.messages[-1]


def test_move_copy_preview_replaces_selected_native_sources(runtime):
    """An active transform paints one ghost, not source plus ghost."""
    Qt,AppQt,bridge,_=runtime
    first={'type':'line','coords':[(0,0),(4,0)]}
    second={'type':'line','coords':[(10,0),(14,0)]}
    draw=prepared_draw([first,second])
    tool=controller(bridge,draw,'block')
    tool.selection_manager.apply([
        tool._native_feature(0,first), tool._native_feature(1,second)
    ])
    tool.enter_copy_mode()
    assert tool.transform_hidden_drawing_ids()==set()
    tool._offset_origin=(0.,0.)
    tool._hover=(5.,5.)
    assert tool.transform_hidden_drawing_ids()=={id(first),id(second)}
    previews=tool.preview_shapes()
    assert len([shape for shape in previews if shape[0]=='drawing_preview'])==2
    tool.exit_copy_mode()
    assert tool.transform_hidden_drawing_ids()==set()


@pytest.mark.parametrize('copy_mode', [False, True])
@pytest.mark.parametrize('drag', [False, True])
def test_transform_commit_matches_preview_in_every_repaint(runtime, copy_mode, drag):
    Qt,AppQt,bridge,_=runtime
    original={'type':'line','coords':[(100.,200.),(110.,210.)]}
    draw=prepared_draw([original])
    tool=controller(bridge,draw,'block')
    tool.selection_manager.apply([tool._native_feature(0,original)])
    (tool.enter_copy_mode if copy_mode else tool.enter_move_mode)()
    tool.press(Pos(100.,200.),Qt.LeftButton)
    tool.move(Pos(130.,250.),0)
    expected=tool.preview_shapes()[0][1]
    frames=[]
    draw._paint=lambda:frames.append((tool._offset_origin, tool.preview_shapes()))
    if drag:
        tool.release(Pos(130.,250.),Qt.LeftButton)
    else:
        tool.press(Pos(130.,250.),Qt.LeftButton)
        tool.release(Pos(130.,250.),Qt.LeftButton)
    assert draw.drawings[-1]['coords']==expected
    assert len(draw.drawings)==(2 if copy_mode else 1)
    assert original['coords']==[(100.,200.),(110.,210.)]
    assert frames and all(anchor is None and not shapes for anchor,shapes in frames)
    assert len(draw.actions)==1


def test_text_copy_uses_rendered_entities_without_modifying_them(runtime,monkeypatch):
    Qt,AppQt,bridge,_=runtime
    text={'type':'text','text':'FIRST GRID','position':(10.,20.),'height':4.,'layer':'T'}
    draw=prepared_draw([],attachments=[{'filename':'file.snt','_render_entities':[text]}])
    tool=controller(bridge,draw,'snt_text')
    tool._select_text_rectangle((0,0),(50,50))
    clipboard=SimpleNamespace(value=None,setText=lambda v:setattr(clipboard,'value',v))
    monkeypatch.setattr(AppQt,'clipboard',lambda:clipboard,raising=False)
    tool.copy_selected_snt_text_to_clipboard()
    assert clipboard.value=='FIRST GRID'
    assert text['text']=='FIRST GRID'


def test_snt_text_rotate_mirror_extend_are_blocked(runtime):
    Qt,AppQt,bridge,_=runtime
    text={'type':'text','text':'READ ONLY','position':(10.,20.),'height':4.}
    draw=prepared_draw([],attachments=[{'filename':'file.snt','_render_entities':[text]}])
    tool=controller(bridge,draw,'snt_text')
    tool._select_text_rectangle((0,0),(100,100))
    tool.rotate_selected()
    tool.mirror_selected()
    tool.extend()
    assert draw.drawings==[]
    assert len(draw.app.bar.messages)>=3
    assert text['position']==(10.,20.)
