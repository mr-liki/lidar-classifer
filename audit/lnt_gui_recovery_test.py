"""Verify the recovered GUI entry points without creating Qt/VTK windows."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gui.app_window import NakshaApp
from gui import shading_display
import ast

tree=ast.parse((ROOT/'gui/app_window.py').read_text(encoding='utf-8-sig'))
app_class=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NakshaApp')
init=next(n for n in app_class.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
viewport=next(n for n in ast.walk(init) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='vtk_widget' for t in n.targets))
owner=next(n for n in ast.walk(init) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='render_backend' for t in n.targets))
assert isinstance(owner.value,ast.Call) and getattr(owner.value.func,'id','')=='AppRenderBackendOwner'
assert owner.lineno>viewport.lineno
print('PASS main-window construction creates the native renderer owner after the viewport')

source=str(ROOT/'native/lnt/test_shading.lnt')
app=SimpleNamespace(data=None,_lnt_main_view=None)
with patch('PySide6.QtWidgets.QFileDialog.getOpenFileNames',return_value=([source],'')), \
     patch('gui.lnt_open.open_with_lnt',return_value=True) as loader, \
     patch('gui.file_loader_worker.FileLoaderWorker',side_effect=AssertionError('legacy loader called')):
    NakshaApp.open_file(app)
    loader.assert_called_once_with(app,source)
print('PASS file-open routes directly to .lnt without the legacy array loader')

native=SimpleNamespace(set_app_display_mode=Mock(),_push_applied_palette=Mock(),shading_name='shaded_class')
app=SimpleNamespace(data=None,_lnt_main_view=native,_suspend_curve_tool_safely=Mock(),
                    _is_applying_class_map=False,view_palettes={},point_border_percent=0,
                    _apply_import_class_block=Mock())
NakshaApp.set_display_mode(app,'shaded_class')
native.set_app_display_mode.assert_called_with('shaded_class')
assert NakshaApp._can_show_shading_controls(app)
print('PASS Shading ribbon and controls use the native view with no VTK point array')

NakshaApp.apply_class_map(app,{'classes':{2:{'color':(170,120,60),'show':True},5:{'color':(40,200,60),'show':True}},'target_view':0,'color_mode':1})
native._push_applied_palette.assert_called_once()
native.set_app_display_mode.assert_called_with('shaded_class')
assert app.class_palette[2]['color']==(170,120,60)
assert not app._is_applying_class_map
print('PASS palette Apply updates native classification colours and shading')

shading_display.update_shaded_class(app,azimuth=123,angle=67,ambient=.3)
assert (app.last_shade_azimuth,app.shading_sharpness_angle,app.shade_ambient)==(123.,67.,.3)
native.set_app_display_mode.assert_called_with('shaded_class')
print('PASS lighting updates reach native shading without VTK triangulation')

# Execute the dialog's native dispatch with no VTK scene or Python point records.
dialog_tree = ast.parse((ROOT / 'gui/display_mode.py').read_text(encoding='utf-8-sig'))
dispatch = next(n for n in ast.walk(dialog_tree) if isinstance(n, ast.If)
                and "getattr(app, '_lnt_main_view', None)" in ast.unparse(n.test))
native._push_applied_palette.return_value = True
app.set_display_mode = Mock()
for idx, mode in enumerate(('class', 'shaded_class', 'depth', 'intensity',
                             'rgb', 'elevation', 'surface', 'line')):
    env = dict(app=app, self=SimpleNamespace(current_slot=0), idx=idx,
               fast_path_handled=False)
    exec(compile(ast.Module(body=dispatch.body, type_ignores=[]), '<native Apply>', 'exec'), env)
    assert env['fast_path_handled']
    app.set_display_mode.assert_called_with(mode)
print('PASS Display Mode Apply dispatches every main-view mode to native')

from gui.lnt_open import LntMainView
import numpy as np
app.class_palette = {2: {'color': (170, 120, 60), 'show': True},
                     5: {'color': (40, 200, 60), 'show': False}}
view = SimpleNamespace(app=app, native=SimpleNamespace(set_classification_lut=Mock(return_value=True)))
assert LntMainView._push_applied_palette(view)
packed = view.native.set_classification_lut.call_args.args[0]
assert packed.nbytes == 1024 and packed.dtype == np.uint8
assert tuple(packed[2]) == (170, 120, 60, 1)
assert tuple(packed[5]) == (40, 200, 60, 0)
assert tuple(packed[255]) == (128, 128, 128, 1)
print('PASS native Apply uploads exactly 1024 bytes with explicit visibility')
