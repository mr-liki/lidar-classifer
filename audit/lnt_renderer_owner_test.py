"""Real Vulkan owner bootstrap on a Qt viewport, without a VTK OpenGL window."""
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout
from PySide6.QtCore import Qt,QCoreApplication,QEvent
import vtk
from gui.render_backend import AppRenderBackendOwner

application=QApplication([])
parent=QWidget();parent.resize(800,600)
viewport=QWidget(parent);viewport.resize(800,600)
viewport.camera=vtk.vtkCamera()
viewport.renderer=vtk.vtkRenderer();viewport.renderer.SetActiveCamera(viewport.camera)
viewport.interactor=viewport
viewport.render=lambda:None
viewport.actors={}
layout=QVBoxLayout(parent);layout.addWidget(viewport)
app=SimpleNamespace(data=None,display_mode='class',vtk_widget=viewport)
owner=AppRenderBackendOwner(app,viewport)
try:
    assert owner.vulkan_backend is not None,owner.last_error
    assert owner.vulkan_widget is not None
    assert owner.vulkan_backend._handle,owner.last_error
    print('PASS real Qt viewport owner creates a live Vulkan renderer for .lnt loading',flush=True)
finally:
    owner.shutdown()
    application.processEvents()
    QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)
    parent.close();application.quit()
