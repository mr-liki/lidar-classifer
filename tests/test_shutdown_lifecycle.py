"""Check actual process exit: returning from app.exec() does not prove safe teardown."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = r'''
import sys,faulthandler
faulthandler.enable()
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from PySide6.QtCore import QSettings,QTimer,Qt,QObject,QEvent
from PySide6.QtWidgets import QApplication,QMessageBox
import shiboken6
QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,sys.argv[3])
from gui.app_window import NakshaApp
from gui.lnt_open import _show
app=QApplication([])
answers=iter((QMessageBox.No,QMessageBox.Yes))
QMessageBox.question=lambda *a,**k:next(answers,QMessageBox.Yes)
QMessageBox.information=lambda *a,**k:QMessageBox.Ok
QMessageBox.warning=lambda *a,**k:QMessageBox.Ok
win=NakshaApp()
win.show()
owner=win.render_backend
surface=owner.vulkan_widget
assert owner.active and owner.active_backend=="vulkan", "native Vulkan required"
legacy=win.vtk_widget.interactor
class LegacyExposure(QObject):
    shows=0
    def eventFilter(self,obj,event):
        if event.type()==QEvent.Show and getattr(win,"_shutdown_in_progress",False):
            self.shows+=1
        return False
exposure=LegacyExposure()
legacy.installEventFilter(exposure)
assert win.testAttribute(Qt.WA_DeleteOnClose)
checks=[]
def opened():
    if sys.argv[1]!="empty":
        _show(win,sys.argv[2],"Shutdown regression")
        assert win._lnt_main_view.native.renderer_handle()
    if sys.argv[1]=="dialogs":
        from gui.display_mode import DisplayModeDialog
        win.display_mode_dialog=DisplayModeDialog(win)
        win.display_mode_dialog.show()
    QTimer.singleShot(500,close)
def close():
    assert not win.close(),"cancelled close was accepted"
    assert win.isVisible() and win.color_thread.isRunning()
    if sys.argv[1]!="empty":assert win._lnt_main_view.native.renderer_handle()
    checks.append("cancel preserved session")
    assert win.close(),"confirmed close was rejected"
    assert win.color_thread is None
    assert win._shutdown_in_progress, "accepted close released shutdown guard"
    assert not owner.active and owner._geometry_mirror is None
    assert surface._owner is None and not owner.vulkan_backend._handle
    assert exposure.shows==0, "shutdown exposed legacy viewport"
    checks.append("confirmed close joined worker")
    if sys.argv[4]=="quit":app.quit()
QTimer.singleShot(400,opened)
QTimer.singleShot(20000,app.quit)
code=app.exec()
assert len(checks)==2,checks
assert not shiboken6.isValid(win),"window survived beyond Qt event loop"
print("SHUTDOWN_REGRESSION_OK",sys.argv[1],flush=True)
sys.exit(code)
'''

@pytest.mark.parametrize("exit_path",["last_window","quit"])
@pytest.mark.parametrize("scenario",["empty","lnt","dialogs"])
def test_main_window_shutdown_process(tmp_path,scenario,exit_path):
    source=ROOT/"native/lnt/test.lnt"
    if scenario!="empty" and not source.exists():
        pytest.skip("native LNT fixture unavailable")
    if not (ROOT/"native/naksha_vulkan/build_msvc/naksha_vulkan.dll").exists():
        pytest.skip("Vulkan DLL unavailable")
    document=tmp_path/"shutdown.lnt"
    if scenario!="empty":shutil.copyfile(source,document)
    script=tmp_path/"shutdown.py"
    script.write_text(SCRIPT,encoding="utf-8")
    env=dict(os.environ,PYTHONIOENCODING="utf-8",QT_QPA_PLATFORM="windows",
             NAKSHA_RENDER_BACKEND="vulkan",NAKSHA_VULKAN_MAIN_VIEWPORT="1",
             NAKSHA_VULKAN_PREVIEW="0",NAKSHA_VULKAN_DISABLE_VTK_LIDAR="1")
    result=subprocess.run([sys.executable,"-u",str(script),scenario,str(document),str(tmp_path/"settings"),exit_path],
                          cwd=ROOT,env=env,capture_output=True,text=True,encoding="utf-8",
                          errors="replace",timeout=40)
    output=result.stdout+result.stderr
    (tmp_path/"process.log").write_text(output,encoding="utf-8")
    assert result.returncode==0,f"exit {result.returncode:#x}\n{output}"
    assert "SHUTDOWN_REGRESSION_OK" in output,output
    assert "Error calling Python override" not in output,output
