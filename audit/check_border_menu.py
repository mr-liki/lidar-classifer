"""Qt regression: border menu remains anchored and cannot outlive its dialog."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6.QtCore import QSettings,QPoint
from PySide6.QtWidgets import QApplication,QMainWindow
app=QApplication([])
settings=QSettings(str(ROOT/'native/naksha_vulkan/build_structured/menu_test.ini'),QSettings.IniFormat)
settings.clear()
with patch('PySide6.QtCore.QSettings',side_effect=lambda *a,**k:settings):
    from gui.display_mode import DisplayModeDialog
    owner=QMainWindow();owner.resize(900,700);owner.show()
    owner.class_palette={};owner.view_palettes={};owner.data={}
    owner._naksha_border_logic_modes={0:1}
    owner._lnt_main_view=SimpleNamespace(native=None)
    owner.view_borders={0:45};owner.display_mode='class'
    dialog=DisplayModeDialog(owner);dialog.move(300,120);dialog.show();app.processEvents()
    try:
        def opened():
            dialog.border_setting_btn.click();app.processEvents()
            menu=dialog.border_setting_menu
            assert menu.isVisible()
            assert menu.parentWidget() is dialog.border_setting_btn
            expected=dialog.border_setting_btn.mapToGlobal(QPoint(0,dialog.border_setting_btn.height()))
            assert abs(menu.pos().x()-expected.x())<8,(menu.pos(),expected)
        opened();dialog.move(dialog.x()+40,dialog.y()+30);app.processEvents()
        assert not dialog.border_setting_menu.isVisible(),'Detached popup after move'
        opened();dialog.resize(dialog.width()+20,dialog.height()+10);app.processEvents()
        assert not dialog.border_setting_menu.isVisible(),'Detached popup after resize'
        opened();dialog.hide();app.processEvents()
        assert not dialog.border_setting_menu.isVisible(),'Detached popup after hide'
        dialog._raise_visible_window_family();app.processEvents()
        assert not dialog.border_setting_menu.isVisible(),'Owner activation resurrected popup'
        print('PASS menu is button-owned, correctly anchored, and dismissed on move/resize/hide')
    finally:
        dialog._allow_native_close=True;dialog.close();owner.close();app.processEvents()
