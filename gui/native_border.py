"""Per-view border controls for native LNT/Vulkan renderers only."""
import ctypes
from PySide6.QtCore import QSettings

_dll = None


def set_border(handle, percent, mode):
    global _dll
    if _dll is None:
        from gui.lnt_native import VULKAN_DLL
        dll = ctypes.CDLL(VULKAN_DLL)
        fn = dll.nkv_set_point_border
        fn.argtypes = [ctypes.c_uint64, ctypes.c_float, ctypes.c_int]
        fn.restype = ctypes.c_int
        _dll = dll
    return bool(_dll.nkv_set_point_border(int(handle), float(percent), int(mode)))


def settings_for(app, slot):
    settings = QSettings('NakshaAI', 'LidarApp')
    saved = settings.value('global_view_borders', {}) or {}
    percent = (getattr(app, 'view_borders', {}) or {}).get(
        slot, getattr(app, 'point_border_percent', saved.get(str(slot), 0.)) if slot == 0
        else saved.get(str(slot), 0.))
    dialog = getattr(app, 'display_mode_dialog', None)
    modes = getattr(dialog, 'view_border_modes', None) or getattr(app, '_naksha_border_logic_modes', {}) or {}
    mode = modes.get(slot)
    if dialog is not None and getattr(dialog, 'current_slot', -1) == slot:
        mode = dialog.get_border_mode()
    if mode is None:
        saved_modes = settings.value('global_view_border_modes', {}) or {}
        mode = saved_modes.get(str(slot), settings.value('global_border_logic_mode', 1))
    return max(0., min(100., float(percent))), max(0, min(2, int(mode)))


def apply_border(app, slot, percent=None, mode=None):
    """Return False only when the view is not owned by the native workflow."""
    native = getattr(app, '_lnt_main_view', None)
    sections = getattr(app, '_native_sections', None)
    if native is None and not getattr(sections, 'views', {}):
        return False
    saved_percent, saved_mode = settings_for(app, slot)
    percent = saved_percent if percent is None else float(percent)
    mode = saved_mode if mode is None else int(mode)
    modes = getattr(app, '_naksha_border_logic_modes', None)
    if not isinstance(modes, dict):
        app._naksha_border_logic_modes = modes = {}
    modes[slot] = mode
    if slot == 0:
        app.point_border_percent = percent
        if native is not None and native.native is not None:
            if not set_border(native.native.renderer_handle(), percent, mode):
                raise RuntimeError('Native border update failed')
            native.native.request_redraw()
    else:
        canvas = getattr(sections, 'views', {}).get(slot)
        if canvas is not None and canvas.backend._handle:
            if not set_border(canvas.backend._handle, percent, mode):
                raise RuntimeError('Native section border update failed')
            canvas._dirty = True
    return True
