"""Canonical left/right modifier handling for NAKSHA runtime shortcuts.

Windows exposes left and right Control/Alt/Shift via distinct virtual keys.
Both sides map to the existing saved shortcut names (ctrl/alt/shift).
No global hook or system-wide keyboard registration is installed.
"""

import sys


def windows_modifier_state():
    """Best-effort *current* per-side state, or None outside Windows."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        get_state = ctypes.windll.user32.GetAsyncKeyState
        keys = {
            "left_ctrl": 0xA2, "right_ctrl": 0xA3,
            "left_alt": 0xA4, "right_alt": 0xA5,
            "left_shift": 0xA0, "right_shift": 0xA1,
            "ctrl_generic": 0x11, "alt_generic": 0x12,
            "shift_generic": 0x10,
        }
        return {name: bool(get_state(vk) & 0x8000) for name, vk in keys.items()}
    except (AttributeError, OSError, ValueError):
        return None


def canonical_modifier_names(*, ctrl=False, alt=False, shift=False,
                             pressed=(), physical=None, function_key=False):
    """Return the stored shortcut modifier label in Ctrl, Alt, Shift order.

    A Windows layout may synthesize left Ctrl while right Alt (AltGr) is held.
    For F-keys, treat this *specific* pattern as right Alt, so the user's
    Alt+F1...F12 presets work on either physical Alt. Text/character keys
    retain the OS modifiers to avoid altering AltGr text-input semantics.
    """
    # A missed KeyRelease can leave Qt-side tracking stale after focus moves.
    # When Windows side-key state is available, trust it over that cache.
    pressed = set(pressed or ()) if physical is None else set()
    control = bool(ctrl) or "ctrl" in pressed
    alternate = bool(alt) or "alt" in pressed
    shifted = bool(shift) or "shift" in pressed
    if physical is not None:
        control = (control or physical.get("left_ctrl", False)
                   or physical.get("right_ctrl", False)
                   or physical.get("ctrl_generic", False))
        alternate = (alternate or physical.get("left_alt", False)
                     or physical.get("right_alt", False)
                     or physical.get("alt_generic", False))
        shifted = (shifted or physical.get("left_shift", False)
                   or physical.get("right_shift", False)
                   or physical.get("shift_generic", False))
        if (function_key and physical.get("right_alt", False)
                and not physical.get("left_alt", False)
                and physical.get("left_ctrl", False)
                and not physical.get("right_ctrl", False)):
            # Windows synthesized Ctrl for AltGr (right Alt).
            control = False
    parts = []
    if control:
        parts.append("ctrl")
    if alternate:
        parts.append("alt")
    if shifted:
        parts.append("shift")
    return "+".join(parts) if parts else "none"


def qt_event_modifier_names(event, Qt, pressed=()):
    """Resolve a real QKeyEvent; supports either side of all three modifiers."""
    key = event.key()
    # Synthesized QKeyEvents have no native virtual key; use their explicit
    # modifiers instead of mixing in unrelated physical keys held at dispatch.
    physical = None
    try:
        if event.nativeVirtualKey():
            physical = windows_modifier_state()
    except (AttributeError, TypeError):
        physical = windows_modifier_state()
    return canonical_modifier_names(
        ctrl=bool(event.modifiers() & Qt.ControlModifier),
        alt=bool(event.modifiers() & Qt.AltModifier),
        shift=bool(event.modifiers() & Qt.ShiftModifier),
        pressed=pressed,
        physical=physical,
        function_key=(Qt.Key_F1 <= key <= Qt.Key_F12 or key == Qt.Key_AltGr),
    )


def native_virtual_key_modifier_names(keyname, *, syskey_alt=False):
    """Resolve current physical modifiers for a Win32 WM_KEYDOWN message."""
    physical = windows_modifier_state() or {}
    return canonical_modifier_names(
        # WM_SYSKEYDOWN carries the Alt context in its message type even when
        # the synchronous keystate query is between transitions.
        alt=syskey_alt,
        physical=physical,
        function_key=keyname.startswith("F") and keyname[1:].isdigit(),
    )
