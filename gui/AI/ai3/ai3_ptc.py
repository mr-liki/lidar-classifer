"""The class-definition (.ptc) file AI model 3.0 takes as its input schema.

Display Mode stores the loaded file on the app only when the user presses Apply
(``app.active_ptc_path`` + ``app.active_ptc_schema``). AI model 3.0 may run only with such a file.
"""
from __future__ import annotations

import os
from typing import Optional

NEEDS_PTC_TITLE = "PTC file required"
NEEDS_PTC_TEXT = (
    "AI model 3.0 needs the class-definition .ptc file as its input.\n\n"
    "Load the .ptc file in Display Mode and press Apply, then run AI model 3.0 again."
)


def applied_ptc_path(app) -> Optional[str]:
    """Path of the .ptc applied in Display Mode, or None when none is applied (or it is gone)."""
    if app is None:
        return None
    try:
        if not getattr(app, "active_ptc_schema", None):
            return None
        path = getattr(app, "active_ptc_path", None)
    except Exception:
        return None
    if path and os.path.isfile(str(path)):
        return str(path)
    return None


def ptc_hint(app) -> str:
    """One line for the dialog: which PTC will be used, or what is missing."""
    path = applied_ptc_path(app)
    if path:
        return f"Input PTC: {os.path.basename(path)}"
    return "Requires a .ptc: load it in Display Mode and press Apply."
