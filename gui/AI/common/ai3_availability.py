"""Optional AI model 3.0 integration readiness, without importing optional engines.

This build contains Basic, Advanced and Premium AI. AI3 is a separate optional
source package. File-presence checks avoid triggering heavy dependencies while
constructing the PySide6 model-selection dialogs. AI3 import failures are also
handled at the actual import sites in gui.ai_dialog.
"""
from pathlib import Path

_REQUIRED_AI3_FILES = (
    "ai3_ptc.py",
    "ai3_setup_dialog.py",
    "ai3_worker.py",
    "ai3_controller.py",
)


def ai3_installation_status(root=None):
    """Return (ready, missing filenames) for the genuine local AI3 integration."""
    if root is None:
        root = Path(__file__).resolve().parents[1] / "ai3"
    root = Path(root)
    missing = tuple(name for name in _REQUIRED_AI3_FILES if not (root / name).is_file())
    return (len(missing) == 0, missing)


def ai3_missing_message(missing):
    names = "\n".join("- " + item for item in missing)
    return (
        "The AI model 3.0 engine is not installed in this NAKSHA-AI LiDAR build.\n\n"
        "Missing source files under gui/AI/ai3/:\n" + names + "\n\n"
        "Restore the genuine AI3 engine modules and its model assets from your "
        "AI3 development branch or authorized release package. "
        "Installing PyTorch alone does not supply this engine.\n\n"
        "Basic AI, Advanced AI and Premium AI remain available. "
        "No LiDAR data has been changed."
    )
