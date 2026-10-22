import ast
from pathlib import Path
from types import SimpleNamespace

from gui.native_highlight import NativeHighlightOverlay
from gui.prj_block_identifier import PRJBlockIdentifierDialog


def test_prj_identify_blue_width_six_and_three_blinks(monkeypatch):
    tree = ast.parse(Path("gui/prj_block_identifier.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PRJBlockIdentifierDialog")
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
                {"_highlight_prj_boundaries", "_clear_native_prj_highlight", "_auto_clear_highlights"}]
    timers = []
    namespace = {"QTimer": SimpleNamespace(singleShot=lambda delay, fn: timers.append((delay, fn)))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "prj_methods", "exec"), namespace)
    frames = []
    overlay = NativeHighlightOverlay(object(), SimpleNamespace(set_shapes=lambda shapes: frames.append(shapes)))
    monkeypatch.setattr("gui.native_highlight.highlight_overlay_for", lambda app: overlay)
    dialog = namespace["PRJBlockIdentifierDialog"]()
    dialog.app = SimpleNamespace(_native_highlight_overlay=overlay)
    dialog._highlight_actors = []
    block = dict(label="Block1", boundary_coords=[(0, 0), (10, 0), (10, 20)])
    assert dialog._highlight_prj_boundaries([block]) == (0, 0, 10, 20)
    style = frames[0][0][2]
    assert style["color"] == "#0078ff"
    assert style["width"] == 6
    assert [delay for delay, fn in timers] == [200, 400, 600, 800, 1000, 1200]
    for delay, fn in timers:
        fn()
    assert [bool(frame) for frame in frames] == [True, False, True, False, True, False, True]
    dialog._auto_clear_highlights()
    assert not overlay.groups
    assert not dialog._highlight_actors
    timers[-1][1]()  # Cleared highlights cannot reappear from stale callbacks.
    assert not overlay.groups


def test_native_overlay_does_not_import_vtk():
    source = Path("gui/native_highlight.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name != "vtk" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "vtk"


def test_native_prj_navigation_is_repeatable_without_attachment_layer(monkeypatch):
    frames = []
    modes = []

    native = SimpleNamespace(
        frame_region=lambda x, y, width: frames.append((x, y, width))
    )
    widget = SimpleNamespace(width=lambda: 1200, height=lambda: 600)
    view = SimpleNamespace(
        native=native,
        owner=SimpleNamespace(vulkan_widget=widget),
        set_2d=lambda enabled: modes.append(enabled),
    )
    history = []
    app = SimpleNamespace(
        _lnt_main_view=view,
        _prj_fly_camera_active=True,
        _commit_main_view_history=lambda reason: history.append(reason),
    )
    dialog = PRJBlockIdentifierDialog.__new__(PRJBlockIdentifierDialog)
    dialog.app = app

    monkeypatch.setattr("gui.native_attachments.is_native_main", lambda _app: True)
    # If camera navigation regresses to constructing an attachment overlay,
    # this sentinel turns the old failure into a focused test failure.
    monkeypatch.setattr(
        "gui.native_attachments.layer_for",
        lambda _app: (_ for _ in ()).throw(AssertionError("attachment layer used")),
    )

    dialog.fly_to_location(100.0, 200.0, target_scale=100.0)
    dialog.fly_to_location(300.0, 400.0, target_scale=50.0)

    assert frames == [(100.0, 200.0, 400.0), (300.0, 400.0, 200.0)]
    assert modes == [True, True]
    assert history == ["prj_fly_to", "prj_fly_to"]
    assert app._prj_fly_camera_active is False
