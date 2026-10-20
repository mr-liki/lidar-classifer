import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def manager_type():
    # Exercise the shared edit methods without requiring a VTK installation.
    tree = ast.parse(Path("gui/grid_label_system.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "GridLabelManager")
    names = {"_capture_label_state", "_apply_label_state", "_label_font_scale_factor",
             "_iter_all_grid_label_actors", "_iter_native_grid_labels", "_force_render_after_label_edit",
             "_undo_last_label_edit"}
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 "label_methods", "exec"), namespace)
    return namespace["GridLabelManager"]


def test_native_bulk_resize_rebuild_and_undo_preserve_text_color_and_base_height():
    entity = dict(type="TEXT", text="Block001", height=2.0, color=(255, 0, 0))
    document = {"entities": [entity], "bounds": (0, 1, 0, 1)}
    calls = []
    def put_entities(kind, name, entities, bounds):
        calls.append(entities[0]["height"])
        document.pop("_labels_dirty", None)
        return {}
    layer = SimpleNamespace(documents={("snt", "survey.snt"): document}, put_entities=put_entities)
    manager = manager_type()()
    manager.app = SimpleNamespace(_native_vector_layer=layer)
    manager._get_main_interactor_handles = lambda: pytest.fail("Native font sizing must not access VTK")
    actor, = manager._iter_native_grid_labels()
    before = manager._capture_label_state(actor)
    for size, height in [(150, 8.0), (1, 2.0 / 75.0)]:
        assert manager._apply_label_state(actor, before["text"], size, before["color"])
        manager._force_render_after_label_edit()
        assert entity["height"] == pytest.approx(height)
    assert entity["text"] == "Block001"
    assert entity["color"] == (255, 0, 0)
    manager._label_edit_undo_stack = [[before]]
    manager._undo_last_label_edit()
    assert entity["height"] == 2.0
    assert calls == pytest.approx([8.0, 2.0 / 75.0, 2.0])


def test_font_size_excludes_user_drawings():
    manager = manager_type()()
    manager.app = SimpleNamespace(_native_vector_layer=SimpleNamespace(documents={
        ("drawing", "user"): {"entities": [dict(type="text", text="annotation")]}}))
    assert list(manager._iter_all_grid_label_actors()) == []


def test_native_document_rebuild_retains_layer_handles_and_visibility(monkeypatch):
    from gui.native_vector_overlay import NativeVectorLayer, VectorLayerHandle
    layer = NativeVectorLayer.__new__(NativeVectorLayer)
    layer._v_used = layer._i_used = 0
    layer._pin_origin = lambda groups: None
    layer._reserve = lambda *args, **kwargs: True
    layer._upload_document = lambda document: None
    layer._push_line_width = lambda: None
    layer.refresh = lambda: None
    key = ("snt", "survey.snt")
    handle = VectorLayerHandle(layer, key, "BL", None, [])
    handle._visible = False
    document = {"handles": {"BL": handle}, "gpu": {}}
    layer.documents = {key: document}
    monkeypatch.setattr("gui.native_label_layout.native_text_entities", lambda entity: None)
    entity = dict(type="text", layer="BL", text="Block001", position=(0, 0, 0), height=2)
    result = layer.put_entities(*key, [entity])
    assert result["BL"][0] is handle
    assert not handle.GetVisibility()
    assert layer.documents[key] is document
    assert document["entities"][0] is entity


def test_native_font_size_does_not_read_vtk_actor_stores():
    class App:
        _native_vector_layer = SimpleNamespace(documents={})

        @property
        def snt_actors(self):
            pytest.fail("Native font sizing must not inspect VTK actors")

        dxf_actors = snt_actors

    manager = manager_type()()
    manager.app = App()
    assert list(manager._iter_native_grid_labels()) == []
