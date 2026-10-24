from types import SimpleNamespace
from unittest.mock import Mock

from gui.native_draw import get_native_draw


def test_existing_native_draw_is_discoverable_in_3d_without_creating_one():
    drawing = Mock()
    view = SimpleNamespace(native=Mock(), mode_2d=False, drawing=drawing)
    app = SimpleNamespace(_lnt_main_view=view)

    assert get_native_draw(app, existing_any_mode=True) is drawing
    assert get_native_draw(app) is None
    assert get_native_draw(app, create=True, existing_any_mode=True) is None


def test_missing_native_draw_is_not_created_in_3d():
    view = SimpleNamespace(native=Mock(), mode_2d=False, drawing=None)
    app = SimpleNamespace(_lnt_main_view=view)

    assert get_native_draw(app, existing_any_mode=True) is None
    assert view.drawing is None


def test_clear_dialog_deletes_existing_3d_native_drawing(monkeypatch):
    from gui.draw_clear_dialog import show_draw_clear_dialog

    drawings = [{'type': 'line', 'coords': [(0.0, 0.0), (1.0, 1.0)]}]

    def delete(selected_ids):
        before = len(drawings)
        drawings[:] = [item for item in drawings if id(item) not in selected_ids]
        return before - len(drawings)

    draw = SimpleNamespace(drawings=drawings, delete_drawings=delete)
    view = SimpleNamespace(native=Mock(), mode_2d=False, drawing=draw)
    status = Mock()
    app = SimpleNamespace(_lnt_main_view=view, statusBar=lambda: status)
    monkeypatch.setattr(
        'gui.native_draw.get_native_draw',
        lambda app, create=False, allow_snt=False, existing_any_mode=False:
            draw if existing_any_mode and not create else None,
    )

    assert show_draw_clear_dialog(app) == 1
    assert drawings == []
    status.showMessage.assert_called_once()
