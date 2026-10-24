"""Native .lnt view: interactive cut while moving, full detail once the camera is still."""
from types import SimpleNamespace

from gui.lnt_open import LntMainView


class FakeNative:
    def __init__(self):
        self.camera = (0.0, 0.0)
        self.budget = None
        self.sse = None
        self.upload_ms = None

    def world_to_screen(self, x, y):
        return (x + self.camera[0], y + self.camera[1])

    def set_point_budget(self, points):
        self.budget = points

    def set_sse(self, px):
        self.sse = px

    def set_upload_budget_ms(self, ms):
        self.upload_ms = ms


def make_view(shading='class'):
    view = SimpleNamespace(native=FakeNative(), sse=8.0, shading_name=shading,
                           _idle_detail=False, _camera_probe=None, _moved_at=0.0,
                           IDLE_POINT_BUDGET=LntMainView.IDLE_POINT_BUDGET,
                           MOTION_POINT_BUDGET=LntMainView.MOTION_POINT_BUDGET,
                           IDLE_AFTER_S=LntMainView.IDLE_AFTER_S,
                           IDLE_UPLOAD_MS=LntMainView.IDLE_UPLOAD_MS,
                           MOTION_UPLOAD_MS=LntMainView.MOTION_UPLOAD_MS)
    view._set_idle_detail = lambda on: LntMainView._set_idle_detail(view, on)
    view.update = lambda now: LntMainView._update_idle_detail(view, now)
    return view


def test_still_camera_refines_to_every_visible_original_point():
    view = make_view()
    view.update(0.0)                       # first probe counts as a move
    view.update(0.2)
    assert not view._idle_detail and view.native.budget is None
    view.update(0.35)
    assert view._idle_detail
    assert view.native.budget == LntMainView.IDLE_POINT_BUDGET
    assert view.native.sse == 0.0
    assert view.native.upload_ms == LntMainView.IDLE_UPLOAD_MS


def test_camera_motion_restores_the_interactive_cut_and_dial():
    view = make_view()
    view.update(0.0); view.update(0.5)
    assert view._idle_detail
    view.native.camera = (5.0, 0.0)         # pan
    view.update(0.51)
    assert not view._idle_detail
    assert view.native.budget == LntMainView.MOTION_POINT_BUDGET
    assert view.native.sse == 8.0
    assert view.native.upload_ms == LntMainView.MOTION_UPLOAD_MS
    view.update(0.7)                        # still moving recently: no refinement yet
    assert not view._idle_detail
    view.update(0.9)
    assert view._idle_detail


def test_shaded_and_surface_keep_the_interactive_point_cut():
    for mode in ('shaded', 'shaded_class', 'surface'):
        view = make_view(mode)
        view.update(0.0); view.update(5.0)
        assert not view._idle_detail and view.native.budget is None


def test_switching_to_surface_while_idle_drops_the_large_budget():
    view = make_view()
    view.update(0.0); view.update(0.5)
    view.shading_name = 'surface'
    view.update(0.6)
    assert not view._idle_detail
    assert view.native.budget == LntMainView.MOTION_POINT_BUDGET
