from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QWidget

from gui.native_main_classify import MainCanvas
from gui.native_vector_overlay import _MvpScreenMapping


@pytest.fixture
def scene():
    qt = QApplication.instance() or QApplication([])
    widget = QWidget()
    widget.resize(640, 480)
    widget.show()
    matrix = np.eye(4)
    backend = SimpleNamespace(_last_extent=(640, 480),
        get_last_mvp=Mock(side_effect=lambda: matrix.copy()),
        get_render_origin_xyz=Mock(return_value=(595200., 682060., 0.)))
    owner = SimpleNamespace(vulkan_widget=widget, vulkan_backend=backend)
    mapping = _MvpScreenMapping(owner)
    view = SimpleNamespace(owner=owner, app=SimpleNamespace(), native=mapping)
    canvas = MainCanvas(view)
    yield qt, widget, matrix, backend, mapping, canvas
    widget.close()


def test_main_measurement_uses_rendered_frame_instead_of_live_controller(scene):
    _, _, matrix, backend, _, canvas = scene
    backend.present_count = 1
    canvas.view.native = SimpleNamespace(
        projection_snapshot=Mock(side_effect=AssertionError('live controller queried')))
    _, project = canvas.projection_snapshot()
    assert project(595200., 682060.) == (320., 240.)
    matrix[0, 3] = .25
    backend.render_count = 2  # a skipped render must not move measurements
    assert canvas.projection_snapshot()[1](595200., 682060.) == (320., 240.)
    backend.present_count = 2
    assert canvas.projection_snapshot()[1](595200., 682060.) == (400., 240.)
    assert project(595200., 682060.) == (320., 240.)


def test_section_measurement_tracks_presented_camera_not_pending_pan(scene):
    from gui.native_sections import SectionCanvas
    _, _, matrix, backend, _, _ = scene
    backend.present_count = 1
    section = SimpleNamespace(backend=backend, width=lambda: 640, height=lambda: 480,
                              devicePixelRatioF=lambda: 1.5,
                              centre=np.array([595200., 0., 100.]), scale=10.)
    matrix[:] = 0.
    matrix[0, 0] = matrix[1, 2] = matrix[3, 3] = 1.
    _, project = SectionCanvas.projection_snapshot(section)
    assert project(595200., 0.) == (320., 240.)
    section.centre += 100.  # input changes ahead of the next rendered frame
    section.scale = 20.
    assert SectionCanvas.projection_snapshot(section)[1](595200., 0.) == (320., 240.)
    matrix[0, 3] = .25
    backend.present_count += 1
    assert SectionCanvas.projection_snapshot(section)[1](595200., 0.) == (400., 240.)
    assert project(595200., 0.) == (320., 240.)
