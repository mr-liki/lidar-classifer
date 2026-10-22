from types import SimpleNamespace

import laspy
import numpy as np
import pytest

from gui.data_loader import _extract_classification_array
from gui.lnt_open import _laz_to_las

from gui.grid_label_system import (
    GridLabelManager,
    _block_hit_priority,
    _buffer_polygon_xy,
    _point_in_polygon_xy,
)
from gui.save_pointcloud import (
    _inspect_fenced_parent_session,
    _save_fenced_parent_files,
    save_current_pointcloud_in_place,
)


def _write_las(path, classes):
    header = laspy.LasHeader(point_format=6, version="1.4")
    las = laspy.LasData(header)
    count = len(classes)
    las.x = np.arange(count, dtype=float)
    las.y = np.zeros(count, dtype=float)
    las.z = np.zeros(count, dtype=float)
    las.classification = np.asarray(classes, dtype=np.uint8)
    las.write(path)


def test_data_backed_block_has_priority_over_overlapping_grid_polygon():
    grid = {"source": "prj_boundary", "file_path": None}
    block_data = {"source": "snt_text", "file_path": "MARCENA000018.laz"}

    assert _block_hit_priority(block_data) > _block_hit_priority(grid)


def test_block_hit_polygon_includes_every_corner_and_edge():
    polygon = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]

    for point in polygon:
        assert _point_in_polygon_xy(*point, polygon)
    assert _point_in_polygon_xy(50.0, 0.0, polygon)
    assert _point_in_polygon_xy(100.0, 50.0, polygon)


def test_legacy_las_extended_class_51_is_not_collapsed_to_19():
    las = SimpleNamespace(
        header=SimpleNamespace(version="1.2", point_format=SimpleNamespace(id=3)),
        point_format=SimpleNamespace(
            dimension_names={"classification", "synthetic", "key_point", "withheld"}
        ),
        classification=np.asarray([19, 19, 7], dtype=np.uint8),
        synthetic=np.asarray([1, 0, 0], dtype=np.uint8),
        key_point=np.zeros(3, dtype=np.uint8),
        withheld=np.zeros(3, dtype=np.uint8),
        vlrs=[],
    )

    assert _extract_classification_array(las).tolist() == [51, 19, 7]


def test_lnt_laz_bridge_promotes_legacy_packed_class_51(tmp_path):
    source = tmp_path / "legacy.laz"
    promoted = tmp_path / "promoted.las"
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = [1.0, 2.0, 3.0]
    las.y = [1.0, 2.0, 3.0]
    las.z = [1.0, 2.0, 3.0]
    las.classification = np.asarray([19, 19, 7], dtype=np.uint8)
    las.synthetic = np.asarray([1, 0, 0], dtype=bool)
    las.red = np.asarray([65535, 0, 0], dtype=np.uint16)
    las.green = np.zeros(3, dtype=np.uint16)
    las.blue = np.zeros(3, dtype=np.uint16)
    try:
        las.write(source)
    except laspy.LaspyException as exc:
        pytest.skip(f"LAZ backend unavailable: {exc}")

    _laz_to_las(
        str(source), str(promoted), lambda *_args: None, lambda: False, chunk=2
    )
    loaded = laspy.read(promoted)

    # Format 7 is format 6 plus RGB: the promotion must keep the legacy
    # colour channels, not only the packed class byte.
    assert loaded.header.point_format.id == 7
    assert np.asarray(loaded.classification).tolist() == [51, 19, 7]
    assert np.asarray(loaded.red).tolist() == [65535, 0, 0]
    assert np.asarray(loaded.green).tolist() == [0, 0, 0]


def test_buffer_polygon_contains_primary_and_requested_neighbor_strip():
    primary = [(0, 0), (100, 0), (100, 100), (0, 100)]
    buffered = _buffer_polygon_xy(primary, 10.0)

    assert _point_in_polygon_xy(50, 50, buffered)
    assert _point_in_polygon_xy(105, 50, buffered)
    assert not _point_in_polygon_xy(111, 50, buffered)
    assert min(x for x, _y in buffered) == pytest.approx(-10.0)
    assert max(x for x, _y in buffered) == pytest.approx(110.0)


def test_primary_grid_resolution_prefers_matching_snt_owner():
    manager = object.__new__(GridLabelManager)
    manager.app = SimpleNamespace()
    manager._ensure_snt_block_index = lambda: [
        {"grid_name": "51000_216000", "snt_filename": "other.snt", "points_2d": [(0, 0), (1, 0), (1, 1)]},
        {"grid_name": "51000_216000", "snt_filename": "selected.snt", "points_2d": [(10, 0), (11, 0), (11, 1)]},
    ]

    resolved = manager._resolve_primary_buffer_block("51000_216000", "selected.snt")

    assert resolved["snt_filename"] == "selected.snt"


def test_world_buffer_polygon_is_not_screen_converted_again():
    manager = object.__new__(GridLabelManager)
    manager.app = SimpleNamespace(snt_block_polygons=[{
        "grid_name": "M-34-61-D-b-3-4-4-3",
        "points_2d": [(458000, 246000), (459000, 246000), (459000, 247000), (458000, 247000)],
    }])
    manager.screen_to_world_polygon = lambda _polygon: (_ for _ in ()).throw(
        AssertionError("world buffer must not be converted as screen coordinates")
    )

    scan = manager.get_snt_blocks_intersecting_polygon(
        [(457990, 245990), (459010, 245990), (459010, 247010), (457990, 247010)],
        coordinates_are_world=True,
    )

    assert [block["grid_name"] for block in scan["intersecting_blocks"]] == [
        "M-34-61-D-b-3-4-4-3"
    ]


def test_buffered_classifications_write_to_each_original_file(tmp_path):
    first = tmp_path / "51000_216000.las"
    second = tmp_path / "51500_216000.las"
    _write_las(first, [1, 1, 1, 1])
    _write_las(second, [2, 2, 2, 2])

    session_id = "buffer-session"
    app = SimpleNamespace(
        data={
            "xyz": np.zeros((3, 3), dtype=float),
            "classification": np.asarray([11, 12, 13], dtype=np.uint8),
            "_fence_session_id": session_id,
            "_fence_source_file_ids": np.asarray([0, 1, 1], dtype=np.int32),
            "_fence_source_point_indices": np.asarray([2, 0, 3], dtype=np.int64),
        },
        _fence_parent_session={
            "session_id": session_id,
            "source_files": [str(first), str(second)],
            "operation": "buffered_grid",
            "primary_grid": "51000_216000",
            "buffer_width": 10.0,
            "duplicate_owner_indices": np.empty(0, dtype=np.int64),
            "duplicate_source_file_ids": np.empty(0, dtype=np.int32),
            "duplicate_source_point_indices": np.empty(0, dtype=np.int64),
        },
    )

    session, error, is_multisource = _inspect_fenced_parent_session(app)
    assert error is None
    assert is_multisource
    assert session["operation"] == "buffered_grid"
    assert _save_fenced_parent_files(app, session, in_place=True, show_messages=False)

    first_saved = laspy.read(first)
    second_saved = laspy.read(second)
    assert np.asarray(first_saved.classification).tolist() == [1, 1, 11, 1]
    assert np.asarray(second_saved.classification).tolist() == [12, 2, 2, 13]


def test_pathless_buffer_session_autosaves_to_parent_files(tmp_path):
    first = tmp_path / "primary.las"
    second = tmp_path / "neighbor.las"
    _write_las(first, [1, 1])
    _write_las(second, [2, 2])
    session_id = "pathless-buffer"
    app = SimpleNamespace(
        loaded_file=None,
        last_save_path=None,
        data={
            "xyz": np.zeros((2, 3), dtype=float),
            "classification": np.asarray([7, 8], dtype=np.uint8),
            "_fence_session_id": session_id,
            "_fence_source_file_ids": np.asarray([0, 1], dtype=np.int32),
            "_fence_source_point_indices": np.asarray([1, 0], dtype=np.int64),
        },
        _fence_parent_session={
            "session_id": session_id,
            "source_files": [str(first), str(second)],
            "operation": "buffered_grid",
            "primary_grid": "primary",
            "buffer_width": 10.0,
            "duplicate_owner_indices": np.empty(0, dtype=np.int64),
            "duplicate_source_file_ids": np.empty(0, dtype=np.int32),
            "duplicate_source_point_indices": np.empty(0, dtype=np.int64),
        },
    )

    assert save_current_pointcloud_in_place(app)
    assert np.asarray(laspy.read(first).classification).tolist() == [1, 7]
    assert np.asarray(laspy.read(second).classification).tolist() == [8, 2]
