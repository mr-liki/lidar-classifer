"""Fail-closed AI3 full-file LAZ -> resident Vulkan/LNT classification handoff.

This module does not change model output, point coordinates, PTC mappings,
LNT topology or the Vulkan renderer. It proves source-order identity before
using the output's classification field. Import laspy only at finalization.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np


def find_ai3_output(source_file: str, started_wall: float | None) -> Path:
    """Select exactly one output produced by the current AI run.

    AI3's observed output convention is:
      <source-dir>/AI3_Classified/_ai3_work_<pid>_<epoch>/engine_classified.laz
    Never select an arbitrary stale output from an earlier run.
    """
    if started_wall is None:
        raise RuntimeError("Missing AI3 run start time; cannot identify this run's output")
    directory = Path(source_file).parent / "AI3_Classified"
    if not directory.is_dir():
        raise RuntimeError(f"AI3 output directory does not exist: {directory}")
    candidates = []
    for name in ("engine_classified.laz", "engine_classified.las"):
        for candidate in directory.glob(f"_ai3_work_*/{name}"):
            if candidate.is_file():
                try:
                    stamp = candidate.stat().st_mtime
                except OSError:
                    continue
                # Small tolerance for file-system timestamp precision; never
                # admit arbitrarily old outputs into a new classification.
                if stamp >= float(started_wall) - 5.0:
                    candidates.append(candidate)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one current AI3 engine result, found {len(candidates)} "
            f"under {directory}. Refusing to commit a stale/ambiguous result."
        )
    return candidates[0]


def validated_classes(source_file: str, ai3_result_file: str | Path, expected_count: int,
                      chunk_points: int = 250_000) -> np.ndarray:
    """Return AI3 class bytes only when all other point dimensions are identical.

    Same count alone is insufficient: wrong point order would silently corrupt
    the entire classification. Compare every non-classification LAS dimension
    bit-for-bit in source order before touching the live LNT session.
    """
    import laspy

    source_file = Path(source_file)
    ai3_result_file = Path(ai3_result_file)
    if not source_file.is_file() or not ai3_result_file.is_file():
        raise RuntimeError("AI3 source or result file not found")
    if not expected_count or expected_count < 0:
        raise RuntimeError("AI3 expected point count is invalid")
    if chunk_points < 1:
        raise ValueError("chunk_points must be positive")

    with laspy.open(str(source_file)) as orig, laspy.open(str(ai3_result_file)) as result:
        src_header, out_header = orig.header, result.header
        src_dims = list(src_header.point_format.dimension_names)
        out_dims = list(out_header.point_format.dimension_names)
        if int(src_header.point_count) != expected_count or int(out_header.point_count) != expected_count:
            raise RuntimeError(
                "AI3 point count differs from the loaded LNT: "
                f"source={src_header.point_count}, output={out_header.point_count}, "
                f"LNT={expected_count}"
            )
        if src_header.point_format.id != out_header.point_format.id or src_dims != out_dims:
            raise RuntimeError("AI3 output changed the LAS point format/dimension schema")
        if "classification" not in src_dims:
            raise RuntimeError("Source LAS has no classification dimension")

        classes = np.empty(expected_count, dtype=np.uint8)
        offset = 0
        src_iter = orig.chunk_iterator(chunk_points)
        out_iter = result.chunk_iterator(chunk_points)
        while offset < expected_count:
            try:
                before = next(src_iter)
                after = next(out_iter)
            except StopIteration as exc:
                raise RuntimeError("AI3 output ended before the expected point count") from exc
            count = len(before)
            if count == 0 or len(after) != count or offset + count > expected_count:
                raise RuntimeError(f"AI3 source/output chunk length mismatch at {offset}")
            for dimension in src_dims:
                if dimension == "classification":
                    continue
                src_field = np.ascontiguousarray(np.asarray(before[dimension]))
                out_field = np.ascontiguousarray(np.asarray(after[dimension]))
                if src_field.dtype != out_field.dtype or src_field.shape != out_field.shape or \
                        src_field.tobytes() != out_field.tobytes():
                    raise RuntimeError(
                        f"AI3 output is NOT source-order identical at point {offset:,}, "
                        f"dimension '{dimension}'. No LNT data was changed."
                    )
            classes[offset:offset + count] = np.asarray(after["classification"], dtype=np.uint8)
            offset += count
        try:
            next(src_iter)
            raise RuntimeError("Source has unexpected trailing points")
        except StopIteration:
            pass
        try:
            next(out_iter)
            raise RuntimeError("AI3 output has unexpected trailing points")
        except StopIteration:
            pass
    return classes
