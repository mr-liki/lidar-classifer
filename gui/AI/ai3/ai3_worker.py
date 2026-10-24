"""GUI bridge for AI model 3.0.

Full-file mode
    The engine classifies the whole source LAS/LAZ; the result is streamed into the viewer's
    classification array and a classified copy of the *source* is written next to it.

Fence mode
    The pipeline needs surrounding points (ground height, 50 m inference tiles), so it runs on a
    LOCAL cropped copy: the bounding box of the selected fence points plus a context buffer.
    Only the points inside the fence are merged back; the context points are never modified in
    the viewer or in the written copy.  A fence result is therefore not guaranteed identical to a
    full-file run (the cropped cloud has less context at its edges); the full file is the
    reference.

The classified copy is always built from the SOURCE file with only the classification replaced:
the engine's own output goes through colour handling for the model and is not a faithful copy of
the user's data.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import laspy
import numpy as np
from PySide6.QtCore import QThread, Signal

from .ai3_controller import Ai3Cancelled, Ai3Controller, Ai3Error, _log

# The 5-class model's internal indices used by the dialog's class_mapping, and the power-line
# keys of power_mapping (gui.AI.basic.inference.InferenceConfig.WIRE/POLE_INTERNAL_CODE).
CLASS_KEYS = {0: "ground", 1: "low_veg", 2: "medium_veg", 3: "high_veg", 4: "building"}
WIRE_INTERNAL_KEY = 5
POLE_INTERNAL_KEY = 6

# Host no-PTC convention (the one Premium keeps): LAS 0 unclassified, 7 low point, 18 high noise.
DEFAULT_UNCLASSIFIED = 0
DEFAULT_LOW_NOISE = 7
DEFAULT_HIGH_NOISE = 18


def _next_free(preferred: int, used: set) -> int:
    code = int(preferred)
    while code in used and code < 255:
        code += 1
    return code


def build_engine_mapping(class_mapping: dict, power_mapping: dict, advanced_config: dict,
                         enable_power_lines: bool) -> dict:
    """The mapping JSON the engine reads, from the dialog's own codes.

    Class codes come straight from the dialog.  Unclassified / low point / high noise come from
    the active PTC when it defines them, else the host convention (0 / 7 / 18).  A support code
    that would collide with a class code the user chose (the zero-based preset uses 0 for
    ground) moves to the next free number, so a point can never be delivered under a class it
    does not belong to.
    """
    classes: Dict[str, int] = {}
    for index, key in CLASS_KEYS.items():
        if index not in class_mapping:
            raise ValueError(f"The class mapping has no output code for '{key}'.")
        classes[key] = int(class_mapping[index])
    classes["wire"] = int((power_mapping or {}).get(WIRE_INTERNAL_KEY, 14))
    classes["pole"] = int((power_mapping or {}).get(POLE_INTERNAL_KEY, 15))

    used = set(classes.values()) if enable_power_lines else {classes[k] for k in CLASS_KEYS.values()}
    optional = dict((advanced_config or {}).get("_ptc_optional_semantic_codes", {}) or {})
    unclassified = _next_free(optional.get("uncategorized", DEFAULT_UNCLASSIFIED), used)
    # support classes may share a code with each other (a PTC often has one noise class)
    low_noise = _next_free(optional.get("low_point", DEFAULT_LOW_NOISE), used | {unclassified})
    high_noise = _next_free(optional.get("high_noise", DEFAULT_HIGH_NOISE), used | {unclassified})
    classes.update({"unclassified": unclassified, "low_noise": low_noise, "high_noise": high_noise})
    return {"classes": classes, "power_lines_enabled": bool(enable_power_lines)}


def _las14_point_format_for(source_format_id: int) -> int:
    source_format_id = int(source_format_id)
    if source_format_id in (0, 1):
        return 6
    if source_format_id in (2, 3):
        return 7
    if source_format_id == 4:
        return 9
    if source_format_id == 5:
        return 10
    if source_format_id in (6, 7, 8, 9, 10):
        return source_format_id
    return 7


def _extended_header(source_header) -> "laspy.LasHeader":
    """Header able to store classification codes 0-255 (LAS 1.4), CRS and extra dimensions kept."""
    source_pf = int(source_header.point_format.id)
    if source_pf >= 6:
        return source_header.copy()
    header = laspy.LasHeader(version="1.4", point_format=_las14_point_format_for(source_pf))
    header.scales = np.asarray(source_header.scales).copy()
    header.offsets = np.asarray(source_header.offsets).copy()
    for attr in ("system_identifier", "generating_software", "global_encoding"):
        try:
            setattr(header, attr, getattr(source_header, attr))
        except Exception:
            pass
    try:
        header.vlrs.extend(source_header.vlrs)
    except Exception:
        pass
    try:
        for dim in source_header.point_format.extra_dimensions:
            header.add_extra_dim(laspy.ExtraBytesParams(name=dim.name, type=dim.dtype))
    except Exception:
        pass
    return header


class Ai3Worker(QThread):
    progress = Signal(int, str)
    finished = Signal()
    error = Signal(str)

    STREAM_CHUNK = 1_000_000
    # Context kept around the fence. The pipeline smooths ground over a 16 m window and the model
    # looks at 50 m tiles, so the default is generous; NAKSHA_AI3_FENCE_BUFFER_M overrides it.
    FENCE_CONTEXT_BUFFER_M = float(os.environ.get("NAKSHA_AI3_FENCE_BUFFER_M", "50.0"))
    MIN_ENGINE_POINTS = 5_000          # below this the pipeline has too little to work with
    ORDER_CHECK_STEP = 997             # every n-th point is compared when checking point order
    ORDER_CHECK_TOL = 0.011            # metres: the LAS coordinate resolution is 0.01 m

    def __init__(self, data_dict, class_mapping, power_mapping, controller: Optional[Ai3Controller] = None,
                 target_indices=None, advanced_config=None, enable_power_lines=False,
                 ptc_active=False, ptc_path=None):
        super().__init__()
        self.ptc_path = str(ptc_path) if ptc_path else None
        self.data_dict = data_dict
        self.class_mapping = dict(class_mapping or {})
        self.power_mapping = dict(power_mapping or {})
        self.controller = controller or Ai3Controller()
        self.target_indices = target_indices
        self.advanced_config = dict(advanced_config or {})
        self.enable_power_lines = bool(enable_power_lines)
        self.ptc_active = bool(ptc_active)
        self.ai_mode = "ai3"
        self.output_path: Optional[Path] = None
        self.warnings: list = []
        self._cancel_requested = False
        self._work_root: Optional[Path] = None

    # ------------------------------------------------------------------ control
    def cancel(self) -> None:
        self._cancel_requested = True
        try:
            self.controller.cancel()
        except Exception:
            pass

    def _check_cancel(self) -> None:
        if self._cancel_requested:
            raise Ai3Cancelled("AI model 3.0 classification was cancelled.")

    # ------------------------------------------------------------------ data access
    @staticmethod
    def _source_path(data_dict) -> Path:
        source = data_dict.get("_source_file_path")
        if not source:
            raise Ai3Error("The source LAS/LAZ path is missing; AI model 3.0 needs the original file.")
        path = Path(source).resolve()
        if not path.is_file():
            raise Ai3Error(f"Source LAS/LAZ not found: {path}")
        return path

    @staticmethod
    def _point_count(path: Path) -> int:
        with laspy.open(str(path), mode="r") as reader:
            return int(reader.header.point_count)

    @staticmethod
    def _live_classes(data_dict, expected: int) -> np.ndarray:
        classes = data_dict.get("classification")
        if classes is None:
            raise Ai3Error("The viewer classification array is missing.")
        classes = np.asarray(classes)
        if len(classes) != expected:
            raise Ai3Error(f"Viewer/source point-count mismatch: viewer={len(classes):,}, "
                           f"source={expected:,}")
        return classes

    @staticmethod
    def _xyz(data_dict, expected: int) -> np.ndarray:
        xyz = data_dict.get("xyz")
        if xyz is None:
            raise Ai3Error("The viewer XYZ array is missing; fence mode needs it.")
        xyz = np.asarray(xyz)
        if xyz.ndim != 2 or xyz.shape[1] < 2 or len(xyz) != expected:
            raise Ai3Error(f"Viewer XYZ has an unexpected shape {xyz.shape}.")
        return xyz

    @staticmethod
    def _valid_targets(target_indices, n_total: int) -> np.ndarray:
        target = np.asarray(target_indices, dtype=np.int64).ravel()
        target = target[(target >= 0) & (target < n_total)]
        if target.size == 0:
            raise Ai3Error("Fence mode received no valid target points.")
        return np.unique(target)

    def _scaled(self, lo: float, hi: float):
        """Progress callback that maps the engine's 0-100 onto [lo, hi] of the whole run."""
        def emit(percent: int, message: str) -> None:
            self.progress.emit(int(lo + (hi - lo) * max(0, min(100, percent)) / 100.0), message)
        return emit

    # ------------------------------------------------------------------ fence input
    def _build_fence_local_input(self, source: Path, subset_path: Path, target_indices):
        """Crop the source to the fence's bounding box + context. Returns
        (target_global, target_local, subset_count)."""
        n_total = self._point_count(source)
        target = self._valid_targets(target_indices, n_total)
        xyz = self._xyz(self.data_dict, n_total)
        target_xy = np.asarray(xyz[target, :2], dtype=np.float64)
        buffer_m = max(float(self.FENCE_CONTEXT_BUFFER_M), 0.0)
        min_x, max_x = float(target_xy[:, 0].min() - buffer_m), float(target_xy[:, 0].max() + buffer_m)
        min_y, max_y = float(target_xy[:, 1].min() - buffer_m), float(target_xy[:, 1].max() + buffer_m)

        subset_path.parent.mkdir(parents=True, exist_ok=True)
        subset_path.unlink(missing_ok=True)
        self.progress.emit(2, f"Cropping the fence area: {len(target):,} target points + "
                              f"{buffer_m:.0f} m context...")
        parts, source_pos, subset_count = [], 0, 0
        with laspy.open(str(source), mode="r") as reader:
            header = reader.header.copy()
            with laspy.open(str(subset_path), mode="w", header=header,
                            do_compress=subset_path.suffix.lower() == ".laz") as writer:
                for points in reader.chunk_iterator(self.STREAM_CHUNK):
                    self._check_cancel()
                    x = np.asarray(points.x, dtype=np.float64)
                    y = np.asarray(points.y, dtype=np.float64)
                    mask = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
                    if np.any(mask):
                        parts.append(np.flatnonzero(mask).astype(np.int64) + source_pos)
                        writer.write_points(points[mask])
                        subset_count += int(mask.sum())
                    source_pos += len(points)
        if source_pos != n_total:
            raise Ai3Error(f"Fence crop scan mismatch: {source_pos:,}/{n_total:,}")
        if subset_count < self.MIN_ENGINE_POINTS or not parts:
            raise Ai3Error(f"The fence area holds only {subset_count:,} points; AI model 3.0 needs "
                           f"at least {self.MIN_ENGINE_POINTS:,}. Select a larger fence.")
        subset_global = np.concatenate(parts)
        target_local = np.searchsorted(subset_global, target)
        if np.any(target_local >= len(subset_global)) or \
                not np.array_equal(subset_global[target_local], target):
            raise Ai3Error("The fence crop lost some of the selected target points.")
        return target, target_local.astype(np.int64), int(subset_count)

    # ------------------------------------------------------------------ engine output
    def _read_engine_classes(self, reference: Path, engine_output: Path) -> np.ndarray:
        """Classification array of the engine output, after checking it kept the point order.

        The engine's PDAL steps keep point order, and everything downstream (merging by index)
        depends on that, so it is verified rather than assumed: every ORDER_CHECK_STEP-th point of
        the reference file is compared with the same index of the output.
        """
        n_ref = self._point_count(reference)
        n_out = self._point_count(engine_output)
        if n_ref != n_out:
            raise Ai3Error(f"The engine returned {n_out:,} points for {n_ref:,} input points.")
        classes = np.empty(n_out, dtype=np.uint8)
        pos = 0
        with laspy.open(str(reference), mode="r") as ref, \
                laspy.open(str(engine_output), mode="r") as out:
            for ref_pts, out_pts in zip(ref.chunk_iterator(self.STREAM_CHUNK),
                                        out.chunk_iterator(self.STREAM_CHUNK)):
                self._check_cancel()
                count = len(out_pts)
                if len(ref_pts) != count:
                    raise Ai3Error("The engine output could not be aligned with its input.")
                s = slice(0, count, self.ORDER_CHECK_STEP)
                for axis in "xyz":
                    delta = np.abs(np.asarray(getattr(ref_pts, axis))[s]
                                   - np.asarray(getattr(out_pts, axis))[s])
                    if delta.size and float(delta.max()) > self.ORDER_CHECK_TOL:
                        raise Ai3Error("The engine changed the point order; refusing to apply "
                                       "its classification to the wrong points.")
                classes[pos:pos + count] = np.asarray(out_pts.classification, dtype=np.uint8)
                pos += count
        if pos != n_out:
            raise Ai3Error(f"Engine output read mismatch: {pos:,}/{n_out:,}")
        return classes

    # ------------------------------------------------------------------ deliverable
    def _write_classified_copy(self, source: Path, final_classes: np.ndarray, out_path: Path) -> None:
        """The source file with only the classification replaced (LAS 1.4 when codes need it)."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.unlink(missing_ok=True)
        need_ext = int(final_classes.max(initial=0)) > 31
        start = 0
        with laspy.open(str(source), mode="r") as reader:
            source_pf = int(reader.header.point_format.id)
            # legacy point formats store classification in 5 bits: only those need LAS 1.4
            header = (_extended_header(reader.header) if (need_ext and source_pf < 6)
                      else reader.header.copy())
            with laspy.open(str(out_path), mode="w", header=header,
                            do_compress=out_path.suffix.lower() == ".laz") as writer:
                for points in reader.chunk_iterator(self.STREAM_CHUNK):
                    self._check_cancel()
                    count = len(points)
                    if header.point_format.id != reader.header.point_format.id:
                        points = laspy.PackedPointRecord.from_point_record(points, header.point_format)
                    points.classification = final_classes[start:start + count]
                    writer.write_points(points)
                    start += count
        if start != len(final_classes):
            raise Ai3Error(f"Classified copy write mismatch: {start:,}/{len(final_classes):,}")

    def _ground_pct(self) -> float:
        """Share of the Ground points set to Unclassified, from the dialog (default 20 %)."""
        cfg = self.advanced_config or {}
        if "ground_to_uncategorized_percent" in cfg:
            pct = float(cfg["ground_to_uncategorized_percent"])
        elif "ground_to_uncategorized_ratio" in cfg:
            pct = float(cfg["ground_to_uncategorized_ratio"]) * 100.0
        else:
            pct = 20.0
        return min(max(pct, 0.0), 100.0)

    def _write_report(self, path: Path, **fields) -> None:
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"integration": "Naksha GUI AI model 3.0", **fields}, handle,
                          indent=2, sort_keys=True, default=str)
        except Exception:
            pass

    # ------------------------------------------------------------------ main
    def run(self) -> None:
        fence_mode = self.target_indices is not None
        try:
            if not self.ptc_path or not Path(self.ptc_path).is_file():
                raise Ai3Error("AI model 3.0 needs the class-definition .ptc file. Load it in "
                               "Display Mode and press Apply, then run again.")
            _log(f"[AI model 3.0] input PTC: {self.ptc_path}")
            self.progress.emit(1, "Preparing AI model 3.0...")
            source = self._source_path(self.data_dict)
            total = self._point_count(source)
            live = self._live_classes(self.data_dict, total)
            original = np.asarray(live, dtype=np.uint8).copy()


            out_dir = source.parent / "AI3_Classified"
            out_dir.mkdir(parents=True, exist_ok=True)
            final_path = out_dir / f"{source.stem}_ai3_classified{source.suffix}"
            self._work_root = out_dir / f"_ai3_work_{os.getpid()}_{int(time.time())}"
            shutil.rmtree(self._work_root, ignore_errors=True)
            self._work_root.mkdir(parents=True, exist_ok=True)
            self.output_path = final_path
            engine_out = self._work_root / "engine_classified.laz"

            if fence_mode:
                local_in = self._work_root / f"{source.stem}_fence_local.laz"
                target_global, target_local, subset_n = self._build_fence_local_input(
                    source, local_in, self.target_indices)
                self.progress.emit(
                    8, f"Fence mode: {len(target_global):,} target points, engine input "
                       f"{subset_n:,}/{total:,} points ({100.0 * (1 - subset_n / total):.1f}% less).")
                result = self.controller.run(local_in, engine_out, self.ptc_path, self._work_root,
                                             power_lines=self.enable_power_lines,
                                             ground_unclass_pct=self._ground_pct(),
                                             progress_callback=self._scaled(8, 88))
                self._check_cancel()
                self.progress.emit(90, "Applying the result inside the selected fence(s)...")
                local_classes = self._read_engine_classes(local_in, engine_out)
                target_classes = local_classes[target_local]
                final_classes = original.copy()
                final_classes[target_global] = target_classes
                live[target_global] = target_classes
                processed = int(len(target_global))
                engine_points = subset_n
            else:
                self.progress.emit(3, f"Full-file mode: {total:,} points.")
                result = self.controller.run(source, engine_out, self.ptc_path, self._work_root,
                                             power_lines=self.enable_power_lines,
                                             ground_unclass_pct=self._ground_pct(),
                                             progress_callback=self._scaled(3, 88))
                self._check_cancel()
                self.progress.emit(90, "Loading the classification into the viewer...")
                final_classes = self._read_engine_classes(source, engine_out)
                live[:] = final_classes
                processed = total
                engine_points = total

            self.warnings = list(result.warnings)
            self.progress.emit(95, "Writing the classified copy of the source file...")
            self._write_classified_copy(source, final_classes, final_path)
            self._write_report(
                out_dir / f"{source.stem}_ai3_report.json",
                status="COMPLETED", source=str(source), output=str(final_path),
                fence_mode=fence_mode, points_total=total, points_processed=processed,
                engine_input_points=engine_points, ptc=self.ptc_path, warnings=self.warnings,
                engine_report=result.report)
            self.progress.emit(100, "AI model 3.0 classification complete.")
            self.finished.emit()

        except Ai3Cancelled as exc:
            self.error.emit(str(exc))
        except Ai3Error as exc:
            self.error.emit(str(exc))
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            if self._work_root is not None:
                shutil.rmtree(self._work_root, ignore_errors=True)


# Name the dialog imports, mirroring the Premium module's alias.
Ai3InferenceWorker = Ai3Worker
