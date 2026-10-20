"""VTK-free faceted Shaded Classification math from shading_display.py.

Preserves the legacy representative selection, face filtering and geographic
hillshade formulas; returns arrays for the native Vulkan mesh renderer.
"""
from __future__ import annotations
import os
import time
from collections import OrderedDict
import numpy as np
from scipy.spatial import Delaunay, cKDTree
try:
    import triangle as tr
    HAS_TRIANGLE = True
except ImportError:
    HAS_TRIANGLE = False
HAS_SCIPY = True
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


_ADAPTIVE_SHADING_VERSION = 1


_ADAPTIVE_DEFAULT_TARGET = 3_000_000


_SHADING_FAST_TARGET = 1_000_000


_FEATURE_AWARE_VERSION = 2


_SHADING_LINE_FILTER_VERSION = 2


def _adaptive_boundary_representatives(xyz, baseline_indices, boundary_mask,
                                       baseline_precision, target_max):
    """Add edge-focused representatives while preserving every baseline point.

    The baseline Patch-7 set is never reduced. Boundary points are selected on
    a finer grid and only new representatives are appended. If candidates
    exceed the quality budget, deterministic spatial-order thinning keeps the
    additions distributed across the complete tile.
    """
    t0 = time.perf_counter()
    baseline_indices = np.asarray(baseline_indices, dtype=np.int64)
    boundary_mask = np.asarray(boundary_mask, dtype=bool).reshape(-1)
    n_base = len(baseline_indices)
    budget = max(int(target_max), n_base)
    extra_budget = budget - n_base
    if extra_budget <= 0 or boundary_mask.size != len(xyz) or not np.any(boundary_mask):
        return baseline_indices, {
            "enabled": False, "reason": "no_budget_or_boundary", "base": n_base,
            "boundary_points": int(np.count_nonzero(boundary_mask)), "extra": 0,
            "target": budget, "elapsed": time.perf_counter() - t0,
        }

    boundary_local = np.flatnonzero(boundary_mask)
    # Phase 1 conservative fine grid. Environment override allows production
    # tuning without changing code. 0.55 gives ~3.3x potential cell density,
    # but the hard representative budget prevents runaway meshes.
    fine_ratio = float(os.environ.get("NAKSHA_SHADING_ADAPTIVE_FINE_RATIO", "0.55"))
    fine_ratio = min(max(fine_ratio, 0.30), 0.95)
    fine_precision = max(float(baseline_precision) * fine_ratio, 0.0025)
    fine_sub = _grid_dedup_at_precision(xyz[boundary_local], fine_precision)
    fine_local = boundary_local[np.asarray(fine_sub, dtype=np.int64)]

    selected = np.zeros(len(xyz), dtype=np.uint8)
    selected[baseline_indices] = 1
    extras = fine_local[selected[fine_local] == 0]
    candidate_extra = len(extras)
    if candidate_extra > extra_budget:
        # _grid_dedup_at_precision returns deterministic grid order. Evenly
        # sampling that order retains spatial coverage better than truncation.
        pick = np.linspace(0, candidate_extra - 1, extra_budget, dtype=np.int64)
        extras = extras[pick]

    result = np.concatenate((baseline_indices, extras.astype(np.int64, copy=False)))
    meta = {
        "enabled": True,
        "reason": "boundary_refinement",
        "base": n_base,
        "boundary_points": len(boundary_local),
        "fine_candidates": len(fine_local),
        "candidate_extra": candidate_extra,
        "extra": len(extras),
        "target": budget,
        "fine_ratio": fine_ratio,
        "fine_precision": fine_precision,
        "elapsed": time.perf_counter() - t0,
    }
    return result, meta


def _feature_aware_filter_faces(faces, xyz_unique, representative_boundary, spacing):
    """Conservatively reject long triangles that bridge strong boundary jumps.

    This is a post-Delaunay quality filter, not a topology generator. It only
    acts on mixed boundary/non-boundary triangles that are simultaneously long,
    vertically discontinuous, and steep. A removal-ratio guard preserves the
    original mesh when thresholds would remove too much topology.
    """
    t0 = time.perf_counter()
    enabled = os.environ.get("NAKSHA_SHADING_FEATURE_AWARE", "0").strip().lower() not in ("0", "false", "off", "no")
    meta = {
        "enabled": False, "reason": "disabled", "input_faces": int(len(faces)),
        "output_faces": int(len(faces)), "removed": 0, "elapsed": 0.0,
    }
    if not enabled:
        meta["elapsed"] = time.perf_counter() - t0
        return faces, meta
    if representative_boundary is None or len(representative_boundary) != len(xyz_unique):
        meta.update(reason="boundary_unavailable", elapsed=time.perf_counter() - t0)
        return faces, meta
    if len(faces) == 0 or not HAS_NUMBA:
        meta.update(reason="empty_or_numba_unavailable", elapsed=time.perf_counter() - t0)
        return faces, meta

    edge_factor = float(os.environ.get("NAKSHA_SHADING_FEATURE_EDGE_FACTOR", "3.0"))
    min_edge = float(os.environ.get("NAKSHA_SHADING_FEATURE_MIN_EDGE", "1.0"))
    z_factor = float(os.environ.get("NAKSHA_SHADING_FEATURE_Z_FACTOR", "0.75"))
    min_z_jump = float(os.environ.get("NAKSHA_SHADING_FEATURE_MIN_Z_JUMP", "0.35"))
    min_slope = float(os.environ.get("NAKSHA_SHADING_FEATURE_MIN_SLOPE", "0.75"))
    max_remove_ratio = float(os.environ.get("NAKSHA_SHADING_FEATURE_MAX_REMOVE_RATIO", "0.02"))

    min_edge_len = max(float(spacing) * max(edge_factor, 1.0), max(min_edge, 0.0))
    min_jump = max(float(spacing) * max(z_factor, 0.0), max(min_z_jump, 0.0))
    remove = _numba_feature_discontinuity_mask(
        np.asarray(faces, dtype=np.int32),
        np.asarray(xyz_unique, dtype=np.float64),
        np.asarray(representative_boundary, dtype=np.uint8),
        min_edge_len * min_edge_len,
        min_jump,
        max(min_slope, 0.0),
    )
    removed = int(np.count_nonzero(remove))
    ratio = removed / max(len(faces), 1)
    if removed == 0:
        meta.update(
            reason="no_crossing_discontinuities", candidate_removed=0,
            removal_ratio=0.0, elapsed=time.perf_counter() - t0,
        )
        return faces, meta
    if ratio > max_remove_ratio:
        meta.update(
            reason="safety_guard", candidate_removed=removed,
            removal_ratio=ratio, max_remove_ratio=max_remove_ratio,
            elapsed=time.perf_counter() - t0,
        )
        return faces, meta

    filtered = faces[~remove]
    meta.update(
        enabled=True, reason="mixed_boundary_discontinuity", removed=removed,
        candidate_removed=removed, removal_ratio=ratio,
        output_faces=int(len(filtered)), edge_factor=edge_factor,
        min_edge=min_edge_len, min_z_jump=min_jump, min_slope=min_slope,
        elapsed=time.perf_counter() - t0,
    )
    return filtered, meta


def _compute_shading_geometry_backend(
        xyz_raw, classes_raw, visible_classes, azimuth, angle, ambient,
        max_edge_factor, single_class_max_edge, data_hash,
        representative_seed=None, boundary_flags=None, quality_mode="normal",
        source_global_indices=None, line_filter_active=False):
    profile_start = time.perf_counter()
    stage_start = profile_start
    timings = OrderedDict()

    def checkpoint(name):
        nonlocal stage_start
        now = time.perf_counter()
        timings[name] = now - stage_start
        stage_start = now

    nv = len(visible_classes)
    is_sc = (nv == 1)
    representative_cache_hit = bool(representative_seed)
    adaptive_meta = {"enabled": False, "reason": "representative_cache" if representative_cache_hit else "not_evaluated"}

    if representative_cache_hit:
        # PATCH 4A: reuse the exact final representative selection. The seed
        # is accepted only after geometry, visible-preset, and full
        # classification-content validation in the GUI thread. No point-count
        # reduction or alternate representative rule is used here.
        offset = representative_seed["offset"]
        unique_indices_global = representative_seed["unique_indices"]
        xyz_unique = representative_seed["xyz_unique"]
        spacing = float(representative_seed["spacing"])
        lod_factor = float(representative_seed.get("lod_factor", 1.0))
        base_grid_unique_count = int(representative_seed.get("base_grid_unique_count", len(xyz_unique)))
        # Shaded Class is faceted for every visibility combination, including
        # the all-classes preset.  Do not revive the former smooth policy from
        # a representative-selection cache created by an older build.
        smooth_all_classes = False
        precision = float(representative_seed.get("precision", 0.0))
        adaptive_meta = dict(representative_seed.get("adaptive_meta", {
            "enabled": False, "reason": "representative_cache"
        }))
        representative_boundary = representative_seed.get("representative_boundary")
        if representative_boundary is not None:
            representative_boundary = np.asarray(representative_boundary, dtype=bool).reshape(-1)
            if len(representative_boundary) != len(xyz_unique):
                representative_boundary = None

        # Keep the established profile field names stable. Cache lookup and
        # validation are profiled separately by the caller.
        for name in ("visibility_mask", "visible_indices", "visible_xyz_copy",
                     "finite_filter", "normalize_extent", "dedup_pass1",
                     "dedup_lod_extra", "adaptive_refinement"):
            timings[name] = 0.0
        stage_start = time.perf_counter()

        xr2 = xyz_unique[:, 0].max() - xyz_unique[:, 0].min()
        yr2 = xyz_unique[:, 1].max() - xyz_unique[:, 1].min()
        data_extent = max(xr2, yr2)
        xy = xyz_unique[:, :2]
        checkpoint("unique_materialize")
    else:
        classes = classes_raw.astype(np.int32)
        mc = int(np.max(classes)) + 1 if len(classes) > 0 else 256
        vl = np.zeros(mc, dtype=bool)
        for v in visible_classes:
            if v < mc:
                vl[v] = True
        vm = vl[classes]
        checkpoint("visibility_mask")

        vi = np.flatnonzero(vm)
        # Keep per-face lighting/class colour even when every loaded class is
        # visible. Previously this special-cased all classes to vertex-smoothed
        # shading, making the same mode change appearance when one class was
        # toggled off.
        smooth_all_classes = False
        checkpoint("visible_indices")
        if len(vi) < 3:
            timings["total_backend"] = time.perf_counter() - profile_start
            return {"empty": True, "profile_timings": timings}

        xv = xyz_raw[vi]
        checkpoint("visible_xyz_copy")

        finite_mask = np.isfinite(xv).all(axis=1)
        if not np.all(finite_mask):
            vi = vi[finite_mask]
            xv = xv[finite_mask]
        visible_boundary = None
        representative_boundary = None
        if boundary_flags is not None:
            try:
                bf = np.asarray(boundary_flags, dtype=bool).reshape(-1)
                if len(bf) == len(xyz_raw):
                    visible_boundary = bf[vi]
            except Exception:
                visible_boundary = None
        checkpoint("finite_filter")
        if len(vi) < 3:
            timings["total_backend"] = time.perf_counter() - profile_start
            return {"empty": True, "profile_timings": timings}

        offset = xv.min(axis=0)
        xyz = (xv - offset).astype(np.float64)
        xr = xyz[:, 0].max() - xyz[:, 0].min()
        yr = xyz[:, 1].max() - xyz[:, 1].min()
        area = max(xr * yr, 1.0)
        n_pts = len(xyz)
        natural_spacing = np.sqrt(area / n_pts)
        checkpoint("normalize_extent")

        quality_mode = str(quality_mode or "normal").strip().lower()
        if quality_mode not in ("fast", "normal", "slow"):
            quality_mode = "normal"
        adaptive_enabled = os.environ.get("NAKSHA_SHADING_ADAPTIVE", "1").strip().lower() not in ("0", "false", "off", "no")
        use_count_first = False
        precision = max(natural_spacing * 0.3, 0.005)
        lod_factor = 1.0

        if quality_mode == "slow":
            # MicroStation-style Slow: retain every eligible finite point.
            # The confirmation and memory warning live in Display Mode before
            # this deliberately expensive background operation starts.
            unique_local = np.arange(n_pts, dtype=np.int64)
            n_unique = n_pts
            base_grid_unique_count = n_pts
            checkpoint("dedup_pass1")
            checkpoint("dedup_lod_extra")
            adaptive_target = n_pts
            adaptive_meta = {
                "enabled": False, "reason": "slow_all_points",
                "base": n_pts, "extra": 0, "target": n_pts,
            }
            print(
                "SHADING_ADAPTIVE status=disabled reason=slow_all_points "
                f"base={n_pts} final={n_pts} target={n_pts}"
            )
        else:
            final_target = (
                _SHADING_FAST_TARGET if quality_mode == "fast"
                else int(os.environ.get(
                    "NAKSHA_SHADING_ADAPTIVE_TARGET", str(_ADAPTIVE_DEFAULT_TARGET)
                ))
            )
            TARGET_MAX = (
                max(3, final_target // 2) if HAS_TRIANGLE
                else min(final_target, 500_000)
            )

            # Count-first grid deduplication avoids materializing an oversized
            # representative array before the final precision is known.
            use_count_first = n_pts > int(TARGET_MAX * 1.30)
            if use_count_first:
                n_unique = _grid_unique_count_fast_at_precision(xyz, precision)
                unique_local = None
            else:
                unique_local = _grid_dedup_at_precision(xyz, precision)
                n_unique = len(unique_local)
            base_grid_unique_count = int(n_unique)
            checkpoint("dedup_pass1")

            if n_unique > TARGET_MAX:
                lod_factor = np.sqrt(n_unique / TARGET_MAX)
                lod_factor = min(max(lod_factor, 1.0), 5.0)
                precision2 = precision * lod_factor
                n_unique2 = _grid_unique_count_fast_at_precision(xyz, precision2)
                if n_unique2 > TARGET_MAX * 1.3:
                    lod_factor2 = np.sqrt(n_unique2 / TARGET_MAX)
                    precision3 = precision2 * lod_factor2
                    unique_local = _grid_dedup_at_precision(xyz, precision3)
                    n_unique = len(unique_local)
                    lod_factor *= lod_factor2
                    precision = precision3
                else:
                    unique_local = _grid_dedup_at_precision(xyz, precision2)
                    n_unique = len(unique_local)
                    precision = precision2
            elif unique_local is None:
                unique_local = _grid_dedup_at_precision(xyz, precision)
                n_unique = len(unique_local)
            checkpoint("dedup_lod_extra")

            adaptive_target = final_target
            if adaptive_enabled and HAS_TRIANGLE and visible_boundary is not None:
                unique_local, adaptive_meta = _adaptive_boundary_representatives(
                    xyz, unique_local, visible_boundary, precision, adaptive_target
                )
                print(
                    "SHADING_ADAPTIVE "
                    f"status={'enabled' if adaptive_meta.get('enabled') else 'fallback'} "
                    f"reason={adaptive_meta.get('reason')} "
                    f"base={adaptive_meta.get('base', len(unique_local))} "
                    f"boundary_points={adaptive_meta.get('boundary_points', 0)} "
                    f"fine_candidates={adaptive_meta.get('fine_candidates', 0)} "
                    f"extra={adaptive_meta.get('extra', 0)} "
                    f"final={len(unique_local)} target={adaptive_meta.get('target', adaptive_target)} "
                    f"elapsed={adaptive_meta.get('elapsed', 0.0)*1000.0:.1f}ms"
                )
            else:
                adaptive_meta = {
                    "enabled": False,
                    "reason": "disabled" if not adaptive_enabled else (
                        "triangle_unavailable" if not HAS_TRIANGLE else "boundary_unavailable"
                    ),
                    "base": len(unique_local), "extra": 0,
                    "target": adaptive_target,
                }
                print(
                    "SHADING_ADAPTIVE status=fallback "
                    f"reason={adaptive_meta['reason']} "
                    f"base={len(unique_local)} final={len(unique_local)}"
                )
        checkpoint("adaptive_refinement")

        xyz_unique = xyz[unique_local]
        selected_input_indices = vi[unique_local]
        if source_global_indices is not None:
            source_global_indices = np.asarray(
                source_global_indices, dtype=np.int64
            ).reshape(-1)
            if len(source_global_indices) != len(xyz_raw):
                raise ValueError(
                    "flight-line source/global index map length mismatch: "
                    f"{len(source_global_indices)} != {len(xyz_raw)}"
                )
            unique_indices_global = source_global_indices[selected_input_indices]
        else:
            unique_indices_global = np.asarray(
                selected_input_indices, dtype=np.int64
            )
        if visible_boundary is not None:
            representative_boundary = np.asarray(visible_boundary[unique_local], dtype=bool)

        xr2 = xyz_unique[:, 0].max() - xyz_unique[:, 0].min()
        yr2 = xyz_unique[:, 1].max() - xyz_unique[:, 1].min()
        data_extent = max(xr2, yr2)
        spacing = np.sqrt((xr2 * yr2) / max(len(xyz_unique), 1))
        xy = xyz_unique[:, :2]
        checkpoint("unique_materialize")

    faces = _do_triangulate(xy)
    checkpoint("delaunay")

    if len(faces) > 0:
        # Valid skinny faces belong to the Delaunay coverage. Spacing and
        # shape thresholds punch holes in irregularly sampled surfaces.
        min_area = 0.0
        min_aspect = 0.0
        if HAS_NUMBA:
            keep = _numba_degenerate_filter(
                faces, xy, float(min_area), float(min_aspect)
            )
            if np.any(~keep):
                faces = faces[keep]
        else:
            p0 = xy[faces[:, 0]]
            p1 = xy[faces[:, 1]]
            p2 = xy[faces[:, 2]]
            cz = ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) -
                  (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
            ta = np.abs(cz) * 0.5
            e0 = ((p1 - p0) ** 2).sum(1)
            e1 = ((p2 - p1) ** 2).sum(1)
            e2 = ((p0 - p2) ** 2).sum(1)
            me = np.maximum(np.maximum(e0, e1), e2)
            nd = (ta > min_area) & (ta / np.maximum(me, 1e-20) > min_aspect)
            if np.any(~nd):
                faces = faces[nd]
    checkpoint("degenerate_filter")

    # Keep continuous coverage unless the user explicitly requests a limit.
    explicit_edge = float(single_class_max_edge or 0.0)
    if len(faces) > 0 and explicit_edge > 0.0:
        faces = _filter_edges_by_absolute(faces, xy, explicit_edge)
        max_edge_factor = explicit_edge / max(spacing, 1e-9)
    checkpoint("edge_filter")

    faces, feature_meta = _feature_aware_filter_faces(
        faces, xyz_unique, representative_boundary, spacing
    )
    print(
        "SHADING_FEATURE_AWARE "
        f"status={'enabled' if feature_meta.get('enabled') else 'fallback'} "
        f"reason={feature_meta.get('reason')} "
        f"input_faces={feature_meta.get('input_faces', len(faces))} "
        f"removed={feature_meta.get('removed', 0)} "
        f"output_faces={feature_meta.get('output_faces', len(faces))} "
        f"ratio={feature_meta.get('removal_ratio', 0.0):.6f} "
        f"elapsed={feature_meta.get('elapsed', 0.0)*1000.0:.1f}ms"
    )
    checkpoint("feature_filter")

    if str(quality_mode or "normal").lower() == "slow" and len(faces) > 0:
        try:
            used = np.zeros(len(xyz_unique), dtype=np.bool_)
            face_chunk = 2_000_000
            for begin in range(0, len(faces), face_chunk):
                fc = faces[begin:begin + face_chunk]
                used[fc[:, 0]] = True
                used[fc[:, 1]] = True
                used[fc[:, 2]] = True
            orphan_count = int(len(used) - np.count_nonzero(used))
            coverage = 100.0 * (len(used) - orphan_count) / max(len(used), 1)
            print(
                "SHADING_POINT_COVERAGE mode=slow "
                f"input_vertices={len(used):,} referenced_vertices={len(used)-orphan_count:,} "
                f"orphan_vertices={orphan_count:,} coverage={coverage:.6f}%"
            )
        except Exception as exc:
            print(f"SHADING_POINT_COVERAGE status=unavailable reason={exc}")

    face_normals = None
    vertex_normals = None
    vertex_shade = None
    shade = None

    if len(faces) > 0:
        face_normals = _compute_face_normals(xyz_unique, faces)
    else:
        face_normals = np.array([]).reshape(0, 3)
    checkpoint("face_normals")

    if len(faces) > 0:
        shade = _compute_face_shade(
            xyz_unique, faces, azimuth, angle, ambient,
            face_normals=face_normals,
        )
    else:
        shade = np.array([])
    checkpoint("face_shade")

    if len(faces) > 0 and smooth_all_classes:
        vertex_normals = _compute_vertex_normals(
            xyz_unique, faces, face_normals,
        )
        checkpoint("vertex_normals")
        vertex_shade = _compute_shading(
            vertex_normals, azimuth, angle, ambient,
            z_values=xyz_unique[:, 2],
        )
        checkpoint("vertex_shade")
    else:
        vertex_normals = np.array([]).reshape(0, 3) if len(faces) == 0 else None
        vertex_shade = np.array([]) if len(faces) == 0 else None
        timings["vertex_normals"] = 0.0
        timings["vertex_shade"] = 0.0

    timings["total_backend"] = time.perf_counter() - profile_start

    return {
        "empty": False,
        "offset": offset,
        "unique_indices": unique_indices_global,
        "xyz_unique": xyz_unique,
        "xyz_final": xyz_unique + offset,
        "spacing": spacing,
        "max_edge_factor": max_edge_factor,
        "visible_classes_hash": hash(frozenset(visible_classes)),
        "n_visible_classes": nv,
        "visible_classes_set": visible_classes.copy(),
        "single_class_id": list(visible_classes)[0] if is_sc else None,
        "smooth_all_classes": smooth_all_classes,
        "faces": faces,
        "face_normals": face_normals,
        "vertex_normals": vertex_normals,
        "vertex_shade": vertex_shade,
        "shade": shade,
        "last_azimuth": azimuth,
        "last_angle": angle,
        "last_ambient": ambient,
        "data_hash": data_hash,
        "lod_factor": lod_factor,
        "dedup_strategy": ("representative_cache" if representative_cache_hit
                           else ("count_then_materialize" if use_count_first else "legacy_direct")),
        "representative_cache_hit": representative_cache_hit,
        "precision": precision,
        "base_grid_unique_count": base_grid_unique_count,
        "adaptive_meta": adaptive_meta,
        "adaptive_version": _ADAPTIVE_SHADING_VERSION,
        "representative_boundary": representative_boundary,
        "feature_meta": feature_meta,
        "feature_version": _FEATURE_AWARE_VERSION,
        "source_index_space": "canonical_global",
        "line_filter_active": bool(line_filter_active),
        "line_filter_version": _SHADING_LINE_FILTER_VERSION,
        "profile_timings": timings,
    }


def triangulate_with_triangle(xy):
    if not HAS_TRIANGLE: raise ImportError("no triangle")
    return tr.triangulate({'vertices': xy.astype(np.float64)}, 'Qz')['triangles'].astype(np.int32)


def triangulate_scipy_direct(xy):
    return Delaunay(xy).simplices.astype(np.int32)


def _do_triangulate(xy):
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3:
        return np.empty((0, 3), dtype=np.int32)
    # Guard native triangulators against NaN/Inf and degenerate lines.
    if not np.isfinite(xy).all():
        return np.empty((0, 3), dtype=np.int32)
    if np.ptp(xy[:, 0]) <= 1e-12 or np.ptp(xy[:, 1]) <= 1e-12:
        return np.empty((0, 3), dtype=np.int32)
    if HAS_TRIANGLE:
        try:
            return triangulate_with_triangle(xy)
        except Exception:
            pass
    try:
        return triangulate_scipy_direct(xy)
    except Exception:
        return np.empty((0, 3), dtype=np.int32)


def _filter_edges_by_absolute(faces, xy, max_edge_length):
    if len(faces) == 0: return faces
    if HAS_NUMBA:
        return faces[_numba_edge_filter(faces, xy, max_edge_length * max_edge_length)]
    v0, v1, v2 = xy[faces[:,0]], xy[faces[:,1]], xy[faces[:,2]]
    e0 = ((v1-v0)**2).sum(1); e1 = ((v2-v1)**2).sum(1); e2 = ((v0-v2)**2).sum(1)
    return faces[np.maximum(np.maximum(e0, e1), e2) <= max_edge_length**2]


def _filter_edges_3d_abs(faces, xyz, max_xy, max_slope_ratio=10.0):
    if len(faces) == 0: return faces
    xy = xyz[:, :2]
    if HAS_NUMBA:
        return faces[_numba_edge_filter(faces, xy, max_xy * max_xy)]
    v0, v1, v2 = xy[faces[:,0]], xy[faces[:,1]], xy[faces[:,2]]
    e0 = ((v1-v0)**2).sum(1); e1 = ((v2-v1)**2).sum(1); e2 = ((v0-v2)**2).sum(1)
    return faces[np.maximum(np.maximum(e0, e1), e2) <= max_xy**2]


def _compute_face_normals(xyz, faces):
    if len(faces) == 0: return np.array([]).reshape(0, 3)
    if HAS_NUMBA: return _compute_face_normals_fast(xyz, faces)
    p0, p1, p2 = xyz[faces[:,0]], xyz[faces[:,1]], xyz[faces[:,2]]
    fn = np.cross(p1-p0, p2-p0)
    l = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.maximum(l, 1e-10)
    m = (fn[:,2] < 0) & (np.abs(fn[:,2]) > 0.3); fn[m] *= -1
    return fn


def _compute_vertex_normals(xyz, faces, face_normals):
    if len(faces) == 0:
        vn = np.zeros((len(xyz), 3), dtype=np.float64); vn[:,2] = 1.0; return vn.astype(np.float32)
    if HAS_NUMBA:
        return _compute_vertex_normals_fast(xyz, faces, face_normals).astype(np.float32)
    n = len(xyz); vn = np.zeros((n, 3), dtype=np.float64)
    p0, p1, p2 = xyz[faces[:,0]], xyz[faces[:,1]], xyz[faces[:,2]]
    a = 0.5 * np.linalg.norm(np.cross(p1-p0, p2-p0), axis=1)
    w = face_normals * a[:, np.newaxis]
    np.add.at(vn, faces[:,0], w); np.add.at(vn, faces[:,1], w); np.add.at(vn, faces[:,2], w)
    l = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = vn / np.maximum(l, 1e-10); vn[l.ravel() < 1e-10] = [0,0,1]
    return vn.astype(np.float32)


def _compute_shading(normals, azimuth, angle, ambient, z_values=None):
    """Standard MicroStation/ArcGIS hillshade algorithm.
    
    Uses the standard formula:
      Hillshade = cos(zenith)*cos(slope) + sin(zenith)*sin(slope)*cos(azimuth_math - aspect)
    Which equals N . L where L is constructed with geographic-to-math azimuth conversion.
    """
    if len(normals) == 0: return np.array([])
    zenith_rad = np.radians(90.0 - angle)
    az_math_rad = np.radians(360.0 - azimuth + 90.0)
    lx = np.sin(zenith_rad) * np.cos(az_math_rad)
    ly = np.sin(zenith_rad) * np.sin(az_math_rad)
    lz = np.cos(zenith_rad)
    if HAS_NUMBA:
        return _compute_shading_fast(normals, z_values if z_values is not None else np.empty(0, dtype=np.float64),
                                     lx, ly, lz, ambient, 0., 0.)
    ld = np.array([lx, ly, lz], dtype=np.float64)
    NdL = (normals * ld).sum(1)
    ni = np.maximum(NdL, 0.0)
    ni = np.maximum(ni, ambient)
    return np.clip(ni, 0., 1.).astype(np.float32)


def _compute_face_shade(xyz, faces, azimuth, angle, ambient, face_normals=None, z_values=None):
    if xyz is None or faces is None or len(faces) == 0: return np.array([], dtype=np.float32)
    if face_normals is None or len(face_normals) != len(faces): face_normals = _compute_face_normals(xyz, faces)
    fz = _fast_centroids_z(xyz, faces) if HAS_NUMBA else xyz[faces, 2].mean(axis=1)
    return _compute_shading(face_normals, azimuth, angle, ambient, z_values=fz)


def _grid_unique_count_at_precision(xyz, precision):
    """Return the exact number of XY grid cells at ``precision``.

    This intentionally mirrors ``_grid_dedup_at_precision`` cell construction
    but does not sort by Z or build representative indices. It is used only
    when the first-pass representatives would be discarded by the existing
    adaptive LOD logic.
    """
    xy_grid = np.floor(xyz[:, :2] / precision).astype(np.int64)
    gx = xy_grid[:, 0] - xy_grid[:, 0].min()
    gy = xy_grid[:, 1] - xy_grid[:, 1].min()
    gy_span = int(gy.max()) + 1

    if gx.max() < 2**30 and gy_span < 2**30:
        grid_key = gx * gy_span + gy
        return int(np.unique(grid_key).size)

    # Extremely large coordinate spans cannot be packed safely into the same
    # scalar key. np.unique(axis=0) preserves the exact legacy cell count.
    return int(np.unique(xy_grid, axis=0).shape[0])


def _grid_unique_count_fast_at_precision(xyz, precision):
    """Return the exact XY-cell count using a dense Numba bitmap when safe.

    The result is identical to ``_grid_unique_count_at_precision``. The legacy
    NumPy unique implementation remains the fallback for sparse/extreme grids
    or accelerator failures. No representative points are selected here.
    """
    min_gx = int(np.floor(np.min(xyz[:, 0]) / precision))
    max_gx = int(np.floor(np.max(xyz[:, 0]) / precision))
    min_gy = int(np.floor(np.min(xyz[:, 1]) / precision))
    max_gy = int(np.floor(np.max(xyz[:, 1]) / precision))
    gx_span = max_gx - min_gx + 1
    gy_span = max_gy - min_gy + 1
    n_cells = gx_span * gy_span

    count_limit = int(os.environ.get(
        "NAKSHA_SHADING_DENSE_COUNT_MAX_CELLS", "50000000"
    ))
    # One uint8 per cell. The default limit bounds the temporary bitmap to
    # roughly 50 MB while covering the 17-32M-cell grids seen in production.
    if HAS_NUMBA and 0 < n_cells <= count_limit:
        try:
            t0 = time.perf_counter()
            count = int(_numba_dense_grid_unique_count(
                xyz, float(precision), min_gx, min_gy, gx_span, gy_span
            ))
            print(
                f"SHADING_DEDUP_COUNT strategy=numba_dense points={len(xyz)} "
                f"cells={n_cells} unique={count} "
                f"elapsed={(time.perf_counter()-t0)*1000:.1f}ms"
            )
            return count
        except Exception as exc:
            print(f"SHADING_DEDUP_COUNT fallback=numpy_unique reason={exc}")

    t0 = time.perf_counter()
    count = _grid_unique_count_at_precision(xyz, precision)
    print(
        f"SHADING_DEDUP_COUNT strategy=numpy_unique points={len(xyz)} "
        f"cells={n_cells} unique={count} "
        f"elapsed={(time.perf_counter()-t0)*1000:.1f}ms"
    )
    return count


def _grid_dedup_at_precision(xyz, precision):
    """Grid dedup at given precision. Returns local highest-Z indices.

    PATCH 6B uses an O(N) dense-cell Numba selector when the XY grid is
    compact enough. It preserves the same final cell count and highest-Z
    representative rule as the legacy global lexsort. The original path is
    retained as a fallback for sparse/extreme grids or any accelerator error.
    """
    xy_grid = np.floor(xyz[:, :2] / precision).astype(np.int64)
    min_gx = int(xy_grid[:, 0].min())
    min_gy = int(xy_grid[:, 1].min())
    gx = xy_grid[:, 0] - min_gx
    gy = xy_grid[:, 1] - min_gy
    gx_span = int(gx.max()) + 1
    gy_span = int(gy.max()) + 1

    # Dense terrain grids are normally only ~2-4M cells for the current
    # 1.5M representative target. Cap memory conservatively (~24 bytes/cell
    # including temporary/JIT arrays) and require reasonable occupancy.
    n_cells = gx_span * gy_span
    dense_limit = int(os.environ.get("NAKSHA_SHADING_DENSE_GRID_MAX_CELLS", "12000000"))
    occupancy_ok = n_cells <= max(len(xyz) * 2, 1)
    if HAS_NUMBA and 0 < n_cells <= dense_limit and occupancy_ok:
        try:
            t0 = time.perf_counter()
            result = _numba_dense_grid_highest_z(
                xyz, float(precision), min_gx, min_gy, gx_span, gy_span
            )
            print(
                f"SHADING_DEDUP_SELECTOR strategy=numba_dense points={len(xyz)} "
                f"cells={n_cells} selected={len(result)} "
                f"elapsed={(time.perf_counter()-t0)*1000:.1f}ms"
            )
            return result
        except Exception as exc:
            print(f"SHADING_DEDUP_SELECTOR fallback=legacy reason={exc}")

    if gx.max() < 2**30 and gy_span < 2**30:
        grid_key = gx * gy_span + gy
        sort_idx = np.lexsort((-xyz[:, 2], grid_key))
        sorted_keys = grid_key[sort_idx]
        unique_mask = np.empty(len(sorted_keys), dtype=bool)
        unique_mask[0] = True
        unique_mask[1:] = np.diff(sorted_keys) != 0
    else:
        sort_idx = np.lexsort((-xyz[:, 2], xy_grid[:, 1], xy_grid[:, 0]))
        xy_sorted = xy_grid[sort_idx]
        d = np.diff(xy_sorted, axis=0)
        unique_mask = np.concatenate([[True], (d[:, 0] != 0) | (d[:, 1] != 0)])
    print(
        f"SHADING_DEDUP_SELECTOR strategy=legacy_sort points={len(xyz)} "
        f"cells={n_cells} selected={int(np.count_nonzero(unique_mask))}"
    )
    return sort_idx[unique_mask]



if HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _compute_face_normals_fast(xyz, faces):
        n_faces = faces.shape[0]
        fn = np.empty((n_faces, 3), dtype=np.float64)
        for i in prange(n_faces):
            v0x, v0y, v0z = xyz[faces[i, 0]]
            v1x, v1y, v1z = xyz[faces[i, 1]]
            v2x, v2y, v2z = xyz[faces[i, 2]]
            ax, ay, az = v1x - v0x, v1y - v0y, v1z - v0z
            bx, by, bz = v2x - v0x, v2y - v0y, v2z - v0z
            nx = ay * bz - az * by
            ny = az * bx - ax * bz
            nz = ax * by - ay * bx
            length = np.sqrt(nx * nx + ny * ny + nz * nz)
            if length > 1e-10:
                nx, ny, nz = nx / length, ny / length, nz / length
            if nz < 0 and abs(nz) > 0.3:
                nx, ny, nz = -nx, -ny, -nz
            fn[i, 0] = nx; fn[i, 1] = ny; fn[i, 2] = nz
        return fn

    @njit(fastmath=True)
    def _compute_vertex_normals_fast(xyz, faces, fn):
        n_verts = xyz.shape[0]
        vn = np.zeros((n_verts, 3), dtype=np.float64)
        n_faces = faces.shape[0]
        for i in range(n_faces):
            v0, v1, v2 = faces[i, 0], faces[i, 1], faces[i, 2]
            p0x, p0y, p0z = xyz[v0]; p1x, p1y, p1z = xyz[v1]; p2x, p2y, p2z = xyz[v2]
            axx, ay, az = p1x - p0x, p1y - p0y, p1z - p0z
            bx, by, bz = p2x - p0x, p2y - p0y, p2z - p0z
            cx = ay * bz - az * by; cy = az * bx - axx * bz; cz = axx * by - ay * bx
            area = 0.5 * np.sqrt(cx * cx + cy * cy + cz * cz)
            wx, wy, wz = fn[i, 0] * area, fn[i, 1] * area, fn[i, 2] * area
            vn[v0, 0] += wx; vn[v0, 1] += wy; vn[v0, 2] += wz
            vn[v1, 0] += wx; vn[v1, 1] += wy; vn[v1, 2] += wz
            vn[v2, 0] += wx; vn[v2, 1] += wy; vn[v2, 2] += wz
        for i in range(n_verts):
            nx, ny, nz = vn[i, 0], vn[i, 1], vn[i, 2]
            length = np.sqrt(nx * nx + ny * ny + nz * nz)
            if length < 1e-10:
                vn[i, 0] = 0.0; vn[i, 1] = 0.0; vn[i, 2] = 1.0
            else:
                vn[i, 0] = nx / length; vn[i, 1] = ny / length; vn[i, 2] = nz / length
        return vn

    @njit(parallel=True, fastmath=True)
    def _compute_shading_fast(normals, z_values, lx, ly, lz, ambient, z_lo, z_range):
        """Standard MicroStation/ArcGIS hillshade: N.L dot product only."""
        n = normals.shape[0]
        shade = np.empty(n, dtype=np.float32)
        for i in prange(n):
            nx, ny, nz = normals[i, 0], normals[i, 1], normals[i, 2]
            ndotl = nx * lx + ny * ly + nz * lz
            intensity = max(ndotl, 0.0)
            if intensity < ambient:
                intensity = ambient
            shade[i] = min(max(intensity, 0.0), 1.0)
        return shade

    @njit(parallel=True, fastmath=True)
    def _fast_centroids_z(xyz, faces):
        n = faces.shape[0]
        fz = np.empty(n, dtype=np.float64)
        for i in prange(n):
            fz[i] = (xyz[faces[i, 0], 2] + xyz[faces[i, 1], 2] + xyz[faces[i, 2], 2]) / 3.0
        return fz

    @njit(parallel=True, cache=True)
    def _numba_affected_face_mask(faces, changed_vertex):
        n = faces.shape[0]
        affected = np.empty(n, dtype=np.bool_)
        for i in prange(n):
            affected[i] = (
                changed_vertex[faces[i, 0]]
                or changed_vertex[faces[i, 1]]
                or changed_vertex[faces[i, 2]]
            )
        return affected

    @njit(parallel=True, cache=True)
    def _numba_fill_affected_face_mask(faces, changed_vertex, affected):
        """Fill a caller-owned mask to avoid a 26M bool allocation per edit."""
        n = faces.shape[0]
        for i in prange(n):
            affected[i] = (
                changed_vertex[faces[i, 0]]
                or changed_vertex[faces[i, 1]]
                or changed_vertex[faces[i, 2]]
            )

    @njit(parallel=True, fastmath=True)
    def _numba_edge_filter(faces, xy, max_edge_sq):
        n = faces.shape[0]
        keep = np.empty(n, dtype=np.bool_)
        for i in prange(n):
            x0, y0 = xy[faces[i, 0], 0], xy[faces[i, 0], 1]
            x1, y1 = xy[faces[i, 1], 0], xy[faces[i, 1], 1]
            x2, y2 = xy[faces[i, 2], 0], xy[faces[i, 2], 1]
            e0 = (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0)
            e1 = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)
            e2 = (x0-x2)*(x0-x2) + (y0-y2)*(y0-y2)
            mx = e0
            if e1 > mx: mx = e1
            if e2 > mx: mx = e2
            keep[i] = mx <= max_edge_sq
        return keep

    @njit(parallel=True, fastmath=True)
    def _numba_feature_discontinuity_mask(faces, xyz, boundary, min_edge_sq, min_z_jump, min_slope):
        n = faces.shape[0]
        remove = np.zeros(n, dtype=np.bool_)
        for i in prange(n):
            a = faces[i, 0]; b = faces[i, 1]; c = faces[i, 2]
            bc = int(boundary[a]) + int(boundary[b]) + int(boundary[c])
            # Only mixed feature/non-feature triangles can bridge a detected break.
            if bc == 0 or bc == 3:
                continue
            x0, y0, z0 = xyz[a, 0], xyz[a, 1], xyz[a, 2]
            x1, y1, z1 = xyz[b, 0], xyz[b, 1], xyz[b, 2]
            x2, y2, z2 = xyz[c, 0], xyz[c, 1], xyz[c, 2]
            e0 = (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0)
            e1 = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)
            e2 = (x0-x2)*(x0-x2) + (y0-y2)*(y0-y2)
            mx = e0
            if e1 > mx: mx = e1
            if e2 > mx: mx = e2
            if mx <= min_edge_sq:
                continue
            zmin = z0
            if z1 < zmin: zmin = z1
            if z2 < zmin: zmin = z2
            zmax = z0
            if z1 > zmax: zmax = z1
            if z2 > zmax: zmax = z2
            dz = zmax - zmin
            if dz <= min_z_jump:
                continue
            slope = dz / (np.sqrt(mx) + 1e-12)
            if slope >= min_slope:
                remove[i] = True
        return remove

    @njit(parallel=True, fastmath=True)
    def _numba_degenerate_filter(faces, xy, min_area, min_aspect):
        n = faces.shape[0]
        keep = np.empty(n, dtype=np.bool_)
        for i in prange(n):
            x0, y0 = xy[faces[i, 0], 0], xy[faces[i, 0], 1]
            x1, y1 = xy[faces[i, 1], 0], xy[faces[i, 1], 1]
            x2, y2 = xy[faces[i, 2], 0], xy[faces[i, 2], 1]
            d0x, d0y = x1 - x0, y1 - y0
            cross_z = d0x * (y2 - y0) - d0y * (x2 - x0)
            tri_area = abs(cross_z) * 0.5
            d1x, d1y = x2 - x1, y2 - y1
            d2x, d2y = x0 - x2, y0 - y2
            e0 = d0x*d0x + d0y*d0y; e1 = d1x*d1x + d1y*d1y; e2 = d2x*d2x + d2y*d2y
            mx = e0
            if e1 > mx: mx = e1
            if e2 > mx: mx = e2
            keep[i] = (tri_area > min_area) and (tri_area / (mx + 1e-10) > min_aspect)
        return keep


    @njit(cache=True)
    def _numba_dense_grid_highest_z(xyz, precision, min_gx, min_gy, gx_span, gy_span):
        """Exact highest-Z representative per dense XY grid cell.

        Output order is ascending packed grid key, matching the legacy
        lexsort result. Equal-Z ties use the lowest source index so results
        are deterministic.
        """
        n_cells = gx_span * gy_span
        best_z = np.empty(n_cells, dtype=np.float64)
        best_idx = np.empty(n_cells, dtype=np.int64)
        for c in range(n_cells):
            best_z[c] = -np.inf
            best_idx[c] = -1
        for i in range(xyz.shape[0]):
            gx = int(np.floor(xyz[i, 0] / precision)) - min_gx
            gy = int(np.floor(xyz[i, 1] / precision)) - min_gy
            key = gx * gy_span + gy
            z = xyz[i, 2]
            old = best_idx[key]
            if z > best_z[key] or (z == best_z[key] and (old < 0 or i < old)):
                best_z[key] = z
                best_idx[key] = i
        count = 0
        for c in range(n_cells):
            if best_idx[c] >= 0:
                count += 1
        out = np.empty(count, dtype=np.int64)
        j = 0
        for c in range(n_cells):
            idx = best_idx[c]
            if idx >= 0:
                out[j] = idx
                j += 1
        return out


    @njit(cache=True)
    def _numba_dense_grid_unique_count(xyz, precision, min_gx, min_gy, gx_span, gy_span):
        """Exact occupied-cell count without sorting or representatives.

        This is used for adaptive-LOD decision passes. It preserves the exact
        cell count used by the legacy selector while avoiding a global sort and
        avoiding construction of representative indices that would be discarded.
        """
        occupied = np.zeros(gx_span * gy_span, dtype=np.uint8)
        for i in range(xyz.shape[0]):
            gx = int(np.floor(xyz[i, 0] / precision)) - min_gx
            gy = int(np.floor(xyz[i, 1] / precision)) - min_gy
            occupied[gx * gy_span + gy] = 1
        count = 0
        for i in range(occupied.shape[0]):
            count += occupied[i]
        return count

