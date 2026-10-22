"""VTK-free Surface math, preserved from surface_mode.py for native sections.

Only arrays enter and leave this module. The bounded NumPy paths use the same
selection, triangulation, ramp and lighting formulas as the legacy renderer.
"""
from __future__ import annotations
import os
import time
import numpy as np
from scipy.spatial import Delaunay, cKDTree
try:
    import triangle as _surface_triangle
    _SURFACE_HAS_TRIANGLE = True
except ImportError:
    _SURFACE_HAS_TRIANGLE = False
try:
    from numba import njit, prange
    _SURFACE_HAS_NUMBA = True
except ImportError:
    _SURFACE_HAS_NUMBA = False


_SURFACE_FAST_TARGET = 1_000_000


_SURFACE_NORMAL_TARGET = 3_000_000


def normalize_surface_quality(value):
    """Return persisted Surface quality key used by UI and caches."""
    quality = str(value or "normal").strip().lower()
    return quality if quality in ("fast", "normal", "slow") else "normal"


def _surface_quality_target(quality_mode: str, eligible_count: int) -> int:
    quality_mode = normalize_surface_quality(quality_mode)
    if quality_mode == "slow":
        return max(int(eligible_count), 3)
    if quality_mode == "fast":
        return min(max(int(eligible_count), 3), _SURFACE_FAST_TARGET)
    return min(max(int(eligible_count), 3), _SURFACE_NORMAL_TARGET)


def _surface_grid_shape(points: np.ndarray, precision: float):
    precision = max(float(precision), 1e-9)
    min_gx = int(np.floor(float(np.min(points[:, 0])) / precision))
    max_gx = int(np.floor(float(np.max(points[:, 0])) / precision))
    min_gy = int(np.floor(float(np.min(points[:, 1])) / precision))
    max_gy = int(np.floor(float(np.max(points[:, 1])) / precision))
    gx_span = max_gx - min_gx + 1
    gy_span = max_gy - min_gy + 1
    return min_gx, min_gy, gx_span, gy_span


def _surface_grid_count(points: np.ndarray, precision: float) -> int:
    min_gx, min_gy, gx_span, gy_span = _surface_grid_shape(points, precision)
    n_cells = int(gx_span) * int(gy_span)
    dense_limit = int(os.environ.get("NAKSHA_SURFACE_DENSE_COUNT_MAX_CELLS", "50000000"))
    if _SURFACE_HAS_NUMBA and 0 < n_cells <= dense_limit:
        try:
            return int(_surface_dense_grid_count(
                points, float(precision), min_gx, min_gy, gx_span, gy_span
            ))
        except Exception:
            pass

    grid = np.floor(points[:, :2] / float(precision)).astype(np.int64, copy=False)
    gx = grid[:, 0] - grid[:, 0].min()
    gy = grid[:, 1] - grid[:, 1].min()
    gy_span2 = int(gy.max()) + 1
    if gx.max() < 2**30 and gy_span2 < 2**30:
        return int(np.unique(gx * gy_span2 + gy).size)
    return int(np.unique(grid, axis=0).shape[0])


def _surface_grid_select(points: np.ndarray, precision: float, heights=None) -> np.ndarray:
    if heights is None:
        heights = points[:, 2] if points.shape[1] > 2 else np.zeros(len(points))
    min_gx, min_gy, gx_span, gy_span = _surface_grid_shape(points, precision)
    n_cells = int(gx_span) * int(gy_span)
    dense_limit = int(os.environ.get("NAKSHA_SURFACE_DENSE_GRID_MAX_CELLS", "12000000"))
    occupancy_ok = n_cells <= max(int(len(points) * 2), 1)
    if _SURFACE_HAS_NUMBA and 0 < n_cells <= dense_limit and occupancy_ok:
        try:
            result = _surface_dense_grid_first(
                points, float(precision), min_gx, min_gy, gx_span, gy_span, heights
            )
            # Keep deterministic source order after selecting each cell's highest return.
            return np.sort(result.astype(np.int64, copy=False))
        except Exception:
            pass

    grid = np.floor(points[:, :2] / float(precision)).astype(np.int64, copy=False)
    gx = grid[:, 0] - grid[:, 0].min()
    gy = grid[:, 1] - grid[:, 1].min()
    gy_span2 = int(gy.max()) + 1
    order = np.argsort(-np.asarray(heights), kind="stable")
    if gx.max() < 2**30 and gy_span2 < 2**30:
        _, idx = np.unique((gx * gy_span2 + gy)[order], return_index=True)
    else:
        _, idx = np.unique(grid[order], axis=0, return_index=True)
    return np.sort(order[idx].astype(np.int64, copy=False))


def _select_surface_representatives(
    local_points: np.ndarray,
    target_max: int,
    user_precision: float = 0.0,
    heights=None,
):
    """Count-first Surface LOD selection, analogous to shading representatives."""
    n = int(len(local_points))
    target_max = max(3, min(int(target_max), n))
    if n <= target_max:
        return np.arange(n, dtype=np.int64), 0.0, {
            "strategy": "all_eligible", "selected": n, "target": target_max,
        }

    xr = max(float(np.ptp(local_points[:, 0])), 1e-9)
    yr = max(float(np.ptp(local_points[:, 1])), 1e-9)
    area = max(xr * yr, 1.0)

    # A slightly denser first estimate prevents irregular project outlines from
    # undershooting the requested representative count.
    precision = max(float(user_precision or 0.0), np.sqrt(area / target_max) * 0.82, 1e-6)
    count = _surface_grid_count(local_points, precision)

    # Two exact count-first corrections are enough in production and avoid the
    # old "dedup millions first, then stride them away" cost.
    for _ in range(2):
        if count <= 0:
            break
        ratio = float(count) / float(target_max)
        if 0.82 <= ratio <= 1.08:
            break
        precision = max(float(user_precision or 0.0), precision * np.sqrt(max(ratio, 1e-9)))
        count = _surface_grid_count(local_points, precision)

    idx = _surface_grid_select(local_points, precision, heights)
    if len(idx) > target_max:
        # Deterministic, low-cost final cap. The grid step has already made the
        # candidates spatially representative; this only clips small overshoot.
        pick = np.linspace(0, len(idx) - 1, target_max, dtype=np.int64)
        idx = idx[pick]

    return idx.astype(np.int64, copy=False), float(precision), {
        "strategy": "grid_count_first",
        "selected": int(len(idx)),
        "target": int(target_max),
        "grid_count": int(count),
    }


def _triangulate_surface_xy(xy: np.ndarray):
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3:
        return np.empty((0, 3), dtype=np.int32), "none"
    if not np.isfinite(xy).all():
        return np.empty((0, 3), dtype=np.int32), "none"
    if np.ptp(xy[:, 0]) <= 1e-12 or np.ptp(xy[:, 1]) <= 1e-12:
        return np.empty((0, 3), dtype=np.int32), "none"

    if _SURFACE_HAS_TRIANGLE:
        try:
            result = _surface_triangle.triangulate(
                {"vertices": np.ascontiguousarray(xy, dtype=np.float64)}, "Qz"
            )
            faces = result.get("triangles") if isinstance(result, dict) else None
            if faces is not None and len(faces):
                return np.asarray(faces, dtype=np.int32), "triangle"
        except Exception as exc:
            print(f"SURFACE_TRIANGULATOR fallback=scipy reason={exc}")

    return Delaunay(xy).simplices.astype(np.int32, copy=False), "scipy"


def _normalize_surface_ramp(ramp):
    normalized = []
    if not ramp:
        return normalized

    for pos, color in ramp:
        try:
            rgb = _as_palette_color(color)
            normalized.append((float(pos), rgb))
        except Exception:
            continue
    return normalized


def _compute_surface_face_colors(
    points: np.ndarray,
    faces: np.ndarray,
    azimuth: float,
    angle: float,
    ambient: float,
    z_lo: float,
    z_hi: float,
    ramp=None,
) -> np.ndarray:
    """Compute Surface cell colors without multi-GB temporary triangle arrays.

    The mathematical result is unchanged: elevation ramp * Lambert shade.
    Numba is used when available; otherwise the same NumPy math is executed in
    bounded chunks so Slow/all-point Surface does not allocate ``points[faces]``
    for tens of millions of triangles at once.
    """
    if faces is None or len(faces) == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    pts = np.asarray(points, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)

    az = np.deg2rad(float(azimuth))
    el = np.deg2rad(float(angle))
    light = np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
        dtype=np.float64,
    )
    light /= np.linalg.norm(light) + 1e-12

    z = pts[:, 2]
    lo = float(z_lo)
    hi = float(z_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.percentile(z, 1.0))
        hi = float(np.percentile(z, 99.0))
        if hi <= lo:
            lo, hi = float(np.min(z)), float(np.max(z))
    if hi <= lo:
        hi = lo + 1.0

    valid_ramp = None
    if ramp:
        try:
            stops = np.asarray([float(pos) for pos, _ in ramp], dtype=np.float64)
            colors = np.asarray([list(color[:3]) for _, color in ramp], dtype=np.float64)
            if len(stops) >= 2 and len(colors) == len(stops):
                valid_ramp = (stops, colors)
        except Exception:
            valid_ramp = None

    if valid_ramp is None:
        stops = np.array([0.00, 0.25, 0.50, 0.75, 1.00], dtype=np.float64)
        colors = np.array(
            [
                [0, 0, 255],
                [0, 200, 255],
                [0, 255, 0],
                [255, 255, 0],
                [255, 80, 0],
            ],
            dtype=np.float64,
        )
    else:
        stops, colors = valid_ramp

    # Fast exact streaming kernel. This avoids ~5-7 GB of temporary arrays on
    # a 60-70M-face Slow Surface.
    if _SURFACE_HAS_NUMBA:
        try:
            return _surface_face_colors_numba(
                np.ascontiguousarray(pts),
                np.ascontiguousarray(f),
                float(light[0]),
                float(light[1]),
                float(light[2]),
                float(ambient),
                float(lo),
                float(hi),
                np.ascontiguousarray(stops),
                np.ascontiguousarray(colors),
            )
        except Exception as exc:
            print(f"SURFACE_ACCEL face_colors=fallback reason={type(exc).__name__}: {exc}")

    # Memory-bounded NumPy fallback. Keep the chunk modest because each chunk
    # still materializes three triangle point arrays and cross-product vectors.
    chunk = max(50_000, int(os.environ.get("NAKSHA_SURFACE_COLOR_CHUNK", "750000")))
    out = np.empty((len(f), 3), dtype=np.uint8)
    ambient_f = float(ambient)
    z_span = float(hi - lo)

    for begin in range(0, len(f), chunk):
        finish = min(begin + chunk, len(f))
        fc = f[begin:finish]
        p0 = pts[fc[:, 0]]
        p1 = pts[fc[:, 1]]
        p2 = pts[fc[:, 2]]
        v1 = p1 - p0
        v2 = p2 - p0
        normals = np.cross(v1, v2)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        shade = ambient_f + (1.0 - ambient_f) * np.clip(normals @ light, 0.0, 1.0)

        z_face = (p0[:, 2] + p1[:, 2] + p2[:, 2]) / 3.0
        norm = np.clip((z_face - lo) / z_span, 0.0, 1.0)
        rgb = np.column_stack(
            [
                np.interp(norm, stops, colors[:, 0]),
                np.interp(norm, stops, colors[:, 1]),
                np.interp(norm, stops, colors[:, 2]),
            ]
        )
        rgb *= shade[:, None]
        out[begin:finish] = np.clip(rgb, 0, 255).astype(np.uint8)

    return out


def _compute_surface_geometry_backend(
    xyz_all: np.ndarray,
    global_idx: np.ndarray,
    precision: float,
    target_max: int,
    max_edge: float,
    azimuth: float,
    angle: float,
    ambient: float,
    ramp,
    progress_callback=None,
    quality_mode="normal",
    z_bounds=None,
) -> dict:
    profile_start = time.perf_counter()
    timings = {}
    last = profile_start

    def _checkpoint(name):
        nonlocal last
        now = time.perf_counter()
        timings[name] = now - last
        last = now

    def _emit(value, message):
        if callable(progress_callback):
            try:
                progress_callback(int(value), str(message))
            except Exception:
                pass

    if global_idx is None or int(getattr(global_idx, "size", 0)) < 3:
        return {"empty": True, "profile_timings": timings}

    quality_mode = normalize_surface_quality(quality_mode)
    xyz = xyz_all[global_idx].astype(np.float64, copy=False)
    if len(xyz) < 3:
        return {"empty": True, "profile_timings": timings}
    _checkpoint("visible_xyz_copy")

    _emit(20, f"Filtering Surface points ({quality_mode})...")
    finite = np.isfinite(xyz).all(axis=1)
    if not np.all(finite):
        xyz = xyz[finite]
        global_idx = global_idx[finite]
    if len(xyz) < 3:
        return {"empty": True, "profile_timings": timings}
    _checkpoint("finite_filter")

    # Only XY needs normalization for representative selection/triangulation.
    # The previous Slow path materialized two separate Nx3 translated arrays
    # (``local`` and ``local_pts``) before making the Nx2 triangulation buffer.
    # At 34M points that was ~2.2 GB of avoidable memory traffic.  Preserve the
    # exact same local XY coordinates while keeping global XYZ only once.
    offset = xyz.min(axis=0)
    max_xyz = xyz.max(axis=0)
    xy_range = max_xyz[:2] - offset[:2]
    area = max(float(xy_range[0] * xy_range[1]), 1.0)
    spacing = float(np.sqrt(area / max(len(xyz), 1)))
    local_xy = np.ascontiguousarray(xyz[:, :2] - offset[:2], dtype=np.float64)
    _checkpoint("normalize_extent")

    target_max = _surface_quality_target(quality_mode, len(xyz))
    user_precision = max(float(precision or 0.0), 0.0)

    _emit(35, f"Selecting {quality_mode.title()} Surface representatives...")
    if quality_mode == "slow":
        # Do not allocate a 26M-entry arange just to select every point.
        unique_local_idx = None
        selected_precision = 0.0
        rep_meta = {
            "strategy": "all_points",
            "selected": int(len(xyz)),
            "target": int(target_max),
        }
    elif len(xyz) <= target_max:
        unique_local_idx = np.arange(len(xyz), dtype=np.int64)
        selected_precision = 0.0
        rep_meta = {
            "strategy": "under_target",
            "selected": int(len(xyz)),
            "target": int(target_max),
        }
    else:
        # Keep compact local XY and pass authoritative Z separately so grid
        # representatives retain peaks without allocating another Nx3 array.
        unique_local_idx, selected_precision, rep_meta = _select_surface_representatives(
            local_xy, target_max, user_precision, heights=xyz[:, 2]
        )
    _checkpoint("representative_select")

    if unique_local_idx is not None and len(unique_local_idx) < 3:
        return {"empty": True, "profile_timings": timings}

    if unique_local_idx is None or len(unique_local_idx) == len(xyz):
        # Avoid a second full copy in Slow/all-points mode.
        pts = xyz
        xy = local_xy
        unique_global_idx = global_idx.astype(np.int64, copy=False)
    else:
        pts = xyz[unique_local_idx].copy()
        xy = np.ascontiguousarray(local_xy[unique_local_idx], dtype=np.float64)
        unique_global_idx = global_idx[unique_local_idx].astype(np.int64, copy=False)
        del local_xy
    _checkpoint("representative_materialize")

    # Use the highest return at duplicate plan positions, including Slow mode.
    order = np.lexsort((-pts[:, 2], xy[:, 1], xy[:, 0]))
    ordered_xy = xy[order]
    first = np.r_[True, np.any(ordered_xy[1:] != ordered_xy[:-1], axis=1)]
    if not first.all():
        keep = np.sort(order[first])
        pts, xy = pts[keep], xy[keep]
        unique_global_idx = unique_global_idx[keep]
    del order, ordered_xy, first

    _emit(55, f"Triangulating Surface mesh ({quality_mode})...")
    faces, triangulator = _triangulate_surface_xy(xy)
    _checkpoint("delaunay")
    if len(faces) == 0:
        return {"empty": True, "profile_timings": timings}

    max_edge = float(max_edge or 0.0)
    # Zero means continuous coverage; only an explicit edge limit clips faces.
    _emit(75, "Finalizing Surface triangles...")
    faces = _filter_long_edges(faces, xy, max_edge)
    _checkpoint("edge_filter")
    if len(faces) == 0:
        return {"empty": True, "profile_timings": timings}

    if z_bounds is not None and len(z_bounds) >= 2:
        z_lo, z_hi = float(z_bounds[0]), float(z_bounds[1])
    else:
        z_lo = float(np.percentile(xyz_all[:, 2], 1.0))
        z_hi = float(np.percentile(xyz_all[:, 2], 99.0))
    _checkpoint("z_bounds")

    _emit(88, "Applying Surface elevation shading...")
    colors = _compute_surface_face_colors(
        pts, faces, azimuth, angle, ambient, z_lo, z_hi, ramp=ramp
    )
    _checkpoint("face_colors")
    timings["total_backend"] = time.perf_counter() - profile_start

    return {
        "empty": False,
        "points": pts,
        "faces": faces,
        "colors": colors,
        "unique_global_idx": unique_global_idx,
        "z_lo": z_lo,
        "z_hi": z_hi,
        "quality_mode": quality_mode,
        "triangulator": triangulator,
        "representative_meta": rep_meta,
        "selected_precision": selected_precision,
        "profile_timings": timings,
    }


def _as_palette_color(color):
    try:
        if hasattr(color, "red"):
            return (int(color.red()), int(color.green()), int(color.blue()))
        return tuple(int(c) for c in color[:3])
    except Exception:
        return (128, 128, 128)


def _filter_long_edges(faces: np.ndarray, xy: np.ndarray, max_edge: float) -> np.ndarray:
    """Filter triangles by absolute edge length without giant edge temporaries."""
    if faces is None or len(faces) == 0:
        return np.empty((0, 3), dtype=np.int32)
    if max_edge <= 0:
        return np.asarray(faces, dtype=np.int32)

    f = np.asarray(faces, dtype=np.int32)
    xy_arr = np.asarray(xy, dtype=np.float64)
    max_edge_sq = float(max_edge) * float(max_edge)

    if _SURFACE_HAS_NUMBA:
        try:
            keep = _surface_edge_keep_mask_numba(
                np.ascontiguousarray(f),
                np.ascontiguousarray(xy_arr),
                max_edge_sq,
            )
            return f[keep].astype(np.int32, copy=False)
        except Exception as exc:
            print(f"SURFACE_ACCEL edge_filter=fallback reason={type(exc).__name__}: {exc}")

    # Chunked NumPy fallback: exact squared-distance predicate, no sqrt needed.
    chunk = max(100_000, int(os.environ.get("NAKSHA_SURFACE_EDGE_CHUNK", "1500000")))
    keep = np.empty(len(f), dtype=bool)
    for begin in range(0, len(f), chunk):
        finish = min(begin + chunk, len(f))
        fc = f[begin:finish]
        p0 = xy_arr[fc[:, 0]]
        p1 = xy_arr[fc[:, 1]]
        p2 = xy_arr[fc[:, 2]]
        d01 = np.sum((p0 - p1) * (p0 - p1), axis=1)
        d12 = np.sum((p1 - p2) * (p1 - p2), axis=1)
        d20 = np.sum((p2 - p0) * (p2 - p0), axis=1)
        keep[begin:finish] = (
            (d01 <= max_edge_sq)
            & (d12 <= max_edge_sq)
            & (d20 <= max_edge_sq)
        )
    return f[keep].astype(np.int32, copy=False)



if _SURFACE_HAS_NUMBA:
    @njit(cache=True)
    def _surface_dense_grid_count(points, precision, min_gx, min_gy, gx_span, gy_span):
        occupied = np.zeros(gx_span * gy_span, dtype=np.uint8)
        for i in range(points.shape[0]):
            gx = int(np.floor(points[i, 0] / precision)) - min_gx
            gy = int(np.floor(points[i, 1] / precision)) - min_gy
            occupied[gx * gy_span + gy] = 1
        count = 0
        for i in range(occupied.shape[0]):
            count += occupied[i]
        return count


    @njit(cache=True)
    def _surface_dense_grid_first(points, precision, min_gx, min_gy, gx_span, gy_span, heights=None):
        """Select the highest source point in every occupied XY grid cell."""
        best_idx = np.empty(gx_span * gy_span, dtype=np.int64)
        for c in range(best_idx.shape[0]):
            best_idx[c] = -1
        for i in range(points.shape[0]):
            gx = int(np.floor(points[i, 0] / precision)) - min_gx
            gy = int(np.floor(points[i, 1] / precision)) - min_gy
            key = gx * gy_span + gy
            previous = best_idx[key]
            if previous < 0:
                best_idx[key] = i
            elif heights is not None:
                if heights[i] > heights[previous]:
                    best_idx[key] = i
            elif points.shape[1] > 2 and points[i, 2] > points[previous, 2]:
                best_idx[key] = i
        count = 0
        for c in range(best_idx.shape[0]):
            if best_idx[c] >= 0:
                count += 1
        out = np.empty(count, dtype=np.int64)
        j = 0
        for c in range(best_idx.shape[0]):
            idx = best_idx[c]
            if idx >= 0:
                out[j] = idx
                j += 1
        return out



if _SURFACE_HAS_NUMBA:
    @njit(cache=True, parallel=True)
    def _surface_edge_keep_mask_numba(faces, xy, max_edge_sq):
        """Exact long-edge predicate without allocating three Nx2 edge arrays."""
        n = faces.shape[0]
        keep = np.empty(n, dtype=np.bool_)
        for i in prange(n):
            i0 = int(faces[i, 0])
            i1 = int(faces[i, 1])
            i2 = int(faces[i, 2])

            dx01 = xy[i0, 0] - xy[i1, 0]
            dy01 = xy[i0, 1] - xy[i1, 1]
            dx12 = xy[i1, 0] - xy[i2, 0]
            dy12 = xy[i1, 1] - xy[i2, 1]
            dx20 = xy[i2, 0] - xy[i0, 0]
            dy20 = xy[i2, 1] - xy[i0, 1]

            e01 = dx01 * dx01 + dy01 * dy01
            e12 = dx12 * dx12 + dy12 * dy12
            e20 = dx20 * dx20 + dy20 * dy20
            keep[i] = (
                e01 <= max_edge_sq
                and e12 <= max_edge_sq
                and e20 <= max_edge_sq
            )
        return keep


    @njit(cache=True, parallel=True)
    def _surface_face_colors_numba(
        points,
        faces,
        light_x,
        light_y,
        light_z,
        ambient,
        z_lo,
        z_hi,
        stops,
        ramp_colors,
    ):
        """Compute the exact elevation-ramp + Lambert color in one streaming pass."""
        n = faces.shape[0]
        out = np.empty((n, 3), dtype=np.uint8)
        denom_z = z_hi - z_lo
        nstops = stops.shape[0]

        for i in prange(n):
            ia = int(faces[i, 0])
            ib = int(faces[i, 1])
            ic = int(faces[i, 2])

            ax = points[ia, 0]
            ay = points[ia, 1]
            az = points[ia, 2]
            bx = points[ib, 0]
            by = points[ib, 1]
            bz = points[ib, 2]
            cx = points[ic, 0]
            cy = points[ic, 1]
            cz = points[ic, 2]

            v1x = bx - ax
            v1y = by - ay
            v1z = bz - az
            v2x = cx - ax
            v2y = cy - ay
            v2z = cz - az

            nx = v1y * v2z - v1z * v2y
            ny = v1z * v2x - v1x * v2z
            nz = v1x * v2y - v1y * v2x
            nn = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
            dot = (nx * light_x + ny * light_y + nz * light_z) / nn
            if dot < 0.0:
                dot = 0.0
            elif dot > 1.0:
                dot = 1.0
            shade = ambient + (1.0 - ambient) * dot

            zn = ((az + bz + cz) / 3.0 - z_lo) / denom_z
            if zn < 0.0:
                zn = 0.0
            elif zn > 1.0:
                zn = 1.0

            # np.interp-equivalent linear interpolation over a tiny ramp.
            if zn <= stops[0]:
                r = ramp_colors[0, 0]
                g = ramp_colors[0, 1]
                b = ramp_colors[0, 2]
            elif zn >= stops[nstops - 1]:
                r = ramp_colors[nstops - 1, 0]
                g = ramp_colors[nstops - 1, 1]
                b = ramp_colors[nstops - 1, 2]
            else:
                hi = 1
                while hi < nstops and zn > stops[hi]:
                    hi += 1
                lo = hi - 1
                span = stops[hi] - stops[lo]
                if span <= 1e-15:
                    t = 0.0
                else:
                    t = (zn - stops[lo]) / span
                r = ramp_colors[lo, 0] + (ramp_colors[hi, 0] - ramp_colors[lo, 0]) * t
                g = ramp_colors[lo, 1] + (ramp_colors[hi, 1] - ramp_colors[lo, 1]) * t
                b = ramp_colors[lo, 2] + (ramp_colors[hi, 2] - ramp_colors[lo, 2]) * t

            r *= shade
            g *= shade
            b *= shade
            if r < 0.0:
                r = 0.0
            elif r > 255.0:
                r = 255.0
            if g < 0.0:
                g = 0.0
            elif g > 255.0:
                g = 255.0
            if b < 0.0:
                b = 0.0
            elif b > 255.0:
                b = 255.0

            out[i, 0] = np.uint8(r)
            out[i, 1] = np.uint8(g)
            out[i, 2] = np.uint8(b)

        return out

