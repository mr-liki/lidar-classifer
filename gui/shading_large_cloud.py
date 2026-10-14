"""Bounded-memory Slow shading for very large point clouds.

The ordinary shading cache intentionally owns one global TIN.  Native
triangulators need several temporary copies of that TIN and can terminate the
process before Python can raise MemoryError.  This module builds the same
all-input-point presentation as independent spatial TIN tiles with a one-cell
halo.  Faces are owned by their centroid, so adjacent tiles overlap for
triangulation without drawing duplicate faces.
"""
from __future__ import annotations

import gc
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import vtk
from vtkmodules.util import numpy_support


_ACTOR_PREFIX = "shaded_mesh_large_tile_"
_MAX_RESIDENT_PRESETS = 2


def _resident_preset_cache(app):
    cache = getattr(app, "_large_shading_preset_cache", None)
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict()
        app._large_shading_preset_cache = cache
    return cache


def clear_large_shading_preset_cache(app):
    """Release detached large-mesh bundles at a dataset lifecycle boundary."""
    cache = getattr(app, "_large_shading_preset_cache", None)
    released = len(cache) if isinstance(cache, dict) else 0
    if isinstance(cache, dict):
        for bundle in cache.values():
            if isinstance(bundle, dict):
                bundle["actors"] = []
        cache.clear()
    app._large_shading_preset_cache = OrderedDict()
    if released:
        print(
            "SHADING_LARGE_PRESET_CACHE status=cleared "
            f"presets={released}"
        )


def _detach_active_preset(app):
    """Detach the active tile graph and retain its NumPy/VTK geometry."""
    actors = list(getattr(app, "_large_shading_tile_actors", None) or [])
    signature = getattr(app, "_large_shading_geometry_signature", None)
    if not actors or signature is None:
        return False
    cache = _resident_preset_cache(app)
    bundle = {
        "actors": actors,
        "geometry_signature": signature,
        "visible_classes": tuple(getattr(
            app, "_large_shading_visible_classes", (),
        )),
        "classification_revision": int(getattr(
            app, "_large_shading_classification_revision", 0,
        )),
        "palette_signature": getattr(
            app, "_large_shading_palette_signature", None,
        ),
        "lighting_signature": getattr(
            app, "_large_shading_lighting_signature", None,
        ),
        "build_state": getattr(app, "_large_shading_build_state", None),
        "pending_color_indices": np.empty(0, dtype=np.int64),
        "pending_topology_indices": np.empty(0, dtype=np.int64),
    }
    plotter = app.vtk_widget
    for actor in actors:
        try:
            plotter.remove_actor(actor, render=False)
        except Exception:
            try:
                tile_id = int(getattr(actor, "_naksha_large_tile_id", -1))
                plotter.remove_actor(f"{_ACTOR_PREFIX}{tile_id:04d}", render=False)
            except Exception:
                pass
        try:
            actor.SetVisibility(False)
        except Exception:
            pass
    cache[signature] = bundle
    cache.move_to_end(signature)
    while len(cache) > _MAX_RESIDENT_PRESETS:
        _old_key, old = cache.popitem(last=False)
        old["actors"] = []
    app._large_shading_tile_actors = []
    print(
        "SHADING_LARGE_PRESET_CACHE status=stored "
        f"classes={len(bundle['visible_classes'])} tiles={len(actors)} "
        f"resident_presets={len(cache)}"
    )
    return True


def _mark_inactive_presets_dirty(app, indices, old_classes, new_classes):
    """Journal sparse edits for detached topology presets."""
    ci = np.unique(np.asarray(indices, dtype=np.int64).ravel())
    if ci.size == 0:
        return
    old = None if old_classes is None else np.asarray(old_classes).ravel()
    new = None if new_classes is None else np.asarray(new_classes).ravel()
    known = set(int(c) for c in (getattr(app, "class_palette", {}) or {}))
    for bundle in _resident_preset_cache(app).values():
        prior = np.asarray(bundle.get("pending_color_indices", ()), dtype=np.int64)
        bundle["pending_color_indices"] = np.union1d(prior, ci)
        visible = set(int(c) for c in bundle.get("visible_classes", ()))
        topology_dirty = False
        if old is not None and new is not None and old.size == new.size == ci.size:
            va = np.asarray(sorted(visible), dtype=np.int64)
            topology_dirty = bool(np.any(np.isin(old, va) != np.isin(new, va)))
        elif not known or not known.issubset(visible):
            topology_dirty = True
        if topology_dirty:
            prior_topology = np.asarray(
                bundle.get("pending_topology_indices", ()), dtype=np.int64,
            )
            bundle["pending_topology_indices"] = np.union1d(
                prior_topology, ci,
            )


def _restore_resident_preset(app, signature, azimuth, angle, ambient):
    """Reattach a cached tile graph and replay its sparse dirty journal."""
    cache = _resident_preset_cache(app)
    bundle = cache.pop(signature, None)
    if not bundle:
        return False
    actors = list(bundle.get("actors") or [])
    if not actors:
        return False
    plotter = app.vtk_widget
    for actor in actors:
        tile_id = int(getattr(actor, "_naksha_large_tile_id", 0))
        plotter.add_actor(
            actor, name=f"{_ACTOR_PREFIX}{tile_id:04d}",
            reset_camera=False, render=False, remove_existing_actor=True,
        )
        actor.SetVisibility(True)

    app._large_shading_tile_actors = actors
    app._large_shading_geometry_signature = signature
    app._large_shading_visible_classes = bundle["visible_classes"]
    app._large_shading_classification_revision = bundle[
        "classification_revision"
    ]
    app._large_shading_palette_signature = bundle["palette_signature"]
    app._large_shading_lighting_signature = bundle["lighting_signature"]
    app._large_shading_build_state = bundle["build_state"]
    app._large_shading_tiled = True
    app._shaded_mesh_actor = actors[0]

    requested_palette = large_shading_palette_signature(
        app, bundle["visible_classes"],
    )
    palette_changed = requested_palette != bundle.get("palette_signature")
    if palette_changed:
        classes_raw = np.asarray(app.data["classification"])
        lut = _palette_color_lut(
            app, classes_raw, set(bundle["visible_classes"]),
        )
        for actor in actors:
            owners = getattr(actor, "_naksha_large_array_owners", None)
            source_ids = getattr(actor, "_naksha_large_source_indices", None)
            if owners is None or source_ids is None:
                return False
            colors = owners[2]
            tile_classes = np.asarray(classes_raw[source_ids], dtype=np.int64)
            colors[:] = lut[np.clip(tile_classes, 0, len(lut) - 1)]
            poly = actor.GetMapper().GetInput()
            poly.GetPointData().GetScalars().Modified()
            poly.Modified()
        app._large_shading_palette_signature = requested_palette

    topology_ci = np.asarray(
        bundle.get("pending_topology_indices", ()), dtype=np.int64,
    )
    color_ci = np.asarray(
        bundle.get("pending_color_indices", ()), dtype=np.int64,
    )
    if topology_ci.size:
        if not _refresh_large_shading_topology_local(
            app, topology_ci, operation="preset_cache_restore",
        ):
            return False
    elif color_ci.size:
        classes = np.asarray(app.data["classification"])[color_ci]
        if not refresh_large_shading_sparse(
            app, color_ci, old_classes=classes, new_classes=classes,
            operation="preset_cache_restore",
        ):
            return False

    current_actors = list(getattr(app, "_large_shading_tile_actors", None) or [])
    for actor in current_actors:
        from gui.shading_display import _configure_microstation_color_blend_lighting
        _configure_microstation_color_blend_lighting(app, actor)
    app._large_shading_lighting_signature = (
        float(azimuth), float(angle), float(ambient),
    )
    app._large_shading_classification_revision = int(
        getattr(app, "classification_revision", 0) or 0
    )
    _mark_large_shading_active(app, current_actors)
    plotter.render()
    print(
        "SHADING_LARGE_PRESET_CACHE status=restored "
        f"tiles={len(current_actors)} color_dirty={color_ci.size} "
        f"topology_dirty={topology_ci.size} "
        f"palette_recolor={int(palette_changed)} full_rebuild=0"
    )
    return True


def _large_shading_is_active(app):
    return str(getattr(app, "display_mode", "") or "").lower() == "shaded_class"


def _mark_large_shading_active(app, actors):
    """Synchronize resident-cache state after a tiled shading activation."""
    app._shading_parked = False
    app._shading_parked_actor_visibility = {
        f"{_ACTOR_PREFIX}{int(getattr(actor, '_naksha_large_tile_id', i)):04d}": True
        for i, actor in enumerate(actors)
    }
    for actor in actors:
        try:
            actor.SetVisibility(True)
        except Exception:
            pass


def _palette_color_lut(app, classes, visible_classes):
    """Build a compact uint8 class palette for the current display state."""
    palette = getattr(app, "class_palette", {}) or {}
    class_max = int(np.max(classes)) if np.size(classes) else 0
    max_code = max([255, class_max] + [int(c) for c in visible_classes] +
                   [int(c) for c in palette.keys()])
    lut = np.zeros((max_code + 1, 3), dtype=np.uint8)
    for code, entry in palette.items():
        ci = int(code)
        if 0 <= ci <= max_code and ci in visible_classes:
            lut[ci] = np.asarray(
                entry.get("color", (128, 128, 128)), dtype=np.uint8,
            )[:3]
    return lut


def large_shading_palette_signature(app, visible_classes):
    """Return the visible class colours that define a tiled presentation."""
    palette = getattr(app, "class_palette", {}) or {}
    signature = []
    for code in sorted(int(c) for c in visible_classes):
        entry = palette.get(code, {}) or {}
        color = np.asarray(
            entry.get("color", (128, 128, 128)), dtype=np.uint8,
        ).reshape(-1)
        signature.append((code, tuple(int(v) for v in color[:3])))
    return tuple(signature)


def refresh_large_shading_sparse(
    app, changed_indices, *, old_classes=None, new_classes=None,
    operation="classification",
):
    """Patch changed point RGB in resident large-cloud tiles.

    A source point can occur in several halo tiles.  Each tile keeps its sorted
    canonical source IDs, allowing O(tiles * changed * log(tile)) lookup without
    allocating a 47-million-entry reverse map.  Connectivity and normals remain
    resident because class-only edits do not change the TIN when visibility
    membership is stable.
    """
    if not bool(getattr(app, "_large_shading_tiled", False)):
        return False
    actors = list(getattr(app, "_large_shading_tile_actors", None) or [])
    if not actors:
        return False
    data = getattr(app, "data", None)
    if not isinstance(data, dict) or data.get("classification") is None:
        return False

    ci = np.asarray(changed_indices, dtype=np.int64).ravel()
    if ci.size == 0:
        return True
    ci = np.unique(ci)
    classes = np.asarray(data["classification"])
    ci = ci[(ci >= 0) & (ci < len(classes))]
    if ci.size == 0:
        return True

    if operation != "preset_cache_restore":
        _mark_inactive_presets_dirty(app, ci, old_classes, new_classes)

    visible = set(int(c) for c in getattr(
        app, "_large_shading_visible_classes", (),
    ))
    if not visible:
        return False

    # Explicit history data is the strongest topology contract.  A class edit
    # that crosses the visible/hidden boundary requires a local TIN rebuild.
    if old_classes is not None and new_classes is not None:
        old = np.asarray(old_classes).ravel()
        new = np.asarray(new_classes).ravel()
        if old.size == new.size == ci.size:
            va = np.asarray(sorted(visible), dtype=np.int64)
            if np.any(np.isin(old, va) != np.isin(new, va)):
                return _refresh_large_shading_topology_local(
                    app, ci, operation=operation,
                )
        elif old.size or new.size:
            return False
    else:
        known = set(int(c) for c in (getattr(app, "class_palette", {}) or {}))
        if not known or not known.issubset(visible):
            return False

    current = np.asarray(classes[ci], dtype=np.int64)
    lut = _palette_color_lut(app, current, visible)
    colors_for_changed = lut[np.clip(current, 0, len(lut) - 1)]
    touched_tiles = touched_vertices = 0
    started = time.perf_counter()

    for actor in actors:
        source_ids = getattr(actor, "_naksha_large_source_indices", None)
        owners = getattr(actor, "_naksha_large_array_owners", None)
        if source_ids is None or owners is None:
            return False
        source_ids = np.asarray(source_ids)
        positions = np.searchsorted(source_ids, ci)
        in_range = positions < source_ids.size
        if not np.any(in_range):
            continue
        changed_pos = np.flatnonzero(in_range)
        local_pos = positions[in_range]
        matched = source_ids[local_pos] == ci[changed_pos]
        if not np.any(matched):
            continue
        changed_pos = changed_pos[matched]
        local_pos = local_pos[matched]
        colors = owners[2]
        colors[local_pos] = colors_for_changed[changed_pos]
        poly = actor.GetMapper().GetInput()
        scalars = poly.GetPointData().GetScalars()
        scalars.Modified()
        poly.Modified()
        actor.GetMapper().Modified()
        touched_tiles += 1
        touched_vertices += int(local_pos.size)

    if touched_tiles == 0:
        return False
    app._large_shading_classification_revision = int(
        getattr(app, "classification_revision", 0) or 0
    )
    if _large_shading_is_active(app):
        app.vtk_widget.render()
    print(
        "SHADING_LARGE_SPARSE status=patched topology_rebuild=0 "
        f"operation={operation} changed={ci.size} tiles={touched_tiles} "
        f"tile_vertices={touched_vertices} "
        f"elapsed={(time.perf_counter()-started)*1000:.1f}ms"
    )
    return True


def large_slow_threshold() -> int:
    raw = str(os.environ.get("NAKSHA_SHADING_TILED_THRESHOLD", "8000000"))
    try:
        return max(1_000_000, int(raw.replace(",", "")))
    except Exception:
        return 8_000_000


def should_use_large_slow_path(quality_mode, point_count) -> bool:
    return str(quality_mode).lower() == "slow" and int(point_count) >= large_slow_threshold()


def _remove_old_tiles(app):
    plotter = app.vtk_widget
    for name in list((getattr(plotter, "actors", {}) or {}).keys()):
        actor = (getattr(plotter, "actors", {}) or {}).get(name)
        if str(name).startswith(_ACTOR_PREFIX) or bool(
            getattr(actor, "_is_shading_mesh", False)
        ):
            try:
                plotter.remove_actor(name, render=False)
            except Exception:
                pass
    app._large_shading_tile_actors = []


def _release_full_point_tiles(app):
    """Release duplicate full-resolution VTK point buffers, retaining LOD."""
    try:
        from gui.unified_actor_manager import _phase2c2_evict_full_tile
        tile_ids = list((getattr(app, "_main_tile_actors", {}) or {}).keys())
        for tile_id in tile_ids:
            _phase2c2_evict_full_tile(app, int(tile_id), reason="large_shading")
        if tile_ids:
            print(
                "SHADING_LARGE_POINT_RESIDENCY "
                f"evicted_full_tiles={len(tile_ids)} lod_retained="
                f"{len(getattr(app, '_main_tile_lod_actors', {}) or {})}"
            )
    except Exception as exc:
        print(f"SHADING_LARGE_POINT_RESIDENCY status=skip reason={exc}")
    gc.collect()


def _vtk_tile(local_points, faces, point_colors, face_normals):
    """Create a compact tile with barycentric point RGB and flat face normals."""
    points_np = np.ascontiguousarray(local_points, dtype=np.float32)
    faces_np = np.ascontiguousarray(faces, dtype=np.int32)
    colors_np = np.ascontiguousarray(point_colors, dtype=np.uint8)
    normals_np = np.ascontiguousarray(face_normals, dtype=np.float32)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(points_np, deep=False))

    n_faces = int(len(faces_np))
    offsets = np.arange(0, (n_faces + 1) * 3, 3, dtype=np.int32)
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_support.numpy_to_vtk(offsets, deep=False),
        numpy_support.numpy_to_vtk(faces_np.reshape(-1), deep=False),
    )

    rgb = numpy_support.numpy_to_vtk(colors_np, deep=False)
    rgb.SetName("RGB")
    normals_vtk = numpy_support.numpy_to_vtk(normals_np, deep=False)
    normals_vtk.SetName("Normals")
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetPolys(cells)
    # VTK interpolates point RGB barycentrically across each triangle. Cell
    # normals keep illumination flat/faceted, so only colour is degraded.
    poly.GetPointData().SetScalars(rgb)
    poly.GetCellData().SetNormals(normals_vtk)
    # Retain NumPy owners because deep=False is deliberate for peak memory.
    poly._naksha_array_owners = (
        points_np, faces_np, colors_np, offsets, normals_np,
    )
    return poly


def _build_one_tile(
    app, desc, sidx, xyz_raw, classes_raw, visible_lut, color_lut,
    max_code, source_global_indices=None,
):
    """Build one owned-core TIN actor from its halo point set."""
    from gui.shading_display import (
        _compute_face_normals, _configure_microstation_color_blend_lighting,
        _do_triangulate,
    )

    gi = sidx.indices_for_cell_rect(
        int(desc.ix0) - 1, int(desc.ix1) + 1,
        int(desc.iy0) - 1, int(desc.iy1) + 1,
        sort_canonical=True,
    )
    if gi.size < 3:
        return None, 0, 0
    cls = np.asarray(classes_raw[gi], dtype=np.int64)
    keep = visible_lut[np.clip(cls, 0, max_code)]
    if not np.any(keep):
        return None, 0, 0
    gi, cls = gi[keep], cls[keep]
    points = np.asarray(xyz_raw[gi], dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    if not np.all(finite):
        points, cls, gi = points[finite], cls[finite], gi[finite]
    if len(points) < 3:
        return None, 0, 0

    offset = points.min(axis=0)
    local = points - offset
    faces = _do_triangulate(local[:, :2])
    if len(faces) == 0:
        return None, 0, 0
    bounds = tuple(float(v) for v in desc.bounds)
    centroids = (
        local[faces[:, 0], :2] + local[faces[:, 1], :2] +
        local[faces[:, 2], :2]
    ) / 3.0
    centroids += offset[:2]
    own = (
        (centroids[:, 0] >= bounds[0]) & (centroids[:, 0] <= bounds[1]) &
        (centroids[:, 1] >= bounds[2]) & (centroids[:, 1] <= bounds[3])
    )
    faces = faces[own]
    if len(faces) == 0:
        return None, 0, 0

    normals = _compute_face_normals(local, faces)
    colors = color_lut[np.clip(cls, 0, max_code)]
    poly = _vtk_tile(local, faces, colors, normals)
    name = f"{_ACTOR_PREFIX}{int(desc.tile_id):04d}"
    actor = app.vtk_widget.add_mesh(
        poly, scalars="RGB", rgb=True, preference="point",
        lighting=True, smooth_shading=False, show_edges=False,
        name=name, render=False,
    )
    actor.SetPosition(*[float(v) for v in offset])
    actor._is_shading_mesh = True
    actor._naksha_large_shading_tile = True
    canonical_ids = (
        np.asarray(source_global_indices)[gi]
        if source_global_indices is not None else gi
    )
    actor._naksha_large_source_indices = np.ascontiguousarray(
        canonical_ids, dtype=np.int32
    )
    actor._naksha_large_array_owners = poly._naksha_array_owners
    actor._naksha_large_tile_id = int(desc.tile_id)
    actor._naksha_large_tile_descriptor = desc
    actor.SetVisibility(_large_shading_is_active(app))
    prop = actor.GetProperty()
    prop.SetInterpolationToFlat(); prop.EdgeVisibilityOff()
    _configure_microstation_color_blend_lighting(app, actor)
    return actor, int(len(local)), int(len(faces))


def _refresh_large_shading_topology_local(
    app, changed_indices, *, operation="classification",
):
    """Rebuild only core tiles whose halo can be affected by membership edits."""
    state = getattr(app, "_large_shading_build_state", None)
    if not isinstance(state, dict):
        return False
    sidx = state.get("spatial_index")
    descriptors = state.get("descriptors") or []
    xyz_raw = state.get("xyz_raw")
    classes_raw = state.get("classes_raw")
    if sidx is None or xyz_raw is None or classes_raw is None or not descriptors:
        return False

    ci = np.unique(np.asarray(changed_indices, dtype=np.int64).ravel())
    ci = ci[(ci >= 0) & (ci < len(xyz_raw))]
    if ci.size == 0:
        return True
    # Select against the exact changed fine-grid cells. Expanding an entire
    # render-tile rectangle rebuilt every tile along its long border (6-9 tiles
    # for a one-cell brush edit). A one-cell expansion around the actual edit
    # is sufficient because each TIN tile itself was built with that same halo.
    try:
        changed_xy = np.asarray(sidx.points[ci, :2], dtype=np.float64)
        changed_ix = (
            (changed_xy[:, 0] - float(sidx.x_min)) * float(sidx._inv_cell_w)
        ).astype(np.int32)
        changed_iy = (
            (changed_xy[:, 1] - float(sidx.y_min)) * float(sidx._inv_cell_h)
        ).astype(np.int32)
        np.clip(changed_ix, 0, int(sidx.nx) - 1, out=changed_ix)
        np.clip(changed_iy, 0, int(sidx.ny) - 1, out=changed_iy)
        changed_cells = np.unique(
            np.column_stack((changed_ix, changed_iy)), axis=0
        )
        changed_cell_count = int(len(changed_cells))
        selection_mode = "exact_cells"
        affected = [
            desc for desc in descriptors
            if np.any(
                (changed_cells[:, 0] >= int(desc.ix0) - 1) &
                (changed_cells[:, 0] <= int(desc.ix1) + 1) &
                (changed_cells[:, 1] >= int(desc.iy0) - 1) &
                (changed_cells[:, 1] <= int(desc.iy1) + 1)
            )
        ]
    except Exception:
        # Compatibility fallback for non-grid test/adaptor indexes.
        changed_cell_count = -1
        selection_mode = "tile_fallback"
        core_ids = set(
            int(v) for v in sidx.tile_ids_for_indices(ci, descriptors)
        )
        core_ids.discard(-1)
        core = [d for d in descriptors if int(d.tile_id) in core_ids]
        affected = [
            desc for desc in descriptors
            if any(
                int(desc.ix0) <= int(c.ix1) + 1 and
                int(desc.ix1) >= int(c.ix0) - 1 and
                int(desc.iy0) <= int(c.iy1) + 1 and
                int(desc.iy1) >= int(c.iy0) - 1
                for c in core
            )
        ]
    if not affected:
        return False

    visible = set(int(c) for c in app._large_shading_visible_classes)
    max_code = max([255] + list(visible) +
                   [int(c) for c in getattr(app, "class_palette", {})])
    visible_lut = np.zeros(max_code + 1, dtype=np.bool_)
    for code in visible:
        if 0 <= code <= max_code:
            visible_lut[code] = True
    color_lut = _palette_color_lut(app, classes_raw, visible)
    old_by_id = {
        int(getattr(a, "_naksha_large_tile_id", -1)): a
        for a in (getattr(app, "_large_shading_tile_actors", None) or [])
    }
    started = time.perf_counter()
    rebuilt_vertices = rebuilt_faces = 0
    for desc in affected:
        tile_started = time.perf_counter()
        name = f"{_ACTOR_PREFIX}{int(desc.tile_id):04d}"
        if int(desc.tile_id) in old_by_id:
            try:
                app.vtk_widget.remove_actor(name, render=False)
            except Exception:
                pass
            old_by_id.pop(int(desc.tile_id), None)
        try:
            actor, nv, nf = _build_one_tile(
                app, desc, sidx, xyz_raw, classes_raw, visible_lut,
                color_lut, max_code, state.get("source_global_indices"),
            )
        except Exception as exc:
            # The old actor may already have been detached. Invalidate the
            # resident signature so the caller's established fallback performs
            # one canonical rebuild instead of recoloring an incomplete set.
            app._large_shading_geometry_signature = None
            print(
                "SHADING_LARGE_LOCAL_TOPOLOGY status=failed "
                f"tile={int(desc.tile_id)} reason={type(exc).__name__}:{exc}"
            )
            return False
        if actor is not None:
            old_by_id[int(desc.tile_id)] = actor
        rebuilt_vertices += nv; rebuilt_faces += nf
        print(
            "SHADING_LARGE_LOCAL_TILE status=ready "
            f"tile={int(desc.tile_id)} points={nv:,} faces={nf:,} "
            f"elapsed={(time.perf_counter()-tile_started):.2f}s"
        )

    actors = [old_by_id[k] for k in sorted(old_by_id)]
    if not actors:
        app._large_shading_geometry_signature = None
        return False
    app._large_shading_tile_actors = actors
    app._shaded_mesh_actor = actors[0]
    app._large_shading_classification_revision = int(
        getattr(app, "classification_revision", 0) or 0
    )
    if _large_shading_is_active(app):
        app._shading_parked = False
        app.vtk_widget.render()
    print(
        "SHADING_LARGE_LOCAL_TOPOLOGY status=patched full_rebuild=0 "
        f"operation={operation} changed={ci.size} tiles={len(affected)} "
        f"cells={changed_cell_count} selection={selection_mode} "
        f"vertices={rebuilt_vertices:,} faces={rebuilt_faces:,} "
        f"elapsed={(time.perf_counter()-started):.2f}s"
    )
    return True


def build_large_slow_shading(
    app, xyz_raw, classes_raw, visible_classes, azimuth, angle, ambient,
    *, source_global_indices=None,
):
    """Build an all-point tiled TIN and return True when this path owns it."""
    from PySide6.QtCore import QCoreApplication, QEventLoop
    from gui.spatial_index import get_or_build_index
    from gui.shading_display import (
        _compute_face_normals, _configure_microstation_color_blend_lighting,
        _do_triangulate,
    )

    n_points = int(len(xyz_raw))
    if n_points < 3:
        return False

    # Classification revisions do not invalidate XY triangulation.  Sparse
    # class edits patch resident point RGB through refresh_large_shading_sparse.
    geometry_signature = (
        id(xyz_raw), n_points, tuple(sorted(int(c) for c in visible_classes)),
    )
    max_code = max([255] + [int(c) for c in visible_classes] +
                   [int(c) for c in getattr(app, "class_palette", {}).keys()])
    visible_lut = np.zeros(max_code + 1, dtype=np.bool_)
    color_lut = np.zeros((max_code + 1, 3), dtype=np.float32)
    for code, entry in getattr(app, "class_palette", {}).items():
        ci = int(code)
        if 0 <= ci <= max_code and ci in visible_classes:
            visible_lut[ci] = True
            color_lut[ci] = entry.get("color", (128, 128, 128))

    old_actors = getattr(app, "_large_shading_tile_actors", None) or []
    if geometry_signature == getattr(app, "_large_shading_geometry_signature", None) and old_actors:
        recolor_started = time.perf_counter()
        for actor in old_actors:
            poly = actor.GetMapper().GetInput()
            owners = getattr(actor, "_naksha_large_array_owners", None)
            tile_indices = getattr(actor, "_naksha_large_source_indices", None)
            if owners is None or tile_indices is None:
                return False
            _local, _faces, colors, _offsets, _normals = owners
            tile_classes = np.asarray(classes_raw[tile_indices], dtype=np.int64)
            colors[:] = color_lut[np.clip(tile_classes, 0, max_code)].astype(
                np.uint8, copy=False
            )
            poly.GetPointData().GetScalars().Modified()
            poly.Modified()
            actor.SetVisibility(True)
            _configure_microstation_color_blend_lighting(app, actor)
        app._shaded_mesh_actor = old_actors[0]
        app._large_shading_visible_classes = tuple(
            sorted(int(c) for c in visible_classes)
        )
        app._large_shading_classification_revision = int(
            getattr(app, "classification_revision", 0) or 0
        )
        app._large_shading_palette_signature = large_shading_palette_signature(
            app, visible_classes
        )
        _mark_large_shading_active(app, old_actors)
        app.vtk_widget.render()
        app._large_shading_lighting_signature = (
            float(azimuth), float(angle), float(ambient)
        )
        print(
            f"SHADING_LARGE_TILED status=recolored tiles={len(old_actors)} "
            f"elapsed={(time.perf_counter()-recolor_started):.2f}s topology_rebuild=0"
        )
        return True

    # A different visibility topology (for example F10 all-class -> F11
    # ground-only) gets its own resident bundle. Detach the current actor graph
    # without destroying its PolyData, then restore the requested graph if it
    # was built before. This is the large-cloud equivalent of the ordinary
    # shading preset cache.
    if old_actors:
        _detach_active_preset(app)
    if _restore_resident_preset(
        app, geometry_signature, azimuth, angle, ambient,
    ):
        return True

    started = time.perf_counter()
    _remove_old_tiles(app)
    _release_full_point_tiles(app)
    plotter = app.vtk_widget
    sidx = get_or_build_index(xyz_raw, method="grid2d")
    target = int(os.environ.get("NAKSHA_SHADING_TILE_POINTS", "1200000"))
    target = max(250_000, min(2_000_000, target))
    descriptors = sidx.build_render_tiles(max_points=target)
    if not descriptors:
        raise RuntimeError("spatial index produced no shading tiles")

    actors = []
    total_vertices = total_faces = 0
    cpu_count = int(os.cpu_count() or 1)
    requested_workers = int(os.environ.get("NAKSHA_SHADING_TILE_WORKERS", "8"))
    worker_count = max(1, min(requested_workers, cpu_count, len(descriptors), 8))
    executor = (
        ProcessPoolExecutor(max_workers=worker_count)
        if worker_count > 1 else None
    )
    print(
        "SHADING_LARGE_PARALLEL "
        f"workers={worker_count} "
        f"backend={'process_triangle' if executor else 'serial_triangle'} "
        f"tiles={len(descriptors)} bounded_batch={worker_count}"
    )

    def prepare_tile(desc):
        gi = sidx.indices_for_cell_rect(
            int(desc.ix0) - 1, int(desc.ix1) + 1,
            int(desc.iy0) - 1, int(desc.iy1) + 1,
            sort_canonical=True,
        )
        if gi.size < 3:
            return None
        cls = np.asarray(classes_raw[gi], dtype=np.int64)
        keep = visible_lut[np.clip(cls, 0, max_code)]
        if not np.any(keep):
            return None
        gi, cls = gi[keep], cls[keep]
        points = np.asarray(xyz_raw[gi], dtype=np.float64)
        finite = np.isfinite(points).all(axis=1)
        if not np.all(finite):
            points, cls, gi = points[finite], cls[finite], gi[finite]
        if len(points) < 3:
            return None
        offset = points.min(axis=0)
        local = points - offset
        return (
            desc, gi, cls, local, offset,
            tuple(float(v) for v in desc.bounds),
        )

    def serial_compute(local, bounds, offset):
        faces = _do_triangulate(local[:, :2])
        if len(faces) == 0:
            return faces, np.empty((0, 3), dtype=np.float32)
        centroids = (
            local[faces[:, 0], :2] + local[faces[:, 1], :2]
            + local[faces[:, 2], :2]
        ) / 3.0
        centroids += offset[:2]
        own = (
            (centroids[:, 0] >= bounds[0]) & (centroids[:, 0] <= bounds[1])
            & (centroids[:, 1] >= bounds[2]) & (centroids[:, 1] <= bounds[3])
        )
        faces = np.ascontiguousarray(faces[own], dtype=np.int32)
        return faces, _compute_face_normals(local, faces)

    try:
        from gui.shading_tile_compute import compute_owned_tin

        for batch_start in range(0, len(descriptors), worker_count):
            batch_desc = descriptors[batch_start:batch_start + worker_count]
            prepared = [prepare_tile(desc) for desc in batch_desc]
            jobs = []
            for item in prepared:
                if item is None or executor is None:
                    jobs.append(None)
                    continue
                _desc, _gi, _cls, _local, _offset, _bounds = item
                jobs.append(executor.submit(
                    compute_owned_tin, _local, _bounds, _offset,
                ))

            for batch_pos, item in enumerate(prepared):
                if item is None:
                    continue
                sequence = batch_start + batch_pos
                desc, gi, cls, local, offset, bounds = item
                wait_started = time.perf_counter()
                try:
                    if jobs[batch_pos] is None:
                        faces, normals = serial_compute(local, bounds, offset)
                    else:
                        faces, normals = jobs[batch_pos].result()
                except Exception as exc:
                    print(
                        "SHADING_LARGE_PARALLEL status=tile_fallback "
                        f"tile={int(desc.tile_id)} "
                        f"reason={type(exc).__name__}:{exc}"
                    )
                    faces, normals = serial_compute(local, bounds, offset)
                if len(faces) == 0:
                    continue
                compute_wait = time.perf_counter() - wait_started
                attach_started = time.perf_counter()
                colors = color_lut[np.clip(cls, 0, max_code)].astype(
                    np.uint8, copy=False
                )
                poly = _vtk_tile(local, faces, colors, normals)
                name = f"{_ACTOR_PREFIX}{int(desc.tile_id):04d}"
                actor = plotter.add_mesh(
                    poly, scalars="RGB", rgb=True, preference="point",
                    lighting=True, smooth_shading=False, show_edges=False,
                    name=name, render=False,
                )
                actor.SetPosition(*[float(v) for v in offset])
                actor._is_shading_mesh = True
                actor._naksha_large_shading_tile = True
                canonical_ids = (
                    np.asarray(source_global_indices)[gi]
                    if source_global_indices is not None else gi
                )
                actor._naksha_large_source_indices = np.ascontiguousarray(
                    canonical_ids, dtype=np.int32
                )
                actor._naksha_large_array_owners = poly._naksha_array_owners
                actor._naksha_large_tile_id = int(desc.tile_id)
                actor._naksha_large_tile_descriptor = desc
                prop = actor.GetProperty()
                prop.SetInterpolationToFlat(); prop.EdgeVisibilityOff()
                _configure_microstation_color_blend_lighting(app, actor)
                actors.append(actor)
                total_vertices += int(len(local))
                total_faces += int(len(faces))
                print(
                    "SHADING_LARGE_TILE status=ready "
                    f"tile={sequence+1}/{len(descriptors)} "
                    f"points={len(local):,} faces={len(faces):,} "
                    f"compute_wait={compute_wait:.2f}s "
                    f"attach_elapsed={(time.perf_counter()-attach_started):.2f}s "
                    "colors=barycentric_point_rgb lighting=flat_cell_normals"
                )
                QCoreApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
            del prepared, jobs
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if not actors:
        raise RuntimeError("no visible finite points produced a shading tile")
    app._large_shading_tile_actors = actors
    app._large_shading_geometry_signature = geometry_signature
    app._large_shading_visible_classes = tuple(
        sorted(int(c) for c in visible_classes)
    )
    app._large_shading_classification_revision = int(
        getattr(app, "classification_revision", 0) or 0
    )
    app._large_shading_palette_signature = large_shading_palette_signature(
        app, visible_classes
    )
    app._large_shading_build_state = {
        "spatial_index": sidx,
        "descriptors": descriptors,
        "xyz_raw": xyz_raw,
        "classes_raw": classes_raw,
        "source_global_indices": source_global_indices,
    }
    app._large_shading_lighting_signature = (
        float(azimuth), float(angle), float(ambient)
    )
    app._large_shading_tiled = True
    app._shaded_mesh_actor = actors[0]
    app._shaded_mesh_polydata = None
    _mark_large_shading_active(app, actors)
    app.vtk_widget.renderer.ResetCameraClippingRange()
    app.vtk_widget.render()
    print(
        "SHADING_LARGE_TILED status=ready all_input_points=1 bounded_memory=1 "
        f"source_points={n_points:,} tiles={len(actors)} "
        f"tile_vertices={total_vertices:,} faces={total_faces:,} "
        f"elapsed={(time.perf_counter()-started):.2f}s "
        "presentation=barycentric_point_rgb flat_lighting=1"
    )
    return True
