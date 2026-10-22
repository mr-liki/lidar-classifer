"""Read-only bounded LNT planning for Ground PTD and Below Surface.

No native writes occur here.  Every committed identity is a validated packed-leaf
(node index, local point index).  Work regions are nonoverlapping; contexts may
overlap, but only core/fence points may be changed.  The GUI/native adapter owns
one atomic Undo operation after preflight and source-file revision verification.
"""
from __future__ import annotations

from collections import OrderedDict
import math
import os
from pathlib import Path

import numpy as np

from gui.native_lnt_directory import (
    LNTDirectoryError, POINT, _exactly, inspect_directory,
)


class ScopedPlanningError(LNTDirectoryError):
    pass


def _check_cancel(cancelled):
    if cancelled is not None and cancelled():
        raise ScopedPlanningError('Classification cancelled before any LNT write.')


def _check_file(path, info):
    stat = os.stat(path)
    if (stat.st_size, stat.st_mtime_ns) != (info['size'], info['mtime_ns']):
        raise ScopedPlanningError('LNT was changed during analysis; no edits were committed.')


def _prepare_area(polygons):
    from matplotlib.path import Path as MplPath
    if not polygons:
        return []
    prepared = []
    for coords in polygons:
        a = np.asarray(coords, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] < 2 or len(a) < 3 or not np.all(np.isfinite(a[:, :2])):
            raise ScopedPlanningError('Invalid selected fence coordinates; no changes made.')
        xy = a[:, :2]
        prepared.append((xy.min(0), xy.max(0), MplPath(xy)))
    return prepared


def _inside(xy, prepared):
    if not prepared:
        return np.ones(len(xy), dtype=bool)
    mask = np.zeros(len(xy), dtype=bool)
    for lo, hi, shape in prepared:
        coarse = ((xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) &
                  (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1]))
        if np.any(coarse):
            mask[coarse] |= shape.contains_points(xy[coarse])
    return mask


def _intersects_poly_bbox(bbox, prepared):
    if not prepared:
        return True
    lo = np.asarray(bbox[:2]); hi = np.asarray(bbox[2:])
    return any(np.all(lo <= right) and np.all(hi >= left)
               for left, right, _ in prepared)


def _regions(info, prepared, cell_size):
    coords = np.asarray(info['header_bounds'], dtype=np.float64)
    xmin, ymin, xmax, ymax = map(float, (coords[0], coords[1], coords[3], coords[4]))
    if prepared:
        xmin = max(xmin, min(float(lo[0]) for lo, _, _ in prepared))
        ymin = max(ymin, min(float(lo[1]) for lo, _, _ in prepared))
        xmax = min(xmax, max(float(hi[0]) for _, hi, _ in prepared))
        ymax = min(ymax, max(float(hi[1]) for _, hi, _ in prepared))
    if xmin > xmax or ymin > ymax:
        return []
    # Quantization may place a few packed coordinates outside header bounds.
    # An outward ulp + metric margin is used at the outermost edges, never to
    # move interior tile ownership borders (so every record has one core).
    step = float(cell_size)
    if not 0 < step < float('inf'):
        raise ScopedPlanningError('Invalid spatial processing cell size.')
    nx = max(1, int(math.ceil((xmax - xmin) / step)))
    ny = max(1, int(math.ceil((ymax - ymin) / step)))
    if nx * ny > 2_000_000:
        raise ScopedPlanningError('Too many spatial cells; invalid LNT extent or tile setting.')
    eps = max(0.5, float(max(abs(xmin), abs(ymin), abs(xmax), abs(ymax))) *
              np.finfo(np.float32).eps * 4.0)
    regions = []
    for ix in range(nx):
        xa = xmin + step*ix
        xb = xmin + step*(ix+1) if ix+1 < nx else xmax+eps
        if ix == 0:
            xa -= eps
        for iy in range(ny):
            ya = ymin + step*iy
            yb = ymin + step*(iy+1) if iy+1 < ny else ymax+eps
            if iy == 0:
                ya -= eps
            bbox = (xa, ya, xb, yb)
            if _intersects_poly_bbox(bbox, prepared):
                regions.append(bbox)
    return regions


def _identity(nodes, locals_):
    return (np.asarray(nodes, dtype=np.uint64) << np.uint64(32) |
            np.asarray(locals_, dtype=np.uint64))


def _read_window(path, info, bounds, allowed_classes=None, max_context=2_800_000,
                 chunk_size=180_000, cancelled=None):
    """Exact leaf reader filtered by XY box and optionally class, no entire-LNT snapshot."""
    xmin, ymin, xmax, ymax = map(float, bounds)
    if not np.isfinite((xmin, ymin, xmax, ymax)).all() or xmin > xmax or ymin > ymax:
        raise ScopedPlanningError('Invalid context window.')
    origin = info['origin']
    valid_classes = None if allowed_classes is None else np.asarray(sorted({int(x) for x in allowed_classes}), dtype=np.uint8)
    xyz_parts, classes_parts, nodes_parts, locals_parts = [], [], [], []
    found = 0
    with open(path, 'rb') as handle:
        for leaf_no, (node_id, offset, count, center, half) in enumerate(info['leaves']):
            if leaf_no % 32 == 0:
                _check_cancel(cancelled)
            center_abs = origin + np.asarray(center, dtype=np.float64)
            if (center_abs[0] + half < xmin or center_abs[0] - half > xmax or
                center_abs[1] + half < ymin or center_abs[1] - half > ymax):
                continue
            scale = 2.0 * float(half) / 65535.0
            xbase = center_abs[0] - float(half)
            ybase = center_abs[1] - float(half)
            zbase = center_abs[2] - float(half)
            for start in range(0, count, chunk_size):
                _check_cancel(cancelled)
                take = min(chunk_size, count-start)
                handle.seek(offset + start*POINT.itemsize)
                raw = np.frombuffer(_exactly(handle, take*POINT.itemsize, f'leaf {node_id}'), dtype=POINT)
                xy = np.empty((take, 2), dtype=np.float64)
                xy[:, 0] = xbase + scale*raw['x'].astype(np.float64)
                xy[:, 1] = ybase + scale*raw['y'].astype(np.float64)
                within = ((xy[:, 0] >= xmin) & (xy[:, 0] < xmax) &
                          (xy[:, 1] >= ymin) & (xy[:, 1] < ymax))
                if valid_classes is not None:
                    within &= np.isin(raw['classification'], valid_classes)
                rows = np.flatnonzero(within)
                if not len(rows):
                    continue
                found += len(rows)
                if found > max_context:
                    raise ScopedPlanningError(f'Context window contains over {max_context:,} relevant points.')
                xyz = np.empty((len(rows), 3), dtype=np.float64)
                xyz[:, :2] = xy[rows]
                xyz[:, 2] = zbase + scale*raw['z'][rows].astype(np.float64)
                xyz_parts.append(xyz)
                classes_parts.append(raw['classification'][rows].copy())
                nodes_parts.append(np.full(len(rows), int(node_id), dtype=np.uint32))
                locals_parts.append(rows.astype(np.uint32) + np.uint32(start))
    if not xyz_parts:
        return {'xyz': np.empty((0, 3), np.float64),
                'classification': np.empty(0, np.uint8),
                'node': np.empty(0, np.uint32),
                'local': np.empty(0, np.uint32)}
    xyz = np.concatenate(xyz_parts)
    if not np.all(np.isfinite(xyz)):
        raise ScopedPlanningError('Non-finite coordinate in LNT leaf window.')
    out = {'xyz': xyz,
           'classification': np.concatenate(classes_parts),
           'node': np.concatenate(nodes_parts),
           'local': np.concatenate(locals_parts)}
    if len(np.unique(_identity(out['node'], out['local']))) != len(xyz):
        raise ScopedPlanningError('Duplicate LNT point identity in spatial window.')
    return out


def _class_set(values):
    if values is None or (isinstance(values, (list, tuple)) and (not values or None in values)):
        return None
    return set(map(int, values))


def _split_rect(bounds):
    x0, y0, x1, y1 = bounds
    if (x1-x0) >= (y1-y0):
        mid = x0 + (x1-x0)/2.0
        return (x0, y0, mid, y1), (mid, y0, x1, y1)
    mid = y0 + (y1-y0)/2.0
    return (x0, y0, x1, mid), (x0, mid, x1, y1)


def _plan_result(records, info, stats):
    if records:
        # The same underlying node/local can never be planned twice, even
        # when selected user fences overlap or tile context overlaps.
        ids = np.asarray([entry[0] for entry in records], np.uint64)
        if len(np.unique(ids)) != len(ids):
            raise ScopedPlanningError('Duplicate planned point identity; no changes written.')
        nodes = np.asarray([int(k >> 32) for k in ids], np.uint32)
        local = np.asarray([int(k & 0xffffffff) for k in ids], np.uint32)
        old = np.asarray([entry[1] for entry in records], np.uint8)
        new = np.asarray([entry[2] for entry in records], np.uint8)
    else:
        nodes = np.empty(0, np.uint32)
        local = np.empty(0, np.uint32)
        old = np.empty(0, np.uint8)
        new = np.empty(0, np.uint8)
    return dict(node=nodes, local=local, old_classes=old, new_classes=new,
                file_stat=(info['size'], info['mtime_ns']), stats=stats)


def plan_streamed_ground(path, polygons, params, *, expected_total=None,
                         max_changes=8_000_000, progress=None, cancelled=None,
                         tile_size=110.0, max_context=2_800_000):
    """Bounded local PTD planning with trusted-ground/outside-fence context.

    IMPORTANT: a globally triangulated PTD has no finite exact tile halo.
    This is a spatially local PTD extension for large clouds, *not* a claim of
    bit-for-bit identity with a whole-cloud global TIN. Tile contexts overlap
    and all writes remain exact and fence-scoped. A conservative halo is based
    on the actual configured seed-search cell size and TIN thresholds.
    """
    from gui.lidar_classification_tools import classify_ground_ptd
    path = Path(path)
    info = inspect_directory(path, max_points=2**63-1)
    if expected_total is not None and info['total'] != int(expected_total):
        raise ScopedPlanningError('Active cloud count differs from LNT directory.')
    if info['leaf_sum'] != info['total']:
        raise ScopedPlanningError('LNT leaf count differs from header.')
    prepared = _prepare_area(polygons)
    sources = _class_set(params['from_classes'])
    trusted = _class_set(params.get('current_ground'))
    target = int(params['to_class'])
    if not 0 <= target <= 255:
        raise ScopedPlanningError('Invalid Ground target classification.')
    if sources is not None and trusted is not None and (sources & trusted) and params.get('seed_method') != 'lowest_only':
        raise ScopedPlanningError('Current Ground class must not overlap source classes.')
    halo = max(65.0, float(params.get('max_building_size', 60.0))*1.5,
               float(params.get('iteration_distance', 1.4))*6.0,
               float(params.get('edge_length_threshold', 5.0))*3.0)
    if not np.isfinite(halo) or halo <= 0:
        raise ScopedPlanningError('Invalid Ground search extent.')
    if sources is None or trusted is None:
        allowed = None
    else:
        allowed = sources | (trusted if params.get('seed_method') != 'lowest_only' else set())
    regions = _regions(info, prepared, tile_size)
    all_changes = []
    nregions, processed, eligible_count, skipped = 0, 0, 0, 0
    tasks = [(rect, 0) for rect in reversed(regions)]
    while tasks:
        _check_cancel(cancelled)
        core, depth = tasks.pop()
        if not _intersects_poly_bbox(core, prepared):
            continue
        pad = (core[0]-halo, core[1]-halo, core[2]+halo, core[3]+halo)
        try:
            tile = _read_window(path, info, pad, allowed_classes=allowed,
                                max_context=max_context, cancelled=cancelled)
        except ScopedPlanningError as exc:
            if 'Context window contains over' not in str(exc) or depth >= 12:
                raise
            tasks.extend([(sub, depth+1) for sub in reversed(_split_rect(core))])
            continue
        nregions += 1
        xyz, original = tile['xyz'], tile['classification']
        if not len(xyz):
            continue
        inside = ((xyz[:, 0] >= core[0]) & (xyz[:, 0] < core[2]) &
                  (xyz[:, 1] >= core[1]) & (xyz[:, 1] < core[3]))
        if prepared:
            inside &= _inside(xyz[:, :2], prepared)
        eligible = inside & (original != target)
        if sources is not None:
            eligible &= np.isin(original, list(sources))
        count = int(np.count_nonzero(eligible))
        if not count:
            continue
        eligible_count += count
        # PTD sees full tile + surrounding context, but the write mask is
        # ALWAYS intersected with the non-overlapping core and original fence.
        updated = original.copy()
        def _emit(pct, msg=''):
            _check_cancel(cancelled)
            if progress is not None:
                base = 6 + int(85 * (1-len(tasks)/max(1, len(regions))))
                progress(min(92, max(5, base)),
                         f'Ground PTD cell {nregions:,}: {str(msg)[:95]}')
        # The polygon is a write selection for the legacy PTD. Preserve its
        # seed-selection semantics even though the read context extends well
        # beyond the fence and current tile core. The PTD implementation
        # deliberately uses trusted Ground OUTSIDE the selected fence as
        # read-only terrain support; never turn all outside-fence source
        # points into eligible seed/candidate inputs by omitting this mask.
        fence_in_context = _inside(xyz[:, :2], prepared) if prepared else None
        result = classify_ground_ptd(xyz=xyz, classification=updated,
                                     fence_mask=fence_in_context, progress_cb=_emit,
                                     abort_check=cancelled, **params) or {}
        reason = str(result.get('stop_reason', '')).strip().lower()
        if reason in ('cancelled', 'canceled', 'aborted'):
            raise ScopedPlanningError('Ground PTD cancelled before write.')
        # Some stopping states signify genuinely invalid analysis and should
        # never be considered successful classification runs.
        unsafe = ('tin construction failed', 'non-finite tin coordinates',
                  'degenerate tin extent')
        if reason in unsafe:
            raise ScopedPlanningError(f'Ground PTD terrain analysis failed: {reason}.')
        changed = np.flatnonzero(eligible & (updated != original))
        if len(changed):
            ids = _identity(tile['node'][changed], tile['local'][changed])
            all_changes.extend((int(key), int(before), int(after)) for key, before, after
                               in zip(ids, original[changed], updated[changed]))
        processed += 1
        if len(all_changes) > max_changes:
            raise ScopedPlanningError(f'Ground changes exceed reversible {max_changes:,}-record Undo budget.')
        if progress is not None:
            progress(6 + int(85*(1-len(tasks)/max(1, len(regions)))),
                     f'Ground PTD: {processed:,} cells, {len(all_changes):,} changes planned')
    _check_cancel(cancelled)
    _check_file(path, info)
    plan = _plan_result(all_changes, info, {'areas': nregions,
                 'source_candidates': eligible_count, 'planned_changes': len(all_changes),
                 'method': 'local PTD with overlapping terrain context'})
    return plan, info


def _fit_flags(query, neighbor_points, surface_type, limit, z_tolerance):
    """Match classify_below_surface's planar/curved residual tests per query batch."""
    if len(query) == 0:
        return np.empty(0, dtype=bool)
    curved = surface_type == 'curved'
    n = len(query)
    flags = np.zeros(n, dtype=bool)
    if not curved:
        A = np.empty((n, neighbor_points.shape[1], 3), dtype=np.float64)
        A[:, :, :2] = neighbor_points[:, :, :2]
        A[:, :, 2] = 1.0
        b = neighbor_points[:, :, 2]
        ata = np.einsum('nki,nkj->nij', A, A)
        atb = np.einsum('nki,nk->ni', A, b)
        try:
            coef = np.linalg.solve(ata, atb)
        except Exception:
            coef = np.full((n, 3), np.nan)
        bad = ~np.all(np.isfinite(coef), axis=1)
        for row in np.flatnonzero(bad):
            try:
                coef[row], _, _, _ = np.linalg.lstsq(A[row], b[row], rcond=None)
            except Exception:
                coef[row] = np.nan
        fitted = np.einsum('nki,ni->nk', A, coef)
        norm = np.sqrt(coef[:, 0]**2 + coef[:, 1]**2 + 1.0)
        deviation = np.std((b-fitted)/norm[:, None], axis=1)
        predicted = np.einsum('ni,ni->n', np.column_stack((query[:, 0], query[:, 1],
                                                          np.ones(n))), coef)
        offset = (predicted-query[:, 2])/norm
        flags = (np.all(np.isfinite(coef), axis=1) & np.isfinite(deviation) &
                 np.isfinite(offset) & (offset > 0) & (offset >= z_tolerance) &
                 ((deviation < 1e-9) | (offset > limit*deviation)))
    else:
        for row in range(n):
            neighbors = neighbor_points[row]
            if len(neighbors) < 6:
                continue
            pt = query[row]
            cx, cy = neighbors[:, :2].mean(axis=0)
            xn, yn = neighbors[:, 0]-cx, neighbors[:, 1]-cy
            design = np.column_stack((xn*xn, yn*yn, xn*yn, xn, yn,
                                      np.ones(len(neighbors))))
            try:
                coef, _, _, _ = np.linalg.lstsq(design, neighbors[:, 2], rcond=None)
            except Exception:
                continue
            fitted = design @ coef
            px, py = pt[0]-cx, pt[1]-cy
            zhat = np.dot((px*px, py*py, px*py, px, py, 1.0), coef)
            dx = 2*coef[0]*px + coef[2]*py + coef[3]
            dy = 2*coef[1]*py + coef[2]*px + coef[4]
            normal = np.sqrt(dx*dx + dy*dy + 1)
            dev = np.std((neighbors[:, 2]-fitted)/normal)
            offset = (zhat-pt[2])/normal
            flags[row] = (offset > 0 and offset >= z_tolerance and
                          (dev < 1e-9 or offset > limit*dev))
    return flags


def plan_streamed_below_surface(path, polygons, params, *, expected_total=None,
                               max_changes=8_000_000, progress=None, cancelled=None,
                               tile_size=135.0, max_context=2_800_000,
                               initial_halo=6.0, max_halo=1500.0):
    """Exact k-nearest local fitting for large LNT clouds, including fence edges.

    Every core query's k-nearest points must lie strictly inside the read halo
    (or the complete cloud extent). That certificate makes the local kNN set
    exact, even though no full-cloud array is built. Iterative mode operates
    on a planned classification overlay and commits only at the very end.
    """
    from scipy.spatial import cKDTree
    path = Path(path)
    info = inspect_directory(path, max_points=2**63-1)
    if expected_total is not None and info['total'] != int(expected_total):
        raise ScopedPlanningError('Active cloud count differs from LNT directory.')
    if info['leaf_sum'] != info['total']:
        raise ScopedPlanningError('LNT leaf count differs from header.')
    prepared = _prepare_area(polygons)
    src_codes = _class_set(params['from_classes'])
    target = int(params['to_class'])
    requested = max(3, int(params.get('num_neighbors', 25)))
    iterations = max(1, int(params.get('max_iterations', 5))) if params.get('iterative') else 1
    limit = max(0.0, float(params.get('limit', 4.0)))
    ztol = max(0.0, float(params.get('z_tolerance', .1)))
    surface = str(params.get('surface_type', 'planar'))
    if surface not in ('planar', 'curved'):
        raise ScopedPlanningError('Unknown Below Surface fitting type.')
    if not 0 <= target <= 255 or not np.isfinite((limit, ztol)).all():
        raise ScopedPlanningError('Invalid Below Surface parameters.')
    cloud_bounds = np.asarray(info['header_bounds'])[[0, 1, 3, 4]]
    # Packed leaves may deviate slightly from the exported LAS header envelope.
    edgepad = max(.5, float(np.max(np.abs(cloud_bounds))) *
                  float(np.finfo(np.float32).eps) * 4)
    cloud_bounds[:2] -= edgepad
    cloud_bounds[2:] += edgepad
    regions = _regions(info, prepared, tile_size)
    changes = {}   # uint64 identity -> original class (all planned changes to target)
    source_candidates = 0
    cells_done = 0
    for iteration in range(iterations):
        _check_cancel(cancelled)
        newly_changed = {}
        overlay = np.fromiter(changes.keys(), dtype=np.uint64, count=len(changes))
        if overlay.size:
            overlay.sort()
        tasks = [(rect, 0) for rect in reversed(regions)]
        while tasks:
            _check_cancel(cancelled)
            core, depth = tasks.pop()
            if not _intersects_poly_bbox(core, prepared):
                continue
            if progress is not None:
                progress(6 + int(85*(iteration + (1-len(tasks)/max(1,len(regions))))/iterations),
                         f'Below Surface: scanning cell {cells_done+1:,}, pass {iteration+1}/{iterations}')
            halo = float(initial_halo)
            if halo <= 0 or not np.isfinite(halo):
                raise ScopedPlanningError('Invalid Below Surface halo.')
            while True:
                _check_cancel(cancelled)
                ctx = (core[0]-halo, core[1]-halo, core[2]+halo, core[3]+halo)
                try:
                    tile = _read_window(path, info, ctx, allowed_classes=src_codes,
                                        max_context=max_context, cancelled=cancelled)
                except ScopedPlanningError as exc:
                    if 'Context window contains over' not in str(exc) or depth >= 12:
                        raise
                    tasks.extend((r, depth+1) for r in reversed(_split_rect(core)))
                    tile = None
                    break
                xyz = tile['xyz']
                if not len(xyz):
                    break
                original = tile['classification']
                key = _identity(tile['node'], tile['local'])
                actual = original.copy()
                if overlay.size:
                    actual[np.isin(key, overlay, assume_unique=False)] = target
                ref = np.ones(len(xyz), dtype=bool)
                if prepared:
                    ref &= _inside(xyz[:, :2], prepared)
                if src_codes is not None:
                    ref &= np.isin(actual, list(src_codes))
                in_core = ((xyz[:, 0] >= core[0]) & (xyz[:, 0] < core[2]) &
                           (xyz[:, 1] >= core[1]) & (xyz[:, 1] < core[3]))
                candidates = np.flatnonzero(ref & in_core & (actual != target))
                if iteration == 0:
                    # Count only once (this tile may be retried with a larger halo).
                    pass
                if not len(candidates):
                    break
                refs = np.flatnonzero(ref)
                if len(refs) < requested+1:
                    if halo < max_halo:
                        halo = min(max_halo, max(halo*2, halo+1))
                        continue
                    # Sparse cloud: if the search covers the entire LNT and
                    # still lacks requested neighbors, the legacy algorithm
                    # also has no valid fit. Otherwise fail closed.
                    if ctx[0] > cloud_bounds[0] or ctx[1] > cloud_bounds[1] or ctx[2] < cloud_bounds[2] or ctx[3] < cloud_bounds[3]:
                        raise ScopedPlanningError('Not enough source neighbors within bounded search; no changes made.')
                    break
                tree = cKDTree(xyz[refs])
                # Certificate: kth-nearest 3D distance to any candidate is
                # below the minimum XY distance to an uninspected region.
                # k includes the candidate itself as in the legacy routine.
                cert_ok = True
                for start in range(0, len(candidates), 4096):
                    q = candidates[start:start+4096]
                    kth = tree.query(xyz[q], k=requested+1, workers=1)[0]
                    kth = kth[:, -1] if np.ndim(kth) == 2 else np.asarray([kth[-1]])
                    xy = xyz[q, :2]
                    margins = []
                    for axis, lower, upper, minc, maxc in ((0, ctx[0], ctx[2], cloud_bounds[0], cloud_bounds[2]),
                                                           (1, ctx[1], ctx[3], cloud_bounds[1], cloud_bounds[3])):
                        if lower > minc:
                            margins.append(xy[:, axis] - lower)
                        if upper < maxc:
                            margins.append(upper - xy[:, axis])
                    radius = min((np.min(a) for a in margins), default=float('inf'))
                    if np.any(kth >= radius - 1e-7):
                        cert_ok = False
                        break
                if not cert_ok:
                    if halo < max_halo:
                        halo = min(max_halo, max(halo*2, halo+1))
                        continue
                    raise ScopedPlanningError('Below Surface exact KNN search reached halo limit; no changes made.')
                # The core's K-nearest sources are now mathematically bounded.
                if iteration == 0:
                    source_candidates += len(candidates)
                cells_done += 1
                for start in range(0, len(candidates), 2048):
                    _check_cancel(cancelled)
                    if progress is not None and (start // 2048) % 16 == 0:
                        progress(6 + int(85*(iteration + (1-len(tasks)/max(1,len(regions))))/iterations),
                                 f'Below Surface: fitting {start:,}/{len(candidates):,} points')
                    q = candidates[start:start+2048]
                    _, neighbors = tree.query(xyz[q], k=requested+1, workers=1)
                    if neighbors.ndim == 1:
                        neighbors = neighbors[None, :]
                    # The legacy classifier discards the first self-neighbor.
                    neighborhood = xyz[refs[neighbors[:, 1:]]]
                    flags = _fit_flags(xyz[q], neighborhood, surface, limit, ztol)
                    for ii in q[flags]:
                        identity = int(key[ii])
                        if identity not in changes and identity not in newly_changed:
                            newly_changed[identity] = int(original[ii])
                break
            if len(changes) + len(newly_changed) > max_changes:
                raise ScopedPlanningError(f'Below Surface exceeds {max_changes:,} reversible Undo entries.')
            if progress is not None and cells_done % 3 == 0:
                progress(6 + int(85*(iteration + (1-len(tasks)/max(1,len(regions))))/iterations),
                         f'Below Surface: pass {iteration+1}/{iterations}, {cells_done:,} cells, '
                         f'{len(changes)+len(newly_changed):,} changes planned')
        if not newly_changed:
            break
        changes.update(newly_changed)
    _check_cancel(cancelled)
    _check_file(path, info)
    records = [(key, cls, target) for key, cls in changes.items()]
    return _plan_result(records, info, {'cells': cells_done, 'source_candidates': source_candidates,
                               'iterations': iteration+1, 'planned_changes': len(changes),
                               'fit': surface}), info
