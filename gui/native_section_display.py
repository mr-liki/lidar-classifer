"""Array-only display preparation for Vulkan cross/cut sections."""
from __future__ import annotations

import numpy as np

MESH_MODES = frozenset(('shaded_class', 'surface'))
POINT_MODES = {'class': 1, 'depth': 6, 'intensity': 2, 'rgb': 0,
               'elevation': 3, 'line': 0}


def rainbow(t, ramp=None):
    """The legacy five-stop elevation ramp, or the user's custom ramp."""
    t = np.clip(np.asarray(t, dtype=np.float32), 0, 1)
    stops = sorted(ramp or [])
    if len(stops) < 2:
        stops = [(0, (0, 0, 255)), (.25, (0, 255, 255)), (.5, (0, 255, 0)),
                 (.75, (255, 255, 0)), (1, (255, 0, 0))]
    pos = np.array([p for p, _ in stops], dtype=np.float32)
    cols = np.array([c for _, c in stops], dtype=np.float32)
    return np.column_stack([np.interp(t, pos, cols[:, i]) for i in range(3)]).astype(np.uint8)


def line_colors(ids, colors=None):
    ids = np.asarray(ids, dtype=np.uint32)
    colors = colors or {}
    rgb = np.column_stack(((ids*67+53)%256,(ids*131+97)%256,(ids*193+181)%256)).astype(np.uint8)
    for value, color in colors.items():
        rgb[ids == int(value)] = color
    return rgb


def depth_colors(depth, app):
    # Section cameras look along +local depth. Translation cancels under
    # percentile normalisation, exactly as the legacy view-direction formula.
    values = np.asarray(depth, dtype=np.float64)
    lo, hi = np.percentile(values, [float(getattr(app, 'depth_clip_low', 1.)),
                                     float(getattr(app, 'depth_clip_high', 99.))])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    t = (np.clip((values - lo) / (hi - lo), 0., 1.) if hi > lo
         else np.full(values.shape, .5))
    t = np.power(t, float(getattr(app, 'depth_gamma', 1.)))
    scheme = getattr(app, 'depth_color_scheme', 'grayscale')
    if scheme == 'rainbow':
        return rainbow(t)
    if scheme == 'heatmap':
        return rainbow(t, [(0, (0, 0, 128)), (.33, (128, 0, 128)),
                           (.67, (255, 0, 0)), (1, (255, 255, 0))])
    grey = np.clip((t if scheme == 'inverted' else 1. - t) * 255., 0, 255).astype(np.uint8)
    return np.repeat(grey[:, None], 3, axis=1)


def color_range(values, low, high, ignore_zero=False):
    """Legacy percentile stretch, including flat and non-finite data."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if ignore_zero and np.any(values > 0):
        values = values[values > 0]
    if not len(values):
        return 0., 1.
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    # A constant maps to the legacy midpoint (0.5), not to black/blue.
    return (float(lo), float(hi)) if hi > lo else (float(lo) - .5, float(lo) + .5)


def visibility(classes, palette, blocked=(), source_ids=None, lines=None):
    shown = np.full(256, not bool(palette), dtype=bool)
    for code, entry in palette.items():
        if 0 <= int(code) < 256:
            shown[int(code)] = bool(entry.get('show', True))
    for code in blocked or ():
        if 0 <= int(code) < 256:
            shown[int(code)] = False
    mask = shown[np.asarray(classes, dtype=np.uint8)]
    if source_ids is not None:
        hidden = [int(k) for k, show in (lines or {}).items() if not show]
        if hidden:
            mask &= ~np.isin(source_ids, hidden)
    return mask


def draw_ranges(mask):
    """Contiguous runs keep selection/store indices stable in every mode."""
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    first = np.flatnonzero(edges == 1).astype(np.uint32)
    return first, (np.flatnonzero(edges == -1) - first).astype(np.uint32)


def build_mesh_data(xyz, display, classes, palette, mode, options):
    """Use the legacy terrain math, then place vertices in this section's frame.

    xyz contains only this corridor/cut's selected source records. display is
    their existing local frame, including side/nested-cut transforms. Geometry
    never depends on Main View's current actor, palette or presentation mode.

    The returned mapping keeps the geometry and per-face shade separately from
    the baked colours, so a later classification edit can recolour the SAME
    mesh instead of triangulating the section again.
    """
    from gui.section_surface_math import _compute_surface_geometry_backend
    from gui.section_shading_math import _compute_shading_geometry_backend

    azimuth, angle, ambient = options['light']
    shade = None
    if mode == 'surface':
        result = _compute_surface_geometry_backend(
            xyz, np.arange(len(xyz), dtype=np.int64), options['precision'], len(xyz),
            options['max_edge'], azimuth, angle, ambient, options['ramp'],
            quality_mode=options['quality'], z_bounds=options['z_bounds'])
        if result.get('empty', False):
            return None
        selected, faces, colors = (result['unique_global_idx'], result['faces'], result['colors'])
    else:
        result = _compute_shading_geometry_backend(
            xyz, classes, set(map(int, np.unique(classes))), azimuth, angle, ambient,
            3., None, None, quality_mode=options['quality'])
        if result.get('empty', False) or not len(result['faces']):
            return None
        selected, faces = result['unique_indices'], result['faces']
        shade = np.ascontiguousarray(result['shade'], dtype=np.float64)
        colors = shade_mesh_colors(classes[selected][faces[:, 0]], palette, shade)
    return {
        'positions': np.ascontiguousarray(display[selected], dtype=np.float64),
        'faces': np.ascontiguousarray(faces, dtype=np.int32),
        'colors': np.ascontiguousarray(colors, dtype=np.uint8),
        'selected': np.ascontiguousarray(selected, dtype=np.int64),
        'shade': shade,
        'mode': mode,
    }


def shade_mesh_colors(face_classes, palette, shade):
    """Bake the palette colour of each face through the cached hillshade."""
    lut = np.full((256, 3), 128., dtype=np.float64)
    for code, entry in (palette or {}).items():
        if 0 <= int(code) < 256:
            lut[int(code)] = entry.get('color', (128, 128, 128))
    return np.clip(lut[np.asarray(face_classes, dtype=np.uint8)] * np.asarray(shade)[:, None],
                   0, 255).astype(np.uint8)


def recolor_mesh(data, classes, palette):
    """New face colours for an existing Shaded Classification mesh.

    ``data`` is a build_mesh_data() result; ``classes`` are the current class
    codes of the masked records it was built from. Surface meshes bake
    elevation, not class, so their colours are returned unchanged.
    """
    shade = None if data is None else data.get('shade')
    if shade is None:
        return None if data is None else data['colors']
    face_classes = np.asarray(classes)[data['selected']][data['faces'][:, 0]]
    return shade_mesh_colors(face_classes, palette, shade)


def build_mesh(xyz, display, classes, palette, mode, options):
    """Tuple form of build_mesh_data(), kept for callers/tests."""
    data = build_mesh_data(xyz, display, classes, palette, mode, options)
    return None if data is None else (data['positions'], data['faces'], data['colors'])


def source_line_ids(path, xyz, tolerance, target_crs_wkt=None):
    """Recover LAS Point Source IDs for LNT records without loading the LAS.

    LNT v2 discards this attribute. Scan the source in bounded chunks, matching
    its original coordinates to the reconstructed section records. Refuse
    ambiguous coincident returns from different lines instead of inventing IDs.
    """
    import laspy
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=np.float64)
    if not len(xyz):
        return np.empty(0, dtype=np.uint16)
    tolerance = max(float(tolerance), 1e-8)
    lo, hi = xyz.min(axis=0) - tolerance, xyz.max(axis=0) + tolerance
    best = np.full(len(xyz), np.inf)
    result = np.zeros(len(xyz), dtype=np.uint16)
    ambiguous = np.zeros(len(xyz), dtype=bool)
    from gui.lnt_collection import is_collection, read_collection
    group = read_collection(path) if is_collection(path) else None
    paths = group['sources'] if group else [path]
    def chunks():
        from pyproj import CRS, Transformer
        target_wkt = group.get('target_crs') if group else target_crs_wkt
        target = CRS.from_user_input(target_wkt) if target_wkt else None
        for source in paths:
            with laspy.open(source) as reader:
                crs = reader.header.parse_crs()
                transform = Transformer.from_crs(crs,target,always_xy=True) if target and crs and crs!=target else None
                for chunk in reader.chunk_iterator(500_000):
                    coords = (chunk.x,chunk.y,chunk.z)
                    if transform: coords = transform.transform(*coords,errcheck=True)
                    yield np.column_stack(coords), np.asarray(chunk.point_source_id)
    for points, ids in chunks():
        inside = ((points >= lo) & (points <= hi)).all(axis=1)
        if not inside.any():
            continue
        points = points[inside]
        ids = ids[inside]
        # Query source neighbours per section point: duplicates in the
        # section receive the same ID only when the source is unambiguous.
        source_tree = cKDTree(points)
        d, rows = source_tree.query(xyz, k=2, distance_upper_bound=tolerance)
        valid = np.isfinite(d[:, 0])
        tied = valid & np.isfinite(d[:, 1])
        at = np.flatnonzero(tied)
        if len(at):
            ambiguous[at] |= ids[rows[at, 0]] != ids[rows[at, 1]]
        at = np.flatnonzero(valid)
        equal = np.isfinite(best[at])
        ambiguous[at[equal]] |= result[at[equal]] != ids[rows[at[equal], 0]]
        better = at[d[at, 0] < best[at]]
        result[better] = ids[rows[better, 0]]
        best[better] = d[better, 0]
    if not np.isfinite(best).all():
        raise ValueError('Source LAS does not match every selected LNT record')
    if ambiguous.any():
        raise ValueError('Coincident returns from different flight lines cannot be distinguished in LNT v2')
    return result
