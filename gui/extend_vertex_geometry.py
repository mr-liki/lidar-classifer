"""Angle-preserving Extend for existing editable open Draw polylines.

Extending an N-shaped SmartLine never creates or splits geometry. A middle
vertex cannot move while its neighbors stay fixed *and* preserve the direction
of the next segment. Instead, Extend changes the chosen incident edge while
translating the untouched connected side rigidly. The other side's segment
lengths, directions, vertex angles, and stored Z values remain intact.

Endpoint Extend behavior remains unchanged. All results are computed without
mutating the source or its drawing metadata.
"""
from __future__ import annotations

import math

OPEN_LINE_TYPES = frozenset({'smartline', 'line', 'line_segment', 'polyline'})
FORBIDDEN_SOURCES = frozenset({'block_creation', 'cell_creation', 'snt_readonly'})
EPSILON = 1e-10


def open_vertices(drawing):
    """Return immutable vertices for an editable *open* Draw line."""
    if not isinstance(drawing, dict) or drawing.get('type') not in OPEN_LINE_TYPES:
        return ()
    if (drawing.get('source') in FORBIDDEN_SOURCES
            or drawing.get('imported') or drawing.get('snt_committed')
            or drawing.get('snt_stored') or drawing.get('classified_fence')):
        return ()
    raw = drawing.get('coords')
    if raw is None:
        return ()
    try:
        coords = tuple(tuple(float(v) for v in p) for p in raw)
        if (len(coords) < 2 or any(len(p) < 2 or not all(map(math.isfinite, p))
                                    for p in coords)):
            return ()
        if math.dist(coords[0][:2], coords[-1][:2]) <= EPSILON:
            return ()   # no edits to closed polygons/rings
        return coords
    except (TypeError, ValueError, IndexError, OverflowError):
        return ()


def nearest_vertex(drawings, cursor, tolerance):
    """Pick an actual control vertex, not an interpolated segment midpoint."""
    if tolerance <= 0 or not math.isfinite(tolerance):
        return None
    try:
        cx, cy = float(cursor[0]), float(cursor[1])
    except (ValueError, TypeError, IndexError):
        return None
    if not (math.isfinite(cx) and math.isfinite(cy)):
        return None
    best = (tolerance, None)
    for drawing in drawings:
        for index, point in enumerate(open_vertices(drawing)):
            distance = math.hypot(cx - point[0], cy - point[1])
            if distance <= best[0]:
                best = (distance, (drawing, index))
    return best[1]


def adjacent_sides(drawing, index):
    """Legal existing edge(s) for a vertex, in source-polyline order."""
    coords = open_vertices(drawing)
    if not coords or not (0 <= index < len(coords)):
        return ()
    result = []
    if index > 0 and math.dist(coords[index][:2], coords[index - 1][:2]) > EPSILON:
        result.append('previous')
    if index < len(coords) - 1 and math.dist(coords[index][:2], coords[index + 1][:2]) > EPSILON:
        result.append('next')
    return tuple(result)


def _neighbor(coords, index, side):
    if not (0 <= index < len(coords)):
        return None
    allowed = []
    if index > 0 and math.dist(coords[index][:2], coords[index - 1][:2]) > EPSILON:
        allowed.append('previous')
    if index < len(coords) - 1 and math.dist(coords[index][:2], coords[index + 1][:2]) > EPSILON:
        allowed.append('next')
    if side is None and len(allowed) == 1:
        side = allowed[0]
    if side not in allowed:
        return None
    return coords[index - 1 if side == 'previous' else index + 1]


def _new_coords(coords, index, target, side=None):
    """Extend a chosen edge without turning the *other* edges of an N.

    For middle vertex B in A-B-C-D:

    - previous edge (or 360 Free's default): move B,C,D together by delta;
      A-B is stretched, while B-C and C-D retain direction, length, angles.
    - next edge: move A,B together by delta; B-C is stretched while A-B
      and C-D retain their directions.

    Only the selected edge is lengthened/redirected; unaffected chain sections
    translate as a unit. Endpoints A and D retain the existing behavior.
    No vertices, drawings, line chains, or branches are added or removed.
    """
    if not (0 <= index < len(coords)):
        return None
    current = coords[index]
    if (len(target) < 2 or not all(map(math.isfinite, target)) or
            math.hypot(target[0] - current[0], target[1] - current[1]) <= EPSILON):
        return None
    dx, dy = target[0] - current[0], target[1] - current[1]
    new = list(coords)
    if 0 < index < len(coords) - 1:
        # Free defaults to extending the PREVIOUS edge: no new mode or prompt.
        moved = (range(index + 1) if side == 'next'
                 else range(index, len(coords)))
    else:
        # Previously working endpoint Extend stays exactly as before.
        moved = (index,)
    for j in moved:
        old = coords[j]
        new[j] = (old[0] + dx, old[1] + dy, *old[2:])
    # Preserve the previous edit's validation scope: only incident edges.
    # The other edges have been translated without changing their lengths.
    for neighbor in (index - 1, index + 1):
        if 0 <= neighbor < len(coords) and math.dist(new[index][:2], new[neighbor][:2]) <= EPSILON:
            return None
    if math.dist(new[0][:2], new[-1][:2]) <= EPSILON:
        return None
    return 'replace', tuple(new)


def extension_from_vertex(drawing, index, cursor, mode, side=None):
    """Reshape exactly one existing vertex using the pointer position.

    Free: unrestricted XY drag of the selected edge's endpoint, translating
    subsequent chain vertices to avoid changing the NEXT edge's direction.
    Fixed: constrain movement outward along the chosen incident edge.
    Interior Fixed/Distance requires an explicit previous/next side.
    Always returns ('replace', same_number_of_vertices) or None.
    """
    coords = open_vertices(drawing)
    if not coords or mode not in ('free', 'fixed') or not (0 <= index < len(coords)):
        return None
    try:
        x, y = float(cursor[0]), float(cursor[1])
    except (ValueError, TypeError, IndexError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    anchor = coords[index]
    if mode == 'fixed':
        neighbor = _neighbor(coords, index, side)
        if neighbor is None:
            return None
        dx, dy = anchor[0] - neighbor[0], anchor[1] - neighbor[1]
        length = math.hypot(dx, dy)
        if length <= EPSILON:
            return None
        ux, uy = dx / length, dy / length
        additional = (x - anchor[0]) * ux + (y - anchor[1]) * uy
        if additional <= EPSILON:
            return None
        x, y = anchor[0] + ux * additional, anchor[1] + uy * additional
    return _new_coords(coords, index, (x, y), side=side)


def extension_by_distance(drawing, index, distance, side=None):
    """Extend an existing incident segment a signed-off *positive* distance.

    Endpoints choose their only incident edge. An interior vertex requires
    previous/next to avoid silently choosing the wrong edge of an N shape.
    Does not add a point: for a middle joint, it translates the connected
    side as a rigid unit so its next segment does not turn.
    """
    coords = open_vertices(drawing)
    if not coords or not (0 <= index < len(coords)):
        return None
    try:
        distance = float(distance)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(distance) or distance <= EPSILON:
        return None
    neighbor = _neighbor(coords, index, side)
    if neighbor is None:
        return None
    anchor = coords[index]
    dx, dy = anchor[0] - neighbor[0], anchor[1] - neighbor[1]
    length = math.hypot(dx, dy)
    if length <= EPSILON:
        return None
    return _new_coords(coords, index, (
        anchor[0] + distance * dx / length,
        anchor[1] + distance * dy / length,
    ), side=side)


# A user-drawn Line chain is stored as consecutive two-vertex records in the
# BRUSH native editor. Treating only one record as a polyline wrongly breaks
# shared N corners. These virtual views exist ONLY during Extend interaction;
# they are never inserted into the drawing list or saved as a new entity.

def _same_xy(a, b):
    return a[0] == b[0] and a[1] == b[1]


def _joined_line_records(left, right):
    if not (isinstance(left, dict) and isinstance(right, dict)):
        return False
    if left.get('type') != right.get('type') or left.get('type') not in ('line', 'line_segment'):
        return False
    a, b = open_vertices(left), open_vertices(right)
    if len(a) != 2 or len(b) != 2 or not _same_xy(a[-1], b[0]):
        return False
    chain_a, chain_b = left.get('line_chain'), right.get('line_chain')
    if chain_a is not None or chain_b is not None:
        return chain_a is not None and chain_a == chain_b
    # Older VTK Line emits separate line_segment records with shared marker
    # object identities, one actor per segment. Never infer unrelated joins
    # from their being coincident on screen or sequential in the list.
    if left.get('type') == 'line_segment':
        marker = left.get('end_marker')
        return marker is not None and marker is right.get('start_marker')
    return False


def extend_virtual_line(drawings, drawing):
    """Produce an ephemeral joined view of a *proven* Line chain, if any."""
    if not isinstance(drawing, dict) or drawing.get('type') not in ('line', 'line_segment'):
        return drawing
    if len(open_vertices(drawing)) != 2:
        return drawing
    position = next((i for i, item in enumerate(drawings) if item is drawing), -1)
    if position < 0:
        return drawing
    start = position
    end = position
    while start > 0 and _joined_line_records(drawings[start - 1], drawings[start]):
        start -= 1
    while end + 1 < len(drawings) and _joined_line_records(drawings[end], drawings[end + 1]):
        end += 1
    if start == end:
        return drawing
    segments = tuple(drawings[start:end + 1])
    vertices = [tuple(open_vertices(segments[0])[0])]
    vertices.extend(tuple(open_vertices(item)[1]) for item in segments)
    # Do not silently convert closed rings or malformed chains into an open N.
    candidate = {'type': 'smartline', 'coords': vertices,
                 '_extend_chain_records': segments}
    # A connected closed chain is never an editable open line, and must not
    # fall back to editing a single segment of that protected closed ring.
    return candidate


def extend_candidates(drawings, preferred):
    """Expand selection groups into logical chains without changing records."""
    emitted = set()
    for drawing in preferred:
        if not isinstance(drawing, dict):
            continue
        virtual = extend_virtual_line(drawings, drawing)
        if not open_vertices(virtual):
            continue
        originals = virtual.get('_extend_chain_records', (drawing,))
        key = tuple(id(item) for item in originals)
        if key in emitted:
            continue
        emitted.add(key)
        yield virtual


def extend_target_live(drawings, target):
    """A pending click must not write to stale or removed drawings."""
    originals = target.get('_extend_chain_records', (target,))
    live = {id(d): d for d in drawings}
    if not all(live.get(id(item)) is item for item in originals):
        return False
    if len(originals) > 1:
        # Chain membership and its connectivity must not have changed during
        # a pending modal dialog / viewport callback.
        reassembled = extend_virtual_line(drawings, originals[0])
        if (len(reassembled.get('_extend_chain_records', ())) != len(originals) or
                any(a is not b for a, b in zip(reassembled.get('_extend_chain_records', ()), originals)) or
                tuple(open_vertices(reassembled)) != tuple(open_vertices(target))):
            return False
    return True


def extend_replacements(drawings, target, new_coords):
    """Return (old, new-coords) records changed by an atomic vertex edit.

    When a virtual Line chain is extended at its middle, the unchanged side
    is translated with the chosen corner so adjacent segment angles remain
    intact. The existing records are updated together; there are no new
    vertices, Draw entities, or branches.
    """
    if not extend_target_live(drawings, target):
        return ()
    records = target.get('_extend_chain_records')
    if not records:
        old = open_vertices(target)
        if len(old) != len(new_coords) or tuple(old) == tuple(new_coords):
            return ()
        return ((target, tuple(new_coords)),)
    if len(new_coords) != len(records) + 1:
        return ()
    updates = []
    for i, record in enumerate(records):
        old = open_vertices(record)
        if len(old) != 2:
            return ()
        changed = []
        for j in (0, 1):
            target_pos = new_coords[i + j]
            # Retain each source control point's non-XY dimensions, e.g. Z.
            changed.append((float(target_pos[0]), float(target_pos[1]), *old[j][2:]))
        if tuple(changed) != old:
            if math.hypot(changed[1][0]-changed[0][0],
                          changed[1][1]-changed[0][1]) <= EPSILON:
                return ()
            updates.append((record, tuple(changed)))
    return tuple(updates)


def vertex_displacement_meters(before, after, canvas_crs=None):
    """World-XY displacement of one selected vertex, expressed in metres.

    A projected CRS is converted from its native axis units (e.g. US survey
    feet); a geographic CRS uses a geodesic distance, not degrees as metres.
    With no declared canvas CRS, follow the existing Measure-tool convention
    that drawing world units represent metres. The display must never sum the
    displacement of translated neighbouring N-shape vertices.
    """
    try:
        x0, y0 = float(before[0]), float(before[1])
        x1, y1 = float(after[0]), float(after[1])
    except (IndexError, TypeError, ValueError, OverflowError):
        return None
    if not all(map(math.isfinite, (x0, y0, x1, y1))):
        return None
    dx, dy = x1 - x0, y1 - y0
    if canvas_crs is not None:
        try:
            from pyproj import CRS
            crs = CRS.from_user_input(canvas_crs)
            if crs.is_geographic:
                _, _, distance = crs.get_geod().inv(x0, y0, x1, y1)
                return abs(float(distance)) if math.isfinite(distance) else None
            if crs.is_projected and len(crs.axis_info) >= 2:
                fx = float(crs.axis_info[0].unit_conversion_factor)
                fy = float(crs.axis_info[1].unit_conversion_factor)
                if math.isfinite(fx) and math.isfinite(fy) and fx > 0 and fy > 0:
                    return math.hypot(dx * fx, dy * fy)
        except Exception:
            # Unit metadata is optional; a malformed/unavailable CRS must not
            # interrupt the already successful geometry edit.
            pass
    return math.hypot(dx, dy)


def extended_edge_length_meters(drawing, index, new_coords, side=None):
    """Final LENGTH of the edited edge, using Measure -> Line's metric.

    This is not the selected vertex's displacement. Extend's geometry edits
    one existing segment between adjacent control vertices, so the reference
    points are that segment's final endpoints (A and C in A--B--C).

    Measure -> Line uses Euclidean distance of saved world XYZ coordinates
    (``numpy.linalg.norm(p2 - p1)``); this matches that same convention without
    altering the separate measurement tool, reprojecting points or summing a
    polyline's unrelated edges. When vertices only contain XY, use XY.

    ``side`` is the explicitly chosen 'previous'/'next' incident edge for a
    middle vertex. Free mode defaults to the previous edge, consistent with
    _new_coords. The first vertex uses its next edge, and the last its previous.
    Line-chain virtual geometry is supported without creating any drawings.
    """
    original = open_vertices(drawing)
    if not original or not isinstance(index, int) or not 0 <= index < len(original):
        return None
    if side is None:
        side = 'next' if index == 0 else 'previous'
    if side not in adjacent_sides(drawing, index):
        return None
    try:
        if len(new_coords) != len(original):
            return None
        other_index = index - 1 if side == 'previous' else index + 1
        endpoint_a = tuple(float(x) for x in new_coords[other_index])
        endpoint_c = tuple(float(x) for x in new_coords[index])
    except (ValueError, TypeError, IndexError, OverflowError):
        return None
    # All editable Draw polyline vertices use XY or XYZ. Stay in the same
    # world-space metric as Measure -> Line; do not convert via canvas CRS.
    if (len(endpoint_a) not in (2, 3) or len(endpoint_a) != len(endpoint_c)
            or not all(map(math.isfinite, endpoint_a + endpoint_c))):
        return None
    distance = math.dist(endpoint_a, endpoint_c)
    return distance if math.isfinite(distance) else None
