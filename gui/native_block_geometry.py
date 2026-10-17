"""Working block geometry ported from E:/naksha-lidar, 2026-10-17.

The splitting, offsets, end caps, priority clipping and small-fragment rules are
copied from gui/block_creation_settings_dialog.py. Only presentation is adapted:
return authored drawing dictionaries for the native viewport, without Qt/VTK.
"""
from types import SimpleNamespace
import numpy as np

SIDE_BLOCK_SMALL_RATIO = 0.50


_SMALL_BLOCK_MERGE_CONFIG = {
    "enabled": True,
    # Verbose diagnostics: logs every candidate, its neighbours, the scores of
    # every possible parent and the reason a merge was accepted or refused.
    "debug": False,
    # Optional absolute cap in squared model units.  None disables it; the
    # decision is then purely relative to the neighbouring blocks.
    "smallBlockAreaThreshold": None,
    # A piece counts as a sliver when it is at most this fraction of the largest
    # piece of its own chunk, or of the smallest block that is larger than it.
    # A corridor cut into comparable halves (an X crossing) stays untouched.
    "smallBlockRelativeAreaRatio": 0.25,
    # User's Block Settings -> Distance for the current run.  Filled in per call
    # by the caller; 0 disables the distance-based side-block rule.
    "blockDistance": 0.0,
    # Fraction of the configured distance below which a side/junction block also
    # counts as a small fragment.  Central definition - never inline a 0.5.
    "sideBlockSmallRatio": SIDE_BLOCK_SMALL_RATIO,
    # Numeric slack on the strict "<" comparison, so a block whose chainage span
    # evaluates to 124.9999999998 for an intended 125.0 does not flicker into a
    # merge.  Deliberately tiny: this must not become a fuzzy threshold.
    "sideBlockDistanceTolerance": 1e-6,
    # Two block boundaries closer than this are the SAME separator.  Noding and
    # floating-point noise must not turn one shared edge into "no neighbour".
    "snapTolerance": 1e-6,
    # Model-space tolerance for the geometry cleanliness checks.
    "geometryTolerance": 1e-6,
    # Line contact shorter than this is vertex-only contact -> not a neighbour.
    "minimumAttachmentLength": 1e-6,
    # Two candidate parents whose attachment ratios differ by less than this
    # (relative) are treated as ambiguous and deliberately left unmerged.
    "ambiguousMergeTolerance": 0.05,
    # Relative tolerance of the area-preservation check on a union.
    "areaTolerance": 1e-6,
    # The dissolved interface must not survive anywhere on the merged boundary.
    "separatorResidualTolerance": 1e-6,
    # Re-derive every drawn block edge from the FINAL faces after a merge, so a
    # separator can never survive inside a block that grew.
    "rebuildLinework": True,
}


def _polyline_length(points):
    """Arc length along a polyline - a block's chainage span along its own CL.

    The distance-based small-block rule measures a block along the CL that
    generated it, so a curved CL keeps its true along-the-line length instead of
    the straight-line distance between the block's first and last vertices.
    """
    if points is None:
        return 0.0
    try:
        pts = np.asarray(points, dtype=np.float64)
    except Exception:
        return 0.0
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return 0.0
    deltas = np.diff(pts[:, :2], axis=0)
    if deltas.size == 0:
        return 0.0
    return float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))


def _normalize_polygon(geom, grid=0.0, area_floor=1e-9):
    """Snap a polygon onto a common precision grid and make it valid.

    Snap rounding is what makes noding noise disappear: two boundaries that only
    differed by a hair end up with *exactly* the same coordinates, so their
    complete shared edge is later found by an exact intersection and a real
    polygon union can dissolve it.  Self-intersecting (bow-tie) or spiky faces
    are repaired, and zero-area leftovers are dropped.

    Returns a list of valid polygons.
    """
    if geom is None or geom.is_empty:
        return []
    snapped = geom
    if grid and grid > 0.0:
        try:
            from shapely import set_precision

            snapped = set_precision(geom, grid)
        except Exception:
            snapped = geom
    if snapped is None or snapped.is_empty:
        return []
    if not snapped.is_valid:
        try:
            snapped = snapped.buffer(0)
        except Exception:
            return []
    parts = (
        list(snapped.geoms)
        if getattr(snapped, "geom_type", "") == "MultiPolygon"
        else [snapped]
    )
    out = []
    for part in parts:
        if getattr(part, "geom_type", "") != "Polygon" or part.is_empty:
            continue
        if not part.is_valid:
            try:
                part = part.buffer(0)
            except Exception:
                continue
            if getattr(part, "geom_type", "") != "Polygon" or part.is_empty:
                continue
        if float(part.area) <= area_floor:
            continue
        out.append(part)
    return out


def _block_piece_geometries(corridor_geometries, area_floor=1e-9, grid=0.0):
    """Realize each corridor against the higher-priority corridors before it.

    Mirrors the polygonization the export path performs, so the pieces seen here
    are exactly the faces that would be written out as separate blocks.  Every
    corridor and every resulting face is snapped to the model-space ``grid`` and
    repaired, so the decomposition is topologically clean and neighbouring faces
    share exactly coincident edges.
    """
    from shapely.ops import unary_union
    from shapely.strtree import STRtree

    normalized = []
    for geom in corridor_geometries:
        parts = _normalize_polygon(geom, grid, area_floor)
        normalized.append(unary_union(parts) if len(parts) > 1 else (parts[0] if parts else None))

    # Index the corridors once, so each one is only clipped against the (few)
    # higher-priority corridors it actually overlaps instead of re-unioning the
    # whole prefix of the list for every block - that prefix union is quadratic
    # and dominates the stage on a large selection.
    indexed = [geom for geom in normalized if geom is not None and not geom.is_empty]
    tree = STRtree(indexed) if indexed else None
    owners = {id(geom): index for index, geom in enumerate(normalized) if geom is not None}

    realized = []
    for index, geom in enumerate(normalized):
        if geom is None or geom.is_empty:
            realized.append([])
            continue
        higher_geoms = []
        if tree is not None:
            try:
                for position in tree.query(geom):
                    candidate = indexed[int(position)]
                    if owners.get(id(candidate), index) < index:
                        higher_geoms.append(candidate)
            except Exception:
                higher_geoms = [
                    other for other in normalized[:index] if other is not None and not other.is_empty
                ]
        effective = geom.difference(unary_union(higher_geoms)) if higher_geoms else geom
        if effective.is_empty:
            realized.append([])          # fully absorbed by higher-priority blocks
            continue
        pieces = (
            list(effective.geoms)
            if getattr(effective, "geom_type", "") == "MultiPolygon"
            else [effective]
        )
        kept = []
        for piece in pieces:
            # The overlay already ran on grid-snapped inputs, so the faces only
            # need a validity/area check - re-snapping every face is expensive
            # and changes nothing.
            kept.extend(_normalize_polygon(piece, 0.0, area_floor))
        realized.append(kept)
    return realized


def _boundary_overlap(geom_a, geom_b, tolerance=1e-6, minimum_length=1e-6):
    """The COMPLETE separator between two blocks, as one geometry.

    Returns ``(separator, exact_length, tolerant_length)``.  The exact stage
    intersects the two boundaries and keeps *all* linear components (a noded
    separator arrives as e1+e2+e3 and is never reduced to the first segment).
    When noding nudged one of the two copies of the line, the tolerant stage
    matches each boundary against a thin band around the other one, so a shared
    edge whose coordinates differ by a hair is still recognised as a neighbour.
    A vertex-only contact yields an empty separator and zero lengths.
    """
    if geom_a is None or geom_b is None or geom_a.is_empty or geom_b.is_empty:
        return None, 0.0, 0.0
    try:
        exact = geom_a.boundary.intersection(geom_b.boundary)
    except Exception:
        exact = None
    exact_length = float(exact.length) if exact is not None and not exact.is_empty else 0.0
    tolerant, tolerant_length = None, 0.0
    if tolerance > 0.0 and exact_length <= minimum_length:
        # Only the exact stage failing makes the (expensive) tolerant match
        # worthwhile, and only for polygons that are actually close by.
        try:
            if geom_a.distance(geom_b) <= tolerance:
                band = geom_b.boundary.buffer(
                    tolerance, quad_segs=1, cap_style=2, join_style=2
                )
                tolerant = geom_a.boundary.intersection(band)
                if tolerant is not None and tolerant.is_empty:
                    tolerant = None
                else:
                    tolerant_length = float(tolerant.length)
        except Exception:
            tolerant, tolerant_length = None, 0.0
    # A tolerant match no longer than a few band widths is the two boundaries
    # grazing each other at a shared corner - that is point contact, not an
    # interface, and must never make two blocks neighbours.
    if tolerant_length <= max(minimum_length, 10.0 * tolerance):
        tolerant, tolerant_length = None, 0.0
    if tolerant_length > exact_length:
        geometry, length = tolerant, tolerant_length
    else:
        geometry, length = exact, exact_length
    if length <= minimum_length:
        return None, exact_length, tolerant_length
    return geometry, exact_length, tolerant_length


def _shared_boundary_length(geom_a, geom_b, minimum_length=1e-6, tolerance=1e-6):
    """Length of the boundary two polygons really share (0 for vertex-only contact)."""
    separator, _exact, _tolerant = _boundary_overlap(
        geom_a, geom_b, tolerance, minimum_length
    )
    return 0.0 if separator is None else float(separator.length)


def _search_envelope(geom, tolerance):
    """Bounding box of ``geom`` grown by ``tolerance``.

    A spatial-index query only reports boxes that overlap, so a neighbour that
    sits a few microns away (snap rounding, noding noise) would be invisible and
    the fragment would be reported as having no parent at all.
    """
    from shapely.geometry import box

    minx, miny, maxx, maxy = geom.bounds
    return box(minx - tolerance, miny - tolerance, maxx + tolerance, maxy + tolerance)


def _fragment_components(fragment_keys, block_pieces, minimum_shared, tolerance=1e-6):
    """Group small faces that touch each other, so a chain LARGE-s1-s2 merges as one."""
    parent_of = {key: key for key in fragment_keys}

    def _find(key):
        while parent_of[key] != key:
            parent_of[key] = parent_of[parent_of[key]]
            key = parent_of[key]
        return key

    keys = list(fragment_keys)
    for index, key_a in enumerate(keys):
        for key_b in keys[index + 1:]:
            if not _shared_boundary_length(
                block_pieces[key_a[0]][key_a[1]],
                block_pieces[key_b[0]][key_b[1]],
                minimum_shared,
                tolerance,
            ):
                continue
            root_a, root_b = _find(key_a), _find(key_b)
            if root_a != root_b:
                parent_of[root_b] = root_a

    components = {}
    for key in keys:
        components.setdefault(_find(key), []).append(key)
    return [sorted(members) for members in components.values()]


def _plan_small_fragment_merges(block_pieces, config, source_ids=None, longitudinal_spans=None):
    """Decide which small faces are absorbed by which neighbouring block.

    ``block_pieces`` is one list of shapely polygons per block, in priority
    order.  ``source_ids`` optionally carries the originating CL line of every
    block, so two chunks cut from the SAME CL line are never merged into each
    other (that would weld ordinary distance-split blocks together).

    ``longitudinal_spans`` optionally carries ``(startChainage, endChainage,
    length)`` per block - how far the block runs ALONG its own CL.  It feeds the
    additive distance rule: a SIDE / JUNCTION block shorter than half of the
    configured Block Distance becomes a small-fragment candidate too.

    Classification runs on the ORIGINAL faces only and the resulting set is
    frozen: a parent that grew during a merge can never become a new candidate.

    Returns ``{"merges", "skipped", "details", "fragments", "distanceFragments"}``:
    * ``merges``  - ``{(block, piece) of the parent: [(block, piece), ...] faces}``
    * ``skipped`` - ``[((block, piece), reason), ...]`` left untouched on purpose
    * ``details`` - per-face separator, attachment ratio, areas, neighbour count
    * ``distanceFragments`` - candidates that ONLY the distance rule added
    """
    result = {
        "merges": {},
        "skipped": [],
        "details": {},
        "fragments": [],
        "distanceFragments": [],
    }
    if not config.get("enabled", True):
        return result

    ratio = float(config.get("smallBlockRelativeAreaRatio") or 0.0)
    raw_threshold = config.get("smallBlockAreaThreshold")
    threshold = float(raw_threshold) if raw_threshold is not None else None
    tolerance = float(config.get("snapTolerance") or 0.0)
    minimum_attachment = float(config.get("minimumAttachmentLength") or 1e-6)
    ambiguity = float(config.get("ambiguousMergeTolerance") or 0.0)
    debug = bool(config.get("debug"))

    # Additive distance rule configuration (see SIDE_BLOCK_SMALL_RATIO).
    block_distance = float(config.get("blockDistance") or 0.0)
    side_ratio = float(config.get("sideBlockSmallRatio") or SIDE_BLOCK_SMALL_RATIO)
    distance_tolerance = float(config.get("sideBlockDistanceTolerance") or 0.0)
    short_threshold = block_distance * side_ratio if block_distance > 0.0 else None
    primary_source = min(source_ids) if source_ids else None

    def _span_for(block_index):
        """``(start, end, length)`` along the CL for one block, or ``None``."""
        if not longitudinal_spans or block_index >= len(longitudinal_spans):
            return None
        span = longitudinal_spans[block_index]
        if not span:
            return None
        try:
            return float(span[0]), float(span[1]), float(span[2])
        except (TypeError, ValueError, IndexError):
            return None

    keys = [
        (block_index, piece_index)
        for block_index, pieces in enumerate(block_pieces)
        for piece_index in range(len(pieces))
    ]
    if not keys:
        return result

    def _area(key):
        return float(block_pieces[key[0]][key[1]].area)

    def _same_source(block_a, block_b):
        return source_ids is not None and source_ids[block_a] == source_ids[block_b]

    # ---- Phase 1: true edge adjacency and the COMPLETE separator per pair ----
    # The separator is kept in full (every noded component), exact first and
    # tolerant second, so a boundary split into e1+e2+e3 is never half-matched.
    from shapely.strtree import STRtree

    tree = STRtree([block_pieces[block][piece] for block, piece in keys])
    adjacency = {}
    for key in keys:
        piece = block_pieces[key[0]][key[1]]
        try:
            positions = tree.query(_search_envelope(piece, tolerance))
        except Exception:
            continue
        for position in positions:
            other_key = keys[int(position)]
            if other_key == key or other_key[0] == key[0]:
                continue
            if _same_source(key[0], other_key[0]):
                continue
            other = block_pieces[other_key[0]][other_key[1]]
            separator, exact_length, tolerant_length = _boundary_overlap(
                piece, other, tolerance, minimum_attachment
            )
            if separator is None:
                continue
            adjacency.setdefault(key, {})[other_key] = {
                "separator": separator,
                "exactLength": exact_length,
                "tolerantLength": tolerant_length,
                "length": float(separator.length),
            }

    # ---- Phase 2: classify the ORIGINAL faces, then freeze the candidate set ----
    fragments = set()
    # Candidates that ONLY the distance rule contributed, tracked separately so
    # the two criteria stay distinguishable in the logs.
    distance_fragments = set()
    reasons = {}
    for block_index, pieces in enumerate(block_pieces):
        if not pieces:
            continue
        largest_own = max(float(piece.area) for piece in pieces)

        # A SIDE / JUNCTION block is a block generated from a CL other than the
        # highest-priority one that is ALSO in contact with another block's
        # corridor.  Cross-CL contact can only exist where two corridors meet,
        # i.e. at a junction, so an ordinary terminal chunk of one long CL never
        # qualifies - and a side block merely shorter than the target distance
        # does not qualify on its own either.
        span = _span_for(block_index)
        side_block = bool(
            source_ids is not None
            and primary_source is not None
            and source_ids[block_index] != primary_source
            and any((block_index, index) in adjacency for index in range(len(pieces)))
        )
        short_side = bool(
            side_block
            and short_threshold is not None
            and span is not None
            and span[2] < short_threshold - distance_tolerance
        )

        for piece_index, piece in enumerate(pieces):
            key = (block_index, piece_index)
            area = float(piece.area)
            neighbours = adjacency.get(key) or {}
            bigger = [other for other in neighbours if _area(other) > area]
            is_small, reason = False, "not_small"
            if len(pieces) > 1 and ratio > 0.0 and area <= ratio * largest_own:
                is_small, reason = True, "leftover_of_own_chunk"
            elif bigger and ratio > 0.0 and area <= ratio * min(_area(other) for other in bigger):
                is_small, reason = True, "small_vs_neighbour"
            if not is_small and threshold is not None and area <= threshold:
                is_small, reason = True, "below_area_threshold"
            existing_small = is_small

            # Additive rule: an eligible side block running less than half of the
            # configured Block Distance along its CL is a small fragment as well.
            # It only ever ADDS candidates - the existing detection above still
            # decides on its own, and ``fragments`` is a set, so a face that both
            # tests flag is still exactly one candidate.
            if short_side:
                reasons.setdefault(key, []).append("short_side_block")
                distance_fragments.add(key)
                if not is_small:
                    is_small, reason = True, "short_side_block"

            if is_small:
                fragments.add(key)
            if debug and (is_small or bigger):
                print(
                    f"[SmallBlockMerge] classify block{block_index + 1}/piece{piece_index} "
                    f"area={area:.4f} ownChunkLargest={largest_own:.4f} "
                    f"biggerNeighbours={len(bigger)} isSmall={is_small} reason={reason}"
                )
            if side_block and span is not None and (debug or short_side):
                print(
                    f"[SmallBlockCandidate] id=block{block_index + 1}/piece{piece_index} "
                    f"type=SIDE_BLOCK sourceCL={source_ids[block_index]} "
                    f"startChainage={span[0]:.2f} endChainage={span[1]:.2f} "
                    f"blockDistance={span[2]:.2f} "
                    f"configuredDistance={block_distance:.2f} "
                    f"threshold50={short_threshold:.2f} "
                    f"existingSmall={existing_small} shortSide={short_side} "
                    f"finalClassification="
                    f"{'SMALL_FRAGMENT' if is_small else 'NORMAL_BLOCK'}"
                )

    result["fragments"] = sorted(fragments)
    result["distanceFragments"] = sorted(distance_fragments)
    if not fragments:
        return result

    def _adjacent_keys(key):
        return list((adjacency.get(key) or {}).keys())

    # ---- Phase 3: one parent per connected component of small faces ----
    # Components are processed smallest-first and every face can be claimed by
    # exactly one of them: two components that grow into each other must not both
    # hand the same face to the same parent (that would double its area).
    components = _fragment_components(fragments, block_pieces, minimum_attachment, tolerance)
    components = sorted(components, key=lambda members: (len(members), members))
    claimed = set()
    for component in components:
        component = [key for key in component if key not in claimed]
        if not component:
            continue
        component_area = sum(_area(key) for key in component)
        total_perimeter = sum(float(block_pieces[b][p].length) for b, p in component)

        candidates = {}
        separator_lengths = {}
        for key in component:
            for parent_key, info in (adjacency.get(key) or {}).items():
                if parent_key in fragments:
                    continue          # a real block is always preferred over a sliver
                if _area(parent_key) <= component_area:
                    continue          # only ever absorb into a LARGER neighbour
                separator_lengths.setdefault(parent_key, []).append(float(info["length"]))
                entry = candidates.setdefault(parent_key, {"shared": 0.0, "area": 0.0})
                entry["shared"] += float(info["length"])
                entry["area"] = max(entry["area"], _area(parent_key))

        if not candidates or total_perimeter <= 0.0:
            claimed.update(component)
            for key in component:
                result["skipped"].append((key, "no_parent"))
            if debug:
                for key in component:
                    print(
                        f"[SmallBlockMerge] fragment=block{key[0] + 1}/piece{key[1]} "
                        f"area={_area(key):.4f} isSmall=true candidateParents=0 "
                        f"result=NO_PARENT_FOUND"
                    )
            continue

        scored = [
            (info["shared"] / total_perimeter, info["shared"], info["area"], key)
            for key, info in candidates.items()
        ]
        scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        best = scored[0]

        # ---- Phase 4: grow the component along touching small faces ----
        # LARGE - s1 - s2 is allowed to become LARGE + (s1 + s2), but never
        # across another chunk of a CL line already taking part in this merge.
        parent_area = _area(best[3])
        members = set(component)
        blocked_sources = set()
        if source_ids is not None:
            blocked_sources = {source_ids[best[3][0]]}
            blocked_sources.update(source_ids[block_index] for block_index, _ in component)
        growing = ratio > 0.0
        while growing:
            growing = False
            for member in list(members):
                for neighbour_key in _adjacent_keys(member):
                    if neighbour_key in members:
                        continue
                    if source_ids is not None and source_ids[neighbour_key[0]] in blocked_sources:
                        continue
                    if neighbour_key in claimed:
                        continue
                    if _area(neighbour_key) > ratio * parent_area:
                        continue
                    members.add(neighbour_key)
                    if source_ids is not None:
                        blocked_sources.add(source_ids[neighbour_key[0]])
                    growing = True
        component = sorted(members)

        if parent_area <= sum(_area(key) for key in component):
            claimed.update(component)
            for key in component:
                result["skipped"].append((key, "no_parent"))
            continue

        if len(scored) > 1 and best[0] > 0.0:
            runner_up = scored[1]
            if best[0] - runner_up[0] <= ambiguity * best[0]:
                claimed.update(component)
                for key in component:
                    result["skipped"].append((key, "ambiguous"))
                if debug:
                    print(
                        f"[SmallBlockMerge] result=SKIPPED_AMBIGUOUS "
                        f"parentA=block{best[3][0] + 1} scoreA={best[0]:.3f} "
                        f"parentB=block{runner_up[3][0] + 1} scoreB={runner_up[0]:.3f}"
                    )
                continue

        claimed.update(component)
        result["merges"].setdefault(best[3], []).extend(component)
        for block_index, piece_index in component:
            key = (block_index, piece_index)
            separator = (adjacency.get(key) or {}).get(best[3])
            if separator is None:
                for other in component:
                    separator = (adjacency.get(key) or {}).get(other)
                    if separator is not None:
                        break
            result["details"][key] = {
                "parent": best[3],
                "separator": (separator or {}).get("separator"),
                "sharedLength": float(sum(separator_lengths.get(best[3], [0.0]))),
                "exactSharedLength": float((separator or {}).get("exactLength", 0.0)),
                "attachmentRatio": float(best[0]),
                "area": _area(key),
                "neighbourArea": float(candidates[best[3]]["area"]),
                "neighbourCount": len(candidates),
                # Which rule(s) made this face a candidate.  Lets a debug overlay
                # or reader tell an existing small fragment from a block that is
                # small only because of the distance rule.
                "reasons": tuple(reasons.get(key, ())),
            }
    return result


def _snap_pair(first, second, tolerance):
    """Move each geometry's vertices onto the other one within ``tolerance``.

    Two faces that came from different corridors routinely share an edge that is
    *almost* coincident (the two copies of the line differ by ~1e-14).  A plain
    union then keeps that hair-thin crack and the separator survives on the
    merged boundary.  Reconciling the interface first makes the two copies
    exactly equal, so the overlay can dissolve the whole thing.
    """
    if tolerance <= 0.0:
        return first, second
    try:
        import shapely

        first_snapped = shapely.snap(first, second, tolerance)
        second_snapped = shapely.snap(second, first_snapped, tolerance)
    except Exception:
        return first, second
    if first_snapped is None or first_snapped.is_empty:
        first_snapped = first
    if second_snapped is None or second_snapped.is_empty:
        second_snapped = second
    return first_snapped, second_snapped


def _union_faces(parent_geom, fragment_geoms, config):
    """Union a parent block with its small faces; ``None`` when unusable.

    Returns ``(merged, reason, detail)``.  The union is a real polygon overlay -
    no edge is deleted by hand - and the result is only accepted when it is one
    valid connected polygon, its area equals parent + fragments, and the whole
    dissolved interface has left the merged boundary.
    """
    from shapely.ops import unary_union

    tolerance = float(config.get("geometryTolerance") or 1e-6)
    snap_tolerance = float(config.get("snapTolerance") or tolerance)
    area_tolerance = float(config.get("areaTolerance") or 1e-6)
    residual_tolerance = float(config.get("separatorResidualTolerance") or tolerance)
    minimum_attachment = float(config.get("minimumAttachmentLength") or 1e-6)

    # Reconcile every interface before the overlay (see _snap_pair).
    parent_snapped = parent_geom
    fragments_snapped = []
    for fragment in fragment_geoms:
        parent_snapped, fragment_snapped = _snap_pair(
            parent_snapped, fragment, snap_tolerance
        )
        fragments_snapped.append(fragment_snapped)

    try:
        merged = unary_union([parent_snapped, *fragments_snapped])
    except Exception as exc:
        return None, "UNION_INVALID", str(exc)
    if merged is None or merged.is_empty:
        return None, "UNION_INVALID", "empty"
    if merged.geom_type != "Polygon":
        return None, "MULTIPOLYGON_RESULT", merged.geom_type
    if not merged.is_valid:
        merged = merged.buffer(0)
        if merged.is_empty or merged.geom_type != "Polygon":
            return None, "UNION_INVALID", "still invalid after buffer(0)"

    expected = float(parent_snapped.area) + sum(
        float(geom.area) for geom in fragments_snapped
    )
    merged_area = float(merged.area)
    if abs(merged_area - expected) > max(tolerance, expected * area_tolerance):
        return None, "AREA_CHECK_FAILED", f"{merged_area:.6f} != {expected:.6f}"

    # Every interface that was supposed to be dissolved must now be interior
    # geometry - never a drawable part of the merged exterior boundary.  The
    # residual is judged against the separator length, not a bare epsilon, so a
    # sub-micron noding crumb is fine while a surviving half-edge is not.
    pairs = [(parent_snapped, geom) for geom in fragments_snapped]
    for index, first in enumerate(fragments_snapped):
        for second in fragments_snapped[index + 1:]:
            pairs.append((first, second))
    residual = 0.0
    for first, second in pairs:
        separator, _exact, _tolerant = _boundary_overlap(
            first, second, snap_tolerance, minimum_attachment
        )
        if separator is None:
            continue
        try:
            leftover = separator.intersection(merged.boundary)
        except Exception:
            continue
        if not leftover.is_empty:
            residual = max(
                residual, float(leftover.length) / max(float(separator.length), tolerance)
            )
    if residual > residual_tolerance:
        return None, "RESIDUAL_SEPARATOR", f"fraction={residual:.6f}"

    return merged, "SUCCESS", f"area {expected:.4f} -> {merged_area:.4f}"


class NativeBlockGeometry:
    def __init__(self, app, selected, above, below, distance):
        self.app = app
        self.selected_cl_lines = selected
        self._generated_block_drawings = []
        self.mode_combo = SimpleNamespace(currentData=lambda: 'asymmetric')
        self.distance_spin = SimpleNamespace(value=lambda: distance)
        self.tol_above_spin = SimpleNamespace(value=lambda: above)
        self.tol_below_spin = SimpleNamespace(value=lambda: below)

    def _remove_generated_block_drawings(self):
        self._generated_block_drawings.clear()

    def _discard_block_drawings(self, record):
        discarded = {id(d) for d in record.get('drawings', [])}
        self._generated_block_drawings[:] = [d for d in self._generated_block_drawings
                                            if id(d) not in discarded]
        record['drawings'] = []

    @staticmethod
    def _native_outline(coords):
        return {'type': 'smartline', 'source': 'block_creation', 'coords': coords,
                'color': (0, 179, 255), 'width': 3, 'line_style': 'solid',
                'original_color': (0., .7, 1.), 'original_width': 3,
                'original_style': 'solid'}

    def _replace_block_outline(self, record, world_rings, block_color=(0., .7, 1.), z_reference=None):
        heights = [self._outline_z_for_xy(record, ring, z_reference) for ring in world_rings]
        self._discard_block_drawings(record)
        drawings = []
        for ring, zs in zip(world_rings, heights):
            if len(ring) < 2:
                continue
            drawing = self._native_outline([(float(x), float(y), float(z))
                                            for (x, y), z in zip(ring, zs)])
            drawings.append(drawing)
            self._generated_block_drawings.append(drawing)
        record['drawings'] = drawings
        self._tag_block_vertex_groups(drawings)

    @staticmethod
    def _clean_block_edit_chain(points):
        cleaned = []
        for point in list(points or []):
            try:
                xyz = (
                    float(point[0]),
                    float(point[1]),
                    float(point[2]) if len(point) > 2 else 0.0,
                )
            except (TypeError, ValueError, IndexError):
                continue
            if not cleaned or np.hypot(xyz[0] - cleaned[-1][0], xyz[1] - cleaned[-1][1]) > 1e-9:
                cleaned.append(xyz)
        return cleaned


    def _tag_block_vertex_groups(self, drawings):
        """Remember which preview vertices are one logical block corner."""
        groups = []
        for drawing in list(drawings or []):
            original = self._clean_block_edit_chain(drawing.get("coords"))
            drawing["block_creation_original_coords"] = [tuple(point) for point in original]
            vertex_groups = [None] * len(original)
            drawing["block_creation_vertex_groups"] = vertex_groups

            for vertex_index, point in enumerate(original):
                group = next(
                    (
                        candidate
                        for candidate in groups
                        if np.hypot(point[0] - candidate["point"][0], point[1] - candidate["point"][1])
                        <= 1e-6
                    ),
                    None,
                )
                if group is None:
                    group = {"point": point, "members": []}
                    groups.append(group)
                group["members"].append((drawing, vertex_index))

        for group_index, group in enumerate(groups, 1):
            if len(group["members"]) < 2:
                continue
            group_id = f"block-junction-{group_index}"
            for drawing, vertex_index in group["members"]:
                drawing["block_creation_vertex_groups"][vertex_index] = group_id


    def _current_source_path(self):
        for attr in ("loaded_file", "current_file_path", "last_save_path"):
            value = getattr(self.app, attr, None)
            if value:
                return str(value)
        data = getattr(self.app, "data", None)
        if isinstance(data, dict):
            value = data.get("_source_file_path") or data.get("filename")
            if value:
                return str(value)
        return None


    def _split_polyline_by_distance(self, pts, max_distance):
        if len(pts) < 2 or max_distance <= 1e-9:
            return [pts]

        chunks = []
        current = [pts[0]]
        current_len = 0.0
        prev = pts[0]

        for idx in range(1, len(pts)):
            seg_start = prev
            seg_end = pts[idx]
            seg_vec = seg_end - seg_start
            seg_len = float(np.hypot(seg_vec[0], seg_vec[1]))

            if seg_len <= 1e-12:
                current.append(seg_end)
                prev = seg_end
                continue

            segment_cursor = seg_start
            remaining_seg_len = seg_len

            while current_len + remaining_seg_len > max_distance + 1e-9:
                needed = max_distance - current_len
                if needed <= 1e-9:
                    if len(current) >= 2:
                        chunks.append(np.array(current, dtype=np.float64))
                    current = [segment_cursor]
                    current_len = 0.0
                    continue

                t = needed / remaining_seg_len
                split_pt = segment_cursor + t * (seg_end - segment_cursor)
                current.append(split_pt)

                if len(current) >= 2:
                    chunks.append(np.array(current, dtype=np.float64))

                current = [split_pt]
                segment_cursor = split_pt
                remaining_seg_len = float(np.hypot(
                    seg_end[0] - segment_cursor[0],
                    seg_end[1] - segment_cursor[1],
                ))
                current_len = 0.0

                if remaining_seg_len <= 1e-12:
                    break

            if remaining_seg_len > 1e-12:
                current.append(seg_end)
                current_len += remaining_seg_len

            prev = seg_end

        if len(current) >= 2:
            chunks.append(np.array(current, dtype=np.float64))

        return chunks


    @staticmethod
    def _ring_to_world(geom, origin):
        """Closed world-coord ring of (x, y) tuples for one shapely polygon."""
        try:
            coords = list(geom.exterior.coords)
        except Exception:
            return []
        ox, oy = float(origin[0]), float(origin[1])
        return [(float(x) + ox, float(y) + oy) for x, y in coords]


    def _outline_z_for_xy(self, record, xy_points, z_reference=None):
        """Lift rebuilt outline vertices back onto the block's original 3-D edges."""
        reference = list(z_reference or [])
        for drawing in record.get("drawings") or []:
            for point in drawing.get("coords") or []:
                if point is not None and len(point) > 2:
                    reference.append((float(point[0]), float(point[1]), float(point[2])))
        if not reference:
            return [0.0] * len(xy_points)
        ref_xy = np.array([[point[0], point[1]] for point in reference], dtype=np.float64)
        ref_z = np.array([point[2] for point in reference], dtype=np.float64)
        heights = []
        for x, y in xy_points:
            distances = np.hypot(ref_xy[:, 0] - float(x), ref_xy[:, 1] - float(y))
            heights.append(float(ref_z[int(np.argmin(distances))]))
        return heights


    @staticmethod
    def _linework_chains(geometry):
        """Every linear component of a shapely geometry as a list of 2-D points."""
        if geometry is None or geometry.is_empty:
            return []
        chains = []
        for part in (getattr(geometry, "geoms", None) or [geometry]):
            kind = getattr(part, "geom_type", "")
            if kind == "LineString":
                coords = [(float(point[0]), float(point[1])) for point in part.coords]
                if len(coords) >= 2:
                    chains.append(coords)
            elif kind in ("MultiLineString", "GeometryCollection"):
                chains.extend(NativeBlockGeometry._linework_chains(part))
        return chains


    def _rebuild_block_linework(self, block_records, block_faces, origin, z_extra=None, tolerance=1e-6):
        """Re-derive EVERY drawn block edge from the final face geometry.

        ``drawn(block) = boundary(faces of the block)`` minus everything covered
        by the faces of higher-priority blocks.  That is the same linework the
        per-chain clipping produces, except it is computed from the FINAL faces:
        after a merge no separator can survive inside the block that grew, and no
        boundary is drawn twice by two different blocks.
        """
        from shapely.ops import unary_union
        from shapely.strtree import STRtree

        if not block_records:
            return
        all_faces = []
        owners = []
        for record_index, faces in enumerate(block_faces):
            for geom in faces:
                all_faces.append(geom)
                owners.append(record_index)
        if not all_faces:
            return
        tree = STRtree(all_faces)
        ox, oy = float(origin[0]), float(origin[1])
        z_extra = z_extra or {}

        for index, record in enumerate(block_records):
            faces = block_faces[index]
            if not faces:
                self._discard_block_drawings(record)
                continue
            own = unary_union(faces) if len(faces) > 1 else faces[0]
            if own is None or own.is_empty:
                self._discard_block_drawings(record)
                continue
            lines = own.boundary
            try:
                positions = tree.query(_search_envelope(own, tolerance))
            except Exception:
                positions = range(len(all_faces))
            higher = [all_faces[int(pos)] for pos in positions if owners[int(pos)] < index]
            if higher:
                try:
                    blocker = unary_union(higher) if len(higher) > 1 else higher[0]
                    lines = lines.difference(blocker)
                except Exception as exc:
                    print(f"[SmallBlockMerge] linework clip failed for block {index + 1}: {exc}")

            world_chains = [
                [(float(x) + ox, float(y) + oy) for x, y in chain]
                for chain in self._linework_chains(lines)
            ]
            if not world_chains:
                self._discard_block_drawings(record)
                continue
            self._replace_block_outline(
                record, world_chains, z_reference=z_extra.get(index)
            )


    def _verify_linework(self, block_records, block_faces, tolerance=1e-6):
        """Count drawn edges that cross the INSIDE of another block.

        After a correct rebuild this must be zero: every drawn edge belongs to
        the boundary of the block that drew it or of a higher-priority block.
        """
        from shapely.geometry import Point

        faces = [geom for group in block_faces for geom in group]
        if not faces:
            return 0
        violations = 0
        for index, record in enumerate(block_records):
            own = set(id(geom) for geom in block_faces[index])
            for drawing in record.get("drawings") or []:
                coords = np.asarray(drawing.get("coords"), dtype=np.float64)
                if coords.ndim != 2 or coords.shape[0] < 2:
                    continue
                step = max(1, coords.shape[0] - 1)
                for offset in range(0, coords.shape[0] - 1, step):
                    a = coords[offset, :2]
                    b = coords[min(offset + step, coords.shape[0] - 1), :2]
                    probe = Point(0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
                    for geom in faces:
                        if id(geom) in own:
                            continue
                        if not geom.covers(probe):
                            continue
                        if geom.boundary.distance(probe) > tolerance:
                            violations += 1
                            break
        return violations


    def _block_setting_distance(self, block_records=None):
        """The user's live Block Settings -> Distance, else the recorded value."""
        try:
            value = float(self.distance_spin.value())
        except Exception:
            value = 0.0
        if value > 0.0:
            return value
        for record in block_records or []:
            try:
                recorded = float(record.get("distance") or 0.0)
            except (TypeError, ValueError):
                recorded = 0.0
            if recorded > 0.0:
                return recorded
        return 0.0


    @staticmethod
    def _record_chainage_span(record):
        """``(start, end, length)`` of one block along its own CL, or ``None``.

        ``length`` is the block's chainage span measured ALONG the CL polyline,
        which is what the distance-based small-block rule compares against the
        configured Block Distance.  It is never derived from polygon area, a
        bounding box or a longest edge; for a curved CL the arc length and the
        straight-line endpoint distance genuinely differ.
        """
        def _number(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        start = _number(record.get("start_chainage"))
        end = _number(record.get("end_chainage"))
        length = _number(record.get("longitudinal_distance"))
        if length is None:
            if start is not None and end is not None:
                length = abs(end - start)
            else:
                length = _polyline_length(record.get("cl_pts"))
        if not length or length <= 0.0:
            return None
        if start is None:
            start = 0.0
        if end is None:
            end = start + length
        return float(start), float(end), float(length)


    def _dissolve_small_intersection_fragments(self, block_records, origin):
        """Absorb small faces near CL junctions into one neighbouring block.

        Runs once, on freshly generated records, before they are displayed,
        exported or queried.  Only small faces can start a merge, the absorbed
        area is kept (a real polygon union), the whole shared interface
        disappears and normal blocks are left untouched.  The drawn block edges
        are re-derived from the final faces afterwards, so no stale separator
        can survive inside a block that grew.
        """
        config = dict(_SMALL_BLOCK_MERGE_CONFIG)
        if not config.get("enabled", True) or not block_records:
            return block_records
        try:
            from shapely.geometry import Polygon
        except Exception:
            return block_records

        # The short side-block rule is driven by the user's live Block Settings
        # -> Distance, so the threshold always matches what is on screen.
        block_distance = self._block_setting_distance(block_records)
        config["blockDistance"] = block_distance
        config["sideBlockSmallRatio"] = SIDE_BLOCK_SMALL_RATIO

        ox, oy = float(origin[0]), float(origin[1])

        def _local_polygon(record):
            ring = np.asarray(record.get("polygon"), dtype=np.float64)
            if ring.ndim != 2 or ring.shape[0] < 3:
                return None
            coords = [(float(px) - ox, float(py) - oy) for px, py in ring[:, :2]]
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            try:
                geom = Polygon(coords)
                if not geom.is_valid:
                    geom = geom.buffer(0)
            except Exception:
                return None
            if geom.is_empty or geom.geom_type != "Polygon":
                return None
            return geom

        grid = float(config.get("snapTolerance") or 0.0)
        pieces = [
            list(group)
            for group in _block_piece_geometries(
                [_local_polygon(r) for r in block_records], grid=grid
            )
        ]
        plan = _plan_small_fragment_merges(
            pieces,
            config,
            [record.get("cl_priority", index) for index, record in enumerate(block_records)],
            [self._record_chainage_span(record) for record in block_records],
        )

        def _log_summary(merged_count, rejected):
            if not plan["fragments"]:
                return
            distance_candidates = len(plan.get("distanceFragments") or [])
            print(
                f"[SmallBlockMerge] summary fragments={len(plan['fragments'])} "
                f"merged={merged_count} skipped={len(plan['skipped'])} "
                f"rejected={rejected} areaRatio={config.get('smallBlockRelativeAreaRatio')} "
                f"areaThreshold={config.get('smallBlockAreaThreshold')} "
                f"blockDistance={block_distance:.4f} "
                f"sideBlockThreshold="
                f"{block_distance * SIDE_BLOCK_SMALL_RATIO:.4f} "
                f"distanceRuleCandidates={distance_candidates} "
                f"snapTolerance={config.get('snapTolerance')}"
            )

        if not plan["merges"]:
            for key, reason in plan["skipped"]:
                print(
                    f"[SmallBlockMerge] fragment=block{key[0] + 1}/piece{key[1]} "
                    f"result=SKIPPED_{str(reason).upper()}"
                )
            _log_summary(0, 0)
            return block_records

        absorbed = set()
        merged_count = 0
        rejected = 0
        changed_blocks = set()
        z_extra = {}
        # parent block index -> [(donor block index, CL chain)] collected while
        # planning, applied after we know which donors ended up fully removed.
        carry_requests = {}
        for parent_key, fragment_keys in plan["merges"].items():
            parent_geom = pieces[parent_key[0]][parent_key[1]]
            fragment_geoms = [pieces[block][piece] for block, piece in fragment_keys]
            merged, reason, detail = _union_faces(parent_geom, fragment_geoms, config)
            if merged is None:
                rejected += 1
                print(
                    f"[SmallBlockMerge] parent=block{parent_key[0] + 1} "
                    f"fragments={len(fragment_keys)} result=REJECTED_{reason} detail={detail}"
                )
                for key in fragment_keys:
                    plan["skipped"].append((key, str(reason).lower()))
                continue

            area_before = float(parent_geom.area)
            # Keep the absorbed blocks' original 3-D edges as a z reference for
            # the parts of the merged outline they contributed.
            for block_index, _piece_index in fragment_keys:
                if block_index == parent_key[0]:
                    continue
                bucket = z_extra.setdefault(parent_key[0], [])
                for drawing in block_records[block_index].get("drawings") or []:
                    for point in drawing.get("coords") or []:
                        if point is not None and len(point) > 2:
                            bucket.append(
                                (float(point[0]), float(point[1]), float(point[2]))
                            )

            pieces[parent_key[0]][parent_key[1]] = merged
            changed_blocks.add(parent_key[0])
            for key in fragment_keys:
                absorbed.add(key)
                changed_blocks.add(key[0])
                merged_count += 1
                # Remember the absorbed block's own CL chain: if that block ends
                # up fully removed it must not take its CL out of the export.
                # Applied after the survivor pass, because a block that is only
                # PARTLY absorbed stays a block of its own and keeps its CL, so
                # copying it to the parent as well would duplicate the line.
                if key[0] != parent_key[0]:
                    chain = list(block_records[key[0]].get("cl_pts") or [])
                    if len(chain) >= 2:
                        carry_requests.setdefault(parent_key[0], []).append(
                            (key[0], chain)
                        )
                info = plan["details"].get(key) or {}
                print(
                    f"[SmallBlockMerge] fragment=block{key[0] + 1}/piece{key[1]} "
                    f"area={float(info.get('area', 0.0)):.4f} isSmall=true "
                    f"neighbourCount={int(info.get('neighbourCount', 0))} "
                    f"selectedParent=block{parent_key[0] + 1} "
                    f"exactSharedLength={float(info.get('exactSharedLength', 0.0)):.4f} "
                    f"totalSeparatorLength={float(info.get('sharedLength', 0.0)):.4f} "
                    f"attachmentRatio={float(info.get('attachmentRatio', 0.0)):.3f} "
                    f"mergeAreaBefore={area_before:.4f} "
                    f"mergeAreaAfter={float(merged.area):.4f} "
                    f"rules={'+'.join(info.get('reasons') or ()) or 'area'} "
                    f"result=SUCCESS"
                )

        if not changed_blocks:
            _log_summary(0, rejected)
            return block_records

        survivors = []
        survivor_faces = []
        for block_index, record in enumerate(block_records):
            remaining = [
                geom
                for piece_index, geom in enumerate(pieces[block_index])
                if (block_index, piece_index) not in absorbed
            ]
            pieces[block_index] = remaining
            if not remaining:
                self._discard_block_drawings(record)
                print(f"[SmallBlockMerge] block={block_index + 1} result=ABSORBED_REMOVED")
                continue
            if block_index in changed_blocks:
                world_rings = [self._ring_to_world(geom, origin) for geom in remaining]
                record["effective_polygon"] = [ring for ring in world_rings if ring]
            survivors.append(record)
            survivor_faces.append(remaining)

        # A block that was fully absorbed would take its CL out of the export
        # with it, so hand that CL to the surviving block that took its points.
        # Donors that still exist are skipped: they export their own CL already.
        survivor_indices = {
            index for index in range(len(block_records)) if pieces[index]
        }
        for parent_index, requests in carry_requests.items():
            for donor_index, chain in requests:
                if donor_index in survivor_indices:
                    continue
                target = parent_index
                seen = set()
                while target not in survivor_indices and target not in seen:
                    seen.add(target)
                    following = next(
                        (
                            other_parent
                            for other_parent, other_requests in carry_requests.items()
                            if any(donor == target for donor, _chain in other_requests)
                        ),
                        None,
                    )
                    if following is None:
                        break
                    target = following
                if target not in survivor_indices:
                    continue
                carried = block_records[target].setdefault("absorbed_cl_chains", [])
                if chain not in carried:
                    carried.append(chain)

        # Single source of truth: the drawn edges are re-derived from the final
        # faces, so the dissolved separators cannot survive anywhere.
        if config.get("rebuildLinework", True) and survivors:
            self._rebuild_block_linework(
                survivors, survivor_faces, origin, z_extra, grid if grid > 0.0 else 1e-6
            )
            if config.get("debug"):
                stale = self._verify_linework(
                    survivors, survivor_faces, float(config.get("geometryTolerance") or 1e-6)
                )
                print(f"[SmallBlockMerge] verify staleInteriorEdges={stale}")

        for key, reason in plan["skipped"]:
            if key not in absorbed:
                print(
                    f"[SmallBlockMerge] fragment=block{key[0] + 1}/piece{key[1]} "
                    f"result=SKIPPED_{str(reason).upper()}"
                )
        _log_summary(merged_count, rejected)
        return survivors


    def _generate_block_lines(self):
        if not self.selected_cl_lines:
            print("No CL lines selected")
            return []

        self._remove_generated_block_drawings()

        mode = self.mode_combo.currentData()
        distance = float(self.distance_spin.value())

        if mode == "symmetric":
            total_width = float(self.total_width_spin.value())
            tol_above = total_width * 0.5
            tol_below = total_width * 0.5
        else:
            tol_above = float(self.tol_above_spin.value())
            tol_below = float(self.tol_below_spin.value())

        block_color = (0.0, 0.7, 1.0)
        count = 0
        block_records = []

        print(f"CL count={len(self.selected_cl_lines)}")

        # Compute a shared world-coordinate origin so that all block geometry math
        # (normal offsets, polygon clipping, point-in-polygon ray casting) operates on
        # small numbers.  Large survey coordinates (e.g. 600 000 m) cause catastrophic
        # cancellation in the floating-point arithmetic used by the clipping helpers,
        # producing wrong inside/outside results and distorted block shapes for SNT CL
        # lines.  Subtracting this origin here and adding it back to native drawing
        # coordinates eliminates cancellation without changing authored coordinates.
        first_coords = next(
            (d.get("coords", []) for d in self.selected_cl_lines if d.get("coords")),
            [],
        )
        if first_coords:
            _ox = float(first_coords[0][0])
            _oy = float(first_coords[0][1])
            _oz = float(first_coords[0][2]) if len(first_coords[0]) > 2 else 0.0
        else:
            _ox = _oy = _oz = 0.0

        def _to_local(coords_list):
            """Shift a list of (x,y,z) tuples to local coords."""
            return [
                (float(p[0]) - _ox, float(p[1]) - _oy,
                 float(p[2]) - _oz if len(p) > 2 else -_oz)
                for p in coords_list
            ]

        def _to_world(coords_list):
            """Shift a list of (x,y,z) tuples back to world coords."""
            return [
                (float(p[0]) + _ox, float(p[1]) + _oy, float(p[2]) + _oz)
                for p in coords_list
            ]

        # ---- Pass 1: compute block geometry for every CL line in selection-priority order ----
        all_block_data = []   # list-of-lists: one inner list per CL line

        for cl_priority, cl_drawing in enumerate(self.selected_cl_lines):
            coords = cl_drawing.get("coords", [])
            print(f"CL#{cl_priority} coords len={len(coords)}")
            if len(coords) < 2:
                continue
            pts = np.array(_to_local(coords), dtype=np.float64)
            chunks = self._split_polyline_by_distance(pts, distance) if distance > 0 else [pts]

            cl_blocks = []
            chainage = 0.0
            for idx, chunk_pts in enumerate(chunks):
                block_data = self._compute_block_geometry(
                    chunk_pts, tol_above, tol_below,
                    is_first_block=(idx == 0),
                    is_last_block=(idx == len(chunks) - 1),
                )
                if block_data is not None:
                    block_data["cl_pts"] = [tuple(p) for p in chunk_pts]
                    # Chainage provenance: how far this block runs ALONG its own
                    # CL.  The distance-based small-block rule measures the block
                    # here rather than from its polygon shape, so a curved CL
                    # keeps its true along-the-line length.
                    chunk_length = _polyline_length(chunk_pts)
                    block_data["start_chainage"] = chainage
                    block_data["end_chainage"] = chainage + chunk_length
                    block_data["longitudinal_distance"] = chunk_length
                    chainage += chunk_length
                    cl_blocks.append(block_data)
            all_block_data.append(cl_blocks)

        # ---- Pass 2: draw, clipping lower-priority blocks against higher-priority polygons ----
        # All geometry in this pass is in local (origin-subtracted) coordinates so that
        # the polygon clipping math operates on small numbers (no catastrophic cancellation).
        # Chains are converted back to world coords before being stored in
        # native drawings and block records.
        priority_polygons = []  # closed 2-D polygon arrays, highest priority first (local coords)

        for cl_priority, cl_blocks in enumerate(all_block_data):
            for block_data in cl_blocks:
                above_pts       = block_data["above_pts"]
                below_pts       = block_data["below_pts"]
                start_cap_above = block_data["start_cap_above"]
                start_cap_below = block_data["start_cap_below"]
                end_cap_above   = block_data["end_cap_above"]
                end_cap_below   = block_data["end_cap_below"]

                line_sets = [
                    (above_pts,                          "above"),
                    (below_pts,                          "below"),
                    ([start_cap_above, start_cap_below], "start_cap"),
                    ([end_cap_above,   end_cap_below],   "end_cap"),
                ]

                drawing_group = []
                for seg_pts, seg_name in line_sets:
                    if len(seg_pts) < 2:
                        continue

                    # Clip this polyline against every higher-priority block polygon
                    # (clipping uses local coords — correct, no large-number precision issue)
                    surviving_chains = [seg_pts]
                    for higher_poly in priority_polygons:
                        next_chains = []
                        for chain in surviving_chains:
                            next_chains.extend(
                                self._clip_polyline_outside_polygon(chain, higher_poly)
                            )
                        surviving_chains = next_chains

                    for chain in surviving_chains:
                        if len(chain) < 2:
                            continue
                        world_chain = _to_world(chain)
                        drawing = self._native_outline(world_chain)
                        self._generated_block_drawings.append(drawing)
                        drawing_group.append(drawing)
                        count += 1

                # Register this block polygon (local coords) for clipping subsequent blocks.
                # Build a world-coord polygon separately for export and point-assignment.
                poly_local = self._build_block_polygon(above_pts, below_pts)
                if poly_local is not None:
                    self._tag_block_vertex_groups(drawing_group)
                    priority_polygons.append(poly_local)
                    # Shift the polygon back to world coords for downstream use
                    poly_world = poly_local + np.array([_ox, _oy], dtype=np.float64)
                    block_records.append({
                        "priority": len(block_records) + 1,
                        "cl_priority": cl_priority + 1,
                        "distance": distance,
                        "source_path": self._current_source_path(),
                        "polygon": poly_world,
                        "cl_pts": _to_world(list(block_data.get("cl_pts", []))),
                        # Block chainage along its own CL (per chunk, so the last
                        # chunk of a CL carries the remainder distance).
                        "start_chainage": float(block_data.get("start_chainage", 0.0)),
                        "end_chainage": float(block_data.get("end_chainage", 0.0)),
                        "longitudinal_distance": float(
                            block_data.get("longitudinal_distance", 0.0)
                        ),
                        "above_pts": _to_world(list(above_pts)),
                        "below_pts": _to_world(list(below_pts)),
                        "drawings": drawing_group,
                    })

        # ---- Post-processing: dissolve small slivers at CL intersections ----
        # Where CL lines meet, a corridor can be clipped into a large remainder
        # plus a small leftover sliver (a stub past the other corridor, or a wedge
        # trapped between two blocks).  Such a sliver is absorbed into the block it
        # shares a boundary with, so the shared internal separation line goes away
        # and the area stays inside ONE block.
        block_records = self._dissolve_small_intersection_fragments(
            block_records, (_ox, _oy)
        )
        count = sum(len(record.get("drawings") or []) for record in block_records)
        print(f"Created {count} block lines")
        self._generated_blocks_metadata = block_records
        setattr(self.app, "block_creation_generated_blocks", block_records)
        setattr(self.app, "block_creation_generated_source", self._current_source_path())
        return block_records


    def _compute_block_geometry(self, pts, tol_above, tol_below, is_first_block, is_last_block):
        """Compute offset boundary lists and cap endpoints for one chunk."""
        above_pts = []
        below_pts = []

        for i in range(len(pts)):
            if i == 0:
                direction = pts[1] - pts[0]
                length = np.hypot(direction[0], direction[1])
                if length < 1e-12:
                    above_pts.append(tuple(pts[i]))
                    below_pts.append(tuple(pts[i]))
                    continue
                d = direction[:2] / length
                n = np.array([-d[1], d[0]])
                miter_factor = 1.0
            elif i == len(pts) - 1:
                direction = pts[i] - pts[i - 1]
                length = np.hypot(direction[0], direction[1])
                if length < 1e-12:
                    above_pts.append(tuple(pts[i]))
                    below_pts.append(tuple(pts[i]))
                    continue
                d = direction[:2] / length
                n = np.array([-d[1], d[0]])
                miter_factor = 1.0
            else:
                p_prev = pts[i - 1][:2]
                p_curr = pts[i][:2]
                p_next = pts[i + 1][:2]

                v1 = p_curr - p_prev
                v1_len = np.hypot(v1[0], v1[1])
                v2 = p_next - p_curr
                v2_len = np.hypot(v2[0], v2[1])

                if v1_len > 1e-12 and v2_len > 1e-12:
                    v1_u = v1 / v1_len
                    v2_u = v2 / v2_len

                    n1 = np.array([-v1_u[1], v1_u[0]])
                    n2 = np.array([-v2_u[1], v2_u[0]])

                    n_sum = n1 + n2
                    n_sum_len = np.hypot(n_sum[0], n_sum[1])

                    if n_sum_len > 1e-6:
                        n = n_sum / n_sum_len
                        cos_half = float(np.dot(n, n1))
                        if cos_half > 1e-6:
                            miter_factor = 1.0 / cos_half
                            # Cap the miter factor to achieve a balance between line and vertex
                            miter_factor = min(miter_factor, 1.5)
                        else:
                            miter_factor = 1.0
                    else:
                        n = n1
                        miter_factor = 1.0
                else:
                    direction = (pts[i + 1] - pts[i - 1]) * 0.5
                    length = np.hypot(direction[0], direction[1])
                    if length < 1e-12:
                        above_pts.append(tuple(pts[i]))
                        below_pts.append(tuple(pts[i]))
                        continue
                    d = direction[:2] / length
                    n = np.array([-d[1], d[0]])
                    miter_factor = 1.0

            z = float(pts[i][2]) if len(pts[i]) > 2 else 0.0
            above_pts.append((
                float(pts[i][0] + n[0] * tol_above * miter_factor),
                float(pts[i][1] + n[1] * tol_above * miter_factor),
                z,
            ))
            below_pts.append((
                float(pts[i][0] - n[0] * tol_below * miter_factor),
                float(pts[i][1] - n[1] * tol_below * miter_factor),
                z,
            ))

        if len(above_pts) < 2 or len(below_pts) < 2:
            return None

        d_start = (pts[1] - pts[0])[:2]
        d_start_len = np.hypot(d_start[0], d_start[1])
        d_start = d_start / d_start_len if d_start_len > 1e-12 else np.array([1.0, 0.0])

        d_end = (pts[-1] - pts[-2])[:2]
        d_end_len = np.hypot(d_end[0], d_end[1])
        d_end = d_end / d_end_len if d_end_len > 1e-12 else np.array([1.0, 0.0])

        n_start = np.array([-d_start[1], d_start[0]])
        n_end   = np.array([-d_end[1],   d_end[0]])

        z0    = float(pts[0][2])  if len(pts[0])  > 2 else 0.0
        z_end = float(pts[-1][2]) if len(pts[-1]) > 2 else 0.0

        if is_first_block:
            start_cap_above = (
                float(pts[0][0] - d_start[0] * tol_above + n_start[0] * tol_above),
                float(pts[0][1] - d_start[1] * tol_above + n_start[1] * tol_above),
                z0,
            )
            start_cap_below = (
                float(pts[0][0] - d_start[0] * tol_below - n_start[0] * tol_below),
                float(pts[0][1] - d_start[1] * tol_below - n_start[1] * tol_below),
                z0,
            )
            above_pts.insert(0, start_cap_above)
            below_pts.insert(0, start_cap_below)
        else:
            start_cap_above = above_pts[0]
            start_cap_below = below_pts[0]

        if is_last_block:
            end_cap_above = (
                float(pts[-1][0] + d_end[0] * tol_above + n_end[0] * tol_above),
                float(pts[-1][1] + d_end[1] * tol_above + n_end[1] * tol_above),
                z_end,
            )
            end_cap_below = (
                float(pts[-1][0] + d_end[0] * tol_below - n_end[0] * tol_below),
                float(pts[-1][1] + d_end[1] * tol_below - n_end[1] * tol_below),
                z_end,
            )
            above_pts.append(end_cap_above)
            below_pts.append(end_cap_below)
        else:
            end_cap_above = above_pts[-1]
            end_cap_below = below_pts[-1]

        return {
            "above_pts":       above_pts,
            "below_pts":       below_pts,
            "start_cap_above": start_cap_above,
            "start_cap_below": start_cap_below,
            "end_cap_above":   end_cap_above,
            "end_cap_below":   end_cap_below,
        }


    def _build_block_polygon(self, above_pts, below_pts):
        """Return a closed 2-D polygon (Nx2 numpy array) from block boundary lists."""
        try:
            ring = []
            for p in above_pts:
                ring.append((float(p[0]), float(p[1])))
            for p in reversed(below_pts):
                ring.append((float(p[0]), float(p[1])))
            if len(ring) < 3:
                return None
            return np.array(ring, dtype=np.float64)
        except Exception:
            return None


    @staticmethod
    def _point_in_polygon_2d(px, py, polygon):
        """Ray-casting point-in-polygon test (2-D). polygon is Nx2 numpy array."""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / (yj - yi + 1e-300) + xi
            ):
                inside = not inside
            j = i
        return inside


    @staticmethod
    def _segment_polygon_intersections(p1, p2, polygon):
        """
        Find all t in (0,1) where segment p1->p2 crosses a polygon edge.
        Returns sorted list of (t, (ix, iy)).
        """
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        dx, dy = x2 - x1, y2 - y1

        hits = []
        n = len(polygon)
        for i in range(n):
            ax, ay = polygon[i]
            bx, by = polygon[(i + 1) % n]
            ex, ey = bx - ax, by - ay

            denom = dx * ey - dy * ex
            if abs(denom) < 1e-12:
                continue

            t = ((ax - x1) * ey - (ay - y1) * ex) / denom
            u = ((ax - x1) * dy - (ay - y1) * dx) / denom

            if 1e-9 < t < 1.0 - 1e-9 and 0.0 <= u <= 1.0:
                hits.append((t, (x1 + t * dx, y1 + t * dy)))

        hits.sort(key=lambda h: h[0])
        return hits


    def _clip_polyline_outside_polygon(self, polyline_pts, polygon):
        """
        Clip polyline_pts retaining only the portions that lie OUTSIDE the polygon.
        Returns a list of sub-chains (each is a list of (x,y,z) tuples).
        """
        if polygon is None or len(polygon) < 3 or len(polyline_pts) < 2:
            return [polyline_pts]

        def z_of(p):
            return float(p[2]) if len(p) > 2 else 0.0

        def get_pt(p1, p2, t):
            x = float(p1[0]) + t * (float(p2[0]) - float(p1[0]))
            y = float(p1[1]) + t * (float(p2[1]) - float(p1[1]))
            z = z_of(p1) + t * (z_of(p2) - z_of(p1))
            return (x, y, z)

        # Collect all kept sub-segments
        kept_segments = []

        for seg_idx in range(len(polyline_pts) - 1):
            p1 = polyline_pts[seg_idx]
            p2 = polyline_pts[seg_idx + 1]

            # Find intersections in (0, 1)
            hits = self._segment_polygon_intersections(p1, p2, polygon)
            t_vals = [0.0]
            for t, _ in hits:
                if t > 1e-9 and t < 1.0 - 1e-9:
                    t_vals.append(t)
            t_vals.append(1.0)
            t_vals.sort()

            for idx in range(len(t_vals) - 1):
                ta = t_vals[idx]
                tb = t_vals[idx + 1]
                if tb - ta < 1e-9:
                    continue

                # Check the midpoint of this sub-segment
                t_mid = (ta + tb) * 0.5
                mid_pt = get_pt(p1, p2, t_mid)

                if not self._point_in_polygon_2d(mid_pt[0], mid_pt[1], polygon):
                    # Outside! Keep this sub-segment
                    sub_start = get_pt(p1, p2, ta)
                    sub_end = get_pt(p1, p2, tb)
                    kept_segments.append((sub_start, sub_end))

        # Assemble the kept segments into continuous chains
        chains = []
        for start_pt, end_pt in kept_segments:
            if chains and np.hypot(chains[-1][-1][0] - start_pt[0], chains[-1][-1][1] - start_pt[1]) < 1e-6:
                chains[-1].append(end_pt)
            else:
                chains.append([start_pt, end_pt])

        return [c for c in chains if len(c) >= 2]


