"""Read-only exact LNT v2 leaf snapshot and directory verification.

This is an independent, conservative audit of the packed formats in
native/lnt/include/LNTFormat.h and LNTMath.h. The native DLL remains the only
writer. This module never edits an LNT file.
"""
from __future__ import annotations

import os
from pathlib import Path
import struct
import numpy as np

HEADER = struct.Struct('<4sHHQ6d')
DIR = struct.Struct('<QQ')
ENTRY = struct.Struct('<QQII3ffBB10s')
POINT = np.dtype([
    ('x', '<u2'), ('y', '<u2'), ('z', '<u2'),
    ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
    ('intensity', 'u1'), ('classification', 'u1'),
], align=False)
assert HEADER.size == 64 and DIR.size == 16 and ENTRY.size == 52 and POINT.itemsize == 11


class LNTDirectoryError(ValueError):
    pass


def _exactly(handle, size, label):
    data = handle.read(size)
    if len(data) != size:
        raise LNTDirectoryError(f'Incomplete {label}: expected {size} bytes, read {len(data)}.')
    return data


def inspect_directory(path, max_points=8_000_000):
    """Validate the directory and return exact leaf metadata (no payload read)."""
    path = Path(path)
    before = path.stat()
    with path.open('rb') as handle:
        magic, version, reserved, index_at, *bounds = HEADER.unpack(_exactly(handle, 64, 'header'))
        if magic != b'LNT\x00' or version != 2 or reserved:
            raise LNTDirectoryError('Unsupported or malformed LNT v2 header.')
        if not np.all(np.isfinite(bounds)) or any(bounds[i] > bounds[i+3] for i in range(3)):
            raise LNTDirectoryError('Invalid LNT header coordinates.')
        if not (64 <= index_at <= before.st_size - DIR.size):
            raise LNTDirectoryError('LNT directory offset exceeds the file.')
        handle.seek(index_at)
        total, num_nodes = DIR.unpack(_exactly(handle, DIR.size, 'directory header'))
        if total > max_points:
            raise LNTDirectoryError(f'{total:,} LNT points exceed exact-read limit {max_points:,}.')
        if num_nodes > (before.st_size - index_at - DIR.size) // ENTRY.size:
            raise LNTDirectoryError('LNT node directory is truncated.')
        if num_nodes > 2_000_000:
            raise LNTDirectoryError('Unreasonably large LNT directory; refusing inspection.')
        raw_entries = _exactly(handle, num_nodes * ENTRY.size, 'node directory')
    leaves = []
    seen_ids = set()
    occupied = []
    leaf_count = 0
    lod_count = 0
    for index in range(num_nodes):
        (node_id, offset, byte_length, point_count, cx, cy, cz,
         half_size, children, level, reserved_bytes) = ENTRY.unpack_from(raw_entries, index*ENTRY.size)
        if node_id in seen_ids:
            raise LNTDirectoryError(f'Duplicate LNT node ID at index {index}.')
        seen_ids.add(node_id)
        if byte_length != point_count * POINT.itemsize:
            raise LNTDirectoryError(f'Bad point-chunk size at node {index}.')
        if (not 64 <= offset <= index_at or byte_length > index_at - offset):
            raise LNTDirectoryError(f'Node {index} chunk escapes the LNT payload.')
        if not np.all(np.isfinite([cx, cy, cz, half_size])) or half_size < 0:
            raise LNTDirectoryError(f'Invalid node coordinate envelope at index {index}.')
        if level > 21 or any(reserved_bytes):
            raise LNTDirectoryError(f'Invalid level or reserved node bytes at index {index}.')
        if point_count:
            occupied.append((offset, offset+byte_length, index))
        if children == 0:
            leaf_count += point_count
            if point_count:
                leaves.append((index, offset, point_count, (cx, cy, cz), half_size))
        else:
            lod_count += point_count
    occupied.sort()
    for prior, current in zip(occupied, occupied[1:]):
        if prior[1] > current[0]:
            raise LNTDirectoryError(f'Payload chunks overlap: nodes {prior[2]}, {current[2]}.')
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise LNTDirectoryError('LNT changed while its directory was inspected.')
    return dict(path=str(path), total=int(total), leaf_sum=int(leaf_count),
                lod_sum=int(lod_count), nodes=int(num_nodes), leaves=leaves,
                origin=np.asarray(bounds[:3], dtype=np.float64),
                header_bounds=np.asarray(bounds, dtype=np.float64),
                size=before.st_size, mtime_ns=before.st_mtime_ns)


def read_exact_leaf_snapshot(path, expected_total=None, max_points=8_000_000):
    """Reconstruct all identities from each packed leaf, fail-closed on corruption.

    Leaf identity is (directory node index, point index within that leaf),
    exactly like native/lnt/include/LNTSectionQuery.h's SectionHit.
    """
    info = inspect_directory(path, max_points=max_points)
    n = info['total']
    if expected_total is not None and n != int(expected_total):
        raise LNTDirectoryError(f'LNT trailer count {n} disagrees with expected {expected_total}.')
    if info['leaf_sum'] != n:
        raise LNTDirectoryError(f'LNT leaf total {info["leaf_sum"]} does not match trailer {n}.')
    xyz = np.empty((n, 3), dtype=np.float64)
    classes = np.empty(n, dtype=np.uint8)
    nodes = np.empty(n, dtype=np.uint32)
    local = np.empty(n, dtype=np.uint32)
    origin = info['origin']
    at = 0
    with open(path, 'rb') as handle:
        for node_index, offset, count, center, half in info['leaves']:
            handle.seek(offset)
            raw = np.frombuffer(_exactly(handle, count * POINT.itemsize, f'leaf {node_index}'),
                                dtype=POINT)
            end = at + count
            for axis, key in enumerate(('x', 'y', 'z')):
                lo = float(origin[axis]) + float(center[axis]) - float(half)
                xyz[at:end, axis] = lo + (2.0 * float(half) * raw[key].astype(np.float64)) / 65535.0
            classes[at:end] = raw['classification']
            nodes[at:end] = node_index
            local[at:end] = np.arange(count, dtype=np.uint32)
            at = end
    if at != n or not np.isfinite(xyz).all():
        raise LNTDirectoryError('Incomplete leaf reconstruction or nonfinite point coordinates.')
    after = os.stat(path)
    if (info['size'], info['mtime_ns']) != (after.st_size, after.st_mtime_ns):
        raise LNTDirectoryError('LNT changed during leaf snapshot; refusing classification.')
    return {'xyz': xyz, 'classification': classes, 'node': nodes, 'local': local}, info


def verify_native_subset(native_tile, file_tile):
    """Every native-returned record must match the independent packed LNT read."""
    want = ('xyz', 'classification', 'node', 'local')
    if not all(k in native_tile and k in file_tile for k in want):
        raise LNTDirectoryError('Missing required identity/coordinate arrays.')
    known = (np.asarray(file_tile['node'], dtype=np.uint64) << 32) | np.asarray(file_tile['local'], dtype=np.uint64)
    returned = (np.asarray(native_tile['node'], dtype=np.uint64) << 32) | np.asarray(native_tile['local'], dtype=np.uint64)
    if not np.all(known[1:] > known[:-1]):
        raise LNTDirectoryError('File leaf identities are not strictly sorted.')
    if len(np.unique(returned)) != len(returned):
        raise LNTDirectoryError('Native query duplicated point identities.')
    positions = np.searchsorted(known, returned)
    if np.any(positions >= len(known)) or not np.array_equal(known[positions], returned):
        raise LNTDirectoryError('Native query contains identities absent from the LNT leaf directory.')
    if not np.array_equal(np.asarray(native_tile['classification']), file_tile['classification'][positions]):
        raise LNTDirectoryError('Native query classes disagree with leaf data; there may be an active edit.')
    if not np.allclose(np.asarray(native_tile['xyz']), file_tile['xyz'][positions], rtol=0, atol=1e-7):
        raise LNTDirectoryError('Native query coordinates disagree with reconstructed leaf data.')
    return len(known) - len(returned)


def read_exact_polygon_snapshot(path, polygons, expected_total=None, max_points=8_000_000,
                                progress=None, cancelled=None, chunk_size=250_000,
                                operation_name='Height'):
    """Read only EXACT native leaf identities inside selected XY fences.

    This avoids the full-cloud exact-query cap for large LNT files, without
    guessing a spatial tile halo or silently dropping an unqueried leaf. All
    leaf metadata is validated first, and every intersecting leaf is read
    in bounded chunks. No class write is performed by this function.

    The caller still has to refuse operations needing reference points
    outside the supplied fence (a global TIN cannot be inferred here).
    """
    from matplotlib.path import Path as PolygonPath

    if not polygons:
        raise LNTDirectoryError(
            f'Select at least one fence for large-cloud {operation_name}. '
            'An exact unfenced classification requires a dedicated streaming path.')
    prepared = []
    for vertices in polygons:
        poly = np.asarray(vertices, dtype=np.float64)
        if (poly.ndim != 2 or len(poly) < 3 or poly.shape[1] < 2
                or not np.isfinite(poly).all()):
            raise LNTDirectoryError(f'Invalid {operation_name} fence geometry; no points were read.')
        poly = poly[:, :2]
        prepared.append((poly.min(axis=0), poly.max(axis=0), PolygonPath(poly)))

    # The global 8M cap applies to the returned, editable fence scope, NOT
    # to the header count. Directory validation and leaf-count reconciliation
    # remain whole-cloud operations, with no whole-cloud point arrays.
    info = inspect_directory(path, max_points=2**63 - 1)
    total = info['total']
    if expected_total is not None and int(expected_total) != total:
        raise LNTDirectoryError(
            f'LNT directory total {total:,} differs from active view '
            f'{int(expected_total):,}; no changes were made.')
    if info['leaf_sum'] != total:
        raise LNTDirectoryError(
            f'LNT leaf count {info["leaf_sum"]:,} != header total {total:,}.')
    max_points = int(max_points)
    if max_points < 1:
        raise LNTDirectoryError('Invalid positive limit for the selected fence.')
    chunk_size = max(1, min(int(chunk_size), 500_000))

    xyz_parts, class_parts, node_parts, local_parts = [], [], [], []
    found = 0
    visited = 0
    origin = info['origin']
    leaf_count = len(info['leaves'])
    with open(path, 'rb') as handle:
        for leaf_no, (node_index, offset, count, center, half) in enumerate(info['leaves']):
            if cancelled is not None and cancelled():
                raise LNTDirectoryError('Fence read cancelled; no classification changed.')
            # Update progress even when most leaves are outside the fence so
            # the GUI remains responsive while scanning huge LNT directories.
            if progress is not None and leaf_no % 64 == 0:
                progress(5 + int(16 * leaf_no / max(1, leaf_count)),
                         f'{operation_name}: scanned {leaf_no:,}/{leaf_count:,} leaves; '
                         f'{found:,} exact fence points')
            center = origin + np.asarray(center, dtype=np.float64)
            leaf_lo = center[:2] - float(half)
            leaf_hi = center[:2] + float(half)
            # Every packed point lies inside this validated node envelope.
            if not any(np.all(leaf_hi >= lo) and np.all(leaf_lo <= hi)
                       for lo, hi, _ in prepared):
                continue
            visited += 1
            for start in range(0, count, chunk_size):
                if cancelled is not None and cancelled():
                    raise LNTDirectoryError('Fence read cancelled; no classification changed.')
                size = min(chunk_size, count - start)
                handle.seek(offset + start * POINT.itemsize)
                pts = np.frombuffer(_exactly(handle, size * POINT.itemsize,
                                            f'leaf {node_index} points {start}+{size}'),
                                    dtype=POINT)
                xy = np.empty((size, 2), dtype=np.float64)
                for axis, key in enumerate(('x', 'y')):
                    xy[:, axis] = (center[axis] - float(half) +
                                   (2.0 * float(half) / 65535.0) *
                                   pts[key].astype(np.float64))
                inside = np.zeros(size, dtype=bool)
                for lo, hi, shape in prepared:
                    pre = ((xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) &
                           (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1]))
                    if np.any(pre):
                        inside[pre] |= shape.contains_points(xy[pre])
                rows = np.flatnonzero(inside)
                if not len(rows):
                    continue
                found += len(rows)
                if found > max_points:
                    raise LNTDirectoryError(
                        f'Selected fence(s) contain over {max_points:,} points. '
                        'Split/narrow the fences and retry; no changes were made.')
                pos = np.empty((len(rows), 3), dtype=np.float64)
                pos[:, :2] = xy[rows]
                pos[:, 2] = (center[2] - float(half) +
                             (2.0 * float(half) / 65535.0) *
                             pts['z'][rows].astype(np.float64))
                xyz_parts.append(pos)
                class_parts.append(pts['classification'][rows].copy())
                node_parts.append(np.full(len(rows), node_index, dtype=np.uint32))
                local_parts.append((rows + start).astype(np.uint32))

    after = os.stat(path)
    if (info['size'], info['mtime_ns']) != (after.st_size, after.st_mtime_ns):
        raise LNTDirectoryError(
            'LNT file was modified during fence read; no classification changed.')
    def combine(parts, empty_shape, dtype):
        return np.concatenate(parts) if parts else np.empty(empty_shape, dtype=dtype)
    result = dict(
        xyz=combine(xyz_parts, (0, 3), np.float64),
        classification=combine(class_parts, (0,), np.uint8),
        node=combine(node_parts, (0,), np.uint32),
        local=combine(local_parts, (0,), np.uint32),
        file_stat=(info['size'], info['mtime_ns']),
    )
    # Ensure no duplicate (node_index, local_index) identities. A valid
    # directory cannot duplicate nodes, but fail closed if a future decoder
    # violates the uniqueness contract.
    if len(result['node']) != found or len(result['local']) != found:
        raise LNTDirectoryError('Incomplete exact fence snapshot; no changes made.')
    if not np.all(np.isfinite(result['xyz'])):
        raise LNTDirectoryError('Nonfinite exact fence coordinates; no changes made.')
    if cancelled is not None and cancelled():
        raise LNTDirectoryError('Fence read cancelled; no classification changed.')
    if progress is not None:
        progress(25, f'Verified {found:,} exact points in {visited:,} LNT leaves')
    return result, info



def read_exact_corridor_snapshot(
        path, alignment_coords, corridor_width, max_ground_distance,
        expected_total=None, max_points=8_000_000, progress=None,
        cancelled=None, chunk_size=250_000):
    """Exact packed-LNT leaf read within a Find Wires corridor + ground halo.

    Every source candidate must be within corridor_width/2 of a CL segment.
    Every usable Ground reference must be within max_ground_distance of a
    source candidate. By the triangle inequality the union of line buffers
    radius (width/2 + ground_distance) contains every such point. The
    classifier still applies its ORIGINAL exact XY corridor and HAG tests.

    A prepared GEOS shape provides reliable pruning for diagonal/curved lines:
    unlike an axis-aligned full-CL bbox it doesn't accidentally capture the
    whole 47M-point cloud. The validated LNT leaf directory yields stable
    node/local point IDs; the native DLL remains the ONLY writer.

    If the context exceeds max_points, fail closed before modifying any data.
    """
    try:
        from shapely.geometry import LineString
        from shapely.ops import unary_union
        import shapely
        if not hasattr(shapely, 'contains_xy'):
            raise ImportError('Shapely 2.0+ is required for exact buffered reads')
    except ImportError as exc:
        raise LNTDirectoryError(
            'Find Wires exact corridor reader requires Shapely 2.0 or newer. '
            'Install a compatible Python dependency before classifying.') from exc

    width = float(corridor_width)
    reach = float(max_ground_distance)
    if not np.isfinite(width) or width <= 0 or not np.isfinite(reach) or reach <= 0:
        raise LNTDirectoryError('Find Wires width and Ground distance must be finite and positive.')
    try:
        sequences = np.asarray(alignment_coords, dtype=np.float64)
        if sequences.ndim == 2:
            alignment_coords = [sequences]
    except (TypeError, ValueError):
        pass
    centerlines = []
    for coords in ([] if alignment_coords is None else alignment_coords):
        line = np.asarray(coords, dtype=np.float64)
        if (line.ndim != 2 or line.shape[1] < 2 or len(line) < 2
                or not np.isfinite(line[:, :2]).all()):
            raise LNTDirectoryError('Invalid selected centerline; no points were read.')
        geom = LineString(line[:, :2])
        if geom.is_empty or geom.length <= 0:
            raise LNTDirectoryError('Zero-length centerline; no points were read.')
        centerlines.append(geom)
    if not centerlines:
        raise LNTDirectoryError('Select at least one centerline for Find Wires.')
    # Small floating-point allowance around exact radius boundary.
    all_xy = np.vstack([np.asarray(g.coords) for g in centerlines])
    epsilon = max(1e-6, 16 * np.finfo(float).eps * np.max(np.abs(all_xy)))
    area = unary_union(centerlines).buffer(width / 2.0 + reach + epsilon, quad_segs=8)
    if area.is_empty or not area.is_valid:
        raise LNTDirectoryError('Unable to construct a valid centerline search corridor.')
    shapely.prepare(area)
    area_bounds = np.asarray(area.bounds, dtype=np.float64)
    if not np.isfinite(area_bounds).all():
        raise LNTDirectoryError('Invalid centerline buffer bounds.')

    info = inspect_directory(path, max_points=2**63 - 1)
    total = info['total']
    if expected_total is not None and total != int(expected_total):
        raise LNTDirectoryError(
            f'LNT directory total {total:,} differs from active cloud '
            f'{int(expected_total):,}; no changes were made.')
    if info['leaf_sum'] != total:
        raise LNTDirectoryError('Incomplete LNT leaf directory; no changes were made.')
    max_points = int(max_points)
    if max_points < 1:
        raise LNTDirectoryError('Invalid maximum buffered read size.')
    chunk_size = max(1, min(int(chunk_size), 500_000))

    xyz_parts, class_parts, node_parts, local_parts = [], [], [], []
    count_in_scope = 0
    intersected_leaves = 0
    origin = info['origin']
    leaves = info['leaves']
    with open(path, 'rb') as handle:
        for leaf_num, (node_index, offset, leaf_count, local_center, half) in enumerate(leaves):
            if cancelled is not None and cancelled():
                raise LNTDirectoryError('Find Wires read cancelled; no changes were made.')
            if progress is not None and leaf_num % 64 == 0:
                progress(5 + int(17 * leaf_num / max(1, len(leaves))),
                         f'Find Wires: checked {leaf_num:,}/{len(leaves):,} leaves; '
                         f'{count_in_scope:,} spatial context points')
            center = origin + np.asarray(local_center, dtype=np.float64)
            lo = center[:2] - float(half)
            hi = center[:2] + float(half)
            if (hi[0] < area_bounds[0] or lo[0] > area_bounds[2]
                    or hi[1] < area_bounds[1] or lo[1] > area_bounds[3]):
                continue
            if not shapely.intersects(area, shapely.box(lo[0], lo[1], hi[0], hi[1])):
                continue
            intersected_leaves += 1
            for start in range(0, leaf_count, chunk_size):
                if cancelled is not None and cancelled():
                    raise LNTDirectoryError('Find Wires read cancelled; no changes were made.')
                size = min(chunk_size, leaf_count - start)
                handle.seek(offset + start * POINT.itemsize)
                data = np.frombuffer(_exactly(handle, size * POINT.itemsize,
                    f'Find Wires leaf {node_index} rows {start}+{size}'), dtype=POINT)
                xy = np.empty((size, 2), dtype=np.float64)
                for axis, key in enumerate(('x', 'y')):
                    xy[:, axis] = (center[axis] - float(half) +
                                   (2.0 * float(half) / 65535.0) *
                                   data[key].astype(np.float64))
                rows = np.flatnonzero(shapely.contains_xy(area, xy[:, 0], xy[:, 1]))
                if not len(rows):
                    continue
                count_in_scope += len(rows)
                if count_in_scope > max_points:
                    raise LNTDirectoryError(
                        f'Find Wires buffered corridor has over {max_points:,} points. '
                        'Select a shorter CL or reduce maximum Ground distance. '
                        'No changes were made.')
                coords = np.empty((len(rows), 3), dtype=np.float64)
                coords[:, :2] = xy[rows]
                coords[:, 2] = (center[2] - float(half) +
                                 (2.0 * float(half) / 65535.0) *
                                 data['z'][rows].astype(np.float64))
                xyz_parts.append(coords)
                class_parts.append(data['classification'][rows].copy())
                node_parts.append(np.full(len(rows), node_index, dtype=np.uint32))
                local_parts.append((rows + start).astype(np.uint32))

    current = os.stat(path)
    if (info['size'], info['mtime_ns']) != (current.st_size, current.st_mtime_ns):
        raise LNTDirectoryError('LNT file changed during Find Wires read; no changes made.')
    if cancelled is not None and cancelled():
        raise LNTDirectoryError('Find Wires cancelled; no changes were made.')
    def combine(parts, dtype, empty_shape):
        return np.concatenate(parts) if parts else np.empty(empty_shape, dtype=dtype)
    result = {
        'xyz': combine(xyz_parts, np.float64, (0, 3)),
        'classification': combine(class_parts, np.uint8, (0,)),
        'node': combine(node_parts, np.uint32, (0,)),
        'local': combine(local_parts, np.uint32, (0,)),
        'file_stat': (info['size'], info['mtime_ns']),
    }
    if (len(result['xyz']) != count_in_scope
            or len(result['node']) != count_in_scope
            or len(result['local']) != count_in_scope
            or len(result['classification']) != count_in_scope
            or not np.all(np.isfinite(result['xyz']))):
        raise LNTDirectoryError('Find Wires exact corridor returned invalid identities or coordinates.')
    if progress is not None:
        progress(25, f'Find Wires: verified {count_in_scope:,} exact point IDs '
                     f'from {intersected_leaves:,} intersecting leaves')
    return result, info



def plan_streamed_find_wires(
        path, alignment_coords, corridor_width, from_classes, to_class,
        ground_class, min_height_above_ground, expected_total=None,
        max_changes=8_000_000, progress=None, cancelled=None,
        chunk_size=100_000, ground_cache_bytes=64 * 1024 * 1024):
    """Plan exact Find Wires edits WITHOUT a buffered whole-corridor snapshot.

    The LNT leaf directory is verified once. Candidate points are streamed in
    bounded packed-leaf chunks, tested against the precise full centerline
    geometry, then compared to the globally nearest *actual* Ground point.
    Ground points are queried lazily from directory leaves, in increasing
    conservative XY bounding-box distance order, and may belong to any leaf.
    Thus there is no arbitrary 50m Ground cutoff or 8M *context* record cap.

    Only changed node/local point IDs are retained for the one Undo-safe native
    commit performed by the caller. No LNT data is modified here. If more
    than max_changes points need modification, refuse BEFORE any native write.
    """
    try:
        import shapely
        from shapely.geometry import LineString
        from shapely.ops import unary_union
        from scipy.spatial import cKDTree
        from collections import OrderedDict
        if not hasattr(shapely, 'contains_xy'):
            raise ImportError('Shapely 2.0+ required')
    except ImportError as exc:
        raise LNTDirectoryError(
            'Find Wires streaming requires Shapely 2.0+ and SciPy.') from exc

    path = Path(path)
    width = float(corridor_width)
    hag = float(min_height_above_ground)
    if not np.isfinite(width) or width <= 0 or not np.isfinite(hag) or hag < 0:
        raise LNTDirectoryError('Find Wires width/HAG must be finite and nonnegative (width > 0).')
    if not 0 <= int(ground_class) <= 255 or not 0 <= int(to_class) <= 255:
        raise LNTDirectoryError('Invalid Ground or destination classification.')
    max_changes = int(max_changes)
    if max_changes < 1:
        raise LNTDirectoryError('Find Wires Undo/change budget must be positive.')
    chunk_size = max(1, min(int(chunk_size), 250_000))
    ground_cache_bytes = max(0, int(ground_cache_bytes))
    try:
        arr = np.asarray(alignment_coords, dtype=float)
        if arr.ndim == 2:
            alignment_coords = [arr]
    except (ValueError, TypeError):
        pass
    lines = []
    for coords in ([] if alignment_coords is None else alignment_coords):
        coords = np.asarray(coords, dtype=float)
        if (coords.ndim != 2 or coords.shape[1] < 2 or len(coords) < 2
                or not np.isfinite(coords[:, :2]).all()):
            raise LNTDirectoryError('Invalid Find Wires centerline coordinates.')
        geom = LineString(coords[:, :2])
        if geom.is_empty or not np.isfinite(geom.length) or geom.length <= 0:
            raise LNTDirectoryError('Find Wires centerline is empty or zero-length.')
        lines.append(geom)
    if not lines:
        raise LNTDirectoryError('Select one or more centerlines before Find Wires.')
    cl_geometry = unary_union(lines)
    all_line_xy = np.vstack([np.asarray(g.coords) for g in lines])
    epsilon = max(1e-6, 16 * np.finfo(float).eps * np.max(np.abs(all_line_xy)))
    half_width = width / 2.0
    # GEOS round endcaps are inscribed polygons: inflate the coarse pruning
    # buffer enough to enclose the true circular boundary at EVERY angle.
    # The final exact GEOS line distance below enforces the user width.
    quad_segs = 16
    outer = (half_width + epsilon) / np.cos(np.pi / (4 * quad_segs))
    candidate_area = cl_geometry.buffer(outer, quad_segs=quad_segs)
    if candidate_area.is_empty or not candidate_area.is_valid:
        raise LNTDirectoryError('Invalid Find Wires centerline corridor geometry.')
    shapely.prepare(candidate_area)
    area_bounds = np.asarray(candidate_area.bounds, dtype=float)

    any_class = from_classes is None or None in from_classes
    if not any_class:
        codes = [int(v) for v in from_classes]
        if not codes or any(v < 0 or v > 255 for v in codes):
            raise LNTDirectoryError('Select valid From Classes (0-255).')
        source_codes = np.asarray(codes, dtype=np.uint8)
    else:
        source_codes = None

    info = inspect_directory(path, max_points=2**63 - 1)
    if expected_total is not None and info['total'] != int(expected_total):
        raise LNTDirectoryError(
            f'LNT file contains {info["total"]:,} records, not the active '
            f'{int(expected_total):,}. No changes were made.')
    if info['leaf_sum'] != info['total']:
        raise LNTDirectoryError('Find Wires leaf count differs from the LNT total.')
    origin = info['origin']
    leaves = info['leaves']
    n_leaves = len(leaves)
    lo = np.empty((n_leaves, 2), dtype=float)
    hi = np.empty((n_leaves, 2), dtype=float)
    for j, (_, _, _, center, half) in enumerate(leaves):
        mid = origin[:2] + np.asarray(center[:2], dtype=float)
        lo[j] = mid - float(half)
        hi[j] = mid + float(half)

    cache = OrderedDict()
    cache_bytes = 0
    MAX_CACHED_LEAF_POINTS = 150_000
    stats = {'candidate_points': 0, 'ground_leaf_visits': 0,
             'ground_chunks': 0, 'candidate_leaves': 0, 'ground_missing': 0}
    stage_pct = 4

    def check_cancel():
        if cancelled is not None and cancelled():
            raise LNTDirectoryError('Find Wires cancelled before writing any changes.')

    def ground_arrays(raw, center, half):
        hits = np.flatnonzero(raw['classification'] == int(ground_class))
        if not len(hits):
            return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
        xy = np.empty((len(hits), 2), dtype=float)
        for ax, key in enumerate(('x', 'y')):
            xy[:, ax] = (center[ax] - half + (2.0 * half / 65535.0)
                         * raw[key][hits].astype(float))
        z = (center[2] - half + (2.0 * half / 65535.0)
             * raw['z'][hits].astype(float))
        return xy, z

    def ground_nearest(handle, candidate_xy):
        nonlocal cache_bytes
        # The node AABBs are conservative for all decoded packed points.
        # If the next node's minimum possible distance is larger than the
        # farthest current best distance, no unvisited node can improve any
        # candidate: that is the exact global-nearest stopping condition.
        amin = candidate_xy.min(axis=0)
        amax = candidate_xy.max(axis=0)
        gaps = np.maximum(np.maximum(lo - amax, amin - hi), 0.0)
        bound = np.hypot(gaps[:, 0], gaps[:, 1])
        order = np.argsort(bound, kind='stable')
        distances = np.full(len(candidate_xy), np.inf, dtype=float)
        nearest_z = np.full(len(candidate_xy), np.nan, dtype=float)

        def update(tree, z):
            if tree is None:
                return
            # cKDTree's distance_upper_bound is exclusive. Enlarge it
            # slightly to retain boundary candidates across platforms.
            current = float(np.max(distances))
            upper = (current + max(1e-8, 1e-9 * current)
                     if np.isfinite(current) else np.inf)
            d, indices = tree.query(candidate_xy, k=1, workers=1,
                                    distance_upper_bound=upper)
            good = np.isfinite(d) & (indices < len(z)) & (d < distances)
            if np.any(good):
                distances[good] = d[good]
                nearest_z[good] = z[indices[good]]

        for rank, j in enumerate(order):
            if rank % 32 == 0:
                if progress is not None:
                    progress(stage_pct, f'Find Wires: searching exact Ground references '
                             f'(visited {stats["ground_leaf_visits"]:,} leaves)')
                check_cancel()
            if np.all(np.isfinite(distances)) and bound[j] > float(np.max(distances)) + 1e-8:
                break
            if np.all(distances == 0):
                break
            stats['ground_leaf_visits'] += 1
            idx, offset, count, local_center, half = leaves[j]
            if not count:
                continue
            if j in cache:
                tree, z, memory = cache[j]
                cache.move_to_end(j)
                update(tree, z)
                continue
            center = origin + np.asarray(local_center, dtype=float)
            half = float(half)
            if count <= MAX_CACHED_LEAF_POINTS:
                handle.seek(offset)
                raw = np.frombuffer(_exactly(handle, count * POINT.itemsize,
                            f'Ground reference leaf {idx}'), dtype=POINT)
                xy, z = ground_arrays(raw, center, half)
                tree = cKDTree(xy) if len(z) else None
                update(tree, z)
                # Keep a small LRU of real reference points; never cache
                # large leaves, which would reintroduce a hidden RAM limit.
                memory = int(len(z) * 40 + 128)
                if memory <= ground_cache_bytes:
                    while cache and cache_bytes + memory > ground_cache_bytes:
                        _, (_, _, freed) = cache.popitem(last=False)
                        cache_bytes -= freed
                    cache[j] = (tree, z, memory)
                    cache_bytes += memory
            else:
                # Very large LNT leaves are scanned in fixed-size batches;
                # never materialize all Ground points into a KD-tree.
                for start in range(0, count, chunk_size):
                    check_cancel()
                    size = min(chunk_size, count - start)
                    handle.seek(offset + start * POINT.itemsize)
                    raw = np.frombuffer(_exactly(handle, size * POINT.itemsize,
                              f'Ground reference leaf {idx} chunk'), dtype=POINT)
                    stats['ground_chunks'] += 1
                    if progress is not None and stats['ground_chunks'] % 32 == 0:
                        progress(stage_pct, 'Find Wires: streaming Ground reference chunks')
                    xy, z = ground_arrays(raw, center, half)
                    if len(z):
                        update(cKDTree(xy), z)
                    if np.all(distances == 0):
                        break
        return distances, nearest_z

    node_parts = []
    local_parts = []
    old_parts = []
    change_count = 0
    with path.open('rb') as handle:
        for leaf_j, (node_index, offset, count, local_center, half) in enumerate(leaves):
            stage_pct = 4 + int(80 * leaf_j / max(n_leaves, 1))
            check_cancel()
            if progress is not None and leaf_j % 32 == 0:
                progress(stage_pct,
                    f'Find Wires: streamed {leaf_j:,}/{n_leaves:,} leaves; '
                    f'{stats["candidate_points"]:,} eligible corridor points')
            if not count:
                continue
            if (hi[leaf_j, 0] < area_bounds[0] or lo[leaf_j, 0] > area_bounds[2]
                    or hi[leaf_j, 1] < area_bounds[1] or lo[leaf_j, 1] > area_bounds[3]
                    or not shapely.intersects(candidate_area, shapely.box(
                        *lo[leaf_j], *hi[leaf_j]))):
                continue
            stats['candidate_leaves'] += 1
            center = origin + np.asarray(local_center, dtype=float)
            half = float(half)
            for start in range(0, count, chunk_size):
                check_cancel()
                size = min(chunk_size, count - start)
                handle.seek(offset + start * POINT.itemsize)
                raw = np.frombuffer(_exactly(handle, size * POINT.itemsize,
                               f'Find Wires candidate leaf {node_index}'), dtype=POINT)
                if source_codes is None:
                    source = np.flatnonzero((raw['classification'] != int(to_class))
                                                & (raw['classification'] != int(ground_class)))
                else:
                    source = np.flatnonzero(np.isin(raw['classification'], source_codes)
                        & (raw['classification'] != int(to_class))
                        & (raw['classification'] != int(ground_class)))
                if not len(source):
                    continue
                xy = np.empty((len(source), 2), dtype=float)
                for ax, key in enumerate(('x','y')):
                    xy[:, ax] = (center[ax] - half + (2.0 * half / 65535.0)
                        * raw[key][source].astype(float))
                close = shapely.contains_xy(candidate_area, xy[:,0], xy[:,1])
                if not np.any(close):
                    continue
                source = source[close]
                xy = xy[close]
                # GEOS distance to actual polylines, NOT just the approximate
                # buffer polygon. This preserves the requested width exactly.
                distance = shapely.distance(shapely.points(xy[:,0], xy[:,1]), cl_geometry)
                inside = distance <= half_width + epsilon
                if not np.any(inside):
                    continue
                source = source[inside]
                xy = xy[inside]
                stats['candidate_points'] += len(source)
                distances, ground_z = ground_nearest(handle, xy)
                has_ground = np.isfinite(distances)
                stats['ground_missing'] += int(np.count_nonzero(~has_ground))
                if not np.any(has_ground):
                    continue
                heights = (center[2] - half + (2.0 * half / 65535.0)
                           * raw['z'][source].astype(float)) - ground_z
                changed_rows = source[has_ground & (heights >= hag)]
                if not len(changed_rows):
                    continue
                change_count += len(changed_rows)
                if change_count > max_changes:
                    raise LNTDirectoryError(
                        f'Find Wires would change more than {max_changes:,} points, '
                        'exceeding the single-Undo transaction budget. No writes were made.')
                node_parts.append(np.full(len(changed_rows), node_index, dtype=np.uint32))
                local_parts.append((start + changed_rows).astype(np.uint32))
                old_parts.append(raw['classification'][changed_rows].copy())

    check_cancel()
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (info['size'], info['mtime_ns']):
        raise LNTDirectoryError('LNT file changed during Find Wires analysis. No changes were made.')
    if stats['candidate_points'] and stats['ground_missing'] == stats['candidate_points']:
        raise LNTDirectoryError(
            f'No Ground class {int(ground_class)} points exist in the active '
            'LNT cloud. No changes were made.')
    node = np.concatenate(node_parts) if node_parts else np.empty(0, dtype=np.uint32)
    local = np.concatenate(local_parts) if local_parts else np.empty(0, dtype=np.uint32)
    old = np.concatenate(old_parts) if old_parts else np.empty(0, dtype=np.uint8)
    if not (len(node) == len(local) == len(old) == change_count):
        raise LNTDirectoryError('Find Wires produced inconsistent point IDs; no writes were made.')
    # Directory-order leaf iteration and row-order chunk iteration make these
    # identities strictly increasing and unique without sorting millions.
    if len(node) > 1:
        ids = (node.astype(np.uint64) << 32) | local.astype(np.uint64)
        if not np.all(ids[1:] > ids[:-1]):
            raise LNTDirectoryError('Duplicate/out-of-order Find Wires IDs; no writes were made.')
    if progress is not None:
        progress(90, f'Find Wires: validated {stats["candidate_points"]:,} corridor '
                     f'candidates; {len(node):,} changes planned')
    return dict(node=node, local=local, old_classes=old,
                new_classes=np.full(len(old), int(to_class), dtype=np.uint8),
                file_stat=(info['size'], info['mtime_ns']), stats=stats), info



def plan_streamed_centerline_poles(
        path, centerline_coords, radius, down_height, from_classes, to_class,
        visible_classes=None, expected_total=None, max_changes=8_000_000,
        progress=None, cancelled=None, chunk_size=100_000):
    """Plan the legacy Ctrline vertex-radius algorithm on an arbitrarily large LNT.

    The legacy classifier builds a KD-tree of all eligible source points, then
    for EVERY distinct selected centerline vertex selects points within its XY
    radius whose Z is >= that vertex's local maximum minus down_height. These
    are *vertex circles*, not a continuous polyline corridor.

    First pass: bounded packed-leaf reads compute the exact local highest Z
    for every vertex. Second pass: bounded reads collect the union of eligible
    native node/local identities. All overlap is deduplicated within each
    physical source chunk. Only the changed IDs are held for a single reversible
    native commit; this function NEVER writes to the LNT file.
    """
    from scipy.spatial import cKDTree

    radius = float(radius)
    down_height = float(down_height)
    if not np.isfinite(radius) or radius <= 0:
        raise LNTDirectoryError('Ctrline vertex radius must be positive and finite.')
    if not np.isfinite(down_height) or down_height < 0:
        raise LNTDirectoryError('Ctrline downward height must be nonnegative and finite.')
    if not isinstance(to_class, (int, np.integer)) or not 0 <= int(to_class) <= 255:
        raise LNTDirectoryError('Ctrline destination class must be 0-255.')
    max_changes = int(max_changes)
    if max_changes <= 0:
        raise LNTDirectoryError('Ctrline Undo/change budget must be positive.')
    chunk_size = max(1, min(int(chunk_size), 250_000))

    # Match _normalize_centerline_collection() without importing Qt modules.
    if centerline_coords is None:
        raise LNTDirectoryError('Select one or more centerlines before Ctrline.')
    try:
        array = np.asarray(centerline_coords, dtype=np.float64)
        lines = [array] if array.ndim == 2 else list(centerline_coords)
    except (ValueError, TypeError):
        lines = list(centerline_coords)
    normalized = []
    for line in lines:
        try:
            arr = np.asarray(line, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise LNTDirectoryError('Invalid Ctrline centerline coordinates.') from exc
        if (arr.ndim != 2 or len(arr) < 2 or arr.shape[1] < 2
                or not np.isfinite(arr).all()
                or np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1).sum() <= 0):
            raise LNTDirectoryError('Ctrline requires nonempty, finite selected lines.')
        normalized.append(arr[:, :2])
    if not normalized:
        raise LNTDirectoryError('Select one or more centerlines before Ctrline.')
    # np.unique is used by the original classifier (same ordering/duplicates).
    vertices = np.unique(np.vstack(normalized), axis=0)
    vertex_tree = cKDTree(vertices)

    def codes_to_mask(classes):
        if from_codes is not None:
            keep = np.isin(classes, from_codes)
        else:
            keep = np.ones(len(classes), dtype=bool)
        if visible_codes is not None:
            keep &= np.isin(classes, visible_codes)
        return keep

    if from_classes is None or None in from_classes:
        from_codes = None
    else:
        selected = list(from_classes)
        if not selected or any(not isinstance(v, (int, np.integer))
                               or not 0 <= int(v) <= 255 for v in selected):
            raise LNTDirectoryError('Ctrline From Classes must be valid 0-255 IDs.')
        from_codes = np.asarray(selected, dtype=np.uint8)
    if visible_classes is None:
        visible_codes = None
    else:
        if any(not isinstance(v, (int, np.integer)) or not 0 <= int(v) <= 255
               for v in visible_classes):
            raise LNTDirectoryError('Ctrline display class IDs must be 0-255.')
        visible_codes = np.asarray(list(visible_classes), dtype=np.uint8)

    info = inspect_directory(path, max_points=2**63 - 1)
    if expected_total is not None and int(expected_total) != info['total']:
        raise LNTDirectoryError(
            f'Ctrline LNT has {info["total"]:,} points, but the active cloud '
            f'has {int(expected_total):,}. No changes were made.')
    if info['leaf_sum'] != info['total']:
        raise LNTDirectoryError('Ctrline LNT leaf total does not match directory header.')
    origin = info['origin']
    if not np.isfinite(origin).all():
        raise LNTDirectoryError('Ctrline LNT origin has nonfinite coordinates.')

    def check_cancel():
        if cancelled is not None and cancelled():
            raise LNTDirectoryError('Ctrline cancelled before any changes were written.')

    # Conservative leaf pruning: a point can be inside the vertex circle only
    # if the vertex is within radius of the square node's XY bounding box.
    # A KD-tree radius of half*sqrt(2)+radius is a quick overapproximation;
    # the following AABB check eliminates its false positives safely.
    relevant_leaves = []
    for node_index, offset, count, local_center, half in info['leaves']:
        check_cancel()
        if not count:
            continue
        center = origin + np.asarray(local_center, dtype=float)
        candidate_vertices = np.asarray(vertex_tree.query_ball_point(
            center[:2], r=radius + float(half) * np.sqrt(2.0)), dtype=np.intp)
        if not len(candidate_vertices):
            continue
        near = vertices[candidate_vertices]
        intersects = ((np.maximum(np.abs(near - center[:2]) - float(half), 0.0)
                       ** 2).sum(axis=1) <= radius * radius)
        candidate_vertices = candidate_vertices[intersects]
        if len(candidate_vertices):
            relevant_leaves.append((node_index, offset, count, center,
                                    float(half), candidate_vertices))

    top = np.full(len(vertices), -np.inf, dtype=np.float64)
    stats = dict(vertices=len(vertices), relevant_leaves=len(relevant_leaves),
                 source_points_scanned=0, touched_vertices=0)

    def visit_chunks(handle, phase):
        """Yield bounded source records + matching CL vertices in leaf order."""
        total_leaves = len(relevant_leaves)
        for leaf_i, (node_index, offset, count, center, half, v_ids) in enumerate(relevant_leaves):
            check_cancel()
            if progress is not None and leaf_i % 16 == 0:
                pct = (4 if phase == 1 else 48) + int((40 * leaf_i) / max(1, total_leaves))
                progress(pct, f'Ctrline pass {phase}/2: '
                         f'{leaf_i:,}/{total_leaves:,} candidate leaves')
            for start in range(0, count, chunk_size):
                check_cancel()
                size = min(chunk_size, count - start)
                handle.seek(offset + start * POINT.itemsize)
                raw = np.frombuffer(_exactly(handle, size * POINT.itemsize,
                                      f'Ctrline pass {phase} leaf {node_index}'),
                                    dtype=POINT)
                ids = np.flatnonzero(codes_to_mask(raw['classification']))
                if not len(ids):
                    continue
                xy = np.empty((len(ids), 2), dtype=np.float64)
                for axis, key in enumerate(('x', 'y')):
                    xy[:, axis] = (center[axis] - half + (2.0 * half / 65535.0)
                                   * raw[key][ids].astype(np.float64))
                # cKDTree exactly matches the legacy query_ball_point radius
                # test after we reconstruct the LNT's native coordinates.
                kdtree = cKDTree(xy)
                z = (center[2] - half + (2.0 * half / 65535.0)
                     * raw['z'][ids].astype(np.float64))
                yield node_index, start, ids, raw, z, v_ids, kdtree

    # Limit fanout as well as leaf chunk size: 1,000 coincident centerline
    # vertices * 100,000 source points would otherwise create a giant nested
    # list of hit indices in one query_ball_point call.
    vertex_batch = 8
    with open(path, 'rb') as handle:
        for _, _, _, _, z, v_ids, kdtree in visit_chunks(handle, phase=1):
            stats['source_points_scanned'] += len(z)
            for begin in range(0, len(v_ids), vertex_batch):
                check_cancel()
                current = v_ids[begin:begin + vertex_batch]
                groups = kdtree.query_ball_point(vertices[current], r=radius, workers=1)
                for vertex_id, hits in zip(current, groups):
                    if hits:
                        top[vertex_id] = max(top[vertex_id], float(np.max(z[hits])))
    stats['touched_vertices'] = int(np.isfinite(top).sum())
    check_cancel()

    node_parts, local_parts, old_parts = [], [], []
    changed_count = 0
    with open(path, 'rb') as handle:
        for node_index, start, ids, raw, z, v_ids, kdtree in visit_chunks(handle, phase=2):
            selected = np.zeros(len(ids), dtype=bool)
            for begin in range(0, len(v_ids), vertex_batch):
                check_cancel()
                current = v_ids[begin:begin + vertex_batch]
                groups = kdtree.query_ball_point(vertices[current], r=radius, workers=1)
                for vertex_id, hits in zip(current, groups):
                    if hits and np.isfinite(top[vertex_id]):
                        hit_indices = np.asarray(hits, dtype=np.intp)
                        selected[hit_indices[z[hit_indices] >= top[vertex_id] - down_height]] = True
            # No native write is needed when the class is already the target.
            selected &= raw['classification'][ids] != int(to_class)
            found = ids[selected]
            if not len(found):
                continue
            changed_count += len(found)
            if changed_count > max_changes:
                raise LNTDirectoryError(
                    f'Ctrline would change more than {max_changes:,} points, '
                    'exceeding the single-Undo transaction budget. No changes were made.')
            node_parts.append(np.full(len(found), node_index, dtype=np.uint32))
            local_parts.append((start + found).astype(np.uint32))
            old_parts.append(raw['classification'][found].copy())
    check_cancel()
    after = Path(path).stat()
    if (after.st_size, after.st_mtime_ns) != (info['size'], info['mtime_ns']):
        raise LNTDirectoryError('LNT changed during Ctrline analysis. No changes were made.')
    node = np.concatenate(node_parts) if node_parts else np.empty(0, dtype=np.uint32)
    local = np.concatenate(local_parts) if local_parts else np.empty(0, dtype=np.uint32)
    old = np.concatenate(old_parts) if old_parts else np.empty(0, dtype=np.uint8)
    if len(node) != changed_count or len(local) != changed_count or len(old) != changed_count:
        raise LNTDirectoryError('Ctrline produced inconsistent record IDs. No changes were made.')
    if len(node) > 1:
        identities = (node.astype(np.uint64) << 32) | local.astype(np.uint64)
        if not np.all(identities[1:] > identities[:-1]):
            raise LNTDirectoryError('Ctrline produced duplicate/out-of-order identities.')
    if progress is not None:
        progress(90, f'Ctrline: {stats["touched_vertices"]:,} vertices with '
                 f'sources, {len(node):,} exact edits planned')
    return dict(node=node, local=local, old_classes=old,
                new_classes=np.full(len(old), int(to_class), dtype=np.uint8),
                file_stat=(info['size'], info['mtime_ns']), stats=stats), info

def plan_exact_class_conversion(path, from_classes, to_class, visible_classes=None,
                                expected_total=None, max_changes=8_000_000,
                                progress=None, cancelled=None, chunk_size=250_000):
    """Plan an exact unfenced Convert against all LNT leaf classifications.

    Directory and leaf totals are verified first; only class bytes and stable
    directory node / leaf-local indices are retained for changed records.
    This never edits disk. It explicitly rejects more than ``max_changes``
    changes because the current native Undo stack is array-backed.

    A full-cloud read cap must NOT be confused with a cap on the number of
    records we can *inspect*: a 237M-point file with fewer than 8M edits can
    be classified safely after one complete read-only scan.
    """
    target = int(to_class)
    if not 0 <= target <= 255:
        raise LNTDirectoryError('To-class outside uint8 range.')
    allowed = np.zeros(256, dtype=bool)
    allowed[:] = True if from_classes is None else False
    if from_classes is not None:
        for item in from_classes:
            val = int(item)
            if not 0 <= val <= 255:
                raise LNTDirectoryError('From-class outside uint8 range.')
            allowed[val] = True
    if visible_classes is not None:
        visibility = np.zeros(256, dtype=bool)
        for item in visible_classes:
            val = int(item)
            if not 0 <= val <= 255:
                raise LNTDirectoryError('Visible class outside uint8 range.')
            visibility[val] = True
        allowed &= visibility
    allowed[target] = False
    max_changes = int(max_changes)
    if max_changes < 1:
        raise LNTDirectoryError('Convert requires a positive Undo-memory budget.')
    chunk_size = max(1, min(int(chunk_size), 500_000))
    info = inspect_directory(path, max_points=2**63 - 1)
    total = info['total']
    if expected_total is not None and total != int(expected_total):
        raise LNTDirectoryError('Active LNT point count disagrees with its validated directory.')
    if total != info['leaf_sum']:
        raise LNTDirectoryError('LNT leaf counts disagree with directory total.')
    node_parts, local_parts, old_parts = [], [], []
    visited = 0
    count = 0
    with open(path, 'rb') as handle:
        for leaf_no, (node_index, offset, points, _center, _half) in enumerate(info['leaves']):
            if cancelled is not None and cancelled():
                raise LNTDirectoryError('Convert cancelled during preflight; no changes were made.')
            if progress is not None and leaf_no % 64 == 0:
                progress(5 + int(80 * leaf_no / max(len(info['leaves']), 1)),
                         f'Convert: audited {visited:,}/{total:,} LNT points; '
                         f'{count:,} eligible changes')
            for start in range(0, points, chunk_size):
                if cancelled is not None and cancelled():
                    raise LNTDirectoryError('Convert cancelled during preflight; no changes were made.')
                size = min(chunk_size, points - start)
                handle.seek(offset + start * POINT.itemsize)
                block = np.frombuffer(_exactly(handle, size * POINT.itemsize,
                    f'Convert leaf {node_index} points {start}+{size}'), dtype=POINT)
                classes = block['classification']
                rows = np.flatnonzero(allowed[classes])
                count += len(rows)
                if count > max_changes:
                    raise LNTDirectoryError(
                        f'Convert would change over {max_changes:,} points, '
                        'exceeding the current reversible Undo-memory budget. '
                        'Narrow the From class selection or use a smaller file. '
                        'No changes were made.')
                if len(rows):
                    node_parts.append(np.full(len(rows), node_index, dtype=np.uint32))
                    local_parts.append((rows + start).astype(np.uint32))
                    old_parts.append(classes[rows].copy())
                visited += size
    if visited != total:
        raise LNTDirectoryError('Convert leaf scan did not read the complete LNT file.')
    now = os.stat(path)
    if (info['size'], info['mtime_ns']) != (now.st_size, now.st_mtime_ns):
        raise LNTDirectoryError('LNT changed while Convert was reading; no changes made.')
    if cancelled is not None and cancelled():
        raise LNTDirectoryError('Convert cancelled before classification; no changes made.')
    def combine(parts, dtype):
        return np.concatenate(parts) if parts else np.empty(0, dtype=dtype)
    return dict(node=combine(node_parts, np.uint32),
                local=combine(local_parts, np.uint32),
                old_classes=combine(old_parts, np.uint8),
                file_stat=(info['size'], info['mtime_ns'])), info
