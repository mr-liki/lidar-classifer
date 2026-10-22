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
