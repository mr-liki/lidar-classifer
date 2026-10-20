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
