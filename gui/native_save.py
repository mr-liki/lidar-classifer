"""Save full native documents and their LAS/LAZ companions, without a render readback."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

CHUNK = 1_000_000
USER_CLASS = 'naksha_class'


def _publish(files, stage_dir):
    """Rollback a partially published pair, including pre-existing sidecars."""
    backups, applied = {}, []
    try:
        for i, (staged, target) in enumerate(files):
            backup = str(stage_dir / f'previous-{i}')
            if os.path.isfile(target):
                try:
                    os.link(target, backup)
                except OSError:
                    shutil.copyfile(target, backup)
                backups[target] = backup
            if staged is None:
                if os.path.isfile(target):
                    os.remove(target)
            else:
                os.replace(staged, target)
            applied.append(target)
    except Exception:
        for target in reversed(applied):
            if target in backups:
                os.replace(backups[target], target)
            elif os.path.isfile(target):
                os.remove(target)
        raise


def _class_digest(slots, mapping, drawings):
    """Hash native classes without building another tens-of-megabytes bytes object."""
    digest = hashlib.sha256()
    digest.update(memoryview(slots))
    digest.update(json.dumps(mapping, sort_keys=True).encode('utf-8'))
    digest.update(drawings or b'')
    return digest.digest()


def _compatible_point_scaling(source_header, destination_header):
    """A source record is reusable only with identical physical XYZ scaling."""
    return (np.array_equal(source_header.scales, destination_header.scales)
            and np.array_equal(source_header.offsets, destination_header.offsets))


def _binary_prefix_compatible(source_dtype, destination_dtype):
    """Permit raw field copies ONLY for a byte-identical prefix layout.

    A new Naksha user-class extra dimension is sometimes appended to an
    existing LAS point format. The old bytes can be copied directly into a
    larger record only when *every* original field has the identical offset
    and dtype, and every additional output field begins AFTER the original
    record. Never apply this optimization to LAS 1.2 -> 1.4 format upgrades.
    """
    if (source_dtype.names is None or destination_dtype.names is None
            or source_dtype.hasobject or destination_dtype.hasobject
            or source_dtype.itemsize > destination_dtype.itemsize):
        return False
    for name in source_dtype.names:
        src = source_dtype.fields[name]
        dst = destination_dtype.fields.get(name)
        if dst is None or src[0] != dst[0] or src[1] != dst[1]:
            return False
    for name in destination_dtype.names:
        if name not in source_dtype.fields:
            if destination_dtype.fields[name][1] < source_dtype.itemsize:
                return False
    return True


def _full_source_record(chunk, header, laspy, *, allow_fast):
    """Build output with no lost LAS fields; avoid per-dimension copies.

    Reuse the decompressed chunk in-place if its point format matches exactly.
    For identical base formats with one appended extra field, copy raw bytes
    into the pre-zeroed wider record. Otherwise use laspy's existing safe
    dimension conversion logic, including LAS 1.2 -> 1.4 scan-angle handling.
    Filters and attribute selection run after this faithful record copy, so
    Save As retains its existing filtering and dimension-selection behavior.
    """
    if allow_fast and _compatible_point_scaling(chunk, header):
        if chunk.point_format == header.point_format:
            return chunk, 'direct'
        if chunk.point_format.id == header.point_format.id:
            pts = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
            if (chunk.array.flags.c_contiguous and pts.array.flags.c_contiguous
                    and _binary_prefix_compatible(chunk.array.dtype, pts.array.dtype)):
                input_size = chunk.array.dtype.itemsize
                pts.array.view(np.uint8).reshape(len(chunk), -1)[:, :input_size] = (
                    chunk.array.view(np.uint8).reshape(len(chunk), input_size))
                return pts, 'prefix'
    pts = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
    pts.copy_fields_from(chunk)
    if chunk.point_format.id < 6 <= header.point_format.id:
        pts.scan_angle = np.rint(np.asarray(chunk.scan_angle_rank) / .006).astype(np.int16)
    return pts, 'converted'


def native_active(app):
    view = getattr(app, '_lnt_main_view', None)
    return view is not None and getattr(view, 'native', None) is not None


def _source_for(app):
    source = getattr(app, '_lnt_source_path', None) or getattr(app, 'loaded_file', None)
    if source and Path(source).suffix.lower() == '.lnt':
        for suffix in ('.laz', '.las'):
            candidate = Path(source).with_suffix(suffix)
            if candidate.is_file():
                return str(candidate)
    return source


def flush_edits(app):
    """Run on the GUI thread, and verify success rather than trusting timer flush()."""
    bridge = getattr(app, '_lnt_ai_bridge', None)
    if bridge is not None and bridge.active:
        bridge.commit()
    store = getattr(app, '_lnt_section_store', None)
    if store is not None and store.view is app._lnt_main_view:
        store.flush()
        if not np.array_equal(store.data['classification'], store._synced):
            raise OSError(getattr(store, 'last_error', '') or 'Pending native edits could not be saved.')


def read_classes(path):
    from gui.lnt_class_counts import read_directory
    total, entries = read_directory(path)
    classes = np.empty(total, dtype=np.uint8)
    leaves = np.flatnonzero(entries['children_mask'] == 0)
    starts = np.concatenate(([0], np.cumsum(entries['point_count'][leaves], dtype=np.int64)))
    if int(starts[-1]) != total:
        raise ValueError('Native leaf point count does not match the file header.')
    with open(path, 'rb') as handle:
        for i, node in enumerate(leaves):
            entry = entries[node]
            count = int(entry['point_count'])
            handle.seek(int(entry['byte_offset']))
            raw = handle.read(count * 11)
            if len(raw) != count * 11:
                raise ValueError('Truncated native point records.')
            classes[starts[i]:starts[i + 1]] = np.frombuffer(raw, np.uint8)[10::11]
    return classes, entries, leaves, starts


def _mapping_cache_paths(lnt):
    """Where the source->document permutation is kept between sessions."""
    base = os.fspath(lnt)
    return base + '.sourceorder.npy', base + '.sourceorder.json'


def _signature_json(signature):
    """The mapping signature in a JSON-friendly form."""
    path, size, mtime_ns, digest = signature
    return [str(path), int(size), int(mtime_ns), digest.hex()]


def _load_mapping_cache(lnt, signature):
    """The permutation from a previous session, when it still matches.

    Building it costs a full pass over the source plus a sort (measured at
    ~9.5 s for 15 million points), and it is needed again by every export -
    including the backup written when the application closes, which is exactly
    the moment a user is waiting. The signature covers the source's bytes and
    the document's directory, so a stale cache can never be used.
    """
    import json
    order_path, meta_path = _mapping_cache_paths(lnt)
    try:
        with open(meta_path, encoding='utf-8') as handle:
            meta = json.load(handle)
        if meta.get('signature') != _signature_json(signature):
            return None
        order = np.load(order_path)
        if order.dtype.kind not in 'iu' or order.size <= 0:
            return None
        return order
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _save_mapping_cache(lnt, signature, order):
    """Keep the permutation beside the document for the next session."""
    import json
    order_path, meta_path = _mapping_cache_paths(lnt)
    temp_order = order_path + '.tmp.npy'
    temp_meta = meta_path + '.tmp'
    try:
        stored = order
        if stored.dtype == np.int64 and int(stored.max(initial=0)) < 2 ** 31:
            stored = stored.astype(np.int32)          # half the size on disk
        np.save(temp_order, stored)
        payload = {'signature': _signature_json(signature), 'points': int(order.size)}
        with open(temp_meta, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle)
        # The metadata is written last: without it the array is ignored.
        os.replace(temp_order, order_path)
        os.replace(temp_meta, meta_path)
    except (OSError, ValueError, TypeError):
        for temp in (temp_order, temp_meta):
            try:
                os.remove(temp)
            except OSError:
                pass


def source_mapping(source, lnt, entries, leaves, starts, cache, progress):
    """Cache only the permutation; repeated saves never sort coordinates again."""
    import laspy
    from gui.lnt_ai_bridge import _lnt_bounds, morton_keys
    stat = os.stat(source)
    signature = (str(Path(source).resolve()), stat.st_size, stat.st_mtime_ns,
                 hashlib.sha256(entries.tobytes()).digest())
    if cache.get('signature') == signature:
        return cache['order']
    total = int(starts[-1])
    order = _load_mapping_cache(lnt, signature)
    if order is not None and order.size == total:
        progress(25, 'Reusing the point mapping from the last session…')
        cache.clear()
        cache.update(signature=signature, order=order)
        return order
    bounds_min, bounds_max = _lnt_bounds(lnt)
    keys = np.empty(total, dtype=np.uint64)
    # Memory is bounded by keys/order, not a full XYZ + all LAS attributes array.
    with laspy.open(source) as reader:
        if reader.header.point_count != total:
            raise ValueError('The source point count differs from the native document. '
                             'Save As a new file to export the current native points.')
        at = 0
        for points in reader.chunk_iterator(CHUNK):
            xyz = np.column_stack((points.x, points.y, points.z))
            keys[at:at + len(points)] = morton_keys(xyz, bounds_min, bounds_max)
            at += len(points)
            progress(int(25 * at / max(1, total)), 'Matching source points (first save only)…')
    order = np.argsort(keys, kind='stable')
    del keys
    # Validate every record against its mapped source, in source chunks. This
    # also rejects a stale/foreign source before any destination is replaced.
    inverse = np.empty(total, dtype=order.dtype)
    inverse[order] = np.arange(total, dtype=order.dtype)
    cumulative = starts[1:]
    mm = np.memmap(lnt, dtype=np.uint8, mode='r')
    try:
        with laspy.open(source) as reader:
            at = 0
            for points in reader.chunk_iterator(CHUNK):
                records = inverse[at:at + len(points)]
                owners = np.searchsorted(cumulative, records, side='right')
                nodes = entries[leaves[owners]]
                offsets = (nodes['byte_offset'].astype(np.int64) +
                           (records - starts[owners]) * 11)
                rel = np.empty((len(points), 3), dtype=np.float64)
                for axis in range(3):
                    rel[:, axis] = (mm[offsets + 2 * axis].astype(np.uint16) |
                                    (mm[offsets + 2 * axis + 1].astype(np.uint16) << 8))
                half = nodes['half_size'].astype(np.float64)
                world = bounds_min + nodes['center'].astype(np.float64) - half[:, None]
                world += rel * (2 * half[:, None] / 65535.)
                xyz = np.column_stack((points.x, points.y, points.z))
                if np.any(np.abs(world - xyz) > (0.51 * 2 * half / 65535. + 1e-9)[:, None]):
                    raise ValueError('The native file no longer matches its LAS/LAZ source. '
                                     'The original file was left unchanged.')
                at += len(points)
    finally:
        del mm
    cache.clear()
    cache.update(signature=signature, order=order)
    _save_mapping_cache(lnt, signature, order)
    return order


def _native_chunks(lnt, entries, progress):
    """Standalone .lnt export: decode full-detail leaves, never resident LODs."""
    from gui.lnt_ai_bridge import _lnt_bounds
    # LNT v2 records: 3 uint16 local coordinates, RGB, intensity, class byte.
    POINT = np.dtype([('rel', '<u2', (3,)), ('rgb', 'u1', (3,)),
                      ('intensity', 'u1'), ('class', 'u1')])
    bmin, _ = _lnt_bounds(lnt)
    with open(lnt, 'rb') as handle:
        for entry in entries[entries['children_mask'] == 0]:
            count = int(entry['point_count'])
            handle.seek(int(entry['byte_offset']))
            for start in range(0, count, CHUNK):
                n = min(CHUNK, count - start)
                raw = handle.read(n * POINT.itemsize)
                if len(raw) != n * POINT.itemsize:
                    raise ValueError('Truncated native point records.')
                points = np.frombuffer(raw, dtype=POINT)
                half = float(entry['half_size'])
                xyz = bmin + entry['center'].astype(float) - half
                xyz = xyz + points['rel'].astype(float) * (2 * half / 65535.)
                yield xyz, points


def pending_deletions(app):
    store = getattr(app, '_lnt_section_store', None)
    if store is None or not hasattr(store, 'deletion_identities'):
        return None
    node, local = store.deletion_identities()
    return (node, local) if len(node) else None


def write_pair(source, lnt, destination, *, mapping=None, cache=None,
               classes=None, fences=None, attributes=None, version=None,
               lines=None, progress=lambda n, text: None, drawings=None,
               companion=True, deleted=None):
    """Stage both outputs before replacing either. All heavy work stays off Qt's thread.

    Source LAS dimensions, CRS, VLRs and EVLRs are preserved. Class numbers above
    255 additionally live in naksha_class extra bytes; classification holds slots.

    `companion=False` publishes ONLY the LAS/LAZ destination: the native
    document and its class map are left alone. That is the backup mode - the
    classified result is written to a NEW file and the open document (and the
    source it was built from) are never replaced.
    """
    import laspy
    total_started = time.perf_counter()
    cache = cache if cache is not None else {}
    mapping = mapping or {}
    destination = str(Path(destination).resolve())
    ext = Path(destination).suffix.lower()
    if ext not in ('.las', '.laz'):
        raise ValueError('Choose a LAS or LAZ destination.')
    target_lnt = str(Path(destination).with_suffix('.lnt'))
    source = str(source) if source and Path(source).suffix.lower() in ('.las', '.laz') and os.path.isfile(source) else None
    slots, entries, leaves, starts = read_classes(lnt)
    lut = np.arange(256, dtype=np.uint64)
    for code, slot in mapping.items():
        lut[int(slot)] = int(code)
    excluded = np.zeros(len(slots), dtype=bool)
    if deleted is not None:
        node, local = (np.asarray(a, dtype=np.int64) for a in deleted)
        positions = np.searchsorted(leaves, node)
        if (node.shape != local.shape or np.any(positions >= len(leaves))
                or np.any(leaves[positions] != node) or np.any(local < 0)
                or np.any(local >= entries['point_count'][node])):
            raise ValueError('Invalid deleted native record identity.')
        excluded[starts[positions] + local] = True
    deleting = bool(excluded.any())
    subset = classes is not None or bool(fences) or lines is not None
    filtering = subset or deleting
    same_lnt = os.path.normcase(os.path.abspath(lnt)) == os.path.normcase(target_lnt)
    if subset and same_lnt:
        raise ValueError('Save a filtered subset to a new filename to keep the open document intact.')
    same_source = source and os.path.normcase(os.path.abspath(source)) == os.path.normcase(destination)
    digest = _class_digest(slots, mapping, drawings)
    # A repeated unchanged Save needs neither source-order matching nor LAZ
    # recompression. The only trusted shortcut is an earlier successful Save
    # to this very destination, with unchanged source metadata and companion.
    if (same_source and companion and not filtering and attributes is None and version is None
            and cache.get('saved_path') == os.path.normcase(os.path.abspath(destination))
            and cache.get('saved_digest') == digest and os.path.isfile(target_lnt)
            and cache.get('saved_stat') == os.stat(source).st_mtime_ns):
        progress(100, 'Both files are already saved.')
        print(f'[NATIVE-SAVE-PERF] unchanged=True total={time.perf_counter() - total_started:.2f}s '
              'no LAZ recompression', flush=True)
        return {'path': destination, 'lnt': target_lnt, 'count': len(slots), 'unchanged': True}
    mapping_started = time.perf_counter()
    if source:
        order = source_mapping(source, lnt, entries, leaves, starts, cache, progress)
        source_slots = np.empty_like(slots)
        source_slots[order] = slots
        source_excluded = np.empty_like(excluded)
        source_excluded[order] = excluded
        with laspy.open(source) as reader:
            header = copy.deepcopy(reader.header)
        # Legacy point formats only hold 0..31. Upgrade without abusing LAS flags.
        wanted_version = version or str(header.version)
        high = int(slots.max()) if slots.size else 0
        if high > 31 and header.point_format.id < 6:
            wanted_version = '1.4'
            fmt = {4: 9, 5: 10}.get(header.point_format.id,
                                   7 if 'red' in header.point_format.dimension_names else 6)
        else:
            fmt = header.point_format.id
        minimum = '1.4' if fmt >= 6 else ('1.3' if fmt in (4, 5) else '1.2' if fmt in (2, 3) else '1.1')
        if tuple(map(int, wanted_version.split('.'))) < tuple(map(int, minimum.split('.'))):
            raise ValueError(f'Point format {fmt} requires LAS {minimum} or newer. Choose a newer version.')
        point_format = laspy.PointFormat(fmt)
        for dimension in header.point_format.extra_dimensions:
            point_format.add_extra_dimension(laspy.ExtraBytesParams(
                name=dimension.name, type=dimension.dtype,
                description=dimension.description, scales=dimension.scales,
                offsets=dimension.offsets))
        header.set_version_and_point_format(laspy.header.Version.from_str(wanted_version), point_format)
    else:
        source_slots = slots
        source_excluded = excluded
        header = laspy.LasHeader(point_format=7, version='1.4')
        from gui.lnt_ai_bridge import _lnt_bounds
        low, high_bounds = _lnt_bounds(lnt)
        header.offsets = low
        header.scales = np.maximum(0.000001, (high_bounds - low) / 2_000_000_000.)
        if lines is not None:
            raise ValueError('Standalone native files do not contain flight-line IDs.')
    has_user_classes = bool(mapping) and (attributes is None or 'Class' in attributes)
    if has_user_classes and USER_CLASS not in header.point_format.dimension_names:
        header.add_extra_dim(laspy.ExtraBytesParams(name=USER_CLASS, type=np.uint64,
                                                  description='Naksha user class number'))

    map_seconds = time.perf_counter() - mapping_started
    folder = str(Path(destination).parent)
    # A private staging directory gives the builder normal .las/.laz extensions.
    # mkdir uses normal inherited Windows permissions (unlike Python 3.14 mkdtemp).
    import uuid
    stage_dir = Path(folder) / ('.naksha-save-' + uuid.uuid4().hex)
    stage_dir.mkdir()
    staged = str(stage_dir / ('points' + ext))
    staged_lnt = str(stage_dir / 'points.lnt')
    count = 0
    source_record_modes = {'direct': 0, 'prefix': 0, 'converted': 0}
    export_started = time.perf_counter()
    try:
        fast_las = False
        if source and Path(source).suffix.lower() == '.las' and ext == '.las' and not filtering and attributes is None:
            with laspy.open(source) as original:
                fast_las = (original.header.point_format == header.point_format and
                            original.header.version == header.version)
        if fast_las:
            progress(30, 'Copying LAS and updating class bytes?')
            shutil.copyfile(source, staged)
            with laspy.mmap(staged) as mapped:
                for at in range(0, len(source_slots), CHUNK):
                    current = source_slots[at:at + CHUNK]
                    mapped.classification[at:at + len(current)] = current
                    if has_user_classes:
                        mapped[USER_CLASS][at:at + len(current)] = lut[current]
            count = len(source_slots)
        else:
            with laspy.open(staged, mode='w', header=header, do_compress=ext == '.laz') as writer:
                reader = laspy.open(source) if source else None
                try:
                    chunks = reader.chunk_iterator(CHUNK) if reader else _native_chunks(lnt, entries, progress)
                    at = 0
                    for chunk in chunks:
                        if reader:
                            n = len(chunk)
                            pts, record_mode = _full_source_record(
                                chunk, header, laspy,
                                allow_fast=True,  # only succeeds for proven matching layouts
                            )
                            source_record_modes[record_mode] += 1
                            xyz = None
                        else:
                            xyz, raw = chunk
                            n = len(raw)
                            pts = laspy.ScaleAwarePointRecord.zeros(n, header=header)
                            pts.x, pts.y, pts.z = xyz.T
                            pts.red, pts.green, pts.blue = (raw['rgb'].astype(np.uint16) * 257).T
                            pts.intensity = raw['intensity'].astype(np.uint16) * 257
                        current = source_slots[at:at + n]
                        pts.classification = current
                        # Allocate user-code/selection masks only when needed.
                        # Filtered Save As semantics remain exactly as before.
                        if has_user_classes or classes is not None:
                            user_codes = lut[current]
                        if has_user_classes:
                            pts[USER_CLASS] = user_codes
                        if filtering:
                            keep = ~source_excluded[at:at + n].copy()
                            if classes is not None:
                                keep &= np.isin(user_codes, classes)
                            if lines is not None:
                                keep &= np.isin(np.asarray(pts.point_source_id), lines)
                            if fences:
                                from gui.save_points_dialog import fence_inside_mask
                                if xyz is None:
                                    xyz = np.column_stack((pts.x, pts.y, pts.z))
                                keep &= fence_inside_mask(xyz, fences)
                        if attributes is not None:
                            from gui.save_pointcloud import _apply_attribute_selection
                            _apply_attribute_selection(pts, attributes)
                            if 'Class' not in attributes and USER_CLASS in pts.point_format.dimension_names:
                                pts[USER_CLASS] = np.zeros(n, dtype=np.uint64)
                        out = pts[keep] if filtering else pts
                        writer.write_points(out)
                        count += len(out)
                        at += n
                        progress(30 + int(50 * at / max(1, len(slots))), f'Writing {count:,} points…')
                    if reader and reader.header.evlrs:
                        writer.write_evlrs(reader.header.evlrs)
                finally:
                    if reader:
                        reader.close()
        encode_seconds = time.perf_counter() - export_started
        if count == 0 and not deleting and (len(slots) or subset):
            raise ValueError('No points match the Save As selection.')
        companion_started = time.perf_counter()
        if count == 0 and companion:
            # A valid empty v2 document replaces any old companion, atomically.
            import struct
            with open(lnt, 'rb') as native_file:
                header_bytes = bytearray(native_file.read(64))
            struct.pack_into('<Q', header_bytes, 8, 64)
            Path(staged_lnt).write_bytes(header_bytes + struct.pack('<QQ', 0, 0))
            Path(staged_lnt + '.lines').write_bytes(b'LNTLINE1' + struct.pack('<Q', 0))
        elif companion and (filtering or attributes is not None):
            from gui.lnt_open import build_lnt
            build_lnt(staged, lambda n, text: progress(80 + int(n * .15), text), lambda: False)
        elif companion and not same_lnt:
            progress(85, 'Copying the native document…')
            shutil.copyfile(lnt, staged_lnt)
        companion_seconds = time.perf_counter() - companion_started
        # Every output is staged before publication; replacement failures roll back.
        files = []
        if companion and os.path.isfile(staged_lnt):
            files.append((staged_lnt, target_lnt))
            if deleting or count == 0:
                lines_file = staged_lnt + '.lines'
                files.append((lines_file if os.path.isfile(lines_file) else None, target_lnt + '.lines'))
        classmap_targets = [target_lnt + '.classmap.json'] if companion else []
        classmap_targets.append(destination + '.classmap.json')
        for i, target in enumerate(classmap_targets):
            temp = None
            if mapping:
                payload = json.dumps({'version': 1, 'code_to_slot': {str(c): s for c, s in mapping.items()}}, indent=1)
                temp = str(stage_dir / f'map-{i}.json')
                Path(temp).write_text(payload, encoding='utf-8')
            files.append((temp, target))
        temp = None
        if drawings:
            temp = str(stage_dir / 'drawings')
            Path(temp).write_bytes(drawings)
        files.append((temp, destination + '.drawings.json'))
        if source:
            original = str(Path(source).with_suffix('.prj'))
            target = str(Path(destination).with_suffix('.prj'))
            if os.path.isfile(original) and os.path.normcase(original) != os.path.normcase(target):
                temp = str(stage_dir / 'projection.prj')
                shutil.copyfile(original, temp)
                files.append((temp, target))
        files.append((staged, destination))
        publish_started = time.perf_counter()
        _publish(files, stage_dir)
        publish_seconds = time.perf_counter() - publish_started
        # A source save must not invalidate the edited native cache on next open.
        if companion and os.path.isfile(target_lnt):
            stamp = max(time.time_ns(), os.stat(destination).st_mtime_ns + 1_000_000)
            os.utime(target_lnt, ns=(stamp, stamp))
            if Path(destination).suffix.lower() in ('.las', '.laz'):
                from gui.lnt_open import stamp_source_cache
                stamp_source_cache(destination, target_lnt)
        if same_source and not filtering and attributes is None:
            cache['saved_digest'] = digest
            cache['saved_path'] = os.path.normcase(os.path.abspath(destination))
            cache['saved_stat'] = os.stat(destination).st_mtime_ns
            # Content changed but point order did not; retain the verified permutation.
            stat = os.stat(destination)
            cache['signature'] = (str(Path(source).resolve()), stat.st_size, stat.st_mtime_ns,
                                  hashlib.sha256(entries.tobytes()).digest())
        progress(100, 'Saved LAS/LAZ and native document.')
        print(f'[NATIVE-SAVE-PERF] mapping={map_seconds:.2f}s '
              f'encode={encode_seconds:.2f}s companion={companion_seconds:.2f}s '
              f'publish={publish_seconds:.2f}s '
              f'total={time.perf_counter() - total_started:.2f}s '
              f'chunks={source_record_modes} points={count:,} '
              f'companion={companion} output={ext}', flush=True)
        return {'path': destination, 'lnt': target_lnt, 'count': count, 'unchanged': False}
    finally:
        shutil.rmtree(stage_dir)


def _same_save_path(left, right):
    """Compare on-disk destinations without confusing a session LNT with its companion."""
    if not left or not right:
        return False
    return os.path.normcase(os.path.abspath(os.fspath(left))) == os.path.normcase(os.path.abspath(os.fspath(right)))


def _should_reopen_after_save(source, destination, *, show_dialog, version, reopen):
    """Keep an already loaded Vulkan/LNT session for ordinary in-place Save.

    The private session's .lnt path necessarily differs from its published
    companion's path. That difference is *not* evidence that the document needs
    reloading. Explicit Save As, a new output path, a format change, or an
    explicit caller override retain their original reopen behavior.
    """
    if reopen is not None:
        return bool(reopen)
    return bool(show_dialog or version is not None or not _same_save_path(source, destination))


def save_native(app, path=None, *, show_dialog=False, reopen=None, version=None):
    """Shared ribbon, shortcut and close/save entry point. Returns actual completion."""
    from PySide6.QtCore import QThread, Signal, Qt
    from PySide6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
    if getattr(app, '_native_save_running', False):
        return False
    view = app._lnt_main_view
    source = _source_for(app)
    source_exists = source and Path(source).suffix.lower() in ('.las', '.laz') and os.path.isfile(source)
    classes = lines = fences = attributes = None
    original_document = getattr(view, 'original_lnt_path', None) or view.lnt_path
    destination = path or (source if source_exists else str(Path(original_document).with_suffix('.las')))
    if show_dialog:
        from gui.save_points_dialog import SavePointsDialog, collect_save_fences, _fence_ring
        import laspy
        default = 'LAS 1.4'
        dimensions = ('classification', 'intensity', 'red', 'green', 'blue')
        if source_exists:
            with laspy.open(source) as reader:
                default = f'{Path(source).suffix[1:].upper()} {reader.header.version}'
                dimensions = tuple(reader.header.point_format.dimension_names)
        palette = getattr(app, 'class_palette', {}) or {}
        class_items = [(int(c), f"{c} - {rule.get('description', '')}") for c, rule in palette.items()]
        def line_provider():
            found = np.zeros(65536, dtype=bool)
            with laspy.open(source) as reader:
                for points in reader.chunk_iterator(CHUNK):
                    found[np.asarray(points.point_source_id)] = True
            return [(int(i), str(i)) for i in np.flatnonzero(found)]
        options = SavePointsDialog(app, class_items, default_format=default,
                                  available_dimensions=dimensions, fence_items=collect_save_fences(app),
                                  line_provider=line_provider if source_exists else None)
        if options.exec() != options.Accepted:
            return False
        classes, attributes = options.selected_classes(), options.selected_attributes()
        lines = options.selected_lines()
        fences = [{'coords': _fence_ring(f, getattr(app, 'digitizer', None))}
                  for f in options.selected_fences()]
        selected_format = options.selected_format()
        version = selected_format.split()[-1]
        extension = '.laz' if selected_format.startswith('LAZ') else '.las'
        selected, _ = QFileDialog.getSaveFileName(app, 'Save As — LAS/LAZ and native LNT',
                                                str(Path(destination).with_suffix(extension)),
                                                f'{selected_format} (*{extension})')
        if not selected:
            return False
        destination = str(Path(selected).with_suffix(extension))
        target_lnt = str(Path(destination).with_suffix('.lnt'))
        if os.path.normcase(os.path.abspath(target_lnt)) != os.path.normcase(os.path.abspath(view.lnt_path)) and os.path.exists(target_lnt):
            if QMessageBox.question(app, 'Replace native companion?',
                                    f'Also replace the existing native file?\n{target_lnt}',
                                    QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return False
    # Ordinary Save writes back to the loaded LAS/LAZ without discarding the
    # active private LNT session, Vulkan buffers, section stores or camera.
    # Save As and explicit version/new-path exports retain the old reopen path.
    reopen_after_save = _should_reopen_after_save(
        source if source_exists else None, destination,
        show_dialog=show_dialog, version=version, reopen=reopen,
    )
    deleted = pending_deletions(app)
    if deleted is not None:
        reopen_after_save = True
    prepare_started = time.perf_counter()
    try:
        flush_edits(app)
        from gui.class_slots import _state
        mapping = dict(_state(app)['code_to_slot'])
        from gui.save_pointcloud import _serialize_drawings
        drawings = _serialize_drawings(app)
    except Exception as exc:
        QMessageBox.critical(app, 'Save failed', str(exc))
        return False
    prepare_seconds = time.perf_counter() - prepare_started
    cache = getattr(view, '_save_mapping_cache', None)
    if cache is None:
        view._save_mapping_cache = cache = {}

    class Worker(QThread):
        progress = Signal(int, str)

        def run(self):
            self.result, self.error = None, ''
            try:
                self.result = write_pair(source, view.lnt_path, destination, mapping=mapping,
                                         cache=cache, classes=classes, lines=lines, fences=fences, deleted=deleted,
                                         attributes=attributes, version=version, drawings=drawings,
                                         progress=lambda n, text: self.progress.emit(n, text))
            except Exception as exc:
                self.error = str(exc)

    class SaveProgress(QProgressDialog):
        def reject(self):
            # Esc cannot dismiss a running transactional save and block Qt in wait().
            pass

        def closeEvent(self, event):
            event.ignore()

    dialog = SaveProgress('Saving LAS/LAZ and native document…', None, 0, 100, app)
    dialog.setWindowTitle('Save points')
    dialog.setWindowModality(Qt.ApplicationModal)
    dialog.setWindowFlag(Qt.WindowCloseButtonHint, False)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    worker = Worker(app)
    def update(n, text):
        dialog.setValue(n)
        dialog.setLabelText(text)
    worker.progress.connect(update)
    worker.finished.connect(dialog.accept)
    app._native_save_running = True
    started = time.perf_counter()
    try:
        worker.start()
        dialog.exec()
        worker.wait()
    finally:
        app._native_save_running = False
        dialog.deleteLater()
        worker.deleteLater()
    if worker.error:
        print(f'[NATIVE-SAVE-PERF] FAILED after={time.perf_counter() - started:.2f}s '
              f'gui_prepare={prepare_seconds:.2f}s', flush=True)
        QMessageBox.critical(app, 'Save failed', worker.error)
        return False
    print(f'[NATIVE-SAVE-PERF] gui_prepare={prepare_seconds:.2f}s '
          f'worker_wall={time.perf_counter() - started:.2f}s', flush=True)
    # The active LNT is a *private writable session*; its filename almost
    # always differs from the companion saved next to the LAZ. When the whole
    # source file was saved in place, the in-memory classifications are now a
    # committed checkpoint even though those two LNT paths differ. This keeps
    # Clear Project -> Don't Save from accidentally undoing a successful Save.
    full_source_save = (
        source_exists and _same_save_path(source, destination)
        and classes is None and lines is None and not fences and attributes is None
    )
    checkpoint_needed = (
        _same_save_path(worker.result['lnt'], view.lnt_path) or full_source_save
    )
    if checkpoint_needed and deleted is None:
        store = getattr(app, '_lnt_section_store', None)
        if store is not None and store.view is view:
            try:
                store.mark_saved()
            except Exception as exc:
                # The on-disk export already committed; never report it as a
                # failed export or initiate another reload to mask this issue.
                print(f'[NATIVE-SAVE] WARNING: session checkpoint failed: {exc}', flush=True)
                QMessageBox.warning(
                    app, 'Save checkpoint warning',
                    'The LAS/LAZ was saved, but the in-memory section checkpoint '
                    f'could not be updated: {exc}\n'
                    'Do not discard further edits until this is resolved.',
                )
    if deleted is not None:
        # Compaction assigns new node/local identities. Old row history cannot
        # be replayed against the replacement document.
        app.undo_stack.clear()
        app.redo_stack.clear()
        cache.clear()
    app.last_save_path = destination
    app.last_save_format = Path(destination).suffix[1:]
    app.statusBar().showMessage(f'Saved {worker.result["count"]:,} points — LAS/LAZ + LNT '
                                f'({time.perf_counter() - started:.2f} s)', 6000)
    if reopen_after_save and (deleted is not None or not _same_save_path(worker.result['lnt'], view.lnt_path)):
        from gui.lnt_open import open_with_lnt
        print(f'[NATIVE-SAVE] reopening exported document: {destination}', flush=True)
        open_with_lnt(app, destination)
    else:
        print('[NATIVE-SAVE] kept current Vulkan/LNT session; '
              'no close/reopen or tile/cache rebuild', flush=True)
    return True
