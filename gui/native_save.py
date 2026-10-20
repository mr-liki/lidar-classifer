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


def write_pair(source, lnt, destination, *, mapping=None, cache=None,
               classes=None, fences=None, attributes=None, version=None,
               lines=None, progress=lambda n, text: None, drawings=None):
    """Stage both outputs before replacing either. All heavy work stays off Qt's thread.

    Source LAS dimensions, CRS, VLRs and EVLRs are preserved. Class numbers above
    255 additionally live in naksha_class extra bytes; classification holds slots.
    """
    import laspy
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
    subset = classes is not None or bool(fences) or lines is not None
    same_lnt = os.path.normcase(os.path.abspath(lnt)) == os.path.normcase(target_lnt)
    if subset and same_lnt:
        raise ValueError('Save a filtered subset to a new filename to keep the open document intact.')
    same_source = source and os.path.normcase(os.path.abspath(source)) == os.path.normcase(destination)
    if source:
        order = source_mapping(source, lnt, entries, leaves, starts, cache, progress)
        source_slots = np.empty_like(slots)
        source_slots[order] = slots
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
    # Repeated unchanged Save is a histogram-sized comparison, with no LAZ recompression.
    digest = hashlib.sha256(slots.tobytes() + json.dumps(mapping, sort_keys=True).encode() + (drawings or b'')).digest()
    if (same_source and not subset and attributes is None and version is None and
            cache.get('saved_digest') == digest and cache.get('saved_stat') == os.stat(source).st_mtime_ns):
        progress(100, 'Both files are already saved.')
        return {'path': destination, 'lnt': target_lnt, 'count': len(slots), 'unchanged': True}
    folder = str(Path(destination).parent)
    # A private staging directory gives the builder normal .las/.laz extensions.
    # mkdir uses normal inherited Windows permissions (unlike Python 3.14 mkdtemp).
    import uuid
    stage_dir = Path(folder) / ('.naksha-save-' + uuid.uuid4().hex)
    stage_dir.mkdir()
    staged = str(stage_dir / ('points' + ext))
    staged_lnt = str(stage_dir / 'points.lnt')
    count = 0
    try:
        fast_las = False
        if source and Path(source).suffix.lower() == '.las' and ext == '.las' and not subset and attributes is None:
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
                            pts = laspy.ScaleAwarePointRecord.zeros(n, header=header)
                            pts.copy_fields_from(chunk)
                            if chunk.point_format.id < 6 <= header.point_format.id:
                                pts.scan_angle = np.rint(np.asarray(chunk.scan_angle_rank) / .006).astype(np.int16)
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
                        user_codes = lut[current]
                        if has_user_classes:
                            pts[USER_CLASS] = user_codes
                        keep = np.ones(n, dtype=bool)
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
                        out = pts[keep] if subset else pts
                        writer.write_points(out)
                        count += len(out)
                        at += n
                        progress(30 + int(50 * at / max(1, len(slots))), f'Writing {count:,} points…')
                    if reader and reader.header.evlrs:
                        writer.write_evlrs(reader.header.evlrs)
                finally:
                    if reader:
                        reader.close()
        if count == 0:
            raise ValueError('No points match the Save As selection.')
        if subset or attributes is not None:
            from gui.lnt_open import build_lnt
            build_lnt(staged, lambda n, text: progress(80 + int(n * .15), text), lambda: False)
        elif not same_lnt:
            progress(85, 'Copying the native document…')
            shutil.copyfile(lnt, staged_lnt)
        # Every output is staged before publication; replacement failures roll back.
        files = []
        if os.path.isfile(staged_lnt):
            files.append((staged_lnt, target_lnt))
        for i, target in enumerate((target_lnt + '.classmap.json', destination + '.classmap.json')):
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
        _publish(files, stage_dir)
        # A source save must not invalidate the edited native cache on next open.
        stamp = max(time.time_ns(), os.stat(destination).st_mtime_ns + 1_000_000)
        os.utime(target_lnt, ns=(stamp, stamp))
        if Path(destination).suffix.lower() in ('.las', '.laz') and os.path.isfile(target_lnt):
            from gui.lnt_open import stamp_source_cache
            stamp_source_cache(destination, target_lnt)
        if same_source and not subset and attributes is None:
            cache['saved_digest'] = digest
            cache['saved_stat'] = os.stat(destination).st_mtime_ns
            # Content changed but point order did not; retain the verified permutation.
            stat = os.stat(destination)
            cache['signature'] = (str(Path(source).resolve()), stat.st_size, stat.st_mtime_ns,
                                  hashlib.sha256(entries.tobytes()).digest())
        progress(100, 'Saved LAS/LAZ and native document.')
        return {'path': destination, 'lnt': target_lnt, 'count': count, 'unchanged': False}
    finally:
        shutil.rmtree(stage_dir)


def save_native(app, path=None, *, show_dialog=False, reopen=True, version=None):
    """Shared ribbon, shortcut and close/save entry point. Returns actual completion."""
    from PySide6.QtCore import QThread, Signal, Qt
    from PySide6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
    if getattr(app, '_native_save_running', False):
        return False
    view = app._lnt_main_view
    source = _source_for(app)
    source_exists = source and Path(source).suffix.lower() in ('.las', '.laz') and os.path.isfile(source)
    classes = lines = fences = attributes = None
    destination = path or (source if source_exists else str(Path(view.lnt_path).with_suffix('.las')))
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
    try:
        flush_edits(app)
        from gui.class_slots import _state
        mapping = dict(_state(app)['code_to_slot'])
        from gui.save_pointcloud import _serialize_drawings
        drawings = _serialize_drawings(app)
    except Exception as exc:
        QMessageBox.critical(app, 'Save failed', str(exc))
        return False
    cache = getattr(view, '_save_mapping_cache', None)
    if cache is None:
        view._save_mapping_cache = cache = {}

    class Worker(QThread):
        progress = Signal(int, str)

        def run(self):
            self.result, self.error = None, ''
            try:
                self.result = write_pair(source, view.lnt_path, destination, mapping=mapping,
                                         cache=cache, classes=classes, lines=lines, fences=fences,
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
        QMessageBox.critical(app, 'Save failed', worker.error)
        return False
    if os.path.normcase(os.path.abspath(worker.result['lnt'])) == os.path.normcase(os.path.abspath(view.lnt_path)):
        store = getattr(app, '_lnt_section_store', None)
        if store is not None and store.view is view:
            store.mark_saved()
    app.last_save_path = destination
    app.last_save_format = Path(destination).suffix[1:]
    app.statusBar().showMessage(f'Saved {worker.result["count"]:,} points — LAS/LAZ + LNT '
                                f'({time.perf_counter() - started:.2f} s)', 6000)
    if reopen and os.path.normcase(os.path.abspath(worker.result['lnt'])) != os.path.normcase(os.path.abspath(view.lnt_path)):
        from gui.lnt_open import open_with_lnt
        open_with_lnt(app, destination)
    return True
