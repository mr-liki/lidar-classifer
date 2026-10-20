"""Exact-record By Class routing for a Vulkan/LNT main view.

The original VTK classifier functions are reused unchanged.  This adapter is
intentionally conservative: LNT stores uint8 classes, and native queries lack
point_source_id.  Unsupported input is rejected BEFORE any class write.

A single exact snapshot is used so neighbourhood algorithms do not acquire
approximate tile seams.  Bigger clouds need a separately audited, out-of-core
implementation and are never partially classified by this adapter.
"""
from __future__ import annotations

import os
import numpy as np

# Matches the native query cap used by the existing gui.native_by_class helper.
MAX_EXACT_POINTS = 8_000_000


class NativeByClassError(RuntimeError):
    pass


class NativeByClassCancelled(NativeByClassError):
    pass


def active_native(app):
    from gui.native_by_class import want_native
    return bool(want_native(app))


def checked_class(code, description='Class'):
    if code is None:
        raise NativeByClassError(f'{description} must be selected.')
    try:
        value = int(code)
    except (ValueError, TypeError, OverflowError) as exc:
        raise NativeByClassError(f'{description} must be an integer.') from exc
    if value != code or not 0 <= value <= 255:
        raise NativeByClassError(
            f'{description} {code} cannot be stored in an 8-bit LNT classification. '
            'This Vulkan build supports class IDs 0-255; use the VTK path '
            'for higher class IDs until a compatible native DLL exists.')
    return value


def validate_class_parameters(params):
    """Reject accidental uint8 wrap in top-level and macro class selectors."""
    params = dict(params or {})
    for key in ('to_class', 'reference_class', 'ground_class'):
        if key in params:
            checked_class(params[key], key)
    for key in ('from_classes', 'in_classes', 'current_ground', 'ground_classes'):
        if key not in params or params[key] is None:
            continue
        values = params[key]
        if isinstance(values, (str, bytes)):
            raise NativeByClassError(f'{key} must be class ID values, not text.')
        if isinstance(values, (int, np.integer)):
            values = [values]
        for code in values:
            if code is not None:
                checked_class(code, key)
    for step in params.get('steps', []) or []:
        validate_class_parameters(step)
        validate_class_parameters(step.get('params', {}))


def _available_cloud(app, *, allow_busy=False):
    from gui.native_by_class import native_main_view
    from gui.flight_line_filter import flight_line_visibility_for_slot
    view = native_main_view(app)
    if view is None:
        raise NativeByClassError('Native LNT query/write API is not available.')
    hidden = [key for key, on in flight_line_visibility_for_slot(app, 0).items() if not on]
    if hidden:
        raise NativeByClassError(
            'Cannot exactly honor hidden flight lines: the native query API '
            'does not supply point_source_id. Re-enable all flight lines or '
            'run this operation using the VTK data path.')
    if not os.path.isfile(view.lnt_path):
        raise NativeByClassError('Native LNT cloud file is missing.')
    if not os.access(view.lnt_path, os.W_OK):
        raise NativeByClassError('Native LNT cloud is read-only; classification not attempted.')
    if not allow_busy and getattr(app, '_native_byclass_busy', False):
        raise NativeByClassError('Another native By Class operation is already running.')
    # The existing bridge is a separate editing session with pending changes.
    from gui.lnt_ai_bridge import active_bridge
    if active_bridge(app) is not None:
        raise NativeByClassError('Close the active AI classification session before By Class.')
    return view


def read_snapshot(app, progress=None):
    """Read all source records with stable native node/local point identities."""
    from gui.native_by_class import cloud_bbox
    view = _available_cloud(app, allow_busy=True)
    x0, y0, x1, y1, total = cloud_bbox(view)
    if total > MAX_EXACT_POINTS:
        raise NativeByClassError(
            f'This cloud contains {total:,} points; the verified exact-query '
            f'limit is {MAX_EXACT_POINTS:,}. No changes were made. '
            'Large-cloud parity requires an audited out-of-core implementation.')
    if progress is not None:
        progress(1, f'Reading {total:,} exact LNT records')
    # The LAS/LNT header XY range can differ slightly from the coordinates
    # stored in native leaves (notably when node coordinates are float32).
    # First preserve the original exact query; on an incomplete count RETRY
    # progressively wider bounds. NEVER accept a partial snapshot or suppress
    # duplicate node/local identities. This is not a substitute for fixing a
    # native DLL that still drops records on the widest AABB.
    extrema = np.asarray((x0, y0, x1, y1), dtype=np.float64)
    if not np.all(np.isfinite(extrema)) or x0 > x1 or y0 > y1 or total < 0:
        raise NativeByClassError('Native LNT header reported invalid bounds/count.')
    ulp = tuple(float(np.nextafter(v, -np.inf)) for v in (x0, y0))
    urp = tuple(float(np.nextafter(v, np.inf)) for v in (x1, y1))
    # At projected coordinates of millions of metres, float32 quantization
    # can be tenths of a metre, not one float64 ULP.
    pad = max(1.0, 8.0 * float(np.max(np.abs(extrema))) *
              float(np.finfo(np.float32).eps))
    reach = max(1e12, float(np.max(np.abs(extrema))) * 2.0 + pad)
    attempts = (
        (ulp + (-1e12,), urp + (1e12,), 'header AABB'),
        ((float(x0-pad), float(y0-pad), -1e12),
         (float(x1+pad), float(y1+pad), 1e12), 'float32-padded AABB'),
        ((-reach, -reach, -reach), (reach, reach, reach), 'full-world AABB'),
    )
    required = ('xyz', 'classification', 'node', 'local')
    tried = []
    tile = None
    for index, (lo, hi, label) in enumerate(attempts):
        if index and progress is not None:
            progress(4 + index * 3, f'Retrying exact native query: {label}')
        tile = view.native.query_points(lo, hi, max_points=MAX_EXACT_POINTS)
        if tile is None:
            raise NativeByClassError(
                f'Native LNT {label} query failed: {view.native.last_error}. '
                'No classification was changed.')
        if any(k not in tile for k in required):
            raise NativeByClassError('Native query omitted required point identities.')
        sizes = [len(tile[k]) for k in required]
        if len(set(sizes)) != 1:
            raise NativeByClassError(
                f'Native {label} query returned inconsistent array lengths: {sizes}. '
                'No classification was changed.')
        tried.append((label, sizes[0]))
        print(f'[Native By Class] exact snapshot attempt={label} '
              f'count={sizes[0]:,}/{total:,}')
        if sizes[0] == total:
            break
        if sizes[0] > total:
            raise NativeByClassError(
                f'Native query returned {sizes[0]:,} points but header says '
                f'{total:,}; refusing mismatched identities. No changes made.')
    else:
        details = ', '.join(f'{label}: {count:,}' for label, count in tried)
        # Independent read-only verification of the LNT v2 leaf directory.
        # The query can miss a record even on a full-world AABB. Never infer
        # that a header count is correct: validate every packed leaf, then
        # compare EVERY native-returned identity, class, and coordinate with
        # that independent snapshot before allowing any classification write.
        path = getattr(view, 'lnt_path', None)
        if not path or not os.path.isfile(path):
            raise NativeByClassError(
                f'Incomplete native query: expected {total:,} records; {details}. '
                'Even full-world querying missed records; no readable LNT file for independent verification. No classification was changed.')
        try:
            from gui.native_lnt_directory import (read_exact_leaf_snapshot,
                                                   verify_native_subset)
            fallback, info = read_exact_leaf_snapshot(path, expected_total=total,
                                                       max_points=MAX_EXACT_POINTS)
            missing = verify_native_subset(tile, fallback)
            if missing <= 0:
                raise ValueError('Expected missing identities but leaf audit did not find any.')
            print(f'[Native By Class] LNT leaf audit passed: {total:,} records, '
                  f'{info["nodes"]:,} nodes, recovered {missing:,} omitted query record(s); '
                  'all overlapping native records matched exactly.')
            if progress is not None:
                progress(16, f'Independently verified all {total:,} LNT leaf records')
            tile = fallback
        except Exception as exc:
            raise NativeByClassError(
                f'Incomplete native query: expected {total:,} records; {details}. '
                f'Independent LNT leaf audit refused recovery: {exc}. '
                'No classification was changed.') from exc
    if np.asarray(tile['xyz']).shape != (total, 3):
        raise NativeByClassError('Native query returned invalid XYZ geometry.')
    if np.asarray(tile['classification']).dtype != np.uint8:
        raise NativeByClassError('Native classification dtype is not uint8.')
    if not np.all(np.isfinite(np.asarray(tile['xyz']))):
        raise NativeByClassError('LNT query returned nonfinite coordinates; no changes made.')
    # A count-matching query can still contain duplicates with missing points.
    # Verify unique native identities BEFORE classifier evaluation or writes.
    node = np.asarray(tile['node'], dtype=np.uint64)
    local = np.asarray(tile['local'], dtype=np.uint64)
    identities = (node << 32) | local
    if len(np.unique(identities)) != total:
        raise NativeByClassError(
            'Native query contains duplicated point identities; '
            'complete-cloud parity cannot be guaranteed. No changes made.')
    return view, tile


def _palette_eligible(app, classes):
    palette = getattr(app, 'class_palette', None)
    if not isinstance(palette, dict) or not palette:
        return np.ones(len(classes), dtype=bool)
    visible = [int(c) for c, cfg in palette.items()
               if isinstance(cfg, dict) and cfg.get('show', True)]
    # An explicitly ALL-HIDDEN palette must never turn into all-eligible.
    return np.isin(classes, visible) if visible else np.zeros(len(classes), dtype=bool)


def _polygon_coords(dialog):
    fences = getattr(dialog, 'selected_fences', None)
    # Algorithm dialogs keep selection on the embedded selector. An empty
    # top-level selection list must not hide an active embedded selection.
    if not fences:
        selector = getattr(dialog, '_fence_sel', None)
        embedded = getattr(selector, 'selected_fences', None)
        if embedded:
            fences = embedded
    if not fences:
        return []
    polys = []
    for fence in fences:
        coords = None
        getter = getattr(dialog, '_shape_coords', None)
        if not callable(getter):
            getter = getattr(getattr(dialog, '_fence_sel', None), '_shape_coords', None)
        if callable(getter):
            coords = getter(fence)
        if coords is None and isinstance(fence, dict):
            coords = fence.get('coords')
        try:
            arr = np.asarray(coords, dtype=np.float64)
        except (TypeError, ValueError):
            raise NativeByClassError('A selected fence has invalid coordinates.')
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2 or not np.isfinite(arr).all():
            raise NativeByClassError('A selected fence must have at least three finite XY points.')
        polys.append(arr[:, :2])
    return polys


def _fence_mask(xyz, polygons):
    """Match the VTK dialog's matplotlib polygon test without silent fallback.

    The older native tiled helper catches polygon errors and can return an
    empty mask.  For this exact-record path, an invalid polygon must fail
    loudly; otherwise a user could mistake an empty result for a valid run.
    """
    if not polygons:
        return np.ones(len(xyz), dtype=bool)
    from matplotlib.path import Path
    xy = np.asarray(xyz)[:, :2]
    inside = np.zeros(len(xy), dtype=bool)
    for coords in polygons:
        poly = np.asarray(coords, dtype=np.float64)
        if poly.ndim != 2 or len(poly) < 3 or poly.shape[1] < 2 or not np.isfinite(poly).all():
            raise NativeByClassError('Fence polygon has invalid coordinates.')
        poly = poly[:, :2]
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        in_bounds = ((xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) &
                     (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1]))
        if np.any(in_bounds):
            inside[in_bounds] |= Path(poly).contains_points(xy[in_bounds])
    return inside


def _height_mask(xyz, dialog):
    from gui.native_by_class import height_filter_mask
    extra = height_filter_mask(xyz, dialog)
    return np.ones(len(xyz), dtype=bool) if extra is None else extra


def compute_convert(xyz, classes, eligible, from_classes, to_class):
    mask = eligible.copy()
    mask &= (classes != to_class)
    if from_classes is not None:
        mask &= np.isin(classes, from_classes)
    classes[mask] = to_class
    return int(np.count_nonzero(mask))


def compute_close(xyz, classes, eligible, from_classes, to_class, radius,
                  filter_class=None, height_mask=None):
    from scipy.spatial import cKDTree
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0:
        raise NativeByClassError('Close radius must be positive and finite.')
    eligible = eligible & np.all(np.isfinite(xyz), axis=1)
    if height_mask is None:
        height_mask = np.ones(len(xyz), dtype=bool)
    # The original VTK Close uses scoped To-class points as references.
    refs = eligible & (classes == to_class)
    candidates = eligible & height_mask & (classes != to_class)
    if from_classes is not None:
        candidates &= np.isin(classes, from_classes)
    if not np.any(refs):
        # This is handled by the caller with a second explicit confirmation.
        return None
    if not np.any(candidates):
        return 0
    references = xyz[refs]
    if filter_class is not None:
        filter_refs = xyz[eligible & (classes == filter_class)]
        if not len(filter_refs):
            return 0
        distances, _ = cKDTree(filter_refs).query(references,
                                                    distance_upper_bound=radius)
        references = references[distances < radius]
        if not len(references):
            return 0
    inds = np.flatnonzero(candidates)
    distances, _ = cKDTree(references).query(xyz[inds], distance_upper_bound=radius)
    hits = inds[distances < radius]
    classes[hits] = to_class
    return int(len(hits))


def compute_height(xyz, classes, eligible, from_classes, to_class,
                   ground_class, min_height, max_height, max_triangle,
                   fence_polys=None):
    from gui.height_reference_surface import heights_above_reference_tin, has_tin_xy_support
    if min_height > max_height:
        raise NativeByClassError('Minimum height exceeds maximum height.')
    refs = xyz[classes == ground_class]
    if not len(refs):
        raise NativeByClassError(f'No reference-class {ground_class} points.')
    if fence_polys:
        local = refs[_fence_mask(refs, fence_polys)]
        if has_tin_xy_support(local):
            refs = local  # fallback to global refs if the fence cannot form a TIN
    candidates = eligible & (classes != to_class)
    if from_classes is not None:
        candidates &= np.isin(classes, from_classes)
    inds = np.flatnonzero(candidates)
    if not len(inds):
        return 0
    try:
        heights, supported, _ = heights_above_reference_tin(
            refs, xyz[inds], float(max_triangle))
    except Exception as exc:
        raise NativeByClassError(f'Height reference surface failed: {exc}') from exc
    selected = inds[supported & (heights >= min_height) & (heights <= max_height)]
    classes[selected] = to_class
    return int(len(selected))


def _dialog_sources(dialog):
    from PySide6.QtCore import Qt
    selected = dialog.from_list.selectedItems()
    if not selected:
        raise NativeByClassError('Select at least one From class.')
    values = [entry.data(Qt.UserRole) for entry in selected]
    return None if None in values else [checked_class(x, 'From class') for x in values]


def _ask(dialog, title, detail):
    from PySide6.QtWidgets import QMessageBox
    return (QMessageBox.question(dialog, title, detail,
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) == QMessageBox.Yes)


def _report(dialog, message, error=False):
    from PySide6.QtWidgets import QMessageBox
    if error:
        QMessageBox.warning(dialog, 'Native By Class', message)
    else:
        QMessageBox.information(dialog, 'Native By Class', message)


def _with_progress(dialog, name, operation, preview=False):
    """One GUI-thread query + compute + write, with cancellation before write."""
    from PySide6.QtWidgets import QApplication, QProgressDialog
    from PySide6.QtCore import Qt
    app = dialog.app
    _available_cloud(app)
    app._native_byclass_busy = True
    progress = QProgressDialog(f'{name}: reading cloud...', 'Cancel', 0, 100, dialog)
    progress.setWindowTitle(f'{name} (Vulkan exact)')
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)
    progress.show()
    try:
        def report(pct, msg=''):
            progress.setValue(max(0, min(100, int(pct))))
            if msg:
                progress.setLabelText(str(msg)[:180])
            QApplication.processEvents()
        view, tile = read_snapshot(app, progress=report)
        if progress.wasCanceled():
            raise NativeByClassCancelled('Cancelled before classification.')
        xyz = tile['xyz']
        original = tile['classification'].copy()
        updated = original.copy()
        if preview:
            report(30, 'Computing exact preview...')
        else:
            report(30, 'Running the original classification rules...')
        operation(xyz, updated, report, progress.wasCanceled)
        if progress.wasCanceled():
            raise NativeByClassCancelled('Cancelled; no changes were written.')
        changed = np.flatnonzero(updated != original)
        if preview:
            report(100, f'Preview: {len(changed):,} points')
            return len(changed)
        if len(changed):
            report(95, f'Writing {len(changed):,} changed LNT records...')
            nodes = np.ascontiguousarray(tile['node'][changed], dtype=np.uint32)
            locals_ = np.ascontiguousarray(tile['local'][changed], dtype=np.uint32)
            new = np.ascontiguousarray(updated[changed], dtype=np.uint8)
            old = np.ascontiguousarray(original[changed], dtype=np.uint8)
            if not view.native.set_classes(nodes, locals_, new):
                # Native DLL may have partially written before its failure;
                # attempt to restore the original classes immediately.
                view.native.set_classes(nodes, locals_, old)
                raise NativeByClassError('LNT write failed. Original classes were '
                                         'submitted for recovery; inspect the file. '
                                         + str(view.native.last_error))
            step = dict(lnt_path=view.lnt_path, lnt_node=nodes,
                        lnt_local=locals_, old_classes=old, new_classes=new)
            app.undo_stack.append(step)
            from gui.native_history_routing import record_native_action
            record_native_action(app, 'classification')
            for stack in (getattr(app, 'redo_stack', None),
                          getattr(app, 'redostack', None)):
                if isinstance(stack, list):
                    stack.clear()
            from gui.memory_manager import trim_undo_stack
            trim_undo_stack(app)
            app.classification_revision = int(getattr(app, 'classification_revision', 0)) + 1
            from gui.lnt_ai_bridge import sync_store_rows
            sync_store_rows(app, view, nodes, locals_, new)
            view.native.frame(0.0)
            try:
                from gui.point_count_widget import refresh_point_statistics
                refresh_point_statistics(app)
            except Exception:
                pass
        report(100, f'{name}: {len(changed):,} points changed')
        try:
            app.statusBar().showMessage(f'{name} (Vulkan): {len(changed):,} points changed', 7000)
        except Exception:
            pass
        return int(len(changed))
    finally:
        progress.close()
        app._native_byclass_busy = False


def _check_algorithm_result(result):
    """VTK classifiers expose stop_reason for NORMAL completion as well.

    Only confirmed error/cancel statuses should abort the native write. Unknown
    reasons still fail closed; do not assume all non-empty reasons are failures.
    """
    if result is None:
        return
    if not isinstance(result, dict):
        raise NativeByClassError('Classifier returned an invalid result type.')
    reason = str(result.get('stop_reason') or '').strip().lower()
    if not reason:
        return
    if reason in ('cancelled', 'canceled', 'aborted'):
        raise NativeByClassCancelled('Classifier cancelled.')
    accepted = {
        'completed', 'stable', 'thresholds reached', 'all candidates processed',
        'no candidates inside tin', 'no candidates', 'no points',
        'no eligible points', 'no eligible source points',
        'too few source points', 'too few finite source points',
        'fewer than 2 source points', 'fewer than 3 valid ground seeds',
        'fewer than 3 tin points', 'fewer than 3 sampled tin points',
        'completed one lowest-tier pass', 'completed one 3d count pass',
        'no source candidates after excluding trusted current-ground guide',
    }
    if reason not in accepted:
        raise NativeByClassError(f'Classifier stopped: {reason}')


def route_algorithm_dialog(dialog, func, kwargs, name):
    """Handle _BaseClassifyDialog._run_algorithm when a native LNT cloud is active."""
    if not active_native(dialog.app):
        return False
    try:
        kwargs = dict(kwargs)
        validate_class_parameters(kwargs)
        selector = getattr(dialog, '_fence_sel', None)
        fences = getattr(selector, 'selected_fences', None)
        polygons = _polygon_coords(dialog) if fences else []
        # The VTK caller cannot construct a mask when app.data is None.
        kwargs.pop('fence_mask', None)
        if not _ask(dialog, name, f'Run {name} on the native LNT cloud?\n\n'
                'Original VTK classifier rules will be used. Changes are written '
                'to the LNT file and can be undone. Back up the point cloud first.'):
            return True
        def calculate(xyz, updated, report, cancelled):
            eligible = _palette_eligible(dialog.app, updated) & _fence_mask(xyz, polygons)
            if not np.any(eligible):
                return
            before = updated.copy()
            def algorithm_progress(p, msg=''):
                report(30 + float(p)*.60, msg)
            result = func(xyz=xyz, classification=updated, fence_mask=eligible,
                          progress_cb=algorithm_progress, abort_check=cancelled,
                          **kwargs) or {}
            _check_algorithm_result(result)
            # Preserve native scope even for classifiers that only partly
            # implement VTK's fence_mask contract.
            updated[~eligible] = before[~eligible]
        count = _with_progress(dialog, name, calculate)
        # Only a successful nonempty exact native point write makes a fence
        # "used for classification". Preview/selection/cancellation do not.
        # Saved macros store their step fence coordinates without Python refs;
        # the marker helper checks unambiguous live drawings, including compact
        # native rectangles/circles expanded into world-coordinate polygons.
        if count > 0:
            used_fences = list(fences or ())
            if name == 'By Class Macro':
                for step in kwargs.get('steps', ()):
                    if step.get('operation') in ('Fence', 'Close', 'Height'):
                        used_fences.extend(step.get('lines', ()) or ())
            if used_fences:
                from gui.draw_clear_selection import mark_used_classification_fences
                mark_used_classification_fences(dialog.app, used_fences, operation=name)
        _report(dialog, f'{name}: {count:,} exact native records reclassified.')
    except NativeByClassCancelled:
        pass
    except Exception as exc:
        _report(dialog, str(exc), error=True)
    return True


def route_convert_dialog(dialog, mode, preview=False):
    """Handle original Convert, Close, Height, Fence dialogs on LNT."""
    if not active_native(dialog.app):
        return False
    try:
        if mode == 'fence':
            sources = dialog._get_selected_from_classes()
            if sources == []:
                raise NativeByClassError('Select at least one From class.')
        elif mode == 'height':
            # Unlike Close/Fence, ByClassHeightDialog does not implement
            # _get_selected_from_classes(); its VTK handler reads from_list.
            sources = _dialog_sources(dialog)
        elif mode == 'close':
            sources = dialog._get_selected_from_classes()
            if sources == []:
                raise NativeByClassError('Select at least one From class.')
        else:
            sources = _dialog_sources(dialog)
        target = checked_class(dialog.to_combo.currentData(), 'To class')
        if sources is not None:
            sources = [checked_class(x, 'From class') for x in sources]
        if mode == 'height':
            ground = checked_class(dialog.ref_class_combo.currentData(),
                                   'Reference class')
            if sources is not None and ground in sources:
                raise NativeByClassError('Reference class cannot be in From classes.')
        else:
            ground = None
        # Convert is always class-wide; Close/Height/Fence honor their fences.
        # Never accidentally scope Convert to stale dialog geometry.
        polygons = [] if mode == 'convert' else _polygon_coords(dialog)
        print(f'[Native By Class] mode={mode} preview={bool(preview)} '
              f'polygon_count={len(polygons)} sources={sources} target={target}')
        if mode == 'fence' and not polygons:
            raise NativeByClassError('Select at least one valid fence. No points changed.')
        if mode in ('height', 'close', 'fence') and sources is not None and target in sources:
            raise NativeByClassError('To class must not be among From classes.')
        scope_note = (
            '\n\nConvert is class-wide when no fence is selected. '
            'For a polygon-limited conversion, use By Class > Fence.'
            if mode == 'convert' and not polygons else ''
        )
        if not preview and not _ask(dialog, f'{mode.title()} on Vulkan',
                    f'Apply {mode.title()} to the LNT point cloud?\n\n'
                    'This uses the original VTK operation rules with the exact '
                    'native records. The LNT file will be changed; keep a backup.'
                    + scope_note):
            return True
        def calculate(xyz, updated, report, cancelled):
            initial = updated.copy()
            eligible = _palette_eligible(dialog.app, initial)
            if polygons and mode != 'height':
                inside = _fence_mask(xyz, polygons)
                n_inside = int(np.count_nonzero(inside))
                print(f'[Native By Class] {mode} polygon scope: '
                      f'{n_inside:,}/{len(xyz):,} native records inside '
                      f'{len(polygons)} selected fence(s)')
                report(32, f'{len(polygons)} fence(s): {n_inside:,} points inside')
                if (mode == 'fence' and not preview and n_inside == len(xyz)
                        and not _ask(dialog, 'Fence Includes Every Point',
                            f'Your selected fence(s) include all {len(xyz):,} '
                            'LNT records. Convert all eligible points anyway?')):
                    raise NativeByClassCancelled('All-points fence conversion declined.')
                eligible &= inside
            elif mode == 'convert':
                print('[Native By Class] Convert has no selected fence: '
                      'using class-wide scope (original Convert behavior)')
            if mode == 'convert':
                compute_convert(xyz, updated, eligible, sources, target)
            elif mode == 'fence':
                eligible &= _height_mask(xyz, dialog)
                compute_convert(xyz, updated, eligible, sources, target)
            elif mode == 'close':
                eligible &= _fence_mask(xyz, polygons)
                count = compute_close(xyz, updated, eligible, sources, target,
                      dialog.radius_spin.value(),
                      filter_class=dialog.filter_combo.currentData()
                        if dialog.filter_checkbox.isChecked() else None,
                      height_mask=_height_mask(xyz, dialog))
                if count is None:
                    # The VTK Close has a separate explicit "convert all" option
                    # when reference To-class points are absent.
                    if not _ask(dialog, 'No Reference Points',
                            'No To-class reference points found. Convert all eligible '
                            'From-class points instead?'):
                        raise NativeByClassCancelled('No target reference points.')
                    compute_convert(xyz, updated,
                        eligible & _height_mask(xyz, dialog), sources, target)
            elif mode == 'height':
                if polygons:
                    eligible &= _fence_mask(xyz, polygons)
                compute_height(xyz, updated, eligible, sources, target, ground,
                     dialog.min_height_spin.value(), dialog.max_height_spin.value(),
                     dialog.max_triangle_spin.value(), fence_polys=polygons)
            else:
                raise NativeByClassError(f'Unsupported By Class mode: {mode}')
            if cancelled():
                raise NativeByClassCancelled('Cancelled; nothing written.')
        count = _with_progress(dialog, mode.title(), calculate, preview=preview)
        if not preview and count > 0 and mode != 'convert' and polygons:
            fences_used = getattr(dialog, 'selected_fences', None)
            if not fences_used:
                selector = getattr(dialog, '_fence_sel', None)
                fences_used = getattr(selector, 'selected_fences', None)
            if fences_used:
                from gui.draw_clear_selection import mark_used_classification_fences
                mark_used_classification_fences(dialog.app, fences_used, operation=mode)
        label = getattr(dialog, 'preview_label', None) or getattr(dialog, 'info_label', None)
        if label is not None:
            label.setText(f'{"Preview" if preview else "Converted"}: {count:,} points')
        if not preview:
            ribbon = getattr(dialog, 'ribbon_parent', None)
            if ribbon is not None:
                ribbon.update_status(f'Converted {count:,} native points', 'success')
            _report(dialog, f'{mode.title()}: {count:,} native records reclassified.')
    except NativeByClassCancelled:
        pass
    except Exception as exc:
        _report(dialog, str(exc), error=True)
    return True
