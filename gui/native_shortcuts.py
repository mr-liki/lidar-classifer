"""Point-cloud shortcut dispatch for the native LNT/Vulkan viewports."""
from PySide6.QtWidgets import QApplication

from gui.native_sections import manager, CUT_SLOT
from gui.native_section_classify import DRAWABLE_TOOLS


TOOL_NAMES = {
    'AboveLine': 'above_line', 'BelowLine': 'below_line', 'ParallelLine': 'parallel_line',
    'Rectangle': 'rectangle', 'Circle': 'circle', 'Polygon': 'polygon',
    'Freehand': 'freehand', 'Brush': 'brush', 'Point': 'point',
}
DISPLAY_MODES = {'Class': 'class', 'RGB': 'rgb', 'Intensity': 'intensity',
                 'Elevation': 'elevation', 'Depth': 'depth', 'Line': 'line', 'Surface': 'surface'}

# The Measure ribbon emits these lowercase ids (menu_sidebar_system.py); the
# shortcut names on the right are what execute_native_shortcut() dispatches on,
# so both entry points reach the same native Vulkan tool.
MEASURE_NAMES = {
    'measure_line': 'MeasureLine',
    'measure_path': 'MeasurePath',
    'measure_grid_area': 'MeasureGridArea',
    'measure_block_area': 'MeasureBlockArea',
    'MeasureLine': 'MeasureLine',
    'MeasurePath': 'MeasurePath',
    'MeasureGridArea': 'MeasureGridArea',
    'MeasureBlockArea': 'MeasureBlockArea',
}


def focused_canvas(app):
    sections = getattr(app, '_native_sections', None)
    focused = QApplication.focusWidget()
    if sections is not None:
        for canvas in sections.views.values():
            if focused is canvas or (focused is not None and canvas.isAncestorOf(focused)):
                return canvas
    return None


def main_widget(app):
    view = getattr(app, '_lnt_main_view', None)
    return getattr(getattr(view, 'owner', None), 'vulkan_widget', None)


def _sync_class_picker_for_native_shortcut(app, tool_name, from_cls, to_cls):
    """Keep the floating ClassPicker in lock-step with native shortcuts.

    Native LNT/Vulkan shortcuts are dispatched before the legacy ``execute_tool``
    path. That fast path used to update ``app.from_classes`` / ``app.to_class``
    but never created or refreshed ClassPicker, so the popup stayed hidden (or
    showed the previous shortcut's classes) until the user clicked a classify
    tool in the ribbon. Make the native shortcut path publish the exact same
    live UI state as manual classification activation.

    The picker is deliberately non-modal. ``ensure_visible()`` may briefly
    activate its top-level window, so the caller restores focus to the target
    Vulkan canvas immediately afterwards.
    """
    normalized_from = None if from_cls == [] else from_cls
    app.from_classes = normalized_from
    app.to_class = to_cls
    app.active_classify_tool = tool_name

    # Keep right-click reactivation/history helpers aligned with the shortcut
    # that was just armed instead of retaining the previous manual tool state.
    app.last_classify_tool = tool_name
    app.last_classify_from_classes = normalized_from
    app.last_classify_to_class = to_cls

    try:
        digitizer = getattr(app, 'digitizer', None)
        if digitizer is not None:
            digitizer.enabled = False
    except Exception:
        pass

    picker = None
    get_live_picker = getattr(app, '_get_live_class_picker', None)
    if callable(get_live_picker):
        try:
            picker = get_live_picker()
        except Exception:
            picker = None
    else:
        picker = getattr(app, 'class_picker', None)

    try:
        if picker is None:
            from gui.class_picker import ClassPicker
            picker = ClassPicker(app, parent=app)
            app.class_picker = picker
            picker.configure_for_background_mode()
            print('CLASS_PICKER_SHORTCUT action=create')
        else:
            print('CLASS_PICKER_SHORTCUT action=refresh')

        # sync_with_app() updates title + From/To selections without rebuilding
        # the catalog unless the live PTC schema actually changed.
        picker.sync_with_app()
        picker.ensure_visible()
        print(
            f'CLASS_PICKER_SHORTCUT tool={tool_name} '
            f'from={normalized_from} to={to_cls} visible=YES'
        )
        return True
    except (RuntimeError, ReferenceError, AttributeError):
        # A deleted C++ QWidget can leave a Python wrapper behind. Clear it so
        # the next shortcut creates a fresh picker instead of remaining stale.
        try:
            app.class_picker = None
        except Exception:
            pass
    except Exception as exc:
        print(f'ClassPicker native shortcut sync failed: {exc}')

    return False


def native_classification_history(app, redo=False):
    view = getattr(app, '_lnt_main_view', None)
    store = getattr(app, '_lnt_section_store', None)
    if view is None or getattr(view, 'native', None) is None:
        return False
    source = getattr(app, 'redo_stack' if redo else 'undo_stack', [])
    destination = getattr(app, 'undo_stack' if redo else 'redo_stack', [])
    if source and isinstance(source[-1], dict) and 'lnt_node' in source[-1]:
        return _native_ai_history(app, view, source, destination, redo)
    if store is None or not source:
        app.statusBar().showMessage('Nothing to redo' if redo else 'Nothing to undo', 2500)
        return True
    entry = source[-1]
    if not isinstance(entry, dict) or not all(k in entry for k in ('indices', 'old_classes', 'new_classes')):
        app.statusBar().showMessage('This history entry is not a native point-cloud edit.', 3000)
        return True
    entry = source.pop()
    shown = store.data['classification'][entry['indices']].copy()
    store.data['classification'][entry['indices']] = entry['new_classes' if redo else 'old_classes']
    destination.append(entry)
    app.classification_revision = int(getattr(app, 'classification_revision', 0)) + 1
    store.show(entry['indices'], shown)            # recolour now; the write follows
    sections = getattr(app, '_native_sections', None)
    if sections is not None:
        sections.refresh_edits(force=True)
    return True


def snt_viewport_measure(app, name, key_label=''):
    """Measure tools on the Vulkan viewport of an SNT-only session (no .lnt).

    The SNT attach promotes the engine's surface to the visible canvas, so the
    VTK MeasurementTool underneath never sees a click. The promoted viewport's
    view (_SntViewportView: engine MVP world<->screen, no point stream) hosts
    the same NativeMeasure the .lnt main view uses; NativeMeasureInput feeds it
    the Vulkan widget's mouse/key events. Returns False when the engine is not
    the visible canvas, so the caller keeps the VTK tool.
    """
    if name not in ('MeasureLine', 'MeasurePath', 'MeasureGridArea', 'MeasureBlockArea'):
        return False
    from gui.native_vector_overlay import (_view_for, promote_snt_viewport,
                                           snt_viewport_active)
    if not snt_viewport_active(app):
        # Fresh session with nothing loaded: the canvas is still VTK's, so
        # arming the native overlay would draw where nobody looks. Take the
        # viewport for the native tools instead - the same §5 promotion an
        # SNT attach uses. It declines (-> VTK fallback stays) when a
        # VTK-rendered cloud owns the data or the engine cannot serve the
        # surface, so a data-bearing VTK session keeps its VTK tool.
        if not promote_snt_viewport(app):
            return False
    view = _view_for(app)
    widget = getattr(getattr(view, 'owner', None), 'vulkan_widget', None)
    if view is None or widget is None:
        return False
    from gui.native_main_classify import MainCanvas
    from gui.native_measure import NativeMeasure, ensure_snt_input
    stand_down_vtk_measurement(app)
    measurement = getattr(view, 'measurement', None)
    if measurement is None:
        measurement = view.measurement = NativeMeasure(MainCanvas(view))
    ensure_snt_input(view)
    creation = getattr(view, 'creation', None)
    if creation is not None and creation.active:
        creation.disarm()                 # creation takes clicks first; measuring wins now
    _arm_measurement(measurement, name)
    widget.setFocus()
    app.statusBar().showMessage(
        (f'[{key_label}] ' if key_label else '') + _MEASURE_HINTS[name], 4000)
    return True


_MEASURE_HINTS = {
    'MeasureLine': 'Click connected points; right-click finishes the total length.',
    'MeasurePath': 'Click points; right-click finishes the path.',
    'MeasureGridArea': 'Click inside a grid cell to measure its area.',
    'MeasureBlockArea': 'Click inside a block to measure its area.',
    'CrossSectionMeasure': 'Click 2 points in a cross-section view to measure.',
}


def _arm_measurement(measurement, name):
    measurement.cancel()
    # XS (CrossSectionMeasure) is the Line tool, confined to cross-section views.
    measurement.path = name == 'MeasurePath'
    measurement.style_key = 'cross_section' if name == 'CrossSectionMeasure' else None
    # Grid/block-area picking (cell or block polygon -> exact area) runs in
    # the same overlay tool; the mode switches click handling wholesale.
    measurement.area_mode = {'MeasureGridArea': 'grid',
                             'MeasureBlockArea': 'block'}.get(name)
    measurement.active = True
    refresh_measure_ribbon(measurement.app)


_MEASURE_RIBBON_IDS = {'MeasureLine': 'measure_line', 'MeasurePath': 'measure_path',
                       'MeasureGridArea': 'measure_grid_area',
                       'MeasureBlockArea': 'measure_block_area'}


def _measurement_ribbon_id(measurement):
    """Ribbon id of a native measurement overlay (None for the XS line)."""
    if getattr(measurement, 'style_key', None) == 'cross_section':
        return None
    if getattr(measurement, 'area_mode', None) == 'grid':
        return 'measure_grid_area'
    if getattr(measurement, 'area_mode', None) == 'block':
        return 'measure_block_area'
    return 'measure_path' if getattr(measurement, 'path', False) else 'measure_line'


def active_measure_ribbon_id(app):
    """The Line/Path/Block/Grid tool actually armed right now, if any.

    Native overlays win over the VTK tool: when a native viewport owns the
    clicks, the VTK MeasurementTool is only a stand-in.
    """
    for measurement in native_measurements(app):
        if getattr(measurement, 'active', False):
            ribbon_id = _measurement_ribbon_id(measurement)
            if ribbon_id is not None:
                return ribbon_id
    tool = getattr(app, 'measurement_tool', None)
    if (tool is not None and getattr(tool, 'active', False)
            and getattr(tool, 'is_measuring', False)):
        mode = str(getattr(tool, 'mode', '') or '')
        if mode in _MEASURE_RIBBON_IDS.values():
            return mode
    return None


def xs_measure_active(app):
    """XS (CrossSectionMeasure) armed: VTK tool, native flag, or an XS overlay.

    The native flag stays set while XS waits for section windows to open, so
    it counts as armed even when no measurement overlay exists yet - exactly
    like the button's own semantics.
    """
    tool = getattr(app, 'cross_section_measurement_tool', None)
    if tool is not None and getattr(tool, 'active', False):
        return True
    if getattr(app, '_native_xs_active', False):
        return True
    for measurement in native_measurements(app):
        if (getattr(measurement, 'style_key', None) == 'cross_section'
                and getattr(measurement, 'active', False)):
            return True
    return False


def refresh_measure_ribbon(app):
    """Match the Measure ribbon highlight to the armed tools (or clear them)."""
    try:
        ribbon = getattr(getattr(app, 'ribbon_manager', None), 'ribbons', {}).get('measure')
        sync = getattr(ribbon, 'sync_active_tool', None)
        if sync is not None:
            sync(active_measure_ribbon_id(app), xs_measure_active(app))
    except Exception as exc:      # noqa: BLE001 - sync must never break a tool
        print(f'⚠️ Measure ribbon sync failed: {exc}')


def deactivate_native_measurement(app, tool_id):
    """Stand down the native Line/Path/Block/Grid tool behind one ribbon id."""
    cancelled = False
    for measurement in native_measurements(app):
        if (getattr(measurement, 'active', False)
                and _measurement_ribbon_id(measurement) == tool_id):
            measurement.cancel()      # also refreshes the ribbon
            cancelled = True
    return cancelled


def stand_down_vtk_measurement(app):
    """A native measure owns the clicks now: the VTK tool must not observe.

    deactivate_completely() (not deactivate()) uninstalls the VTK observers,
    so Esc, right-click and Ctrl+Z reach only the native overlay while a
    Vulkan canvas - .lnt main view or promoted SNT viewport - is on screen.
    """
    tool = getattr(app, 'measurement_tool', None)
    if tool is None:
        return
    if not (getattr(tool, 'active', False)
            or getattr(tool, 'is_measuring', False)
            or getattr(tool, '_observer_tags', None)):
        return                    # never armed: leave unrelated VTK state alone
    try:
        if hasattr(tool, 'deactivate_completely'):
            tool.deactivate_completely()
        elif getattr(tool, 'active', False) or getattr(tool, 'is_measuring', False):
            tool.deactivate()
    except Exception as exc:      # noqa: BLE001 - arming must never fail
        print(f'⚠️ VTK measurement stand-down failed: {exc}')


def _xs_button(app):
    ribbon = getattr(app, 'ribbon_manager', None)
    measure = getattr(ribbon, 'ribbons', {}).get('measure') if ribbon is not None else None
    return getattr(measure, 'cross_section_measure_btn', None)


def arm_section_measure(canvas, on):
    """XS on one cross-section canvas: a two-click Line NativeMeasure, or stood down."""
    measurement = getattr(canvas, 'measurement', None)
    if on:
        if measurement is None:
            from gui.native_measure import NativeMeasure
            measurement = canvas.measurement = NativeMeasure(canvas)
        _arm_measurement(measurement, 'CrossSectionMeasure')
    elif measurement is not None and measurement.active:
        measurement.cancel()          # finished measurements stay on screen


def native_section_measure(app, on):
    """XS (Measure ribbon > Cross Section) in a native .lnt session.

    Arms the Line measurement on every open cross-section window - never the
    main view or the cut view - and on windows opened while it is on
    (NativeSections.ensure_dock). Returns False without a native session, so
    the caller keeps the VTK CrossSectionMeasurementTool.
    """
    view = getattr(app, '_lnt_main_view', None)
    if view is None or getattr(view, 'native', None) is None:
        return False
    app._native_xs_active = bool(on)
    sections = manager(app)
    if on:
        # Classification and the section-drawing gestures own the same clicks.
        app.active_classify_tool = None
        sections.cancel_cut()
        sections.cancel_cross()
        for classifier in [getattr(view, 'classifier', None)] + [
                c.classifier for c in sections.views.values()]:
            if classifier is not None:
                classifier.cancel()
        # Native owns XS on the Vulkan section canvases now: stand the VTK
        # CrossSectionMeasurementTool down so its observers stop stealing the
        # same clicks (and Ctrl+Z/Ctrl+Y) from the native overlay.
        try:
            tool = getattr(app, 'cross_section_measurement_tool', None)
            if tool is not None and getattr(tool, 'active', False):
                tool.deactivate()
        except Exception as exc:      # noqa: BLE001
            print(f'⚠️ VTK XS stand-down failed: {exc}')
    else:
        # Toggled off in a native session: make sure the VTK tool is not left
        # armed either (the ribbon handler returns early on the native path).
        try:
            tool = getattr(app, 'cross_section_measurement_tool', None)
            if tool is not None and getattr(tool, 'active', False):
                tool.deactivate()
        except Exception as exc:      # noqa: BLE001
            print(f'⚠️ VTK XS stand-down failed: {exc}')
    for slot, canvas in sections.views.items():
        if slot != CUT_SLOT:
            arm_section_measure(canvas, on)
    refresh_measure_ribbon(app)       # keep the XS button honest for direct callers
    return True


def set_xs(app, on):
    """Switch XS through its ribbon button (the one source of truth), or directly."""
    button = _xs_button(app)
    if button is not None:
        if button.isChecked() != bool(on):
            button.setChecked(bool(on))   # toggled -> ribbon sync -> native_section_measure
        elif bool(on) != bool(getattr(app, '_native_xs_active', False)):
            native_section_measure(app, on)
    else:
        native_section_measure(app, on)


# Tools that compete with XS for cross-section clicks switch it off (as the
# VTK tool's stand-downs did); everything else leaves it armed.
_XS_EXCLUSIVE = {'CrossSectionRect', 'CutSectionRect', 'CutFromCross', 'CutFromCut', 'Escape'}


def _native_ai_history(app, view, source, destination, redo):
    """Undo/redo one AI run on an .lnt: rewrite its points' classes by .lnt identity."""
    import numpy as np
    entry = source[-1]
    if entry.get('lnt_path') != view.lnt_path:
        app.statusBar().showMessage('That AI run belongs to another point cloud.', 3000)
        return True
    from gui.lnt_ai_bridge import active_bridge, sync_store_rows
    if active_bridge(app) is not None:
        app.statusBar().showMessage('Close the AI classification window before undoing it.', 3000)
        return True
    node, local = entry['lnt_node'], entry['lnt_local']
    classes = np.asarray(entry['new_classes' if redo else 'old_classes'], dtype=np.uint8)
    if not view.native.set_classes(node, local, classes):
        app.statusBar().showMessage(f'AI {"redo" if redo else "undo"} failed: '
                                    f'{view.native.last_error}', 4000)
        return True
    destination.append(source.pop())
    app.classification_revision = int(getattr(app, 'classification_revision', 0)) + 1
    sync_store_rows(app, view, node, local, classes)
    view.native.frame(0.0)
    try:
        from gui.point_count_widget import refresh_point_statistics
        refresh_point_statistics(app)
    except Exception:
        pass
    app.statusBar().showMessage(f'AI classification {"redone" if redo else "undone"} '
                                f'({len(classes):,} points)', 3000)
    return True


def execute_native_shortcut(app, tool, from_cls=None, to_cls=None, preset=None, key_label='',
                            pan_mode=None):
    view = getattr(app, '_lnt_main_view', None)
    if view is None or getattr(view, 'native', None) is None:
        name = MEASURE_NAMES.get(TOOL_NAMES.get(tool, tool), tool)
        return snt_viewport_measure(app, name, key_label)
    name = TOOL_NAMES.get(tool, tool)
    name = MEASURE_NAMES.get(name, name)      # Measure ribbon ids are lowercase
    supported = set(DRAWABLE_TOOLS) | set(DISPLAY_MODES) | {
        'CrossSectionRect', 'CutSectionRect', 'CutFromCross', 'CutFromCut',
        'TopView', 'Pan', 'DisplayMode', 'ShadingMode', 'SyncViews', 'FitView', 'Escape',
        'MeasureLine', 'MeasurePath', 'MeasureGridArea', 'MeasureBlockArea',
        'CrossSectionMeasure', 'ClearMeasurements',
        'Save', 'SaveAs', 'DrawSettings', 'LoadBlockGrid'}
    if name not in supported:
        return False
    sections = manager(app)
    canvas = focused_canvas(app)
    app._native_shortcut_target = canvas.slot if canvas is not None else 0
    app.active_classify_tool = None
    # All point tools release their native gesture owners, without VTK styles.
    sections.cancel_cut()
    sections.cancel_cross()
    for current in sections.views.values():
        current.classifier.cancel()
        current.drag = None
    main_classifier = getattr(view, 'classifier', None)
    if main_classifier is not None:
        main_classifier.cancel()
    for owner in [view] + list(sections.views.values()):
        measurement = getattr(owner, 'measurement', None)
        if measurement is not None:
            measurement.cancel()
    drawing = getattr(view, 'drawing', None)
    if drawing is not None:
        drawing.cancel()
    app._left_pan_shortcut_active = False
    if getattr(app, '_native_xs_active', False) and name != 'CrossSectionMeasure':
        if name in DRAWABLE_TOOLS or name in _XS_EXCLUSIVE:
            set_xs(app, False)
        else:
            native_section_measure(app, True)   # re-arm: the loop above cancelled it
    prefix = f'[{key_label}] ' if key_label else ''

    def status(message):
        app.statusBar().showMessage(prefix + message, 4000)

    def focus_main():
        widget = main_widget(app)
        if widget is not None:
            widget.setFocus()

    if name == 'CrossSectionMeasure':
        # Same as the XS button: toggle it (VTK shortcut semantics).
        on = not getattr(app, '_native_xs_active', False)
        set_xs(app, on)
        if on and not any(slot != CUT_SLOT for slot in sections.views):
            status('XS on - draw a cross section, then click 2 points in its view.')
        else:
            status(_MEASURE_HINTS[name] if on else 'Cross-section measurement disabled')
    elif name in ('MeasureLine', 'MeasurePath', 'MeasureGridArea', 'MeasureBlockArea'):
        from gui.native_measure import NativeMeasure
        if canvas is None:
            from gui.native_main_classify import MainCanvas
            if not view.mode_2d:
                refresh_measure_ribbon(app)   # nothing was armed: keep the ribbon honest
                status('Native measurements require the 2D plan view.')
                return True
            target = MainCanvas(view)
            owner = view
        else:
            target = owner = canvas
        stand_down_vtk_measurement(app)
        measurement = getattr(owner, 'measurement', None)
        if measurement is None:
            measurement = owner.measurement = NativeMeasure(target)
        _arm_measurement(measurement, name)
        target.setFocus()
        status(_MEASURE_HINTS[name])
    elif name == 'ClearMeasurements':
        for owner in [view] + list(sections.views.values()):
            measurement = getattr(owner, 'measurement', None)
            if measurement is not None:
                measurement.clear()
        status('Native measurements cleared')
    elif name in ('Save', 'SaveAs'):
        store = getattr(app, '_lnt_section_store', None)
        if store is not None:
            store.last_error = ''
            store.flush()
            if store.last_error:
                status(store.last_error)
                return True
        if name == 'SaveAs':
            from PySide6.QtWidgets import QFileDialog
            import shutil
            from pathlib import Path
            source = Path(view.lnt_path)
            destination, _ = QFileDialog.getSaveFileName(app, 'Save Native LNT As',
                str(source.with_name(source.stem + '_copy.lnt')), 'Native point cloud (*.lnt)')
            if not destination:
                return True
            destination = Path(destination)
            if destination.suffix.lower() != '.lnt':
                destination = destination.with_suffix('.lnt')
            if destination.resolve() != source.resolve():
                shutil.copy2(source, destination)
        status('Native point cloud saved')
    elif name == 'LoadBlockGrid':
        loader = getattr(getattr(app, 'grid_label_manager', None), 'load_block_or_grid_under_cursor', None)
        if callable(loader):
            from gui.data_loader import skip_import_prompt
            with skip_import_prompt():
                loader()
    elif name == 'DrawSettings':
        digitizer = getattr(app, 'digitizer', None)
        if digitizer is not None and preset:
            from gui.draw_settings_dialog import save_draw_settings
            styles = getattr(digitizer, 'draw_tool_styles', {})
            for name, style in preset.get('tools', {}).items():
                styles[name] = dict(style)
            digitizer.draw_tool_styles = styles
            save_draw_settings(styles)
            status('Draw settings preset applied')
    elif name == 'Escape':
        status('Native point tool deactivated')
    elif name == 'CrossSectionRect':
        sections.select_view()
        focus_main()
    elif name in ('CutSectionRect', 'CutFromCross', 'CutFromCut'):
        source = canvas.slot if canvas is not None and canvas.slot != CUT_SLOT else None
        sections.arm_cut(name == 'CutFromCut', source)
    elif name in DRAWABLE_TOOLS:
        if canvas is None:
            from gui.native_main_classify import ensure_classifier
            main_classifier = ensure_classifier(view)
            if main_classifier is None:
                status('Classification requires the native 2D plan view.')
                return True

        app.active_classify_target = ('cut' if canvas.slot == CUT_SLOT else 'cross') if canvas else 'main'

        # Native shortcut dispatch bypasses gui.execute_tool, therefore it must
        # explicitly publish the shortcut's From/To/tool state to ClassPicker.
        # Do this before restoring canvas focus so the popup can show/update
        # immediately without stealing the next classification key/click.
        _sync_class_picker_for_native_shortcut(app, name, from_cls, to_cls)

        if canvas is not None:
            sections.active_slot = canvas.slot
            canvas.setFocus()
        else:
            focus_main()

        normalized_from = None if from_cls == [] else from_cls
        if normalized_from is None:
            from_label = 'Any'
        elif isinstance(normalized_from, (list, tuple)):
            from_label = ', '.join(str(value) for value in normalized_from)
        else:
            from_label = str(normalized_from)
        to_label = 'Any' if to_cls is None else str(to_cls)
        status(
            f'{tool} ready in {"Main View" if canvas is None else "View " + str(canvas.slot)} '
            f'[{from_label} -> {to_label}]'
        )
    elif name == 'Pan':
        # Mode comes from the Shortcut Manager's Pan cell; consumed by
        # gui/native_pan.py in the main view and every cross/cut canvas.
        app._pan_shortcut_mode = pan_mode if pan_mode in ('left', 'tap') else 'left'
        app._left_pan_shortcut_active = True
        if canvas is not None:
            canvas.setFocus()
        else:
            focus_main()
        if app._pan_shortcut_mode == 'tap':
            status('Pan active — tap to start, tap again / Esc to stop')
        else:
            status('Pan active — left-drag to pan')
    elif name == 'FitView':
        if canvas is not None:
            canvas.fit()
        else:
            view.native.reset()
    elif name == 'TopView':
        view.set_2d(True)
        view.native.plan_view()
        app.current_view = 'top'
        focus_main()
    elif name == 'SyncViews':
        if preset:
            mapping = dict(getattr(app, 'view_sync_map', {}))
            for row in preset.get('rows', []):
                left, right = int(row.get('view', 1))-1, int(row.get('source', 1))-1
                target, source = (left, right) if preset.get('version', 1) >= 2 else (right, left)
                if row.get('mode', 0) == 1 and target != source:
                    mapping[target] = source
                elif preset.get('version', 1) >= 2:
                    mapping.pop(target, None)
                else:
                    mapping = {t: s for t, s in mapping.items() if s != left}
            sections.apply_sync_map(mapping)
        else:
            app.open_sync_views_dialog()
    elif name == 'DisplayMode':
        if not preset:
            app.open_display_mode()
            return True
        from gui.shortcut_manager import rebase_display_preset_to_current_ptc
        rebased = rebase_display_preset_to_current_ptc(preset, app)
        for slot, palette in rebased.get('views', {}).items():
            _apply_display(app, view, sections, int(slot), palette, 'class', status)
    else:
        mode = 'shaded_class' if name == 'ShadingMode' else DISPLAY_MODES[name]
        slot = int((preset or {}).get('target_view', (preset or {}).get('slot',
                   canvas.slot if canvas is not None else 0)))
        canvas = sections.views.get(slot) if slot else None
        palette = dict(canvas.palette if canvas is not None else
                       getattr(app, 'view_palettes', {}).get(slot, getattr(app, 'class_palette', {})))
        for code, rule in (preset or {}).get('classes', {}).items():
            code = int(code)
            if code in palette:
                palette[code] = dict(palette[code], show=bool(rule.get('show', False)))
        if name in ('ShadingMode', 'Surface'):
            settings = preset or {}
            app.last_shade_azimuth = float(settings.get('azimuth', 45.))
            app.shading_sharpness_angle = float(settings.get('angle', 45.))
            app.shade_ambient = float(settings.get('ambient', .25))
            if name == 'Surface':
                app.last_shade_angle = app.shading_sharpness_angle
            if canvas is not None:
                canvas.shading_parameters = (app.last_shade_azimuth,
                    app.shading_sharpness_angle, app.shade_ambient)
        _apply_display(app, view, sections, slot, palette, mode, status)
        if canvas is None:
            focus_main()
    return True


def _apply_display(app, view, sections, slot, palette, mode, status):
    if (slot and mode in ('line', 'surface')) or (not slot and mode in ('line', 'depth')):
        status(f'{mode.title()} is not available in this native viewport.')
        return
    if not hasattr(app, 'view_palettes'):
        app.view_palettes = {}
    app.view_palettes[slot] = palette
    if slot:
        sections.apply(slot, palette, mode)
    else:
        app.class_palette = palette
        view._push_applied_palette()
        view.set_app_display_mode(mode)
    status(f'{mode.replace("_", " ").title()} applied to {"Main View" if not slot else "View " + str(slot)}')


def native_measurements(app):
    """Every native measurement overlay: main view first, then section canvases."""
    owners = []
    for name in ('_lnt_main_view', '_snt_viewport_view'):
        view = getattr(app, name, None)
        if view is not None:
            owners.append(view)
    sections = getattr(app, '_native_sections', None)
    if sections is not None:
        owners.extend(sections.views.values())
    for owner in owners:
        measurement = getattr(owner, 'measurement', None)
        if measurement is not None:
            yield measurement


def cancel_native_measurements(app):
    """Stand down native measurement gestures (the VTK tool is taking over)."""
    for measurement in native_measurements(app):
        if getattr(measurement, 'active', False):
            measurement.cancel()


def clear_native_measurements(app) -> int:
    """Erase native measurement lines (Measure ribbon Clear / ClearMeasurements)."""
    cleared = 0
    for measurement in native_measurements(app):
        measurement.clear()
        cleared += 1
    return cleared
