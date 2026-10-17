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


def native_classification_history(app, redo=False):
    view = getattr(app, '_lnt_main_view', None)
    store = getattr(app, '_lnt_section_store', None)
    if view is None or getattr(view, 'native', None) is None:
        return False
    if store is None:
        app.statusBar().showMessage('Nothing to redo' if redo else 'Nothing to undo', 2500)
        return True
    source = getattr(app, 'redo_stack' if redo else 'undo_stack', [])
    destination = getattr(app, 'undo_stack' if redo else 'redo_stack', [])
    if not source:
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
    from gui.native_vector_overlay import _view_for, snt_viewport_active
    if not snt_viewport_active(app):
        return False
    view = _view_for(app)
    widget = getattr(getattr(view, 'owner', None), 'vulkan_widget', None)
    if view is None or widget is None:
        return False
    from gui.native_main_classify import MainCanvas
    from gui.native_measure import NativeMeasure, NativeMeasureInput
    measurement = getattr(view, 'measurement', None)
    if measurement is None:
        measurement = view.measurement = NativeMeasure(MainCanvas(view))
    if getattr(view, '_measure_input', None) is None:
        view._measure_input = NativeMeasureInput(view, widget)
    _arm_measurement(measurement, name)
    widget.setFocus()
    app.statusBar().showMessage(
        (f'[{key_label}] ' if key_label else '') + _MEASURE_HINTS[name], 4000)
    return True


_MEASURE_HINTS = {
    'MeasureLine': 'Click 2 points to measure.',
    'MeasurePath': 'Click points; right-click finishes the path.',
    'MeasureGridArea': 'Click inside a grid cell to measure its area.',
    'MeasureBlockArea': 'Click inside a block to measure its area.',
    'CrossSectionMeasure': 'Click points; right-click finishes the path.',
}


def _arm_measurement(measurement, name):
    measurement.cancel()
    measurement.path = name in ('MeasurePath', 'CrossSectionMeasure')
    # Grid/block-area picking (cell or block polygon -> exact area) runs in
    # the same overlay tool; the mode switches click handling wholesale.
    measurement.area_mode = {'MeasureGridArea': 'grid',
                             'MeasureBlockArea': 'block'}.get(name)
    measurement.active = True


def execute_native_shortcut(app, tool, from_cls=None, to_cls=None, preset=None, key_label=''):
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
    prefix = f'[{key_label}] ' if key_label else ''

    def status(message):
        app.statusBar().showMessage(prefix + message, 4000)

    def focus_main():
        widget = main_widget(app)
        if widget is not None:
            widget.setFocus()

    if name in ('MeasureLine', 'MeasurePath', 'MeasureGridArea', 'MeasureBlockArea',
                'CrossSectionMeasure'):
        from gui.native_measure import NativeMeasure
        if name == 'CrossSectionMeasure' and canvas is None:
            canvas = sections.views.get(sections.active_slot)
            if canvas is None:
                status('Draw a cross section before measuring it.')
                return True
        if canvas is None:
            from gui.native_main_classify import MainCanvas
            if not view.mode_2d:
                status('Native measurements require the 2D plan view.')
                return True
            target = MainCanvas(view)
            owner = view
        else:
            target = owner = canvas
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
        app.from_classes = None if from_cls == [] else from_cls
        app.to_class = to_cls
        app.active_classify_tool = name
        app.active_classify_target = ('cut' if canvas.slot == CUT_SLOT else 'cross') if canvas else 'main'
        if canvas is not None:
            sections.active_slot = canvas.slot
            canvas.setFocus()
        else:
            focus_main()
        status(f'{tool} ready in {"Main View" if canvas is None else "View " + str(canvas.slot)}')
    elif name == 'Pan':
        app._left_pan_shortcut_active = True
        if canvas is not None:
            canvas.setFocus()
        else:
            focus_main()
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
