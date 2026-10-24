"""Coordinate native Draw and LNT-classification Undo/Redo chronologically.

Only newly recorded actions enter this lightweight timeline. Existing history
stacks are retained and remain accessible when no timeline entry is present.
Never rewrite or rerun a classification algorithm during undo/redo.
"""
from __future__ import annotations


def record_native_action(app, kind):
    """Mark an *already committed* Draw/classification operation."""
    if kind not in ('draw', 'classification'):
        raise ValueError(f'Unsupported history kind: {kind}')
    undo = getattr(app, '_native_action_timeline_undo', None)
    if undo is None:
        undo = app._native_action_timeline_undo = []
    redo = getattr(app, '_native_action_timeline_redo', None)
    if redo is None:
        redo = app._native_action_timeline_redo = []
    # New mutation invalidates redo history across both domains, consistent
    # with conventional Undo/Redo. Keep the previously committed Undo entries.
    if redo:
        redo.clear()
        if kind == 'draw':
            for name in ('redo_stack', 'redostack'):
                stack = getattr(app, name, None)
                if isinstance(stack, list):
                    stack.clear()
        else:
            draw = getattr(getattr(app, '_lnt_main_view', None), 'drawing', None)
            if draw is not None:
                getattr(draw, '_object_redo', []).clear()
    undo.append(kind)
    if len(undo) > 100:
        del undo[:-100]


def _has_native_drawing_history(draw, redo=False):
    if draw is None:
        return False
    stack = getattr(draw, '_object_redo' if redo else '_object_actions', None)
    return bool(stack)


def _has_classification_history(app, redo=False):
    return bool(getattr(app, 'redo_stack' if redo else 'undo_stack', None))


def _has_pending_drawing(draw, redo=False):
    if draw is None or not getattr(draw, 'active', lambda: False)():
        return False
    if getattr(draw, 'vertex_target', None) is not None:
        return not redo
    if getattr(draw, 'points', None):
        return True
    if redo and getattr(draw, 'point_redo', None):
        return True
    if getattr(draw, 'tool', None) == 'line' and (
        getattr(draw, '_line_session_refs', None) or
        getattr(draw, '_line_session_undone', None)
    ):
        return True
    if getattr(draw, 'tool', None) == 'freehand' and (
        getattr(draw, '_freehand_undo_strokes', None) or
        getattr(draw, '_freehand_redo_strokes', None) or
        getattr(draw, 'dragging', False)
    ):
        return True
    return False


def route_native_history(app, drawing=None, *, redo=False):
    """Execute exactly one eligible action, or return False for VTK fallback.

    A pending vertex/drag always gets first priority. Completed actions use
    the recorded cross-tool order, irrespective of which dialog has focus.
    """
    if _has_pending_drawing(drawing, redo=redo):
        return bool(drawing.history(redo=redo))

    source = getattr(app, '_native_action_timeline_redo' if redo else
                     '_native_action_timeline_undo', None)
    dest = getattr(app, '_native_action_timeline_undo' if redo else
                   '_native_action_timeline_redo', None)
    if source:
        kind = source[-1]
        if kind == 'draw' and _has_native_drawing_history(drawing, redo):
            success = bool(drawing.history(redo=redo))
        elif kind == 'classification' and _has_classification_history(app, redo):
            fn = getattr(app, 'redo_classification' if redo else 'undo_classification', None)
            stack = getattr(app, 'redo_stack' if redo else 'undo_stack', None)
            before_length = len(stack)
            # The legacy app reports True for some *blocked* LNT histories
            # (different file, busy bridge, native write failure). An actual
            # Undo/Redo must consume exactly one source entry to count.
            handled = bool(fn()) if callable(fn) else False
            success = handled and len(stack) == before_length - 1
            if not success:
                return handled
        else:
            # A history transaction exists but cannot be applied. Do not
            # skip it and accidentally modify an older, unrelated action.
            return False
        if success:
            source.pop()
            if dest is None:
                name = '_native_action_timeline_undo' if redo else '_native_action_timeline_redo'
                dest = []
                setattr(app, name, dest)
            dest.append(kind)
        return success

    # Compatibility with existing Stage 6 histories recorded before this
    # hotfix. Prioritize nonempty native Draw history while its ribbon/tool
    # owns input; fall back to point classification when Draw is idle.
    if drawing is not None and getattr(drawing, 'active', lambda: False)() and (
        _has_native_drawing_history(drawing, redo=redo)
    ):
        return bool(drawing.history(redo=redo))
    if _has_classification_history(app, redo=redo):
        fn = getattr(app, 'redo_classification' if redo else 'undo_classification', None)
        if callable(fn):
            return bool(fn())
    if _has_native_drawing_history(drawing, redo=redo):
        return bool(drawing.history(redo=redo))
    return False
