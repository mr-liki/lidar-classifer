"""Transient highlights (identify / fence / markers) on the native Vulkan viewport.

WHY THIS EXISTS
    The PRJ Block Identifier drew its selection outlines and markers as VTK
    actors. With the native Vulkan main viewport installed (an .lnt session, or
    an SNT-only session whose canvas was promoted - gui/native_vector_overlay)
    the VTK canvas sits hidden behind the engine surface, so those actors were
    never seen.

WHERE THEY ARE DRAWN
    On a dedicated Qt CanvasOverlay over the same canvas the SNT labels use.
    Highlights are a handful of short rings that come and go every few seconds;
    putting them in the engine's vector arena would mean appending to (and on
    every clear, compacting) the arena that holds the attached SNT linework and
    re-pushing its draw ranges. The overlay touches none of that: attached
    linework draw ranges and arena offsets are never written by this module.

    Shapes are named groups ('identify', 'fence', 'marker', ...), so each dialog
    feature clears only its own highlight.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional


def native_viewport_active(app) -> bool:
    """True when the engine surface - not VTK - is the visible main canvas."""
    try:
        from gui.native_vector_overlay import snt_viewport_active
        return bool(snt_viewport_active(app))
    except Exception:
        return False


def _view_for(app):
    from gui.native_vector_overlay import _view_for as view_for
    return view_for(app)


def _make_overlay(view):
    """A CanvasOverlay on the view's main canvas, or None when there is none."""
    from gui.native_attachments import ensure_classifier
    from gui.native_section_classify import CanvasOverlay
    classifier = ensure_classifier(view)
    if classifier is None:
        return None
    return CanvasOverlay(classifier.canvas)


class HighlightHandle:
    """One highlight group; stands in for the VTK actor the dialog used to keep."""

    def __init__(self, overlay: "NativeHighlightOverlay", group: str):
        self._overlay = overlay
        self.group = group

    def remove(self) -> None:
        self._overlay.clear_group(self.group)

    def GetVisibility(self) -> int:
        return int(self.group in self._overlay.groups)


class NativeHighlightOverlay:
    def __init__(self, view, overlay):
        self.view = view
        self.overlay = overlay
        self.groups: Dict[str, List] = {}

    def set_group(self, group: str, shapes: Iterable) -> HighlightHandle:
        shapes = [s for s in shapes if s is not None]
        if shapes:
            self.groups[group] = shapes
        else:
            self.groups.pop(group, None)
        self._push()
        return HighlightHandle(self, group)

    def clear_group(self, group: str) -> None:
        if self.groups.pop(group, None) is not None:
            self._push()

    def clear(self) -> None:
        self.groups.clear()
        self._push()

    def _push(self) -> None:
        self.overlay.set_shapes([s for shapes in self.groups.values() for s in shapes])

    def shutdown(self) -> None:
        self.groups.clear()
        try:
            self.overlay.set_shapes([])
            self.overlay._timer.stop()
            self.overlay.close()
        except Exception:
            pass


def highlight_overlay_for(app) -> Optional[NativeHighlightOverlay]:
    """The app's highlight overlay, or None when no native viewport is visible.

    Rebuilt when the main view changes (an .lnt opened or closed); live groups
    carry over to the new canvas.
    """
    if not native_viewport_active(app):
        return None
    view = _view_for(app)
    if view is None:
        return None
    layer = getattr(app, "_native_highlight_overlay", None)
    if layer is not None and layer.view is view:
        return layer
    overlay = _make_overlay(view)
    if overlay is None:
        return None
    carried = dict(layer.groups) if layer is not None else {}
    if layer is not None:
        layer.shutdown()
    layer = NativeHighlightOverlay(view, overlay)
    app._native_highlight_overlay = layer
    for group, shapes in carried.items():
        layer.set_group(group, shapes)
    return layer


def clear_highlight_group(app, group: str) -> None:
    """Drop one group, without creating an overlay when none exists yet."""
    layer = getattr(app, "_native_highlight_overlay", None)
    if layer is not None:
        layer.clear_group(group)


def remove_highlight(app, item) -> bool:
    """Removes `item` if it is a native highlight handle; False for VTK actors."""
    if isinstance(item, HighlightHandle):
        item.remove()
        return True
    return False


def _hex(rgb) -> str:
    values = [float(v) for v in tuple(rgb)[:3]]
    if values and max(values) <= 1.0:
        values = [v * 255.0 for v in values]
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(v)))) for v in values)


def ring_shape(coords, color=(255, 255, 0), width: int = 2):
    """A closed world-space outline as a CanvasOverlay 'drawing' shape."""
    pts = []
    for p in coords or ():
        try:
            pts.append((float(p[0]), float(p[1])))
        except (TypeError, ValueError, IndexError):
            continue
    if len(pts) < 2:
        return None
    if len(pts) > 2 and pts[0] != pts[-1]:
        pts.append(pts[0])
    # 'block_creation' keeps the overlay from dotting every vertex.
    return ("drawing", pts, {"type": "polyline", "source": "block_creation",
                             "color": _hex(color), "width": int(width),
                             "style": "solid"})

