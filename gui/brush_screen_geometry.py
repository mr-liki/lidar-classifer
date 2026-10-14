"""Brush geometry in the render surface's pixel coordinates."""
import ctypes
import sys
from ctypes import wintypes

import numpy as np


def native_cursor_position(widget, render_window, user32=None):
    """Return cursor/client bounds from the HWND VTK actually renders into.

    QWidget geometry can disagree with its native child after a floating dock
    is reparented. ScreenToClient bypasses that geometry cache. All dimensions
    here use the same native coordinate system, including monitor DPI.
    """
    if user32 is None:
        if sys.platform != 'win32':
            return None
        user32 = ctypes.windll.user32
    raw = render_window.GetGenericWindowId()
    if isinstance(raw, str) and raw.startswith('_'):
        handle = int(raw.split('_')[1], 16)
    else:
        handle = int(raw or 0)
    if not handle:
        handle = int(widget.winId())
    hwnd = wintypes.HWND(handle)
    point, rect = wintypes.POINT(), wintypes.RECT()
    if not (user32.GetCursorPos(ctypes.byref(point))
            and user32.ScreenToClient(hwnd, ctypes.byref(point))
            and user32.GetClientRect(hwnd, ctypes.byref(rect))):
        return None
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None
    return float(point.x - rect.left), float(point.y - rect.top), float(width), float(height)


def project_points(renderer, points):
    """Vectorized equivalent of VTK WorldToDisplay, including the viewport."""
    camera = renderer.GetActiveCamera()
    matrix = camera.GetCompositeProjectionTransformMatrix(renderer.GetTiledAspectRatio(), 0., 1.)
    matrix = np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
    points = np.asarray(points, dtype=np.float64)
    clip = points @ matrix[:, :3].T + matrix[:, 3]
    valid = np.isfinite(clip).all(axis=1) & (clip[:, 3] > 1e-12)
    ndc = np.zeros((len(points), 3), dtype=np.float64)
    np.divide(clip[:, :3], clip[:, 3, None], out=ndc, where=valid[:, None])
    valid &= (ndc[:, 2] >= 0.) & (ndc[:, 2] <= 1.)
    # WorldToDisplay uses the fractional viewport bounds, not GetOrigin/Size's
    # integer-rounded bounds (which differ at odd window sizes).
    viewport = np.asarray(renderer.GetViewport(), dtype=float)
    window_size = np.asarray(renderer.GetRenderWindow().GetSize(), dtype=float)
    origin = viewport[:2] * window_size
    size = (viewport[2:] - viewport[:2]) * window_size
    display = origin + (ndc[:, :2] + 1.) * size / 2.
    return display, valid


def swept_brush_mask(points, start, end, radius, shape='circle'):
    """Exact circle capsule or axis-aligned square swept along a cursor segment."""
    points = np.asarray(points, dtype=float)
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    delta = end - start
    if shape == 'rectangle':
        # Intersect each point's square with the segment using slab intervals.
        lower, upper = np.zeros(len(points)), np.ones(len(points))
        valid = np.ones(len(points), dtype=bool)
        for axis in (0, 1):
            offset = points[:, axis] - start[axis]
            if abs(delta[axis]) < 1e-12:
                valid &= np.abs(offset) <= radius
            else:
                a, b = (offset - radius) / delta[axis], (offset + radius) / delta[axis]
                lower = np.maximum(lower, np.minimum(a, b))
                upper = np.minimum(upper, np.maximum(a, b))
        return valid & (lower <= upper)
    length_sq = float(delta @ delta)
    t = np.zeros(len(points)) if length_sq < 1e-12 else np.clip((points - start) @ delta / length_sq, 0., 1.)
    difference = points - (start + t[:, None] * delta)
    return np.einsum('ij,ij->i', difference, difference) <= radius * radius
