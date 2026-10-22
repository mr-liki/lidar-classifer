"""Read-only SNT label -> editable native text overlay conversion.

Only a new Draw object is returned.  Neither the source entity nor its
attachment is modified.  World-space height/rotation are preserved so pasted
labels follow the Vulkan view's pan and zoom rather than being screen stickers.
"""
from __future__ import annotations

import math

from gui.native_selection_targets import text_anchor


def _color(value):
    if isinstance(value, str):
        return value if value.startswith('#') else '#00dcff'
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            values = [float(x) for x in value[:3]]
            if not all(math.isfinite(x) for x in values):
                raise ValueError('invalid color')
            if all(0 <= x <= 1 for x in values):
                values = [x * 255 for x in values]
            return '#%02x%02x%02x' % tuple(round(max(0, min(255, x))) for x in values)
        except (ValueError, TypeError):
            pass
    return '#00dcff'


def copied_label(entity, destination):
    """Return a new independently editable text label at world-XY *destination*.

    Position should already include group-offset when copying multiple labels.
    Invalid SNT metadata is safely ignored instead of producing malformed Draw
    geometry that could break the renderer or later edits.
    """
    if not isinstance(entity, dict):
        return None
    original = text_anchor(entity)
    if original is None:
        return None
    text = str(entity.get('text') or entity.get('value') or '')
    if not text:
        return None
    try:
        pos = tuple(float(v) for v in destination[:2])
    except (ValueError, TypeError, IndexError):
        return None
    if len(pos) != 2 or not all(math.isfinite(v) for v in pos):
        return None
    drawing = {
        'type': 'text', 'coords': [pos], 'text': text,
        'source': 'snt_text_copy', 'color': _color(entity.get('color')),
        'font_family': str(entity.get('font_family') or 'Arial'),
        'bold': bool(entity.get('bold', False)),
        'italic': bool(entity.get('italic', False)),
    }
    for name, default in (('rotation', 0.), ('height', 0.)):
        try:
            value = float(entity.get(name) or default)
        except (TypeError, ValueError):
            value = default
        drawing[name] = value if math.isfinite(value) else default
    try:
        size = float(entity.get('font_size') or 16)
        drawing['font_size'] = max(6, min(512, round(size))) if math.isfinite(size) else 16
    except (TypeError, ValueError):
        drawing['font_size'] = 16
    return drawing


def copy_group(features, target):
    """Paste all selected SNT labels with their original relative positions."""
    picked = []
    for feature in features:
        if feature.get('source') != 'snt':
            continue
        entity = feature.get('entity')
        anchor = text_anchor(entity) if isinstance(entity, dict) else None
        if anchor is not None:
            picked.append((entity, anchor))
    if not picked:
        return []
    origin = picked[0][1]
    try:
        point = (float(target[0]), float(target[1]))
        if not all(map(math.isfinite, point)):
            return []
    except (ValueError, TypeError, IndexError):
        return []
    copies = []
    for entity, anchor in picked:
        item = copied_label(entity, (point[0] + anchor[0] - origin[0],
                                     point[1] + anchor[1] - origin[1]))
        if item is not None:
            copies.append(item)
    return copies
