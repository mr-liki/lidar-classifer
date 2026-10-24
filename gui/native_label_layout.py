"""World-sized CAD text projected with the same frame as its geometry."""
import math
from functools import lru_cache
from PySide6.QtGui import QTransform


@lru_cache(maxsize=512)
def _glyph_contours(text):
    """Flatten font outlines once in font coordinates, independently of zoom."""
    from PySide6.QtGui import QFont, QFontMetricsF, QPainterPath
    font = QFont()
    font.setPixelSize(64)
    path = QPainterPath()
    path.addText(0., 0., font, text)
    contours = tuple(tuple((p.x(), p.y()) for p in polygon)
                     for polygon in path.toSubpathPolygons())
    return max(QFontMetricsF(font).capHeight(), 1.), contours


def native_text_entities(entity):
    """World-space glyph contours for the existing Vulkan line-strip pass.

    Returning None leaves unsupported/non-world-sized text on the UI overlay.
    Font extraction runs at attachment time, never during camera navigation.
    """
    height = float(entity.get('height', 0.) or 0.)
    position = entity.get('position')
    if height <= 0. or position is None or len(position) < 2:
        return None
    text = str(entity.get('text', ''))
    cap_height, contours = _glyph_contours(text)
    angle = math.radians(float(entity.get('rotation', 0.) or 0.))
    c, s = math.cos(angle), math.sin(angle)
    scale = height / cap_height
    x, y = float(position[0]), float(position[1])
    z = float(position[2]) if len(position) > 2 else 0.
    return [dict(type='polyline', layer=entity.get('layer', '0'),
                 color=entity.get('color') or (255, 255, 255), closed=True,
                 points=[(x + scale * (gx*c + gy*s),
                          y + scale * (gx*s - gy*c), z) for gx, gy in ring])
            for ring in contours if len(ring) >= 2]


def world_text_transform(project, x, y, height, rotation, cap_height):
    angle = math.radians(rotation)
    c, s = math.cos(angle), math.sin(angle)
    px, py = project(x, y)
    bx, by = project(x + height*c, y + height*s)
    dx, dy = project(x + height*s, y - height*c)
    divisor = max(float(cap_height), 1.)
    return QTransform((bx-px)/divisor, (by-py)/divisor,
                      (dx-px)/divisor, (dy-py)/divisor, px, py)


def native_draw_text_layout(project, anchor, entity, base_font=None):
    """One font and one transform for painting AND hit-testing native Draw text.

    A new native label has a positive CAD-world ``height`` frozen when it is
    placed, so zoom changes its screen geometry exactly like nearby linework.
    Older drawings without height retain their previous screen-sized behavior
    until edited (or replaced).  Imported SNT text has its own rendering path.
    """
    from PySide6.QtGui import QFont, QFontMetricsF

    font = QFont(base_font) if base_font is not None else QFont()
    family = str(entity.get('font_family') or 'Arial')
    font.setFamily(family)
    font.setBold(bool(entity.get('bold', False)))
    font.setItalic(bool(entity.get('italic', False)))
    try:
        points = max(1., min(999., float(entity.get('font_size') or 75.)))
    except (ValueError, TypeError):
        points = 75.
    try:
        height = float(entity.get('height') or 0.)
    except (ValueError, TypeError):
        height = 0.
    rotation = float(entity.get('rotation', 0.) or 0.)
    if math.isfinite(height) and height > 0.:
        # The same reference-glyph size is used for all zooms.  Never use
        # QFont.pointSize() after setPixelSize(): it is -1 by Qt convention.
        font.setPixelSize(64)
        cap_height = max(1., QFontMetricsF(font).capHeight())
        transform = world_text_transform(project, anchor[0], anchor[1],
                                         height, rotation, cap_height)
    else:
        # Backward compatibility: older native text stored font_size in pixels,
        # not CAD-world height. Keep its appearance until the first edit.
        font.setPixelSize(max(6, min(512, int(points))))
        px, py = project(anchor[0], anchor[1])
        transform = QTransform().translate(px, py).rotate(-rotation)
    return font, transform


def native_draw_text_height(entity, world_per_logical_pixel):
    """Capture CAD letter height once at placement, independently of later zoom.

    Font size is a typographic point size; QFontMetricsF provides the actual
    cap height at current logical-DPI.  Existing labels are scaled by their
    original height when subsequently edited, not by current zoom.
    """
    from PySide6.QtGui import QFont, QFontMetricsF

    try:
        font_size = max(1., min(999., float(entity.get('font_size') or 75.)))
    except (TypeError, ValueError):
        font_size = 75.
    font = QFont(str(entity.get('font_family') or 'Arial'))
    font.setPointSizeF(font_size)
    font.setBold(bool(entity.get('bold', False)))
    font.setItalic(bool(entity.get('italic', False)))
    px = max(float(QFontMetricsF(font).capHeight()), 1.)
    return px * max(float(world_per_logical_pixel), 1.e-12)
