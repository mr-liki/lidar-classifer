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
