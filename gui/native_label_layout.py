"""World-sized CAD text projected with the same frame as its geometry."""
import math
from PySide6.QtGui import QTransform


def world_text_transform(project, x, y, height, rotation, cap_height):
    angle = math.radians(rotation)
    c, s = math.cos(angle), math.sin(angle)
    px, py = project(x, y)
    bx, by = project(x + height*c, y + height*s)
    dx, dy = project(x + height*s, y - height*c)
    divisor = max(float(cap_height), 1.)
    return QTransform((bx-px)/divisor, (by-py)/divisor,
                      (dx-px)/divisor, (dy-py)/divisor, px, py)
