"""Geometry-only helpers for Vulkan plan-view selection.

No VTK or Qt imports: tests can exercise selection without opening a renderer.
All calculations use the same XY world coordinates as native_draw.MainCanvas.
"""
from __future__ import annotations

import math


def point_in_polygon(point, vertices):
    if len(vertices) < 3:
        return False
    x, y = point
    inside = False
    for a, b in zip(vertices, list(vertices[1:]) + [vertices[0]]):
        if (a[1] > y) != (b[1] > y):
            cut = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < cut:
                inside = not inside
    return inside


def segment_crosses(a, b, c, d):
    """Include endpoint/tangency intersections; unlike snap, this is selection."""
    def orient(u, v, w):
        return (v[0]-u[0])*(w[1]-u[1]) - (v[1]-u[1])*(w[0]-u[0])
    def on_segment(u, v, w):
        return (min(u[0], v[0])-1e-10 <= w[0] <= max(u[0], v[0])+1e-10
                and min(u[1], v[1])-1e-10 <= w[1] <= max(u[1], v[1])+1e-10)
    ab_c, ab_d = orient(a, b, c), orient(a, b, d)
    cd_a, cd_b = orient(c, d, a), orient(c, d, b)
    if ab_c * ab_d < 0 and cd_a * cd_b < 0:
        return True
    return any(abs(o) <= 1e-9 and on_segment(u, v, w) for o, u, v, w in (
        (ab_c, a, b, c), (ab_d, a, b, d),
        (cd_a, c, d, a), (cd_b, c, d, b)))


def rect_selects(coords, a, b, inside=False):
    if not coords:
        return False
    left, right = sorted((a[0], b[0])); bottom, top = sorted((a[1], b[1]))
    def within(p):
        return left <= p[0] <= right and bottom <= p[1] <= top
    if inside:
        return all(within(p) for p in coords)
    if any(within(p) for p in coords):
        return True
    corners = [(left,bottom), (right,bottom), (right,top), (left,top)]
    edges = list(zip(corners, corners[1:] + corners[:1]))
    if any(segment_crosses(p,q,a,b) for p,q in zip(coords, coords[1:]) for a,b in edges):
        return True
    return len(coords) >= 4 and coords[0] == coords[-1] and point_in_polygon(corners[0],coords)


def polygon_selects(coords, region, inside=False):
    if not coords or len(region) < 3:
        return False
    within = lambda p: point_in_polygon(p,region) or any(
        point_on_segment(p,a,b) for a,b in zip(region,region[1:]+region[:1]))
    if inside:
        return all(within(p) for p in coords)
    if any(within(p) for p in coords):
        return True
    if any(segment_crosses(a,b,c,d) for a,b in zip(coords,coords[1:])
           for c,d in zip(region,region[1:]+region[:1])):
        return True
    return len(coords)>=4 and coords[0]==coords[-1] and point_in_polygon(region[0],coords)


def point_on_segment(p,a,b):
    vx,vy=b[0]-a[0],b[1]-a[1]
    scale=math.hypot(vx,vy)
    if scale <= 1e-15:
        return math.dist(p,a)<1e-10
    return (abs((p[0]-a[0])*vy-(p[1]-a[1])*vx) <= 1e-9*max(1,scale)
            and min(a[0],b[0])-1e-9<=p[0]<=max(a[0],b[0])+1e-9
            and min(a[1],b[1])-1e-9<=p[1]<=max(a[1],b[1])+1e-9)


def line_selects(coords, a, b, tol=0):
    if len(coords) < 2:
        return bool(coords and point_on_segment(coords[0],a,b))
    return any(segment_crosses(a,b,c,d) for c,d in zip(coords,coords[1:]))


def translate(coords, dx, dy):
    return [(float(p[0])+dx,float(p[1])+dy) for p in coords]


def rotate(coords, degrees, pivot):
    theta=math.radians(degrees); c,s=math.cos(theta),math.sin(theta)
    x0,y0=pivot
    return [(x0 + c*(p[0]-x0)-s*(p[1]-y0),
             y0 + s*(p[0]-x0)+c*(p[1]-y0)) for p in coords]


def mirror(coords, axis, pivot):
    x0,y0=pivot
    if axis=='vertical':return [(2*x0-p[0],p[1]) for p in coords]
    if axis=='horizontal':return [(p[0],2*y0-p[1]) for p in coords]
    raise ValueError('Mirror axis must be vertical or horizontal')
