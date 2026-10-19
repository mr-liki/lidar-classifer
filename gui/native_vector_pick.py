"""Read-only 2-D vector picking for an attached SNT Vulkan plan view.

No VTK actors are consulted: SNT geometry shown in the native main viewport
lives in ``attachment['_render_entities']``. Never change those entities here.
The x-interval index is lazy, reusable, and invalidated on attachment changes.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import math
from pathlib import Path


def coords_of(entity):
    """Finite XY coordinates, using the same entity types/points as SNT GPU linework."""
    if not isinstance(entity, dict):
        return []
    kind = str(entity.get('type', '')).strip().lower()
    if kind == 'point':
        pos=entity.get('position')
        if pos is None:pos=entity.get('point')
        try:
            coords=[(float(pos[0]),float(pos[1]))]
            return coords if all(math.isfinite(v) for v in coords[0]) else []
        except (IndexError,ValueError,TypeError):
            return []
    if kind not in ('line', 'polyline', 'lwpolyline', 'polygon', '3dface'):
        return []
    raw = entity.get('points')
    if raw is None or len(raw) == 0:
        raw = entity.get('vertices')
    if raw is None:
        return []
    try:
        coords = [(float(p[0]), float(p[1])) for p in raw]
        if len(coords) < 2 or not all(math.isfinite(v) for p in coords for v in p):
            return []
    except (IndexError, ValueError, TypeError):
        return []
    if kind in ('polygon', '3dface') or entity.get('closed', False):
        if len(coords) >= 3 and coords[0] != coords[-1]:
            coords.append(coords[0])
    return coords


def segment_projection(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx*dx + dy*dy
    t = max(0.0, min(1.0, ((point[0]-a[0])*dx + (point[1]-a[1])*dy)/length2)) if length2 else 0.0
    q = (a[0]+t*dx, a[1]+t*dy)
    return math.hypot(point[0]-q[0],point[1]-q[1]), q, t


def nearest_segment(point, coords):
    best = None
    for si,(a,b) in enumerate(zip(coords,coords[1:])):
        dist, foot, fraction = segment_projection(point,a,b)
        if best is None or dist < best[0]:
            best = (dist,si,foot,fraction)
    return best


def segment_intersection(a,b,c,d):
    """Interior-interior segment crossing (not a coincident/endpoint snap)."""
    rx, ry = b[0]-a[0], b[1]-a[1]
    sx, sy = d[0]-c[0], d[1]-c[1]
    denominator = rx*sy-ry*sx
    scale = max(math.hypot(rx,ry)*math.hypot(sx,sy), 1.)
    if abs(denominator) <= scale*1e-12:
        return None
    dx,dy = c[0]-a[0], c[1]-a[1]
    t = (dx*sy-dy*sx)/denominator
    u = (dx*ry-dy*rx)/denominator
    if 1.e-10 < t < 1.-1.e-10 and 1.e-10 < u < 1.-1.e-10:
        return (a[0]+t*rx, a[1]+t*ry)
    return None


def polygon_centroid(coords):
    poly = list(coords)
    if len(poly)<3:
        return None
    if poly[0] != poly[-1]:
        poly.append(poly[0])
    twice_area=cx=cy=0.
    for a,b in zip(poly,poly[1:]):
        cross=a[0]*b[1]-b[0]*a[1]
        twice_area += cross
        cx += (a[0]+b[0])*cross
        cy += (a[1]+b[1])*cross
    if abs(twice_area)<1.e-16:
        return None
    return (cx/(3.*twice_area), cy/(3.*twice_area))


def offset_open_line(points, distance):
    """Constant perpendicular polyline offset, using true miter intersections.

    Produces a straightforward 2-D CAD offset for open non-degenerate lines;
    caps excessively sharp joins to avoid unbounded spikes.
    """
    if len(points)<2:
        return None
    coords=[]
    for p in points:
        p=tuple(map(float,p[:2]))
        if not coords or math.hypot(p[0]-coords[-1][0],p[1]-coords[-1][1])>1.e-12:
            coords.append(p)
    if len(coords)<2 or (len(coords)>=3 and coords[0] == coords[-1]):
        return None
    normal=[]
    for a,b in zip(coords,coords[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1]
        length=math.hypot(dx,dy)
        normal.append((-dy/length,dx/length))
    output=[]
    for i,p in enumerate(coords):
        if i==0:
            n=normal[0]; scale=distance
        elif i==len(coords)-1:
            n=normal[-1]; scale=distance
        else:
            a,b=normal[i-1],normal[i]
            # Offset-line intersection: n_i dot (intersection - p) = d.
            ux,uy=a[0]+b[0],a[1]+b[1]
            den=ux*b[0]+uy*b[1]
            if abs(den)<1.e-7:
                n=b; scale=distance
            else:
                n=(ux,uy); scale=distance/den
                if math.hypot(scale*n[0],scale*n[1]) > 8.*abs(distance):
                    n=b; scale=distance
        output.append((p[0]+scale*n[0],p[1]+scale*n[1]))
    return output


def _path_sample(pts, fraction):
    segments=[];total=0.
    for a,b in zip(pts,pts[1:]):
        length=math.hypot(b[0]-a[0],b[1]-a[1]);segments.append((a,b,length));total+=length
    if total<1.e-12:return tuple(pts[0])
    at=max(0.,min(1.,fraction))*total
    for a,b,length in segments:
        if at <= length or (a,b,length)==segments[-1]:
            t=max(0.,min(1.,at/length)) if length else 0.
            return (a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]))
        at-=length
    return tuple(pts[-1])


def _fractions(pts):
    cum=[0.];total=0.
    for a,b in zip(pts,pts[1:]):
        total+=math.hypot(b[0]-a[0],b[1]-a[1]);cum.append(total)
    return [v/total for v in cum] if total>1.e-12 else []


def centerline_between(a,b):
    """Arc-length aligned midpoint line; reverse opposite-direction input.

    Supports unequal vertex counts without truncating the longer side.
    Caller must provide two *open* lines (not polygon rings).
    """
    if len(a)<2 or len(b)<2 or a[0]==a[-1] or b[0]==b[-1]:
        return None
    a=[tuple(p[:2]) for p in a];b=[tuple(p[:2]) for p in b]
    forward=(math.dist(a[0],b[0])+math.dist(a[-1],b[-1]))
    reverse=(math.dist(a[0],b[-1])+math.dist(a[-1],b[0]))
    if reverse<forward:
        b=list(reversed(b))
    ta,tb=_fractions(a),_fractions(b)
    if not ta or not tb:return None
    fractions=sorted(set(round(t,12) for t in (ta+tb)))
    result=[]
    for t in fractions:
        p,q=_path_sample(a,t),_path_sample(b,t)
        mid=((p[0]+q[0])/2.,(p[1]+q[1])/2.)
        if not result or math.dist(result[-1],mid)>1.e-9:result.append(mid)
    return result if len(result)>=2 else None


class SntPickIndex:
    """X-interval index over the *rendered* SNT entity geometry.

    This makes ordinary pick queries logarithmic for local linework instead of
    traversing every attached entity on each mouse move. It deliberately keeps
    source geometry immutable. In very large scenes index construction may take
    noticeable time on the first pick; further queries reuse that index.
    """
    def __init__(self, attachments):
        records=[]
        for attachment in attachments:
            entities=attachment.get('_render_entities')
            if entities is None:
                entities=attachment.get('entities') or ()
            name=attachment.get('full_path') or attachment.get('filename') or ''
            key=('snt',str(Path(str(name))))
            for index,entity in enumerate(entities):
                coords=coords_of(entity)
                if len(coords)<1:continue
                lo_x=min(q[0] for q in coords);hi_x=max(q[0] for q in coords)
                lo_y=min(q[1] for q in coords);hi_y=max(q[1] for q in coords)
                records.append((lo_x,hi_x,lo_y,hi_y,key,
                                str(entity.get('layer','0') or '0'),index,entity))
        records.sort(key=lambda f:f[0])
        self.records=records
        self.min_x=[f[0] for f in records]
        self.prefix_max_x=[]
        maximum=-math.inf
        for f in records:
            maximum=max(maximum,f[1]);self.prefix_max_x.append(maximum)

    def nearby(self,point,tolerance,visible=None):
        x,y=point;left=bisect_left(self.prefix_max_x,x-tolerance)
        right=bisect_right(self.min_x,x+tolerance)
        for rec in self.records[left:right]:
            lo_x,hi_x,lo_y,hi_y,doc_key,layer,entity_idx,entity=rec
            if not (lo_y-tolerance<=y<=hi_y+tolerance):continue
            if visible is not None and not visible(doc_key,layer):continue
            yield {'source':'snt','coords':coords_of(entity), 'entity':entity,
                   'key':(doc_key,entity_idx), 'doc_key':doc_key,
                   'layer':layer,'type':str(entity.get('type','')).lower()}

    # Range-index lookup for the Select Block/Shape/Line tools.
    def query_bbox(self, min_x, max_x, min_y, max_y, visible=None):
        left=bisect_left(self.prefix_max_x,min_x)
        right=bisect_right(self.min_x,max_x)
        for rec in self.records[left:right]:
            lo_x,hi_x,lo_y,hi_y,doc_key,layer,entity_idx,entity=rec
            if hi_y<min_y or lo_y>max_y or hi_x<min_x:continue
            if visible is not None and not visible(doc_key,layer):continue
            yield {'source':'snt', 'coords':coords_of(entity),'entity':entity,
                   'key':(doc_key,entity_idx),'doc_key':doc_key,
                   'layer':layer,'type':str(entity.get('type','')).lower()}

