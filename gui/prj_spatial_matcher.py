"""Conservative, name-independent point-file matching for PRJ Identify.

The PRJ Block Identifier can identify a block from its PRJ boundary and then
look for the point cloud that belongs to it.  Matching by NAME only works when
the block label and the file name agree; in the field they often do not.

This module matches a point file to a block by GEOMETRY instead: the file's
own XY samples must lie inside the PRJ boundary polygon.  It is deliberately
strict - an ambiguous, unreadable or invalid input fails closed instead of
guessing, because a wrong match loads the wrong block.

No VTK and no rendering: only the LAS/LAZ headers and a few sampled points
are read, so millions of points are never loaded to identify a block.
"""
from __future__ import annotations

import numpy as np


def _polygon_bounds(boundary):
    """(xmin, ymin, xmax, ymax) of a boundary ring, or None when unusable."""
    try:
        points = np.asarray(boundary, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return None
    if not np.all(np.isfinite(points[:, :2])):
        return None
    xy = points[:, :2]
    return (float(xy[:, 0].min()), float(xy[:, 1].min()),
            float(xy[:, 0].max()), float(xy[:, 1].max()))


def _polygon_area(boundary):
    """Shoelace area of the boundary ring (0.0 when degenerate)."""
    points = np.asarray(boundary, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return 0.0
    xy = points[:, :2]
    x, y = xy[:, 0], xy[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _inside_fraction(polygon, x, y):
    """Fraction of the (x, y) samples that lie inside the polygon ring."""
    try:
        from shapely.geometry import Polygon
        from shapely import intersects_xy
        shape = Polygon(polygon)
        if not shape.is_valid or shape.is_empty:
            return 0.0
        return float(np.mean(intersects_xy(shape, np.asarray(x), np.asarray(y))))
    except Exception:
        return _inside_fraction_numpy(polygon, np.asarray(x), np.asarray(y))


def _inside_fraction_numpy(polygon, x, y):
    """Ray-casting point-in-polygon (fallback when shapely is unavailable)."""
    ring = np.asarray(polygon, dtype=np.float64)[:, :2]
    if ring.shape[0] < 3:
        return 0.0
    inside = np.zeros(len(x), dtype=bool)
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = float(ring[i, 0]), float(ring[i, 1])
        xj, yj = float(ring[j, 0]), float(ring[j, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            crosses = ((yi > y) != (yj > y))
            x_at = (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
        inside ^= crosses & (x < x_at)
        j = i
    return float(np.mean(inside)) if len(x) else 0.0


def find_spatial_lidar_file(paths, boundary, sample_size=1024):
    """Return the unique file whose distributed XY samples lie in the block.

    Headers only reject disjoint files; overlapping bounding boxes alone never
    establish a match. Ambiguous, unreadable, and invalid inputs fail closed.
    Sampling avoids loading millions of points just to identify a block.
    """
    import laspy

    if not boundary or _polygon_area(boundary) <= 0.0:
        return None
    bounds = _polygon_bounds(boundary)
    if bounds is None:
        return None
    xmin, ymin, xmax, ymax = bounds

    matches = []
    for path in dict.fromkeys(str(p) for p in paths or ()):
        try:
            with laspy.open(path) as reader:
                header = reader.header
                count = int(header.point_count)
                if not count or (header.maxs[0] < xmin or header.mins[0] > xmax
                                 or header.maxs[1] < ymin or header.mins[1] > ymax):
                    continue
                size = min(int(sample_size), count)
                # Require agreement at the beginning, middle, and end; a single
                # prefix sample can misidentify an uncut survey strip.
                starts = sorted({0, (count - size) // 2, count - size})
                fractions = []
                for start in starts:
                    reader.seek(start)
                    points = reader.read_points(size)
                    fractions.append(_inside_fraction(
                        boundary, np.asarray(points.x), np.asarray(points.y)))
                if min(fractions) >= 0.95:
                    matches.append(path)
        except Exception as exc:
            # An unreadable overlapping candidate prevents a safe unique match.
            print(f"[PRJ spatial] Cannot inspect {path}: {exc}")
            return None
    return matches[0] if len(matches) == 1 else None
