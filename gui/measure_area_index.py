"""Block and grid polygon index for area measurements - pure data, no VTK.

Shared by the VTK MeasurementTool (which inherits it) and the native Vulkan
NativeMeasure overlay: block polygons come from TerraScan PRJ files next to
attached DXF/DWG overlays, PRJ-matched SNT blocks and closed digitizer
drawings; grid cells come from app.snt_block_polygons.
"""
from pathlib import Path

import numpy as np


class AreaIndex:
    def __init__(self, app):
        self.app = app
        self._block_boundary_cache = {}
        self._block_boundary_index = []

    def _is_valid_world_point(self, pos):
        """Return True when a picked world position is usable for measurement."""
        if pos is None:
            return False
        try:
            arr = np.asarray(pos, dtype=np.float64).reshape(-1)
        except Exception:
            return False
        return arr.size >= 3 and np.all(np.isfinite(arr[:3]))


    def _calculate_polygon_area(self, points):
        """
        Calculate the area of a polygon using the Shoelace formula (2D projection).
        Implements Gauss's area formula as specified by the user.
        """
        if len(points) < 3:
            return 0.0

        # 1. Use a local relative coordinate system to maintain high precision
        # (Subtract first point from all points to avoid large coordinate issues)
        origin = np.array(points[0])
        local_pts = [(p[0] - origin[0], p[1] - origin[1]) for p in points]

        # 2. Apply Shoelace formula: A = 0.5 * |sum(x_i*y_{i+1} - x_{i+1}*y_i)|
        area = 0.0
        n = len(local_pts)

        for i in range(n):
            j = (i + 1) % n
            # (x_i * y_{j}) - (x_j * y_i)
            area += local_pts[i][0] * local_pts[j][1]
            area -= local_pts[j][0] * local_pts[i][1]

        return abs(area) / 2.0

    def _normalize_block_label(self, label):
        """Normalize labels so PRJ block names and DXF text labels can be matched."""
        return "".join(ch.lower() for ch in str(label or "") if ch.isalnum())

    def _normalize_block_polygon(self, coords):
        """Drop duplicate closing vertices and malformed points from a block polygon."""
        normalized = []
        for coord in coords or []:
            try:
                x = float(coord[0])
                y = float(coord[1])
            except Exception:
                continue
            if normalized and abs(normalized[-1][0] - x) < 1e-9 and abs(normalized[-1][1] - y) < 1e-9:
                continue
            normalized.append((x, y))

        if len(normalized) >= 2:
            first_x, first_y = normalized[0]
            last_x, last_y = normalized[-1]
            if abs(first_x - last_x) < 1e-9 and abs(first_y - last_y) < 1e-9:
                normalized.pop()
        return normalized



    def _calculate_polygon_centroid_2d(self, points):
        """Return a stable centroid for a 2D polygon."""
        if not points:
            return (0.0, 0.0)
        if len(points) < 3:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))

        signed_area = 0.0
        cx = 0.0
        cy = 0.0
        for i in range(len(points)):
            j = (i + 1) % len(points)
            cross = points[i][0] * points[j][1] - points[j][0] * points[i][1]
            signed_area += cross
            cx += (points[i][0] + points[j][0]) * cross
            cy += (points[i][1] + points[j][1]) * cross

        if abs(signed_area) < 1e-9:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))

        signed_area *= 0.5
        scale = 1.0 / (6.0 * signed_area)
        return (float(cx * scale), float(cy * scale))

    def _calculate_polygon_area_2d(self, points):
        """
        Backward-compatible 2D area helper.
        Keeps legacy block-area call sites working while delegating to the
        shared Shoelace implementation.
        """
        return float(self._calculate_polygon_area(points))

    def _point_in_polygon_2d(self, point_xy, polygon):
        """Return True when the XY point lies inside or very near the polygon."""
        if len(polygon) < 3:
            return False

        x = float(point_xy[0])
        y = float(point_xy[1])
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]

            edge_dx = x2 - x1
            edge_dy = y2 - y1
            edge_len_sq = edge_dx * edge_dx + edge_dy * edge_dy
            if edge_len_sq > 1e-12:
                t = ((x - x1) * edge_dx + (y - y1) * edge_dy) / edge_len_sq
                t = max(0.0, min(1.0, t))
                nearest_x = x1 + t * edge_dx
                nearest_y = y1 + t * edge_dy
                if (nearest_x - x) ** 2 + (nearest_y - y) ** 2 < 1e-6:
                    return True

            intersects = ((y1 > y) != (y2 > y))
            if intersects:
                cross_x = (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
                if x <= cross_x:
                    inside = not inside
        return inside

    def _iter_block_project_files(self):
        """Yield PRJ files that can describe block boundaries for attached overlays."""
        candidates = []
        seen = set()

        loaded_file = getattr(self.app, "loaded_file", None)
        if loaded_file:
            loaded_path = Path(str(loaded_file))
            if loaded_path.suffix.lower() == ".prj" and loaded_path.exists():
                key = str(loaded_path.resolve()).lower()
                if key not in seen:
                    seen.add(key)
                    candidates.append(loaded_path)

        for store_name in ("dxf_actors", "dwg_actors"):
            for overlay in getattr(self.app, store_name, []) or []:
                full_path = overlay.get("full_path") or overlay.get("filename")
                if not full_path:
                    continue
                base_path = Path(str(full_path))
                for suffix in (".prj", ".PRJ"):
                    prj_path = base_path.with_suffix(suffix)
                    if not prj_path.exists():
                        continue
                    try:
                        key = str(prj_path.resolve()).lower()
                    except Exception:
                        key = str(prj_path).lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(prj_path)

        return candidates

    def _parse_terrascan_prj_blocks(self, prj_path):
        """Parse TerraScan PRJ block boundaries into searchable polygons."""
        blocks = []
        try:
            lines = Path(prj_path).read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as exc:
            print(f"⚠️ Failed to read PRJ block file {prj_path}: {exc}")
            return blocks

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line.startswith("Block "):
                i += 1
                continue

            block_file = line.replace("Block ", "", 1).strip()
            block_label = Path(block_file).stem or block_file
            coords = []
            j = i + 1
            while j < len(lines):
                coord_line = lines[j].strip()
                if not coord_line:
                    j += 1
                    continue
                if coord_line.startswith("Block "):
                    break

                parts = coord_line.split()
                if len(parts) >= 2:
                    try:
                        coords.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
                j += 1

            polygon = self._normalize_block_polygon(coords)
            if len(polygon) >= 3:
                area = self._calculate_polygon_area_2d(polygon)
                centroid = self._calculate_polygon_centroid_2d(polygon)
                xs = [pt[0] for pt in polygon]
                ys = [pt[1] for pt in polygon]
                blocks.append({
                    "label": block_label,
                    "label_key": self._normalize_block_label(block_label),
                    "points": polygon,
                    "area": area,
                    "centroid": centroid,
                    "bbox": (min(xs), max(xs), min(ys), max(ys)),
                    "source_type": "prj",
                    "source_prj": str(prj_path),
                })

            i = j

        return blocks

    def _parse_attached_snt_blocks(self):
        """Convert attached SNT block polygons into the measurement block index.

        Includes legacy PRJ matches and current PRJ-authority records.
        Raw grid outlines (snt_grid) and unnamed BL-layer polygons (snt_text) are
        excluded — they are reference geometry, not actual file blocks.
        """
        blocks = []
        for idx, entry in enumerate(getattr(self.app, "snt_block_polygons", []) or []):
            # Only PRJ-matched blocks have file definitions — skip all others.
            if entry.get("source") not in {"prj", "prj_authority"}:
                continue

            coords = (
                entry.get("points_2d")
                or entry.get("points")
                or entry.get("vertices")
                or entry.get("polygon")
                or []
            )
            polygon = self._normalize_block_polygon(coords)
            if len(polygon) < 3:
                continue

            source_snt = (
                entry.get("snt_filename")
                or entry.get("source_snt")
                or entry.get("filename")
                or ""
            )
            label = (
                entry.get("grid_name")
                or entry.get("block_name")
                or entry.get("label")
                or ""
            )
            if not label:
                source_name = Path(str(source_snt)).stem if source_snt else "SNT block"
                label = f"{source_name} #{idx + 1}"

            area = self._calculate_polygon_area_2d(polygon)
            centroid = self._calculate_polygon_centroid_2d(polygon)
            xs = [pt[0] for pt in polygon]
            ys = [pt[1] for pt in polygon]
            blocks.append({
                "label": str(label),
                "label_key": self._normalize_block_label(label),
                "points": polygon,
                "area": area,
                "centroid": centroid,
                "bbox": (min(xs), max(xs), min(ys), max(ys)),
                "source_type": "snt",
                "source_snt": str(source_snt) if source_snt else None,
                "source_prj": entry.get("prj_path"),
            })

        return blocks

    def _parse_attached_snt_grids(self):
        """Adapt the live SNT grid index into measurement entries.

        Current SNT attachments replace raw ``snt_grid`` cells with
        ``prj_authority`` records whenever an adjacent PRJ supplies the exact,
        authoritative boundary.  Files without PRJ data continue to expose
        ``snt_grid`` records.  Both are grid cells for measurement purposes.
        The labelled BBOX tiles (``app.snt_grid_tiles``) are the square grid
        cells when real blocks exist; picking takes the smallest containing
        cell, so a tile wins over the PRJ block around it.
        """
        grids = []
        entries = list(getattr(self.app, "snt_block_polygons", []) or [])
        entries += list(getattr(self.app, "snt_grid_tiles", []) or [])
        for idx, entry in enumerate(entries):
            index_source = entry.get("source")
            if index_source not in {"snt_grid", "prj_authority", "snt_bbox"}:
                continue

            coords = (
                entry.get("points_2d")
                or entry.get("points")
                or entry.get("vertices")
                or entry.get("polygon")
                or []
            )
            polygon = self._normalize_block_polygon(coords)
            if len(polygon) < 3:
                continue

            source_snt = entry.get("snt_filename") or entry.get("source_snt") or ""
            label = entry.get("grid_name") or entry.get("label") or f"Grid #{idx + 1}"
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            grids.append({
                "label": str(label),
                "label_key": self._normalize_block_label(label),
                "points": polygon,
                "area": self._calculate_polygon_area_2d(polygon),
                "centroid": self._calculate_polygon_centroid_2d(polygon),
                "bbox": (min(xs), max(xs), min(ys), max(ys)),
                "source_type": "snt_grid",
                "source_snt": str(source_snt) if source_snt else None,
                "source_layer": entry.get("poly_layer") or entry.get("layer"),
                "index_source": index_source,
                "source_prj": entry.get("prj_path"),
            })
        return grids

    def _load_grid_boundaries(self):
        """Return live grid geometry without changing the Block boundary cache."""
        return self._parse_attached_snt_grids()

    def _parse_digitizer_drawings(self):
        """Convert digitizer drawn closed shapes into block-like polygon entries."""
        blocks = []
        digitizer = getattr(self.app, 'digitizer', None)
        if digitizer is None:
            return blocks

        CLOSED_TYPES = {'polygon', 'polyline', 'rectangle', 'freehand', 'orthopolygon'}

        for idx, drawing in enumerate(getattr(digitizer, 'drawings', []) or []):
            draw_type = drawing.get('type', '')
            if draw_type not in CLOSED_TYPES:
                continue

            coords = drawing.get('coords', [])
            polygon_2d = []
            for c in coords:
                try:
                    polygon_2d.append((float(c[0]), float(c[1])))
                except (IndexError, TypeError, ValueError):
                    continue

            polygon_2d = self._normalize_block_polygon(polygon_2d)
            if len(polygon_2d) < 3:
                continue

            area = self._calculate_polygon_area_2d(polygon_2d)
            centroid = self._calculate_polygon_centroid_2d(polygon_2d)
            xs = [pt[0] for pt in polygon_2d]
            ys = [pt[1] for pt in polygon_2d]

            layer = drawing.get('layer') or ''
            label = f"{draw_type.capitalize()} #{idx + 1}" + (f" [{layer}]" if layer else "")

            blocks.append({
                'label': label,
                'label_key': self._normalize_block_label(label),
                'points': polygon_2d,
                'area': area,
                'centroid': centroid,
                'bbox': (min(xs), max(xs), min(ys), max(ys)),
                'source_type': 'digitizer',
                'drawing_type': draw_type,
            })

        return blocks

    def _load_block_boundaries(self, force=False):
        """Load block polygons from PRJ files, SNT overlays, and digitizer drawings."""
        if force:
            self._block_boundary_cache = {}
            self._block_boundary_index = []

        blocks = []
        for prj_path in self._iter_block_project_files():
            cache_key = str(prj_path)
            if cache_key not in self._block_boundary_cache:
                self._block_boundary_cache[cache_key] = self._parse_terrascan_prj_blocks(prj_path)
            blocks.extend(self._block_boundary_cache.get(cache_key, []))

        blocks.extend(self._parse_attached_snt_blocks())
        blocks.extend(self._parse_digitizer_drawings())

        self._block_boundary_index = blocks
        return self._block_boundary_index

    def _find_block_by_label(self, label):
        """Find a parsed block whose label matches the clicked grid label."""
        label_key = self._normalize_block_label(label)
        if not label_key:
            return None

        for block in self._load_block_boundaries():
            block_key = block.get("label_key", "")
            if not block_key:
                continue
            if block_key == label_key or block_key in label_key or label_key in block_key:
                return block
        return None

    def _find_block_at_world_point(self, world_point):
        """Find the block polygon containing the clicked world XY location."""
        if not self._is_valid_world_point(world_point):
            return None

        x = float(world_point[0])
        y = float(world_point[1])

        for block in self._load_block_boundaries():
            xmin, xmax, ymin, ymax = block["bbox"]
            if x < xmin or x > xmax or y < ymin or y > ymax:
                continue
            if self._point_in_polygon_2d((x, y), block["points"]):
                return block
        return None

    def _find_grid_by_label(self, label):
        """Resolve a grid label without mixing in PRJ/block definitions."""
        label_key = self._normalize_block_label(label)
        if not label_key:
            return None
        for grid in self._load_grid_boundaries():
            grid_key = grid.get("label_key", "")
            if grid_key and (
                grid_key == label_key
                or grid_key in label_key
                or label_key in grid_key
            ):
                return grid
        return None

    def _find_grid_at_world_point(self, world_point):
        """Return the smallest grid cell containing the clicked world XY."""
        if not self._is_valid_world_point(world_point):
            return None
        x, y = float(world_point[0]), float(world_point[1])
        candidates = []
        for grid in self._load_grid_boundaries():
            xmin, xmax, ymin, ymax = grid["bbox"]
            if xmin <= x <= xmax and ymin <= y <= ymax:
                if self._point_in_polygon_2d((x, y), grid["points"]):
                    candidates.append(grid)
        if not candidates:
            return None
        return min(candidates, key=lambda item: float(item.get("area", float("inf"))))
