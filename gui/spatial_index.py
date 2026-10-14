"""Shared point-cloud spatial indexing used by classification and section tools.

Phase 2 goals
-------------
* One index per canonical ``xyz`` array.  No independent Main-brush / section
  / rectangle indexes for the same dataset.
* Large datasets use an XY CSR-style uniform grid so plan-view selection cost
  is proportional to intersecting cells/candidates rather than total points.
* Canonical XYZ is never converted or rewritten.  The index stores only point
  ids and grid metadata.
* 3-D KD-tree compatibility remains available lazily for legacy callers that
  explicitly access ``.tree`` or need nearest-neighbour / 3-D radius queries.

The large-cloud grid is intentionally display/data neutral: it returns canonical
point ids and therefore does not alter classification semantics, LAS ordering,
undo/redo, cross sections, shading, or save/export behaviour.
"""

from __future__ import annotations

import threading
import time
import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


_INDEX_CACHE = {
    "index": None,
    "point_id": None,
    "n_points": 0,
    "fp": None,
}
_INDEX_LOCK = threading.RLock()
_BUILD_LOCK = threading.Lock()
_CACHE_GENERATION = 0
_BUILD_STATE = {
    "point_id": None,
    "thread": None,
}

@dataclass(frozen=True)
class RenderTileDescriptor:
    tile_id: int
    ix0: int
    ix1: int
    iy0: int
    iy1: int
    point_count: int
    bounds: tuple



def _fingerprint(xyz: np.ndarray):
    n = int(len(xyz))
    if n <= 0:
        return (0, 0.0, 0.0, 0.0, 0.0)
    mid = n // 2
    try:
        return (
            n,
            float(xyz[0, 0]),
            float(xyz[mid, 0]),
            float(xyz[-1, 0]),
            float(xyz[mid, 1]),
        )
    except Exception:
        return (n, id(xyz))


def _cache_matches(xyz: np.ndarray) -> bool:
    return (
        _INDEX_CACHE["index"] is not None
        and _INDEX_CACHE["point_id"] == id(xyz)
        and _INDEX_CACHE["n_points"] == len(xyz)
        and _INDEX_CACHE.get("fp") == _fingerprint(xyz)
    )


def invalidate_cache() -> None:
    """Drop the dataset-owned spatial index cache without waiting on a build."""
    global _CACHE_GENERATION
    with _INDEX_LOCK:
        _CACHE_GENERATION += 1
        _INDEX_CACHE.update(index=None, point_id=None, n_points=0, fp=None)
        _BUILD_STATE["point_id"] = None
        _BUILD_STATE["thread"] = None


def get_cached_index(xyz: np.ndarray):
    """Return an already-built index for *xyz* without triggering a build."""
    with _INDEX_LOCK:
        return _INDEX_CACHE["index"] if _cache_matches(xyz) else None


def get_or_build_index(xyz: np.ndarray, method: str = "auto") -> "SpatialIndex":
    """Return the single shared index for the current canonical XYZ array.

    The expensive build is serialized by ``_BUILD_LOCK`` but does not hold the
    cache-state lock.  Therefore loading/clearing a project can invalidate an
    in-flight old build immediately instead of freezing until it finishes.
    """
    with _INDEX_LOCK:
        if _cache_matches(xyz):
            return _INDEX_CACHE["index"]

    with _BUILD_LOCK:
        with _INDEX_LOCK:
            if _cache_matches(xyz):
                return _INDEX_CACHE["index"]
            generation = _CACHE_GENERATION

        started = time.perf_counter()
        idx = build_spatial_index_auto(xyz, method)
        elapsed = (time.perf_counter() - started) * 1000.0

        installed = False
        with _INDEX_LOCK:
            if generation == _CACHE_GENERATION:
                _INDEX_CACHE["index"] = idx
                _INDEX_CACHE["point_id"] = id(xyz)
                _INDEX_CACHE["n_points"] = len(xyz)
                _INDEX_CACHE["fp"] = _fingerprint(xyz)
                installed = True

        print(
            "PHASE2_SPATIAL_INDEX status=ready "
            f"backend={idx.method} points={len(xyz):,} "
            f"build_ms={elapsed:.1f} memory_mb={idx.memory_bytes / (1024*1024):.1f} "
            f"installed={int(installed)}"
        )
        return idx


def schedule_shared_index_build(app, xyz: np.ndarray, *, name: str = "NakshaSharedSpatialIndex"):
    """Build the shared index once on a daemon thread and install on ``app``.

    If the same dataset is already indexed, installation is immediate.  If a
    build for this exact array is already running, no second thread is started.
    """
    cached = get_cached_index(xyz)
    if cached is not None:
        app.spatial_index = cached
        return cached

    # A new canonical XYZ array supersedes the previous dataset index. Drop the
    # old cache before allocating the new order array so both cannot coexist.
    with _INDEX_LOCK:
        old_id = _INDEX_CACHE.get("point_id")
    if old_id is not None and old_id != id(xyz):
        invalidate_cache()

    point_id = id(xyz)
    with _INDEX_LOCK:
        running = _BUILD_STATE.get("thread")
        if (
            _BUILD_STATE.get("point_id") == point_id
            and running is not None
            and running.is_alive()
        ):
            return None

        def _worker():
            try:
                idx = get_or_build_index(xyz)
                current = getattr(app, "data", None) or {}
                if id(current.get("xyz")) == point_id:
                    app.spatial_index = idx
                    print("PHASE2_SPATIAL_INDEX status=installed shared=1")
                else:
                    print("PHASE2_SPATIAL_INDEX status=discarded reason=data_replaced")
            except Exception as exc:
                print(f"PHASE2_SPATIAL_INDEX status=failed error={exc}")
            finally:
                with _INDEX_LOCK:
                    if _BUILD_STATE.get("point_id") == point_id:
                        _BUILD_STATE["thread"] = None

        t = threading.Thread(target=_worker, name=name, daemon=True)
        _BUILD_STATE["point_id"] = point_id
        _BUILD_STATE["thread"] = t
        t.start()
        return None



_GRID_CACHE_SCHEMA = 1


def _phase2c_grid_cache_dir(points) -> Optional[Path]:
    try:
        from gui.point_cache import cache_dir_from_array
        return cache_dir_from_array(points)
    except Exception:
        return None


def _phase2c_grid_meta_path(cache_dir: Path) -> Path:
    return cache_dir / "spatial_grid_meta.json"


def _phase2c_try_load_grid(points, target_points_per_cell: int):
    cdir = _phase2c_grid_cache_dir(points)
    if cdir is None:
        return None
    meta_path = _phase2c_grid_meta_path(cdir)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("schema", -1)) != _GRID_CACHE_SCHEMA:
            return None
        if int(meta.get("n_points", -1)) != int(len(points)):
            return None
        if int(meta.get("target_points_per_cell", -1)) != int(target_points_per_cell):
            return None
        if tuple(meta.get("fingerprint") or ()) != tuple(_fingerprint(points)):
            return None
        order = np.load(cdir / meta["point_order"], mmap_mode="r", allow_pickle=False)
        offsets = np.load(cdir / meta["cell_offsets"], mmap_mode="r", allow_pickle=False)
        prefix = np.load(cdir / meta["prefix"], mmap_mode="r", allow_pickle=False)
        if len(order) != len(points):
            return None
        return {
            "cache_dir": cdir,
            "x_min": float(meta["x_min"]), "x_max": float(meta["x_max"]),
            "y_min": float(meta["y_min"]), "y_max": float(meta["y_max"]),
            "nx": int(meta["nx"]), "ny": int(meta["ny"]),
            "cell_w": float(meta["cell_w"]), "cell_h": float(meta["cell_h"]),
            "point_order": order, "cell_offsets": offsets, "prefix": prefix,
        }
    except Exception:
        return None


def _phase2c_save_grid(index) -> None:
    cdir = _phase2c_grid_cache_dir(index.points)
    if cdir is None:
        return
    t0 = time.perf_counter()
    try:
        files = {
            "point_order": "spatial_point_order.npy",
            "cell_offsets": "spatial_cell_offsets.npy",
            "prefix": "spatial_prefix.npy",
        }
        for key, fn in files.items():
            arr = index.point_order if key == "point_order" else (
                index.cell_offsets if key == "cell_offsets" else index._cell_count_prefix
            )
            path = cdir / fn
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "wb") as fh:
                np.save(fh, np.asarray(arr), allow_pickle=False)
            os.replace(tmp, path)
        meta = {
            "schema": _GRID_CACHE_SCHEMA,
            "n_points": int(len(index.points)),
            "target_points_per_cell": int(index.target_points_per_cell),
            "fingerprint": list(_fingerprint(index.points)),
            "x_min": float(index.x_min), "x_max": float(index.x_max),
            "y_min": float(index.y_min), "y_max": float(index.y_max),
            "nx": int(index.nx), "ny": int(index.ny),
            "cell_w": float(index.cell_w), "cell_h": float(index.cell_h),
            **files,
        }
        tmp_meta = _phase2c_grid_meta_path(cdir).with_suffix(".json.tmp")
        tmp_meta.write_text(json.dumps(meta, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp_meta, _phase2c_grid_meta_path(cdir))
        print(
            "PHASE2C_SPATIAL_CACHE status=built "
            f"points={len(index.points):,} build_ms={(time.perf_counter()-t0)*1000.0:.1f}"
        )
    except Exception as exc:
        print(f"PHASE2C_SPATIAL_CACHE status=build_failed error={exc}")


class SpatialIndex:
    """Compatibility spatial index with a large-cloud XY grid backend.

    ``grid2d`` is the Phase-2 backend.  Legacy 3-D callers continue to work:
    ``.tree`` is built lazily only when a 3-D operation actually asks for it.
    """

    def __init__(
        self,
        points: np.ndarray,
        method: str = "grid2d",
        grid_size: int = 50,
        *,
        target_points_per_cell: int = 2048,
    ):
        self.points = points
        self.method = method
        self.grid_size = int(grid_size)
        self.target_points_per_cell = max(256, int(target_points_per_cell))
        self._tree = None
        self._tree_lock = threading.Lock()
        self.memory_bytes = 0

        if method == "kdtree":
            self._build_kdtree()
        elif method in ("grid", "grid2d"):
            self.method = "grid2d"
            self._build_grid2d()
        elif method == "octree":
            self._build_kdtree()  # exact compatibility without recursive copies
            self.method = "kdtree"
        else:
            raise ValueError(f"Unknown spatial index method: {method}")

    @property
    def tree(self):
        """Lazy 3-D KD-tree for legacy nearest-neighbour callers."""
        if self._tree is None:
            with self._tree_lock:
                if self._tree is None:
                    t0 = time.perf_counter()
                    print(
                        "PHASE2_SPATIAL_INDEX status=lazy_3d_build "
                        f"points={len(self.points):,}"
                    )
                    self._tree = cKDTree(
                        self.points, leafsize=48, balanced_tree=True, compact_nodes=True
                    )
                    print(
                        "PHASE2_SPATIAL_INDEX status=lazy_3d_ready "
                        f"build_ms={(time.perf_counter()-t0)*1000.0:.1f}"
                    )
        return self._tree

    @property
    def valid(self) -> bool:
        """Compatibility with the historical SpatialGridIndex.valid flag."""
        try:
            if self.points is None:
                return False
            if self.method == "grid2d":
                return (hasattr(self, "point_order") and hasattr(self, "cell_offsets")
                        and len(self.cell_offsets) == int(self.nx) * int(self.ny) + 1)
            return self._tree is not None
        except Exception:
            return False

    def _build_kdtree(self):
        self._tree = cKDTree(
            self.points, leafsize=48, balanced_tree=True, compact_nodes=True
        )
        self.memory_bytes = 0  # SciPy owns internal allocation; don't guess here.

    def _build_grid2d(self):
        pts = self.points
        n = int(len(pts))

        # Phase 2C warm path: the permanent CSR grid can be memory-mapped
        # directly from the point cache.  No 47M-point cell-id/argsort pass.
        _cached = _phase2c_try_load_grid(pts, self.target_points_per_cell)
        if _cached is not None:
            self.x_min = _cached["x_min"]; self.x_max = _cached["x_max"]
            self.y_min = _cached["y_min"]; self.y_max = _cached["y_max"]
            self.nx = _cached["nx"]; self.ny = _cached["ny"]
            self.cell_w = _cached["cell_w"]; self.cell_h = _cached["cell_h"]
            self._inv_cell_w = 1.0 / self.cell_w
            self._inv_cell_h = 1.0 / self.cell_h
            self.point_order = _cached["point_order"]
            self.cell_offsets = _cached["cell_offsets"]
            self._cell_count_prefix = _cached["prefix"]
            self.memory_bytes = int(
                self.point_order.nbytes + self.cell_offsets.nbytes + self._cell_count_prefix.nbytes
            )
            self._phase2c_cache_hit = True
            print(
                "PHASE2C_SPATIAL_CACHE status=hit "
                f"points={n:,} mmap=1 memory_mb={self.memory_bytes/(1024*1024):.1f}"
            )
            return

        self._phase2c_cache_hit = False
        if n <= 0:
            self.x_min = self.y_min = 0.0
            self.x_max = self.y_max = 0.0
            self.nx = self.ny = 1
            self.cell_w = self.cell_h = 1.0
            self.point_order = np.empty(0, dtype=np.uint32)
            self.cell_offsets = np.zeros(2, dtype=np.uint64)
            self.memory_bytes = self.point_order.nbytes + self.cell_offsets.nbytes
            return

        # Canonical coordinates remain untouched and retain their original dtype.
        x_min = float(np.min(pts[:, 0])); x_max = float(np.max(pts[:, 0]))
        y_min = float(np.min(pts[:, 1])); y_max = float(np.max(pts[:, 1]))
        rx = max(x_max - x_min, 1e-9)
        ry = max(y_max - y_min, 1e-9)
        aspect = max(1e-6, rx / ry)

        desired_cells = max(64, int(np.ceil(n / self.target_points_per_cell)))
        # Avoid pathological huge metadata arrays on sparse geographic extents.
        desired_cells = min(desired_cells, 1_000_000)
        nx = max(8, int(round(np.sqrt(desired_cells * aspect))))
        ny = max(8, int(np.ceil(desired_cells / nx)))
        nx = min(nx, 4096); ny = min(ny, 4096)

        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.nx, self.ny = int(nx), int(ny)
        self.cell_w = rx / self.nx
        self.cell_h = ry / self.ny
        self._inv_cell_w = 1.0 / self.cell_w
        self._inv_cell_h = 1.0 / self.cell_h
        n_cells = self.nx * self.ny

        # Build one compact cell id per point.  Chunks cap temporary XY arrays.
        cell_ids = np.empty(n, dtype=np.uint32)
        chunk = 2_000_000
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            ix = ((pts[s:e, 0] - x_min) * self._inv_cell_w).astype(np.int32)
            iy = ((pts[s:e, 1] - y_min) * self._inv_cell_h).astype(np.int32)
            np.clip(ix, 0, self.nx - 1, out=ix)
            np.clip(iy, 0, self.ny - 1, out=iy)
            cell_ids[s:e] = (ix.astype(np.uint32) * np.uint32(self.ny) + iy.astype(np.uint32))

        counts = np.bincount(cell_ids, minlength=n_cells).astype(np.uint64, copy=False)
        offsets = np.empty(n_cells + 1, dtype=np.uint64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        del counts

        # NumPy argsort returns platform intp.  Convert the permanent point-id
        # permutation to uint32 for clouds below 4.29B points.
        order64 = np.argsort(cell_ids, kind="stable")
        if n <= np.iinfo(np.uint32).max:
            point_order = order64.astype(np.uint32, copy=True)
        else:
            point_order = order64.astype(np.uint64, copy=True)
        del order64, cell_ids

        self.point_order = point_order
        self.cell_offsets = offsets
        cell_counts = np.diff(offsets).reshape(self.nx, self.ny)
        prefix = np.zeros((self.nx + 1, self.ny + 1), dtype=np.uint64)
        prefix[1:, 1:] = cell_counts.cumsum(axis=0).cumsum(axis=1)
        self._cell_count_prefix = prefix
        self.memory_bytes = int(point_order.nbytes + offsets.nbytes + prefix.nbytes)
        _phase2c_save_grid(self)

    def _region_count(self, ix0: int, ix1: int, iy0: int, iy1: int) -> int:
        if self.method != "grid2d":
            return 0
        ix0 = max(0, min(self.nx - 1, int(ix0))); ix1 = max(0, min(self.nx - 1, int(ix1)))
        iy0 = max(0, min(self.ny - 1, int(iy0))); iy1 = max(0, min(self.ny - 1, int(iy1)))
        if ix0 > ix1 or iy0 > iy1:
            return 0
        p = self._cell_count_prefix
        # Cast each prefix term to Python int before subtraction. NumPy uint64
        # intermediate subtraction can underflow even though the final region
        # count is non-negative, producing warnings or a wrapped count.
        return (int(p[ix1 + 1, iy1 + 1]) - int(p[ix0, iy1 + 1])
                - int(p[ix1 + 1, iy0]) + int(p[ix0, iy0]))

    def indices_for_cell_rect(self, ix0: int, ix1: int, iy0: int, iy1: int, *, sort_canonical: bool = True) -> np.ndarray:
        if self.method != "grid2d":
            return np.empty(0, dtype=np.int64)
        ix0 = max(0, min(self.nx - 1, int(ix0))); ix1 = max(0, min(self.nx - 1, int(ix1)))
        iy0 = max(0, min(self.ny - 1, int(iy0))); iy1 = max(0, min(self.ny - 1, int(iy1)))
        if ix0 > ix1 or iy0 > iy1:
            return np.empty(0, dtype=np.int64)
        pieces = []
        for ix in range(ix0, ix1 + 1):
            c0 = ix * self.ny + iy0; c1 = ix * self.ny + iy1
            st = int(self.cell_offsets[c0]); en = int(self.cell_offsets[c1 + 1])
            if en > st:
                pieces.append(self.point_order[st:en])
        if not pieces:
            return np.empty(0, dtype=np.int64)
        out = (pieces[0].astype(np.int64, copy=True) if len(pieces) == 1
               else np.concatenate(pieces).astype(np.int64, copy=False))
        if sort_canonical and out.size > 1:
            out.sort()
        return out

    def build_render_tiles(self, max_points: int = 2_000_000) -> list[RenderTileDescriptor]:
        """Recursively split the fine grid into rectangular render tiles."""
        if self.method != "grid2d" or len(self.points) == 0:
            return []
        limit = max(250_000, int(max_points))
        pending = [(0, self.nx - 1, 0, self.ny - 1)]
        regions = []
        while pending:
            ix0, ix1, iy0, iy1 = pending.pop()
            count = self._region_count(ix0, ix1, iy0, iy1)
            if count <= 0:
                continue
            if count <= limit or (ix0 == ix1 and iy0 == iy1):
                regions.append((ix0, ix1, iy0, iy1, count)); continue
            xspan = ix1 - ix0 + 1; yspan = iy1 - iy0 + 1
            if xspan >= yspan and xspan > 1:
                half = count / 2.0; best = ix0; err = float('inf')
                for cut in range(ix0, ix1):
                    e = abs(self._region_count(ix0, cut, iy0, iy1) - half)
                    if e < err: best, err = cut, e
                pending.append((best + 1, ix1, iy0, iy1)); pending.append((ix0, best, iy0, iy1))
            elif yspan > 1:
                half = count / 2.0; best = iy0; err = float('inf')
                for cut in range(iy0, iy1):
                    e = abs(self._region_count(ix0, ix1, iy0, cut) - half)
                    if e < err: best, err = cut, e
                pending.append((ix0, ix1, best + 1, iy1)); pending.append((ix0, ix1, iy0, best))
            else:
                regions.append((ix0, ix1, iy0, iy1, count))
        regions.sort(key=lambda r: (r[0], r[2], r[1], r[3]))
        tiles = []
        for tid, (ix0, ix1, iy0, iy1, count) in enumerate(regions):
            xmin = self.x_min + ix0 * self.cell_w; xmax = self.x_min + (ix1 + 1) * self.cell_w
            ymin = self.y_min + iy0 * self.cell_h; ymax = self.y_min + (iy1 + 1) * self.cell_h
            tiles.append(RenderTileDescriptor(tid, ix0, ix1, iy0, iy1, int(count),
                (float(xmin), float(xmax), float(ymin), float(ymax))))
        return tiles

    def tile_id_array_for_indices(self, indices, tiles: list[RenderTileDescriptor]) -> np.ndarray:
        """Return one render-tile id per canonical point index.

        The output is aligned with the input and uses ``-1`` for invalid or
        unmapped indices.  Phase 2C.2 uses this to distinguish an intentionally
        non-resident tile from a genuine mapping failure without scanning every
        resident actor.
        """
        raw = np.asarray(indices, dtype=np.int64).ravel()
        found = np.full(raw.size, -1, dtype=np.int32)
        if raw.size == 0 or not tiles:
            return found
        valid = (raw >= 0) & (raw < len(self.points))
        if not np.any(valid):
            return found
        idx = raw[valid]
        ix = ((self.points[idx, 0] - self.x_min) * self._inv_cell_w).astype(np.int32)
        iy = ((self.points[idx, 1] - self.y_min) * self._inv_cell_h).astype(np.int32)
        np.clip(ix, 0, self.nx - 1, out=ix); np.clip(iy, 0, self.ny - 1, out=iy)
        local = np.full(idx.size, -1, dtype=np.int32)
        for t in tiles:
            m = (ix >= t.ix0) & (ix <= t.ix1) & (iy >= t.iy0) & (iy <= t.iy1)
            local[m] = int(t.tile_id)
        found[valid] = local
        return found

    def tile_ids_for_indices(self, indices, tiles: list[RenderTileDescriptor]) -> np.ndarray:
        found = self.tile_id_array_for_indices(indices, tiles)
        return np.unique(found[found >= 0])

    def _cell_range(self, x_min, x_max, y_min, y_max):
        if x_max < self.x_min or x_min > self.x_max or y_max < self.y_min or y_min > self.y_max:
            return None
        ix0 = max(0, min(self.nx - 1, int(np.floor((x_min - self.x_min) * self._inv_cell_w))))
        ix1 = max(0, min(self.nx - 1, int(np.floor((x_max - self.x_min) * self._inv_cell_w))))
        iy0 = max(0, min(self.ny - 1, int(np.floor((y_min - self.y_min) * self._inv_cell_h))))
        iy1 = max(0, min(self.ny - 1, int(np.floor((y_max - self.y_min) * self._inv_cell_h))))
        if ix0 > ix1 or iy0 > iy1:
            return None
        return ix0, ix1, iy0, iy1

    def _candidate_rows(self, x_min, x_max, y_min, y_max):
        cr = self._cell_range(x_min, x_max, y_min, y_max)
        if cr is None:
            return []
        ix0, ix1, iy0, iy1 = cr
        rows = []
        offsets = self.cell_offsets
        order = self.point_order
        # Cell ids are ix*ny+iy, so a consecutive y-range is one contiguous
        # span in the sorted permutation for each x row.
        for ix in range(ix0, ix1 + 1):
            c0 = ix * self.ny + iy0
            c1 = ix * self.ny + iy1
            s = int(offsets[c0]); e = int(offsets[c1 + 1])
            if e > s:
                rows.append(order[s:e])
        return rows

    def query_rectangle_2d(self, x_min, x_max, y_min, y_max) -> np.ndarray:
        x_min, x_max = sorted((float(x_min), float(x_max)))
        y_min, y_max = sorted((float(y_min), float(y_max)))
        if self.method != "grid2d":
            pts = self.points
            mask = (
                (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
                (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)
            )
            return np.flatnonzero(mask).astype(np.int64, copy=False)

        rows = self._candidate_rows(x_min, x_max, y_min, y_max)
        if not rows:
            return np.empty(0, dtype=np.int64)
        candidates = rows[0] if len(rows) == 1 else np.concatenate(rows)

        # Exact filter in bounded chunks: large cross-section boxes can contain
        # millions of candidates without creating a giant Nx2 temporary copy.
        out = []
        chunk = 2_000_000
        for s in range(0, len(candidates), chunk):
            c = candidates[s:s+chunk]
            px = self.points[c, 0]
            py = self.points[c, 1]
            keep = (px >= x_min) & (px <= x_max) & (py >= y_min) & (py <= y_max)
            if np.any(keep):
                out.append(c[keep])
        if not out:
            return np.empty(0, dtype=np.int64)
        result = out[0] if len(out) == 1 else np.concatenate(out)
        return result.astype(np.int64, copy=False)

    def query_circle_2d(self, center, radius: float) -> np.ndarray:
        cx, cy = float(center[0]), float(center[1])
        r = max(0.0, float(radius))
        if r <= 0.0:
            return np.empty(0, dtype=np.int64)
        candidates = self.query_rectangle_2d(cx-r, cx+r, cy-r, cy+r)
        if candidates.size == 0:
            return candidates
        dx = self.points[candidates, 0] - cx
        dy = self.points[candidates, 1] - cy
        return candidates[(dx*dx + dy*dy) <= r*r]

    def query_polygon_2d(self, polygon) -> np.ndarray:
        poly = np.asarray(polygon, dtype=np.float64)
        if poly.ndim != 2 or poly.shape[0] < 3:
            return np.empty(0, dtype=np.int64)
        xs = poly[:, 0]; ys = poly[:, 1]
        candidates = self.query_rectangle_2d(xs.min(), xs.max(), ys.min(), ys.max())
        if candidates.size == 0:
            return candidates
        from matplotlib.path import Path
        path = Path(poly[:, :2], closed=True)
        out = []
        chunk = 1_000_000
        for s in range(0, candidates.size, chunk):
            c = candidates[s:s+chunk]
            keep = path.contains_points(self.points[c, :2])
            if np.any(keep):
                out.append(c[keep])
        if not out:
            return np.empty(0, dtype=np.int64)
        return (out[0] if len(out) == 1 else np.concatenate(out)).astype(np.int64, copy=False)

    def query_ball_point(self, center, radius):
        # Preserve exact legacy 3-D semantics.  Large datasets build the 3-D
        # tree only when a legacy caller truly needs it.
        return self.tree.query_ball_point(center, radius)

    def query_rectangle(self, x_min, x_max, y_min, y_max, z_min=None, z_max=None):
        idx = self.query_rectangle_2d(x_min, x_max, y_min, y_max)
        if idx.size == 0 or (z_min is None and z_max is None):
            return idx
        z = self.points[idx, 2]
        keep = np.ones(idx.size, dtype=bool)
        if z_min is not None:
            keep &= z >= float(z_min)
        if z_max is not None:
            keep &= z <= float(z_max)
        return idx[keep]

    def query_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        return self.query_rectangle_2d(xmin, xmax, ymin, ymax)

    # Compatibility with the ClassificationInteractor's old SpatialGridIndex.
    def query_rectangle_minmax(self, u_min, z_min, u_max, z_max):
        return self.query_rectangle_2d(u_min, u_max, z_min, z_max)

    def query_radius(self, u, z=None, radius=None):
        # Old 2-D grid signature: query_radius(x, y, r)
        if radius is not None and z is not None and np.isscalar(u):
            return self.query_circle_2d((u, z), radius)
        # Generic legacy signature: query_radius(center3d, radius)
        if radius is None:
            return self.query_ball_point(u, z)
        return self.query_ball_point(u, radius)

    def dirty_cells_for_indices(self, indices) -> np.ndarray:
        """Return unique Phase-2 spatial cells touched by canonical point ids."""
        idx = np.asarray(indices, dtype=np.int64).ravel()
        if idx.size == 0 or self.method != "grid2d":
            return np.empty(0, dtype=np.uint32)
        idx = idx[(idx >= 0) & (idx < len(self.points))]
        if idx.size == 0:
            return np.empty(0, dtype=np.uint32)
        ix = ((self.points[idx, 0] - self.x_min) * self._inv_cell_w).astype(np.int32)
        iy = ((self.points[idx, 1] - self.y_min) * self._inv_cell_h).astype(np.int32)
        np.clip(ix, 0, self.nx - 1, out=ix); np.clip(iy, 0, self.ny - 1, out=iy)
        return np.unique(ix.astype(np.uint32) * np.uint32(self.ny) + iy.astype(np.uint32))


def build_spatial_index_auto(points, method: str = "auto") -> SpatialIndex:
    n = int(len(points))
    if method == "auto":
        # cKDTree is excellent for smaller/general 3-D datasets, but for tens of
        # millions of LiDAR points it is expensive memory-wise and rectangle
        # queries in the old code still scanned all N.  Use the shared XY grid
        # for large interactive editing and build a 3-D tree lazily if required.
        method = "kdtree" if n < 2_000_000 else "grid2d"
    return SpatialIndex(points, method=method)


def query_section_box(
    xyz: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    depth: float = 0.5,
    use_cache: bool = True,
) -> np.ndarray:
    """Return canonical IDs inside the section AABB using the shared index."""
    if use_cache:
        return get_or_build_index(xyz).query_rectangle_2d(x_min, x_max, y_min, y_max)
    mask = (
        (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) &
        (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max)
    )
    return np.flatnonzero(mask).astype(np.int64, copy=False)
