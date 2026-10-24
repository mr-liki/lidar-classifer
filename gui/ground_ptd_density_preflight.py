"""Read-only preflight for the native Ground PTD LNT context-window planner.

Never changes classes, PTD parameters, halo, node identities, or LNT bytes.  The
4-m occupancy index counts exactly the same allowed classifications and packed
XY coordinates as native_ground_below_stream._read_window. Its prefix sums are
*conservative bounds*; ambiguous comparisons use a matching exact streamed
count.  This permits safe, early selection of oversized contexts instead of
spending minutes in PTD and then failing at subdivision depth 12.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np


class DensityPreflightError(RuntimeError):
    pass


def _checked_cancel(cancelled):
    if cancelled is not None and cancelled():
        raise DensityPreflightError('Ground analysis cancelled before any LNT write.')


class DensityIndex:
    def __init__(self, path, info, allowed_classes, dtype, exactly,
                 *, eligible_classes=None, bin_size=4., chunks=180_000, cancelled=None):
        self.bin_size = float(bin_size)
        if not np.isfinite(self.bin_size) or self.bin_size <= 0:
            raise DensityPreflightError('Invalid density preflight grid size.')
        self.exact_queries = self.bounds_queries = self.chunks = 0
        self.total_relevant = 0
        self._path = Path(path)
        self._info = info
        self._dtype = dtype
        self._exactly = exactly
        self._chunks = int(chunks)
        valid = None if allowed_classes is None else sorted({int(x) for x in allowed_classes})
        if valid is not None and any(c < 0 or c > 255 for c in valid):
            raise DensityPreflightError('Invalid source/trusted classification in density preflight.')
        self._allowed = valid
        self._eligible_table = (None if eligible_classes is None else
                                np.isin(np.arange(256), sorted({int(c) for c in eligible_classes})))
        self._allowed_table = None if valid is None else np.isin(np.arange(256), valid)
        leaves = info['leaves']
        origin = np.asarray(info['origin'], np.float64)
        if not leaves:
            raise DensityPreflightError('LNT has no packed point leaves.')
        minx = min(origin[0] + float(center[0]) - float(half)
                   for _, _, _, center, half in leaves)
        miny = min(origin[1] + float(center[1]) - float(half)
                   for _, _, _, center, half in leaves)
        maxx = max(origin[0] + float(center[0]) + float(half)
                   for _, _, _, center, half in leaves)
        maxy = max(origin[1] + float(center[1]) + float(half)
                   for _, _, _, center, half in leaves)
        if not np.isfinite((minx, miny, maxx, maxy)).all():
            raise DensityPreflightError('Non-finite LNT leaf envelope.')
        self.minx, self.miny = math.floor(minx), math.floor(miny)
        self.nx = int(math.ceil((maxx - self.minx) / self.bin_size)) + 2
        self.ny = int(math.ceil((maxy - self.miny) / self.bin_size)) + 2
        if self.nx < 1 or self.ny < 1 or self.nx * self.ny > 6_000_000:
            raise DensityPreflightError('LNT spatial density grid exceeds preflight memory bound.')
        flat = np.zeros(self.nx * self.ny, np.int64)
        eligible_flat = (np.zeros(len(flat), np.int64)
                         if self._eligible_table is not None else None)
        with open(path, 'rb') as f:
            for leaf_idx, (node, offset, count, center, half) in enumerate(leaves):
                if leaf_idx % 32 == 0:
                    _checked_cancel(cancelled)
                scale = 2.0 * float(half) / 65535.0
                base_x = float(origin[0]) + float(center[0]) - float(half)
                base_y = float(origin[1]) + float(center[1]) - float(half)
                for start in range(0, int(count), self._chunks):
                    _checked_cancel(cancelled)
                    take = min(self._chunks, int(count) - start)
                    f.seek(int(offset) + start * dtype.itemsize)
                    raw = np.frombuffer(exactly(f, take * dtype.itemsize,
                                               f'density leaf {node}'), dtype=dtype)
                    if self._allowed_table is None:
                        rx = raw['x']
                        ry = raw['y']
                        cls = raw['classification']
                    else:
                        selected = self._allowed_table[raw['classification']]
                        if not np.any(selected):
                            continue
                        rx = raw['x'][selected]
                        ry = raw['y'][selected]
                        cls = raw['classification'][selected]
                    x = base_x + scale * rx.astype(np.float64)
                    y = base_y + scale * ry.astype(np.float64)
                    ix = np.floor((x - self.minx) / self.bin_size).astype(np.int64)
                    iy = np.floor((y - self.miny) / self.bin_size).astype(np.int64)
                    if (np.any(ix < 0) or np.any(iy < 0) or
                            np.any(ix >= self.nx) or np.any(iy >= self.ny)):
                        raise DensityPreflightError('Packed LNT point outside counted leaf envelope.')
                    # Packed LNT leaves are spatially local. Count only the
                    # index span touched by this leaf chunk instead of
                    # allocating a full-map bincount array 3,346 times.
                    keys = ix * self.ny + iy
                    first, last = int(keys.min()), int(keys.max())
                    flat[first:last+1] += np.bincount(
                        keys-first, minlength=last-first+1)
                    if eligible_flat is not None:
                        eligible_rows = self._eligible_table[cls]
                        if np.any(eligible_rows):
                            eligible_flat[first:last+1] += np.bincount(
                                keys[eligible_rows]-first,
                                minlength=last-first+1)
                    self.total_relevant += len(ix)
                    self.chunks += 1
        matrix = flat.reshape(self.nx, self.ny)
        self.prefix = np.zeros((self.nx + 1, self.ny + 1), dtype=np.int64)
        self.prefix[1:, 1:] = matrix.cumsum(axis=0).cumsum(axis=1)
        if int(self.prefix[-1, -1]) != self.total_relevant:
            raise DensityPreflightError('LNT density index count mismatch.')
        self.eligible_prefix = None
        if eligible_flat is not None:
            self.eligible_prefix = np.zeros_like(self.prefix)
            self.eligible_prefix[1:, 1:] = (eligible_flat.reshape(self.nx, self.ny)
                                           .cumsum(axis=0).cumsum(axis=1))

    def _sum_bins(self, i0, j0, i1, j1, prefix=None):
        i0 = min(self.nx, max(0, int(i0)))
        i1 = min(self.nx, max(0, int(i1)))
        j0 = min(self.ny, max(0, int(j0)))
        j1 = min(self.ny, max(0, int(j1)))
        if i1 <= i0 or j1 <= j0:
            return 0
        p = self.prefix if prefix is None else prefix
        return int(p[i1, j1] - p[i0, j1] - p[i1, j0] + p[i0, j0])

    def count_bounds(self, bounds, *, eligible=False):
        """Return (lower,upper) exact conservative counts for half-open bbox."""
        self.bounds_queries += 1
        x0, y0, x1, y1 = (float(v) for v in bounds)
        if not np.isfinite((x0, y0, x1, y1)).all() or x0 > x1 or y0 > y1:
            raise DensityPreflightError('Invalid Ground context bounds.')
        if x0 == x1 or y0 == y1:
            return (0, 0)
        a = (x0 - self.minx) / self.bin_size
        b = (x1 - self.minx) / self.bin_size
        c = (y0 - self.miny) / self.bin_size
        d = (y1 - self.miny) / self.bin_size
        # A one-bin floating-point guard is intentional. The LNT uses metre
        # coordinates near 4.8M and packed-uint16 quantization, so boundary
        # subtraction can round across an exact bin edge. Never let these
        # fast counts claim a window fits when the original exact reader
        # would reject it. Ambiguous windows are counted exactly.
        prefix = self.eligible_prefix if eligible else self.prefix
        if prefix is None:
            raise DensityPreflightError('Source eligibility histogram was not built.')
        lower = self._sum_bins(math.ceil(a)+1, math.ceil(c)+1,
                               math.floor(b)-1, math.floor(d)-1, prefix)
        upper = self._sum_bins(math.floor(a)-1, math.floor(c)-1,
                               math.ceil(b)+1, math.ceil(d)+1, prefix)
        if lower > upper:
            raise DensityPreflightError('Invalid LNT density interval.')
        return lower, upper

    def _count_exact_at_most(self, bounds, max_count, cancelled=None):
        """Stop at max_count+1. Match the original LNT decoder and class filter."""
        self.exact_queries += 1
        xmin, ymin, xmax, ymax = map(float, bounds)
        origin = np.asarray(self._info['origin'], np.float64)
        found = 0
        with open(self._path, 'rb') as f:
            for leaf_no, (node, offset, count, center, half) in enumerate(self._info['leaves']):
                if leaf_no % 32 == 0:
                    _checked_cancel(cancelled)
                abs_center = origin + np.asarray(center, np.float64)
                if (abs_center[0] + half < xmin or abs_center[0] - half > xmax or
                    abs_center[1] + half < ymin or abs_center[1] - half > ymax):
                    continue
                scale = 2.0 * float(half) / 65535.0
                xbase = abs_center[0] - float(half)
                ybase = abs_center[1] - float(half)
                for start in range(0, int(count), self._chunks):
                    _checked_cancel(cancelled)
                    take = min(self._chunks, int(count) - start)
                    f.seek(int(offset) + start * self._dtype.itemsize)
                    raw = np.frombuffer(self._exactly(f, take * self._dtype.itemsize,
                                  f'density leaf {node}'), dtype=self._dtype)
                    x = xbase + scale * raw['x'].astype(np.float64)
                    y = ybase + scale * raw['y'].astype(np.float64)
                    mask = ((x >= xmin) & (x < xmax) &
                            (y >= ymin) & (y < ymax))
                    if self._allowed_table is not None:
                        mask &= self._allowed_table[raw['classification']]
                    found += int(np.count_nonzero(mask))
                    if found > max_count:
                        return found
        return found

    def fits(self, bounds, limit, cancelled=None):
        low, high = self.count_bounds(bounds)
        if high <= limit:
            return True
        if low > limit:
            return False
        return self._count_exact_at_most(bounds, limit, cancelled=cancelled) <= limit


def ram_budgeted_extended_limit(base, workers, requested=6_000_000):
    """Budget a peak-larger window conservatively, regardless of the GPU VRAM."""
    base, workers, requested = int(base), int(workers), int(requested)
    if base <= 0 or workers <= 0 or requested < base:
        raise DensityPreflightError('Invalid memory-aware Ground limits.')
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
    except (ImportError, AttributeError):
        # No unknown-memory guessing or disabling the density guard.
        raise DensityPreflightError('psutil required for safe dense-context RAM budgeting.')
    reserve = 4 * 1024**3  # preserve OS + renderer and other application data
    bytes_per_pt = 1400    # bounds SciPy/PTD temporary arrays, not raw LNT bytes
    allowance = max(0, (available - reserve) // bytes_per_pt)
    # Large tasks run alone, normal tasks can run two at a time.
    allowable = min(8_000_000, allowance, requested)
    # Never block the previously working 2.8M path solely because Windows
    # reports low free RAM; just decline to enlarge that proven bound.
    return max(base, allowable), available


def plan_windows(regions, index, *, halo, base_limit, dense_limit,
                 adaptive, prefer_dense=False, cancelled=None, max_depth=12, intersects=None):
    """Precompute ALL read windows before any PTD or LNT write.

    The ordinary non-oversize 2.8M regions retain their original tiling.
    Adaptive rescue preserves original base-limit splits by default, using
    RAM-guarded large contexts only where depth-12 original would abort.
    `prefer_dense=True` (separately opt-in) avoids recursive splitting and
    speeds dense areas, but changes the local tiling/CPU reference semantics.
    Always compare GPU/CPU on the same saved plan and source classes.
    """
    planned = []
    dense = 0
    splits = 0
    skipped_ineligible = 0
    for top in regions:
        current = [(top, 0)]
        outputs = []
        while current:
            _checked_cancel(cancelled)
            core, depth = current.pop()
            if intersects is not None and not intersects(core):
                continue
            if index.eligible_prefix is not None and index.count_bounds(
                    core, eligible=True)[1] == 0:
                skipped_ineligible += 1
                continue
            padded = (core[0]-halo, core[1]-halo,
                      core[2]+halo, core[3]+halo)
            if index.fits(padded, base_limit, cancelled):
                outputs.append((core, base_limit))
                continue
            if (adaptive and dense_limit > base_limit and
                    (prefer_dense or depth >= max_depth) and index.fits(
                        padded, dense_limit, cancelled)):
                outputs.append((core, dense_limit))
                dense += 1
                continue
            if depth >= max_depth:
                low, high = index.count_bounds(padded)
                raise DensityPreflightError(
                    'Ground context cannot fit RAM-safe limit: '
                    f'core={tuple(round(float(q), 3) for q in core)} '
                    f'halo={halo:.2f}m points_at_least={low:,} '
                    f'upper={high:,} allowed={dense_limit if adaptive else base_limit:,}. '
                    'No Ground work started and no classifications were changed.')
            left, right = _split_rect(core)
            current.extend(((right, depth + 1), (left, depth + 1)))
            splits += 1
        planned.append(outputs)
    return planned, {'total_cells': sum(map(len, planned)),
                     'dense_cells': dense, 'splits': splits,
                     'prefer_dense': bool(prefer_dense),
                     'skipped_ineligible_cells': skipped_ineligible,
                     'preflight_counts': index.bounds_queries,
                     'fallback_exact_counts': index.exact_queries,
                     'relevant_source_points': index.total_relevant}


def _split_rect(bounds):
    x0, y0, x1, y1 = bounds
    if x1 - x0 >= y1 - y0:
        mid = x0 + (x1 - x0) / 2.0
        return (x0, y0, mid, y1), (mid, y0, x1, y1)
    mid = y0 + (y1 - y0) / 2.0
    return (x0, y0, x1, mid), (x0, mid, x1, y1)
