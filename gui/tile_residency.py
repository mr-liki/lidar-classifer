"""Policy helpers for Main View full-tile residency.

Phase 2C.3 adds *anti-thrash hot residency* on top of Phase 2C.2.

Why this exists
---------------
Phase 2C.2 correctly bounded the number of full-resolution VTK actors, but it
trimmed back to the soft budget immediately after every settled zoom.  A quick
zoom back out then had to rebuild the same 1.4-1.7M point actors (~90-110 ms
per tile on the reference GTX-1660 workstation).  That made production zooming
feel like "LOD loading" even though the LOD actors themselves were already
resident.

The new policy separates three concepts:

* soft budget  -- long-idle target for memory efficiency;
* hot budget   -- short-term working-set cache used while the operator is
                  actively navigating;
* visible set  -- always protected, even when it exceeds both budgets.

Tiles are never evicted merely because one wheel gesture temporarily made the
viewport smaller.  Recently-visible full actors are kept hot for a grace
period and are trimmed only after an idle delay or real memory pressure.

Safety contract
---------------
* Every tile intersecting the settled/editing viewport is protected from
  eviction.
* Exact-visible behavior therefore remains unchanged.
* Canonical mmap data remains authoritative; this module only chooses actor
  residency.
* Memory pressure can override the hot cache and trim back to the soft budget.
"""
from __future__ import annotations

import os
import time
from typing import Iterable, Mapping, Sequence, Tuple


DEFAULT_RESIDENT_FULL_TILES = 8
MIN_RESIDENT_FULL_TILES = 3
MAX_RESIDENT_FULL_TILES = 64
DEFAULT_RESIDENCY_GRACE_SECONDS = 12.0
DEFAULT_IDLE_TRIM_MS = 60000
DEFAULT_MIN_FREE_RAM_GB = 6.0


def residency_enabled() -> bool:
    raw = str(os.getenv("NAKSHA_MAIN_TILE_RESIDENCY", "1")).strip().lower()
    return raw not in ("0", "false", "off", "no")


def _available_ram_gb() -> float | None:
    try:
        import psutil
        return float(psutil.virtual_memory().available) / float(1024 ** 3)
    except Exception:
        return None


def resident_full_tile_budget() -> int:
    """Return the long-idle *soft* full-resolution actor budget."""
    explicit = str(os.getenv("NAKSHA_MAIN_RESIDENT_FULL_TILES", "")).strip()
    if explicit:
        try:
            return max(MIN_RESIDENT_FULL_TILES, min(MAX_RESIDENT_FULL_TILES, int(explicit)))
        except Exception:
            pass

    budget = DEFAULT_RESIDENT_FULL_TILES
    avail_gb = _available_ram_gb()
    if avail_gb is not None:
        if avail_gb >= 24.0:
            budget = 12
        elif avail_gb >= 16.0:
            budget = 10
        elif avail_gb >= 10.0:
            budget = 8
        elif avail_gb >= 6.0:
            budget = 6
        else:
            budget = 4
    return max(MIN_RESIDENT_FULL_TILES, min(MAX_RESIDENT_FULL_TILES, int(budget)))


def resident_full_tile_hot_budget(soft_budget: int | None = None) -> int:
    """Return short-term hot-cache capacity used during production navigation.

    This is deliberately larger than the soft budget.  It does *not* increase
    peak memory beyond what an exact zoomed-out frame already required; it only
    avoids immediately throwing those already-built actors away.
    """
    explicit = str(os.getenv("NAKSHA_MAIN_RESIDENT_HOT_TILES", "")).strip()
    if explicit:
        try:
            return max(MIN_RESIDENT_FULL_TILES, min(MAX_RESIDENT_FULL_TILES, int(explicit)))
        except Exception:
            pass

    soft = int(soft_budget if soft_budget is not None else resident_full_tile_budget())
    avail_gb = _available_ram_gb()
    if avail_gb is None:
        return min(MAX_RESIDENT_FULL_TILES, max(soft, soft + 6))

    # Keep a generous but bounded navigation working set when RAM is available.
    # Exact-visible tiles can still exceed this number; correctness always wins.
    if avail_gb >= 32.0:
        hot = max(soft, 32)
    elif avail_gb >= 24.0:
        hot = max(soft, 28)
    elif avail_gb >= 16.0:
        hot = max(soft, 24)
    elif avail_gb >= 10.0:
        hot = max(soft, 18)
    elif avail_gb >= 6.0:
        hot = max(soft, 14)
    else:
        hot = soft
    return max(soft, min(MAX_RESIDENT_FULL_TILES, int(hot)))


def residency_grace_seconds() -> float:
    raw = str(os.getenv("NAKSHA_MAIN_RESIDENT_GRACE_SECONDS", "")).strip()
    if raw:
        try:
            return max(0.0, min(300.0, float(raw)))
        except Exception:
            pass
    return float(DEFAULT_RESIDENCY_GRACE_SECONDS)


def residency_idle_trim_ms() -> int:
    raw = str(os.getenv("NAKSHA_MAIN_RESIDENT_IDLE_TRIM_MS", "")).strip()
    if raw:
        try:
            return max(1000, min(600000, int(float(raw))))
        except Exception:
            pass
    return int(DEFAULT_IDLE_TRIM_MS)


def min_free_ram_gb() -> float:
    raw = str(os.getenv("NAKSHA_MAIN_MIN_FREE_RAM_GB", "")).strip()
    if raw:
        try:
            return max(1.0, min(64.0, float(raw)))
        except Exception:
            pass
    return float(DEFAULT_MIN_FREE_RAM_GB)


def memory_pressure() -> bool:
    """Return True only for real system-memory pressure.

    Under pressure the hot cache yields immediately to the soft budget.
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        avail_gb = float(vm.available) / float(1024 ** 3)
        avail_frac = float(vm.available) / max(float(vm.total), 1.0)
        return bool(avail_gb < min_free_ram_gb() or avail_frac < 0.08)
    except Exception:
        return False


def tiles_intersecting(descriptors: Sequence, bounds: Tuple[float, float, float, float] | None) -> list[int]:
    """Return descriptor tile ids intersecting XY bounds."""
    if bounds is None:
        return [int(d.tile_id) for d in descriptors]
    xmin, xmax, ymin, ymax = [float(v) for v in bounds]
    out = []
    for d in descriptors:
        tx0, tx1, ty0, ty1 = d.bounds
        if not (tx1 < xmin or tx0 > xmax or ty1 < ymin or ty0 > ymax):
            out.append(int(d.tile_id))
    return out


def source_point_count(descriptor_by_id: Mapping[int, object], tile_ids: Iterable[int]) -> int:
    total = 0
    for tid in tile_ids:
        d = descriptor_by_id.get(int(tid))
        if d is not None:
            total += int(getattr(d, "point_count", 0) or 0)
    return int(total)


def choose_evictions(
    resident_ids: Iterable[int],
    protected_ids: Iterable[int],
    last_touch: Mapping[int, float],
    soft_budget: int,
    *,
    hot_budget: int | None = None,
    grace_seconds: float = 0.0,
    now: float | None = None,
    force_soft: bool = True,
) -> list[int]:
    """Choose LRU non-protected tiles to evict.

    Backward compatibility: with the default ``force_soft=True`` this behaves
    like the Phase-2C.2 helper and trims to ``soft_budget`` immediately.

    Production-navigation mode passes ``force_soft=False``.  In that mode the
    target is ``hot_budget`` and tiles touched within ``grace_seconds`` are not
    evicted.  This is the anti-thrash rule that prevents wheel zoom from
    repeatedly destroying/rebuilding the same full actors.
    """
    resident = {int(v) for v in resident_ids}
    protected = {int(v) for v in protected_ids}
    soft = max(0, int(soft_budget))
    hot = max(soft, int(hot_budget if hot_budget is not None else soft))
    target = soft if force_soft else hot
    effective_budget = max(int(target), len(protected))
    excess = max(0, len(resident) - effective_budget)
    if excess <= 0:
        return []

    candidates = [tid for tid in resident if tid not in protected]
    candidates.sort(key=lambda tid: (float(last_touch.get(tid, 0.0)), tid))

    if not force_soft and grace_seconds > 0.0:
        stamp = float(time.monotonic() if now is None else now)
        aged = [
            tid for tid in candidates
            if (stamp - float(last_touch.get(tid, 0.0))) >= float(grace_seconds)
        ]
        candidates = aged

    return candidates[:excess]


def touch(access_map: dict, tile_ids: Iterable[int], when: float | None = None) -> None:
    stamp = float(time.monotonic() if when is None else when)
    for tid in tile_ids:
        access_map[int(tid)] = stamp
