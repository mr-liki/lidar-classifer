"""Read-only full-cloud Ground density/memory precheck.

Run from the BRUSH repository root using `python -m scripts.preflight_ground_lnt`.
No Vulkan DLL, classification writes, or Ground PTD execution is performed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import monotonic


def main():
    p = argparse.ArgumentParser(description='Read-only Ground full-cloud LNT preflight')
    p.add_argument('lnt', help='existing original .lnt file (never modified)')
    p.add_argument('--from-classes', default='0', help='comma-separated source class codes, or any')
    p.add_argument('--trusted', default='1', help='comma-separated current Ground codes, or any')
    p.add_argument('--to-class', default=1, type=int)
    p.add_argument('--halo', default=90.0, type=float)
    p.add_argument('--tile-size', default=110.0, type=float)
    p.add_argument('--base-cap', default=2_800_000, type=int)
    p.add_argument('--dense-cap', default=6_000_000, type=int)
    p.add_argument('--workers', default=2, type=int)
    args = p.parse_args()
    from gui.native_ground_below_stream import (
        _regions, _preflight_ground_windows, _class_set, inspect_directory)

    def parse(text):
        return None if text.strip().lower() in ('any', 'all', '*') else [
            int(item.strip()) for item in text.split(',') if item.strip()]

    src, trusted = _class_set(parse(args.from_classes)), _class_set(parse(args.trusted))
    if src is not None and trusted is not None and (src & trusted):
        raise SystemExit('Source and trusted class sets must not overlap.')
    allowed = None if src is None or trusted is None else src | trusted
    path = Path(args.lnt).expanduser().resolve(strict=True)
    if path.suffix.lower() != '.lnt':
        raise SystemExit('Only existing .lnt files are supported by this read-only scanner.')
    info = inspect_directory(path, max_points=2**63-1)
    regions = _regions(info, [], args.tile_size)
    os.environ['NAKSHA_GROUND_PREFLIGHT'] = '1'
    os.environ['NAKSHA_GROUND_DENSE_ADAPTIVE'] = '1'
    os.environ['NAKSHA_GROUND_DENSE_MAX_CONTEXT'] = str(args.dense_cap)
    print('[GROUND PRECHECK] READ ONLY; no Ground classifications will be calculated or changed.')
    print(f'[GROUND PRECHECK] total={info["total"]:,} roots={len(regions)} '
          f'halo={args.halo}m sources={sorted(src) if src is not None else "Any"} '
          f'trusted={sorted(trusted) if trusted is not None else "Any"}')
    began = monotonic()
    plans, stats = _preflight_ground_windows(
        path, info, regions, allowed, args.halo, args.base_cap, [],
        workers=args.workers, cancelled=None, params_sources=src,
        params_target=args.to_class, trace=False)
    assert plans is not None
    print('[GROUND PRECHECK] PASS: all contexts fit RAM-budgeted limits, '
          f'planned_cells={stats["total_cells"]:,}, dense={stats["dense_cells"]:,}, '
          f'empty-skipped={stats["skipped_ineligible_cells"]:,}, '
          f'elapsed={monotonic()-began:.2f}s')
    print('[GROUND PRECHECK] This does NOT predict elapsed PTD processing time.')


if __name__ == '__main__':
    main()
