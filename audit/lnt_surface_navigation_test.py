"""Check surface coverage during motion, before the new viewport has settled.

Uses a read-only source session (persistence disabled); accepts any .lnt path.
"""
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ['NAKSHA_LNT_SURFACE_PERSIST'] = '0'
os.environ['NAKSHA_LNT_SURFACE_GPU_MB'] = '400'
from lnt_surface_lod_view_test import View, header
from gui import lnt_native as nat


def main():
    source = Path(sys.argv[1])
    fixture = '--fixture' in sys.argv
    before = source.stat()
    h, count, _ = header(source)
    side = h[7] - h[4]
    view = View(nat, source)
    durations, holes, fractions = [], [], []
    try:
        assert view.nv.set_shading(nat.SHADE_NORMAL)
        started = time.monotonic()
        settled = view.settle(timeout=120)
        overview_seconds = time.monotonic() - started
        assert settled['source_covered'] == count
        # Change scale and centre every frame: do not wait for fine tiles.
        for i in range(32):
            t = i / 31.
            width = side * (.12 + .25 * (1 + np.cos(t * 4 * np.pi)))
            x = h[4] + side * (.5 + .12 * np.sin(t * 2 * np.pi))
            y = h[5] + side * (.5 + .12 * np.cos(t * 2 * np.pi))
            view.nv.frame_region(x, y, width)
            begin = time.perf_counter()
            assert view.nv.frame(.016)
            durations.append(time.perf_counter() - begin)
            hole, lit, image = view.holes()
            holes.append(hole)
            fractions.append(lit / (image.shape[0] * image.shape[1]))
            if fixture:
                assert hole == 0, f'interior void during navigation frame {i}: {hole}'
                assert fractions[-1] > .99, f'viewport lost its fallback during frame {i}'
        final = view.settle(timeout=120)
        assert final['source_covered'] > 0
        print(f'PASS source={count:,} overview={overview_seconds:.3f}s '
              f'motion_frames={len(durations)} frame_p95_ms={np.percentile(durations,95)*1000:.2f} '
              f'max_enclosed_void_pixels={max(holes)} min_coverage={min(fractions):.4f} '
              f'gpu_vertices={final["gpu_vertices"]:,}', flush=True)
    finally:
        view.close()
    after = source.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)


if __name__ == '__main__':
    main()
    sys.stdout.flush()
    os._exit(0)
