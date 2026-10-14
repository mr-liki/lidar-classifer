"""Timing measurements for the viewport surface LOD on a private copy of a local fixture.

    python audit/lnt_surface_lod_measure.py [fixture.lnt]
Phases (each in its own process): first session (build on demand + archive), reopen
(archive only), and the cut-based path on the same zoomed view for comparison.
Numbers depend on this machine, disk and GPU load; they are measurements, not targets.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'audit'))
WORK = ROOT / 'audit/lod_measure.lnt'


def session(label):
    import lnt_surface_lod_view_test as T
    nat = T.load()
    h, points, base = T.header(WORK)
    side = h[7] - h[4]
    v = T.View(nat, WORK)
    nv = v.nv
    walls = []

    def frame():
        t = time.perf_counter()
        assert nv.frame(.016)
        walls.append((time.perf_counter() - t) * 1000)

    def settle(tag, timeout=600):
        walls.clear()
        t0 = time.monotonic()
        first = None
        calm = 0
        while time.monotonic() - t0 < timeout:
            frame()
            assert not nv.surface_error(), nv.surface_error()
            s = v.stats() if v.lod_active() else {}
            if first is None and s.get('drawn_tiles', 0) > 0:
                first = time.monotonic() - t0
            if v.lod_active() and not nv.surface_building():
                calm += 1
                if calm >= 5:
                    break
            elif not v.lod_active() and nv.surface_method() and not nv.surface_building():
                calm += 1
                if calm >= 5:
                    break
            else:
                calm = 0
            time.sleep(.002)
        total = time.monotonic() - t0
        s = v.stats() if v.lod_active() else {}
        ws = sorted(walls)
        p95 = ws[int(len(ws) * .95)] if ws else 0
        extra = (f"first tiles {first:.2f}s, " if first is not None else '')
        if s:
            print(f"[{label}] {tag}: {extra}settled {total:.2f}s | drawn {s['drawn_tiles']} tiles "
                  f"({s['drawn_exact_tiles']} full-res), {s['triangles_drawn']:,} tris, "
                  f"source covered {s['source_covered']:,} ({s['source_exact']:,} at full res) | "
                  f"built {s['built']} loaded {s['loaded']} | frame call p95 {p95:.1f} ms, max {ws[-1]:.1f} ms",
                  flush=True)
        else:
            counts = nv.surface_point_counts()
            print(f"[{label}] {tag}: settled {total:.2f}s (cut path) | source {counts[0]:,} | "
                  f"frame call p95 {p95:.1f} ms, max {ws[-1]:.1f} ms", flush=True)
        return s

    try:
        assert nv.set_shading(nat.SHADE_NORMAL)
        if label != 'cut-path':
            settle('overview')
        regions = [(.25, .25), (.7, .4), (.5, .8)]
        for x, y in regions:
            nv.frame_region(h[4] + side * x, h[5] + side * y, side * .05)
            settle(f'zoom {x:.2f},{y:.2f} (5% width)')
            if label == 'cut-path':
                break
        if label != 'cut-path':
            walls.clear()
            t0 = time.monotonic()
            for _ in range(120):
                nv.pan(25, 0)
                frame()
                time.sleep(.016)
            ws = sorted(walls)
            s = settle('after a 120-step pan')
            print(f"[{label}] pan frames: p95 {ws[int(len(ws) * .95)]:.1f} ms, max {ws[-1]:.1f} ms (render thread)", flush=True)
            nv.reset()
            settle('back to overview')
            if label == 'first':
                t0 = time.monotonic()
                v.flush()
                print(f"[{label}] flush {time.monotonic() - t0:.2f}s; archive now {(os.path.getsize(WORK) - base) / 1e6:.1f} MB "
                      f"after {base / 1e6:.1f} MB of original data", flush=True)
    finally:
        t0 = time.monotonic()
        v.close()
        print(f"[{label}] close {time.monotonic() - t0:.2f}s", flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        session(sys.argv[2])
        sys.stdout.flush()
        os._exit(0)
    fixture = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'audit/full_detail_50m.lnt'
    shutil.copyfile(fixture, WORK)
    import lnt_surface_lod_view_test as T
    _, points, base = T.header(WORK)
    with open(WORK, 'r+b') as f:
        f.truncate(base)
    print(f'fixture {fixture.name}: {points:,} points, {base / 1e6:.1f} MB', flush=True)
    for label, env in (('first', {}), ('reopen', {}), ('cut-path', {'NAKSHA_LNT_SURFACE_LOD': '0', 'NAKSHA_LNT_SURFACE_PERSIST': '0'})):
        e = dict(os.environ, PYTHONUTF8='1', **env)
        r = subprocess.run([sys.executable, __file__, str(fixture), label], env=e, cwd=ROOT)
        if r.returncode:
            raise SystemExit(f'{label} failed')
    WORK.unlink(missing_ok=True)
