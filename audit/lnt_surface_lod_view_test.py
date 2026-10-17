"""GPU acceptance for the persistent viewport surface LOD (native DLLs only, no Qt/VTK).

Works on private copies of audit/lod_fixture_4m.lnt; never touches a real dataset.
    python audit/lnt_surface_lod_view_test.py            # runs every phase in subprocesses
Environment: NAKSHA_TEST_LNT_DLL / NAKSHA_TEST_VULKAN_DLL select the DLLs under test.
"""
import ctypes as C
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'audit/lod_fixture_4m.lnt'
WORK = ROOT / 'audit/lod_view_test.lnt'
STATS = ['drawn_tiles', 'drawn_exact_tiles', 'source_covered', 'source_exact', 'represented_drawn',
         'triangles_drawn', 'coverage_drawn', 'gpu_tiles', 'gpu_vertices', 'cpu_tiles', 'cpu_bytes',
         'queued', 'inflight', 'built', 'loaded', 'uploaded', 'gpu_evicted', 'cpu_evicted',
         'load_failures', 'archive_tiles', 'commits', 'archive_bytes', 'total_points', 'wanted_missing']
W, H = 800, 600


def header(path):
    with open(path, 'rb') as f:
        h = struct.unpack('<4sHHQ6d', f.read(64))
        f.seek(h[3])
        points, nodes = struct.unpack('<QQ', f.read(16))
    return h, points, h[3] + 16 + nodes * 52


class View:
    def __init__(self, nat, path):
        self.nat = nat
        dll = nat.lib()
        for name, res in [('lnt_view_surface_lod_active', C.c_int), ('lnt_view_surface_flush', C.c_int),
                          ('lnt_view_renderer_handle', C.c_uint64), ('lnt_view_set_surface_lod_px', C.c_int),
                          ('lnt_view_surface_persist_message', C.c_char_p)]:
            getattr(dll, name).restype = res
        dll.lnt_view_surface_lod_active.argtypes = [C.c_void_p]
        dll.lnt_view_surface_flush.argtypes = [C.c_void_p]
        dll.lnt_view_renderer_handle.argtypes = [C.c_void_p]
        dll.lnt_view_set_surface_lod_px.argtypes = [C.c_void_p, C.c_double]
        dll.lnt_view_surface_persist_message.argtypes = [C.c_void_p]
        dll.lnt_view_surface_lod_stats.argtypes = [C.c_void_p, C.POINTER(C.c_uint64), C.c_int]
        dll.lnt_view_surface_lod_stats.restype = C.c_int
        self.dll = dll
        err = C.create_string_buffer(512)
        nv = nat.NativeView.__new__(nat.NativeView)
        nv.h = dll.lnt_view_open(None, W, H, os.fsencode(path), os.fsencode(nat.VULKAN_DLL), 512, 8.0, err, 512)
        assert nv.h, err.value
        nv.stats = nat.LntViewStats()
        nv.last_error = ''
        self.nv = nv
        self.vk = C.CDLL(nat.VULKAN_DLL)
        self.vk.nkv_surface_tiles_get_stats.argtypes = [C.c_uint64, C.POINTER(C.c_uint64), C.c_uint32]
        self.palette = np.zeros((256, 4), dtype=np.uint8)
        self.palette[:, :3] = 160
        self.palette[:, 3] = 1
        self.palette[2] = (170, 120, 60, 1)
        self.palette[5] = (40, 200, 60, 1)
        self.palette[6] = (230, 40, 40, 1)
        assert nv.set_classification_lut(self.palette)

    def stats(self):
        out = (C.c_uint64 * len(STATS))()
        n = self.dll.lnt_view_surface_lod_stats(self.nv.h, out, len(STATS))
        return dict(zip(STATS, list(out)[:n]))

    def engine(self):
        out = (C.c_uint64 * 16)()
        self.vk.nkv_surface_tiles_get_stats(self.dll.lnt_view_renderer_handle(self.nv.h), out, 16)
        return list(out)

    def lod_active(self):
        return bool(self.dll.lnt_view_surface_lod_active(self.nv.h))

    def settle(self, timeout=90, quiet=6):
        deadline = time.monotonic() + timeout
        calm = 0
        while time.monotonic() < deadline:
            assert self.nv.frame(.016), 'frame failed'
            err = self.nv.surface_error()
            assert not err, err
            if self.lod_active() and not self.nv.surface_building():
                calm += 1
                if calm >= quiet:
                    for _ in range(3):
                        assert self.nv.frame(.016)
                        time.sleep(.01)
                    return self.stats()
            else:
                calm = 0
            time.sleep(.004)
        raise AssertionError(f'surface did not settle: {self.stats()}')

    def holes(self):
        """Background pixels enclosed by the surface (not connected to the image border):
        cracks and gaps, as opposed to the surface's own silhouette."""
        from scipy import ndimage
        img = self.nv.capture()
        assert img is not None
        lit = np.any(img[:, :, :3] > 0, axis=2)
        labels, _ = ndimage.label(~lit)
        border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        enclosed = (~lit) & ~np.isin(labels, border)
        return int(enclosed.sum()), int(lit.sum()), img

    def flush(self):
        """Commit every built tile, then refresh the per-frame statistics."""
        assert self.dll.lnt_view_surface_flush(self.nv.h)
        for _ in range(3):
            assert self.nv.frame(.016)
        return self.stats()

    def close(self):
        self.nv.close()


def load(env_extra=None):
    sys.path.insert(0, str(ROOT))
    from gui import lnt_native as nat
    nat.LNT_DLL = os.environ.get('NAKSHA_TEST_LNT_DLL', str(ROOT / 'native/lnt/main_shading_lod.dll'))
    nat.VULKAN_DLL = os.environ.get('NAKSHA_TEST_VULKAN_DLL',
                                    str(ROOT / 'native/naksha_vulkan/build_refresh/naksha_vulkan.dll'))
    return nat


def fmt(s, *keys):
    return ', '.join(f'{k} {s[k]:,}' for k in keys)


def phase_build():
    nat = load()
    h, points, base = header(WORK)
    side = h[7] - h[4]
    v = View(nat, WORK)
    nv = v.nv
    try:
        assert nv.set_shading(nat.SHADE_NORMAL)
        t0 = time.monotonic()
        s = v.settle()
        first = time.monotonic() - t0
        assert v.lod_active() and s['drawn_tiles'] > 0
        assert s['source_covered'] == points, s
        assert s['drawn_exact_tiles'] < s['drawn_tiles'], 'overview should be drawn from coarse tiles'
        overview = s
        hole_lod, lit_lod, _ = v.holes()
        print(f'PASS first overview in {first:.1f}s: {fmt(s, "drawn_tiles", "drawn_exact_tiles", "source_covered", "represented_drawn", "triangles_drawn", "built")}'
              f' - all {points:,} source records covered through the hierarchy, {s["represented_drawn"]:,} representatives drawn', flush=True)
        # Full resolution everywhere in view (threshold 0.25 px): only exact tiles.
        assert v.dll.lnt_view_set_surface_lod_px(nv.h, 0.25)
        s = v.settle()
        assert s['drawn_exact_tiles'] == s['drawn_tiles'] and s['source_exact'] == points, s
        assert s['represented_drawn'] == points, 'full detail must represent every visible record individually'
        assert overview['triangles_drawn'] < s['triangles_drawn'] / 2, (overview['triangles_drawn'], s['triangles_drawn'])
        hole_full, lit_full, _ = v.holes()
        print(f'PASS full resolution: {fmt(s, "drawn_tiles", "triangles_drawn", "coverage_drawn")}; every record a vertex or coverage point; '
              f'overview drew {overview["triangles_drawn"] / s["triangles_drawn"]:.1%} of these triangles', flush=True)
        assert hole_lod <= hole_full + 5, (hole_lod, hole_full)
        print(f'PASS no LOD holes: {hole_lod} background pixels inside the surface at overview vs {hole_full} at full resolution', flush=True)
        assert v.dll.lnt_view_set_surface_lod_px(nv.h, 2.0)
        s = v.settle()
        # Zoom into a corner: full resolution recovered locally.
        nv.frame_region(h[4] + side * .2, h[5] + side * .2, side * .06)
        s = v.settle()
        assert s['drawn_exact_tiles'] == s['drawn_tiles'] and s['source_exact'] == s['source_covered'] > 0, s
        built_zoom = s['built']
        print(f'PASS close zoom: {fmt(s, "drawn_tiles", "source_exact", "triangles_drawn")} - full resolution', flush=True)
        for _ in range(8):
            nv.pan(100, 0)
            assert nv.frame(.016)
        s = v.settle()
        built_pan = s['built']
        assert built_pan == built_zoom, 'panning across previously refined tiles must reuse their geometry'
        assert s['drawn_exact_tiles'] == s['drawn_tiles'] and s['source_exact'] > 0, s
        print(f'PASS pan across cached full-detail tiles: no triangulation (built {built_pan})', flush=True)
        uploads = v.engine()[10]
        nv.reset()
        s = v.settle()
        nv.frame_region(h[4] + side * .2, h[5] + side * .2, side * .06)
        s = v.settle()
        assert s['built'] == built_pan, 'returning to cached views must not triangulate again'
        print(f'PASS revisit overview + corner: no triangulation ({s["built"]} built in total), {v.engine()[10] - uploads} GPU tile uploads', flush=True)
        # 3D perspective: near tiles finer than far tiles; seams closed.
        nv.reset()
        nv.set_2d(False)
        for _ in range(6):
            nv.orbit(0, -40)
            assert nv.frame(.016)
        for _ in range(16):              # approach until near tiles need full resolution
            s3 = v.settle()
            if 0 < s3['drawn_exact_tiles'] < s3['drawn_tiles']:
                break
            nv.zoom(1.5)
        s3 = v.settle()
        hole3, lit3, img3 = v.holes()
        assert v.dll.lnt_view_set_surface_lod_px(nv.h, 0.25)
        f3 = v.settle()
        hole3f, lit3f, _ = v.holes()
        assert v.dll.lnt_view_set_surface_lod_px(nv.h, 2.0)
        assert 0 < s3['drawn_exact_tiles'] < s3['drawn_tiles'], s3
        assert hole3 <= hole3f + 5, (hole3, hole3f)
        print(f'PASS perspective mixed LOD: {s3["drawn_exact_tiles"]}/{s3["drawn_tiles"]} tiles at full resolution, '
              f'{hole3} background pixels inside the surface (full resolution: {hole3f})', flush=True)
        nv.set_2d(True)
        nv.plan_view()
        nv.reset()
        s = v.settle()
        # Persist.
        s = v.flush()
        assert s['commits'] >= 1 and s['archive_tiles'] == s['built'], s
        assert os.path.getsize(WORK) > base
        print(f'PASS archive: {s["commits"]} commit(s), {s["archive_tiles"]} tiles, {s["archive_bytes"] / 1e6:.1f} MB appended after byte {base:,}', flush=True)
        # Palette change: colour only.
        built = s['built']
        recolors = v.engine()[15]
        v.palette[5] = (20, 120, 220, 1)
        assert nv.set_classification_lut(v.palette)
        s = v.settle()
        assert s['built'] == built and v.engine()[15] > recolors, (s, v.engine())
        print('PASS palette change recolours resident tiles without triangulation', flush=True)
        # Leave Shading: point streaming resumes; return is instant.
        assert nv.set_shading(nat.SHADE_CLASSIFICATION)
        for _ in range(60):
            assert nv.frame(.016)
            time.sleep(.005)
        assert nv.stats.points_rendered > 0, 'point streaming did not resume after leaving Shading'
        assert nv.set_shading(nat.SHADE_NORMAL)
        s = v.settle()
        assert s['built'] == built
        print('PASS point streaming resumes outside Shading; returning to Shading triangulates nothing', flush=True)
        # Class visibility changes geometry: a separate settings slot.
        v.palette[6, 3] = 0
        assert nv.set_classification_lut(v.palette)
        s = v.settle()
        assert s['built'] > built and s['source_covered'] == points and s['represented_drawn'] < overview['represented_drawn'] + 1, s
        hidden_built = s['built']
        v.flush()
        v.palette[6, 3] = 1
        assert nv.set_classification_lut(v.palette)
        s = v.settle()
        assert s['built'] == hidden_built and s['loaded'] > 0, s
        print(f'PASS hiding a class builds its own tiles ({hidden_built - built}); restoring it reloads the stored tiles ({s["loaded"]} loaded, none rebuilt)', flush=True)
        v.flush()
    finally:
        v.close()


def phase_reopen():
    nat = load()
    h, points, base = header(WORK)
    side = h[7] - h[4]
    size = os.path.getsize(WORK)
    v = View(nat, WORK)
    nv = v.nv
    try:
        assert nv.set_shading(nat.SHADE_NORMAL)
        t0 = time.monotonic()
        s = v.settle()
        dt = time.monotonic() - t0
        assert s['built'] == 0 and s['loaded'] > 0 and s['source_covered'] == points, s
        nv.frame_region(h[4] + side * .2, h[5] + side * .2, side * .06)
        s = v.settle()
        assert s['built'] == 0, s
        print(f'PASS reopen: overview in {dt:.2f}s from the archive, zoom in - {s["loaded"]} tiles loaded, 0 triangulated', flush=True)
    finally:
        v.close()
    assert os.path.getsize(WORK) == size, 'a reopen without new tiles must not write'


def phase_edit():
    nat = load()
    h, points, base = header(WORK)
    side = h[7] - h[4]
    v = View(nat, WORK)
    nv = v.nv
    try:
        assert nv.set_shading(nat.SHADE_NORMAL)
        s = v.settle()
        assert s['built'] == 0
        x0, y0 = h[4] + side * .5, h[5] + side * .5
        q = nv.query_points((x0, y0, -1e9), (x0 + 5, y0 + 5, 1e9))
        assert q is not None and len(q['node']) > 0
        assert nv.set_classes(q['node'], q['local'], np.full(len(q['node']), 6, np.uint8)), nv.last_error
        s = v.settle()
        assert s['built'] == 0, s
        v.flush()
        print(f'PASS class edit patched stored tiles: {s["built"]} rebuilt, archive re-keyed', flush=True)
    finally:
        v.close()
    v = View(nat, WORK)
    try:
        assert v.nv.set_shading(nat.SHADE_NORMAL)
        s = v.settle()
        assert s['built'] == 0 and s['loaded'] > 0, s
        print('PASS reopen after the edit uses the re-keyed archive (0 triangulated)', flush=True)
    finally:
        v.close()


def phase_evict():
    """Tiny GPU/CPU budgets: eviction keeps working sets bounded and revisits do not rebuild."""
    nat = load()
    h, points, base = header(WORK)
    side = h[7] - h[4]
    v = View(nat, WORK)
    nv = v.nv
    try:
        assert nv.set_shading(nat.SHADE_NORMAL)
        v.settle()
        caps = v.engine()
        built = None
        for rnd in range(2):
            for x in (.15, .5, .85):
                for y in (.15, .5, .85):
                    nv.frame_region(h[4] + side * x, h[5] + side * y, side * .12)
                    s = v.settle()
                    e = v.engine()
                    assert e[1] <= e[2] and e[3] <= e[4], e
                    assert s['drawn_exact_tiles'] == s['drawn_tiles'] > 0
            if rnd == 0:
                built = s['built']
                v.flush()
        s = v.stats()
        assert s['gpu_evicted'] > 0 and s['built'] == built, s
        print(f'PASS bounded caches: arena {caps[2]:,} vertices; {s["gpu_evicted"]} GPU / {s["cpu_evicted"]} CPU evictions over two sweeps, '
              f'second sweep reloaded instead of triangulating ({s["loaded"]} loaded)', flush=True)
    finally:
        v.close()


def phase_readonly():
    nat = load()
    os.chmod(WORK, 0o444)
    size = os.path.getsize(WORK)
    v = View(nat, WORK)
    try:
        assert v.nv.set_shading(nat.SHADE_NORMAL)
        s = v.settle()
        assert s['built'] > 0
        v.flush()
        msg = (v.dll.lnt_view_surface_persist_message(v.nv.h) or b'').decode()
        assert 'Access is denied' in msg or 'denied' in msg.lower(), msg
        print(f'PASS read-only file: Shading works, real save error reported: "{msg}"', flush=True)
    finally:
        v.close()
        os.chmod(WORK, 0o666)
    assert os.path.getsize(WORK) == size


def phase_corrupt():
    nat = load()
    sys.path.insert(0, str(ROOT))
    h, points, base = header(WORK)
    # Flip one byte in the middle of the first stored tile record.
    with open(WORK, 'r+b') as f:
        f.seek(base + 72 + 40)
        b = f.read(1)
        f.seek(base + 72 + 40)
        f.write(bytes([b[0] ^ 0x77]))
    v = View(nat, WORK)
    try:
        assert v.nv.set_shading(nat.SHADE_NORMAL)
        s = v.settle()
        assert s['load_failures'] >= 1 and s['built'] >= 1, s
        print(f'PASS corrupted stored tile rejected ({s["load_failures"]}) and rebuilt; {s["loaded"]} intact tiles loaded', flush=True)
    finally:
        v.close()


def phase_badleaf():
    """A leaf whose payload cannot be read: its tiles fail, coarser tiles cover the area,
    and the reason reaches the GUI as the Shading error."""
    nat = load()
    raw = bytearray(FIXTURE.read_bytes())
    index = struct.unpack_from('<Q', raw, 8)[0]
    nodes = struct.unpack_from('<Q', raw, index + 8)[0]
    for i in range(nodes):
        off = index + 16 + 52 * i
        if raw[off + 40] == 0 and struct.unpack_from('<I', raw, off + 20)[0]:
            damaged_id = struct.unpack_from('<Q', raw, off)[0]
            struct.pack_into('<I', raw, off + 16, struct.unpack_from('<I', raw, off + 16)[0] + 1)
            break
    WORK.write_bytes(raw[:index + 16 + 52 * nodes])
    h, points, base = header(WORK)
    v = View(nat, WORK)
    nv = v.nv
    try:
        assert nv.set_shading(nat.SHADE_NORMAL)
        assert v.dll.lnt_view_set_surface_lod_px(nv.h, 0.25)     # request every full-resolution tile
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            assert nv.frame(.016)
            if nv.surface_error() and not nv.surface_building():
                break
            time.sleep(.005)
        err = nv.surface_error()
        s = v.stats()
        assert f'tile source node {damaged_id}:' in err and 'byte_length' in err, err
        assert s['drawn_tiles'] > 0 and s['drawn_exact_tiles'] < s['drawn_tiles'] and s['source_covered'] == points, s
        print(f'PASS unreadable leaf: "{err}"; {s["drawn_tiles"] - s["drawn_exact_tiles"]} coarser tiles keep all '
              f'{points:,} source records covered', flush=True)
    finally:
        v.close()


PHASES = {
    'build': ({'NAKSHA_LNT_SURFACE_GPU_MB': '400'}, phase_build),
    'reopen': ({'NAKSHA_LNT_SURFACE_GPU_MB': '400'}, phase_reopen),
    'evict': ({'NAKSHA_LNT_SURFACE_GPU_MB': '80', 'NAKSHA_LNT_SURFACE_CPU_MB': '96'}, phase_evict),
    'readonly': ({'NAKSHA_LNT_SURFACE_GPU_MB': '200'}, phase_readonly),
    'corrupt': ({'NAKSHA_LNT_SURFACE_GPU_MB': '200'}, phase_corrupt),
    'edit': ({'NAKSHA_LNT_SURFACE_GPU_MB': '200'}, phase_edit),
    'badleaf': ({'NAKSHA_LNT_SURFACE_GPU_MB': '400', 'NAKSHA_LNT_SURFACE_PERSIST': '0'}, phase_badleaf),
}


def run(name, fresh=False):
    env = dict(os.environ)
    env.update(PHASES[name][0])
    env['PYTHONUTF8'] = '1'
    if fresh:
        shutil.copyfile(FIXTURE, WORK)
        _, _, base = header(WORK)
        with open(WORK, 'r+b') as f:
            f.truncate(base)
    print(f'--- phase {name}', flush=True)
    r = subprocess.run([sys.executable, __file__, name], env=env, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f'phase {name} failed ({r.returncode})')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        PHASES[sys.argv[1]][1]()
        sys.stdout.flush()
        os._exit(0)   # skip interpreter teardown of the GPU driver (known noisy)
    run('build', fresh=True)
    run('reopen')
    run('corrupt')
    run('evict')
    run('edit')
    run('readonly', fresh=True)
    os.chmod(WORK, 0o666)
    run('badleaf')
    os.chmod(WORK, 0o666)
    WORK.unlink(missing_ok=True)
    print('ALL SURFACE LOD GPU CHECKS PASSED', flush=True)
