"""A failed leaf read must withhold the whole surface and expose a GUI error."""
import argparse
import ctypes as C
import os
from pathlib import Path
import struct
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The cut-based path's contract; the tiled LOD path's read failures are covered by
# audit/lnt_surface_lod_view_test.py (coarser tile shown + error exposed).
os.environ['NAKSHA_LNT_SURFACE_LOD'] = '0'
from gui import lnt_native as nat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dll')
    args = parser.parse_args()
    if args.dll:
        nat.LNT_DLL = str(Path(args.dll).resolve())
    raw = bytearray((ROOT/'native/lnt/test_shading.lnt').read_bytes())
    index = struct.unpack_from('<Q', raw, 8)[0]
    nodes = struct.unpack_from('<Q', raw, index+8)[0]
    for i in range(nodes):
        offset = index+16+52*i
        if raw[offset+40] == 0 and struct.unpack_from('<I', raw, offset+20)[0]:
            length = struct.unpack_from('<I', raw, offset+16)[0]
            struct.pack_into('<I', raw, offset+16, length+1)
            break
    else:
        raise AssertionError('no leaf in the test fixture')
    with tempfile.TemporaryDirectory(prefix='surface_error_', dir=ROOT/'audit') as scratch:
        path = Path(scratch)/'bad_leaf.lnt'
        path.write_bytes(raw)
        dll = nat.lib()
        error = C.create_string_buffer(512)
        nv = nat.NativeView.__new__(nat.NativeView)
        nv.h = dll.lnt_view_open(None, 640, 480, os.fsencode(path),
                               os.fsencode(nat.VULKAN_DLL), 64, 1000000, error, len(error))
        assert nv.h, error.value
        nv.stats = nat.LntViewStats()
        try:
            nv.set_point_budget(1000)
            assert not nv.surface_error()
            assert nv.set_shading(nat.SHADE_NORMAL)
            end = time.monotonic()+10
            while time.monotonic()<end:
                assert nv.frame(.016)
                if nv.surface_error():
                    break
                time.sleep(.016)
            assert 'leaf read failed' in nv.surface_error(), nv.surface_error()
            assert nv.surface_point_counts() == (0, 0, 0), 'partial surface was published'
            print('PASS failed leaf read exposes an error and publishes no partial surface', flush=True)
        finally:
            nv.close()


if __name__ == '__main__':
    main()
