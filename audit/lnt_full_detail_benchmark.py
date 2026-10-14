"""Read-only full-detail Shading benchmark with point counts, capture and peak RAM."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gui import lnt_native as nat
import numpy as np
import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--dll')
    parser.add_argument('--capture')
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if args.dll:
        nat.LNT_DLL = str(Path(args.dll).resolve())
    dll = nat.lib()
    error = C.create_string_buffer(512)
    nv = nat.NativeView.__new__(nat.NativeView)
    nv.h = dll.lnt_view_open(None, 800, 600, os.fsencode(args.dataset),
                           os.fsencode(nat.VULKAN_DLL), 128, 8, error, len(error))
    assert nv.h, error.value.decode(errors='replace')
    nv.stats = nat.LntViewStats()
    process = psutil.Process()
    try:
        palette = np.zeros((256, 4), dtype=np.uint8)
        palette[:, :3] = (128, 128, 128)
        palette[:, 3] = 1
        for cls, rgb in {2: (160, 120, 60), 5: (30, 160, 40),
                         6: (200, 60, 60), 1: (180, 180, 180)}.items():
            palette[cls, :3] = rgb
        nv.set_point_budget(1000)
        nv.set_sse(1000000)
        assert nv.set_classification_lut(palette)
        assert nv.set_surface_shading(315, 45, .18, 0)
        nv.set_surface_single_elevation(45)
        dataset = nv.info()[0]
        started = time.monotonic()
        assert nv.set_shading(nat.SHADE_NORMAL)
        peak_rss = 0
        while time.monotonic() - started < args.timeout:
            assert nv.frame(.016)
            peak_rss = max(peak_rss, process.memory_info().rss)
            source, visible, coverage = nv.surface_point_counts()
            if source == visible == dataset:
                break
            if hasattr(nv, 'surface_error') and nv.surface_error():
                raise RuntimeError(nv.surface_error())
            time.sleep(.016)
        else:
            raise TimeoutError(f'full-detail build incomplete: {nv.surface_point_counts()}')
        elapsed = time.monotonic() - started
        if args.capture:
            from PIL import Image
            Image.fromarray(nv.capture()[:, :, :3]).save(args.capture)
        memory = process.memory_info()
        print(json.dumps({'dataset': args.dataset, 'seconds_to_surface': elapsed,
                          'source': source, 'visible': visible, 'coverage': coverage,
                          'triangle_referenced_records': visible-coverage,
                          'peak_working_set_gb': getattr(memory, 'peak_wset', peak_rss)/1e9,
                          'peak_commit_gb': getattr(memory, 'peak_pagefile', memory.vms)/1e9}),
              flush=True)
    finally:
        nv.close()


if __name__ == '__main__':
    main()
