"""Make a repeatable all-points LAS fixture in bounded chunks, then convert it."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import laspy
import numpy as np
from gui import lnt_native as nat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--points', type=int, default=50_000_000)
    parser.add_argument('--dll')
    args = parser.parse_args()
    if args.dll:
        nat.LNT_DLL = str(Path(args.dll).resolve())
    if args.output.exists() or args.output.with_suffix('.lnt').exists():
        raise FileExistsError('fixture already exists; use a fresh output path')
    hdr = laspy.LasHeader(point_format=3, version='1.2')
    hdr.scales = [.001] * 3
    hdr.offsets = [500000, 4000000, 0]
    rng = np.random.default_rng(932)
    started = time.monotonic()
    with laspy.open(args.output, mode='w', header=hdr) as writer:
        for start in range(0, args.points, 1_000_000):
            n = min(1_000_000, args.points - start)
            x, y = rng.uniform(0, 1000, (2, n))
            z = 100 + 8*np.sin(x/45)*np.cos(y/60)
            cls = np.full(n, 2, dtype=np.uint8)
            trees = ((x-650)**2 + (y-700)**2) < 100**2
            z[trees] += 5 + 2*np.sin(x[trees])
            cls[trees] = 5
            building = (x > 150) & (x < 250) & (y > 200) & (y < 300)
            z[building] += 9
            cls[building] = 6
            points = laspy.ScaleAwarePointRecord.zeros(n, header=hdr)
            points.x, points.y, points.z = x+500000, y+4000000, z
            points.classification = cls
            points.intensity = rng.integers(0, 65536, n, dtype=np.uint16)
            writer.write_points(points)
    print(f'LAS: {args.points:,} points in {time.monotonic()-started:.1f}s', flush=True)
    print(nat.convert_las(str(args.output), str(args.output.with_suffix('.lnt'))), flush=True)


if __name__ == '__main__':
    main()
