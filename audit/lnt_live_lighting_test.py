"""Real GPU pixel regression for lighting changes on resident tiled surfaces."""
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import laspy
import numpy as np
from gui import lnt_native as nat


def main():
    with tempfile.TemporaryDirectory(prefix="lnt_lighting_") as folder:
        source, target = Path(folder)/"terrain.las", Path(folder)/"terrain.lnt"
        x, y = np.meshgrid(np.linspace(0, 30, 100), np.linspace(0, 30, 100))
        z = 4*np.sin(x*.4)*np.cos(y*.3)
        las = laspy.LasData(laspy.LasHeader(point_format=7, version="1.4"))
        las.x, las.y, las.z = x.ravel(), y.ravel(), z.ravel()
        las.classification = np.where(x.ravel() < 15, 2, 5).astype(np.uint8)
        las.write(source)
        nat.convert_las(str(source), str(target))
        view = nat.NativeView(0, 640, 480, str(target), 64, 0.)
        try:
            palette = np.full((256, 4), 255, dtype=np.uint8)
            palette[:, :3] = (180, 140, 90)
            assert view.set_classification_lut(palette)
            assert view.set_shading(nat.SHADE_NORMAL)
            deadline = time.monotonic()+30
            while time.monotonic() < deadline:
                assert view.frame(.016)
                stats = view.surface_lod_stats()
                if view.surface_lod_active() and stats and stats["triangles_drawn"] > 0 and not view.surface_building():
                    break
                time.sleep(.02)
            assert view.surface_lod_active(), "test must exercise tiled shading"

            def pixels(az, sharp, ambient):
                assert view.set_surface_shading(az, sharp, ambient)
                for _ in range(5):
                    assert view.frame(.016)
                    time.sleep(.025)
                image = view.capture()
                assert image is not None
                return image[:, :, :3].astype(float)

            base = pixels(45, 45, .25)
            for label, params in [("azimuth", (225, 45, .25)),
                                  ("sharpness", (45, 180, .25)),
                                  ("brightness", (45, 45, .8))]:
                changed = pixels(*params)
                delta = float(np.mean(np.abs(changed-base)))
                assert delta > .5, (label, delta)
                print(f"PASS {label}: real pixel mean delta={delta:.3f}")
        finally:
            view.close()


if __name__ == "__main__":
    main()
