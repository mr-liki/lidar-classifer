"""Shading-control parity: does Azimuth / Sharpness / Ambient change the native surface
exactly the way it changes the VTK shaded-classification view?

This is the control-level gate the facet gate cannot give you. The Vulkan shaded-class
proxy is a *bake*: the VTK view bakes every solid-cell colour as

    RGB = trunc(classRGB * _crisp_shade_chunk(azimuth, sharpness, ambient))

with lighting switched OFF on that actor, which is why moving the Sharpness slider changes
VTK pixels at all. If the native side only multiplies the class colour by the plain
hillshade max(N.L, ambient), then Sharpness does nothing there, Ambient ignores the 0.08
anti-black floor, and the over-90 light-elevation drop is missing - the three controls look
"connected" but produce a different picture.

Method
------
For each parameter set, compute the reference colours through PRODUCTION code
(gui/shading_display.py: _compute_shading_geometry_backend + _build_crisp_base_face_colors,
driven by the same sharpness -> effective-elevation derivation update_shaded_class uses),
then bake the SAME points through the native mesher (audit/tin_policy_probe.cpp,
crisp_shade = 1) and compare face by face.

Both meshes are compared on shared facets only, so a cocircular diagonal flip or a
duplicate-return choice cannot be mistaken for a shading difference.

Usage:
    python audit/lnt_bake_shading_parity.py [C:\\path\\file.laz|.las]
Exit code 0 = PASS.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audit"))

from gui import shading_display as sd  # noqa: E402
import tin_policy_parity as tin  # noqa: E402

PROBE_EXE = ROOT / "audit" / "tin_policy_probe.exe"

# The three controls as the panel exposes them, sampled at both ends and across the
# Sharpness overdrive knee (90 / 200) where the derivation switches branch.
SETS = [
    ("defaults", 45.0, 45.0, 0.25),
    ("sharpness_0", 45.0, 0.0, 0.25),
    ("sharpness_90", 45.0, 90.0, 0.25),
    ("sharpness_120", 45.0, 120.0, 0.25),
    ("sharpness_200", 45.0, 200.0, 0.25),
    ("sharpness_500", 45.0, 500.0, 0.25),
    ("sharpness_999", 45.0, 999.0, 0.25),
    ("azimuth_0", 0.0, 45.0, 0.25),
    ("azimuth_135", 135.0, 45.0, 0.25),
    ("azimuth_225", 225.0, 45.0, 0.25),
    ("azimuth_315", 315.0, 45.0, 0.25),
    ("azimuth_270_sharp_180", 270.0, 180.0, 0.25),
    ("ambient_0", 45.0, 45.0, 0.0),
    ("ambient_floor", 45.0, 45.0, 0.05),
    ("ambient_0.5", 45.0, 45.0, 0.5),
    ("ambient_0.9", 45.0, 45.0, 0.9),
    ("dark_overdrive", 200.0, 250.0, 0.30),
]

MAX_DIFF = 1          # one RGB level, i.e. float32-vs-float64 rounding
PALETTE = {2: (170, 120, 60), 5: (40, 200, 60), 6: (230, 40, 40), 9: (60, 140, 230)}


def palette_bytes(palette: dict) -> bytes:
    table = np.full((256, 3), 160, dtype=np.uint8)
    for code, rgb in palette.items():
        table[code] = rgb
    return table.tobytes()


def class_palette(palette: dict) -> dict:
    return {code: {"color": rgb, "show": True} for code, rgb in palette.items()}


def reference_colors(xyz, cls, visible, azimuth, sharpness, ambient):
    """The VTK view's baked cell colours for one parameter set, through production code."""
    app = SimpleNamespace(
        class_palette=class_palette(PALETTE),
        shade_ambient=float(ambient),
        shading_sharpness_angle=float(sharpness),
        shading_light_elevation=45.0,
    )
    # What update_shaded_class does before it builds geometry: Sharpness -> effective
    # light elevation (over-90 lowers the sun; 0..90 keeps the configured 45 degrees).
    _, overdrive = sd._shading_sharpness_response(float(sharpness))
    effective_elevation = sd._shading_effective_light_elevation(app, overdrive)

    res = sd._compute_shading_geometry_backend(
        xyz, cls, set(visible), float(azimuth), float(effective_elevation), float(ambient),
        3.0, None, f"bake-{azimuth}-{sharpness}-{ambient}", quality_mode="slow",
    )
    if res.get("empty"):
        return None, None, None
    cache = SimpleNamespace(
        faces=np.asarray(res["faces"], dtype=np.int32),
        xyz_unique=np.asarray(res["xyz_unique"], dtype=np.float64),
        face_normals=np.asarray(res["face_normals"], dtype=np.float64),
        shade=np.asarray(res["shade"], dtype=np.float64),
        vertex_normals=res.get("vertex_normals"),
        last_azimuth=float(azimuth),
        last_angle=float(effective_elevation),
        last_ambient=float(ambient),
        smooth_all_classes=False,
    )
    vertex_classes = cls[np.asarray(res["unique_indices"])]
    colors = sd._build_crisp_base_face_colors(app, cache, vertex_classes, set(visible))
    # Only PURE faces are baked by the VTK actor. A mixed-colour face gets a hidden
    # fallback colour there and is drawn by the separate barycentric overlay actor, so
    # comparing its baked colour would compare two different models.
    pure = (vertex_classes[cache.faces[:, 0]] == vertex_classes[cache.faces[:, 1]]) & \
           (vertex_classes[cache.faces[:, 0]] == vertex_classes[cache.faces[:, 2]])
    return np.asarray(res["xyz_final"], dtype=np.float64), cache.faces, colors, pure


def native_colors(points_bin: Path, out_bin: Path, palette_bin: Path,
                  azimuth, sharpness, ambient, visible):
    result = subprocess.run(
        [str(PROBE_EXE), str(points_bin), str(out_bin),
         str(azimuth), "45.0", str(ambient),
         ",".join(str(c) for c in visible), str(sharpness), "1", str(palette_bin)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"probe failed: {result.stderr}")
    raw = np.fromfile(out_bin, dtype=np.int64, count=5)
    nverts, ntris = int(raw[0]), int(raw[1])
    with open(out_bin, "rb") as f:
        f.seek(5 * 8)
        positions = np.fromfile(f, dtype="<f8", count=3 * nverts).reshape(nverts, 3)
        faces = np.fromfile(f, dtype="<i4", count=3 * ntris).reshape(ntris, 3)
        vclass = np.fromfile(f, dtype="u1", count=nverts)
        colors = np.fromfile(f, dtype="u1", count=3 * ntris).reshape(ntris, 3)
    pure = (vclass[faces[:, 0]] == vclass[faces[:, 1]]) & (vclass[faces[:, 0]] == vclass[faces[:, 2]])
    return positions, faces, colors, pure


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else r"H:\TESTING CONTIUES\123.laz"
    if not os.path.exists(path):
        print(f"[bake] missing dataset: {path}")
        return 2
    if not PROBE_EXE.exists():
        print(f"[bake] missing {PROBE_EXE} (build with audit/build_tin_policy_probe.bat)")
        return 2

    xyz, cls = tin.load_points(path)
    counts = np.bincount(cls, minlength=256)
    visible = [int(c) for c in np.argsort(counts)[::-1][:6] if counts[c] > 0 and c != 0][:2]
    for code in visible:
        PALETTE.setdefault(code, (160, 160, 160))
    print(f"[bake] {path}: {len(xyz):,} points, visible classes {visible}", flush=True)

    failures = []
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder)
        points_bin = tmp / "points.bin"
        palette_bin = tmp / "palette.bin"
        tin.write_points(points_bin, xyz, cls)
        palette_bin.write_bytes(palette_bytes(PALETTE))

        for label, azimuth, sharpness, ambient in SETS:
            vtk_xyz, vtk_faces, vtk_rgb, vtk_pure = reference_colors(
                xyz, cls, visible, azimuth, sharpness, ambient)
            if vtk_rgb is None:
                failures.append(f"{label}: no VTK geometry")
                continue
            nat_xyz, nat_faces, nat_rgb, nat_pure = native_colors(
                points_bin, tmp / f"{label}.bin", palette_bin,
                azimuth, sharpness, ambient, visible)

            vkeys = tin.face_keys(vtk_xyz, vtk_faces)
            nkeys = tin.face_keys(nat_xyz, nat_faces)
            vmap = {k: i for i, k in enumerate(vkeys) if vtk_pure[i]}
            nmap = {k: i for i, k in enumerate(nkeys) if nat_pure[i]}
            shared = vmap.keys() & nmap.keys()
            if not shared:
                failures.append(f"{label}: no shared facets")
                continue
            vi = np.fromiter((vmap[k] for k in shared), dtype=np.int64, count=len(shared))
            ni = np.fromiter((nmap[k] for k in shared), dtype=np.int64, count=len(shared))
            delta = np.abs(vtk_rgb[vi].astype(np.int16) - nat_rgb[ni].astype(np.int16))
            worst = int(delta.max())
            mean = float(delta.mean())
            same = 100.0 * float(np.count_nonzero(delta.max(axis=1) <= MAX_DIFF)) / len(vi)
            print(f"  {label:<24} az={azimuth:5.0f} sharp={sharpness:5.0f} amb={ambient:.2f}"
                  f"  pure={len(vi):,}  max={worst} mean={mean:.4f}  within1={same:6.2f}%",
                  flush=True)
            if worst > MAX_DIFF:
                failures.append(f"{label}: max {worst} levels")

    print("\n" + "=" * 70)
    if failures:
        print(f"SHADING CONTROL PARITY: FAIL ({len(failures)} of {len(SETS)})")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"SHADING CONTROL PARITY: PASS ({len(SETS)}/{len(SETS)} parameter sets)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
