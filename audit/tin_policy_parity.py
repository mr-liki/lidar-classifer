"""Tin policy parity: does the native shaded-class TIN keep the SAME facets as VTK?

The Vulkan shaded-class viewport and the VTK/CPU Shaded Classification view are
supposed to show identical facets for identical panel values. The lighting math
is proven equal by vulkan_shading_parity_test.py (per-face RGB, mean 0.00). What
that gate cannot see is WHICH triangles exist: it samples faces the native engine
already drew. This audit closes that gap.

Method
------
1. Load a real LAS/LAZ window and pick the classes the viewport shows.
2. Build the reference mesh through the PRODUCTION VTK builder
   (gui/shading_display._compute_shading_geometry_backend, quality "slow" =
   every point, the policy the native mesher mirrors).
3. Build the native mesh through the PRODUCTION native builder
   (native/lnt/include/LNTSurfaceMesh.h: plan_surface -> build_surface_mesh),
   driven by audit/tin_policy_probe.cpp.
4. Compare the two triangle sets facet by facet (unordered coordinate triple):

   * policy: no face may exceed the VTK absolute plan-edge limit
     (10 % of the plan extent, 20 % with one visible class), and no face may be
     numerically degenerate -- in either mesh;
   * structure: how many faces exist, and how many are the SAME face;
   * appearance: for the faces both meshes contain, the per-face hillshade and
     facet normal must be equal (the visible consequence of a different TIN is
     a different facet normal, which is a different colour).

Delaunay is unique except for cocircular ties, so matched faces should dominate;
unmatched faces are reported by kind (long-edge, degenerate, tie) so a policy
difference can never hide behind a tie-break statistic.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gui import shading_display as shading  # noqa: E402

PROBE_SRC = ROOT / "audit" / "tin_policy_probe.cpp"
PROBE_EXE = ROOT / "audit" / "tin_policy_probe.exe"
BUILD_BAT = ROOT / "audit" / "build_tin_policy_probe.bat"

AZIMUTH, ELEVATION, AMBIENT = 45.0, 45.0, 0.25
MAX_EDGE_FACTOR = 3.0
WINDOW = 60.0          # metres, square window around the densest area
MAX_POINTS = 300_000

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}", flush=True)


def build_probe() -> None:
    if PROBE_EXE.exists() and PROBE_EXE.stat().st_mtime > PROBE_SRC.stat().st_mtime:
        return
    print(f"[tin] building {PROBE_EXE.name} ...", flush=True)
    subprocess.run(["cmd", "/c", str(BUILD_BAT)], cwd=str(ROOT), check=True)


def load_points(path: str) -> tuple[np.ndarray, np.ndarray]:
    import laspy

    with laspy.open(path) as fh:
        las = fh.read()
    xyz = np.column_stack([
        np.asarray(las.x, dtype=np.float64),
        np.asarray(las.y, dtype=np.float64),
        np.asarray(las.z, dtype=np.float64),
    ])
    cls = np.asarray(las.classification, dtype=np.uint8)
    # Dense window: the biggest class keeps the probe honest about real TINs.
    counts = np.bincount(cls, minlength=256)
    counts[0] = 0
    seed = int(np.argmax(counts))
    c = xyz[cls == seed]
    centre = c.mean(axis=0)
    keep = ((np.abs(xyz[:, 0] - centre[0]) <= WINDOW / 2.0) &
            (np.abs(xyz[:, 1] - centre[1]) <= WINDOW / 2.0))
    xyz, cls = xyz[keep], cls[keep]
    if len(xyz) > MAX_POINTS:
        step = int(np.ceil(len(xyz) / MAX_POINTS))
        xyz, cls = xyz[::step], cls[::step]
    return xyz, cls


def write_points(path: Path, xyz: np.ndarray, cls: np.ndarray) -> None:
    with open(path, "wb") as f:
        np.array([len(xyz)], dtype=np.int64).tofile(f)
        rec = np.empty(len(xyz), dtype=[("xyz", "<f8", 3), ("cls", "u1")])
        rec["xyz"] = xyz
        rec["cls"] = cls
        rec.tofile(f)


def run_probe(points: Path, out: Path, classes: list[int]) -> dict:
    result = subprocess.run(
        [str(PROBE_EXE), str(points), str(out), str(AZIMUTH), str(ELEVATION), str(AMBIENT),
         ",".join(str(c) for c in classes)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"probe failed: {result.stderr}")
    raw = np.fromfile(out, dtype=np.int64, count=5)
    nverts, ntris, visible, referenced, dropped = (int(v) for v in raw)
    with open(out, "rb") as f:
        f.seek(5 * 8)
        positions = np.fromfile(f, dtype="<f8", count=3 * nverts).reshape(nverts, 3)
        faces = np.fromfile(f, dtype="<i4", count=3 * ntris).reshape(ntris, 3)
        vclass = np.fromfile(f, dtype="u1", count=nverts)
        colors = np.fromfile(f, dtype="u1", count=3 * ntris).reshape(ntris, 3)
    return {
        "positions": positions, "faces": faces, "vertex_class": vclass, "colors": colors,
        "visible": visible, "referenced": referenced, "dropped": dropped,
        "stderr": result.stderr,
    }


def plan_edges(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Longest plan (XY) edge of every face, exactly as the policy is written."""
    if len(faces) == 0:
        return np.empty(0)
    p = positions[faces]
    d01 = np.hypot(p[:, 0, 0] - p[:, 1, 0], p[:, 0, 1] - p[:, 1, 1])
    d12 = np.hypot(p[:, 1, 0] - p[:, 2, 0], p[:, 1, 1] - p[:, 2, 1])
    d20 = np.hypot(p[:, 2, 0] - p[:, 0, 0], p[:, 2, 1] - p[:, 0, 1])
    return np.maximum(np.maximum(d01, d12), d20)


def plan_area(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.empty(0)
    p = positions[faces]
    return np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) -
                  (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])) * 0.5


def face_keys(positions: np.ndarray, faces: np.ndarray) -> list:
    """Unordered coordinate triples: the identity of a facet, not of an index."""
    keys = []
    for tri in faces:
        pts = positions[tri]
        keys.append(frozenset((float(a), float(b), float(c)) for a, b, c in pts))
    return keys


def edge_set(keys: list) -> set:
    """Every facet edge (as an unordered vertex pair) of a face-key collection."""
    edges = set()
    for key in keys:
        pts = list(key)
        for i in range(3):
            for j in range(i + 1, 3):
                edges.add(frozenset((pts[i], pts[j])))
    return edges


def shares_edge(key, edges: set) -> bool:
    pts = list(key)
    return any(frozenset((pts[i], pts[j])) in edges
               for i in range(3) for j in range(i + 1, 3))


def compare_case(label: str, xyz: np.ndarray, cls: np.ndarray, visible: list[int],
                 tmp: Path) -> None:
    print(f"\n[tin] {label}: {len(xyz):,} points, classes {visible}", flush=True)
    vtk = shading._compute_shading_geometry_backend(
        xyz, cls, set(visible), AZIMUTH, ELEVATION, AMBIENT,
        MAX_EDGE_FACTOR, None, f"tin-policy-{label}", quality_mode="slow",
    )
    if vtk.get("empty"):
        check(f"{label}: VTK geometry built", False, "empty result")
        return
    # xyz_unique is offset-relative (the CPU builder normalises by the cut's min XY);
    # xyz_final puts it back into world space, which is what the native mesh holds.
    vtk_xyz = np.asarray(vtk["xyz_final"], dtype=np.float64)
    vtk_faces = np.asarray(vtk["faces"], dtype=np.int32)
    vtk_shade = np.asarray(vtk["shade"], dtype=np.float64)
    extent = float(max(np.ptp(vtk_xyz[:, 0]), np.ptp(vtk_xyz[:, 1])))
    limit = extent * (0.20 if len(visible) == 1 else 0.10)

    points_bin = tmp / f"{label}.points.bin"
    out_bin = tmp / f"{label}.native.bin"
    write_points(points_bin, xyz, cls)
    native = run_probe(points_bin, out_bin, visible)

    nat_xyz, nat_faces = native["positions"], native["faces"]
    print(f"  VTK   : {len(vtk_xyz):,} vertices {len(vtk_faces):,} triangles "
          f"(feature_meta reason={vtk.get('feature_meta', {}).get('reason')})", flush=True)
    print(f"  native: {native['referenced'] and len(nat_xyz):,} vertices {len(nat_faces):,} triangles "
          f"(referenced {native['referenced']:,} of {native['visible']:,} visible, "
          f"dropped {native['dropped']:,})", flush=True)
    print(f"  plan extent {extent:,.3f} m -> VTK absolute plan-edge limit {limit:,.3f} m", flush=True)

    vtk_referenced = int(np.unique(vtk_faces).size) if len(vtk_faces) else 0
    check(f"{label}: both meshes reference the same points",
          native["referenced"] == vtk_referenced,
          f"native {native['referenced']:,} vs VTK {vtk_referenced:,} of {native['visible']:,} visible")

    # ---- policy ---------------------------------------------------------
    for name, xyz_m, faces_m in (("VTK", vtk_xyz, vtk_faces), ("native", nat_xyz, nat_faces)):
        edges = plan_edges(xyz_m, faces_m)
        over = int(np.count_nonzero(edges > limit * (1 + 1e-9)))
        area = plan_area(xyz_m, faces_m)
        degenerate = int(np.count_nonzero(area <= 1e-12))
        print(f"  {name:<6}: max plan edge {edges.max() if len(edges) else 0:,.3f} m, "
              f"{over} over the limit, {degenerate} degenerate, "
              f"mean area {area.mean() if len(area) else 0:,.4f} m2", flush=True)
        check(f"{label}: {name} keeps no face beyond the VTK plan-edge limit", over == 0, f"{over} faces")
        check(f"{label}: {name} keeps no numerically degenerate face", degenerate == 0, f"{degenerate} faces")

    # ---- structure ------------------------------------------------------
    vtk_keys, nat_keys = face_keys(vtk_xyz, vtk_faces), face_keys(nat_xyz, nat_faces)
    vtk_set, nat_set = set(vtk_keys), set(nat_keys)
    shared = vtk_set & nat_set
    union = vtk_set | nat_set
    jaccard = len(shared) / max(len(union), 1)
    v_only = list(vtk_set - nat_set)
    n_only = list(nat_set - vtk_set)
    print(f"  facets : {len(shared):,} shared, {len(v_only):,} VTK-only, "
          f"{len(n_only):,} native-only (Jaccard {jaccard:.4f})", flush=True)
    check(f"{label}: meshes agree on the facet set", jaccard >= 0.98, f"Jaccard {jaccard:.4f}")
    if v_only or n_only:
        # Two allowed reasons for a facet to differ, and nothing else:
        #
        # 1. A cocircular tie. Two triangulators pick opposite diagonals of the same
        #    quadrilateral; the points, the count and the surface are unchanged. The
        #    two facets then still share an edge with the other mesh.
        # 2. A duplicate plan position. LiDAR returns several Z at one XY; a 2.5D TIN
        #    can only keep one. The native mesher keeps the HIGHEST return (documented
        #    in mesh_one_tile), the CPU path keeps whichever duplicate Triangle's
        #    jettison survives - not reproducible by inspection. Such a facet has a
        #    vertex that only one of the two meshes references.
        vref = {tuple(p) for p in vtk_xyz[np.unique(vtk_faces)]}
        nref = {tuple(p) for p in nat_xyz[np.unique(nat_faces)]}
        ambiguous = vref ^ nref
        v_edges, n_edges = edge_set(n_only), edge_set(v_only)
        tie_v = sum(1 for k in v_only if shares_edge(k, v_edges))
        tie_n = sum(1 for k in n_only if shares_edge(k, n_edges))
        unexplained_v = [k for k in v_only
                         if not shares_edge(k, v_edges)
                         and not any(p in ambiguous for p in k)]
        unexplained_n = [k for k in n_only
                         if not shares_edge(k, n_edges)
                         and not any(p in ambiguous for p in k)]
        print(f"  ties   : {tie_v:,}/{len(v_only):,} VTK-only and {tie_n:,}/{len(n_only):,} "
              "native-only facets share an edge with the other mesh (diagonal flips)", flush=True)
        print(f"  dups   : {len(ambiguous):,} vertices sit at a plan position both meshes see with "
              "a different return (native keeps the highest, VTK keeps Triangle's survivor)",
              flush=True)
        check(f"{label}: every facet difference is a tie flip or a duplicate return",
              not unexplained_v and not unexplained_n,
              f"{len(unexplained_v)} VTK / {len(unexplained_n)} native unexplained")

    # ---- appearance on shared facets -----------------------------------
    if shared:
        vtk_index = {k: i for i, k in enumerate(vtk_keys)}
        nat_index = {k: i for i, k in enumerate(nat_keys)}
        pairs = [(vtk_index[k], nat_index[k]) for k in shared]
        vi = np.array([p[0] for p in pairs])
        ni = np.array([p[1] for p in pairs])
        vn = np.asarray(vtk["face_normals"], dtype=np.float64)[vi]
        nn = shading._compute_face_normals(nat_xyz, nat_faces[ni])
        cos = np.clip((vn * nn).sum(axis=1), -1.0, 1.0)
        # The CPU builder flips a normal only when |nz| > 0.3, so a steep facet keeps
        # whichever winding its triangulator chose. Both windings are lit identically
        # (audit/lnt_steep_shading_parity.py proves the GPU does the same), so the
        # geometric claim is "same plane", not "same arrow".
        flipped = int(np.count_nonzero(cos < 0))
        angle = np.degrees(np.arccos(np.clip(np.abs(cos), -1.0, 1.0)))
        # The native facet colour is trunc(classRGB * shade) against a flat 160 grey
        # probe palette, so it recovers the shade to ~1/160.
        nat_shade = native["colors"][ni].astype(np.float64).mean(axis=1) / 160.0
        dshade = np.abs(nat_shade - vtk_shade[vi])
        print(f"  shading: max plane deviation {angle.max():.4f} deg "
              f"({flipped:,} of {len(cos):,} facets wound the other way), "
              f"max |shade diff| {dshade.max():.4f}", flush=True)
        check(f"{label}: shared facets lie on the same plane", angle.max() <= 0.01,
              f"max {angle.max():.4f} deg")
        check(f"{label}: shared facets keep the same hillshade", dshade.max() <= 0.01,
              f"max {dshade.max():.4f}")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else r"H:\TESTING CONTIUES\123.laz"
    if not os.path.exists(path):
        print(f"[tin] missing dataset: {path}")
        return 2
    build_probe()
    xyz, cls = load_points(path)
    print(f"[tin] {path}: {len(xyz):,} points in a {WINDOW:g} m window", flush=True)

    counts = np.bincount(cls, minlength=256)
    ranked = [int(c) for c in np.argsort(counts)[::-1][:6] if counts[c] > 0 and c != 0]
    if len(ranked) < 2:
        print("[tin] need at least two populated classes")
        return 2
    multi = ranked[:2]
    single = ranked[:1]

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        compare_case("multi", xyz, cls, multi, tmp)
        compare_case("single", xyz, cls, single, tmp)

    failed = [name for name, ok, _ in CHECKS if not ok]
    print("\n" + "=" * 70)
    print(f"checks: {len(CHECKS) - len(failed)}/{len(CHECKS)} passed")
    print("TIN POLICY PARITY: " + ("PASS" if not failed else "FAIL"))
    if failed:
        for name in failed:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
