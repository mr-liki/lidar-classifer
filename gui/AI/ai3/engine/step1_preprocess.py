import json
import sys
import time

import laspy
import numpy as np
import pdal
from scipy.spatial import cKDTree


HAG_HEALTH_SAMPLE = 200_000
# Negative HAG shallower than this is ground roughness and is clamped to 0; it matches
# Step 3's outlier_detection.negative_hag_threshold, below which a point is low noise.
MICRO_NEGATIVE_HAG = 0.5


def report_hag_health(las, sample: int = HAG_HEALTH_SAMPLE) -> dict:
    """Print how well HeightAboveGround is anchored on this particular cloud.

    The number that matters is how far each point had to reach for its ground
    reference.  A file whose points sit metres from the nearest ground return
    (voids under a bridge, a cliff face, a canopy over a lake, ground the filter
    classified as vegetation) gets an HAG that is an extrapolation, and every
    rule keyed on HAG inherits the doubt - so the pipeline says so out loud
    instead of leaving it to be discovered on a screenshot.

    Returns the same numbers for the run manifest.
    """
    cls = np.asarray(las.classification)
    hag = np.asarray(las.HeightAboveGround, dtype=np.float64)
    stats = {
        "hag_min": float(hag.min()),
        "hag_p50": float(np.percentile(hag, 50)),
        "hag_p95": float(np.percentile(hag, 95)),
        "hag_max": float(hag.max()),
        "hag_negative": int((hag < 0).sum()),
    }
    ground = cls == 2
    stats["ground_points"] = int(ground.sum())
    if ground.sum() >= 3:
        gx = np.asarray(las.x, dtype=np.float64)[ground]
        gy = np.asarray(las.y, dtype=np.float64)[ground]
        tree = cKDTree(np.column_stack((gx, gy)), copy_data=False,
                       balanced_tree=False, compact_nodes=False)
        n = len(hag)
        idx = np.arange(n) if n <= sample else np.linspace(
            0, n - 1, sample).astype(np.int64)
        d, _ = tree.query(np.column_stack((
            np.asarray(las.x, dtype=np.float64)[idx],
            np.asarray(las.y, dtype=np.float64)[idx])), k=1, workers=-1)
        stats["ground_dist_p50"] = float(np.percentile(d, 50))
        stats["ground_dist_p90"] = float(np.percentile(d, 90))
        stats["ground_dist_p99"] = float(np.percentile(d, 99))
        stats["ground_dist_far"] = int((d > 2.0).sum())
        stats["ground_dist_sampled"] = int(len(d))
        print(f"            - Ground anchor: nearest ground point p50 {stats['ground_dist_p50']:.2f} m,"
              f" p90 {stats['ground_dist_p90']:.2f} m, p99 {stats['ground_dist_p99']:.2f} m"
              f" ({stats['ground_dist_far']:,} of {len(d):,} sampled points are > 2 m from any"
              f" ground return; their HAG is an extrapolation)")
    print(f"            - HAG p50 {stats['hag_p50']:.3f} m, p95 {stats['hag_p95']:.3f} m,"
          f" range {stats['hag_min']:.1f} .. {stats['hag_max']:.1f} m,"
          f" {stats['hag_negative']:,} points below ground")
    return stats


def run_step1_preprocessing(input_laz: str, output_laz: str):
    print(f"[Step 1] Starting Preprocessing & Height Normalization on: {input_laz}")
    start_time = time.time()

    # 1. Define PDAL pipeline
    pdal_json = [
        {"type": "readers.las", "filename": input_laz},
        # Marks statistical outliers as Classification 7 (noise); points are kept,
        # and SMRF below ignores them when building the ground surface.
        {
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": 8,
            "multiplier": 3.0,
        },
        # SMRF, not CSF. Measured on FUNIVIA000007 (median ground slope 28 deg,
        # ~200 m of relief): this SMRF finds ground in 92.7 % of 2 m cells; CSF
        # finds it in 12.6-16.2 % at resolution 1 m whatever the rigidness
        # (1 = steep, 3 = flat), and 40.9 % at 0.5 m / 1000 iterations, with
        # worse hold-out error in every case. Raising slope to 0.7 adds 3 % of
        # cells but makes the surface rougher and its gap error larger, so 0.2
        # stays.
        {
            "type": "filters.smrf",
            "ignore": "Classification[7:7]",
            "slope": 0.2,
            "threshold": 0.45,
            "window": 16.0,
            "scalar": 1.2,
        },
        # Populates 'HeightAboveGround' (Z_rel = Z_raw - Z_ground) against a
        # triangulated ground surface (TIN): each point is measured vertically
        # to the plane of the ground triangle beneath it, built from its 10
        # nearest ground returns.
        #
        # This replaces hag_nn count=3 (an inverse-distance average of the 3
        # nearest ground points), which is biased on a slope whenever those 3
        # points sit on one side - exactly the situation under a canopy gap.
        # Measured by removing ground in discs and scoring the surface at the
        # removed points (true error, not agreement):
        #   FUNIVIA000007 (76 % of ground > 15 deg), 3 m gaps: p90 1.38 -> 0.61 m,
        #     share > 0.15 m 67 % -> 41 %;  1.5 m gaps: p90 0.61 -> 0.37 m
        #   DU3039046 FUNIVIA-250-01, 3 m gaps: p90 1.14 -> 0.59 m
        #   123 (gentler, dense ground), 3 m gaps: p90 0.25 -> 0.22 m
        # On the steep tile 2.3 % of points change vegetation band; on 123, 0.5 %.
        #
        # allow_extrapolation stays on (PDAL default): with it off, a point
        # outside its local triangulation is written as HAG = 0, which would
        # present an elevated point with no ground beneath it as ground-level.
        {"type": "filters.hag_delaunay", "count": 10, "allow_extrapolation": True},
        # Micro-terrain noise only: a return a few centimetres below the TIN is
        # ground roughness, not a point under the terrain, so it is set to 0.
        # Anything at or below -0.5 m is left negative on purpose - Step 3 Rule C
        # (outlier_detection.negative_hag_threshold = -0.5) uses it to find the
        # multipath returns that sit tens of metres under the slope (down to
        # -395 m on DU3039046), and clamping those would deliver them as ground.
        {"type": "filters.assign",
         "value": f"HeightAboveGround = 0 WHERE HeightAboveGround < 0 "
                  f"&& HeightAboveGround > {-MICRO_NEGATIVE_HAG}"},
        {
            "type": "writers.las",
            "filename": output_laz,
            "extra_dims": "HeightAboveGround=float32",
            "forward": "all",
        },
    ]

    # 2. Execute PDAL pipeline
    print("        Executing PDAL outlier, SMRF & HAG_DELAUNAY (TIN) filters...")
    pipeline = pdal.Pipeline(json.dumps(pdal_json))
    point_count = pipeline.execute()
    elapsed = time.time() - start_time
    print(f"        Processed {point_count:,} points in {elapsed:.2f} seconds.")

    # 3. Validation using laspy
    print("        Validating output file...")
    with laspy.open(input_laz) as f:
        in_count = f.header.point_count
        in_dims = set(f.header.point_format.dimension_names)

    las = laspy.read(output_laz)

    # a. Point count must match the input
    if len(las.points) != in_count:
        raise ValueError(f"[X] Point count mismatch: input {in_count:,} vs output {len(las.points):,}")
    print(f"        [OK] Point count matches input: {in_count:,}")

    # original dimensions must survive
    missing = in_dims - set(las.point_format.dimension_names)
    if missing:
        raise ValueError(f"[X] Original dimensions lost: {sorted(missing)}")
    print("        [OK] Original dimensions preserved (X, Y, Z, intensity, returns, ...)")

    # b. HeightAboveGround must exist, be float32 and contain no NaN
    if "HeightAboveGround" not in las.point_format.dimension_names:
        raise ValueError("[X] 'HeightAboveGround' dimension missing in output!")
    hag = np.asarray(las.HeightAboveGround)
    if hag.dtype != np.float32:
        raise ValueError(f"[X] HeightAboveGround dtype is {hag.dtype}, expected float32")
    if not np.isfinite(hag).all():
        raise ValueError(f"[X] HeightAboveGround has {(~np.isfinite(hag)).sum():,} NaN/inf values")
    print("        [OK] 'HeightAboveGround' present, float32, no NaN/inf")
    hag_health = report_hag_health(las)

    # c. Summary statistics
    print(f"            - Min Z_rel:  {hag.min():.3f} m")
    print(f"            - Max Z_rel:  {hag.max():.3f} m")
    print(f"            - Mean Z_rel: {hag.mean():.3f} m")
    print(f"            - Std Z_rel:  {hag.std():.3f} m")

    ground = int((np.asarray(las.classification) == 2).sum())
    print(f"            - Ground-classified points: {ground:,}")
    print(f"[Step 1] Done in {time.time() - start_time:.2f} seconds -> {output_laz}")

    return {"hag_health": hag_health, "ground_classified": ground}



if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "input/123_backup.las"
    dst = sys.argv[2] if len(sys.argv) > 2 else "normalized_temp.laz"
    run_step1_preprocessing(src, dst)
